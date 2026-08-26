"""
arrival.py
==========

**L0, half one: when does the next request arrive?**

Ported from Vidur's `request_generator/*_request_interval_generator.py`.  The
whole interface is one method -- `next_interval() -> float | None` -- answering
"how long until the next request?".  Gaps accumulate into arrival times.  It
says nothing about *what* the request contains; that is `lengths.py`.

WHY THIS IS THE BIGGEST LEVER IN THE STACK
------------------------------------------
Identical requests per second, identical total energy, very different **peak**.
And peak is what sizes a breaker.

    static   0.5 0.5 0.5 0.5 0.5      a metronome; batch stays ~constant
    poisson  0.12 0.83 0.31 0.05 1.42 some collide, some leave long gaps
    gamma    cv > 1                   bursty: six land together, then nothing

The simplest component in the whole simulator moves the headline number more
than the scheduler does.

TWO DELIBERATE DEVIATIONS FROM VIDUR
------------------------------------
1. **Isolated RNGs.**  Vidur draws from the global `random`/`numpy` stream via
   `set_seeds()`, so four of its six random components share one sequence.  That
   is reproducible but not *attributable*: you cannot re-roll arrival times
   while holding lengths fixed, because every draw shifts the sequence for
   everyone downstream.  Each generator here owns a `numpy.random.Generator`
   spawned from one master seed -- reproducible *and* independently
   controllable, which is what an uncertainty layer needs.
2. **`numpy` instead of `scipy` for gamma.**  Same distribution, one less
   dependency, and it uses the isolated stream.

Vidur's constant-rate caveat is inherited and worth restating: `poisson` and
`gamma` are random in *when*, but their rate never moves.  Only `TraceInterval`
gives a genuinely time-varying lambda(t).  See `piecewise_poisson` for a
ten-line synthetic lambda(t) that Vidur has no generator for.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import numpy as np


class IntervalGenerator:
    """Interface: `next_interval()` -> seconds until the next arrival.

    Returning `None` means the source is exhausted (only `TraceInterval` ever
    does this), which stops generation.
    """

    name = "abstract"

    def next_interval(self) -> Optional[float]:
        raise NotImplementedError

    def describe(self) -> str:
        return self.name


class StaticInterval(IntervalGenerator):
    """Fixed rate, zero jitter -- a metronome.

    Vidur returns 0 here, meaning *every request arrives at once*.  That is a
    useful degenerate case (it isolates the scheduler by removing the arrival
    process entirely) but it is not "static rate", so both are offered:
    `qps=None` reproduces Vidur exactly, a number gives evenly spaced arrivals.
    """

    name = "static"

    def __init__(self, qps: Optional[float] = None):
        if qps is not None and qps <= 0:
            raise ValueError("qps must be > 0")
        self.qps = qps

    def next_interval(self) -> float:
        return 0.0 if self.qps is None else 1.0 / self.qps

    def describe(self) -> str:
        return "static (all at t=0)" if self.qps is None else f"static @ {self.qps} qps"


class PoissonInterval(IntervalGenerator):
    """Exponential gaps -- the standard model for many independent users.

    Faithful to Vidur, **including the tail clip**: intervals are capped at
    `3 / qps`.  That is a real behavioural choice, not an implementation detail
    -- it removes the long quiet gaps a true exponential produces, so the
    generated stream is slightly *more* regular than Poisson.  Set
    `clip_sigmas=None` for an unclipped exponential.
    """

    name = "poisson"

    def __init__(self, qps: float, seed: int = 0, clip_sigmas: Optional[float] = 3.0):
        if qps <= 0:
            raise ValueError("qps must be > 0")
        self.qps = qps
        self.rng = np.random.default_rng(seed)
        self.clip_sigmas = clip_sigmas
        self.max_interval = (clip_sigmas / qps) if clip_sigmas else math.inf

    def next_interval(self) -> float:
        u = self.rng.random()
        interval = -math.log(1.0 - u) / self.qps
        return min(interval, self.max_interval)

    def describe(self) -> str:
        return f"poisson @ {self.qps} qps"


class GammaInterval(IntervalGenerator):
    """Poisson with a burstiness dial.

    `cv` is the coefficient of variation, exactly as in Vidur:
    `shape = 1 / cv^2`, `scale = 1 / (qps * shape)`, so the mean stays `1/qps`
    whatever `cv` is -- the dial changes the *shape* of the traffic, never its
    rate.

        cv = 0.5   more regular than Poisson
        cv = 1.0   exactly Poisson
        cv = 2.0   bursty: long quiet, then a rush

    Real arrivals clump more than Poisson does, because people react to the same
    events, so `cv > 1` is closer to reality than the default.
    """

    name = "gamma"

    def __init__(self, qps: float, cv: float = 1.0, seed: int = 0):
        if qps <= 0:
            raise ValueError("qps must be > 0")
        if cv <= 0:
            raise ValueError("cv must be > 0")
        self.qps = qps
        self.cv = cv
        self.shape = 1.0 / (cv ** 2)
        self.scale = 1.0 / (qps * self.shape)
        self.rng = np.random.default_rng(seed)

    def next_interval(self) -> float:
        return float(self.rng.gamma(self.shape, self.scale))

    def describe(self) -> str:
        return f"gamma @ {self.qps} qps, cv={self.cv}"


class TraceInterval(IntervalGenerator):
    """Replay arrival times that really happened.

    **The only generator here with a time-varying rate.** The other three hold
    `qps` constant forever, which matters more than it sounds: constant-rate
    fluctuations are independent across GPUs and cancel as sqrt(N), so a
    constant-lambda facility looks artificially smoother the bigger it gets.  A
    real lambda(t) moves every GPU together and does not cancel.

    `time_scale_factor` is the load axis: 0.5 replays the same curve in half the
    wall-clock, doubling load **without changing its shape**.
    """

    name = "trace"

    def __init__(self, arrival_times_s: Sequence[float], time_scale_factor: float = 1.0):
        t = np.asarray(list(arrival_times_s), dtype=float)
        if t.size == 0:
            raise ValueError("trace is empty")
        t = np.sort(t)
        t = (t - t.min()) * float(time_scale_factor)
        self.times = t
        self.time_scale_factor = time_scale_factor
        # Vidur starts at index 1 and returns diffs; index 0 is the origin.
        self._i = 1

    @classmethod
    def from_csv(cls, path: str, column: str = "arrival_time",
                 time_scale_factor: float = 1.0) -> "TraceInterval":
        """Read a CSV column of arrival times -- seconds or parseable datetimes."""
        import pandas as pd
        df = pd.read_csv(path)
        if column not in df.columns:
            raise ValueError(f"{path!r} has no column {column!r}; found {list(df.columns)}")
        col = df[column]
        if not np.issubdtype(col.dtype, np.number):
            col = pd.to_datetime(col)
            col = (col - col.min()).dt.total_seconds()
        return cls(col.to_numpy(dtype=float), time_scale_factor=time_scale_factor)

    def next_interval(self) -> Optional[float]:
        if self._i >= len(self.times):
            return None
        gap = float(self.times[self._i] - self.times[self._i - 1])
        self._i += 1
        return gap

    def describe(self) -> str:
        return f"trace ({len(self.times)} arrivals, scale={self.time_scale_factor})"


def piecewise_poisson(rates: Sequence[float], segment_s: float, duration_s: float,
                      seed: int = 0) -> TraceInterval:
    """A synthetic **time-varying** lambda(t), which Vidur has no generator for.

    Thinning: generate at the peak rate, then keep each arrival with probability
    `lambda(t) / lambda_max`.  What survives has exactly the rate profile
    `lambda(t)`.  Give `rates` a diurnal shape and you get synthetic traffic
    with a real envelope -- the thing that makes a fleet swing rather than
    average out.

    >>> arr = piecewise_poisson([0.3, 4, 8, 5, 0.5], segment_s=60, duration_s=300)
    """
    rates = [float(r) for r in rates]
    if not rates or min(rates) < 0 or max(rates) <= 0:
        raise ValueError("rates must be non-negative with at least one positive value")
    if segment_s <= 0 or duration_s <= 0:
        raise ValueError("segment_s and duration_s must be > 0")

    rng = np.random.default_rng(seed)
    lam_max = max(rates)
    kept: List[float] = []
    t = 0.0
    while t < duration_s:
        t += -math.log(1.0 - rng.random()) / lam_max
        if t >= duration_s:
            break
        seg = min(int(t // segment_s), len(rates) - 1)
        if rng.random() < rates[seg] / lam_max:
            kept.append(t)
    if not kept:
        raise ValueError("thinning kept no arrivals -- raise the rates or the duration")
    return TraceInterval(kept)


INTERVAL_GENERATORS = {
    "static": StaticInterval,
    "poisson": PoissonInterval,
    "gamma": GammaInterval,
    "trace": TraceInterval,
}
