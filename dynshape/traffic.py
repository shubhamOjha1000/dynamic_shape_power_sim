"""
traffic.py
==========

**L0 assembled**: an interval generator x a length generator -> a list of
`SimRequest`.  This is Vidur's `SyntheticRequestGenerator`, with the seeding
fixed.

THE SEEDING FIX, AND WHY IT IS NOT COSMETIC
-------------------------------------------
Vidur seeds once, globally (`set_seeds(config.seed)`), and four of its six
random components then draw from that one shared stream.  The run is perfectly
reproducible, so this is not a reproducibility bug.  It is an **attribution**
bug, and it bites in two places:

1. The question "how much does the power trace vary from output-length
   randomness *alone*?" needs everything else held fixed while one source is
   re-rolled.  With one global stream, changing the seed re-rolls arrivals,
   lengths and routing at once -- you get a spread you cannot attribute.
2. Anything that *consumes* a random number shifts the sequence for everyone
   downstream.  Switch a length generator and the arrival times change too, so
   an experiment meant to compare two length distributions on identical traffic
   silently compares them on different traffic.

The fix is `SeedSequence.spawn`: independent child streams that stay fully
reproducible from one master seed.  Both properties at once.  Worth doing
before any ensemble work, not after -- retrofitting makes every earlier result
non-comparable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from .arrival import (GammaInterval, IntervalGenerator, PoissonInterval,
                      StaticInterval, TraceInterval)
from .entities import SimRequest, reset_ids
from .lengths import (FixedLength, LengthGenerator, TraceLength, UniformLength,
                      ZipfLength)

#: Child stream order.  Fixed, so adding a fifth consumer later cannot shift
#: the four that already exist.
_STREAMS = ("arrival", "length", "prefix", "spare")


def spawn_seeds(master_seed: int, n: int = len(_STREAMS)) -> List[int]:
    """`n` independent, reproducible child seeds from one master seed."""
    return [int(s.generate_state(1)[0]) for s in
            np.random.SeedSequence(master_seed).spawn(n)]


@dataclass
class TrafficConfig:
    """What traffic to generate.

    interval : 'static' | 'poisson' | 'gamma' | 'trace'
    length   : 'fixed'  | 'uniform' | 'zipf'   | 'trace'

    Either `num_requests` or `duration_s` bounds the stream; a trace source
    stops when it runs out regardless.

    `prefix_cache_fraction` gives every request a cached prefix of that
    fraction of its prompt -- the cheapest way to see what prefix caching does
    to the prefill share of a trace.  0.0 disables it (Vidur's behaviour).
    """

    interval: str = "poisson"
    length: str = "zipf"

    qps: float = 2.0
    cv: float = 1.0                      # gamma only

    prefill_tokens: int = 512            # fixed only
    decode_tokens: int = 128             # fixed only
    min_tokens: int = 64                 # uniform / zipf
    max_tokens: int = 4096               # uniform / zipf, and the KV sizing cap
    theta: float = 0.6                   # zipf only
    scramble: bool = False               # zipf only
    prefill_to_decode_ratio: float = 4.0

    num_requests: Optional[int] = 100
    duration_s: Optional[float] = None
    seed: int = 0

    prefix_cache_fraction: float = 0.0

    # Pre-built generators win over the string names, for trace replay and tests.
    interval_generator: Optional[IntervalGenerator] = None
    length_generator: Optional[LengthGenerator] = None

    def __post_init__(self):
        if self.num_requests is None and self.duration_s is None \
                and self.interval_generator is None and self.interval != "trace":
            raise ValueError("bound the stream with num_requests or duration_s")
        if self.num_requests is not None and self.num_requests < 1:
            raise ValueError("num_requests must be >= 1")
        if not (0.0 <= self.prefix_cache_fraction < 1.0):
            raise ValueError("prefix_cache_fraction must be in [0, 1)")

    # -- construction --------------------------------------------------------

    def build_interval(self, seed: int) -> IntervalGenerator:
        if self.interval_generator is not None:
            return self.interval_generator
        if self.interval == "static":
            return StaticInterval(qps=self.qps)
        if self.interval == "poisson":
            return PoissonInterval(qps=self.qps, seed=seed)
        if self.interval == "gamma":
            return GammaInterval(qps=self.qps, cv=self.cv, seed=seed)
        if self.interval == "trace":
            raise ValueError(
                "interval='trace' needs a prebuilt TraceInterval passed as "
                "interval_generator -- there is no default trace file")
        raise ValueError(f"unknown interval generator {self.interval!r}")

    def build_length(self, seed: int) -> LengthGenerator:
        if self.length_generator is not None:
            return self.length_generator
        if self.length == "fixed":
            return FixedLength(self.prefill_tokens, self.decode_tokens)
        if self.length == "uniform":
            return UniformLength(self.min_tokens, self.max_tokens,
                                 self.prefill_to_decode_ratio, seed=seed)
        if self.length == "zipf":
            return ZipfLength(self.min_tokens, self.max_tokens, self.theta,
                              self.scramble, self.prefill_to_decode_ratio, seed=seed)
        if self.length == "trace":
            raise ValueError(
                "length='trace' needs a prebuilt TraceLength passed as "
                "length_generator -- there is no default trace file")
        raise ValueError(f"unknown length generator {self.length!r}")


def generate_traffic(config: Optional[TrafficConfig] = None,
                     reset_request_ids: bool = True) -> List[SimRequest]:
    """The L0 output: `[SimRequest]` sorted by arrival time.

    >>> reqs = generate_traffic(TrafficConfig(interval='poisson', qps=4,
    ...                                       length='zipf', num_requests=50))
    >>> len(reqs)
    50
    """
    cfg = config or TrafficConfig()
    if reset_request_ids:
        reset_ids()

    s_arrival, s_length, s_prefix, _ = spawn_seeds(cfg.seed)
    intervals = cfg.build_interval(s_arrival)
    lengths = cfg.build_length(s_length)
    prefix_rng = np.random.default_rng(s_prefix)

    requests: List[SimRequest] = []
    t = 0.0
    while True:
        if cfg.num_requests is not None and len(requests) >= cfg.num_requests:
            break
        if cfg.duration_s is not None and t >= cfg.duration_s:
            break

        gap = intervals.next_interval()
        if gap is None:
            break
        t += float(gap)

        p, d = lengths.next_lengths()
        if p is None or d is None:
            break

        cached = 0
        if cfg.prefix_cache_fraction > 0:
            # Jitter the hit rate a little; a fixed fraction on every request is
            # a strong and unrealistic assumption.
            frac = float(np.clip(
                prefix_rng.normal(cfg.prefix_cache_fraction,
                                  cfg.prefix_cache_fraction * 0.25), 0.0, 0.95))
            cached = min(int(p * frac), p - 1)

        requests.append(SimRequest(arrived_at=t, num_prefill_tokens=int(p),
                                   num_decode_tokens=int(d),
                                   cached_prefix_tokens=cached))

    if cfg.duration_s is not None:
        requests = [r for r in requests if r.arrived_at < cfg.duration_s]
    requests.sort(key=lambda r: r.arrived_at)
    if not requests:
        raise ValueError("no requests generated -- check the generator bounds")
    return requests


def traffic_summary(requests: Sequence[SimRequest]) -> dict:
    """Read this before trusting a run: it catches a scale factor or a P:D
    ratio that pushed the workload somewhere unrealistic."""
    p = np.array([r.num_prefill_tokens for r in requests], dtype=float)
    d = np.array([r.num_decode_tokens for r in requests], dtype=float)
    a = np.array([r.arrived_at for r in requests], dtype=float)
    span = float(a.max() - a.min())
    return {
        "requests": len(requests),
        "span_s": span,
        "arrival_rate_qps": (len(requests) / span) if span > 0 else float("inf"),
        "prefill_tokens_mean": float(p.mean()),
        "prefill_tokens_p99": float(np.percentile(p, 99)),
        "decode_tokens_mean": float(d.mean()),
        "pd_ratio_median": float(np.median(p / d)),
        "total_prefill_tokens": int(p.sum()),
        "total_decode_tokens": int(d.sum()),
        "cached_prefix_tokens": int(sum(r.cached_prefix_tokens for r in requests)),
    }
