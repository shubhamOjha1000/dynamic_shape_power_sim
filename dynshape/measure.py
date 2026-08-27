"""
measure.py
==========

**Reading real watts off a real GPU**, so a predicted trace has something to be
wrong against.

Everything else in this package predicts.  This module measures, and the two
must be compared on terms that do not quietly favour the prediction.

WHAT NVML ACTUALLY GIVES YOU
----------------------------
`nvmlDeviceGetPowerUsage` returns **board** power in milliwatts -- the whole
card, including memory and VRM losses, not the die.  Three consequences worth
knowing before comparing anything to it:

* It is already **averaged over a hardware window** of a few milliseconds, so
  polling faster than that returns the same number repeatedly.  It cannot show
  you a 200-microsecond kernel spike, which is exactly why a predicted
  *instantaneous* peak has no measured counterpart (see `peak_power_series`).
* It includes the **idle floor**, so a comparison must either include idle on
  both sides or subtract it from both.  This module measures the floor rather
  than assuming the 47.35 W from the artifact's table.
* The reading lags the work by a millisecond or two.  Over a 250 ms bin that is
  noise; over a 10 ms bin it is a visible phase shift.

THE COMPARISON HAS TO BE FAIR
-----------------------------
Both sides get binned onto the same fixed grid by the same box filter, and the
alignment is done on the *start of work*, not on wall clock zero -- the harness
spends seconds loading a model before the first request, and counting that as
predicted idle would flatter the prediction enormously.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class PowerSample:
    """One NVML reading."""
    t_s: float
    watts: float


class PowerSampler:
    """Poll NVML board power on a background thread.

    ``interval_s`` is a floor, not a guarantee -- NVML's own refresh is a few
    milliseconds, so anything faster than ~5 ms mostly returns repeats.  250 ms
    bins are the target here, so 10 ms is ample and cheap.

    >>> s = PowerSampler().start()
    ...                                     # run the workload
    >>> t, w = s.stop()
    """

    def __init__(self, interval_s: float = 0.01, device: int = 0):
        if interval_s <= 0:
            raise ValueError("interval_s must be > 0")
        self.interval_s = interval_s
        self.device = device
        self.samples: List[PowerSample] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._handle = None
        self._t0: Optional[float] = None

    # -- device ---------------------------------------------------------------

    def _nvml(self):
        import pynvml
        pynvml.nvmlInit()
        return pynvml

    def device_info(self) -> Dict:
        """Name, power cap and clocks -- print this before trusting a comparison.

        The LUT this project predicts from was measured on an **A100-40GB-PCIe at
        900 MHz**.  Measuring on a different card, or on the same card at a
        different clock, compares two different operating points, and the
        disagreement that produces is not the model being wrong.
        """
        pynvml = self._nvml()
        h = pynvml.nvmlDeviceGetHandleByIndex(self.device)
        name = pynvml.nvmlDeviceGetName(h)
        if isinstance(name, bytes):
            name = name.decode()
        info = {
            "name": name,
            "power_limit_w": pynvml.nvmlDeviceGetPowerManagementLimit(h) / 1000.0,
            "sm_clock_mhz": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
            "max_sm_clock_mhz": pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_SM),
            "memory_total_gb": pynvml.nvmlDeviceGetMemoryInfo(h).total / 1024 ** 3,
        }
        try:
            info["persistence_mode"] = bool(pynvml.nvmlDeviceGetPersistenceMode(h))
        except Exception:
            info["persistence_mode"] = None
        return info

    def matches_lut_hardware(self, info: Optional[Dict] = None) -> Tuple[bool, str]:
        """Is this the card the lookup tables were measured on?

        Returns `(ok, why)`.  A mismatch does not make the run useless -- the
        *shape* of the trace is still comparable -- but absolute watts are not,
        and saying so is the difference between a validation and a coincidence.
        """
        info = info or self.device_info()
        name = info["name"]
        if "A100" not in name:
            return False, (
                f"{name} is not an A100. The LUT was measured on an "
                "A100-40GB-PCIe; absolute watts from another card are a "
                "different operating point, so compare shape only.")
        if abs(info["sm_clock_mhz"] - 900) > 60:
            return False, (
                f"{name} is at {info['sm_clock_mhz']} MHz, not the 900 MHz the "
                "tables were measured at. Lock it with "
                "`nvidia-smi -lgc 900` (needs root) or expect a level offset.")
        return True, f"{name} at {info['sm_clock_mhz']} MHz -- matches the LUT"

    # -- sampling -------------------------------------------------------------

    def _loop(self):
        pynvml = self._nvml()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device)
        while not self._stop.is_set():
            try:
                mw = pynvml.nvmlDeviceGetPowerUsage(self._handle)
            except Exception:
                break
            self.samples.append(PowerSample(time.perf_counter() - self._t0, mw / 1000.0))
            self._stop.wait(self.interval_s)

    def start(self) -> "PowerSampler":
        self.samples = []
        self._stop.clear()
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float = 2.0):
        import numpy as np
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
        t = np.array([s.t_s for s in self.samples], dtype=float)
        w = np.array([s.watts for s in self.samples], dtype=float)
        return t, w

    @property
    def t0(self) -> Optional[float]:
        """`perf_counter` at the moment sampling began -- the shared time origin."""
        return self._t0


def idle_baseline(seconds: float = 5.0, interval_s: float = 0.02,
                  device: int = 0) -> Dict:
    """Measure the idle floor rather than assuming it.

    The artifact's DVFS table says 47.35 W at 900 MHz for its A100.  A different
    card, a different driver, or anything else resident on the GPU moves that,
    and the floor is a constant offset under **every** sample in the run -- so an
    unmeasured floor is a systematic error in the comparison, not a rounding one.
    """
    import numpy as np

    s = PowerSampler(interval_s=interval_s, device=device).start()
    time.sleep(seconds)
    _t, w = s.stop()
    if w.size == 0:
        raise RuntimeError("NVML returned no samples; is pynvml installed and a GPU visible?")
    return {"mean_w": float(w.mean()), "median_w": float(np.median(w)),
            "min_w": float(w.min()), "max_w": float(w.max()), "samples": int(w.size)}


def bin_mean(t_s, watts, dt_s: float = 0.25, t_start: float = 0.0,
             t_end: Optional[float] = None):
    """Point samples -> fixed-grid means.  The same box filter, from the other side.

    NVML gives irregularly spaced points; `EngineTrace.resample` gives an
    energy-conserving mean over spans.  Both end up as "the mean power over this
    window", which is the only form in which they can be compared.
    """
    import numpy as np

    t_s = np.asarray(t_s, dtype=float)
    watts = np.asarray(watts, dtype=float)
    if t_s.size == 0:
        return np.zeros(0), np.zeros(0)
    if t_end is None:
        t_end = float(t_s.max())

    keep = (t_s >= t_start) & (t_s <= t_end)
    t_s, watts = t_s[keep] - t_start, watts[keep]
    span = t_end - t_start
    n = max(1, int(np.ceil(span / dt_s)))

    idx = np.clip((t_s // dt_s).astype(int), 0, n - 1)
    total = np.bincount(idx, weights=watts, minlength=n)
    count = np.bincount(idx, minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    # A bin with no sample is a gap in the *measurement*, not a zero-power
    # moment; carry the last reading forward rather than inventing a trough.
    for i in range(n):
        if np.isnan(mean[i]):
            mean[i] = mean[i - 1] if i else (watts[0] if watts.size else 0.0)
    return np.arange(n) * dt_s, mean


def compare_to_trace(measured_t, measured_w, trace, dt_s: float = 0.25,
                     measured_t0: float = 0.0, subtract_idle: Optional[float] = None):
    """Line a measured trace up against a predicted one and score the agreement.

    `measured_t0` is when the **first request was issued**, in the sampler's own
    clock.  Alignment is on the start of work rather than on sampling zero: the
    harness spends seconds loading a model first, and counting that as predicted
    idle would flatter the prediction enormously.

    `subtract_idle` removes a floor from **both** sides before scoring, which
    answers a different and often more useful question -- does the model get the
    *dynamic* power right? -- since a board-power comparison is otherwise
    dominated by a constant both sides agree on for free.
    """
    import numpy as np

    from .engine_plot import _ks_distance

    pred_t_ms, pred_w = trace.resample(dt_ms=dt_s * 1000.0)
    dur_s = float(pred_t_ms[-1] / 1000.0 + dt_s) if pred_t_ms.size else 0.0
    meas_t, meas_w = bin_mean(measured_t, measured_w, dt_s=dt_s,
                              t_start=measured_t0, t_end=measured_t0 + dur_s)

    n = min(meas_w.size, pred_w.size)
    if n < 4:
        raise ValueError(
            f"only {n} overlapping bins at dt={dt_s}s -- shorten dt_s, or check "
            "that measured_t0 really is when the first request went out")
    y, p = meas_w[:n], pred_w[:n]

    if subtract_idle is not None:
        y = np.maximum(y - subtract_idle, 0.0)
        p = np.maximum(p - subtract_idle, 0.0)

    def acf(x, max_lag):
        xc = x - x.mean()
        den = xc @ xc
        if den <= 0:
            return np.zeros(max_lag)
        return np.array([xc[:-k] @ xc[k:] / den for k in range(1, max_lag + 1)])

    rmse = float(np.sqrt(np.mean((p - y) ** 2)))
    rng = float(np.ptp(y))
    out = {
        "bins": int(n), "dt_s": dt_s, "window_s": n * dt_s,
        "measured_mean_w": float(y.mean()), "predicted_mean_w": float(p.mean()),
        "measured_peak_w": float(y.max()), "predicted_peak_w": float(p.max()),
        "measured_energy_j": float(y.sum() * dt_s),
        "predicted_energy_j": float(p.sum() * dt_s),
        "energy_error_pct": float(100 * abs(p.sum() - y.sum()) / y.sum()) if y.sum() else float("nan"),
        "mean_bias_pct": float(100 * (p.mean() - y.mean()) / y.mean()) if y.mean() else float("nan"),
        "rmse_w": rmse,
        "nrmse_range": rmse / rng if rng > 0 else float("nan"),
        "nrmse_mean": rmse / y.mean() if y.mean() > 0 else float("nan"),
        "ks_agreement": float(1.0 - _ks_distance(y, p)),
        "idle_subtracted_w": subtract_idle,
    }
    lag = min(60, n - 2)
    if lag >= 2:
        ay, ap = acf(y, lag), acf(p, lag)
        tss = float(np.sum((ay - ay.mean()) ** 2))
        out["acf_r2"] = float(1 - np.sum((ap - ay) ** 2) / tss) if tss > 0 else float("nan")
    else:
        out["acf_r2"] = float("nan")

    out["_series"] = {"t_s": (np.arange(n) * dt_s), "measured": y, "predicted": p}
    return out
