"""
predictor.py
============

Turn one kernel shape into `(time_ms, power_W, energy_J)`.

Two backends behind one interface:

  GeeBackend       the real EnergAIzer predictor -- `estimator.lookup(...)`.
                   3.1-3.8% MAPE per kernel type, and the reason this project
                   exists.  Needs the pre-collected LUT database, which is a
                   separate multi-hundred-MB download.

  AnalyticBackend  a roofline fallback so the pipeline, the tests and the plots
                   run with **no LUT and no GPU**.  Clearly SYNTHETIC: it gets
                   the *trends* right (compute- vs memory-bound, batch scaling,
                   precision) and the absolute numbers roughly.  Never report a
                   number from it as an EnergAIzer prediction.

CACHING IS NOT AN OPTIMISATION
------------------------------
A real lookup costs on the order of 50 ms.  A single 32x2048 prefill expands to
242 kernels; a few hundred requests is ~100k lookups, which is hours.  But the
distinct *shapes* number in the low thousands, because the same kernel repeats
12 times per model (once per transformer block) and batch sizes recur.  The
cache turns hours into minutes and is therefore mandatory, at simulation scope.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from .work import BYTES_PER_ELEMENT, kernel_work, uses_tensor_core

# ---------------------------------------------------------------------------
# A100-40GB-PCIe (EnergAIzer's `yz8`).  Used by the analytic backend only.
# ---------------------------------------------------------------------------

A100_PCIE = {
    "peak_tc_bf16_flops": 312e12,   # bf16/fp16 Tensor Core
    "peak_cuda_fp32_flops": 19.5e12,
    "hbm_bytes_s": 1_555e9,
    "tdp_w": 250.0,
    "idle_w": 47.0,                 # config/multiple/idle_power.yaml @ 900 MHz
    "launch_overhead_ms": 0.002,    # the epsilon in t = lambda*t_ideal + epsilon
}

#: Kept as an alias so anything importing it still works; the table now lives in
#: `work.py`, beside the FLOP/byte arithmetic that uses it.
_BYTES = BYTES_PER_ELEMENT


def query_type_of(q: Dict, op: Tuple[str, ...]) -> Tuple[str, ...]:
    """The op-type tuple EnergAIzer's `lookup` expects (already in the trace)."""
    return tuple(op)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class Backend:
    """Interface: shape in, (time_ms, power_W, energy_J) out."""

    name = "abstract"
    is_measured_model = False

    def predict(self, q: Dict, op: Tuple[str, ...], freq: int) -> Tuple[float, float, float]:
        raise NotImplementedError

    def stats(self) -> Dict:
        """Anything the backend wants reported beside the numbers."""
        return {}


def _scalar(x) -> float:
    """Unwrap whatever `lookup` hands back into a plain float.

    Necessary, not defensive: EnergAIzer's own `lookup` converts *energy* and
    *time* from a pandas Series but leaves *power* alone, so the third value can
    arrive as a one-element Series, a numpy scalar or a float depending on which
    estimator branch ran.
    """
    if hasattr(x, "values"):
        x = x.values
    if hasattr(x, "item") and getattr(x, "size", 1) == 1:
        return float(x.item())
    if isinstance(x, (list, tuple)):
        return float(x[0])
    return float(x)


class GeeBackend(Backend):
    """The real EnergAIzer estimator -- measured lookup tables, not a roofline.

    `lookup(..., lookup_target='all')` returns `(time_ms, power_W, energy_J)`.
    Those three come from related but distinct prediction paths, so they need not
    satisfy `energy == power x time` exactly, and downstream code assumes they
    do: an iteration's average power is its summed energy over its summed time.

    `reconcile` decides what to do about that, explicitly:

        'report'  return all three as measured, and record the largest
                  disagreement seen so it can be inspected.  The default,
                  because silently rewriting a measured number is worse than
                  carrying a small inconsistency you can see.
        'energy'  trust energy, derive power = E/t.  Makes per-kernel power
                  agree with every aggregate that is computed from energy.
        'power'   trust power, derive energy = P*t.

    THE COST OF A LOOKUP
    --------------------
    Every *distinct* shape re-solves a small quadratic program (cvxpy, via
    `gee/optimization_utils/cvxpy_qp.py`) to fit the analytical model's
    coefficients against nearby measured entries -- once for time and again for
    power.  That is 50-300 ms per shape, against roughly 1 microsecond for the
    roofline.  A run with 8,000 distinct shapes is therefore twenty minutes to
    an hour, and the cache is not an optimisation but the only reason it
    finishes at all.

    `use_precomputed_coeff=True` takes the fitted coefficients from a table
    instead of re-solving, which is one to two orders of magnitude faster.  The
    estimator falls back to the full solve per shape when no precomputed entry
    exists, so it degrades in accuracy only where it has to.  It is **off by
    default** because that is what the artifact's own end-to-end runs use, and
    a speed knob that silently changes numbers should be opt-in.
    """

    name = "energaizer (measured LUT)"
    is_measured_model = True

    def __init__(self, estimator, reconcile: str = "report",
                 use_precomputed_coeff: bool = False):
        if reconcile not in ("report", "energy", "power"):
            raise ValueError("reconcile must be 'report', 'energy' or 'power'")
        self.estimator = estimator
        self.reconcile = reconcile
        self.use_precomputed_coeff = use_precomputed_coeff
        self.max_inconsistency = 0.0
        self.n_predictions = 0

    def predict(self, q: Dict, op: Tuple[str, ...], freq: int) -> Tuple[float, float, float]:
        t, p, e = self.estimator.lookup(
            dict(q), tuple(op), target_freq=freq, lookup_target="all",
            use_precomputed_coeff=self.use_precomputed_coeff,
        )
        t_ms, power_w, energy_j = _scalar(t), _scalar(p), _scalar(e)

        implied = power_w * t_ms / 1000.0
        if energy_j > 0:
            self.max_inconsistency = max(
                self.max_inconsistency, abs(implied - energy_j) / energy_j)
        self.n_predictions += 1

        if self.reconcile == "energy":
            power_w = energy_j / (t_ms / 1000.0) if t_ms > 0 else power_w
        elif self.reconcile == "power":
            energy_j = implied

        return t_ms, power_w, energy_j

    def stats(self) -> Dict:
        return {
            "reconcile": self.reconcile,
            "max_energy_vs_power_x_time": self.max_inconsistency,
            "lut_lookups": self.n_predictions,
        }


class AnalyticBackend(Backend):
    """SYNTHETIC roofline stand-in -- no LUT, no GPU, no measurement.

    time  = max(flops / peak_flops, bytes / bandwidth) / efficiency + launch_overhead
    power = idle + (tdp - idle) * utilisation

    where `utilisation` blends the compute and memory roofline fractions.  A
    compute-bound bf16 GEMM lands near TDP; a bandwidth-bound elementwise kernel
    lands well below it -- which is the qualitative fact the graphs are meant to
    show.
    """

    name = "analytic (SYNTHETIC)"
    is_measured_model = False

    def __init__(self, gpu: Optional[Dict] = None, efficiency: float = 0.75,
                 mem_efficiency: float = 0.85):
        self.gpu = dict(gpu or A100_PCIE)
        self.efficiency = efficiency
        self.mem_efficiency = mem_efficiency

    # -- work / traffic per op ---------------------------------------------

    def _work(self, q: Dict, op: Tuple[str, ...]) -> Tuple[float, float, float]:
        """(flops, bytes, peak_flops).

        FLOPs and bytes come from `work.kernel_work`, which is also what the
        engine reports as its per-iteration work vector -- so the roofline and
        the reported arithmetic cannot drift apart.  Only the peak-rate choice
        is the backend's own business.
        """
        try:
            flops, byts, _weights = kernel_work(q, op)
        except NotImplementedError:
            raise NotImplementedError(f"analytic backend has no model for op {op!r}")
        peak = (self.gpu["peak_tc_bf16_flops"] if uses_tensor_core(q, op)
                else self.gpu["peak_cuda_fp32_flops"])
        return flops, byts, peak

    # -- interface ----------------------------------------------------------

    def predict(self, q: Dict, op: Tuple[str, ...], freq: int) -> Tuple[float, float, float]:
        flops, byts, peak = self._work(q, op)

        t_compute = (flops / peak / self.efficiency) if flops else 0.0
        t_memory = (byts / self.gpu["hbm_bytes_s"] / self.mem_efficiency) if byts else 0.0
        t_s = max(t_compute, t_memory)

        # Clock scaling: the roofline above is quoted at the A100's nominal
        # 1410 MHz boost, so a lower target clock stretches compute time.
        t_s *= (1410.0 / max(freq, 1)) if flops else 1.0
        t_s += self.gpu["launch_overhead_ms"] / 1000.0

        denom = max(t_s, 1e-12)
        u_compute = min(1.0, t_compute / denom)
        u_memory = min(1.0, t_memory / denom)
        # Tensor cores burn far more than the memory system per unit of busy
        # time, so the two rails are weighted differently.
        util = min(1.0, 0.80 * u_compute + 0.45 * u_memory)

        power_w = self.gpu["idle_w"] + (self.gpu["tdp_w"] - self.gpu["idle_w"]) * util
        time_ms = t_s * 1000.0
        return time_ms, power_w, power_w * t_s


# ---------------------------------------------------------------------------
# Caching wrapper
# ---------------------------------------------------------------------------

@dataclass
class CachedPredictor:
    """Memoised `(time_ms, power_W, energy_J)` per distinct (shape, op, freq).

    Cache at *simulation* scope, not per request -- shapes repeat across
    requests far more than within one.
    """

    backend: Backend
    freq: int = 900
    cache: Dict = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    def key(self, q: Dict, op: Tuple[str, ...]) -> Tuple:
        return (tuple(sorted(q.items(), key=lambda kv: kv[0])), tuple(op), self.freq)

    def predict(self, q: Dict, op: Tuple[str, ...]) -> Tuple[float, float, float]:
        k = self.key(q, op)
        if k in self.cache:
            self.hits += 1
            return self.cache[k]
        self.misses += 1
        v = self.backend.predict(q, op, self.freq)
        self.cache[k] = v
        return v

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def stats(self) -> Dict:
        return {
            "backend": self.backend.name,
            "is_measured_model": self.backend.is_measured_model,
            "freq_mhz": self.freq,
            "distinct_shapes": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            **self.backend.stats(),
        }


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------

def build_predictor(pkg_path: Optional[str] = None, lut_dir: Optional[str] = None,
                    gpu_yaml: Optional[str] = None, lut_yaml: Optional[str] = None,
                    freq: int = 900, force_analytic: bool = False,
                    require_measured: bool = False,
                    reconcile: str = "report",
                    use_precomputed_coeff: bool = False) -> CachedPredictor:
    """Real EnergAIzer if the LUT is present, otherwise the analytic fallback.

    By default this **degrades rather than raises**, so a Colab cell that skipped
    the multi-hundred-MB download still produces a graph -- but it says so, and
    every figure carries the backend name.

    Pass `require_measured=True` when the whole point of the run is the measured
    model.  Silent fallback is right for a demo and wrong there: you want to know
    you did not get EnergAIzer, not to read SYNTHETIC numbers under a heading
    that says measured.  `dynshape.energaizer.build_gee_predictor` is the same
    thing with better diagnostics.
    """
    def fallback(why: str) -> CachedPredictor:
        if require_measured:
            raise RuntimeError(
                f"require_measured=True but the measured model is unavailable: {why}")
        print(f"[predictor] {why} -> falling back to the SYNTHETIC analytic model")
        return CachedPredictor(backend=AnalyticBackend(), freq=freq)

    if force_analytic:
        if require_measured:
            raise ValueError("force_analytic and require_measured are contradictory")
        return CachedPredictor(backend=AnalyticBackend(), freq=freq)

    if not (pkg_path and lut_dir):
        return fallback("no artifact path or LUT directory given")

    import glob
    import os
    import sys

    if not glob.glob(os.path.join(lut_dir, "*.csv")):
        return fallback(f"no LUT csv in {lut_dir}")

    if pkg_path not in sys.path:
        sys.path.insert(0, pkg_path)
    try:
        from gee import get_gee
        estimator = get_gee(
            gpu_yaml_path=gpu_yaml or os.path.join(pkg_path, "config", "gpu", "yz8.yaml"),
            lut_yaml_path=lut_yaml or os.path.join(
                pkg_path, "experiments_endtoend", "exp_config", "a100_lut_config.yaml"),
            dvfs_aware=False,
            lut_folder_abs_path=lut_dir,
        )
    except Exception as e:                                   # pragma: no cover
        return fallback(f"could not build EnergAIzer ({e!r})")

    return CachedPredictor(
        backend=GeeBackend(estimator, reconcile=reconcile,
                           use_precomputed_coeff=use_precomputed_coeff),
        freq=freq)
