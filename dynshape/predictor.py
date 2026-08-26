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

_BYTES = {"fp32": 4, "tf32": 4, "bf16": 2, "fp16": 2, "int8": 1, "fp8": 1}


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


class GeeBackend(Backend):
    """The real EnergAIzer estimator."""

    name = "energaizer"
    is_measured_model = True

    def __init__(self, estimator):
        self.estimator = estimator

    def predict(self, q: Dict, op: Tuple[str, ...], freq: int) -> Tuple[float, float, float]:
        t, p, e = self.estimator.lookup(
            dict(q), tuple(op), target_freq=freq, lookup_target="all"
        )
        return float(t), float(p), float(e)


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

    def _gemm(self, q: Dict) -> Tuple[float, float, float]:
        b = q.get("batch", 1)
        m, n, k = q["dimM"], q["dimN"], q["dimK"]
        w = _BYTES.get(q.get("precM", "bf16"), 2)
        flops = 2.0 * b * m * n * k
        byts = float(b) * (m * k + k * n + m * n) * w
        peak = (self.gpu["peak_tc_bf16_flops"] if q.get("useTensorCore", False)
                else self.gpu["peak_cuda_fp32_flops"])
        return flops, byts, peak

    def _memory_op(self, elements: float, w: int, passes: float) -> Tuple[float, float, float]:
        # No meaningful math; the kernel is defined by how many times it must
        # stream its data.
        return 0.0, elements * w * passes, self.gpu["peak_cuda_fp32_flops"]

    def _work(self, q: Dict, op: Tuple[str, ...]) -> Tuple[float, float, float]:
        head = op[0]
        w = _BYTES.get(q.get("prec", q.get("precM", "bf16")), 2)
        if head == "gemm":
            return self._gemm(q)
        if head == "elementwise":
            return self._memory_op(q["dim"], w, 3.0)          # read, read, write
        if head == "layernorm":
            return self._memory_op(q["batch"] * q["dim"], w, 3.0)   # 2 passes + write
        if head == "softmax":
            return self._memory_op(q["batch"] * q["dim"], w, 3.0)   # max, exp/sum, write
        raise NotImplementedError(f"analytic backend has no model for op {op!r}")

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
            "distinct_shapes": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
        }


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------

def build_predictor(pkg_path: Optional[str] = None, lut_dir: Optional[str] = None,
                    gpu_yaml: Optional[str] = None, lut_yaml: Optional[str] = None,
                    freq: int = 900, force_analytic: bool = False) -> CachedPredictor:
    """Real EnergAIzer if the LUT is present, otherwise the analytic fallback.

    Never raises for a missing LUT -- it degrades, loudly, so a Colab cell that
    skipped the 500 MB download still produces a graph.
    """
    if force_analytic or not (pkg_path and lut_dir):
        return CachedPredictor(backend=AnalyticBackend(), freq=freq)

    import glob
    import os
    import sys

    if not glob.glob(os.path.join(lut_dir, "*.csv")):
        print(f"[predictor] no LUT csv in {lut_dir} -> falling back to SYNTHETIC analytic model")
        return CachedPredictor(backend=AnalyticBackend(), freq=freq)

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
        print(f"[predictor] could not build EnergAIzer ({e!r}) -> SYNTHETIC analytic model")
        return CachedPredictor(backend=AnalyticBackend(), freq=freq)

    return CachedPredictor(backend=GeeBackend(estimator), freq=freq)
