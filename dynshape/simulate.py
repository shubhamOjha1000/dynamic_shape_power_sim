"""
simulate.py
===========

Lay the expanded kernels on a timeline and read off **time and average power**.

  Request(batch, seqlen, mode)
        -> ShapeRewriter.expand(...)        242 kernel shapes, any (B, S, mode)
        -> CachedPredictor.predict(...)     (time_ms, power_W, energy_J) each
        -> KernelRecord[] laid end to end   t_start accumulates
        -> RequestRecord[]                  per-pass time and average power

Two modelling choices, both inherited from the v1 design and both worth stating
because they are choices, not facts:

GAPS ARE PER REQUEST, NOT PER KERNEL.
    EnergAIzer's latency correction `t = lambda * t_ideal + epsilon` already
    folds per-launch overhead into each kernel's time, so adding a gap after
    every kernel double-counts it.  Within a forward pass the CPU runs ahead of
    the GPU anyway, so those gaps do not materialise as idle time.  A real gap
    exists only *between* passes, at the synchronisation point.

GAP POWER IS IDLE, NOT ZERO.
    The power model is `P = sum(alpha * C * V^2 * f) + P_idle(f)`.  A gap is the
    same equation with every utilisation at zero, which leaves P_idle -- about
    47 W on an A100 at 900 MHz.  Chip *dynamic* power goes to zero; the board
    does not.

KERNEL COSTS ARE ASSUMED ADDITIVE.
    No overlap, no memory-system contention, no cache carry-over between
    kernels.  This is the largest untested assumption in the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .predictor import CachedPredictor
from .template import ShapeRewriter
from .workload import Request

IDLE_W = 47.0            # A100 @ 900 MHz, config/multiple/idle_power.yaml
GAP_MS = 0.05            # inter-pass synchronisation / launch gap


@dataclass
class KernelRecord:
    req_idx: int
    kernel_idx: int
    kind: str            # 'KERNEL' | 'GAP'
    op: str
    t_start_ms: float
    time_ms: float
    power_w: float
    energy_j: float
    shape: str

    @property
    def t_end_ms(self) -> float:
        return self.t_start_ms + self.time_ms


@dataclass
class RequestRecord:
    idx: int
    batch: int
    seqlen: int
    mode: str
    tokens: int
    t_start_ms: float
    time_ms: float
    avg_power_w: float
    energy_j: float
    n_kernels: int
    n_skipped: int = 0

    @property
    def tokens_per_s(self) -> float:
        return self.tokens / (self.time_ms / 1000.0) if self.time_ms > 0 else 0.0

    @property
    def energy_per_token_mj(self) -> float:
        return (self.energy_j / self.tokens * 1000.0) if self.tokens else 0.0


@dataclass
class Trace:
    """Everything the run produced, plus the joins the graphs need."""

    kernels: List[KernelRecord] = field(default_factory=list)
    requests: List[RequestRecord] = field(default_factory=list)
    predictor_stats: Dict = field(default_factory=dict)
    skipped_ops: Dict[str, int] = field(default_factory=dict)

    # -- aggregate ----------------------------------------------------------

    @property
    def total_time_ms(self) -> float:
        return self.kernels[-1].t_end_ms if self.kernels else 0.0

    @property
    def total_energy_j(self) -> float:
        return sum(k.energy_j for k in self.kernels)

    @property
    def avg_power_w(self) -> float:
        t = self.total_time_ms / 1000.0
        return self.total_energy_j / t if t > 0 else 0.0

    @property
    def peak_power_w(self) -> float:
        return max((k.power_w for k in self.kernels), default=0.0)

    def summary(self) -> Dict:
        return {
            "requests": len(self.requests),
            "kernels": sum(1 for k in self.kernels if k.kind == "KERNEL"),
            "total_time_ms": self.total_time_ms,
            "total_energy_j": self.total_energy_j,
            "avg_power_w": self.avg_power_w,
            "peak_power_w": self.peak_power_w,
            **self.predictor_stats,
        }

    # -- the (time, power) join the graphs consume --------------------------

    def power_steps(self) -> Tuple[List[float], List[float]]:
        """The trace as a staircase: (times_ms, powers_W) for `plt.step`.

        Each kernel contributes its start; a final point closes the last step,
        so `step(..., where='post')` draws the true piecewise-constant signal
        rather than interpolating between kernels.
        """
        ts = [k.t_start_ms for k in self.kernels]
        ps = [k.power_w for k in self.kernels]
        if self.kernels:
            ts.append(self.kernels[-1].t_end_ms)
            ps.append(self.kernels[-1].power_w)
        return ts, ps

    def to_dataframes(self):
        """(kernels_df, requests_df) -- pandas, for slicing and plotting."""
        import pandas as pd
        kdf = pd.DataFrame([k.__dict__ for k in self.kernels])
        rdf = pd.DataFrame([{**r.__dict__,
                             "tokens_per_s": r.tokens_per_s,
                             "energy_per_token_mj": r.energy_per_token_mj}
                            for r in self.requests])
        return kdf, rdf


# ---------------------------------------------------------------------------


def _shape_label(q: Dict, op: Tuple[str, ...]) -> str:
    if op and op[0] == "gemm":
        return f"{q.get('batch', 1)}x{q['dimM']}x{q['dimN']}x{q['dimK']}"
    if "dim" in q and "batch" in q:
        return f"{q['batch']}x{q['dim']}"
    if "dim" in q:
        return str(q["dim"])
    return ""


def simulate(
    requests: Sequence[Request],
    rewriter: ShapeRewriter,
    predictor: CachedPredictor,
    idle_w: float = IDLE_W,
    gap_ms: float = GAP_MS,
    skip_unsupported: bool = True,
    progress: bool = False,
) -> Trace:
    """Run a dynamic shape stream through the predictor and build the trace.

    `skip_unsupported` keeps a run alive when the backend has no model for one
    op type: the kernel is dropped and counted rather than aborting.  The count
    is reported in `Trace.skipped_ops`, so a run that quietly lost 30% of its
    kernels is visible rather than silently cheap.
    """
    trace = Trace()
    t = 0.0

    for n, req in enumerate(requests):
        if progress and n % 10 == 0:
            print(f"  [{n}/{len(requests)}] {req.label}", flush=True)

        kernels = rewriter.expand(req.batch, req.seqlen, req.mode)

        r_start, r_energy, n_ok, n_skip = t, 0.0, 0, 0
        for ki, (q, op) in enumerate(kernels):
            try:
                time_ms, power_w, energy_j = predictor.predict(q, op)
            except Exception as e:
                if not skip_unsupported:
                    raise
                name = op[0] if op else "?"
                trace.skipped_ops[name] = trace.skipped_ops.get(name, 0) + 1
                n_skip += 1
                continue

            trace.kernels.append(KernelRecord(
                req_idx=req.idx, kernel_idx=ki, kind="KERNEL", op=" ".join(op),
                t_start_ms=t, time_ms=time_ms, power_w=power_w, energy_j=energy_j,
                shape=_shape_label(q, op),
            ))
            t += time_ms
            r_energy += energy_j
            n_ok += 1

        # One gap per pass, at idle power -- see the module docstring.
        if gap_ms > 0:
            gap_j = idle_w * gap_ms / 1000.0
            trace.kernels.append(KernelRecord(
                req_idx=req.idx, kernel_idx=-1, kind="GAP", op="gap",
                t_start_ms=t, time_ms=gap_ms, power_w=idle_w, energy_j=gap_j,
                shape="",
            ))
            t += gap_ms
            r_energy += gap_j

        busy_ms = t - r_start
        trace.requests.append(RequestRecord(
            idx=req.idx, batch=req.batch, seqlen=req.seqlen, mode=req.mode,
            tokens=req.tokens, t_start_ms=r_start, time_ms=busy_ms,
            avg_power_w=(r_energy / (busy_ms / 1000.0)) if busy_ms > 0 else 0.0,
            energy_j=r_energy, n_kernels=n_ok, n_skipped=n_skip,
        ))

    trace.predictor_stats = predictor.stats()
    return trace
