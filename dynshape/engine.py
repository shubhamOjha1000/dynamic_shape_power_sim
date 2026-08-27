"""
engine.py
=========

**The run loop.**  Traffic in, per-kernel power trace out.

    L0  traffic.generate_traffic()      when requests arrive, and how big
    L1  ChunkedPrefillScheduler         who is in this batch
    L2  mixed.build_iteration_kernels   what kernels that batch launches
    L3  CachedPredictor                 what each kernel costs

TIMING AUTHORITY
----------------
There is exactly one clock, and EnergAIzer drives it.  An iteration lasts
`sum(kernel times) + gap`, so the scheduler's timeline and the power numbers
derive from the same source by construction.  Vidur ships its own execution-time
predictor; running both would let the two disagree about how long every
iteration lasted, and there would be no principled way to reconcile them.

THREE KINDS OF TIME
-------------------
    KERNEL   a launched kernel, at its predicted power
    GAP      the inter-iteration synchronisation point -- scheduler and
             sampling -- at idle power, one per iteration, not one per kernel
    IDLE     the engine has nothing to run and is waiting for an arrival

The third only exists once there is a real arrival process, and it is the reason
average power over wall-clock is not the same as average power over busy time.
At low load a GPU spends most of its life in IDLE at ~47 W, and a report that
quietly averages only the busy segments will overstate facility draw
substantially.  Both numbers are reported, separately.

WHAT A MIXED ITERATION LOOKS LIKE
---------------------------------
Because decodes come off the top of the token budget and prefill fills the rest,
most iterations under load contain both.  That is visible in the trace: the
power staircase stops alternating between a compute-bound prefill spike and a
memory-bound decode floor, and settles into a middle band instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .entities import Batch, SimRequest
from .kvcache import HardwareConfig, ModelConfig
from .mixed import build_iteration_kernels_tagged, iteration_token_shapes, mixed_report
from .predictor import CachedPredictor
from .scheduler import ChunkedPrefillScheduler, SchedulerConfig
from .simulate import GAP_MS, IDLE_W, _shape_label
from .template import ShapeRewriter
from .work import WORK_FIELDS, empty_work, kernel_work


@dataclass
class EngineConfig:
    """How to run the engine, as opposed to how to schedule it."""

    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)

    fuse_linear: bool = True
    logits_last_token_only: bool = False

    gap_ms: float = GAP_MS
    idle_w: float = IDLE_W

    #: Per-kernel rows are only kept for the first N ms **after the first
    #: iteration starts**.  A busy run is millions of kernels; the zoomed
    #: staircase you would actually plot is the opening slice.  None keeps
    #: everything, 0 keeps none.  Per-iteration records are always kept.
    #:
    #: Measured from the first iteration rather than from t=0 on purpose: under
    #: light load the engine idles for hundreds of milliseconds before the first
    #: request arrives, and a window anchored at zero can close before any work
    #: has happened at all -- producing an empty zoom panel with no warning.
    record_kernels_until_ms: Optional[float] = 250.0

    skip_unsupported: bool = True
    max_iterations: int = 500_000


@dataclass
class IterationRecord:
    """One forward pass: what was in it, what it cost, and what state it left."""

    idx: int
    t_start_ms: float
    duration_ms: float
    gap_ms: float

    decode_batch: int
    prefill_chunks: int
    prefill_tokens: int
    decode_tokens: int
    total_tokens: int
    context_mean: float
    context_max: int

    n_kernels: int
    n_skipped: int
    energy_j: float
    avg_power_w: float
    peak_power_w: float

    energy_fused_j: float
    energy_attn_prefill_j: float
    energy_attn_decode_j: float

    n_waiting: int
    n_running: int
    kv_blocks: int
    kv_utilisation: float
    preemptions_total: int

    # -- the work vector -----------------------------------------------------
    # FLOPs and bytes implied by the shapes, independent of any power model.
    # Reported alongside the watts so the two error sources stay separable: if a
    # predicted trace disagrees with a real one, these say whether the *work* is
    # wrong (scheduler or shapes) or the *conversion to watts* is (predictor).
    linear_flops: float = 0.0
    linear_bytes: float = 0.0
    weight_bytes: float = 0.0
    attn_flops: float = 0.0
    attn_bytes: float = 0.0
    #: FLOPs split by phase.  Exact, not apportioned: a fused GEMM's rows each
    #: belong to one request and FLOPs are linear in the row count.  Bytes are
    #: deliberately *not* split -- the weight matrix is read once for the whole
    #: batch, which is the entire point of fusing it.
    prefill_flops: float = 0.0
    decode_flops: float = 0.0

    @property
    def t_end_ms(self) -> float:
        return self.t_start_ms + self.duration_ms + self.gap_ms

    @property
    def arithmetic_intensity(self) -> float:
        """FLOPs per byte -- which side of the roofline this iteration sits on."""
        b = self.linear_bytes + self.attn_bytes
        return (self.linear_flops + self.attn_flops) / b if b else 0.0

    @property
    def is_mixed(self) -> bool:
        return self.decode_batch > 0 and self.prefill_chunks > 0

    @property
    def tokens_per_s(self) -> float:
        busy = (self.duration_ms + self.gap_ms) / 1000.0
        return self.total_tokens / busy if busy > 0 else 0.0


@dataclass
class Segment:
    """A stretch of wall clock at one power level -- KERNEL, GAP or IDLE."""

    kind: str
    t_start_ms: float
    time_ms: float
    power_w: float
    energy_j: float
    op: str = ""
    shape: str = ""
    tag: str = ""
    iteration: int = -1

    @property
    def t_end_ms(self) -> float:
        return self.t_start_ms + self.time_ms


@dataclass
class EngineTrace:
    """Everything the run produced."""

    iterations: List[IterationRecord] = field(default_factory=list)
    segments: List[Segment] = field(default_factory=list)
    requests: List[SimRequest] = field(default_factory=list)
    idle_segments: List[Segment] = field(default_factory=list)
    predictor_stats: Dict = field(default_factory=dict)
    scheduler_report: Dict = field(default_factory=dict)
    shape_report: str = ""
    skipped_ops: Dict[str, int] = field(default_factory=dict)
    kernels_truncated_at_ms: Optional[float] = None
    config: Optional[EngineConfig] = None

    # -- wall clock ----------------------------------------------------------

    @property
    def total_time_ms(self) -> float:
        ends = [i.t_end_ms for i in self.iterations] + \
               [s.t_end_ms for s in self.idle_segments]
        return max(ends) if ends else 0.0

    @property
    def busy_time_ms(self) -> float:
        return sum(i.duration_ms + i.gap_ms for i in self.iterations)

    @property
    def idle_time_ms(self) -> float:
        return sum(s.time_ms for s in self.idle_segments)

    @property
    def busy_energy_j(self) -> float:
        return sum(i.energy_j for i in self.iterations)

    @property
    def idle_energy_j(self) -> float:
        return sum(s.energy_j for s in self.idle_segments)

    @property
    def total_energy_j(self) -> float:
        return self.busy_energy_j + self.idle_energy_j

    @property
    def avg_power_w(self) -> float:
        """Over **wall clock**, idle included -- what a facility meter reads."""
        t = self.total_time_ms / 1000.0
        return self.total_energy_j / t if t > 0 else 0.0

    @property
    def avg_busy_power_w(self) -> float:
        """Over busy time only -- what a kernel-level report would show."""
        t = self.busy_time_ms / 1000.0
        return self.busy_energy_j / t if t > 0 else 0.0

    @property
    def peak_iteration_power_w(self) -> float:
        return max((i.avg_power_w for i in self.iterations), default=0.0)

    @property
    def duty_cycle(self) -> float:
        return self.busy_time_ms / self.total_time_ms if self.total_time_ms else 0.0

    # -- workload ------------------------------------------------------------

    @property
    def completed_requests(self) -> List[SimRequest]:
        return [r for r in self.requests if r.completed]

    def request_metrics(self) -> List[Dict]:
        return [r.to_dict() for r in self.requests]

    def summary(self) -> Dict:
        done = self.completed_requests
        ttfts = [r.ttft_s for r in done if r.ttft_s is not None]
        e2es = [r.e2e_s for r in done if r.e2e_s is not None]
        # Inter-token latency is the other half of the chunked-prefill trade and
        # the half it is actually for: TTFT alone shows only the cost, never the
        # benefit, so a comparison reported on TTFT is guaranteed to make
        # chunking look bad.
        itls = [1000.0 * gap for r in done for gap in r.itl_s()]
        mixed = sum(1 for i in self.iterations if i.is_mixed)
        gen = sum(r.num_generated_tokens for r in self.requests)
        return {
            "iterations": len(self.iterations),
            "mixed_iterations": mixed,
            "mixed_fraction": mixed / len(self.iterations) if self.iterations else 0.0,
            "requests": len(self.requests),
            "completed": len(done),
            "wall_time_s": self.total_time_ms / 1000.0,
            "busy_time_s": self.busy_time_ms / 1000.0,
            "duty_cycle": self.duty_cycle,
            "total_energy_j": self.total_energy_j,
            "avg_power_w_wallclock": self.avg_power_w,
            "avg_power_w_busy": self.avg_busy_power_w,
            "peak_iteration_power_w": self.peak_iteration_power_w,
            "energy_per_output_token_mj": (1000.0 * self.total_energy_j / gen) if gen else 0.0,
            "output_tokens_per_s": gen / (self.total_time_ms / 1000.0) if self.total_time_ms else 0.0,
            "ttft_p50_s": _pct(ttfts, 50),
            "ttft_p99_s": _pct(ttfts, 99),
            "itl_p50_ms": _pct(itls, 50),
            "itl_p99_ms": _pct(itls, 99),
            "e2e_p50_s": _pct(e2es, 50),
            "preemptions": self.scheduler_report.get("preemptions", 0),
            "restart_work_tokens": sum(r.restart_work_tokens for r in self.requests),
            **self.predictor_stats,
        }

    # -- the joins the graphs consume ---------------------------------------

    def power_steps(self) -> Tuple[List[float], List[float]]:
        """Iteration-resolution staircase over wall clock, idle included."""
        events = [(i.t_start_ms, i.avg_power_w, i.duration_ms + i.gap_ms)
                  for i in self.iterations]
        events += [(s.t_start_ms, s.power_w, s.time_ms) for s in self.idle_segments]
        events.sort()
        ts = [e[0] for e in events]
        ps = [e[1] for e in events]
        if events:
            last = events[-1]
            ts.append(last[0] + last[2])
            ps.append(last[1])
        return ts, ps

    def resample(self, dt_ms: float = 1.0, smooth_tau_ms: Optional[float] = None,
                 include_idle: bool = True):
        """The trace on a **fixed time grid** -- `(times_ms, power_W)` arrays.

        This is what a power meter produces, and what the event-based staircase
        above is not.  Iterations have variable width, so `power_steps()` cannot
        be compared against an NVML capture, averaged across runs, or fed to
        anything expecting a uniform series.  Resampling fixes all three.

        **Energy-conserving by construction.** Each iteration's energy is spread
        across the bins it overlaps in proportion to the overlap, so the integral
        of the returned series equals `total_energy_j` whatever `dt_ms` is.  A
        box filter, which is exactly what an integrating meter does over its
        aperture.

        `smooth_tau_ms` then applies a one-pole RC response.  A real sensor never
        sees the square edges this trace produces -- board capacitance smooths
        microsecond transitions -- so comparing a raw simulated trace against a
        measured one at the same sample rate compares two different things.
        Typical board time constants are single-digit milliseconds.

        Idle stretches are included by default: leaving them out is what makes a
        duty-cycled machine look like it draws its busy power all day.
        """
        import numpy as np

        if dt_ms <= 0:
            raise ValueError("dt_ms must be > 0")
        total = self.total_time_ms
        if total <= 0:
            return np.zeros(0), np.zeros(0)

        n = max(1, int(np.ceil(total / dt_ms)))
        energy = np.zeros(n)

        spans = [(i.t_start_ms, i.t_end_ms, i.energy_j) for i in self.iterations]
        if include_idle:
            spans += [(s.t_start_ms, s.t_end_ms, s.energy_j)
                      for s in self.idle_segments]

        for a, b, e in spans:
            if b <= a or e == 0.0:
                continue
            watts = e / ((b - a) / 1000.0)
            first = min(n - 1, int(a // dt_ms))
            last = min(n - 1, int((b - 1e-12) // dt_ms))
            for k in range(first, last + 1):
                lo = max(a, k * dt_ms)
                hi = min(b, (k + 1) * dt_ms)
                if hi > lo:
                    energy[k] += watts * (hi - lo) / 1000.0

        # The final bin is usually a partial one; dividing it by a full dt would
        # understate its power.
        widths = np.full(n, dt_ms)
        widths[-1] = total - (n - 1) * dt_ms or dt_ms
        power = energy / (widths / 1000.0)

        if smooth_tau_ms:
            alpha = 1.0 - float(np.exp(-dt_ms / smooth_tau_ms))
            out = np.empty_like(power)
            acc = float(power[0])
            for k, x in enumerate(power):
                acc += (float(x) - acc) * alpha
                out[k] = acc
            power = out

        return np.arange(n) * dt_ms, power

    def work_totals(self) -> Dict[str, float]:
        """The work vector summed over the run -- backend-independent totals."""
        return {f: sum(getattr(i, f) for i in self.iterations) for f in WORK_FIELDS}

    def kernel_power_steps(self) -> Tuple[List[float], List[float]]:
        """Kernel-resolution staircase -- only where kernels were recorded."""
        segs = sorted(self.segments, key=lambda s: s.t_start_ms)
        ts = [s.t_start_ms for s in segs]
        ps = [s.power_w for s in segs]
        if segs:
            ts.append(segs[-1].t_end_ms)
            ps.append(segs[-1].power_w)
        return ts, ps

    def to_dataframes(self):
        """(iterations_df, requests_df, segments_df)."""
        import pandas as pd
        idf = pd.DataFrame([{**i.__dict__, "is_mixed": i.is_mixed,
                             "tokens_per_s": i.tokens_per_s,
                             "arithmetic_intensity": i.arithmetic_intensity}
                            for i in self.iterations])
        rdf = pd.DataFrame(self.request_metrics())
        sdf = pd.DataFrame([s.__dict__ for s in self.segments])
        return idf, rdf, sdf


def _pct(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    import numpy as np
    return float(np.percentile(np.asarray(values, dtype=float), q))


# ---------------------------------------------------------------------------


def run_engine(
    requests: Sequence[SimRequest],
    rewriter: ShapeRewriter,
    predictor: CachedPredictor,
    config: Optional[EngineConfig] = None,
    progress_every: int = 0,
) -> EngineTrace:
    """Run an arrival stream through the scheduler and price every kernel.

    `requests` come from `traffic.generate_traffic()`; they are consumed in
    arrival order, and the engine idles between them when there is nothing to
    run.
    """
    cfg = config or EngineConfig()
    if not requests:
        raise ValueError("no requests to run")

    pending = sorted(requests, key=lambda r: (r.arrived_at, r.id))
    sched = ChunkedPrefillScheduler(cfg.scheduler, cfg.model, cfg.hardware)
    trace = EngineTrace(requests=list(pending), config=cfg)
    trace.shape_report = str(mixed_report(rewriter))
    trace.kernels_truncated_at_ms = cfg.record_kernels_until_ms

    clock = 0.0
    cursor = 0
    n_done = 0
    it = 0
    first_iteration_ms: Optional[float] = None

    def record_kernels_now() -> bool:
        lim = cfg.record_kernels_until_ms
        if lim is None:
            return True
        anchor = clock if first_iteration_ms is None else first_iteration_ms
        return (clock - anchor) < lim

    while n_done < len(pending):
        if it >= cfg.max_iterations:
            raise RuntimeError(
                f"stopped after {it} iterations with {len(pending) - n_done} "
                "requests unfinished -- raise max_iterations or shorten the run")

        while cursor < len(pending) and pending[cursor].arrived_at * 1000.0 <= clock:
            sched.add_request(pending[cursor])
            cursor += 1

        batch = sched.get_next_batch()

        if batch is None:
            if cursor >= len(pending):
                # Nothing left to arrive and nothing runnable.  Either every
                # request finished, or one cannot fit in the pool at all.
                if sched.is_idle():
                    # The loop condition should already have ended the run; if it
                    # has not, a request went missing rather than completing, and
                    # a silent break would hand back a trace with a hole in it.
                    raise RuntimeError(
                        f"the scheduler went idle with {len(pending) - n_done} "
                        "requests unaccounted for -- this is a bug in the "
                        "scheduler's bookkeeping, not a configuration problem")
                raise RuntimeError(
                    f"deadlock: {sched.num_waiting} requests waiting, "
                    f"{sched.num_running} running, and none can be allocated. "
                    f"The KV pool holds {sched.allocator.capacity_tokens} tokens; "
                    "a prompt longer than that can never be admitted.")
            # Idle until the next arrival -- real wall clock at real idle power.
            next_ms = pending[cursor].arrived_at * 1000.0
            if next_ms > clock:
                dt = next_ms - clock
                trace.idle_segments.append(Segment(
                    kind="IDLE", t_start_ms=clock, time_ms=dt,
                    power_w=cfg.idle_w, energy_j=cfg.idle_w * dt / 1000.0,
                    op="idle", iteration=-1))
                clock = next_ms
            continue

        if first_iteration_ms is None:
            first_iteration_ms = clock

        batch.on_schedule(clock / 1000.0)
        pieces = batch.pieces
        shapes = iteration_token_shapes(pieces)

        tagged = build_iteration_kernels_tagged(
            rewriter, pieces,
            fuse_linear=cfg.fuse_linear,
            logits_last_token_only=cfg.logits_last_token_only)

        t_iter = clock
        energy = 0.0
        peak = 0.0
        by_tag = {"fused": 0.0, "attn:prefill": 0.0, "attn:decode": 0.0,
                  "whole:prefill": 0.0, "whole:decode": 0.0}
        n_ok = n_skip = 0
        keep = record_kernels_now()

        work = empty_work()
        prefill_share = (shapes["prefill_tokens"] / shapes["total_tokens"]
                         if shapes["total_tokens"] else 0.0)

        for (q, op), tag in tagged:
            # Work accounting is independent of whether the predictor has a model
            # for this op, so it happens before the prediction and survives a skip.
            k_flops, k_bytes, k_weights = kernel_work(q, op, strict=False)
            if tag.endswith("prefill") or tag.endswith("decode"):
                work["attn_flops"] += k_flops
                work["attn_bytes"] += k_bytes
                if tag.endswith("prefill"):
                    work["prefill_flops"] += k_flops
                else:
                    work["decode_flops"] += k_flops
            else:
                work["linear_flops"] += k_flops
                work["linear_bytes"] += k_bytes
                work["weight_bytes"] += k_weights
                # Exact, not apportioned: a fused GEMM's rows each belong to one
                # request and FLOPs are linear in the row count.
                work["prefill_flops"] += k_flops * prefill_share
                work["decode_flops"] += k_flops * (1.0 - prefill_share)

            try:
                time_ms, power_w, energy_j = predictor.predict(q, op)
            except Exception:
                if not cfg.skip_unsupported:
                    raise
                name = op[0] if op else "?"
                trace.skipped_ops[name] = trace.skipped_ops.get(name, 0) + 1
                n_skip += 1
                continue

            if keep:
                trace.segments.append(Segment(
                    kind="KERNEL", t_start_ms=t_iter, time_ms=time_ms,
                    power_w=power_w, energy_j=energy_j, op=" ".join(op),
                    shape=_shape_label(q, op), tag=tag, iteration=it))
            t_iter += time_ms
            energy += energy_j
            by_tag[tag] = by_tag.get(tag, 0.0) + energy_j
            peak = max(peak, power_w)
            n_ok += 1

        duration_ms = t_iter - clock
        gap_j = cfg.idle_w * cfg.gap_ms / 1000.0
        if keep and cfg.gap_ms > 0:
            trace.segments.append(Segment(
                kind="GAP", t_start_ms=t_iter, time_ms=cfg.gap_ms,
                power_w=cfg.idle_w, energy_j=gap_j, op="gap", iteration=it))
        total_energy = energy + (gap_j if cfg.gap_ms > 0 else 0.0)
        span_ms = duration_ms + cfg.gap_ms

        trace.iterations.append(IterationRecord(
            idx=it, t_start_ms=clock, duration_ms=duration_ms, gap_ms=cfg.gap_ms,
            decode_batch=shapes["decode_batch"],
            prefill_chunks=shapes["prefill_chunks"],
            prefill_tokens=shapes["prefill_tokens"],
            decode_tokens=shapes["decode_batch"],
            total_tokens=shapes["total_tokens"],
            context_mean=shapes["context_mean"],
            context_max=shapes["context_max"],
            n_kernels=n_ok, n_skipped=n_skip,
            energy_j=total_energy,
            avg_power_w=(total_energy / (span_ms / 1000.0)) if span_ms > 0 else 0.0,
            peak_power_w=peak,
            energy_fused_j=by_tag["fused"],
            energy_attn_prefill_j=by_tag["attn:prefill"] + by_tag["whole:prefill"],
            energy_attn_decode_j=by_tag["attn:decode"] + by_tag["whole:decode"],
            n_waiting=sched.num_waiting, n_running=sched.num_running,
            kv_blocks=sched.allocator.num_allocated_blocks,
            kv_utilisation=sched.allocator.utilisation,
            preemptions_total=sched.num_preemptions,
            **work,
        ))

        clock += span_ms
        batch.on_batch_end(clock / 1000.0)
        # Counted incrementally, not by rescanning: a completed request is freed
        # and never re-admitted, so it can only be counted once.
        n_done += sum(1 for r in batch.requests if r.completed)
        sched.on_batch_end(batch)
        it += 1

        if progress_every and it % progress_every == 0:
            print(f"  iter {it:>6}  t={clock / 1000.0:7.2f}s  "
                  f"done {n_done}/{len(pending)}  "
                  f"kv {sched.allocator.utilisation:5.1%}", flush=True)

    trace.predictor_stats = predictor.stats()
    trace.scheduler_report = sched.report()
    return trace
