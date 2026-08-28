"""
fleet.py
========

**N replicas, one router, one facility meter.**

`engine.run_engine` prices one GPU.  This prices several, each a full model copy
with its own scheduler, its own KV pool and its own power trace, fed by one
arrival stream through a `routing.Router`.

    L0    traffic.generate_traffic()   one arrival stream
    L0.5  routing.Router               which replica  <- THE NEW LAYER
    L1    ChunkedPrefillScheduler      per replica, independent
    L2/L3 execute_iteration()          shared with run_engine, byte for byte

WHY THIS EXISTS
---------------
Facility peak is not N times a single GPU's peak.  It depends on whether the
replicas are busy *at the same time*, and the router is the only component that
decides that.  Everything else in this simulator prices work; this is the layer
that decides how work **coincides**.

    round_robin / random ->  replicas rise and fall independently
                             independent wobbles cancel as sqrt(N)
    lor                  ->  pushes work at idle replicas, smooths the fleet

The number that comes out is the **coincidence factor**: facility peak divided
by the sum of the individual replica peaks.  1.0 means every replica peaked
together and the fleet is as spiky as one GPU scaled up; 1/N means they never
did.  It is the quantity that sizes a breaker, and choosing a load balancer sets
it without anyone noticing.

Read `dynamic_coincidence_factor`, not the raw one.  Each replica draws ~47 W
whether or not it is doing anything, so on any fleet below full duty cycle that
floor dominates both halves of the ratio and drags it toward 1.0 regardless of
the router.  That is not an artefact of the simulator -- it is what a facility
meter actually sees, and it is why `NxP_idle` is reported as its own line.

**The policy only matters when the fleet is crowded.**  Below saturation every
replica is empty when a request arrives, all reactive routers tie, and the tie
rule *is* the policy: `lor` then piles everything onto replica 0.  Sweep load
before concluding anything about routers (`notebooks/Multi_Replica_Routing_Colab.ipynb`).

THE CLOCK, WITH N OF THEM
-------------------------
Each replica has its own clock, because each runs its own iterations and
EnergAIzer's kernel times drive them (design decision: the predictor *is* the
clock).  The fleet loop is therefore event-driven rather than lockstep:

    repeat:
        t_arrival = when the next request shows up
        t_replica = the earliest clock among replicas that have work
        if t_arrival <= t_replica:  route that arrival        <- router runs here
        else:                       step that replica one iteration

The ordering matters for `lor` and only for `lor`: it guarantees **no replica
advances past an unrouted arrival**, so the load a reactive router observes is
the load as of the arrival instant and never a stale or a future one.  For
`round_robin` and `random` the interleaving is irrelevant, which is exactly what
`reactive = False` means.

A replica that runs out of work stops being stepped and its clock stands still;
when a request is finally routed to it, it emits one IDLE segment covering the
whole wait.  Idle is **47 W, not zero** -- an idle GPU is a powered-on GPU, and
across a fleet that floor is most of the bill at low duty cycle.

WHAT IS DELIBERATELY NOT MODELLED
---------------------------------
Replicas are identical, requests never migrate, and the router has no affinity
for prefix-cache hits -- returning a follow-up turn to the replica already
holding its KV is a large win that real systems increasingly chase, and neither
Vidur nor FSTS models it.  There is no inter-replica network, and the router
itself is free.  Tensor and pipeline parallelism remain out of scope: these are
N independent single-GPU replicas, not one model sharded across N devices.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .engine import EngineConfig, EngineTrace, Segment, execute_iteration
from .entities import SimRequest
from .predictor import CachedPredictor
from .routing import (ReplicaLoad, Router, ROUTING_POLICIES, build_router,
                      routing_balance)
from .scheduler import ChunkedPrefillScheduler
from .template import ShapeRewriter


@dataclass
class FleetConfig:
    """How many replicas, and how to spread requests over them.

    `engine` is shared by every replica -- identical hardware, identical
    scheduler settings.  Heterogeneous fleets are out of scope (see the module
    docstring), and pretending otherwise would need a per-replica LUT anyway.

    `routing_seed` is its **own** stream, so switching `routing` from
    `round_robin` to `random` cannot shift the arrival times: the two policies
    are compared on identical traffic by construction.  That is the one thing
    Vidur's single global seed makes impossible.
    """

    num_replicas: int = 4
    routing: str = "round_robin"
    routing_seed: int = 0
    engine: EngineConfig = field(default_factory=EngineConfig)

    #: Only replica 0 keeps per-kernel segments by default.  N copies of the
    #: opening staircase is N times the memory for one extra plot nobody reads.
    record_kernels_on_replica: Optional[int] = 0

    max_total_iterations: int = 5_000_000

    def __post_init__(self):
        if self.num_replicas < 1:
            raise ValueError("num_replicas must be >= 1")
        if self.routing not in ROUTING_POLICIES:
            raise ValueError(
                f"unknown routing policy {self.routing!r}; choose from "
                f"{list(ROUTING_POLICIES)}")

    def build_router(self) -> Router:
        return build_router(self.routing, self.num_replicas, seed=self.routing_seed)


# ---------------------------------------------------------------------------


class _Replica:
    """One GPU: a scheduler, a clock, an inbox and a trace.

    Thin on purpose.  All the pricing lives in `engine.execute_iteration`, which
    `run_engine` calls too, so a fleet run and a single-replica run cannot drift
    apart in how they cost an iteration.
    """

    def __init__(self, replica_id: int, cfg: EngineConfig,
                 rewriter: ShapeRewriter, predictor: CachedPredictor,
                 record_kernels: bool = False):
        self.replica_id = replica_id
        self.cfg = cfg
        self.rewriter = rewriter
        self.predictor = predictor
        self.record_kernels = record_kernels

        self.sched = ChunkedPrefillScheduler(cfg.scheduler, cfg.model, cfg.hardware)
        self.trace = EngineTrace(config=cfg)
        self.trace.kernels_truncated_at_ms = (
            cfg.record_kernels_until_ms if record_kernels else 0.0)

        self.clock = 0.0
        self.it = 0
        self.inbox: List[SimRequest] = []
        self.cursor = 0
        self.first_iteration_ms: Optional[float] = None

    # -- what the router is allowed to see -----------------------------------

    def load(self) -> ReplicaLoad:
        """This replica's load *right now*, as `routing.ReplicaLoad`.

        `queued_requests` counts the scheduler's waiting queue **plus** anything
        routed here that its clock has not reached yet.  Both have arrived from
        the router's point of view -- a request does not stop existing because
        the GPU it was sent to is mid-iteration -- and counting only the first
        would make a replica look emptier the further behind it fell, which is
        precisely backwards.
        """
        unreleased = len(self.inbox) - self.cursor
        return ReplicaLoad(
            replica_id=self.replica_id,
            queued_requests=self.sched.num_waiting + unreleased,
            running_requests=self.sched.num_running,
            pending_tokens=self.sched.pending_tokens + sum(
                r.total_tokens for r in self.inbox[self.cursor:]),
            kv_utilisation=self.sched.allocator.utilisation,
        )

    def receive(self, request: SimRequest) -> None:
        self.inbox.append(request)
        self.trace.requests.append(request)

    def has_work(self) -> bool:
        return self.cursor < len(self.inbox) or not self.sched.is_idle()

    # -- one iteration -------------------------------------------------------

    def _keep_kernels(self) -> bool:
        if not self.record_kernels:
            return False
        lim = self.cfg.record_kernels_until_ms
        if lim is None:
            return True
        anchor = self.clock if self.first_iteration_ms is None else self.first_iteration_ms
        return (self.clock - anchor) < lim

    def step(self) -> int:
        """Advance by one iteration, or idle forward.  Returns requests completed."""
        while (self.cursor < len(self.inbox)
               and self.inbox[self.cursor].arrived_at * 1000.0 <= self.clock):
            self.sched.add_request(self.inbox[self.cursor])
            self.cursor += 1

        batch = self.sched.get_next_batch()

        if batch is None:
            if self.cursor >= len(self.inbox):
                # Nothing routed here is still to come, and nothing runnable.
                # `has_work()` should have kept the fleet loop from calling us.
                raise RuntimeError(
                    f"replica {self.replica_id}: {self.sched.num_waiting} waiting, "
                    f"{self.sched.num_running} running, and none can be "
                    f"allocated. The KV pool holds "
                    f"{self.sched.allocator.capacity_tokens} tokens; a prompt "
                    "longer than that can never be admitted.")
            # Idle until the next request routed here arrives -- real wall clock
            # at the real idle floor, not zero.
            next_ms = self.inbox[self.cursor].arrived_at * 1000.0
            dt = next_ms - self.clock
            if dt > 0:
                self.trace.idle_segments.append(Segment(
                    kind="IDLE", t_start_ms=self.clock, time_ms=dt,
                    power_w=self.cfg.idle_w,
                    energy_j=self.cfg.idle_w * dt / 1000.0,
                    op="idle", iteration=-1))
                self.clock = next_ms
            return 0

        if self.first_iteration_ms is None:
            self.first_iteration_ms = self.clock

        record = execute_iteration(
            batch, self.sched, self.rewriter, self.predictor, self.cfg,
            idx=self.it, clock_ms=self.clock,
            record_kernels=self._keep_kernels(),
            segments=self.trace.segments, skipped_ops=self.trace.skipped_ops)
        self.trace.iterations.append(record)

        self.clock += record.duration_ms + record.gap_ms
        batch.on_batch_end(self.clock / 1000.0)
        done = sum(1 for r in batch.requests if r.completed)
        self.sched.on_batch_end(batch)
        self.it += 1
        return done

    def finish(self, predictor: CachedPredictor, shape_report: str) -> EngineTrace:
        self.trace.predictor_stats = predictor.stats()
        self.trace.scheduler_report = self.sched.report()
        self.trace.shape_report = shape_report
        return self.trace


# ---------------------------------------------------------------------------


@dataclass
class FleetTrace:
    """N replica traces, plus the facility-level view that only exists together."""

    replicas: List[EngineTrace]
    routing: str
    assignment: Dict[int, int]
    requests: List[SimRequest]
    router_reactive: bool = False
    config: Optional[FleetConfig] = None

    @property
    def num_replicas(self) -> int:
        return len(self.replicas)

    @property
    def idle_w(self) -> float:
        return self.config.engine.idle_w if self.config else 0.0

    @property
    def total_time_ms(self) -> float:
        """The fleet's wall clock: until the **last** replica finishes."""
        return max((r.total_time_ms for r in self.replicas), default=0.0)

    @property
    def tail_idle_energy_j(self) -> float:
        """Energy the early finishers burn waiting for the last one.

        Charged explicitly rather than left out.  A replica that finished at 12 s
        in a 20 s run did not stop drawing power at 12 s, and dropping those 8
        seconds is how a fleet report quietly understates the floor.
        """
        total = self.total_time_ms
        return sum(self.idle_w * (total - r.total_time_ms) / 1000.0
                   for r in self.replicas)

    @property
    def total_energy_j(self) -> float:
        return sum(r.total_energy_j for r in self.replicas) + self.tail_idle_energy_j

    @property
    def avg_power_w(self) -> float:
        t = self.total_time_ms / 1000.0
        return self.total_energy_j / t if t > 0 else 0.0

    @property
    def duty_cycle(self) -> float:
        """Fraction of fleet GPU-time spent running kernels."""
        gpu_ms = self.total_time_ms * self.num_replicas
        busy = sum(r.busy_time_ms for r in self.replicas)
        return busy / gpu_ms if gpu_ms > 0 else 0.0

    # -- the facility view ---------------------------------------------------

    def resample(self, dt_ms: float = 250.0,
                 smooth_tau_ms: Optional[float] = None):
        """`(times_ms, per_replica[N, k], facility[k])` on one shared grid.

        Every replica is resampled onto the *fleet's* window, with its tail
        padded at the idle floor, so the columns really do add up: the facility
        series is the sum of the replica series bin for bin, and its integral is
        `total_energy_j`.  That is what makes a coincidence factor meaningful --
        peaks computed on grids of different lengths would not be comparable.
        """
        total = self.total_time_ms
        if total <= 0:
            return np.zeros(0), np.zeros((self.num_replicas, 0)), np.zeros(0)
        rows = []
        times = None
        for r in self.replicas:
            t, p = r.resample(dt_ms=dt_ms, smooth_tau_ms=smooth_tau_ms,
                              total_ms=total, pad_power_w=self.idle_w)
            times = t
            rows.append(p)
        n = min(len(p) for p in rows)
        per_replica = np.vstack([p[:n] for p in rows])
        return times[:n], per_replica, per_replica.sum(axis=0)

    def facility_peak_w(self, dt_ms: float = 250.0) -> float:
        _, _, total = self.resample(dt_ms=dt_ms)
        return float(total.max()) if total.size else 0.0

    def coincidence(self, dt_ms: float = 250.0) -> Dict[str, float]:
        """**The headline.** How much the replicas peaked at the same moment.

        `facility_peak_w` is the number that sizes a breaker.  The two factors
        beside it say *why* it is that big:

            coincidence_factor          facility peak / sum of replica peaks
            dynamic_coincidence_factor  the same, with the idle floor removed

        1.0 means every replica peaked in the same bin -- no smoothing at all.
        1/N means they never did.

        **Read the dynamic one.**  Every replica draws ~47 W whatever it is
        doing, and that floor sits in the numerator and the denominator of the
        raw factor alike, dragging it toward 1.0 no matter how the work is
        spread.  On a fleet whose duty cycle is low the raw factor is mostly a
        statement about the floor, not about the router.  Subtracting `N x
        idle_w` measures what actually varies.

        One honest limit on both: a router that piles everything onto one
        replica scores ~1.0, because a single machine supplies the whole peak
        and there is nothing to be out of phase *with*.  Concentration and
        synchronisation are different failures with the same score, so read
        `peak_to_mean` and `balance()` alongside.
        """
        _, per_replica, total = self.resample(dt_ms=dt_ms)
        if not total.size:
            return {}
        floor = self.idle_w
        replica_peaks = per_replica.max(axis=1)
        dyn_replica_peaks = np.maximum(per_replica - floor, 0.0).max(axis=1)
        dyn_total = np.maximum(total - floor * self.num_replicas, 0.0)

        peak_sum = float(replica_peaks.sum())
        dyn_peak_sum = float(dyn_replica_peaks.sum())
        facility_peak = float(total.max())
        dyn_peak = float(dyn_total.max())
        mean = float(total.mean())
        return {
            "aperture_ms": dt_ms,
            "facility_peak_w": facility_peak,
            "facility_mean_w": mean,
            "facility_floor_w": floor * self.num_replicas,
            "dynamic_peak_w": dyn_peak,
            "sum_of_replica_peaks_w": peak_sum,
            "coincidence_factor": facility_peak / peak_sum if peak_sum else 0.0,
            "dynamic_coincidence_factor": (dyn_peak / dyn_peak_sum
                                           if dyn_peak_sum else 0.0),
            "peak_to_mean": facility_peak / mean if mean else 0.0,
            "replica_peaks_w": [float(x) for x in replica_peaks],
        }

    def aperture_table(self, apertures_ms: Sequence[float] = (50, 250, 1000)) -> List[Dict]:
        """The same fleet read through several meter apertures.

        A peak is not a number until you say over what window -- a breaker
        responds in milliseconds, a PDU averages over seconds.  Energy is
        invariant across the rows; the peak is not.
        """
        rows = []
        for dt in apertures_ms:
            c = self.coincidence(dt_ms=dt)
            if not c:
                continue
            rows.append({
                "aperture_ms": dt,
                "facility_peak_w": c["facility_peak_w"],
                "facility_mean_w": c["facility_mean_w"],
                "dynamic_peak_w": c["dynamic_peak_w"],
                "coincidence_factor": c["coincidence_factor"],
                "dynamic_coincidence_factor": c["dynamic_coincidence_factor"],
                "peak_to_mean": c["peak_to_mean"],
                "energy_j": self.total_energy_j,
            })
        return rows

    # -- balance and per-replica ---------------------------------------------

    def balance(self) -> Dict:
        """How evenly the router split the work, by count and by tokens."""
        return routing_balance(self.assignment, self.requests, self.num_replicas)

    def per_replica_summary(self) -> List[Dict]:
        rows = []
        total = self.total_time_ms
        for k, r in enumerate(self.replicas):
            gen = sum(req.num_generated_tokens for req in r.requests)
            rows.append({
                "replica": k,
                "requests": len(r.requests),
                "completed": len(r.completed_requests),
                "iterations": len(r.iterations),
                "busy_time_s": r.busy_time_ms / 1000.0,
                "duty_cycle": r.busy_time_ms / total if total else 0.0,
                "energy_j": r.total_energy_j
                            + self.idle_w * (total - r.total_time_ms) / 1000.0,
                "avg_power_w": r.avg_power_w,
                "peak_iteration_power_w": r.peak_iteration_power_w,
                "output_tokens": gen,
                "preemptions": r.scheduler_report.get("preemptions", 0),
            })
        return rows

    def summary(self, dt_ms: float = 250.0) -> Dict:
        done = [r for r in self.requests if r.completed]
        ttfts = [r.ttft_s for r in done if r.ttft_s is not None]
        e2es = [r.e2e_s for r in done if r.e2e_s is not None]
        itls = [1000.0 * g for r in done for g in r.itl_s()]
        gen = sum(r.num_generated_tokens for r in self.requests)
        bal = self.balance()
        out = {
            "routing": self.routing,
            "num_replicas": self.num_replicas,
            "requests": len(self.requests),
            "completed": len(done),
            "wall_time_s": self.total_time_ms / 1000.0,
            "fleet_duty_cycle": self.duty_cycle,
            "total_energy_j": self.total_energy_j,
            "tail_idle_energy_j": self.tail_idle_energy_j,
            "avg_power_w": self.avg_power_w,
            "energy_per_output_token_mj": (1000.0 * self.total_energy_j / gen) if gen else 0.0,
            "output_tokens_per_s": gen / (self.total_time_ms / 1000.0) if self.total_time_ms else 0.0,
            "requests_max_over_mean": bal["requests"]["max_over_mean"],
            "tokens_max_over_mean": bal["tokens"]["max_over_mean"],
            "ttft_p50_s": _pct(ttfts, 50),
            "ttft_p99_s": _pct(ttfts, 99),
            "itl_p99_ms": _pct(itls, 99),
            "e2e_p50_s": _pct(e2es, 50),
            "preemptions": sum(r.scheduler_report.get("preemptions", 0)
                               for r in self.replicas),
        }
        c = self.coincidence(dt_ms=dt_ms)
        out.update({k: v for k, v in c.items() if k != "replica_peaks_w"})
        return out

    def to_dataframes(self):
        """(fleet_power_df, per_replica_df, requests_df)."""
        import pandas as pd
        t, per_replica, total = self.resample(dt_ms=250.0)
        power = pd.DataFrame({"t_ms": t, "facility_w": total})
        for k in range(self.num_replicas):
            power[f"replica_{k}_w"] = per_replica[k]
        rdf = pd.DataFrame(self.per_replica_summary())
        qdf = pd.DataFrame([{**r.to_dict(), "replica": self.assignment.get(r.id, -1)}
                            for r in self.requests])
        return power, rdf, qdf


def _pct(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), q))


# ---------------------------------------------------------------------------


def clone_requests(requests: Sequence[SimRequest]) -> List[SimRequest]:
    """Fresh, unrun copies -- same ids, same arrivals, same shapes.

    `SimRequest` carries its own lifecycle, so a request that has been through
    one run is *used*: its KV length, restart count and token times are all
    non-empty.  Comparing two routers on "the same traffic" means giving each a
    clone, not the same objects.
    """
    return [copy.deepcopy(r) for r in requests]


def run_fleet(
    requests: Sequence[SimRequest],
    rewriter: ShapeRewriter,
    predictor: CachedPredictor,
    config: Optional[FleetConfig] = None,
    progress_every: int = 0,
) -> FleetTrace:
    """Run one arrival stream across N replicas and price every kernel on each.

    The predictor is **shared** across replicas on purpose: identical hardware
    means identical shapes cost identical amounts, and one cache serving N
    replicas is what keeps a fleet run the same wall-clock cost as a single-GPU
    one.  `predictor.stats()` is therefore fleet-wide, and appears identically in
    every replica's `predictor_stats`.
    """
    cfg = config or FleetConfig()
    if not requests:
        raise ValueError("no requests to run")

    used = [r for r in requests if r.scheduled or r.num_processed_tokens]
    if used:
        raise ValueError(
            f"{len(used)} of {len(requests)} requests have already been run "
            "(they carry processed tokens or a schedule time). Pass "
            "clone_requests(...) -- reusing request objects silently continues "
            "the previous run instead of starting a new one.")

    router = cfg.build_router()
    router.reset()

    pending = sorted(requests, key=lambda r: (r.arrived_at, r.id))
    reps = [
        _Replica(k, cfg.engine, rewriter, predictor,
                 record_kernels=(cfg.record_kernels_on_replica == k))
        for k in range(cfg.num_replicas)
    ]

    assignment: Dict[int, int] = {}
    cursor = 0
    n_done = 0
    steps = 0
    inf = float("inf")

    while n_done < len(pending):
        if steps >= cfg.max_total_iterations:
            raise RuntimeError(
                f"stopped after {steps} fleet iterations with "
                f"{len(pending) - n_done} requests unfinished -- raise "
                "max_total_iterations or shorten the run")

        t_arrival = pending[cursor].arrived_at * 1000.0 if cursor < len(pending) else inf
        busy = [r for r in reps if r.has_work()]
        t_replica = min((r.clock for r in busy), default=inf)

        if t_arrival <= t_replica:
            # Route at the arrival instant.  No replica has advanced past it, so
            # a reactive router sees the load as of now -- never stale, never
            # ahead.  This ordering is the whole reason the loop is event-driven.
            request = pending[cursor]
            cursor += 1
            replica_id = router.route(request, [r.load() for r in reps])
            if not 0 <= replica_id < cfg.num_replicas:
                raise ValueError(
                    f"router {router.name!r} returned replica {replica_id}, "
                    f"outside [0, {cfg.num_replicas})")
            reps[replica_id].receive(request)
            assignment[request.id] = replica_id
            continue

        if not busy:
            raise RuntimeError(
                f"every replica is out of work with {len(pending) - n_done} "
                "requests unaccounted for -- this is a bookkeeping bug, not a "
                "configuration problem")

        replica = min(busy, key=lambda r: (r.clock, r.replica_id))
        n_done += replica.step()
        steps += 1

        if progress_every and steps % progress_every == 0:
            print(f"  step {steps:>7}  t={replica.clock / 1000.0:7.2f}s  "
                  f"done {n_done}/{len(pending)}  routed {cursor}", flush=True)

    from .mixed import mixed_report
    shape_report = str(mixed_report(rewriter))
    traces = [r.finish(predictor, shape_report) for r in reps]

    return FleetTrace(
        replicas=traces,
        routing=cfg.routing,
        assignment=assignment,
        requests=list(pending),
        router_reactive=router.reactive,
        config=cfg,
    )


def compare_routing(
    requests: Sequence[SimRequest],
    rewriter: ShapeRewriter,
    predictor: CachedPredictor,
    policies: Sequence[str] = ("round_robin", "random", "lor", "lor_tokens"),
    config: Optional[FleetConfig] = None,
    dt_ms: float = 250.0,
    progress: bool = False,
) -> Tuple[Dict[str, FleetTrace], "object"]:
    """The experiment the routing layer exists for: same traffic, N routers.

    Returns `(traces_by_policy, dataframe)`.  Every policy gets a **clone** of
    the request list, so the arrival times, prompt lengths and output lengths are
    identical down to the token -- the only thing that varies is which replica
    each request landed on.  Read `coincidence_factor` first: that is the column
    the router actually moves.
    """
    import pandas as pd

    base = config or FleetConfig()
    out: Dict[str, FleetTrace] = {}
    rows = []
    for policy in policies:
        cfg = dataclasses.replace(base, routing=policy)
        if progress:
            print(f"routing = {policy} ...", flush=True)
        trace = run_fleet(clone_requests(requests), rewriter, predictor, cfg)
        out[policy] = trace
        rows.append(trace.summary(dt_ms=dt_ms))
    return out, pd.DataFrame(rows)
