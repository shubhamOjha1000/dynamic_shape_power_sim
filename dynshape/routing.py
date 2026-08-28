"""
routing.py
==========

**L0.5: which replica gets this request?**

Several GPUs each run their own complete copy of the model.  A request arrives
and something has to pick one.  That decision is *multi-replica routing* --
Vidur's `global_scheduler` -- and it is the last L0/L1 feature this simulator
was missing.

    request arrives
          |
      [ ROUTER ]  <- picks one
       |   |   |
       A   B   C   each a full model copy, each with its own scheduler,
                   its own KV pool and its own power trace

**One router decision per request.**  After that the chosen replica's own
scheduler takes over -- batching, KV admission, preemption -- independently of
the others.  Nothing migrates.

WHY THIS IS A POWER FEATURE, NOT A LATENCY FEATURE
--------------------------------------------------
Facility peak depends on whether replicas are busy *at the same time*, and the
router is the only thing in the stack that decides that.  It is the correlation
knob, even though nobody labels it that way:

    round_robin / random  ->  replicas rise and fall largely independently
                              peaks land where they land
    lor                   ->  actively pushes work toward idle replicas
                              -> smooths the fleet
                              -> LOWER facility peak

A load balancer is chosen by systems engineers for latency reasons.  It quietly
changes facility peak power, which is the number that sizes a breaker.  With a
fleet loop underneath (`fleet.py`) that trade is finally measurable.

THE FOUR POLICIES, AND WHICH ONE IS VIDUR'S
-------------------------------------------
| name           | rule                                    | source            |
|----------------|-----------------------------------------|-------------------|
| `round_robin`  | take turns in order                     | Vidur, exact      |
| `random`       | uniform over replicas                   | Vidur, reseeded   |
| `lor`          | fewest *queued* requests                | Vidur, exact      |
| `lor_requests` | fewest queued **+ running** requests     | ours              |
| `lor_tokens`   | fewest outstanding *tokens*              | ours              |
| `lor_kv`       | lowest KV utilisation                    | ours              |

The last three exist because of a caution worth stating loudly.

**Vidur's LOR counts the waiting queue only.**  `num_pending_requests` is
`len(self._request_queue)` -- requests that have arrived and have *not yet been
admitted*.  A replica with a hundred requests mid-decode and an empty queue
reports zero.  So under any load light enough that queues drain each iteration,
every replica reports 0, `min()` returns the first key, and **LOR degenerates to
"always replica 0"** -- a hot spot, not a balancer.  Vidur's own comment calls it
"a very simple implementation... to keep wiring simple", which is fair; this is
not a bug report, it is a statement about what the number means.  It is pinned
by a test (`test_routing.py::test_vidur_lor_collapses_when_queues_are_empty`)
because it is exactly the kind of thing that looks like a simulator bug later.

`lor_requests` is what most people mean when they say LOR.  `lor_tokens` is
closer to what real balancers do -- an 8000-token prefill and a 50-token chat
turn are not "one each" -- and `lor_kv` routes on the resource that actually
runs out.

THE SEEDING FIX, CARRIED THROUGH
--------------------------------
Vidur's `random` router draws `randint(1, num_replicas) - 1` from the **global**
stream that its Poisson arrivals and uniform lengths also draw from.  Switching
from `round_robin` to `random` therefore consumes random numbers that nothing
consumed before, which shifts every arrival time downstream: an experiment meant
to compare two routers on identical traffic silently compares them on different
traffic.  Here the router owns a `SeedSequence.spawn` child of its own (the
stream `traffic.py` reserved as `"spare"` and this module now claims), so the
comparison is on the same arrivals by construction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .entities import SimRequest

#: Every policy name `build_router` accepts.
ROUTING_POLICIES = ("round_robin", "random", "lor", "lor_requests",
                    "lor_tokens", "lor_kv")


@dataclass(frozen=True)
class ReplicaLoad:
    """What a router is allowed to see about one replica, at one instant.

    Deliberately small.  A router that could see everything would be a scheduler,
    and the whole point of the seam is that the two are separate machines: the
    router decides *where*, the replica scheduler decides *when*.

    queued_requests : arrived at this replica, not yet admitted.  **This is
                      Vidur's `num_pending_requests`** -- the number `lor` uses.
    running_requests: admitted and holding KV blocks.
    pending_tokens  : prompt tokens not yet processed, summed over queued and
                      running requests.  Work rather than headcount.
    kv_utilisation  : fraction of the KV block pool allocated, in [0, 1].
    """

    replica_id: int
    queued_requests: int = 0
    running_requests: int = 0
    pending_tokens: int = 0
    kv_utilisation: float = 0.0

    @property
    def outstanding_requests(self) -> int:
        """Queued **and** running -- what LOR is usually assumed to count."""
        return self.queued_requests + self.running_requests


class Router(ABC):
    """Pick a replica for one request.  One decision, never revisited."""

    #: Policy name, as it appears in reports and figures.
    name: str = "router"

    #: Does this router read replica state?  False means the assignment can be
    #: computed from the arrival list alone, before any simulation runs -- which
    #: `assign_static` relies on, and which is itself the interesting property:
    #: a router that cannot see load cannot react to it.
    reactive: bool = False

    @abstractmethod
    def route(self, request: SimRequest, loads: Sequence[ReplicaLoad]) -> int:
        """Return the replica index this request is assigned to."""

    def reset(self) -> None:
        """Forget any per-run state.  Called before each run."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


class RoundRobinRouter(Router):
    """Take turns in order.  Vidur's `RoundRobinGlobalScheduler`, exactly.

    Perfectly even *counts* -- and counts are not load.  One 4000-token request
    and one 50-token request are "one each".  Worse, request sizes and arrival
    order are uncorrelated, so nothing stops two giants three apart landing on
    the same replica purely by alignment.
    """

    name = "round_robin"
    reactive = False

    def __init__(self, num_replicas: int):
        if num_replicas < 1:
            raise ValueError("num_replicas must be >= 1")
        self.num_replicas = int(num_replicas)
        self._counter = 0

    def reset(self) -> None:
        self._counter = 0

    def route(self, request: SimRequest, loads: Sequence[ReplicaLoad]) -> int:
        replica_id = self._counter % self.num_replicas
        self._counter += 1
        return replica_id


class RandomRouter(Router):
    """Uniform over replicas, from an **isolated** stream.

    Vidur's `RandomGlobalScheduler` computes `randint(1, num_replicas) - 1`,
    which is uniform on `[0, num_replicas)` -- the same distribution -- but draws
    it from the global RNG.  The distribution is what matters for the physics;
    the stream is what matters for comparing two runs (see the module docstring).

    This is also, exactly, FSTS's paper-described "shared-intensity with
    independent thinning": one arrival stream, split by coin flip.  Each replica
    then sees a Poisson process of rate lambda/N, independent of the others.
    """

    name = "random"
    reactive = False

    def __init__(self, num_replicas: int, seed: int = 0):
        if num_replicas < 1:
            raise ValueError("num_replicas must be >= 1")
        self.num_replicas = int(num_replicas)
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)

    def reset(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def route(self, request: SimRequest, loads: Sequence[ReplicaLoad]) -> int:
        return int(self._rng.integers(0, self.num_replicas))


class LORRouter(Router):
    """Least outstanding requests -- the only policy that reacts to load.

    `metric` chooses what "outstanding" counts:

        'queue'     len(waiting queue)          <- Vidur's, exactly
        'requests'  queued + running
        'tokens'    prompt tokens not yet processed
        'kv'        KV pool utilisation

    Ties go to the lowest replica id, which is what Python's `min()` over
    Vidur's `pending_requests_map` does too (dict insertion order is replica
    order).  That tie rule is not cosmetic: with `metric='queue'` on an
    unsaturated fleet **every** replica ties at zero, so the tie rule *is* the
    policy.  See the module docstring.
    """

    _METRICS = {
        "queue": lambda load: float(load.queued_requests),
        "requests": lambda load: float(load.outstanding_requests),
        "tokens": lambda load: float(load.pending_tokens),
        "kv": lambda load: float(load.kv_utilisation),
    }

    reactive = True

    def __init__(self, num_replicas: int, metric: str = "queue"):
        if num_replicas < 1:
            raise ValueError("num_replicas must be >= 1")
        if metric not in self._METRICS:
            raise ValueError(
                f"unknown LOR metric {metric!r}; choose from "
                f"{sorted(self._METRICS)}")
        self.num_replicas = int(num_replicas)
        self.metric = metric
        self.name = "lor" if metric == "queue" else f"lor_{metric}"

    def route(self, request: SimRequest, loads: Sequence[ReplicaLoad]) -> int:
        if len(loads) != self.num_replicas:
            raise ValueError(
                f"router was built for {self.num_replicas} replicas but saw "
                f"{len(loads)} loads")
        key = self._METRICS[self.metric]
        best_id, best_val = 0, float("inf")
        for k, load in enumerate(loads):
            val = key(load)
            if val < best_val:          # strict: ties keep the lower index
                best_id, best_val = k, val
        return best_id


def build_router(policy: str, num_replicas: int, seed: int = 0) -> Router:
    """`'round_robin' | 'random' | 'lor' | 'lor_requests' | 'lor_tokens' | 'lor_kv'`."""
    if policy == "round_robin":
        return RoundRobinRouter(num_replicas)
    if policy == "random":
        return RandomRouter(num_replicas, seed=seed)
    if policy == "lor":
        return LORRouter(num_replicas, metric="queue")
    if policy.startswith("lor_"):
        return LORRouter(num_replicas, metric=policy[len("lor_"):])
    raise ValueError(
        f"unknown routing policy {policy!r}; choose from {list(ROUTING_POLICIES)}")


def assign_static(requests: Sequence[SimRequest], router: Router,
                  num_replicas: int) -> Dict[int, int]:
    """Assignment for a **non-reactive** router, without running anything.

    Useful for checking a split before paying for a simulation, and for the
    thinning argument: `random` over one Poisson stream gives N independent
    Poisson streams of rate lambda/N, which is a claim about the arrival times
    alone and needs no engine to verify.

    Raises for `lor`, which cannot be resolved without live replica state --
    that difference *is* the feature.
    """
    if router.reactive:
        raise ValueError(
            f"{router.name} reads replica load, so its assignment only exists "
            "while the fleet runs -- use run_fleet()")
    router.reset()
    empty = [ReplicaLoad(replica_id=k) for k in range(num_replicas)]
    return {r.id: router.route(r, empty)
            for r in sorted(requests, key=lambda r: (r.arrived_at, r.id))}


def routing_balance(assignment: Dict[int, int],
                    requests: Sequence[SimRequest],
                    num_replicas: int) -> Dict:
    """How evenly an assignment split the work -- by count *and* by tokens.

    The two disagree, and the disagreement is the point: `round_robin` is exact
    on counts and can be badly skewed on tokens, because it never looks at how
    big a request is.  `max_over_mean` is the number to read -- 1.0 is perfect,
    and it is the factor by which the busiest replica is over-subscribed.
    """
    by_id = {r.id: r for r in requests}
    counts = np.zeros(num_replicas)
    tokens = np.zeros(num_replicas)
    prefill = np.zeros(num_replicas)
    for rid, k in assignment.items():
        req = by_id.get(rid)
        if req is None:
            continue
        counts[k] += 1
        tokens[k] += req.total_tokens
        prefill[k] += req.num_prefill_tokens

    def spread(x: np.ndarray) -> Dict[str, float]:
        mean = float(x.mean()) if x.size else 0.0
        return {
            "min": float(x.min()) if x.size else 0.0,
            "max": float(x.max()) if x.size else 0.0,
            "mean": mean,
            "max_over_mean": (float(x.max()) / mean) if mean > 0 else 0.0,
        }

    return {
        "num_replicas": num_replicas,
        "requests": spread(counts),
        "tokens": spread(tokens),
        "prefill_tokens": spread(prefill),
        "per_replica_requests": counts.astype(int).tolist(),
        "per_replica_tokens": tokens.astype(int).tolist(),
    }
