"""
Tests for the router itself -- no engine, no predictor, no shapes.

A router is a pure function of (arrival order, replica load), so it can be
tested against hand-written loads.  What is worth pinning here is not that the
arithmetic works but the three behaviours that will otherwise be mistaken for
simulator bugs later:

  * Vidur's `lor` counts the *queue only*, so on an unsaturated fleet every
    replica ties at zero and the tie rule becomes the policy
  * `random` is uniform but seeded from its **own** stream, so switching
    policies cannot shift the arrival times
  * `round_robin` is exact on counts and can still be badly skewed on tokens
"""

import numpy as np
import pytest

from dynshape.entities import SimRequest, reset_ids
from dynshape.routing import (LORRouter, RandomRouter, ReplicaLoad,
                              ROUTING_POLICIES, RoundRobinRouter, assign_static,
                              build_router, routing_balance)


def loads(*specs):
    """`loads((queued, running, tokens, kv), ...)` -> a list of ReplicaLoad."""
    return [ReplicaLoad(replica_id=k, queued_requests=q, running_requests=r,
                        pending_tokens=t, kv_utilisation=u)
            for k, (q, r, t, u) in enumerate(specs)]


def req(arrived=0.0, prefill=100, decode=10):
    return SimRequest(arrived_at=arrived, num_prefill_tokens=prefill,
                      num_decode_tokens=decode)


# -- round robin ------------------------------------------------------------


def test_round_robin_takes_turns_in_order():
    r = RoundRobinRouter(3)
    empty = loads((0, 0, 0, 0.0), (0, 0, 0, 0.0), (0, 0, 0, 0.0))
    assert [r.route(req(), empty) for _ in range(7)] == [0, 1, 2, 0, 1, 2, 0]


def test_round_robin_ignores_load_completely():
    """It is not that round robin balances badly -- it cannot see load at all."""
    r = RoundRobinRouter(2)
    swamped = loads((999, 999, 10 ** 6, 0.99), (0, 0, 0, 0.0))
    assert r.route(req(), swamped) == 0        # straight into the busy one


def test_reset_restarts_the_rotation():
    r = RoundRobinRouter(3)
    empty = loads(*[(0, 0, 0, 0.0)] * 3)
    [r.route(req(), empty) for _ in range(4)]
    r.reset()
    assert r.route(req(), empty) == 0


# -- random -----------------------------------------------------------------


def test_random_is_uniform_over_replicas():
    r = RandomRouter(4, seed=0)
    empty = loads(*[(0, 0, 0, 0.0)] * 4)
    picks = np.array([r.route(req(), empty) for _ in range(4000)])
    counts = np.bincount(picks, minlength=4)
    assert set(np.unique(picks)) == {0, 1, 2, 3}
    # 4000 draws over 4 bins: 1000 +- ~100 at 3 sigma.
    assert counts.min() > 800 and counts.max() < 1200


def test_random_is_reproducible_and_reset_restores_it():
    empty = loads(*[(0, 0, 0, 0.0)] * 3)
    a = RandomRouter(3, seed=7)
    first = [a.route(req(), empty) for _ in range(20)]
    a.reset()
    assert [a.route(req(), empty) for _ in range(20)] == first
    b = RandomRouter(3, seed=7)
    assert [b.route(req(), empty) for _ in range(20)] == first


def test_random_draws_from_its_own_stream_not_the_global_one():
    """The seeding fix, as a test.

    Vidur's router draws from the global RNG that its arrivals also use, so
    *choosing a router* perturbs the traffic.  Here routing consumes nothing
    global, so a numpy global draw is unaffected by how much routing happened.
    """
    np.random.seed(1234)
    before = np.random.random()

    np.random.seed(1234)
    r = RandomRouter(8, seed=99)
    empty = loads(*[(0, 0, 0, 0.0)] * 8)
    for _ in range(500):
        r.route(req(), empty)
    after = np.random.random()

    assert before == after


# -- LOR --------------------------------------------------------------------


def test_lor_picks_the_emptiest_queue():
    r = build_router("lor", 3)
    assert r.route(req(), loads((5, 0, 0, 0.0), (1, 0, 0, 0.0), (9, 0, 0, 0.0))) == 1


def test_vidur_lor_collapses_when_queues_are_empty():
    """**Vidur's LOR is a hot spot below saturation, and this pins it.**

    `num_pending_requests` is `len(self._request_queue)` -- the *waiting* queue.
    A replica mid-decode with a hundred admitted requests reports zero.  So on
    any fleet whose queues drain each iteration every replica ties at zero,
    `min()` returns the first key, and every request goes to replica 0.

    Not a bug in this port -- it is what the source does, and it is the reason
    `lor_requests` exists beside it.  Written down because otherwise it looks
    like a routing bug the first time a load sweep produces [196, 4, 0, 0].
    """
    busy_but_unqueued = loads((0, 100, 50_000, 0.9),
                              (0, 0, 0, 0.0),
                              (0, 0, 0, 0.0))
    vidur = build_router("lor", 3)
    assert vidur.route(req(), busy_but_unqueued) == 0        # into the busy one

    # Counting running requests too is what most people mean by LOR, and it
    # sends the request somewhere sane.
    ours = build_router("lor_requests", 3)
    assert ours.route(req(), busy_but_unqueued) == 1


def test_lor_metrics_disagree_and_that_is_the_point():
    """Same fleet, three metrics, three different answers.

    Replica 0: many small requests.  Replica 1: one enormous prefill.
    Replica 2: moderate count, KV nearly full.
    """
    state = loads((6, 0, 600, 0.10),        # 6 tiny requests
                  (1, 0, 8000, 0.35),       # 1 giant
                  (3, 0, 3000, 0.95))       # KV nearly gone
    assert build_router("lor", 3).route(req(), state) == 1          # fewest queued
    assert build_router("lor_tokens", 3).route(req(), state) == 0   # least work
    assert build_router("lor_kv", 3).route(req(), state) == 0       # most room


def test_lor_ties_go_to_the_lowest_replica_id():
    """Matches `min()` over Vidur's dict, whose insertion order is replica order."""
    r = build_router("lor", 4)
    assert r.route(req(), loads(*[(2, 0, 0, 0.0)] * 4)) == 0


def test_lor_rejects_a_mismatched_fleet_size():
    r = build_router("lor", 4)
    with pytest.raises(ValueError, match="built for 4 replicas"):
        r.route(req(), loads(*[(0, 0, 0, 0.0)] * 3))


def test_lor_rejects_an_unknown_metric():
    with pytest.raises(ValueError, match="unknown LOR metric"):
        LORRouter(2, metric="vibes")


# -- the registry -----------------------------------------------------------


def test_every_advertised_policy_builds():
    for name in ROUTING_POLICIES:
        router = build_router(name, 3, seed=0)
        assert router.name == name
        assert router.route(req(), loads(*[(0, 0, 0, 0.0)] * 3)) in (0, 1, 2)


def test_only_lor_is_reactive():
    """The property that decides whether an assignment can be precomputed."""
    assert not build_router("round_robin", 2).reactive
    assert not build_router("random", 2).reactive
    assert build_router("lor", 2).reactive


def test_unknown_policy_names_are_refused():
    with pytest.raises(ValueError, match="unknown routing policy"):
        build_router("least_loaded_probably", 2)


# -- static assignment and balance ------------------------------------------


def test_static_assignment_matches_a_live_run_for_round_robin():
    reset_ids()
    reqs = [req(arrived=0.1 * i) for i in range(9)]
    a = assign_static(reqs, RoundRobinRouter(3), 3)
    assert [a[r.id] for r in reqs] == [0, 1, 2, 0, 1, 2, 0, 1, 2]


def test_static_assignment_refuses_lor():
    reset_ids()
    reqs = [req(arrived=0.1 * i) for i in range(4)]
    with pytest.raises(ValueError, match="reads replica load"):
        assign_static(reqs, build_router("lor", 2), 2)


def test_round_robin_is_exact_on_counts_and_can_be_skewed_on_tokens():
    """Counts are not load -- the caution the comparison document opens with."""
    reset_ids()
    # Two giants land three apart, so round robin puts both on replica 0.
    sizes = [4000, 50, 50, 4000, 50, 50]
    reqs = [req(arrived=0.01 * i, prefill=n) for i, n in enumerate(sizes)]
    a = assign_static(reqs, RoundRobinRouter(3), 3)
    bal = routing_balance(a, reqs, 3)

    assert bal["requests"]["max_over_mean"] == pytest.approx(1.0)   # perfect
    assert bal["tokens"]["max_over_mean"] > 2.0                     # not perfect
    assert bal["per_replica_requests"] == [2, 2, 2]


def test_balance_ignores_requests_it_was_not_given():
    reset_ids()
    reqs = [req(arrived=0.0), req(arrived=1.0)]
    bal = routing_balance({reqs[0].id: 0, 999_999: 1}, reqs, 2)
    assert bal["per_replica_requests"] == [1, 0]
