"""
Tests for the fleet loop.

Four things are worth pinning, and only one of them is about routing:

  1. **A one-replica fleet is the single-GPU engine.**  Iteration for iteration,
     joule for joule.  If that ever drifts, `execute_iteration` has been forked
     and every fleet number becomes incomparable to every engine number.
  2. **Energy is conserved on the shared grid.**  The facility series must
     integrate to `total_energy_j`, tail padding included -- otherwise a
     coincidence factor is computed from a series that is not the trace.
  3. **The idle floor is charged.**  A replica that finished early is idle, not
     absent.  Dropping those watts is how a fleet report understates the floor.
  4. Routing changes *where* work goes without changing *what* work there is.
"""

import numpy as np
import pytest

from dynshape import (EngineConfig, FleetConfig, SchedulerConfig, ShapeRewriter,
                      TrafficConfig, build_predictor, clone_requests,
                      compare_routing, generate_traffic, run_engine, run_fleet)
from dynshape.entities import SimRequest, reset_ids


@pytest.fixture(scope="module")
def engine_config():
    # Small pool and short prompts: fast, and crowded enough that the scheduler
    # has real decisions to make.
    return EngineConfig(scheduler=SchedulerConfig(max_tokens=1024),
                        record_kernels_until_ms=0.0)


@pytest.fixture(scope="module")
def traffic():
    return generate_traffic(TrafficConfig(interval="poisson", qps=40, length="zipf",
                                          num_requests=40, max_tokens=1024, seed=3))


# -- 1. one replica is the engine ------------------------------------------


def test_a_one_replica_fleet_reproduces_run_engine_exactly(
        rewriter, predictor, engine_config, traffic):
    """The seam that keeps fleet numbers comparable with single-GPU numbers.

    With one replica there is nothing to route, so any difference is a
    difference in how an iteration was costed -- which is exactly what sharing
    `execute_iteration` is supposed to make impossible.
    """
    solo = run_engine(clone_requests(traffic), rewriter, predictor, engine_config)
    fleet = run_fleet(clone_requests(traffic), rewriter, predictor,
                      FleetConfig(num_replicas=1, routing="round_robin",
                                  engine=engine_config))
    one = fleet.replicas[0]

    assert len(one.iterations) == len(solo.iterations)
    assert one.total_energy_j == pytest.approx(solo.total_energy_j, rel=1e-12)
    assert one.total_time_ms == pytest.approx(solo.total_time_ms, rel=1e-12)
    for a, b in zip(one.iterations, solo.iterations):
        assert a.total_tokens == b.total_tokens
        assert a.decode_batch == b.decode_batch
        assert a.energy_j == pytest.approx(b.energy_j, rel=1e-12)
        assert a.t_start_ms == pytest.approx(b.t_start_ms, rel=1e-12)


def test_every_routing_policy_sends_every_request_somewhere(
        rewriter, predictor, engine_config, traffic):
    for policy in ("round_robin", "random", "lor", "lor_requests", "lor_tokens",
                   "lor_kv"):
        trace = run_fleet(clone_requests(traffic), rewriter, predictor,
                          FleetConfig(num_replicas=3, routing=policy,
                                      engine=engine_config))
        assert len(trace.assignment) == len(traffic)
        assert set(trace.assignment.values()) <= {0, 1, 2}
        assert all(r.completed for r in trace.requests), policy
        # Every request appears in exactly one replica's trace.
        seen = [r.id for rep in trace.replicas for r in rep.requests]
        assert sorted(seen) == sorted(r.id for r in trace.requests)


# -- 2. energy conservation on the shared grid ------------------------------


@pytest.mark.parametrize("dt_ms", [10.0, 50.0, 250.0, 1000.0])
def test_the_facility_series_integrates_to_the_fleet_energy(
        rewriter, predictor, engine_config, traffic, dt_ms):
    """A coincidence factor read off a series that loses energy is meaningless."""
    trace = run_fleet(clone_requests(traffic), rewriter, predictor,
                      FleetConfig(num_replicas=3, routing="round_robin",
                                  engine=engine_config))
    t, per_replica, total = trace.resample(dt_ms=dt_ms)
    integral = float(total.sum() * dt_ms / 1000.0)
    # The last bin is partial, so allow it: dt x the fleet's idle floor.
    slack = dt_ms / 1000.0 * trace.idle_w * trace.num_replicas
    assert integral == pytest.approx(trace.total_energy_j, abs=slack + 1e-6)


def test_the_facility_series_is_the_sum_of_the_replica_series(
        rewriter, predictor, engine_config, traffic):
    trace = run_fleet(clone_requests(traffic), rewriter, predictor,
                      FleetConfig(num_replicas=4, routing="random",
                                  engine=engine_config))
    _, per_replica, total = trace.resample(dt_ms=100.0)
    assert per_replica.shape[0] == 4
    np.testing.assert_allclose(per_replica.sum(axis=0), total, rtol=1e-12)


def test_padding_a_short_replica_holds_it_at_idle_not_at_zero(
        rewriter, predictor, engine_config):
    """A finished GPU is a powered-on GPU.

    Two requests, one enormous and one tiny, on two replicas under round robin:
    replica 1 finishes long before replica 0 and must keep drawing its floor for
    the rest of the run.
    """
    reset_ids()
    reqs = [SimRequest(arrived_at=0.0, num_prefill_tokens=900, num_decode_tokens=400),
            SimRequest(arrived_at=0.0, num_prefill_tokens=64, num_decode_tokens=2)]
    trace = run_fleet(reqs, rewriter, predictor,
                      FleetConfig(num_replicas=2, routing="round_robin",
                                  engine=engine_config))
    short, long = sorted(trace.replicas, key=lambda r: r.total_time_ms)
    assert short.total_time_ms < long.total_time_ms      # it really did finish first

    _, per_replica, _ = trace.resample(dt_ms=5.0)
    k = trace.replicas.index(short)
    # Its last bins are the idle floor, not zero.
    assert per_replica[k][-1] == pytest.approx(trace.idle_w, rel=1e-6)
    assert trace.tail_idle_energy_j > 0


def test_tail_idle_energy_is_included_in_the_fleet_total(
        rewriter, predictor, engine_config, traffic):
    trace = run_fleet(clone_requests(traffic), rewriter, predictor,
                      FleetConfig(num_replicas=4, routing="lor",
                                  engine=engine_config))
    naive = sum(r.total_energy_j for r in trace.replicas)
    assert trace.total_energy_j == pytest.approx(naive + trace.tail_idle_energy_j)
    assert trace.total_energy_j >= naive


# -- 3. routing moves work without changing it ------------------------------


def test_the_same_traffic_does_the_same_work_under_every_router(
        rewriter, predictor, engine_config, traffic):
    """Routing decides *where*, never *what*.

    Total tokens generated must be identical across policies -- the requests are
    the same requests.  Energy is *not* expected to match: batch sizes differ, so
    the fused kernels amortise differently, which is the burstiness result
    arriving by another route.
    """
    totals = {}
    for policy in ("round_robin", "random", "lor_tokens"):
        trace = run_fleet(clone_requests(traffic), rewriter, predictor,
                          FleetConfig(num_replicas=3, routing=policy,
                                      engine=engine_config))
        totals[policy] = sum(r.num_generated_tokens for r in trace.requests)
    assert len(set(totals.values())) == 1, totals


def test_round_robin_splits_counts_exactly(rewriter, predictor, engine_config,
                                           traffic):
    trace = run_fleet(clone_requests(traffic), rewriter, predictor,
                      FleetConfig(num_replicas=4, routing="round_robin",
                                  engine=engine_config))
    counts = trace.balance()["per_replica_requests"]
    assert max(counts) - min(counts) <= 1


def test_vidur_lor_concentrates_on_replica_zero_below_saturation(
        rewriter, predictor, engine_config, traffic):
    """The queue-metric collapse, end to end rather than in isolation.

    `test_routing.py` pins the rule; this pins that it really does produce a hot
    spot in a run, so a lopsided split is recognised as the policy rather than
    investigated as a bug.
    """
    lor = run_fleet(clone_requests(traffic), rewriter, predictor,
                    FleetConfig(num_replicas=4, routing="lor",
                                engine=engine_config))
    fixed = run_fleet(clone_requests(traffic), rewriter, predictor,
                      FleetConfig(num_replicas=4, routing="lor_requests",
                                  engine=engine_config))
    assert lor.balance()["requests"]["max_over_mean"] > \
        fixed.balance()["requests"]["max_over_mean"]


def test_a_reactive_router_actually_reads_the_load(
        rewriter, predictor, engine_config):
    """Two requests at the same instant must not both go to replica 0.

    The first is routed and *received* before the second is routed, so a
    load-reading router sees a non-empty replica 0.  A router that only saw
    stale state -- or state sampled once per iteration -- would send both.
    """
    reset_ids()
    reqs = [SimRequest(arrived_at=0.0, num_prefill_tokens=512, num_decode_tokens=64),
            SimRequest(arrived_at=0.0, num_prefill_tokens=512, num_decode_tokens=64)]
    trace = run_fleet(reqs, rewriter, predictor,
                      FleetConfig(num_replicas=2, routing="lor",
                                  engine=engine_config))
    assert sorted(trace.assignment.values()) == [0, 1]


# -- 4. the guards ----------------------------------------------------------


def test_reusing_request_objects_is_refused_loudly(rewriter, predictor,
                                                   engine_config, traffic):
    """Silently continuing a finished run is the worst possible failure mode."""
    used = clone_requests(traffic)
    run_fleet(used, rewriter, predictor,
              FleetConfig(num_replicas=2, engine=engine_config))
    with pytest.raises(ValueError, match="already been run"):
        run_fleet(used, rewriter, predictor,
                  FleetConfig(num_replicas=2, engine=engine_config))


def test_clone_requests_gives_independent_unrun_copies(traffic):
    a = clone_requests(traffic)
    assert [r.id for r in a] == [r.id for r in traffic]
    assert [r.arrived_at for r in a] == [r.arrived_at for r in traffic]
    assert all(not r.scheduled and r.num_processed_tokens == 0 for r in a)
    a[0].on_batch_end(1.0, 1)
    assert traffic[0].num_processed_tokens == 0     # really independent


def test_an_unknown_policy_is_refused_at_config_time():
    with pytest.raises(ValueError, match="unknown routing policy"):
        FleetConfig(routing="whatever")


def test_zero_replicas_is_refused():
    with pytest.raises(ValueError, match="num_replicas must be >= 1"):
        FleetConfig(num_replicas=0)


# -- reporting --------------------------------------------------------------


def test_coincidence_is_bounded_and_the_floor_is_reported(
        rewriter, predictor, engine_config, traffic):
    trace = run_fleet(clone_requests(traffic), rewriter, predictor,
                      FleetConfig(num_replicas=4, routing="round_robin",
                                  engine=engine_config))
    c = trace.coincidence(dt_ms=250.0)
    assert 0.0 < c["coincidence_factor"] <= 1.0 + 1e-9
    assert 0.0 <= c["dynamic_coincidence_factor"] <= 1.0 + 1e-9
    assert c["facility_floor_w"] == pytest.approx(4 * trace.idle_w)
    assert c["facility_peak_w"] >= c["facility_mean_w"] >= c["facility_floor_w"]
    # The dynamic factor is the one that is not swamped by the floor.
    assert c["dynamic_coincidence_factor"] <= c["coincidence_factor"] + 1e-9


def test_energy_is_invariant_across_apertures_and_the_peak_is_not(
        rewriter, predictor, engine_config, traffic):
    """The §7 result, at fleet scale: a peak is not a number without an aperture."""
    trace = run_fleet(clone_requests(traffic), rewriter, predictor,
                      FleetConfig(num_replicas=3, routing="round_robin",
                                  engine=engine_config))
    rows = trace.aperture_table((10.0, 100.0, 1000.0))
    energies = [r["energy_j"] for r in rows]
    peaks = [r["facility_peak_w"] for r in rows]
    assert max(energies) == pytest.approx(min(energies), rel=1e-9)
    assert peaks[0] >= peaks[-1]


def test_compare_routing_runs_each_policy_on_identical_traffic(
        rewriter, predictor, engine_config, traffic):
    traces, table = compare_routing(
        traffic, rewriter, predictor,
        policies=("round_robin", "random", "lor"),
        config=FleetConfig(num_replicas=3, engine=engine_config))

    assert list(table["routing"]) == ["round_robin", "random", "lor"]
    assert set(traces) == {"round_robin", "random", "lor"}
    # Identical traffic: same requests, same arrivals, same prompt sizes.
    for name, tr in traces.items():
        assert [r.arrived_at for r in tr.requests] == \
               [r.arrived_at for r in traffic], name
    # And the caller's own list is untouched -- clones were used.
    assert all(not r.scheduled for r in traffic)
    assert "coincidence_factor" in table.columns


def test_per_replica_summary_accounts_for_every_request(
        rewriter, predictor, engine_config, traffic):
    trace = run_fleet(clone_requests(traffic), rewriter, predictor,
                      FleetConfig(num_replicas=4, routing="lor_tokens",
                                  engine=engine_config))
    rows = trace.per_replica_summary()
    assert sum(r["requests"] for r in rows) == len(traffic)
    assert sum(r["completed"] for r in rows) == len(traffic)
    assert sum(r["energy_j"] for r in rows) == pytest.approx(trace.total_energy_j)


def test_the_dataframes_join_up(rewriter, predictor, engine_config, traffic):
    trace = run_fleet(clone_requests(traffic), rewriter, predictor,
                      FleetConfig(num_replicas=3, routing="round_robin",
                                  engine=engine_config))
    power, per_replica, requests = trace.to_dataframes()
    assert list(power.columns) == ["t_ms", "facility_w", "replica_0_w",
                                   "replica_1_w", "replica_2_w"]
    assert len(per_replica) == 3
    assert len(requests) == len(traffic)
    assert set(requests["replica"]) <= {0, 1, 2}


def test_the_fleet_dashboard_renders(rewriter, predictor, engine_config, traffic):
    import matplotlib
    matplotlib.use("Agg")
    from dynshape.fleet_plot import plot_fleet_dashboard, plot_routing_comparison

    trace = run_fleet(clone_requests(traffic), rewriter, predictor,
                      FleetConfig(num_replicas=3, routing="round_robin",
                                  engine=engine_config))
    fig = plot_fleet_dashboard(trace)
    assert len(fig.axes) >= 4
    fig2 = plot_routing_comparison({"round_robin": trace})
    assert len(fig2.axes) == 2
