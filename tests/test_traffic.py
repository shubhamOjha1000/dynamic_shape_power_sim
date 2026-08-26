"""L0 assembled, and the seeding discipline that makes it attributable."""

import numpy as np
import pytest

from dynshape.arrival import TraceInterval
from dynshape.entities import reset_ids
from dynshape.lengths import TraceLength
from dynshape.traffic import (TrafficConfig, generate_traffic, spawn_seeds,
                              traffic_summary)


def test_generates_the_requested_count_in_arrival_order():
    reqs = generate_traffic(TrafficConfig(num_requests=50, seed=0))
    assert len(reqs) == 50
    assert all(a.arrived_at <= b.arrived_at for a, b in zip(reqs, reqs[1:]))
    assert all(r.num_prefill_tokens >= 1 and r.num_decode_tokens >= 1 for r in reqs)


def test_duration_bound_stops_the_stream():
    reqs = generate_traffic(TrafficConfig(num_requests=None, duration_s=5.0,
                                          interval="poisson", qps=10.0, seed=1))
    assert all(r.arrived_at < 5.0 for r in reqs)
    assert len(reqs) > 10


def test_same_seed_is_byte_identical():
    a = generate_traffic(TrafficConfig(num_requests=30, seed=3))
    b = generate_traffic(TrafficConfig(num_requests=30, seed=3))
    assert [(r.arrived_at, r.num_prefill_tokens, r.num_decode_tokens) for r in a] \
        == [(r.arrived_at, r.num_prefill_tokens, r.num_decode_tokens) for r in b]


def test_different_seed_gives_different_traffic():
    a = generate_traffic(TrafficConfig(num_requests=30, seed=3))
    b = generate_traffic(TrafficConfig(num_requests=30, seed=4))
    assert [r.arrived_at for r in a] != [r.arrived_at for r in b]


def test_child_streams_are_independent_which_is_the_whole_point():
    """Vidur's shared global seed makes this impossible.

    Changing only the LENGTH generator must leave arrival times untouched --
    otherwise an experiment meant to compare two length distributions on
    identical traffic silently compares them on different traffic.
    """
    base = TrafficConfig(num_requests=40, seed=9, length="zipf")
    other = TrafficConfig(num_requests=40, seed=9, length="uniform")
    a = generate_traffic(base)
    b = generate_traffic(other)
    assert [r.arrived_at for r in a] == [r.arrived_at for r in b]
    assert [r.num_prefill_tokens for r in a] != [r.num_prefill_tokens for r in b]


def test_changing_the_arrival_generator_leaves_lengths_untouched():
    a = generate_traffic(TrafficConfig(num_requests=40, seed=9, interval="poisson"))
    b = generate_traffic(TrafficConfig(num_requests=40, seed=9, interval="gamma", cv=2.5))
    assert [r.num_prefill_tokens for r in a] == [r.num_prefill_tokens for r in b]
    assert [r.arrived_at for r in a] != [r.arrived_at for r in b]


def test_spawn_seeds_is_deterministic_and_distinct():
    s = spawn_seeds(42)
    assert s == spawn_seeds(42)
    assert len(set(s)) == len(s)
    assert s != spawn_seeds(43)


def test_pd_ratio_moves_the_prefill_share():
    """One number decides compute-bound vs memory-bound, and therefore roughly
    a factor of two in power."""
    story = traffic_summary(generate_traffic(TrafficConfig(
        num_requests=300, seed=2, prefill_to_decode_ratio=0.5)))
    summarise = traffic_summary(generate_traffic(TrafficConfig(
        num_requests=300, seed=2, prefill_to_decode_ratio=19.0)))
    assert story["pd_ratio_median"] < 1.0
    assert summarise["pd_ratio_median"] > 10.0
    assert summarise["total_prefill_tokens"] > 3 * story["total_prefill_tokens"]


def test_prefix_cache_reduces_executed_prefill_work():
    """The 21x error class: without prefix caching the highest-power phase is
    overstated on every single request."""
    off = generate_traffic(TrafficConfig(num_requests=200, seed=5,
                                         prefix_cache_fraction=0.0))
    on = generate_traffic(TrafficConfig(num_requests=200, seed=5,
                                        prefix_cache_fraction=0.7))
    w_off = sum(r.num_prefill_tokens for r in off)
    w_on = sum(r.num_prefill_tokens for r in on)
    assert w_on < w_off
    assert all(r.cached_prefix_tokens == 0 for r in off)
    assert sum(r.cached_prefix_tokens for r in on) > 0
    # Every request still has at least one token of real work.
    assert all(r.num_prefill_tokens >= 1 for r in on)


def test_prebuilt_trace_generators_compose():
    reset_ids()
    cfg = TrafficConfig(
        num_requests=None, duration_s=None,
        interval_generator=TraceInterval([0.0, 1.0, 2.5, 3.0]),
        length_generator=TraceLength([100, 200, 300], [10, 20, 30], shuffle=False),
    )
    reqs = generate_traffic(cfg)
    assert [r.num_prefill_tokens for r in reqs] == [100, 200, 300]
    assert [r.arrived_at for r in reqs] == pytest.approx([1.0, 2.5, 3.0])


def test_unbounded_synthetic_stream_is_rejected():
    with pytest.raises(ValueError):
        TrafficConfig(num_requests=None, duration_s=None, interval="poisson")


def test_trace_names_without_a_generator_fail_loudly():
    with pytest.raises(ValueError, match="prebuilt"):
        generate_traffic(TrafficConfig(interval="trace", num_requests=5))
    with pytest.raises(ValueError, match="prebuilt"):
        generate_traffic(TrafficConfig(length="trace", num_requests=5))


def test_summary_reports_the_realised_rate():
    reqs = generate_traffic(TrafficConfig(num_requests=500, interval="poisson",
                                          qps=8.0, seed=6))
    s = traffic_summary(reqs)
    assert s["requests"] == 500
    assert s["arrival_rate_qps"] == pytest.approx(8.0, rel=0.2)
