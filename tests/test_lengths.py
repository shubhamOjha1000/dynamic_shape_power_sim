"""L0, half two: the length generators.

The P:D ratio is the knob that moves power most -- it decides whether the
workload is compute-bound prefill or memory-bound decode -- so most of these
tests are about the split rather than the draw.
"""

import numpy as np
import pytest

from dynshape.lengths import (FixedLength, TraceLength, UniformLength,
                              ZipfGenerator, ZipfLength, _split)


# -- the shared split -------------------------------------------------------

@pytest.mark.parametrize("ratio,expect_prefill_share", [
    (0.5, 1 / 3),      # "write me a story"  -- decode-heavy
    (1.0, 1 / 2),      # balanced
    (4.0, 4 / 5),      # ordinary chat
    (19.0, 19 / 20),   # "summarise this document" -- prefill-heavy
])
def test_pd_ratio_splits_the_total_as_documented(ratio, expect_prefill_share):
    p, d = _split(1000.0, ratio)
    assert p + d == pytest.approx(1000, abs=2)
    assert p / (p + d) == pytest.approx(expect_prefill_share, abs=0.01)


def test_split_never_returns_an_unrunnable_half():
    """A request with zero prefill or zero decode tokens is not a request."""
    for total in (1, 2, 3, 5):
        for ratio in (0.1, 1.0, 50.0):
            p, d = _split(float(total), ratio)
            assert p >= 1 and d >= 1


def test_split_rejects_a_nonpositive_ratio():
    with pytest.raises(ValueError):
        _split(100.0, 0.0)


# -- fixed ------------------------------------------------------------------

def test_fixed_is_identical_every_time():
    g = FixedLength(512, 128)
    assert [g.next_lengths() for _ in range(3)] == [(512, 128)] * 3


def test_fixed_rejects_empty_halves():
    with pytest.raises(ValueError):
        FixedLength(0, 128)


# -- uniform ----------------------------------------------------------------

def test_uniform_covers_the_range_evenly():
    g = UniformLength(100, 1000, prefill_to_decode_ratio=4.0, seed=0)
    totals = np.array([sum(g.next_lengths()) for _ in range(5000)])
    assert totals.min() < 150 and totals.max() > 950
    # Evenly spread: the median sits near the midpoint, unlike zipf.
    assert np.median(totals) == pytest.approx(550, rel=0.1)


def test_uniform_over_represents_large_requests_relative_to_zipf():
    """The documented caution, made checkable: uniform produces far too many
    large requests, inflating both KV pressure and the prefill share."""
    u = UniformLength(64, 4096, seed=1)
    z = ZipfLength(64, 4096, theta=0.9, seed=1)
    u_big = np.mean([sum(u.next_lengths()) > 2048 for _ in range(3000)])
    z_big = np.mean([sum(z.next_lengths()) > 2048 for _ in range(3000)])
    assert u_big > 3 * z_big


# -- zipf -------------------------------------------------------------------

def test_zipf_stays_inside_its_range():
    g = ZipfGenerator(10, 500, theta=0.8, seed=0)
    v = np.array([g.next() for _ in range(5000)])
    assert v.min() >= 10 and v.max() <= 500


def _many(g, n):
    return [g.next() for _ in range(n)]


def _many_lengths(g, n):
    return [g.next_lengths() for _ in range(n)]


def test_zipf_is_a_long_tail_not_a_bell():
    """A few huge, many tiny.  Those occasional giants are what fill the KV
    cache and trigger preemption, so the tail matters more than the mean."""
    v = np.array(_many(ZipfGenerator(1, 1000, theta=0.9, seed=2), 5000))
    assert np.median(v) < v.mean()
    assert np.percentile(v, 99) > 10 * np.median(v)


def test_higher_theta_is_a_heavier_tail():
    flat = np.array(_many(ZipfGenerator(1, 1000, theta=0.1, seed=3), 4000))
    steep = np.array(_many(ZipfGenerator(1, 1000, theta=0.95, seed=3), 4000))
    assert np.median(steep) < np.median(flat)


def test_zipf_scramble_breaks_rank_ordering_but_keeps_the_range():
    g = ZipfGenerator(1, 100, theta=0.7, scramble=True, seed=4)
    v = np.array(_many(g, 2000))
    assert v.min() >= 1 and v.max() <= 100


def test_zipf_theta_must_be_below_one():
    with pytest.raises(ValueError):
        ZipfGenerator(1, 100, theta=1.0)


def test_zipf_length_is_reproducible():
    a = _many_lengths(ZipfLength(64, 2048, seed=11), 20)
    b = _many_lengths(ZipfLength(64, 2048, seed=11), 20)
    assert a == b


# -- trace ------------------------------------------------------------------

def test_trace_replays_then_stops():
    g = TraceLength([100, 200], [10, 20], shuffle=False)
    assert g.next_lengths() == (100, 10)
    assert g.next_lengths() == (200, 20)
    assert g.next_lengths() == (None, None)


def test_trace_clipping_trims_both_sides_proportionally():
    """The careful bit: an over-long request keeps its P:D character instead of
    having one side lopped off."""
    g = TraceLength([800], [200], max_tokens=500, shuffle=False)
    p, d = g.next_lengths()
    assert p + d <= 500
    # Original ratio was 4:1; it survives the clip.
    assert p / d == pytest.approx(4.0, rel=0.15)


def test_trace_scale_factors_are_a_what_if_knob():
    """'What if people paste twice as much context?' is one number."""
    g = TraceLength([100], [50], prefill_scale_factor=2.0, max_tokens=10000,
                    shuffle=False)
    assert g.next_lengths() == (200, 50)


def test_trace_never_emits_an_empty_half_after_clipping():
    g = TraceLength([10000], [10000], max_tokens=4, shuffle=False)
    p, d = g.next_lengths()
    assert p >= 1 and d >= 1


def test_trace_shuffle_is_seeded():
    a = TraceLength(list(range(1, 51)), [5] * 50, seed=1).prefill.tolist()
    b = TraceLength(list(range(1, 51)), [5] * 50, seed=1).prefill.tolist()
    c = TraceLength(list(range(1, 51)), [5] * 50, seed=2).prefill.tolist()
    assert a == b and a != c


def test_trace_rejects_mismatched_columns():
    with pytest.raises(ValueError):
        TraceLength([1, 2, 3], [1, 2])


def test_pd_percentiles_are_reported():
    g = TraceLength([400] * 10, [100] * 10, max_tokens=10000)
    assert g.pd_ratio_percentiles["p50"] == pytest.approx(4.0)
