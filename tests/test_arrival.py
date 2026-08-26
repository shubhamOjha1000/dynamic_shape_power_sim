"""L0, half one: the arrival generators.

The properties worth pinning are distributional, so they are asserted as
tolerances on large samples rather than on exact values -- except for
reproducibility, which must be exact.
"""

import math

import numpy as np
import pytest

from dynshape.arrival import (GammaInterval, PoissonInterval, StaticInterval,
                              TraceInterval, piecewise_poisson)


def draw(gen, n):
    return np.array([gen.next_interval() for _ in range(n)])


# -- static -----------------------------------------------------------------

def test_static_without_qps_reproduces_vidur():
    """Vidur returns 0: every request arrives at once.  Degenerate but useful --
    it removes the arrival process entirely and isolates the scheduler."""
    g = StaticInterval()
    assert all(g.next_interval() == 0.0 for _ in range(5))


def test_static_with_qps_is_a_metronome():
    g = StaticInterval(qps=4.0)
    assert draw(g, 5).tolist() == [0.25] * 5


def test_static_rejects_nonpositive_qps():
    with pytest.raises(ValueError):
        StaticInterval(qps=0)


# -- poisson ----------------------------------------------------------------

def test_poisson_mean_matches_qps():
    g = PoissonInterval(qps=5.0, seed=0, clip_sigmas=None)
    x = draw(g, 20000)
    assert x.mean() == pytest.approx(1 / 5.0, rel=0.05)


def test_poisson_is_exponential_not_uniform():
    """An exponential has cv == 1; a uniform on the same range has cv ~ 0.58.
    This is the property that makes some requests collide and others leave long
    gaps, which is what moves peak power."""
    x = draw(PoissonInterval(qps=2.0, seed=1, clip_sigmas=None), 20000)
    assert x.std() / x.mean() == pytest.approx(1.0, rel=0.05)


def test_poisson_clip_is_faithful_to_vidur_and_truncates_the_tail():
    """Vidur caps intervals at 3/qps.  Not an implementation detail: it removes
    the long quiet gaps, so the stream is slightly MORE regular than Poisson."""
    qps = 2.0
    g = PoissonInterval(qps=qps, seed=2)
    x = draw(g, 5000)
    assert x.max() <= 3.0 / qps + 1e-12
    # An unclipped draw from the same seed exceeds the cap, so the clip bites.
    assert draw(PoissonInterval(qps=qps, seed=2, clip_sigmas=None), 5000).max() > 3.0 / qps


def test_poisson_is_reproducible_and_seed_sensitive():
    a = draw(PoissonInterval(qps=3.0, seed=7), 50)
    b = draw(PoissonInterval(qps=3.0, seed=7), 50)
    c = draw(PoissonInterval(qps=3.0, seed=8), 50)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


# -- gamma ------------------------------------------------------------------

def test_gamma_at_cv_one_is_poisson():
    x = draw(GammaInterval(qps=2.0, cv=1.0, seed=3), 20000)
    assert x.mean() == pytest.approx(0.5, rel=0.05)
    assert x.std() / x.mean() == pytest.approx(1.0, rel=0.06)


@pytest.mark.parametrize("cv", [0.5, 1.0, 2.0])
def test_gamma_dial_moves_variability_but_not_the_rate(cv):
    """The whole point of the dial: it changes the SHAPE of the traffic, never
    its rate.  Same requests per second, very different peak."""
    x = draw(GammaInterval(qps=4.0, cv=cv, seed=4), 30000)
    assert x.mean() == pytest.approx(0.25, rel=0.06)
    assert x.std() / x.mean() == pytest.approx(cv, rel=0.08)


def test_gamma_shape_matches_vidurs_formula():
    g = GammaInterval(qps=10.0, cv=0.25)
    assert g.shape == pytest.approx(1 / 0.25 ** 2)
    assert g.scale == pytest.approx(1 / (10.0 * g.shape))


def test_burstier_traffic_clumps_more():
    """The lever behind peak power: identical rate, different clumping."""
    def max_in_window(cv, window=1.0, n=20000):
        gaps = draw(GammaInterval(qps=10.0, cv=cv, seed=5), n)
        times = np.cumsum(gaps)
        counts, _ = np.histogram(times, bins=np.arange(0, times[-1], window))
        return counts.max()

    assert max_in_window(3.0) > max_in_window(0.5)


# -- trace ------------------------------------------------------------------

def test_trace_replays_gaps_and_then_stops():
    g = TraceInterval([0.0, 1.0, 1.5, 4.0])
    assert g.next_interval() == pytest.approx(1.0)
    assert g.next_interval() == pytest.approx(0.5)
    assert g.next_interval() == pytest.approx(2.5)
    assert g.next_interval() is None


def test_time_scale_factor_is_the_load_axis():
    """0.5 replays the same curve in half the wall clock -- double the load
    without changing its shape."""
    slow = draw(TraceInterval([0, 1, 2, 4], time_scale_factor=1.0), 3)
    fast = draw(TraceInterval([0, 1, 2, 4], time_scale_factor=0.5), 3)
    assert np.allclose(fast, slow / 2)


def test_trace_sorts_and_rebases():
    g = TraceInterval([10.0, 12.0, 11.0])
    assert g.times[0] == 0.0
    assert list(g.times) == [0.0, 1.0, 2.0]


def test_empty_trace_is_rejected():
    with pytest.raises(ValueError):
        TraceInterval([])


# -- synthetic lambda(t) ----------------------------------------------------

def test_piecewise_poisson_produces_a_time_varying_rate():
    """The thing Vidur has no generator for: a rate that MOVES.  Constant-rate
    fluctuations cancel as sqrt(N) across a fleet; a real lambda(t) does not."""
    quiet, busy = 0.5, 20.0
    gen = piecewise_poisson([quiet, busy], segment_s=100.0, duration_s=200.0, seed=0)
    times = gen.times
    n_quiet = int((times < 100).sum())
    n_busy = int((times >= 100).sum())
    assert n_busy > 5 * n_quiet


def test_piecewise_poisson_rejects_an_impossible_request():
    with pytest.raises(ValueError):
        piecewise_poisson([0.0], segment_s=1.0, duration_s=1.0)
