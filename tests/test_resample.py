"""Fixed-rate power reporting and the work vector -- what FSTS reports and the
event-based staircase could not.
"""

import numpy as np
import pytest

from dynshape.engine import EngineConfig, run_engine
from dynshape.entities import SimRequest, reset_ids
from dynshape.scheduler import SchedulerConfig
from dynshape.traffic import TrafficConfig, generate_traffic
from dynshape.work import WORK_FIELDS, gemm_work, kernel_work


def engine_cfg(**kw):
    return EngineConfig(
        scheduler=SchedulerConfig(chunk_size=512, max_num_seqs=16, max_tokens=4096),
        record_kernels_until_ms=0.0, **kw)


def traffic(n=12, qps=200.0):
    reset_ids()
    return generate_traffic(TrafficConfig(
        num_requests=n, interval='poisson', qps=qps, length='fixed',
        prefill_tokens=256, decode_tokens=12, seed=0))


# -- the work model ---------------------------------------------------------

def test_gemm_work_is_the_textbook_formula():
    q = {"batch": 2, "dimM": 64, "dimN": 128, "dimK": 32, "precM": "bf16"}
    flops, byts, weights = gemm_work(q)
    assert flops == 2 * 2 * 64 * 128 * 32
    assert byts == 2 * (64 * 32 + 32 * 128 + 64 * 128) * 2
    assert weights == 2 * 32 * 128 * 2          # the K x N operand alone


def test_weight_bytes_are_part_of_bytes_not_extra():
    q = {"batch": 1, "dimM": 8, "dimN": 16, "dimK": 4, "precM": "bf16"}
    _, byts, weights = gemm_work(q)
    assert 0 < weights < byts


def test_fusing_amortises_the_weight_read():
    """The whole reason fusing wins: the K x N operand is read once per launch
    whatever M is, so doubling the rows does not double the traffic."""
    small = {"batch": 1, "dimM": 1, "dimN": 2304, "dimK": 768, "precM": "bf16"}
    big = dict(small, dimM=64)
    _, b_small, w_small = gemm_work(small)
    _, b_big, w_big = gemm_work(big)
    assert w_small == w_big                     # weights unchanged
    assert b_big < 64 * b_small                 # traffic grows far less than 64x


def test_memory_ops_have_no_flops():
    for op, q in ((("elementwise",), {"dim": 1000, "prec": "bf16"}),
                  (("softmax",), {"batch": 8, "dim": 128, "prec": "bf16"}),
                  (("layernorm",), {"batch": 8, "dim": 768, "prec": "bf16"})):
        flops, byts, weights = kernel_work(q, op)
        assert flops == 0.0 and byts > 0 and weights == 0.0


def test_unknown_ops_raise_or_return_zero_by_request():
    with pytest.raises(NotImplementedError):
        kernel_work({}, ("something_new",))
    assert kernel_work({}, ("something_new",), strict=False) == (0.0, 0.0, 0.0)


def test_the_analytic_backend_uses_the_same_arithmetic(predictor):
    """One source of truth: if these drifted apart, the reported work vector and
    the reported watts would describe different kernels."""
    from dynshape.predictor import AnalyticBackend
    b = AnalyticBackend()
    q = {"batch": 1, "dimM": 512, "dimN": 2304, "dimK": 768,
         "precM": "bf16", "useTensorCore": True}
    flops, byts, _peak = b._work(q, ("gemm",))
    w_flops, w_bytes, _ = kernel_work(q, ("gemm",))
    assert (flops, byts) == (w_flops, w_bytes)


# -- the work vector on a real run ------------------------------------------

def test_every_iteration_carries_a_work_vector(rewriter, predictor):
    trace = run_engine(traffic(), rewriter, predictor, engine_cfg())
    for i in trace.iterations:
        for f in WORK_FIELDS:
            assert getattr(i, f) >= 0.0
        assert i.linear_flops > 0
        assert i.attn_flops > 0


def test_flops_split_by_phase_is_exact_not_apportioned(rewriter, predictor):
    """A fused GEMM's rows each belong to one request and FLOPs are linear in the
    row count, so prefill + decode must reconstruct the total exactly."""
    trace = run_engine(traffic(), rewriter, predictor, engine_cfg())
    for i in trace.iterations:
        total = i.linear_flops + i.attn_flops
        assert i.prefill_flops + i.decode_flops == pytest.approx(total, rel=1e-9)


def test_a_decode_only_iteration_has_no_prefill_flops(rewriter, predictor):
    trace = run_engine(traffic(qps=2.0), rewriter, predictor, engine_cfg())
    for i in trace.iterations:
        if i.prefill_chunks == 0:
            assert i.prefill_flops == 0.0


def test_arithmetic_intensity_separates_the_phases(rewriter, predictor):
    """Prefill is compute-bound and decode is memory-bound; FLOP/byte is the
    quantity that says so, and it needs no power model to compute."""
    trace = run_engine(traffic(n=20, qps=300.0), rewriter, predictor, engine_cfg())
    dec = [i.arithmetic_intensity for i in trace.iterations
           if i.decode_batch > 0 and i.prefill_chunks == 0]
    pre = [i.arithmetic_intensity for i in trace.iterations if i.prefill_chunks > 0]
    if not (dec and pre):
        pytest.skip("this traffic produced no contrast")
    assert np.mean(pre) > np.mean(dec)


def test_work_totals_sum_the_iterations(rewriter, predictor):
    trace = run_engine(traffic(), rewriter, predictor, engine_cfg())
    totals = trace.work_totals()
    assert set(totals) == set(WORK_FIELDS)
    assert totals["linear_flops"] == pytest.approx(
        sum(i.linear_flops for i in trace.iterations))


# -- resampling -------------------------------------------------------------

@pytest.mark.parametrize("dt", [0.1, 1.0, 5.0, 25.0])
def test_resampling_conserves_energy_at_any_rate(rewriter, predictor, dt):
    """The property that makes the resample trustworthy: it is a box filter, so
    the integral of the returned series is the trace's energy whatever the
    sample rate. If this drifts, the series is not the same trace."""
    trace = run_engine(traffic(qps=5.0), rewriter, predictor, engine_cfg())
    t, p = trace.resample(dt_ms=dt)
    # The last bin is partial, so integrate it at its true width.
    widths = np.full(len(p), dt)
    widths[-1] = trace.total_time_ms - (len(p) - 1) * dt or dt
    integral = float(np.sum(p * widths) / 1000.0)
    assert integral == pytest.approx(trace.total_energy_j, rel=1e-9)


def test_the_grid_really_is_uniform(rewriter, predictor):
    """The point of the exercise: `power_steps` has variable-width steps and
    cannot be compared against a sensor capture or averaged across runs."""
    trace = run_engine(traffic(), rewriter, predictor, engine_cfg())
    t, p = trace.resample(dt_ms=2.0)
    assert len(t) == len(p)
    assert np.allclose(np.diff(t), 2.0)

    ts, _ = trace.power_steps()
    gaps = np.diff(np.asarray(ts))
    assert gaps.std() > 1e-6, "event steps were uniform; nothing to fix"


def test_idle_pulls_the_resampled_series_down_to_idle_power(rewriter, predictor):
    trace = run_engine(traffic(qps=1.0), rewriter, predictor, engine_cfg())
    _, p = trace.resample(dt_ms=1.0)
    assert trace.idle_time_ms > 0
    assert p.min() == pytest.approx(trace.config.idle_w, rel=0.05)
    assert p.max() > p.min()


def test_excluding_idle_drops_exactly_the_idle_energy(rewriter, predictor):
    """Leaving idle out is what makes a duty-cycled machine look like it draws
    its busy power all day, so the option exists but is not the default."""
    trace = run_engine(traffic(qps=1.0), rewriter, predictor, engine_cfg())
    dt = 1.0

    def integral(include):
        _, p = trace.resample(dt_ms=dt, include_idle=include)
        w = np.full(len(p), dt)
        w[-1] = trace.total_time_ms - (len(p) - 1) * dt or dt
        return float(np.sum(p * w) / 1000.0)

    assert integral(True) == pytest.approx(trace.total_energy_j, rel=1e-9)
    assert integral(False) == pytest.approx(trace.busy_energy_j, rel=1e-9)


def test_smoothing_reduces_variation_without_moving_the_mean(rewriter, predictor):
    """A real sensor never sees the square edges a kernel trace produces --
    board capacitance smooths microsecond transitions. Charge is conserved, so
    the mean must survive."""
    trace = run_engine(traffic(n=20, qps=300.0), rewriter, predictor, engine_cfg())
    _, raw = trace.resample(dt_ms=0.5)
    _, smooth = trace.resample(dt_ms=0.5, smooth_tau_ms=10.0)
    assert smooth.std() < raw.std()
    assert smooth.mean() == pytest.approx(raw.mean(), rel=0.05)


def test_a_longer_time_constant_smooths_more(rewriter, predictor):
    trace = run_engine(traffic(n=20, qps=300.0), rewriter, predictor, engine_cfg())
    _, fast = trace.resample(dt_ms=0.5, smooth_tau_ms=1.0)
    _, slow = trace.resample(dt_ms=0.5, smooth_tau_ms=50.0)
    assert slow.std() < fast.std()


def test_resample_rejects_a_nonpositive_step(rewriter, predictor):
    trace = run_engine(traffic(), rewriter, predictor, engine_cfg())
    with pytest.raises(ValueError):
        trace.resample(dt_ms=0.0)


def test_a_coarse_grid_hides_the_peak_a_fine_one_shows(rewriter, predictor):
    """Why the sample rate has to be reported next to any peak-power number:
    the same trace has different peaks at different apertures."""
    trace = run_engine(traffic(n=20, qps=300.0), rewriter, predictor, engine_cfg())
    _, fine = trace.resample(dt_ms=0.5)
    _, coarse = trace.resample(dt_ms=50.0)
    assert fine.max() > coarse.max()


# -- PowerTrace-Sim figure style and metrics ---------------------------------

def _two_traces(rewriter, predictor):
    a = run_engine(traffic(n=14, qps=250.0), rewriter, predictor, engine_cfg())
    b = run_engine(traffic(n=14, qps=250.0), rewriter, predictor, engine_cfg())
    return a, b


def test_ks_distance_matches_the_textbook_definition():
    from dynshape.engine_plot import _ks_distance

    # Identical samples -> distance 0.
    assert _ks_distance([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)
    # Completely separated -> distance 1.
    assert _ks_distance([0, 0, 0], [9, 9, 9]) == pytest.approx(1.0)
    # Half-overlapping, worked by hand.
    assert _ks_distance([0, 1], [1, 2]) == pytest.approx(0.5)


def test_agreement_of_a_trace_with_itself_is_perfect(rewriter, predictor):
    from dynshape.engine_plot import trace_agreement

    t = run_engine(traffic(n=14, qps=250.0), rewriter, predictor, engine_cfg())
    m = trace_agreement(t, t, dt_s=0.02)
    assert m["energy_error_pct"] == pytest.approx(0.0, abs=1e-9)
    assert m["mean_bias_pct"] == pytest.approx(0.0, abs=1e-9)
    assert m["nrmse_range"] == pytest.approx(0.0, abs=1e-9)
    assert m["ks_agreement"] == pytest.approx(1.0)


def test_energy_error_alone_would_hide_a_completely_different_trace(
        rewriter, predictor):
    """The reason FSTS reports five metrics and not one: two traces can carry
    the same total energy while one is flat and the other swings between idle
    and peak, and for anything that sizes a breaker those are different traces.
    """
    from dynshape.engine_plot import _ks_distance, trace_agreement

    t = run_engine(traffic(n=14, qps=250.0), rewriter, predictor, engine_cfg())
    _, p = t.resample(dt_ms=20.0)

    # A flat line carrying exactly the same energy.
    flat = np.full_like(p, p.mean())
    assert abs(flat.sum() - p.sum()) / p.sum() == pytest.approx(0.0, abs=1e-9)
    # ...and a KS distance that is nowhere near zero, if the trace varies at all.
    if p.std() > 1e-9:
        assert _ks_distance(p, flat) > 0.2

    m = trace_agreement(t, t, dt_s=0.02)
    assert set(m) >= {"energy_error_pct", "mean_bias_pct", "nrmse_range",
                      "acf_r2", "ks_agreement"}


def test_mean_bias_is_signed(rewriter, predictor):
    """Unsigned error cannot distinguish systematic over-prediction from noise,
    and systematic is the kind that survives aggregation."""
    from dynshape.engine_plot import trace_agreement

    t = run_engine(traffic(n=14, qps=250.0), rewriter, predictor, engine_cfg())
    hot = run_engine(traffic(n=14, qps=250.0), rewriter, predictor,
                     engine_cfg(idle_w=200.0))
    m = trace_agreement(t, hot, dt_s=0.02)
    assert m["mean_bias_pct"] > 0


def test_agreement_refuses_a_window_too_coarse_for_the_run(rewriter, predictor):
    from dynshape.engine_plot import trace_agreement

    t = run_engine(traffic(n=6, qps=250.0), rewriter, predictor, engine_cfg())
    with pytest.raises(ValueError, match="too short"):
        trace_agreement(t, t, dt_s=1000.0)


def test_the_paper_figure_renders_and_keeps_its_conventions(rewriter, predictor):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from dynshape.engine_plot import STANFORD_RED, plot_power_trace_paper

    a, b = _two_traces(rewriter, predictor)
    ax = plot_power_trace_paper([("EnergAIzer LUT", a), ("Roofline", b)],
                                dt_s=0.02)
    # Alpha-gradient lines are LineCollections, not Line2D.
    assert len(ax.collections) == 2
    assert ax.get_ylim()[0] == 0.0
    assert ax.get_xlim()[0] == 0.0
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["EnergAIzer LUT", "Roofline"]
    assert "Power per GPU (W)" in ax.get_ylabel()
    plt.close("all")


def test_the_first_series_is_the_black_reference(rewriter, predictor):
    """In FSTS the black line is hardware. Nothing here is, so the first series
    is whatever the caller nominates as the reference -- and the label has to
    say what it actually is rather than borrowing the word 'measured'."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from dynshape.engine_plot import STANFORD_RED, plot_power_trace_paper

    a, b = _two_traces(rewriter, predictor)
    ax = plot_power_trace_paper([("ref", a), ("candidate", b)], dt_s=0.02)
    first = ax.collections[0].get_colors()[0][:3]
    second = ax.collections[1].get_colors()[0][:3]
    assert tuple(first) == pytest.approx(to_rgba("black")[:3])
    assert tuple(second) == pytest.approx(to_rgba(STANFORD_RED)[:3])
    plt.close("all")


def test_alpha_really_increases_along_the_line(rewriter, predictor):
    """The convention that lets you see which way time runs where two dense
    traces overlap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from dynshape.engine_plot import plot_power_trace_paper

    a, _ = _two_traces(rewriter, predictor)
    ax = plot_power_trace_paper([("ref", a)], dt_s=0.02)
    alphas = ax.collections[0].get_colors()[:, 3]
    assert alphas[-1] > alphas[0]
    plt.close("all")


def test_a_single_series_still_renders(rewriter, predictor):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from dynshape.engine_plot import plot_power_trace_paper

    a, _ = _two_traces(rewriter, predictor)
    ax = plot_power_trace_paper([("only", a)], dt_s=0.02)
    assert len(ax.collections) == 1
    plt.close("all")


def test_the_paper_figure_rejects_an_empty_call():
    from dynshape.engine_plot import plot_power_trace_paper
    with pytest.raises(ValueError):
        plot_power_trace_paper([])


def test_averaging_four_native_samples_equals_one_second_resampling(
        rewriter, predictor):
    """FSTS samples at 250 ms and plots four-sample means. Because both are box
    filters on aligned bins, that must equal resampling straight to 1 s -- their
    `one_second_pair` and our `resample` are the same operation."""
    from dynshape.engine_plot import fsts_grids

    t = run_engine(traffic(n=30, qps=250.0), rewriter, predictor, engine_cfg())
    _, native, _, plotted = fsts_grids(t)
    usable = (native.size // 4) * 4
    if usable < 4:
        pytest.skip("run too short for a one-second bin")
    manual = native[:usable].reshape(-1, 4).mean(axis=1)
    assert np.allclose(manual, plotted[:manual.size], rtol=1e-9, atol=1e-9)


def test_the_native_grid_is_250ms_because_that_is_the_sensor_rate(
        rewriter, predictor):
    """`nvidia-smi ... -lms 250` in their profiling jobs. Anything finer is
    simulator-only detail no meter in that campaign recorded."""
    from dynshape.engine_plot import FSTS_NATIVE_DT_S, fsts_grids

    assert FSTS_NATIVE_DT_S == 0.25
    t = run_engine(traffic(n=30, qps=250.0), rewriter, predictor, engine_cfg())
    t_n, _, _, _ = fsts_grids(t)
    assert np.allclose(np.diff(t_n), 250.0)


def test_the_peak_differs_between_the_two_grids(rewriter, predictor):
    """Which is why the grid has to be reported next to any peak: a number read
    off the 250 ms series is not the number in their 1 s tables."""
    from dynshape.engine_plot import fsts_grids

    t = run_engine(traffic(n=30, qps=250.0), rewriter, predictor, engine_cfg())
    _, native, _, plotted = fsts_grids(t)
    if native.std() < 1e-9:
        pytest.skip("trace is flat; nothing for the aperture to smooth")
    assert native.max() >= plotted.max()
