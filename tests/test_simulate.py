"""The timeline: contiguous, energy-conserving, and pointed the right way."""

import math

import matplotlib
matplotlib.use("Agg")            # headless -- no display needed

import pytest

from dynshape import (AnalyticBackend, CachedPredictor, RandomShapeGenerator,
                      Request, WorkloadConfig, simulate, sweep)
from dynshape.simulate import GAP_MS, IDLE_W


@pytest.fixture(scope="module")
def small_trace(rewriter):
    reqs = RandomShapeGenerator(
        WorkloadConfig(batch_choices=(1, 4, 8), seq_min=128, seq_max=1024,
                       seq_round_to=128), seed=0).sample(12)
    return simulate(reqs, rewriter, CachedPredictor(backend=AnalyticBackend()))


# --- structure -------------------------------------------------------------

def test_every_request_appears(small_trace):
    assert len(small_trace.requests) == 12
    assert {r.idx for r in small_trace.requests} == set(range(12))


def test_kernel_count_matches_the_expansion(small_trace, rewriter):
    per_request = rewriter.n_kernels() + 1          # + the gap
    assert len(small_trace.kernels) == 12 * per_request
    assert sum(1 for k in small_trace.kernels if k.kind == "GAP") == 12


def test_timeline_is_contiguous_and_monotone(small_trace):
    t = 0.0
    for k in small_trace.kernels:
        assert k.time_ms > 0
        assert math.isclose(k.t_start_ms, t, rel_tol=0, abs_tol=1e-9), "gap in the timeline"
        t = k.t_end_ms
    assert math.isclose(small_trace.total_time_ms, t)


def test_request_windows_tile_the_timeline(small_trace):
    t = 0.0
    for r in small_trace.requests:
        assert math.isclose(r.t_start_ms, t, abs_tol=1e-9)
        t += r.time_ms
    assert math.isclose(t, small_trace.total_time_ms)


# --- energy ----------------------------------------------------------------

def test_energy_is_conserved_per_kernel(small_trace):
    for k in small_trace.kernels:
        assert math.isclose(k.energy_j, k.power_w * k.time_ms / 1000.0, rel_tol=1e-9)


def test_energy_is_conserved_per_request(small_trace):
    for r in small_trace.requests:
        assert math.isclose(r.energy_j, r.avg_power_w * r.time_ms / 1000.0, rel_tol=1e-9)


def test_trace_energy_equals_the_sum_of_its_parts(small_trace):
    assert math.isclose(small_trace.total_energy_j,
                        sum(r.energy_j for r in small_trace.requests), rel_tol=1e-9)
    assert math.isclose(small_trace.avg_power_w,
                        small_trace.total_energy_j / (small_trace.total_time_ms / 1000.0),
                        rel_tol=1e-9)


def test_average_power_sits_between_idle_and_peak(small_trace):
    assert IDLE_W <= small_trace.avg_power_w <= small_trace.peak_power_w <= 250.0


# --- gaps ------------------------------------------------------------------

def test_gaps_are_at_idle_power_not_zero(small_trace):
    gaps = [k for k in small_trace.kernels if k.kind == "GAP"]
    assert gaps
    assert all(k.power_w == IDLE_W and k.time_ms == GAP_MS for k in gaps)
    assert all(k.energy_j > 0 for k in gaps), "idle is not free"


def test_one_gap_per_request_not_per_kernel(small_trace):
    assert sum(1 for k in small_trace.kernels if k.kind == "GAP") == len(small_trace.requests)


def test_gaps_can_be_disabled(rewriter):
    tr = simulate([Request(0, 4, 256, "prefill")], rewriter,
                  CachedPredictor(backend=AnalyticBackend()), gap_ms=0.0)
    assert not any(k.kind == "GAP" for k in tr.kernels)


# --- direction of the physics ---------------------------------------------

def test_prefill_is_slower_and_hotter_than_decode(rewriter):
    p = CachedPredictor(backend=AnalyticBackend())
    tr = simulate([Request(0, 8, 2048, "prefill"), Request(1, 8, 2048, "decode")],
                  rewriter, p)
    pre, dec = tr.requests
    assert pre.time_ms > dec.time_ms * 50, "prefill should dominate"
    assert pre.avg_power_w > dec.avg_power_w, "prefill is compute-bound, decode is not"


def test_longer_prompts_take_longer(rewriter):
    p = CachedPredictor(backend=AnalyticBackend())
    reqs = [Request(i, 8, s, "prefill") for i, s in enumerate((128, 512, 2048))]
    tr = simulate(reqs, rewriter, p)
    ts = [r.time_ms for r in tr.requests]
    assert ts[0] < ts[1] < ts[2]


def test_bigger_batches_take_longer(rewriter):
    p = CachedPredictor(backend=AnalyticBackend())
    reqs = [Request(i, b, 512, "prefill") for i, b in enumerate((1, 4, 16))]
    tr = simulate(reqs, rewriter, p)
    ts = [r.time_ms for r in tr.requests]
    assert ts[0] < ts[1] < ts[2]


def test_energy_per_token_falls_as_batch_grows(rewriter):
    """Batching amortises weight traffic -- the reason anyone batches at all."""
    p = CachedPredictor(backend=AnalyticBackend())
    reqs = [Request(i, b, 1, "decode") for i, b in enumerate((1, 8, 32))]
    tr = simulate(reqs, rewriter, p, gap_ms=0.0)
    e = [r.energy_per_token_mj for r in tr.requests]
    assert e[0] > e[1] > e[2]


# --- robustness ------------------------------------------------------------

class _PartialBackend(AnalyticBackend):
    name = "partial (test)"

    def predict(self, q, op, freq):
        if op[0] == "softmax":
            raise NotImplementedError("no softmax model")
        return super().predict(q, op, freq)


def test_unsupported_ops_are_skipped_and_counted(rewriter):
    tr = simulate([Request(0, 4, 256, "prefill")], rewriter,
                  CachedPredictor(backend=_PartialBackend()))
    assert tr.skipped_ops.get("softmax") == 12
    assert tr.requests[0].n_skipped == 12
    assert tr.requests[0].n_kernels == rewriter.n_kernels() - 12


def test_unsupported_ops_can_be_made_fatal(rewriter):
    with pytest.raises(NotImplementedError):
        simulate([Request(0, 4, 256, "prefill")], rewriter,
                 CachedPredictor(backend=_PartialBackend()), skip_unsupported=False)


def test_summary_reports_provenance(small_trace):
    s = small_trace.summary()
    assert s["is_measured_model"] is False
    assert s["requests"] == 12 and s["kernels"] > 0


# --- the (time, power) join the graphs consume -----------------------------

def test_power_steps_are_plottable(small_trace):
    ts, ps = small_trace.power_steps()
    assert len(ts) == len(ps) == len(small_trace.kernels) + 1
    assert all(b >= a for a, b in zip(ts, ts[1:])), "time must not go backwards"
    assert math.isclose(ts[-1], small_trace.total_time_ms)


def test_dataframes_round_trip(small_trace):
    kdf, rdf = small_trace.to_dataframes()
    assert len(kdf) == len(small_trace.kernels)
    assert len(rdf) == len(small_trace.requests)
    assert {"t_start_ms", "time_ms", "power_w", "energy_j"} <= set(kdf.columns)
    assert {"avg_power_w", "tokens_per_s", "energy_per_token_mj"} <= set(rdf.columns)


def test_all_figures_render(small_trace, rewriter):
    import matplotlib.pyplot as plt
    from dynshape.plot import (plot_dashboard, plot_power_timeline,
                               plot_request_scatter, plot_shape_sweep)

    plot_power_timeline(small_trace)
    plot_request_scatter(small_trace)
    plot_dashboard(small_trace)

    grid = simulate(sweep([1, 8], [128, 512, 1024]), rewriter,
                    CachedPredictor(backend=AnalyticBackend()))
    plot_shape_sweep(grid)
    plt.close("all")


def test_empty_trace_does_not_divide_by_zero():
    from dynshape.simulate import Trace
    t = Trace()
    assert t.total_time_ms == 0 and t.avg_power_w == 0 and t.peak_power_w == 0
    assert t.power_steps() == ([], [])
