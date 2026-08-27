"""Measuring a real GPU, and replaying traffic through a real engine.

The NVML half needs hardware and skips without it. The alignment, binning and
scoring are pure arithmetic and are tested against synthetic series where the
right answer is known by construction -- which is the only way to know the
comparison is not quietly flattering the prediction.
"""

import numpy as np
import pytest

from dynshape.measure import bin_mean, compare_to_trace
from dynshape.replay import (GPT2_MAX_POSITIONS, ReplayRequest, build_replay_traffic,
                             check_fits_context, load_spec, prompt_token_ids,
                             save_spec, spec_summary, to_spec)


# -- binning point samples ---------------------------------------------------

def test_bin_mean_averages_within_each_window():
    t = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    w = np.array([10., 20., 30., 40., 50., 60.])
    grid, mean = bin_mean(t, w, dt_s=0.25, t_start=0.0, t_end=0.5)
    assert grid.tolist() == [0.0, 0.25]
    assert mean[0] == pytest.approx(20.0)      # 10, 20, 30
    assert mean[1] == pytest.approx(45.0)      # 40, 50


def test_a_window_with_no_sample_carries_the_last_reading():
    """A gap in the measurement is not a zero-power moment, and filling it with
    zero would invent a trough the GPU never had."""
    t = np.array([0.0, 0.05, 0.9])
    w = np.array([100., 100., 100.])
    _, mean = bin_mean(t, w, dt_s=0.25, t_start=0.0, t_end=1.0)
    assert (mean > 0).all()
    assert mean[1] == pytest.approx(100.0)
    assert mean[2] == pytest.approx(100.0)


def test_binning_respects_the_window():
    t = np.arange(0, 2.0, 0.01)
    w = np.where(t < 1.0, 50.0, 200.0)
    _, mean = bin_mean(t, w, dt_s=0.5, t_start=1.0, t_end=2.0)
    assert (mean > 150).all(), "samples before t_start leaked in"


def test_bin_mean_on_no_samples():
    g, m = bin_mean(np.array([]), np.array([]), dt_s=0.25)
    assert g.size == 0 and m.size == 0


# -- scoring -----------------------------------------------------------------

def _trace(rewriter, predictor, n=14, qps=250.0):
    from dynshape.engine import EngineConfig, run_engine
    from dynshape.entities import reset_ids
    from dynshape.scheduler import SchedulerConfig
    from dynshape.traffic import TrafficConfig, generate_traffic
    reset_ids()
    reqs = generate_traffic(TrafficConfig(
        num_requests=n, interval="poisson", qps=qps, length="fixed",
        prefill_tokens=256, decode_tokens=12, seed=0))
    return run_engine(reqs, rewriter, predictor, EngineConfig(
        scheduler=SchedulerConfig(chunk_size=512, max_num_seqs=16, max_tokens=4096),
        record_kernels_until_ms=0.0))


def test_a_perfect_measurement_scores_perfectly(rewriter, predictor):
    """Feed the prediction back in as the measurement: every metric must say the
    two agree. If this drifts, the scoring is broken, not the model."""
    t = _trace(rewriter, predictor)
    pt_ms, pw = t.resample(dt_ms=250.0)
    # dense point samples reproducing the predicted series exactly
    mt = np.arange(0, pt_ms[-1] / 1000.0 + 0.25, 0.005)
    mw = np.array([pw[min(len(pw) - 1, int(x / 0.25))] for x in mt])

    r = compare_to_trace(mt, mw, t, dt_s=0.25, measured_t0=0.0)
    assert r["mean_bias_pct"] == pytest.approx(0.0, abs=1e-6)
    assert r["energy_error_pct"] == pytest.approx(0.0, abs=1e-6)
    assert r["rmse_w"] == pytest.approx(0.0, abs=1e-6)
    assert r["ks_agreement"] == pytest.approx(1.0)


def test_a_uniform_over_prediction_shows_up_as_signed_bias(rewriter, predictor):
    """Systematic bias is the kind that survives aggregation, so it has to be
    reported signed rather than as an absolute error."""
    t = _trace(rewriter, predictor)
    pt_ms, pw = t.resample(dt_ms=250.0)
    mt = np.arange(0, pt_ms[-1] / 1000.0 + 0.25, 0.005)
    mw = np.array([pw[min(len(pw) - 1, int(x / 0.25))] for x in mt]) / 1.25

    r = compare_to_trace(mt, mw, t, dt_s=0.25, measured_t0=0.0)
    assert r["mean_bias_pct"] == pytest.approx(25.0, rel=0.02)
    assert r["acf_r2"] > 0.99, "a pure scale change must not disturb the shape score"


def test_subtracting_idle_changes_the_question(rewriter, predictor):
    """Board power is dominated by a floor both sides agree on for free, so it
    flatters the model; removing it asks whether the DYNAMIC power is right."""
    t = _trace(rewriter, predictor)
    pt_ms, pw = t.resample(dt_ms=250.0)
    mt = np.arange(0, pt_ms[-1] / 1000.0 + 0.25, 0.005)
    idle = 47.0
    # measured dynamic power is half what was predicted, on the same floor
    mw = np.array([idle + (pw[min(len(pw) - 1, int(x / 0.25))] - idle) * 0.5 for x in mt])

    board = compare_to_trace(mt, mw, t, dt_s=0.25, measured_t0=0.0)
    dyn = compare_to_trace(mt, mw, t, dt_s=0.25, measured_t0=0.0, subtract_idle=idle)
    assert dyn["mean_bias_pct"] > board["mean_bias_pct"]
    assert dyn["mean_bias_pct"] == pytest.approx(100.0, rel=0.05)


def test_alignment_uses_the_start_of_work(rewriter, predictor):
    """The harness idles for seconds loading a model before the first request;
    counting that as predicted idle would flatter the prediction enormously."""
    t = _trace(rewriter, predictor)
    pt_ms, pw = t.resample(dt_ms=250.0)
    dur = pt_ms[-1] / 1000.0 + 0.25
    lead = 5.0
    mt = np.arange(0, lead + dur, 0.005)
    mw = np.where(mt < lead, 47.0,
                  [pw[min(len(pw) - 1, int(max(0.0, x - lead) / 0.25))] for x in mt])

    aligned = compare_to_trace(mt, mw, t, dt_s=0.25, measured_t0=lead)
    naive = compare_to_trace(mt, mw, t, dt_s=0.25, measured_t0=0.0)
    assert abs(aligned["mean_bias_pct"]) < abs(naive["mean_bias_pct"])
    assert aligned["mean_bias_pct"] == pytest.approx(0.0, abs=1e-6)


def test_too_few_overlapping_bins_is_refused(rewriter, predictor):
    t = _trace(rewriter, predictor)
    with pytest.raises(ValueError, match="overlapping bins"):
        compare_to_trace(np.array([0.0, 1.0]), np.array([50.0, 50.0]), t, dt_s=1000.0)


# -- replay ------------------------------------------------------------------

def test_replay_traffic_fits_the_context_window():
    """The simulator will price a 2048-token context happily -- its arithmetic
    has no notion of a limit -- but GPT-2 cannot run one."""
    reqs = build_replay_traffic(num_requests=80, seed=1)
    worst = max(r.num_prefill_tokens + r.num_decode_tokens for r in reqs)
    assert worst <= GPT2_MAX_POSITIONS
    assert spec_summary(to_spec(reqs))["fits_gpt2_window"]


def test_traffic_that_does_not_fit_is_refused_not_clamped():
    """Silently trimming would make the executed workload differ from the priced
    one, which is the single thing the replay exists to prevent."""
    over = [ReplayRequest(index=0, arrival_s=0.0, n_prompt=900, n_decode=400)]
    with pytest.raises(ValueError, match="context window"):
        check_fits_context(over)


def test_build_replay_traffic_rejects_an_impossible_cap():
    with pytest.raises(ValueError, match="window"):
        build_replay_traffic(num_requests=10, max_total_tokens=4096)


def test_spec_preserves_arrival_order_and_sizes():
    reqs = build_replay_traffic(num_requests=40, seed=2)
    spec = to_spec(reqs)
    assert [s.index for s in spec] == list(range(40))
    assert all(a.arrival_s <= b.arrival_s for a, b in zip(spec, spec[1:]))
    assert sorted(s.n_prompt for s in spec) == sorted(r.num_prefill_tokens for r in reqs)


def test_spec_round_trips_through_json(tmp_path):
    spec = to_spec(build_replay_traffic(num_requests=20, seed=3))
    p = str(tmp_path / "spec.json")
    save_spec(spec, p)
    assert load_spec(p) == spec


def test_prompt_token_ids_are_exact_and_reproducible():
    """Token ids, not text: tokenising a string of the right character count
    gives a token count that is only approximately right, and on the prefill
    axis that moves the largest GEMM in the model."""
    a = prompt_token_ids(377, seed=7)
    assert len(a) == 377
    assert a == prompt_token_ids(377, seed=7)
    assert a != prompt_token_ids(377, seed=8)
    assert all(0 <= t < 50256 for t in a)


def test_prompt_token_ids_rejects_an_empty_prompt():
    with pytest.raises(ValueError):
        prompt_token_ids(0)


# -- hardware, when present --------------------------------------------------

def test_nvml_sampling_if_a_gpu_is_here():
    pytest.importorskip("pynvml")
    import pynvml
    try:
        pynvml.nvmlInit()
        if pynvml.nvmlDeviceGetCount() == 0:
            pytest.skip("no GPU visible")
    except Exception as e:
        pytest.skip(f"NVML unavailable: {e}")

    import time
    from dynshape.measure import PowerSampler
    s = PowerSampler(interval_s=0.02).start()
    time.sleep(0.5)
    t, w = s.stop()
    assert t.size > 5 and (w > 0).all()
    ok, why = s.matches_lut_hardware()
    assert isinstance(ok, bool) and isinstance(why, str)
