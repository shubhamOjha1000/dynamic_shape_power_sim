"""The whole stack: traffic -> scheduler -> mixed kernels -> power trace."""

import pytest

from dynshape.engine import EngineConfig, run_engine
from dynshape.entities import SimRequest, reset_ids
from dynshape.kvcache import HardwareConfig, ModelConfig
from dynshape.scheduler import SchedulerConfig
from dynshape.traffic import TrafficConfig, generate_traffic


def small_engine(**sched_kw):
    return EngineConfig(
        scheduler=SchedulerConfig(**{"chunk_size": 512, "max_num_seqs": 16,
                                     "max_tokens": 4096, **sched_kw}),
        record_kernels_until_ms=50.0,
    )


def tiny_traffic(n=6, **kw):
    reset_ids()
    return generate_traffic(TrafficConfig(**{
        "num_requests": n, "interval": "poisson", "qps": 4.0,
        "length": "fixed", "prefill_tokens": 256, "decode_tokens": 8,
        "seed": 0, **kw}))


# -- it runs, and it finishes -----------------------------------------------

def test_every_request_completes(rewriter, predictor):
    reqs = tiny_traffic(6)
    trace = run_engine(reqs, rewriter, predictor, small_engine())
    assert len(trace.completed_requests) == 6
    assert all(r.num_generated_tokens == r.num_decode_tokens
               for r in trace.completed_requests)


def test_time_and_energy_are_positive_and_consistent(rewriter, predictor):
    trace = run_engine(tiny_traffic(5), rewriter, predictor, small_engine())
    assert trace.total_time_ms > 0
    assert trace.total_energy_j > 0
    assert trace.busy_time_ms + trace.idle_time_ms == pytest.approx(
        trace.total_time_ms, rel=1e-6)
    assert trace.busy_energy_j + trace.idle_energy_j == pytest.approx(
        trace.total_energy_j, rel=1e-9)


def test_iterations_tile_the_busy_time_without_gaps_or_overlap(rewriter, predictor):
    trace = run_engine(tiny_traffic(5), rewriter, predictor, small_engine())
    events = sorted([(i.t_start_ms, i.t_end_ms) for i in trace.iterations]
                    + [(s.t_start_ms, s.t_end_ms) for s in trace.idle_segments])
    for (a_start, a_end), (b_start, _) in zip(events, events[1:]):
        assert a_end == pytest.approx(b_start, abs=1e-6)


# -- the thing this layer was built for --------------------------------------

def test_prefill_and_decode_really_are_batched_together(rewriter, predictor):
    """The headline requirement: mixed iterations, not alternating ones."""
    reqs = tiny_traffic(24, qps=60.0, prefill_tokens=400, decode_tokens=40)
    trace = run_engine(reqs, rewriter, predictor, small_engine())
    mixed = [i for i in trace.iterations if i.is_mixed]
    assert mixed, "no iteration carried prefill and decode at once"
    assert trace.summary()["mixed_fraction"] > 0.05
    for i in mixed:
        assert i.decode_batch > 0 and i.prefill_chunks > 0
        assert i.total_tokens == i.prefill_tokens + i.decode_tokens


def test_a_mixed_iteration_costs_more_than_either_half_alone(rewriter, predictor):
    """Additivity made visible: the fused GEMMs grow with the summed token
    count, so a mixed pass is strictly more work than the decodes alone."""
    reqs = tiny_traffic(20, qps=60.0, prefill_tokens=400, decode_tokens=30)
    trace = run_engine(reqs, rewriter, predictor, small_engine())
    mixed = [i for i in trace.iterations if i.is_mixed]
    pure_decode = [i for i in trace.iterations
                   if i.decode_batch > 0 and i.prefill_chunks == 0]
    if not (mixed and pure_decode):
        pytest.skip("this traffic produced no contrast to compare")
    assert max(i.duration_ms for i in mixed) > min(i.duration_ms for i in pure_decode)


def test_a_long_prompt_spans_several_iterations(rewriter, predictor):
    reqs = [SimRequest(arrived_at=0.0, num_prefill_tokens=2000, num_decode_tokens=4)]
    trace = run_engine(reqs, rewriter, predictor, small_engine(chunk_size=512))
    prefill_iters = [i for i in trace.iterations if i.prefill_tokens > 0]
    assert len(prefill_iters) == 4
    assert sum(i.prefill_tokens for i in prefill_iters) == 2000


# -- idle, which only exists once there is an arrival process ----------------

def test_low_load_leaves_the_gpu_idle_and_the_two_averages_diverge(
        rewriter, predictor):
    """A report that quietly averages only busy segments overstates facility
    draw.  Both numbers are reported, and at low load they must differ."""
    reqs = tiny_traffic(4, qps=0.5)
    trace = run_engine(reqs, rewriter, predictor, small_engine())
    assert trace.idle_time_ms > 0
    assert trace.duty_cycle < 0.5
    assert trace.avg_power_w < trace.avg_busy_power_w
    assert trace.avg_power_w > 0


def test_all_at_once_traffic_leaves_no_idle(rewriter, predictor):
    reset_ids()
    reqs = [SimRequest(arrived_at=0.0, num_prefill_tokens=128, num_decode_tokens=4)
            for _ in range(4)]
    trace = run_engine(reqs, rewriter, predictor, small_engine())
    assert trace.idle_time_ms == 0
    assert trace.duty_cycle == pytest.approx(1.0)


# -- load moves the trace ----------------------------------------------------

def test_higher_load_raises_duty_cycle_and_average_power(rewriter, predictor):
    quiet = run_engine(tiny_traffic(8, qps=0.5), rewriter, predictor, small_engine())
    busy = run_engine(tiny_traffic(8, qps=50.0), rewriter, predictor, small_engine())
    assert busy.duty_cycle > quiet.duty_cycle
    assert busy.avg_power_w > quiet.avg_power_w


def test_prefill_heavy_traffic_costs_more_energy_than_decode_heavy(
        rewriter, predictor):
    """P:D is the knob that moves power most: compute-bound prefill against
    memory-bound decode, on the same request count."""
    def energy(p, d):
        reset_ids()
        reqs = generate_traffic(TrafficConfig(
            num_requests=8, interval="static", qps=20.0, length="fixed",
            prefill_tokens=p, decode_tokens=d, seed=0))
        return run_engine(reqs, rewriter, predictor, small_engine()).busy_energy_j

    assert energy(1900, 100) > energy(100, 100)


# -- fusing vs concatenating -------------------------------------------------

def test_fusing_launches_fewer_kernels_for_the_same_work(rewriter, predictor):
    reqs = tiny_traffic(12, qps=60.0, prefill_tokens=400, decode_tokens=20)
    fused = run_engine(reqs, rewriter, predictor, small_engine())
    reset_ids()
    reqs2 = tiny_traffic(12, qps=60.0, prefill_tokens=400, decode_tokens=20)
    cfg = small_engine()
    cfg.fuse_linear = False
    concat = run_engine(reqs2, rewriter, predictor, cfg)

    n_fused = sum(i.n_kernels for i in fused.iterations)
    n_concat = sum(i.n_kernels for i in concat.iterations)
    assert n_fused < n_concat
    # Not fusing pays a launch, an HBM round trip and a low-occupancy tail for
    # every request separately, so it is also slower.
    assert concat.busy_time_ms > fused.busy_time_ms


# -- energy attribution ------------------------------------------------------

def test_energy_attribution_adds_up(rewriter, predictor):
    trace = run_engine(tiny_traffic(6), rewriter, predictor, small_engine())
    for i in trace.iterations:
        parts = i.energy_fused_j + i.energy_attn_prefill_j + i.energy_attn_decode_j
        gap = trace.config.idle_w * trace.config.gap_ms / 1000.0
        assert parts + gap == pytest.approx(i.energy_j, rel=1e-9)


def test_decode_only_iterations_have_no_prefill_attention_energy(rewriter, predictor):
    trace = run_engine(tiny_traffic(4, qps=1.0), rewriter, predictor, small_engine())
    for i in trace.iterations:
        if i.prefill_chunks == 0:
            assert i.energy_attn_prefill_j == 0.0


# -- preemption --------------------------------------------------------------

def test_a_starved_kv_pool_produces_restarts_and_wasted_work(rewriter, predictor):
    """The recompute spike, end to end: real kernels, real joules, no new
    output, and nothing in the arrival pattern predicts it."""
    reset_ids()
    reqs = [SimRequest(arrived_at=0.0, num_prefill_tokens=200, num_decode_tokens=150)
            for _ in range(3)]
    cfg = small_engine(num_blocks=48, block_size=16, chunk_size=512)
    cfg.max_iterations = 20000
    cfg.record_kernels_until_ms = 0.0      # thousands of iterations; keep it light
    trace = run_engine(reqs, rewriter, predictor, cfg)
    s = trace.summary()
    assert s["preemptions"] > 0
    assert s["restart_work_tokens"] > 0
    assert any(i.kv_utilisation > 0.9 for i in trace.iterations)


def test_a_prompt_larger_than_the_pool_fails_loudly(rewriter, predictor):
    reset_ids()
    reqs = [SimRequest(arrived_at=0.0, num_prefill_tokens=4000, num_decode_tokens=4)]
    cfg = small_engine(num_blocks=8, block_size=16)     # 128 tokens of pool
    with pytest.raises(RuntimeError, match="deadlock"):
        run_engine(reqs, rewriter, predictor, cfg)


# -- recording and reporting -------------------------------------------------

def test_kernel_records_are_truncated_but_iterations_are_not(rewriter, predictor):
    reqs = tiny_traffic(10, qps=40.0)
    cfg = small_engine()
    cfg.record_kernels_until_ms = 5.0
    trace = run_engine(reqs, rewriter, predictor, cfg)
    assert trace.segments
    assert trace.total_time_ms > 5.0
    # The cutoff is applied per iteration, so an iteration that starts before
    # the limit records all of its kernels -- including any that run past it.
    assert all(trace.iterations[s.iteration].t_start_ms < 5.0 for s in trace.segments)
    assert len(trace.iterations) > len({s.iteration for s in trace.segments})


def test_kernel_recording_can_be_switched_off(rewriter, predictor):
    cfg = small_engine()
    cfg.record_kernels_until_ms = 0.0
    trace = run_engine(tiny_traffic(4), rewriter, predictor, cfg)
    assert trace.segments == []
    assert trace.iterations


def test_summary_reports_both_averages_and_the_mix(rewriter, predictor):
    trace = run_engine(tiny_traffic(8, qps=30.0), rewriter, predictor, small_engine())
    s = trace.summary()
    for key in ("iterations", "mixed_fraction", "wall_time_s", "duty_cycle",
                "avg_power_w_wallclock", "avg_power_w_busy",
                "energy_per_output_token_mj", "ttft_p50_s", "backend"):
        assert key in s, key
    assert s["completed"] == 8
    assert s["avg_power_w_busy"] >= s["avg_power_w_wallclock"]


def test_the_trace_carries_its_backend_so_a_figure_cannot_lie(rewriter, predictor):
    trace = run_engine(tiny_traffic(3), rewriter, predictor, small_engine())
    assert "SYNTHETIC" in trace.predictor_stats["backend"]
    assert trace.predictor_stats["is_measured_model"] is False


def test_dataframes_round_trip(rewriter, predictor):
    pytest.importorskip("pandas")
    trace = run_engine(tiny_traffic(4), rewriter, predictor, small_engine())
    idf, rdf, sdf = trace.to_dataframes()
    assert len(idf) == len(trace.iterations)
    assert len(rdf) == 4
    assert "is_mixed" in idf.columns


def test_power_steps_are_monotonic_in_time(rewriter, predictor):
    trace = run_engine(tiny_traffic(5, qps=2.0), rewriter, predictor, small_engine())
    ts, ps = trace.power_steps()
    assert len(ts) == len(ps)
    assert all(b >= a for a, b in zip(ts, ts[1:]))
    assert min(ps) >= trace.config.idle_w - 1e-9


def test_running_with_no_requests_is_rejected(rewriter, predictor):
    with pytest.raises(ValueError):
        run_engine([], rewriter, predictor, small_engine())


def test_the_cache_actually_hits(rewriter, predictor):
    """The 50 ms-per-lookup arithmetic: a run is only affordable because shapes
    repeat.  A hit rate near zero means the cache key is wrong."""
    trace = run_engine(tiny_traffic(10, qps=30.0), rewriter, predictor,
                       small_engine())
    assert trace.predictor_stats["hit_rate"] > 0.5
