"""The predictor backends and the cache that makes the whole thing tractable."""

import math

import pytest

from dynshape import AnalyticBackend, CachedPredictor, build_predictor

GEMM = ("gemm", "tc", "bf16_bf16")


def q_gemm(batch=1, m=4096, n=4096, k=4096, prec="bf16", tc=True):
    return {"batch": batch, "dimM": m, "dimN": n, "dimK": k,
            "precM": prec, "precA": prec, "useTensorCore": tc}


# --- the analytic fallback -------------------------------------------------

def test_backend_is_labelled_synthetic():
    b = AnalyticBackend()
    assert b.is_measured_model is False
    assert "SYNTHETIC" in b.name


def test_outputs_are_self_consistent():
    """energy must equal power x time -- otherwise every aggregate is wrong."""
    b = AnalyticBackend()
    for q in (q_gemm(), q_gemm(8, 512, 512, 512), q_gemm(1, 4096, 4096, 128)):
        t_ms, p_w, e_j = b.predict(q, GEMM, 900)
        assert t_ms > 0 and p_w > 0
        assert math.isclose(e_j, p_w * t_ms / 1000.0, rel_tol=1e-9)


def test_power_is_bounded_by_idle_and_tdp():
    b = AnalyticBackend()
    shapes = [q_gemm(1, 64, 64, 64), q_gemm(), q_gemm(32, 4096, 4096, 4096)]
    for q in shapes:
        _, p, _ = b.predict(q, GEMM, 900)
        assert 47.0 <= p <= 250.0


def test_time_grows_with_work():
    b = AnalyticBackend()
    t_small, _, _ = b.predict(q_gemm(1, 512, 512, 512), GEMM, 900)
    t_big, _, _ = b.predict(q_gemm(1, 4096, 4096, 4096), GEMM, 900)
    assert t_big > t_small * 100          # 512^3 -> 4096^3 is 512x the FLOPs


def test_time_grows_with_batch():
    b = AnalyticBackend()
    ts = [b.predict(q_gemm(bs, 1024, 1024, 1024), GEMM, 900)[0] for bs in (1, 2, 4, 8)]
    assert all(x < y for x, y in zip(ts, ts[1:]))


def test_compute_bound_draws_more_than_memory_bound():
    """The qualitative fact the graphs exist to show."""
    b = AnalyticBackend()
    _, p_compute, _ = b.predict(q_gemm(1, 4096, 4096, 4096), GEMM, 900)
    _, p_memory, _ = b.predict({"dim": 4096 * 4096, "op": "pointwise_add", "prec": "bf16"},
                               ("elementwise",), 900)
    assert p_compute > p_memory


def test_tensor_core_beats_cuda_core():
    b = AnalyticBackend()
    t_tc, _, _ = b.predict(q_gemm(prec="bf16", tc=True), GEMM, 900)
    t_cuda, _, _ = b.predict(q_gemm(prec="fp32", tc=False), ("gemm", "cuda", "fp32_fp32"), 900)
    assert t_cuda > t_tc


def test_lower_clock_is_slower():
    b = AnalyticBackend()
    t_900, _, _ = b.predict(q_gemm(), GEMM, 900)
    t_1400, _, _ = b.predict(q_gemm(), GEMM, 1400)
    assert t_900 > t_1400


def test_all_template_ops_are_supported(rewriter):
    b = AnalyticBackend()
    ops = {op for _, op in rewriter.expand(4, 256, "prefill")}
    for q, op in rewriter.expand(4, 256, "prefill"):
        b.predict(q, op, 900)             # must not raise
    assert {o[0] for o in ops} == {"gemm", "elementwise", "layernorm", "softmax"}


def test_unknown_op_raises():
    with pytest.raises(NotImplementedError):
        AnalyticBackend().predict({"dim": 10}, ("flashattention_v2",), 900)


# --- the cache -------------------------------------------------------------

def test_cache_returns_identical_values():
    p = CachedPredictor(backend=AnalyticBackend())
    first = p.predict(q_gemm(), GEMM)
    second = p.predict(q_gemm(), GEMM)
    assert first == second
    assert (p.misses, p.hits) == (1, 1)


def test_cache_key_ignores_dict_ordering():
    p = CachedPredictor(backend=AnalyticBackend())
    a = {"batch": 1, "dimM": 512, "dimN": 512, "dimK": 512,
         "precM": "bf16", "precA": "bf16", "useTensorCore": True}
    b = {k: a[k] for k in reversed(list(a))}
    p.predict(a, GEMM)
    p.predict(b, GEMM)
    assert p.misses == 1 and p.hits == 1


def test_cache_separates_distinct_shapes_and_frequencies():
    p = CachedPredictor(backend=AnalyticBackend())
    p.predict(q_gemm(1, 512, 512, 512), GEMM)
    p.predict(q_gemm(1, 1024, 512, 512), GEMM)
    assert p.misses == 2 and len(p.cache) == 2

    p900 = CachedPredictor(backend=AnalyticBackend(), freq=900)
    p1400 = CachedPredictor(backend=AnalyticBackend(), freq=1400)
    assert p900.key(q_gemm(), GEMM) != p1400.key(q_gemm(), GEMM)


def test_cache_actually_pays_off_on_a_real_expansion(rewriter):
    """242 kernels per pass, but only a few dozen distinct shapes -- because the
    12 transformer blocks are 12 literal repetitions of the same kernels."""
    p = CachedPredictor(backend=AnalyticBackend())
    for q, op in rewriter.expand(8, 512, "prefill"):
        p.predict(q, op)
    assert p.misses < 40, f"expected few distinct shapes, saw {p.misses}"
    assert p.hit_rate > 0.8


def test_build_predictor_falls_back_without_a_lut(tmp_path):
    p = build_predictor(pkg_path=str(tmp_path), lut_dir=str(tmp_path))
    assert p.backend.is_measured_model is False
    assert build_predictor(force_analytic=True).backend.is_measured_model is False


def test_stats_shape():
    p = CachedPredictor(backend=AnalyticBackend())
    p.predict(q_gemm(), GEMM)
    s = p.stats()
    assert set(s) == {"backend", "is_measured_model", "distinct_shapes",
                      "hits", "misses", "hit_rate"}
