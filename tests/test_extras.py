"""The kernels the traced template leaves out: counts, sizes, placement, and
the guarantee that adding them never changes a template entry.
"""

import pytest

from dynshape.engine import EngineConfig, price_kernel, run_engine
from dynshape.entities import Piece, reset_ids
from dynshape.extras import (ELEMENTWISE_MIN_DIM, ExtraKernel, ExtraSet, MeasuredPrice,
                             Mono, gpt2_hf_eager)
from dynshape.mixed import (_piece_extras, build_iteration_kernels_tagged,
                            extra_slots, fusible_mask)
from dynshape.scheduler import SchedulerConfig
from dynshape.traffic import TrafficConfig, generate_traffic

X = gpt2_hf_eager()

#: Kernels per forward pass beyond the template, by profiler family, as profiled
#: on a Colab A100 (A100_Energy_Gap_Colab.ipynb, section 5).
PROFILED_PREFILL = {"gemm": 1, "cat": 24, "where / mask": 13, "elementwise": 14,
                    "fill": 25, "copy": 13, "index / embed": 6}
PROFILED_DECODE = dict(PROFILED_PREFILL, copy=1)


def pre(tokens, context=0, rid=0):
    return Piece(kind="prefill", request_id=rid, tokens=tokens, context=context)


def dec(context, rid=0):
    return Piece(kind="decode", request_id=rid, tokens=1, context=context)


def extras_by_name(tagged):
    out = {}
    for (q, _op), tag in tagged:
        phase, _, name = tag.partition("+")
        if name:
            out.setdefault(name, []).append((q, phase))
    return out


def families(tagged):
    family = {x.name: x.family for x in X.kernels}
    out = {}
    for name, rows in extras_by_name(tagged).items():
        out[family[name]] = out.get(family[name], 0) + len(rows)
    return out


# -- the counts the profile found ----------------------------------------------

def test_a_prefill_pass_launches_the_profiled_338_kernels(rewriter):
    tagged = build_iteration_kernels_tagged(rewriter, [pre(1000)], fuse_linear=False,
                                            extras=X)
    assert len(tagged) == 338
    assert families(tagged) == PROFILED_PREFILL


def test_a_decode_step_launches_the_profiled_326_kernels(rewriter):
    """One token's heads merge without a copy, so a decode step has 12 fewer."""
    tagged = build_iteration_kernels_tagged(rewriter, [dec(512)], fuse_linear=False,
                                            extras=X)
    assert len(tagged) == 326
    assert families(tagged) == PROFILED_DECODE


def test_the_count_helpers_agree_with_the_expansion(rewriter):
    blocks = len(extra_slots(rewriter).at["before_attn"])
    assert blocks == 12
    assert X.launches(1000, blocks) == 96
    assert X.launches(1, blocks) == 84
    assert X.family_counts(1000, blocks) == PROFILED_PREFILL


# -- adding extras never touches the template ----------------------------------

def test_extras_only_insert_they_never_alter_a_template_entry(rewriter):
    pieces = [dec(128, rid=0), pre(500, rid=1), pre(64, context=200, rid=2)]
    for fuse in (True, False):
        plain = build_iteration_kernels_tagged(rewriter, pieces, fuse_linear=fuse)
        with_x = build_iteration_kernels_tagged(rewriter, pieces, fuse_linear=fuse,
                                                extras=X)
        assert [kt for kt in with_x if "+" not in kt[1]] == plain


def test_no_extras_is_the_old_output(rewriter):
    pieces = [dec(300, rid=0), pre(200, rid=1)]
    for fuse in (True, False):
        assert build_iteration_kernels_tagged(rewriter, pieces, fuse_linear=fuse) == \
            build_iteration_kernels_tagged(rewriter, pieces, fuse_linear=fuse, extras=None)


# -- sizes ---------------------------------------------------------------------

def test_decode_extras_are_sized_for_context_plus_one(rewriter):
    """A decode step attends over its context and itself: k = 513 at context 512."""
    by = extras_by_name(build_iteration_kernels_tagged(
        rewriter, [dec(512)], fuse_linear=False, extras=X))
    k = 513
    assert {q["dim"] for q, _ in by["kv_cache_cat"]} == {768 * k}
    assert {q["dim"] for q, _ in by["causal_where"]} == {12 * k}
    assert {q["dim"] for q, _ in by["attn_mask_add"]} == {12 * k}
    assert {q["dim"] for q, _ in by["scale_fill"]} == {ELEMENTWISE_MIN_DIM}
    assert "merge_heads_copy" not in by
    assert [q for q, _ in by["lm_head"]] == [
        {"batch": 1, "dimM": 1, "dimN": 50257, "dimK": 768,
         "precM": "bf16", "precA": "bf16", "useTensorCore": True}]


def test_prefill_extras_are_sized_for_the_chunk(rewriter):
    """A 200-token chunk on 300 tokens of context: q = 200, k = 500."""
    by = extras_by_name(build_iteration_kernels_tagged(
        rewriter, [pre(200, context=300)], fuse_linear=False, extras=X))
    assert {q["dim"] for q, _ in by["kv_cache_cat"]} == {768 * 500}
    assert {q["dim"] for q, _ in by["causal_where"]} == {12 * 200 * 500}
    assert {q["dim"] for q, _ in by["merge_heads_copy"]} == {768 * 200}
    assert {q["dim"] for q, _ in by["embed_lookup"]} == {768 * 200}
    assert {q["dim"] for q, _ in by["mask_where"]} == {200 * 500}
    assert [q["dimM"] for q, _ in by["lm_head"]] == [200]


# -- placement -----------------------------------------------------------------

def test_extras_land_where_the_real_pass_launches_them(rewriter):
    """Block 0 reads: layernorm, QKV, the K/V appends, Q.K, the scale fill, the
    scale, the mask value, the causal where, the mask add, softmax, A.V, the
    head merge, the output projection.  The LM head comes last."""
    tagged = build_iteration_kernels_tagged(rewriter, [pre(256)], fuse_linear=False,
                                            extras=X)
    names = [tag.partition("+")[2] or op[0] for (_q, op), tag in tagged]
    prologue = sum(x.count for x in X.kernels if x.anchor == "prologue")
    assert prologue == 11
    assert names[prologue:prologue + 15] == [
        "elementwise", "layernorm", "gemm",
        "kv_cache_cat", "kv_cache_cat", "gemm",
        "scale_fill", "elementwise",
        "mask_value_fill", "causal_where", "attn_mask_add", "softmax",
        "gemm", "merge_heads_copy", "gemm"]
    assert names[-2:] == ["layernorm", "lm_head"]


def test_fused_extras_split_like_the_template(rewriter):
    """Extras that never read the key axis launch once per iteration at the
    summed token count, like a linear layer; the rest launch per request, like
    attention -- and everything inside an attention run is per request."""
    pieces = [pre(100, rid=0), dec(50, rid=1)]
    tagged = build_iteration_kernels_tagged(rewriter, pieces, extras=X)
    by = extras_by_name(tagged)

    assert [(q["dimM"], ph) for q, ph in by["lm_head"]] == [(101, "fused")]
    assert [q["dim"] for q, _ in by["embed_lookup"]] == [768 * 101] * 2
    # Only the prefill chunk needs its heads merged.
    assert [q["dim"] for q, _ in by["merge_heads_copy"]] == [768 * 100] * 12
    assert len(by["kv_cache_cat"]) == 2 * 12 * 2
    assert {ph for _, ph in by["kv_cache_cat"]} == {"attn:prefill", "attn:decode"}
    # Constant, but inside the attention run: once per request per block.
    assert len(by["scale_fill"]) == 12 * 2

    n_template = fusible_mask(rewriter).count(True) + 2 * 48
    assert len(tagged) == n_template + 20 + 152


def test_logits_last_token_only_shrinks_the_lm_head(rewriter):
    def head(tagged):
        return [q["dimM"] for (q, _op), tag in tagged if tag.endswith("+lm_head")]

    one = [pre(2000)]
    assert head(build_iteration_kernels_tagged(
        rewriter, one, fuse_linear=False, extras=X)) == [2000]
    assert head(build_iteration_kernels_tagged(
        rewriter, one, fuse_linear=False, extras=X, logits_last_token_only=True)) == [1]
    three = [pre(300, rid=0), dec(40, rid=1), dec(41, rid=2)]
    assert head(build_iteration_kernels_tagged(
        rewriter, three, extras=X, logits_last_token_only=True)) == [3]


# -- pricing, persistence, validation ------------------------------------------

def test_every_extra_can_be_priced(predictor):
    for piece in (pre(1000), dec(512), pre(1), pre(64, context=900)):
        for _x, (q, op) in _piece_extras(X, piece):
            t_ms, _p, e_j = predictor.predict(q, op)
            assert t_ms > 0 and e_j > 0


def test_an_extra_set_survives_a_json_round_trip(tmp_path):
    path = str(tmp_path / "extras.json")
    X.save(path)
    assert ExtraSet.load(path) == X


def test_bad_specs_are_refused():
    fields = {"dim": 1, "op": "scalar_add", "prec": "bf16"}
    with pytest.raises(ValueError):
        ExtraKernel("x", "fill", "middle", ("elementwise",), fields)
    with pytest.raises(ValueError):
        ExtraKernel("x", "fill", "prologue", ("elementwise",), fields, count=0)
    k = ExtraKernel("x", "fill", "prologue", ("elementwise",), fields)
    with pytest.raises(ValueError):
        ExtraSet((k, k))


def test_mono_reads_keys_only_when_it_grows_with_k():
    by = {x.name: x for x in X.kernels}
    assert by["kv_cache_cat"].reads_keys and by["causal_where"].reads_keys
    assert not by["lm_head"].reads_keys and not by["merge_heads_copy"].reads_keys
    assert Mono(12, 1, 1)(3, 5) == 180


# -- through the engine --------------------------------------------------------

def _traffic(n=4):
    reset_ids()
    return generate_traffic(TrafficConfig(
        num_requests=n, interval="poisson", qps=4.0, length="fixed",
        prefill_tokens=256, decode_tokens=8, seed=0))


def test_the_engine_prices_extras_and_phase_energy_still_adds_up(rewriter, predictor):
    cfg = EngineConfig(scheduler=SchedulerConfig(chunk_size=512, max_num_seqs=16,
                                                 max_tokens=4096),
                       fuse_linear=False, record_kernels_until_ms=None, extras=X)
    trace = run_engine(_traffic(), rewriter, predictor, cfg)
    assert len(trace.completed_requests) == 4
    assert "gpt2-hf-eager" in trace.shape_report
    gap_j = cfg.idle_w * cfg.gap_ms / 1000.0
    for it in trace.iterations:
        passes = it.prefill_chunks + it.decode_batch
        assert 326 * passes <= it.n_kernels + it.n_skipped <= 338 * passes
        parts = it.energy_fused_j + it.energy_attn_prefill_j + it.energy_attn_decode_j
        assert parts + gap_j == pytest.approx(it.energy_j, rel=1e-9)
    assert any("+" in s.tag for s in trace.segments if s.kind == "KERNEL")


# -- measured prices -------------------------------------------------------------

P = MeasuredPrice(field="dimM", sizes=(1, 100, 1000), time_ms=(0.1, 1.0, 4.0),
                  dyn_mj=(1.0, 50.0, 400.0))


def test_a_measured_price_hits_its_points_and_interpolates_on_log_axes():
    for s, t, e in zip(P.sizes, P.time_ms, P.dyn_mj):
        ms, mj = P(s)
        assert ms == pytest.approx(t) and mj == pytest.approx(e)
    ms, mj = P(10)                        # halfway between 1 and 100 on a log axis
    assert ms == pytest.approx((0.1 * 1.0) ** 0.5)
    assert mj == pytest.approx(30.0 * ms)  # power halfway between 10 W and 50 W


def test_a_measured_price_clamps_below_and_scales_above():
    assert P(0.5) == P(1)
    ms, mj = P(2000)
    assert ms == pytest.approx(8.0) and mj == pytest.approx(800.0)


def test_negative_measured_energy_reads_as_zero_power():
    p = MeasuredPrice(field="dim", sizes=(128,), time_ms=(0.003,), dyn_mj=(-0.001,))
    assert p(128) == (0.003, 0.0)


def test_bad_prices_are_refused():
    with pytest.raises(ValueError):
        MeasuredPrice(field="dim", sizes=(10, 5), time_ms=(1, 1), dyn_mj=(1, 1))
    with pytest.raises(ValueError):
        MeasuredPrice(field="dim", sizes=(1,), time_ms=(0,), dyn_mj=(1,))
    with pytest.raises(ValueError):
        X.with_prices({"no_such_kernel": P})
    with pytest.raises(ValueError):           # the LM head has no `dim` field
        X.with_prices({"lm_head": MeasuredPrice(field="dim", sizes=(1,), time_ms=(1,),
                                                dyn_mj=(1,))})


def test_prices_attach_by_name_and_survive_a_json_round_trip(tmp_path):
    priced = X.with_prices({"lm_head": P})
    assert priced.priced == ("lm_head",) and X.priced == ()
    assert priced.kernel("lm_head").price == P
    path = str(tmp_path / "priced.json")
    priced.save(path)
    assert ExtraSet.load(path) == priced
    assert priced.with_prices({"lm_head": None}) == X


def test_price_kernel_uses_a_measured_price_and_nothing_else(predictor):
    priced = X.with_prices({"lm_head": P})
    q, op = priced.kernel("lm_head").entry(100, 100)
    t_ms, p_w, e_j = price_kernel(predictor, q, op, "whole:prefill+lm_head", priced,
                                  idle_w=50.0)
    assert t_ms == pytest.approx(1.0)
    assert e_j == pytest.approx((50.0 * 1.0 + 50.0) / 1000.0)
    assert p_w == pytest.approx(e_j / 0.001)
    # Unpriced extras and template entries still go to the predictor.
    cq, cop = priced.kernel("kv_cache_cat").entry(1, 513)
    assert price_kernel(predictor, cq, cop, "whole:decode+kv_cache_cat", priced) == \
        predictor.predict(cq, cop)
    assert price_kernel(predictor, q, op, "whole:prefill", priced) == predictor.predict(q, op)


def test_the_engine_charges_a_measured_price(rewriter, predictor):
    slow = MeasuredPrice(field="dimM", sizes=(1, 1e6), time_ms=(500.0, 500.0),
                         dyn_mj=(0.0, 0.0))
    cfg = EngineConfig(scheduler=SchedulerConfig(chunk_size=512, max_num_seqs=16,
                                                 max_tokens=4096),
                       fuse_linear=False, record_kernels_until_ms=None,
                       extras=X.with_prices({"lm_head": slow}))
    trace = run_engine(_traffic(2), rewriter, predictor, cfg)
    assert "measured prices for lm_head" in trace.shape_report
    for it in trace.iterations:
        assert it.duration_ms >= 500.0 * (it.prefill_chunks + it.decode_batch)
    heads = [s for s in trace.segments if s.tag.endswith("+lm_head")]
    assert heads and all(s.time_ms == pytest.approx(500.0) for s in heads)
    assert all(s.power_w == pytest.approx(cfg.idle_w) for s in heads)
