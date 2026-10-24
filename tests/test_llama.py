"""LLaMA templates generated from the model shape: counts, the learned scaling
law, the query/key split for decode, and the engine on top.
"""

import pytest

from dynshape import ShapeRewriter
from dynshape.engine import EngineConfig, run_engine
from dynshape.entities import Piece, reset_ids
from dynshape.extras import llama_hf_eager
from dynshape.llama import (ANCHORS, LLAMA_2_13B, LlamaShape, llama_template,
                            llama_template_labeled, n_template_entries,
                            write_llama_templates)
from dynshape.mixed import (_lm_head_indices, build_iteration_kernels_tagged,
                            extra_slots, fusible_mask, mixed_report)
from dynshape.scheduler import SchedulerConfig
from dynshape.traffic import TrafficConfig, generate_traffic

TINY = LlamaShape(name="tiny-llama", hidden=256, heads=4, kv_heads=4,
                  intermediate=688, layers=2, vocab=32000, max_position=2048)
GQA = LlamaShape(name="tiny-gqa", hidden=512, heads=8, kv_heads=2,
                 intermediate=1376, layers=2, vocab=32000, max_position=2048)


def rewriter_for(shape, tmp_path):
    d = str(tmp_path / shape.name)
    write_llama_templates(shape, d)
    return ShapeRewriter.from_dir(d, model=shape.name, b0=ANCHORS[0][0], s0=ANCHORS[0][1])


@pytest.fixture(scope="module")
def rw13(tmp_path_factory):
    return rewriter_for(LLAMA_2_13B, tmp_path_factory.mktemp("llama13"))


def pre(tokens, context=0, rid=0):
    return Piece(kind="prefill", request_id=rid, tokens=tokens, context=context)


def dec(context, rid=0):
    return Piece(kind="decode", request_id=rid, tokens=1, context=context)


# -- the template itself --------------------------------------------------------

def test_the_13b_template_has_the_predicted_length():
    assert n_template_entries(LLAMA_2_13B) == 1778
    assert len(llama_template(LLAMA_2_13B, 1, 128)) == 1778
    labels = [lab for _e, lab in llama_template_labeled(LLAMA_2_13B, 1, 128)]
    assert labels.count("aten::mm") == 7 * 40 + 1           # 7 linears per layer + LM head
    assert labels.count("aten::_softmax") == 40


def test_the_learned_law_reproduces_the_generator_anywhere(tmp_path):
    for shape in (TINY, GQA):
        rw = rewriter_for(shape, tmp_path)
        for b, s in ((1, 1000), (3, 77), (1, 1), (5, 2048)):
            assert rw.expand(b, s, "prefill") == llama_template(shape, b, s), (shape.name, b, s)


def test_the_classifiers_agree_and_find_one_attention_run_per_layer(rw13):
    rep = mixed_report(rw13)
    assert rep.classes_agree, rep.disagreements
    assert rep.n_runs == 40
    assert fusible_mask(rw13).count(False) == 7 * 40
    assert len(extra_slots(rw13).at["before_attn"]) == 40


def test_the_lm_head_is_found_and_is_last(rw13):
    assert _lm_head_indices(rw13) == [len(rw13.base) - 1]


def test_decode_shapes_come_out_of_the_query_key_split(rw13):
    """No decode law is learned: the split must send the linear layers to the
    query axis (one token) and attention to the key axis (context + 1)."""
    kernels = rw13.expand(1, 512, "decode")
    gemms = [q for q, op in kernels if op[0] == "gemm" and q["batch"] == 1 and q["dimK"] > 1]
    assert {q["dimM"] for q in gemms} == {1}
    qk = [q for q, op in kernels if op[0] == "gemm" and q["batch"] == 40 and q["dimN"] != 128]
    assert {(q["dimM"], q["dimN"], q["dimK"]) for q in qk} == {(1, 513, 128)}
    av = [q for q, op in kernels if op[0] == "gemm" and q["batch"] == 40 and q["dimN"] == 128]
    assert {(q["dimM"], q["dimN"], q["dimK"]) for q in av} == {(1, 128, 513)}
    soft = [q for q, op in kernels if op[0] == "softmax"]
    assert {(q["batch"], q["dim"]) for q in soft} == {(40, 513)}


def test_a_prefill_chunk_attends_over_its_context(rw13):
    tagged = build_iteration_kernels_tagged(rw13, [pre(200, context=300)], fuse_linear=False)
    soft = [q for (q, op), _t in tagged if op[0] == "softmax"]
    assert {(q["batch"], q["dim"]) for q in soft} == {(40 * 200, 500)}
    lm = tagged[-1][0][0]
    assert (lm["dimM"], lm["dimN"]) == (200, 32000)


# -- with the extras -----------------------------------------------------------

def test_a_pass_launches_the_predicted_kernels(rw13):
    x = llama_hf_eager(LLAMA_2_13B.hidden, LLAMA_2_13B.kv_hidden)
    assert x.launches(1000, 40) == 130 and x.launches(1, 40) == 90
    for piece, want in ((pre(1000), 1908), (dec(512), 1868)):
        tagged = build_iteration_kernels_tagged(rw13, [piece], fuse_linear=False, extras=x)
        assert len(tagged) == want
    by = {}
    for (q, _op), tag in build_iteration_kernels_tagged(rw13, [dec(512)], fuse_linear=False,
                                                        extras=x):
        by.setdefault(tag.partition("+")[2], []).append(q)
    assert {q["dim"] for q in by["kv_cache_cat"]} == {5120 * 513}
    assert "merge_heads_copy" not in by


def test_every_entry_can_be_priced(rw13, predictor):
    x = llama_hf_eager(LLAMA_2_13B.hidden, LLAMA_2_13B.kv_hidden)
    for piece in (pre(1000), dec(512), pre(1)):
        seen = set()
        for (q, op), _tag in build_iteration_kernels_tagged(rw13, [piece], fuse_linear=False,
                                                            extras=x):
            key = (tuple(sorted(q.items())), op)
            if key in seen:
                continue
            seen.add(key)
            t_ms, _p, e_j = predictor.predict(q, op)
            assert t_ms > 0 and e_j > 0


def test_the_engine_runs_a_llama(tmp_path, predictor):
    rw = rewriter_for(TINY, tmp_path)
    x = llama_hf_eager(TINY.hidden, TINY.kv_hidden)
    reset_ids()
    reqs = generate_traffic(TrafficConfig(num_requests=3, interval="poisson", qps=4.0,
                                          length="fixed", prefill_tokens=64,
                                          decode_tokens=4, seed=0))
    cfg = EngineConfig(scheduler=SchedulerConfig(chunk_size=256, max_num_seqs=8,
                                                 max_tokens=2048, num_blocks=512),
                       model=TINY.model_config(), fuse_linear=False,
                       record_kernels_until_ms=0.0, extras=x)
    trace = run_engine(reqs, rw, predictor, cfg)
    assert len(trace.completed_requests) == 3
    per_prefill = n_template_entries(TINY) + x.launches(64, TINY.layers)
    per_decode = n_template_entries(TINY) + x.launches(1, TINY.layers)
    for it in trace.iterations:
        assert it.n_kernels + it.n_skipped == (it.prefill_chunks * per_prefill
                                               + it.decode_batch * per_decode)


def test_the_shape_reads_a_hf_config():
    class Cfg:
        _name_or_path = "meta-llama/Llama-2-13b-hf"
        hidden_size, num_attention_heads, num_key_value_heads = 5120, 40, 40
        intermediate_size, num_hidden_layers = 13824, 40
        vocab_size, max_position_embeddings = 32000, 4096
    s = LlamaShape.from_hf_config(Cfg())
    assert s == LlamaShape(name="llama-2-13b-hf")
    assert abs(s.parameters() / 1e9 - 13.0) < 0.1
    assert s.kv_bytes_per_token() == 2 * 2 * 5120 * 40


# -- Qwen3: QK-norm and grouped-query attention ----------------------------------

from dynshape.llama import QWEN3_14B, llama_extras

TINY_QWEN = LlamaShape(name="tiny-qwen3", hidden=256, heads=4, kv_heads=2, intermediate=688,
                       layers=2, vocab=32000, max_position=2048, head_size=64, qk_norm=True)


def test_the_qwen3_14b_template_and_extras_have_the_predicted_counts(tmp_path):
    assert n_template_entries(QWEN3_14B) == 2418
    assert len(llama_template(QWEN3_14B, 1, 128)) == 2418
    x = llama_extras(QWEN3_14B)
    assert x.launches(1000, 40) == 210 and x.launches(1, 40) == 170
    rw = rewriter_for(QWEN3_14B, tmp_path)
    assert mixed_report(rw).classes_agree and mixed_report(rw).n_runs == 40
    for piece, want in ((pre(1000), 2628), (dec(512), 2588)):
        assert len(build_iteration_kernels_tagged(rw, [piece], fuse_linear=False, extras=x)) == want


def test_qwen3_shapes_use_the_kv_and_head_widths(tmp_path):
    rw = rewriter_for(TINY_QWEN, tmp_path)
    for b, s in ((1, 1000), (3, 77)):
        assert rw.expand(b, s, "prefill") == llama_template(TINY_QWEN, b, s)
    x = llama_extras(TINY_QWEN)
    by = {}
    for (q, _op), tag in build_iteration_kernels_tagged(rw, [dec(100)], fuse_linear=False,
                                                        extras=x):
        by.setdefault(tag.partition("+")[2], []).append(q)
    assert {q["dim"] for q in by["kv_cache_cat"]} == {2 * 64 * 101}     # kv_heads x head_dim x k
    assert {q["dim"] for q in by["repeat_kv"]} == {4 * 64 * 101}        # every query head
    gemms = [q for q in by[""] if "dimN" in q and q["batch"] == 1 and q["dimK"] == 256]
    assert sorted({q["dimN"] for q in gemms}) == [128, 256, 688, 32000]  # k/v, q, gate/up, head


def test_a_qwen3_config_reads_its_head_dim_and_qk_norm():
    class Cfg:
        _name_or_path, model_type = "Qwen/Qwen3-14B", "qwen3"
        hidden_size, num_attention_heads, num_key_value_heads, head_dim = 5120, 40, 8, 128
        intermediate_size, num_hidden_layers = 17408, 40
        vocab_size, max_position_embeddings, tie_word_embeddings = 151936, 40960, False
    s = LlamaShape.from_hf_config(Cfg())
    assert s == QWEN3_14B
    assert abs(s.parameters() / 1e9 - 14.8) < 0.1
    assert s.kv_bytes_per_token() == 2 * 2 * 1024 * 40
