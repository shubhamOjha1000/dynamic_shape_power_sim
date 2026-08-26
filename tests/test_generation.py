"""The decode dynamics: KV cache grows one token per step, so shapes move.

A generation is not one decode call. It is a prefill, then N decode steps whose
context grows by one each time -- and every one of those steps is a different
attention shape.
"""

import pytest

from dynshape import (AnalyticBackend, CachedPredictor, conversation,
                      generation, simulate)

HEADS = 12
HEAD_DIM = 64


def test_generation_is_one_prefill_then_n_decodes():
    reqs = generation(batch=4, prompt_len=1000, n_new_tokens=6, context_bucket=1)
    assert len(reqs) == 7
    assert reqs[0].mode == "prefill" and reqs[0].seqlen == 1000
    assert all(r.mode == "decode" for r in reqs[1:])
    assert {r.idx for r in reqs} == set(range(7))


def test_context_grows_by_one_per_step():
    reqs = generation(batch=4, prompt_len=1000, n_new_tokens=5, context_bucket=1)
    assert [r.seqlen for r in reqs[1:]] == [1000, 1001, 1002, 1003, 1004]


def test_prefill_can_be_omitted():
    reqs = generation(batch=2, prompt_len=64, n_new_tokens=3,
                      context_bucket=1, include_prefill=False)
    assert len(reqs) == 3 and all(r.mode == "decode" for r in reqs)


def test_bucketing_rounds_up_and_holds_the_shape_steady():
    """Vidur rounds context up to a granularity too -- (x + g - 1) // g * g."""
    reqs = generation(batch=1, prompt_len=1000, n_new_tokens=200, context_bucket=128)
    ctxs = [r.seqlen for r in reqs[1:]]
    assert all(c % 128 == 0 for c in ctxs)
    # Raw context runs 1000 -> 1199, so it lands in three buckets.
    assert ctxs[0] == 1024, "1000 rounds up to 1024"
    assert ctxs[-1] == 1280, "1199 rounds up to 1280"
    assert sorted(set(ctxs)) == [1024, 1152, 1280]
    assert len(ctxs) == 200, "200 steps, but only 3 distinct shapes among them"


def test_each_step_really_is_a_different_attention_shape(rewriter):
    """The point of the whole exercise -- the strip gets longer every token."""
    reqs = generation(batch=4, prompt_len=1000, n_new_tokens=4, context_bucket=1)

    seen = []
    for r in reqs[1:]:
        kernels = rewriter.expand(r.batch, r.seqlen, r.mode)
        qk = next(q for q, op in kernels
                  if op[0] == "gemm" and q.get("batch") == 4 * HEADS
                  and q["dimK"] == HEAD_DIM)
        proj = next(q for q, op in kernels if op[0] == "gemm" and q.get("batch") == 1)
        assert qk["dimM"] == 1, "one query row per sequence"
        assert proj["dimM"] == 4, "projection stays at batch rows the whole way"
        seen.append(qk["dimN"])

    # context + 1 keys at each step -- the token attends over its own K/V too.
    assert seen == [1001, 1002, 1003, 1004], "the KV strip must lengthen each step"


def test_decode_gets_slower_and_hotter_as_context_grows(rewriter):
    """Attention work grows linearly with context; the projections do not."""
    reqs = generation(batch=8, prompt_len=512, n_new_tokens=1500, context_bucket=128)
    trace = simulate(reqs, rewriter, CachedPredictor(backend=AnalyticBackend()),
                     gap_ms=0.0)
    dec = [r for r in trace.requests if r.mode == "decode"]

    assert dec[-1].seqlen > dec[0].seqlen, "the context must have grown"
    assert dec[-1].time_ms > dec[0].time_ms * 1.3, "later steps should be slower"
    assert dec[-1].avg_power_w > dec[0].avg_power_w, "and draw more"

    # Monotone in context, not just at the endpoints.
    by_ctx = sorted({(r.seqlen, r.time_ms) for r in dec})
    times = [t for _, t in by_ctx]
    assert times == sorted(times), "step time must not fall as context grows"


def test_bucketing_keeps_the_cache_alive(rewriter):
    """Exact per-token contexts make every step a distinct shape; bucketing is
    what turns a 2000-step generation back into something tractable."""
    p = CachedPredictor(backend=AnalyticBackend())
    simulate(generation(1, 512, 400, context_bucket=128), rewriter, p, gap_ms=0.0)
    bucketed = p.stats()["distinct_shapes"]

    p2 = CachedPredictor(backend=AnalyticBackend())
    simulate(generation(1, 512, 400, context_bucket=1), rewriter, p2, gap_ms=0.0)
    exact = p2.stats()["distinct_shapes"]

    assert exact > bucketed * 10, f"bucketing should collapse shapes ({exact} vs {bucketed})"
    assert p.hit_rate > 0.95


# --- multi-turn ------------------------------------------------------------

def test_conversation_carries_context_across_turns():
    """Turn 2 prefills only the new text but decodes from an already-long cache."""
    reqs = conversation([(500, 3), (20, 3)], batch=1, context_bucket=1)
    assert [r.mode for r in reqs] == ["prefill", "decode", "decode", "decode",
                                      "prefill", "decode", "decode", "decode"]
    assert [r.seqlen for r in reqs] == [500, 500, 501, 502, 20, 523, 524, 525]


def test_conversation_prefill_shrinks_while_decode_context_grows(rewriter):
    """The asymmetry: later turns are cheap to start, expensive to continue."""
    reqs = conversation([(500, 5), (20, 5), (20, 5)], batch=1, context_bucket=1)
    pre = [r for r in reqs if r.mode == "prefill"]
    dec = [r for r in reqs if r.mode == "decode"]

    assert pre[0].seqlen > pre[1].seqlen == pre[2].seqlen
    assert dec[0].seqlen < dec[-1].seqlen


# --- validation ------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    dict(batch=1, prompt_len=0, n_new_tokens=4),
    dict(batch=1, prompt_len=64, n_new_tokens=0),
])
def test_generation_validates_its_inputs(kwargs):
    with pytest.raises(ValueError):
        generation(**kwargs)


def test_conversation_validates_its_turns():
    with pytest.raises(ValueError):
        conversation([(0, 4)])
    with pytest.raises(ValueError):
        conversation([(10, 0)])
