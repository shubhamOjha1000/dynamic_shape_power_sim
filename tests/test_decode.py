"""`mode` as a real third axis, not a rescaling.

Decode has no shipped trace anywhere in the EnergAIzer artifact -- all 90
workload files are `modeprefill`.  So these tests do the only two honest things
available: check that the query/key split *reduces to prefill exactly* when
Sq == Sk (a strong constraint, since prefill is verified against real traces),
and check that decode shapes are internally consistent with what
`run_model.get_input()` actually feeds the model in decode mode (one token per
sequence, plus a length-`seqlen` KV cache).
"""

import pytest

from dynshape import rewrite_dims, split_seq_exponent

HEADS = 12          # GPT-2 base
HIDDEN = 768
HEAD_DIM = 64


# --- the split rule itself -------------------------------------------------

def test_split_rule_cases():
    gemm = ("gemm", "tc", "bf16_bf16")
    assert split_seq_exponent(gemm, "dimN", 0) == (0, 0)      # architecture constant
    assert split_seq_exponent(gemm, "dim", 2) == (1, 1)       # B*h*Sq*Sk score tensor
    assert split_seq_exponent(gemm, "dimM", 1) == (1, 0)      # query rows
    assert split_seq_exponent(gemm, "dimN", 1) == (0, 1)      # key columns
    assert split_seq_exponent(gemm, "dimK", 1) == (0, 1)      # key inner axis
    assert split_seq_exponent(("softmax", "bf16"), "dim", 1) == (0, 1)
    assert split_seq_exponent(("softmax", "bf16"), "batch", 1) == (1, 0)
    assert split_seq_exponent(("elementwise",), "dim", 1) == (1, 0)


def test_split_rule_rejects_unmodelled_exponent():
    with pytest.raises(ValueError, match="outside the modelled"):
        split_seq_exponent(("gemm",), "dimM", 3)


def test_every_field_in_the_template_has_a_split_rule(rewriter):
    """Guards the moment a new model introduces a field class we cannot split."""
    for (q, op), r in zip(rewriter.base, rewriter.rules):
        for k, (a, b) in r.items():
            p, kk = split_seq_exponent(op, k, b)
            assert p + kk == b, f"{op} {k}: split {p}+{kk} does not sum to {b}"


# --- the reduction property ------------------------------------------------

@pytest.mark.parametrize("batch,seqlen", [(1, 128), (8, 512), (13, 377), (32, 4096)])
def test_three_axis_reduces_to_two_axis(rewriter, batch, seqlen):
    """With Sq == Sk the split must be invisible -- which ties the (unvalidated)
    decode path to the (validated) prefill path."""
    two = rewrite_dims(rewriter.base, rewriter.rules, rewriter.b0, rewriter.s0,
                       batch, seqlen)
    assert rewriter.expand(batch, seqlen, "prefill") == two


# --- decode structure ------------------------------------------------------

@pytest.mark.parametrize("batch,context", [(1, 128), (4, 1000), (32, 2048)])
def test_decode_token_count_is_one_per_sequence(rewriter, batch, context):
    """Decode pushes B tokens, not B*context -- the whole reason it is cheap."""
    dec = rewriter.expand(batch, context, "decode")

    # The projection GEMMs are unbatched with M = token count.
    proj_m = {q["dimM"] for q, op in dec if op[0] == "gemm" and q.get("batch") == 1}
    assert proj_m == {batch}, f"expected M == {batch} tokens, got {proj_m}"

    # layernorm rows are also the token count.
    ln_rows = {q["batch"] for q, op in dec if op[0] == "layernorm"}
    assert ln_rows == {batch}


@pytest.mark.parametrize("batch,context", [(1, 128), (4, 1000), (32, 2048)])
def test_decode_attention_is_a_strip_not_a_square(rewriter, batch, context):
    """prefill: M = S, N = S.  decode: M = 1, N = context."""
    dec = rewriter.expand(batch, context, "decode")
    attn = [q for q, op in dec if op[0] == "gemm" and q.get("batch") == batch * HEADS]
    assert attn, "no batched attention GEMMs found"

    for q in attn:
        assert q["dimM"] == 1, f"decode query length must be 1, got {q['dimM']}"

    # Q.K^T : (1 x head_dim) @ (head_dim x context)
    qk = [q for q in attn if q["dimK"] == HEAD_DIM]
    assert qk and all(q["dimN"] == context for q in qk)

    # scores.V : (1 x context) @ (context x head_dim)
    av = [q for q in attn if q["dimN"] == HEAD_DIM]
    assert av and all(q["dimK"] == context for q in av)

    assert len(qk) == len(av) == HEADS, "one of each per transformer block"


@pytest.mark.parametrize("batch,context", [(2, 256), (8, 1024)])
def test_decode_softmax_and_mask_match_the_attention_shape(rewriter, batch, context):
    """Cross-entry consistency: what attention produces is what softmax consumes."""
    dec = rewriter.expand(batch, context, "decode")

    sm = [q for q, op in dec if op[0] == "softmax"]
    assert sm and all(q["dim"] == context for q in sm), "softmax reduces over context"
    assert all(q["batch"] == batch * HEADS for q in sm), "one row per head per sequence"

    # The score tensor carried by the (1,1) elementwise op is B*h*Sq*Sk.
    scores = [q["dim"] for q, op in dec if op[0] == "elementwise"
              if q["dim"] == batch * HEADS * context]
    assert scores, "no B*heads*1*context score tensor found in decode"


def test_decode_is_far_cheaper_than_prefill_at_the_same_length(rewriter):
    """Sanity in the direction everyone expects: same context, 1 token vs S."""
    b, s = 8, 1024
    pre = rewriter.expand(b, s, "prefill")
    dec = rewriter.expand(b, s, "decode")
    assert len(pre) == len(dec)

    def gemm_flops(entries):
        return sum(q.get("batch", 1) * q["dimM"] * q["dimN"] * q["dimK"]
                   for q, op in entries if op[0] == "gemm")

    ratio = gemm_flops(pre) / gemm_flops(dec)
    assert ratio > 100, f"decode should be orders of magnitude cheaper, ratio was {ratio:.1f}"


def test_decode_context_one_is_the_degenerate_case(rewriter):
    """context == 1 must still produce runnable (>= 1) dimensions everywhere."""
    dec = rewriter.expand(4, 1, "decode")
    for q, op in dec:
        for k, v in q.items():
            if isinstance(v, int) and not isinstance(v, bool):
                assert v >= 1, f"{op} {k} collapsed to {v}"
