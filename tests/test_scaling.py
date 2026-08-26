"""The core claim: three traces are enough to reproduce every other shape.

If `test_rewrite_matches_every_shipped_template` passes, the dynamic-shape
machinery is exact on all 25 shapes anybody actually measured -- which is the
cheapest strong evidence available anywhere in this project.
"""

import json

import pytest

from conftest import (B0, S0, SHIPPED_BATCHES, SHIPPED_SEQLENS, TEMPLATE_DIR,
                      template_path)
from dynshape import (ShapeRewriter, learn_scaling, load_template,
                      parse_shape_from_name, rewrite_dims)


# --- filename parsing ------------------------------------------------------

def test_parse_shape_from_name():
    assert parse_shape_from_name(template_path(8, 128)) == (8, 128, "prefill")
    assert parse_shape_from_name("x_b32_s4096_modedecode.json") == (32, 4096, "decode")


def test_parse_shape_rejects_junk():
    with pytest.raises(ValueError):
        parse_shape_from_name("not_a_workload.json")


# --- the learned rules -----------------------------------------------------

def test_template_shape(rewriter):
    assert rewriter.n_kernels() == 242, "GPT-2 traces to 242 kernels"
    assert (rewriter.b0, rewriter.s0) == (B0, S0)


def test_learned_exponents_are_the_expected_classes(rewriter):
    """Every (op, field) class the template contains, against what the
    architecture implies.  A field name can carry more than one class -- GEMM
    `dimN` is an architecture constant in the projections and the key axis in
    attention -- so these are sets, not single values."""
    classes = {}
    for op, fieldname, a, b, _, _ in rewriter.field_report():
        classes.setdefault((op, fieldname), set()).add((a, b))

    GEMM = ("gemm", "tc", "bf16_bf16")
    # Projection / MLP GEMMs are unbatched with M = tokens = B*S; attention
    # GEMMs are batched by B*n_heads with M = query length.
    assert classes[(GEMM, "dimM")] == {(1, 1), (0, 1)}
    assert classes[(GEMM, "batch")] == {(0, 0), (1, 0)}
    # N and K are 3*hidden / hidden in the projections, and the key axis in
    # attention.
    assert classes[(GEMM, "dimN")] == {(0, 0), (0, 1)}
    assert classes[(GEMM, "dimK")] == {(0, 0), (0, 1)}
    # layernorm rows are tokens; its width is the hidden size.
    assert classes[(("layernorm", "bf16"), "dim")] == {(0, 0)}
    assert classes[(("layernorm", "bf16"), "batch")] == {(1, 1)}
    # softmax reduces over the key axis, one row per head per query token.
    assert classes[(("softmax", "bf16"), "dim")] == {(0, 1)}
    assert classes[(("softmax", "bf16"), "batch")] == {(1, 1)}
    # elementwise carries both the token tensors (B*S*hidden) and the score
    # tensors (B*heads*S*S) -- the only place a squared exponent appears.
    assert classes[(("elementwise",), "dim")] == {(1, 1), (1, 2)}


def test_booleans_and_strings_are_never_rescaled(rewriter):
    """`isinstance(True, int)` is True in Python -- the classic way to corrupt
    `useTensorCore` into a rescaled integer."""
    for r in rewriter.rules:
        assert "useTensorCore" not in r
        assert "precM" not in r and "precA" not in r and "prec" not in r and "op" not in r

    out = rewriter.expand(batch=32, seqlen=512, mode="prefill")
    for q, op in out:
        if "useTensorCore" in q:
            assert q["useTensorCore"] is True
        if "prec" in q:
            assert isinstance(q["prec"], str)


# --- the headline test -----------------------------------------------------

@pytest.mark.parametrize("batch", SHIPPED_BATCHES)
@pytest.mark.parametrize("seqlen", SHIPPED_SEQLENS)
def test_rewrite_matches_every_shipped_template(rewriter, batch, seqlen):
    """Rewritten shapes must equal the independently traced file, field for field."""
    generated = rewriter.expand(batch, seqlen, mode="prefill")
    reference = load_template(template_path(batch, seqlen))

    assert len(generated) == len(reference)
    for i, ((gq, gop), (rq, rop)) in enumerate(zip(generated, reference)):
        assert gop == rop, f"entry {i}: op mismatch"
        assert gq == rq, f"entry {i}: {gq} != {rq}"


def test_two_axis_rewrite_agrees_with_expand(rewriter):
    """`rewrite_dims` (2-axis) and `expand(mode='prefill')` (3-axis) coincide."""
    two = rewrite_dims(rewriter.base, rewriter.rules, rewriter.b0, rewriter.s0, 16, 1024)
    three = rewriter.expand(16, 1024, mode="prefill")
    assert two == three


def test_untraced_shape_is_produced(rewriter):
    """A shape nobody ever traced -- the entire point."""
    out = rewriter.expand(batch=13, seqlen=377, mode="prefill")
    assert len(out) == 242
    tokens = 13 * 377
    gemm_m = [q["dimM"] for q, op in out if op[0] == "gemm" and q.get("batch") == 1]
    assert gemm_m and all(m == tokens for m in gemm_m)


def test_identity_rewrite(rewriter):
    assert rewriter.expand(B0, S0, mode="prefill") == rewriter.base


# --- failure modes ---------------------------------------------------------

def test_non_integer_exponent_raises_loudly():
    base = [({"dim": 100}, ("elementwise",))]
    alt_b = [({"dim": 200}, ("elementwise",))]
    alt_s = [({"dim": 150}, ("elementwise",))]     # log2(1.5) is not an integer
    with pytest.raises(ValueError, match="non-integer exponent"):
        learn_scaling(base, alt_b, alt_s, batch_ratio=2, seq_ratio=2)


def test_mismatched_trace_lengths_raise():
    base = [({"dim": 100}, ("elementwise",))]
    with pytest.raises(ValueError, match="differ in length"):
        learn_scaling(base, base * 2, base, batch_ratio=2, seq_ratio=2)


def test_mismatched_op_sequence_raises():
    base = [({"dim": 100}, ("elementwise",))]
    other = [({"dim": 200}, ("softmax", "bf16"))]
    with pytest.raises(ValueError, match="op type differs"):
        learn_scaling(base, other, base, batch_ratio=2, seq_ratio=2)


def test_unit_ratio_raises():
    base = [({"dim": 100}, ("elementwise",))]
    with pytest.raises(ValueError, match="ratios must differ"):
        learn_scaling(base, base, base, batch_ratio=1, seq_ratio=2)


def test_bad_mode_and_bad_dims_raise(rewriter):
    with pytest.raises(ValueError, match="mode must be"):
        rewriter.expand(4, 128, mode="mixed")
    with pytest.raises(ValueError, match=">= 1"):
        rewriter.expand(0, 128, mode="prefill")
    with pytest.raises(ValueError, match=">= 1"):
        rewriter.expand(4, 0, mode="prefill")


def test_from_files_rejects_wrong_anchors():
    with pytest.raises(ValueError, match="hold seqlen fixed"):
        ShapeRewriter.from_files(template_path(8, 128), template_path(16, 512),
                                 template_path(8, 512))
    with pytest.raises(ValueError, match="hold batch fixed"):
        ShapeRewriter.from_files(template_path(8, 128), template_path(16, 128),
                                 template_path(16, 512))


def test_alternate_anchor_pair_gives_identical_rules():
    """Learning from a different pair of anchors must recover the same law."""
    a = ShapeRewriter.from_files(template_path(8, 128), template_path(16, 128),
                                 template_path(8, 512))
    b = ShapeRewriter.from_files(template_path(8, 128), template_path(32, 128),
                                 template_path(8, 4096))
    assert a.rules == b.rules
    assert a.expand(2, 2048, "prefill") == b.expand(2, 2048, "prefill")
