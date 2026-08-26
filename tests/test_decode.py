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

from dynshape import load_template, rewrite_dims, split_seq_exponent

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
    """prefill: M = S, N = S.  decode: M = 1, N = context + 1.

    The +1 is MEASURED, not assumed: the generated token's own K/V are appended
    to the cache before attention runs, so it attends over one more key than the
    cache holds.  A GPT-2 decode trace at a nominal context of 128 shows
    attention dimN = 129 and softmax dim = 129.
    """
    dec = rewriter.expand(batch, context, "decode")
    attn = [q for q, op in dec if op[0] == "gemm" and q.get("batch") == batch * HEADS]
    assert attn, "no batched attention GEMMs found"

    for q in attn:
        assert q["dimM"] == 1, f"decode query length must be 1, got {q['dimM']}"

    keys = context + 1

    # Q.K^T : (1 x head_dim) @ (head_dim x keys)
    qk = [q for q in attn if q["dimK"] == HEAD_DIM]
    assert qk and all(q["dimN"] == keys for q in qk)

    # scores.V : (1 x keys) @ (keys x head_dim)
    av = [q for q in attn if q["dimN"] == HEAD_DIM]
    assert av and all(q["dimK"] == keys for q in av)

    assert len(qk) == len(av) == HEADS, "one of each per transformer block"


@pytest.mark.parametrize("batch,context", [(2, 256), (8, 1024)])
def test_decode_softmax_and_mask_match_the_attention_shape(rewriter, batch, context):
    """Cross-entry consistency: what attention produces is what softmax consumes."""
    dec = rewriter.expand(batch, context, "decode")

    keys = context + 1
    sm = [q for q, op in dec if op[0] == "softmax"]
    assert sm and all(q["dim"] == keys for q in sm), "softmax reduces over the keys"
    assert all(q["batch"] == batch * HEADS for q in sm), "one row per head per sequence"

    # The score tensor carried by the (1,1) elementwise op is B*h*Sq*Sk.
    scores = [q["dim"] for q, op in dec if op[0] == "elementwise"
              if q["dim"] == batch * HEADS * keys]
    assert scores, "no B*heads*1*(context+1) score tensor found in decode"


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


# --- the second law: real decode traces, when they exist -------------------

import json
import os

from dynshape import ShapeRewriter
from conftest import TEMPLATE_DIR, template_path


def _write(path, entries):
    with open(path, "w") as f:
        json.dump([[q, list(op)] for q, op in entries], f)


def _fake_decode_templates(tmp_path, n_entries=2, batch_scale=1, ctx_scale=1):
    """Hand-made decode anchors so the *plumbing* can be tested without
    pretending to own real decode measurements.

    Two entries whose fields scale as (batch^1, ctx^0) and (batch^1, ctx^1).
    """
    def make(b, s):
        return [({"batch": 12 * b, "dimM": 1, "dimN": s, "dimK": 64,
                  "precM": "bf16", "precA": "bf16", "useTensorCore": True},
                 ("gemm", "tc", "bf16_bf16")),
                ({"batch": 12 * b, "dim": s, "prec": "bf16"},
                 ("softmax", "bf16"))][:n_entries]

    for b, s in ((8, 128), (16, 128), (8, 512)):
        _write(os.path.join(tmp_path, f"gpt2model_gpt2_pbf16_b{b}_s{s}_modedecode.json"),
               make(b, s))
    return [os.path.join(tmp_path, f"gpt2model_gpt2_pbf16_b{b}_s{s}_modedecode.json")
            for b, s in ((8, 128), (16, 128), (8, 512))]


def test_shipped_decode_traces_are_measured(rewriter):
    """The repo now ships real decode anchors, so decode is no longer inferred."""
    assert rewriter.decode_source == "measured"
    assert rewriter.decode_base is not None
    assert rewriter.n_kernels("decode") == rewriter.n_kernels("prefill") == 242
    assert rewriter.decode_seq_offset == 1, "the +1 key must be detected from the traces"


def test_falls_back_to_inferred_without_decode_traces(tmp_path):
    """Remove the decode anchors and the inferred split takes over again."""
    import shutil
    for b, s in [(8, 128), (16, 128), (8, 512)]:
        shutil.copy(template_path(b, s), str(tmp_path))
    rw = ShapeRewriter.from_dir(str(tmp_path))
    assert rw.decode_source == "inferred"
    assert rw.decode_base is None
    # and it must still agree with the measurement it was corrected against
    assert rw.expand(8, 128, "decode") == rewriter_measured_expand()


def rewriter_measured_expand():
    from conftest import TEMPLATE_DIR
    return ShapeRewriter.from_dir(TEMPLATE_DIR).expand(8, 128, "decode")


def test_measured_decode_law_reproduces_the_holdout():
    """The independent check: predict a real trace never used to learn the law.

    The three anchors in templates/gpt2/ fit the law; tests/holdout/ holds a
    fourth real trace that `from_dir` cannot see. If this passes, decode
    generalises rather than being memorised.
    """
    from conftest import HOLDOUT_DIR, TEMPLATE_DIR

    rw = ShapeRewriter.from_dir(TEMPLATE_DIR)
    assert rw.decode_source == "measured"

    path = os.path.join(HOLDOUT_DIR,
                        "gpt2model_gpt2_pbf16_b16_s512_modedecode.json")
    held = load_template(path)
    assert rw.expand(16, 512, "decode") == held, "held-out decode trace not reproduced"


def test_holdout_is_not_reachable_as_an_anchor():
    """Guard the independence property itself."""
    from conftest import HOLDOUT_DIR, TEMPLATE_DIR

    assert os.path.abspath(HOLDOUT_DIR) != os.path.abspath(TEMPLATE_DIR)
    assert not [f for f in os.listdir(TEMPLATE_DIR) if "b16_s512_modedecode" in f], (
        "the held-out trace has been copied into templates/ -- the independent "
        "decode check is now a memorisation test")


def test_decode_traces_take_over_when_present(tmp_path):
    paths = _fake_decode_templates(str(tmp_path))
    rw = ShapeRewriter.from_files(
        template_path(8, 128), template_path(16, 128), template_path(8, 512),
        decode_paths=paths)

    assert rw.decode_source == "measured"
    assert rw.n_kernels("prefill") == 242
    assert rw.n_kernels("decode") == 2, "decode uses its OWN template, not prefill's"

    # Prefill is untouched by the presence of a decode law.
    assert rw.expand(32, 512, "prefill") == rewriter_prefill(rw, 32, 512)

    # Decode now comes from the decode law: scaled from its own (b8, s128).
    dec = rw.expand(16, 512, "decode")
    assert dec[0][0]["batch"] == 12 * 16
    assert dec[0][0]["dimN"] == 512
    assert dec[0][0]["dimM"] == 1


def rewriter_prefill(rw, b, s):
    from dynshape import rewrite_dims
    return rewrite_dims(rw.base, rw.rules, rw.b0, rw.s0, b, s)


def test_from_dir_picks_up_decode_templates_automatically(tmp_path):
    import shutil
    for b, s in ((8, 128), (16, 128), (8, 512)):
        shutil.copy(template_path(b, s), str(tmp_path))
    assert ShapeRewriter.from_dir(str(tmp_path)).decode_source == "inferred"

    _fake_decode_templates(str(tmp_path))
    assert ShapeRewriter.from_dir(str(tmp_path)).decode_source == "measured"


def test_a_differing_kernel_count_is_reported_not_swallowed(tmp_path, capsys):
    """If real decode traces to a different number of kernels, the inferred
    rule was wrong -- and that must be visible, not silently absorbed."""
    paths = _fake_decode_templates(str(tmp_path), n_entries=1)
    ShapeRewriter.from_files(template_path(8, 128), template_path(16, 128),
                             template_path(8, 512), decode_paths=paths)
    out = capsys.readouterr().out
    assert "kernel list DOES differ" in out
    assert "1 kernels" in out and "242" in out


def test_decode_paths_are_validated(tmp_path):
    paths = _fake_decode_templates(str(tmp_path))
    with pytest.raises(ValueError, match="exactly three"):
        ShapeRewriter.from_files(template_path(8, 128), template_path(16, 128),
                                 template_path(8, 512), decode_paths=paths[:2])
    with pytest.raises(ValueError, match="must be decode traces"):
        ShapeRewriter.from_files(
            template_path(8, 128), template_path(16, 128), template_path(8, 512),
            decode_paths=[template_path(8, 128), template_path(16, 128),
                          template_path(8, 512)])


def test_anchor_modes_must_agree(tmp_path):
    paths = _fake_decode_templates(str(tmp_path))
    with pytest.raises(ValueError, match="same mode"):
        ShapeRewriter._learn_from_anchors(template_path(8, 128), paths[1],
                                          template_path(8, 512))


def test_the_key_axis_is_context_plus_one(rewriter):
    """Pin the +1 on its own, with its provenance.

    Tracing GPT-2 decode found 48 of 242 entries differing from the inferred
    rule -- 12 blocks x 4 attention entries -- every one of them off by exactly
    one token. This is the assertion that keeps that from regressing.
    """
    for ctx in (1, 127, 128, 1000, 4096):
        dec = rewriter.expand(4, ctx, "decode")
        qk = next(q for q, op in dec
                  if op[0] == "gemm" and q.get("batch") == 4 * HEADS
                  and q["dimK"] == HEAD_DIM)
        assert qk["dimN"] == ctx + 1, f"context {ctx} must give {ctx + 1} keys"


def test_measured_decode_law_detects_the_offset(tmp_path, rewriter):
    """A decode law learned from real traces must recover offset = 1 itself.

    Without it `learn_scaling` refuses -- 129 -> 513 is affine in S, giving an
    exponent of 0.9958 rather than 1 -- which is exactly what the first real
    decode traces produced.
    """
    import json
    import shutil

    for b, s in [(8, 128), (16, 128), (8, 512)]:
        shutil.copy(template_path(b, s), str(tmp_path))
        json.dump([[q, list(op)] for q, op in rewriter.expand(b, s, "decode")],
                  open(os.path.join(str(tmp_path),
                       f"gpt2model_gpt2_pbf16_b{b}_s{s}_modedecode.json"), "w"))

    rw = ShapeRewriter.from_dir(str(tmp_path))
    assert rw.decode_source == "measured"
    assert rw.decode_seq_offset == 1, "the +1 must be detected, not configured"

    # And it must generalise to a shape none of the three anchors covers.
    assert rw.expand(16, 512, "decode") == rewriter.expand(16, 512, "decode")
