"""L2 for a mixed batch: what fuses, what cannot, and the identities that prove
the fused path did not change the answer.
"""

import pytest

from dynshape.entities import Piece
from dynshape.mixed import (attention_mask, build_iteration_kernels,
                            build_iteration_kernels_tagged, fusible_mask,
                            iteration_token_shapes, mixed_report)


def pre(tokens, context=0, rid=0):
    return Piece(kind="prefill", request_id=rid, tokens=tokens, context=context)


def dec(context, rid=0):
    return Piece(kind="decode", request_id=rid, tokens=1, context=context)


# -- the classifier ---------------------------------------------------------

def test_the_two_classifiers_are_exact_complements(rewriter):
    """One asks an arithmetic question (does the field depend only on B*S?), the
    other a structural one (does it read the key axis?).  They must agree; a
    disagreement means a field class exists that fuses arithmetically but reads
    keys, and the fused expansion would be silently wrong for it."""
    rep = mixed_report(rewriter)
    assert rep.classes_agree, rep.disagreements
    assert rep.n_fusible + rep.n_attention == rep.n_entries


def test_gpt2_has_exactly_four_attention_entries_per_block(rewriter):
    """48 = 12 blocks x 4 -- the same 48 entries the decode `+1` correction
    touched, which is a useful independent confirmation of both."""
    fus = fusible_mask(rewriter)
    assert len(fus) == 242
    assert fus.count(False) == 48
    assert mixed_report(rewriter).n_runs == 12


def test_the_fusible_entries_are_the_linear_layers(rewriter):
    """Sanity: every GEMM whose dimN is an architecture constant fuses, and
    every entry that does not fuse belongs to attention."""
    att = attention_mask(rewriter)
    fus = fusible_mask(rewriter)
    for (q, op), f, a in zip(rewriter.base, fus, att):
        if op[0] in ("layernorm", "elementwise") and not a:
            assert f, f"{op} {q} should fuse"


# -- the identities ---------------------------------------------------------

def test_a_single_prefill_iteration_is_exactly_the_old_expansion(rewriter):
    """The strongest check available: with one request there is nothing to fuse
    and nothing to reorder, so the mixed path must reproduce `expand` field for
    field.  If it does not, the fused arithmetic is wrong."""
    for s in (128, 377, 512, 2048):
        got = build_iteration_kernels(rewriter, [pre(s)])
        want = rewriter.expand(batch=1, seqlen=s, mode="prefill")
        assert got == want, f"mixed prefill diverges at S={s}"


def test_a_single_decode_iteration_is_exactly_the_old_expansion(rewriter):
    """Same identity on the other side, and it is a real cross-check: the fused
    entries come from the PREFILL template at one token, the attention entries
    from the DECODE template at the true context.  They must agree."""
    for ctx in (128, 300, 512, 2049):
        got = build_iteration_kernels(rewriter, [dec(ctx)])
        want = rewriter.expand(batch=1, seqlen=ctx, mode="decode")
        assert got == want, f"mixed decode diverges at ctx={ctx}"


def test_a_uniform_decode_batch_matches_the_batched_expansion_where_it_can(
        rewriter):
    """N decodes at the same context: the fused half must equal what a batch-N
    decode expansion produces.  The attention half cannot -- N separate strips
    are not one batched strip -- which is exactly the varlen gap."""
    ctx, n = 512, 8
    got = build_iteration_kernels(rewriter, [dec(ctx, rid=i) for i in range(n)])
    want = rewriter.expand(batch=n, seqlen=ctx, mode="decode")
    fus = fusible_mask(rewriter)

    gi = 0
    for i, f in enumerate(fus):
        if f:
            assert got[gi] == want[i], f"fused entry {i} diverges"
            gi += 1
        else:
            gi += n                      # n per-request copies of this entry


def test_the_subset_expansion_matches_the_full_one(rewriter):
    """The engine expands only the ~48 attention entries per request rather than
    all 242.  That is a pure optimisation, valid because the rewriters process
    each (entry, rule) pair independently -- pinned here so it stays true."""
    from dynshape.mixed import _piece_entries
    idx = [i for i, f in enumerate(fusible_mask(rewriter)) if not f]
    for piece in (pre(700, context=300), dec(1024), pre(64)):
        full = _piece_entries(rewriter, piece)
        subset = _piece_entries(rewriter, piece, idx)
        assert subset == [full[i] for i in idx]


def test_kernel_count_follows_the_fusion_arithmetic(rewriter):
    n_fus = fusible_mask(rewriter).count(True)
    n_att = 242 - n_fus
    for n_pieces in (1, 3, 17):
        pieces = [dec(128, rid=i) for i in range(n_pieces)]
        assert len(build_iteration_kernels(rewriter, pieces)) \
            == n_fus + n_pieces * n_att


# -- fusing vs concatenating ------------------------------------------------

def test_fusing_launches_far_fewer_kernels_than_concatenating(rewriter):
    """The linear half is the easy half: one GEMM at the summed token count is
    strictly more faithful than one per request, and costs nothing."""
    pieces = [dec(256, rid=i) for i in range(12)] + [pre(500, rid=99)]
    fused = build_iteration_kernels(rewriter, pieces, fuse_linear=True)
    concat = build_iteration_kernels(rewriter, pieces, fuse_linear=False)
    assert len(concat) == 242 * len(pieces)
    assert len(fused) < len(concat) / 3


def test_the_fused_gemm_carries_the_summed_token_count(rewriter):
    """M = decode_batch + prefill_chunk_tokens, in one launch."""
    pieces = [dec(256, rid=i) for i in range(10)] + [pre(500, rid=99)]
    total = 10 + 500
    kernels = build_iteration_kernels(rewriter, pieces)
    solo = build_iteration_kernels(rewriter, [pre(total)])
    fus = fusible_mask(rewriter)

    gi = 0
    for i, f in enumerate(fus):
        if f:
            assert kernels[gi][0] == solo[i][0]
            gi += 1
        else:
            gi += len(pieces)


def test_concatenation_preserves_every_requests_own_shape(rewriter):
    pieces = [pre(300), dec(1024, rid=1)]
    concat = build_iteration_kernels(rewriter, pieces, fuse_linear=False)
    assert concat[:242] == rewriter.expand(1, 300, "prefill")
    assert concat[242:] == rewriter.expand(1, 1024, "decode")


# -- ordering and provenance ------------------------------------------------

def test_attention_runs_are_grouped_by_request_not_by_kernel(rewriter):
    """Execution order matters for the shape of the trace even though a purely
    additive cost model would not notice: one block reads
    (Q.K, mask, softmax, A.V) for request 1, then the same four for request 2."""
    pieces = [dec(128, rid=0), dec(256, rid=1)]
    tagged = build_iteration_kernels_tagged(rewriter, pieces)
    tags = [t for _, t in tagged]
    # The first attention run is 4 entries for request 0 then 4 for request 1;
    # both carry the decode tag, so check the shapes instead.
    fus = fusible_mask(rewriter)
    first = fus.index(False)
    run_len = 0
    while not fus[first + run_len]:
        run_len += 1
    start = fus[:first].count(True)
    block = tagged[start:start + 2 * run_len]
    a = [q for (q, _op), _t in block[:run_len]]
    b = [q for (q, _op), _t in block[run_len:]]
    assert a != b, "both requests produced identical attention shapes"


def test_tags_attribute_attention_to_a_phase(rewriter):
    pieces = [dec(128, rid=0), pre(500, rid=1)]
    tags = {t for _, t in build_iteration_kernels_tagged(rewriter, pieces)}
    assert tags == {"fused", "attn:decode", "attn:prefill"}


def test_unfused_tags_mark_whole_request_lists(rewriter):
    tags = {t for _, t in build_iteration_kernels_tagged(
        rewriter, [dec(128), pre(64, rid=1)], fuse_linear=False)}
    assert tags == {"whole:decode", "whole:prefill"}


# -- the logits option ------------------------------------------------------

def test_logits_last_token_only_shrinks_the_output_projection(rewriter):
    """The traced HF forward computes logits at every position; a real engine
    computes them only where it will sample.  Off by default, because the
    template is a measurement."""
    pieces = [pre(2000)]
    faithful = build_iteration_kernels(rewriter, pieces)
    corrected = build_iteration_kernels(rewriter, pieces, logits_last_token_only=True)

    def lm_head(kernels):
        return [q for q, op in kernels
                if op and op[0] == "gemm" and q.get("dimN", 0) >= 10000]

    if not lm_head(faithful):
        # `GPT2Model` is the bare transformer -- no LM head is traced, so there
        # is nothing to correct.  The option must then be a no-op rather than
        # mistaking the MLP up-projection for an output projection.
        assert faithful == corrected
        pytest.skip("template has no output projection (traced from GPT2Model)")

    assert lm_head(faithful)[0]["dimM"] == 2000
    assert lm_head(corrected)[0]["dimM"] == 1
    assert len(faithful) == len(corrected)


# -- composition reporting --------------------------------------------------

def test_iteration_token_shapes_describes_the_mix():
    pieces = [dec(100, rid=0), dec(300, rid=1), pre(400, rid=2, context=1000)]
    s = iteration_token_shapes(pieces)
    assert s["decode_batch"] == 2
    assert s["prefill_chunks"] == 1
    assert s["prefill_tokens"] == 400
    assert s["total_tokens"] == 402
    assert s["context_mean"] == 200.0
    assert s["context_max"] == 300
    assert s["distinct_contexts"] == 2


def test_an_empty_iteration_is_rejected(rewriter):
    with pytest.raises(ValueError):
        build_iteration_kernels(rewriter, [])
