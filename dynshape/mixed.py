"""
mixed.py
========

**L2 for a mixed batch**: one iteration's `Piece` list -> a kernel list.

This is the seam between the scheduler and the shape rewriter, and the only
genuinely new modelling in this layer.  Chunked prefill means most iterations
contain decodes *and* prefill chunks in the same forward pass, and the two do
not share a shape.

WHAT FUSES, AND WHAT CANNOT
---------------------------
The template is a flat list of 242 entries in execution order.  Every entry
falls into exactly one of two classes, and the class is **derived from the
learned exponents**, not hard-coded per kernel family:

    value = const * B^a * S^b

    a == b   the field depends only on the *product* B*S -- the token count.
             Set B=1 and S=total_tokens and the value is exact, whatever mix of
             requests produced those tokens.  Projections, MLP, layernorms,
             elementwise: everything except attention.

    a != b   batch and sequence enter separately, so there is no single token
             count that reproduces the value.  These are the attention entries,
             and they must be emitted **per request**.

That is not a coincidence.  A linear layer sees a bag of token rows and does not
care which request each row came from; attention cares about nothing else.  So
the arithmetic classifier and the physical intuition give the same answer, and
for GPT-2 it lands on exactly 48 entries -- 12 blocks x 4 attention kernels,
the same 48 that the decode `+1` correction touched.

Fusing the linear layers is **exact, not an approximation**.  Emitting one GEMM
at `M = summed tokens` is strictly more faithful than emitting one per request,
and it costs nothing.

WHERE THIS STILL DIVERGES FROM A REAL ENGINE
--------------------------------------------
Attention cannot be fused this way.  vLLM issues a single ragged launch --
`flash_attn_varlen_func` with `cu_seqlens_q` / `cu_seqlens_k` -- whose "shape" is
a pair of offset arrays, each request contributing its own query length (chunk
size, or 1 for decode) and its own KV length.  EnergAIzer's LUT is keyed on
rectangles, so a ragged launch can only be approximated as the sum of its
per-request rectangles, losing the shared launch and the tail quantisation.

Under **eager** attention there is nothing to lose: HuggingFace runs per-request
kernels anyway, so this *is* what happens, and every shape involved is measured.
That is the self-consistent choice for v1.  The cost is that it drifts further
from vLLM -- four kernels each paying their own launch, HBM round-trip and
low-occupancy tail where vLLM pays one fused pass -- so expect systematic
over-prediction of decode time.

ORDERING
--------
Kernels are emitted in true execution order.  A contiguous run of attention
entries is replayed **once per request** before the next fused entry, so one
transformer block reads: fused layernorm, fused QKV, then (Q.K, mask, softmax,
A.V) for request 1, the same four for request 2, ..., then the fused output
projection.  Grouping by request rather than by kernel matters for the shape of
the trace even though a purely additive cost model would not notice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .entities import Piece
from .template import Entry, ShapeRewriter, rewrite_dims, rewrite_dims_qk


#: Smallest `dimN` that could plausibly be a vocabulary rather than a hidden
#: dimension.  GPT-2's is 50257; its widest hidden dimension is 3072.
_MIN_VOCAB = 10000


def fusible_mask(rewriter: ShapeRewriter) -> List[bool]:
    """Per template entry: can it be expanded once at the summed token count?

    True iff every scalable field has `a == b`, i.e. depends only on `B * S`.
    Constants (0, 0) qualify trivially.

    Memoised on the rewriter: the engine asks this once per iteration and the
    answer depends only on the learned rules, which never change after
    construction.
    """
    cached = getattr(rewriter, "_fusible_cache", None)
    if cached is not None:
        return cached
    mask = [all(a == b for (a, b) in rule.values()) for rule in rewriter.rules]
    try:
        object.__setattr__(rewriter, "_fusible_cache", mask)
    except Exception:      # a frozen dataclass without the slot -- just recompute
        pass
    return mask


def attention_mask(rewriter: ShapeRewriter) -> List[bool]:
    """Per template entry: does it read the key axis?

    Independent of `fusible_mask` -- computed from the query/key split rather
    than from `a == b`.  The two should be exact complements; `mixed_report`
    checks that they are, and a mismatch means a new model has introduced a
    field class this layer has no rule for.
    """
    from .template import split_seq_exponent
    out = []
    for (q, op), rule in zip(rewriter.base, rewriter.rules):
        out.append(any(split_seq_exponent(op, k, b)[1] != 0
                       for k, (a, b) in rule.items()))
    return out


def _lm_head_indices(rewriter: ShapeRewriter,
                     vocab: Optional[int] = None) -> List[int]:
    """Entries whose `dimN` is the vocabulary -- the output projection.

    `vocab=None` auto-detects it as the largest constant `dimN` in the template,
    which beats hard-coding 50257: a padded vocabulary (50304 is common) would
    silently match nothing.
    """
    gemms = [(i, q) for i, (q, op) in enumerate(rewriter.base)
             if op and op[0] == "gemm" and "dimN" in q]
    if not gemms:
        return []
    if vocab is None:
        constant = [q["dimN"] for i, q in gemms
                    if rewriter.rules[i].get("dimN", (0, 0)) == (0, 0)]
        # A vocabulary is an order of magnitude wider than any hidden dimension.
        # Without that floor, a template traced from `GPT2Model` -- which has no
        # LM head at all -- would hand back the MLP up-projection instead and the
        # correction would silently mangle it.
        candidates = [n for n in constant if n >= _MIN_VOCAB]
        if not candidates:
            return []
        vocab = max(candidates)
    return [i for i, q in gemms if q["dimN"] == vocab]


def _piece_entries(rewriter: ShapeRewriter, piece: Piece,
                   indices: Optional[Sequence[int]] = None) -> List[Entry]:
    """The template expanded for one request's contribution alone.

    `indices` restricts the expansion to those template positions, returned in
    the same order.  This is a pure optimisation and it is exactly equivalent:
    `rewrite_dims` and `rewrite_dims_qk` process each (entry, rule) pair
    independently, with no cross-entry state, so slicing the inputs slices the
    output.  When fusing, only the ~48 attention entries are ever read from a
    per-piece expansion, and computing the other 194 for every request in the
    batch is the single hottest waste in the engine loop.
    """
    def take(entries, rules):
        if indices is None:
            return entries, rules
        return [entries[i] for i in indices], [rules[i] for i in indices]

    if piece.kind == "decode" and rewriter.decode_base is not None:
        # Measured decode law: its own template, its own exponents, and the
        # `+1` already folded into `seq_offset`.
        base, rules = take(rewriter.decode_base, rewriter.decode_rules)
        return rewrite_dims(base, rules, rewriter.decode_b0, rewriter.decode_s0,
                            batch=1, seqlen=max(1, piece.context),
                            seq_offset=rewriter.decode_seq_offset)

    base, rules = take(rewriter.base, rewriter.rules)
    if piece.kind == "decode":
        return rewrite_dims_qk(base, rules, rewriter.b0, rewriter.s0, batch=1,
                               seq_q=1, seq_k=max(1, piece.key_len))
    # A prefill chunk attends to everything before it as well as to itself.
    return rewrite_dims_qk(base, rules, rewriter.b0, rewriter.s0, batch=1,
                           seq_q=max(1, piece.query_len),
                           seq_k=max(1, piece.key_len))


def build_iteration_kernels_tagged(
    rewriter: ShapeRewriter,
    pieces: Sequence[Piece],
    fuse_linear: bool = True,
    logits_last_token_only: bool = False,
    vocab: Optional[int] = None,
) -> List[Tuple[Entry, str]]:
    """Kernel list for one mixed iteration, each entry tagged with its origin.

    Tags are `'fused'`, `'attn:prefill'` or `'attn:decode'`.  They cost nothing
    to produce and they are what lets the engine attribute energy to a phase --
    "how much of this iteration's joules went to decode attention?" is otherwise
    unanswerable once the linear layers have been fused, because a fused GEMM
    genuinely belongs to both phases at once.

    fuse_linear
        True  -- one GEMM per linear layer at the summed token count, plus
                 per-request attention.  What a real engine does for the half
                 that can be fused.
        False -- concatenate whole per-request kernel lists.  Cheaper to reason
                 about and what the v1 design originally proposed; kept so the
                 two can be compared directly, since the difference is the
                 launch-overhead-and-tail cost of not fusing.

    logits_last_token_only
        The traced HuggingFace forward computes logits at **every** position, so
        a fused prefill chunk of 2000 tokens produces a 2000 x 50257 GEMM.  A
        real engine computes logits only for tokens it will sample from: one per
        finishing prefill chunk, one per decode.  Off by default, because the
        template is a measurement and silently rewriting it would break the
        property that every shipped shape is reproducible from a trace.  Turn it
        on when comparing against vLLM, where it is worth several percent of
        prefill time.
    """
    if not pieces:
        raise ValueError("an iteration needs at least one piece")

    base = rewriter.base
    n = len(base)
    if rewriter.decode_base is not None and len(rewriter.decode_base) != n:
        raise ValueError(
            f"decode template has {len(rewriter.decode_base)} entries and prefill "
            f"has {n}; index alignment is what lets a mixed batch mix the two")

    tags = [f"attn:{p.kind}" for p in pieces]

    if not fuse_linear:
        out: List[Tuple[Entry, str]] = []
        for piece in pieces:
            entries = _piece_entries(rewriter, piece)
            out.extend((e, f"whole:{piece.kind}") for e in entries)
        return out

    fusible = fusible_mask(rewriter)
    # Only the attention entries are ever read per request when fusing.
    per_request_idx = [i for i, f in enumerate(fusible) if not f]
    slot = {orig: k for k, orig in enumerate(per_request_idx)}
    per_piece = [_piece_entries(rewriter, p, per_request_idx) for p in pieces]

    total_tokens = sum(p.tokens for p in pieces)
    # Every fusible field depends only on B*S, so batch=1 and seqlen=total is
    # the same token count the real fused launch sees.
    fused = rewrite_dims(base, rewriter.rules, rewriter.b0, rewriter.s0,
                         batch=1, seqlen=max(1, total_tokens))

    if logits_last_token_only:
        # At most one sampled position per request per iteration.  Exact for
        # decode; an over-count of at most one row for a prefill chunk that does
        # not finish the prompt, which is negligible beside the 2000 -> N
        # reduction this is correcting.
        emitted = len(pieces)
        for i in _lm_head_indices(rewriter, vocab):
            q, op = fused[i]
            q = dict(q)
            q["dimM"] = max(1, emitted)
            fused[i] = (q, op)

    out: List[Tuple[Entry, str]] = []
    i = 0
    while i < n:
        if fusible[i]:
            out.append((fused[i], "fused"))
            i += 1
            continue
        # Replay the whole contiguous attention run once per request, so the
        # trace reads block by block, request by request.
        j = i
        while j < n and not fusible[j]:
            j += 1
        lo, hi = slot[i], slot[j - 1] + 1
        for entries, tag in zip(per_piece, tags):
            out.extend((e, tag) for e in entries[lo:hi])
        i = j
    return out


def build_iteration_kernels(
    rewriter: ShapeRewriter,
    pieces: Sequence[Piece],
    fuse_linear: bool = True,
    logits_last_token_only: bool = False,
    vocab: Optional[int] = None,
) -> List[Entry]:
    """`build_iteration_kernels_tagged` with the tags stripped."""
    return [entry for entry, _tag in build_iteration_kernels_tagged(
        rewriter, pieces, fuse_linear=fuse_linear,
        logits_last_token_only=logits_last_token_only, vocab=vocab)]


@dataclass
class MixedReport:
    """What the classifier found -- print this once before trusting a run."""

    n_entries: int
    n_fusible: int
    n_attention: int
    n_runs: int
    classes_agree: bool
    disagreements: List[int]

    def __str__(self) -> str:
        agree = "agree" if self.classes_agree else \
                f"DISAGREE at entries {self.disagreements[:8]}"
        return (f"{self.n_entries} template entries: {self.n_fusible} fuse at the "
                f"summed token count, {self.n_entries - self.n_fusible} are "
                f"per-request across {self.n_runs} contiguous attention runs "
                f"({self.n_attention} read the key axis; the two classifiers {agree})")


def mixed_report(rewriter: ShapeRewriter) -> MixedReport:
    """Cross-check the two independent classifiers.

    `fusible_mask` asks an arithmetic question (does the field depend only on
    B*S?).  `attention_mask` asks a structural one (does it read the key axis?).
    They should be exact complements.  If they are not, a model has introduced a
    field class that fuses arithmetically but reads keys, or vice versa -- and
    the fused expansion would be silently wrong for it.
    """
    fus = fusible_mask(rewriter)
    att = attention_mask(rewriter)
    bad = [i for i, (f, a) in enumerate(zip(fus, att)) if f == a]

    runs, prev = 0, True
    for f in fus:
        if not f and prev:
            runs += 1
        prev = f

    return MixedReport(
        n_entries=len(fus),
        n_fusible=sum(fus),
        n_attention=sum(att),
        n_runs=runs,
        classes_agree=not bad,
        disagreements=bad,
    )


def iteration_token_shapes(pieces: Sequence[Piece]) -> Dict:
    """The composition, as the numbers that drive the shapes."""
    decode = [p for p in pieces if p.kind == "decode"]
    prefill = [p for p in pieces if p.kind == "prefill"]
    ctx = [p.context for p in decode]
    return {
        "decode_batch": len(decode),
        "prefill_chunks": len(prefill),
        "prefill_tokens": sum(p.tokens for p in prefill),
        "total_tokens": sum(p.tokens for p in pieces),
        "context_mean": (sum(ctx) / len(ctx)) if ctx else 0.0,
        "context_max": max(ctx) if ctx else 0,
        "distinct_contexts": len(set(ctx)),
    }
