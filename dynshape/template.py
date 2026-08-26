"""
template.py
===========

Make EnergAIzer's input shape -- (batch, seq_len, mode) -- **dynamic**.

THE PROBLEM
-----------
EnergAIzer consumes a *traced* workload file: a flat JSON list of kernels in
execution order, produced by running the real model once on a real GPU at one
fixed (batch, seqlen, mode).  The shipped GPT-2 files are literally named

    gpt2model_gpt2_pbf16_b8_s128_modeprefill.json

so every shape you can evaluate is a shape somebody already traced.  Ask for
b=13, s=377 and there is no file.

THE FIX
-------
Trace **three** times, then *derive* every other shape arithmetically.

Every numeric field in a traced entry is a monomial in the input dimensions:

    value = const * B^a * S^b

`learn_scaling()` recovers the integer exponents (a, b) per entry per field by
differencing three traces that move one input at a time.  `rewrite_dims()` then
evaluates that monomial at any (B, S) you want.  No architecture knowledge, no
hard-coded kernel families, no per-model rules -- it works for anything the
tracer can trace.

VALIDATION
----------
Learned from (b8,s128) + (b16,s128) + (b8,s512), the rewriter reproduces **all
25 shipped GPT-2 prefill files exactly** -- every field of every one of the 242
entries, across b in {1,2,8,16,32} x s in {128,512,1024,2048,4096}.  See
`tests/test_scaling.py`.

MODE
----
`mode` is the third dynamic axis and it is *not* a rescaling of the first two.
Prefill attention is square (every token attends to every token); decode
attention is a strip (one query row against the whole KV cache).  No exponent
turns a square into a strip.

So decode is handled by *splitting* the sequence exponent into two independent
axes -- query length Sq and key/context length Sk -- with prefill being the
special case Sq == Sk == S, and decode being Sq == 1, Sk == context.  See
`split_seq_exponent()`.  That split is the one hand-written rule in this file
and it is flagged INFERRED: no decode workload ships with the artifact
(`grep -c decode` over all 90 shipped files returns 0), so it is validated only
structurally, not against a real decode trace.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

# A traced entry is [query_dict, op_type_list]; we keep op_type as a tuple.
Entry = Tuple[Dict, Tuple[str, ...]]

#: Exponents on (batch, seq) per field, per entry.
Rules = List[Dict[str, Tuple[int, int]]]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

_NAME_RE = re.compile(r"_b(\d+)_s(\d+)_mode([a-z]+)")


def load_template(path: str) -> List[Entry]:
    """Load a traced workload JSON as a list of (query_dict, op_type_tuple)."""
    with open(path) as f:
        raw = json.load(f)
    return [(dict(q), tuple(op)) for q, op in raw]


def parse_shape_from_name(path: str) -> Tuple[int, int, str]:
    """Recover (batch, seqlen, mode) from an EnergAIzer workload filename."""
    m = _NAME_RE.search(os.path.basename(path))
    if not m:
        raise ValueError(f"cannot parse b/s/mode out of {os.path.basename(path)!r}")
    return int(m.group(1)), int(m.group(2)), m.group(3)


def _is_scalable(v) -> bool:
    """True for numeric fields that can carry an exponent.

    `bool` is excluded explicitly: `isinstance(True, int)` is True in Python, so
    `useTensorCore` would otherwise be treated as the number 1 and rescaled into
    nonsense.  Zeros are excluded because log(0/0) is undefined.
    """
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v != 0


# ---------------------------------------------------------------------------
# Learning the scaling law
# ---------------------------------------------------------------------------

def learn_scaling(
    t_base: Sequence[Entry],
    t_alt_batch: Sequence[Entry],
    t_alt_seq: Sequence[Entry],
    batch_ratio: float,
    seq_ratio: float,
    tol: float = 1e-9,
) -> Rules:
    """Recover the exponents (a, b) on (batch, seq) for every numeric field.

    Parameters
    ----------
    t_base      : template traced at (B0, S0)
    t_alt_batch : same model traced at (B0 * batch_ratio, S0)
    t_alt_seq   : same model traced at (B0, S0 * seq_ratio)
    batch_ratio, seq_ratio : the factors used above (need not be 2)

    Raises
    ------
    ValueError
        If the three traces disagree in length or op sequence, or if any field
        yields a non-integer exponent.  The loud failure is the point: a
        non-integer exponent means `value = const * B^a * S^b` does not hold for
        that field, and silently rounding it would bake a wrong shape into every
        downstream prediction.
    """
    if batch_ratio == 1 or seq_ratio == 1:
        raise ValueError("ratios must differ from 1 -- otherwise nothing varies")
    if not (len(t_base) == len(t_alt_batch) == len(t_alt_seq)):
        raise ValueError(
            f"traces differ in length: {len(t_base)}, {len(t_alt_batch)}, {len(t_alt_seq)} "
            "-- the kernel sequence must be shape-independent for differencing to work"
        )

    rules: Rules = []
    for i, (q, op) in enumerate(t_base):
        if t_alt_batch[i][1] != op or t_alt_seq[i][1] != op:
            raise ValueError(f"entry {i}: op type differs between traces ({op})")

        r: Dict[str, Tuple[int, int]] = {}
        for k, v in q.items():
            if not _is_scalable(v):
                continue
            try:
                vb, vs = t_alt_batch[i][0][k], t_alt_seq[i][0][k]
            except KeyError as e:
                raise ValueError(f"entry {i}: field {k!r} missing from a trace") from e

            a = math.log(vb / v, batch_ratio)
            b = math.log(vs / v, seq_ratio)
            if abs(a - round(a)) > tol or abs(b - round(b)) > tol:
                raise ValueError(
                    f"entry {i} field {k!r} (op={op}): non-integer exponent "
                    f"({a:.6f}, {b:.6f}) -- scaling is not a pure power law here"
                )
            r[k] = (round(a), round(b))
        rules.append(r)
    return rules


# ---------------------------------------------------------------------------
# Rewriting -- two-axis (prefill only)
# ---------------------------------------------------------------------------

def rewrite_dims(
    t_base: Sequence[Entry],
    rules: Rules,
    b0: int,
    s0: int,
    batch: int,
    seqlen: int,
    seq_offset: int = 0,
) -> List[Entry]:
    """Rescale a traced template from (b0, s0) to (batch, seqlen).

    `seq_offset` shifts the sequence axis, so the law is

        value = const * B^a * (S + offset)^b

    Prefill needs no shift.  **Decode needs offset = 1**, measured rather than
    assumed: tracing GPT-2 decode with a `seqlen`-token KV cache shows attention
    over `seqlen + 1` keys, because the new token's own K/V are appended before
    attention runs.  At a nominal context of 128 the attention `dimN` is 129, and
    512 gives 513 -- affine in S, so a pure power law fits with an exponent of
    0.9958 instead of 1 and `learn_scaling` refuses it.  Shifted by one it is
    exactly 1.
    """
    out: List[Entry] = []
    for (q, op), r in zip(t_base, rules):
        nq = dict(q)
        for k, (a, b) in r.items():
            nq[k] = _round_pos(
                q[k] * (batch / b0) ** a
                * ((seqlen + seq_offset) / (s0 + seq_offset)) ** b
            )
        out.append((nq, op))
    return out


def _round_pos(x: float) -> int:
    """Round to a positive integer -- dimensions of 0 are not runnable kernels."""
    return max(1, int(round(x)))


# ---------------------------------------------------------------------------
# The third axis: mode
# ---------------------------------------------------------------------------

def split_seq_exponent(op: Tuple[str, ...], field: str, b: int) -> Tuple[int, int]:
    """Split a sequence exponent `b` into (query-length, key-length) exponents.

    Rationale, field class by field class -- these are the only classes that
    occur in the traced GPT-2 template, and every one of them is checked by
    `tests/test_decode.py`:

      b == 0   architecture constant (hidden size, head dim, 3*hidden)  -> (0,0)
      b == 2   the B*heads*Sq*Sk score tensors (mask add)               -> (1,1)
      b == 1   one sequence axis, and which one depends on where it sits:
                 GEMM `dimN` / `dimK`  -- the key/inner axis of attention -> key
                 softmax `dim`         -- the reduction axis              -> key
                 everything else       -- `dimM`, `batch`, token counts   -> query

    Prefill sets Sq == Sk, so the split is invisible there; that identity is
    itself a test (`test_three_axis_reduces_to_two_axis`).

    INFERRED, NOT VALIDATED: no decode workload ships with the EnergAIzer
    artifact, so this rule is derived from `run_model.get_input()` (decode feeds
    one token plus a length-`seqlen` KV cache) and checked for internal
    consistency -- never against a real decode trace.
    """
    if b == 0:
        return (0, 0)
    if b == 2:
        return (1, 1)
    if b == 1:
        if field in ("dimN", "dimK"):
            return (0, 1)
        if op and op[0] == "softmax" and field == "dim":
            return (0, 1)
        return (1, 0)
    raise ValueError(
        f"op={op} field={field!r}: sequence exponent {b} is outside the modelled "
        "set {0,1,2}; the query/key split is undefined for it"
    )


def rewrite_dims_qk(
    t_base: Sequence[Entry],
    rules: Rules,
    b0: int,
    s0: int,
    batch: int,
    seq_q: int,
    seq_k: int,
) -> List[Entry]:
    """Rescale a template with query length and key length varied independently.

    `value = const * B^a * Sq^p * Sk^q` where `p + q == b` from `learn_scaling`.

        prefill(B, S)      ->  seq_q = S,  seq_k = S
        decode(B, context) ->  seq_q = 1,  seq_k = context
    """
    out: List[Entry] = []
    for (q, op), r in zip(t_base, rules):
        nq = dict(q)
        for k, (a, b) in r.items():
            p, kk = split_seq_exponent(op, k, b)
            nq[k] = _round_pos(
                q[k] * (batch / b0) ** a * (seq_q / s0) ** p * (seq_k / s0) ** kk
            )
        out.append((nq, op))
    return out


# ---------------------------------------------------------------------------
# The user-facing object
# ---------------------------------------------------------------------------

@dataclass
class ShapeRewriter:
    """A traced template plus its learned scaling law -- an *any-shape* expander.

    >>> rw = ShapeRewriter.from_dir("templates/gpt2")
    >>> kernels = rw.expand(batch=13, seqlen=377, mode="prefill")   # never traced
    >>> len(kernels)
    242

    **Two laws, not one.** Prefill always has its own law, learned from three
    prefill anchors and verified exactly against all 25 shipped templates.
    Decode can have a second, independent law learned from three *decode*
    anchors -- and when it does, the inferred query/key split of
    `split_seq_exponent` is never used:

        prefill anchors (3) -> prefill law -> any prefill shape   verified
        decode  anchors (3) -> decode  law -> any decode shape    verifiable

    Drop `..._modedecode.json` files into the template directory and `from_dir`
    picks them up automatically. Until then decode falls back to the inferred
    split, and `decode_source` returns 'inferred' so a caller can say so.

    Producing the three decode anchors is three runs of the artifact's own
    harness -- `run_model.py --mode decode --trace` at (b8,s128), (b16,s128),
    (b8,s512) -- then `parse_trace.py`. That single step answers the kernel-list
    question, deletes the inferred rule, and makes decode testable the same way
    prefill is.
    """

    base: List[Entry]
    rules: Rules
    b0: int
    s0: int
    model: str = "gpt2"

    #: An optional SECOND law, learned from real decode traces. When present it
    #: is used verbatim for `mode='decode'` and the inferred query/key split is
    #: never consulted. See `decode_source`.
    decode_base: Optional[List[Entry]] = None
    decode_rules: Optional[Rules] = None
    decode_b0: Optional[int] = None
    decode_s0: Optional[int] = None
    #: Shift on decode's sequence axis, detected when the law is learned.
    #: Measured as 1 for GPT-2 -- see `rewrite_dims`.
    decode_seq_offset: int = 0

    # -- construction --------------------------------------------------------

    @staticmethod
    def _learn_from_anchors(base_path: str, alt_batch_path: str, alt_seq_path: str,
                            seq_offsets: Sequence[int] = (0, 1)):
        """(base, rules, b0, s0, mode, offset) from three traces.

        `seq_offsets` are the shifts to try on the sequence axis, in order.  A
        prefill template fits at 0; decode needs 1, because the token being
        generated attends over `context + 1` keys.  The first shift that yields
        integer exponents everywhere wins -- so the shift is *detected*, and if
        none fits, the error from the last attempt propagates rather than a law
        being invented.
        """
        b0, s0, mode0 = parse_shape_from_name(base_path)
        b1, s1, mode1 = parse_shape_from_name(alt_batch_path)
        b2, s2, mode2 = parse_shape_from_name(alt_seq_path)

        if s1 != s0:
            raise ValueError("alt_batch trace must hold seqlen fixed")
        if b2 != b0:
            raise ValueError("alt_seq trace must hold batch fixed")
        if not (mode0 == mode1 == mode2):
            raise ValueError(
                f"all three anchors must be the same mode, got "
                f"{mode0!r}, {mode1!r}, {mode2!r}"
            )

        base = load_template(base_path)
        alt_b = load_template(alt_batch_path)
        alt_s = load_template(alt_seq_path)

        last: Optional[Exception] = None
        for off in seq_offsets:
            try:
                rules = learn_scaling(
                    base, alt_b, alt_s,
                    batch_ratio=b1 / b0,
                    seq_ratio=(s2 + off) / (s0 + off),
                )
                return base, rules, b0, s0, mode0, off
            except ValueError as e:
                last = e
        raise last

    @classmethod
    def from_files(cls, base_path: str, alt_batch_path: str, alt_seq_path: str,
                   model: str = "gpt2",
                   decode_paths: Optional[Sequence[str]] = None) -> "ShapeRewriter":
        """Learn the prefill law, and optionally a real decode law beside it.

        `decode_paths` is the same three-anchor pattern traced with
        `--mode decode`: (base, alt_batch, alt_seq). Supply it and decode stops
        being inferred -- see the class docstring.
        """
        base, rules, b0, s0, mode0, _off = cls._learn_from_anchors(
            base_path, alt_batch_path, alt_seq_path, seq_offsets=(0,))
        if mode0 != "prefill":
            raise ValueError("the base template must be a prefill trace")

        d_base = d_rules = d_b0 = d_s0 = None
        d_off = 0
        if decode_paths is not None:
            if len(decode_paths) != 3:
                raise ValueError("decode_paths needs exactly three anchor traces")
            d_base, d_rules, d_b0, d_s0, d_mode, d_off = cls._learn_from_anchors(
                *decode_paths)
            if d_off:
                print(f"[dynshape] decode's sequence axis is shifted by +{d_off}: "
                      f"a token generated with a context of S attends over S+{d_off} "
                      f"keys, because its own K/V are appended before attention.")
            if d_mode != "decode":
                raise ValueError(f"decode_paths must be decode traces, got {d_mode!r}")
            if len(d_base) != len(base):
                # NOT an error -- this is exactly the thing worth discovering.
                print(f"[dynshape] note: decode traces to {len(d_base)} kernels, "
                      f"prefill to {len(base)}. The kernel list DOES differ between "
                      f"modes; the inferred split rule would have been wrong.")

        return cls(base=base, rules=rules, b0=b0, s0=s0, model=model,
                   decode_base=d_base, decode_rules=d_rules,
                   decode_b0=d_b0, decode_s0=d_s0, decode_seq_offset=d_off)

    @classmethod
    def from_dir(cls, template_dir: str, model: str = "gpt2",
                 b0: int = 8, s0: int = 128) -> "ShapeRewriter":
        """Pick the three anchor traces out of a directory of workload files.

        Needs (b0, s0), one file at a different batch, and one at a different
        seqlen.  Anything else in the directory is left alone -- the extra files
        become validation targets for `tests/test_scaling.py`.
        """
        files = [f for f in sorted(os.listdir(template_dir)) if f.endswith(".json")]
        by_mode: Dict[str, Dict] = {"prefill": {}, "decode": {}}
        for f in files:
            try:
                b, s, mode = parse_shape_from_name(f)
            except ValueError:
                continue
            if mode in by_mode:
                by_mode[mode][(b, s)] = os.path.join(template_dir, f)

        def anchors(by_shape, mode):
            """The three files that move one input each, or None."""
            if (b0, s0) not in by_shape:
                return None
            alt_b = next((by_shape[(b, s0)] for b, s in sorted(by_shape)
                          if s == s0 and b != b0), None)
            alt_s = next((by_shape[(b0, s)] for b, s in sorted(by_shape)
                          if b == b0 and s != s0), None)
            if alt_b is None or alt_s is None:
                return None
            return (by_shape[(b0, s0)], alt_b, alt_s)

        pre = anchors(by_mode["prefill"], "prefill")
        if pre is None:
            raise FileNotFoundError(
                f"need prefill templates at b={b0} s={s0}, one at another batch and "
                f"one at another seqlen; found {sorted(by_mode['prefill'])} in {template_dir}"
            )
        # Decode anchors are optional. When they are absent the inferred
        # query/key split is used instead, and `decode_source` says so.
        dec = anchors(by_mode["decode"], "decode")
        return cls.from_files(*pre, model=model, decode_paths=dec)

    # -- use -----------------------------------------------------------------

    def expand(self, batch: int, seqlen: int, mode: str = "prefill") -> List[Entry]:
        """Kernel list for ANY (batch, seqlen, mode) -- the whole point of this file.

        mode='prefill' : `seqlen` is the prompt length; attention is square.
        mode='decode'  : `seqlen` is the KV-cache/context length; one new token
                         per sequence, so attention is a 1 x context strip.
        """
        if batch < 1 or seqlen < 1:
            raise ValueError(f"batch and seqlen must be >= 1, got ({batch}, {seqlen})")
        if mode == "prefill":
            return rewrite_dims_qk(self.base, self.rules, self.b0, self.s0,
                                   batch, seq_q=seqlen, seq_k=seqlen)
        if mode == "decode":
            if self.decode_base is not None:
                # Measured decode law: decode has its own template and its own
                # exponents, so the query/key split is never consulted.
                return rewrite_dims(self.decode_base, self.decode_rules,
                                    self.decode_b0, self.decode_s0, batch, seqlen,
                                    seq_offset=self.decode_seq_offset)
            # seq_k is context + 1, not context: the generated token's own K/V are
            # appended before attention, so it attends over one more key than the
            # cache holds. Measured -- tracing GPT-2 decode at a nominal context of
            # 128 gives attention dimN 129, softmax dim 129 and a B*heads*1*129
            # score tensor. Getting this wrong misses 48 of 242 entries, every one
            # of them an attention shape.
            return rewrite_dims_qk(self.base, self.rules, self.b0, self.s0,
                                   batch, seq_q=1, seq_k=seqlen + 1)
        raise ValueError(f"mode must be 'prefill' or 'decode', got {mode!r}")

    @property
    def decode_source(self) -> str:
        """Where decode shapes come from -- always report this beside a number.

        'measured'  : learned from real decode traces, validatable like prefill.
        'inferred'  : derived from the prefill law by `split_seq_exponent`.
                      Structurally consistent, never checked against hardware.
        """
        return "measured" if self.decode_base is not None else "inferred"

    def n_kernels(self, mode: str = "prefill") -> int:
        if mode == "decode" and self.decode_base is not None:
            return len(self.decode_base)
        return len(self.base)

    def field_report(self) -> List[Tuple[Tuple[str, ...], str, int, int, int, int]]:
        """Every distinct (op, field, a, b, p, q) class the template contains.

        Useful for eyeballing what was learned -- and for spotting the moment a
        new model introduces a field class the query/key split has no rule for.
        """
        seen = {}
        for (q, op), r in zip(self.base, self.rules):
            for k, (a, b) in r.items():
                p, kk = split_seq_exponent(op, k, b)
                seen[(op, k, a, b, p, kk)] = seen.get((op, k, a, b, p, kk), 0) + 1
        return sorted(seen, key=lambda t: (-seen[t], str(t)))
