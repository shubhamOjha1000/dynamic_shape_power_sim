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
from typing import Dict, List, Sequence, Tuple

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
) -> List[Entry]:
    """Rescale a traced template from (b0, s0) to (batch, seqlen).

    Exact for prefill.  For decode use `rewrite_dims_qk` instead -- decode is a
    different attention shape, not a rescaled one.
    """
    out: List[Entry] = []
    for (q, op), r in zip(t_base, rules):
        nq = dict(q)
        for k, (a, b) in r.items():
            nq[k] = _round_pos(q[k] * (batch / b0) ** a * (seqlen / s0) ** b)
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
    """

    base: List[Entry]
    rules: Rules
    b0: int
    s0: int
    model: str = "gpt2"

    # -- construction --------------------------------------------------------

    @classmethod
    def from_files(cls, base_path: str, alt_batch_path: str, alt_seq_path: str,
                   model: str = "gpt2") -> "ShapeRewriter":
        b0, s0, mode0 = parse_shape_from_name(base_path)
        b1, s1, _ = parse_shape_from_name(alt_batch_path)
        b2, s2, _ = parse_shape_from_name(alt_seq_path)

        if s1 != s0:
            raise ValueError("alt_batch trace must hold seqlen fixed")
        if b2 != b0:
            raise ValueError("alt_seq trace must hold batch fixed")
        if mode0 != "prefill":
            raise ValueError("the base template must be a prefill trace")

        base = load_template(base_path)
        rules = learn_scaling(
            base,
            load_template(alt_batch_path),
            load_template(alt_seq_path),
            batch_ratio=b1 / b0,
            seq_ratio=s2 / s0,
        )
        return cls(base=base, rules=rules, b0=b0, s0=s0, model=model)

    @classmethod
    def from_dir(cls, template_dir: str, model: str = "gpt2",
                 b0: int = 8, s0: int = 128) -> "ShapeRewriter":
        """Pick the three anchor traces out of a directory of workload files.

        Needs (b0, s0), one file at a different batch, and one at a different
        seqlen.  Anything else in the directory is left alone -- the extra files
        become validation targets for `tests/test_scaling.py`.
        """
        files = [f for f in sorted(os.listdir(template_dir)) if f.endswith(".json")]
        by_shape = {}
        for f in files:
            try:
                b, s, mode = parse_shape_from_name(f)
            except ValueError:
                continue
            if mode == "prefill":
                by_shape[(b, s)] = os.path.join(template_dir, f)

        if (b0, s0) not in by_shape:
            raise FileNotFoundError(f"no prefill template at b={b0}, s={s0} in {template_dir}")

        alt_b = next((by_shape[(b, s0)] for b, s in sorted(by_shape) if s == s0 and b != b0), None)
        alt_s = next((by_shape[(b0, s)] for b, s in sorted(by_shape) if b == b0 and s != s0), None)
        if alt_b is None or alt_s is None:
            raise FileNotFoundError(
                "need one template at a different batch and one at a different seqlen; "
                f"found shapes {sorted(by_shape)}"
            )
        return cls.from_files(by_shape[(b0, s0)], alt_b, alt_s, model=model)

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
            return rewrite_dims_qk(self.base, self.rules, self.b0, self.s0,
                                   batch, seq_q=1, seq_k=seqlen)
        raise ValueError(f"mode must be 'prefill' or 'decode', got {mode!r}")

    def n_kernels(self) -> int:
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
