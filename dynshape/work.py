"""
work.py
=======

**How much arithmetic and how much traffic a kernel shape implies** -- FLOPs and
bytes, from the shape alone.

No backend, no LUT, no GPU.  This is pure geometry: a GEMM of `B x M x N x K`
does `2BMNK` multiply-adds and touches `B(MK + KN + MN)` elements whatever
hardware runs it.

WHY THIS EXISTS SEPARATELY FROM THE PREDICTOR
---------------------------------------------
FSTS reports a per-iteration **work vector** -- gemm flops, attention flops,
attention bytes, split by phase -- alongside its power number, and that is a
better thing to report than watts alone for two reasons:

1. **It is backend-independent.** Watts come from a model that can be wrong;
   FLOPs come from the shapes, which are measured.  A work vector lets someone
   re-derive power under a *different* power model without re-running the
   scheduler, and lets you sanity-check a suspicious wattage against the
   arithmetic that produced it.
2. **It separates the two error sources.**  If a predicted trace disagrees with
   a real one, the work vector says whether the simulator got the *work* wrong
   (a scheduler or shape bug) or the *conversion to watts* wrong (a predictor
   bug).  With only watts reported, those two are indistinguishable.

The analytic backend uses these same functions, so its roofline and the reported
work vector cannot drift apart.

WHAT IS SEPARABLE BY PHASE, AND WHAT IS NOT
-------------------------------------------
A fused GEMM's rows each belong to exactly one request, and FLOPs are linear in
the row count, so **FLOPs split between prefill and decode exactly** by token
share.  Bytes do not: the weight matrix is read once for the whole batch, which
is the entire point of fusing.  So the work vector splits flops by phase and
leaves bytes whole, and says so rather than inventing a rule.
"""

from __future__ import annotations

from typing import Dict, Tuple

#: Bytes per element, by precision tag as it appears in a traced entry.
BYTES_PER_ELEMENT = {"fp32": 4, "tf32": 4, "bf16": 2, "fp16": 2, "int8": 1, "fp8": 1}

#: How many times a memory-bound kernel streams its data.  Elementwise reads two
#: operands and writes one; layernorm and softmax make two reduction passes and
#: write once.  Both land on three.
_STREAM_PASSES = 3.0


def element_bytes(q: Dict) -> int:
    return BYTES_PER_ELEMENT.get(q.get("prec", q.get("precM", "bf16")), 2)


def gemm_work(q: Dict) -> Tuple[float, float, float]:
    """(flops, bytes, weight_bytes) for one GEMM.

    `weight_bytes` is the `K x N` operand -- the part that is read once per
    launch regardless of how many token rows ride along, and therefore the part
    that fusing amortises.  It is included in `bytes`, not additional to it.
    """
    b = q.get("batch", 1)
    m, n, k = q["dimM"], q["dimN"], q["dimK"]
    w = BYTES_PER_ELEMENT.get(q.get("precM", "bf16"), 2)
    flops = 2.0 * b * m * n * k
    byts = float(b) * (m * k + k * n + m * n) * w
    weight_bytes = float(b) * k * n * w
    return flops, byts, weight_bytes


def kernel_work(q: Dict, op: Tuple[str, ...],
                strict: bool = True) -> Tuple[float, float, float]:
    """(flops, bytes, weight_bytes) for any traced entry.

    `strict=False` returns zeros for an op with no model instead of raising --
    what the engine wants, so one unknown kernel type cannot abort a whole run's
    work accounting.
    """
    head = op[0] if op else ""
    w = element_bytes(q)

    if head == "gemm":
        return gemm_work(q)
    if head == "elementwise":
        return 0.0, float(q["dim"]) * w * _STREAM_PASSES, 0.0
    if head in ("layernorm", "softmax"):
        return 0.0, float(q["batch"] * q["dim"]) * w * _STREAM_PASSES, 0.0

    if strict:
        raise NotImplementedError(f"no work model for op {op!r}")
    return 0.0, 0.0, 0.0


def uses_tensor_core(q: Dict, op: Tuple[str, ...]) -> bool:
    return bool(op) and op[0] == "gemm" and bool(q.get("useTensorCore", False))


#: The per-iteration work vector, in the order it is reported.  Named here so the
#: engine, the dataframe columns and the docs cannot disagree about it.
WORK_FIELDS = (
    "linear_flops", "linear_bytes", "weight_bytes",
    "attn_flops", "attn_bytes",
    "prefill_flops", "decode_flops",
)


def empty_work() -> Dict[str, float]:
    return {k: 0.0 for k in WORK_FIELDS}
