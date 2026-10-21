"""
extras.py
=========

**The kernels the traced template leaves out.**

The 242-entry GPT-2 template is EnergAIzer's own trace, turned into entries by
its own `parse_trace.py`.  That parser keeps only the ops EnergAIzer has a model
for -- GEMMs, softmax, layernorm and a fixed list of elementwise math -- and
silently drops everything else.  The trace is also of `GPT2Model`, the bare
transformer, so it has no LM head at all.

A real forward pass launches those kernels anyway.  Profiling the executor the
validation notebooks run -- HuggingFace GPT-2, bf16, eager attention, one
`DynamicCache` per request -- on a Colab A100 gave, per forward pass
(`A100_Energy_Gap_Colab.ipynb`, section 5):

    family         extra  what launches it
    gemm           +1     the LM head: logits at every position, q x 50257 x 768
    cat            24     DynamicCache.update: torch.cat of K and V, 2 per block
    where / mask   13     the causal torch.where in each block, + 1 building the mask
    elementwise    +14    the attention-mask add in each block, + 2 building the mask
    fill           25     torch.full scalars, 2 per block (1/sqrt(d), mask value), + 1
    copy           13     merging heads, 1 per block (prefill only) + 1 host-to-device copy
    index / embed  6      token and position embedding lookups, + 4 aranges

That is 96 more kernels per prefill pass (242 -> 338) and 84 more per decode
step (242 -> 326: one token's heads merge without a copy), exactly the counts
profiled.  `gpt2_hf_eager()` is that list.

HOW THEY ARE PRICED
-------------------
EnergAIzer has no table for cat, copy, fill, where or gather, so each extra
borrows the closest kernel it does have, matched on memory traffic:

    one input, one output   (cat, copy, gather, fill, arange)  ->  scalar_add
    two inputs, one output  (where, the mask add)              ->  pointwise_add
    the LM head                                                ->  a GEMM, as is

These are stand-ins, and the notebooks check them against measured GPU time.
The real `where` and mask add broadcast a q x k mask across the heads and run
PyTorch's non-vectorised elementwise kernel, which is slower than the vectorised
add it is priced as.  Elementwise sizes are floored at `ELEMENTWISE_MIN_DIM`,
because EnergAIzer's elementwise tables start at 127 elements; a one-element
fill is the same single launch as a 128-element one.

SIZES
-----
Every numeric field is a monomial in the two numbers a piece carries,

    value = const * q^a * k^b      q = new tokens, k = keys attended (context + q)

the same form `template.py` learns for the template, with the query and key
axes apart from the start.  An extra that never involves k depends only on
token counts and fuses like a linear layer; one that involves k is per request,
like attention.  `mixed.py` uses exactly that split.

WHERE THEY GO
-------------
Each extra names an anchor, resolved by `mixed.extra_slots` against the
template's attention runs (the contiguous non-fusible entries, one per block):

    prologue        before the first template entry   embeddings, building the mask
    before_attn     before each attention run         the KV-cache append
    after_qk        after each run's first entry      the 1/sqrt(d) fill
    before_softmax  before each run's softmax         mask value, causal where, mask add
    after_attn      after each attention run          merging heads
    epilogue        after the last template entry     the LM head
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

from .template import Entry, _round_pos

#: EnergAIzer's elementwise tables start at 127 elements.
ELEMENTWISE_MIN_DIM = 128

ANCHORS = ("prologue", "before_attn", "after_qk", "before_softmax",
           "after_attn", "epilogue")

#: Anchors that occur once per forward pass; every other anchor occurs once per
#: attention run, i.e. once per block.
PASS_ANCHORS = ("prologue", "epilogue")


@dataclass(frozen=True)
class Mono:
    """`const * q**q_exp * k**k_exp`: one field's size law."""

    const: float
    q_exp: int = 0
    k_exp: int = 0

    def __call__(self, q: int, k: int) -> float:
        return self.const * q ** self.q_exp * k ** self.k_exp


def _as_value(v):
    """A field value as stored: numbers become monomials, the rest stay literal.

    A three-element list is a serialised monomial.  `bool` is checked before the
    numbers because `isinstance(True, int)` is True, and `useTensorCore` must stay
    a flag.
    """
    if isinstance(v, (Mono, str, bool)):
        return v
    if isinstance(v, (int, float)):
        return Mono(v)
    if isinstance(v, (list, tuple)) and len(v) == 3:
        return Mono(v[0], int(v[1]), int(v[2]))
    raise ValueError(f"cannot read {v!r} as a field value")


def _to_json_value(v):
    return [v.const, v.q_exp, v.k_exp] if isinstance(v, Mono) else v


@dataclass(frozen=True)
class ExtraKernel:
    """One kind of kernel the template does not have.

    name       unique within a set; also the suffix of the kernel's tag
    family     the profiler family it stands in for (cat, copy, fill, ...)
    anchor     where it is launched, one of `ANCHORS`
    op, fields the EnergAIzer entry it is priced as; numbers become `Mono`
    count      launches at each slot (2 for the K and V appends)
    min_query  launched only for pieces with at least this many new tokens
    logits     True for the LM head, so `logits_last_token_only` can shrink it
    """

    name: str
    family: str
    anchor: str
    op: Tuple[str, ...]
    fields: Tuple[Tuple[str, object], ...]
    count: int = 1
    min_query: int = 1
    logits: bool = False
    note: str = ""

    def __post_init__(self):
        if self.anchor not in ANCHORS:
            raise ValueError(f"{self.name}: anchor {self.anchor!r} is not one of {ANCHORS}")
        if self.count < 1 or self.min_query < 1:
            raise ValueError(f"{self.name}: count and min_query must be >= 1")
        pairs = self.fields.items() if isinstance(self.fields, Mapping) else self.fields
        object.__setattr__(self, "op", tuple(self.op))
        object.__setattr__(self, "fields", tuple((str(k), _as_value(v)) for k, v in pairs))

    @property
    def reads_keys(self) -> bool:
        """True if any field grows with the key length -- per request, like attention."""
        return any(isinstance(v, Mono) and v.k_exp != 0 for _, v in self.fields)

    @property
    def per_block(self) -> bool:
        return self.anchor not in PASS_ANCHORS

    def entry(self, q: int, k: int) -> Entry:
        """The EnergAIzer entry for a piece with `q` new tokens attending over `k` keys."""
        out: Dict = {}
        for name, v in self.fields:
            if isinstance(v, Mono):
                x = _round_pos(v(q, k))
                if name == "dim" and self.op[:1] == ("elementwise",):
                    x = max(x, ELEMENTWISE_MIN_DIM)
                out[name] = x
            else:
                out[name] = v
        return out, self.op

    def to_dict(self) -> Dict:
        return {"name": self.name, "family": self.family, "anchor": self.anchor,
                "op": list(self.op),
                "fields": {k: _to_json_value(v) for k, v in self.fields},
                "count": self.count, "min_query": self.min_query,
                "logits": self.logits, "note": self.note}

    @classmethod
    def from_dict(cls, d: Mapping) -> "ExtraKernel":
        return cls(name=d["name"], family=d["family"], anchor=d["anchor"],
                   op=tuple(d["op"]), fields=dict(d["fields"]),
                   count=int(d.get("count", 1)), min_query=int(d.get("min_query", 1)),
                   logits=bool(d.get("logits", False)), note=d.get("note", ""))


@dataclass(frozen=True)
class ExtraSet:
    """The extra kernels of one executor, in launch order within each anchor."""

    kernels: Tuple[ExtraKernel, ...]
    name: str = ""
    source: str = ""

    def __post_init__(self):
        object.__setattr__(self, "kernels", tuple(self.kernels))
        names = [x.name for x in self.kernels]
        dup = sorted({n for n in names if names.count(n) > 1})
        if dup:
            raise ValueError(f"duplicate extra kernel names: {dup}")

    def launches(self, query_len: int, n_blocks: int) -> int:
        """Extra kernels one forward pass launches for a piece of `query_len` tokens."""
        return sum(self.family_counts(query_len, n_blocks).values())

    def family_counts(self, query_len: int, n_blocks: int) -> Dict[str, int]:
        """The same, by profiler family -- what a kernel profile is compared against."""
        out: Dict[str, int] = {}
        for x in self.kernels:
            if query_len < x.min_query:
                continue
            out[x.family] = out.get(x.family, 0) + x.count * (n_blocks if x.per_block else 1)
        return out

    def to_dict(self) -> Dict:
        return {"name": self.name, "source": self.source,
                "kernels": [x.to_dict() for x in self.kernels]}

    @classmethod
    def from_dict(cls, d: Mapping) -> "ExtraSet":
        return cls(kernels=tuple(ExtraKernel.from_dict(x) for x in d["kernels"]),
                   name=d.get("name", ""), source=d.get("source", ""))

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1)

    @classmethod
    def load(cls, path: str) -> "ExtraSet":
        with open(path) as f:
            return cls.from_dict(json.load(f))


def gpt2_hf_eager(hidden: int = 768, heads: int = 12, vocab: int = 50257,
                  prec: str = "bf16") -> ExtraSet:
    """What HuggingFace GPT-2 launches beyond the template.

    Eager attention, one `DynamicCache` per request, logits at every position:
    the executor of `A100_Validation_Colab.ipynb`.  The table in the module
    docstring is this list, counted by family.
    """
    ew = ("elementwise",)

    def x(name, family, anchor, op, size, **kw):
        return ExtraKernel(name, family, anchor, ew,
                           {"dim": size, "op": op, "prec": prec}, **kw)

    one = Mono(1)
    per_q, per_k, per_qk = Mono(1, 1, 0), Mono(1, 0, 1), Mono(1, 1, 1)
    kernels = (
        # Before the blocks: the two embedding lookups, then transformers'
        # `create_causal_mask` building one q x k mask for the whole pass.
        x("embed_lookup", "index / embed", "prologue", "scalar_add", Mono(hidden, 1, 0),
          count=2, note="token and position embedding gathers, one row per new token"),
        x("arange_q", "index / embed", "prologue", "scalar_add", per_q,
          note="cache_position = arange(past, past + q)"),
        x("arange_k", "index / embed", "prologue", "scalar_add", per_k,
          note="the mask's key positions, arange(k)"),
        x("arange_scalar", "index / embed", "prologue", "scalar_add", one, count=2,
          note="the mask's batch and head aranges, one element each"),
        x("mask_offset", "elementwise", "prologue", "scalar_add", per_k,
          note="key positions += offset"),
        x("mask_compare", "elementwise", "prologue", "scalar_add", per_qk,
          note="the causal rule kv_idx <= q_idx, over q x k"),
        x("mask_scalar_copy", "copy", "prologue", "scalar_add", one,
          note="torch.tensor(0.0, device=cuda): a host-to-device copy"),
        x("mask_fill", "fill", "prologue", "scalar_add", one,
          note="the one bf16 scalar fill per pass outside the blocks"),
        x("mask_where", "where / mask", "prologue", "pointwise_add", per_qk,
          note="boolean mask -> 0 / dtype min, over q x k"),
        # Every block.
        x("kv_cache_cat", "cat", "before_attn", "scalar_add", Mono(hidden, 0, 1), count=2,
          note="DynamicCache.update: torch.cat of K, then of V, k x hidden each"),
        x("scale_fill", "fill", "after_qk", "scalar_add", one,
          note="torch.full([], sqrt(head_dim)) for the score scale"),
        x("mask_value_fill", "fill", "before_softmax", "scalar_add", one,
          note="torch.full([], dtype min) for the causal where"),
        x("causal_where", "where / mask", "before_softmax", "pointwise_add",
          Mono(heads, 1, 1), note="torch.where(causal, scores, min), heads x q x k"),
        x("attn_mask_add", "elementwise", "before_softmax", "pointwise_add",
          Mono(heads, 1, 1), note="scores + attention_mask, heads x q x k"),
        x("merge_heads_copy", "copy", "after_attn", "scalar_add", Mono(hidden, 1, 0),
          min_query=2, note="transpose(1, 2).contiguous(); free for a single token"),
        # After the last block.
        ExtraKernel("lm_head", "gemm", "epilogue", ("gemm", "tc", f"{prec}_{prec}"),
                    {"batch": 1, "dimM": per_q, "dimN": vocab, "dimK": hidden,
                     "precM": prec, "precA": prec, "useTensorCore": True},
                    logits=True, note="logits at every position, q x vocab x hidden"),
    )
    return ExtraSet(kernels, name="gpt2-hf-eager",
                    source="HuggingFace GPT-2, eager attention, DynamicCache, "
                           "profiled on an A100 (A100_Energy_Gap_Colab.ipynb)")
