"""
llama.py
========

**LLaMA-family kernel templates (LLaMA, Qwen3), generated from the model's shape.**

The GPT-2 templates are EnergAIzer traces.  Tracing a 13-14B model the same way
would need the full model at every anchor shape, and EnergAIzer's parser would
again drop every kernel it has no model for.  But a kernel's shape depends only
on the architecture, not on the weights, so this module writes the kernel list
out directly: one entry for every CUDA kernel HuggingFace's eager causal-LM
forward launches, in launch order, as EnergAIzer entries.

    from dynshape.llama import QWEN3_14B, write_llama_templates
    write_llama_templates(QWEN3_14B, "templates/qwen3-14b")
    rw = ShapeRewriter.from_dir("templates/qwen3-14b", model="qwen3-14b", b0=1, s0=128)

`write_llama_templates` writes three prefill anchors and `ShapeRewriter` learns
their scaling law as it does for GPT-2, so the rest of the simulator does not
change.  The kernels that grow with the KV cache rather than with the new
tokens -- the cache appends and, with grouped-query attention, the K/V repeat --
and the mask, embedding and head-merging kernels are extras
(`extras.llama_hf_eager`), exactly as for GPT-2.

WHAT ONE PASS LAUNCHES (prefill; n = new tokens, k = keys, eager attention)
----------------------------------------------------------------------------
    rotary (once)   position cast, cos/sin GEMM, cat, cos, mul, sin, mul, 2 casts   9
    per layer       RMSNorm (cast, pow, mean, +eps, rsqrt, mul, cast, weight mul)  8
                    q, k, v projections                                            3
                    Qwen3 only: RMSNorm on q and on k, per head (QK-norm)          16
                    RoPE on q and on k (mul, neg, cat, mul, add)                   10
                    Q.K^T, scale, mask add, cast, softmax, cast, A.V               7
                    o projection, residual add                                     2
                    RMSNorm                                                        8
                    gate, SiLU, up, mul, down, residual add                        6
    end (once)      RMSNorm, LM head                                               9
    extras          embedding, mask building (10), KV cats (2 L), head merge (L),
                    and with grouped-query attention the K/V repeat (2 L)

LLaMA-2-13B: 1,778 template entries + 130 extras = 1,908 kernels per prefill pass,
1,868 per decode step.  Qwen3-14B: 2,418 + 210 = 2,628 and 2,588.  These counts
are a prediction from transformers' source; the Colab notebook checks them
against the profiler.

FIELD CONVENTIONS
-----------------
Linear layers are written as GPT-2's are, `dimM = tokens`, so the query/key split
of `template.split_seq_exponent` sends them to the query axis and decode needs no
separate law; the LM head keeps `dimN = vocab`, which is how `mixed` finds it.
Attention follows the GPT-2 template: Q.K^T is `M = q, N = k`, A.V is `M = q,
K = k`, softmax is `batch = heads * q, dim = k`, score-sized elementwise kernels
are `heads * q * k`.  Kernels of `n` elements (the RMSNorm's per-token statistics,
the position cast) are sized `n * 128`, because EnergAIzer's elementwise tables
start at 127 elements and these are launch-bound either way.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .kvcache import ModelConfig
from .template import Entry

#: EnergAIzer's elementwise tables start at 127 elements.
_PER_TOKEN_FLOOR = 128


@dataclass(frozen=True)
class LlamaShape:
    """The numbers that set every kernel shape of a LLaMA-family model.

    head_size  the attention head width; 0 means hidden / heads (LLaMA). Qwen3
               sets it explicitly in its config.
    qk_norm    True for Qwen3: an RMSNorm on q and on k, per head, before RoPE.
    """

    name: str = "llama-2-13b"
    hidden: int = 5120
    heads: int = 40
    kv_heads: int = 40
    intermediate: int = 13824
    layers: int = 40
    vocab: int = 32000
    max_position: int = 4096
    head_size: int = 0
    qk_norm: bool = False
    tied_embeddings: bool = False

    def __post_init__(self):
        if self.heads % self.kv_heads:
            raise ValueError("heads must be a multiple of kv_heads")
        if not self.head_size and self.hidden % self.heads:
            raise ValueError("hidden must divide into heads when head_size is not given")

    @property
    def head_dim(self) -> int:
        return self.head_size or self.hidden // self.heads

    @property
    def q_hidden(self) -> int:
        """Width of the attention output, heads x head_dim (= hidden for LLaMA)."""
        return self.heads * self.head_dim

    @property
    def kv_hidden(self) -> int:
        return self.kv_heads * self.head_dim

    @classmethod
    def from_hf_config(cls, cfg, name: Optional[str] = None) -> "LlamaShape":
        """From a transformers `LlamaConfig` or `Qwen3Config`."""
        heads = cfg.num_attention_heads
        return cls(name=name or str(getattr(cfg, "_name_or_path", "llama")).split("/")[-1].lower(),
                   hidden=cfg.hidden_size, heads=heads,
                   kv_heads=getattr(cfg, "num_key_value_heads", None) or heads,
                   intermediate=cfg.intermediate_size, layers=cfg.num_hidden_layers,
                   vocab=cfg.vocab_size, max_position=cfg.max_position_embeddings,
                   head_size=int(getattr(cfg, "head_dim", None) or 0),
                   qk_norm=getattr(cfg, "model_type", "") == "qwen3",
                   tied_embeddings=bool(getattr(cfg, "tie_word_embeddings", False)))

    def model_config(self) -> ModelConfig:
        """The scheduler's view of the model: layers and KV bytes per token.

        `ModelConfig.num_parameters` is GPT-2's formula, so size the KV pool with
        `SchedulerConfig(num_blocks=...)` rather than from the weight count.
        """
        return ModelConfig(name=self.name, num_layers=self.layers, num_heads=self.heads,
                           num_kv_heads=self.kv_heads, head_dim=self.head_dim,
                           hidden=self.hidden, ffn_hidden=self.intermediate,
                           vocab=self.vocab, max_position=self.max_position)

    def parameters(self) -> int:
        """Weight count: embeddings, LM head, attention, MLP and norms."""
        h, qw, kv = self.hidden, self.q_hidden, self.kv_hidden
        layer = (h * qw + 2 * h * kv + qw * h + 3 * h * self.intermediate + 2 * h
                 + (2 * self.head_dim if self.qk_norm else 0))
        heads = 1 if self.tied_embeddings else 2
        return heads * self.vocab * h + self.layers * layer + h

    def kv_bytes_per_token(self, bytes_per_value: int = 2) -> int:
        return 2 * bytes_per_value * self.kv_hidden * self.layers


LLAMA_2_7B = LlamaShape(name="llama-2-7b", hidden=4096, heads=32, kv_heads=32,
                        intermediate=11008, layers=32)
LLAMA_2_13B = LlamaShape()
QWEN3_14B = LlamaShape(name="qwen3-14b", hidden=5120, heads=40, kv_heads=8,
                       intermediate=17408, layers=40, vocab=151936, max_position=40960,
                       head_size=128, qk_norm=True)


def _ew(n: int, op: str, prec: str = "bf16") -> Entry:
    return {"dim": int(n), "op": op, "prec": prec}, ("elementwise",)


def _gemm(m: int, n: int, k: int, batch: int = 1, prec: str = "bf16") -> Entry:
    tc = prec in ("bf16", "fp16")
    return ({"batch": int(batch), "dimM": int(m), "dimN": int(n), "dimK": int(k),
             "precM": prec, "precA": prec, "useTensorCore": tc},
            ("gemm", "tc" if tc else "cuda", f"{prec}_{prec}"))


def _rmsnorm(n: int, h: int) -> List[Tuple[Entry, str]]:
    stat = n * _PER_TOKEN_FLOOR
    return [(_ew(n * h, "typecast_to_fp32", "bf16"), "aten::copy_"),
            (_ew(n * h, "unspecified_scalar", "fp32"), "aten::pow"),
            (_ew(n * h, "scalar_add", "fp32"), "aten::mean"),
            (_ew(stat, "scalar_add", "fp32"), "aten::add"),
            (_ew(stat, "unspecified_scalar", "fp32"), "aten::rsqrt"),
            (_ew(n * h, "pointwise_mul", "fp32"), "aten::mul"),
            (_ew(n * h, "typecast_to_bf16", "fp32"), "aten::copy_"),
            (_ew(n * h, "pointwise_mul", "bf16"), "aten::mul")]


def _rope(n: int, width: int) -> List[Tuple[Entry, str]]:
    """q * cos + rotate_half(q) * sin, where rotate_half is cat(-x2, x1)."""
    return [(_ew(n * width, "pointwise_mul"), "aten::mul"),
            (_ew(n * width // 2, "scalar_mul"), "aten::neg"),
            (_ew(n * width, "scalar_add"), "aten::cat"),
            (_ew(n * width, "pointwise_mul"), "aten::mul"),
            (_ew(n * width, "pointwise_add"), "aten::add")]


def llama_template_labeled(shape: LlamaShape, batch: int, seqlen: int
                           ) -> List[Tuple[Entry, str]]:
    """The prefill kernel list at (batch, seqlen), each entry with the aten op
    expected to launch it -- the label is what the notebook matches against the
    profiler."""
    if batch < 1 or seqlen < 1:
        raise ValueError("batch and seqlen must be >= 1")
    b, s = batch, seqlen
    n = b * s
    h, qw, kv, d = shape.hidden, shape.q_hidden, shape.kv_hidden, shape.head_dim
    scores = b * shape.heads * s * s
    out: List[Tuple[Entry, str]] = []

    # Rotary embedding, computed once per pass in fp32 with autocast off.
    out += [(_ew(n * _PER_TOKEN_FLOOR, "scalar_add", "fp32"), "aten::copy_"),
            (_gemm(n, d // 2, 1, prec="fp32"), "aten::bmm"),
            (_ew(n * d, "scalar_add", "fp32"), "aten::cat"),
            (_ew(n * d, "unspecified_scalar", "fp32"), "aten::cos"),
            (_ew(n * d, "scalar_mul", "fp32"), "aten::mul"),
            (_ew(n * d, "unspecified_scalar", "fp32"), "aten::sin"),
            (_ew(n * d, "scalar_mul", "fp32"), "aten::mul"),
            (_ew(n * d, "typecast_to_bf16", "fp32"), "aten::copy_"),
            (_ew(n * d, "typecast_to_bf16", "fp32"), "aten::copy_")]

    for _ in range(shape.layers):
        out += _rmsnorm(n, h)
        if shape.qk_norm:          # Qwen3: q_proj, q_norm, k_proj, k_norm, v_proj
            out += [(_gemm(n, qw, h), "aten::mm")] + _rmsnorm(n, qw)
            out += [(_gemm(n, kv, h), "aten::mm")] + _rmsnorm(n, kv)
            out += [(_gemm(n, kv, h), "aten::mm")]
        else:
            out += [(_gemm(n, qw, h), "aten::mm"),
                    (_gemm(n, kv, h), "aten::mm"),
                    (_gemm(n, kv, h), "aten::mm")]
        out += _rope(n, qw) + _rope(n, kv)
        # The attention run: everything here reads the key axis.
        out += [(_gemm(s, s, d, batch=b * shape.heads), "aten::bmm"),
                (_ew(scores, "scalar_mul"), "aten::mul"),
                (_ew(scores, "pointwise_add"), "aten::add"),
                (_ew(scores, "typecast_to_fp32", "bf16"), "aten::copy_"),
                (({"batch": b * shape.heads * s, "dim": s, "prec": "fp32"},
                  ("softmax", "fp32")), "aten::_softmax"),
                (_ew(scores, "typecast_to_bf16", "fp32"), "aten::copy_"),
                (_gemm(s, d, s, batch=b * shape.heads), "aten::bmm")]
        out += [(_gemm(n, h, qw), "aten::mm"),
                (_ew(n * h, "pointwise_add"), "aten::add")]
        out += _rmsnorm(n, h)
        out += [(_gemm(n, shape.intermediate, h), "aten::mm"),
                (_ew(n * shape.intermediate, "silu"), "aten::silu"),
                (_gemm(n, shape.intermediate, h), "aten::mm"),
                (_ew(n * shape.intermediate, "pointwise_mul"), "aten::mul"),
                (_gemm(n, h, shape.intermediate), "aten::mm"),
                (_ew(n * h, "pointwise_add"), "aten::add")]

    out += _rmsnorm(n, h)
    out.append((_gemm(n, shape.vocab, h), "aten::mm"))
    return out


def llama_template(shape: LlamaShape, batch: int, seqlen: int) -> List[Entry]:
    """The prefill kernel list at (batch, seqlen), as EnergAIzer entries."""
    return [e for e, _label in llama_template_labeled(shape, batch, seqlen)]


def n_template_entries(shape: LlamaShape) -> int:
    return 9 + (44 + (16 if shape.qk_norm else 0)) * shape.layers + 9


def llama_extras(shape: LlamaShape):
    """The extra kernels for this shape (`extras.llama_hf_eager`)."""
    from .extras import llama_hf_eager
    return llama_hf_eager(shape.q_hidden, shape.kv_hidden)


#: The three prefill anchors: (b0, s0), another batch, another seqlen.
ANCHORS = ((1, 128), (2, 128), (1, 512))


def write_llama_templates(shape: LlamaShape, template_dir: str,
                          anchors=ANCHORS) -> List[str]:
    """Write the anchor templates in EnergAIzer's file format and naming.

    Load them with `ShapeRewriter.from_dir(template_dir, model=shape.name,
    b0=anchors[0][0], s0=anchors[0][1])`.
    """
    os.makedirs(template_dir, exist_ok=True)
    paths = []
    for b, s in anchors:
        path = os.path.join(template_dir, f"{shape.name}_pbf16_b{b}_s{s}_modeprefill.json")
        with open(path, "w") as f:
            json.dump([[q, list(op)] for q, op in llama_template(shape, b, s)], f)
        paths.append(path)
    return paths
