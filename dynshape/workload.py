"""
workload.py
===========

A **dynamic shape stream** -- random batch sizes, varied sequence lengths, both
modes.  Deliberately NOT a serving simulator.

WHAT THIS IS NOT
----------------
There is no L0 (arrival process) and no L1 (scheduler) here.  No Poisson, no KV
admission, no chunked prefill, no preemption.  Those layers decide *which*
shapes a real engine would produce; this module just produces a spread of
shapes so the dynamic-shape machinery downstream has something varied to chew
on.  Swapping this file for a real Vidur timeline is the entire L0/L1
integration -- everything after it stays as-is.

SEEDING
-------
One `numpy.random.Generator` per generator object, never the global stream.
Same seed -> byte-identical stream; different seed -> a different sample of the
same distribution.  (This is the isolated-RNG discipline that a shared global
seed makes impossible: with one global seed you cannot re-roll batch sizes while
holding sequence lengths fixed, because every draw shifts the sequence for
everyone downstream.)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def _spawn_rngs(seed: int, n: int) -> List[np.random.Generator]:
    """`n` independent, reproducible child streams from one master seed.

    Uses `SeedSequence.spawn` rather than `Generator.spawn`, which only exists
    on NumPy >= 1.25 -- this way the package runs on older environments too.
    """
    return [np.random.default_rng(c) for c in np.random.SeedSequence(seed).spawn(n)]


@dataclass(frozen=True)
class Request:
    """One forward pass to evaluate: the three dynamic inputs, plus an id."""

    idx: int
    batch: int
    seqlen: int
    mode: str          # 'prefill' | 'decode'

    @property
    def tokens(self) -> int:
        """Tokens actually pushed through the model this pass."""
        return self.batch * self.seqlen if self.mode == "prefill" else self.batch

    @property
    def label(self) -> str:
        return f"b{self.batch}_s{self.seqlen}_{self.mode}"

    def as_dict(self) -> Dict:
        d = asdict(self)
        d["tokens"] = self.tokens
        return d


@dataclass
class WorkloadConfig:
    """Knobs for the random shape stream.

    batch_choices   : batch sizes to draw from (uniform unless weights given)
    seq_min/seq_max : sequence length range, drawn log-uniform so short prompts
                      are as well represented as long ones
    seq_round_to    : quantise the draw (1 = no quantisation).  Real engines pad
                      to CUDA-graph buckets; this mimics that and, as a side
                      effect, makes the predictor cache actually hit.
    decode_fraction : probability a request is decode rather than prefill
    """

    batch_choices: Sequence[int] = (1, 2, 4, 8, 16, 32)
    batch_weights: Optional[Sequence[float]] = None
    seq_min: int = 64
    seq_max: int = 4096
    seq_round_to: int = 1
    decode_fraction: float = 0.5

    def __post_init__(self):
        if not len(self.batch_choices):
            raise ValueError("batch_choices is empty")
        if min(self.batch_choices) < 1:
            raise ValueError("batch sizes must be >= 1")
        if not (1 <= self.seq_min <= self.seq_max):
            raise ValueError(f"need 1 <= seq_min <= seq_max, got ({self.seq_min}, {self.seq_max})")
        if not (0.0 <= self.decode_fraction <= 1.0):
            raise ValueError("decode_fraction must be in [0, 1]")
        if self.batch_weights is not None:
            if len(self.batch_weights) != len(self.batch_choices):
                raise ValueError("batch_weights must match batch_choices in length")
            if abs(sum(self.batch_weights) - 1.0) > 1e-9:
                raise ValueError("batch_weights must sum to 1")


def quantise(x: int, step: int) -> int:
    """Round up to a multiple of `step`, never below `step`."""
    if step <= 1:
        return max(1, int(x))
    return max(step, int(np.ceil(x / step)) * step)


class RandomShapeGenerator:
    """Draws (batch, seqlen, mode) triples.

    >>> g = RandomShapeGenerator(seed=0)
    >>> reqs = g.sample(5)
    >>> [r.label for r in reqs]        # doctest: +SKIP
    ['b8_s1130_decode', 'b1_s216_prefill', ...]
    """

    def __init__(self, config: Optional[WorkloadConfig] = None, seed: int = 0):
        self.config = config or WorkloadConfig()
        self.seed = seed
        # Independent child streams, so batch / seqlen / mode can each be
        # re-rolled without disturbing the other two.
        self.rng_batch, self.rng_seq, self.rng_mode = _spawn_rngs(seed, 3)

    def reset(self) -> "RandomShapeGenerator":
        self.rng_batch, self.rng_seq, self.rng_mode = _spawn_rngs(self.seed, 3)
        return self

    def _batch(self) -> int:
        c = self.config
        return int(self.rng_batch.choice(np.asarray(c.batch_choices), p=c.batch_weights))

    def _seqlen(self) -> int:
        c = self.config
        if c.seq_min == c.seq_max:
            return quantise(c.seq_min, c.seq_round_to)
        # log-uniform: a 100-token prompt is as likely as a 3000-token one,
        # which a plain uniform draw would badly under-represent.
        lo, hi = np.log(c.seq_min), np.log(c.seq_max)
        return quantise(int(round(float(np.exp(self.rng_seq.uniform(lo, hi))))), c.seq_round_to)

    def _mode(self) -> str:
        return "decode" if self.rng_mode.random() < self.config.decode_fraction else "prefill"

    def sample(self, n: int) -> List[Request]:
        """n random requests -- the dynamic shape stream."""
        if n < 1:
            raise ValueError("n must be >= 1")
        return [Request(idx=i, batch=self._batch(), seqlen=self._seqlen(), mode=self._mode())
                for i in range(n)]


def sweep(batches: Sequence[int], seqlens: Sequence[int],
          modes: Sequence[str] = ("prefill", "decode")) -> List[Request]:
    """A deterministic grid instead of a random stream.

    Random shapes show that arbitrary shapes *work*; a grid shows how power and
    time *move* with each axis.  Both are useful, so both ship.
    """
    out, i = [], 0
    for mode in modes:
        for b in batches:
            for s in seqlens:
                out.append(Request(idx=i, batch=int(b), seqlen=int(s), mode=mode))
                i += 1
    return out


def generation(batch: int, prompt_len: int, n_new_tokens: int,
               context_bucket: int = 128, start_idx: int = 0,
               include_prefill: bool = True) -> List[Request]:
    """One autoregressive generation, step by step -- the decode dynamics.

    A real generation is not one decode call. It is a prefill of the prompt,
    then `n_new_tokens` decode steps whose **KV cache grows by one token each
    time**, so every step is a slightly different shape:

        prefill  b x 1024 tokens        attn 1024 x 1024   (square)
        decode   ctx = 1024             attn    1 x 1024   (strip)
        decode   ctx = 1025             attn    1 x 1025   longer strip
        decode   ctx = 1026             attn    1 x 1026        ...

    Attention work therefore grows linearly across the generation while the
    projection GEMMs stay fixed at `batch` rows -- which is why a long
    conversation gets slower and slightly hotter the further it runs.

    `context_bucket` quantises the context length, so the shape changes once
    every `context_bucket` tokens instead of every single token. Rounding is
    **up**, matching Vidur, whose `kv_cache_prediction_granularity` defaults to
    64 and rounds with `(x + g - 1) // g * g`. Set it to 1 for exact per-token
    contexts -- correct, but it makes every step a distinct shape and collapses
    the predictor cache (see `CachedPredictor`).
    """
    if prompt_len < 1:
        raise ValueError("prompt_len must be >= 1")
    if n_new_tokens < 1:
        raise ValueError("n_new_tokens must be >= 1")

    reqs: List[Request] = []
    i = start_idx
    if include_prefill:
        reqs.append(Request(idx=i, batch=batch, seqlen=prompt_len, mode="prefill"))
        i += 1

    for step in range(n_new_tokens):
        # After the prompt and `step` generated tokens, the cache holds
        # prompt_len + step tokens, which is what this step attends over.
        ctx = prompt_len + step
        reqs.append(Request(idx=i, batch=batch,
                            seqlen=quantise(ctx, context_bucket), mode="decode"))
        i += 1
    return reqs


def conversation(turns: Sequence[Tuple[int, int]], batch: int = 1,
                 context_bucket: int = 128) -> List[Request]:
    """A multi-turn chat: each turn prefills only the new text, then generates.

    `turns` is [(new_prompt_tokens, tokens_to_generate), ...]. The KV cache
    carries across turns, so turn 2's prefill is short but its decode steps
    start from an already-long context -- the shape asymmetry that makes
    later turns cheap to start and expensive to continue.
    """
    reqs: List[Request] = []
    ctx = 0
    for new_tokens, n_gen in turns:
        if new_tokens < 1 or n_gen < 1:
            raise ValueError("each turn needs >= 1 prompt token and >= 1 generated token")
        reqs.append(Request(idx=len(reqs), batch=batch, seqlen=new_tokens, mode="prefill"))
        ctx += new_tokens
        for step in range(n_gen):
            reqs.append(Request(idx=len(reqs), batch=batch,
                                seqlen=quantise(ctx + step, context_bucket),
                                mode="decode"))
        ctx += n_gen
    return reqs
