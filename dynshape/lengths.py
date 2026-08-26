"""
lengths.py
==========

**L0, half two: how big is the request?**

Ported from Vidur's `request_generator/*_request_length_generator.py`.  One call,
two numbers out: `next_lengths() -> (prefill_tokens, decode_tokens)` -- the
prompt the user typed, and how many tokens the model will generate back.

WHY THIS DECIDES THE WATTAGE
----------------------------
The arrival generator sets batch size, which is the M dimension of every GEMM.
The length distribution controls something different -- **which phase
dominates** -- and that is the difference between compute-bound and
memory-bound:

    P:D  0.5   "write me a story"      short prompt, long generation
                -> one quick prefill, then hundreds of decode steps
                -> mostly memory-bound, low and flat
    P:D 19.0   "summarise this doc"    long prompt, short answer
                -> a big compute burst, then almost nothing
                -> spiky and compute-bound, roughly double the peak

Same requests per second, same GPU, roughly double the peak power, from one
number.  Vidur curates datasets spanning P:D 0.65 to 15.7 for exactly this
reason.

THE SHARED TRICK
----------------
Three of the four generators work identically: draw a **total** size, then split
it with one knob.

    decode  = total / (1 + prefill_to_decode_ratio)
    prefill = total - decode

WHAT NONE OF THEM MODEL
-----------------------
Correlation between prompt and output length.  In reality the two are related
-- long documents get long summaries -- but all four draw a total and split it
by a fixed ratio, so that structure is simply absent from every option.  Stated
here because it is a property of the workload model, not a bug to be fixed
locally.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np

EPS = 1e-8


def _split(total: float, pd_ratio: float) -> Tuple[int, int]:
    """Vidur's split, with the clamping that keeps both halves runnable."""
    if pd_ratio <= 0:
        raise ValueError("prefill_to_decode_ratio must be > 0")
    decode = math.ceil(total / (1.0 + pd_ratio))
    prefill = int(round(total - decode))
    return max(1, prefill), max(1, int(decode))


class LengthGenerator:
    """Interface: `next_lengths()` -> (prefill_tokens, decode_tokens).

    `(None, None)` means the source is exhausted (trace replay only).
    """

    name = "abstract"

    def next_lengths(self) -> Tuple[Optional[int], Optional[int]]:
        raise NotImplementedError

    def describe(self) -> str:
        return self.name


class FixedLength(LengthGenerator):
    """Every request identical.  No variation at all.

    Useful for changing exactly one thing and watching its effect cleanly --
    which is the only situation where it is the right choice.
    """

    name = "fixed"

    def __init__(self, prefill_tokens: int, decode_tokens: int):
        if prefill_tokens < 1 or decode_tokens < 1:
            raise ValueError("prefill_tokens and decode_tokens must be >= 1")
        self.prefill_tokens = int(prefill_tokens)
        self.decode_tokens = int(decode_tokens)

    def next_lengths(self) -> Tuple[int, int]:
        return self.prefill_tokens, self.decode_tokens

    def describe(self) -> str:
        return f"fixed ({self.prefill_tokens}p / {self.decode_tokens}d)"


class UniformLength(LengthGenerator):
    """Total drawn uniformly, then split.

    **Not how real traffic looks**, and worth avoiding for power work: a
    950-token request is as common as a 150-token one, which inflates both KV
    pressure and the prefill share of the trace.  It spreads samples evenly
    across the range, which is occasionally what you want when probing.
    """

    name = "uniform"

    def __init__(self, min_tokens: int, max_tokens: int,
                 prefill_to_decode_ratio: float = 4.0, seed: int = 0):
        if not (1 <= min_tokens <= max_tokens):
            raise ValueError("need 1 <= min_tokens <= max_tokens")
        self.min_tokens = int(min_tokens)
        self.max_tokens = int(max_tokens)
        self.pd_ratio = float(prefill_to_decode_ratio)
        self.rng = np.random.default_rng(seed)

    def next_lengths(self) -> Tuple[int, int]:
        total = self.rng.uniform(self.min_tokens, self.max_tokens)
        return _split(total, self.pd_ratio)

    def describe(self) -> str:
        return f"uniform ({self.min_tokens}-{self.max_tokens}, P:D {self.pd_ratio})"


class ZipfGenerator:
    """Vidur's `utils/zipf_generator.py`, ported to an isolated RNG.

    A few huge, many tiny -- the long tail.  Closest of the four to real chat
    traffic: most people send a sentence, occasionally someone pastes a whole
    document.  **Those occasional giants are what fill the KV cache and trigger
    preemption**, so the tail matters far more than the mean.

    `theta` is the skew (0 -> near-uniform, ~0.99 -> very heavy tail).
    `scramble` breaks the rank/size monotonicity by hashing, which decorrelates
    consecutive draws.
    """

    def __init__(self, min_val: int, max_val: int, theta: float = 0.6,
                 scramble: bool = False, seed: int = 0):
        if min_val < 1 or max_val < min_val:
            raise ValueError("need 1 <= min_val <= max_val")
        if not (0.0 <= theta < 1.0):
            raise ValueError("theta must be in [0, 1)")
        self._min = int(min_val)
        self._max = int(max_val)
        self._items = self._max - self._min + 1
        self._theta = float(theta)
        self._zeta_2 = self._zeta(2, self._theta)
        self._alpha = 1.0 / (1.0 - self._theta)
        self._zetan = self._zeta(self._items, self._theta)
        self._eta = ((1 - np.power(2.0 / self._items, 1 - self._theta))
                     / (1 - self._zeta_2 / (self._zetan + EPS)))
        self._scramble = scramble
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    @staticmethod
    def _zeta(count: float, theta: float) -> float:
        return float(np.sum(1.0 / np.power(np.arange(1, count), theta)))

    def _next(self) -> int:
        u = self._rng.random()
        uz = u * self._zetan
        if uz < 1.0:
            return self._min
        if uz < 1.0 + np.power(0.5, self._theta):
            return self._min + 1
        return self._min + int(self._items * np.power(
            self._eta * u - self._eta + 1, self._alpha))

    def next(self) -> int:
        v = self._next()
        if self._scramble:
            v = self._min + hash(str(v) + str(self._seed)) % self._items
        # The closed form can overshoot by a rounding step at the extremes.
        return int(min(max(v, self._min), self._max))


class ZipfLength(LengthGenerator):
    """Zipf-distributed totals, then split.  The realistic synthetic option."""

    name = "zipf"

    def __init__(self, min_tokens: int, max_tokens: int, theta: float = 0.6,
                 scramble: bool = False, prefill_to_decode_ratio: float = 4.0,
                 seed: int = 0):
        self.zipf = ZipfGenerator(min_tokens, max_tokens, theta, scramble, seed)
        self.pd_ratio = float(prefill_to_decode_ratio)
        self.min_tokens = int(min_tokens)
        self.max_tokens = int(max_tokens)
        self.theta = theta

    def next_lengths(self) -> Tuple[int, int]:
        return _split(float(self.zipf.next()), self.pd_ratio)

    def describe(self) -> str:
        return (f"zipf ({self.min_tokens}-{self.max_tokens}, theta={self.theta}, "
                f"P:D {self.pd_ratio})")


class TraceLength(LengthGenerator):
    """Replay measured (prefill, decode) pairs.

    Ported from Vidur, including the careful clipping: an over-long request has
    prompt and output trimmed **proportionally** rather than one side lopped
    off, so its P:D character survives the clamp.

    `prefill_scale_factor` is a genuinely useful what-if -- "what if prompts get
    twice as long as people paste more context?" is `prefill_scale_factor=2.0`.
    """

    name = "trace"

    def __init__(self, prefill_tokens: Sequence[int], decode_tokens: Sequence[int],
                 prefill_scale_factor: float = 1.0, decode_scale_factor: float = 1.0,
                 max_tokens: int = 4096, seed: int = 0, shuffle: bool = True):
        p = np.asarray(list(prefill_tokens), dtype=float) * float(prefill_scale_factor)
        d = np.asarray(list(decode_tokens), dtype=float) * float(decode_scale_factor)
        if p.size == 0 or p.size != d.size:
            raise ValueError("prefill and decode columns must be non-empty and equal length")

        p = p.astype(int)
        d = d.astype(int)
        total = p + d
        over = np.clip(total - int(max_tokens), 0, None)
        # Deduct proportionally, so the P:D ratio survives the clip.
        with np.errstate(divide="ignore", invalid="ignore"):
            p_ratio = np.where(total > 0, p / total, 0.5)
            d_ratio = np.where(total > 0, d / total, 0.5)
        p = p - np.ceil(over * p_ratio).astype(int)
        d = d - np.ceil(over * d_ratio).astype(int)
        p = np.clip(p, 1, None)
        d = np.clip(d, 1, None)

        if shuffle:
            order = np.random.default_rng(seed).permutation(p.size)
            p, d = p[order], d[order]

        self.prefill = p
        self.decode = d
        self.max_tokens = int(max_tokens)
        self._i = 0

    @classmethod
    def from_csv(cls, path: str, prefill_column: str = "num_prefill_tokens",
                 decode_column: str = "num_decode_tokens", **kwargs) -> "TraceLength":
        import pandas as pd
        df = pd.read_csv(path)
        for c in (prefill_column, decode_column):
            if c not in df.columns:
                raise ValueError(f"{path!r} has no column {c!r}; found {list(df.columns)}")
        return cls(df[prefill_column].to_numpy(), df[decode_column].to_numpy(), **kwargs)

    @property
    def pd_ratio_percentiles(self) -> dict:
        """P:D percentiles -- read this to catch a scale factor that pushed the
        distribution somewhere unrealistic, since P:D is the knob that moves
        power most."""
        r = self.prefill / self.decode
        return {f"p{q}": float(np.percentile(r, q)) for q in (25, 50, 75, 90, 95, 99)}

    def next_lengths(self) -> Tuple[Optional[int], Optional[int]]:
        if self._i >= len(self.prefill):
            return None, None
        p, d = int(self.prefill[self._i]), int(self.decode[self._i])
        self._i += 1
        return p, d

    def describe(self) -> str:
        return f"trace ({len(self.prefill)} rows, max_tokens={self.max_tokens})"


LENGTH_GENERATORS = {
    "fixed": FixedLength,
    "uniform": UniformLength,
    "zipf": ZipfLength,
    "trace": TraceLength,
}
