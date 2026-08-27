"""
replay.py
=========

**Turning simulated traffic into something a real engine will execute** -- the
same requests, the same sizes, arriving at the same times.

Without this the comparison is meaningless in a way that is easy to miss.  The
scheduler's behaviour is a function of *when* requests arrive: fire them all at
once and vLLM batches them all at once, and the resulting power trace is a
picture of a different workload than the one the simulator priced.

THE CONSTRAINT THAT BITES FIRST
-------------------------------
**GPT-2's context window is 1024 tokens.**  The simulator does not know that --
`rewrite_dims` is arithmetic, so it will happily produce a shape for a
2048-token context and the predictor will happily price it.  Nothing is
inconsistent; those shapes simply do not correspond to anything GPT-2 can run.
Real vLLM refuses them outright.

So a run that is going to be validated has to be generated inside the window in
the first place, which is what `build_replay_traffic` is for.  This is worth
stating plainly rather than clamping quietly: **every simulated run so far with
`max_tokens=2048` was extrapolating past the model's context**, and that is a
property of those runs, not a bug in the replay.

EXACT LENGTHS, NOT APPROXIMATE ONES
-----------------------------------
Two details make the executed workload match the priced one token for token:

* prompts are sent as **token ids**, not text.  Tokenising a string of the right
  character count gives a token count that is merely close, and "close" on the
  prefill axis moves the largest GEMM in the model.
* outputs are pinned with `min_tokens == max_tokens` and `ignore_eos=True`.
  Otherwise GPT-2 emits an end-of-text token whenever it likes and the decode
  phase -- which is most of the trace -- ends early and at a different length
  for every request.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

#: GPT-2's positional embedding table. Prompt + generated must fit inside it.
GPT2_MAX_POSITIONS = 1024

#: GPT-2's vocabulary, for synthesising prompt token ids.
GPT2_VOCAB = 50257


@dataclass(frozen=True)
class ReplayRequest:
    """One request, in the only three terms both engines agree on."""

    index: int
    arrival_s: float
    n_prompt: int
    n_decode: int

    @property
    def total_tokens(self) -> int:
        return self.n_prompt + self.n_decode


def check_fits_context(requests: Sequence, max_positions: int = GPT2_MAX_POSITIONS) -> None:
    """Raise if any request would run past the model's context window.

    Loud rather than clamped: silently trimming would make the executed workload
    differ from the priced one, which is exactly the thing this module exists to
    prevent.
    """
    bad = [(r.index if hasattr(r, "index") else i, r.n_prompt + r.n_decode)
           for i, r in enumerate(requests)
           if getattr(r, "n_prompt", getattr(r, "num_prefill_tokens", 0))
           + getattr(r, "n_decode", getattr(r, "num_decode_tokens", 0)) > max_positions]
    if bad:
        worst = max(b[1] for b in bad)
        raise ValueError(
            f"{len(bad)} of {len(requests)} requests exceed the {max_positions}-token "
            f"context window (worst: {worst} tokens). The simulator will price these "
            "happily -- its shape arithmetic has no notion of a context limit -- but "
            "GPT-2 cannot run them. Regenerate with build_replay_traffic().")


def build_replay_traffic(num_requests: int = 120, qps: float = 6.0, cv: float = 2.0,
                         min_tokens: int = 64, max_total_tokens: int = 960,
                         theta: float = 0.85, prefill_to_decode_ratio: float = 4.0,
                         seed: int = 0, interval: str = "gamma"):
    """Traffic that fits inside GPT-2's context window, so it can be replayed.

    `max_total_tokens` defaults to 960 rather than 1024: vLLM counts the prompt
    plus every generated token against the window, and leaving a little headroom
    avoids losing requests to an off-by-a-few at the boundary.
    """
    from .traffic import TrafficConfig, generate_traffic

    if max_total_tokens > GPT2_MAX_POSITIONS:
        raise ValueError(
            f"max_total_tokens={max_total_tokens} exceeds GPT-2's "
            f"{GPT2_MAX_POSITIONS}-token window")

    reqs = generate_traffic(TrafficConfig(
        interval=interval, qps=qps, cv=cv,
        length="zipf", min_tokens=min_tokens, max_tokens=max_total_tokens,
        theta=theta, prefill_to_decode_ratio=prefill_to_decode_ratio,
        num_requests=num_requests, seed=seed))
    check_fits_context(reqs, GPT2_MAX_POSITIONS)
    return reqs


def to_spec(requests: Sequence) -> List[ReplayRequest]:
    """`SimRequest` list -> the replay spec, in arrival order."""
    out = []
    for i, r in enumerate(sorted(requests, key=lambda x: x.arrived_at)):
        out.append(ReplayRequest(index=i, arrival_s=float(r.arrived_at),
                                 n_prompt=int(r.num_prefill_tokens),
                                 n_decode=int(r.num_decode_tokens)))
    return out


def save_spec(spec: Sequence[ReplayRequest], path: str) -> str:
    with open(path, "w") as f:
        json.dump([{"index": s.index, "arrival_s": s.arrival_s,
                    "n_prompt": s.n_prompt, "n_decode": s.n_decode} for s in spec], f)
    return path


def load_spec(path: str) -> List[ReplayRequest]:
    with open(path) as f:
        return [ReplayRequest(**d) for d in json.load(f)]


def prompt_token_ids(n_tokens: int, seed: int = 0, vocab: int = GPT2_VOCAB,
                     avoid_special: bool = True) -> List[int]:
    """Exactly `n_tokens` token ids, drawn reproducibly.

    Token *ids*, not text: tokenising a string of the right character count gives
    a token count that is only approximately right, and on the prefill axis that
    moves the largest GEMM in the model.

    Content is random because the shapes -- which is all either side models --
    do not depend on it.  Worth being explicit that this makes the *outputs*
    meaningless; it is a power benchmark, not a quality one.
    """
    import numpy as np

    if n_tokens < 1:
        raise ValueError("n_tokens must be >= 1")
    hi = vocab - 1 if avoid_special else vocab
    rng = np.random.default_rng(seed)
    return rng.integers(0, hi, size=n_tokens).tolist()


def spec_summary(spec: Sequence[ReplayRequest]) -> Dict:
    import numpy as np

    p = np.array([s.n_prompt for s in spec], dtype=float)
    d = np.array([s.n_decode for s in spec], dtype=float)
    a = np.array([s.arrival_s for s in spec], dtype=float)
    span = float(a.max() - a.min()) if a.size > 1 else 0.0
    return {
        "requests": len(spec),
        "span_s": span,
        "arrival_rate_qps": (len(spec) / span) if span > 0 else float("inf"),
        "prompt_tokens_mean": float(p.mean()), "prompt_tokens_max": int(p.max()),
        "decode_tokens_mean": float(d.mean()), "decode_tokens_max": int(d.max()),
        "total_tokens_max": int((p + d).max()),
        "total_prompt_tokens": int(p.sum()), "total_decode_tokens": int(d.sum()),
        "fits_gpt2_window": bool((p + d).max() <= GPT2_MAX_POSITIONS),
    }
