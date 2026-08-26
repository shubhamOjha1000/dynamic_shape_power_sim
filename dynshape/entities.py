"""
entities.py
===========

The two objects the scheduler moves around: a **request** and a **batch**.

Ported from Vidur's `entities/request.py` and `entities/batch.py`, trimmed to
one replica and one pipeline stage, and renamed `SimRequest` so it cannot be
confused with `workload.Request` (which is a *shape*, not a request -- one
forward pass to evaluate, with no lifecycle at all).

THE ONE PIECE OF LIFECYCLE THAT MATTERS DOWNSTREAM
--------------------------------------------------
`num_processed_tokens` is the KV cache length, and it moves in a way that is
easy to get wrong.  Vidur's rule, reproduced exactly:

    prefill chunk of n tokens   ->  num_processed_tokens += n
    prefill just completed      ->  is_prefill_complete = True, and
                                    num_processed_tokens += 1
                                    (the prefill pass emits the first token)
    decode step                 ->  num_processed_tokens += 1

So at a decode iteration `num_processed_tokens` is exactly the number of keys
already in the cache, and this step attends over one more than that.  That `+1`
is not a fudge -- it was measured (see `template.rewrite_dims`), and it is why
`Piece.key_len` adds one for decode and not for prefill.

RESTART, THE POWER EVENT
------------------------
When KV memory runs out the scheduler picks a victim and calls `restart()`.
Everything the victim had generated is thrown away and re-prefilled: all its
processed tokens become *prompt* tokens again.  Real GPU work, real watts, no
new output, and **nothing in the arrival pattern predicts it**.  FSTS cannot
produce this event at all -- it has no preemption -- which is the main reason
the block accounting here comes from Vidur.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

_ids = itertools.count()


def _next_id() -> int:
    return next(_ids)


def reset_ids() -> None:
    """Restart id numbering -- tests want request 0 to be request 0."""
    global _ids
    _ids = itertools.count()


class SimRequest:
    """One user request, from arrival to completion.

    `cached_prefix_tokens` is FSTS's prefix-cache annotation, which Vidur has no
    concept of: when a prompt starts with text the GPU has already processed
    (a system prompt, earlier chat turns, the same retrieved documents), the
    engine reuses the stored KV instead of recomputing it.  Only
    `num_prefill_tokens - cached_prefix_tokens` is actually executed, but the
    full prompt still occupies KV.  Omitting this overstates the *highest-power*
    phase on every single request -- a 2000-token system prompt with a 100-token
    question is a 21x overstatement of prefill work -- and because it is the
    same error in the same direction every time, it survives aggregation.

    It is a static annotation: whoever builds the workload must know the hit
    rate in advance.  A real engine discovers hits dynamically and evicts under
    pressure; neither Vidur nor FSTS models that.
    """

    __slots__ = (
        "id", "arrived_at", "_num_prefill_tokens", "_num_decode_tokens",
        "cached_prefix_tokens", "_num_processed_tokens", "_is_prefill_complete",
        "_scheduled", "_completed", "_num_restarts", "scheduled_at",
        "prefill_completed_at", "completed_at", "token_times", "_restart_work_tokens",
    )

    def __init__(self, arrived_at: float, num_prefill_tokens: int,
                 num_decode_tokens: int, cached_prefix_tokens: int = 0,
                 request_id: Optional[int] = None):
        if num_prefill_tokens < 1:
            raise ValueError("num_prefill_tokens must be >= 1")
        if num_decode_tokens < 1:
            raise ValueError("num_decode_tokens must be >= 1")
        if not (0 <= cached_prefix_tokens < num_prefill_tokens):
            raise ValueError(
                "cached_prefix_tokens must be >= 0 and strictly less than "
                "num_prefill_tokens -- a fully cached prompt has no work to do")

        self.id = _next_id() if request_id is None else request_id
        self.arrived_at = float(arrived_at)
        # What is *executed*: the cached prefix is skipped, but it still occupies
        # KV, which is why the two are tracked separately.
        self._num_prefill_tokens = int(num_prefill_tokens) - int(cached_prefix_tokens)
        self._num_decode_tokens = int(num_decode_tokens)
        self.cached_prefix_tokens = int(cached_prefix_tokens)

        self._num_processed_tokens = 0
        self._is_prefill_complete = False
        self._scheduled = False
        self._completed = False
        self._num_restarts = 0
        self._restart_work_tokens = 0

        self.scheduled_at: Optional[float] = None
        self.prefill_completed_at: Optional[float] = None
        self.completed_at: Optional[float] = None
        self.token_times: List[float] = []

    # -- shape ---------------------------------------------------------------

    @property
    def num_prefill_tokens(self) -> int:
        """Prompt tokens actually executed (cached prefix already deducted)."""
        return self._num_prefill_tokens

    @property
    def num_decode_tokens(self) -> int:
        return self._num_decode_tokens

    @property
    def total_tokens(self) -> int:
        return self._num_prefill_tokens + self._num_decode_tokens

    @property
    def num_processed_tokens(self) -> int:
        """KV length owned by this request, excluding the cached prefix."""
        return self._num_processed_tokens

    @property
    def context_tokens(self) -> int:
        """Full attention context: the cached prefix plus what we processed."""
        return self.cached_prefix_tokens + self._num_processed_tokens

    @property
    def kv_tokens(self) -> int:
        """KV slots this request occupies, prefix included."""
        return self.cached_prefix_tokens + self._num_processed_tokens

    # -- state ---------------------------------------------------------------

    @property
    def is_prefill_complete(self) -> bool:
        return self._is_prefill_complete

    @property
    def scheduled(self) -> bool:
        return self._scheduled

    @property
    def completed(self) -> bool:
        return self._completed

    @property
    def num_restarts(self) -> int:
        return self._num_restarts

    @property
    def restart_work_tokens(self) -> int:
        """Prompt tokens re-executed because of restarts -- pure wasted work."""
        return self._restart_work_tokens

    @property
    def num_generated_tokens(self) -> int:
        return max(0, self._num_processed_tokens - self._num_prefill_tokens)

    # -- lifecycle -----------------------------------------------------------

    def on_batch_schedule(self, t: float) -> None:
        if not self._scheduled:
            self._scheduled = True
            if self._num_restarts == 0:
                self.scheduled_at = t

    def on_batch_end(self, t: float, num_tokens_processed: int) -> None:
        self._num_processed_tokens += num_tokens_processed
        if self._num_processed_tokens > self.total_tokens:
            raise AssertionError(
                f"request {self.id} processed {self._num_processed_tokens} of "
                f"{self.total_tokens} tokens -- the scheduler over-issued work")

        if self._num_processed_tokens == self._num_prefill_tokens:
            self._is_prefill_complete = True
            # Prefill emits the first output token as it completes.
            self._num_processed_tokens += 1
            if self.prefill_completed_at is None:
                self.prefill_completed_at = t
            self.token_times.append(t)
        elif self._is_prefill_complete:
            self.token_times.append(t)

        if self._num_processed_tokens == self.total_tokens:
            self._completed = True
            self.completed_at = t

    def restart(self) -> None:
        """Throw away the KV and re-prefill everything decoded so far."""
        total = self._num_prefill_tokens + self._num_decode_tokens
        self._restart_work_tokens += self._num_processed_tokens
        self._num_prefill_tokens = self._num_processed_tokens
        self._num_decode_tokens = total - self._num_prefill_tokens
        if self._num_decode_tokens < 1:
            raise AssertionError(
                f"request {self.id} restarted with nothing left to generate -- "
                "it should have completed and been freed")
        self._num_processed_tokens = 0
        self._is_prefill_complete = False
        self._scheduled = False
        self._completed = False
        self._num_restarts += 1
        self.token_times.clear()

    # -- metrics -------------------------------------------------------------

    @property
    def ttft_s(self) -> Optional[float]:
        return None if not self.token_times else self.token_times[0] - self.arrived_at

    @property
    def e2e_s(self) -> Optional[float]:
        return None if self.completed_at is None else self.completed_at - self.arrived_at

    @property
    def scheduling_delay_s(self) -> Optional[float]:
        return None if self.scheduled_at is None else self.scheduled_at - self.arrived_at

    def itl_s(self) -> List[float]:
        return [b - a for a, b in zip(self.token_times, self.token_times[1:])]

    def to_dict(self) -> dict:
        itl = self.itl_s()
        return {
            "id": self.id,
            "arrived_at": self.arrived_at,
            "num_prefill_tokens": self._num_prefill_tokens,
            "num_decode_tokens": self._num_decode_tokens,
            "cached_prefix_tokens": self.cached_prefix_tokens,
            "num_processed_tokens": self._num_processed_tokens,
            "completed": self._completed,
            "scheduled_at": self.scheduled_at,
            "prefill_completed_at": self.prefill_completed_at,
            "completed_at": self.completed_at,
            "scheduling_delay_s": self.scheduling_delay_s,
            "ttft_s": self.ttft_s,
            "e2e_s": self.e2e_s,
            "mean_itl_ms": (1000.0 * sum(itl) / len(itl)) if itl else None,
            "num_restarts": self._num_restarts,
            "restart_work_tokens": self._restart_work_tokens,
        }

    def __repr__(self) -> str:
        return (f"SimRequest(id={self.id}, t={self.arrived_at:.3f}, "
                f"{self._num_prefill_tokens}p/{self._num_decode_tokens}d, "
                f"processed={self._num_processed_tokens})")


@dataclass(frozen=True)
class Piece:
    """One request's contribution to one iteration -- what L2 needs to know.

    This is the entire interface between the scheduler and the kernel expander.
    Everything above it is queueing; everything below it is shapes.

    query_len : rows of new tokens this request pushes through
    key_len   : keys attention reads, which is *not* the same number
    """

    kind: str          # 'prefill' | 'decode'
    request_id: int
    tokens: int        # new tokens processed this iteration
    context: int       # KV already present before this iteration

    @property
    def query_len(self) -> int:
        return self.tokens

    @property
    def key_len(self) -> int:
        """Keys attended over.

        prefill: the prior context plus this chunk -- the chunk's own tokens are
                 written to the cache before attention, and a chunk attends to
                 everything before it as well as itself.
        decode:  the context plus the one new token, for the same reason.  The
                 `+1` here is measured, not assumed.
        """
        return self.context + self.tokens


class Batch:
    """One iteration's work: which requests, and how many tokens each.

    A batch is **mixed by construction** -- decodes and prefill chunks in the
    same forward pass.  That is the whole point of chunked prefill, and it is
    what makes `num_prefill_tokens` and `num_decode_tokens` both non-zero for
    most iterations in a busy engine.
    """

    def __init__(self, requests: List[SimRequest], num_tokens: List[int]):
        if len(requests) != len(num_tokens):
            raise ValueError("requests and num_tokens must be the same length")
        if not requests:
            raise ValueError("a batch needs at least one request")
        if any(n < 1 for n in num_tokens):
            raise ValueError("every request in a batch must process >= 1 token")
        self.requests = list(requests)
        self.num_tokens = [int(n) for n in num_tokens]
        # Snapshot the phase *now*: on_batch_end flips is_prefill_complete, so
        # asking afterwards gives the wrong answer for the iteration just run.
        self._pieces = tuple(
            Piece(kind="decode" if r.is_prefill_complete else "prefill",
                  request_id=r.id, tokens=n, context=r.context_tokens)
            for r, n in zip(self.requests, self.num_tokens)
        )

    @property
    def pieces(self) -> Tuple[Piece, ...]:
        return self._pieces

    @property
    def size(self) -> int:
        return len(self.requests)

    @property
    def total_num_tokens(self) -> int:
        return sum(self.num_tokens)

    @property
    def num_prefill_tokens(self) -> int:
        return sum(p.tokens for p in self._pieces if p.kind == "prefill")

    @property
    def num_decode_tokens(self) -> int:
        return sum(p.tokens for p in self._pieces if p.kind == "decode")

    @property
    def decode_batch(self) -> int:
        return sum(1 for p in self._pieces if p.kind == "decode")

    @property
    def prefill_batch(self) -> int:
        return sum(1 for p in self._pieces if p.kind == "prefill")

    @property
    def is_mixed(self) -> bool:
        return self.decode_batch > 0 and self.prefill_batch > 0

    @property
    def request_ids(self) -> List[int]:
        return [r.id for r in self.requests]

    def on_schedule(self, t: float) -> None:
        for r in self.requests:
            r.on_batch_schedule(t)

    def on_batch_end(self, t: float) -> None:
        for r, n in zip(self.requests, self.num_tokens):
            r.on_batch_end(t, n)

    def __repr__(self) -> str:
        return (f"Batch(size={self.size}, {self.decode_batch}d+{self.prefill_batch}p, "
                f"tokens={self.total_num_tokens})")
