"""
scheduler.py
============

**L1: who is in this batch?**

That is the entire job.  Every iteration the engine answers one question, and a
scheduler policy is just a different answer.  This one is **decode-first chunked
prefill** -- vLLM V1's default, and what SGLang and TensorRT-LLM do too.

    decodes = [every running request whose prefill is done]   # protected
    budget  = chunk_size - len(decodes)                       # what is left
    prefill = fills the remainder, sliced to fit

Two rules, and they produce the mixed batches this simulator exists to price:
**decodes go first, always; prefill takes whatever is left, sliced to fit.**

WHAT THIS BUYS, AND WHAT IT COSTS
---------------------------------
Thirty people are chatting and a new user pastes a 4000-token document.

    without chunking   iteration 1: all 4000 prefill tokens
                                    the 30 chatters get nothing -- everyone stutters
                                    ~400 W, pure compute-bound
    with chunking      iteration 1: 30 decode + 2018 prefill tokens
                       iteration 2: 30 decode + 1982 prefill tokens
                                    nobody stutters; ~250 W each, mixed

Existing users are protected; new users are sacrificed.  Because decodes come
off the top, the busier the system the less budget survives for prefill -- at
1800 chatters a 4000-token prompt takes 17 iterations instead of 2.  That is a
deliberate trade, not a bug: losing your place mid-sentence feels worse than
waiting a little longer for the first word.

For power it means every iteration now contains **both** compute-bound prefill
and memory-bound decode instead of alternating between them: same total work,
same energy, a much smoother ramp.  And mildly counterintuitively, **a busier
GPU can draw less instantaneous power**, because decode is memory-bound and
cheaper per token than prefill.

THE HYBRID, STATED PLAINLY
--------------------------
The batching policy is FSTS's (vLLM V1 semantics).  The memory accounting is
Vidur's (blocks, watermark, preemption/restart).  FSTS has no preemption at all,
so the recompute spike -- a real high-power event triggered by KV exhaustion
rather than by any request arriving -- cannot occur in it.  Vidur models that
fully, and it is the single most interesting thing L1 contributes to a power
trace, so it is worth the hybrid.

One faithful detail carried over from Vidur rather than FSTS: decodes consume
budget one token at a time and are *skipped* if the budget runs out, instead of
being taken off the top unconditionally.  With the usual settings
(`max_num_seqs` 256 below `chunk_size` 2048) the two are identical; they differ
only when the seat cap exceeds the token budget, where Vidur's form degrades
gracefully and FSTS's would go negative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .entities import Batch, SimRequest
from .kvcache import BlockAllocator, HardwareConfig, MemoryPlanner, ModelConfig


@dataclass
class SchedulerConfig:
    """FSTS's four-field `EngineConfig`, plus Vidur's block knobs.

    chunk_size      : token budget per iteration (vLLM's `--max-num-batched-tokens`)
    max_num_seqs    : seat cap -- also what stops prefill being fully starved,
                      since the worst case still leaves `chunk_size - max_num_seqs`
                      tokens of budget for prefill
    block_size      : KV page size in tokens
    num_blocks      : None -> size the pool from hardware (see `BlockAllocator`)
    max_tokens      : longest request the workload permits; **sizes the pool**
    enable_chunked_prefill : False gives the pre-Sarathi behaviour -- a prefill
                      takes a whole iteration to itself.  Kept because the
                      contrast is the clearest way to see what chunking does to
                      a power trace.
    """

    chunk_size: int = 2048
    max_num_seqs: int = 256
    block_size: int = 16
    num_blocks: Optional[int] = None
    watermark_blocks_fraction: float = 0.01
    max_tokens: int = 4096
    enable_chunked_prefill: bool = True

    def __post_init__(self):
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if self.max_num_seqs < 1:
            raise ValueError("max_num_seqs must be >= 1")
        if self.enable_chunked_prefill and self.max_num_seqs > self.chunk_size:
            raise ValueError(
                f"max_num_seqs ({self.max_num_seqs}) above chunk_size "
                f"({self.chunk_size}) would let decodes consume the whole budget "
                "and starve prefill entirely -- the seat cap is what prevents that")


class ChunkedPrefillScheduler:
    """Decode-first chunked prefill over a block-allocated KV pool."""

    def __init__(self, config: Optional[SchedulerConfig] = None,
                 model: Optional[ModelConfig] = None,
                 hardware: Optional[HardwareConfig] = None):
        self.config = config or SchedulerConfig()
        self.model = model or ModelConfig()
        self.hardware = hardware or HardwareConfig()
        self.planner = MemoryPlanner(self.model, self.hardware, self.config.max_tokens)
        self.allocator = BlockAllocator.from_memory(
            self.model, self.hardware, self.config.max_tokens,
            block_size=self.config.block_size,
            watermark_blocks_fraction=self.config.watermark_blocks_fraction,
            num_blocks=self.config.num_blocks,
        )

        self._queue: List[SimRequest] = []      # arrived, holding no blocks
        self._running: List[SimRequest] = []    # admitted, holding blocks, idle now
        self.num_preemptions = 0
        self.num_batches = 0

    # -- state ---------------------------------------------------------------

    @property
    def num_waiting(self) -> int:
        return len(self._queue)

    @property
    def num_running(self) -> int:
        return len(self.allocator.allocation_map)

    @property
    def pending_tokens(self) -> int:
        """Tokens sent here and not yet processed -- queued **and** running.

        Work rather than headcount, which is the distinction a load balancer
        cares about: an 8000-token prefill and a 50-token chat turn are "one
        each" to Vidur's LOR and 160x apart here.  Only meaningful between
        iterations, when `_running` holds every admitted request.
        """
        return sum(max(0, r.total_tokens - r.num_processed_tokens)
                   for r in self._queue + self._running)

    def is_idle(self) -> bool:
        return not self._queue and not self._running

    def add_request(self, request: SimRequest) -> None:
        self._queue.append(request)

    # -- allocation ----------------------------------------------------------

    def _reservation_tokens(self, request: SimRequest) -> int:
        """What a fresh admission must reserve: the whole prompt, prefix included.

        The cached prefix is not recomputed but it still occupies KV, so it is
        charged for.  This follows FSTS, which counts `initial_context + n_in`;
        a real engine sharing prefix blocks across requests would charge less,
        and neither simulator models that sharing.
        """
        return request.cached_prefix_tokens + request.num_prefill_tokens

    def _can_allocate_request(self, request: SimRequest) -> bool:
        if not self.allocator.is_allocated(request.id):
            need = self.allocator.blocks_for(self._reservation_tokens(request))
            return self.allocator.can_allocate(need, respect_watermark=True)
        # An already-running request only ever needs one more page.
        return self.allocator.free_blocks >= 1

    def _allocate_request(self, request: SimRequest) -> None:
        if not self.allocator.is_allocated(request.id):
            self.allocator.allocate(
                request.id,
                self.allocator.blocks_for(self._reservation_tokens(request)))
            return
        reserved = self.allocator.tokens_reserved(request.id)
        required = max(0, request.kv_tokens - reserved)
        if required == 0:
            return
        if required > 1:
            raise AssertionError(
                f"request {request.id} needs {required} tokens beyond its "
                f"reservation -- growth should be one token per iteration")
        self.allocator.allocate(request.id, 1)

    def _make_room(self, request: SimRequest) -> bool:
        """Preempt until `request` fits.  False if `request` was itself the victim.

        Victims are taken from the *back* of the running pool -- newest arrival
        first -- so the requests closest to finishing keep their cache.  A victim
        is restarted, not paused: its whole prompt plus everything it had already
        generated must be prefilled again.  Real GPU work, real watts, no new
        output.
        """
        while not self._can_allocate_request(request):
            if self._running:
                victim = self._running.pop(-1)
                victim.restart()
                self.allocator.free(victim.id)
                self._queue.insert(0, victim)
                self.num_preemptions += 1
            else:
                request.restart()
                self.allocator.free(request.id)
                self._queue.insert(0, request)
                self.num_preemptions += 1
                return False
        return True

    # -- the policy ----------------------------------------------------------

    def get_next_batch(self) -> Optional[Batch]:
        """One iteration's batch, or None when there is nothing runnable.

        None does not mean "finished" -- it also means "everything that could run
        is waiting for an arrival".  The engine distinguishes the two.
        """
        if not self.config.enable_chunked_prefill:
            batch = self._next_batch_unchunked()
        else:
            batch = self._next_batch_chunked()
        if batch is not None:
            self.num_batches += 1
        return batch

    def _next_batch_chunked(self) -> Optional[Batch]:
        cfg = self.config
        requests: List[SimRequest] = []
        num_tokens: List[int] = []
        skipped: List[SimRequest] = []
        running_prefills: List[SimRequest] = []
        num_batch_tokens = 0

        # --- 1. decodes, off the top ---------------------------------------
        # `_running` is drained here; `on_batch_end` refills it.  Whatever is
        # still in it during this call is a candidate victim for preemption.
        while self._running:
            if len(requests) >= cfg.max_num_seqs:
                break
            request = self._running.pop(0)

            if not request.is_prefill_complete:
                running_prefills.append(request)
                continue

            if num_batch_tokens >= cfg.chunk_size:
                skipped.append(request)
                continue

            if not self._make_room(request):
                continue                       # it was restarted and re-queued

            self._allocate_request(request)
            num_batch_tokens += 1
            requests.append(request)
            num_tokens.append(1)

        # --- 2. partially-prefilled requests take what budget is left -------
        # These already hold their blocks: the whole prompt was reserved at
        # admission, so continuing a prefill needs no new allocation.
        for request in running_prefills:
            remaining = request.num_prefill_tokens - request.num_processed_tokens
            n = max(0, min(remaining, cfg.chunk_size - num_batch_tokens))
            if n == 0:
                skipped.append(request)
                continue
            num_batch_tokens += n
            requests.append(request)
            num_tokens.append(n)

        # Skipped requests go back into the pool in FIFO order, so a request
        # squeezed out by the budget is first in line next iteration.
        self._running = sorted(skipped + self._running, key=lambda r: r.arrived_at)

        # --- 3. admit new arrivals, first come first served ------------------
        # No preemption here: a *new* request never evicts a running one.  If it
        # does not fit, it waits.
        while self._queue:
            if len(requests) >= cfg.max_num_seqs:
                break
            if len(self.allocator.allocation_map) >= cfg.max_num_seqs:
                break
            if not self._can_allocate_request(self._queue[0]):
                break

            head = self._queue[0]
            remaining = head.num_prefill_tokens - head.num_processed_tokens
            n = min(remaining, cfg.chunk_size - num_batch_tokens)
            if n <= 0:
                break

            request = self._queue.pop(0)
            self._allocate_request(request)
            num_batch_tokens += n
            requests.append(request)
            num_tokens.append(n)

        return Batch(requests, num_tokens) if requests else None

    def _next_batch_unchunked(self) -> Optional[Batch]:
        """The pre-Sarathi contrast: a prefill owns its iteration outright.

        Prefill takes priority, so a newly admitted prompt is processed whole
        while every chatter waits a turn -- the "spike, quiet, spike, quiet"
        trace that chunking flattens.  Simplified to **one prefill per
        iteration**: real vLLM V0 packs several under a padded token budget
        (`len(tokens) * max(tokens)`, counting the longest not the sum), which
        changes how many companions fit but not the alternating character this
        mode exists to show.
        """
        cfg = self.config

        # A waiting prompt that fits wins the iteration.
        if self._queue and self._can_allocate_request(self._queue[0]):
            request = self._queue.pop(0)
            self._allocate_request(request)
            n = request.num_prefill_tokens - request.num_processed_tokens
            if n > 0:
                return Batch([request], [n])
            self._queue.insert(0, request)

        # Otherwise every running request decodes.
        requests: List[SimRequest] = []
        num_tokens: List[int] = []
        skipped: List[SimRequest] = []
        while self._running:
            if len(requests) >= cfg.max_num_seqs:
                break
            request = self._running.pop(0)
            if not request.is_prefill_complete:
                # Cannot happen while prefills run whole, but a restart can
                # leave one here; give it its own iteration next time.
                skipped.append(request)
                continue
            if not self._make_room(request):
                continue
            self._allocate_request(request)
            requests.append(request)
            num_tokens.append(1)

        self._running = sorted(skipped + self._running, key=lambda r: r.arrived_at)
        return Batch(requests, num_tokens) if requests else None

    def on_batch_end(self, batch: Batch) -> None:
        """Return the batch's requests to the pool, freeing anything finished."""
        for request in batch.requests:
            if request.completed:
                self.allocator.free(request.id)
            elif not self.allocator.is_allocated(request.id):
                # Restarted mid-construction by someone else's preemption; it is
                # already back in the waiting queue, so do not resurrect it.
                continue
            else:
                self._running.append(request)
        self._running.sort(key=lambda r: r.arrived_at)

    # -- reporting -----------------------------------------------------------

    def state(self) -> Dict:
        return {
            "waiting": self.num_waiting,
            "running": self.num_running,
            "allocated_blocks": self.allocator.num_allocated_blocks,
            "kv_utilisation": self.allocator.utilisation,
            "preemptions": self.num_preemptions,
        }

    def report(self) -> Dict:
        return {
            "policy": ("decode-first chunked prefill (vLLM V1)"
                       if self.config.enable_chunked_prefill
                       else "no chunking -- prefill takes a whole iteration"),
            "chunk_size": self.config.chunk_size,
            "max_num_seqs": self.config.max_num_seqs,
            **self.allocator.report(),
            "preemptions": self.num_preemptions,
        }
