"""
kvcache.py
==========

**Where the KV pool comes from, and how it is spent.**

Both halves are Vidur's, and the choice is deliberate.  FSTS sizes the pool
better (automatically, from hardware minus weights) but spends it with a single
running sum and never preempts.  Vidur spends it with a real block allocator and
*does* preempt -- and preemption is a genuine high-power event that nothing in
the arrival stream predicts.  So: **Vidur's allocator, with FSTS's sizing
available as an alternative**, since the two are algebraically identical anyway.

THEY REALLY ARE IDENTICAL
-------------------------
It is tempting to call block accounting and a token counter merely "close".
They are the same number:

    vidur_tokens = num_blocks * block_size
                 = (max_tokens / block_size) * max_batch * block_size
                 = max_tokens * free / (kv_per_token * max_tokens)
                 = free / kv_per_token
                 = fsts_capacity

Vidur simply routes through a per-request worst case and multiplies it back
out.  Integer flooring is the only source of divergence.  **The pool is the same
size in both; only the policy for spending it differs.**

WHY THE ALLOCATOR SETS THE WATTAGE
----------------------------------
Blocks -> concurrency -> batch size -> the M dimension of every GEMM.  Better
accounting fits several times more requests in the same pool, and batch size is
the M dimension of every GEMM in the model.  The memory allocator quietly sets
the wattage of every iteration.

THE COUPLING WORTH KNOWING ABOUT
--------------------------------
Vidur sizes the pool from `max_tokens` -- the longest request the *workload
config* permits.  Change the workload's cap and the memory pool changes with it,
which is a surprising dependency between two layers that ought to be
independent.  Ported as-is, and flagged here rather than silently fixed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from typing import Dict, List, Optional

GB = 1024 ** 3


@dataclass(frozen=True)
class ModelConfig:
    """Just enough of a model to count parameters and KV bytes.

    Defaults are GPT-2 124M, which is what the shipped templates were traced
    from -- the model is fixed in v1, so these numbers and the kernel shapes
    must describe the same network.
    """

    name: str = "gpt2"
    num_layers: int = 12
    num_heads: int = 12
    num_kv_heads: int = 12          # GPT-2 is MHA: kv_heads == heads
    head_dim: int = 64
    hidden: int = 768
    ffn_hidden: int = 3072
    vocab: int = 50257
    max_position: int = 1024
    param_bytes: int = 2            # bf16 weights
    kv_bytes: int = 2               # bf16 cache

    def num_parameters(self) -> int:
        """GPT-2's parameter count, written out rather than approximated."""
        h = self.hidden
        embed = self.vocab * h + self.max_position * h
        attn = (h * 3 * h + 3 * h) + (h * h + h)          # qkv + out projection
        mlp = (h * self.ffn_hidden + self.ffn_hidden) + (self.ffn_hidden * h + h)
        norms = 2 * (2 * h)                                # two LNs, weight+bias
        return embed + self.num_layers * (attn + mlp + norms) + 2 * h

    def weight_bytes(self) -> int:
        return self.num_parameters() * self.param_bytes

    def kv_bytes_per_token(self) -> int:
        """2 (K and V) x bytes x head_dim x kv_heads x layers."""
        return (2 * self.kv_bytes * self.head_dim
                * self.num_kv_heads * self.num_layers)


@dataclass(frozen=True)
class HardwareConfig:
    """A100-40GB-PCIe -- EnergAIzer's `yz8`, the GPU the LUT was measured on."""

    name: str = "a100-40gb-pcie"
    memory_gb: float = 40.0
    memory_margin_fraction: float = 0.1      # Vidur's default headroom
    tensor_parallel: int = 1


class MemoryPlanner:
    """Vidur's `scheduler/utils/memory_planner.py`.

    Sizes the pool once, at startup, from a **per-request worst case**: how many
    requests of `max_request_tokens` fit in memory after the weights.  That
    worst case is why the coupling to the workload's `max_tokens` exists.
    """

    def __init__(self, model: ModelConfig, hardware: HardwareConfig,
                 max_request_tokens: int):
        if max_request_tokens < 1:
            raise ValueError("max_request_tokens must be >= 1")
        self.model = model
        self.hardware = hardware
        self.max_request_tokens = int(max_request_tokens)

    def available_bytes(self) -> float:
        return (self.hardware.memory_gb * GB
                * (1 - self.hardware.memory_margin_fraction))

    def parameter_bytes_per_device(self) -> float:
        return self.model.weight_bytes() / self.hardware.tensor_parallel

    def kv_bytes_per_request(self) -> float:
        """A full-length request's KV, on one device."""
        return (self.model.kv_bytes_per_token() * self.max_request_tokens
                / self.hardware.tensor_parallel)

    def max_batch_size(self) -> int:
        free = self.available_bytes() - self.parameter_bytes_per_device()
        if free <= 0:
            raise ValueError(
                f"weights ({self.model.weight_bytes() / GB:.2f} GB) exceed the "
                f"memory budget ({self.available_bytes() / GB:.2f} GB)")
        n = int(free // self.kv_bytes_per_request())
        if n < 1:
            raise ValueError(
                "not enough memory for even one request of "
                f"{self.max_request_tokens} tokens")
        return n

    def kv_capacity_tokens(self) -> float:
        """FSTS's `derive_kv_capacity_tokens()` -- the same pool, stated in tokens.

        Free memory divided by bytes per token, with no per-request worst case
        in the way.  Compare against `BlockAllocator.capacity_tokens` to see the
        integer-flooring gap, which is the only place the two formulations
        disagree.
        """
        free = self.available_bytes() - self.parameter_bytes_per_device()
        return free / (self.model.kv_bytes_per_token() / self.hardware.tensor_parallel)

    def report(self) -> Dict:
        return {
            "model": self.model.name,
            "parameters_m": self.model.num_parameters() / 1e6,
            "weight_bytes_gb": self.model.weight_bytes() / GB,
            "kv_bytes_per_token": self.model.kv_bytes_per_token(),
            "available_gb": self.available_bytes() / GB,
            "max_request_tokens": self.max_request_tokens,
            "max_batch_size": self.max_batch_size(),
            "kv_capacity_tokens_fsts": self.kv_capacity_tokens(),
        }


class BlockAllocator:
    """Vidur's block accounting: `can_allocate` / `allocate` / `free`.

    Pages, not a reservation.  "Here is page 7.  Full?  Page 412."  Pages need
    not be adjacent, so you keep an index of who owns which -- and that index is
    the whole of block accounting: three operations on a counter and a dict.

    The alternative, reserving each request's worst case upfront, is what
    `faster_transformer` does, and it wastes most of the pool: a user who *might*
    write 4096 tokens but writes 300 leaves 237 pages blank for the whole
    conversation.

    `watermark_blocks_fraction` holds a sliver in reserve so that admitting a new
    prompt cannot immediately starve the requests already decoding.
    """

    def __init__(self, num_blocks: int, block_size: int = 16,
                 watermark_blocks_fraction: float = 0.01):
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        if not (0.0 <= watermark_blocks_fraction < 1.0):
            raise ValueError("watermark_blocks_fraction must be in [0, 1)")
        self.num_blocks = int(num_blocks)
        self.block_size = int(block_size)
        self.watermark_blocks = int(watermark_blocks_fraction * num_blocks)
        self.num_allocated_blocks = 0
        self.allocation_map: Dict[int, int] = {}

    # -- sizing --------------------------------------------------------------

    @classmethod
    def from_memory(cls, model: ModelConfig, hardware: HardwareConfig,
                    max_request_tokens: int, block_size: int = 16,
                    watermark_blocks_fraction: float = 0.01,
                    num_blocks: Optional[int] = None) -> "BlockAllocator":
        """Vidur's sizing, or an explicit override.

        An explicit `num_blocks` is not a hack -- it is the only way to see
        preemption on a small model.  GPT-2's KV is so cheap that an A100 holds
        roughly a million tokens, so the pool never fills and the recompute spike
        never fires.  Shrinking the pool deliberately is how you study the
        behaviour that a 70B model would hit naturally.
        """
        if num_blocks is not None:
            return cls(num_blocks, block_size, watermark_blocks_fraction)
        planner = MemoryPlanner(model, hardware, max_request_tokens)
        max_blocks_per_sequence = max(1, max_request_tokens // block_size)
        derived = max_blocks_per_sequence * planner.max_batch_size()
        return cls(derived, block_size, watermark_blocks_fraction)

    @property
    def capacity_tokens(self) -> int:
        return self.num_blocks * self.block_size

    @property
    def free_blocks(self) -> int:
        return self.num_blocks - self.num_allocated_blocks

    @property
    def utilisation(self) -> float:
        return self.num_allocated_blocks / self.num_blocks

    # -- spending ------------------------------------------------------------

    def blocks_for(self, num_tokens: int) -> int:
        return ceil(num_tokens / self.block_size)

    def is_allocated(self, request_id: int) -> bool:
        return request_id in self.allocation_map

    def can_allocate(self, num_blocks: int, respect_watermark: bool = False) -> bool:
        need = num_blocks + (self.watermark_blocks if respect_watermark else 0)
        return self.free_blocks >= need

    def allocate(self, request_id: int, num_blocks: int) -> None:
        if num_blocks < 0:
            raise ValueError("num_blocks must be >= 0")
        if num_blocks > self.free_blocks:
            raise AssertionError(
                f"over-allocation: {num_blocks} blocks requested, "
                f"{self.free_blocks} free -- can_allocate was not checked")
        self.num_allocated_blocks += num_blocks
        self.allocation_map[request_id] = self.allocation_map.get(request_id, 0) + num_blocks

    def free(self, *request_ids: int) -> None:
        for rid in request_ids:
            if rid not in self.allocation_map:
                raise KeyError(f"request {rid} holds no blocks")
            self.num_allocated_blocks -= self.allocation_map.pop(rid)
        if self.num_allocated_blocks < 0:
            raise AssertionError("freed more blocks than were allocated")

    def tokens_reserved(self, request_id: int) -> int:
        return self.allocation_map.get(request_id, 0) * self.block_size

    def report(self) -> Dict:
        return {
            "num_blocks": self.num_blocks,
            "block_size": self.block_size,
            "capacity_tokens": self.capacity_tokens,
            "watermark_blocks": self.watermark_blocks,
            "allocated_blocks": self.num_allocated_blocks,
            "utilisation": self.utilisation,
            "requests_holding_blocks": len(self.allocation_map),
        }
