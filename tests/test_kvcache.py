"""KV sizing and block accounting."""

import pytest

from dynshape.kvcache import (GB, BlockAllocator, HardwareConfig, MemoryPlanner,
                              ModelConfig)


def test_gpt2_parameter_count_is_right():
    """124M is the published figure; if this drifts, the weight footprint and
    therefore the KV pool size drift with it."""
    n = ModelConfig().num_parameters()
    assert 120e6 < n < 165e6


def test_kv_bytes_per_token_matches_the_formula():
    m = ModelConfig()
    # 2 (K and V) x 2 bytes x 64 head_dim x 12 kv_heads x 12 layers
    assert m.kv_bytes_per_token() == 2 * 2 * 64 * 12 * 12 == 36864


def test_gqa_model_needs_less_cache():
    mha = ModelConfig(num_kv_heads=12)
    gqa = ModelConfig(num_kv_heads=2)
    assert gqa.kv_bytes_per_token() * 6 == mha.kv_bytes_per_token()


# -- the finding: the two sizing formulas agree -----------------------------

def test_block_pool_and_token_capacity_agree_up_to_integer_flooring():
    """Vidur's blocks and FSTS's token counter are algebraically identical --
    Vidur just routes through a per-request worst case and multiplies it back
    out.  The pool is the same size in both; only the spending policy differs.
    """
    model, hw = ModelConfig(), HardwareConfig()
    max_tokens = 4096
    planner = MemoryPlanner(model, hw, max_tokens)
    alloc = BlockAllocator.from_memory(model, hw, max_tokens, block_size=16)

    fsts = planner.kv_capacity_tokens()
    vidur = alloc.capacity_tokens
    # Flooring twice (blocks per sequence, then requests) can only lose, never
    # gain, and it loses less than one full request's worth.
    assert vidur <= fsts
    assert vidur > fsts - max_tokens


def test_planner_refuses_a_model_that_does_not_fit():
    huge = ModelConfig(num_layers=200, hidden=16384, ffn_hidden=65536, vocab=200000)
    with pytest.raises(ValueError):
        MemoryPlanner(huge, HardwareConfig(memory_gb=8.0), 4096).max_batch_size()


def test_pool_size_is_coupled_to_the_workload_cap():
    """The footgun worth knowing about: change `max_tokens` in the WORKLOAD and
    the MEMORY pool changes with it, across two layers that ought to be
    independent."""
    model, hw = ModelConfig(), HardwareConfig()
    small = BlockAllocator.from_memory(model, hw, 1024).num_blocks
    large = BlockAllocator.from_memory(model, hw, 8192).num_blocks
    assert small != large


# -- spending ---------------------------------------------------------------

def test_allocate_and_free_are_symmetric():
    a = BlockAllocator(num_blocks=100, block_size=16)
    a.allocate(1, 10)
    a.allocate(2, 20)
    assert a.num_allocated_blocks == 30
    assert a.free_blocks == 70
    a.free(1, 2)
    assert a.num_allocated_blocks == 0
    assert a.allocation_map == {}


def test_growing_a_request_adds_to_its_entry():
    a = BlockAllocator(num_blocks=100, block_size=16)
    a.allocate(1, 4)
    a.allocate(1, 1)
    assert a.allocation_map[1] == 5
    assert a.tokens_reserved(1) == 80


def test_over_allocation_is_a_loud_failure_not_a_silent_overflow():
    a = BlockAllocator(num_blocks=10, block_size=16)
    with pytest.raises(AssertionError):
        a.allocate(1, 11)


def test_freeing_an_unknown_request_raises():
    a = BlockAllocator(num_blocks=10)
    with pytest.raises(KeyError):
        a.free(99)


def test_watermark_holds_a_reserve_against_new_admissions():
    """Admitting a new prompt must not be able to starve requests already
    decoding, so a slice of the pool is withheld from newcomers only."""
    a = BlockAllocator(num_blocks=100, block_size=16, watermark_blocks_fraction=0.1)
    assert a.watermark_blocks == 10
    a.allocate(1, 85)
    assert not a.can_allocate(10, respect_watermark=True)    # newcomer: blocked
    assert a.can_allocate(10, respect_watermark=False)       # incumbent: allowed


def test_blocks_for_rounds_up():
    a = BlockAllocator(num_blocks=100, block_size=16)
    assert a.blocks_for(1) == 1
    assert a.blocks_for(16) == 1
    assert a.blocks_for(17) == 2


def test_paging_wastes_less_than_worst_case_reservation():
    """Reserving `max_blocks_per_sequence` per request -- what faster_transformer
    does -- leaves most of the pool blank.  Paging wastes at most one block."""
    a = BlockAllocator(num_blocks=1000, block_size=16)
    prompt = 300
    paged = a.blocks_for(prompt)
    reserved = 4096 // 16
    assert paged * 16 - prompt < 16
    assert paged < reserved / 8


def test_explicit_num_blocks_overrides_the_derived_pool():
    """The only way to see preemption on a small model: GPT-2's KV is so cheap
    that an A100 holds roughly a million tokens and the pool never fills."""
    model, hw = ModelConfig(), HardwareConfig()
    derived = BlockAllocator.from_memory(model, hw, 4096)
    assert derived.capacity_tokens > 500_000
    tiny = BlockAllocator.from_memory(model, hw, 4096, num_blocks=200)
    assert tiny.num_blocks == 200


def test_invalid_configs_are_rejected():
    with pytest.raises(ValueError):
        BlockAllocator(num_blocks=0)
    with pytest.raises(ValueError):
        BlockAllocator(num_blocks=10, watermark_blocks_fraction=1.0)
