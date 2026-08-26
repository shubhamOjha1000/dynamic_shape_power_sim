"""L1: decode-first chunked prefill over a block-allocated KV pool.

The behaviours worth pinning are the ones that change the power trace: that
decodes are protected, that prefill is sliced rather than serialised, that
batches really are mixed, and that KV exhaustion produces a restart.
"""

import pytest

from dynshape.entities import SimRequest, reset_ids
from dynshape.kvcache import HardwareConfig, ModelConfig
from dynshape.scheduler import ChunkedPrefillScheduler, SchedulerConfig


def sched(**kw):
    reset_ids()
    cfg = SchedulerConfig(**{"chunk_size": 512, "max_num_seqs": 8,
                             "block_size": 16, "max_tokens": 4096, **kw})
    return ChunkedPrefillScheduler(cfg, ModelConfig(), HardwareConfig())


def req(p, d=8, t=0.0):
    return SimRequest(arrived_at=t, num_prefill_tokens=p, num_decode_tokens=d)


def run(s, n_iters):
    """Drive n iterations, returning the batches actually produced."""
    out = []
    for _ in range(n_iters):
        b = s.get_next_batch()
        if b is None:
            break
        b.on_schedule(0.0)
        b.on_batch_end(0.0)
        s.on_batch_end(b)
        out.append(b)
    return out


# -- chunking ---------------------------------------------------------------

def test_a_long_prompt_is_sliced_to_the_token_budget():
    s = sched(chunk_size=512)
    s.add_request(req(2000))
    b = s.get_next_batch()
    assert b.total_num_tokens == 512
    assert b.num_prefill_tokens == 512


def test_a_short_prompt_is_not_padded_up_to_the_budget():
    s = sched(chunk_size=512)
    s.add_request(req(100))
    assert s.get_next_batch().total_num_tokens == 100


def test_a_prompt_finishes_across_the_expected_number_of_iterations():
    s = sched(chunk_size=512)
    s.add_request(req(2000, d=1))
    batches = run(s, 10)
    prefill_iters = [b for b in batches if b.num_prefill_tokens]
    assert len(prefill_iters) == 4                    # 512+512+512+464
    assert sum(b.num_prefill_tokens for b in prefill_iters) == 2000


# -- decode-first, and the mixing it produces -------------------------------

def test_decodes_are_taken_off_the_top_before_prefill():
    """The protected class.  Everyone mid-conversation gets their next word,
    without exception; prefill takes whatever is left."""
    s = sched(chunk_size=512)
    for _ in range(3):
        s.add_request(req(50, d=20))
    run(s, 3)                                          # get all three decoding
    s.add_request(req(2000, d=5, t=1.0))               # a big newcomer
    b = s.get_next_batch()
    assert b.decode_batch == 3
    assert b.num_decode_tokens == 3
    assert b.num_prefill_tokens == 512 - 3             # exactly the remainder


def test_batches_really_are_mixed():
    """The whole reason this layer exists: prefill and decode in the same
    forward pass, not alternating between passes."""
    s = sched(chunk_size=512)
    s.add_request(req(100, d=50))
    run(s, 2)
    s.add_request(req(1500, d=5, t=0.5))
    b = s.get_next_batch()
    assert b.is_mixed
    assert b.decode_batch >= 1 and b.prefill_batch >= 1


def test_the_busier_the_system_the_less_budget_survives_for_prefill():
    """Decodes come off the top, so a newcomer's first chunk shrinks as the
    number of chatters grows -- new users are sacrificed to protect existing
    ones."""
    def first_chunk(n_chatters):
        s = sched(chunk_size=512, max_num_seqs=256)
        for _ in range(n_chatters):
            s.add_request(req(20, d=100))
        run(s, n_chatters + 2)
        s.add_request(req(2000, d=5, t=9.0))
        return s.get_next_batch().num_prefill_tokens

    assert first_chunk(2) > first_chunk(40) > first_chunk(200)


def test_seat_cap_is_what_stops_prefill_being_starved_entirely():
    with pytest.raises(ValueError, match="starve"):
        SchedulerConfig(chunk_size=256, max_num_seqs=512)


# -- KV accounting ----------------------------------------------------------

def test_admission_reserves_the_whole_prompt_upfront():
    s = sched(block_size=16)
    r = req(300)
    s.add_request(r)
    s.get_next_batch()
    assert s.allocator.allocation_map[r.id] == 19        # ceil(300/16)


def test_a_decoding_request_grows_by_one_block_at_a_time():
    s = sched(chunk_size=512, block_size=16)
    r = req(32, d=60)                                    # 32 = exactly 2 blocks
    s.add_request(r)
    run(s, 1)
    before = s.allocator.allocation_map[r.id]
    run(s, 20)
    after = s.allocator.allocation_map[r.id]
    assert after > before
    assert after - before <= 2                           # 20 tokens ~ 2 pages


def test_blocks_are_freed_when_a_request_completes():
    s = sched()
    r = req(50, d=3)
    s.add_request(r)
    run(s, 10)
    assert r.completed
    assert r.id not in s.allocator.allocation_map
    assert s.allocator.num_allocated_blocks == 0


def test_a_request_that_does_not_fit_waits_rather_than_evicting():
    """A NEW request never preempts a running one; it queues."""
    s = sched(num_blocks=40, block_size=16, max_num_seqs=8)   # 640 tokens
    a = req(500, d=20)
    s.add_request(a)
    run(s, 2)
    b = req(500, d=20, t=1.0)
    s.add_request(b)
    batch = s.get_next_batch()
    assert b.id not in s.allocator.allocation_map
    assert s.num_waiting == 1
    assert batch is not None and a.id in batch.request_ids


# -- preemption, the power event --------------------------------------------

def test_kv_exhaustion_restarts_a_victim_and_wastes_real_work():
    """A recompute spike: blocks are freed, the victim's generated tokens are
    thrown away, and its whole prompt must be prefilled again.  Nothing in the
    arrival pattern predicts it."""
    s = sched(num_blocks=48, block_size=16, max_num_seqs=16, chunk_size=512)
    rs = [req(200, d=200, t=0.0) for _ in range(3)]
    for r in rs:
        s.add_request(r)
    run(s, 400)
    assert s.num_preemptions > 0
    assert sum(r.num_restarts for r in rs) > 0
    assert sum(r.restart_work_tokens for r in rs) > 0


def test_the_victim_is_the_newest_arrival_not_the_oldest():
    """Requests closest to finishing keep their cache; the newcomer pays."""
    s = sched(num_blocks=48, block_size=16, max_num_seqs=16, chunk_size=512)
    old = req(200, d=300, t=0.0)
    new = req(200, d=300, t=5.0)
    s.add_request(old)
    s.add_request(new)
    run(s, 300)
    assert new.num_restarts >= old.num_restarts


def test_total_token_budget_is_conserved_through_a_restart():
    s = sched(num_blocks=48, block_size=16, chunk_size=512)
    rs = [req(150, d=150) for _ in range(3)]
    for r in rs:
        s.add_request(r)
    run(s, 400)
    for r in rs:
        assert r.total_tokens == 300


# -- the unchunked contrast -------------------------------------------------

def test_without_chunking_a_prefill_owns_its_iteration():
    """The 'spike, quiet, spike, quiet' trace that chunking flattens."""
    s = sched(chunk_size=512, enable_chunked_prefill=False)
    s.add_request(req(50, d=20))
    run(s, 2)
    s.add_request(req(2000, d=5, t=1.0))
    b = s.get_next_batch()
    assert b.decode_batch == 0
    assert b.num_prefill_tokens == 2000            # whole prompt, one pass
    assert not b.is_mixed


def test_without_chunking_decodes_still_run_when_no_prompt_waits():
    s = sched(enable_chunked_prefill=False)
    s.add_request(req(50, d=20))
    batches = run(s, 5)
    assert any(b.decode_batch > 0 for b in batches)


# -- housekeeping -----------------------------------------------------------

def test_idle_scheduler_returns_no_batch():
    assert sched().get_next_batch() is None


def test_everything_eventually_completes():
    s = sched(chunk_size=512, max_num_seqs=8)
    rs = [req(120 + 40 * i, d=6, t=0.0) for i in range(6)]
    for r in rs:
        s.add_request(r)
    run(s, 2000)
    assert all(r.completed for r in rs)
    assert s.allocator.num_allocated_blocks == 0
    assert s.is_idle()


def test_report_names_the_policy():
    assert "chunked prefill" in sched().report()["policy"]
    assert "whole iteration" in sched(enable_chunked_prefill=False).report()["policy"]
