"""The request lifecycle -- the part that is easy to get wrong and silently
poisons every downstream shape."""

import pytest

from dynshape.entities import Batch, Piece, SimRequest, reset_ids


def req(p=100, d=10, arrived=0.0, cached=0):
    return SimRequest(arrived_at=arrived, num_prefill_tokens=p,
                      num_decode_tokens=d, cached_prefix_tokens=cached)


# -- prefill -> decode transition ------------------------------------------

def test_prefill_completion_emits_the_first_token():
    """Vidur's rule, and it is not obvious: the pass that finishes the prompt
    also produces token one, so num_processed_tokens jumps by n+1."""
    r = req(p=100, d=10)
    r.on_batch_end(1.0, 100)
    assert r.is_prefill_complete
    assert r.num_processed_tokens == 101
    assert r.token_times == [1.0]
    assert r.num_generated_tokens == 1


def test_chunked_prefill_does_not_complete_early():
    r = req(p=100, d=10)
    r.on_batch_end(1.0, 40)
    assert not r.is_prefill_complete
    assert r.num_processed_tokens == 40
    assert r.token_times == []
    r.on_batch_end(2.0, 60)
    assert r.is_prefill_complete
    assert r.num_processed_tokens == 101


def test_decode_steps_grow_the_context_by_one():
    r = req(p=100, d=3)
    r.on_batch_end(1.0, 100)          # prefill + token 1
    r.on_batch_end(2.0, 1)            # token 2
    assert r.num_processed_tokens == 102
    r.on_batch_end(3.0, 1)            # token 3 -- done
    assert r.completed
    assert r.completed_at == 3.0
    assert len(r.token_times) == 3


def test_over_issuing_work_is_a_loud_failure():
    r = req(p=10, d=1)
    r.on_batch_end(1.0, 10)           # completes: 10 prefill + 1 decode
    assert r.completed
    with pytest.raises(AssertionError):
        r.on_batch_end(2.0, 1)


# -- the cached prefix ------------------------------------------------------

def test_cached_prefix_shrinks_executed_work_but_not_the_context():
    """The prompt still occupies KV; only the compute is skipped."""
    r = req(p=1000, d=5, cached=900)
    assert r.num_prefill_tokens == 100          # executed
    assert r.kv_tokens == 900                   # occupied, before any work
    r.on_batch_end(1.0, 100)
    assert r.is_prefill_complete
    assert r.context_tokens == 900 + 101


def test_a_fully_cached_prompt_is_rejected():
    with pytest.raises(ValueError):
        req(p=100, cached=100)


# -- restart ----------------------------------------------------------------

def test_restart_converts_generated_tokens_back_into_prompt_work():
    """The power event: everything generated is thrown away and re-prefilled.
    Real watts, no new output, and nothing in the arrival stream predicts it."""
    r = req(p=100, d=50)
    r.on_batch_end(1.0, 100)
    for t in range(2, 12):
        r.on_batch_end(float(t), 1)
    assert r.num_processed_tokens == 111

    r.restart()
    assert r.num_prefill_tokens == 111          # all of it is prompt now
    assert r.num_decode_tokens == 150 - 111
    assert r.num_processed_tokens == 0
    assert not r.is_prefill_complete
    assert r.num_restarts == 1
    assert r.restart_work_tokens == 111
    assert r.total_tokens == 150                # conserved


def test_restart_preserves_the_original_scheduling_time():
    r = req(p=10, d=5, arrived=1.0)
    r.on_batch_schedule(2.0)
    assert r.scheduled_at == 2.0
    r.on_batch_end(3.0, 10)
    r.restart()
    r.on_batch_schedule(9.0)
    assert r.scheduled_at == 2.0                # the first admission, not the retry


# -- pieces -----------------------------------------------------------------

def test_decode_piece_attends_over_context_plus_one():
    """The measured `+1`: the new token's own K/V are appended before attention
    runs, so it reads one more key than the cache holds."""
    p = Piece(kind="decode", request_id=0, tokens=1, context=128)
    assert p.query_len == 1
    assert p.key_len == 129


def test_prefill_chunk_attends_to_prior_context_and_itself():
    p = Piece(kind="prefill", request_id=0, tokens=512, context=1536)
    assert p.query_len == 512
    assert p.key_len == 2048


# -- batch ------------------------------------------------------------------

def test_batch_snapshots_the_phase_before_it_changes():
    """on_batch_end flips is_prefill_complete, so a batch that asked afterwards
    would mislabel the iteration it just ran."""
    r = req(p=10, d=5)
    b = Batch([r], [10])
    assert b.pieces[0].kind == "prefill"
    b.on_batch_end(1.0)
    assert r.is_prefill_complete
    assert b.pieces[0].kind == "prefill"        # still describes what ran


def test_mixed_batch_reports_both_phases():
    a, c = req(p=20, d=5), req(p=30, d=5)
    a.on_batch_end(1.0, 20)                     # a is now decoding
    b = Batch([a, c], [1, 30])
    assert b.is_mixed
    assert b.decode_batch == 1 and b.prefill_batch == 1
    assert b.num_decode_tokens == 1
    assert b.num_prefill_tokens == 30
    assert b.total_num_tokens == 31


def test_empty_or_zero_token_batches_are_rejected():
    with pytest.raises(ValueError):
        Batch([], [])
    with pytest.raises(ValueError):
        Batch([req()], [0])
    with pytest.raises(ValueError):
        Batch([req()], [1, 2])


def test_reset_ids_makes_runs_comparable():
    reset_ids()
    assert req().id == 0
    reset_ids()
    assert req().id == 0
