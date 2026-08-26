"""The random shape stream: reproducible, in range, and actually varied."""

import pytest

from dynshape import RandomShapeGenerator, WorkloadConfig, sweep
from dynshape.workload import quantise


def test_same_seed_same_stream():
    a = RandomShapeGenerator(seed=7).sample(50)
    b = RandomShapeGenerator(seed=7).sample(50)
    assert a == b


def test_different_seed_different_stream():
    a = RandomShapeGenerator(seed=1).sample(50)
    b = RandomShapeGenerator(seed=2).sample(50)
    assert a != b


def test_reset_replays():
    g = RandomShapeGenerator(seed=3)
    first = g.sample(20)
    assert g.sample(20) != first          # the stream advanced
    assert g.reset().sample(20) == first  # and rewinds


def test_independent_substreams_isolate_the_axes():
    """Batch, seqlen and mode draw from separate child streams, so consuming
    one does not shift the others -- the property a single global seed makes
    impossible."""
    g = RandomShapeGenerator(seed=5)
    baseline = g.reset().sample(30)

    g2 = RandomShapeGenerator(seed=5).reset()
    for _ in range(11):                    # burn batch draws only
        g2._batch()
    shifted = g2.sample(30)

    assert [r.seqlen for r in shifted] == [r.seqlen for r in baseline]
    assert [r.mode for r in shifted] == [r.mode for r in baseline]
    assert [r.batch for r in shifted] != [r.batch for r in baseline]


def test_values_stay_in_range():
    cfg = WorkloadConfig(batch_choices=(1, 4, 16), seq_min=32, seq_max=512)
    for r in RandomShapeGenerator(cfg, seed=0).sample(300):
        assert r.batch in (1, 4, 16)
        assert 32 <= r.seqlen <= 512
        assert r.mode in ("prefill", "decode")


def test_stream_is_actually_varied():
    reqs = RandomShapeGenerator(seed=0).sample(200)
    assert len({r.batch for r in reqs}) >= 4
    assert len({r.seqlen for r in reqs}) >= 50
    assert len({r.mode for r in reqs}) == 2


def test_decode_fraction_is_honoured():
    cfg_all = WorkloadConfig(decode_fraction=1.0)
    assert all(r.mode == "decode" for r in RandomShapeGenerator(cfg_all, seed=0).sample(40))
    cfg_none = WorkloadConfig(decode_fraction=0.0)
    assert all(r.mode == "prefill" for r in RandomShapeGenerator(cfg_none, seed=0).sample(40))

    cfg_half = WorkloadConfig(decode_fraction=0.5)
    reqs = RandomShapeGenerator(cfg_half, seed=0).sample(2000)
    frac = sum(r.mode == "decode" for r in reqs) / len(reqs)
    assert 0.45 < frac < 0.55


def test_quantisation_buckets():
    cfg = WorkloadConfig(seq_min=64, seq_max=4096, seq_round_to=128)
    for r in RandomShapeGenerator(cfg, seed=0).sample(100):
        assert r.seqlen % 128 == 0 and r.seqlen >= 128
    assert quantise(1, 128) == 128
    assert quantise(129, 128) == 256
    assert quantise(256, 128) == 256
    assert quantise(377, 1) == 377


def test_token_accounting():
    from dynshape import Request
    assert Request(0, 8, 512, "prefill").tokens == 8 * 512
    assert Request(0, 8, 512, "decode").tokens == 8       # one new token per sequence


def test_config_validation():
    with pytest.raises(ValueError):
        WorkloadConfig(batch_choices=())
    with pytest.raises(ValueError):
        WorkloadConfig(batch_choices=(0, 4))
    with pytest.raises(ValueError):
        WorkloadConfig(seq_min=512, seq_max=64)
    with pytest.raises(ValueError):
        WorkloadConfig(decode_fraction=1.5)
    with pytest.raises(ValueError):
        WorkloadConfig(batch_choices=(1, 2), batch_weights=(0.5,))
    with pytest.raises(ValueError):
        WorkloadConfig(batch_choices=(1, 2), batch_weights=(0.5, 0.9))


def test_sweep_is_a_complete_grid():
    reqs = sweep([1, 8], [128, 512, 1024])
    assert len(reqs) == 2 * 3 * 2
    assert len({r.idx for r in reqs}) == len(reqs)
    assert {(r.batch, r.seqlen, r.mode) for r in reqs} == {
        (b, s, m) for m in ("prefill", "decode") for b in (1, 8) for s in (128, 512, 1024)
    }


def test_sample_rejects_zero():
    with pytest.raises(ValueError):
        RandomShapeGenerator(seed=0).sample(0)
