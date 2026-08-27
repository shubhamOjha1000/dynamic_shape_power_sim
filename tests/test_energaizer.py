"""The measured-LUT path.

Most of this runs without the artifact or the database, because the parts worth
pinning are the *diagnostics* -- what happens when a piece is missing. The tests
that need the real estimator skip cleanly, and say why.
"""

import os

import pytest

from dynshape.energaizer import (LUT_FREQ_MHZ, LUT_SENTINEL, ArtifactPaths,
                                 find_artifact_root,
                                 find_lut_dir, locate_artifact, lut_status,
                                 required_luts)
from dynshape.predictor import AnalyticBackend, GeeBackend, _scalar, build_predictor

#: Set DYNSHAPE_ENERGAIZER=/path/to/artifact to run the measured-model tests.
ARTIFACT = os.environ.get("DYNSHAPE_ENERGAIZER")
needs_artifact = pytest.mark.skipif(
    not ARTIFACT, reason="set DYNSHAPE_ENERGAIZER to the artifact checkout")


# -- unwrapping whatever lookup returns --------------------------------------

def test_scalar_unwraps_every_shape_lookup_can_return():
    """Not defensive: EnergAIzer converts energy and time from a pandas Series
    but leaves power alone, so the third value's type depends on which estimator
    branch ran."""
    import numpy as np
    pd = pytest.importorskip("pandas")

    assert _scalar(3.5) == 3.5
    assert _scalar(np.float64(3.5)) == 3.5
    assert _scalar(np.array([3.5])) == 3.5
    assert _scalar(pd.Series([3.5])) == 3.5
    assert _scalar([3.5]) == 3.5


# -- the reconciliation policy -----------------------------------------------

class FakeEstimator:
    """Returns a deliberately inconsistent (t, p, e): E != P x t."""

    def __init__(self, t_ms=2.0, power_w=300.0, energy_j=0.5):
        self.t, self.p, self.e = t_ms, power_w, energy_j

    def lookup(self, query, query_type, target_freq=None, lookup_target="all",
               **kw):
        return (self.t, self.p, self.e)


def test_report_mode_returns_the_measured_numbers_untouched():
    """Silently rewriting a measured number is worse than carrying a small
    inconsistency you can see."""
    b = GeeBackend(FakeEstimator(), reconcile="report")
    t, p, e = b.predict({}, ("gemm",), 900)
    assert (t, p, e) == (2.0, 300.0, 0.5)
    # P x t = 0.6 J against a reported 0.5 J -> 20% apart, and recorded.
    assert b.stats()["max_energy_vs_power_x_time"] == pytest.approx(0.2)


def test_energy_mode_makes_power_agree_with_every_aggregate():
    b = GeeBackend(FakeEstimator(), reconcile="energy")
    t, p, e = b.predict({}, ("gemm",), 900)
    assert e == 0.5
    assert p == pytest.approx(0.5 / (2.0 / 1000.0))
    assert p * t / 1000.0 == pytest.approx(e)


def test_power_mode_derives_energy_instead():
    b = GeeBackend(FakeEstimator(), reconcile="power")
    t, p, e = b.predict({}, ("gemm",), 900)
    assert p == 300.0
    assert e == pytest.approx(300.0 * 2.0 / 1000.0)


def test_a_consistent_estimator_reports_no_inconsistency():
    b = GeeBackend(FakeEstimator(t_ms=2.0, power_w=300.0, energy_j=0.6))
    b.predict({}, ("gemm",), 900)
    assert b.stats()["max_energy_vs_power_x_time"] == pytest.approx(0.0, abs=1e-12)


def test_an_unknown_reconcile_policy_is_rejected():
    with pytest.raises(ValueError):
        GeeBackend(FakeEstimator(), reconcile="whatever")


def test_the_backend_declares_itself_measured():
    b = GeeBackend(FakeEstimator())
    assert b.is_measured_model is True
    assert "SYNTHETIC" not in b.name
    assert AnalyticBackend().is_measured_model is False


def test_stats_flow_through_the_cache():
    """The reconciliation gap has to reach the trace summary, or nobody sees it."""
    from dynshape.predictor import CachedPredictor
    p = CachedPredictor(backend=GeeBackend(FakeEstimator()), freq=900)
    p.predict({"dimM": 1}, ("gemm",))
    s = p.stats()
    assert s["is_measured_model"] is True
    assert "max_energy_vs_power_x_time" in s
    assert s["freq_mhz"] == 900


# -- failing loudly when asked to ---------------------------------------------

def test_require_measured_refuses_to_degrade_silently():
    """Silent fallback is right for a demo and wrong when the whole point of the
    run is the measured model."""
    with pytest.raises(RuntimeError, match="require_measured"):
        build_predictor(require_measured=True)


def test_require_measured_and_force_analytic_are_contradictory():
    with pytest.raises(ValueError):
        build_predictor(force_analytic=True, require_measured=True)


def test_without_require_measured_it_degrades_and_says_so(capsys):
    p = build_predictor(pkg_path="/nonexistent", lut_dir="/nonexistent")
    assert p.backend.is_measured_model is False
    assert "SYNTHETIC" in capsys.readouterr().out


def test_locate_artifact_names_the_missing_file(tmp_path):
    (tmp_path / "gee").mkdir()
    with pytest.raises(FileNotFoundError, match="GPU config"):
        locate_artifact(str(tmp_path))


def test_locate_artifact_rejects_a_path_that_is_not_there():
    with pytest.raises(FileNotFoundError, match="does not exist"):
        locate_artifact("/definitely/not/here")


def test_find_lut_dir_searches_rather_than_assuming(tmp_path):
    """The tarball's internal layout has changed between releases, and a
    hard-coded path fails with 'no LUT' when the files are one level deeper."""
    deep = tmp_path / "database" / "data" / "precollected" / "yz8"
    deep.mkdir(parents=True)
    (deep / LUT_SENTINEL).write_text("x")
    assert find_lut_dir(str(tmp_path)) == str(deep)


def test_find_lut_dir_returns_none_when_absent(tmp_path):
    assert find_lut_dir(str(tmp_path)) is None


def _fake_lut_yaml(tmp_path):
    """A LUT config shaped like the artifact's: a nested tree of `path:` leaves."""
    y = tmp_path / "lut.yaml"
    y.write_text(
        "lut_config:\n"
        "  softmax:\n"
        "    bf16:\n"
        "      - path: 'sm_bf16.csv'\n"
        "  gemm:\n"
        "    tc:\n"
        "      bf16_bf16:\n"
        "        - path: 'gemm_bf16.csv'\n"
        "  elementwise:\n"
        "    - path: 'ew.csv'\n")
    return str(y)


def test_required_luts_flattens_the_nested_config(tmp_path):
    """The config nests op / kernel-class / precision to arbitrary depth, so the
    filenames have to be walked out rather than indexed."""
    assert required_luts(_fake_lut_yaml(tmp_path)) == [
        "ew.csv", "gemm_bf16.csv", "sm_bf16.csv"]


def test_lut_status_reports_exactly_which_tables_are_missing(tmp_path):
    """A partially extracted database builds an estimator that raises on some op
    types, and skipping those kernels hands back a trace that is simply cheaper
    than reality -- so 'incomplete' has to be distinguishable from 'absent'."""
    lut_dir = tmp_path / "data"
    lut_dir.mkdir()
    (lut_dir / "ew.csv").write_text("x" * 1024)

    paths = ArtifactPaths(root=str(tmp_path), gpu_yaml="g",
                          lut_yaml=_fake_lut_yaml(tmp_path), lut_dir=str(lut_dir))
    st = lut_status(paths)
    assert st["complete"] is False
    assert st["present"] == ["ew.csv"]
    assert set(st["missing"]) == {"gemm_bf16.csv", "sm_bf16.csv"}
    assert st["total_mb"] > 0


def test_lut_status_with_no_directory_at_all(tmp_path):
    paths = ArtifactPaths(root=str(tmp_path), gpu_yaml="g",
                          lut_yaml=_fake_lut_yaml(tmp_path), lut_dir=None)
    st = lut_status(paths)
    assert paths.has_lut is False
    assert st["complete"] is False
    assert st["missing"] == st["wanted"]


def test_a_complete_database_reports_complete(tmp_path):
    lut_dir = tmp_path / "data"
    lut_dir.mkdir()
    for name in ("ew.csv", "gemm_bf16.csv", "sm_bf16.csv"):
        (lut_dir / name).write_text("x" * 512)
    paths = ArtifactPaths(root=str(tmp_path), gpu_yaml="g",
                          lut_yaml=_fake_lut_yaml(tmp_path), lut_dir=str(lut_dir))
    assert lut_status(paths)["complete"] is True


# -- with the real artifact ---------------------------------------------------

@needs_artifact
def test_the_lut_config_names_the_tables_we_expect():
    paths = locate_artifact(ARTIFACT)
    names = required_luts(paths.lut_yaml)
    assert any("gemmex_bf16bf16" in n for n in names)
    assert any("softmax" in n for n in names)
    assert any("layernorm" in n for n in names)
    assert any("elementwise" in n for n in names)


@needs_artifact
def test_status_is_complete_or_says_exactly_what_is_missing():
    st = lut_status(locate_artifact(ARTIFACT))
    if not st["complete"]:
        pytest.skip(f"LUT incomplete, missing {len(st['missing'])} tables")
    assert st["total_mb"] > 1.0


@needs_artifact
def test_measured_predictions_are_positive_and_self_consistent(rewriter):
    from dynshape.energaizer import build_gee_predictor

    pred = build_gee_predictor(ARTIFACT, verbose=False)
    kernels = rewriter.expand(batch=8, seqlen=512, mode="prefill")
    for q, op in kernels[:40]:
        t, p, e = pred.predict(q, op)
        assert t > 0 and p > 0 and e > 0
    assert pred.stats()["freq_mhz"] == LUT_FREQ_MHZ


@needs_artifact
def test_measured_and_roofline_agree_on_direction_if_not_magnitude(rewriter):
    """The reason to bother with the LUT: the roofline is an ansatz that agrees
    with the reasoning that produced it, so it can never falsify anything. These
    should order the same way -- and where they disagree is the finding."""
    from dynshape.energaizer import build_gee_predictor
    from dynshape.predictor import CachedPredictor

    gee = build_gee_predictor(ARTIFACT, verbose=False)
    ana = CachedPredictor(backend=AnalyticBackend(), freq=900)

    small = rewriter.expand(batch=1, seqlen=128, mode="prefill")
    big = rewriter.expand(batch=8, seqlen=1024, mode="prefill")

    def total_time(pred, kernels):
        return sum(pred.predict(q, op)[0] for q, op in kernels)

    assert total_time(gee, big) > total_time(gee, small)
    assert total_time(ana, big) > total_time(ana, small)


def test_find_artifact_root_locates_a_vendored_checkout(tmp_path):
    """The artifact lives inside a larger repo, so the clone lands one level
    above it -- the depth is not something this module should hardcode."""
    inner = tmp_path / "some" / "energaizer-ispass26-artifact-main"
    (inner / "gee").mkdir(parents=True)
    (inner / "gee" / "gee_utils.py").write_text("def get_gee(): pass")
    assert find_artifact_root(str(tmp_path)) == str(inner)


def test_find_artifact_root_accepts_the_root_itself(tmp_path):
    (tmp_path / "gee").mkdir()
    assert find_artifact_root(str(tmp_path)) == str(tmp_path)


def test_find_artifact_root_returns_none_when_absent(tmp_path):
    assert find_artifact_root(str(tmp_path)) is None


# -- the two things the authors' own demo revealed -----------------------------

def test_flatten_lut_makes_a_nested_archive_visible(tmp_path):
    """Load-bearing, not tidying: the LUT config references tables by BARE
    FILENAME against one `lut_folder_abs_path`, so a CSV that extracted a
    directory deeper is invisible to the estimator.  The failure is not 'file
    not found' -- it is a working estimator that raises on one op type, which
    `skip_unsupported` then silently drops."""
    from dynshape.energaizer import flatten_lut, lut_base_dir

    base = tmp_path / "database" / "data"
    nested = base / "precollected_database" / "yz8"
    nested.mkdir(parents=True)
    (nested / "a.csv").write_text("x")
    (nested / "b.csv").write_text("x")

    assert not [f for f in os.listdir(base) if f.endswith(".csv")]
    flat = flatten_lut(str(tmp_path))
    assert flat == lut_base_dir(str(tmp_path))
    assert sorted(f for f in os.listdir(flat) if f.endswith(".csv")) == ["a.csv", "b.csv"]


def test_flatten_lut_is_idempotent(tmp_path):
    from dynshape.energaizer import flatten_lut

    nested = tmp_path / "database" / "data" / "sub"
    nested.mkdir(parents=True)
    (nested / "a.csv").write_text("x")
    first = flatten_lut(str(tmp_path))
    second = flatten_lut(str(tmp_path))
    assert first == second
    assert sorted(f for f in os.listdir(first) if f.endswith(".csv")) == ["a.csv"]


def test_flatten_lut_with_nothing_to_flatten(tmp_path):
    from dynshape.energaizer import flatten_lut
    assert flatten_lut(str(tmp_path)) is None


def test_find_lut_dir_prefers_the_flattened_directory(tmp_path):
    """After flattening, the sentinel is reachable from `database/data`, and
    that is the directory the estimator must be pointed at -- not the
    subdirectory the archive happened to create."""
    from dynshape.energaizer import lut_base_dir

    nested = tmp_path / "database" / "data" / "precollected"
    nested.mkdir(parents=True)
    (nested / LUT_SENTINEL).write_text("x")
    assert find_lut_dir(str(tmp_path)) == lut_base_dir(str(tmp_path))


@needs_artifact
def test_measured_idle_power_is_not_the_round_number_we_assumed():
    """`simulate.IDLE_W` has been 47.0 W all along -- read off a config file.
    The measured table says 47.35 W at 900 MHz, and it is not flat: idle runs
    44.5 W at 210 MHz and 67.3 W at 1410 MHz."""
    from dynshape.energaizer import idle_power_table, measured_idle_power_w
    from dynshape.simulate import IDLE_W

    table = idle_power_table(ARTIFACT)
    assert measured_idle_power_w(ARTIFACT, 900) == pytest.approx(47.35, abs=0.01)
    assert abs(measured_idle_power_w(ARTIFACT, 900) - IDLE_W) < 1.0
    # Not flat -- boost clock pays half again as much for doing nothing.
    assert table[1410] > 1.4 * table[210]


@needs_artifact
def test_idle_power_falls_back_to_the_nearest_measured_clock():
    from dynshape.energaizer import measured_idle_power_w
    assert measured_idle_power_w(ARTIFACT, 901) == pytest.approx(
        measured_idle_power_w(ARTIFACT, 900))


@needs_artifact
def test_gee_imports_without_torch():
    """The artifact's own demo says inference needs no torch, and the import
    graph agrees: only `gee/frontend_utils.py` imports it, and `gee/__init__.py`
    does not pull that module in."""
    import ast

    init = os.path.join(ARTIFACT, "gee", "__init__.py")
    tree = ast.parse(open(init).read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[-1])
    assert "frontend_utils" not in imported
