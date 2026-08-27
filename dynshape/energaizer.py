"""
energaizer.py
=============

**Wiring the real thing in.**  Everything else in this package can run on the
analytic roofline; this module is what replaces it with EnergAIzer's measured
lookup tables.

The difference matters more than "a better model".  The roofline is an *ansatz*:
`max(flops/peak, bytes/bandwidth)`, with power blended from the two utilisation
fractions.  It gets trends right by construction, which is exactly why it can
never falsify anything -- it agrees with the reasoning that produced it.
EnergAIzer's numbers come from **kernels that were actually run and measured** on
an A100, so it can disagree with the roofline, and where it does is where the
interesting information is.

WHAT IT NEEDS, AND WHY NONE OF IT SHIPS
---------------------------------------
Three things, and the third is the reason this is not just an import:

    the artifact repo   github.com/.../energaizer-ispass26-artifact  (code)
    two YAML configs    config/gpu/yz8.yaml, exp_config/a100_lut_config.yaml
    the LUT database    a separate ~GB download, NOT in the repo

`database/` in the artifact holds only the *harness that collects* a database,
not a database.  The measured CSVs live behind a Google Drive link, which is why
every figure in this project has been stamped SYNTHETIC until now.

WHAT COMES BACK
---------------
`lookup(query, query_type, target_freq=900, lookup_target='all')` returns
`(time_ms, power_W, energy_J)` -- checked against the artifact's own Colab demo,
which calls it exactly that way and labels the three.  The units are confirmed
twice over by its DVFS branch recomputing `e = p * t / 1000`, which is only
consistent if `t` is milliseconds and `p` is watts.

TWO THINGS THE DEMO MADE CLEAR THAT ARE EASY TO GET WRONG
----------------------------------------------------------
* **The LUT config references its tables by bare filename**, resolved against a
  single `lut_folder_abs_path`.  A CSV that extracted one directory deeper is
  invisible -- see `flatten_lut`, which is why that function exists.
* **torch is not needed.**  Only `gee/frontend_utils.py` imports it, and
  `gee/__init__.py` does not pull that module in.  torch is for *generating*
  workload JSONs, not for looking them up.

Those three are predicted by related but distinct paths, so they need not satisfy
`energy == power x time` exactly.  Downstream code assumes they do: an
iteration's average power is its summed energy over its summed time.  Rather than
silently pick one, `GeeBackend` measures the disagreement and reports it, and
`reconcile=` says explicitly what to do about it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

#: The artifact's own Google Drive id for `precollected_database.tar.gz`, from
#: `misc/download_files.sh`.
LUT_DRIVE_ID = "1krvqRFDnaqrJUT06V2psIua0wQr6ETAE"

#: One CSV that must exist for the database to be considered present.  Chosen
#: because it is the table GPT-2 leans on hardest: every projection and MLP GEMM.
LUT_SENTINEL = "yz8_gemmex_bf16bf16_freq900_noflush_lut_v2.csv"

#: The A100-40GB-PCIe tables are all measured at 900 MHz, so this is not a free
#: parameter -- asking for another frequency without a DVFS-aware estimator
#: silently extrapolates.
LUT_FREQ_MHZ = 900


@dataclass
class ArtifactPaths:
    """Where the pieces are, once found."""

    root: str
    gpu_yaml: str
    lut_yaml: str
    lut_dir: Optional[str]

    @property
    def has_lut(self) -> bool:
        return self.lut_dir is not None

    def describe(self) -> str:
        lut = self.lut_dir or "NOT FOUND -- the analytic fallback will be used"
        return (f"artifact : {self.root}\n"
                f"gpu yaml : {self.gpu_yaml}\n"
                f"lut yaml : {self.lut_yaml}\n"
                f"lut dir  : {lut}")


def lut_base_dir(root: str) -> str:
    """`database/data` -- where `misc/download_files.sh` extracts the archive."""
    return os.path.join(os.path.abspath(os.path.expanduser(root)),
                        "database", "data")


def flatten_lut(root: str, quiet: bool = True) -> Optional[str]:
    """Make every measured CSV visible in one directory, and return it.

    **This is load-bearing, not tidying.**  The LUT config references its tables
    by *bare filename*, resolved against whatever single directory is passed as
    `lut_folder_abs_path`.  The archive does not necessarily extract flat, so a
    CSV sitting one directory deeper is simply invisible to the estimator -- and
    the failure is not "file not found" but a working estimator that raises on
    one op type, which `skip_unsupported` then silently drops.

    Symlinks rather than copies, so a ~GB database is not duplicated.  Falls
    back to copying on filesystems that refuse symlinks.
    """
    import glob
    import shutil

    base = lut_base_dir(root)
    if not os.path.isdir(base):
        return None

    for src in glob.glob(os.path.join(base, "**", "*.csv"), recursive=True):
        dest = os.path.join(base, os.path.basename(src))
        if os.path.abspath(src) == os.path.abspath(dest) or os.path.exists(dest):
            continue
        try:
            os.symlink(os.path.abspath(src), dest)
        except OSError:                                       # pragma: no cover
            shutil.copy2(src, dest)
        if not quiet:
            print(f"[energaizer] linked {os.path.basename(src)} into database/data")

    return base if glob.glob(os.path.join(base, "*.csv")) else None


def find_lut_dir(root: str, sentinel: str = LUT_SENTINEL) -> Optional[str]:
    """Locate the measured CSVs, flattening them into one directory first.

    Searched rather than hard-coded because the archive's internal layout has
    changed between releases, and a hard-coded path fails with "no LUT" when the
    files are one directory deeper.
    """
    flat = flatten_lut(root)
    if flat and os.path.isfile(os.path.join(flat, sentinel)):
        return flat

    # No `database/data`, or the sentinel is not there: fall back to wherever it
    # actually lives, which at least gets the common tables loaded.
    for dirpath, _dirnames, filenames in os.walk(root):
        if sentinel in filenames:
            return dirpath
    return flat


def locate_artifact(root: str, lut_dir: Optional[str] = None) -> ArtifactPaths:
    """Resolve and check every path the estimator needs.

    Raises with the specific missing file rather than a generic failure, because
    "EnergAIzer could not be built" is the least useful possible message when
    three separate things could be absent.
    """
    root = os.path.abspath(os.path.expanduser(root))
    if not os.path.isdir(root):
        raise FileNotFoundError(f"artifact root does not exist: {root}")

    gpu_yaml = os.path.join(root, "config", "gpu", "yz8.yaml")
    lut_yaml = os.path.join(root, "experiments_endtoend", "exp_config",
                            "a100_lut_config.yaml")
    for label, path in (("GPU config", gpu_yaml), ("LUT config", lut_yaml)):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"{label} missing: {path}\n"
                f"Is {root} really the energaizer artifact checkout?")

    if lut_dir is None:
        lut_dir = find_lut_dir(root)
    elif not os.path.isdir(lut_dir):
        raise FileNotFoundError(f"lut_dir does not exist: {lut_dir}")

    return ArtifactPaths(root=root, gpu_yaml=gpu_yaml, lut_yaml=lut_yaml,
                         lut_dir=lut_dir)


def required_luts(lut_yaml: str) -> List[str]:
    """Every CSV filename the LUT config references, flattened out of the tree."""
    import yaml

    with open(lut_yaml) as f:
        cfg = yaml.safe_load(f)["lut_config"]

    found: List[str] = []

    def walk(node):
        if isinstance(node, dict):
            if "path" in node:
                found.append(node["path"])
                return
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(cfg)
    return sorted(set(found))


def lut_status(paths: ArtifactPaths) -> Dict:
    """Which measured tables the config wants, and which are actually on disk.

    Worth printing before a run: a *partially* extracted database is the failure
    mode that produces a working estimator which quietly raises on one op type,
    and `skip_unsupported` would then drop those kernels and hand back a trace
    that is simply cheaper than reality.
    """
    wanted = required_luts(paths.lut_yaml)
    if not paths.has_lut:
        return {"lut_dir": None, "wanted": wanted, "present": [],
                "missing": wanted, "complete": False, "total_mb": 0.0}

    present, missing, total = [], [], 0
    for name in wanted:
        p = os.path.join(paths.lut_dir, name)
        if os.path.isfile(p):
            present.append(name)
            total += os.path.getsize(p)
        else:
            missing.append(name)

    return {"lut_dir": paths.lut_dir, "wanted": wanted, "present": present,
            "missing": missing, "complete": not missing,
            "total_mb": total / 1024 ** 2}


def download_lut(root: str, quiet: bool = False) -> str:
    """Fetch and extract `precollected_database.tar.gz` into `database/data`.

    Tries `gdown` first -- it handles Drive's large-file confirmation page and is
    already installed on Colab -- and falls back to the artifact's own bash
    script.  Returns the directory the CSVs ended up in.
    """
    root = os.path.abspath(os.path.expanduser(root))
    dest = os.path.join(root, "database", "data")
    os.makedirs(dest, exist_ok=True)

    existing = find_lut_dir(root)
    if existing:
        if not quiet:
            print(f"[energaizer] LUT already present at {existing}")
        return existing

    tarball = os.path.join(dest, "precollected_database.tar.gz")
    got = False
    try:
        import gdown
        gdown.download(id=LUT_DRIVE_ID, output=tarball, quiet=quiet)
        got = os.path.isfile(tarball) and os.path.getsize(tarball) > 1024 ** 2
    except Exception as e:                                    # pragma: no cover
        if not quiet:
            print(f"[energaizer] gdown failed ({e!r}); trying the artifact script")

    if not got:                                               # pragma: no cover
        script = os.path.join(root, "misc", "gdrive_download.sh")
        if not os.path.isfile(script):
            raise FileNotFoundError(
                f"no gdown and no {script}; download the LUT manually from "
                f"https://drive.google.com/file/d/{LUT_DRIVE_ID}/view")
        subprocess.run(["bash", script,
                        f"https://drive.google.com/file/d/{LUT_DRIVE_ID}/view"],
                       cwd=dest, check=True)
    elif os.path.isfile(tarball):
        subprocess.run(["tar", "-xzf", tarball], cwd=dest, check=True)
        os.remove(tarball)

    found = find_lut_dir(root)
    if found is None:
        raise RuntimeError(
            f"the archive extracted but {LUT_SENTINEL} is not under {root}; "
            "the database layout may have changed")
    if not quiet:
        print(f"[energaizer] LUT ready at {found}")
    return found


def idle_power_table(root: str, gpu: str = "yz8") -> Dict[int, float]:
    """The **measured** idle power, per core clock, from the artifact.

    Worth using rather than a constant.  `simulate.IDLE_W` has been 47.0 W all
    along -- a round number read off a config file -- and the measured table says
    **47.35 W at 900 MHz**.  More importantly it is not flat: idle runs 44.5 W at
    210 MHz and **67.3 W at 1410 MHz**, so a run at boost clock pays half again
    as much for doing nothing.

    That is also the mechanism behind the U-shaped energy-vs-clock curve the
    artifact's own demo draws: on the voltage floor, raising the clock finishes
    the work sooner and pays this fixed floor for less time (energy falls), until
    voltage starts climbing and both the V^2 term and this idle number rise
    faster than the runtime shrinks.
    """
    import json

    path = os.path.join(os.path.abspath(os.path.expanduser(root)),
                        "config", "dvfs", gpu, "idle_power.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no idle power table at {path}")
    with open(path) as f:
        return {int(k): float(v) for k, v in json.load(f).items()}


def measured_idle_power_w(root: str, freq: int = LUT_FREQ_MHZ,
                          gpu: str = "yz8") -> float:
    """Idle power at one clock, measured.  47.35 W at 900 MHz on the A100."""
    table = idle_power_table(root, gpu)
    if freq in table:
        return table[freq]
    nearest = min(table, key=lambda f: abs(f - freq))
    print(f"[energaizer] no measured idle power at {freq} MHz; "
          f"using {nearest} MHz ({table[nearest]:.2f} W)")
    return table[nearest]


def build_estimator(paths: ArtifactPaths, dvfs_aware: bool = False):
    """The `gee` object itself.

    The artifact is not a package -- its modules import each other by top-level
    name (`from gee import get_gee`), so its root has to go on `sys.path` rather
    than being imported from a file path.
    """
    if not paths.has_lut:
        raise FileNotFoundError(
            "no LUT database found. Run `download_lut(root)` first, or point "
            "`lut_dir` at an existing copy. Without it EnergAIzer has nothing "
            "to look up -- the artifact ships the collection harness, not the "
            "measurements.")

    if paths.root not in sys.path:
        sys.path.insert(0, paths.root)

    try:
        from gee import get_gee
    except ImportError as e:
        raise ImportError(
            f"could not import `gee` from {paths.root} ({e}). It needs "
            "numpy, pandas, pyyaml, scipy, scikit-learn, cvxpy and opt-einsum. "
            "Note torch is NOT required: only `gee/frontend_utils.py` imports "
            "it, and `gee/__init__.py` does not pull that module in -- torch is "
            "for *generating* workload JSONs, not for looking them up.") from e

    return get_gee(gpu_yaml_path=paths.gpu_yaml,
                   lut_yaml_path=paths.lut_yaml,
                   dvfs_aware=dvfs_aware,
                   lut_folder_abs_path=paths.lut_dir)


def build_gee_predictor(root: str, lut_dir: Optional[str] = None,
                        freq: int = LUT_FREQ_MHZ,
                        reconcile: str = "report",
                        use_precomputed_coeff: bool = False,
                        verbose: bool = True):
    """A `CachedPredictor` backed by measured tables.

    Unlike `build_predictor`, this **raises** rather than degrading to the
    roofline.  Silent fallback is right for a demo notebook and wrong here: if
    you asked for EnergAIzer you want to know when you did not get it, not to
    read SYNTHETIC numbers under a heading that says measured.
    """
    from .predictor import CachedPredictor, GeeBackend

    paths = locate_artifact(root, lut_dir=lut_dir)
    if verbose:
        print(paths.describe())

    status = lut_status(paths)
    if not status["complete"]:
        raise FileNotFoundError(
            "the LUT database is incomplete -- missing "
            f"{len(status['missing'])} of {len(status['wanted'])} tables: "
            f"{status['missing'][:4]}{' ...' if len(status['missing']) > 4 else ''}\n"
            "A partial database builds an estimator that raises on some op "
            "types, and skipping those kernels would hand back a trace that is "
            "simply cheaper than reality.")
    if verbose:
        print(f"lut      : {len(status['present'])} tables, "
              f"{status['total_mb']:.0f} MB")

    if freq != LUT_FREQ_MHZ:
        print(f"[energaizer] WARNING: the A100 tables are measured at "
              f"{LUT_FREQ_MHZ} MHz; asking for {freq} MHz extrapolates.")

    estimator = build_estimator(paths)
    return CachedPredictor(
        backend=GeeBackend(estimator, reconcile=reconcile,
                           use_precomputed_coeff=use_precomputed_coeff),
        freq=freq)


#: The artifact is vendored inside this repo rather than published standalone,
#: so the clone lands one level above the artifact root.
DEFAULT_ARTIFACT_REPO = "https://github.com/shubhamOjha1000/single_kernel_GPU_model.git"


def find_artifact_root(start: str) -> Optional[str]:
    """The directory that *is* the artifact -- the one holding `gee/`.

    Searched rather than assumed, because the artifact is vendored inside a
    larger repo and its depth there is not something this module should hardcode.
    """
    start = os.path.abspath(os.path.expanduser(start))
    if os.path.isdir(os.path.join(start, "gee")):
        return start
    for dirpath, dirnames, _files in os.walk(start):
        if "gee" in dirnames and os.path.isfile(
                os.path.join(dirpath, "gee", "gee_utils.py")):
            return dirpath
    return None


def clone_artifact(dest: str = "energaizer-artifact",
                   url: str = DEFAULT_ARTIFACT_REPO,
                   quiet: bool = False) -> str:
    """Clone the repo holding the artifact, and return the **artifact root**."""
    dest = os.path.abspath(os.path.expanduser(dest))
    found = find_artifact_root(dest) if os.path.isdir(dest) else None
    if found:
        if not quiet:
            print(f"[energaizer] artifact already at {found}")
        return found

    subprocess.run(["git", "clone", "--depth", "1", url, dest], check=True)
    found = find_artifact_root(dest)
    if found is None:
        raise RuntimeError(
            f"cloned {url} into {dest} but found no directory containing "
            "`gee/gee_utils.py` -- is that the right repository?")
    if not quiet:
        print(f"[energaizer] artifact at {found}")
    return found


def benchmark_lookup(predictor, kernels, n: int = 30) -> Dict:
    """Measure the **cold** lookup rate, so a run's cost is knowable in advance.

    A measured lookup is not a table read.  EnergAIzer re-solves a small
    quadratic program per distinct shape -- once for time, again for power -- so
    it costs 50-300 ms where the roofline costs about a microsecond.  Multiply
    that by the few thousand distinct shapes a serving run produces and the
    difference between "two minutes" and "two hours" is a factor nobody should
    discover by waiting.

    `kernels` is a `[(query, op), ...]` list; only shapes the predictor has not
    already cached are timed, since a cache hit measures nothing.
    """
    import time

    timed = 0
    elapsed = 0.0
    for q, op in kernels:
        if predictor.key(q, op) in predictor.cache:
            continue
        t0 = time.perf_counter()
        predictor.predict(q, op)
        elapsed += time.perf_counter() - t0
        timed += 1
        if timed >= n:
            break

    if timed == 0:
        raise ValueError("every shape was already cached; nothing to time")
    return {"cold_lookups_timed": timed,
            "seconds_per_lookup": elapsed / timed,
            "lookups_per_second": timed / elapsed}


def project_run_minutes(seconds_per_lookup: float, distinct_shapes: int) -> float:
    """Wall-clock minutes for a run with this many distinct shapes.

    Cache *hits* are free by comparison -- a dict lookup against a QP solve -- so
    the distinct-shape count is the whole cost model.
    """
    return seconds_per_lookup * distinct_shapes / 60.0


def estimate_distinct_shapes(n_requests: int, per_request: float = 29.0) -> int:
    """Rough distinct-shape count for a serving run of this size.

    The constant is measured, not derived: a 400-request run of the reference
    traffic reported 11,508 distinct shapes, so about 29 per request.  It scales
    with how varied the traffic is -- every distinct KV length and every distinct
    fused token count is a new shape -- so treat it as an order of magnitude, not
    a prediction.  Bucketing contexts (`SchedulerConfig`) is what holds it down.
    """
    return int(n_requests * per_request)
