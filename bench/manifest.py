"""Environment manifest capture for benchmark runs (stdlib + installed packages).

Alongside the git revision, every manifest records the working-tree state at
capture (``working_tree.clean`` plus the porcelain file list when dirty) so
gate adjudication can check a run's provenance mechanically (M60, per the
M51 review's observation that two runs with identical ``git_revision`` could
not otherwise be told apart by tree state).
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

THREAD_ENV_VARS = (
  "NUMBA_NUM_THREADS",
  "OMP_NUM_THREADS",
  "OPENBLAS_NUM_THREADS",
  "MKL_NUM_THREADS",
  "VECLIB_MAXIMUM_THREADS",
  "NUMBA_CACHE_DIR",
)

# Caller -> callee source pairs whose numba cache coherence is recorded in
# every run manifest (NUMBA_CACHING.md §6). A cached caller embeds its jit
# callees' machine code, so a caller .nbi cache entry older than the callee
# source makes the run's timings cache-suspect even when the parity gates pass
# (performance-only callee edits leave numerics unchanged). These pairs are the
# fem kernel call-graph edges that cross module boundaries; their callers
# currently carry documented cache=False, so any caller cache entry at all is
# worth recording.
CACHE_COHERENCE_PAIRS: tuple[tuple[str, str], ...] = (
  ("pyfem/v3/fem/element.py", "pyfem/v3/fem/quadrature.py"),
  ("pyfem/v3/fem/element.py", "pyfem/v3/fem/shapes.py"),
  ("pyfem/v3/fem/element.py", "pyfem/v3/fem/kinematics.py"),
  ("pyfem/v3/fem/tl_element.py", "pyfem/v3/fem/quadrature.py"),
  ("pyfem/v3/fem/tl_element.py", "pyfem/v3/fem/shapes.py"),
  ("pyfem/v3/fem/tl_element.py", "pyfem/v3/fem/tl_kinematics.py"),
)


def check_cache_coherence(root: Path | None = None) -> list[dict[str, Any]]:
  """Check that each declared caller's live .nbi cache entries are newer than
  the callee source (NUMBA_CACHING.md §6). An entry counts as live only when
  it is at least as new as the caller source itself — numba's per-file
  (mtime, size) stamp gate empties the index of any caller edited since the
  entry was written, so such dead entries are excluded. A caller with no live
  entries is trivially coherent (e.g. documented cache=False kernels, whose
  stale leftover files are inert). When NUMBA_CACHE_DIR redirects the cache
  out of the source tree, no entries are found here; thread_env records that
  setting.
  """
  base = root if root is not None else Path(__file__).resolve().parents[1]
  pairs: list[dict[str, Any]] = []
  for caller_rel, callee_rel in CACHE_COHERENCE_PAIRS:
    caller = base / caller_rel
    callee = base / callee_rel
    cache_dir = caller.parent / "__pycache__"
    entries = (
      sorted(cache_dir.glob(f"{caller.stem}.*.nbi")) if cache_dir.is_dir() else []
    )
    caller_mtime = caller.stat().st_mtime
    live_mtimes = [
      entry.stat().st_mtime
      for entry in entries
      if entry.stat().st_mtime >= caller_mtime
    ]
    callee_mtime = callee.stat().st_mtime if callee.is_file() else None
    coherent = callee_mtime is not None and all(
      mtime >= callee_mtime for mtime in live_mtimes
    )
    pairs.append(
      {
        "caller": caller_rel,
        "callee": callee_rel,
        "callee_source_mtime_utc": (
          datetime.fromtimestamp(callee_mtime, UTC).isoformat()
          if callee_mtime is not None
          else None
        ),
        "caller_cache_entries": len(entries),
        "caller_cache_live_entries": len(live_mtimes),
        "caller_cache_oldest_live_mtime_utc": (
          datetime.fromtimestamp(min(live_mtimes), UTC).isoformat()
          if live_mtimes
          else None
        ),
        "coherent": coherent,
      }
    )
  return pairs


def _cpu_brand() -> str:
  if sys.platform == "darwin":
    try:
      out = subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string"],
        capture_output=True,
        text=True,
        check=True,
      )
      return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
      pass
  return platform.processor() or platform.machine()


def _git_revision() -> str:
  try:
    out = subprocess.run(
      ["git", "rev-parse", "HEAD"],
      capture_output=True,
      text=True,
      check=True,
      cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    return out.stdout.strip()
  except (OSError, subprocess.CalledProcessError):
    return "unknown"


def _git_working_tree() -> dict[str, Any]:
  """Working-tree cleanliness at capture: ``clean`` flag + dirty file list.

  ``clean`` is True only when ``git status --porcelain`` is empty (no
  modified tracked files, no untracked files). When dirty, ``dirty_files``
  lists the porcelain entries verbatim (``"<XY> <path>"``, untracked as
  ``??``) so adjudication can see exactly what diverges from
  ``git_revision``. Both values are None when git itself fails (e.g. a
  tarball install), matching ``_git_revision``'s "unknown".
  """
  try:
    out = subprocess.run(
      ["git", "status", "--porcelain"],
      capture_output=True,
      text=True,
      check=True,
      cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
  except (OSError, subprocess.CalledProcessError):
    return {"clean": None, "dirty_files": None}
  entries = sorted(line for line in out.stdout.splitlines() if line.strip())
  return {"clean": not entries, "dirty_files": entries}


def _numpy_linalg_backend() -> dict[str, Any]:
  import numpy as np

  info: dict[str, Any] = {}
  try:
    config = np.__config__.show(mode="dicts")  # numpy >= 1.26
    blas = config.get("Build Dependencies", {}).get("blas", {})
    lapack = config.get("Build Dependencies", {}).get("lapack", {})
    info["blas"] = {k: blas.get(k) for k in ("name", "version") if k in blas}
    info["lapack"] = {k: lapack.get(k) for k in ("name", "version") if k in lapack}
  except (TypeError, AttributeError):
    info["blas"] = {"name": "unavailable (numpy __config__ dict mode missing)"}
  return info


def collect_manifest(threads: int | None = None) -> dict[str, Any]:
  """Capture CPU, OS, Python, library versions, BLAS backend, thread env,
  and git working-tree state."""
  import numba
  import numpy
  import scipy

  try:
    import threadpoolctl  # noqa: F401

    threadpoolctl_available = True
  except ImportError:
    threadpoolctl_available = False

  manifest: dict[str, Any] = {
    "created_utc": __import__("datetime")
    .datetime.now(__import__("datetime").timezone.utc)
    .isoformat(timespec="seconds"),
    "git_revision": _git_revision(),
    "working_tree": _git_working_tree(),
    "cpu": _cpu_brand(),
    "machine": platform.machine(),
    "os": f"{platform.system()} {platform.release()} ({platform.version()})",
    "python": sys.version.split()[0],
    "python_executable": sys.executable,
    "numpy": numpy.__version__,
    "scipy": scipy.__version__,
    "numba": numba.__version__,
    "linalg_backend": _numpy_linalg_backend(),
    "thread_env": {var: os.environ.get(var, "") for var in THREAD_ENV_VARS},
    "numba_threads_default": int(numba.config.NUMBA_NUM_THREADS),
    "threadpoolctl_available": threadpoolctl_available,
  }
  if threads is not None:
    manifest["threads_requested"] = threads
  pairs = check_cache_coherence()
  manifest["numba_cache_coherence"] = {
    "coherent": all(pair["coherent"] for pair in pairs),
    "pairs": pairs,
  }
  return manifest
