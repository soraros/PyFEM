"""Environment manifest capture for benchmark runs (stdlib + installed packages)."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from typing import Any

THREAD_ENV_VARS = (
  "NUMBA_NUM_THREADS",
  "OMP_NUM_THREADS",
  "OPENBLAS_NUM_THREADS",
  "MKL_NUM_THREADS",
  "VECLIB_MAXIMUM_THREADS",
  "NUMBA_CACHE_DIR",
)


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
  """Capture CPU, OS, Python, library versions, BLAS backend, and thread env."""
  import numba
  import numpy
  import scipy

  manifest: dict[str, Any] = {
    "created_utc": __import__("datetime")
    .datetime.now(__import__("datetime").timezone.utc)
    .isoformat(timespec="seconds"),
    "git_revision": _git_revision(),
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
  }
  if threads is not None:
    manifest["threads_requested"] = threads
  return manifest
