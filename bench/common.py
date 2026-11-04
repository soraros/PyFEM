"""Shared timing, statistics, RSS, and JSON helpers for the benchmark harness."""

from __future__ import annotations

import json
import os
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = REPO_ROOT / "bench"
RESULTS_DIR = BENCH_DIR / "results"
GENERATED_DIR = BENCH_DIR / "generated"

SCHEMA_VERSION = 1

# Metric names shared by runs, the seeded baseline, and the regression gate.
STAGE_METRICS = (
  "meshgen",
  "load",
  "kernel",
  "scatter",
  "dedup",
  "assemble_coo",
  "assemble",
  "constrain",
  "factor",
  "backsolve",
  "solve",
  "e2e",
)
COLD_METRICS = ("import", "load", "assemble", "solve", "wall", "wall_truecold")


@dataclass
class MetricStats:
  """Timing statistics for one metric over repeated measurements."""

  min_ms: float
  median_ms: float
  mean_ms: float
  reps: int
  samples_ms: list[float] = field(repr=False)
  rss_delta_mb: float | None = None
  note: str = ""

  def to_json(self) -> dict[str, Any]:
    data = asdict(self)
    return data


@dataclass
class BenchRecord:
  """One benchmark cell: a (workload, side, mode, threads) measurement."""

  category: str
  workload: str
  side: str
  mode: str
  threads: int
  n_elems: int
  n_dofs: int
  metrics: dict[str, MetricStats]
  logical_bytes: dict[str, int] = field(default_factory=dict)
  correctness: dict[str, Any] = field(default_factory=dict)

  def to_json(self) -> dict[str, Any]:
    return {
      "category": self.category,
      "workload": self.workload,
      "side": self.side,
      "mode": self.mode,
      "threads": self.threads,
      "n_elems": self.n_elems,
      "n_dofs": self.n_dofs,
      "metrics": {k: v.to_json() for k, v in self.metrics.items()},
      "logical_bytes": self.logical_bytes,
      "correctness": self.correctness,
    }


def stats_from_samples(
  samples_ms: list[float], *, rss_delta_mb: float | None = None, note: str = ""
) -> MetricStats:
  """Summarize measured samples; ``None`` means that RSS was not measured."""
  return MetricStats(
    min_ms=min(samples_ms),
    median_ms=statistics.median(samples_ms),
    mean_ms=statistics.fmean(samples_ms),
    reps=len(samples_ms),
    samples_ms=samples_ms,
    rss_delta_mb=rss_delta_mb,
    note=note,
  )


def single_sample(milliseconds: float, note: str = "") -> MetricStats:
  """Construct one fixed measurement with unknown RSS."""
  return stats_from_samples([milliseconds], note=note)


def rss_mb() -> float:
  """Process peak RSS (high-water mark) in MiB."""
  usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  if sys.platform == "darwin":
    return usage / (1024 * 1024)
  return usage / 1024


def time_call(fn: Callable[[], Any]) -> float:
  """Time one call in milliseconds."""
  t0 = time.perf_counter()
  fn()
  return (time.perf_counter() - t0) * 1e3


def run_fresh_worker(
  module: str,
  arguments: list[str],
  *,
  threads: int,
  true_cold: bool,
  cache_prefix: str,
) -> tuple[dict[str, Any], float]:
  """Run one pinned worker process and decode its final JSON stdout line."""
  env = os.environ.copy()
  env["NUMBA_NUM_THREADS"] = str(threads)
  for var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
  ):
    env[var] = str(threads)
  temporary = tempfile.TemporaryDirectory(prefix=cache_prefix) if true_cold else None
  if temporary is not None:
    env["NUMBA_CACHE_DIR"] = temporary.name
  try:
    start = time.perf_counter()
    output = subprocess.run(
      [sys.executable, "-m", module, *arguments],
      env=env,
      cwd=REPO_ROOT,
      capture_output=True,
      text=True,
      check=True,
    )
    wall_ms = (time.perf_counter() - start) * 1e3
  finally:
    if temporary is not None:
      temporary.cleanup()
  return json.loads(output.stdout.strip().splitlines()[-1]), wall_ms


def measure(
  fn: Callable[[], Any],
  *,
  warmup: int = 1,
  min_reps: int = 5,
  max_reps: int = 30,
  budget_s: float = 0.6,
) -> MetricStats:
  """
  Time ``fn`` with warmup and an adaptive repetition count.

  The repetition count targets ``budget_s`` of total measured time, clamped to
  [min_reps, max_reps]; per-repetition samples yield min/median/mean. The RSS
  delta is the process high-water-mark increase across the whole rep block,
  not a per-rep allocation profile.
  """
  for _ in range(warmup):
    fn()
  first_ms = time_call(fn)
  reps = int(budget_s * 1e3 / max(first_ms, 1e-3))
  reps = max(min_reps, min(max_reps, reps))
  samples = [first_ms]
  rss_before = rss_mb()
  for _ in range(reps - 1):
    samples.append(time_call(fn))
  rss_after = rss_mb()
  return stats_from_samples(samples, rss_delta_mb=max(0.0, rss_after - rss_before))


def measure_fixed(
  fn: Callable[[], Any],
  *,
  warmup: int,
  reps: int,
  note: str = "",
) -> MetricStats:
  """Time fixed repetitions; RSS remains unknown rather than a zero delta."""
  for _ in range(warmup):
    fn()
  samples = [time_call(fn) for _ in range(reps)]
  return stats_from_samples(samples, note=note)


def write_run(
  path: Path,
  manifest: dict[str, Any],
  records: list[BenchRecord],
  *,
  coverage: str = "full",
) -> None:
  """Write a benchmark run JSON (manifest + records).

  ``coverage`` is ``"full"`` only when the run produced the complete workload
  matrix (all default sizes/materials, skims, warm + cold); anything reduced
  is ``"partial"`` and the regression gate treats unproduced baseline cells
  as informational skips rather than failures.
  """
  payload = {
    "schema": SCHEMA_VERSION,
    "kind": "benchmark-run",
    "coverage": coverage,
    "manifest": manifest,
    "records": [r.to_json() for r in records],
  }
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_run(path: Path) -> dict[str, Any]:
  """Read a benchmark run or seeded baseline JSON."""
  return json.loads(path.read_text(encoding="utf-8"))
