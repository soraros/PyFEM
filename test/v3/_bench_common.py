"""Shared helpers for v3 scale benchmarks (not collected by pytest)."""

from __future__ import annotations

import resource
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class ScaleRow:
  label: str
  n_elems: int
  n_dofs: int
  asm_ms: float
  solve_ms: float
  repeat_ms: float
  rss_mb: float
  nnz_coo: int = 0


def require_v3_python() -> None:
  if sys.version_info < (3, 13):
    raise SystemExit("requires Python 3.13+")


def rss_mb() -> float:
  usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  if sys.platform == "darwin":
    return usage / (1024 * 1024)
  return usage / 1024


def timeit_ms(fn: Callable[[], None], *, warmup: int, repeats: int) -> float:
  for _ in range(warmup):
    fn()
  t0 = time.perf_counter()
  for _ in range(repeats):
    fn()
  return (time.perf_counter() - t0) / repeats * 1e3


def print_linear_table(rows: list[ScaleRow], *, include_nnz: bool = True) -> None:
  if include_nnz:
    print(
      f"{'mesh':>12} {'elems':>8} {'dofs':>8} {'nnz_coo':>10} "
      f"{'asm_ms':>10} {'solve_ms':>10} {'repeat_ms':>11} {'rss_mb':>8}"
    )
    for row in rows:
      print(
        f"{row.label:>12} {row.n_elems:8d} {row.n_dofs:8d} {row.nnz_coo:10d} "
        f"{row.asm_ms:10.3f} {row.solve_ms:10.3f} {row.repeat_ms:11.3f} "
        f"{row.rss_mb:8.1f}"
      )
    return
  print(
    f"{'mesh':>12} {'elems':>8} {'dofs':>8} "
    f"{'asm_ms':>10} {'solve_ms':>10} {'repeat_ms':>11} {'rss_mb':>8}"
  )
  for row in rows:
    print(
      f"{row.label:>12} {row.n_elems:8d} {row.n_dofs:8d} "
      f"{row.asm_ms:10.3f} {row.solve_ms:10.3f} {row.repeat_ms:11.3f} "
      f"{row.rss_mb:8.1f}"
    )


Q8_PATCH_SIZES: list[tuple[str, int, int]] = [
  ("2x2", 2, 2),
  ("4x4", 4, 4),
  ("8x8", 8, 8),
  ("16x16", 16, 16),
  ("32x32", 32, 32),
  ("64x64", 64, 64),
]


def run_q8_patch_sweep(benchmark_fn) -> list[ScaleRow]:
  rows: list[ScaleRow] = []
  for label, nx, ny in Q8_PATCH_SIZES:
    try:
      rows.append(benchmark_fn(label, nx, ny))
    except MemoryError:
      print(f"{label}: skipped (MemoryError)")
      break
  return rows
