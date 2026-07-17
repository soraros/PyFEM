"""Structural truss/spring scale benchmark (not collected by pytest)."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

if sys.version_info < (3, 13):
  raise SystemExit("requires Python 3.13+")

from _bench_common import rss_mb, timeit_ms

from pyfem.v3 import load_problem, solve_riks
from pyfem.v3._prototype_assembly import assemble_tangent_loaded
from pyfem.v3.mesh.truss_fan import build_truss_fan_loaded

ROOT = Path(__file__).resolve().parents[2]
SKIM_PRO = ROOT / "skims" / "shallow_truss_riks" / "skim.pro"


@dataclass
class StructuralScaleRow:
  label: str
  n_elems: int
  n_dofs: int
  tangent_ms: float
  riks_ms: float
  rss_mb: float


def _benchmark_fan(n_rays: int) -> StructuralScaleRow:
  loaded = build_truss_fan_loaded(n_rays, max_lam=2.0)
  state = np.zeros(loaded.problem.n_dofs, dtype=np.float64)
  tangent_ms = timeit_ms(
    lambda: assemble_tangent_loaded(loaded, state),
    warmup=3,
    repeats=20,
  )
  riks_ms = timeit_ms(lambda: solve_riks(loaded), warmup=1, repeats=5)
  return StructuralScaleRow(
    label=f"fan_{n_rays}",
    n_elems=loaded.problem.n_elems,
    n_dofs=loaded.problem.n_dofs,
    tangent_ms=tangent_ms,
    riks_ms=riks_ms,
    rss_mb=rss_mb(),
  )


def _print_table(rows: list[StructuralScaleRow]) -> None:
  print(
    f"{'mesh':>10} {'elems':>8} {'dofs':>8} "
    f"{'tangent_ms':>12} {'riks_ms':>10} {'rss_mb':>8}"
  )
  for row in rows:
    print(
      f"{row.label:>10} {row.n_elems:>8} {row.n_dofs:>8} "
      f"{row.tangent_ms:>12.2f} {row.riks_ms:>10.2f} {row.rss_mb:>8.1f}"
    )


def main() -> None:
  skim = load_problem(SKIM_PRO)
  skim_ms = timeit_ms(lambda: solve_riks(skim), warmup=1, repeats=10)
  print(f"reference shallow_truss_riks skim riks_ms={skim_ms:.2f}")
  print()

  rows = [_benchmark_fan(n) for n in (8, 32, 128, 512)]
  _print_table(rows)


if __name__ == "__main__":
  main()
