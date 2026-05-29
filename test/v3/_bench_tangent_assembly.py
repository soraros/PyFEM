"""Tangent assembly scale benchmark (not collected by pytest)."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

if sys.version_info < (3, 13):
  raise SystemExit("requires Python 3.13+")

from _bench_common import rss_mb, run_q8_patch_sweep, timeit_ms

from pyfem.v3 import load_problem, solve_linear
from pyfem.v3.assembly import assemble_loaded, assemble_tangent_loaded
from pyfem.v3.mesh.refined_patch import build_uniform_q8_loaded
from pyfem.v3.solver.context import prepare_cached_linear

ROOT = Path(__file__).resolve().parents[2]
LOADED_SKIM_PRO = ROOT / "skims" / "patch_test8_loaded" / "skim.pro"


@dataclass
class TangentScaleRow:
  label: str
  n_elems: int
  n_dofs: int
  linear_ms: float
  tangent_ms: float
  fint_ms: float
  repeat_fint_ms: float
  rss_mb: float


def _benchmark_patch(label: str, nx: int, ny: int) -> TangentScaleRow:
  loaded = build_uniform_q8_loaded(nx, ny)
  problem = loaded.problem
  state = prepare_cached_linear(loaded).solve()
  linear_ms = timeit_ms(lambda: assemble_loaded(loaded), warmup=3, repeats=20)
  tangent_ms = timeit_ms(
    lambda: assemble_tangent_loaded(loaded, state),
    warmup=3,
    repeats=20,
  )
  ctx = prepare_cached_linear(loaded)
  fint_ms = timeit_ms(lambda: ctx.internal_force(state), warmup=3, repeats=30)
  repeat_fint_ms = timeit_ms(lambda: ctx.internal_force(state), warmup=0, repeats=50)
  return TangentScaleRow(
    label=label,
    n_elems=problem.n_elems,
    n_dofs=problem.n_dofs,
    linear_ms=linear_ms,
    tangent_ms=tangent_ms,
    fint_ms=fint_ms,
    repeat_fint_ms=repeat_fint_ms,
    rss_mb=rss_mb(),
  )


def _print_table(rows: list[TangentScaleRow]) -> None:
  print(
    f"{'mesh':>12} {'elems':>8} {'dofs':>8} "
    f"{'linear_ms':>11} {'tangent_ms':>11} {'fint_ms':>9} {'repeat_fint':>12} {'rss_mb':>8}"
  )
  for row in rows:
    print(
      f"{row.label:>12} {row.n_elems:8d} {row.n_dofs:8d} "
      f"{row.linear_ms:11.3f} {row.tangent_ms:11.3f} {row.fint_ms:9.3f} "
      f"{row.repeat_fint_ms:12.3f} {row.rss_mb:8.1f}"
    )


def main() -> None:
  print("=== v3 tangent assembly scale (warm Numba, fused K_e + f_int) ===")
  rows = run_q8_patch_sweep(_benchmark_patch)
  _print_table(rows)

  print("\n=== patch_test8_loaded skim (5 elems) ===")
  loaded = load_problem(LOADED_SKIM_PRO)
  state = solve_linear(loaded)
  tangent_skim = timeit_ms(
    lambda: assemble_tangent_loaded(loaded, state),
    warmup=3,
    repeats=30,
  )
  ctx = prepare_cached_linear(loaded)
  repeat_fint = timeit_ms(lambda: ctx.internal_force(state), warmup=0, repeats=50)
  print(f"  tangent assemble: {tangent_skim:.3f} ms/call")
  print(f"  cached K @ u:     {repeat_fint:.3f} ms/call")


if __name__ == "__main__":
  main()
