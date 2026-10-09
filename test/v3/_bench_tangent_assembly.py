"""Tangent assembly scale benchmark (not collected by pytest)."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

if sys.version_info < (3, 13):
  raise SystemExit("requires Python 3.13+")

import numba
from _bench_common import rss_mb, run_q8_patch_sweep, timeit_ms
from scipy.sparse import csr_array

from pyfem.v3 import load_problem, solve_linear
from pyfem.v3._prototype_assembly import assemble_loaded, assemble_tangent_loaded
from pyfem.v3.fem.assembly import compile_csr_pattern, dedup_coo_values
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


@dataclass
class DedupScaleRow:
  label: str
  n_elems: int
  nnz_coo: int
  scipy_ms: float
  fused_ms: float
  fused_1t_ms: float


def _benchmark_dedup(label: str, nx: int, ny: int) -> DedupScaleRow:
  """COO -> CSR duplicate summation: scipy ``tocsr`` vs the fused pattern path.

  The fused path precompiles the CSR pattern once (untimed here — a fixed
  topology cost, like driver-plan compilation) and reassembles values only;
  the scipy side re-sorts every call. Both produce a ``csr_array``.
  """
  coo = assemble_loaded(build_uniform_q8_loaded(nx, ny)).stiffness
  pattern = compile_csr_pattern(coo.row, coo.col, coo.shape)

  def fused() -> None:
    data = dedup_coo_values(pattern, coo.data)
    csr_array((data, pattern.indices, pattern.indptr), shape=pattern.shape)

  scipy_ms = timeit_ms(lambda: coo.tocsr(), warmup=2, repeats=10)
  fused_ms = timeit_ms(fused, warmup=2, repeats=10)
  previous = numba.get_num_threads()
  try:
    numba.set_num_threads(1)
    fused_1t_ms = timeit_ms(fused, warmup=2, repeats=10)
  finally:
    numba.set_num_threads(previous)
  return DedupScaleRow(
    label=label,
    n_elems=nx * ny,
    nnz_coo=int(coo.nnz),
    scipy_ms=scipy_ms,
    fused_ms=fused_ms,
    fused_1t_ms=fused_1t_ms,
  )


def _print_dedup_table(rows: list[DedupScaleRow]) -> None:
  print(
    f"{'mesh':>12} {'elems':>8} {'nnz_coo':>10} "
    f"{'scipy_ms':>10} {'fused_ms':>10} {'fused_1t_ms':>12}"
  )
  for row in rows:
    print(
      f"{row.label:>12} {row.n_elems:8d} {row.nnz_coo:10d} "
      f"{row.scipy_ms:10.3f} {row.fused_ms:10.3f} {row.fused_1t_ms:12.3f}"
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

  print("\n=== COO -> CSR dedup: scipy tocsr vs precompiled-pattern fused ===")
  _print_dedup_table(run_q8_patch_sweep(_benchmark_dedup))


if __name__ == "__main__":
  main()
