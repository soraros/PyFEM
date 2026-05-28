"""One-off scale benchmark: v3 assembly/solve vs mesh size (not collected by pytest)."""

from __future__ import annotations

import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

if sys.version_info < (3, 13):
  raise SystemExit("requires Python 3.13+")

from pyfem.fem.Assembly import assembleExternalForce, assembleTangentStiffness, prepare
from pyfem.io.InputReader import InputRead
from pyfem.v3 import load_problem
from pyfem.v3.assembly import assemble_loaded
from pyfem.v3.mesh import build_dof_map
from pyfem.v3.mesh.refined_patch import build_uniform_q8_patch
from pyfem.v3.pack import pack_problem
from pyfem.v3.solver.context import prepare_linear_solve
from pyfem.v3.solver.linear import solve_linear
from pyfem.v3.types import LoadedProblem

ROOT = Path(__file__).resolve().parents[2]
SKIM_PRO = ROOT / "skims" / "patch_test8" / "skim.pro"
LOADED_SKIM_PRO = ROOT / "skims" / "patch_test8_loaded" / "skim.pro"


@dataclass
class ScaleRow:
  label: str
  n_elems: int
  n_dofs: int
  nnz_coo: int
  asm_ms: float
  solve_ms: float
  repeat_ms: float
  rss_mb: float


def _rss_mb() -> float:
  usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  if sys.platform == "darwin":
    return usage / (1024 * 1024)
  return usage / 1024


def _timeit(fn, *, warmup: int, repeats: int) -> float:
  for _ in range(warmup):
    fn()
  t0 = time.perf_counter()
  for _ in range(repeats):
    fn()
  return (time.perf_counter() - t0) / repeats * 1e3


def _make_loaded_patch(nx: int, ny: int) -> LoadedProblem:
  mesh, constraints = build_uniform_q8_patch(nx, ny)
  dof_map = build_dof_map(mesh)
  problem = pack_problem(mesh, dof_map, 1.0e6, 0.25, constraints)
  return LoadedProblem(
    problem=problem,
    name=f"patch_{nx}x{ny}",
    element_type="SmallStrainContinuum",
    material_type="PlaneStress",
    solver_type="LinearSolver",
    element_group="ContElem",
    mesh_path=None,
  )


def _legacy_solve_ms() -> float | None:
  props, globdat = InputRead(str(SKIM_PRO))

  def run_once() -> None:
    prepare(props, globdat)
    k, _ = assembleTangentStiffness(props, globdat)
    fext = assembleExternalForce(props, globdat)
    globdat.dofs.solve(k, fext)

  return _timeit(run_once, warmup=2, repeats=20)


def _benchmark_patch(label: str, nx: int, ny: int) -> ScaleRow:
  loaded = _make_loaded_patch(nx, ny)
  problem = loaded.problem
  nnz_coo = problem.n_elems * 256
  asm = _timeit(lambda: assemble_loaded(loaded), warmup=3, repeats=20)
  ctx = prepare_linear_solve(loaded)
  solve = _timeit(lambda: ctx.solve(), warmup=3, repeats=30)
  repeat = _timeit(lambda: ctx.solve(), warmup=0, repeats=50)
  rss = _rss_mb()
  return ScaleRow(
    label=label,
    n_elems=problem.n_elems,
    n_dofs=problem.n_dofs,
    nnz_coo=nnz_coo,
    asm_ms=asm,
    solve_ms=solve,
    repeat_ms=repeat,
    rss_mb=rss,
  )


def _print_table(rows: list[ScaleRow]) -> None:
  print(
    f"{'mesh':>12} {'elems':>8} {'dofs':>8} {'nnz_coo':>10} "
    f"{'asm_ms':>10} {'solve_ms':>10} {'repeat_ms':>11} {'rss_mb':>8}"
  )
  for row in rows:
    print(
      f"{row.label:>12} {row.n_elems:8d} {row.n_dofs:8d} {row.nnz_coo:10d} "
      f"{row.asm_ms:10.3f} {row.solve_ms:10.3f} {row.repeat_ms:11.3f} {row.rss_mb:8.1f}"
    )


def main() -> None:
  print("=== v3 uniform Q8 patch scale (warm Numba) ===")
  sizes = [
    ("2x2", 2, 2),
    ("4x4", 4, 4),
    ("8x8", 8, 8),
    ("16x16", 16, 16),
    ("32x32", 32, 32),
    ("64x64", 64, 64),
  ]
  rows: list[ScaleRow] = []
  for label, nx, ny in sizes:
    try:
      rows.append(_benchmark_patch(label, nx, ny))
    except MemoryError:
      print(f"{label}: skipped (MemoryError)")
      break

  _print_table(rows)

  print("\n=== legacy PatchTest8 skim (5 elems, reference) ===")
  leg_ms = _legacy_solve_ms()
  if leg_ms is not None:
    print(f"  legacy solve-only: {leg_ms:.3f} ms/call")

  print("\n=== patch_test8_loaded context repeat (5 elems) ===")
  loaded_context = load_problem(LOADED_SKIM_PRO)
  ctx = prepare_linear_solve(loaded_context)
  ctx.solve()
  repeat_loaded = _timeit(lambda: ctx.solve(), warmup=0, repeats=50)
  print(f"  repeat solve: {repeat_loaded:.3f} ms/call")

  print("\n=== skim parity spot-check (5 elems) ===")
  loaded_skim = load_problem(SKIM_PRO)
  v3_state = solve_linear(loaded_skim)
  print(f"  v3 skim solve ok, ||u||={np.linalg.norm(v3_state):.6e}")


if __name__ == "__main__":
  main()
