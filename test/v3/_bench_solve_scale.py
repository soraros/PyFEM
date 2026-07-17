"""Q8 patch scale benchmark (PlaneStress and PlaneStrain; not collected by pytest)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if sys.version_info < (3, 13):
  raise SystemExit("requires Python 3.13+")

from _bench_common import (
  ScaleRow,
  print_linear_table,
  rss_mb,
  run_q8_patch_sweep,
  timeit_ms,
)

from pyfem.fem.Assembly import assembleExternalForce, assembleTangentStiffness, prepare
from pyfem.io.InputReader import InputRead
from pyfem.v3 import load_problem, solve_linear
from pyfem.v3._prototype_assembly import assemble_loaded
from pyfem.v3.mesh.refined_patch import build_uniform_q8_loaded
from pyfem.v3.solver.context import prepare_cached_linear

ROOT = Path(__file__).resolve().parents[2]
SKIM_PRO = ROOT / "skims" / "patch_test8" / "skim.pro"
LOADED_SKIM_PRO = ROOT / "skims" / "patch_test8_loaded" / "skim.pro"
PLANE_STRAIN_SKIM = ROOT / "skims" / "patch_test8_plane_strain" / "skim.pro"


def _benchmark_patch(label: str, nx: int, ny: int, *, material_type: str) -> ScaleRow:
  loaded = build_uniform_q8_loaded(nx, ny, material_type=material_type)
  problem = loaded.problem
  asm = timeit_ms(lambda: assemble_loaded(loaded), warmup=3, repeats=20)
  ctx = prepare_cached_linear(loaded)
  solve = timeit_ms(lambda: ctx.solve(), warmup=3, repeats=30)
  repeat = timeit_ms(lambda: ctx.solve(), warmup=0, repeats=50)
  return ScaleRow(
    label=label,
    n_elems=problem.n_elems,
    n_dofs=problem.n_dofs,
    nnz_coo=problem.n_elems * 256,
    asm_ms=asm,
    solve_ms=solve,
    repeat_ms=repeat,
    rss_mb=rss_mb(),
  )


def _legacy_solve_ms() -> float:
  props, globdat = InputRead(str(SKIM_PRO))

  def run_once() -> None:
    prepare(props, globdat)
    k, _ = assembleTangentStiffness(props, globdat)
    fext = assembleExternalForce(props, globdat)
    globdat.dofs.solve(k, fext)

  return timeit_ms(run_once, warmup=2, repeats=20)


def run_material(material_type: str) -> None:
  print(f"=== v3 uniform Q8 patch scale — {material_type} (warm Numba) ===")
  rows = run_q8_patch_sweep(
    lambda label, nx, ny: _benchmark_patch(label, nx, ny, material_type=material_type),
  )
  print_linear_table(rows)

  skim = PLANE_STRAIN_SKIM if material_type == "PlaneStrain" else SKIM_PRO
  print(f"\n=== skim parity spot-check (5 elems, {material_type}) ===")
  loaded_skim = load_problem(skim)
  v3_state = solve_linear(loaded_skim)
  print(f"  v3 skim solve ok, ||u||={np.linalg.norm(v3_state):.6e}")

  if material_type == "PlaneStrain":
    print("\n=== 16x16 stress vs strain assembly (same mesh, different C) ===")
    stress = build_uniform_q8_loaded(16, 16, material_type="PlaneStress")
    strain = build_uniform_q8_loaded(16, 16, material_type="PlaneStrain")
    asm_stress = timeit_ms(lambda: assemble_loaded(stress), warmup=3, repeats=20)
    asm_strain = timeit_ms(lambda: assemble_loaded(strain), warmup=3, repeats=20)
    print(f"  PlaneStress asm: {asm_stress:.3f} ms/call")
    print(f"  PlaneStrain asm: {asm_strain:.3f} ms/call")


def main() -> None:
  run_material("PlaneStress")

  print("\n=== legacy PatchTest8 skim (5 elems, reference) ===")
  print(f"  legacy solve-only: {_legacy_solve_ms():.3f} ms/call")

  print("\n=== patch_test8_loaded context repeat (5 elems) ===")
  ctx = prepare_cached_linear(load_problem(LOADED_SKIM_PRO))
  ctx.solve()
  repeat_loaded = timeit_ms(lambda: ctx.solve(), warmup=0, repeats=50)
  print(f"  repeat solve: {repeat_loaded:.3f} ms/call")

  print()
  run_material("PlaneStrain")


if __name__ == "__main__":
  if len(sys.argv) > 1 and sys.argv[1] == "PlaneStrain":
    run_material("PlaneStrain")
  elif len(sys.argv) > 1 and sys.argv[1] == "PlaneStress":
    run_material("PlaneStress")
  else:
    main()
