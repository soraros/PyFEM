"""Cold-mode worker: measure one case in a fresh interpreter and print JSON.

Invoked by ``bench.run cold`` as ``python -m bench.cold_worker`` so every
measurement pays the real import + first-dispatch cost. With the numba cache
present this is the CI cold mode; with ``NUMBA_CACHE_DIR`` pointed at an empty
directory it is the true-cold (JIT compile) mode, documented in the README.
"""

from __future__ import annotations

import argparse
import json
import time

from bench.common import rss_mb
from bench.workloads import skim_cases


def _run_v3_q8(nx: int, material_type: str) -> dict[str, float]:
  import numpy as np

  from pyfem.v3._prototype_assembly import assemble_loaded
  from pyfem.v3.mesh.refined_patch import (
    build_uniform_q8_loaded,
    patch_displacement,
  )
  from pyfem.v3.solver.context import CachedLinearSystem

  t0 = time.perf_counter()
  loaded = build_uniform_q8_loaded(nx, nx, material_type=material_type)
  load_ms = (time.perf_counter() - t0) * 1e3

  t0 = time.perf_counter()
  system = assemble_loaded(loaded)
  k_csr = system.stiffness.tocsr()
  assemble_ms = (time.perf_counter() - t0) * 1e3

  t0 = time.perf_counter()
  state = CachedLinearSystem.from_system(loaded.problem, system).solve()
  solve_ms = (time.perf_counter() - t0) * 1e3
  del k_csr

  coords = loaded.problem.coords
  exact = np.zeros(loaded.problem.n_dofs)
  for node in range(coords.shape[0]):
    exact[2 * node], exact[2 * node + 1] = patch_displacement(
      float(coords[node, 0]), float(coords[node, 1])
    )
  denom = np.maximum(np.abs(exact), 1.0e-30)
  analytic_rel = float((np.abs(state - exact) / denom).max())
  return {
    "load_ms": load_ms,
    "assemble_ms": assemble_ms,
    "solve_ms": solve_ms,
    "analytic_rel": analytic_rel,
  }


def _run_v3_skim(name: str) -> dict[str, float]:
  from pyfem.v3 import load_problem, solve_linear, solve_nonlinear, solve_riks

  (case,) = [c for c in skim_cases() if c.name == name]
  t0 = time.perf_counter()
  loaded = load_problem(case.pro_path)
  load_ms = (time.perf_counter() - t0) * 1e3

  t0 = time.perf_counter()
  if case.kind == "linear":
    solve_linear(loaded)
  elif case.kind == "nonlinear":
    solve_nonlinear(loaded)
  else:
    solve_riks(loaded)
  solve_ms = (time.perf_counter() - t0) * 1e3
  return {"load_ms": load_ms, "solve_ms": solve_ms}


def _run_legacy_q8(nx: int, material_type: str) -> dict[str, float]:
  from bench.legacy_cases import (
    legacy_assembly,
    legacy_load,
    legacy_solve,
    pro_path_for,
  )

  pro_path = pro_path_for(nx, nx, material_type)
  if not pro_path.exists():
    msg = f"generated case missing: {pro_path} (run the gates/warm phase first)"
    raise SystemExit(msg)
  t0 = time.perf_counter()
  props, globdat = legacy_load(pro_path)
  load_ms = (time.perf_counter() - t0) * 1e3

  t0 = time.perf_counter()
  matrix = legacy_assembly(props, globdat)
  assemble_ms = (time.perf_counter() - t0) * 1e3

  t0 = time.perf_counter()
  legacy_solve(props, globdat, matrix)
  solve_ms = (time.perf_counter() - t0) * 1e3
  return {"load_ms": load_ms, "assemble_ms": assemble_ms, "solve_ms": solve_ms}


def _run_legacy_skim(name: str) -> dict[str, float]:
  from bench.legacy_cases import (
    legacy_linear_state,
    legacy_load,
    legacy_nonlinear_state,
    legacy_riks_state,
  )

  (case,) = [c for c in skim_cases() if c.name == name]
  t0 = time.perf_counter()
  legacy_load(case.pro_path)
  load_ms = (time.perf_counter() - t0) * 1e3

  t0 = time.perf_counter()
  if case.kind == "linear":
    legacy_linear_state(case.pro_path)
  elif case.kind == "nonlinear":
    legacy_nonlinear_state(case.pro_path)
  else:
    legacy_riks_state(case.pro_path)
  solve_ms = (time.perf_counter() - t0) * 1e3
  return {"load_ms": load_ms, "solve_ms": solve_ms}


def _run_v3_family(spec: str) -> tuple[dict[str, float], int, int]:
  """One cold family case (v3 side): deck-stack or Riks-driver load + solve."""
  from bench.family_v3 import (
    cantilever_family_load,
    cantilever_family_solve,
    riks_fan_load,
    riks_fan_solve,
  )
  from bench.legacy_cases import cantilever_pro_path_for

  name, _, size = spec.partition(":")
  if name == "tl-cantilever":
    nx_str, _, ny_str = size.partition("x")
    nx, ny = int(nx_str), int(ny_str)
    pro_path = cantilever_pro_path_for(nx, ny)
    if not pro_path.exists():
      msg = f"generated case missing: {pro_path} (run the gates/warm phase first)"
      raise SystemExit(msg)
    t0 = time.perf_counter()
    prepared = cantilever_family_load(pro_path)
    load_ms = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    cantilever_family_solve(prepared)
    solve_ms = (time.perf_counter() - t0) * 1e3
    n_elems = nx * ny
    n_dofs = 2 * ((ny + 1) * (2 * nx + 1) + ny * (nx + 1))
    return {"load_ms": load_ms, "solve_ms": solve_ms}, n_elems, n_dofs
  if name == "riks-fan":
    n_rays = int(size)
    t0 = time.perf_counter()
    prepared = riks_fan_load(n_rays)
    load_ms = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    riks_fan_solve(prepared)
    solve_ms = (time.perf_counter() - t0) * 1e3
    return {"load_ms": load_ms, "solve_ms": solve_ms}, n_rays, 2 * (n_rays + 1)
  msg = f"Unknown family case {spec!r}"
  raise SystemExit(msg)


def _run_legacy_family(spec: str) -> tuple[dict[str, float], int, int]:
  """One cold family case (legacy side): load + full solver loop."""
  from bench.legacy_cases import (
    cantilever_pro_path_for,
    fan_pro_path_for,
    legacy_load,
    legacy_nonlinear_solver,
    legacy_riks_solver,
  )

  name, _, size = spec.partition(":")
  if name == "tl-cantilever":
    nx_str, _, ny_str = size.partition("x")
    nx, ny = int(nx_str), int(ny_str)
    pro_path = cantilever_pro_path_for(nx, ny)
    solver = legacy_nonlinear_solver
    n_elems = nx * ny
    n_dofs = 2 * ((ny + 1) * (2 * nx + 1) + ny * (nx + 1))
  elif name == "riks-fan":
    n_rays = int(size)
    pro_path = fan_pro_path_for(n_rays)
    solver = legacy_riks_solver
    n_elems = n_rays
    n_dofs = 2 * (n_rays + 1)
  else:
    msg = f"Unknown family case {spec!r}"
    raise SystemExit(msg)
  if not pro_path.exists():
    msg = f"generated case missing: {pro_path} (run the gates/warm phase first)"
    raise SystemExit(msg)
  t0 = time.perf_counter()
  props, globdat = legacy_load(pro_path)
  load_ms = (time.perf_counter() - t0) * 1e3
  t0 = time.perf_counter()
  solver(props, globdat, pro_path)
  solve_ms = (time.perf_counter() - t0) * 1e3
  return {"load_ms": load_ms, "solve_ms": solve_ms}, n_elems, n_dofs


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--case",
    required=True,
    help="q8patch:<n>:<material>, skim:<name>, or family:<name>:<size>",
  )
  parser.add_argument("--side", choices=("v3", "legacy"), required=True)
  args = parser.parse_args()

  t_import_start = time.perf_counter()
  import pyfem  # noqa: F401

  if args.side == "v3":
    import pyfem.v3  # noqa: F401
  import_ms = (time.perf_counter() - t_import_start) * 1e3

  kind, _, rest = args.case.partition(":")
  if kind == "q8patch":
    n_str, _, material_type = rest.partition(":")
    nx = int(n_str)
    timings = (
      _run_v3_q8(nx, material_type)
      if args.side == "v3"
      else _run_legacy_q8(nx, material_type)
    )
    n_elems = nx * nx
    n_dofs = 2 * ((2 * nx + 1) * (2 * nx + 1) - n_elems)
  elif kind == "skim":
    timings = _run_v3_skim(rest) if args.side == "v3" else _run_legacy_skim(rest)
    n_elems, n_dofs = 0, 0
  elif kind == "family":
    timings, n_elems, n_dofs = (
      _run_v3_family(rest) if args.side == "v3" else _run_legacy_family(rest)
    )
  else:
    msg = f"Unknown case kind {kind!r}"
    raise SystemExit(msg)

  payload = {
    "case": args.case,
    "side": args.side,
    "import_ms": import_ms,
    "peak_rss_mb": rss_mb(),
    **timings,
    "n_elems": n_elems,
    "n_dofs": n_dofs,
  }
  print(json.dumps(payload))


if __name__ == "__main__":
  main()
