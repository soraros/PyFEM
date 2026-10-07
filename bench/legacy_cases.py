"""Legacy-side cases and drivers: generated ``.dat``/``.pro`` files and solver runs.

The generated uniform Q8 patches reuse :func:`build_uniform_q8_patch`, so the
legacy mesh is identical to the v3 one by construction (``%.17g`` coordinates,
same node numbering, same prescribed boundary field). Legacy drivers mirror
``test/v3/_legacy_parity.py`` — including the solver-settings overrides, which
are essential (legacy solvers do not read those fields natively) and come from
the numba-free :mod:`bench.legacy_settings` replica, so importing this module
never pulls numba into legacy benchmark processes.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stdout
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix

from bench.common import GENERATED_DIR
from bench.legacy_settings import nonlinear_settings, riks_settings
from pyfem.fem.Assembly import (
  assembleExternalForce,
  assembleTangentStiffness,
)
from pyfem.io.InputReader import InputRead
from pyfem.solvers.LinearSolver import LinearSolver
from pyfem.solvers.NonlinearSolver import NonlinearSolver
from pyfem.solvers.RiksSolver import RiksSolver
from pyfem.util.dataStructures import GlobalData, Properties

SILENCE_LOG_LEVEL = logging.CRITICAL


def silence_legacy_logging() -> None:
  """Mute the legacy root logger for the whole bench process."""
  logging.disable(SILENCE_LOG_LEVEL)


@contextmanager
def quiet_legacy() -> Iterator[None]:
  """
  Suppress legacy solver stdout chatter (e.g. the stray ``BaseModule`` print).

  Timing is unaffected; only the solver's own banner output is swallowed.
  """
  with redirect_stdout(io.StringIO()):
    yield


def pro_path_for(nx: int, ny: int, material_type: str) -> Path:
  """Location of the generated legacy ``.pro`` for a uniform Q8 patch."""
  tag = material_type.lower()
  return GENERATED_DIR / f"q8patch_{nx}x{ny}_{tag}.pro"


def write_legacy_q8_patch(
  nx: int,
  ny: int,
  *,
  youngs_modulus: float = 1.0e6,
  poisson_ratio: float = 0.25,
  material_type: str = "PlaneStress",
) -> Path:
  """
  Write legacy ``.dat``/``.pro`` for the uniform Q8 patch and return the pro path.

  Sections follow ``examples/ch02/PatchTest8.dat``: nodes, one ``ContElem``
  group in serendipity order, and the patch displacement field prescribed on
  the boundary nodes. The mesh-builder import is lazy so that legacy cold
  processes (which only read the generated files) stay numba-free.
  """
  from pyfem.v3.mesh.refined_patch import build_uniform_q8_patch

  mesh, constraints = build_uniform_q8_patch(nx, ny)
  pro_path = pro_path_for(nx, ny, material_type)
  dat_path = pro_path.with_suffix(".dat")
  pro_path.parent.mkdir(parents=True, exist_ok=True)

  node_lines = [
    f"  {node_id} {x:.17g} {y:.17g};" for node_id, (x, y) in enumerate(mesh.coords)
  ]
  elem_lines = [
    f'  {elem_id + 1} "ContElem" {" ".join(str(int(n)) for n in conn_row)};'
    for elem_id, conn_row in enumerate(mesh.conn)
  ]
  cons_lines = [
    f"  {item.dof_type}[{item.node_id}] = {item.value:.17g};" for item in constraints
  ]
  dat_path.write_text(
    "<Nodes>\n"
    + "\n".join(node_lines)
    + "\n</Nodes>\n\n<Elements>\n"
    + "\n".join(elem_lines)
    + "\n</Elements>\n\n<NodeConstraints>\n"
    + "\n".join(cons_lines)
    + "\n</NodeConstraints>\n\n<ExternalForces>\n\n</ExternalForces>\n",
    encoding="utf-8",
  )
  pro_path.write_text(
    f'input = "{dat_path.name}";\n\n'
    "ContElem =\n{\n"
    '  type = "SmallStrainContinuum";\n\n'
    "  material =\n  {\n"
    f'    type = "{material_type}";\n'
    f"    E    = {youngs_modulus:.6g};\n"
    f"    nu   = {poisson_ratio};\n"
    "  };\n};\n\n"
    "solver =\n{\n"
    '  type = "LinearSolver";\n};\n',
    encoding="utf-8",
  )
  return pro_path


def legacy_load(pro_path: Path) -> tuple[Properties, GlobalData]:
  """Legacy load path: ``InputRead`` (mesh, element, DOF, constraint parsing)."""
  return InputRead(str(pro_path))


def legacy_linear_state(pro_path: Path) -> np.ndarray:
  """Canonical legacy reference: full ``LinearSolver.run`` (three element loops)."""
  props, globdat = legacy_load(pro_path)
  with quiet_legacy():
    LinearSolver(props, globdat).run(props, globdat)
  return np.asarray(globdat.state)


def legacy_linear_state_fast(props: Properties, globdat: GlobalData) -> np.ndarray:
  """
  Single-assembly legacy linear state for large gate sizes.

  Identical math to ``LinearSolver.run`` for the generated uniform patches
  (no external loads, so ``fext`` is zero); skips the internal-force and
  commit element loops that do not change the solved state. Verified against
  the canonical path by :func:`bench.gates.check_fast_reference_selftest`.
  """
  stiff, _ = assembleTangentStiffness(props, globdat)
  fext = np.zeros(len(globdat.dofs), dtype=float)
  return np.asarray(globdat.dofs.solve(stiff, fext))


def legacy_assembly(props: Properties, globdat: GlobalData) -> coo_matrix:
  """One legacy tangent assembly (the timed legacy 'assemble' metric)."""
  matrix, _ = assembleTangentStiffness(props, globdat)
  return matrix


def legacy_solve(
  props: Properties, globdat: GlobalData, matrix: coo_matrix
) -> np.ndarray:
  """One legacy constrained solve on an assembled matrix (SuperLU single-shot)."""
  fext = np.zeros(len(globdat.dofs), dtype=float)
  return np.asarray(globdat.dofs.solve(matrix, fext))


def legacy_external_load(props: Properties, globdat: GlobalData) -> np.ndarray:
  """Legacy external load vector (skim cases with nodal loads)."""
  return np.asarray(assembleExternalForce(props, globdat))


def legacy_nonlinear_solver(
  props: Properties, globdat: GlobalData, pro_path: Path
) -> np.ndarray:
  """Legacy ``NonlinearSolver`` loop on an already-loaded case (no load cost)."""
  with quiet_legacy():
    solver = NonlinearSolver(props, globdat)
  settings = nonlinear_settings(pro_path.read_text(encoding="utf-8"))
  if settings is not None:
    solver.tol = settings.tol
    solver.iterMax = settings.iter_max
    solver.maxCycle = settings.max_cycle
    solver.dtime = settings.dtime
    if settings.load_table is not None:
      solver.loadTable = settings.load_table
  while globdat.active:
    solver.run(props, globdat)
  return np.asarray(globdat.state)


def legacy_nonlinear_state(pro_path: Path) -> np.ndarray:
  """Legacy ``NonlinearSolver`` driven to completion (mirrors the parity tests)."""
  props, globdat = legacy_load(pro_path)
  return legacy_nonlinear_solver(props, globdat, pro_path)


def legacy_riks_solver(
  props: Properties, globdat: GlobalData, pro_path: Path
) -> np.ndarray:
  """Legacy ``RiksSolver`` loop on an already-loaded case (no load cost)."""
  with quiet_legacy():
    solver = RiksSolver(props, globdat)
  settings = riks_settings(pro_path.read_text(encoding="utf-8"))
  if settings is not None:
    solver.tol = settings.tol
    solver.iterMax = settings.iter_max
    solver.optiter = settings.opt_iter
    solver.fixedStep = settings.fixed_step
    solver.maxLam = settings.max_lam
    solver.maxFactor = settings.max_factor
  while globdat.active:
    solver.run(props, globdat)
  return np.asarray(globdat.state)


def legacy_riks_state(pro_path: Path) -> np.ndarray:
  """Legacy ``RiksSolver`` driven to completion (mirrors the parity tests)."""
  props, globdat = legacy_load(pro_path)
  return legacy_riks_solver(props, globdat, pro_path)


def cantilever_pro_path_for(nx: int, ny: int) -> Path:
  """Location of the generated legacy ``.pro`` for a refined cantilever8."""
  return GENERATED_DIR / f"tl_cantilever_{nx}x{ny}.pro"


def write_legacy_cantilever(nx: int, ny: int) -> Path:
  """
  Write legacy ``.dat``/``.pro`` for the refined cantilever8 and return the pro path.

  The shipped skim's 8x1 serendipity-quad8 FiniteStrainContinuum strip
  refined to ``nx x ny`` cells over the same 8.0 x 0.5 geometry: clamped
  left edge, 0.01 tip load at the top-right corner, PlaneStress E=100
  nu=0.3, NonlinearSolver fixedStep maxCycle=20. Both sides read the
  identical files (M3's same-files comparability). Node rows and the
  corner/mid interleaved counterclockwise connectivity follow the book
  deck's convention; pure-stdlib construction keeps legacy cold processes
  numba-free.
  """
  width, height = 8.0 / nx, 0.5 / ny
  node_id: dict[tuple[str, int, int], int] = {}
  coords: list[tuple[float, float]] = []
  for j in range(ny + 1):
    for i in range(2 * nx + 1):
      node_id[("f", j, i)] = len(coords)
      coords.append((0.5 * width * i, height * j))
    if j < ny:
      for i in range(nx + 1):
        node_id[("m", j, i)] = len(coords)
        coords.append((width * i, height * (j + 0.5)))

  conn: list[list[int]] = []
  for j in range(ny):
    for i in range(nx):
      conn.append(
        [
          node_id[("f", j, 2 * i)],
          node_id[("f", j, 2 * i + 1)],
          node_id[("f", j, 2 * i + 2)],
          node_id[("m", j, i + 1)],
          node_id[("f", j + 1, 2 * i + 2)],
          node_id[("f", j + 1, 2 * i + 1)],
          node_id[("f", j + 1, 2 * i)],
          node_id[("m", j, i)],
        ]
      )

  clamped = [node_id[("f", j, 0)] for j in range(ny + 1)] + [
    node_id[("m", j, 0)] for j in range(ny)
  ]
  tip = node_id[("f", ny, 2 * nx)]

  pro_path = cantilever_pro_path_for(nx, ny)
  dat_path = pro_path.with_suffix(".dat")
  pro_path.parent.mkdir(parents=True, exist_ok=True)

  node_lines = [f"  {nid} {x:.17g} {y:.17g};" for nid, (x, y) in enumerate(coords)]
  elem_lines = [
    f'  {elem_id + 1} "ContElem" {" ".join(str(n) for n in conn_row)};'
    for elem_id, conn_row in enumerate(conn)
  ]
  cons_lines = [f"  {dof}[{nid}] = 0.0;" for nid in clamped for dof in ("u", "v")]
  dat_path.write_text(
    "<Nodes>\n"
    + "\n".join(node_lines)
    + "\n</Nodes>\n\n<Elements>\n"
    + "\n".join(elem_lines)
    + "\n</Elements>\n\n<NodeConstraints>\n"
    + "\n".join(cons_lines)
    + "\n</NodeConstraints>\n\n<ExternalForces>\n"
    + f"  v[{tip}] = 0.01;\n"
    + "</ExternalForces>\n",
    encoding="utf-8",
  )
  pro_path.write_text(
    f'input = "{dat_path.name}";\n\n'
    "ContElem =\n{\n"
    '  type = "FiniteStrainContinuum";\n\n'
    "  material =\n  {\n"
    '    type = "PlaneStress";\n'
    "    E    = 100.0;\n"
    "    nu   = 0.3;\n"
    "  };\n};\n\n"
    "solver =\n{\n"
    '  type = "NonlinearSolver";\n\n'
    "  fixedStep = true;\n"
    "  maxCycle  = 20;\n};\n",
    encoding="utf-8",
  )
  return pro_path


def fan_pro_path_for(n_rays: int) -> Path:
  """Location of the generated legacy ``.pro`` for a truss-only Riks fan."""
  return GENERATED_DIR / f"riks_fan_{n_rays}.pro"


def write_legacy_truss_fan(n_rays: int) -> Path:
  """
  Write legacy ``.dat``/``.pro`` for the truss-only shallow-truss fan.

  ``n_rays`` Truss members (E=5e6, Area=1.0) from fully constrained base
  nodes on [-10, 10] (span 20) to the apex (0, 0.5) loaded by v = -100;
  RiksSolver tol=1e-10 iterMax=25 fixedStep maxLam=10 — the landed M33
  oracle's deck and settings (``test/v3/test_v3_driver_parity.py``), which
  ``n_rays=2`` reproduces exactly. The fan drops the skim's spring on both
  sides like that oracle. ``bench.family_v3`` mirrors these constants on
  the v3 side; the parity gate protects the correspondence.
  """
  coords = [(-10.0 + i * 20.0 / max(n_rays - 1, 1), 0.0) for i in range(n_rays)] + [
    (0.0, 0.5)
  ]
  apex = n_rays

  pro_path = fan_pro_path_for(n_rays)
  dat_path = pro_path.with_suffix(".dat")
  pro_path.parent.mkdir(parents=True, exist_ok=True)

  node_lines = [f"  {nid} {x:.17g} {y:.17g};" for nid, (x, y) in enumerate(coords)]
  elem_lines = [f'  {i + 1} "TrussElem" {i} {apex};' for i in range(n_rays)]
  cons_lines = [f"  {dof}[{nid}] = 0.0;" for nid in range(n_rays) for dof in ("u", "v")]
  dat_path.write_text(
    "<Nodes>\n"
    + "\n".join(node_lines)
    + "\n</Nodes>\n\n<Elements>\n"
    + "\n".join(elem_lines)
    + "\n</Elements>\n\n<NodeConstraints>\n"
    + "\n".join(cons_lines)
    + "\n</NodeConstraints>\n\n<ExternalForces>\n"
    + f"  v[{apex}] = -100.0;\n"
    + "</ExternalForces>\n",
    encoding="utf-8",
  )
  pro_path.write_text(
    f'input = "{dat_path.name}";\n\n'
    "TrussElem =\n{\n"
    '  type = "Truss";\n'
    "  E    = 5e6;\n"
    "  Area = 1.0;\n};\n\n"
    "solver =\n{\n"
    '  type = "RiksSolver";\n\n'
    "  tol       = 1.0e-10;\n"
    "  iterMax   = 25;\n"
    "  fixedStep = true;\n"
    "  maxLam    = 10.0;\n};\n",
    encoding="utf-8",
  )
  return pro_path


def legacy_riks_cycles(
  props: Properties, globdat: GlobalData, pro_path: Path
) -> list[tuple[float, int, np.ndarray]]:
  """
  Legacy ``RiksSolver`` run cycle by cycle, capturing committed points.

  Returns one ``(lam, correction_count, state)`` triple per committed cycle
  (``solverStatus.iiter`` is the legacy per-cycle Newton correction count) —
  the legacy half of the landed M33 per-cycle parity oracle, on an
  already-loaded case (no load cost).
  """
  with quiet_legacy():
    solver = RiksSolver(props, globdat)
  settings = riks_settings(pro_path.read_text(encoding="utf-8"))
  if settings is not None:
    solver.tol = settings.tol
    solver.iterMax = settings.iter_max
    solver.optiter = settings.opt_iter
    solver.fixedStep = settings.fixed_step
    solver.maxLam = settings.max_lam
    solver.maxFactor = settings.max_factor
  cycles: list[tuple[float, int, np.ndarray]] = []
  while globdat.active:
    solver.run(props, globdat)
    cycles.append(
      (
        float(globdat.lam),
        globdat.solverStatus.iiter,
        np.asarray(globdat.state).copy(),
      )
    )
  return cycles


def legacy_tangent_at_state(
  pro_path: Path, state: np.ndarray
) -> tuple[coo_matrix, np.ndarray]:
  """Legacy tangent stiffness and internal force at a given displacement."""
  props, globdat = legacy_load(pro_path)
  globdat.state[:] = state
  matrix, internal_force = assembleTangentStiffness(props, globdat)
  return matrix.tocoo(), np.asarray(internal_force)
