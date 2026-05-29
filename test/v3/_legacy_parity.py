# SPDX-License-Identifier: MIT

"""Shared helpers for v3 vs legacy parity checks (not pytest-collected)."""

from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix

from pyfem.fem.Assembly import assembleExternalForce, assembleTangentStiffness
from pyfem.io.InputReader import InputRead
from pyfem.solvers.LinearSolver import LinearSolver
from pyfem.solvers.NonlinearSolver import NonlinearSolver
from pyfem.util.dataStructures import elementData


def load_parity_tolerances(skim_dir: Path) -> tuple[float, float]:
  with (skim_dir / "parity.toml").open("rb") as fh:
    data = tomllib.load(fh)
  return float(data["rtol"]), float(data["atol"])


def legacy_state(pro_path: Path) -> np.ndarray:
  props, globdat = InputRead(str(pro_path))
  LinearSolver(props, globdat).run(props, globdat)
  return np.asarray(globdat.state)


def legacy_nonlinear_state(pro_path: Path) -> np.ndarray:
  """Run legacy ``NonlinearSolver`` until ``globdat.active`` is false."""
  from pyfem.v3.io.solver_pro import parse_nonlinear_solver_settings

  props, globdat = InputRead(str(pro_path))
  solver = NonlinearSolver(props, globdat)
  settings = parse_nonlinear_solver_settings(pro_path.read_text(encoding="utf-8"))
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


def legacy_riks_state(pro_path: Path) -> np.ndarray:
  """Run legacy ``RiksSolver`` until ``globdat.active`` is false."""
  from pyfem.solvers.RiksSolver import RiksSolver
  from pyfem.v3.io.solver_pro import parse_riks_solver_settings

  props, globdat = InputRead(str(pro_path))
  solver = RiksSolver(props, globdat)
  settings = parse_riks_solver_settings(pro_path.read_text(encoding="utf-8"))
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


def legacy_stiffness_coo(pro_path: Path) -> coo_matrix:
  props, globdat = InputRead(str(pro_path))
  matrix, _ = assembleTangentStiffness(props, globdat)
  return matrix.tocoo()


def legacy_tangent_at_state(
  pro_path: Path,
  state: np.ndarray,
) -> tuple[coo_matrix, np.ndarray]:
  """Legacy tangent stiffness and internal force at a given displacement."""
  props, globdat = InputRead(str(pro_path))
  globdat.state[:] = state
  matrix, internal_force = assembleTangentStiffness(props, globdat)
  return matrix.tocoo(), np.asarray(internal_force)


def legacy_external_load(pro_path: Path) -> np.ndarray:
  props, globdat = InputRead(str(pro_path))
  return np.asarray(assembleExternalForce(props, globdat))


def legacy_element_stiffness(
  pro_path: Path,
  coords: np.ndarray,
  *,
  group: str = "ContElem",
) -> np.ndarray:
  """Single-element tangent stiffness via legacy ``SmallStrainContinuum``."""
  props, globdat = InputRead(str(pro_path))
  element = next(iter(globdat.elements.iterElementGroup(group)))
  n_dof = element.dofCount()
  state = np.zeros(n_dof)
  template = elementData(state, state)
  element.globdat = globdat
  template.coords = coords
  template.stiff.fill(0.0)
  template.fint.fill(0.0)
  if hasattr(element, "mat"):
    element.mat.reset()
  element.getTangentStiffness(template)
  return template.stiff.copy()
