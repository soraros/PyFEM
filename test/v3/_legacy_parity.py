# SPDX-License-Identifier: MIT

"""Shared helpers for v3 vs legacy parity checks (not pytest-collected)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix

from pyfem.fem.Assembly import assembleExternalForce, assembleTangentStiffness
from pyfem.io.InputReader import InputRead
from pyfem.solvers.LinearSolver import LinearSolver
from pyfem.util.dataStructures import elementData


def legacy_state(pro_path: Path) -> np.ndarray:
  props, globdat = InputRead(str(pro_path))
  LinearSolver(props, globdat).run(props, globdat)
  return np.asarray(globdat.state)


def legacy_stiffness_coo(pro_path: Path) -> coo_matrix:
  props, globdat = InputRead(str(pro_path))
  matrix, _ = assembleTangentStiffness(props, globdat)
  return matrix.tocoo()


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
