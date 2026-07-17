# SPDX-License-Identifier: MIT

"""Programmatic Tet4 mesh assemble + solve smoke test."""

from __future__ import annotations

import sys

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from pyfem.v3._prototype_assembly import assemble_linear_system
from pyfem.v3.mesh import build_dof_map
from pyfem.v3.pack import pack_problem
from pyfem.v3.solver.context import prepare_linear_solve
from pyfem.v3.types import LoadedProblem, Mesh, PrescribedDof


def _unit_tet_mesh() -> tuple[Mesh, tuple[PrescribedDof, ...]]:
  coords = np.array(
    [
      [0.0, 0.0, 0.0],
      [1.0, 0.0, 0.0],
      [0.0, 1.0, 0.0],
      [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
  )
  conn = np.array([[0, 1, 2, 3]], dtype=np.int32)
  node_ids = np.arange(4, dtype=np.int32)
  mesh = Mesh(
    coords=coords,
    conn=conn,
    node_ids=node_ids,
    elem_group_id=np.zeros(1, dtype=np.int32),
    node_id_to_index={int(n): int(n) for n in node_ids},
  )
  constraints = (
    PrescribedDof(node_id=0, dof_type="u", value=0.0),
    PrescribedDof(node_id=0, dof_type="v", value=0.0),
    PrescribedDof(node_id=0, dof_type="w", value=0.0),
    PrescribedDof(node_id=1, dof_type="u", value=1.0e-4),
    PrescribedDof(node_id=1, dof_type="v", value=0.0),
    PrescribedDof(node_id=1, dof_type="w", value=0.0),
    PrescribedDof(node_id=2, dof_type="u", value=0.0),
    PrescribedDof(node_id=2, dof_type="v", value=1.0e-4),
    PrescribedDof(node_id=2, dof_type="w", value=0.0),
    PrescribedDof(node_id=3, dof_type="u", value=0.0),
    PrescribedDof(node_id=3, dof_type="v", value=0.0),
    PrescribedDof(node_id=3, dof_type="w", value=1.0e-4),
  )
  return mesh, constraints


def test_tet4_programmatic_assemble_and_solve() -> None:
  mesh, constraints = _unit_tet_mesh()
  dof_map = build_dof_map(mesh)
  problem = pack_problem(
    mesh,
    dof_map,
    1.0e6,
    0.25,
    constraints,
    material_type="Isotropic",
  )
  system = assemble_linear_system(
    problem,
    element_type="SmallStrainContinuum",
    material_type="Isotropic",
  )
  assert system.stiffness.shape[0] == problem.n_dofs
  assert np.all(system.stiffness.diagonal() > 0.0)

  loaded = LoadedProblem(
    problem=problem,
    name="unit_tet4",
    element_type="SmallStrainContinuum",
    material_type="Isotropic",
    solver_type="LinearSolver",
    element_group="ContElem",
    mesh_path=None,
  )
  state = prepare_linear_solve(loaded).solve()
  assert state.shape[0] == problem.n_dofs
  assert state[dof_map.dof_index(3, "w")] == pytest.approx(1.0e-4)
