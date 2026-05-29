# SPDX-License-Identifier: MIT

"""Uniform Q8 patch mesh generator."""

from __future__ import annotations

import sys

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from pyfem.v3.assembly import assemble_linear_system
from pyfem.v3.mesh import build_dof_map
from pyfem.v3.mesh.refined_patch import (
  PATCH_HEIGHT,
  PATCH_WIDTH,
  build_uniform_q8_loaded,
  build_uniform_q8_patch,
  patch_displacement,
)
from pyfem.v3.pack import pack_problem
from pyfem.v3.solver.context import prepare_linear_solve


def test_single_element_patch_counts() -> None:
  mesh, constraints = build_uniform_q8_patch(1, 1)
  assert mesh.n_nodes == 8
  assert mesh.n_elems == 1
  assert mesh.conn.shape == (1, 8)
  assert len(constraints) == 16


def test_four_by_four_element_counts() -> None:
  mesh, _constraints = build_uniform_q8_patch(4, 4)
  assert mesh.n_elems == 16
  assert mesh.n_nodes == 65


def test_boundary_prescribed_matches_patch_field() -> None:
  mesh, constraints = build_uniform_q8_patch(2, 2)
  for item in constraints:
    x, y = mesh.coords[mesh.node_id_to_index[item.node_id]]
    u, v = patch_displacement(float(x), float(y))
    expected = u if item.dof_type == "u" else v
    assert item.value == pytest.approx(expected)


def test_corner_node_coords() -> None:
  mesh, _ = build_uniform_q8_patch(1, 1)
  bl_idx = int(np.where(np.all(mesh.coords == [0.0, 0.0], axis=1))[0][0])
  tr_idx = int(
    np.where(
      np.isclose(mesh.coords[:, 0], PATCH_WIDTH)
      & np.isclose(mesh.coords[:, 1], PATCH_HEIGHT),
    )[0][0],
  )
  np.testing.assert_allclose(mesh.coords[bl_idx], [0.0, 0.0])
  np.testing.assert_allclose(
    mesh.coords[tr_idx],
    [PATCH_WIDTH, PATCH_HEIGHT],
    rtol=0,
    atol=1e-15,
  )


def test_pack_produces_square_system() -> None:
  mesh, constraints = build_uniform_q8_patch(2, 2)
  dof_map = build_dof_map(mesh)
  problem = pack_problem(mesh, dof_map, 1.0e6, 0.25, constraints)
  n_free = problem.n_dofs - len(problem.constraint_dof)
  assert n_free > 0
  assert problem.n_dofs == 2 * mesh.n_nodes


def test_no_orphan_nodes() -> None:
  mesh, _ = build_uniform_q8_patch(4, 4)
  used = set(mesh.conn.ravel())
  assert used == set(range(mesh.n_nodes))


def test_stiffness_diagonal_nonzero() -> None:
  mesh, _ = build_uniform_q8_patch(2, 2)
  dof_map = build_dof_map(mesh)
  problem = pack_problem(mesh, dof_map, 1.0e6, 0.25, ())
  system = assemble_linear_system(
    problem,
    element_type="SmallStrainContinuum",
    material_type="PlaneStress",
  )
  k_diag = system.stiffness.diagonal()
  assert k_diag.shape[0] == problem.n_dofs
  assert np.all(k_diag > 0.0)


def test_factorized_solve_2x2() -> None:
  loaded = build_uniform_q8_loaded(2, 2)
  ctx = prepare_linear_solve(loaded)
  state = ctx.solve()
  assert state.shape[0] == loaded.problem.n_dofs


def test_plane_strain_stiffness_diagonal_nonzero() -> None:
  loaded = build_uniform_q8_loaded(2, 2, material_type="PlaneStrain")
  system = assemble_linear_system(
    loaded.problem,
    element_type="SmallStrainContinuum",
    material_type="PlaneStrain",
  )
  k_diag = system.stiffness.diagonal()
  assert k_diag.shape[0] == loaded.problem.n_dofs
  assert np.all(k_diag > 0.0)


def test_plane_strain_factorized_solve_2x2() -> None:
  loaded = build_uniform_q8_loaded(2, 2, material_type="PlaneStrain")
  ctx = prepare_linear_solve(loaded)
  state = ctx.solve()
  assert state.shape[0] == loaded.problem.n_dofs
  assert np.linalg.norm(state) > 0.0


def test_plane_strain_constitutive_differs_from_plane_stress() -> None:
  stress = build_uniform_q8_loaded(2, 2, material_type="PlaneStress")
  strain = build_uniform_q8_loaded(2, 2, material_type="PlaneStrain")
  assert not np.allclose(stress.problem.constitutive, strain.problem.constitutive)
