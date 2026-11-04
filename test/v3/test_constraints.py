# SPDX-License-Identifier: MIT

"""Native v3 prescribed-displacement constraints."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from scipy.sparse import csr_array
from scipy.sparse.linalg import factorized

from pyfem.v3 import ProblemDefinition, load_problem
from pyfem.v3._prototype_assembly import assemble_loaded, canonical_csr
from pyfem.v3.mesh.refined_patch import build_uniform_q8_loaded
from pyfem.v3.solver.constraints import (
  build_prescribed_constraints,
  reduce_prescribed_tangent,
)
from pyfem.v3.solver.context import CachedLinearSystem

ROOT = Path(__file__).resolve().parents[2]
SKIM = ROOT / "skims" / "patch_test8" / "problem.toml"


def test_prescribed_constraints_matrix_shape() -> None:
  problem = load_problem(SKIM).problem
  constraints = build_prescribed_constraints(problem)
  n_constrained = len(problem.constraint_dof) + len(problem.mpc_slave_dof)
  n_free = problem.n_dofs - n_constrained

  assert constraints.C.shape == (problem.n_dofs, n_free)
  assert constraints.prescribed.shape == (problem.n_dofs,)
  assert constraints.C.nnz == n_free + len(problem.mpc_slave_dof)


def test_prescribed_constraints_sparsity_pattern() -> None:
  problem = load_problem(SKIM).problem
  constraints = build_prescribed_constraints(problem)
  constrained = set(problem.constraint_dof.tolist()) | set(
    problem.mpc_slave_dof.tolist(),
  )

  c_dense = constraints.C.toarray()
  for dof in range(problem.n_dofs):
    if dof in constrained:
      if dof in problem.mpc_slave_dof:
        assert np.count_nonzero(c_dense[dof]) == 1
      else:
        assert np.allclose(c_dense[dof], 0.0)
    else:
      assert c_dense[dof].sum() == 1.0
      assert np.count_nonzero(c_dense[dof]) == 1


def test_prescribed_constraints_reconstruct_full_state() -> None:
  problem = load_problem(SKIM).problem
  constraints = build_prescribed_constraints(problem)
  n_free = constraints.C.shape[1]

  x_free = np.arange(n_free, dtype=np.float64)
  full_state = constraints.C @ x_free + constraints.prescribed

  for dof_id, value in zip(problem.constraint_dof, problem.constraint_val):
    assert full_state[int(dof_id)] == pytest.approx(float(value))

  constrained = set(problem.constraint_dof.tolist()) | set(
    problem.mpc_slave_dof.tolist(),
  )
  free_idx = 0
  for dof in range(problem.n_dofs):
    if dof in constrained:
      continue
    assert full_state[dof] == pytest.approx(float(free_idx))
    free_idx += 1


def test_mpc_tie_links_slave_to_free_master() -> None:
  problem = ProblemDefinition(
    coords=np.zeros((2, 2), dtype=np.float64),
    conn=np.zeros((1, 4), dtype=np.int32),
    global_dofs=np.arange(4, dtype=np.int32).reshape(2, 2),
    constitutive=np.eye(3, dtype=np.float64),
    constraint_dof=np.array([0], dtype=np.int32),
    constraint_val=np.array([0.0], dtype=np.float64),
    mpc_slave_dof=np.array([2], dtype=np.int32),
    mpc_master_dof=np.array([1], dtype=np.int32),
    mpc_factor=np.array([1.5], dtype=np.float64),
    mpc_offset=np.array([0.1], dtype=np.float64),
    external_load=np.zeros(4, dtype=np.float64),
    elem_group_id=np.zeros(1, dtype=np.int32),
    group_kind=np.array([0], dtype=np.int32),
    group_props=np.zeros((1, 2), dtype=np.float64),
  )
  constraints = build_prescribed_constraints(problem)
  x_free = np.array([2.0, 3.0], dtype=np.float64)
  full_state = constraints.C @ x_free + constraints.prescribed

  assert full_state[1] == pytest.approx(2.0)
  assert full_state[2] == pytest.approx(0.1 + 1.5 * 2.0)
  assert full_state[3] == pytest.approx(3.0)


def test_prescribed_constraints_compile_sorted_free_dof_selector() -> None:
  problem = load_problem(SKIM).problem
  constraints = build_prescribed_constraints(problem)
  expected = np.setdiff1d(
    np.arange(problem.n_dofs, dtype=np.int32),
    problem.constraint_dof,
  )

  assert constraints.free_dofs is not None
  assert constraints.free_dofs.dtype == np.int32
  np.testing.assert_array_equal(constraints.free_dofs, expected)


class _SelectionOnlyMatrix:
  def __init__(self, matrix: csr_array) -> None:
    self._matrix = matrix

  def __getitem__(self, key: object) -> csr_array:
    return self._matrix[key]

  def __matmul__(self, other: object) -> None:
    raise AssertionError("prescribed-only extraction used a matrix product")


def test_prescribed_reduction_does_not_use_sparse_matrix_products() -> None:
  problem = load_problem(SKIM).problem
  constraints = build_prescribed_constraints(problem)
  diagonal = np.arange(1, problem.n_dofs + 1, dtype=np.float64)
  k_csr = csr_array(np.diag(diagonal))
  expected = (constraints.C.T @ (k_csr @ constraints.C)).tocsr()

  actual = reduce_prescribed_tangent(constraints, _SelectionOnlyMatrix(k_csr))

  np.testing.assert_array_equal(actual.indptr, expected.indptr)
  np.testing.assert_array_equal(actual.indices, expected.indices)
  np.testing.assert_array_equal(actual.data, expected.data)


@pytest.mark.parametrize("nx", [16, 32, 64])
def test_prescribed_q8_reduced_csr_and_solve_match_sparse_prolongation(
  nx: int,
) -> None:
  loaded = build_uniform_q8_loaded(nx, nx)
  problem = loaded.problem
  system = assemble_loaded(loaded)
  k_csr = canonical_csr(system.stiffness)
  constraints = build_prescribed_constraints(problem)
  expected = (constraints.C.T @ (k_csr @ constraints.C)).tocsr()

  actual = reduce_prescribed_tangent(constraints, k_csr)

  assert constraints.free_dofs is not None
  raw = k_csr[constraints.free_dofs, :][:, constraints.free_dofs].tocsr()
  assert np.count_nonzero(raw.data == 0.0) > 0
  assert np.all(actual.data != 0.0)
  assert actual.has_canonical_format
  assert actual.shape == expected.shape
  np.testing.assert_array_equal(actual.indptr, expected.indptr)
  np.testing.assert_array_equal(actual.indices, expected.indices)
  np.testing.assert_array_equal(
    actual.data.view(np.uint64),
    expected.data.view(np.uint64),
  )

  expected_solve = factorized(expected)
  context = CachedLinearSystem.from_system(problem, system)
  b_red = constraints.C.T @ (system.load - k_csr @ constraints.prescribed)
  expected_x = expected_solve(b_red)
  actual_x = context.factorized_solve(b_red)
  np.testing.assert_array_equal(actual_x.view(np.uint64), expected_x.view(np.uint64))

  expected_state = np.asarray(constraints.C @ expected_x) + constraints.prescribed
  actual_state = context.solve()
  np.testing.assert_array_equal(
    actual_state.view(np.uint64),
    expected_state.view(np.uint64),
  )


def test_mpc_reduction_retains_affine_prolongation_not_selector() -> None:
  problem = ProblemDefinition(
    coords=np.zeros((2, 2), dtype=np.float64),
    conn=np.zeros((1, 4), dtype=np.int32),
    global_dofs=np.arange(4, dtype=np.int32).reshape(2, 2),
    constitutive=np.eye(3, dtype=np.float64),
    constraint_dof=np.array([0], dtype=np.int32),
    constraint_val=np.array([0.0], dtype=np.float64),
    mpc_slave_dof=np.array([2], dtype=np.int32),
    mpc_master_dof=np.array([1], dtype=np.int32),
    mpc_factor=np.array([1.5], dtype=np.float64),
    mpc_offset=np.array([0.1], dtype=np.float64),
    external_load=np.zeros(4, dtype=np.float64),
    elem_group_id=np.zeros(1, dtype=np.int32),
    group_kind=np.array([0], dtype=np.int32),
    group_props=np.zeros((1, 2), dtype=np.float64),
  )
  constraints = build_prescribed_constraints(problem)
  dense = np.arange(problem.n_dofs**2, dtype=np.float64).reshape(
    problem.n_dofs,
    problem.n_dofs,
  )
  k_csr = csr_array(dense + dense.T + problem.n_dofs * np.eye(problem.n_dofs))
  expected = (constraints.C.T @ (k_csr @ constraints.C)).tocsr()

  actual = reduce_prescribed_tangent(constraints, k_csr)

  assert constraints.free_dofs is None
  np.testing.assert_array_equal(actual.indptr, expected.indptr)
  np.testing.assert_array_equal(actual.indices, expected.indices)
  np.testing.assert_array_equal(actual.data, expected.data)

  constrained = np.concatenate((problem.constraint_dof, problem.mpc_slave_dof))
  free = np.setdiff1d(np.arange(problem.n_dofs, dtype=np.int32), constrained)
  selector = k_csr[free, :][:, free].tocsr()
  assert not np.allclose(actual.toarray(), selector.toarray())
