# SPDX-License-Identifier: MIT

"""Native v3 prescribed-displacement constraints."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from pyfem.v3 import ProblemDefinition, load_problem
from pyfem.v3.solver.constraints import build_prescribed_constraints

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
  )
  constraints = build_prescribed_constraints(problem)
  x_free = np.array([2.0, 3.0], dtype=np.float64)
  full_state = constraints.C @ x_free + constraints.prescribed

  assert full_state[1] == pytest.approx(2.0)
  assert full_state[2] == pytest.approx(0.1 + 1.5 * 2.0)
  assert full_state[3] == pytest.approx(3.0)
