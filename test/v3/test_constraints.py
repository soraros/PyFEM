# SPDX-License-Identifier: MIT

"""Native v3 prescribed-displacement constraints."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from pyfem.v3 import load_problem
from pyfem.v3.solver.constraints import build_prescribed_constraints

ROOT = Path(__file__).resolve().parents[2]
SKIM = ROOT / "skims" / "patch_test8" / "problem.toml"


def test_prescribed_constraints_matrix_shape() -> None:
  problem = load_problem(SKIM).problem
  constraints = build_prescribed_constraints(problem)
  n_constrained = len(problem.constraint_dof)
  n_free = problem.n_dofs - n_constrained

  assert constraints.C.shape == (problem.n_dofs, n_free)
  assert constraints.prescribed.shape == (problem.n_dofs,)
  assert constraints.C.nnz == n_free


def test_prescribed_constraints_sparsity_pattern() -> None:
  problem = load_problem(SKIM).problem
  constraints = build_prescribed_constraints(problem)
  constrained = set(problem.constraint_dof.tolist())

  c_dense = constraints.C.toarray()
  for dof in range(problem.n_dofs):
    if dof in constrained:
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

  constrained = set(problem.constraint_dof.tolist())
  free_idx = 0
  for dof in range(problem.n_dofs):
    if dof in constrained:
      continue
    assert full_state[dof] == pytest.approx(float(free_idx))
    free_idx += 1
