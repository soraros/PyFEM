"""Prescribed-displacement constraints."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix as CSRMatrix

from pyfem.v3.types import F64, ProblemDefinition


@dataclass(frozen=True)
class PrescribedConstraints:
  """Prescribed DOFs mapped to a reduced free-DOF system."""

  C: CSRMatrix
  prescribed: F64


def build_prescribed_constraints(problem: ProblemDefinition) -> PrescribedConstraints:
  """Build constraint matrix and prescribed values from packed arrays."""
  n_dofs = problem.n_dofs
  constrained = set(problem.constraint_dof.tolist())
  free_dofs = [i for i in range(n_dofs) if i not in constrained]
  n_free = len(free_dofs)

  row = np.empty(n_free, dtype=np.int32)
  col = np.arange(n_free, dtype=np.int32)
  val = np.ones(n_free, dtype=np.float64)
  for j, dof in enumerate(free_dofs):
    row[j] = dof

  c = coo_matrix((val, (row, col)), shape=(n_dofs, n_free)).tocsr()

  prescribed = np.zeros(n_dofs, dtype=np.float64)
  for dof_id, value in zip(problem.constraint_dof, problem.constraint_val):
    prescribed[int(dof_id)] = float(value)

  return PrescribedConstraints(C=c, prescribed=prescribed)


def apply_prescribed_to_state(state: F64, constraints: PrescribedConstraints) -> None:
  """Add prescribed displacement values into ``state`` (in-place)."""
  state += constraints.prescribed
