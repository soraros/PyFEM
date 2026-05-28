"""Prescribed displacements and MPC tie constraints."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix as CSRMatrix

from pyfem.v3.types import F64, ProblemDefinition


@dataclass(frozen=True)
class PrescribedConstraints:
  """Reduced free-DOF system with prescribed offsets on constrained DOFs."""

  C: CSRMatrix
  prescribed: F64


def build_prescribed_constraints(problem: ProblemDefinition) -> PrescribedConstraints:
  """Build constraint matrix and prescribed values from packed arrays."""
  n_dofs = problem.n_dofs
  prescribed = np.zeros(n_dofs, dtype=np.float64)
  for dof_id, value in zip(problem.constraint_dof, problem.constraint_val):
    prescribed[int(dof_id)] = float(value)

  prescribed_dofs = set(problem.constraint_dof.tolist())
  slave_dofs = set(problem.mpc_slave_dof.tolist())
  constrained = prescribed_dofs | slave_dofs
  free_dofs = [i for i in range(n_dofs) if i not in constrained]
  n_free = len(free_dofs)

  dof_to_col = {dof: col for col, dof in enumerate(free_dofs)}

  row = np.empty(n_free, dtype=np.int32)
  col = np.arange(n_free, dtype=np.int32)
  val = np.ones(n_free, dtype=np.float64)
  for j, dof in enumerate(free_dofs):
    row[j] = dof

  mpc_rows: list[int] = []
  mpc_cols: list[int] = []
  mpc_vals: list[float] = []
  for slave, master, factor, offset in zip(
    problem.mpc_slave_dof,
    problem.mpc_master_dof,
    problem.mpc_factor,
    problem.mpc_offset,
  ):
    slave_id = int(slave)
    master_id = int(master)
    if master_id not in dof_to_col:
      msg = f"MPC master DOF {master_id} is not a free DOF"
      raise ValueError(msg)
    mpc_rows.append(slave_id)
    mpc_cols.append(dof_to_col[master_id])
    mpc_vals.append(float(factor))
    prescribed[slave_id] = float(offset)

  if mpc_rows:
    row = np.concatenate([row, np.asarray(mpc_rows, dtype=np.int32)])
    col = np.concatenate([col, np.asarray(mpc_cols, dtype=np.int32)])
    val = np.concatenate([val, np.asarray(mpc_vals, dtype=np.float64)])

  c = coo_matrix((val, (row, col)), shape=(n_dofs, n_free)).tocsr()
  return PrescribedConstraints(C=c, prescribed=prescribed)


def apply_prescribed_to_state(state: F64, constraints: PrescribedConstraints) -> None:
  """Add prescribed displacement values into ``state`` (in-place)."""
  state += constraints.prescribed
