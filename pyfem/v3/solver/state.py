"""Solver state helpers for nonlinear static analysis."""

from __future__ import annotations

import numpy as np

from pyfem.v3.solver.constraints import (
  PrescribedConstraints,
  apply_prescribed_to_state,
  build_prescribed_constraints,
)
from pyfem.v3.types import F64, ProblemDefinition, SolverState


def initial_solver_state(problem: ProblemDefinition) -> SolverState:
  """Return a zero displacement state with prescribed DOFs applied."""
  constraints = build_prescribed_constraints(problem)
  return solver_state_zeros(problem.n_dofs, constraints=constraints)


def solver_state_zeros(
  n_dofs: int,
  *,
  constraints: PrescribedConstraints | None = None,
) -> SolverState:
  """Allocate zero ``state`` and ``state_increment`` vectors."""
  state = np.zeros(n_dofs, dtype=np.float64)
  state_increment = np.zeros(n_dofs, dtype=np.float64)
  if constraints is not None:
    apply_prescribed_to_state(state, constraints)
  return SolverState(state=state, state_increment=state_increment)


def solver_state_from_displacement(
  displacement: F64,
  *,
  constraints: PrescribedConstraints | None = None,
) -> SolverState:
  """Wrap a displacement vector as :class:`SolverState`."""
  state = np.ascontiguousarray(displacement, dtype=np.float64).copy()
  state_increment = np.zeros(state.shape[0], dtype=np.float64)
  if constraints is not None:
    apply_prescribed_to_state(state, constraints)
  return SolverState(state=state, state_increment=state_increment)
