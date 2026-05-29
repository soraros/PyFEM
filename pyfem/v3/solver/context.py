"""Cached factorization for repeated linear solves."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import factorized

from pyfem.v3.assembly import assemble_linear_system, assemble_loaded
from pyfem.v3.registry import resolve_solver_type
from pyfem.v3.solver.constraints import (
  PrescribedConstraints,
  apply_prescribed_to_state,
  build_prescribed_constraints,
)
from pyfem.v3.types import F64, LinearSystem, LoadedProblem, ProblemDefinition


def factorized_reduced_solve(
  constraints: PrescribedConstraints,
  k_csr: csr_matrix,
) -> Callable[[F64], F64]:
  """Factorize the reduced stiffness ``C.T @ K @ C`` for repeated back-solves."""
  k_red = constraints.C.T @ (k_csr @ constraints.C)
  return factorized(k_red.tocsr())


@dataclass
class LinearSolutionContext:
  """
  Cached reduced-system factorization for repeated solves.

  Valid only while mesh, constraints, and stiffness sparsity pattern are fixed.
  """

  factorized_solve: Callable[[F64], F64]
  constraints: PrescribedConstraints
  k_csr: csr_matrix
  n_dofs: int
  base_load: F64

  def solve(self, load: F64 | None = None) -> F64:
    """Solve with optional replacement load vector."""
    rhs = self.base_load if load is None else load
    a = np.zeros(self.n_dofs, dtype=np.float64)
    apply_prescribed_to_state(a, self.constraints)
    b_red = self.constraints.C.T @ (rhs - self.k_csr @ a)
    x_red = self.factorized_solve(b_red)
    state = self.constraints.C @ x_red
    apply_prescribed_to_state(state, self.constraints)
    return np.asarray(state, dtype=np.float64)

  @classmethod
  def from_loaded(
    cls,
    loaded: LoadedProblem,
    *,
    chunk_size: int | None = None,
  ) -> LinearSolutionContext:
    resolve_solver_type(loaded.solver_type)
    system = assemble_loaded(loaded, chunk_size=chunk_size)
    return cls.from_system(loaded.problem, system)

  @classmethod
  def from_system(
    cls,
    problem: ProblemDefinition,
    system: LinearSystem,
  ) -> LinearSolutionContext:
    constraints = build_prescribed_constraints(problem)
    k_csr = system.stiffness.tocsr()
    solve_red = factorized_reduced_solve(constraints, k_csr)
    return cls(
      factorized_solve=solve_red,
      constraints=constraints,
      k_csr=k_csr,
      n_dofs=system.n_dofs,
      base_load=np.ascontiguousarray(system.load, dtype=np.float64),
    )


def prepare_linear_solve(
  problem: ProblemDefinition | LoadedProblem,
  *,
  element_type: str | None = None,
  material_type: str | None = None,
  solver_type: str | None = None,
  chunk_size: int | None = None,
) -> LinearSolutionContext:
  """Assemble and factorize once for repeated solves."""
  if isinstance(problem, LoadedProblem):
    return LinearSolutionContext.from_loaded(problem, chunk_size=chunk_size)

  if not all((element_type, material_type, solver_type)):
    msg = "element_type, material_type, and solver_type are required"
    raise ValueError(msg)
  resolve_solver_type(solver_type)
  system = assemble_linear_system(
    problem,
    element_type=element_type,
    material_type=material_type,
    chunk_size=chunk_size,
  )
  return LinearSolutionContext.from_system(problem, system)
