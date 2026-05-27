"""Linear static solver with prescribed displacements."""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import spsolve

from pyfem.v3.assembly import assemble_linear_system, assemble_loaded
from pyfem.v3.registry import resolve_solver_type
from pyfem.v3.solver.constraints import build_constrainer
from pyfem.v3.types import F64, LoadedProblem, ProblemDefinition


def solve_linear(
  problem: ProblemDefinition | LoadedProblem,
  *,
  element_type: str | None = None,
  material_type: str | None = None,
  solver_type: str | None = None,
) -> F64:
  """
  Solve ``K u = f`` with displacement boundary conditions.

  Accepts a :class:`LoadedProblem` or a jitable :class:`ProblemDefinition`
  plus registry type strings.
  """
  if isinstance(problem, LoadedProblem):
    resolve_solver_type(problem.solver_type)
    system = assemble_loaded(problem)
    definition = problem.problem
  else:
    if not all((element_type, material_type, solver_type)):
      msg = "element_type, material_type, and solver_type are required"
      raise ValueError(msg)
    resolve_solver_type(solver_type)
    system = assemble_linear_system(
      problem,
      element_type=element_type,
      material_type=material_type,
    )
    definition = problem

  cons = build_constrainer(definition)
  k = system.stiffness.tocsr()
  b = system.load
  n = system.n_dofs

  a = np.zeros(n, dtype=np.float64)
  cons.addConstrainedValues(a)

  k_red = cons.C.T @ (k @ cons.C)
  b_red = cons.C.T @ (b - k @ a)
  x_red = spsolve(k_red, b_red)
  state = cons.C @ x_red
  cons.addConstrainedValues(state)
  return np.asarray(state, dtype=np.float64)
