"""Linear static solver with prescribed displacements."""

from __future__ import annotations

from pyfem.v3.registry import resolve_solver_type
from pyfem.v3.solver.context import prepare_cached_linear
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
    return prepare_cached_linear(problem).solve()

  if not all((element_type, material_type, solver_type)):
    msg = "element_type, material_type, and solver_type are required"
    raise ValueError(msg)
  resolve_solver_type(solver_type)
  return prepare_cached_linear(
    problem,
    element_type=element_type,
    material_type=material_type,
    solver_type=solver_type,
  ).solve()
