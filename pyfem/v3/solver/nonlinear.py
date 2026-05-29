"""Nonlinear static solver (Newton–Raphson with load ramping)."""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import spsolve

from pyfem.v3.assembly import assemble_tangent_loaded
from pyfem.v3.io.load_ramp import load_factor, n_load_steps
from pyfem.v3.registry import (
  resolve_element_type,
  resolve_material_type,
  resolve_solver_type,
)
from pyfem.v3.solver.constraints import (
  PrescribedConstraints,
  build_prescribed_constraints,
)
from pyfem.v3.solver.context import factorized_reduced_solve
from pyfem.v3.solver.tangent_context import TangentAssemblyContext
from pyfem.v3.types import (
  F64,
  LoadedProblem,
  NonlinearSolverSettings,
  ProblemDefinition,
  SolverState,
)

_PRESCRIBED_NORM_TOL = 1.0e-16
_LINEAR_ELEMENT = "SmallStrainContinuum"
_LINEAR_MATERIALS = frozenset({"PlaneStress", "PlaneStrain", "Isotropic"})


def _solver_settings(loaded: LoadedProblem) -> NonlinearSolverSettings:
  if loaded.nonlinear_settings is not None:
    return loaded.nonlinear_settings
  return NonlinearSolverSettings()


def _prescribed_dof_ids(problem: ProblemDefinition) -> np.ndarray:
  if problem.mpc_slave_dof.size:
    return np.unique(
      np.concatenate([problem.constraint_dof, problem.mpc_slave_dof]),
    ).astype(np.int32)
  return problem.constraint_dof


def set_prescribed_displacements(
  state: F64,
  problem: ProblemDefinition,
  constraints: PrescribedConstraints,
  lam: float,
) -> None:
  """Set constrained DOFs to ``lam`` times their reference prescribed values."""
  for dof_id in _prescribed_dof_ids(problem):
    state[int(dof_id)] = lam * constraints.prescribed[int(dof_id)]


def residual_norm(
  f_ext: F64,
  f_int: F64,
  constraints: PrescribedConstraints,
) -> float:
  """L2 residual norm on free DOFs (matches legacy ``DofSpace.norm``)."""
  residual = f_ext - f_int
  r_red = constraints.C.T @ residual
  f_red = constraints.C.T @ f_ext
  norm_r = float(np.linalg.norm(r_red))
  norm_f = float(np.linalg.norm(f_red))
  if norm_f < _PRESCRIBED_NORM_TOL:
    return norm_r
  return norm_r / norm_f


def _can_cache_tangent(loaded: LoadedProblem) -> bool:
  return (
    loaded.element_type == _LINEAR_ELEMENT and loaded.material_type in _LINEAR_MATERIALS
  )


def newton_step(
  loaded: LoadedProblem,
  state: F64,
  lam: float,
  *,
  settings: NonlinearSolverSettings,
  constraints: PrescribedConstraints,
  tangent_ctx: TangentAssemblyContext | None = None,
) -> F64:
  """Run one load step (Newton loop) and return the converged displacement."""
  problem = loaded.problem
  state = np.ascontiguousarray(state, dtype=np.float64)

  set_prescribed_displacements(state, problem, constraints, lam)
  f_ext = lam * problem.external_load

  use_cache = tangent_ctx is not None
  if use_cache:
    f_int = tangent_ctx.internal_force(state)
    k_csr = tangent_ctx.k_csr
  else:
    tangent = assemble_tangent_loaded(loaded, state)
    f_int = tangent.internal_force
    k_csr = tangent.stiffness.tocsr()

  error = residual_norm(f_ext, f_int, constraints)
  iteration = 0
  solve_red = factorized_reduced_solve(constraints, k_csr) if use_cache else None

  while error > settings.tol:
    iteration += 1
    if iteration > settings.iter_max:
      msg = "Newton-Raphson iterations did not converge!"
      raise RuntimeError(msg)

    b_red = constraints.C.T @ (f_ext - f_int)
    if solve_red is not None:
      da_red = solve_red(b_red)
    else:
      k_red = constraints.C.T @ (k_csr @ constraints.C)
      da_red = spsolve(k_red, b_red)
    da = constraints.C @ da_red

    state += da
    set_prescribed_displacements(state, problem, constraints, lam)

    if use_cache:
      f_int = tangent_ctx.internal_force(state)
    else:
      tangent = assemble_tangent_loaded(loaded, state)
      f_int = tangent.internal_force
      k_csr = tangent.stiffness.tocsr()

    error = residual_norm(f_ext, f_int, constraints)

  return np.asarray(state, dtype=np.float64)


def solve_nonlinear(loaded: LoadedProblem) -> SolverState:
  """
  Solve a nonlinear static problem with load ramping and Newton–Raphson.

  Requires ``SmallStrainContinuum`` and a linear elastic material for the
  current implementation.
  """
  resolve_solver_type(loaded.solver_type)
  resolve_element_type(loaded.element_type)
  resolve_material_type(loaded.material_type)

  if loaded.element_type != _LINEAR_ELEMENT:
    msg = f"Nonlinear solve supports {_LINEAR_ELEMENT!r} only"
    raise ValueError(msg)

  settings = _solver_settings(loaded)
  constraints = build_prescribed_constraints(loaded.problem)
  n_steps = n_load_steps(settings)

  state = np.zeros(loaded.problem.n_dofs, dtype=np.float64)
  state_increment = np.zeros(loaded.problem.n_dofs, dtype=np.float64)

  tangent_ctx = (
    TangentAssemblyContext.from_loaded(loaded) if _can_cache_tangent(loaded) else None
  )

  for step in range(1, n_steps + 1):
    lam, _dlam = load_factor(step, settings)
    state = newton_step(
      loaded,
      state,
      lam,
      settings=settings,
      constraints=constraints,
      tangent_ctx=tangent_ctx,
    )
    state_increment = np.zeros(loaded.problem.n_dofs, dtype=np.float64)

  return SolverState(state=state, state_increment=state_increment)
