"""Riks arc-length nonlinear static solver."""

from __future__ import annotations

import numpy as np

from pyfem.v3.assembly import assemble_tangent_loaded
from pyfem.v3.registry import (
  resolve_element_type,
  resolve_material_type,
  resolve_solver_type,
)
from pyfem.v3.solver._settings import riks_settings
from pyfem.v3.solver.constraints import (
  build_prescribed_constraints,
)
from pyfem.v3.solver.context import solve_reduced_displacement
from pyfem.v3.solver.nonlinear import residual_norm
from pyfem.v3.types import LoadedProblem, SolverState

_CYCLE_CAP = 1000


def solve_riks(loaded: LoadedProblem) -> SolverState:
  """Solve a structural problem with Riks arc-length continuation."""
  resolve_solver_type(loaded.solver_type)
  resolve_element_type(loaded.element_type)
  resolve_material_type(loaded.material_type)

  settings = riks_settings(loaded)
  constraints = build_prescribed_constraints(loaded.problem)
  fhat = loaded.problem.external_load

  state = np.zeros(loaded.problem.n_dofs, dtype=np.float64)
  da_prev = np.zeros(loaded.problem.n_dofs, dtype=np.float64)
  dlam_prev = 1.0
  factor = 1.0
  lam = 1.0
  cycle = 0
  active = True

  while active and cycle < _CYCLE_CAP:
    cycle += 1

    if cycle == 1:
      tangent = assemble_tangent_loaded(loaded, state)
      k_csr = tangent.stiffness.tocsr()
      da1 = solve_reduced_displacement(constraints, k_csr, lam * fhat)
      dlam1 = lam
    else:
      da1 = factor * da_prev
      dlam1 = factor * dlam_prev
      lam += dlam1

    state = state + da1
    da = da1.copy()
    dlam = dlam1

    tangent = assemble_tangent_loaded(loaded, state)
    f_int = tangent.internal_force
    k_csr = tangent.stiffness.tocsr()

    f_ext = lam * fhat
    error = residual_norm(f_ext, f_int, constraints)
    iteration = 0

    while error > settings.tol:
      iteration += 1
      if iteration > settings.iter_max:
        msg = "Newton-Raphson iterations did not converge!"
        raise RuntimeError(msg)

      d1 = solve_reduced_displacement(constraints, k_csr, fhat)
      res = f_ext - f_int
      d2 = solve_reduced_displacement(constraints, k_csr, res)
      ddlam = -float(np.dot(da1, d2)) / float(np.dot(da1, d1))
      dda = ddlam * d1 + d2

      dlam += ddlam
      lam += ddlam
      da = da + dda
      state = state + dda
      f_ext = lam * fhat

      tangent = assemble_tangent_loaded(loaded, state)
      f_int = tangent.internal_force
      k_csr = tangent.stiffness.tocsr()
      error = residual_norm(f_ext, f_int, constraints)

    if not settings.fixed_step:
      factor = float(0.5 ** (0.25 * (iteration - settings.opt_iter)))
      if factor > settings.max_factor:
        factor = 1.0

    da_prev = da.copy()
    dlam_prev = dlam

    if lam > settings.max_lam or cycle >= _CYCLE_CAP:
      active = False

  return SolverState(
    state=np.asarray(state, dtype=np.float64),
    state_increment=np.asarray(da_prev, dtype=np.float64),
  )
