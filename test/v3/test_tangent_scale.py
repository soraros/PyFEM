# SPDX-License-Identifier: MIT

"""Tangent assembly scale sanity on programmatic Q8 patches."""

from __future__ import annotations

import sys

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from pyfem.v3.assembly import assemble_tangent_loaded
from pyfem.v3.mesh.refined_patch import build_uniform_q8_loaded
from pyfem.v3.solver.context import prepare_linear_solve
from pyfem.v3.solver.state import initial_solver_state
from pyfem.v3.solver.tangent_context import prepare_tangent_assembly


@pytest.mark.parametrize("nx, ny", [(2, 2), (8, 8)])
def test_tangent_uniform_patch_assembles(nx: int, ny: int) -> None:
  loaded = build_uniform_q8_loaded(nx, ny)
  solver_state = initial_solver_state(loaded.problem)
  system = assemble_tangent_loaded(loaded, solver_state.state)
  assert system.stiffness.shape[0] == loaded.problem.n_dofs
  assert np.all(system.stiffness.diagonal() > 0.0)
  assert system.internal_force.shape[0] == loaded.problem.n_dofs


def test_tangent_context_internal_force_at_solution() -> None:
  loaded = build_uniform_q8_loaded(8, 8)
  state = prepare_linear_solve(loaded).solve()
  tangent = assemble_tangent_loaded(loaded, state)
  ctx = prepare_tangent_assembly(loaded)
  fint = ctx.internal_force(state)
  np.testing.assert_allclose(fint, tangent.internal_force, rtol=0.0, atol=1e-8)
