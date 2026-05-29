# SPDX-License-Identifier: MIT

"""Plane-strain scale sanity on programmatic Q8 patches."""

from __future__ import annotations

import sys

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from pyfem.v3.assembly import assemble_loaded
from pyfem.v3.mesh.refined_patch import build_uniform_q8_loaded
from pyfem.v3.solver.context import prepare_linear_solve


@pytest.mark.parametrize("nx, ny", [(2, 2), (8, 8)])
def test_plane_strain_uniform_patch_assembles_and_solves(nx: int, ny: int) -> None:
  loaded = build_uniform_q8_loaded(nx, ny, material_type="PlaneStrain")
  system = assemble_loaded(loaded)
  assert system.stiffness.shape[0] == loaded.problem.n_dofs
  assert np.all(system.stiffness.diagonal() > 0.0)
  state = prepare_linear_solve(loaded).solve()
  assert state.shape[0] == loaded.problem.n_dofs
  assert np.linalg.norm(state) > 0.0
