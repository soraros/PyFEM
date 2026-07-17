# SPDX-License-Identifier: MIT

"""Programmatic structural mesh generator tests."""

from __future__ import annotations

import sys

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from pyfem.v3 import load_problem, solve_riks
from pyfem.v3._prototype_assembly import assemble_tangent_loaded
from pyfem.v3.mesh.truss_fan import build_truss_fan, build_truss_fan_loaded

ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
SKIM_PRO = ROOT / "skims" / "shallow_truss_riks" / "skim.pro"


def test_truss_fan_counts() -> None:
  mesh, constraints, loads, groups = build_truss_fan(32)
  assert mesh.n_nodes == 34
  assert mesh.n_elems == 33
  assert len(constraints) == 66
  assert len(loads) == 1
  assert len(groups) == 2


def test_truss_fan_n2_matches_shallow_topology() -> None:
  mesh, _, _, _ = build_truss_fan(2, span=20.0, height=0.5)
  assert mesh.n_nodes == 4
  assert mesh.n_elems == 3
  np.testing.assert_allclose(mesh.coords[3], (0.0, 0.5))
  np.testing.assert_allclose(mesh.coords[1], (-10.0, 0.0))
  np.testing.assert_allclose(mesh.coords[2], (10.0, 0.0))


def test_truss_fan_tangent_assembles() -> None:
  loaded = build_truss_fan_loaded(16)
  state = np.zeros(loaded.problem.n_dofs, dtype=np.float64)
  tangent = assemble_tangent_loaded(loaded, state)
  assert tangent.stiffness.shape[0] == loaded.problem.n_dofs
  assert tangent.internal_force.shape == (loaded.problem.n_dofs,)


def test_truss_fan_n2_riks_matches_skim() -> None:
  skim = load_problem(SKIM_PRO)
  fan = build_truss_fan_loaded(2)
  skim_state = solve_riks(skim).state
  fan_state = solve_riks(fan).state
  np.testing.assert_allclose(fan_state, skim_state, rtol=1.0e-10, atol=1.0e-10)
