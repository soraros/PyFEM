# SPDX-License-Identifier: MIT

"""Tangent stiffness and internal-force assembly vs legacy."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from _legacy_parity import legacy_state, legacy_tangent_at_state

from pyfem.v3 import load_problem, solve_linear
from pyfem.v3._prototype_assembly import assemble_loaded, assemble_tangent_loaded
from pyfem.v3.solver.constraints import build_prescribed_constraints
from pyfem.v3.solver.state import initial_solver_state

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
  "skim_name",
  [
    "patch_test8",
    "patch_test4",
    "patch_test3",
    "patch_test8_3d",
    "patch_test8_plane_strain",
    "patch_test8_loaded",
    "patch_test8_mpc",
  ],
)
def test_tangent_matches_legacy_at_zero_state(skim_name: str) -> None:
  skim_pro = ROOT / "skims" / skim_name / "skim.pro"
  loaded = load_problem(skim_pro)
  solver_state = initial_solver_state(loaded.problem)
  v3 = assemble_tangent_loaded(loaded, solver_state.state)
  legacy_k, legacy_fint = legacy_tangent_at_state(skim_pro, solver_state.state)

  assert v3.stiffness.shape == legacy_k.shape
  v3_coo = v3.stiffness.tocoo()
  np.testing.assert_array_equal(v3_coo.row, legacy_k.row)
  np.testing.assert_array_equal(v3_coo.col, legacy_k.col)
  np.testing.assert_allclose(v3_coo.data, legacy_k.data, rtol=0.0, atol=1e-8)
  np.testing.assert_allclose(v3.internal_force, legacy_fint, rtol=0.0, atol=1e-8)


@pytest.mark.parametrize(
  "skim_name",
  [
    "patch_test8",
    "patch_test4",
    "patch_test3",
    "patch_test8_loaded",
  ],
)
def test_tangent_matches_legacy_at_converged_state(skim_name: str) -> None:
  skim_pro = ROOT / "skims" / skim_name / "skim.pro"
  loaded = load_problem(skim_pro)
  state = solve_linear(loaded)
  v3 = assemble_tangent_loaded(loaded, state)
  legacy_k, legacy_fint = legacy_tangent_at_state(skim_pro, state)

  v3_coo = v3.stiffness.tocoo()
  np.testing.assert_allclose(v3_coo.data, legacy_k.data, rtol=0.0, atol=1e-8)
  np.testing.assert_allclose(v3.internal_force, legacy_fint, rtol=0.0, atol=1e-8)


def test_internal_force_balances_external_load_at_solution() -> None:
  skim_pro = ROOT / "skims" / "patch_test8_loaded" / "skim.pro"
  loaded = load_problem(skim_pro)
  state = solve_linear(loaded)
  tangent = assemble_tangent_loaded(loaded, state)
  linear = assemble_loaded(loaded)
  constraints = build_prescribed_constraints(loaded.problem)

  residual = constraints.C.T @ (linear.load - tangent.internal_force)
  np.testing.assert_allclose(residual, 0.0, atol=1e-8)


def test_fused_tangent_matches_matvec_internal_force() -> None:
  skim_pro = ROOT / "skims" / "patch_test8" / "skim.pro"
  loaded = load_problem(skim_pro)
  state = solve_linear(loaded)
  fused = assemble_tangent_loaded(loaded, state)
  linear = assemble_loaded(loaded)
  fint_ref = linear.stiffness @ state
  v3_coo = fused.stiffness.tocoo()
  linear_coo = linear.stiffness.tocoo()
  np.testing.assert_allclose(v3_coo.data, linear_coo.data, rtol=0.0, atol=1e-8)
  np.testing.assert_allclose(fused.internal_force, fint_ref, rtol=0.0, atol=1e-8)


def test_stiffness_times_state_equals_internal_force() -> None:
  skim_pro = ROOT / "skims" / "patch_test8" / "skim.pro"
  loaded = load_problem(skim_pro)
  state = legacy_state(skim_pro)
  tangent = assemble_tangent_loaded(loaded, state)
  matvec = tangent.stiffness @ state
  np.testing.assert_allclose(matvec, tangent.internal_force, rtol=0.0, atol=1e-8)
