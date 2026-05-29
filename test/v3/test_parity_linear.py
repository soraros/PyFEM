# SPDX-License-Identifier: MIT

"""Linear skim parity vs legacy (parametrized)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from _legacy_parity import legacy_state, load_parity_tolerances

from pyfem.v3 import load_problem, solve_linear

ROOT = Path(__file__).resolve().parents[2]

LINEAR_SKIMS = (
  "patch_test8",
  "patch_test3",
  "patch_test4",
  "patch_test8_mpc",
  "patch_test8_plane_strain",
  "patch_test8_loaded",
)


@pytest.mark.parametrize("skim_name", LINEAR_SKIMS)
def test_linear_parity_from_problem_toml(skim_name: str) -> None:
  skim_dir = ROOT / "skims" / skim_name
  rtol, atol = load_parity_tolerances(skim_dir)
  loaded = load_problem(skim_dir / "problem.toml")
  v3_state = solve_linear(loaded)
  legacy = legacy_state(skim_dir / "skim.pro")
  np.testing.assert_allclose(v3_state, legacy, rtol=rtol, atol=atol)


@pytest.mark.parametrize("skim_name", LINEAR_SKIMS)
def test_linear_parity_from_skim_pro(skim_name: str) -> None:
  skim_dir = ROOT / "skims" / skim_name
  rtol, atol = load_parity_tolerances(skim_dir)
  loaded = load_problem(skim_dir / "skim.pro")
  v3_state = solve_linear(loaded)
  legacy = legacy_state(skim_dir / "skim.pro")
  np.testing.assert_allclose(v3_state, legacy, rtol=rtol, atol=atol)


def test_external_load_nonzero() -> None:
  loaded = load_problem(ROOT / "skims" / "patch_test8_loaded" / "problem.toml")
  assert np.linalg.norm(loaded.problem.external_load) > 0.0
