# SPDX-License-Identifier: MIT

"""Parity: v3 linear solve vs legacy LinearSolver on PatchTest4."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from pyfem.io.InputReader import InputRead
from pyfem.solvers.LinearSolver import LinearSolver
from pyfem.v3 import load_problem, solve_linear

ROOT = Path(__file__).resolve().parents[2]
SKIM_DIR = ROOT / "skims" / "patch_test4"
PARITY = SKIM_DIR / "parity.toml"


def _legacy_state(pro_path: Path) -> np.ndarray:
  props, globdat = InputRead(str(pro_path))
  solver = LinearSolver(props, globdat)
  solver.run(props, globdat)
  return np.asarray(globdat.state)


@pytest.fixture(scope="module")
def tolerances() -> tuple[float, float]:
  with PARITY.open("rb") as fh:
    data = tomllib.load(fh)
  return float(data["rtol"]), float(data["atol"])


def test_parity_from_problem_toml(tolerances: tuple[float, float]) -> None:
  rtol, atol = tolerances
  loaded = load_problem(SKIM_DIR / "problem.toml")
  v3_state = solve_linear(loaded)
  legacy_state = _legacy_state(SKIM_DIR / "skim.pro")
  np.testing.assert_allclose(v3_state, legacy_state, rtol=rtol, atol=atol)


def test_parity_from_skim_pro(tolerances: tuple[float, float]) -> None:
  rtol, atol = tolerances
  loaded = load_problem(SKIM_DIR / "skim.pro")
  v3_state = solve_linear(loaded)
  legacy_state = _legacy_state(SKIM_DIR / "skim.pro")
  np.testing.assert_allclose(v3_state, legacy_state, rtol=rtol, atol=atol)
