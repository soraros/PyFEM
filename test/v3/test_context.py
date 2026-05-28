# SPDX-License-Identifier: MIT

"""LinearSolutionContext repeat-solve behavior."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from pyfem.v3 import load_problem
from pyfem.v3.solver.context import prepare_linear_solve
from pyfem.v3.solver.linear import solve_linear

ROOT = Path(__file__).resolve().parents[2]
LOADED_SKIM_PRO = ROOT / "skims" / "patch_test8_loaded" / "skim.pro"


def test_context_matches_solve_linear() -> None:
  loaded = load_problem(LOADED_SKIM_PRO)
  reference = solve_linear(loaded)
  ctx = prepare_linear_solve(loaded)
  state = ctx.solve()
  np.testing.assert_allclose(state, reference, rtol=1.0e-10, atol=1.0e-12)


def test_context_repeat_is_stable() -> None:
  loaded = load_problem(LOADED_SKIM_PRO)
  ctx = prepare_linear_solve(loaded)
  first = ctx.solve()
  second = ctx.solve()
  np.testing.assert_allclose(second, first, rtol=0, atol=0)
