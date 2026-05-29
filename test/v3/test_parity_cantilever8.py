# SPDX-License-Identifier: MIT

"""ch.3 cantilever8 — FiniteStrainContinuum nonlinear parity vs legacy."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from _legacy_parity import legacy_nonlinear_state, legacy_tangent_at_state

from pyfem.v3 import load_problem, solve_nonlinear

ROOT = Path(__file__).resolve().parents[2]
SKIM_PRO = ROOT / "skims" / "cantilever8" / "skim.pro"


def _tolerances() -> tuple[float, float]:
  with (ROOT / "skims" / "cantilever8" / "parity.toml").open("rb") as fh:
    data = tomllib.load(fh)
  return float(data["rtol"]), float(data["atol"])


def test_cantilever8_nonlinear_matches_legacy() -> None:
  rtol, atol = _tolerances()
  loaded = load_problem(SKIM_PRO)
  v3_state = solve_nonlinear(loaded).state
  legacy = legacy_nonlinear_state(SKIM_PRO)
  np.testing.assert_allclose(v3_state, legacy, rtol=rtol, atol=atol)


def test_cantilever8_tangent_matches_legacy_at_converged_state() -> None:
  rtol, atol = _tolerances()
  loaded = load_problem(SKIM_PRO)
  state = solve_nonlinear(loaded).state
  legacy_k, legacy_fint = legacy_tangent_at_state(SKIM_PRO, state)
  from pyfem.v3.assembly import assemble_tangent_loaded

  tangent = assemble_tangent_loaded(loaded, state)
  v3_k = tangent.stiffness.tocoo()
  np.testing.assert_allclose(
    v3_k.data,
    legacy_k.data,
    rtol=rtol,
    atol=atol,
  )
  np.testing.assert_allclose(
    tangent.internal_force,
    legacy_fint,
    rtol=rtol,
    atol=atol,
  )
