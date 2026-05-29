# SPDX-License-Identifier: MIT

"""ch.4 shallow truss — Truss/Spring + Riks parity vs legacy."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from _legacy_parity import legacy_riks_state

from pyfem.v3 import load_problem, solve_riks

ROOT = Path(__file__).resolve().parents[2]
SKIM_PRO = ROOT / "skims" / "shallow_truss_riks" / "skim.pro"


def _tolerances() -> tuple[float, float]:
  with (ROOT / "skims" / "shallow_truss_riks" / "parity.toml").open("rb") as fh:
    data = tomllib.load(fh)
  return float(data["rtol"]), float(data["atol"])


def test_shallow_truss_riks_matches_legacy() -> None:
  rtol, atol = _tolerances()
  loaded = load_problem(SKIM_PRO)
  v3_state = solve_riks(loaded).state
  legacy = legacy_riks_state(SKIM_PRO)
  np.testing.assert_allclose(v3_state, legacy, rtol=rtol, atol=atol)
