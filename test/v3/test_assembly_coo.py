# SPDX-License-Identifier: MIT

"""Global stiffness COO assembly: v3 vs legacy on PatchTest8."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import coo_matrix

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from pyfem.fem.Assembly import assembleTangentStiffness
from pyfem.io.InputReader import InputRead
from pyfem.v3 import load_problem
from pyfem.v3.assembly import assemble_loaded

ROOT = Path(__file__).resolve().parents[2]
SKIM_PRO = ROOT / "skims" / "patch_test8" / "skim.pro"


def _legacy_coo(pro_path: Path) -> coo_matrix:
  props, globdat = InputRead(str(pro_path))
  matrix, _ = assembleTangentStiffness(props, globdat)
  return matrix.tocoo()


def test_assembly_coo_matches_legacy_patch_test8() -> None:
  loaded = load_problem(SKIM_PRO)
  v3 = assemble_loaded(loaded).stiffness.tocoo()
  legacy = _legacy_coo(SKIM_PRO)

  assert v3.shape == legacy.shape
  np.testing.assert_array_equal(v3.row, legacy.row)
  np.testing.assert_array_equal(v3.col, legacy.col)
  np.testing.assert_allclose(v3.data, legacy.data, rtol=0.0, atol=1e-8)
