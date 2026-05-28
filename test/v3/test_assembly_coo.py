# SPDX-License-Identifier: MIT

"""Global stiffness COO assembly: v3 vs legacy on patch-test skims."""

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


def _legacy_coo(pro_path: Path) -> coo_matrix:
  props, globdat = InputRead(str(pro_path))
  matrix, _ = assembleTangentStiffness(props, globdat)
  return matrix.tocoo()


@pytest.mark.parametrize(
  "skim_name",
  ["patch_test8", "patch_test4"],
)
def test_assembly_coo_matches_legacy(skim_name: str) -> None:
  skim_pro = ROOT / "skims" / skim_name / "skim.pro"
  loaded = load_problem(skim_pro)
  v3 = assemble_loaded(loaded).stiffness.tocoo()
  legacy = _legacy_coo(skim_pro)

  assert v3.shape == legacy.shape
  np.testing.assert_array_equal(v3.row, legacy.row)
  np.testing.assert_array_equal(v3.col, legacy.col)
  np.testing.assert_allclose(v3.data, legacy.data, rtol=0.0, atol=1e-8)
