# SPDX-License-Identifier: MIT

"""3D isotropic constitutive matrix."""

from __future__ import annotations

import sys

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from pyfem.v3.materials.isotropic import isotropic_matrix


def _reference_isotropic_h(youngs_modulus: float, poisson_ratio: float) -> np.ndarray:
  """Closed form matching legacy ``Isotropic.H``."""
  nu = poisson_ratio
  e = youngs_modulus
  fac = 1.0 / (2.0 * nu * nu + nu - 1.0)
  lam = fac * e * (nu - 1.0)
  nu_off = -fac * e * nu
  shear = e / (2.0 + 2.0 * nu)
  return np.array(
    [
      [lam, nu_off, nu_off, 0.0, 0.0, 0.0],
      [nu_off, lam, nu_off, 0.0, 0.0, 0.0],
      [nu_off, nu_off, lam, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, shear, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, shear, 0.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, shear],
    ],
    dtype=np.float64,
  )


@pytest.mark.parametrize("e, nu", [(1.0e6, 0.25), (210.0e9, 0.3)])
def test_isotropic_matrix_matches_legacy_formula(e: float, nu: float) -> None:
  np.testing.assert_allclose(
    isotropic_matrix(e, nu),
    _reference_isotropic_h(e, nu),
    rtol=0.0,
    atol=0.0,
  )
