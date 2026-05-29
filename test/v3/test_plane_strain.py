# SPDX-License-Identifier: MIT

"""Plane-strain constitutive matrix."""

from __future__ import annotations

import sys

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from pyfem.v3.materials.plane_strain import plane_strain_matrix


def _reference_plane_strain_h(
  youngs_modulus: float,
  poisson_ratio: float,
) -> np.ndarray:
  """Closed form matching legacy ``PlaneStrain.H``."""
  nu = poisson_ratio
  e = youngs_modulus
  h00 = e * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
  h01 = h00 * nu / (1.0 - nu)
  h22 = h00 * 0.5 * (1.0 - 2.0 * nu) / (1.0 - nu)
  return np.array([[h00, h01, 0.0], [h01, h00, 0.0], [0.0, 0.0, h22]], dtype=np.float64)


@pytest.mark.parametrize("e, nu", [(1.0e6, 0.25), (210.0e9, 0.3)])
def test_plane_strain_matrix_matches_legacy_formula(e: float, nu: float) -> None:
  np.testing.assert_allclose(
    plane_strain_matrix(e, nu),
    _reference_plane_strain_h(e, nu),
    rtol=0.0,
    atol=0.0,
  )
