"""Plane-strain constitutive matrix."""

from __future__ import annotations

import numpy as np

from pyfem.v3.types import F64


def plane_strain_matrix(youngs_modulus: float, poisson_ratio: float) -> F64:
  r"""
  Plane-strain elasticity tensor C (Voigt: ε = [ε_xx, ε_yy, γ_xy]ᵀ).

  Matches legacy ``PlaneStrain.H`` with :math:`\gamma_{xy} = 2\varepsilon_{xy}`.
  """
  nu = poisson_ratio
  e = youngs_modulus
  lam = e * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
  shear = lam * 0.5 * (1.0 - 2.0 * nu) / (1.0 - nu)
  nu_factor = nu / (1.0 - nu)
  return np.array(
    [
      [lam, lam * nu_factor, 0.0],
      [lam * nu_factor, lam, 0.0],
      [0.0, 0.0, shear],
    ],
    dtype=np.float64,
  )
