"""3D isotropic constitutive matrix."""

from __future__ import annotations

import numpy as np

from pyfem.v3.types import F64


def isotropic_matrix(youngs_modulus: float, poisson_ratio: float) -> F64:
  r"""
  3D isotropic elasticity tensor C (Voigt: ε = [ε11, ε22, ε33, γ23, γ13, γ12]ᵀ).

  Matches legacy ``Isotropic.H``.
  """
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
