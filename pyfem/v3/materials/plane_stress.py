"""Plane-stress constitutive matrix."""

from __future__ import annotations

import numpy as np

from pyfem.v3.types import F64


def plane_stress_matrix(youngs_modulus: float, poisson_ratio: float) -> F64:
  r"""
  Plane-stress elasticity tensor C (Voigt: ε = [ε_xx, ε_yy, γ_xy]ᵀ).

  Matches legacy ``PlaneStress.H`` with :math:`\gamma_{xy} = 2\varepsilon_{xy}`.
  The same constant matrix serves the small-strain law (C contracts the
  infinitesimal strain into the Cauchy stress) and the finite-strain
  Saint-Venant-Kirchhoff law (C contracts the Green-Lagrange strain into the
  second Piola-Kirchhoff stress) — the legacy material's single law behind
  both elements.
  """
  nu = poisson_ratio
  e = youngs_modulus
  lam = e / (1.0 - nu * nu)
  shear = e / (2.0 * (1.0 + nu))
  return np.array(
    [
      [lam, lam * nu, 0.0],
      [lam * nu, lam, 0.0],
      [0.0, 0.0, shear],
    ],
    dtype=np.float64,
  )
