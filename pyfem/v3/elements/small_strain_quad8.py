"""Small-strain 2D continuum — 8-node quadrilateral."""

from __future__ import annotations

from pyfem.v3.fem.element import quad8_plane_stress_stiffness
from pyfem.v3.types import F64


def element_stiffness(
  coords: F64,
  constitutive: F64,
) -> F64:
  """Element tangent stiffness K_e = ∫ Bᵀ C B dΩ."""
  return quad8_plane_stress_stiffness(coords, constitutive)
