"""Small-strain 2D continuum — dispatch by nodes per element."""

from __future__ import annotations

from pyfem.v3.fem.element import (
  quad4_plane_stress_stiffness,
  quad8_plane_stress_stiffness,
  tria3_plane_stress_stiffness,
)
from pyfem.v3.types import F64


def element_stiffness(
  coords: F64,
  constitutive: F64,
) -> F64:
  """Element tangent stiffness K_e = ∫ Bᵀ C B dΩ."""
  n_nodes = coords.shape[-2] if coords.ndim == 3 else coords.shape[0]
  if n_nodes == 8:
    return quad8_plane_stress_stiffness(coords, constitutive)
  if n_nodes == 4:
    return quad4_plane_stress_stiffness(coords, constitutive)
  if n_nodes == 3:
    return tria3_plane_stress_stiffness(coords, constitutive)
  msg = f"Unsupported element with {n_nodes} nodes"
  raise ValueError(msg)
