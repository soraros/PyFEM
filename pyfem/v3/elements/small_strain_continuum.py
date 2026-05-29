"""Small-strain continuum — dispatch by spatial rank and nodes per element."""

from __future__ import annotations

from pyfem.v3.fem.element import (
  hex8_stiffness,
  quad4_plane_stress_stiffness,
  quad8_plane_stress_stiffness,
  tet4_stiffness,
  tria3_plane_stress_stiffness,
)
from pyfem.v3.types import F64


def element_stiffness(
  coords: F64,
  constitutive: F64,
) -> F64:
  """Element tangent stiffness K_e = ∫ Bᵀ C B dΩ."""
  if coords.ndim == 2:
    nodal_coords = coords[None, ...]
    single = True
  else:
    nodal_coords = coords
    single = False

  spatial_dim = int(nodal_coords.shape[-1])
  n_nodes = int(nodal_coords.shape[-2])

  if spatial_dim == 2:
    if n_nodes == 8:
      result = quad8_plane_stress_stiffness(nodal_coords, constitutive)
    elif n_nodes == 4:
      result = quad4_plane_stress_stiffness(nodal_coords, constitutive)
    elif n_nodes == 3:
      result = tria3_plane_stress_stiffness(nodal_coords, constitutive)
    else:
      msg = f"Unsupported 2D element with {n_nodes} nodes"
      raise ValueError(msg)
  elif spatial_dim == 3:
    if n_nodes == 8:
      result = hex8_stiffness(nodal_coords, constitutive)
    elif n_nodes == 4:
      result = tet4_stiffness(nodal_coords, constitutive)
    else:
      msg = f"Unsupported 3D element with {n_nodes} nodes"
      raise ValueError(msg)
  else:
    msg = f"Unsupported spatial dimension {spatial_dim}"
    raise ValueError(msg)

  return result[0] if single else result
