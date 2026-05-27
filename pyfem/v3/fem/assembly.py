"""Global assembly (vectorized COO scatter)."""

from __future__ import annotations

import numpy as np

from pyfem.v3.fem.element import quad8_plane_stress_stiffness
from pyfem.v3.types import F64, I32

_NDOF_PER_ELEM = 16
_ENTRIES_PER_ELEM = _NDOF_PER_ELEM * _NDOF_PER_ELEM


def assemble_stiffness_coo(
  coords: F64,
  conn: I32,
  global_dofs: I32,
  constitutive: F64,
  row: I32,
  col: I32,
  val: F64,
) -> int:
  """Fill preallocated COO buffers with element stiffness contributions."""
  nodal_coords = coords[conn]
  element_dofs = global_dofs[conn].reshape(conn.shape[0], _NDOF_PER_ELEM)
  stiffness = quad8_plane_stress_stiffness(nodal_coords, constitutive)

  dof_i = np.repeat(element_dofs, _NDOF_PER_ELEM, axis=1)
  dof_j = np.tile(element_dofs, (1, _NDOF_PER_ELEM))

  row[:] = dof_i.ravel()
  col[:] = dof_j.ravel()
  val[:] = stiffness.reshape(-1)
  return conn.shape[0] * _ENTRIES_PER_ELEM
