"""Global assembly (COO scatter into preallocated buffers)."""

from __future__ import annotations

from numba import njit, prange

from pyfem.v3.fem.element import quad8_plane_stress_stiffness
from pyfem.v3.fem.parallel import PRANGE_MIN_ELEMS
from pyfem.v3.types import F64, I32

_NDOF_PER_ELEM = 16
_ENTRIES_PER_ELEM = _NDOF_PER_ELEM * _NDOF_PER_ELEM


@njit(cache=True, parallel=True)
def _fill_stiffness_coo(
  element_dofs: I32,
  stiffness: F64,
  row: I32,
  col: I32,
  val: F64,
) -> None:
  """Scatter batched element matrices into COO buffers (Q8: 16×16 dofs per elem)."""
  n_elems = element_dofs.shape[0]
  if n_elems < PRANGE_MIN_ELEMS:
    for e in range(n_elems):
      dofs = element_dofs[e]
      ke = stiffness[e]
      base = e * _ENTRIES_PER_ELEM
      k = 0
      for i in range(_NDOF_PER_ELEM):
        for j in range(_NDOF_PER_ELEM):
          idx = base + k
          row[idx] = dofs[i]
          col[idx] = dofs[j]
          val[idx] = ke[i, j]
          k += 1
  else:
    for e in prange(n_elems):
      dofs = element_dofs[e]
      ke = stiffness[e]
      base = e * _ENTRIES_PER_ELEM
      k = 0
      for i in range(_NDOF_PER_ELEM):
        for j in range(_NDOF_PER_ELEM):
          idx = base + k
          row[idx] = dofs[i]
          col[idx] = dofs[j]
          val[idx] = ke[i, j]
          k += 1


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
  _fill_stiffness_coo(element_dofs, stiffness, row, col, val)
  return conn.shape[0] * _ENTRIES_PER_ELEM
