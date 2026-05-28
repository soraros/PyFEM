"""Global assembly (COO scatter into preallocated buffers)."""

from __future__ import annotations

from numba import njit

from pyfem.v3.fem.element import (
  quad4_plane_stress_stiffness,
  quad8_plane_stress_stiffness,
  tria3_plane_stress_stiffness,
)
from pyfem.v3.types import F64, I32


def nodes_per_elem(conn: I32) -> int:
  """Return the number of nodes per element (uniform mesh)."""
  return int(conn.shape[1])


def entries_per_elem(conn: I32) -> int:
  """COO entries contributed by one element stiffness matrix."""
  n_dof = 2 * nodes_per_elem(conn)
  return n_dof * n_dof


@njit(cache=True)
def _fill_stiffness_coo(
  element_dofs: I32,
  stiffness: F64,
  row: I32,
  col: I32,
  val: F64,
) -> None:
  """Scatter batched element matrices into COO buffers."""
  n_elems = element_dofs.shape[0]
  n_dof = element_dofs.shape[1]
  entries_per_elem = n_dof * n_dof
  for e in range(n_elems):
    dofs = element_dofs[e]
    ke = stiffness[e]
    base = e * entries_per_elem
    k = 0
    for i in range(n_dof):
      for j in range(n_dof):
        idx = base + k
        row[idx] = dofs[i]
        col[idx] = dofs[j]
        val[idx] = ke[i, j]
        k += 1


def _batched_stiffness(
  n_nodes: int,
  nodal_coords: F64,
  constitutive: F64,
) -> F64:
  if n_nodes == 8:
    return quad8_plane_stress_stiffness(nodal_coords, constitutive)
  if n_nodes == 4:
    return quad4_plane_stress_stiffness(nodal_coords, constitutive)
  if n_nodes == 3:
    return tria3_plane_stress_stiffness(nodal_coords, constitutive)
  msg = f"Unsupported element with {n_nodes} nodes per element"
  raise ValueError(msg)


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
  n_nodes = nodes_per_elem(conn)
  n_dof = 2 * n_nodes
  nodal_coords = coords[conn]
  element_dofs = global_dofs[conn].reshape(conn.shape[0], n_dof)
  stiffness = _batched_stiffness(n_nodes, nodal_coords, constitutive)
  _fill_stiffness_coo(element_dofs, stiffness, row, col, val)
  return conn.shape[0] * entries_per_elem(conn)
