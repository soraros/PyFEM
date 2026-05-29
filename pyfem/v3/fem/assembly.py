"""Global assembly (COO scatter into preallocated buffers)."""

from __future__ import annotations

import numpy as np
from numba import njit

from pyfem.v3.fem.element import (
  hex8_stiffness,
  quad4_plane_stress_stiffness,
  quad8_plane_stress_stiffness,
  tet4_stiffness,
  tria3_plane_stress_stiffness,
)
from pyfem.v3.types import F64, I32


def nodes_per_elem(conn: I32) -> int:
  """Return the number of nodes per element (uniform mesh)."""
  return int(conn.shape[1])


def entries_per_elem(conn: I32, spatial_dim: int) -> int:
  """COO entries contributed by one element stiffness matrix."""
  n_dof = spatial_dim * nodes_per_elem(conn)
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
  spatial_dim: int,
  nodal_coords: F64,
  constitutive: F64,
) -> F64:
  if spatial_dim == 2:
    if n_nodes == 8:
      return quad8_plane_stress_stiffness(nodal_coords, constitutive)
    if n_nodes == 4:
      return quad4_plane_stress_stiffness(nodal_coords, constitutive)
    if n_nodes == 3:
      return tria3_plane_stress_stiffness(nodal_coords, constitutive)
  if spatial_dim == 3:
    if n_nodes == 8:
      return hex8_stiffness(nodal_coords, constitutive)
    if n_nodes == 4:
      return tet4_stiffness(nodal_coords, constitutive)
  msg = (
    f"Unsupported element with {n_nodes} nodes per element "
    f"in {spatial_dim}D"
  )
  raise ValueError(msg)


@njit(cache=True)
def _batched_element_internal_forces(
  stiffness: F64,
  element_dofs: I32,
  state: F64,
) -> F64:
  """Element internal forces ``f_e = K_e @ u_e`` for small-strain linear elasticity."""
  n_elems = stiffness.shape[0]
  n_dof = stiffness.shape[1]
  element_forces = np.zeros((n_elems, n_dof), dtype=np.float64)
  for e in range(n_elems):
    dofs = element_dofs[e]
    ke = stiffness[e]
    for i in range(n_dof):
      s = 0.0
      for j in range(n_dof):
        s += ke[i, j] * state[dofs[j]]
      element_forces[e, i] = s
  return element_forces


@njit(cache=True)
def _scatter_nodal_forces(
  element_dofs: I32,
  element_forces: F64,
  global_force: F64,
) -> None:
  """Scatter element force vectors into a global internal-force vector."""
  n_elems = element_dofs.shape[0]
  n_dof = element_dofs.shape[1]
  for e in range(n_elems):
    dofs = element_dofs[e]
    fe = element_forces[e]
    for i in range(n_dof):
      global_force[dofs[i]] += fe[i]


def _element_layout(
  coords: F64,
  conn: I32,
  global_dofs: I32,
) -> tuple[int, int, F64, I32]:
  spatial_dim = int(coords.shape[1])
  n_nodes = nodes_per_elem(conn)
  n_dof = spatial_dim * n_nodes
  nodal_coords = coords[conn]
  element_dofs = global_dofs[conn].reshape(conn.shape[0], n_dof)
  return n_nodes, spatial_dim, nodal_coords, element_dofs


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
  return assemble_tangent_coo(
    coords,
    conn,
    global_dofs,
    constitutive,
    row,
    col,
    val,
    state=None,
    global_force=None,
  )


def assemble_tangent_coo(
  coords: F64,
  conn: I32,
  global_dofs: I32,
  constitutive: F64,
  row: I32,
  col: I32,
  val: F64,
  state: F64 | None,
  global_force: F64 | None,
) -> int:
  """
  Fill COO buffers and optionally accumulate internal forces from one ``K_e`` batch.

  When ``state`` and ``global_force`` are set, uses the same element stiffness
  matrices for ``f_e = K_e @ u_e`` (no second quadrature pass).
  """
  n_nodes, spatial_dim, nodal_coords, element_dofs = _element_layout(
    coords,
    conn,
    global_dofs,
  )
  stiffness = _batched_stiffness(n_nodes, spatial_dim, nodal_coords, constitutive)
  _fill_stiffness_coo(element_dofs, stiffness, row, col, val)
  if state is not None and global_force is not None:
    element_forces = _batched_element_internal_forces(stiffness, element_dofs, state)
    _scatter_nodal_forces(element_dofs, element_forces, global_force)
  return conn.shape[0] * entries_per_elem(conn, spatial_dim)


def assemble_internal_force(
  coords: F64,
  conn: I32,
  global_dofs: I32,
  constitutive: F64,
  state: F64,
  global_force: F64,
) -> None:
  """Accumulate global internal forces from element ``K_e @ u_e``."""
  n_nodes, spatial_dim, nodal_coords, element_dofs = _element_layout(
    coords,
    conn,
    global_dofs,
  )
  stiffness = _batched_stiffness(n_nodes, spatial_dim, nodal_coords, constitutive)
  element_forces = _batched_element_internal_forces(stiffness, element_dofs, state)
  _scatter_nodal_forces(element_dofs, element_forces, global_force)
