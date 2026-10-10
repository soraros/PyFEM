"""Global assembly (COO scatter into preallocated buffers)."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numba import get_num_threads, njit, prange

from pyfem.v3.elements.finite_strain_continuum import FINITE_STRAIN_ELEMENT
from pyfem.v3.fem.element import continuum_stiffness_batched
from pyfem.v3.fem.link2 import link2_tangent_batched
from pyfem.v3.fem.tl_element import quad8_tl_tangent_batched
from pyfem.v3.types import F64, GROUP_CONTINUUM, GROUP_SPRING, GROUP_TRUSS, I32

# Parallel-dispatch threshold in COO entries. The numba threading-layer
# dispatch costs ~85 us on the reference machine — about the serial fill
# time of 2**17 entries — so below it the serial kernels win.
_PARALLEL_MIN_ENTRIES = 1 << 17


def nodes_per_elem(conn: I32) -> int:
  """Return the number of nodes per element (uniform mesh)."""
  return int(conn.shape[1])


def entries_per_elem(conn: I32, spatial_dim: int) -> int:
  """COO entries contributed by one element stiffness matrix."""
  n_dof = spatial_dim * nodes_per_elem(conn)
  return n_dof * n_dof


@njit(cache=True)
def _fill_stiffness_coo_serial(
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


@njit(cache=True, parallel=True)
def _fill_stiffness_coo_parallel(
  element_dofs: I32,
  stiffness: F64,
  row: I32,
  col: I32,
  val: F64,
) -> None:
  """Parallel variant of ``_fill_stiffness_coo_serial``.

  Each COO slot is written by exactly one element, so the buffers are
  bit-identical to the serial kernel at any thread count.
  """
  n_elems = element_dofs.shape[0]
  n_dof = element_dofs.shape[1]
  entries_per_elem = n_dof * n_dof
  for e in prange(n_elems):
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


def _fill_stiffness_coo(
  element_dofs: I32,
  stiffness: F64,
  row: I32,
  col: I32,
  val: F64,
) -> None:
  """Scatter batched element matrices into COO buffers (parallel when large).

  The serial kernel also serves single-threaded execution: a prange region
  pays the threading-layer scheduler even at one numba thread (~1.5x slower
  on the 32x32 patch's serial scatter).
  """
  n_dof = element_dofs.shape[1]
  if (
    element_dofs.shape[0] * n_dof * n_dof >= _PARALLEL_MIN_ENTRIES
    and get_num_threads() > 1
  ):
    _fill_stiffness_coo_parallel(element_dofs, stiffness, row, col, val)
  else:
    _fill_stiffness_coo_serial(element_dofs, stiffness, row, col, val)


class CooCsrPattern(NamedTuple):
  """Precompiled canonical CSR topology of one fixed COO entry stream.

  ``permutation`` is the stable (row, col) lexsort of the stream and
  ``segment_offsets`` the start of each unique (row, col) run in sorted
  order, so duplicate accumulation follows the stream's own entry order —
  the stable-order contract ``driver.plan``'s ``refill_tangent`` documents.
  """

  indptr: I32
  indices: I32
  permutation: I32
  segment_offsets: I32
  shape: tuple[int, int]


def compile_csr_pattern(row: I32, col: I32, shape: tuple[int, int]) -> CooCsrPattern:
  """Compile the CSR pattern of a COO stream once, for repeated value refills.

  One-time topology cost (a stable lexsort); each reassembly through
  :func:`dedup_coo_values` is then argsort-free.
  """
  n_rows = int(shape[0])
  entry_count = row.shape[0]
  if entry_count == 0:
    return CooCsrPattern(
      indptr=np.zeros(n_rows + 1, dtype=np.int32),
      indices=np.empty(0, dtype=np.int32),
      permutation=np.empty(0, dtype=np.int32),
      segment_offsets=np.empty(0, dtype=np.int32),
      shape=(n_rows, int(shape[1])),
    )
  permutation = np.lexsort((col, row)).astype(np.int32)
  sorted_row = row[permutation]
  sorted_col = col[permutation]
  boundary = np.empty(entry_count, dtype=np.bool_)
  boundary[0] = True
  np.not_equal(sorted_row[1:], sorted_row[:-1], out=boundary[1:])
  boundary[1:] |= sorted_col[1:] != sorted_col[:-1]
  segment_offsets = np.flatnonzero(boundary).astype(np.int32)
  indices = sorted_col[segment_offsets]
  row_counts = np.bincount(sorted_row[segment_offsets], minlength=n_rows)
  indptr = np.zeros(n_rows + 1, dtype=np.int32)
  np.cumsum(row_counts, out=indptr[1:])
  return CooCsrPattern(
    indptr=indptr,
    indices=indices,
    permutation=permutation,
    segment_offsets=segment_offsets,
    shape=(n_rows, int(shape[1])),
  )


@njit(cache=True)
def _dedup_coo_values_serial(
  val: F64,
  permutation: I32,
  segment_offsets: I32,
  data: F64,
) -> None:
  """Sum each sorted duplicate run of ``val`` into canonical CSR data order."""
  n_segments = segment_offsets.shape[0]
  nnz = val.shape[0]
  for s in range(n_segments):
    start = segment_offsets[s]
    stop = segment_offsets[s + 1] if s + 1 < n_segments else nnz
    acc = 0.0
    for j in range(start, stop):
      acc += val[permutation[j]]
    data[s] = acc


@njit(cache=True, parallel=True)
def _dedup_coo_values_parallel(
  val: F64,
  permutation: I32,
  segment_offsets: I32,
  data: F64,
) -> None:
  """Parallel variant: segments are disjoint output slots, fixed sum order."""
  n_segments = segment_offsets.shape[0]
  nnz = val.shape[0]
  for s in prange(n_segments):
    start = segment_offsets[s]
    stop = segment_offsets[s + 1] if s + 1 < n_segments else nnz
    acc = 0.0
    for j in range(start, stop):
      acc += val[permutation[j]]
    data[s] = acc


def dedup_coo_segments(
  val: F64,
  permutation: I32,
  segment_offsets: I32,
  out: F64 | None = None,
) -> F64:
  """Sum duplicate COO contributions of ``val`` over precomputed segments.

  ``permutation`` is the stable (row, col) lexsort of the COO stream and
  ``segment_offsets`` the start of each unique (row, col) run in sorted
  order — exactly :class:`CooCsrPattern`'s fields, so
  :func:`dedup_coo_values` is this function keyed by a compiled pattern.
  Duplicate runs accumulate strictly left-to-right in the COO stream's own
  (stable-sorted) order, so the result is bit-identical at any thread
  count and on any platform: the parallel kernel assigns each segment one
  disjoint output slot and never reassociates. This strict-sequential
  order is the canonical v3 accumulation order. ``np.add.reduceat`` over
  the same segments agrees except on runs of length >= 3, where numpy's
  SIMD inner loop reorders at ulp level in a platform-dependent way.
  """
  data = np.empty(segment_offsets.shape[0], dtype=np.float64) if out is None else out
  if val.shape[0] >= _PARALLEL_MIN_ENTRIES and get_num_threads() > 1:
    _dedup_coo_values_parallel(val, permutation, segment_offsets, data)
  else:
    _dedup_coo_values_serial(val, permutation, segment_offsets, data)
  return data


def dedup_coo_values(pattern: CooCsrPattern, val: F64, out: F64 | None = None) -> F64:
  """Sum duplicate COO contributions of ``val`` into canonical CSR data.

  Duplicate runs accumulate strictly left-to-right in the COO stream's own
  (stable-sorted) order, so the result is bit-identical at any thread
  count and on any platform. scipy's ``tocsr`` sums the same runs but in
  its unstable per-row ``std::sort`` order, so its result agrees only to
  round-off (observed <= 9.4e-10 abs on the landed meshes, where entries
  reach 2e7).
  """
  return dedup_coo_segments(val, pattern.permutation, pattern.segment_offsets, out)


@njit(cache=True)
def _gather_element_states(
  element_dofs: I32,
  state: F64,
  element_states: F64,
) -> None:
  """Copy global displacements into per-element vectors."""
  n_elems = element_dofs.shape[0]
  n_dof = element_dofs.shape[1]
  for e in range(n_elems):
    for i in range(n_dof):
      element_states[e, i] = state[element_dofs[e, i]]


def _batched_stiffness(
  n_nodes: int,
  spatial_dim: int,
  nodal_coords: F64,
  constitutive: F64,
) -> F64:
  return continuum_stiffness_batched(nodal_coords, constitutive)


def _batched_tl_tangent(
  n_nodes: int,
  spatial_dim: int,
  nodal_coords: F64,
  constitutive: F64,
  element_dofs: I32,
  state: F64,
) -> tuple[F64, F64]:
  if spatial_dim != 2 or n_nodes != 8:
    msg = (
      f"FiniteStrainContinuum supports 2D Q8 only; got {n_nodes} nodes "
      f"in {spatial_dim}D"
    )
    raise ValueError(msg)
  n_dof = spatial_dim * n_nodes
  n_elems = element_dofs.shape[0]
  element_states = np.zeros((n_elems, n_dof), dtype=np.float64)
  _gather_element_states(element_dofs, state, element_states)
  return quad8_tl_tangent_batched(nodal_coords, element_states, constitutive)


def _batched_structural_tangent(
  group_kind: int,
  nodal_coords: F64,
  element_dofs: I32,
  state: F64,
  props: F64,
) -> tuple[F64, F64]:
  if group_kind not in (GROUP_TRUSS, GROUP_SPRING):
    msg = f"Unsupported structural group kind {group_kind}"
    raise ValueError(msg)
  n_elems = element_dofs.shape[0]
  element_states = np.zeros((n_elems, 4), dtype=np.float64)
  _gather_element_states(element_dofs, state, element_states)
  return link2_tangent_batched(
    nodal_coords,
    element_states,
    group_kind,
    props[0],
    props[1],
  )


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


def _entries_for_conn(conn: I32, spatial_dim: int) -> int:
  n_nodes = nodes_per_elem(conn)
  n_dof = spatial_dim * n_nodes
  return int(conn.shape[0]) * n_dof * n_dof


def _commit_group_coo(
  element_dofs: I32,
  stiffness: F64,
  element_forces: F64 | None,
  row: I32,
  col: I32,
  val: F64,
  global_force: F64 | None,
  *,
  row_offset: int,
) -> None:
  """Scatter element stiffness (and optional forces) into global COO buffers."""
  _fill_stiffness_coo(
    element_dofs,
    stiffness,
    row[row_offset:],
    col[row_offset:],
    val[row_offset:],
  )
  if global_force is not None and element_forces is not None:
    _scatter_nodal_forces(element_dofs, element_forces, global_force)


def _assemble_one_group(
  coords: F64,
  conn: I32,
  global_dofs: I32,
  constitutive: F64,
  row: I32,
  col: I32,
  val: F64,
  state: F64 | None,
  global_force: F64 | None,
  *,
  element_type: str,
  group_kind: int,
  group_props: F64,
  row_offset: int,
) -> int:
  spatial_dim = int(coords.shape[1])
  n_nodes = nodes_per_elem(conn)
  n_dof = spatial_dim * n_nodes
  nodal_coords = coords[conn]
  element_dofs = global_dofs[conn].reshape(conn.shape[0], n_dof)

  if group_kind == GROUP_TRUSS or group_kind == GROUP_SPRING:
    if state is None:
      msg = "Structural tangent assembly requires state"
      raise ValueError(msg)
    stiffness, element_forces = _batched_structural_tangent(
      group_kind,
      nodal_coords,
      element_dofs,
      state,
      group_props,
    )
    _commit_group_coo(
      element_dofs,
      stiffness,
      element_forces,
      row,
      col,
      val,
      global_force,
      row_offset=row_offset,
    )
    return _entries_for_conn(conn, spatial_dim)

  if element_type == FINITE_STRAIN_ELEMENT:
    if state is None:
      msg = "FiniteStrainContinuum tangent assembly requires state"
      raise ValueError(msg)
    stiffness, element_forces = _batched_tl_tangent(
      n_nodes,
      spatial_dim,
      nodal_coords,
      constitutive,
      element_dofs,
      state,
    )
    _commit_group_coo(
      element_dofs,
      stiffness,
      element_forces,
      row,
      col,
      val,
      global_force,
      row_offset=row_offset,
    )
    return _entries_for_conn(conn, spatial_dim)

  stiffness = _batched_stiffness(n_nodes, spatial_dim, nodal_coords, constitutive)
  element_forces = (
    _batched_element_internal_forces(stiffness, element_dofs, state)
    if state is not None and global_force is not None
    else None
  )
  _commit_group_coo(
    element_dofs,
    stiffness,
    element_forces,
    row,
    col,
    val,
    global_force,
    row_offset=row_offset,
  )
  return _entries_for_conn(conn, spatial_dim)


def count_tangent_entries(
  conn: I32,
  elem_group_id: I32,
  spatial_dim: int,
  *,
  group_kind: I32 | None = None,
) -> int:
  """Total COO entries for a connectivity slice (multi-group)."""
  total = 0
  if group_kind is None:
    for group_id in np.unique(elem_group_id):
      conn_g = conn[elem_group_id == group_id]
      if conn_g.size:
        total += _entries_for_conn(conn_g, spatial_dim)
    return total
  for group_id in range(int(group_kind.shape[0])):
    conn_g = conn[elem_group_id == group_id]
    if conn_g.size:
      total += _entries_for_conn(conn_g, spatial_dim)
  return total


def assemble_stiffness_coo(
  coords: F64,
  conn: I32,
  global_dofs: I32,
  constitutive: F64,
  row: I32,
  col: I32,
  val: F64,
  *,
  element_type: str = "SmallStrainContinuum",
  elem_group_id: I32 | None = None,
  group_kind: I32 | None = None,
  group_props: F64 | None = None,
) -> int:
  """Fill preallocated COO buffers with element stiffness contributions."""
  zero_state = (
    np.zeros(int(global_dofs.size), dtype=np.float64)
    if element_type == FINITE_STRAIN_ELEMENT
    or (group_kind is not None and np.any(group_kind != GROUP_CONTINUUM))
    else None
  )
  return assemble_tangent_coo(
    coords,
    conn,
    global_dofs,
    constitutive,
    row,
    col,
    val,
    state=zero_state,
    global_force=None,
    element_type=element_type,
    elem_group_id=elem_group_id,
    group_kind=group_kind,
    group_props=group_props,
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
  *,
  element_type: str = "SmallStrainContinuum",
  elem_group_id: I32 | None = None,
  group_kind: I32 | None = None,
  group_props: F64 | None = None,
) -> int:
  """
  Fill COO buffers and optionally accumulate internal forces from element batches.

  Supports multi-group structural meshes when ``elem_group_id``, ``group_kind``,
  and ``group_props`` are provided.
  """
  if elem_group_id is None or group_kind is None or group_props is None:
    elem_group_id = np.zeros(conn.shape[0], dtype=np.int32)
    group_kind = np.array([GROUP_CONTINUUM], dtype=np.int32)
    group_props = np.zeros((1, 2), dtype=np.float64)

  offset = 0
  for group_id in range(int(group_kind.shape[0])):
    mask = elem_group_id == group_id
    if not np.any(mask):
      continue
    conn_g = conn[mask]
    offset += _assemble_one_group(
      coords,
      conn_g,
      global_dofs,
      constitutive,
      row,
      col,
      val,
      state,
      global_force,
      element_type=element_type,
      group_kind=int(group_kind[group_id]),
      group_props=group_props[group_id],
      row_offset=offset,
    )
  return offset
