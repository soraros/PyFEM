"""Global stiffness assembly.

The module's assemble functions return the raw element-major COO stream
(``coo_array``) — the pinned contract the parity tests assert against the
legacy assembly. Production consumers convert that stream to canonical CSR
through :func:`canonical_csr`, which reuses the stream's compiled
:class:`~pyfem.v3.fem.assembly.CooCsrPattern` across assemblies of the same
topology (the driver-plan precedent: topology compilation is a one-time
cost, refills are values-only) and sums duplicates in the canonical
strict-sequential order (:func:`~pyfem.v3.fem.assembly.dedup_coo_values`).
Calling scipy's ``.tocsr()`` on the returned stream remains the reference
path — same canonical CSR topology, summation-order round-off on values.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.sparse import coo_array, csr_array

from pyfem.v3.fem.assembly import (
  CooCsrPattern,
  assemble_stiffness_coo,
  assemble_tangent_coo,
  compile_csr_pattern,
  count_tangent_entries,
  dedup_coo_values,
)
from pyfem.v3.registry import resolve_element_type, resolve_material_type
from pyfem.v3.types import (
  I32,
  LinearSystem,
  LoadedProblem,
  ProblemDefinition,
  TangentSystem,
)

CHUNK_THRESHOLD = 2048
DEFAULT_CHUNK_SIZE = 4096


class _StreamPatternEntry(NamedTuple):
  """One cached CSR pattern and the exact COO stream it was compiled from."""

  row: I32
  col: I32
  pattern: CooCsrPattern


# Process-local CSR pattern cache, keyed by (n_rows, n_cols, entry_count).
# A cached pattern is consumed only after the incoming row/column stream
# compares equal, entry by entry, to the stream the pattern was compiled
# from (the copies stored in the entry): identical streams share a topology,
# and any topology change — different connectivity, DOF map, group layout,
# or chunking — mismatches and recompiles. That validation IS the explicit
# invalidation; there is no staleness window and no identity-based keying.
_CSR_PATTERN_CACHE: dict[tuple[int, int, int], _StreamPatternEntry] = {}


def _cached_csr_pattern(
  row: np.ndarray,
  col: np.ndarray,
  shape: tuple[int, int],
) -> CooCsrPattern:
  """Return the compiled CSR pattern of one COO stream, cached per topology."""
  key = (int(shape[0]), int(shape[1]), int(row.shape[0]))
  entry = _CSR_PATTERN_CACHE.get(key)
  if (
    entry is not None
    and np.array_equal(entry.row, row)
    and np.array_equal(entry.col, col)
  ):
    return entry.pattern
  pattern = compile_csr_pattern(row, col, shape)
  _CSR_PATTERN_CACHE[key] = _StreamPatternEntry(
    row=np.array(row, dtype=np.int32, order="C", copy=True),
    col=np.array(col, dtype=np.int32, order="C", copy=True),
    pattern=pattern,
  )
  return pattern


def canonical_csr(coo: coo_array) -> csr_array:
  """Convert an assembled COO stream to canonical CSR via the cached pattern.

  Duplicate (row, col) contributions are summed by
  :func:`~pyfem.v3.fem.assembly.dedup_coo_values` strictly left-to-right in
  the stream's own stable order — thread-count- and platform-independent
  bits, unlike scipy's ``tocsr`` (unstable per-row ``std::sort`` order).
  The pattern of the stream is compiled once per topology and reused across
  assemblies; the cached entry is re-validated against the full stream on
  every call, so a topology change always recompiles.
  """
  pattern = _cached_csr_pattern(coo.row, coo.col, coo.shape)
  data = dedup_coo_values(pattern, coo.data)
  return csr_array(
    (data, pattern.indices, pattern.indptr),
    shape=pattern.shape,
  )


def _assemble_coo_chunk(
  problem: ProblemDefinition,
  conn_slice: slice,
  *,
  element_type: str,
  tangent: bool,
  state: np.ndarray | None,
  internal_force: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  conn = problem.conn[conn_slice]
  elem_group_id = problem.elem_group_id[conn_slice]
  spatial_dim = int(problem.coords.shape[1])
  nnz = count_tangent_entries(
    conn,
    elem_group_id,
    spatial_dim,
    group_kind=problem.group_kind,
  )
  row = np.empty(nnz, dtype=np.int32)
  col = np.empty(nnz, dtype=np.int32)
  val = np.empty(nnz, dtype=np.float64)
  if tangent:
    assert state is not None and internal_force is not None
    assemble_tangent_coo(
      problem.coords,
      conn,
      problem.global_dofs,
      problem.constitutive,
      row,
      col,
      val,
      state,
      internal_force,
      element_type=element_type,
      elem_group_id=elem_group_id,
      group_kind=problem.group_kind,
      group_props=problem.group_props,
    )
  else:
    assemble_stiffness_coo(
      problem.coords,
      conn,
      problem.global_dofs,
      problem.constitutive,
      row,
      col,
      val,
      element_type=element_type,
      elem_group_id=elem_group_id,
      group_kind=problem.group_kind,
      group_props=problem.group_props,
    )
  return row, col, val


def _assemble_coo_system(
  problem: ProblemDefinition,
  *,
  element_type: str,
  tangent: bool,
  state: np.ndarray | None = None,
  chunk_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
  n_elems = problem.n_elems
  internal_force = np.zeros(problem.n_dofs, dtype=np.float64) if tangent else None
  state_arr = None
  if tangent and state is not None:
    state_arr = np.ascontiguousarray(state, dtype=np.float64)

  if chunk_size is None and n_elems <= CHUNK_THRESHOLD:
    row, col, val = _assemble_coo_chunk(
      problem,
      slice(None),
      element_type=element_type,
      tangent=tangent,
      state=state_arr,
      internal_force=internal_force,
    )
    return row, col, val, internal_force

  size = DEFAULT_CHUNK_SIZE if chunk_size is None else chunk_size
  if size >= n_elems:
    row, col, val = _assemble_coo_chunk(
      problem,
      slice(None),
      element_type=element_type,
      tangent=tangent,
      state=state_arr,
      internal_force=internal_force,
    )
    return row, col, val, internal_force

  row_parts: list[np.ndarray] = []
  col_parts: list[np.ndarray] = []
  val_parts: list[np.ndarray] = []
  for start in range(0, n_elems, size):
    end = min(start + size, n_elems)
    chunk_row, chunk_col, chunk_val = _assemble_coo_chunk(
      problem,
      slice(start, end),
      element_type=element_type,
      tangent=tangent,
      state=state_arr,
      internal_force=internal_force,
    )
    row_parts.append(chunk_row)
    col_parts.append(chunk_col)
    val_parts.append(chunk_val)
  return (
    np.concatenate(row_parts),
    np.concatenate(col_parts),
    np.concatenate(val_parts),
    internal_force,
  )


def assemble_linear_system(
  problem: ProblemDefinition,
  *,
  element_type: str,
  material_type: str,
  chunk_size: int | None = None,
) -> LinearSystem:
  """Assemble global stiffness for a linear elastic static problem."""
  resolve_element_type(element_type)
  resolve_material_type(material_type)

  row, col, val, _ = _assemble_coo_system(
    problem,
    element_type=element_type,
    tangent=False,
    chunk_size=chunk_size,
  )
  n_dofs = problem.n_dofs
  load = np.ascontiguousarray(problem.external_load, dtype=np.float64)
  stiffness = coo_array((val, (row, col)), shape=(n_dofs, n_dofs))
  return LinearSystem(stiffness=stiffness, load=load, n_dofs=n_dofs)


def assemble_loaded(
  loaded: LoadedProblem,
  *,
  chunk_size: int | None = None,
) -> LinearSystem:
  """Assemble using registry metadata on :class:`LoadedProblem`."""
  return assemble_linear_system(
    loaded.problem,
    element_type=loaded.element_type,
    material_type=loaded.material_type,
    chunk_size=chunk_size,
  )


def assemble_tangent_system(
  problem: ProblemDefinition,
  state: np.ndarray,
  *,
  element_type: str,
  material_type: str,
  chunk_size: int | None = None,
) -> TangentSystem:
  """Assemble tangent stiffness and internal force at ``state``."""
  resolve_element_type(element_type)
  resolve_material_type(material_type)

  row, col, val, internal_force = _assemble_coo_system(
    problem,
    element_type=element_type,
    tangent=True,
    state=state,
    chunk_size=chunk_size,
  )
  assert internal_force is not None
  n_dofs = problem.n_dofs
  stiffness = coo_array((val, (row, col)), shape=(n_dofs, n_dofs))
  return TangentSystem(
    stiffness=stiffness,
    internal_force=internal_force,
    n_dofs=n_dofs,
  )


def assemble_tangent_loaded(
  loaded: LoadedProblem,
  state: np.ndarray,
  *,
  chunk_size: int | None = None,
) -> TangentSystem:
  """Assemble tangent system using registry metadata on :class:`LoadedProblem`."""
  return assemble_tangent_system(
    loaded.problem,
    state,
    element_type=loaded.element_type,
    material_type=loaded.material_type,
    chunk_size=chunk_size,
  )
