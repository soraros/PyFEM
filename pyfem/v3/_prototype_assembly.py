"""Global stiffness assembly."""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_array

from pyfem.v3.fem.assembly import (
  assemble_stiffness_coo,
  assemble_tangent_coo,
  count_tangent_entries,
)
from pyfem.v3.registry import resolve_element_type, resolve_material_type
from pyfem.v3.types import (
  LinearSystem,
  LoadedProblem,
  ProblemDefinition,
  TangentSystem,
)

CHUNK_THRESHOLD = 2048
DEFAULT_CHUNK_SIZE = 4096


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
  internal_force = (
    np.zeros(problem.n_dofs, dtype=np.float64) if tangent else None
  )
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
