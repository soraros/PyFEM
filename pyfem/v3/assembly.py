"""Global stiffness assembly."""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_array

from pyfem.v3.fem.assembly import (
  assemble_stiffness_coo,
  entries_per_elem,
)
from pyfem.v3.registry import resolve_element_type, resolve_material_type
from pyfem.v3.types import LinearSystem, LoadedProblem, ProblemDefinition

CHUNK_THRESHOLD = 2048
DEFAULT_CHUNK_SIZE = 4096


def _assemble_coo_buffers(
  problem: ProblemDefinition,
  conn_slice: slice,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  conn = problem.conn[conn_slice]
  n_elems = conn.shape[0]
  spatial_dim = int(problem.coords.shape[1])
  nnz = n_elems * entries_per_elem(conn, spatial_dim)
  row = np.empty(nnz, dtype=np.int32)
  col = np.empty(nnz, dtype=np.int32)
  val = np.empty(nnz, dtype=np.float64)
  assemble_stiffness_coo(
    problem.coords,
    conn,
    problem.global_dofs,
    problem.constitutive,
    row,
    col,
    val,
  )
  return row, col, val


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

  n_dofs = problem.n_dofs
  n_elems = problem.n_elems
  if chunk_size is None and n_elems <= CHUNK_THRESHOLD:
    row, col, val = _assemble_coo_buffers(problem, slice(None))
  else:
    size = DEFAULT_CHUNK_SIZE if chunk_size is None else chunk_size
    if size >= n_elems:
      row, col, val = _assemble_coo_buffers(problem, slice(None))
    else:
      row_parts: list[np.ndarray] = []
      col_parts: list[np.ndarray] = []
      val_parts: list[np.ndarray] = []
      for start in range(0, n_elems, size):
        end = min(start + size, n_elems)
        chunk_row, chunk_col, chunk_val = _assemble_coo_buffers(
          problem,
          slice(start, end),
        )
        row_parts.append(chunk_row)
        col_parts.append(chunk_col)
        val_parts.append(chunk_val)
      row = np.concatenate(row_parts)
      col = np.concatenate(col_parts)
      val = np.concatenate(val_parts)

  load = np.ascontiguousarray(problem.external_load, dtype=np.float64)
  stiffness = coo_array(
    (val, (row, col)),
    shape=(n_dofs, n_dofs),
  )

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
