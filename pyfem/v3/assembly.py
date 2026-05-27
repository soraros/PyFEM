"""Global stiffness assembly."""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_array

from pyfem.v3.fem.assembly import assemble_stiffness_coo
from pyfem.v3.registry import resolve_element_type, resolve_material_type
from pyfem.v3.types import LinearSystem, LoadedProblem, ProblemDefinition

_ENTRIES_PER_ELEM = 16 * 16


def assemble_linear_system(
  problem: ProblemDefinition,
  *,
  element_type: str,
  material_type: str,
) -> LinearSystem:
  """Assemble global stiffness for a linear elastic static problem."""
  resolve_element_type(element_type)
  resolve_material_type(material_type)

  n_dofs = problem.n_dofs
  nnz = problem.n_elems * _ENTRIES_PER_ELEM

  row = np.empty(nnz, dtype=np.int32)
  col = np.empty(nnz, dtype=np.int32)
  val = np.empty(nnz, dtype=np.float64)

  assemble_stiffness_coo(
    problem.coords,
    problem.conn,
    problem.global_dofs,
    problem.constitutive,
    row,
    col,
    val,
  )

  load = np.zeros(n_dofs, dtype=np.float64)
  stiffness = coo_array(
    (val, (row, col)),
    shape=(n_dofs, n_dofs),
  )

  return LinearSystem(stiffness=stiffness, load=load, n_dofs=n_dofs)


def assemble_loaded(loaded: LoadedProblem) -> LinearSystem:
  """Assemble using registry metadata on :class:`LoadedProblem`."""
  return assemble_linear_system(
    loaded.problem,
    element_type=loaded.element_type,
    material_type=loaded.material_type,
  )
