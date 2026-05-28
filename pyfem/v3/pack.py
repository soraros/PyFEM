"""Pack mesh / material / constraints into :class:`ProblemDefinition`."""

from __future__ import annotations

import numpy as np

from pyfem.v3.materials.plane_stress import plane_stress_matrix
from pyfem.v3.types import DofMap, Mesh, NodalLoad, PrescribedDof, ProblemDefinition


def pack_problem(
  mesh: Mesh,
  dof_map: DofMap,
  youngs_modulus: float,
  poisson_ratio: float,
  constraints: tuple[PrescribedDof, ...] = (),
  loads: tuple[NodalLoad, ...] = (),
) -> ProblemDefinition:
  """Build a jitable :class:`ProblemDefinition` from load-time structures."""
  constitutive = plane_stress_matrix(youngs_modulus, poisson_ratio)
  n_dofs = dof_map.n_dofs
  external_load = np.zeros(n_dofs, dtype=np.float64)
  for load in loads:
    external_load[dof_map.dof_index(load.node_id, load.dof_type)] = load.value

  if constraints:
    constraint_dof = np.array(
      [dof_map.dof_index(c.node_id, c.dof_type) for c in constraints],
      dtype=np.int32,
    )
    constraint_val = np.array([c.value for c in constraints], dtype=np.float64)
  else:
    constraint_dof = np.empty(0, dtype=np.int32)
    constraint_val = np.empty(0, dtype=np.float64)

  return ProblemDefinition(
    coords=np.ascontiguousarray(mesh.coords, dtype=np.float64),
    conn=np.ascontiguousarray(mesh.conn, dtype=np.int32),
    global_dofs=np.ascontiguousarray(dof_map.global_dofs, dtype=np.int32),
    constitutive=np.ascontiguousarray(constitutive, dtype=np.float64),
    constraint_dof=constraint_dof,
    constraint_val=constraint_val,
    external_load=np.ascontiguousarray(external_load, dtype=np.float64),
  )
