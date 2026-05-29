"""Build :class:`ProblemDefinition` from load-time structures."""

from __future__ import annotations

import numpy as np

from pyfem.v3.materials.isotropic import isotropic_matrix
from pyfem.v3.materials.plane_strain import plane_strain_matrix
from pyfem.v3.materials.plane_stress import plane_stress_matrix
from pyfem.v3.registry import resolve_material_type
from pyfem.v3.types import (
  DofMap,
  Mesh,
  MpcTie,
  NodalLoad,
  PrescribedDof,
  ProblemDefinition,
)


def _dof_node_type(dof_map: DofMap, dof_id: int) -> tuple[int, str]:
  for row, node_id in enumerate(dof_map.node_ids):
    for col, dof_type in enumerate(dof_map.dof_types):
      if int(dof_map.global_dofs[row, col]) == dof_id:
        return int(node_id), dof_type
  msg = f"Unknown DOF index {dof_id}"
  raise ValueError(msg)


def _resolve_mpc_ties(
  dof_map: DofMap,
  constraints: tuple[PrescribedDof, ...],
  ties: tuple[MpcTie, ...],
) -> tuple[tuple[PrescribedDof, ...], tuple[MpcTie, ...]]:
  """Flatten MPC chains and promote ties to prescribed when the master is fixed."""
  prescribed: dict[int, float] = {
    dof_map.dof_index(item.node_id, item.dof_type): item.value for item in constraints
  }
  pending: dict[int, tuple[float, float, int]] = {
    dof_map.dof_index(tie.slave_node_id, tie.slave_dof_type): (
      tie.offset,
      tie.factor,
      dof_map.dof_index(tie.master_node_id, tie.master_dof_type),
    )
    for tie in ties
  }

  changed = True
  while changed:
    changed = False
    for slave, (offset, factor, master) in list(pending.items()):
      if master in prescribed:
        prescribed[slave] = offset + factor * prescribed[master]
        del pending[slave]
        changed = True
      elif master in pending:
        master_offset, master_factor, master_master = pending[master]
        pending[slave] = (
          offset + factor * master_offset,
          factor * master_factor,
          master_master,
        )
        changed = True

  if pending:
    for _slave, (_offset, _factor, master) in pending.items():
      if master in pending:
        msg = f"MPC master DOF {master} is also a slave (unresolved chain)"
        raise ValueError(msg)

  resolved_constraints = tuple(
    PrescribedDof(node_id=node_id, dof_type=dof_type, value=value)
    for dof_id, value in prescribed.items()
    for node_id, dof_type in [_dof_node_type(dof_map, dof_id)]
  )

  unresolved = tuple(
    MpcTie(
      slave_node_id=_dof_node_type(dof_map, slave)[0],
      slave_dof_type=_dof_node_type(dof_map, slave)[1],
      offset=offset,
      master_node_id=_dof_node_type(dof_map, master)[0],
      master_dof_type=_dof_node_type(dof_map, master)[1],
      factor=factor,
    )
    for slave, (offset, factor, master) in pending.items()
  )

  return resolved_constraints, unresolved


def _constitutive_matrix(
  material_type: str,
  youngs_modulus: float,
  poisson_ratio: float,
  *,
  spatial_rank: int,
) -> np.ndarray:
  key = resolve_material_type(material_type)
  if spatial_rank == 2:
    if key == "plane_stress":
      return plane_stress_matrix(youngs_modulus, poisson_ratio)
    if key == "plane_strain":
      return plane_strain_matrix(youngs_modulus, poisson_ratio)
    msg = f"Material {material_type!r} not supported for 2D meshes"
    raise ValueError(msg)
  if spatial_rank == 3:
    if key == "isotropic":
      return isotropic_matrix(youngs_modulus, poisson_ratio)
    msg = f"Material {material_type!r} not supported for 3D meshes"
    raise ValueError(msg)
  msg = f"Unsupported spatial rank {spatial_rank}"
  raise ValueError(msg)


def pack_problem(
  mesh: Mesh,
  dof_map: DofMap,
  youngs_modulus: float,
  poisson_ratio: float,
  constraints: tuple[PrescribedDof, ...] = (),
  ties: tuple[MpcTie, ...] = (),
  loads: tuple[NodalLoad, ...] = (),
  *,
  material_type: str = "PlaneStress",
) -> ProblemDefinition:
  """Build a jitable :class:`ProblemDefinition` from load-time structures."""
  constitutive = _constitutive_matrix(
    material_type,
    youngs_modulus,
    poisson_ratio,
    spatial_rank=mesh.rank,
  )
  n_dofs = dof_map.n_dofs
  external_load = np.zeros(n_dofs, dtype=np.float64)
  for load in loads:
    external_load[dof_map.dof_index(load.node_id, load.dof_type)] = load.value

  resolved_constraints, unresolved_ties = _resolve_mpc_ties(dof_map, constraints, ties)

  if resolved_constraints:
    constraint_dof = np.array(
      [dof_map.dof_index(c.node_id, c.dof_type) for c in resolved_constraints],
      dtype=np.int32,
    )
    constraint_val = np.array([c.value for c in resolved_constraints], dtype=np.float64)
  else:
    constraint_dof = np.empty(0, dtype=np.int32)
    constraint_val = np.empty(0, dtype=np.float64)

  if unresolved_ties:
    mpc_slave_dof = np.array(
      [dof_map.dof_index(t.slave_node_id, t.slave_dof_type) for t in unresolved_ties],
      dtype=np.int32,
    )
    mpc_master_dof = np.array(
      [dof_map.dof_index(t.master_node_id, t.master_dof_type) for t in unresolved_ties],
      dtype=np.int32,
    )
    mpc_factor = np.array([t.factor for t in unresolved_ties], dtype=np.float64)
    mpc_offset = np.array([t.offset for t in unresolved_ties], dtype=np.float64)
  else:
    mpc_slave_dof = np.empty(0, dtype=np.int32)
    mpc_master_dof = np.empty(0, dtype=np.int32)
    mpc_factor = np.empty(0, dtype=np.float64)
    mpc_offset = np.empty(0, dtype=np.float64)

  return ProblemDefinition(
    coords=np.ascontiguousarray(mesh.coords, dtype=np.float64),
    conn=np.ascontiguousarray(mesh.conn, dtype=np.int32),
    global_dofs=np.ascontiguousarray(dof_map.global_dofs, dtype=np.int32),
    constitutive=np.ascontiguousarray(constitutive, dtype=np.float64),
    constraint_dof=constraint_dof,
    constraint_val=constraint_val,
    mpc_slave_dof=mpc_slave_dof,
    mpc_master_dof=mpc_master_dof,
    mpc_factor=mpc_factor,
    mpc_offset=mpc_offset,
    external_load=np.ascontiguousarray(external_load, dtype=np.float64),
  )
