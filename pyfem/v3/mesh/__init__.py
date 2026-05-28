"""Mesh construction and DOF maps."""

from __future__ import annotations

import numpy as np

from pyfem.v3.types import DOF_TYPES_2D, DofMap, Mesh


def build_dof_map(mesh: Mesh, dof_types: tuple[str, ...] = DOF_TYPES_2D) -> DofMap:
  """Assign consecutive global DOF indices (legacy-compatible ordering)."""
  n_nodes = mesh.n_nodes
  global_dofs = np.arange(n_nodes * len(dof_types), dtype=np.int32).reshape(
    (n_nodes, len(dof_types)),
  )
  return DofMap(
    global_dofs=global_dofs,
    dof_types=dof_types,
    node_ids=mesh.node_ids.copy(),
    node_id_to_index=dict(mesh.node_id_to_index),
  )
