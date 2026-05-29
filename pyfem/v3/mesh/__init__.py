"""Mesh construction and DOF maps."""

from __future__ import annotations

import numpy as np

from pyfem.v3.types import DofMap, Mesh, dof_types_for_rank


def build_dof_map(mesh: Mesh, dof_types: tuple[str, ...] | None = None) -> DofMap:
  """Assign consecutive global DOF indices (legacy-compatible ordering)."""
  if dof_types is None:
    dof_types = dof_types_for_rank(mesh.rank)
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
