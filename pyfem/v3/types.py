"""Shared typed structures for the v3 core."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_array

# v3 array aliases (use in annotations and public APIs)
F64 = NDArray[np.float64]
I32 = NDArray[np.int32]

DOF_TYPES_2D: tuple[str, ...] = ("u", "v")


@dataclass(frozen=True)
class PlaneStressMaterial:
  """Linear plane-stress Hooke material (I/O only)."""

  youngs_modulus: float
  poisson_ratio: float


@dataclass(frozen=True)
class PrescribedDof:
  """Prescribed displacement on a node (I/O only)."""

  node_id: int
  dof_type: str
  value: float


@dataclass(frozen=True)
class NodalLoad:
  """Nodal force on a node (I/O only)."""

  node_id: int
  dof_type: str
  value: float


@dataclass(frozen=True)
class MpcTie:
  """Multi-point tie: slave DOF follows master (I/O only)."""

  slave_node_id: int
  slave_dof_type: str
  offset: float
  master_node_id: int
  master_dof_type: str
  factor: float


@dataclass
class Mesh:
  """Structure-of-arrays mesh (built at load time)."""

  coords: F64
  conn: I32
  node_ids: I32
  elem_group_id: I32
  node_id_to_index: dict[int, int] = field(repr=False)

  @property
  def n_nodes(self) -> int:
    return int(self.coords.shape[0])

  @property
  def n_elems(self) -> int:
    return int(self.conn.shape[0])

  @property
  def rank(self) -> int:
    return int(self.coords.shape[1])


@dataclass
class DofMap:
  """Global DOF numbering (built at load time)."""

  global_dofs: I32
  dof_types: tuple[str, ...]
  node_ids: I32
  node_id_to_index: dict[int, int]

  @property
  def n_dofs(self) -> int:
    return int(self.global_dofs.size)

  def dof_index(self, node_id: int, dof_type: str) -> int:
    row = self.node_id_to_index[node_id]
    col = self.dof_types.index(dof_type)
    return int(self.global_dofs[row, col])


class ProblemDefinition(NamedTuple):
  """
  Numeric problem payload for assembly and solvers.

  Array-only fields suitable for passing to ``@njit`` as a single argument
  or as unpacked ``F64`` / ``I32`` arrays.
  """

  coords: F64
  conn: I32
  global_dofs: I32
  constitutive: F64
  constraint_dof: I32
  constraint_val: F64
  mpc_slave_dof: I32
  mpc_master_dof: I32
  mpc_factor: F64
  mpc_offset: F64
  external_load: F64

  @property
  def n_nodes(self) -> int:
    return int(self.coords.shape[0])

  @property
  def n_elems(self) -> int:
    return int(self.conn.shape[0])

  @property
  def n_dofs(self) -> int:
    return int(self.global_dofs.size)


@dataclass(frozen=True)
class LoadedProblem:
  """Result of ``load_problem``: jitable data plus registry metadata."""

  problem: ProblemDefinition
  name: str
  element_type: str
  material_type: str
  solver_type: str
  element_group: str
  mesh_path: Path | None = None


@dataclass
class LinearSystem:
  """Assembled sparse linear system."""

  stiffness: coo_array
  load: F64
  n_dofs: int
