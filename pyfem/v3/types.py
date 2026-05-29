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
DOF_TYPES_3D: tuple[str, ...] = ("u", "v", "w")

GROUP_CONTINUUM = 0
GROUP_TRUSS = 1
GROUP_SPRING = 2


def dof_types_for_rank(rank: int) -> tuple[str, ...]:
  """Return nodal DOF type names for a spatial rank."""
  if rank == 2:
    return DOF_TYPES_2D
  if rank == 3:
    return DOF_TYPES_3D
  msg = f"Unsupported spatial rank {rank}"
  raise ValueError(msg)


@dataclass(frozen=True)
class PlaneStressMaterial:
  """Linear plane-stress Hooke material (I/O only)."""

  youngs_modulus: float
  poisson_ratio: float


@dataclass(frozen=True)
class PlaneStrainMaterial:
  """Linear plane-strain Hooke material (I/O only)."""

  youngs_modulus: float
  poisson_ratio: float


@dataclass(frozen=True)
class IsotropicMaterial:
  """Linear 3D isotropic Hooke material (I/O only)."""

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


@dataclass(frozen=True)
class ElementGroupSpec:
  """Element group metadata from skim ``.pro`` (I/O only)."""

  name: str
  element_type: str
  props: tuple[float, ...]


@dataclass
class Mesh:
  """Structure-of-arrays mesh (built at load time)."""

  coords: F64
  conn: I32
  node_ids: I32
  elem_group_id: I32
  node_id_to_index: dict[int, int] = field(repr=False)
  group_names: tuple[str, ...] = ()

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
  elem_group_id: I32
  group_kind: I32
  group_props: F64

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
class NonlinearSolverSettings:
  """Nonlinear static solver controls (parsed from skim ``.pro``)."""

  tol: float = 1.0e-3
  iter_max: int = 10
  max_cycle: int = 5
  dtime: float = 1.0
  load_func: str = "t"
  load_table: F64 | None = None


@dataclass(frozen=True)
class RiksSolverSettings:
  """Riks arc-length solver controls (parsed from skim ``.pro``)."""

  tol: float = 1.0e-5
  iter_max: int = 10
  opt_iter: int = 5
  fixed_step: bool = False
  max_lam: float = 1.0e20
  max_factor: float = 1.0e20


@dataclass
class LoadedProblem:
  """Result of ``load_problem``: jitable data plus registry metadata."""

  problem: ProblemDefinition
  name: str
  element_type: str
  material_type: str
  solver_type: str
  element_group: str
  mesh_path: Path | None = None
  nonlinear_settings: NonlinearSolverSettings | None = None
  riks_settings: RiksSolverSettings | None = None
  groups: tuple[ElementGroupSpec, ...] = ()


@dataclass
class LinearSystem:
  """Assembled sparse linear system."""

  stiffness: coo_array
  load: F64
  n_dofs: int


@dataclass
class SolverState:
  """Global displacement state for nonlinear static solves."""

  state: F64
  state_increment: F64


@dataclass
class TangentSystem:
  """Assembled tangent stiffness and internal force at the current state."""

  stiffness: coo_array
  internal_force: F64
  n_dofs: int
