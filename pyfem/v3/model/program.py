"""Immutable compiled-program plans and bound evaluation values."""

from __future__ import annotations

from dataclasses import dataclass

from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.compiled import CompiledSource, DofPlan, EntityIndex, SourceMap
from pyfem.v3.model.identity import InstanceId
from pyfem.v3.model.provenance import CanonicalManifest, ContentFingerprint


@dataclass(frozen=True, slots=True, eq=False)
class AffineConstraintPlan:
  """Canonical CSR-like prolongation and affine prescribed offsets."""

  full_dof_count: int
  reduced_dof_count: int
  free_dofs: FinalizedArray
  row_offsets: FinalizedArray
  column_indices: FinalizedArray
  coefficients: FinalizedArray
  offset_constant: FinalizedArray
  offset_coordinate_coefficients: FinalizedArray


@dataclass(frozen=True, slots=True, eq=False)
class NodalLoadPlan:
  """One canonical attributable row per authored nodal-load declaration."""

  load_ids: tuple[str | int, ...]
  dof_indices: FinalizedArray
  constant_values: FinalizedArray
  coordinate_coefficients: FinalizedArray


@dataclass(frozen=True, slots=True, eq=False)
class ProgramCapabilities:
  """Conservative guarantees derived from the program plans in this slice."""

  fixed_constraint_topology: bool
  fixed_load_topology: bool
  has_constraints: bool
  has_nodal_loads: bool
  has_coordinate_affine_prescribed: bool
  has_coordinate_affine_nodal_loads: bool
  state_dependent: bool
  has_follower_loads: bool
  has_interaction_tangent: bool
  has_program_state: bool
  contribution_channels: tuple[str, ...]


@dataclass(frozen=True, slots=True, eq=False)
class ProgramProvenance:
  """Canonical program meaning and the exact numeric/reduction policy."""

  schema: str
  manifest: CanonicalManifest
  normalized_program_manifest: CanonicalManifest
  floating_dtype: str
  index_dtype: str
  coordinate_order: tuple[str, ...]
  reduction_policy: str


@dataclass(frozen=True, slots=True, eq=False)
class ProgramCoordinateWitness:
  """Compiler-owned normalized coordinate meaning."""

  name: str
  kind: str
  source: CompiledSource


@dataclass(frozen=True, slots=True, eq=False)
class ProgramDofWitness:
  """Compiler-owned exact semantic DOF reference."""

  node_id: str | int
  field_id: str | int
  component: str


@dataclass(frozen=True, slots=True, eq=False)
class ProgramAffineCoefficientWitness:
  """Compiler-owned normalized affine coordinate coefficient."""

  coordinate: str
  coefficient: float
  source: CompiledSource


@dataclass(frozen=True, slots=True, eq=False)
class ProgramAffineValueWitness:
  """Compiler-owned normalized affine value meaning."""

  constant: float
  coefficients: tuple[ProgramAffineCoefficientWitness, ...]
  source: CompiledSource


@dataclass(frozen=True, slots=True, eq=False)
class ProgramPrescribedWitness:
  """Compiler-owned normalized prescribed-DOF declaration."""

  id: str | int
  target: ProgramDofWitness
  value: ProgramAffineValueWitness
  source: CompiledSource


@dataclass(frozen=True, slots=True, eq=False)
class ProgramTieWitness:
  """Compiler-owned normalized one-master affine-tie declaration."""

  id: str | int
  slave: ProgramDofWitness
  master: ProgramDofWitness
  factor: float
  offset: ProgramAffineValueWitness
  source: CompiledSource


type ProgramConstraintWitness = ProgramPrescribedWitness | ProgramTieWitness


@dataclass(frozen=True, slots=True, eq=False)
class ProgramLoadWitness:
  """Compiler-owned normalized nodal-load declaration."""

  id: str | int
  target: ProgramDofWitness
  value: ProgramAffineValueWitness
  source: CompiledSource


@dataclass(frozen=True, slots=True, eq=False)
class ProgramMeaningWitness:
  """Typed normalized meaning used to audit all visible program carriers."""

  coordinates: tuple[ProgramCoordinateWitness, ...]
  constraints: tuple[ProgramConstraintWitness, ...]
  loads: tuple[ProgramLoadWitness, ...]
  source: CompiledSource
  model_dofs: DofPlan


@dataclass(frozen=True, slots=True, eq=False)
class CompiledProgram:
  """Immutable program compatible with one exact live compiled model."""

  instance_id: InstanceId
  content_fingerprint: ContentFingerprint
  provenance: ProgramProvenance
  compatible_model_instance_id: InstanceId
  compatible_model_content_fingerprint: ContentFingerprint
  coordinate_names: tuple[str, ...]
  coordinate_kinds: tuple[str, ...]
  meaning_witness: ProgramMeaningWitness
  constraint_plan: AffineConstraintPlan
  nodal_load_plan: NodalLoadPlan
  capabilities: ProgramCapabilities
  entity_index: EntityIndex
  source_map: SourceMap


@dataclass(frozen=True, slots=True, eq=False)
class ProgramEvaluation:
  """Immutable full-space affine values at one exact structured point."""

  program_instance_id: InstanceId
  program_content_fingerprint: ContentFingerprint
  compatible_model_instance_id: InstanceId
  compatible_model_content_fingerprint: ContentFingerprint
  coordinate_names: tuple[str, ...]
  coordinate_values: FinalizedArray
  prescribed_offsets: FinalizedArray
  prescribed_offset_derivatives: FinalizedArray
  nodal_force: FinalizedArray
  nodal_force_derivatives: FinalizedArray
