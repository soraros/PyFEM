"""Immutable backend-neutral contracts for reference linear assembly."""

from __future__ import annotations

from dataclasses import dataclass

from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.identity import InstanceId
from pyfem.v3.model.program import ProgramEvaluation
from pyfem.v3.model.provenance import CanonicalManifest, ContentFingerprint


@dataclass(frozen=True, slots=True)
class LinearStaticContributionRequest:
  """Exact zero-field marker for the frozen linear contribution family."""


@dataclass(frozen=True, slots=True, eq=False)
class DomainCooPlan:
  """Attributed raw and canonical full/reduced COO topology."""

  full_shape: tuple[int, int]
  full_raw_row_indices: FinalizedArray
  full_raw_column_indices: FinalizedArray
  block_indices: FinalizedArray
  cell_indices: FinalizedArray
  local_row_indices: FinalizedArray
  local_column_indices: FinalizedArray
  full_row_indices: FinalizedArray
  full_column_indices: FinalizedArray
  full_raw_to_canonical: FinalizedArray
  reduced_shape: tuple[int, int]
  reduced_raw_source_indices: FinalizedArray
  reduced_raw_row_indices: FinalizedArray
  reduced_raw_column_indices: FinalizedArray
  reduced_raw_left_factors: FinalizedArray
  reduced_raw_right_factors: FinalizedArray
  reduced_row_indices: FinalizedArray
  reduced_column_indices: FinalizedArray
  reduced_raw_to_canonical: FinalizedArray


@dataclass(frozen=True, slots=True, eq=False)
class NodalVectorContributionPlan:
  """Canonical full-to-reduced vector map in ascending full-DOF order."""

  full_dof_count: int
  reduced_dof_count: int
  full_dof_indices: FinalizedArray
  reduced_dof_indices: FinalizedArray
  coefficients: FinalizedArray


@dataclass(frozen=True, slots=True, eq=False)
class AssemblyPlanProvenance:
  """Canonical prepared-plan meaning and frozen numeric policies."""

  schema: str
  manifest: CanonicalManifest
  floating_dtype: str
  index_dtype: str
  reduction_policy: str
  geometry_policy: str


@dataclass(frozen=True, slots=True, eq=False)
class PreparedAssemblyPlan:
  """Immutable topology compatible with one exact live model/program pair."""

  content_fingerprint: ContentFingerprint
  provenance: AssemblyPlanProvenance
  compatible_model_instance_id: InstanceId
  compatible_model_content_fingerprint: ContentFingerprint
  compatible_program_instance_id: InstanceId
  compatible_program_content_fingerprint: ContentFingerprint
  request: LinearStaticContributionRequest
  domain_coo_plan: DomainCooPlan
  nodal_vector_plan: NodalVectorContributionPlan
  program_operator_recipes: tuple[str, ...]


@dataclass(frozen=True, slots=True, eq=False)
class CanonicalCooOperator:
  """One immutable canonical COO operator with plan-owned topology."""

  shape: tuple[int, int]
  row_indices: FinalizedArray
  column_indices: FinalizedArray
  values: FinalizedArray


@dataclass(frozen=True, slots=True, eq=False)
class LinearStaticContributions:
  """Exact full and affine-reduced linear contribution channels."""

  program_evaluation: ProgramEvaluation
  plan_content_fingerprint: ContentFingerprint
  full_operator: CanonicalCooOperator
  reduced_operator: CanonicalCooOperator
  full_raw_operator_values: FinalizedArray
  reduced_raw_operator_values: FinalizedArray
  full_affine_offset_internal_force: FinalizedArray
  full_affine_offset_internal_force_derivatives: FinalizedArray
  full_offset_corrected_rhs: FinalizedArray
  full_offset_corrected_rhs_derivatives: FinalizedArray
  reduced_external_force: FinalizedArray
  reduced_external_force_derivatives: FinalizedArray
  reduced_affine_offset_internal_force: FinalizedArray
  reduced_affine_offset_internal_force_derivatives: FinalizedArray
  reduced_rhs: FinalizedArray
  reduced_rhs_derivatives: FinalizedArray
