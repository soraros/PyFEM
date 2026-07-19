"""Immutable balance and verification result contracts."""

from __future__ import annotations

from dataclasses import dataclass

from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.identity import InstanceId, StateGeneration
from pyfem.v3.model.program import ProgramEvaluation
from pyfem.v3.model.provenance import CanonicalManifest, ContentFingerprint


@dataclass(frozen=True, slots=True, eq=False)
class LinearBalanceLedger:
  """Candidate-specific full/reduced balance and reaction evidence."""

  ledger_id: InstanceId
  prepared_instance_id: InstanceId
  model_instance_id: InstanceId
  model_content_fingerprint: ContentFingerprint
  program_instance_id: InstanceId
  program_content_fingerprint: ContentFingerprint
  plan_content_fingerprint: ContentFingerprint
  request_manifest: CanonicalManifest
  transaction_id: InstanceId
  trial_id: InstanceId
  base_generation: StateGeneration
  candidate_generation: StateGeneration
  program_evaluation: ProgramEvaluation
  reduced_coordinates: FinalizedArray
  full_primary_values: FinalizedArray
  reduced_primary_image: FinalizedArray
  prescribed_offsets: FinalizedArray
  external_force: FinalizedArray
  internal_force: FinalizedArray
  full_residual: FinalizedArray
  reduced_rhs: FinalizedArray
  reduced_internal_force: FinalizedArray
  reduced_residual: FinalizedArray
  constraint_force: FinalizedArray
  balance: FinalizedArray
  direct_reaction_dof_indices: FinalizedArray
  direct_reactions: FinalizedArray
  constraint_ids: tuple[str | int, ...]
  constraint_violation: FinalizedArray
  constraint_row_scales: FinalizedArray
  constraint_work: float
  external_work: float
  internal_work: float
  full_force_scale: float
  reduced_force_scale: float
  reconstruction_scale: float
  work_scale: float
  full_operator_infinity_norm: float
  reduced_operator_infinity_norm: float
  prolongation_transpose_infinity_norm: float
  verification_tolerance: float


@dataclass(frozen=True, slots=True)
class VerificationCheck:
  """One named normalized numerical check."""

  name: str
  passed: bool
  error: float
  scale: float
  normalized_error: float
  tolerance: float


@dataclass(frozen=True, slots=True)
class VerificationReport:
  """Immutable record-only or freshly recomputed verification report."""

  passed: bool
  level: str
  checks: tuple[VerificationCheck, ...]

  def check(self, name: str) -> VerificationCheck:
    """Return one exact named check."""
    for item in self.checks:
      if item.name == name:
        return item
    msg = f"verification report has no check named {name!r}"
    raise KeyError(msg)
