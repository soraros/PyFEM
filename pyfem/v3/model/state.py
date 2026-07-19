"""Immutable state snapshots for the frozen stateless linear Q8 slice."""

from __future__ import annotations

from dataclasses import dataclass

from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.identity import InstanceId, StateGeneration
from pyfem.v3.model.program import ProgramEvaluation
from pyfem.v3.model.provenance import CanonicalManifest, ContentFingerprint

PHYSICAL_STATE_SCHEMA = "pyfem-v3-physical-state-stateless-q8-v1"
EVOLUTION_STATE_SCHEMA = "pyfem-v3-evolution-state-linear-static-v1"
PROGRAM_HISTORY_SCHEMA = "pyfem-v3-program-history-empty-v1"


@dataclass(frozen=True, slots=True, eq=False)
class PhysicalState:
  """One model-owned, accepted or trial, physical-state snapshot."""

  model_instance_id: InstanceId
  model_content_fingerprint: ContentFingerprint
  schema: str
  generation: StateGeneration
  primary_values: FinalizedArray
  material_histories: tuple[FinalizedArray, ...]
  formulation_histories: tuple[FinalizedArray, ...]


@dataclass(frozen=True, slots=True, eq=False)
class EvolutionState:
  """One request-owned algebraic evolution snapshot."""

  prepared_instance_id: InstanceId
  request_manifest: CanonicalManifest
  schema: str
  generation: StateGeneration
  program_evaluation: ProgramEvaluation
  algebraic_field_ids: tuple[str | int, ...]
  accepted_step_index: int
  predictor: FinalizedArray
  actual_increment: FinalizedArray


@dataclass(frozen=True, slots=True, eq=False)
class ProgramHistory:
  """The exact empty program-owned history used by Phase 1."""

  program_instance_id: InstanceId
  program_content_fingerprint: ContentFingerprint
  schema: str
  generation: StateGeneration
  entries: tuple[()] = ()


@dataclass(frozen=True, slots=True, eq=False)
class CommittedAnalysisState:
  """One atomically accepted composition of all Phase 1 state owners."""

  prepared_instance_id: InstanceId
  generation: StateGeneration
  physical: PhysicalState
  evolution: EvolutionState
  program_history: ProgramHistory
