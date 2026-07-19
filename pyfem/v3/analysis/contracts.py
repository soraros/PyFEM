"""Typed request, transaction, and convergence contracts for linear statics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import final

from pyfem.v3.model import (
  CanonicalManifest,
  CommittedAnalysisState,
  ContentFingerprint,
  FinalizedArray,
  InstanceId,
  ProgramEvaluation,
  StateGeneration,
)
from pyfem.v3.results.contracts import LinearBalanceLedger, VerificationReport

LINEAR_STATIC_REQUEST_SCHEMA = "pyfem-v3-linear-static-request-v1"
LINEAR_STATIC_VERIFICATION_TOLERANCE = 1.0e-12
LINEAR_STATIC_SYMMETRY_EPSILON_FACTOR = 64
LINEAR_STATIC_CHOLESKY_PIVOT_RATIO = 1.0e-12
LINEAR_STATIC_WORKSPACE_BUDGET_BYTES = 256 * 1024 * 1024
LINEAR_STATIC_WORKSPACE_BUDGET_SCOPE = (
  "private-solver-workspace-only; peak=4*n^2+6*n-float64; "
  "explicit-cholesky-factorization; explicit-triangular-solve; "
  "fresh-verification-workspace-measured-separately"
)
LINEAR_STATIC_BACKEND_POLICY = (
  "private-dense-cholesky; factor=explicit-scalar-cholesky; "
  "audit=canonical-Kq; projection=0.5*(Kq+Kq.T); "
  "pivot>1e-12*infinity-norm; solve=explicit-forward-back-substitution; "
  "cache=bitwise-constant"
)


@final
@dataclass(frozen=True, slots=True)
class LinearStatic:
  """The exact zero-field Phase 1 linear-static request."""

  def __init_subclass__(cls, **kwargs: object) -> None:
    del cls, kwargs
    msg = "LinearStatic is runtime-final and cannot be subclassed"
    raise TypeError(msg)


def linear_static_request_manifest(request: object) -> CanonicalManifest:
  """Return the versioned numeric and verification policy for the exact request."""
  if type(request) is not LinearStatic:
    msg = "analysis request must be exactly LinearStatic"
    raise TypeError(msg)
  return CanonicalManifest.capture(
    {
      "schema": LINEAR_STATIC_REQUEST_SCHEMA,
      "floating_dtype": "float64",
      "backend": {
        "family": "private-dense-symmetric-cholesky-reference",
        "audit_operator": "original-canonical-reduced-coo",
        "solver_projection": "0.5*(K_q+K_q.T)",
        "factorization": "explicit-scalar-cholesky",
        "triangular_solve": "explicit-forward-back-substitution",
        "symmetry_epsilon_factor": LINEAR_STATIC_SYMMETRY_EPSILON_FACTOR,
        "unscaled_pivot_ratio": LINEAR_STATIC_CHOLESKY_PIVOT_RATIO,
        "workspace_budget_bytes": LINEAR_STATIC_WORKSPACE_BUDGET_BYTES,
        "workspace_budget_scope": LINEAR_STATIC_WORKSPACE_BUDGET_SCOPE,
        "constant_operator_reuse": "bitwise-identical-only",
      },
      "verification": {
        "norm": "infinity",
        "relative_tolerance": LINEAR_STATIC_VERIFICATION_TOLERANCE,
        "unit_floor": False,
        "zero_scale": "exact-zero-error",
      },
    }
  )


@dataclass(frozen=True, slots=True, eq=False)
class EvolutionFieldLayout:
  """One participating field and its exact algebraic classification."""

  field_id: str | int
  classification: str
  global_size: int


@dataclass(frozen=True, slots=True, eq=False)
class EvolutionLayout:
  """Request-owned evolution storage requirements."""

  schema: str
  fields: tuple[EvolutionFieldLayout, ...]
  evolving_value_count: int


@dataclass(frozen=True, slots=True, eq=False)
class PreparedCapabilities:
  """Conservative guarantees admitted by the Phase 1 analysis."""

  response_class: str
  tangent_is_symmetric: bool
  tangent_is_constant: bool
  state_dependent: bool
  zero_width_history: bool
  fixed_constraint_topology: bool
  fixed_load_topology: bool


@dataclass(frozen=True, slots=True, eq=False)
class LinearPredictor:
  """Typed full-space predictor bound to one linear step transaction."""

  full_increment: FinalizedArray


@dataclass(frozen=True, slots=True, eq=False)
class StepTransaction:
  """Immutable attempt binding one accepted base to one target point."""

  transaction_id: InstanceId
  prepared_instance_id: InstanceId
  model_instance_id: InstanceId
  model_content_fingerprint: ContentFingerprint
  program_instance_id: InstanceId
  program_content_fingerprint: ContentFingerprint
  plan_content_fingerprint: ContentFingerprint
  request_manifest: CanonicalManifest
  base_state: CommittedAnalysisState
  base_generation: StateGeneration
  target_evaluation: ProgramEvaluation
  retry: int
  cutback: int
  predictor: LinearPredictor


@dataclass(frozen=True, slots=True, eq=False)
class LinearConvergenceRecord:
  """Detached convergence and private-backend outcome, never a proof cache."""

  converged: bool
  iteration_count: int
  factorization_bypassed: bool
  operator_infinity_norm: float
  minimum_unscaled_pivot: float | None
  reduced_residual_norm: float
  verification_tolerance: float
  backend_policy: str


@dataclass(frozen=True, slots=True, eq=False)
class TrialAnalysisState:
  """One detached candidate produced from an exact transaction."""

  trial_id: InstanceId
  transaction_id: InstanceId
  prepared_instance_id: InstanceId
  base_generation: StateGeneration
  candidate_generation: StateGeneration
  candidate_state: CommittedAnalysisState
  ledger: LinearBalanceLedger
  convergence: LinearConvergenceRecord
  candidate_verification: VerificationReport


@dataclass(frozen=True, slots=True, eq=False)
class AcceptedTransition:
  """Immutable evidence for one successful atomic acceptance."""

  transaction: StepTransaction
  prepared_instance_id: InstanceId
  transaction_id: InstanceId
  trial_id: InstanceId
  base_generation: StateGeneration
  candidate_generation: StateGeneration
  base_primary_values: FinalizedArray
  actual_increment: FinalizedArray
  target_evaluation: ProgramEvaluation
  committed_state: CommittedAnalysisState
  accepted: bool


@dataclass(frozen=True, slots=True)
class WorkspaceStatistics:
  """Non-authoritative counters for bounded performance evidence."""

  reduced_dof_count: int
  checked_workspace_bytes: int
  evaluation_count: int
  factorization_count: int
  factorization_reuse_count: int
  zero_free_bypass_count: int
