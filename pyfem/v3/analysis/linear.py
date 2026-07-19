"""Prepared ownership, dense reference solve, and one-accept linear flow."""

from __future__ import annotations

import math
from dataclasses import dataclass
from threading import RLock
from typing import NoReturn, final

import numpy as np

from pyfem.v3.analysis.contracts import (
  LINEAR_STATIC_BACKEND_POLICY,
  LINEAR_STATIC_CHOLESKY_PIVOT_RATIO,
  LINEAR_STATIC_SYMMETRY_EPSILON_FACTOR,
  LINEAR_STATIC_VERIFICATION_TOLERANCE,
  LINEAR_STATIC_WORKSPACE_BUDGET_BYTES,
  AcceptedTransition,
  EvolutionFieldLayout,
  EvolutionLayout,
  LinearConvergenceRecord,
  LinearPredictor,
  LinearStatic,
  PreparedCapabilities,
  StepTransaction,
  TrialAnalysisState,
  WorkspaceStatistics,
  linear_static_request_manifest,
)
from pyfem.v3.analysis.diagnostics import (
  AnalysisDiagnostic,
  AnalysisPreparationError,
  AnalysisSolveError,
  StateTransactionError,
)
from pyfem.v3.analysis.numerics import _explicit_cholesky, _strict_ratio_greater
from pyfem.v3.assembly import (
  AssemblyEvaluationError,
  AssemblyPreparationError,
  LinearStaticContributionRequest,
  LinearStaticContributions,
  PreparedAssemblyPlan,
  assemble_reference_linear,
  prepare_assembly_plan,
)
from pyfem.v3.compile import ProgramEvaluationError, evaluate_program
from pyfem.v3.model import (
  EVOLUTION_STATE_SCHEMA,
  PHYSICAL_STATE_SCHEMA,
  PROGRAM_HISTORY_SCHEMA,
  CanonicalManifest,
  CommittedAnalysisState,
  CompiledModel,
  CompiledProgram,
  ContentFingerprint,
  EvolutionState,
  FinalizedArray,
  InstanceId,
  PhysicalState,
  ProgramHistory,
  StateGeneration,
  require_generation_successor,
  require_same_generation,
  require_same_instance,
)
from pyfem.v3.results.diagnostics import SolutionVerificationError
from pyfem.v3.results.solution import Solution
from pyfem.v3.results.verification import (
  build_balance_ledger,
  clone_balance_ledger,
  clone_program_evaluation,
  dense_operator,
  fresh_verify,
  prolongation,
  validate_state_record,
  verify_record_data,
)
from pyfem.v3.spec import ProgramCoordinateValue, ProgramPoint

_FLOAT64 = np.dtype(np.float64)
_EVOLUTION_LAYOUT_SCHEMA = "pyfem-v3-evolution-layout-linear-static-v1"
_PREPARED_CONSTRUCTION_TOKEN = object()


def _preparation_failure(code: str, message: str) -> NoReturn:
  raise AnalysisPreparationError((AnalysisDiagnostic(code, message),))


def _solve_failure(code: str, message: str) -> NoReturn:
  raise AnalysisSolveError((AnalysisDiagnostic(code, message),))


def _transaction_failure(code: str, message: str) -> NoReturn:
  raise StateTransactionError((AnalysisDiagnostic(code, message),))


def _manifest_equal(left: CanonicalManifest, right: CanonicalManifest) -> bool:
  try:
    return left.to_bytes() == right.to_bytes()
  except (OverflowError, RecursionError, TypeError, ValueError):
    return False


def _generation_key(generation: StateGeneration) -> tuple[object, int]:
  if type(generation) is not StateGeneration:
    _transaction_failure(
      "invalid-state-generation",
      "state generation must be exactly StateGeneration",
    )
  return generation._lineage, generation.ordinal


def _instance_key(instance_id: InstanceId) -> object:
  if type(instance_id) is not InstanceId:
    _transaction_failure(
      "invalid-live-identity",
      "transaction identity must be exactly InstanceId",
    )
  return instance_id._token


def _checked_workspace_bytes(reduced_count: int) -> int:
  if type(reduced_count) is not int or reduced_count < 0:
    _preparation_failure(
      "invalid-reduced-size",
      "reduced DOF count must be a nonnegative exact integer",
    )
  matrix_entries = reduced_count * reduced_count
  # Direct retention owns three cached matrices; one fresh audit may coexist on
  # reuse.  The explicit triangular solve peaks at four vector arrays
  # (RHS/pivots/forward/solution); residual construction peaks at five
  # (RHS/pivots/solution/matvec/residual).  Six float64 vector slots therefore
  # remain conservative without any opaque matrix-sized solve scratch.
  value_count = 4 * matrix_entries + 6 * reduced_count
  byte_count = value_count * _FLOAT64.itemsize
  if byte_count > LINEAR_STATIC_WORKSPACE_BUDGET_BYTES:
    _preparation_failure(
      "linear-backend-capacity-exceeded",
      "dense audit, projection, factor, and scratch exceed the frozen "
      "256 MiB workspace budget",
    )
  return byte_count


def _infinity_norm(value: np.ndarray) -> float:
  if value.size == 0:
    return 0.0
  if value.ndim == 1:
    return float(np.max(np.abs(value)))
  largest = 0.0
  for row in value:
    row_max = float(np.max(np.abs(row))) if row.size else 0.0
    if row_max == 0.0:
      row_sum = 0.0
    else:
      scaled = math.fsum(float(abs(item) / row_max) for item in row)
      row_sum = row_max * scaled
      if not math.isfinite(row_sum):
        row_sum = math.inf
    largest = max(largest, row_sum)
  return largest


def _all_finite(value: np.ndarray) -> bool:
  """Check dense arrays without allocating an array-shaped boolean mask."""
  return all(math.isfinite(float(item)) for item in value.flat)


def _bitwise_equal(left: np.ndarray, right: np.ndarray) -> bool:
  return (
    left.shape == right.shape
    and left.dtype == right.dtype
    and all(
      int(left_item) == int(right_item)
      for left_item, right_item in zip(
        left.view(np.uint64).flat,
        right.view(np.uint64).flat,
        strict=True,
      )
    )
  )


def _bounded_cholesky_solve(
  factor: np.ndarray,
  rhs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  """Solve one triangular pair with exactly two owned vector outputs."""
  if (
    type(factor) is not np.ndarray
    or type(rhs) is not np.ndarray
    or factor.dtype != _FLOAT64
    or rhs.dtype != _FLOAT64
    or factor.ndim != 2
    or factor.shape[0] != factor.shape[1]
    or rhs.shape != (factor.shape[0],)
    or not _all_finite(factor)
    or not _all_finite(rhs)
  ):
    _solve_failure(
      "malformed-cholesky-solve-input",
      "explicit triangular solve requires finite float64 factor and RHS arrays",
    )
  reduced_count = rhs.shape[0]
  forward = np.empty(reduced_count, dtype=np.float64)
  coordinates = np.empty(reduced_count, dtype=np.float64)
  for row in range(reduced_count):
    diagonal = float(factor[row, row])
    if not math.isfinite(diagonal) or diagonal <= 0.0:
      _solve_failure(
        "malformed-cholesky-factor",
        "explicit forward substitution requires positive finite diagonals",
      )
    remainder = float(rhs[row])
    for column in range(row):
      remainder -= float(factor[row, column]) * float(forward[column])
    quotient = remainder / diagonal
    if not math.isfinite(quotient):
      _solve_failure(
        "nonfinite-forward-substitution",
        "explicit forward substitution produced a nonfinite value",
      )
    forward[row] = quotient
  for row in range(reduced_count - 1, -1, -1):
    diagonal = float(factor[row, row])
    if not math.isfinite(diagonal) or diagonal <= 0.0:
      _solve_failure(
        "malformed-cholesky-factor",
        "explicit back substitution requires positive finite diagonals",
      )
    remainder = float(forward[row])
    for column in range(row + 1, reduced_count):
      remainder -= float(factor[column, row]) * float(coordinates[column])
    quotient = remainder / diagonal
    if not math.isfinite(quotient):
      _solve_failure(
        "nonfinite-back-substitution",
        "explicit back substitution produced a nonfinite value",
      )
    coordinates[row] = quotient
  return forward, coordinates


def _program_point(evaluation: object) -> ProgramPoint:
  if type(evaluation) is not type(clone_program_evaluation(evaluation)):
    _transaction_failure(
      "invalid-program-evaluation",
      "transaction target evaluation has a foreign type",
    )
  return ProgramPoint(
    tuple(
      ProgramCoordinateValue(name, float(value))
      for name, value in zip(
        evaluation.coordinate_names,
        evaluation.coordinate_values.values,
        strict=True,
      )
    )
  )


def _same_program_evaluation(left: object, right: object) -> bool:
  try:
    require_same_instance(left.program_instance_id, right.program_instance_id)
    require_same_instance(
      left.compatible_model_instance_id,
      right.compatible_model_instance_id,
    )
    return (
      left.program_content_fingerprint == right.program_content_fingerprint
      and left.compatible_model_content_fingerprint
      == right.compatible_model_content_fingerprint
      and left.coordinate_names == right.coordinate_names
      and np.array_equal(left.coordinate_values.values, right.coordinate_values.values)
      and np.array_equal(
        left.prescribed_offsets.values, right.prescribed_offsets.values
      )
      and np.array_equal(
        left.prescribed_offset_derivatives.values,
        right.prescribed_offset_derivatives.values,
      )
      and np.array_equal(left.nodal_force.values, right.nodal_force.values)
      and np.array_equal(
        left.nodal_force_derivatives.values,
        right.nodal_force_derivatives.values,
      )
    )
  except (AttributeError, TypeError, ValueError):
    return False


def _copy_state(
  state: CommittedAnalysisState,
  *,
  prepared_instance_id: InstanceId,
) -> CommittedAnalysisState:
  physical = state.physical
  evolution = state.evolution
  history = state.program_history
  copied_physical = PhysicalState(
    model_instance_id=physical.model_instance_id,
    model_content_fingerprint=physical.model_content_fingerprint,
    schema=physical.schema,
    generation=state.generation,
    primary_values=FinalizedArray(physical.primary_values.values, dtype=np.float64),
    material_histories=tuple(
      FinalizedArray(item.values, dtype=np.float64)
      for item in physical.material_histories
    ),
    formulation_histories=tuple(
      FinalizedArray(item.values, dtype=np.float64)
      for item in physical.formulation_histories
    ),
  )
  copied_evolution = EvolutionState(
    prepared_instance_id=prepared_instance_id,
    request_manifest=evolution.request_manifest,
    schema=evolution.schema,
    generation=state.generation,
    program_evaluation=clone_program_evaluation(evolution.program_evaluation),
    algebraic_field_ids=tuple(evolution.algebraic_field_ids),
    accepted_step_index=evolution.accepted_step_index,
    predictor=FinalizedArray(evolution.predictor.values, dtype=np.float64),
    actual_increment=FinalizedArray(
      evolution.actual_increment.values,
      dtype=np.float64,
    ),
  )
  copied_history = ProgramHistory(
    program_instance_id=history.program_instance_id,
    program_content_fingerprint=history.program_content_fingerprint,
    schema=history.schema,
    generation=state.generation,
    entries=(),
  )
  return CommittedAnalysisState(
    prepared_instance_id=prepared_instance_id,
    generation=state.generation,
    physical=copied_physical,
    evolution=copied_evolution,
    program_history=copied_history,
  )


@dataclass(slots=True)
class _TransactionRecord:
  transaction: StepTransaction
  trials: dict[object, TrialAnalysisState]
  discarded_trials: set[object]
  closed: bool = False


@dataclass(slots=True)
class _LinearWorkspace:
  reduced_dof_count: int
  checked_workspace_bytes: int
  lock: RLock
  audit_operator: np.ndarray | None = None
  solve_operator: np.ndarray | None = None
  factor: np.ndarray | None = None
  evaluation_count: int = 0
  factorization_count: int = 0
  factorization_reuse_count: int = 0
  zero_free_bypass_count: int = 0


def _validate_capabilities(model: CompiledModel, program: CompiledProgram) -> None:
  model_capabilities = model.capabilities
  program_capabilities = program.capabilities
  if (
    model_capabilities.response_class != "linear-elastic"
    or not model_capabilities.fixed_model_coupling
    or model_capabilities.tangent_class != "symmetric-constant-material"
    or not model_capabilities.tangent_is_symmetric
    or not model_capabilities.tangent_is_constant
    or not model_capabilities.conservative_internal_contribution
    or model_capabilities.state_dependent
    or model_capabilities.has_storage
    or model_capabilities.has_mass
    or model_capabilities.has_damping
    or model_capabilities.restart_history_required
  ):
    _preparation_failure(
      "unsupported-linear-capabilities",
      "LinearStatic requires the exact constant symmetric stateless Q8 "
      "capability slice",
    )
  if (
    not program_capabilities.fixed_constraint_topology
    or not program_capabilities.fixed_load_topology
    or program_capabilities.state_dependent
    or program_capabilities.has_follower_loads
    or program_capabilities.has_interaction_tangent
    or program_capabilities.has_program_state
  ):
    _preparation_failure(
      "unsupported-program-capabilities",
      "LinearStatic requires fixed affine constraints and fixed nodal-load topology",
    )
  layout = model.physical_state_layout
  if layout.evolving_value_count != 0 or any(
    item.material_history_width != 0 or item.formulation_history_width != 0
    for item in layout.block_states
  ):
    _preparation_failure(
      "unsupported-state-capabilities",
      "Phase 1 admits only exact zero-width material and formulation history",
    )


@final
class PreparedAnalysis:
  """Sole live owner of Phase 1 transactions and private solver workspace."""

  __slots__ = (
    "instance_id",
    "model",
    "program",
    "request",
    "request_manifest",
    "assembly_plan",
    "evolution_layout",
    "capabilities",
    "_model_owner",
    "_model_content_fingerprint",
    "_program_owner",
    "_program_content_fingerprint",
    "_plan_owner",
    "_plan_content_fingerprint",
    "_lock",
    "_workspace",
    "_transactions",
    "_issued_states",
    "_consumed_generations",
    "_sealed",
  )

  def __init_subclass__(cls, **kwargs: object) -> None:
    del cls, kwargs
    msg = "PreparedAnalysis is runtime-final and cannot be subclassed"
    raise TypeError(msg)

  def __setattr__(self, name: str, value: object) -> None:
    try:
      sealed = object.__getattribute__(self, "_sealed")
    except AttributeError:
      sealed = False
    if sealed:
      msg = "PreparedAnalysis semantic ownership is immutable after preparation"
      raise AttributeError(msg)
    object.__setattr__(self, name, value)

  def __init__(
    self,
    model: CompiledModel,
    program: CompiledProgram,
    request: LinearStatic,
    plan: PreparedAssemblyPlan,
    *,
    _token: object,
  ) -> None:
    if _token is not _PREPARED_CONSTRUCTION_TOKEN:
      _preparation_failure(
        "invalid-prepared-construction",
        "PreparedAnalysis must be created by prepare_analysis",
      )
    _validate_capabilities(model, program)
    request_manifest = linear_static_request_manifest(request)
    reduced_count = plan.domain_coo_plan.reduced_shape[0]
    checked_bytes = _checked_workspace_bytes(reduced_count)
    self.instance_id = InstanceId()
    self.model = model
    self.program = program
    self.request = request
    self.request_manifest = request_manifest
    self.assembly_plan = plan
    self.evolution_layout = EvolutionLayout(
      schema=_EVOLUTION_LAYOUT_SCHEMA,
      fields=tuple(
        EvolutionFieldLayout(
          field_id=item.field_id,
          classification="algebraic",
          global_size=item.global_size,
        )
        for item in model.physical_state_layout.primary_fields
      ),
      evolving_value_count=0,
    )
    self.capabilities = PreparedCapabilities(
      response_class="linear-static",
      tangent_is_symmetric=True,
      tangent_is_constant=True,
      state_dependent=False,
      zero_width_history=True,
      fixed_constraint_topology=True,
      fixed_load_topology=True,
    )
    self._model_owner = model
    self._model_content_fingerprint = model.content_fingerprint
    self._program_owner = program
    self._program_content_fingerprint = program.content_fingerprint
    self._plan_owner = plan
    self._plan_content_fingerprint = plan.content_fingerprint
    self._lock = RLock()
    self._workspace = _LinearWorkspace(
      reduced_dof_count=reduced_count,
      checked_workspace_bytes=checked_bytes,
      lock=RLock(),
    )
    self._transactions: dict[object, _TransactionRecord] = {}
    self._issued_states: dict[tuple[object, int], CommittedAnalysisState] = {}
    self._consumed_generations: set[tuple[object, int]] = set()
    self._sealed = True

  def _validate_owner(self) -> None:
    """Fail before workspace use if prepared semantic ownership was forged."""
    try:
      expected_manifest = linear_static_request_manifest(self.request)
      expected_fields = tuple(
        (item.field_id, "algebraic", item.global_size)
        for item in self.model.physical_state_layout.primary_fields
      )
      actual_fields = tuple(
        (item.field_id, item.classification, item.global_size)
        for item in self.evolution_layout.fields
        if type(item) is EvolutionFieldLayout
      )
      if (
        type(self.instance_id) is not InstanceId
        or type(self.model) is not CompiledModel
        or self.model is not self._model_owner
        or self.model.content_fingerprint != self._model_content_fingerprint
        or type(self.program) is not CompiledProgram
        or self.program is not self._program_owner
        or self.program.content_fingerprint != self._program_content_fingerprint
        or type(self.assembly_plan) is not PreparedAssemblyPlan
        or self.assembly_plan is not self._plan_owner
        or self.assembly_plan.content_fingerprint != self._plan_content_fingerprint
        or not _manifest_equal(self.request_manifest, expected_manifest)
        or type(self.assembly_plan.request) is not LinearStaticContributionRequest
        or self.assembly_plan.compatible_model_content_fingerprint
        != self.model.content_fingerprint
        or self.assembly_plan.compatible_program_content_fingerprint
        != self.program.content_fingerprint
      ):
        raise ValueError
      require_same_instance(
        self._model_owner.instance_id,
        self.model.instance_id,
      )
      require_same_instance(
        self._program_owner.instance_id,
        self.program.instance_id,
      )
      require_same_instance(
        self.model.instance_id,
        self.program.compatible_model_instance_id,
      )
      require_same_instance(
        self.model.instance_id,
        self.assembly_plan.compatible_model_instance_id,
      )
      require_same_instance(
        self.program.instance_id,
        self.assembly_plan.compatible_program_instance_id,
      )
      if (
        type(self.evolution_layout) is not EvolutionLayout
        or self.evolution_layout.schema != _EVOLUTION_LAYOUT_SCHEMA
        or actual_fields != expected_fields
        or len(actual_fields) != len(self.evolution_layout.fields)
        or self.evolution_layout.evolving_value_count != 0
        or type(self.capabilities) is not PreparedCapabilities
        or self.capabilities.response_class != "linear-static"
        or not self.capabilities.tangent_is_symmetric
        or not self.capabilities.tangent_is_constant
        or self.capabilities.state_dependent
        or not self.capabilities.zero_width_history
        or not self.capabilities.fixed_constraint_topology
        or not self.capabilities.fixed_load_topology
        or type(self._workspace) is not _LinearWorkspace
        or self._workspace.reduced_dof_count
        != self.assembly_plan.domain_coo_plan.reduced_shape[0]
        or self._workspace.checked_workspace_bytes
        != _checked_workspace_bytes(self._workspace.reduced_dof_count)
      ):
        raise ValueError
    except (AttributeError, TypeError, ValueError):
      _transaction_failure(
        "malformed-prepared-analysis",
        "prepared semantic owners, request, plan, or capabilities are not exact",
      )

  @property
  def plan_content_fingerprint(self) -> ContentFingerprint:
    """Return the immutable prepared assembly-plan content identity."""
    return self.assembly_plan.content_fingerprint

  def workspace_statistics(self) -> WorkspaceStatistics:
    """Return counters without exposing cache or factorization storage."""
    self._validate_owner()
    with self._workspace.lock:
      return WorkspaceStatistics(
        reduced_dof_count=self._workspace.reduced_dof_count,
        checked_workspace_bytes=self._workspace.checked_workspace_bytes,
        evaluation_count=self._workspace.evaluation_count,
        factorization_count=self._workspace.factorization_count,
        factorization_reuse_count=self._workspace.factorization_reuse_count,
        zero_free_bypass_count=self._workspace.zero_free_bypass_count,
      )

  def _validate_issued_state(self, state: object) -> CommittedAnalysisState:
    try:
      validate_state_record(
        state=state,
        model=self.model,
        program=self.program,
        prepared_instance_id=self.instance_id,
        request_manifest=self.request_manifest,
      )
    except SolutionVerificationError as error:
      _transaction_failure("invalid-committed-state", str(error))
    key = _generation_key(state.generation)
    with self._lock:
      issued = self._issued_states.get(key)
      if issued is not state:
        _transaction_failure(
          "foreign-or-forged-state",
          "state must be the exact immutable snapshot issued by this prepared analysis",
        )
    return state

  def initialize(self, *, point: ProgramPoint) -> CommittedAnalysisState:
    """Create one fresh explicit generation-zero lineage at an exact point."""
    self._validate_owner()
    try:
      evaluation = evaluate_program(self.program, point)
    except ProgramEvaluationError as error:
      _transaction_failure("invalid-initial-point", str(error))
    generation = StateGeneration.initial()
    full_count = self.model.physical_state_layout.global_primary_size
    physical = PhysicalState(
      model_instance_id=self.model.instance_id,
      model_content_fingerprint=self.model.content_fingerprint,
      schema=PHYSICAL_STATE_SCHEMA,
      generation=generation,
      primary_values=FinalizedArray(
        evaluation.prescribed_offsets.values,
        dtype=np.float64,
      ),
      material_histories=tuple(
        FinalizedArray(
          np.zeros(item.material_history_shape, dtype=np.float64),
          dtype=np.float64,
        )
        for item in self.model.physical_state_layout.block_states
      ),
      formulation_histories=tuple(
        FinalizedArray(
          np.zeros(item.formulation_history_shape, dtype=np.float64),
          dtype=np.float64,
        )
        for item in self.model.physical_state_layout.block_states
      ),
    )
    evolution = EvolutionState(
      prepared_instance_id=self.instance_id,
      request_manifest=self.request_manifest,
      schema=EVOLUTION_STATE_SCHEMA,
      generation=generation,
      program_evaluation=clone_program_evaluation(evaluation),
      algebraic_field_ids=tuple(item.field_id for item in self.evolution_layout.fields),
      accepted_step_index=0,
      predictor=FinalizedArray(np.zeros(full_count), dtype=np.float64),
      actual_increment=FinalizedArray(np.zeros(full_count), dtype=np.float64),
    )
    history = ProgramHistory(
      program_instance_id=self.program.instance_id,
      program_content_fingerprint=self.program.content_fingerprint,
      schema=PROGRAM_HISTORY_SCHEMA,
      generation=generation,
      entries=(),
    )
    state = CommittedAnalysisState(
      prepared_instance_id=self.instance_id,
      generation=generation,
      physical=physical,
      evolution=evolution,
      program_history=history,
    )
    try:
      validate_state_record(
        state=state,
        model=self.model,
        program=self.program,
        prepared_instance_id=self.instance_id,
        request_manifest=self.request_manifest,
      )
    except SolutionVerificationError as error:
      _transaction_failure("initial-state-construction-failed", str(error))
    with self._lock:
      self._issued_states[_generation_key(generation)] = state
    return state

  def begin_step(
    self,
    *,
    initial: CommittedAnalysisState,
    point: ProgramPoint,
  ) -> StepTransaction:
    """Bind one exact initialized base and target without touching workspace."""
    self._validate_owner()
    base = self._validate_issued_state(initial)
    if base.generation.ordinal != 0 or base.evolution.accepted_step_index != 0:
      _transaction_failure(
        "unsupported-linear-continuation",
        "Phase 1 direct linear solve accepts exactly one transition from "
        "generation zero",
      )
    base_key = _generation_key(base.generation)
    with self._lock:
      if base_key in self._consumed_generations:
        _transaction_failure(
          "stale-base-generation",
          "the exact base generation has already been consumed",
        )
    try:
      target = evaluate_program(self.program, point)
    except ProgramEvaluationError as error:
      _transaction_failure("invalid-target-point", str(error))
    with np.errstate(over="ignore", invalid="ignore"):
      predictor_values = (
        target.prescribed_offsets.values - base.physical.primary_values.values
      )
    if not bool(np.isfinite(predictor_values).all()):
      _transaction_failure(
        "nonfinite-linear-predictor",
        "linear predictor must remain finite",
      )
    transaction = StepTransaction(
      transaction_id=InstanceId(),
      prepared_instance_id=self.instance_id,
      model_instance_id=self.model.instance_id,
      model_content_fingerprint=self.model.content_fingerprint,
      program_instance_id=self.program.instance_id,
      program_content_fingerprint=self.program.content_fingerprint,
      plan_content_fingerprint=self.assembly_plan.content_fingerprint,
      request_manifest=self.request_manifest,
      base_state=base,
      base_generation=base.generation,
      target_evaluation=clone_program_evaluation(target),
      retry=0,
      cutback=0,
      predictor=LinearPredictor(FinalizedArray(predictor_values, dtype=np.float64)),
    )
    with self._lock:
      if base_key in self._consumed_generations:
        _transaction_failure(
          "stale-base-generation",
          "the exact base generation was consumed before transaction registration",
        )
      self._transactions[_instance_key(transaction.transaction_id)] = (
        _TransactionRecord(transaction, {}, set())
      )
    return transaction

  def _registered_transaction(
    self,
    transaction: object,
    *,
    require_open: bool = True,
  ) -> _TransactionRecord:
    if type(transaction) is not StepTransaction:
      _transaction_failure(
        "foreign-or-forged-transaction",
        "transaction must be exactly StepTransaction",
      )
    key = _instance_key(transaction.transaction_id)
    with self._lock:
      record = self._transactions.get(key)
      if record is None or record.transaction is not transaction:
        _transaction_failure(
          "foreign-or-forged-transaction",
          "transaction was not issued by this prepared analysis",
        )
      if require_open and record.closed:
        _transaction_failure(
          "closed-transaction",
          "transaction is already closed",
        )
      if _generation_key(transaction.base_generation) in self._consumed_generations:
        _transaction_failure(
          "stale-base-generation",
          "transaction base generation has already been consumed",
        )
    return record

  def _solve_reduced(
    self,
    contributions: LinearStaticContributions,
  ) -> tuple[np.ndarray, LinearConvergenceRecord]:
    reduced_count = self._workspace.reduced_dof_count
    try:
      if (
        type(contributions) is not LinearStaticContributions
        or contributions.reduced_operator.shape != (reduced_count, reduced_count)
        or contributions.reduced_rhs.values.shape != (reduced_count,)
      ):
        raise ValueError
      operator = dense_operator(contributions.reduced_operator)
      rhs = np.array(
        contributions.reduced_rhs.values,
        dtype=np.float64,
        copy=True,
      )
    except (AttributeError, IndexError, TypeError, ValueError):
      _solve_failure(
        "malformed-reduced-system",
        "canonical reduced operator and RHS do not match the prepared plan",
      )
    if (
      operator.shape != (reduced_count, reduced_count)
      or rhs.shape != (reduced_count,)
      or not _all_finite(operator)
      or not _all_finite(rhs)
    ):
      _solve_failure(
        "malformed-or-nonfinite-reduced-system",
        "canonical reduced operator and RHS must be finite with exact prepared shapes",
      )
    if reduced_count == 0:
      with self._workspace.lock:
        self._workspace.evaluation_count += 1
        self._workspace.zero_free_bypass_count += 1
      return (
        np.empty(0, dtype=np.float64),
        LinearConvergenceRecord(
          converged=True,
          iteration_count=1,
          factorization_bypassed=True,
          operator_infinity_norm=0.0,
          minimum_unscaled_pivot=None,
          reduced_residual_norm=0.0,
          verification_tolerance=LINEAR_STATIC_VERIFICATION_TOLERANCE,
          backend_policy=LINEAR_STATIC_BACKEND_POLICY,
        ),
      )

    with self._workspace.lock:
      self._workspace.evaluation_count += 1
      if self._workspace.audit_operator is None:
        maximum = max(abs(float(item)) for item in operator.flat)
        asymmetry = max(
          (
            abs(float(operator[row, column]) - float(operator[column, row]))
            for row in range(reduced_count)
            for column in range(row + 1, reduced_count)
          ),
          default=0.0,
        )
        bound = (
          0.0
          if maximum == 0.0
          else LINEAR_STATIC_SYMMETRY_EPSILON_FACTOR
          * np.finfo(np.float64).eps
          * maximum
        )
        if asymmetry > bound:
          _solve_failure(
            "reduced-operator-asymmetry",
            "canonical K_q exceeds the frozen 64-eps symmetry admission bound",
          )
        solve_operator = np.array(operator, dtype=np.float64, copy=True)
        np.add(solve_operator, operator.T, out=solve_operator)
        solve_operator *= 0.5
        if not _all_finite(solve_operator):
          _solve_failure(
            "nonfinite-solver-projection",
            "symmetric solver projection must remain finite",
          )
        operator_scale = _infinity_norm(solve_operator)
        if not math.isfinite(operator_scale) or operator_scale <= 0.0:
          _solve_failure(
            "singular-reduced-operator",
            "nonempty symmetric reduced operator must have positive finite "
            "infinity norm",
          )
        factorization = _explicit_cholesky(solve_operator)
        if factorization is None:
          _solve_failure(
            "cholesky-factorization-failed",
            "reduced operator is indefinite or singular under the frozen "
            "Cholesky policy",
          )
        factor, pivots = factorization
        if not _all_finite(pivots) or any(
          not _strict_ratio_greater(
            float(pivot),
            operator_scale,
            LINEAR_STATIC_CHOLESKY_PIVOT_RATIO,
          )
          for pivot in pivots
        ):
          _solve_failure(
            "near-singular-reduced-operator",
            "an unscaled Cholesky pivot does not exceed 1e-12 times the "
            "operator infinity norm",
          )
        operator.setflags(write=False)
        solve_operator.setflags(write=False)
        factor.setflags(write=False)
        self._workspace.audit_operator = operator
        self._workspace.solve_operator = solve_operator
        self._workspace.factor = factor
        self._workspace.factorization_count += 1
      else:
        if not _bitwise_equal(operator, self._workspace.audit_operator):
          _solve_failure(
            "constant-operator-changed",
            "constant tangent changed under unchanged exact prepared identities",
          )
        if self._workspace.factor is None or self._workspace.solve_operator is None:
          _solve_failure(
            "corrupt-factorization-cache",
            "constant-operator cache is incomplete",
          )
        factor = self._workspace.factor
        solve_operator = self._workspace.solve_operator
        operator_scale = _infinity_norm(solve_operator)
        pivots = np.square(np.diag(factor))
        if (
          factor.shape != operator.shape
          or solve_operator.shape != operator.shape
          or not _all_finite(factor)
          or not _all_finite(solve_operator)
          or not _all_finite(pivots)
          or not math.isfinite(operator_scale)
          or operator_scale <= 0.0
          or any(
            not _strict_ratio_greater(
              float(pivot),
              operator_scale,
              LINEAR_STATIC_CHOLESKY_PIVOT_RATIO,
            )
            for pivot in pivots
          )
        ):
          _solve_failure(
            "corrupt-factorization-cache",
            "cached solver projection or Cholesky factor violates its policy",
          )
        self._workspace.factorization_reuse_count += 1
      minimum_pivot = float(np.min(pivots))
      forward, coordinates = _bounded_cholesky_solve(factor, rhs)
      del forward
      if not _all_finite(coordinates):
        _solve_failure(
          "nonfinite-reduced-solution",
          "reduced solution contains nonfinite values",
        )
      audit_residual = rhs - operator @ coordinates
      residual_norm = _infinity_norm(audit_residual)
      convergence = LinearConvergenceRecord(
        converged=True,
        iteration_count=1,
        factorization_bypassed=False,
        operator_infinity_norm=operator_scale,
        minimum_unscaled_pivot=minimum_pivot,
        reduced_residual_norm=residual_norm,
        verification_tolerance=LINEAR_STATIC_VERIFICATION_TOLERANCE,
        backend_policy=LINEAR_STATIC_BACKEND_POLICY,
      )
    return coordinates, convergence

  def evaluate_trial(self, transaction: StepTransaction) -> TrialAnalysisState:
    """Evaluate, solve, and freshly verify one detached candidate."""
    self._validate_owner()
    record = self._registered_transaction(transaction)
    point = _program_point(transaction.target_evaluation)
    try:
      contributions = assemble_reference_linear(
        self.model,
        self.program,
        self.assembly_plan,
        point,
      )
    except AssemblyEvaluationError as error:
      _solve_failure("assembly-evaluation-failed", str(error))
    if not _same_program_evaluation(
      transaction.target_evaluation,
      contributions.program_evaluation,
    ):
      _solve_failure(
        "target-evaluation-drift",
        "fresh assembly evaluation changed the exact bound program point",
      )
    reduced_coordinates, convergence = self._solve_reduced(contributions)
    projection = prolongation(self.assembly_plan)
    primary = (
      projection @ reduced_coordinates
      + contributions.program_evaluation.prescribed_offsets.values
    )
    if not bool(np.isfinite(primary).all()):
      _solve_failure("nonfinite-primary-solution", "full primary solution is nonfinite")
    base = transaction.base_state
    actual_increment = primary - base.physical.primary_values.values
    if not bool(np.isfinite(actual_increment).all()):
      _solve_failure(
        "nonfinite-primary-increment", "actual full increment is nonfinite"
      )
    candidate_generation = transaction.base_generation.next_accepted()
    physical = PhysicalState(
      model_instance_id=self.model.instance_id,
      model_content_fingerprint=self.model.content_fingerprint,
      schema=PHYSICAL_STATE_SCHEMA,
      generation=candidate_generation,
      primary_values=FinalizedArray(primary, dtype=np.float64),
      material_histories=tuple(
        FinalizedArray(item.values, dtype=np.float64)
        for item in base.physical.material_histories
      ),
      formulation_histories=tuple(
        FinalizedArray(item.values, dtype=np.float64)
        for item in base.physical.formulation_histories
      ),
    )
    evolution = EvolutionState(
      prepared_instance_id=self.instance_id,
      request_manifest=self.request_manifest,
      schema=EVOLUTION_STATE_SCHEMA,
      generation=candidate_generation,
      program_evaluation=clone_program_evaluation(contributions.program_evaluation),
      algebraic_field_ids=tuple(item.field_id for item in self.evolution_layout.fields),
      accepted_step_index=base.evolution.accepted_step_index + 1,
      predictor=FinalizedArray(
        transaction.predictor.full_increment.values,
        dtype=np.float64,
      ),
      actual_increment=FinalizedArray(actual_increment, dtype=np.float64),
    )
    history = ProgramHistory(
      program_instance_id=self.program.instance_id,
      program_content_fingerprint=self.program.content_fingerprint,
      schema=PROGRAM_HISTORY_SCHEMA,
      generation=candidate_generation,
      entries=(),
    )
    candidate = CommittedAnalysisState(
      prepared_instance_id=self.instance_id,
      generation=candidate_generation,
      physical=physical,
      evolution=evolution,
      program_history=history,
    )
    trial_id = InstanceId()
    ledger = build_balance_ledger(
      model=self.model,
      program=self.program,
      plan=self.assembly_plan,
      contributions=contributions,
      request_manifest=self.request_manifest,
      prepared_instance_id=self.instance_id,
      transaction_id=transaction.transaction_id,
      trial_id=trial_id,
      base_generation=transaction.base_generation,
      candidate_generation=candidate_generation,
      reduced_coordinates=reduced_coordinates,
      primary_values=primary,
    )
    candidate_verification = fresh_verify(
      model=self.model,
      program=self.program,
      request_manifest=self.request_manifest,
      prepared_instance_id=self.instance_id,
      plan_content_fingerprint=self.assembly_plan.content_fingerprint,
      transaction_id=transaction.transaction_id,
      trial_id=trial_id,
      base_generation=transaction.base_generation,
      candidate_generation=candidate_generation,
      state=candidate,
      retained_ledger=ledger,
      convergence=convergence,
    )
    if not candidate_verification.passed:
      failed = ", ".join(
        item.name for item in candidate_verification.checks if not item.passed
      )
      _solve_failure(
        "candidate-verification-failed",
        f"fresh candidate verification failed: {failed}",
      )
    trial = TrialAnalysisState(
      trial_id=trial_id,
      transaction_id=transaction.transaction_id,
      prepared_instance_id=self.instance_id,
      base_generation=transaction.base_generation,
      candidate_generation=candidate_generation,
      candidate_state=candidate,
      ledger=ledger,
      convergence=convergence,
      candidate_verification=candidate_verification,
    )
    with self._lock:
      current = self._transactions.get(_instance_key(transaction.transaction_id))
      if current is not record or current.closed:
        _transaction_failure(
          "closed-transaction",
          "transaction closed before trial registration",
        )
      if _generation_key(transaction.base_generation) in self._consumed_generations:
        _transaction_failure(
          "stale-base-generation",
          "base generation was consumed before trial registration",
        )
      current.trials[_instance_key(trial_id)] = trial
    return trial

  def _registered_trial(
    self,
    trial: object,
  ) -> tuple[_TransactionRecord, TrialAnalysisState]:
    if type(trial) is not TrialAnalysisState:
      _transaction_failure(
        "foreign-or-forged-trial",
        "trial must be exactly TrialAnalysisState",
      )
    transaction_key = _instance_key(trial.transaction_id)
    trial_key = _instance_key(trial.trial_id)
    with self._lock:
      record = self._transactions.get(transaction_key)
      if record is None or record.closed:
        _transaction_failure(
          "closed-or-foreign-transaction",
          "trial transaction is absent or already closed",
        )
      if (
        _generation_key(record.transaction.base_generation)
        in self._consumed_generations
      ):
        _transaction_failure(
          "stale-base-generation",
          "trial base generation has already been consumed",
        )
      registered = record.trials.get(trial_key)
      if registered is not trial or trial_key in record.discarded_trials:
        _transaction_failure(
          "foreign-forged-or-discarded-trial",
          "trial is foreign, forged, or already discarded",
        )
    return record, trial

  def accept(self, trial: TrialAnalysisState) -> Solution:
    """Validate completely, copy, then atomically consume one exact base."""
    self._validate_owner()
    record, validated_trial = self._registered_trial(trial)
    transaction = record.transaction
    try:
      require_same_instance(
        self.instance_id,
        validated_trial.prepared_instance_id,
        context="trial prepared analysis",
      )
      require_same_instance(
        transaction.transaction_id,
        validated_trial.transaction_id,
        context="trial transaction",
      )
      require_same_generation(
        transaction.base_generation,
        validated_trial.base_generation,
        context="trial base",
      )
      require_generation_successor(
        validated_trial.base_generation,
        validated_trial.candidate_generation,
        context="trial candidate",
      )
    except (TypeError, ValueError):
      _transaction_failure(
        "invalid-trial-provenance",
        "trial identities or generations do not match its registered transaction",
      )
    if validated_trial.candidate_verification.passed is not True:
      _transaction_failure(
        "unverified-trial",
        "trial must carry a successful fresh candidate verification",
      )

    provisional_transition = AcceptedTransition(
      transaction=transaction,
      prepared_instance_id=self.instance_id,
      transaction_id=transaction.transaction_id,
      trial_id=validated_trial.trial_id,
      base_generation=transaction.base_generation,
      candidate_generation=validated_trial.candidate_generation,
      base_primary_values=FinalizedArray(
        transaction.base_state.physical.primary_values.values,
        dtype=np.float64,
      ),
      actual_increment=FinalizedArray(
        validated_trial.candidate_state.evolution.actual_increment.values,
        dtype=np.float64,
      ),
      target_evaluation=clone_program_evaluation(transaction.target_evaluation),
      committed_state=validated_trial.candidate_state,
      accepted=True,
    )
    try:
      record_report = verify_record_data(
        model=self.model,
        program=self.program,
        request=self.request,
        request_manifest=self.request_manifest,
        prepared_instance_id=self.instance_id,
        plan_content_fingerprint=self.assembly_plan.content_fingerprint,
        state=validated_trial.candidate_state,
        transition=provisional_transition,
        convergence=validated_trial.convergence,
        ledger=validated_trial.ledger,
      )
    except SolutionVerificationError as error:
      _transaction_failure("invalid-trial-record", str(error))
    if not record_report.passed:
      _transaction_failure(
        "inconsistent-trial-record",
        "trial record is numerically inconsistent before acceptance",
      )
    fresh_report = fresh_verify(
      model=self.model,
      program=self.program,
      request_manifest=self.request_manifest,
      prepared_instance_id=self.instance_id,
      plan_content_fingerprint=self.assembly_plan.content_fingerprint,
      transaction_id=transaction.transaction_id,
      trial_id=validated_trial.trial_id,
      base_generation=transaction.base_generation,
      candidate_generation=validated_trial.candidate_generation,
      state=validated_trial.candidate_state,
      retained_ledger=validated_trial.ledger,
      convergence=validated_trial.convergence,
    )
    if not fresh_report.passed:
      _transaction_failure(
        "trial-fresh-verification-failed",
        "trial changed or failed fresh verification before acceptance",
      )

    committed = _copy_state(
      validated_trial.candidate_state,
      prepared_instance_id=self.instance_id,
    )
    transition = AcceptedTransition(
      transaction=transaction,
      prepared_instance_id=self.instance_id,
      transaction_id=transaction.transaction_id,
      trial_id=validated_trial.trial_id,
      base_generation=transaction.base_generation,
      candidate_generation=validated_trial.candidate_generation,
      base_primary_values=FinalizedArray(
        transaction.base_state.physical.primary_values.values,
        dtype=np.float64,
      ),
      actual_increment=FinalizedArray(
        committed.evolution.actual_increment.values,
        dtype=np.float64,
      ),
      target_evaluation=clone_program_evaluation(transaction.target_evaluation),
      committed_state=committed,
      accepted=True,
    )
    solution = Solution(
      model=self.model,
      program=self.program,
      request=self.request,
      request_manifest=self.request_manifest,
      prepared_instance_id=self.instance_id,
      plan_content_fingerprint=self.assembly_plan.content_fingerprint,
      state=committed,
      transition=transition,
      convergence=validated_trial.convergence,
      ledger=clone_balance_ledger(validated_trial.ledger),
    )
    try:
      final_record = solution.verify_record()
    except SolutionVerificationError as error:
      _transaction_failure("accepted-solution-construction-failed", str(error))
    if not final_record.passed:
      _transaction_failure(
        "accepted-solution-construction-failed",
        "copied solution record is inconsistent before atomic acceptance",
      )

    base_key = _generation_key(transaction.base_generation)
    transaction_key = _instance_key(transaction.transaction_id)
    trial_key = _instance_key(validated_trial.trial_id)
    with self._lock:
      current = self._transactions.get(transaction_key)
      if current is not record or current.closed:
        _transaction_failure(
          "double-accept-or-closed-transaction",
          "transaction closed before atomic acceptance",
        )
      if (
        current.trials.get(trial_key) is not validated_trial
        or trial_key in current.discarded_trials
      ):
        _transaction_failure(
          "discarded-or-changed-trial",
          "trial was discarded or its exact registration changed before acceptance",
        )
      if base_key in self._consumed_generations:
        _transaction_failure(
          "stale-sibling-trial",
          "another transaction already consumed the exact base generation",
        )
      self._consumed_generations.add(base_key)
      current.closed = True
      self._issued_states[_generation_key(committed.generation)] = committed
    return solution

  def discard(self, trial: TrialAnalysisState) -> None:
    """Discard one registered trial without changing any accepted state."""
    self._validate_owner()
    record, validated_trial = self._registered_trial(trial)
    trial_key = _instance_key(validated_trial.trial_id)
    with self._lock:
      if record.closed:
        _transaction_failure(
          "closed-transaction",
          "transaction closed before atomic discard",
        )
      if record.trials.get(trial_key) is not validated_trial:
        _transaction_failure(
          "foreign-or-forged-trial",
          "trial registration changed before atomic discard",
        )
      if trial_key in record.discarded_trials:
        _transaction_failure(
          "already-discarded-trial",
          "trial was already discarded",
        )
      record.discarded_trials.add(trial_key)

  def abandon(self, transaction: StepTransaction) -> None:
    """Close an unaccepted transaction without consuming its base."""
    self._validate_owner()
    record = self._registered_transaction(transaction)
    with self._lock:
      if record.closed:
        _transaction_failure("closed-transaction", "transaction is already closed")
      record.closed = True

  def solve(
    self,
    *,
    point: ProgramPoint,
    initial: CommittedAnalysisState | None = None,
    initial_point: ProgramPoint | None = None,
  ) -> Solution:
    """Run the reusable explicit-base or fresh-lineage convenience flow."""
    self._validate_owner()
    if initial is None:
      if initial_point is None:
        _transaction_failure(
          "missing-initial-point",
          "a convenience solve requires an explicit initial_point",
        )
      selected_initial = self.initialize(point=initial_point)
    else:
      if initial_point is not None:
        _transaction_failure(
          "ambiguous-initial-state",
          "provide either an initialized state or an initial_point, not both",
        )
      selected_initial = initial
    transaction = self.begin_step(initial=selected_initial, point=point)
    trial = self.evaluate_trial(transaction)
    return self.accept(trial)


def prepare_analysis(
  model: CompiledModel,
  program: CompiledProgram,
  request: LinearStatic,
) -> PreparedAnalysis:
  """Bind exact compiled inputs to one fresh transaction/cache owner."""
  if type(request) is not LinearStatic:
    _preparation_failure(
      "invalid-analysis-request",
      "analysis request must be exactly LinearStatic",
    )
  try:
    plan = prepare_assembly_plan(
      model,
      program,
      LinearStaticContributionRequest(),
    )
  except AssemblyPreparationError as error:
    _preparation_failure("assembly-preparation-failed", str(error))
  return PreparedAnalysis(
    model,
    program,
    request,
    plan,
    _token=_PREPARED_CONSTRUCTION_TOKEN,
  )
