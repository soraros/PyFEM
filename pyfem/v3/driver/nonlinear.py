"""Purified nonlinear static driver: explicit Newton/cutback loop over the owner.

The driver is the D3/H1 schedule: a typed analysis owning an explicit
predict/evaluate/linearize/solve/check/commit loop. It is forbidden from
owning authoritative state — committed physical coefficients and operator
state rows live only in the M12 :class:`StateTransactionOwner`; the reduced
coordinates are derivable scratch (``q = u[free_dofs]``); the schedule
bookkeeping is local to one ``run`` and returned in the result.

Protocol summary:

- One M12 transaction spans a whole substep: evaluations read the published
  accepted snapshots, commit stages the converged physical vector and every
  operator block's trial rows in one atomic transition, and a rejected or
  cut-back substep leaves committed state byte-identical. Every attempt is
  traceable: committed substeps carry observations, rejected attempts carry
  typed ``REJECTED`` records with their iteration trail, and budget
  exhaustion closes the target with a typed ``FAILED`` record.
- Cutback (operator ``REJECT_STEP``, singular or non-finite solve, a
  non-finite trial iterate, divergence past ``divergence_ratio``, or an
  exhausted iteration budget) halves the substep in program-coordinate
  progress and retries from the SAME committed generation; a committed
  substep regrows the size by ``growth_factor`` capped at the remaining
  progress. Exhausting ``max_cutbacks`` — or rejecting a substep whose size
  is already at or below ``min_substep_size`` of the target interval, where
  no further refinement can advance the schedule meaningfully — yields the
  typed ``STEP_FAILED`` run status, never an exception. The floor is the
  termination guard for limit-point trajectories: a target past a limit
  load commits ever-smaller substeps that each reset the consecutive-rejection
  budget, so without it the schedule asymptotes to the limit point and
  burns unbounded CPU without tripping ``max_cutbacks``. Operators that
  leave the finite float64 envelope raise their own contract error; the
  driver still rejects the open transaction first, so committed state
  survives even that path byte-identical.
- Iteration-budget exhaustion is classified, never silent: the exhausted
  attempt's measured residual trend is typed on the REJECTED or FAILED
  record as a ``BudgetExhaustionObservation``. ``SLOW_CONVERGENCE`` — the
  final measured residual contracts to at most half the first with a
  strict majority of decreasing steps — is the near-miss: the substep ran
  out of budget, not of convergence, and the documented remedy is a larger
  ``max_iterations`` (the M48 finding's episode on the PRE-FIX J2 tangent:
  the load-controlled cantilever with near-yield tangent chatter converged
  in one 33-iteration substep at ``max_iterations=50`` where the default 25
  thrashed it into 294 records / 7270 evaluations of cutback; the M54
  shear-tangent fix restored quadratic convergence — the same deck commits
  in one 7-iteration substep at the default budget). The classification is
  observation-only by policy — the driver never extends the budget itself:
  a past-limit-load trajectory can net-decrease over a budget window
  without converging (5 of 59 failing attempts on the snap-through truss
  do), so trend-keyed continuation would burn extra work inside the very
  limit-point trap the substep floor guards against and would alter the
  failure trail. REJECT, cutback, and FAILED decisions are numerically
  identical with or without the observation.
- Operator ``REJECT_ITERATION`` retries from the same accepted state with a
  damped iterate (half the last Newton increment), budgeted against
  ``max_iterations``; without a previous increment it escalates to cutback.
- Constraints apply exclusively through the M15 coordinate map
  (``full_coefficients``/``reduce_residual``/``reduce_tangent``); reactions
  and constraint work are observed from the FULL residual at commit.
- Program signals reach operators exclusively through the plan: each substep
  binds its fixed trial point once into per-operator ``ProgramSignalInput``
  values, so schedule-owned time replaces any solverStat-style hidden global.
  A port either binds its program coordinate identically or derives the
  coordinate's committed increment (``d<coordinate>``, e.g. ``dtime``): the
  derivation subtracts the coordinate's value at the last COMMITTED point —
  the run's base point for the first substep of a run — so a rejected attempt
  leaves the derivation base untouched and a cut-back retry re-derives from
  the same committed generation. Operators without declared signal ports keep
  receiving an empty signal tuple.
- The tangent is assembled through the compiled plan (values-only refill).
  When every Jacobian channel is compiled ``linear`` the reduced tangent is
  state-independent: it is assembled and factorized once and the
  factorization is reused bitwise across all iterations, substeps, and runs.
  Otherwise each Newton iteration factorizes the refilled reduced tangent.
- Requested parameter sensitivities (``sensitivity_parameters`` on ``run``)
  are implicit-function-theorem solves on committed states, never a
  differentiation of the iteration process — the M48 study measured
  truncated-unrolled differentiation at K=1 reporting exactly zero
  sensitivity on a plastifying step (the elastic predictor has not
  discovered plasticity), a 100% relative error the converged-state IFT
  avoids by construction. At each COMMITTED substep the requested
  ``d(residual)/d(parameter)`` channels are evaluated once at the converged
  point with the ENTERING committed state held fixed — one extra evaluation
  sweep inside the still-open transaction, over only the operators that
  declare a requested parameter — and after commit each parameter solves
  ``K_q (dq/dp) = -P.T dR/dp`` with ONE back-substitution on the
  factorization the converged Newton loop last used (for a
  constant-tangent plan, the state-independent cached one), counted as
  factorization reuse in the workspace counters; no per-parameter
  refactorization ever happens. The one exception: a substep that
  converged with no Newton solve at all (the entering state already
  satisfies the new point) owns no converged factorization, so the
  converged tangent is factorized once from the in-hand Jacobian batches
  and counted truthfully (and cached when the plan is constant-tangent,
  where it is the same state-independent tangent). Sensitivities chain
  forward per committed substep on committed states only — rejected
  attempts never produce them and the budget-exhaustion machinery is
  untouched — and surface as typed ``ParameterSensitivityObservation``
  values on the committed records. With the entering state held fixed the
  per-step derivative is exact for a parameter-independent entering state
  (the virgin state, and every elastic step with zero plastic integration
  points); propagating the entering state's own parameter dependence needs
  state-derivative channels and is the declared follow-up boundary (the
  M48 survey's v2 scope).
"""

from __future__ import annotations

from typing import NoReturn

import numpy as np
from scipy.sparse.linalg import splu

from pyfem.v3.constraints.compile import (
  CompiledConstraintMap,
  admissible_increment,
  evaluate_offsets,
  full_coefficients,
  reaction_forces,
  reduce_residual,
  reduce_tangent,
)
from pyfem.v3.driver.contracts import (
  _NORM_REFERENCE_FLOOR,
  _SLOW_CONVERGENCE_CONTRACTION,
  BudgetExhaustionObservation,
  BudgetExhaustionTrend,
  DriverStatistics,
  DriverStatus,
  IterationRecord,
  NonlinearStaticResult,
  NonlinearStaticSettings,
  ParameterSensitivityObservation,
  SubstepObservation,
  SubstepRecord,
  SubstepStatus,
)
from pyfem.v3.driver.diagnostics import (
  DriverDiagnostic,
  DriverEvaluationError,
)
from pyfem.v3.driver.plan import (
  CompiledSensitivityProgram,
  DriverAssemblyPlan,
  assemble_internal_force,
  assemble_parameter_rhs,
  compile_driver_plan,
  compile_sensitivity_program,
  evaluate_loads,
  evaluate_signals,
  refill_tangent,
)
from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.operator import (
  ChannelRequest,
  EvaluationStatus,
  OperatorEvaluationInput,
  ProgramSignalInput,
  evaluation_derivative_values,
  evaluation_status,
)
from pyfem.v3.model.system import CompiledSystem
from pyfem.v3.spec.diagnostics import SourceContext
from pyfem.v3.spec.program import (
  NodalLoadSpec,
  ProgramCoordinateValue,
  ProgramPoint,
)
from pyfem.v3.state import StateTransaction, StateTransactionOwner


def _evaluation_fail(code: str, message: str) -> NoReturn:
  raise DriverEvaluationError(
    (DriverDiagnostic(code=code, message=message, source=SourceContext()),)
  )


class _DriverWorkspace:
  """Disposable solve caches and counters; never authoritative state."""

  __slots__ = (
    "cached_factorization",
    "committed_substep_count",
    "cutback_count",
    "evaluation_count",
    "factorization_count",
    "factorization_reuse_count",
    "linear_solve_count",
    "rejected_substep_count",
    "residual_assembly_count",
    "stashed_factorization",
    "tangent_refill_count",
  )

  def __init__(self) -> None:
    self.cached_factorization: object = None
    self.committed_substep_count = 0
    self.cutback_count = 0
    self.evaluation_count = 0
    self.factorization_count = 0
    self.factorization_reuse_count = 0
    self.linear_solve_count = 0
    self.rejected_substep_count = 0
    self.residual_assembly_count = 0
    self.stashed_factorization: object = None
    self.tangent_refill_count = 0


def _interpolate(
  coordinate_names: tuple[str, ...],
  base_values: tuple[float, ...],
  target_values: tuple[float, ...],
  progress: float,
) -> ProgramPoint:
  """Affine point interpolation, exact at both progress endpoints."""
  return ProgramPoint(
    tuple(
      ProgramCoordinateValue(
        name,
        (1.0 - progress) * base + progress * target,
      )
      for name, base, target in zip(
        coordinate_names,
        base_values,
        target_values,
        strict=True,
      )
    )
  )


def _classify_budget_exhaustion(
  iterations: tuple[IterationRecord, ...],
) -> BudgetExhaustionObservation:
  """Classify the measured residual trend of one budget-exhausted attempt.

  A budget-exhausted attempt always carries at least one measured residual:
  the first iteration either measures one or rejects the substep outright.
  """
  measured = [
    iteration.residual_norm
    for iteration in iterations
    if iteration.residual_norm is not None
  ]
  pair_count = len(measured) - 1
  decreasing = sum(1 for first, second in zip(measured, measured[1:]) if second < first)
  trend = (
    BudgetExhaustionTrend.SLOW_CONVERGENCE
    if pair_count > 0
    and measured[-1] <= _SLOW_CONVERGENCE_CONTRACTION * measured[0]
    and 2 * decreasing > pair_count
    else BudgetExhaustionTrend.NON_CONVERGENT
  )
  return BudgetExhaustionObservation(
    trend=trend,
    first_residual_norm=measured[0],
    final_residual_norm=measured[-1],
    decreasing_step_count=decreasing,
    measured_step_count=len(measured),
  )


class NonlinearStaticDriver:
  """Typed nonlinear static analysis over one exact system and map pair."""

  def __init__(
    self,
    system: CompiledSystem,
    coordinate_map: CompiledConstraintMap,
    loads: tuple[NodalLoadSpec, ...] = (),
    settings: NonlinearStaticSettings = NonlinearStaticSettings(),
  ) -> None:
    if type(system) is not CompiledSystem:
      msg = "the nonlinear driver requires an exact CompiledSystem"
      raise TypeError(msg)
    if type(coordinate_map) is not CompiledConstraintMap:
      msg = "the nonlinear driver requires an exact CompiledConstraintMap"
      raise TypeError(msg)
    if type(settings) is not NonlinearStaticSettings:
      msg = "the nonlinear driver requires exact NonlinearStaticSettings"
      raise TypeError(msg)
    self._system = system
    self._map = coordinate_map
    self._settings = settings
    self._plan = compile_driver_plan(system, coordinate_map, loads)
    self._owner = StateTransactionOwner(system)
    self._workspace = _DriverWorkspace()
    self._requests = tuple(
      ChannelRequest(
        (operator.header.residual_channels[0].channel_id,),
        (operator.header.jacobian_channels[0].channel_id,),
      )
      for operator in system.operators
    )

  @property
  def plan(self) -> DriverAssemblyPlan:
    """Return the immutable compiled assembly plan."""
    return self._plan

  @property
  def owner(self) -> StateTransactionOwner:
    """Return the sole authoritative state owner."""
    return self._owner

  @property
  def settings(self) -> NonlinearStaticSettings:
    """Return the exact numeric policy."""
    return self._settings

  @property
  def statistics(self) -> DriverStatistics:
    """Return a snapshot of the workspace counters."""
    workspace = self._workspace
    return DriverStatistics(
      evaluation_count=workspace.evaluation_count,
      residual_assembly_count=workspace.residual_assembly_count,
      tangent_refill_count=workspace.tangent_refill_count,
      factorization_count=workspace.factorization_count,
      factorization_reuse_count=workspace.factorization_reuse_count,
      linear_solve_count=workspace.linear_solve_count,
      cutback_count=workspace.cutback_count,
      committed_substep_count=workspace.committed_substep_count,
      rejected_substep_count=workspace.rejected_substep_count,
    )

  def _bound_point_values(self, point: ProgramPoint) -> tuple[float, ...]:
    """Validate one exact point against the map and load coordinates."""
    if type(point) is not ProgramPoint:
      msg = "driver program points must be exact ProgramPoint values"
      raise TypeError(msg)
    evaluation = evaluate_offsets(self._map, point)
    evaluate_loads(self._plan.loads, point)
    return tuple(float(value) for value in evaluation.coordinate_values.values)

  def _solve_reduced(
    self,
    jacobian_batches: tuple[np.ndarray, ...],
    rhs: np.ndarray,
  ) -> np.ndarray | None:
    """Solve the reduced Newton correction, or return None for cutback."""
    plan = self._plan
    workspace = self._workspace
    if plan.constant_tangent and workspace.cached_factorization is not None:
      workspace.factorization_reuse_count += 1
      factorization = workspace.cached_factorization
    else:
      tangent = refill_tangent(plan, jacobian_batches)
      workspace.tangent_refill_count += 1
      reduced_tangent = reduce_tangent(self._map, tangent)
      try:
        factorization = splu(reduced_tangent.tocsc())
      except RuntimeError:
        return None
      workspace.factorization_count += 1
      # Stash every fresh factorization: a committed substep's parameter
      # sensitivity solves reuse the converged loop's last factorization.
      workspace.stashed_factorization = factorization
      if plan.constant_tangent:
        workspace.cached_factorization = factorization
    correction = factorization.solve(rhs)
    workspace.linear_solve_count += 1
    if not bool(np.isfinite(correction).all()):
      return None
    return np.asarray(correction, dtype=np.float64)

  def _evaluate_sensitivity_columns(
    self,
    program: CompiledSensitivityProgram,
    full: FinalizedArray,
    transaction: StateTransaction,
    signal_inputs: tuple[tuple[ProgramSignalInput, ...], ...],
  ) -> tuple[tuple[FinalizedArray, ...] | None, ...]:
    """Evaluate the requested derivative channels at the converged point.

    One extra evaluation per operator declaring a requested parameter. The
    channel semantics hold the accepted state fixed at the ENTERING
    committed state, so this runs inside the still-open transaction
    immediately before commit; the trial rows it returns are discarded
    (the primal evaluations are staged) — they are bitwise identical to
    them by the derivative-twin contract. Operators declaring no requested
    parameter are never evaluated and contribute an exact zero batch.
    """
    columns: list[tuple[FinalizedArray, ...] | None] = []
    evaluated = False
    for operator, slice_, signals in zip(
      self._system.operators,
      program.operator_slices,
      signal_inputs,
      strict=True,
    ):
      if not slice_.derivative_channel_ids:
        columns.append(None)
        continue
      gather = operator.header.ports[0].coefficient_map.values
      evaluation = operator.evaluate(
        OperatorEvaluationInput(
          port_values=(
            FinalizedArray(
              np.array(full.values[gather], dtype=np.float64, order="C", copy=True),
              dtype=np.float64,
            ),
          ),
          accepted_state=transaction.accepted_state(
            operator.header.state_layout.block_id
          ),
          signals=signals,
          request=ChannelRequest((), (), slice_.derivative_channel_ids),
        )
      )
      evaluated = True
      if evaluation_status(evaluation) is not EvaluationStatus.OK:
        _evaluation_fail(
          "sensitivity-evaluation-rejected",
          "the derivative evaluation at a converged point rejected; the "
          "primal evaluation at the same point converged, so the operator "
          "contract is broken",
        )
      values = evaluation_derivative_values(evaluation)
      if len(values) != len(slice_.derivative_channel_ids):
        _evaluation_fail(
          "derivative-channel-count-mismatch",
          "an operator returned fewer derivative values than the requested "
          "channels it declared",
        )
      columns.append(values)
    if evaluated:
      self._workspace.evaluation_count += 1
    return tuple(columns)

  def _solve_sensitivities(
    self,
    program: CompiledSensitivityProgram,
    columns: tuple[tuple[FinalizedArray, ...] | None, ...],
    jacobian_batches: tuple[np.ndarray, ...],
    *,
    solved: bool,
  ) -> tuple[ParameterSensitivityObservation, ...]:
    """Solve the per-parameter IFT systems on the committed factorization.

    Every back-substitution reuses the factorization the converged Newton
    loop last solved with (for a constant-tangent plan, the cached
    state-independent one) — counted as factorization reuse, never a new
    factorization. A substep that converged without any Newton solve owns
    no such factorization; the converged tangent is then factorized once
    from the in-hand Jacobian batches, counted truthfully (and cached when
    the plan is constant-tangent, where it is the same tangent).
    """
    plan = self._plan
    workspace = self._workspace
    fresh = False
    if plan.constant_tangent and workspace.cached_factorization is not None:
      factorization = workspace.cached_factorization
    elif solved and workspace.stashed_factorization is not None:
      factorization = workspace.stashed_factorization
    else:
      tangent = refill_tangent(plan, jacobian_batches)
      workspace.tangent_refill_count += 1
      try:
        factorization = splu(reduce_tangent(self._map, tangent).tocsc())
      except RuntimeError:
        _evaluation_fail(
          "singular-sensitivity-tangent",
          "the converged tangent is singular, so the parameter sensitivity "
          "system has no solution",
        )
      workspace.factorization_count += 1
      fresh = True
      if plan.constant_tangent:
        workspace.cached_factorization = factorization
    zero_batches: dict[int, np.ndarray] = {}
    observations: list[ParameterSensitivityObservation] = []
    for parameter_index, parameter_id in enumerate(program.parameter_ids):
      batches: list[np.ndarray] = []
      for operator_index, (columns_, slice_, plan_slice) in enumerate(
        zip(
          columns,
          program.operator_slices,
          plan.operator_slices,
          strict=True,
        )
      ):
        if columns_ is not None and parameter_id in slice_.parameter_ids:
          batches.append(columns_[slice_.parameter_ids.index(parameter_id)].values)
          continue
        zero = zero_batches.get(operator_index)
        if zero is None:
          zero = np.zeros(
            (plan_slice.entity_count, plan_slice.element_dof_count),
            dtype=np.float64,
          )
          zero_batches[operator_index] = zero
        batches.append(zero)
      rhs = assemble_parameter_rhs(plan, self._map, tuple(batches))
      workspace.residual_assembly_count += 1
      sensitivity = np.asarray(
        factorization.solve(rhs.values),
        dtype=np.float64,
      )
      workspace.linear_solve_count += 1
      # The first solve on a just-factorized corner tangent is not a reuse;
      # every other sensitivity solve reuses a committed factorization.
      if not (fresh and parameter_index == 0):
        workspace.factorization_reuse_count += 1
      if not bool(np.isfinite(sensitivity).all()):
        _evaluation_fail(
          "non-finite-sensitivity-solution",
          f"the parameter sensitivity solve for {parameter_id!r} produced "
          "non-finite coefficients",
        )
      observations.append(
        ParameterSensitivityObservation(
          parameter_id=parameter_id,
          coefficients=admissible_increment(self._map, sensitivity),
        )
      )
    return tuple(observations)

  def _newton_substep(
    self,
    point: ProgramPoint,
    committed_point: ProgramPoint | None,
    sensitivity_program: CompiledSensitivityProgram | None,
  ) -> tuple[
    bool,
    tuple[IterationRecord, ...],
    SubstepObservation | None,
    BudgetExhaustionObservation | None,
  ]:
    """Run one substep Newton loop inside one open owner transaction.

    ``committed_point`` is the last committed program point (the derivation
    base for increment-bound signals); identity-only plans never read it.
    The fourth return is non-``None`` only on the iteration-budget-exhaust
    exit, carrying the measured residual-trend classification.
    ``sensitivity_program`` is the run's validated parameter request (or
    ``None``): its derivative channels are evaluated once at the converged
    point and solved per parameter post-commit — only on the committed
    exit, so rejected attempts never see it.
    """
    plan = self._plan
    workspace = self._workspace
    coordinate_map = self._map
    settings = self._settings
    free_dofs = coordinate_map.free_dofs.values
    transaction = self._owner.begin()
    closed = False
    iterations: list[IterationRecord] = []
    try:
      reduced = np.array(
        transaction.accepted_physical.values[free_dofs],
        dtype=np.float64,
        order="C",
        copy=True,
      )
      external = evaluate_loads(plan.loads, point).values
      signal_inputs = evaluate_signals(plan, point, committed_point)
      external_reduced = reduce_residual(coordinate_map, external).values
      reference_norm = float(np.linalg.norm(external_reduced))
      first_residual_norm: float | None = None
      anchor: np.ndarray | None = None
      increment: np.ndarray | None = None
      increment_norm: float | None = None
      for iteration in range(1, settings.max_iterations + 1):
        full = full_coefficients(coordinate_map, reduced, point)
        evaluations = []
        for operator, request, signals in zip(
          self._system.operators,
          self._requests,
          signal_inputs,
          strict=True,
        ):
          gather = operator.header.ports[0].coefficient_map.values
          port_values = (
            FinalizedArray(
              np.array(full.values[gather], dtype=np.float64, order="C", copy=True),
              dtype=np.float64,
            ),
          )
          evaluations.append(
            operator.evaluate(
              OperatorEvaluationInput(
                port_values=port_values,
                accepted_state=transaction.accepted_state(
                  operator.header.state_layout.block_id
                ),
                signals=signals,
                request=request,
              )
            )
          )
        workspace.evaluation_count += 1
        statuses = tuple(evaluation_status(evaluation) for evaluation in evaluations)
        if EvaluationStatus.REJECT_STEP in statuses:
          iterations.append(
            IterationRecord(iteration, EvaluationStatus.REJECT_STEP, None, None)
          )
          transaction.reject()
          closed = True
          workspace.rejected_substep_count += 1
          return False, tuple(iterations), None, None
        if EvaluationStatus.REJECT_ITERATION in statuses:
          iterations.append(
            IterationRecord(iteration, EvaluationStatus.REJECT_ITERATION, None, None)
          )
          if anchor is None or increment is None:
            transaction.reject()
            closed = True
            workspace.rejected_substep_count += 1
            return False, tuple(iterations), None, None
          increment = 0.5 * increment
          reduced = anchor + increment
          increment_norm = float(np.linalg.norm(increment))
          continue
        residual_batches = tuple(
          evaluation.residual_values[0].values for evaluation in evaluations
        )
        jacobian_batches = tuple(
          evaluation.jacobian_values[0].values for evaluation in evaluations
        )
        internal = assemble_internal_force(plan, residual_batches)
        workspace.residual_assembly_count += 1
        rhs_reduced = reduce_residual(coordinate_map, external - internal).values
        residual_norm = float(np.linalg.norm(rhs_reduced))
        anchor = reduced
        if reference_norm >= _NORM_REFERENCE_FLOOR:
          measure = residual_norm / reference_norm
        else:
          measure = residual_norm
        if measure <= settings.tolerance:
          iterations.append(
            IterationRecord(
              iteration,
              EvaluationStatus.OK,
              residual_norm,
              increment_norm,
            )
          )
          # The derivative channels read the ENTERING committed state, so
          # their evaluation runs inside the still-open transaction; the
          # solves run post-commit. Staging uses the primal evaluations.
          sensitivity_columns = (
            self._evaluate_sensitivity_columns(
              sensitivity_program,
              full,
              transaction,
              signal_inputs,
            )
            if sensitivity_program is not None
            else None
          )
          transaction.stage_physical(full)
          for operator, evaluation in zip(
            self._system.operators,
            evaluations,
            strict=True,
          ):
            transaction.stage_state(
              operator.header.state_layout.block_id,
              evaluation.trial_state,
            )
          transaction.commit()
          closed = True
          workspace.committed_substep_count += 1
          full_residual = internal - external
          reactions = reaction_forces(coordinate_map, full_residual)
          sensitivities: tuple[ParameterSensitivityObservation, ...] = ()
          if sensitivity_program is not None and sensitivity_columns is not None:
            sensitivities = self._solve_sensitivities(
              sensitivity_program,
              sensitivity_columns,
              jacobian_batches,
              solved=increment is not None,
            )
          observation = SubstepObservation(
            reactions=reactions,
            constraint_work=float(np.dot(reactions.values, full.values)),
            full_residual_norm=float(np.linalg.norm(full_residual)),
            reduced_residual_norm=residual_norm,
            sensitivities=sensitivities,
          )
          return True, tuple(iterations), observation, None
        if first_residual_norm is None:
          first_residual_norm = max(residual_norm, _NORM_REFERENCE_FLOOR)
        elif residual_norm > settings.divergence_ratio * first_residual_norm:
          iterations.append(
            IterationRecord(iteration, EvaluationStatus.OK, residual_norm, None)
          )
          transaction.reject()
          closed = True
          workspace.rejected_substep_count += 1
          return False, tuple(iterations), None, None
        correction = self._solve_reduced(jacobian_batches, rhs_reduced)
        if correction is None:
          iterations.append(
            IterationRecord(iteration, EvaluationStatus.OK, residual_norm, None)
          )
          transaction.reject()
          closed = True
          workspace.rejected_substep_count += 1
          return False, tuple(iterations), None, None
        increment = correction
        increment_norm = float(np.linalg.norm(increment))
        trial_reduced = reduced + increment
        if not bool(np.isfinite(trial_reduced).all()):
          iterations.append(
            IterationRecord(
              iteration,
              EvaluationStatus.OK,
              residual_norm,
              increment_norm,
            )
          )
          transaction.reject()
          closed = True
          workspace.rejected_substep_count += 1
          return False, tuple(iterations), None, None
        reduced = trial_reduced
        iterations.append(
          IterationRecord(
            iteration,
            EvaluationStatus.OK,
            residual_norm,
            increment_norm,
          )
        )
      transaction.reject()
      closed = True
      workspace.rejected_substep_count += 1
      exhausted = tuple(iterations)
      return False, exhausted, None, _classify_budget_exhaustion(exhausted)
    except BaseException:
      if not closed:
        transaction.reject()
      raise

  def run(
    self,
    *,
    base_point: ProgramPoint,
    target_points: tuple[ProgramPoint, ...],
    sensitivity_parameters: tuple[str, ...] = (),
  ) -> NonlinearStaticResult:
    """Advance the committed state through the exact target-point schedule.

    ``sensitivity_parameters`` names the spec-level parameters whose
    first-order sensitivities of the committed coefficients are solved per
    committed substep (the implicit-function-theorem protocol in the module
    docstring); the default empty tuple keeps the run primal-only at zero
    extra cost. The request is validated before the first substep: a
    parameter no operator declares a derivative channel for is rejected
    with a coded diagnostic.
    """
    if type(target_points) is not tuple or any(
      type(point) is not ProgramPoint for point in target_points
    ):
      msg = "driver target points must be an exact tuple of ProgramPoint values"
      raise TypeError(msg)
    if type(sensitivity_parameters) is not tuple:
      msg = "driver sensitivity parameters must be an exact tuple of names"
      raise TypeError(msg)
    sensitivity_program = (
      compile_sensitivity_program(self._system, sensitivity_parameters)
      if sensitivity_parameters
      else None
    )
    initial_generation = self._owner.generation
    base_values = self._bound_point_values(base_point)
    target_values = tuple(self._bound_point_values(point) for point in target_points)
    records: list[SubstepRecord] = []
    settings = self._settings
    workspace = self._workspace
    coordinate_names = self._map.coordinate_names
    committed_values = base_values
    requires_committed_point = self._plan.requires_committed_point
    for target_index, (point, values) in enumerate(
      zip(target_points, target_values, strict=True)
    ):
      progress_done = 0.0
      size = 1.0
      cutback_level = 0
      while True:
        trial_progress = min(progress_done + size, 1.0)
        trial_point = _interpolate(
          coordinate_names,
          committed_values,
          values,
          trial_progress,
        )
        # The derivation base for increment-bound signals: the last COMMITTED
        # point (progress zero is the previous target's or the run's base
        # point). Identity-only plans skip the construction entirely.
        committed_point = (
          _interpolate(coordinate_names, committed_values, values, progress_done)
          if requires_committed_point
          else None
        )
        committed, iterations, observation, budget_exhaustion = self._newton_substep(
          trial_point,
          committed_point,
          sensitivity_program,
        )
        if committed:
          records.append(
            SubstepRecord(
              target_index=target_index,
              progress=trial_progress,
              point=trial_point,
              status=SubstepStatus.COMMITTED,
              cutback_level=cutback_level,
              iterations=iterations,
              committed_ordinal=self._owner.generation.ordinal,
              observation=observation,
              budget_exhaustion=None,
            )
          )
          progress_done = trial_progress
          cutback_level = 0
          if progress_done == 1.0:
            break
          size = min(settings.growth_factor * size, 1.0 - progress_done)
          continue
        cutback_level += 1
        if cutback_level > settings.max_cutbacks or size <= settings.min_substep_size:
          records.append(
            SubstepRecord(
              target_index=target_index,
              progress=trial_progress,
              point=trial_point,
              status=SubstepStatus.FAILED,
              cutback_level=cutback_level,
              iterations=iterations,
              committed_ordinal=None,
              observation=None,
              budget_exhaustion=budget_exhaustion,
            )
          )
          return NonlinearStaticResult(
            status=DriverStatus.STEP_FAILED,
            base_point=base_point,
            target_points=target_points,
            records=tuple(records),
            statistics=self.statistics,
            initial_generation=initial_generation,
            final_generation=self._owner.generation,
            failed_target_index=target_index,
          )
        records.append(
          SubstepRecord(
            target_index=target_index,
            progress=trial_progress,
            point=trial_point,
            status=SubstepStatus.REJECTED,
            cutback_level=cutback_level,
            iterations=iterations,
            committed_ordinal=None,
            observation=None,
            budget_exhaustion=budget_exhaustion,
          )
        )
        workspace.cutback_count += 1
        size *= settings.cutback_factor
      committed_values = values
    return NonlinearStaticResult(
      status=DriverStatus.COMPLETED,
      base_point=base_point,
      target_points=target_points,
      records=tuple(records),
      statistics=self.statistics,
      initial_generation=initial_generation,
      final_generation=self._owner.generation,
      failed_target_index=None,
    )
