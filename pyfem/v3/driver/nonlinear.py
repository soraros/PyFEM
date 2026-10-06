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
  progress. Exhausting ``max_cutbacks`` yields the typed ``STEP_FAILED``
  run status, never an exception. Operators that leave the finite float64
  envelope raise their own contract error; the driver still rejects the
  open transaction first, so committed state survives even that path
  byte-identical.
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
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import splu

from pyfem.v3.constraints.compile import (
  CompiledConstraintMap,
  evaluate_offsets,
  full_coefficients,
  reaction_forces,
  reduce_residual,
  reduce_tangent,
)
from pyfem.v3.driver.contracts import (
  _NORM_REFERENCE_FLOOR,
  DriverStatistics,
  DriverStatus,
  IterationRecord,
  NonlinearStaticResult,
  NonlinearStaticSettings,
  SubstepObservation,
  SubstepRecord,
  SubstepStatus,
)
from pyfem.v3.driver.plan import (
  DriverAssemblyPlan,
  assemble_internal_force,
  compile_driver_plan,
  evaluate_loads,
  evaluate_signals,
  refill_tangent,
)
from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.operator import (
  ChannelRequest,
  EvaluationStatus,
  OperatorEvaluationInput,
  evaluation_status,
)
from pyfem.v3.model.system import CompiledSystem
from pyfem.v3.spec.program import (
  NodalLoadSpec,
  ProgramCoordinateValue,
  ProgramPoint,
)
from pyfem.v3.state import StateTransactionOwner


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
      if plan.constant_tangent:
        workspace.cached_factorization = factorization
    correction = factorization.solve(rhs)
    workspace.linear_solve_count += 1
    if not bool(np.isfinite(correction).all()):
      return None
    return np.asarray(correction, dtype=np.float64)

  def _newton_substep(
    self,
    point: ProgramPoint,
    committed_point: ProgramPoint | None,
  ) -> tuple[bool, tuple[IterationRecord, ...], SubstepObservation | None]:
    """Run one substep Newton loop inside one open owner transaction.

    ``committed_point`` is the last committed program point (the derivation
    base for increment-bound signals); identity-only plans never read it.
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
          return False, tuple(iterations), None
        if EvaluationStatus.REJECT_ITERATION in statuses:
          iterations.append(
            IterationRecord(iteration, EvaluationStatus.REJECT_ITERATION, None, None)
          )
          if anchor is None or increment is None:
            transaction.reject()
            closed = True
            workspace.rejected_substep_count += 1
            return False, tuple(iterations), None
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
          observation = SubstepObservation(
            reactions=reactions,
            constraint_work=float(np.dot(reactions.values, full.values)),
            full_residual_norm=float(np.linalg.norm(full_residual)),
            reduced_residual_norm=residual_norm,
          )
          return True, tuple(iterations), observation
        if first_residual_norm is None:
          first_residual_norm = max(residual_norm, _NORM_REFERENCE_FLOOR)
        elif residual_norm > settings.divergence_ratio * first_residual_norm:
          iterations.append(
            IterationRecord(iteration, EvaluationStatus.OK, residual_norm, None)
          )
          transaction.reject()
          closed = True
          workspace.rejected_substep_count += 1
          return False, tuple(iterations), None
        correction = self._solve_reduced(jacobian_batches, rhs_reduced)
        if correction is None:
          iterations.append(
            IterationRecord(iteration, EvaluationStatus.OK, residual_norm, None)
          )
          transaction.reject()
          closed = True
          workspace.rejected_substep_count += 1
          return False, tuple(iterations), None
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
          return False, tuple(iterations), None
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
      return False, tuple(iterations), None
    except BaseException:
      if not closed:
        transaction.reject()
      raise

  def run(
    self,
    *,
    base_point: ProgramPoint,
    target_points: tuple[ProgramPoint, ...],
  ) -> NonlinearStaticResult:
    """Advance the committed state through the exact target-point schedule."""
    if type(target_points) is not tuple or any(
      type(point) is not ProgramPoint for point in target_points
    ):
      msg = "driver target points must be an exact tuple of ProgramPoint values"
      raise TypeError(msg)
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
        committed, iterations, observation = self._newton_substep(
          trial_point,
          committed_point,
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
            )
          )
          progress_done = trial_progress
          cutback_level = 0
          if progress_done == 1.0:
            break
          size = min(settings.growth_factor * size, 1.0 - progress_done)
          continue
        cutback_level += 1
        if cutback_level > settings.max_cutbacks:
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
