"""Purified Riks arc-length driver: two-scope continuation over the owner.

The driver is the D4 typed arc-length analysis: an explicit
predict/evaluate/linearize/two-solve/constrain/check/commit loop over the
same landed machinery as the M18 nonlinear static driver. It is forbidden
from owning authoritative state — committed physical coefficients and
operator state rows live only in the M12 :class:`StateTransactionOwner`;
the reduced coordinates are derivable scratch (``q = u[free_dofs]``); the
continuation bookkeeping lives in the request-owned
:class:`pyfem.v3.state.evolution.ContinuationEvolutionStore` and is
snapshotted into the result.

Two-scope continuation discipline (D3 risk #5 made concrete):

- Scope A — the committed continuation baseline ``{lam, da_prev, dlam_prev,
  factor, total_factor, cycle}`` — is one ``ContinuationEvolutionStore``
  seeded virgin per run at the owner's current generation, or RESUMED from a
  previous run's store through the exact-typed ``resume`` argument gated on
  live-generation identity. It is written ONLY on the substep commit path
  immediately after ``owner.commit()`` via the store's infallible
  ``advance()`` and snapshotted into the result. Reject paths never touch
  it, so a rejected or cut-back attempt leaves both the owner's committed
  state and the store byte-identical; commit atomicity is procedural and
  total (the owner validates-then-writes and cannot fail after staging, and
  the store advance is field writes that cannot fail).
- Scope B — the per-attempt candidate (the anchored predictor ``(da1,
  dlam1)``, the trial ``lam``, the cutback identity, the iteration trail) —
  lives in step-routine locals discarded on reject. Cutback retries the same
  cycle from the SAME committed generation AND the SAME baseline with the
  predictor multiplier shrunk: ``attempt_factor = baseline.factor *
  cutback_factor ** cutback_level``. For cycles after the first this scales
  the committed-increment predictor; for cycle 1 it uniformly scales the
  initial ``lam0`` jump (the beyond-parity extension of the legacy raise —
  legacy never cuts back).
- ``lam`` enters the program ONLY as a bound coordinate value: each attempt
  binds its trial ``lam`` into the per-evaluation ``ProgramPoint`` for
  offsets, loads, signals, and full coefficients. The program remains an
  immutable spec/evaluator, never a ``lam`` store. ``fhat`` is the
  load-coordinate column of ``plan.loads.coordinate_coefficients`` — nothing
  topological is re-derived per iteration.

Legacy behavior ported verbatim (parity contract with ``RiksSolver``):

- Cycle 1 starts at ``lam0 = 1.0`` and solves ``K(u_committed) da1 = lam0 *
  fhat`` with ``dlam1 = lam0``; later cycles scale the last COMMITTED
  increment by the factor (``da1 = factor * da_prev``, ``dlam1 = factor *
  dlam_prev``). The initial baseline mirrors the legacy globdat
  (``lam = 1.0``, ``Dlamprev = 1.0``, ``factor = totalFactor = 1.0``).
- The Newton loop never accepts the predictor state (legacy forces
  ``error = 1`` at loop entry), so every committed cycle applies at least
  one correction. Convergence is the legacy relative measure
  ``norm(res_red) / norm(lam * fhat_red)`` (absolute below the M18 reference
  floor). An attempt that reaches ``max_iterations`` corrections is rejected
  even when the residual just converged — the typed form of the legacy
  ``iterMax`` raise, keeping the same correction budget on converging paths.
- The normal-plane constraint anchors the FROZEN predictor ``da1`` for the
  whole attempt: two solves on the same factorization (``d1 = K \\ fhat``,
  ``d2 = K \\ res``) and ``ddlam = -dot(da1, d2) / dot(da1, d1)``,
  ``dq = ddlam * d1 + d2``, all in reduced coordinates. On homogeneous maps
  (selection-matrix ``P``) the reduced-space dots equal the legacy
  full-space dots exactly: every legacy vector is zero on constrained DOFs.
  Under MPC ties the prolongation mixes free DOFs (``P.T @ P != I``), so the
  reduced-space Riks metric differs from the legacy full-space ``C``-matrix
  metric — a documented, deliberate difference; the oracle decks are
  homogeneous.
- The adaptive factor on commit is ``0.5 ** (0.25 * (iiter - optiter))``
  with ``iiter`` the cycle's correction count, accumulated into
  ``total_factor``; when ``total_factor > max_factor`` the factor resets to
  ``1.0`` but ``total_factor`` is NOT reset — the legacy quirk, ported
  verbatim. Under ``fixed_step`` the factor update is skipped and the
  baseline factor keeps its initial value.
- Termination checks run AFTER the commit and baseline advance
  (``lam > max_lam`` or ``cycle > cycle_cap``), so the trajectory may end
  slightly beyond ``max_lam`` — the legacy behavior, kept for parity. The
  cap uses ``cycle > cycle_cap`` (legacy ``cycle > 1000``), fixing the
  prototype's off-by-one.

Performance discipline (the human directive honored): per Newton correction
exactly ONE operator evaluation round, ONE values-only ``refill_tangent``,
ONE ``splu`` (reused bitwise across the whole run when every Jacobian
channel is compiled ``linear``), and TWO back-substitutions. The cycle-1
predictor adds exactly one refill, one factorization, and one
back-substitution on nonlinear channels (or reuses the cached factorization
on linear ones). Workspace counters prove the discipline:
``tangent_refill_count`` equals the correction count plus the cycle-1
predictor, ``linear_solve_count`` equals twice the correction count plus the
predictor solve.

Operator ``REJECT_ITERATION`` retries from the same accepted state with a
damped iterate — the last correction halved in BOTH ``q`` and ``lam`` (and
in the cycle-increment accumulators) — budgeted against ``max_iterations``;
without a previous correction it escalates to cutback. Operator
``REJECT_STEP``, a singular or non-finite solve, a zero normal-plane
denominator, a non-finite trial iterate, divergence past
``divergence_ratio``, or an exhausted iteration budget all reject the
attempt into the same cutback protocol; exhausting ``max_cutbacks`` yields
the typed ``STEP_FAILED`` run status, never an exception. Observations are
unchanged from M18: at commit the full residual ``internal - lam * fhat``
yields reactions and constraint work through the coordinate map.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NoReturn

import numpy as np
from scipy.sparse.linalg import SuperLU, splu

from pyfem.v3.constraints.compile import (
  CompiledConstraintMap,
  evaluate_offsets,
  full_coefficients,
  reaction_forces,
  reduce_residual,
  reduce_tangent,
)
from pyfem.v3.driver._evaluation import (
  JacobianBatches,
  assemble_evaluation_balance,
  evaluate_operator_round,
  evaluation_channel_batches,
  primal_channel_requests,
  refill_evaluation_tangent,
  restart_nested_trial,
  stage_and_commit_trial,
)
from pyfem.v3.driver.continuation import (
  ArcLengthIterationRecord,
  ArcLengthResult,
  ArcLengthSettings,
  ArcLengthStepRecord,
  ArcLengthTermination,
  arc_length_continuation_from_evolution,
)
from pyfem.v3.driver.contracts import (
  _NORM_REFERENCE_FLOOR,
  DriverStatistics,
  DriverStatus,
  NestedSolveStatistics,
  SubstepObservation,
  SubstepStatus,
)
from pyfem.v3.driver.diagnostics import (
  DriverDiagnostic,
  DriverPreparationError,
)
from pyfem.v3.driver.nested import (
  PreparedNestedExecution,
  begin_nested_trial,
  close_nested_executions,
  collect_nested_activity,
  prepare_nested_executions,
  reject_nested_trial,
  snapshot_nested_activity,
)
from pyfem.v3.driver.plan import (
  DriverAssemblyPlan,
  compile_driver_plan,
  evaluate_loads,
  evaluate_signals,
)
from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.identity import require_same_generation
from pyfem.v3.model.operator import (
  EvaluationStatus,
  NestedEvaluationInput,
  OperatorEvaluation,
)
from pyfem.v3.model.system import CompiledSystem
from pyfem.v3.spec.diagnostics import SourceContext
from pyfem.v3.spec.program import (
  NodalLoadSpec,
  ProgramCoordinateValue,
  ProgramPoint,
)
from pyfem.v3.state import (
  CONTINUATION_EVOLUTION_STATE_SCHEMA,
  ContinuationEvolutionLayout,
  ContinuationEvolutionStore,
  StateCodecError,
  StateTransaction,
  StateTransactionOwner,
)

# Legacy continuation initial value (RiksSolver.__init__): the first cycle
# solves at lam0 = 1.0, and the run preamble probes the program at the same
# value. Ported verbatim, not redesigned.
_INITIAL_LOAD_PARAMETER = 1.0


def _preparation_fail(code: str, message: str) -> NoReturn:
  raise DriverPreparationError(
    (DriverDiagnostic(code=code, message=message, source=SourceContext()),)
  )


class _RiksWorkspace:
  """Disposable solve caches and counters; never authoritative state."""

  __slots__ = (
    "cached_factorization",
    "committed_substep_count",
    "cutback_count",
    "evaluation_count",
    "factorization_count",
    "factorization_reuse_count",
    "linear_solve_count",
    "nested_evaluation_count",
    "nested_factorization_count",
    "nested_factorization_reuse_count",
    "nested_iteration_count",
    "nested_linear_solve_count",
    "rejected_substep_count",
    "residual_assembly_count",
    "tangent_refill_count",
  )

  def __init__(self) -> None:
    self.cached_factorization: SuperLU | None = None
    self.committed_substep_count = 0
    self.cutback_count = 0
    self.evaluation_count = 0
    self.factorization_count = 0
    self.factorization_reuse_count = 0
    self.linear_solve_count = 0
    self.nested_evaluation_count = 0
    self.nested_factorization_count = 0
    self.nested_factorization_reuse_count = 0
    self.nested_iteration_count = 0
    self.nested_linear_solve_count = 0
    self.rejected_substep_count = 0
    self.residual_assembly_count = 0
    self.tangent_refill_count = 0


@dataclass(frozen=True, slots=True, eq=False)
class _AttemptOutcome:
  """One cycle attempt's typed result (committed values or last trial)."""

  committed: bool
  iterations: tuple[ArcLengthIterationRecord, ...]
  observation: SubstepObservation | None
  full: FinalizedArray | None
  lam: float
  point: ProgramPoint


class RiksDriver:
  """Typed Riks arc-length analysis over one exact system and map pair."""

  def __init__(
    self,
    system: CompiledSystem,
    coordinate_map: CompiledConstraintMap,
    loads: tuple[NodalLoadSpec, ...] = (),
    settings: ArcLengthSettings = ArcLengthSettings(),
  ) -> None:
    if type(system) is not CompiledSystem:
      msg = "the Riks driver requires an exact CompiledSystem"
      raise TypeError(msg)
    if type(coordinate_map) is not CompiledConstraintMap:
      msg = "the Riks driver requires an exact CompiledConstraintMap"
      raise TypeError(msg)
    if type(settings) is not ArcLengthSettings:
      msg = "the Riks driver requires exact ArcLengthSettings"
      raise TypeError(msg)
    load_indices = tuple(
      index
      for index, kind in enumerate(coordinate_map.coordinate_kinds)
      if kind == "load"
    )
    if not load_indices:
      _preparation_fail(
        "load-parameter-coordinate-required",
        "the Riks driver requires exactly one program coordinate of kind "
        "'load' to serve as the arc-length parameter",
      )
    if len(load_indices) > 1:
      _preparation_fail(
        "load-parameter-coordinate-ambiguous",
        "the Riks driver requires exactly one program coordinate of kind "
        "'load', found several",
      )
    self._system = system
    self._map = coordinate_map
    self._settings = settings
    self._plan = compile_driver_plan(system, coordinate_map, loads)
    if any(
      role.value == "external" for role, _scatter in self._plan.residual_role_scatters
    ):
      _preparation_fail(
        "unsupported-riks-external-channel",
        "Riks does not support operator EXTERNAL channels without their "
        "load-coordinate derivatives",
      )
    self._evolution_layout = ContinuationEvolutionLayout(
      schema=CONTINUATION_EVOLUTION_STATE_SCHEMA,
      reduced_dof_count=coordinate_map.reduced_dof_count,
      map_fingerprint=coordinate_map.content_fingerprint.digest,
    )
    self._workspace = _RiksWorkspace()
    self._requests = primal_channel_requests(system)
    (load_index,) = load_indices
    self._load_coordinate_name = coordinate_map.coordinate_names[load_index]
    fhat_full = np.array(
      self._plan.loads.coordinate_coefficients.values[:, load_index],
      dtype=np.float64,
      order="C",
      copy=True,
    )
    self._fhat_full = fhat_full
    self._fhat_reduced = np.array(
      reduce_residual(
        coordinate_map,
        FinalizedArray(fhat_full, dtype=np.float64),
      ).values,
      dtype=np.float64,
      order="C",
      copy=True,
    )
    declarations = tuple(
      operator.header.nested_execution for operator in system.operators
    )
    runtimes = prepare_nested_executions(
      tuple(
        None if declaration is None else declaration.recipe
        for declaration in declarations
      )
    )
    try:
      self._nested_inputs = tuple(
        None if runtime is None else NestedEvaluationInput(runtime)
        for runtime in runtimes
      )
      self._owner = StateTransactionOwner(
        system,
        nested_stores=tuple(
          runtime.store for runtime in runtimes if runtime is not None
        ),
      )
    except BaseException:
      close_nested_executions(runtimes)
      raise
    self._nested_executions = runtimes
    self._closed = False

  @property
  def plan(self) -> DriverAssemblyPlan:
    """Return the immutable compiled assembly plan."""
    return self._plan

  @property
  def owner(self) -> StateTransactionOwner:
    """Return the sole authoritative state owner."""
    return self._owner

  @property
  def settings(self) -> ArcLengthSettings:
    """Return the exact numeric policy."""
    return self._settings

  @property
  def nested_executions(self) -> tuple[PreparedNestedExecution | None, ...]:
    """Return this driver's execution-local nested runtimes, aligned to operators."""
    return self._nested_executions

  def close(self) -> None:
    """Release nested host workers and retained buffers; idempotent."""
    if self._closed:
      return
    close_nested_executions(self._nested_executions)
    self._closed = True

  def __enter__(self) -> RiksDriver:
    if self._closed:
      msg = "Riks driver is closed"
      raise RuntimeError(msg)
    return self

  def __exit__(self, *_exc: object) -> None:
    self.close()

  def __del__(self) -> None:
    try:
      self.close()
    except Exception:
      pass

  @property
  def evolution_layout(self) -> ContinuationEvolutionLayout:
    """Return the continuation evolution codec authority for this driver.

    Resume stores are built against this layout — it pins the arc-length
    evolution schema, this map's reduced DOF count, and its content
    fingerprint.
    """
    return self._evolution_layout

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
      nested_iteration_count=workspace.nested_iteration_count,
      nested_evaluation_count=workspace.nested_evaluation_count,
      nested_factorization_count=workspace.nested_factorization_count,
      nested_factorization_reuse_count=workspace.nested_factorization_reuse_count,
      nested_linear_solve_count=workspace.nested_linear_solve_count,
    )

  def _attempt_point(
    self,
    base_values: tuple[ProgramCoordinateValue, ...],
    lam: float,
  ) -> ProgramPoint:
    """Bind one trial load parameter onto the base point's other coordinates."""
    return ProgramPoint(
      (*base_values, ProgramCoordinateValue(self._load_coordinate_name, lam))
    )

  def _evaluate(
    self,
    transaction: StateTransaction,
    reduced: np.ndarray,
    point: ProgramPoint,
  ) -> tuple[
    FinalizedArray,
    tuple[OperatorEvaluation, ...],
    tuple[EvaluationStatus, ...],
  ]:
    """Run one counted operator evaluation round at one trial iterate."""
    full = full_coefficients(self._map, reduced, point)
    signal_inputs = evaluate_signals(self._plan, point)
    evaluations, statuses = evaluate_operator_round(
      self._system,
      self._plan,
      self._requests,
      full,
      transaction,
      signal_inputs,
      self._nested_inputs,
    )
    self._workspace.evaluation_count += 1
    return full, evaluations, statuses

  def _reduced_residual(
    self,
    evaluations: tuple[OperatorEvaluation, ...],
    external: np.ndarray,
  ) -> tuple[np.ndarray, np.ndarray]:
    """Assemble signed full residual and its reduced Newton right-hand side."""
    full_residual, rhs_full, _jacobian = assemble_evaluation_balance(
      self._plan,
      evaluations,
      external,
    )
    self._workspace.residual_assembly_count += 1
    reduced = reduce_residual(self._map, rhs_full).values
    return full_residual, np.asarray(reduced, dtype=np.float64)

  def _factorization(
    self,
    jacobian_batches: JacobianBatches,
  ) -> SuperLU | None:
    """Factor the reduced tangent, or return None for cutback.

    Exactly one values-only refill and one ``splu`` per call on nonlinear
    channels; compiled-linear plans reuse the cached factorization bitwise.
    """
    plan = self._plan
    workspace = self._workspace
    if plan.constant_tangent and workspace.cached_factorization is not None:
      workspace.factorization_reuse_count += 1
      return workspace.cached_factorization
    tangent = refill_evaluation_tangent(plan, jacobian_batches)
    workspace.tangent_refill_count += 1
    reduced_tangent = reduce_tangent(self._map, tangent)
    try:
      factorization = splu(reduced_tangent.tocsc())
    except RuntimeError:
      return None
    workspace.factorization_count += 1
    if plan.constant_tangent:
      workspace.cached_factorization = factorization
    return factorization

  def _back_substitute(
    self,
    factorization: SuperLU,
    rhs: np.ndarray,
  ) -> np.ndarray | None:
    """Run one counted back-substitution; None when the result is non-finite."""
    solution = factorization.solve(rhs)
    self._workspace.linear_solve_count += 1
    if not bool(np.isfinite(solution).all()):
      return None
    return np.asarray(solution, dtype=np.float64)

  def _accumulate_nested_statistics(
    self, statistics: tuple[NestedSolveStatistics, ...]
  ) -> None:
    workspace = self._workspace
    for cost in statistics:
      workspace.nested_iteration_count += cost.iteration_count
      workspace.nested_evaluation_count += cost.evaluation_count
      workspace.nested_factorization_count += cost.factorization_count
      workspace.nested_factorization_reuse_count += cost.factorization_reuse_count
      workspace.nested_linear_solve_count += cost.linear_solve_count

  def _cycle_attempt(
    self,
    baseline: ContinuationEvolutionStore,
    base_values: tuple[ProgramCoordinateValue, ...],
    attempt_factor: float,
  ) -> _AttemptOutcome:
    """Run one cycle attempt inside one open owner transaction."""
    coordinate_map = self._map
    settings = self._settings
    workspace = self._workspace
    free_dofs = coordinate_map.free_dofs.values
    transaction = self._owner.begin()
    closed = False
    nested_open = False
    iterations: list[ArcLengthIterationRecord] = []

    def rejected(lam: float, point: ProgramPoint) -> _AttemptOutcome:
      nonlocal closed
      transaction.reject()
      closed = True
      workspace.rejected_substep_count += 1
      return _AttemptOutcome(
        committed=False,
        iterations=tuple(iterations),
        observation=None,
        full=None,
        lam=lam,
        point=point,
      )

    try:
      begin_nested_trial(self._nested_executions)
      nested_open = True
      reduced_committed = np.array(
        transaction.accepted_physical.values[free_dofs],
        dtype=np.float64,
        order="C",
        copy=True,
      )
      # Predictor: Scope-B candidate anchored for the whole attempt.
      if baseline.cycle == 0:
        lam = _INITIAL_LOAD_PARAMETER * attempt_factor
        dlam1 = lam
        point = self._attempt_point(base_values, lam)
        _full, evaluations, statuses = self._evaluate(
          transaction,
          reduced_committed,
          point,
        )
        if (
          EvaluationStatus.REJECT_STEP in statuses
          or EvaluationStatus.REJECT_ITERATION in statuses
        ):
          return rejected(lam, point)
        factorization = self._factorization(evaluation_channel_batches(evaluations)[1])
        if factorization is None:
          return rejected(lam, point)
        da1 = self._back_substitute(factorization, lam * self._fhat_reduced)
        if da1 is None:
          return rejected(lam, point)
        restart_nested_trial(self._nested_executions)
      else:
        da1 = attempt_factor * baseline.da_prev
        dlam1 = attempt_factor * baseline.dlam_prev
        lam = baseline.lam + dlam1
      reduced = reduced_committed + da1
      da = np.array(da1, dtype=np.float64, order="C", copy=True)
      dlam = dlam1
      if not bool(np.isfinite(reduced).all()):
        return rejected(lam, self._attempt_point(base_values, lam))
      anchor_reduced: np.ndarray | None = None
      anchor_lam: float | None = None
      anchor_da: np.ndarray | None = None
      anchor_dlam: float | None = None
      increment: np.ndarray | None = None
      increment_lam: float | None = None
      increment_norm: float | None = None
      first_residual_norm: float | None = None
      for iteration in range(1, settings.max_iterations + 1):
        point = self._attempt_point(base_values, lam)
        full, evaluations, statuses = self._evaluate(transaction, reduced, point)
        if EvaluationStatus.REJECT_STEP in statuses:
          iterations.append(
            ArcLengthIterationRecord(
              iteration, EvaluationStatus.REJECT_STEP, None, None, lam, None
            )
          )
          return rejected(lam, point)
        if EvaluationStatus.REJECT_ITERATION in statuses:
          iterations.append(
            ArcLengthIterationRecord(
              iteration, EvaluationStatus.REJECT_ITERATION, None, None, lam, None
            )
          )
          if (
            anchor_reduced is None
            or anchor_lam is None
            or anchor_da is None
            or anchor_dlam is None
            or increment is None
            or increment_lam is None
          ):
            return rejected(lam, point)
          increment = 0.5 * increment
          increment_lam = 0.5 * increment_lam
          reduced = anchor_reduced + increment
          lam = anchor_lam + increment_lam
          da = anchor_da + increment
          dlam = anchor_dlam + increment_lam
          increment_norm = float(np.linalg.norm(increment))
          restart_nested_trial(self._nested_executions)
          continue
        external = evaluate_loads(self._plan.loads, point).values
        full_residual, residual_reduced = self._reduced_residual(evaluations, external)
        residual_norm = float(np.linalg.norm(residual_reduced))
        anchor_reduced = reduced
        anchor_lam = lam
        anchor_da = da
        anchor_dlam = dlam
        reference_norm = float(np.linalg.norm(lam * self._fhat_reduced))
        if reference_norm >= _NORM_REFERENCE_FLOOR:
          measure = residual_norm / reference_norm
        else:
          measure = residual_norm
        # The predictor state is never accepted (legacy forces error = 1 at
        # loop entry): every committed cycle applies at least one correction.
        if iteration > 1 and measure <= settings.tolerance:
          iterations.append(
            ArcLengthIterationRecord(
              iteration,
              EvaluationStatus.OK,
              residual_norm,
              increment_norm,
              lam,
              increment_lam,
            )
          )

          def mark_owner_committed() -> None:
            nonlocal closed
            closed = True

          stage_and_commit_trial(
            self._system,
            transaction,
            full,
            evaluations,
            self._nested_executions,
            mark_owner_committed,
          )
          nested_open = False
          workspace.committed_substep_count += 1
          reactions = reaction_forces(coordinate_map, full_residual)
          observation = SubstepObservation(
            reactions=reactions,
            constraint_work=float(np.dot(reactions.values, full.values)),
            full_residual_norm=float(np.linalg.norm(full_residual)),
            reduced_residual_norm=residual_norm,
          )
          corrections = iteration - 1
          # Scope-A baseline advance: commit path only, immediately after the
          # owner commit. The adaptive factor and the totalFactor reset quirk
          # are the legacy formulas, verbatim; the store advance itself is
          # infallible field writes, so owner commit and baseline advance
          # cannot separate.
          factor = baseline.factor
          total_factor = baseline.total_factor
          if not settings.fixed_step:
            factor = float(0.5 ** (0.25 * (corrections - settings.optimal_iterations)))
            total_factor *= factor
          if total_factor > settings.max_factor:
            factor = 1.0
          baseline.advance(
            lam=lam,
            da_prev=np.array(da, dtype=np.float64, order="C", copy=True),
            dlam_prev=dlam,
            factor=factor,
            total_factor=total_factor,
            cycle=baseline.cycle + 1,
            generation=self._owner.generation,
          )
          return _AttemptOutcome(
            committed=True,
            iterations=tuple(iterations),
            observation=observation,
            full=full,
            lam=lam,
            point=point,
          )
        if first_residual_norm is None:
          first_residual_norm = max(residual_norm, _NORM_REFERENCE_FLOOR)
        elif residual_norm > settings.divergence_ratio * first_residual_norm:
          iterations.append(
            ArcLengthIterationRecord(
              iteration, EvaluationStatus.OK, residual_norm, None, lam, None
            )
          )
          return rejected(lam, point)
        factorization = self._factorization(evaluation_channel_batches(evaluations)[1])
        if factorization is None:
          iterations.append(
            ArcLengthIterationRecord(
              iteration, EvaluationStatus.OK, residual_norm, None, lam, None
            )
          )
          return rejected(lam, point)
        d1 = self._back_substitute(factorization, self._fhat_reduced)
        if d1 is None:
          iterations.append(
            ArcLengthIterationRecord(
              iteration, EvaluationStatus.OK, residual_norm, None, lam, None
            )
          )
          return rejected(lam, point)
        d2 = self._back_substitute(factorization, residual_reduced)
        if d2 is None:
          iterations.append(
            ArcLengthIterationRecord(
              iteration, EvaluationStatus.OK, residual_norm, None, lam, None
            )
          )
          return rejected(lam, point)
        denominator = float(np.dot(da1, d1))
        if not math.isfinite(denominator) or denominator == 0.0:
          iterations.append(
            ArcLengthIterationRecord(
              iteration, EvaluationStatus.OK, residual_norm, None, lam, None
            )
          )
          return rejected(lam, point)
        ddlam = -float(np.dot(da1, d2)) / denominator
        dda = ddlam * d1 + d2
        increment = dda
        increment_lam = ddlam
        increment_norm = float(np.linalg.norm(dda))
        evaluated_lam = lam
        reduced = reduced + dda
        lam = lam + ddlam
        da = da + dda
        dlam = dlam + ddlam
        iterations.append(
          ArcLengthIterationRecord(
            iteration,
            EvaluationStatus.OK,
            residual_norm,
            increment_norm,
            evaluated_lam,
            increment_lam,
          )
        )
        if not bool(np.isfinite(reduced).all()) or not math.isfinite(lam):
          return rejected(lam, point)
        restart_nested_trial(self._nested_executions)
      return rejected(lam, point)
    except BaseException:
      if not closed:
        transaction.reject()
      raise
    finally:
      if nested_open:
        reject_nested_trial(self._nested_executions)

  def run(
    self,
    *,
    base_point: ProgramPoint,
    resume: ContinuationEvolutionStore | None = None,
  ) -> ArcLengthResult:
    """Advance the committed state along the autonomous arc-length schedule.

    ``base_point`` binds every declared program coordinate EXCEPT the load
    coordinate of kind ``'load'`` — the continuation owns the load parameter
    and binds it per attempt as a coordinate value. Binding the load
    coordinate in ``base_point`` is a contract error.

    ``resume`` continues a previous run's committed continuation baseline:
    pass the exact :class:`ContinuationEvolutionStore` that run advanced
    (virgin stores are seeded against :attr:`evolution_layout`). Without it
    the run starts a fresh virgin baseline at the owner's current
    generation. A resume store must match this driver's evolution layout
    (schema, reduced DOF count, and constraint-map fingerprint) and its
    generation must be the owner's CURRENT live generation: in-process
    resume is a live-generation identity check; cross-process lineage
    restore is the checkpoint-bundle follow-up's boundary, not this
    driver's.
    """
    if self._closed:
      msg = "Riks driver is closed"
      raise RuntimeError(msg)
    if type(base_point) is not ProgramPoint:
      msg = "the Riks driver base point must be an exact ProgramPoint value"
      raise TypeError(msg)
    if type(base_point.values) is not tuple:
      msg = "driver program point values must be an exact tuple"
      raise TypeError(msg)
    for item in base_point.values:
      if type(item) is not ProgramCoordinateValue:
        msg = "program point values must be exact ProgramCoordinateValue values"
        raise TypeError(msg)
      if item.name == self._load_coordinate_name:
        msg = (
          "the Riks driver base point must not bind the continuation load "
          f"coordinate {item.name!r}"
        )
        raise ValueError(msg)
    base_values = base_point.values
    probe = self._attempt_point(base_values, _INITIAL_LOAD_PARAMETER)
    evaluate_offsets(self._map, probe)
    evaluate_loads(self._plan.loads, probe)
    settings = self._settings
    workspace = self._workspace
    initial_generation = self._owner.generation
    if resume is None:
      baseline = ContinuationEvolutionStore.virgin(
        self._evolution_layout,
        initial_generation,
      )
    else:
      if type(resume) is not ContinuationEvolutionStore:
        msg = "the Riks driver resume state must be an exact ContinuationEvolutionStore"
        raise TypeError(msg)
      if resume.layout != self._evolution_layout:
        msg = (
          "the Riks driver resume store layout contradicts the driver's constraint map"
        )
        raise StateCodecError(msg)
      require_same_generation(
        resume.generation,
        initial_generation,
        context="Riks driver resume",
      )
      baseline = resume
    initial_continuation = arc_length_continuation_from_evolution(baseline.snapshot())
    records: list[ArcLengthStepRecord] = []
    while True:
      cycle = baseline.cycle + 1
      cutback_level = 0
      while True:
        attempt_factor = baseline.factor * settings.cutback_factor**cutback_level
        nested_cursors = snapshot_nested_activity(self._nested_executions)
        try:
          outcome = self._cycle_attempt(baseline, base_values, attempt_factor)
        except BaseException:
          nested_statistics, _ = collect_nested_activity(nested_cursors)
          self._accumulate_nested_statistics(nested_statistics)
          raise
        nested_statistics, nested_failure = collect_nested_activity(nested_cursors)
        self._accumulate_nested_statistics(nested_statistics)
        if outcome.committed:
          break
        cutback_level += 1
        if cutback_level > settings.max_cutbacks:
          records.append(
            ArcLengthStepRecord(
              cycle=cycle,
              point=outcome.point,
              lam=outcome.lam,
              factor=attempt_factor,
              status=SubstepStatus.FAILED,
              cutback_level=cutback_level,
              iterations=outcome.iterations,
              committed_ordinal=None,
              committed_coefficients=None,
              observation=None,
              nested_statistics=nested_statistics,
              nested_failure=nested_failure,
            )
          )
          return ArcLengthResult(
            status=DriverStatus.STEP_FAILED,
            termination_reason=None,
            base_point=base_point,
            records=tuple(records),
            statistics=self.statistics,
            initial_generation=initial_generation,
            final_generation=self._owner.generation,
            initial_continuation=initial_continuation,
            final_continuation=arc_length_continuation_from_evolution(
              baseline.snapshot()
            ),
            failed_cycle=cycle,
          )
        records.append(
          ArcLengthStepRecord(
            cycle=cycle,
            point=outcome.point,
            lam=outcome.lam,
            factor=attempt_factor,
            status=SubstepStatus.REJECTED,
            cutback_level=cutback_level,
            iterations=outcome.iterations,
            committed_ordinal=None,
            committed_coefficients=None,
            observation=None,
            nested_statistics=nested_statistics,
            nested_failure=nested_failure,
          )
        )
        workspace.cutback_count += 1
      records.append(
        ArcLengthStepRecord(
          cycle=cycle,
          point=outcome.point,
          lam=outcome.lam,
          factor=baseline.factor,
          status=SubstepStatus.COMMITTED,
          cutback_level=cutback_level,
          iterations=outcome.iterations,
          committed_ordinal=self._owner.generation.ordinal,
          committed_coefficients=outcome.full,
          observation=outcome.observation,
          nested_statistics=nested_statistics,
        )
      )
      # Termination is checked AFTER the commit and baseline advance: the
      # trajectory may end slightly beyond max_lam (legacy behavior, kept).
      if baseline.lam > settings.max_lam:
        termination = ArcLengthTermination.LOAD_PARAMETER_LIMIT
        break
      if baseline.cycle > settings.cycle_cap:
        termination = ArcLengthTermination.CYCLE_LIMIT
        break
    return ArcLengthResult(
      status=DriverStatus.COMPLETED,
      termination_reason=termination,
      base_point=base_point,
      records=tuple(records),
      statistics=self.statistics,
      initial_generation=initial_generation,
      final_generation=self._owner.generation,
      initial_continuation=initial_continuation,
      final_continuation=arc_length_continuation_from_evolution(baseline.snapshot()),
      failed_cycle=None,
    )
