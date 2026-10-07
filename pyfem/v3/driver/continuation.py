"""Typed contracts for the purified Riks arc-length continuation driver.

The arc-length schedule reports its expected numerical outcomes through the
same typed statuses as the nonlinear static driver (:class:`DriverStatus`,
:class:`SubstepStatus`, :class:`pyfem.v3.model.operator.EvaluationStatus`),
never through exceptions; contract violations (foreign types, malformed
shapes) remain ``TypeError``/``ValueError``/``DriverPreparationError`` at the
boundary. Reaching ``max_lam`` or the cycle cap is a successful schedule
outcome and is reported through the typed
:class:`ArcLengthTermination` reason, keeping the M18 status semantics
stable.

The continuation baseline is the committed Scope-A channel of the two-scope
continuation design: it lives in the request-owned
:class:`pyfem.v3.state.evolution.ContinuationEvolutionStore`, is written only
on the substep commit path immediately after the owner commit, is never
touched by reject paths, and is returned in the result. Per-attempt
candidates (the anchored predictor, the trial load parameter, the cutback
identity, the iteration trail) are step-routine locals discarded on reject.
The adapters below convert between the driver-surface
:class:`ArcLengthContinuationState` and the store's codec-backed
:class:`pyfem.v3.state.evolution.ContinuationEvolutionState`; the two types
share one field set and semantics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from pyfem.v3.driver.contracts import (
  DriverStatistics,
  DriverStatus,
  SubstepObservation,
  SubstepStatus,
)
from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.identity import StateGeneration
from pyfem.v3.model.operator import EvaluationStatus
from pyfem.v3.spec.program import ProgramPoint
from pyfem.v3.state.evolution import ContinuationEvolutionState


class ArcLengthTermination(Enum):
  """Typed successful-schedule termination of an arc-length run."""

  LOAD_PARAMETER_LIMIT = "load-parameter-limit"
  CYCLE_LIMIT = "cycle-limit"


@dataclass(frozen=True, slots=True)
class ArcLengthSettings:
  """Exact numeric policy for the Riks arc-length schedule.

  ``tolerance`` bounds the reduced residual norm relative to the reduced
  ``lam * fhat`` norm (absolute below the M18 reference floor, matching the
  legacy ``DofSpace`` convention). ``max_iterations`` budgets the Newton
  evaluations per cycle attempt and plays the legacy ``iterMax`` role: an
  attempt that reaches ``max_iterations`` corrections is rejected even when
  the residual just converged, the typed form of the legacy raise.
  ``optimal_iterations`` is the legacy ``optiter`` of the adaptive-factor
  formula; ``fixed_step`` pins the predictor factor at its initial value;
  ``max_lam`` and ``max_factor`` are the legacy ``maxLam``/``maxFactor``.
  Termination checks run after each commit (``lam > max_lam`` or
  ``cycle > cycle_cap``), so the trajectory may end slightly beyond
  ``max_lam`` — the legacy behavior, kept for parity. ``max_cutbacks`` and
  ``cutback_factor`` bound the predictor-multiplier shrink cascade per cycle;
  exhausting it is the typed ``STEP_FAILED`` outcome, not an exception.
  ``divergence_ratio`` rejects an attempt whose residual grows past the
  first predictor-state residual by that factor.
  """

  tolerance: float = 1.0e-10
  max_iterations: int = 10
  optimal_iterations: int = 5
  fixed_step: bool = False
  max_lam: float = 1.0e20
  max_factor: float = 1.0e20
  cycle_cap: int = 1000
  max_cutbacks: int = 6
  cutback_factor: float = 0.5
  divergence_ratio: float = 1.0e8

  def __post_init__(self) -> None:
    tolerance = self.tolerance
    if type(tolerance) is not float or not math.isfinite(tolerance) or tolerance <= 0.0:
      msg = "arc-length tolerance must be a positive finite exact float"
      raise ValueError(msg)
    if type(self.max_iterations) is not int or self.max_iterations < 1:
      msg = "arc-length max_iterations must be a positive exact int"
      raise ValueError(msg)
    if type(self.optimal_iterations) is not int or self.optimal_iterations < 1:
      msg = "arc-length optimal_iterations must be a positive exact int"
      raise ValueError(msg)
    if type(self.fixed_step) is not bool:
      msg = "arc-length fixed_step must be an exact bool"
      raise ValueError(msg)
    max_lam = self.max_lam
    if type(max_lam) is not float or not math.isfinite(max_lam) or max_lam <= 0.0:
      msg = "arc-length max_lam must be a positive finite exact float"
      raise ValueError(msg)
    max_factor = self.max_factor
    if (
      type(max_factor) is not float
      or not math.isfinite(max_factor)
      or max_factor <= 0.0
    ):
      msg = "arc-length max_factor must be a positive finite exact float"
      raise ValueError(msg)
    if type(self.cycle_cap) is not int or self.cycle_cap < 1:
      msg = "arc-length cycle_cap must be a positive exact int"
      raise ValueError(msg)
    if type(self.max_cutbacks) is not int or self.max_cutbacks < 0:
      msg = "arc-length max_cutbacks must be a nonnegative exact int"
      raise ValueError(msg)
    cutback_factor = self.cutback_factor
    if (
      type(cutback_factor) is not float
      or not math.isfinite(cutback_factor)
      or not 0.0 < cutback_factor < 1.0
    ):
      msg = "arc-length cutback_factor must lie strictly between zero and one"
      raise ValueError(msg)
    divergence_ratio = self.divergence_ratio
    if (
      type(divergence_ratio) is not float
      or not math.isfinite(divergence_ratio)
      or divergence_ratio <= 1.0
    ):
      msg = "arc-length divergence_ratio must exceed one"
      raise ValueError(msg)


@dataclass(frozen=True, slots=True, eq=False)
class ArcLengthContinuationState:
  """Committed continuation baseline (Scope A), written only at commit.

  ``lam`` is the committed load parameter; ``da_prev`` the committed total
  reduced displacement increment of the last committed cycle; ``dlam_prev``
  its load-parameter part; ``factor`` the predictor multiplier the next
  cycle's first attempt uses; ``total_factor`` the legacy ``totalFactor``
  accumulator of the adaptive-factor reset quirk (it is never reset, ported
  verbatim); ``cycle`` the number of committed cycles of this run.
  """

  lam: float
  da_prev: FinalizedArray
  dlam_prev: float
  factor: float
  total_factor: float
  cycle: int


def arc_length_continuation_from_evolution(
  state: ContinuationEvolutionState,
) -> ArcLengthContinuationState:
  """Adapt one store snapshot into the driver result surface.

  The two types share one field set and semantics; the immutable
  ``da_prev`` carrier passes through without copying.
  """
  if type(state) is not ContinuationEvolutionState:
    msg = (
      "arc-length continuation adaptation requires an exact ContinuationEvolutionState"
    )
    raise TypeError(msg)
  return ArcLengthContinuationState(
    lam=state.lam,
    da_prev=state.da_prev,
    dlam_prev=state.dlam_prev,
    factor=state.factor,
    total_factor=state.total_factor,
    cycle=state.cycle,
  )


def continuation_evolution_from_arc_length(
  state: ArcLengthContinuationState,
) -> ContinuationEvolutionState:
  """Adapt one driver-surface continuation state into the codec state.

  The two types share one field set and semantics; the immutable
  ``da_prev`` carrier passes through without copying.
  """
  if type(state) is not ArcLengthContinuationState:
    msg = (
      "continuation evolution adaptation requires an exact ArcLengthContinuationState"
    )
    raise TypeError(msg)
  return ContinuationEvolutionState(
    lam=state.lam,
    da_prev=state.da_prev,
    dlam_prev=state.dlam_prev,
    factor=state.factor,
    total_factor=state.total_factor,
    cycle=state.cycle,
  )


@dataclass(frozen=True, slots=True, eq=False)
class ArcLengthIterationRecord:
  """One Newton evaluation inside a cycle attempt.

  ``residual_norm`` and ``load_parameter`` describe the evaluated iterate
  (the first record of an attempt covers the predictor state). On an
  unconverged OK record ``increment_norm`` and ``delta_lam`` describe the
  correction computed at that iteration's tail, which leads to the next
  iterate; on the committing record they describe the correction that led to
  the converged iterate. On operator-rejected evaluations
  (``REJECT_STEP``/``REJECT_ITERATION``) ``residual_norm``,
  ``increment_norm``, and ``delta_lam`` are all ``None``. On attempts
  rejected after an OK evaluation (divergence, a singular or non-finite
  solve, a zero or non-finite normal-plane denominator) ``residual_norm``
  stays set while ``increment_norm`` and ``delta_lam`` are ``None``.
  ``load_parameter`` tracks the rejected trial's ``lam`` in both cases.
  """

  iteration: int
  status: EvaluationStatus
  residual_norm: float | None
  increment_norm: float | None
  load_parameter: float
  delta_lam: float | None


@dataclass(frozen=True, slots=True, eq=False)
class ArcLengthStepRecord:
  """One attempted cycle: typed status, continuation values, and trace.

  ``lam`` and ``point`` are the committed load parameter and its bound
  program point for a committed cycle, the last trial values for a rejected
  or failed attempt. ``factor`` is the post-commit baseline predictor factor
  for a committed cycle (what the next predictor will use) and the attempt
  factor for a rejected or failed attempt. ``committed_coefficients`` is the
  committed full-space coefficient vector — the per-cycle program-history
  point returned as per-run data.
  """

  cycle: int
  point: ProgramPoint
  lam: float
  factor: float
  status: SubstepStatus
  cutback_level: int
  iterations: tuple[ArcLengthIterationRecord, ...]
  committed_ordinal: int | None
  committed_coefficients: FinalizedArray | None
  observation: SubstepObservation | None


@dataclass(frozen=True, slots=True, eq=False)
class ArcLengthResult:
  """Typed run outcome: schedule trace, counters, and generation lineage.

  ``termination_reason`` is the typed successful-schedule limit that ended
  the run (``None`` exactly when ``status`` is ``STEP_FAILED``).
  ``initial_continuation``/``final_continuation`` bracket the run's committed
  continuation baseline; ``failed_cycle`` is the cycle whose cutback budget
  was exhausted, if any.
  """

  status: DriverStatus
  termination_reason: ArcLengthTermination | None
  base_point: ProgramPoint
  records: tuple[ArcLengthStepRecord, ...]
  statistics: DriverStatistics
  initial_generation: StateGeneration
  final_generation: StateGeneration
  initial_continuation: ArcLengthContinuationState
  final_continuation: ArcLengthContinuationState
  failed_cycle: int | None
