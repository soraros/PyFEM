"""Typed contracts for the purified nonlinear static driver.

Expected numerical outcomes are reported through the typed statuses below and
through :class:`pyfem.v3.model.operator.EvaluationStatus`, never through
exceptions; contract violations (foreign types, malformed shapes) remain
``TypeError``/``DriverPreparationError`` at the boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.identity import StateGeneration
from pyfem.v3.model.operator import EvaluationStatus
from pyfem.v3.spec.program import ProgramPoint

_NORM_REFERENCE_FLOOR = 1.0e-16
_SLOW_CONVERGENCE_CONTRACTION = 0.5


class DriverStatus(Enum):
  """Typed outcome of one driver run."""

  COMPLETED = "completed"
  STEP_FAILED = "step-failed"


class SubstepStatus(Enum):
  """Typed outcome of one attempted substep."""

  COMMITTED = "committed"
  REJECTED = "rejected"
  FAILED = "failed"


class BudgetExhaustionTrend(Enum):
  """Measured residual-trajectory classification at iteration-budget exhaustion.

  ``SLOW_CONVERGENCE`` is the near-miss: the exhausted attempt's measured
  residual shows sustained contraction, so the substep plausibly ran out of
  iteration budget rather than failing — raising ``max_iterations`` is the
  documented remedy. ``NON_CONVERGENT`` covers every other trajectory shape
  (oscillation, stagnation, growth below the ``divergence_ratio`` trip, or
  too few measured iterations to establish a trend): the substep is
  genuinely failing.
  """

  SLOW_CONVERGENCE = "slow-convergence"
  NON_CONVERGENT = "non-convergent"


@dataclass(frozen=True, slots=True)
class NonlinearStaticSettings:
  """Exact numeric policy for the nonlinear static schedule.

  ``tolerance`` bounds the reduced residual norm relative to the reduced
  external-force norm (absolute below ``_NORM_REFERENCE_FLOOR``, matching the
  legacy ``DofSpace`` convention). ``max_iterations`` budgets Newton
  corrections and operator retry-iteration outcomes together per substep.
  ``max_cutbacks`` bounds the halving cascade per target point; exhausting it
  is the typed ``STEP_FAILED`` outcome, not an exception.
  ``min_substep_size`` floors the substep size as a fraction of the target
  interval: rejecting a substep already at or below the floor means the
  schedule cannot advance by any meaningful increment, which is the same
  typed ``STEP_FAILED`` outcome. Consecutive-rejection sizes bottom out at
  ``cutback_factor ** max_cutbacks`` of the initial size, far above any
  honest floor, so the floor can only trip after micro-commits that make no
  real progress — never on a converging trajectory.
  """

  tolerance: float = 1.0e-10
  max_iterations: int = 25
  max_cutbacks: int = 6
  cutback_factor: float = 0.5
  growth_factor: float = 2.0
  divergence_ratio: float = 1.0e8
  min_substep_size: float = 1.0e-12

  def __post_init__(self) -> None:
    tolerance = self.tolerance
    if type(tolerance) is not float or not math.isfinite(tolerance) or tolerance <= 0.0:
      msg = "nonlinear static tolerance must be a positive finite exact float"
      raise ValueError(msg)
    if type(self.max_iterations) is not int or self.max_iterations < 1:
      msg = "nonlinear static max_iterations must be a positive exact int"
      raise ValueError(msg)
    if type(self.max_cutbacks) is not int or self.max_cutbacks < 0:
      msg = "nonlinear static max_cutbacks must be a nonnegative exact int"
      raise ValueError(msg)
    cutback_factor = self.cutback_factor
    if (
      type(cutback_factor) is not float
      or not math.isfinite(cutback_factor)
      or not 0.0 < cutback_factor < 1.0
    ):
      msg = "nonlinear static cutback_factor must lie strictly between zero and one"
      raise ValueError(msg)
    growth_factor = self.growth_factor
    if (
      type(growth_factor) is not float
      or not math.isfinite(growth_factor)
      or growth_factor < 1.0
    ):
      msg = "nonlinear static growth_factor must be at least one"
      raise ValueError(msg)
    divergence_ratio = self.divergence_ratio
    if (
      type(divergence_ratio) is not float
      or not math.isfinite(divergence_ratio)
      or divergence_ratio <= 1.0
    ):
      msg = "nonlinear static divergence_ratio must exceed one"
      raise ValueError(msg)
    min_substep_size = self.min_substep_size
    if (
      type(min_substep_size) is not float
      or not math.isfinite(min_substep_size)
      or not 0.0 < min_substep_size < 1.0
    ):
      msg = "nonlinear static min_substep_size must lie strictly between zero and one"
      raise ValueError(msg)


@dataclass(frozen=True, slots=True, eq=False)
class IterationRecord:
  """One Newton iteration inside a substep.

  ``residual_norm`` is ``None`` when the operator rejected the evaluation
  (its channels may be empty); ``increment_norm`` is ``None`` before the
  first linear solve of the substep.
  """

  iteration: int
  status: EvaluationStatus
  residual_norm: float | None
  increment_norm: float | None


@dataclass(frozen=True, slots=True, eq=False)
class ParameterSensitivityObservation:
  """First-order sensitivity of the committed coefficients to one parameter.

  ``parameter_id`` is the spec-level parameter name the run requested (for
  the stateful continuum slice, the material law's declared
  ``parameter_names`` entry). ``coefficients`` is the full-space
  ``d(u)/d(parameter)`` at the committed point: the implicit-function-theorem
  solution ``K_q^{-1} (-P.T dR/dp)`` on the substep's converged tangent
  factorization, prolonged to the full basis (``P @ d(q)/d(parameter)`` —
  prescribed offsets do not depend on material parameters). The derivative
  channel is evaluated with the entering committed state held fixed, so the
  observation is exact whenever the entering state is parameter-independent
  (the virgin state — every first step, elastic or plastic), and the
  hardening-parameter columns are exactly zero on every elastic step; from
  the second step on it is the increment's sensitivity, with the entering
  state's own parameter dependence carried by the declared follow-up
  state-derivative channels (the M48 survey's v2 scope).
  """

  parameter_id: str
  coefficients: FinalizedArray


@dataclass(frozen=True, slots=True, eq=False)
class SubstepObservation:
  """Reaction and energy observations from the FULL residual at commit.

  ``reactions`` is the direct constraint-reaction field observed through the
  coordinate map (the full residual on constrained DOFs, zero elsewhere).
  ``constraint_work`` is ``f_c . u`` under the map basis. Both are derived
  from the same full residual that drove convergence; nothing is
  re-evaluated. ``sensitivities`` carries the per-parameter IFT coefficient
  sensitivities of the committed point in request order — empty unless the
  run requested sensitivity parameters.
  """

  reactions: FinalizedArray
  constraint_work: float
  full_residual_norm: float
  reduced_residual_norm: float
  sensitivities: tuple[ParameterSensitivityObservation, ...] = ()


@dataclass(frozen=True, slots=True, eq=False)
class BudgetExhaustionObservation:
  """Measured residual trend behind one iteration-budget exhaustion.

  ``first_residual_norm`` and ``final_residual_norm`` are the first and last
  measured reduced residual norms of the exhausted attempt;
  ``decreasing_step_count`` of the ``measured_step_count - 1`` steps
  decreased the norm. The classification rule is fixed:
  ``SLOW_CONVERGENCE`` when the final measured norm contracts to at most
  ``_SLOW_CONVERGENCE_CONTRACTION`` of the first AND a strict majority of
  measured steps decrease; ``NON_CONVERGENT`` otherwise (including fewer
  than two measured norms, where no trend exists). The fields carry the
  complete evidence behind the classification.
  """

  trend: BudgetExhaustionTrend
  first_residual_norm: float
  final_residual_norm: float
  decreasing_step_count: int
  measured_step_count: int


@dataclass(frozen=True, slots=True, eq=False)
class SubstepRecord:
  """One attempted substep: typed status, protocol trace, and observation.

  ``budget_exhaustion`` is non-``None`` exactly when the attempt ended by
  exhausting the iteration budget — never on ``COMMITTED`` records, and
  never on rejections with another typed cause (operator reject, singular
  or non-finite solve, non-finite iterate, or the divergence trip).
  """

  target_index: int
  progress: float
  point: ProgramPoint
  status: SubstepStatus
  cutback_level: int
  iterations: tuple[IterationRecord, ...]
  committed_ordinal: int | None
  observation: SubstepObservation | None
  budget_exhaustion: BudgetExhaustionObservation | None


@dataclass(frozen=True, slots=True, eq=False)
class DriverStatistics:
  """Workspace counters proving plan reuse and factorization caching."""

  evaluation_count: int
  residual_assembly_count: int
  tangent_refill_count: int
  factorization_count: int
  factorization_reuse_count: int
  linear_solve_count: int
  cutback_count: int
  committed_substep_count: int
  rejected_substep_count: int


@dataclass(frozen=True, slots=True, eq=False)
class NonlinearStaticResult:
  """Typed run outcome: schedule trace, counters, and generation lineage."""

  status: DriverStatus
  base_point: ProgramPoint
  target_points: tuple[ProgramPoint, ...]
  records: tuple[SubstepRecord, ...]
  statistics: DriverStatistics
  initial_generation: StateGeneration
  final_generation: StateGeneration
  failed_target_index: int | None
