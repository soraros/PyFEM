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


class DriverStatus(Enum):
  """Typed outcome of one driver run."""

  COMPLETED = "completed"
  STEP_FAILED = "step-failed"


class SubstepStatus(Enum):
  """Typed outcome of one attempted substep."""

  COMMITTED = "committed"
  REJECTED = "rejected"
  FAILED = "failed"


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
class SubstepObservation:
  """Reaction and energy observations from the FULL residual at commit.

  ``reactions`` is the direct constraint-reaction field observed through the
  coordinate map (the full residual on constrained DOFs, zero elsewhere).
  ``constraint_work`` is ``f_c . u`` under the map basis. Both are derived
  from the same full residual that drove convergence; nothing is
  re-evaluated.
  """

  reactions: FinalizedArray
  constraint_work: float
  full_residual_norm: float
  reduced_residual_norm: float


@dataclass(frozen=True, slots=True, eq=False)
class SubstepRecord:
  """One attempted substep: typed status, protocol trace, and observation."""

  target_index: int
  progress: float
  point: ProgramPoint
  status: SubstepStatus
  cutback_level: int
  iterations: tuple[IterationRecord, ...]
  committed_ordinal: int | None
  observation: SubstepObservation | None


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
