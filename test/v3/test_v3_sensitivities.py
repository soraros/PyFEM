# SPDX-License-Identifier: MIT

"""Driver parameter sensitivities (M57): IFT solves on committed states.

The S1B driver half of the M48 blueprint: a run requesting
``sensitivity_parameters`` solves, at every COMMITTED substep, the implicit
function theorem system ``K_q (dq/dp) = -P.T dR/dp`` per parameter — one
extra evaluation sweep of the declared derivative channels at the converged
point (entering committed state held fixed) and ONE back-substitution per
parameter on the factorization the converged Newton loop last used. This
battery pins:

- request-time validation: undeclared/duplicate parameters and channel-free
  systems are rejected with coded diagnostics before any substep runs;
- the documented M48 J2 cantilever vehicle (load-controlled, single step
  from virgin, tolerance 1e-12): IFT sensitivities reproduce the M48
  agreement class against full-resolve central FD of the solve (observed
  <= 3.4e-10 at the tabulated stencils; pins at 1e-9), including M48's
  recorded dominant ``d(u)/d(initial_yield_stress) = -1.7010667e-04``;
- the cost contract per committed step: one extra evaluation sweep total
  (batched over parameters), one back-substitution and one right-hand-side
  assembly per parameter, ZERO extra factorizations (FD costs two full
  resolves per parameter instead) — and the primal trajectory is untouched
  (counters and schedule structure exactly; floats to rtol 1e-12, covering
  the driver's ambient run-to-run 1-ulp iteration-history floor measured on
  this deck — two identical primal runs differ there while their committed
  states agree bitwise);
- edge cases exact: elastic steps carry exactly zero hardening-parameter
  columns (the elastic map does not reference them) and machine-exact
  elastic-constant columns from the virgin state; a zero-advance (plateau)
  substep — the one schedule that converges with no Newton solve — carries
  exactly zero sensitivities and factorizes the converged tangent once,
  counted truthfully;
- schedule hygiene: sensitivity observations appear only on COMMITTED
  records, never on rejected/failed ones, and the budget-exhaustion
  classification trail is identical with and without the request;
- the M56-deviation-(c) closure, test-side: the compiled operator's
  derivative columns are FD-verified at the exact (entering state,
  converged point) pair the driver consumes them at — the compiler's own
  virgin-state probe point is degenerate for FD (exactly zero columns), so
  the nonzero-state probe lives here.

Fixed-entering-state semantics (the M56 channel contract): the per-step
derivative holds the entering committed state fixed, so it is exact for a
parameter-independent entering state — the virgin state, so every first
step, elastic or plastic — and the hardening-parameter columns are exactly
zero on every elastic step (the elastic map does not reference them). From
the second step on the entering state's stress/strain rows are themselves
parameter-dependent, even on elastic trajectories, so the column is the
increment's sensitivity: the entering state's own parameter dependence
needs state-derivative channels, the M48 survey's declared v2 boundary.
The two-step elastic test pins the boundary by name (total FD = 2x the
step-2 column for the linear map).
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from pyfem.v3 import authoring
from pyfem.v3.compile.continuum import (
  plasticity_reference_registry,
  q8_reference_registry,
)
from pyfem.v3.compile.system import compile_system
from pyfem.v3.constraints import (
  CompiledConstraintMap,
  compile_constraint_map,
  reduce_residual,
)
from pyfem.v3.driver import (
  DriverPreparationError,
  DriverStatus,
  NonlinearStaticDriver,
  NonlinearStaticResult,
  NonlinearStaticSettings,
  SubstepStatus,
  assemble_internal_force,
  assemble_parameter_rhs,
  compile_sensitivity_program,
)
from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.operator import (
  ChannelRequest,
  CompiledOperator,
  OperatorEvaluation,
  OperatorEvaluationInput,
)
from pyfem.v3.spec import ProgramCoordinateSpec
from pyfem.v3.spec.program import ProgramCoordinateValue, ProgramPoint

# The documented M48 cantilever deck (test_v3_driver_nonlinear.py's vehicle).
_E = 210000.0
_NU = 0.3
_SYIELD = 250.0
_HARD = 1000.0
_PARAMETERS = (_E, _NU, _SYIELD, _HARD)
_PARAM_NAMES = (
  "youngs_modulus",
  "poisson_ratio",
  "initial_yield_stress",
  "hardening_slope",
)
_DERIVATIVE_CHANNELS = tuple(f"dinternal-force/d{name}" for name in _PARAM_NAMES)

# Central-FD-of-solve stencils per parameter, chosen from measured h-tables
# on the documented deck (max abs deviation vs stencil: E 1.1e-13 @ 210 /
# 1.1e-15 @ 21 / 6.8e-18 @ 2.1; nu 1.4e-9 @ 3e-4 / 1.3e-11 @ 3e-5 / 3.1e-12
# @ 3e-6; sy0 1.6e-4 @ 2.5 / 3.3e-10 @ 0.25 / 3.3e-12 @ 0.025; hard 3.4e-15
# @ 10 / 1.5e-17 @ 1). The pinned legs sit on the h-independent plateau.
_FD_STEPS = (21.0, 3.0e-5, 0.25, 1.0)
# IFT-vs-FD agreement pin: the M48 agreement class (M48 measured 3.3e-10 on
# this deck), ~3x headroom over the worst tabulated observation.
_IFT_FD_ATOL = 1.0e-9


def _j2_cantilever_deck(
  parameters: tuple[float, ...] = _PARAMETERS,
  *,
  settings: NonlinearStaticSettings | None = None,
) -> tuple[NonlinearStaticDriver, CompiledConstraintMap]:
  mesh = authoring.quad8_patch(4, 1, width=4.0, height=1.0)
  material = authoring.plasticity(*parameters, id="steel")
  model = authoring.small_strain_continuum(mesh, material=material)
  system = compile_system(model, plasticity_reference_registry())
  left = tuple(node.id for node in mesh.nodes if node.coordinates[0] == 0.0)
  right = tuple(
    sorted(
      (node for node in mesh.nodes if node.coordinates[0] == 4.0),
      key=lambda node: node.coordinates[1],
    )
  )
  coordinate_map = compile_constraint_map(
    system,
    constraints=authoring.fixed(left, ("x", "y")),
    coordinates=(ProgramCoordinateSpec(name="load", kind="load"),),
  )
  loads = tuple(
    authoring.nodal_load(node.id, "y", fraction)
    for node, fraction in zip(right, (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0), strict=True)
  )
  driver = NonlinearStaticDriver(
    system,
    coordinate_map,
    loads,
    settings or NonlinearStaticSettings(tolerance=1.0e-12),
  )
  return driver, coordinate_map


def _j2_cantilever(
  parameters: tuple[float, ...] = _PARAMETERS,
  *,
  settings: NonlinearStaticSettings | None = None,
) -> NonlinearStaticDriver:
  return _j2_cantilever_deck(parameters, settings=settings)[0]


def _points(*values: float) -> tuple[ProgramPoint, ...]:
  return tuple(
    ProgramPoint((ProgramCoordinateValue("load", value),)) for value in values
  )


def _run(
  driver: NonlinearStaticDriver,
  *loads: float,
  sensitivity_parameters: tuple[str, ...] = (),
) -> NonlinearStaticResult:
  points = _points(*loads)
  return driver.run(
    base_point=points[0],
    target_points=points[1:],
    sensitivity_parameters=sensitivity_parameters,
  )


def _solved_coefficients(
  parameters: tuple[float, ...],
  schedule: tuple[float, ...],
) -> np.ndarray:
  driver = _j2_cantilever(parameters)
  result = _run(driver, *schedule)
  assert result.status is DriverStatus.COMPLETED
  return np.array(driver.owner.accepted_physical().values, copy=True)


def _fd_of_solve(
  index: int,
  schedule: tuple[float, ...],
  step: float,
) -> np.ndarray:
  """Central FD of the converged solve: two full resolves per parameter."""
  plus = list(_PARAMETERS)
  plus[index] += step
  minus = list(_PARAMETERS)
  minus[index] -= step
  return (
    _solved_coefficients(tuple(plus), schedule)
    - _solved_coefficients(tuple(minus), schedule)
  ) / (2.0 * step)


def _evaluate(
  operator: CompiledOperator,
  values: np.ndarray,
  accepted: np.ndarray,
  request: ChannelRequest,
) -> OperatorEvaluation:
  return operator.evaluate(
    OperatorEvaluationInput(
      port_values=(
        FinalizedArray(
          np.array(values, dtype=np.float64, order="C", copy=True),
          dtype=np.float64,
        ),
      ),
      accepted_state=FinalizedArray(
        np.array(accepted, dtype=np.float64, order="C", copy=True),
        dtype=np.float64,
      ),
      signals=(),
      request=request,
    )
  )


# --- request-time validation ---------------------------------------------------


def test_sensitivity_request_is_validated_before_any_substep() -> None:
  driver = _j2_cantilever()
  # Contract violations on the request shape itself remain TypeErrors.
  with pytest.raises(TypeError, match="exact tuple"):
    _run(driver, 0.0, 20.0, sensitivity_parameters=["initial_yield_stress"])  # type: ignore[arg-type]
  with pytest.raises(TypeError, match="exact tuple"):
    _run(driver, 0.0, 20.0, sensitivity_parameters="initial_yield_stress")  # type: ignore[arg-type]
  with pytest.raises(TypeError, match="exact strings"):
    _run(driver, 0.0, 20.0, sensitivity_parameters=("initial_yield_stress", 5))  # type: ignore[arg-type]
  # A parameter declared twice is ambiguous; a parameter no operator declares
  # a derivative channel for can never assemble an honest right-hand side.
  with pytest.raises(DriverPreparationError) as duplicate:
    _run(
      driver,
      0.0,
      20.0,
      sensitivity_parameters=("initial_yield_stress", "initial_yield_stress"),
    )
  assert duplicate.value.diagnostics[0].code == "duplicate-sensitivity-parameter"
  with pytest.raises(DriverPreparationError) as unknown:
    _run(driver, 0.0, 20.0, sensitivity_parameters=("bulk_modulus",))
  assert unknown.value.diagnostics[0].code == "unknown-sensitivity-parameter"
  assert "bulk_modulus" in unknown.value.diagnostics[0].message
  # Request-time means before any substep: no evaluation ran and the owner
  # never left the initial generation.
  assert driver.statistics.evaluation_count == 0
  assert driver.owner.generation.ordinal == 0
  # A channel-free system (linear-elastic operator, no derivative channels)
  # rejects every sensitivity request — even for spec-level parameter names.
  mesh = authoring.quad8_patch(1, 1)
  model = authoring.small_strain_continuum(
    mesh, material=authoring.linear_elastic(1.0e6, 0.25, id="lin")
  )
  system = compile_system(model, q8_reference_registry())
  coordinate_map = compile_constraint_map(
    system,
    constraints=authoring.fixed(
      tuple(node.id for node in mesh.nodes if node.coordinates[0] == 0.0),
      ("x", "y"),
    ),
    coordinates=(ProgramCoordinateSpec(name="load", kind="load"),),
  )
  elastic_driver = NonlinearStaticDriver(system, coordinate_map, ())
  with pytest.raises(DriverPreparationError) as channel_free:
    _run(elastic_driver, 0.0, 1.0, sensitivity_parameters=("youngs_modulus",))
  assert channel_free.value.diagnostics[0].code == "unknown-sensitivity-parameter"


def test_sensitivity_program_resolution_and_rhs_assembly() -> None:
  """The plan-level units: channel resolution order and rhs assembly path."""
  driver, coordinate_map = _j2_cantilever_deck()
  system = driver.owner.system
  program = compile_sensitivity_program(
    system,
    ("hardening_slope", "youngs_modulus"),
  )
  # The observation order is the request order; the per-operator channel
  # order is the header's declaration order filtered by the request.
  assert program.parameter_ids == ("hardening_slope", "youngs_modulus")
  (slice_,) = program.operator_slices
  assert slice_.parameter_ids == ("youngs_modulus", "hardening_slope")
  assert slice_.derivative_channel_ids == (
    _DERIVATIVE_CHANNELS[0],
    _DERIVATIVE_CHANNELS[3],
  )
  with pytest.raises(TypeError, match="exact CompiledSystem"):
    compile_sensitivity_program(object(), ("youngs_modulus",))  # type: ignore[arg-type]
  with pytest.raises(TypeError, match="non-empty exact tuple"):
    compile_sensitivity_program(system, ())
  plan = driver.plan
  with pytest.raises(TypeError, match="exact DriverAssemblyPlan"):
    assemble_parameter_rhs(object(), coordinate_map, ())  # type: ignore[arg-type]
  with pytest.raises(TypeError, match="exact CompiledConstraintMap"):
    assemble_parameter_rhs(plan, object(), ())  # type: ignore[arg-type]
  # The right-hand side IS the negated residual scatter, reduced through the
  # coordinate map: no parallel assembly path.
  batches = (np.arange(4 * 16, dtype=np.float64).reshape(4, 16),)
  expected = reduce_residual(coordinate_map, -assemble_internal_force(plan, batches))
  np.testing.assert_array_equal(
    assemble_parameter_rhs(plan, coordinate_map, batches).values,
    expected.values,
  )


# --- the documented vehicle: IFT vs FD of the solve ----------------------------


def test_ift_sensitivities_match_fd_of_solve_on_the_j2_cantilever() -> None:
  """The M48 agreement class through the landed channel: <= 3.4e-10 observed.

  Single load-controlled step from the virgin state (a parameter-independent
  entering state, so the fixed-state IFT is the exact total derivative) at
  tolerance 1e-12. The FD oracle is a central difference of two full
  resolves per parameter at the same tolerance — the 2-resolves-per-
  parameter cost the sensitivity channel replaces. A coarse-stencil leg
  convicts the FD reference itself as truncation-limited (M48 measured the
  same class: 114% error at a 1% perturbation), so agreement is pinned on
  the plateau stencils.
  """
  driver, coordinate_map = _j2_cantilever_deck()
  result = _run(driver, 0.0, 20.0, sensitivity_parameters=_PARAM_NAMES)
  assert result.status is DriverStatus.COMPLETED
  (record,) = result.records
  assert record.status is SubstepStatus.COMMITTED
  # The M54 quadratic signature on this deck (pre-fix: 33 iterations).
  assert len(record.iterations) <= 10
  observation = record.observation
  assert observation is not None
  sensitivities = observation.sensitivities
  assert tuple(item.parameter_id for item in sensitivities) == _PARAM_NAMES
  # The mixed elastic/plastic premise of the M48 deck: 6 of 36 IPs plastic.
  rows = driver.owner.accepted_state(driver.owner.block_ids[0]).values
  assert int(np.count_nonzero(rows[:, 18] > 0.0)) == 6
  constrained = coordinate_map.constrained_dofs.values
  assert constrained.size > 0
  for item in sensitivities:
    coefficients = item.coefficients.values
    assert coefficients.shape == (driver.plan.full_dof_count,)
    # Prescribed offsets carry no parameter dependence: exact zeros there.
    assert np.all(coefficients[constrained] == 0.0)
  for index in range(4):
    fd = _fd_of_solve(index, (0.0, 20.0), _FD_STEPS[index])
    deviation = float(np.max(np.abs(fd - sensitivities[index].coefficients.values)))
    assert deviation < _IFT_FD_ATOL, (index, deviation)
  # M48's recorded dominant value, reproduced through the landed path.
  sy0 = sensitivities[2].coefficients.values
  assert float(sy0.min()) == pytest.approx(-1.7010667e-04, rel=1.0e-6)
  # Conviction: at 10x the stencil the FD reference is materially truncated
  # (observed 1.6e-4), so the agreement above is the IFT's exactness, not a
  # shared error.
  coarse = _fd_of_solve(2, (0.0, 20.0), 2.5)
  coarse_deviation = float(np.max(np.abs(coarse - sy0)))
  assert coarse_deviation > 1.0e-6, coarse_deviation


# --- cost accounting and primal non-perturbation -------------------------------


def test_sensitivity_cost_accounting_and_primal_non_perturbation() -> None:
  """Per committed step: +1 evaluation, +k solves/reuses/assemblies, +0 facts.

  Two committed substeps (elastic to load 12, plastifying to 20), two
  requested parameters. The FD alternative costs two full resolves per
  parameter — on this deck each resolve is the primal run's 9 evaluations
  and 8 factorizations — versus 2 evaluations and 4 back-substitutions in
  total here. The primal trajectory is untouched: counters and schedule
  structure exactly, floats to rtol 1e-12 (the deck's iteration history has
  an ambient run-to-run 1-ulp floor — two identical plain runs differ
  there — so cross-run bitwise pins are invalid; the sensitivity channel
  itself is bitwise-transparent where the deck is stable).
  """
  plain = _j2_cantilever()
  plain_result = _run(plain, 0.0, 12.0, 20.0)
  sensed = _j2_cantilever()
  sensed_result = _run(
    sensed,
    0.0,
    12.0,
    20.0,
    sensitivity_parameters=("initial_yield_stress", "youngs_modulus"),
  )
  assert plain_result.status is DriverStatus.COMPLETED
  assert sensed_result.status is DriverStatus.COMPLETED
  plain_stats = plain_result.statistics
  sensed_stats = sensed_result.statistics
  committed = plain_stats.committed_substep_count
  assert committed == 2
  assert plain_stats.cutback_count == 0
  parameter_count = 2
  assert sensed_stats.evaluation_count - plain_stats.evaluation_count == committed
  assert sensed_stats.factorization_count == plain_stats.factorization_count
  assert (
    sensed_stats.factorization_reuse_count - plain_stats.factorization_reuse_count
    == parameter_count * committed
  )
  assert (
    sensed_stats.linear_solve_count - plain_stats.linear_solve_count
    == parameter_count * committed
  )
  assert (
    sensed_stats.residual_assembly_count - plain_stats.residual_assembly_count
    == parameter_count * committed
  )
  assert sensed_stats.tangent_refill_count == plain_stats.tangent_refill_count
  assert sensed_stats.cutback_count == plain_stats.cutback_count
  # The primal trail is untouched: identical schedule structure exactly;
  # floats to rtol 1e-12 (see the module docstring for the ambient ulp
  # floor that invalidates cross-run bitwise pins on iteration history).
  assert len(sensed_result.records) == len(plain_result.records)
  for sensed_record, plain_record in zip(
    sensed_result.records, plain_result.records, strict=True
  ):
    assert sensed_record.status is plain_record.status
    assert sensed_record.committed_ordinal == plain_record.committed_ordinal
    assert len(sensed_record.iterations) == len(plain_record.iterations)
    for sensed_iteration, plain_iteration in zip(
      sensed_record.iterations, plain_record.iterations, strict=True
    ):
      assert sensed_iteration.status is plain_iteration.status
      np.testing.assert_allclose(
        sensed_iteration.residual_norm,
        plain_iteration.residual_norm,
        rtol=1.0e-12,
        atol=0.0,
      )
      if plain_iteration.increment_norm is None:
        assert sensed_iteration.increment_norm is None
      else:
        np.testing.assert_allclose(
          sensed_iteration.increment_norm,
          plain_iteration.increment_norm,
          rtol=1.0e-12,
          atol=0.0,
        )
    assert sensed_record.budget_exhaustion is None
    sensed_observation = sensed_record.observation
    plain_observation = plain_record.observation
    assert sensed_observation is not None and plain_observation is not None
    np.testing.assert_allclose(
      sensed_observation.reactions.values,
      plain_observation.reactions.values,
      rtol=1.0e-12,
      atol=1.0e-14,
    )
    np.testing.assert_allclose(
      sensed_observation.reduced_residual_norm,
      plain_observation.reduced_residual_norm,
      rtol=1.0e-12,
      atol=0.0,
    )
    assert tuple(item.parameter_id for item in sensed_observation.sensitivities) == (
      "initial_yield_stress",
      "youngs_modulus",
    )
    for item in sensed_observation.sensitivities:
      assert bool(np.isfinite(item.coefficients.values).all())
  np.testing.assert_allclose(
    sensed.owner.accepted_physical().values,
    plain.owner.accepted_physical().values,
    rtol=1.0e-12,
    atol=1.0e-15,
  )
  for block in plain.owner.block_ids:
    np.testing.assert_allclose(
      sensed.owner.accepted_state(block).values,
      plain.owner.accepted_state(block).values,
      rtol=1.0e-12,
      atol=1.0e-18,
    )
  # The first (elastic) committed substep carries an exactly zero
  # initial_yield_stress column: the elastic map does not reference it.
  first_observation = sensed_result.records[0].observation
  assert first_observation is not None
  assert np.all(first_observation.sensitivities[0].coefficients.values == 0.0)


# --- edge cases ----------------------------------------------------------------


def test_elastic_steps_are_exact_and_convict_the_fixed_state_boundary() -> None:
  """Zero plastic IPs: exact zeros for hardening, machine-exact elastic columns.

  The two-step schedule also pins the fixed-entering-state boundary by
  name: the step-2 column is the INCREMENT's sensitivity — exactly half the
  total FD-of-solve on this linear regime (the other half is the entering
  state's own parameter dependence, the declared v2 follow-up).
  """
  driver = _j2_cantilever()
  result = _run(driver, 0.0, 2.0, 4.0, sensitivity_parameters=_PARAM_NAMES)
  assert result.status is DriverStatus.COMPLETED
  assert len(result.records) == 2
  rows = driver.owner.accepted_state(driver.owner.block_ids[0]).values
  assert not np.any(rows[:, 18] > 0.0)  # the elastic premise: zero plastic IPs
  first, second = (record.observation for record in result.records)
  assert first is not None and second is not None
  for observation in (first, second):
    for item in observation.sensitivities[2:]:
      assert np.all(item.coefficients.values == 0.0), item.parameter_id
  # Step 1 from the virgin state: the fixed-state IFT is the exact total
  # derivative (observed 1.1e-16 for E, 1.4e-12 for nu at these stencils).
  for index in (0, 1):
    fd = _fd_of_solve(index, (0.0, 2.0), _FD_STEPS[index])
    deviation = float(
      np.max(np.abs(fd - first.sensitivities[index].coefficients.values))
    )
    assert deviation < _IFT_FD_ATOL, (index, deviation)
  # Step 2: the entering committed state is parameter-dependent even for
  # elastic steps (its strain and stress rows are), so the column is the
  # increment's sensitivity — total FD = 2x the column on this linear map
  # (observed ratio 2.0000000000 at the dominant dof).
  for index in (0, 1):
    fd = _fd_of_solve(index, (0.0, 2.0, 4.0), _FD_STEPS[index])
    column = second.sensitivities[index].coefficients.values
    deviation = float(np.max(np.abs(fd - 2.0 * column)))
    assert deviation < 1.0e-6 * max(1.0, float(np.max(np.abs(fd)))), (index, deviation)


def test_plateau_substep_factorizes_once_and_carries_exact_zeros() -> None:
  """The no-Newton-solve corner: one counted factorization, zero columns.

  A zero-advance (plateau) substep converges at iteration 1, so the primal
  loop never factorizes: the sensitivity path factorizes the converged
  tangent once (counted, never hidden) and its single back-substitution is
  the fresh factorization's own solve (not a reuse). The columns are
  exactly zero: the entering state is held fixed, so the evaluation sits at
  zero strain increment on the elastic trial branch, which references no
  parameter — the plateau re-solves the equilibrium the state already
  satisfies.
  """
  driver = _j2_cantilever()
  first = _run(driver, 0.0, 20.0, sensitivity_parameters=("initial_yield_stress",))
  assert first.status is DriverStatus.COMPLETED
  before = driver.statistics
  plateau = _run(driver, 20.0, 20.0, sensitivity_parameters=("initial_yield_stress",))
  assert plateau.status is DriverStatus.COMPLETED
  (record,) = plateau.records
  assert record.status is SubstepStatus.COMMITTED
  assert len(record.iterations) == 1
  observation = record.observation
  assert observation is not None
  (sensitivity,) = observation.sensitivities
  assert np.all(sensitivity.coefficients.values == 0.0)
  after = driver.statistics
  assert after.evaluation_count - before.evaluation_count == 2
  assert after.factorization_count - before.factorization_count == 1
  assert after.factorization_reuse_count - before.factorization_reuse_count == 0
  assert after.linear_solve_count - before.linear_solve_count == 1
  assert after.residual_assembly_count - before.residual_assembly_count == 2
  assert after.tangent_refill_count - before.tangent_refill_count == 1
  assert after.committed_substep_count - before.committed_substep_count == 1


def test_sensitivities_never_touch_rejected_records_or_budget_classification() -> None:
  """The near-miss deck with a sensitivity request: an identical trail.

  The M48-finding deck at a starved budget commits elastic micro-substeps
  (which carry sensitivities) between budget-exhausted attempts (which
  never do), and the records — statuses, cutback levels, budget-exhaustion
  classifications — are pairwise identical with and without the request,
  with residual norms agreeing to rtol 1e-12 (the deck's ambient
  run-to-run ulp floor; see the module docstring).
  """
  settings = NonlinearStaticSettings(
    tolerance=1.0e-12, max_iterations=4, max_cutbacks=2
  )
  plain = _j2_cantilever(settings=settings)
  plain_result = _run(plain, 0.0, 20.0)
  sensed = _j2_cantilever(settings=settings)
  sensed_result = _run(
    sensed, 0.0, 20.0, sensitivity_parameters=("initial_yield_stress",)
  )
  assert plain_result.status is DriverStatus.STEP_FAILED
  assert sensed_result.status is DriverStatus.STEP_FAILED
  assert len(sensed_result.records) == len(plain_result.records)
  committed = 0
  for sensed_record, plain_record in zip(
    sensed_result.records, plain_result.records, strict=True
  ):
    assert sensed_record.status is plain_record.status
    assert sensed_record.cutback_level == plain_record.cutback_level
    plain_exhaustion = plain_record.budget_exhaustion
    sensed_exhaustion = sensed_record.budget_exhaustion
    if plain_exhaustion is None:
      assert sensed_exhaustion is None
    else:
      assert sensed_exhaustion is not None
      assert sensed_exhaustion.trend is plain_exhaustion.trend
      assert (
        sensed_exhaustion.measured_step_count == plain_exhaustion.measured_step_count
      )
    if sensed_record.status is SubstepStatus.COMMITTED:
      committed += 1
      observation = sensed_record.observation
      assert observation is not None
      (sensitivity,) = observation.sensitivities
      assert sensitivity.parameter_id == "initial_yield_stress"
      assert bool(np.isfinite(sensitivity.coefficients.values).all())
      plain_observation = plain_record.observation
      assert plain_observation is not None
      np.testing.assert_allclose(
        observation.reduced_residual_norm,
        plain_observation.reduced_residual_norm,
        rtol=1.0e-12,
        atol=0.0,
      )
    else:
      assert sensed_record.observation is None
  assert committed >= 1  # measured: 4 elastic micro-substeps
  assert sensed_result.final_generation.ordinal == plain_result.final_generation.ordinal
  np.testing.assert_allclose(
    sensed.owner.accepted_physical().values,
    plain.owner.accepted_physical().values,
    rtol=1.0e-12,
    atol=1.0e-15,
  )


# --- M56 deviation (c): the compile-path param_jac FD probe, test-side ---------


def test_compile_path_param_jac_fd_probe_at_the_consumed_point() -> None:
  """FD-verify the compiled operator's derivative columns where the driver
  consumes them: the (entering virgin state, converged point) pair of a
  committed step. The branch margin there is the whole plastic increment —
  not the return-map tolerance — so FD is clean (M56's battery documented
  the committed-state kink for the converged-state probe point).

  The compiler's own virgin-state probe point is degenerate for FD by
  construction: at zero strain the elastic branch columns are exactly zero,
  pinned bitwise here.
  """
  driver = _j2_cantilever()
  operator = driver.owner.system.operators[0]
  entering = driver.owner.accepted_state(driver.owner.block_ids[0]).values
  result = _run(driver, 0.0, 20.0)
  assert result.status is DriverStatus.COMPLETED
  gather = operator.header.ports[0].coefficient_map.values
  converged = driver.owner.accepted_physical().values[gather]
  # The virgin compile-probe point: exactly zero columns (zero strain).
  probe = _evaluate(
    operator,
    np.zeros(gather.shape),
    entering,
    ChannelRequest((), (), _DERIVATIVE_CHANNELS),
  )
  for column in probe.derivative_values:
    assert np.all(column.values == 0.0)
  # The consumed point: FD of the assembled residual channel over perturbed
  # recompiles, at the same point and the same entering state (exactly the
  # channel's fixed-state semantics).
  base = _evaluate(
    operator, converged, entering, ChannelRequest((), (), _DERIVATIVE_CHANNELS)
  )
  columns = base.derivative_values
  assert len(columns) == 4
  worst = 0.0
  for index in range(4):
    step = 1.0e-7 * max(abs(_PARAMETERS[index]), 1.0)
    fd = np.zeros_like(columns[index].values)
    for sign in (1.0, -1.0):
      perturbed = list(_PARAMETERS)
      perturbed[index] += sign * step
      perturbed_operator = _j2_cantilever(tuple(perturbed)).owner.system.operators[0]
      evaluation = _evaluate(
        perturbed_operator,
        converged,
        entering,
        ChannelRequest(("internal-force",), ()),
      )
      fd += sign * evaluation.residual_values[0].values
    fd /= 2.0 * step
    scale = max(1.0, float(np.max(np.abs(fd))))
    worst = max(worst, float(np.max(np.abs(fd - columns[index].values))) / scale)
  # The M56 battery's assembled-column class (observed <= 2.6e-9 there),
  # now at the driver's consumption point.
  assert worst < 1.0e-8, worst


# --- M131: opt-in total path sensitivities -----------------------------------

_TOTAL_FD_STEPS = (0.21, 3.0e-6, 0.025, 0.1)
_TOTAL_SETTINGS = NonlinearStaticSettings(tolerance=1.0e-12, max_iterations=100)


def _run_total(
  driver: NonlinearStaticDriver,
  *loads: float,
  parameters: tuple[str, ...] = _PARAM_NAMES,
  sensitivity_parameters: tuple[str, ...] = (),
) -> NonlinearStaticResult:
  points = _points(*loads)
  return driver.run(
    base_point=points[0],
    target_points=points[1:],
    sensitivity_parameters=sensitivity_parameters,
    total_sensitivity_parameters=parameters,
  )


def _complete_solve_snapshot(
  parameters: tuple[float, ...],
  schedule: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray, NonlinearStaticResult]:
  driver = _j2_cantilever(parameters, settings=_TOTAL_SETTINGS)
  result = _run(driver, *schedule)
  assert result.status is DriverStatus.COMPLETED
  assert len(result.records) == len(schedule) - 1
  assert all(record.status is SubstepStatus.COMMITTED for record in result.records)
  block = driver.owner.block_ids[0]
  return (
    np.array(driver.owner.accepted_physical().values, copy=True),
    np.array(driver.owner.accepted_state(block).values, copy=True),
    result,
  )


def _scaled_total_state_error(actual: np.ndarray, expected: np.ndarray) -> float:
  stress_scale = max(1.0, float(np.max(np.abs(expected[:, :6]))))
  history_scale = max(1.0e-8, float(np.max(np.abs(expected[:, 6:]))))
  return max(
    float(np.max(np.abs(actual[:, :6] - expected[:, :6]))) / stress_scale,
    float(np.max(np.abs(actual[:, 6:] - expected[:, 6:]))) / history_scale,
  )


def test_total_elastic_factor_two_and_distinct_increment_surface() -> None:
  driver = _j2_cantilever()
  result = _run_total(
    driver,
    0.0,
    2.0,
    4.0,
    sensitivity_parameters=_PARAM_NAMES,
  )
  assert result.status is DriverStatus.COMPLETED
  first, second = (record.observation for record in result.records)
  assert first is not None and second is not None
  for p, name in enumerate(_PARAM_NAMES):
    assert first.total_sensitivities[p].parameter_id == name
    np.testing.assert_allclose(
      first.total_sensitivities[p].coefficients.values,
      first.sensitivities[p].coefficients.values,
      rtol=2.0e-12,
      atol=2.0e-15,
    )
    fd = _fd_of_solve(p, (0.0, 2.0, 4.0), _FD_STEPS[p])
    total = second.total_sensitivities[p].coefficients.values
    increment = second.sensitivities[p].coefficients.values
    assert float(np.max(np.abs(fd - total))) < 2.0e-9
    assert float(np.max(np.abs(total - 2.0 * increment))) < 2.0e-9
    block_id, state = second.total_sensitivities[p].state_derivatives[0]
    assert block_id == driver.owner.block_ids[0]
    assert state.values.shape == (36, 19)
    assert second.total_sensitivities[p].generation.ordinal == 2
    if p >= 2:
      assert np.all(total == 0.0)
  with pytest.raises(TypeError, match="exact tuple"):
    driver.run(
      base_point=_points(4.0)[0],
      target_points=_points(5.0),
      total_sensitivity_parameters=list(_PARAM_NAMES),  # type: ignore[arg-type]
    )


def test_total_plastic_load_unload_reload_matches_complete_resolve_fd() -> None:
  schedule = (0.0, 20.0, 10.0, 25.0)
  nominal = _j2_cantilever(settings=_TOTAL_SETTINGS)
  result = _run_total(nominal, *schedule)
  assert result.status is DriverStatus.COMPLETED
  assert len(result.records) == 3
  final_rows = nominal.owner.accepted_state(nominal.owner.block_ids[0]).values
  assert np.count_nonzero(final_rows[:, 18] > 0.0) == 16
  nominal_prefix = [
    _complete_solve_snapshot(_PARAMETERS, schedule[: index + 2]) for index in range(3)
  ]
  assert np.count_nonzero(nominal_prefix[0][1][:, 18] > 0.0) == 6
  assert np.count_nonzero(nominal_prefix[1][1][:, 18] > 0.0) == 6
  np.testing.assert_array_equal(
    nominal_prefix[1][1][:, 18], nominal_prefix[0][1][:, 18]
  )

  worst_u = 0.0
  coarse_state = 0.0
  fine_state = 0.0
  for target_index, record in enumerate(result.records):
    observation = record.observation
    assert observation is not None
    for p, name in enumerate(_PARAM_NAMES):
      fd_legs = []
      for step in (_TOTAL_FD_STEPS[p], 0.5 * _TOTAL_FD_STEPS[p]):
        plus = list(_PARAMETERS)
        minus = list(_PARAMETERS)
        plus[p] += step
        minus[p] -= step
        plus_u, plus_rows, _ = _complete_solve_snapshot(
          tuple(plus), schedule[: target_index + 2]
        )
        minus_u, minus_rows, _ = _complete_solve_snapshot(
          tuple(minus), schedule[: target_index + 2]
        )
        expected_mask = nominal_prefix[target_index][1][:, 18] > 0.0
        assert np.array_equal(plus_rows[:, 18] > 0.0, expected_mask)
        assert np.array_equal(minus_rows[:, 18] > 0.0, expected_mask)
        fd_legs.append(
          ((plus_u - minus_u) / (2.0 * step), (plus_rows - minus_rows) / (2.0 * step))
        )
      total = observation.total_sensitivities[p]
      state = total.state_derivatives[0][1].values
      for leg_index, (fd_u, fd_rows) in enumerate(fd_legs):
        u_error = float(np.max(np.abs(fd_u - total.coefficients.values)))
        state_error = _scaled_total_state_error(state, fd_rows)
        worst_u = max(worst_u, u_error)
        if leg_index == 0:
          coarse_state = max(coarse_state, state_error)
        else:
          fine_state = max(fine_state, state_error)
        assert u_error < 2.0e-9, (target_index, name, leg_index, u_error)
        assert state_error < 5.0e-7, (target_index, name, leg_index, state_error)
      # The independent FD oracle contracts under refinement; it is not a shared
      # analytic formula or a stored nominal output.
      assert float(np.max(np.abs(fd_legs[0][0] - fd_legs[1][0]))) < 5.0e-9
  # Observed: U <= 2.93e-10; coarse state <= 2.66e-7; fine state <= 6.65e-8.
  # The fine-state gate therefore retains more than 7x observed headroom.
  assert worst_u < 2.0e-9
  assert coarse_state < 5.0e-7
  assert fine_state < 5.0e-7


def _assert_same_primal(
  candidate: NonlinearStaticDriver,
  control: NonlinearStaticDriver,
) -> None:
  np.testing.assert_array_equal(
    candidate.owner.accepted_physical().values,
    control.owner.accepted_physical().values,
  )
  for block in control.owner.block_ids:
    np.testing.assert_array_equal(
      candidate.owner.accepted_state(block).values,
      control.owner.accepted_state(block).values,
    )


def test_total_cost_primal_transparency_and_plateau_carry_forward() -> None:
  plain = _j2_cantilever()
  plain_result = _run(plain, 0.0, 12.0, 20.0)
  one = _j2_cantilever()
  one_result = _run_total(one, 0.0, 12.0, 20.0, parameters=("youngs_modulus",))
  four = _j2_cantilever()
  four_result = _run_total(four, 0.0, 12.0, 20.0)
  for result in (one_result, four_result):
    assert result.status is DriverStatus.COMPLETED
    assert result.statistics.committed_substep_count == 2
  _assert_same_primal(one, plain)
  _assert_same_primal(four, plain)
  assert len(one_result.records) == len(plain_result.records)
  assert len(four_result.records) == len(plain_result.records)
  for candidate in (one_result, four_result):
    assert [len(record.iterations) for record in candidate.records] == [
      len(record.iterations) for record in plain_result.records
    ]
    assert (
      candidate.statistics.evaluation_count - plain_result.statistics.evaluation_count
      == 4
    )
    assert (
      candidate.statistics.factorization_count
      - plain_result.statistics.factorization_count
      == 2
    )
    assert (
      candidate.statistics.tangent_refill_count
      - plain_result.statistics.tangent_refill_count
      == 2
    )
  for result, parameter_count in ((one_result, 1), (four_result, 4)):
    assert (
      result.statistics.linear_solve_count - plain_result.statistics.linear_solve_count
      == 2 * parameter_count
    )
    assert (
      result.statistics.residual_assembly_count
      - plain_result.statistics.residual_assembly_count
      == 2 * parameter_count
    )
    assert (
      result.statistics.factorization_reuse_count
      - plain_result.statistics.factorization_reuse_count
      == 2 * (parameter_count - 1)
    )
  assert (
    four_result.statistics.factorization_count
    == one_result.statistics.factorization_count
  )

  accepted = four.owner.accepted_total_derivatives()
  before = four.statistics
  plateau = _run_total(four, 20.0, 20.0)
  after = four.statistics
  assert plateau.status is DriverStatus.COMPLETED
  assert len(plateau.records[0].iterations) == 1
  current = four.owner.accepted_total_derivatives()
  np.testing.assert_array_equal(
    current.coefficients.values, accepted.coefficients.values
  )
  for previous_rows, current_rows in zip(
    accepted.state_derivatives,
    current.state_derivatives,
    strict=True,
  ):
    np.testing.assert_array_equal(previous_rows.values, current_rows.values)
  assert after.evaluation_count - before.evaluation_count == 1
  assert after.factorization_count - before.factorization_count == 0
  assert after.linear_solve_count - before.linear_solve_count == 0
  assert after.tangent_refill_count - before.tangent_refill_count == 0

  both = _j2_cantilever()
  initial = _run_total(
    both,
    0.0,
    20.0,
    sensitivity_parameters=_PARAM_NAMES,
  )
  previous_observation = initial.records[-1].observation
  assert previous_observation is not None
  accepted = both.owner.accepted_total_derivatives()
  plateau = _run_total(
    both,
    20.0,
    20.0,
    sensitivity_parameters=_PARAM_NAMES,
  )
  observation = plateau.records[0].observation
  assert observation is not None
  for p in range(4):
    # The fixed-state RHS is numerically zero; the strain-split identity and
    # back-substitution leave only roundoff (observed <= 3.84e-19 on plastic
    # plateaus), while total U,S carry over bitwise.
    np.testing.assert_allclose(
      observation.sensitivities[p].coefficients.values,
      0.0,
      rtol=0.0,
      atol=1.0e-18,
    )
    np.testing.assert_array_equal(
      observation.total_sensitivities[p].coefficients.values,
      previous_observation.total_sensitivities[p].coefficients.values,
    )
  current = both.owner.accepted_total_derivatives()
  np.testing.assert_array_equal(
    current.coefficients.values, accepted.coefficients.values
  )


def test_total_lineage_fails_after_primal_only_commit() -> None:
  driver = _j2_cantilever()
  assert _run_total(driver, 0.0, 20.0).status is DriverStatus.COMPLETED
  assert _run(driver, 20.0, 21.0).status is DriverStatus.COMPLETED
  before = driver.statistics.evaluation_count
  with pytest.raises(DriverPreparationError) as stale:
    _run_total(driver, 21.0, 22.0)
  assert stale.value.diagnostics[0].code == "invalid-total-sensitivity-lineage"
  assert driver.statistics.evaluation_count == before

  fresh = _j2_cantilever()
  assert _run(fresh, 0.0, 20.0).status is DriverStatus.COMPLETED
  with pytest.raises(DriverPreparationError) as uninitialized:
    _run_total(fresh, 20.0, 21.0)
  assert uninitialized.value.diagnostics[0].code == "invalid-total-sensitivity-lineage"

  ordered = _j2_cantilever()
  assert _run_total(ordered, 0.0, 20.0).status is DriverStatus.COMPLETED
  with pytest.raises(DriverPreparationError) as reordered:
    _run_total(
      ordered,
      20.0,
      21.0,
      parameters=(
        "hardening_slope",
        "youngs_modulus",
        "poisson_ratio",
        "initial_yield_stress",
      ),
    )
  assert reordered.value.diagnostics[0].code == "invalid-total-sensitivity-lineage"


def test_total_cutback_rereads_accepted_derivatives(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  from pyfem.v3.compile.continuum import Q8StatefulContinuumOperator, _new
  from pyfem.v3.model.operator import EvaluationStatus

  driver = _j2_cantilever(settings=_TOTAL_SETTINGS)
  assert _run_total(driver, 0.0, 20.0).status is DriverStatus.COMPLETED
  accepted = driver.owner.accepted_total_derivatives()
  reference = _j2_cantilever(settings=_TOTAL_SETTINGS)
  reference_result = _run_total(reference, 0.0, 20.0, 22.5, 25.0)
  assert reference_result.status is DriverStatus.COMPLETED

  primal_original = Q8StatefulContinuumOperator.evaluate
  path_original = Q8StatefulContinuumOperator.evaluate_path
  calls = 0
  path_calls = 0

  def reject_first(self: object, inputs: object) -> object:
    nonlocal calls
    calls += 1
    if calls == 1:
      return _new(
        OperatorEvaluation,
        residual_values=(),
        jacobian_values=(),
        trial_state=inputs.accepted_state,
        status=EvaluationStatus.REJECT_STEP,
        derivative_values=(),
        nested_references=(),
      )
    return primal_original(self, inputs)

  def capture_path(self: object, inputs: object, seed: object) -> object:
    nonlocal path_calls
    path_calls += 1
    if path_calls == 1:
      np.testing.assert_array_equal(
        seed.accepted_state_derivatives.values,
        accepted.state_derivatives[0].values,
      )
      np.testing.assert_array_equal(
        driver.owner.accepted_total_derivatives().coefficients.values,
        accepted.coefficients.values,
      )
    return path_original(self, inputs, seed)

  monkeypatch.setattr(Q8StatefulContinuumOperator, "evaluate", reject_first)
  monkeypatch.setattr(Q8StatefulContinuumOperator, "evaluate_path", capture_path)
  result = _run_total(driver, 20.0, 25.0)
  assert result.status is DriverStatus.COMPLETED
  assert [record.status for record in result.records] == [
    SubstepStatus.REJECTED,
    SubstepStatus.COMMITTED,
    SubstepStatus.COMMITTED,
  ]
  assert result.records[0].observation is None
  _assert_same_primal(driver, reference)
  current = driver.owner.accepted_total_derivatives()
  expected = reference.owner.accepted_total_derivatives()
  np.testing.assert_allclose(
    current.coefficients.values, expected.coefficients.values, rtol=0.0, atol=1.0e-12
  )
  for actual_rows, expected_rows in zip(
    current.state_derivatives,
    expected.state_derivatives,
    strict=True,
  ):
    np.testing.assert_allclose(
      actual_rows.values, expected_rows.values, rtol=2.0e-10, atol=2.0e-12
    )


def test_total_authoring_surface_and_explicit_nested_fe2_rejection() -> None:
  from pyfem.v3.authoring.transactions import NonlinearStaticSession

  driver = _j2_cantilever()
  session = NonlinearStaticSession(driver)
  result = session.run(
    {"load": 0.0},
    {"load": 2.0},
    total_sensitivities=("youngs_modulus",),
  )
  observation = result.records[0].observation
  assert observation is not None
  assert observation.sensitivities == ()
  assert len(observation.total_sensitivities) == 1
  assert observation.total_sensitivities[0].state_derivatives[0][1].values.shape == (
    36,
    19,
  )

  from test_v3_fe2_operator import _cantilever, _load_point

  nested = _cantilever(elastic=True)
  try:
    with pytest.raises(DriverPreparationError) as rejected:
      nested.driver.run(
        base_point=_load_point(0.0),
        target_points=(_load_point(1.0),),
        total_sensitivity_parameters=("youngs_modulus",),
      )
    assert (
      rejected.value.diagnostics[0].code == "unsupported-total-sensitivity-nested-fe2"
    )
    assert nested.driver.statistics.evaluation_count == 0
  finally:
    nested.driver.close()
