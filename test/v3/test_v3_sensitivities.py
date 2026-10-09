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
  resolves per parameter instead) — and the primal trajectory is bitwise
  untouched by the sensitivity channel;
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
parameter-independent entering state (virgin, or any elastic trajectory,
where the entering plastic strain is identically zero) — and for later
steps of a plastified trajectory it is the increment's sensitivity: the
entering state's own parameter dependence needs state-derivative channels,
the M48 survey's declared v2 boundary. The two-step elastic test pins the
boundary by name (total FD = 2x the step-2 column for the linear map).
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
  evaluation_derivative_values,
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
  total here. The primal trajectory is bitwise identical with and without
  the request.
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
  # The primal trail is untouched: identical records, identical committed
  # state (physical coefficients and operator state rows), bitwise.
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
      assert sensed_iteration.residual_norm == plain_iteration.residual_norm
      assert sensed_iteration.increment_norm == plain_iteration.increment_norm
    assert sensed_record.budget_exhaustion is None
    sensed_observation = sensed_record.observation
    plain_observation = plain_record.observation
    assert sensed_observation is not None and plain_observation is not None
    np.testing.assert_array_equal(
      sensed_observation.reactions.values, plain_observation.reactions.values
    )
    assert (
      sensed_observation.reduced_residual_norm
      == plain_observation.reduced_residual_norm
    )
    assert tuple(item.parameter_id for item in sensed_observation.sensitivities) == (
      "initial_yield_stress",
      "youngs_modulus",
    )
    for item in sensed_observation.sensitivities:
      assert bool(np.isfinite(item.coefficients.values).all())
  np.testing.assert_array_equal(
    sensed.owner.accepted_physical().values,
    plain.owner.accepted_physical().values,
  )
  for block in plain.owner.block_ids:
    assert sensed.owner.encode_state(block) == plain.owner.encode_state(block)
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
  never do), and the records — statuses, cutback levels, residual norms,
  budget-exhaustion classifications — are pairwise identical with and
  without the request.
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
      assert (
        observation.reduced_residual_norm == plain_observation.reduced_residual_norm
      )
    else:
      assert sensed_record.observation is None
  assert committed >= 1  # measured: 4 elastic micro-substeps
  assert sensed_result.final_generation.ordinal == plain_result.final_generation.ordinal
  np.testing.assert_array_equal(
    sensed.owner.accepted_physical().values, plain.owner.accepted_physical().values
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
  for column in evaluation_derivative_values(probe):
    assert np.all(column.values == 0.0)
  # The consumed point: FD of the assembled residual channel over perturbed
  # recompiles, at the same point and the same entering state (exactly the
  # channel's fixed-state semantics).
  base = _evaluate(
    operator, converged, entering, ChannelRequest((), (), _DERIVATIVE_CHANNELS)
  )
  columns = evaluation_derivative_values(base)
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
