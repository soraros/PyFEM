# SPDX-License-Identifier: MIT

"""Sensitivity authoring (M58): qualified parameter names on ``session.run``.

The S1C authoring surface of the M48 blueprint: ``session.run(...,
sensitivities=(...))`` translates plain qualified parameter names (the
material law's declared ``parameter_names``, the M46/M49/M53 metadata
convention) onto the M57 driver IFT channel and returns the landed records
unchanged, committed observations carrying the typed
``ParameterSensitivityObservation`` columns in request order. This battery
pins:

- request shape: plain sequences of non-empty exact strings, lists included;
  malformed requests are ``TypeError``;
- the M53 diagnostics idiom on names no operator differentiates: the landed
  ``unknown-sensitivity-parameter`` code with a field-level diff naming each
  offending entry — ``'constant'`` for a qualified-convention parameter baked
  into the compiled calibration (the parameterized-vs-constant width/type
  diff, pinned literally), ``'unknown'`` for a name no operator declares —
  fired before any substep runs; duplicate names keep the landed driver's own
  diagnostic unchanged;
- the persona: a student authors the documented M48/M57 J2 cantilever deck
  through the helpers, steps the ``nonlinear_static`` session on the load
  ramp, and requests d(committed)/d(initial_yield_stress) plus
  d(committed)/d(youngs_modulus); the columns equal the landed driver oracle
  BITWISE on the committed path, the committed state rows equal the stepped
  M25 kernel oracle bitwise, and M48's recorded dominant value reproduces;
- the elastic vehicle (no plastic IPs): exact zeros for the
  hardening-parameter columns on every elastic step, and the
  fixed-entering-state boundary pinned by name — the step-2 column is the
  INCREMENT's sensitivity: it composes with the step-1 column (the entering
  state's own parameter dependence) into the exact single-step total, so
  reading it as the total halves the answer;
- the cost contract through the surface: default off is zero cost (identical
  statistics), and a sensed run adds one evaluation sweep plus one
  back-substitution and right-hand-side assembly per parameter per committed
  substep, with zero extra factorizations.
"""

from __future__ import annotations

import sys
from dataclasses import astuple

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from pyfem.v3 import authoring
from pyfem.v3.compile.continuum import plasticity_reference_registry
from pyfem.v3.compile.contracts import StatefulContinuumKernelResult
from pyfem.v3.compile.system import compile_system
from pyfem.v3.constraints import CompiledConstraintMap, compile_constraint_map
from pyfem.v3.driver import (
  DriverPreparationError,
  DriverStatus,
  NonlinearStaticDriver,
  NonlinearStaticResult,
  NonlinearStaticSettings,
  ParameterSensitivityObservation,
  SubstepStatus,
)
from pyfem.v3.materials.isotropic_hardening_plasticity import (
  isotropic_hardening_calibration,
  isotropic_hardening_initial_state,
  isotropic_hardening_plasticity_kernel,
  isotropic_hardening_plasticity_metadata,
)
from pyfem.v3.model.operator import OperatorStateLayout
from pyfem.v3.model.system import CompiledSystem
from pyfem.v3.spec import ProgramCoordinateSpec, SourceContext
from pyfem.v3.spec.program import ProgramCoordinateValue, ProgramPoint

# The documented M48/M57 cantilever deck (test_v3_sensitivities.py's vehicle).
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
_SETTINGS = NonlinearStaticSettings(tolerance=1.0e-12)


def _cantilever(
  parameters: tuple[float, ...] = _PARAMETERS,
  *,
  nx: int = 4,
  registry: dict | None = None,
) -> authoring.NonlinearStaticSession:
  """The documented deck through the authoring helpers only."""
  mesh = authoring.quad8_patch(nx, 1, width=4.0, height=1.0)
  model = authoring.small_strain_continuum(
    mesh, material=authoring.plasticity(*parameters, id="steel")
  )
  system = (
    authoring.compile(model) if registry is None else authoring.compile(model, registry)
  )
  left = tuple(node.id for node in mesh.nodes if node.coordinates[0] == 0.0)
  right = tuple(
    sorted(
      (node for node in mesh.nodes if node.coordinates[0] == 4.0),
      key=lambda node: node.coordinates[1],
    )
  )
  return authoring.nonlinear_static(
    system,
    constraints=authoring.fixed(left, ("x", "y")),
    loads=tuple(
      authoring.nodal_load(node.id, "y", fraction)
      for node, fraction in zip(right, (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0), strict=True)
    ),
    settings=_SETTINGS,
  )


def _oracle_deck(
  parameters: tuple[float, ...] = _PARAMETERS,
) -> tuple[NonlinearStaticDriver, CompiledConstraintMap]:
  """The same deck through the landed API directly (the M57 battery vehicle)."""
  mesh = authoring.quad8_patch(4, 1, width=4.0, height=1.0)
  model = authoring.small_strain_continuum(
    mesh, material=authoring.plasticity(*parameters, id="steel")
  )
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
  return NonlinearStaticDriver(system, coordinate_map, loads, _SETTINGS), coordinate_map


def _oracle_run(
  oracle: NonlinearStaticDriver,
  *loads: float,
  sensitivity_parameters: tuple[str, ...] = (),
) -> NonlinearStaticResult:
  points = tuple(
    ProgramPoint((ProgramCoordinateValue("load", value),)) for value in loads
  )
  return oracle.run(
    base_point=points[0],
    target_points=points[1:],
    sensitivity_parameters=sensitivity_parameters,
  )


def _assert_columns_bitwise(
  result: NonlinearStaticResult,
  oracle_result: NonlinearStaticResult,
) -> None:
  """Pairwise bitwise equality of every committed record's typed columns."""
  assert len(result.records) == len(oracle_result.records)
  for record, oracle_record in zip(result.records, oracle_result.records, strict=True):
    assert record.status is SubstepStatus.COMMITTED
    assert oracle_record.status is SubstepStatus.COMMITTED
    observation = record.observation
    oracle_observation = oracle_record.observation
    assert observation is not None and oracle_observation is not None
    assert len(observation.sensitivities) == len(oracle_observation.sensitivities)
    for column, oracle_column in zip(
      observation.sensitivities, oracle_observation.sensitivities, strict=True
    ):
      assert column.parameter_id == oracle_column.parameter_id
      np.testing.assert_array_equal(
        column.coefficients.values, oracle_column.coefficients.values
      )


def _committed_ip_strains(system: CompiledSystem, values: np.ndarray) -> np.ndarray:
  """Recompute the committed per-integration-point 6-Voigt strains.

  The M25 parity harness expression (test_v3_authoring_plasticity.py): the
  operator's own physical strain-displacement map applied to the committed
  coefficient vector.
  """
  operator = system.operators[0]
  payload = operator.payload
  b_matrix = (
    payload.normalized_strain_displacement.values
    / payload.geometry_scales.values[:, None, None, None]
  )
  gather = operator.header.ports[0].coefficient_map.values
  strain3 = np.einsum("epai,ei->epa", b_matrix, values[gather], optimize=True)
  strains = np.zeros((np.prod(strain3.shape[:2]), 6), dtype=np.float64)
  flat = strain3.reshape(-1, 3)
  strains[:, 0] = flat[:, 0]
  strains[:, 1] = flat[:, 1]
  strains[:, 5] = flat[:, 2]
  return strains


class _StudentJ2Binding:
  """The student's own J2 law: the v2 stateful binding protocol in plain methods."""

  def __call__(self, *parameters: float) -> np.ndarray:
    return isotropic_hardening_calibration(*parameters)

  def descriptor_metadata(self) -> dict[str, object]:
    return isotropic_hardening_plasticity_metadata()

  def kernel(
    self,
    strains: np.ndarray,
    accepted_rows: np.ndarray,
    calibration: np.ndarray,
  ) -> StatefulContinuumKernelResult:
    return isotropic_hardening_plasticity_kernel(strains, accepted_rows, calibration)

  def initial_state(
    self,
    parameters: tuple[float, ...],
    layout: OperatorStateLayout,
  ) -> np.ndarray:
    return isotropic_hardening_initial_state(parameters, layout)


# --- request shape ---------------------------------------------------------------


def test_sensitivities_request_shape_is_validated() -> None:
  session = _cantilever(nx=1)
  with pytest.raises(TypeError, match="sequence of qualified parameter names"):
    session.run({"load": 0.0}, {"load": 1.0}, sensitivities="initial_yield_stress")  # type: ignore[arg-type]
  with pytest.raises(TypeError, match="sequence of qualified parameter names"):
    session.run({"load": 0.0}, {"load": 1.0}, sensitivities=5)  # type: ignore[arg-type]
  with pytest.raises(TypeError, match="non-empty exact strings"):
    session.run({"load": 0.0}, {"load": 1.0}, sensitivities=("initial_yield_stress", 5))  # type: ignore[arg-type]
  with pytest.raises(TypeError, match="non-empty exact strings"):
    session.run({"load": 0.0}, {"load": 1.0}, sensitivities=("",))
  assert session.driver.statistics.evaluation_count == 0
  # A plain list is a sequence and translates; observation order is request
  # order.
  result = session.run(
    {"load": 0.0},
    {"load": 2.0},
    sensitivities=["youngs_modulus", "initial_yield_stress"],
  )
  assert result.status is DriverStatus.COMPLETED
  (record,) = result.records
  observation = record.observation
  assert observation is not None
  assert tuple(column.parameter_id for column in observation.sensitivities) == (
    "youngs_modulus",
    "initial_yield_stress",
  )


# --- the M53 diagnostics idiom on the request -------------------------------------


def test_unknown_sensitivity_names_fail_with_field_diffs() -> None:
  """Unknown names: the landed code plus a field-level diff, pre-substep.

  The mixed request fails closed on the offending entry only — the valid
  name produces no diff line — and the trailer lists the declared
  differentiable surface so the student sees the qualified names.
  """
  session = _cantilever(nx=1)
  with pytest.raises(DriverPreparationError) as captured:
    session.run(
      {"load": 0.0},
      {"load": 1.0},
      sensitivities=("initial_yield_stress", "sig0"),
    )
  (diagnostic,) = captured.value.diagnostics
  assert diagnostic.code == "unknown-sensitivity-parameter"
  assert (
    "field 'sensitivities.sig0': expected {'parameter': 'sig0'}, authored "
    "'unknown'" in diagnostic.message
  )
  assert "field 'sensitivities.initial_yield_stress'" not in diagnostic.message
  assert (
    "declared differentiable parameters: ('youngs_modulus', 'poisson_ratio', "
    "'initial_yield_stress', 'hardening_slope'); declared constant parameters: ()"
    in diagnostic.message
  )
  assert diagnostic.source == SourceContext(source="authoring.run:sensitivities")
  # Request-time means before any substep: no evaluation ran and the owner
  # never left the initial generation.
  assert session.driver.statistics.evaluation_count == 0
  assert session.ordinal == 0


def test_constant_parameter_requests_name_the_parameterized_convention() -> None:
  """Constants: the parameterized-vs-constant width/type diff, pinned literally.

  The elastic vehicle declares ``youngs_modulus``/``poisson_ratio`` as
  qualified parameters but compiles them into the calibration as constants
  (the reference elastic binding carries no derivative kernel), so the
  request fails pre-substep with the field-level diff contrasting the
  parameterized declaration the name would need against the constant it is.
  The same holds for a student J2 binding without the derivative twin (the
  M31 teaching-law pattern): the qualified convention names the parameters;
  the binding decides they are constant.
  """
  mesh = authoring.quad8_patch()
  model = authoring.small_strain_continuum(
    mesh, material=authoring.linear_elastic(1.0e6, 0.25, id="lin")
  )
  session = authoring.nonlinear_static(
    authoring.compile(model),
    constraints=authoring.fixed(
      tuple(node.id for node in mesh.nodes if node.coordinates[0] == 0.0),
      ("x", "y"),
    ),
  )
  with pytest.raises(DriverPreparationError) as captured:
    session.run({"load": 0.0}, {"load": 1.0}, sensitivities=("youngs_modulus",))
  (diagnostic,) = captured.value.diagnostics
  assert diagnostic.code == "unknown-sensitivity-parameter"
  assert (
    "field 'sensitivities.youngs_modulus': expected {'parameter': "
    "'youngs_modulus'}, authored 'constant'" in diagnostic.message
  )
  assert (
    "declared differentiable parameters: (); declared constant parameters: "
    "('youngs_modulus', 'poisson_ratio')" in diagnostic.message
  )
  assert diagnostic.source == SourceContext(source="authoring.run:sensitivities")
  assert session.driver.statistics.evaluation_count == 0
  assert session.ordinal == 0
  # The M31 teaching-law pattern: a primal-only student binding behind the
  # qualified plasticity convention declares the same names as constants.
  law = authoring.plasticity_law(
    _StudentJ2Binding(),
    implementation_id="student-j2-v1",
  )
  student = _cantilever(nx=1, registry=authoring.plasticity_registry(material=law))
  with pytest.raises(DriverPreparationError) as student_captured:
    student.run({"load": 0.0}, {"load": 1.0}, sensitivities=("initial_yield_stress",))
  (student_diagnostic,) = student_captured.value.diagnostics
  assert student_diagnostic.code == "unknown-sensitivity-parameter"
  assert (
    "field 'sensitivities.initial_yield_stress': expected {'parameter': "
    "'initial_yield_stress'}, authored 'constant'" in student_diagnostic.message
  )
  assert student.driver.statistics.evaluation_count == 0


def test_duplicate_names_keep_the_landed_diagnostic() -> None:
  """A repeated name is the landed driver's own diagnostic, passed through."""
  session = _cantilever(nx=1)
  with pytest.raises(DriverPreparationError) as captured:
    session.run(
      {"load": 0.0},
      {"load": 1.0},
      sensitivities=("initial_yield_stress", "initial_yield_stress"),
    )
  (diagnostic,) = captured.value.diagnostics
  assert diagnostic.code == "duplicate-sensitivity-parameter"
  assert "is requested twice" in diagnostic.message
  assert "does not match the declared parameter surface" not in diagnostic.message
  assert session.driver.statistics.evaluation_count == 0


# --- the persona: authored, stepped, oracle-checked --------------------------------


def test_persona_steps_the_documented_deck_and_matches_the_oracles_bitwise() -> None:
  """Persona: plasticity() plus session.run sensitivities, oracle-checked.

  The student authors the documented M48/M57 deck end-to-end through the
  helpers and requests d(committed)/d(initial_yield_stress) plus
  d(committed)/d(youngs_modulus) on the single load-controlled step from the
  virgin state (a parameter-independent entering state, so the fixed-state
  IFT columns are the exact total derivatives). The typed columns equal the
  landed driver oracle BITWISE on the committed path; the committed state
  rows equal the stepped M25 kernel oracle bitwise.
  """
  session = _cantilever()
  sensed = session.run(
    {"load": 0.0},
    {"load": 20.0},
    sensitivities=("initial_yield_stress", "youngs_modulus"),
  )
  assert sensed.status is DriverStatus.COMPLETED
  (record,) = sensed.records
  assert record.status is SubstepStatus.COMMITTED
  observation = record.observation
  assert observation is not None
  columns = observation.sensitivities
  assert tuple(column.parameter_id for column in columns) == (
    "initial_yield_stress",
    "youngs_modulus",
  )
  # The typed observations are the landed M57 contract, re-exported.
  assert authoring.ParameterSensitivityObservation is ParameterSensitivityObservation
  for column in columns:
    assert type(column) is ParameterSensitivityObservation
    assert column.coefficients.values.shape == (session.driver.plan.full_dof_count,)
  # The mixed elastic/plastic premise of the M48 deck: 6 of 36 IPs plastic.
  rows = session.accepted_state("cells")
  assert rows.shape == (36, 19)
  assert int(np.count_nonzero(rows[:, 18] > 0.0)) == 6
  # The authoring compile is the landed system (the M31 byte-identity pin),
  # so the driver oracle measures the same deck.
  oracle, coordinate_map = _oracle_deck()
  assert session.system.content_fingerprint == oracle.owner.system.content_fingerprint
  # Prescribed offsets carry no parameter dependence: exact zeros there.
  constrained = coordinate_map.constrained_dofs.values
  assert constrained.size > 0
  for column in columns:
    assert np.all(column.coefficients.values[constrained] == 0.0)
  # BITWISE against the M57 driver oracle on the committed path.
  oracle_result = _oracle_run(
    oracle,
    0.0,
    20.0,
    sensitivity_parameters=("initial_yield_stress", "youngs_modulus"),
  )
  _assert_columns_bitwise(sensed, oracle_result)
  np.testing.assert_array_equal(
    session.accepted_coefficients(),
    oracle.owner.accepted_physical().values,
  )
  # The committed state rows equal the stepped M25 kernel oracle bitwise:
  # the driver stages the trial rows of the converged iterate.
  kernel_oracle = isotropic_hardening_plasticity_kernel(
    _committed_ip_strains(session.system, session.accepted_coefficients()),
    np.zeros(rows.shape, dtype=np.float64),
    isotropic_hardening_calibration(*_PARAMETERS),
  )
  assert kernel_oracle.status is authoring.EvaluationStatus.OK
  np.testing.assert_array_equal(rows, kernel_oracle.trial_rows)
  # M48's recorded dominant value, reproduced through the authoring surface.
  sy0 = columns[0].coefficients.values
  assert float(sy0.min()) == pytest.approx(-1.7010667e-04, rel=1.0e-6)


# --- the elastic vehicle and the fixed-entering-state boundary ---------------------


def test_elastic_regime_columns_are_exact_and_compose_like_increments() -> None:
  """Zero plastic IPs: exact zeros for hardening; the boundary pinned by name.

  The two-step schedule 0 -> 2 -> 4 stays elastic, so every
  hardening-parameter column is exactly zero (the elastic map does not
  reference them) and the elastic columns equal the driver oracle bitwise on
  both committed records. The step-2 elastic column is the INCREMENT's
  sensitivity: composed with the step-1 column (the entering state's own
  parameter dependence) it reproduces the exact single-step total at load 4 —
  reading it as the total halves the answer. Multi-step path propagation
  needs the declared follow-up state-derivative channels (the M48 v2
  boundary).
  """
  session = _cantilever()
  result = session.run(
    {"load": 0.0}, {"load": 2.0}, {"load": 4.0}, sensitivities=_PARAM_NAMES
  )
  assert result.status is DriverStatus.COMPLETED
  assert len(result.records) == 2
  rows = session.accepted_state("cells")
  assert not np.any(rows[:, 18] > 0.0)  # the elastic premise: zero plastic IPs
  first, second = (record.observation for record in result.records)
  assert first is not None and second is not None
  for observation in (first, second):
    for column in observation.sensitivities[2:]:
      assert np.all(column.coefficients.values == 0.0), column.parameter_id
  oracle, _ = _oracle_deck()
  oracle_result = _oracle_run(
    oracle, 0.0, 2.0, 4.0, sensitivity_parameters=_PARAM_NAMES
  )
  _assert_columns_bitwise(result, oracle_result)
  # The boundary by name: the single-step run 0 -> 4 measures the exact total
  # (virgin entering state); the committed columns compose into it linearly
  # (the map is linear here), so step 2's column is exactly the increment's
  # share — never the total itself.
  total_run = _cantilever().run(
    {"load": 0.0}, {"load": 4.0}, sensitivities=("youngs_modulus",)
  )
  (total_record,) = total_run.records
  total_observation = total_record.observation
  assert total_observation is not None
  total = total_observation.sensitivities[0].coefficients.values
  entering = first.sensitivities[0].coefficients.values
  increment = second.sensitivities[0].coefficients.values
  # Observed max abs deviation of the composition: 2.6e-21 on this deck
  # (1.2e-13 relative to the 2.1e-8 column scale; tolerance carries ~1e-5
  # relative headroom over that).
  np.testing.assert_allclose(total, entering + increment, rtol=1.0e-9, atol=1.0e-14)
  # Conviction: reading the step-2 column as the total sensitivity is a ~50%
  # error (measured 0.5000000000 at the dominant dof of this linear regime).
  gap = float(np.max(np.abs(total - increment))) / float(np.max(np.abs(total)))
  assert gap > 0.1, gap


# --- default off = zero cost; the M57 cost contract through the surface -----------


def test_default_run_is_primal_only_and_the_cost_contract_holds() -> None:
  """No request: identical statistics; sensed: +1 sweep, +k solves, +0 facts.

  Two committed substeps (elastic to load 12, plastifying to 20) and two
  requested parameters: the sensed run adds exactly one evaluation sweep per
  committed substep and one back-substitution plus one right-hand-side
  assembly per parameter per committed substep, with ZERO extra
  factorizations — the M57 cost contract measured through the authoring
  surface, columns bitwise equal to the driver oracle on both records.
  """
  plain = _cantilever()
  plain_result = plain.run({"load": 0.0}, {"load": 12.0}, {"load": 20.0})
  defaulted = _cantilever()
  defaulted_result = defaulted.run(
    {"load": 0.0}, {"load": 12.0}, {"load": 20.0}, sensitivities=()
  )
  assert plain_result.status is DriverStatus.COMPLETED
  assert defaulted_result.status is DriverStatus.COMPLETED
  # DriverStatistics is eq=False by contract; the counter tuple is the
  # comparison (identical integers: the empty request is exactly free).
  assert astuple(defaulted_result.statistics) == astuple(plain_result.statistics)
  for record in defaulted_result.records:
    observation = record.observation
    assert observation is not None
    assert observation.sensitivities == ()
  sensed = _cantilever()
  sensed_result = sensed.run(
    {"load": 0.0},
    {"load": 12.0},
    {"load": 20.0},
    sensitivities=("initial_yield_stress", "youngs_modulus"),
  )
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
  oracle, _ = _oracle_deck()
  oracle_result = _oracle_run(
    oracle,
    0.0,
    12.0,
    20.0,
    sensitivity_parameters=("initial_yield_stress", "youngs_modulus"),
  )
  _assert_columns_bitwise(sensed_result, oracle_result)
