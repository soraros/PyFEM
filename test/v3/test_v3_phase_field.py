# SPDX-License-Identifier: MIT

"""Fresh Q4 AT2 spectral-split response, history, and driver acceptance."""

from __future__ import annotations

import sys

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from pyfem.v3.compile.diagnostics import ModelCompilationError
from pyfem.v3.compile.phase_field import (
  PHASE_FIELD_STATE_SCHEMA,
  Q4AT2PhaseFieldOperator,
  phase_field_reference_registry,
)
from pyfem.v3.compile.system import compile_system
from pyfem.v3.constraints import CompiledConstraintMap, compile_constraint_map
from pyfem.v3.driver import (
  DriverStatus,
  NonlinearStaticDriver,
  NonlinearStaticSettings,
  SubstepStatus,
)
from pyfem.v3.fem.phase_field import (
  PhaseFieldPreparationError,
  frozen_history_phase_potential,
  frozen_history_phase_solution,
  spectral_at2_response,
)
from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.operator import (
  ChannelRequest,
  EvaluationStatus,
  OperatorEvaluation,
  OperatorEvaluationInput,
)
from pyfem.v3.model.system import CompiledSystem
from pyfem.v3.spec import (
  CellBlockSpec,
  CellRef,
  CellSpec,
  FieldSpec,
  MaterialParameterSpec,
  MaterialSpec,
  MeshSpec,
  ModelSpec,
  NodeSpec,
  RegionSpec,
  SourceContext,
)
from pyfem.v3.spec.program import (
  AffineCoefficientSpec,
  AffineValueSpec,
  DofRef,
  PrescribedDofSpec,
  ProgramCoordinateSpec,
  ProgramCoordinateValue,
  ProgramPoint,
)
from pyfem.v3.state.codecs import StateCodecError

_E = 2000.0
_NU = 0.25
_LAMBDA = 800.0
_MU = 800.0
_GC = 1.5
_ELL = 0.2
_ETA = 1.0e-6
_STRAIN = np.array([0.004, -0.001, 0.0024], dtype=np.float64)
_H = 0.018207403819453783
_PHASE = np.array([0.08, 0.11, 0.14, 0.09], dtype=np.float64)


def _source(label: str) -> SourceContext:
  return SourceContext(source=label)


def _model(
  *,
  scale: float = 1.0,
  translation: tuple[float, float] = (0.0, 0.0),
  initial_crack: bool = False,
) -> ModelSpec:
  source = _source("q4-at2")
  base = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
  coordinates = scale * base + np.asarray(translation, dtype=np.float64)
  nodes = tuple(
    NodeSpec(
      index + 1,
      tuple(float(value) for value in coordinates[index]),
      _source(f"n{index + 1}"),
    )
    for index in range(4)
  )
  cell = CellSpec("quadrilateral", (1, 2, 3, 4), _source("cell"))
  displacement = FieldSpec("displacement", ("x", "y"), "node", _source("displacement"))
  phase = FieldSpec("phase", ("d",), "node", _source("phase"))
  parameters = [
    MaterialParameterSpec("youngs_modulus", _E),
    MaterialParameterSpec("poisson_ratio", _NU),
    MaterialParameterSpec("critical_energy_release_rate", _GC),
    MaterialParameterSpec("length_scale", _ELL),
    MaterialParameterSpec("residual_stiffness", _ETA),
  ]
  if initial_crack:
    parameters.append(MaterialParameterSpec("initial_crack", 1.0))
  return ModelSpec(
    MeshSpec(
      nodes,
      (
        CellBlockSpec(
          "cells",
          "quadrilateral",
          2,
          2,
          "bilinear-quad4",
          (cell,),
          _source("block"),
        ),
      ),
      source,
    ),
    (displacement, phase),
    (
      MaterialSpec(
        "material",
        "plane-strain-at2-spectral-split",
        tuple(parameters),
        _source("material"),
      ),
    ),
    (
      RegionSpec(
        "domain",
        (CellRef("cells", "quadrilateral"),),
        (displacement.id, phase.id),
        "material",
        "at2-spectral-split-quad4",
        "gauss-2x2",
        _source("region"),
      ),
    ),
    source,
  )


def _system(
  *,
  scale: float = 1.0,
  translation: tuple[float, float] = (0.0, 0.0),
) -> CompiledSystem:
  return compile_system(
    _model(scale=scale, translation=translation),
    phase_field_reference_registry(),
  )


def _affine_displacement(
  coordinates: np.ndarray,
  *,
  scale: float = 1.0,
  origin: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
  relative = coordinates - np.asarray(origin, dtype=np.float64)
  values = np.empty((coordinates.shape[0], 2), dtype=np.float64)
  values[:, 0] = scale * (
    _STRAIN[0] * relative[:, 0] + 0.5 * _STRAIN[2] * relative[:, 1]
  )
  values[:, 1] = scale * (
    0.5 * _STRAIN[2] * relative[:, 0] + _STRAIN[1] * relative[:, 1]
  )
  return values.reshape(-1)


def _unit_displacement(*, scale: float = 1.0) -> np.ndarray:
  coordinates = np.array(
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64
  )
  return _affine_displacement(coordinates, scale=scale)


def _evaluate(
  operator: Q4AT2PhaseFieldOperator,
  displacement: np.ndarray,
  phase: np.ndarray,
  accepted: np.ndarray,
  *,
  jacobian: bool = True,
) -> OperatorEvaluation:
  return operator.evaluate(
    OperatorEvaluationInput(
      (
        FinalizedArray(displacement.reshape(1, 8), dtype=np.float64),
        FinalizedArray(phase.reshape(1, 4), dtype=np.float64),
      ),
      FinalizedArray(np.array(accepted, dtype=np.float64, copy=True), dtype=np.float64),
      (),
      ChannelRequest(
        tuple(channel.channel_id for channel in operator.header.residual_channels),
        (
          tuple(channel.channel_id for channel in operator.header.jacobian_channels)
          if jacobian
          else ()
        ),
      ),
    )
  )


def _residual(evaluation: OperatorEvaluation) -> np.ndarray:
  return np.concatenate(tuple(value.values[0] for value in evaluation.residual_values))


def _dense(evaluation: OperatorEvaluation) -> np.ndarray:
  dense = np.zeros((12, 12), dtype=np.float64)
  dense[:8, :8] = evaluation.jacobian_values[0].values[0]
  dense[:8, 8:] = evaluation.jacobian_values[1].values[0]
  dense[8:, :8] = evaluation.jacobian_values[2].values[0]
  dense[8:, 8:] = evaluation.jacobian_values[3].values[0]
  return dense


def test_m134_spectral_oracle_energy_equality_and_scalar_phase_minimum() -> None:
  response = spectral_at2_response(_STRAIN[None, :], _LAMBDA, _MU)
  np.testing.assert_allclose(
    np.sort(response.eigenvalues[0]),
    0.0015 + np.array([-1.0, 1.0]) * np.sqrt(7.69e-6),
    rtol=0.0,
    atol=2.0e-15,
  )
  assert response.positive_energy[0] == pytest.approx(_H, abs=2.0e-17)
  assert response.negative_energy[0] == pytest.approx(
    0.0012965961805462163, abs=2.0e-17
  )
  full_energy = 0.5 * _LAMBDA * (_STRAIN[0] + _STRAIN[1]) ** 2 + _MU * (
    _STRAIN[0] ** 2 + _STRAIN[1] ** 2 + 0.5 * _STRAIN[2] ** 2
  )
  assert full_energy == pytest.approx(0.019504, abs=2.0e-17)
  assert response.positive_energy[0] + response.negative_energy[0] == pytest.approx(
    full_energy, abs=2.0e-17
  )
  phase_star = frozen_history_phase_solution(_H, _GC, _ELL)
  assert phase_star == pytest.approx(0.004831847578506099, abs=2.0e-17)
  minimum = frozen_history_phase_potential(phase_star, _H, _GC, _ELL)
  for phase in (0.0, 1.0, phase_star - 1.0e-3, phase_star + 1.0e-3):
    assert minimum <= frozen_history_phase_potential(phase, _H, _GC, _ELL)

  operator = _system().operators[0]
  evaluation = _evaluate(
    operator,
    _unit_displacement(),
    np.full(4, phase_star),
    np.zeros((4, 1)),
  )
  np.testing.assert_allclose(
    evaluation.residual_values[1].values.sum(),
    -2.0 * (1.0 - phase_star) * _H + _GC * phase_star / _ELL,
    rtol=0.0,
    atol=2.0e-17,
  )
  assert evaluation.residual_values[1].values.sum() == pytest.approx(0.0, abs=2.0e-17)


@pytest.mark.parametrize(
  ("case", "accepted_value", "active"),
  (
    ("mixed-active", 0.0, True),
    ("mixed-inactive", 0.03, False),
    ("hydrostatic-compression", 0.1, False),
  ),
)
def test_every_residual_jacobian_coefficient_matches_central_fd(
  case: str,
  accepted_value: float,
  active: bool,
) -> None:
  operator = _system().operators[0]
  displacement = _unit_displacement()
  if case == "hydrostatic-compression":
    coordinates = np.array(
      [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64
    )
    displacement = (-0.002 * coordinates).reshape(-1)
  accepted = np.full((4, 1), accepted_value, dtype=np.float64)
  evaluation = _evaluate(operator, displacement, _PHASE, accepted)
  expected = _dense(evaluation)
  step = 1.0e-6
  for index in range(12):
    plus_u = displacement.copy()
    minus_u = displacement.copy()
    plus_d = _PHASE.copy()
    minus_d = _PHASE.copy()
    if index < 8:
      plus_u[index] += step
      minus_u[index] -= step
    else:
      plus_d[index - 8] += step
      minus_d[index - 8] -= step
    plus = _evaluate(operator, plus_u, plus_d, accepted, jacobian=False)
    minus = _evaluate(operator, minus_u, minus_d, accepted, jacobian=False)
    difference = (_residual(plus) - _residual(minus)) / (2.0 * step)
    np.testing.assert_allclose(
      difference,
      expected[:, index],
      rtol=3.0e-6,
      atol=3.0e-7,
      err_msg=f"{case} coefficient {index}",
    )
  if active:
    np.testing.assert_allclose(
      expected[8:, :8], expected[:8, 8:].T, rtol=2.0e-13, atol=2.0e-13
    )
  else:
    np.testing.assert_array_equal(expected[8:, :8], np.zeros((4, 8)))
    if case == "mixed-inactive":
      assert np.max(np.abs(expected[:8, 8:])) > 1.0e-3
      # The blind-transposition control fails: K_ud is not K_du on this branch.
      assert np.max(np.abs(expected[8:, :8] - expected[:8, 8:].T)) > 1.0e-3
    else:
      np.testing.assert_array_equal(expected[:8, 8:], np.zeros((8, 4)))


def test_scaled_translated_q4_uses_physical_weights_without_normalization_drift() -> (
  None
):
  scale = 3.25
  translation = (13.0, -8.0)
  reference = _system().operators[0]
  transformed = _system(scale=scale, translation=translation).operators[0]
  assert transformed.payload.physical_integration_weights.values.sum() == pytest.approx(
    scale**2, rel=2.0e-15
  )
  coordinates = scale * np.array(
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64
  ) + np.asarray(translation)
  displacement = _affine_displacement(coordinates, origin=translation)
  reference_evaluation = _evaluate(
    reference, _unit_displacement(), _PHASE, np.zeros((4, 1))
  )
  transformed_evaluation = _evaluate(
    transformed, displacement, _PHASE, np.zeros((4, 1))
  )
  np.testing.assert_allclose(
    transformed_evaluation.trial_state.values,
    reference_evaluation.trial_state.values,
    rtol=2.0e-14,
    atol=2.0e-14,
  )
  np.testing.assert_allclose(
    transformed_evaluation.residual_values[0].values,
    scale * reference_evaluation.residual_values[0].values,
    rtol=3.0e-13,
    atol=3.0e-13,
  )
  uniform = np.full(4, 0.1)
  reference_phase = _evaluate(
    reference, _unit_displacement(), uniform, np.zeros((4, 1)), jacobian=False
  )
  transformed_phase = _evaluate(
    transformed, displacement, uniform, np.zeros((4, 1)), jacobian=False
  )
  np.testing.assert_allclose(
    transformed_phase.residual_values[1].values,
    scale**2 * reference_phase.residual_values[1].values,
    rtol=3.0e-13,
    atol=3.0e-13,
  )


def test_fresh_trial_rows_order_independence_rejection_and_state_codec() -> None:
  system = _system()
  operator = system.operators[0]
  accepted = np.zeros((4, 1), dtype=np.float64)
  accepted_bytes = accepted.tobytes()
  first = _evaluate(operator, _unit_displacement(scale=0.25), _PHASE, accepted)
  first_trial = first.trial_state.values
  first_trial_bytes = first_trial.tobytes()
  direct = _evaluate(operator, _unit_displacement(), _PHASE, accepted)
  after_other = _evaluate(operator, _unit_displacement(scale=0.25), _PHASE, accepted)
  repeated = _evaluate(operator, _unit_displacement(), _PHASE, accepted)
  np.testing.assert_array_equal(_residual(repeated), _residual(direct))
  np.testing.assert_array_equal(repeated.trial_state.values, direct.trial_state.values)
  assert first_trial.tobytes() == first_trial_bytes
  assert accepted.tobytes() == accepted_bytes
  assert not np.shares_memory(first_trial, accepted)
  assert not np.shares_memory(after_other.trial_state.values, accepted)

  rejected = _evaluate(
    operator,
    _unit_displacement(scale=2.0),
    np.array([0.2, 0.3, 1.2, 0.4]),
    accepted,
  )
  assert rejected.status is EvaluationStatus.REJECT_ITERATION
  assert rejected.residual_values == ()
  assert rejected.jacobian_values == ()
  assert rejected.trial_state.values.tobytes() == accepted_bytes
  assert not np.shares_memory(rejected.trial_state.values, accepted)

  owner = NonlinearStaticDriver(system, compile_constraint_map(system)).owner
  layout = operator.header.state_layout
  assert layout.schema == PHASE_FIELD_STATE_SCHEMA
  assert layout.slots[0].annotation == "monotone-nondecreasing"
  transaction = owner.begin()
  transaction.stage_state(layout.block_id, direct.trial_state)
  transaction.commit()
  payload = owner.encode_state(layout.block_id)
  decoded = owner.decode_state(layout.block_id, payload)
  assert decoded.values.tobytes() == direct.trial_state.values.tobytes()
  with pytest.raises(StateCodecError):
    owner.decode_state(
      layout.block_id,
      payload.replace(b"pyfem-v3-q4-at2-history-v1", b"pyfem-v3-q4-at2-history-v2"),
    )


def test_out_of_range_phase_and_kinks_have_distinct_typed_failures() -> None:
  operator = _system().operators[0]
  accepted = np.zeros((4, 1))
  rejected = _evaluate(operator, _unit_displacement(), np.full(4, -1.0e-8), accepted)
  assert rejected.status is EvaluationStatus.REJECT_ITERATION
  assert rejected.trial_state.values.tobytes() == accepted.tobytes()

  history_kink = spectral_at2_response(
    _STRAIN[None, :], _LAMBDA, _MU, tangent=False
  ).positive_energy[0]
  with pytest.raises(PhaseFieldPreparationError, match=r"psi\+") as history_error:
    _evaluate(operator, _unit_displacement(), _PHASE, np.full((4, 1), history_kink))
  assert history_error.value.code == "phase-field-history-kink"
  with pytest.raises(
    PhaseFieldPreparationError, match="zero eigenvalues or trace"
  ) as spectral_error:
    _evaluate(operator, np.zeros(8), _PHASE, accepted)
  assert spectral_error.value.code == "phase-field-spectral-kink"

  with pytest.raises(ModelCompilationError, match="unsupported-initial-crack-state"):
    compile_system(_model(initial_crack=True), phase_field_reference_registry())


def _point(value: float) -> ProgramPoint:
  return ProgramPoint((ProgramCoordinateValue("load", value),))


def _constraint_map(system: CompiledSystem) -> CompiledConstraintMap:
  displacement = next(space for space in system.spaces if len(space.components) == 2)
  coordinates = np.array(
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64
  )
  unit = _affine_displacement(coordinates).reshape(4, 2)
  constraints = []
  for node in range(1, 5):
    for component_index, component in enumerate(("x", "y")):
      factor = unit[node - 1, component_index]
      constraints.append(
        PrescribedDofSpec(
          f"u-{node}-{component}",
          DofRef(node, displacement.space_id, component),
          AffineValueSpec(coefficients=(AffineCoefficientSpec("load", float(factor)),)),
          _source(f"u-{node}-{component}"),
        )
      )
  return compile_constraint_map(
    system,
    constraints=tuple(constraints),
    coordinates=(ProgramCoordinateSpec("load", "load"),),
  )


def _committed_history(driver: NonlinearStaticDriver) -> np.ndarray:
  operator = driver.owner.system.operators[0]
  return driver.owner.accepted_state(operator.header.state_layout.block_id).values


def _phase_values(
  driver: NonlinearStaticDriver,
  coordinate_map: CompiledConstraintMap,
) -> np.ndarray:
  return driver.owner.accepted_physical().values[coordinate_map.free_dofs.values]


def test_actual_driver_affine_history_sequence_and_independent_phase() -> None:
  system = _system()
  coordinate_map = _constraint_map(system)
  driver = NonlinearStaticDriver(
    system,
    coordinate_map,
    settings=NonlinearStaticSettings(tolerance=1.0e-12),
  )
  expected_multipliers = (1.0, 1.0, 1.0, 2.25)
  for scale, multiplier in zip(
    (1.0, 0.25, 0.75, 1.5), expected_multipliers, strict=True
  ):
    result = driver.run(base_point=_point(0.0), target_points=(_point(scale),))
    assert result.status is DriverStatus.COMPLETED
    assert len(result.records) == 1
    assert result.records[0].status is SubstepStatus.COMMITTED
    expected_history = multiplier * _H
    np.testing.assert_allclose(
      _committed_history(driver),
      np.full((4, 1), expected_history),
      rtol=2.0e-13,
      atol=2.0e-13,
    )
    phase_star = frozen_history_phase_solution(expected_history, _GC, _ELL)
    np.testing.assert_allclose(
      _phase_values(driver, coordinate_map),
      np.full(4, phase_star),
      rtol=2.0e-11,
      atol=2.0e-11,
    )


def test_actual_driver_rejected_scale_two_never_leaks_four_h_on_retry() -> None:
  system = _system()
  coordinate_map = _constraint_map(system)
  initial = NonlinearStaticDriver(system, coordinate_map)
  accepted = initial.run(base_point=_point(0.0), target_points=(_point(1.0),))
  assert accepted.status is DriverStatus.COMPLETED
  owner = initial.owner
  block_id = system.operators[0].header.state_layout.block_id
  assert np.allclose(owner.accepted_state(block_id).values, _H)

  target_phase = frozen_history_phase_solution(2.25 * _H, _GC, _ELL)
  transaction = owner.begin()
  full = np.array(owner.accepted_physical().values, copy=True)
  full[coordinate_map.free_dofs.values] = target_phase
  transaction.stage_physical(full)
  transaction.commit()
  history_before = owner.accepted_state(block_id).values.tobytes()

  driver = NonlinearStaticDriver(
    system,
    coordinate_map,
    settings=NonlinearStaticSettings(max_iterations=1, max_cutbacks=1),
    owner=owner,
  )
  result = driver.run(base_point=_point(1.0), target_points=(_point(2.0),))
  assert result.status is DriverStatus.STEP_FAILED
  assert [record.status for record in result.records[:2]] == [
    SubstepStatus.REJECTED,
    SubstepStatus.COMMITTED,
  ]
  assert result.records[0].point.values[0].value == pytest.approx(2.0)
  assert result.records[1].point.values[0].value == pytest.approx(1.5)
  assert result.records[0].committed_ordinal is None
  assert all(
    record.status is not SubstepStatus.COMMITTED for record in result.records[2:]
  )
  np.testing.assert_allclose(
    owner.accepted_state(block_id).values,
    np.full((4, 1), 2.25 * _H),
    rtol=2.0e-13,
    atol=2.0e-13,
  )
  assert np.max(owner.accepted_state(block_id).values) < 4.0 * _H
  assert history_before == np.full((4, 1), _H).tobytes()


def test_out_of_range_phase_driver_rejection_preserves_accepted_bytes() -> None:
  system = _system()
  coordinate_map = _constraint_map(system)
  driver = NonlinearStaticDriver(
    system,
    coordinate_map,
    settings=NonlinearStaticSettings(max_cutbacks=0),
  )
  owner = driver.owner
  transaction = owner.begin()
  full = np.array(owner.accepted_physical().values, copy=True)
  full[coordinate_map.free_dofs.values] = -0.125
  transaction.stage_physical(full)
  transaction.commit()
  block_id = system.operators[0].header.state_layout.block_id
  physical_before = owner.accepted_physical().values.tobytes()
  history_before = owner.accepted_state(block_id).values.tobytes()
  result = driver.run(base_point=_point(0.0), target_points=(_point(1.0),))
  assert result.status is DriverStatus.STEP_FAILED
  assert result.records[0].iterations[0].status is EvaluationStatus.REJECT_ITERATION
  assert owner.accepted_physical().values.tobytes() == physical_before
  assert owner.accepted_state(block_id).values.tobytes() == history_before
