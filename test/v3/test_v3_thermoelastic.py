# SPDX-License-Identifier: MIT

"""Steady T3 thermoelastic multiport execution and independent oracles."""

from __future__ import annotations

import sys

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from pyfem.v3.compile.system import compile_system
from pyfem.v3.compile.thermoelastic import (
  T3ThermoElasticOperator,
  thermoelastic_reference_registry,
)
from pyfem.v3.constraints import (
  CompiledConstraintMap,
  compile_constraint_map,
  full_coefficients,
  reduce_residual,
)
from pyfem.v3.driver import (
  ArcLengthSettings,
  DriverStatus,
  NonlinearStaticDriver,
  NonlinearStaticSettings,
  RiksDriver,
  SubstepStatus,
  compile_driver_plan,
  evaluate_loads,
)
from pyfem.v3.driver._evaluation import assemble_evaluation_balance
from pyfem.v3.driver.plan import refill_tangent_channels
from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.operator import (
  ChannelRequest,
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
  NodalLoadSpec,
  PrescribedDofSpec,
  ProgramCoordinateSpec,
  ProgramCoordinateValue,
  ProgramPoint,
)

_U = np.array([0.01, -0.02, 0.03, 0.01, -0.01, 0.04], dtype=np.float64)
_T = np.array([0.4, 0.6, 0.8], dtype=np.float64)
_KUU = np.array(
  [
    [80, 40, -60, -20, -20, -20],
    [40, 80, -20, -20, -20, -60],
    [-60, -20, 60, 0, 0, 20],
    [-20, -20, 0, 20, 20, 0],
    [-20, -20, 0, 20, 20, 0],
    [-20, -60, 20, 0, 0, 60],
  ],
  dtype=np.float64,
)
_H = np.array([-1.6, -1.6, 1.6, 0.0, 0.0, 1.6], dtype=np.float64)
_KTT = np.array(
  [[3.0, -1.5, -1.5], [-1.5, 1.5, 0.0], [-1.5, 0.0, 1.5]],
  dtype=np.float64,
)


def _source(label: str) -> SourceContext:
  return SourceContext(source=label)


def _model(*, reordered: bool = False) -> ModelSpec:
  source = _source("reordered" if reordered else "reference")
  nodes = (
    NodeSpec(1, (0.0, 0.0), _source("n1")),
    NodeSpec(2, (1.0, 0.0), _source("n2")),
    NodeSpec(3, (0.0, 1.0), _source("n3")),
  )
  cell = CellSpec("triangle", (1, 2, 3), _source("cell"))
  displacement = FieldSpec(
    "zzz-solid" if reordered else "displacement",
    ("y", "x") if reordered else ("x", "y"),
    "node",
    _source("displacement"),
  )
  temperature = FieldSpec("temperature", ("theta",), "node", _source("temperature"))
  fields = (temperature, displacement) if reordered else (displacement, temperature)
  return ModelSpec(
    MeshSpec(
      nodes,
      (
        CellBlockSpec(
          "cells",
          "triangle",
          2,
          2,
          "linear-tria3",
          (cell,),
          _source("block"),
        ),
      ),
      source,
    ),
    fields,
    (
      MaterialSpec(
        "material",
        "plane-strain-thermoelastic",
        (
          MaterialParameterSpec("youngs_modulus", 100.0),
          MaterialParameterSpec("poisson_ratio", 0.25),
          MaterialParameterSpec("thermal_expansion", 0.02),
          MaterialParameterSpec("conductivity", 3.0),
          MaterialParameterSpec("reference_temperature", 0.25),
        ),
        _source("material"),
      ),
    ),
    (
      RegionSpec(
        "domain",
        (CellRef("cells", "triangle"),),
        (displacement.id, temperature.id),
        "material",
        "steady-thermoelastic-tria3",
        "gauss-tria3-1",
        _source("region"),
      ),
    ),
    source,
  )


def _system(*, reordered: bool = False) -> CompiledSystem:
  return compile_system(_model(reordered=reordered), thermoelastic_reference_registry())


def _evaluate(
  operator: T3ThermoElasticOperator,
  displacement: np.ndarray,
  temperature: np.ndarray,
  *,
  jacobian: bool = True,
) -> OperatorEvaluation:
  return operator.evaluate(
    OperatorEvaluationInput(
      (
        FinalizedArray(displacement.reshape(1, 6), dtype=np.float64),
        FinalizedArray(temperature.reshape(1, 3), dtype=np.float64),
      ),
      FinalizedArray(
        np.zeros(operator.header.state_layout.row_shape), dtype=np.float64
      ),
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


def _evaluate_full(system: CompiledSystem, full: np.ndarray) -> OperatorEvaluation:
  operator = system.operators[0]
  displacement = full[operator.header.ports[0].coefficient_map.values[0]]
  temperature = full[operator.header.ports[1].coefficient_map.values[0]]
  return _evaluate(operator, displacement, temperature)


def _local_dense(operator: T3ThermoElasticOperator) -> np.ndarray:
  dense = np.zeros((9, 9), dtype=np.float64)
  dense[:6, :6] = operator.payload.mechanical_tangent.values[0]
  dense[:6, 6:] = operator.payload.mechanical_thermal_tangent.values[0]
  dense[6:, 6:] = operator.payload.thermal_tangent.values[0]
  return dense


def _point(value: float) -> ProgramPoint:
  return ProgramPoint((ProgramCoordinateValue("load", value),))


def _program(
  system: CompiledSystem,
) -> tuple[CompiledConstraintMap, tuple[NodalLoadSpec, ...]]:
  displacement = next(space for space in system.spaces if len(space.components) == 2)
  constraints = (
    PrescribedDofSpec(
      "fix-1x",
      DofRef(1, displacement.space_id, "x"),
      AffineValueSpec(constant=0.0),
      _source("fix-1x"),
    ),
    PrescribedDofSpec(
      "fix-1y",
      DofRef(1, displacement.space_id, "y"),
      AffineValueSpec(constant=0.0),
      _source("fix-1y"),
    ),
    PrescribedDofSpec(
      "fix-2y",
      DofRef(2, displacement.space_id, "y"),
      AffineValueSpec(constant=0.0),
      _source("fix-2y"),
    ),
    PrescribedDofSpec(
      "offset-3x",
      DofRef(3, displacement.space_id, "x"),
      AffineValueSpec(constant=0.01),
      _source("offset-3x"),
    ),
    PrescribedDofSpec(
      "offset-1T",
      DofRef(1, "temperature", "theta"),
      AffineValueSpec(constant=0.5),
      _source("offset-1T"),
    ),
  )
  coordinate_map = compile_constraint_map(
    system,
    constraints=constraints,
    coordinates=(ProgramCoordinateSpec("load", "load"),),
  )
  loads = (
    NodalLoadSpec(
      "mechanical-x",
      DofRef(2, displacement.space_id, "x"),
      AffineValueSpec(
        coefficients=(AffineCoefficientSpec("load", 0.8),),
      ),
      _source("mechanical-x"),
    ),
    NodalLoadSpec(
      "mechanical-y",
      DofRef(3, displacement.space_id, "y"),
      AffineValueSpec(
        coefficients=(AffineCoefficientSpec("load", -0.4),),
      ),
      _source("mechanical-y"),
    ),
    NodalLoadSpec(
      "thermal-2",
      DofRef(2, "temperature", "theta"),
      AffineValueSpec(
        coefficients=(AffineCoefficientSpec("load", 0.3),),
      ),
      _source("thermal-2"),
    ),
    NodalLoadSpec(
      "thermal-3",
      DofRef(3, "temperature", "theta"),
      AffineValueSpec(constant=0.2),
      _source("thermal-3"),
    ),
  )
  return coordinate_map, loads


def test_exact_t3_matrices_residuals_and_free_expansion() -> None:
  operator = _system().operators[0]
  np.testing.assert_array_equal(operator.payload.mechanical_tangent.values[0], _KUU)
  np.testing.assert_array_equal(operator.payload.thermal_expansion_force.values[0], _H)
  np.testing.assert_array_equal(operator.payload.thermal_tangent.values[0], _KTT)
  expected_u_t = np.repeat(-_H[:, None] / 3.0, 3, axis=1)
  np.testing.assert_allclose(
    operator.payload.mechanical_thermal_tangent.values[0],
    expected_u_t,
    rtol=0.0,
    atol=5.0e-16,
  )
  evaluation = _evaluate(operator, _U, _T)
  np.testing.assert_allclose(
    evaluation.residual_values[0].values[0],
    np.array([-2.04, -3.64, 1.84, 0.2, 0.2, 3.44]),
    rtol=0.0,
    atol=2.0e-15,
  )
  # Tower-corrected M134 arithmetic: the nodal block is G @ G.T, with negative
  # off-diagonal entries for the stated right triangle.
  np.testing.assert_allclose(
    evaluation.residual_values[1].values[0],
    np.array([-0.9, 0.3, 0.6]),
    rtol=0.0,
    atol=2.0e-16,
  )
  free_temperature = 0.9
  free_displacement = (
    0.02 * (free_temperature - 0.25) * np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0])
  )
  free = _evaluate(
    operator,
    free_displacement,
    np.full(3, free_temperature),
    jacobian=False,
  )
  np.testing.assert_allclose(
    free.residual_values[0].values,
    np.zeros((1, 6)),
    rtol=0.0,
    atol=2.0e-15,
  )
  assert tuple(
    channel.target_port_id for channel in operator.header.jacobian_channels
  ) == (
    "u",
    "u",
    "T",
  )
  assert tuple(
    channel.source_port_id for channel in operator.header.jacobian_channels
  ) == (
    "u",
    "T",
    "T",
  )
  assert all(
    not (channel.target_port_id == "T" and channel.source_port_id == "u")
    for channel in operator.header.jacobian_channels
  )


def test_every_local_rectangular_block_matches_central_fd() -> None:
  operator = _system().operators[0]
  step = 1.0e-6
  expected = _local_dense(operator)
  for index in range(9):
    plus_u = _U.copy()
    plus_t = _T.copy()
    minus_u = _U.copy()
    minus_t = _T.copy()
    if index < 6:
      plus_u[index] += step
      minus_u[index] -= step
    else:
      plus_t[index - 6] += step
      minus_t[index - 6] -= step
    plus = _evaluate(operator, plus_u, plus_t, jacobian=False)
    minus = _evaluate(operator, minus_u, minus_t, jacobian=False)
    difference = (
      np.concatenate(
        (plus.residual_values[0].values[0], plus.residual_values[1].values[0])
      )
      - np.concatenate(
        (minus.residual_values[0].values[0], minus.residual_values[1].values[0])
      )
    ) / (2.0 * step)
    # This includes the structurally absent T,u block: its FD column is exactly
    # zero, and no corresponding topology exists in the compiled header.
    np.testing.assert_allclose(difference, expected[:, index], rtol=2.0e-8, atol=2.0e-8)
  np.testing.assert_array_equal(expected[6:, :6], np.zeros((3, 6)))


def test_full_coo_stream_external_signs_and_balance() -> None:
  system = _system()
  coordinate_map, loads = _program(system)
  plan = compile_driver_plan(system, coordinate_map, loads)
  evaluation = _evaluate_full(system, np.concatenate((_U, _T)))
  assert plan.provenance.schema == "pyfem-v3-driver-assembly-plan-multiport-v1"
  assert plan.coo_entry_count == 63
  assert plan.constant_tangent is True
  assert plan.legacy_single_port is False
  tangent = refill_tangent_channels(
    plan,
    (tuple(value.values for value in evaluation.jacobian_values),),
  )
  np.testing.assert_allclose(
    tangent.toarray(), _local_dense(system.operators[0]), rtol=0.0, atol=2.0e-14
  )
  expected_rows: list[np.ndarray] = []
  expected_columns: list[np.ndarray] = []
  for channel in system.operators[0].header.jacobian_channels:
    ports = {
      port.port_id: port.coefficient_map.values
      for port in system.operators[0].header.ports
    }
    target = ports[channel.target_port_id]
    source = ports[channel.source_port_id]
    shape = (1, target.shape[1], source.shape[1])
    expected_rows.append(np.broadcast_to(target[:, :, None], shape).reshape(-1))
    expected_columns.append(np.broadcast_to(source[:, None, :], shape).reshape(-1))
  np.testing.assert_array_equal(
    plan.coo_row_indices.values, np.concatenate(expected_rows)
  )
  np.testing.assert_array_equal(
    plan.coo_column_indices.values,
    np.concatenate(expected_columns),
  )
  external = evaluate_loads(plan.loads, _point(1.0)).values
  full_residual, rhs, _jacobian = assemble_evaluation_balance(
    plan,
    (evaluation,),
    external,
  )
  local_residual = np.concatenate((_KUU @ _U - _H * (_T.mean() - 0.25), _KTT @ _T))
  np.testing.assert_allclose(
    full_residual, local_residual - external, rtol=0.0, atol=2.0e-14
  )
  np.testing.assert_allclose(rhs, external - local_residual, rtol=0.0, atol=2.0e-14)


def _dense_constrained_solution(
  system: CompiledSystem,
  coordinate_map: CompiledConstraintMap,
  loads: tuple[NodalLoadSpec, ...],
  load: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  plan = compile_driver_plan(system, coordinate_map, loads)
  evaluation = _evaluate_full(system, np.zeros(9))
  tangent = refill_tangent_channels(
    plan,
    (tuple(value.values for value in evaluation.jacobian_values),),
  ).toarray()
  constant = assemble_evaluation_balance(plan, (evaluation,), np.zeros(9))[0]
  point = _point(load)
  offset = full_coefficients(
    coordinate_map, np.zeros(coordinate_map.reduced_dof_count), point
  ).values
  external = evaluate_loads(plan.loads, point).values
  prolongation = coordinate_map.prolongation().toarray()
  rhs = prolongation.T @ (external - constant - tangent @ offset)
  reduced = np.linalg.solve(prolongation.T @ tangent @ prolongation, rhs)
  full = full_coefficients(coordinate_map, reduced, point).values
  residual = constant + tangent @ full - external
  return full, residual, tangent, prolongation


def test_constrained_dense_solve_reactions_factorization_and_rollback() -> None:
  system = _system()
  coordinate_map, loads = _program(system)
  expected, expected_residual, _tangent, prolongation = _dense_constrained_solution(
    system,
    coordinate_map,
    loads,
    1.0,
  )
  driver = NonlinearStaticDriver(system, coordinate_map, loads)
  result = driver.run(base_point=_point(0.0), target_points=(_point(0.5), _point(1.0)))
  assert result.status is DriverStatus.COMPLETED
  np.testing.assert_allclose(
    driver.owner.accepted_physical().values, expected, rtol=2.0e-13, atol=2.0e-13
  )
  assert result.statistics.tangent_refill_count == 1
  assert result.statistics.factorization_count == 1
  assert result.statistics.factorization_reuse_count == 1
  assert result.statistics.evaluation_count == 4
  observation = result.records[-1].observation
  assert observation is not None
  expected_reactions = np.zeros(9)
  expected_reactions[coordinate_map.constrained_dofs.values] = expected_residual[
    coordinate_map.constrained_dofs.values
  ]
  np.testing.assert_allclose(
    observation.reactions.values,
    expected_reactions,
    rtol=2.0e-12,
    atol=2.0e-12,
  )
  assert observation.full_residual_norm == pytest.approx(
    float(np.linalg.norm(expected_residual)),
    abs=2.0e-12,
  )
  reduced = reduce_residual(coordinate_map, -expected_residual).values
  np.testing.assert_allclose(reduced, np.zeros(4), rtol=0.0, atol=2.0e-11)
  np.testing.assert_allclose(
    prolongation.T @ expected_residual,
    np.zeros(4),
    rtol=0.0,
    atol=2.0e-11,
  )

  rejected_driver = NonlinearStaticDriver(
    system,
    coordinate_map,
    loads,
    NonlinearStaticSettings(max_iterations=1, max_cutbacks=1),
  )
  before = rejected_driver.owner.accepted_physical().values.tobytes()
  rejected = rejected_driver.run(base_point=_point(0.0), target_points=(_point(1.0),))
  assert rejected.status is DriverStatus.STEP_FAILED
  assert rejected.final_generation.ordinal == 0
  assert rejected_driver.owner.accepted_physical().values.tobytes() == before
  assert all(record.observation is None for record in rejected.records)


def test_reordered_fields_and_components_preserve_source_remapped_response() -> None:
  reference = _system()
  reordered = _system(reordered=True)
  assert reordered.coefficient_count == reference.coefficient_count == 9
  assert [space.space_id for space in reordered.spaces] == ["temperature", "zzz-solid"]
  reference_evaluation = _evaluate(reference.operators[0], _U, _T)
  reordered_evaluation = _evaluate(reordered.operators[0], _U, _T)
  for actual, expected in zip(
    reordered_evaluation.residual_values,
    reference_evaluation.residual_values,
    strict=True,
  ):
    np.testing.assert_array_equal(actual.values, expected.values)
  reference_plan = compile_driver_plan(reference, compile_constraint_map(reference))
  reordered_plan = compile_driver_plan(reordered, compile_constraint_map(reordered))
  reference_dense = refill_tangent_channels(
    reference_plan,
    (tuple(value.values for value in reference_evaluation.jacobian_values),),
  ).toarray()
  reordered_dense = refill_tangent_channels(
    reordered_plan,
    (tuple(value.values for value in reordered_evaluation.jacobian_values),),
  ).toarray()

  def identifiers(system: CompiledSystem) -> list[tuple[str, int, str]]:
    return [
      (
        "T" if space.space_id == "temperature" else "u",
        coefficient[1],
        coefficient[2],
      )
      for space in system.spaces
      for coefficient in space.coefficient_ids
    ]

  reference_ids = identifiers(reference)
  permutation = np.array([reference_ids.index(item) for item in identifiers(reordered)])
  np.testing.assert_array_equal(
    reordered_dense, reference_dense[np.ix_(permutation, permutation)]
  )


def test_nondimensional_riks_trajectory_matches_dense_predictors_and_corrections() -> (
  None
):
  system = _system()
  coordinate_map, loads = _program(system)
  settings = ArcLengthSettings(
    tolerance=1.0e-12,
    fixed_step=True,
    max_lam=0.5,
    max_iterations=10,
  )
  driver = RiksDriver(system, coordinate_map, loads, settings)
  result = driver.run(base_point=ProgramPoint())
  assert result.status is DriverStatus.COMPLETED
  assert len(result.records) == 2
  assert all(record.status is SubstepStatus.COMMITTED for record in result.records)

  plan = driver.plan
  zero_evaluation = _evaluate_full(system, np.zeros(9))
  tangent = refill_tangent_channels(
    plan,
    (tuple(value.values for value in zero_evaluation.jacobian_values),),
  ).toarray()
  constant = assemble_evaluation_balance(plan, (zero_evaluation,), np.zeros(9))[0]
  prolongation = coordinate_map.prolongation().toarray()
  reduced_tangent = prolongation.T @ tangent @ prolongation
  fhat = plan.loads.coordinate_coefficients.values[:, 0]
  fhat_reduced = prolongation.T @ fhat
  d1 = np.linalg.solve(reduced_tangent, fhat_reduced)
  committed_reduced = np.zeros(4)
  committed_lam = 0.0
  previous_da = np.zeros(4)
  previous_dlam = 0.0
  expected_lams: list[float] = []
  expected_fulls: list[np.ndarray] = []
  expected_ddlams: list[float] = []
  expected_predictor_lams: list[float] = []
  for cycle in range(2):
    if cycle == 0:
      predictor_lam = 1.0
      dlam1 = predictor_lam
      da1 = np.linalg.solve(reduced_tangent, predictor_lam * fhat_reduced)
      predictor_reduced = da1
    else:
      da1 = previous_da
      dlam1 = previous_dlam
      predictor_lam = committed_lam + dlam1
      predictor_reduced = committed_reduced + da1
    predictor = full_coefficients(
      coordinate_map,
      predictor_reduced,
      _point(predictor_lam),
    ).values
    predictor_residual = (
      constant + tangent @ predictor - plan.loads.constant.values - predictor_lam * fhat
    )
    d2 = np.linalg.solve(reduced_tangent, prolongation.T @ (-predictor_residual))
    ddlam = -float(da1 @ d2) / float(da1 @ d1)
    dda = ddlam * d1 + d2
    committed_lam = predictor_lam + ddlam
    committed_reduced = predictor_reduced + dda
    previous_da = da1 + dda
    previous_dlam = dlam1 + ddlam
    expected_lams.append(committed_lam)
    expected_ddlams.append(ddlam)
    expected_predictor_lams.append(predictor_lam)
    expected_fulls.append(
      full_coefficients(
        coordinate_map,
        committed_reduced,
        _point(committed_lam),
      ).values
    )
  for record, expected_lam, expected_full, ddlam, predictor_lam in zip(
    result.records,
    expected_lams,
    expected_fulls,
    expected_ddlams,
    expected_predictor_lams,
    strict=True,
  ):
    assert record.lam == pytest.approx(expected_lam, abs=2.0e-13)
    assert record.iterations[0].load_parameter == pytest.approx(
      predictor_lam,
      abs=2.0e-13,
    )
    assert record.iterations[0].delta_lam == pytest.approx(ddlam, abs=2.0e-13)
    assert record.committed_coefficients is not None
    np.testing.assert_allclose(
      record.committed_coefficients.values,
      expected_full,
      rtol=2.0e-12,
      atol=2.0e-12,
    )
    final_residual = (
      constant
      + tangent @ expected_full
      - plan.loads.constant.values
      - expected_lam * fhat
    )
    np.testing.assert_allclose(
      prolongation.T @ final_residual,
      np.zeros(4),
      rtol=0.0,
      atol=2.0e-10,
    )
  assert result.statistics.evaluation_count == 5
  assert result.statistics.residual_assembly_count == 4
  assert result.statistics.tangent_refill_count == 1
  assert result.statistics.factorization_count == 1
  assert result.statistics.factorization_reuse_count == 2
  assert result.statistics.linear_solve_count == 5

  rejected_driver = RiksDriver(
    system,
    coordinate_map,
    loads,
    ArcLengthSettings(max_iterations=1, max_cutbacks=1, max_lam=0.5),
  )
  before = rejected_driver.owner.accepted_physical().values.tobytes()
  rejected = rejected_driver.run(base_point=ProgramPoint())
  assert rejected.status is DriverStatus.STEP_FAILED
  assert rejected.final_generation.ordinal == 0
  assert rejected.final_continuation.cycle == 0
  assert rejected_driver.owner.accepted_physical().values.tobytes() == before
