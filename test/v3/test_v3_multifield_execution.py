# SPDX-License-Identifier: MIT

"""Channel-driven multiport roles, signed assembly, and fail-closed bounds."""

from __future__ import annotations

import dataclasses
import sys

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from pyfem.v3.compile.thermoelastic import T3ThermoElasticOperator
from pyfem.v3.constraints import compile_constraint_map, full_coefficients
from pyfem.v3.driver import (
  DriverPreparationError,
  DriverStatus,
  NonlinearStaticDriver,
  RiksDriver,
  assemble_internal_force,
  compile_driver_plan,
  evaluate_loads,
  refill_tangent,
)
from pyfem.v3.driver._evaluation import assemble_evaluation_balance
from pyfem.v3.driver.contracts import HomogenizationReduction, HomogenizationRequest
from pyfem.v3.driver.plan import (
  compile_sensitivity_program,
  compile_total_sensitivity_program,
  refill_tangent_channels,
)
from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.identity import InstanceId
from pyfem.v3.model.operator import (
  BalanceRole,
  JacobianChannel,
  OperatorHeader,
  PortBinding,
  ResidualChannel,
)
from pyfem.v3.model.provenance import CanonicalManifest, ContentFingerprint
from pyfem.v3.model.system import CompiledSystem

# isort: split

from test_v3_phase_field import (
  _PHASE,
  _unit_displacement,
)
from test_v3_phase_field import (
  _dense as _phase_dense,
)
from test_v3_phase_field import (
  _evaluate as _evaluate_phase,
)
from test_v3_phase_field import (
  _system as _phase_system,
)
from test_v3_thermoelastic import (
  _T,
  _U,
  _evaluate_full,
  _local_dense,
  _point,
  _program,
  _system,
)


def _new[ValueT](cls: type[ValueT], /, **fields: object) -> ValueT:
  value = object.__new__(cls)
  for name, field in fields.items():
    object.__setattr__(value, name, field)
  return value


def _role_system(
  mechanical: BalanceRole,
  thermal: BalanceRole,
) -> CompiledSystem:
  base = _system()
  operator = base.operators[0]
  roles = (mechanical, thermal)
  residuals = tuple(
    _new(
      ResidualChannel,
      channel_id=channel.channel_id,
      target_port_id=channel.target_port_id,
      balance_role=role,
      linear=channel.linear,
    )
    for channel, role in zip(operator.header.residual_channels, roles, strict=True)
  )
  role_by_residual = {
    residual.channel_id: residual.balance_role for residual in residuals
  }
  jacobians = tuple(
    _new(
      JacobianChannel,
      channel_id=channel.channel_id,
      residual_channel_id=channel.residual_channel_id,
      target_port_id=channel.target_port_id,
      source_port_id=channel.source_port_id,
      balance_role=role_by_residual[channel.residual_channel_id],
      linear=channel.linear,
      symmetric=channel.symmetric,
    )
    for channel in operator.header.jacobian_channels
  )
  header = _new(
    OperatorHeader,
    block_id=operator.header.block_id,
    entity_block_id=operator.header.entity_block_id,
    implementations=operator.header.implementations,
    ports=operator.header.ports,
    signal_ports=operator.header.signal_ports,
    residual_channels=residuals,
    jacobian_channels=jacobians,
    state_layout=operator.header.state_layout,
    coupling_policy=operator.header.coupling_policy,
    parameters=operator.header.parameters,
    derivative_channels=operator.header.derivative_channels,
    nested_execution=operator.header.nested_execution,
    tangent_class=operator.header.tangent_class,
  )
  manifest = CanonicalManifest(
    {
      "schema": "test-multiport-role-system-v1",
      "mechanical_role": mechanical.value,
      "thermal_role": thermal.value,
    }
  )
  return _new(
    CompiledSystem,
    instance_id=InstanceId(),
    content_fingerprint=ContentFingerprint.from_manifest(manifest),
    provenance=base.provenance,
    registry_snapshot=base.registry_snapshot,
    point_blocks=base.point_blocks,
    entity_blocks=base.entity_blocks,
    spaces=base.spaces,
    operators=(
      _new(
        T3ThermoElasticOperator,
        header=header,
        entity_block=operator.entity_block,
        payload=operator.payload,
        content_manifest=manifest,
      ),
    ),
    source_attribution=base.source_attribution,
  )


def _signed_dense(system: CompiledSystem) -> np.ndarray:
  dense = _local_dense(system.operators[0])
  for residual in system.operators[0].header.residual_channels:
    if residual.balance_role is not BalanceRole.EXTERNAL:
      continue
    ports = {
      port.port_id: index for index, port in enumerate(system.operators[0].header.ports)
    }
    target = ports[residual.target_port_id]
    row_start = 0 if target == 0 else 6
    row_stop = 6 if target == 0 else 9
    dense[row_start:row_stop] *= -1.0
  return dense


def test_internal_external_algebra_and_newton_consume_both_roles() -> None:
  system = _role_system(BalanceRole.INTERNAL, BalanceRole.EXTERNAL)
  coordinate_map, loads = _program(system)
  plan = compile_driver_plan(system, coordinate_map, loads)
  evaluation = _evaluate_full(system, np.concatenate((_U, _T)))
  external = evaluate_loads(plan.loads, _point(1.0)).values
  full_residual, rhs, jacobian = assemble_evaluation_balance(
    plan,
    (evaluation,),
    external,
  )
  unsigned = np.concatenate(
    (
      evaluation.residual_values[0].values[0],
      -evaluation.residual_values[1].values[0],
    )
  )
  np.testing.assert_allclose(full_residual, unsigned - external, rtol=0.0, atol=2.0e-14)
  np.testing.assert_allclose(rhs, external - unsigned, rtol=0.0, atol=2.0e-14)
  tangent = refill_tangent_channels(plan, jacobian).toarray()
  np.testing.assert_allclose(tangent, _signed_dense(system), rtol=0.0, atol=2.0e-14)

  zero = _evaluate_full(system, np.zeros(9))
  constant = assemble_evaluation_balance(plan, (zero,), np.zeros(9))[0]
  prolongation = coordinate_map.prolongation().toarray()
  point = _point(1.0)
  offset = full_coefficients(coordinate_map, np.zeros(4), point).values
  reduced = np.linalg.solve(
    prolongation.T @ tangent @ prolongation,
    prolongation.T @ (external - constant - tangent @ offset),
  )
  expected = full_coefficients(coordinate_map, reduced, point).values
  driver = NonlinearStaticDriver(system, coordinate_map, loads)
  result = driver.run(base_point=_point(0.0), target_points=(point,))
  assert result.status is DriverStatus.COMPLETED
  np.testing.assert_allclose(
    driver.owner.accepted_physical().values,
    expected,
    rtol=2.0e-13,
    atol=2.0e-13,
  )
  assert result.statistics.tangent_refill_count == 1
  assert result.statistics.factorization_count == 1
  assert result.statistics.factorization_reuse_count == 0


def test_role_and_derivative_bounds_fail_closed() -> None:
  mixed = _role_system(BalanceRole.INTERNAL, BalanceRole.EXTERNAL)
  coordinate_map, loads = _program(mixed)
  with pytest.raises(DriverPreparationError, match="unsupported-riks-external-channel"):
    RiksDriver(mixed, coordinate_map, loads)
  with pytest.raises(DriverPreparationError, match="unsupported-external-sensitivity"):
    compile_sensitivity_program(mixed, ("youngs_modulus",))
  driver = NonlinearStaticDriver(mixed, coordinate_map, loads)
  homogenization = HomogenizationRequest(
    ("load",),
    (HomogenizationReduction("sum", (0,), 1.0),),
  )
  with pytest.raises(
    DriverPreparationError,
    match="unsupported-external-homogenization",
  ):
    driver.run(
      base_point=_point(0.0),
      target_points=(_point(1.0),),
      homogenization=homogenization,
    )

  internal = _system()
  internal_map, _internal_loads = _program(internal)
  with pytest.raises(DriverPreparationError, match="unsupported-multiport-sensitivity"):
    compile_sensitivity_program(internal, ("youngs_modulus",))
  with pytest.raises(
    DriverPreparationError,
    match="unsupported-multiport-total-sensitivity",
  ):
    compile_total_sensitivity_program(
      internal,
      internal_map,
      ("youngs_modulus",),
    )

  constrained = _role_system(BalanceRole.INTERNAL, BalanceRole.CONSTRAINT)
  constrained_map, constrained_loads = _program(constrained)
  with pytest.raises(DriverPreparationError, match="unsupported-constraint-channel"):
    compile_driver_plan(constrained, constrained_map, constrained_loads)


def test_balance_uses_one_bincount_per_role_and_one_tangent_dedup(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  system = _role_system(BalanceRole.INTERNAL, BalanceRole.EXTERNAL)
  coordinate_map, loads = _program(system)
  plan = compile_driver_plan(system, coordinate_map, loads)
  evaluation = _evaluate_full(system, np.concatenate((_U, _T)))
  bincount_calls = 0
  dedup_calls = 0
  original_bincount = np.bincount
  from pyfem.v3.fem.assembly import dedup_coo_segments

  def counted_bincount(*args: object, **kwargs: object) -> np.ndarray:
    nonlocal bincount_calls
    bincount_calls += 1
    return original_bincount(*args, **kwargs)

  def counted_dedup(*args: object, **kwargs: object) -> np.ndarray:
    nonlocal dedup_calls
    dedup_calls += 1
    return dedup_coo_segments(*args, **kwargs)

  monkeypatch.setattr("pyfem.v3.driver.plan.np.bincount", counted_bincount)
  monkeypatch.setattr("pyfem.v3.driver.plan.dedup_coo_segments", counted_dedup)
  _residual, _rhs, jacobian = assemble_evaluation_balance(
    plan,
    (evaluation,),
    evaluate_loads(plan.loads, _point(1.0)).values,
  )
  assert bincount_calls == 2
  refill_tangent_channels(plan, jacobian)
  assert dedup_calls == 1


def _header_with(header: OperatorHeader, **overrides: object) -> OperatorHeader:
  fields = {
    field.name: getattr(header, field.name) for field in dataclasses.fields(header)
  }
  fields.update(overrides)
  return _new(OperatorHeader, **fields)


def _system_with_header(base: CompiledSystem, header: OperatorHeader) -> CompiledSystem:
  operator = base.operators[0]
  manifest = CanonicalManifest({"schema": "test-malformed-multiport-system-v1"})
  return _new(
    CompiledSystem,
    instance_id=InstanceId(),
    content_fingerprint=ContentFingerprint.from_manifest(manifest),
    provenance=base.provenance,
    registry_snapshot=base.registry_snapshot,
    point_blocks=base.point_blocks,
    entity_blocks=base.entity_blocks,
    spaces=base.spaces,
    operators=(
      _new(
        T3ThermoElasticOperator,
        header=header,
        entity_block=operator.entity_block,
        payload=operator.payload,
        content_manifest=manifest,
      ),
    ),
    source_attribution=base.source_attribution,
  )


def test_malformed_multiport_declarations_fail_before_assembly() -> None:
  base = _role_system(BalanceRole.INTERNAL, BalanceRole.INTERNAL)
  header = base.operators[0].header
  rectangular = tuple(
    _new(
      JacobianChannel,
      channel_id=channel.channel_id,
      residual_channel_id=channel.residual_channel_id,
      target_port_id=channel.target_port_id,
      source_port_id=channel.source_port_id,
      balance_role=channel.balance_role,
      linear=channel.linear,
      symmetric=True,
    )
    if channel.target_port_id != channel.source_port_id
    else channel
    for channel in header.jacobian_channels
  )
  system = _system_with_header(
    base,
    _header_with(header, jacobian_channels=rectangular),
  )
  coordinate_map, loads = _program(system)
  with pytest.raises(DriverPreparationError, match="rectangular-symmetric-jacobian"):
    compile_driver_plan(system, coordinate_map, loads)

  duplicate_ports = (
    header.ports[0],
    _new(
      PortBinding,
      port_id=header.ports[0].port_id,
      space_id=header.ports[1].space_id,
      mode=header.ports[1].mode,
      coefficient_map=header.ports[1].coefficient_map,
    ),
  )
  system = _system_with_header(base, _header_with(header, ports=duplicate_ports))
  coordinate_map, loads = _program(system)
  with pytest.raises(DriverPreparationError, match="duplicate-channel-id"):
    compile_driver_plan(system, coordinate_map, loads)

  bad_residuals = (
    _new(
      ResidualChannel,
      channel_id=header.residual_channels[0].channel_id,
      target_port_id="ghost",
      balance_role=BalanceRole.INTERNAL,
      linear=True,
    ),
    header.residual_channels[1],
  )
  system = _system_with_header(
    base, _header_with(header, residual_channels=bad_residuals)
  )
  coordinate_map, loads = _program(system)
  with pytest.raises(DriverPreparationError, match="unknown-residual-target-port"):
    compile_driver_plan(system, coordinate_map, loads)

  bad_temperature_port = _new(
    PortBinding,
    port_id=header.ports[1].port_id,
    space_id=header.ports[1].space_id,
    mode=header.ports[1].mode,
    coefficient_map=FinalizedArray(np.zeros((2, 3), dtype=np.int64), dtype=np.int64),
  )
  system = _system_with_header(
    base,
    _header_with(header, ports=(header.ports[0], bad_temperature_port)),
  )
  coordinate_map, loads = _program(system)
  with pytest.raises(DriverPreparationError, match="operator-entity-count-mismatch"):
    compile_driver_plan(system, coordinate_map, loads)

  bad_displacement_port = _new(
    PortBinding,
    port_id=header.ports[0].port_id,
    space_id=header.ports[0].space_id,
    mode=header.ports[0].mode,
    coefficient_map=FinalizedArray(
      header.ports[0].coefficient_map.values + 99,
      dtype=np.int64,
    ),
  )
  system = _system_with_header(
    base,
    _header_with(header, ports=(bad_displacement_port, header.ports[1])),
  )
  coordinate_map, loads = _program(system)
  with pytest.raises(DriverPreparationError, match="gather-out-of-range"):
    compile_driver_plan(system, coordinate_map, loads)


def test_legacy_assembly_rejects_nonlegacy_plans() -> None:
  system = _system()
  coordinate_map, loads = _program(system)
  plan = compile_driver_plan(system, coordinate_map, loads)
  with pytest.raises(TypeError, match="legacy single-port plan"):
    assemble_internal_force(plan, ())
  with pytest.raises(TypeError, match="legacy single-port plan"):
    refill_tangent(plan, ())


def test_phase_field_four_channels_fill_full_multiport_coo() -> None:
  system = _phase_system()
  operator = system.operators[0]
  plan = compile_driver_plan(system, compile_constraint_map(system), ())
  assert plan.provenance.schema == "pyfem-v3-driver-assembly-plan-multiport-v1"
  assert plan.coo_entry_count == 144
  for accepted in (np.zeros((4, 1)), np.full((4, 1), 0.03)):
    evaluation = _evaluate_phase(
      operator,
      _unit_displacement(),
      _PHASE,
      accepted,
    )
    tangent = refill_tangent_channels(
      plan,
      (tuple(value.values for value in evaluation.jacobian_values),),
    )
    np.testing.assert_allclose(
      tangent.toarray(), _phase_dense(evaluation), rtol=0.0, atol=2.0e-14
    )
    full_residual, rhs, _jacobian = assemble_evaluation_balance(
      plan,
      (evaluation,),
      np.zeros(12),
    )
    expected_residual = np.concatenate(
      tuple(value.values[0] for value in evaluation.residual_values)
    )
    np.testing.assert_allclose(full_residual, expected_residual, rtol=0.0, atol=2.0e-14)
    np.testing.assert_allclose(rhs, -expected_residual, rtol=0.0, atol=2.0e-14)
