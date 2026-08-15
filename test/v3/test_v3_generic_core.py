# SPDX-License-Identifier: MIT

"""Executable correctness proof for the isolated generic semantic IR."""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from pyfem.v3.core.generic import (
  AssemblyResult,
  AttributedJacobian,
  AttributedResidual,
  BalanceRole,
  ContinuumPayload,
  DirectionalSpringPayload,
  DiscreteSpace,
  OperatorBlock,
  OperatorOwner,
  PointEntityBlock,
  PointLoadPayload,
  PreparedExecution,
  assemble,
  compile_continuum_operator,
  compile_directional_spring_operator,
  compile_displacement_space,
  compile_point_entities,
  compile_point_load_operator,
  compile_program,
  compile_system,
  prepare_execution,
  quad4_continuum_descriptor,
  tria3_continuum_descriptor,
)
from pyfem.v3.fem.shapes import bilinear_quad4, linear_tria3
from pyfem.v3.materials.plane_stress import plane_stress_matrix

_Q4_INTEGER_OPERATOR = np.array(
  [
    [4, 1, -2, -1, -2, -1, 0, 1],
    [1, 4, 1, 0, -1, -2, -1, -2],
    [-2, 1, 4, -1, 0, -1, -2, 1],
    [-1, 0, -1, 4, 1, -2, 1, -2],
    [-2, -1, 0, 1, 4, 1, -2, -1],
    [-1, -2, -1, -2, 1, 4, 1, 0],
    [0, -1, -2, 1, -2, 1, 4, -1],
    [1, -2, 1, -2, -1, 0, -1, 4],
  ],
  dtype=np.float64,
)

_T3_INTEGER_OPERATOR = np.array(
  [
    [3, 1, -2, -1, -1, 0],
    [1, 3, 0, -1, -1, -2],
    [-2, 0, 2, 0, 0, 0],
    [-1, -1, 0, 1, 1, 0],
    [-1, -1, 0, 1, 1, 0],
    [0, -2, 0, 0, 0, 2],
  ],
  dtype=np.float64,
)

_EXPECTED_SOLUTION = np.array(
  [0.0, 0.0, 3.0, 2.0, -1.0, 2.0, 0.0, 0.0, 3.5, 0.0],
  dtype=np.float64,
)


@dataclass(frozen=True, slots=True)
class _Fixture:
  points: PointEntityBlock
  space: DiscreteSpace
  continuum: OperatorBlock[ContinuumPayload]
  spring: OperatorBlock[DirectionalSpringPayload]
  loads: OperatorBlock[PointLoadPayload]
  triangle: OperatorBlock[ContinuumPayload] | None
  prepared: PreparedExecution
  coordinate_input: np.ndarray
  connectivity_input: np.ndarray
  load_input: np.ndarray


def _fixture(
  *,
  reverse_model_blocks: bool = False,
  include_triangle: bool = False,
) -> _Fixture:
  coordinates = np.array(
    [
      [0.0, 0.0],
      [1.0, 0.0],
      [1.0, 1.0],
      [0.0, 1.0],
      [2.0, 0.0],
    ],
    dtype=np.float64,
  )
  points = compile_point_entities(
    block_id="points",
    entity_ids=("node-1", "node-2", "node-3", "node-4", "node-5"),
    source_ids=(
      "mesh:node-1",
      "mesh:node-2",
      "mesh:node-3",
      "mesh:node-4",
      "mesh:node-5",
    ),
    reference_coordinates=coordinates,
  )
  space = compile_displacement_space(
    space_id="displacement",
    support=points,
    components=("ux", "uy"),
  )

  connectivity = np.array([[0, 1, 2, 3]], dtype=np.int64)
  continuum = compile_continuum_operator(
    block_id="10-continuum",
    entity_ids=("cell-1",),
    source_ids=("mesh:cell-1",),
    connectivity=connectivity,
    space=space,
    descriptor=quad4_continuum_descriptor(),
    youngs_modulus=1.0,
    poisson_ratio=0.0,
  )
  spring = compile_directional_spring_operator(
    block_id="20-spring",
    entity_ids=("link-1",),
    source_ids=("mesh:link-1",),
    connectivity=np.array([[1, 4]], dtype=np.int64),
    space=space,
    component="ux",
    stiffness=np.array([2.0], dtype=np.float64),
  )

  triangle = None
  model_operators: tuple[
    OperatorBlock[ContinuumPayload] | OperatorBlock[DirectionalSpringPayload],
    ...,
  ] = (continuum, spring)
  if include_triangle:
    triangle = compile_continuum_operator(
      block_id="15-triangle",
      entity_ids=("cell-2",),
      source_ids=("mesh:cell-2",),
      connectivity=np.array([[0, 1, 3]], dtype=np.int64),
      space=space,
      descriptor=tria3_continuum_descriptor(),
      youngs_modulus=1.0,
      poisson_ratio=0.0,
    )
    model_operators = (continuum, triangle, spring)
  if reverse_model_blocks:
    model_operators = tuple(reversed(model_operators))

  load_values = np.array([0.25, 0.75], dtype=np.float64)
  loads = compile_point_load_operator(
    block_id="30-point-load",
    entity_ids=("load-quarter", "load-three-quarters"),
    source_ids=("program:load-quarter", "program:load-three-quarters"),
    point_incidence=np.array([[4], [4]], dtype=np.int64),
    space=space,
    component="ux",
    magnitudes=load_values,
  )
  system = compile_system(space=space, operators=model_operators)
  program = compile_program(space=space, operators=(loads,))
  prepared = prepare_execution(system, program)
  return _Fixture(
    points=points,
    space=space,
    continuum=continuum,
    spring=spring,
    loads=loads,
    triangle=triangle,
    prepared=prepared,
    coordinate_input=coordinates,
    connectivity_input=connectivity,
    load_input=load_values,
  )


def _residual_term(
  result: AssemblyResult,
  operator_id: str,
  entity_id: str,
) -> AttributedResidual:
  return next(
    term
    for term in result.residual_terms
    if term.operator_id == operator_id and term.entity_id == entity_id
  )


def _jacobian_term(
  result: AssemblyResult,
  operator_id: str,
) -> AttributedJacobian:
  return next(term for term in result.jacobian_terms if term.operator_id == operator_id)


def test_reused_low_level_kernels_have_independent_literal_checks() -> None:
  parent_center = np.array([[0.0, 0.0]], dtype=np.float64)
  q4_shapes, q4_gradients = bilinear_quad4(parent_center)
  np.testing.assert_array_equal(
    q4_shapes,
    np.array([[0.25, 0.25, 0.25, 0.25]], dtype=np.float64),
  )
  np.testing.assert_array_equal(
    q4_gradients,
    np.array(
      [[[-0.25, -0.25], [0.25, -0.25], [0.25, 0.25], [-0.25, 0.25]]],
      dtype=np.float64,
    ),
  )

  triangle_center = np.array([[1.0 / 3.0, 1.0 / 3.0]], dtype=np.float64)
  t3_shapes, t3_gradients = linear_tria3(triangle_center)
  np.testing.assert_allclose(
    t3_shapes,
    np.array([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]], dtype=np.float64),
    rtol=0.0,
    atol=2.0e-16,
  )
  np.testing.assert_array_equal(
    t3_gradients,
    np.array([[[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]]], dtype=np.float64),
  )
  np.testing.assert_array_equal(
    plane_stress_matrix(1.0, 0.0),
    np.diag(np.array([1.0, 1.0, 0.5], dtype=np.float64)),
  )


def test_three_unlike_blocks_share_typed_boundary_without_padding() -> None:
  fixture = _fixture()
  result = assemble(
    fixture.prepared,
    np.zeros(fixture.space.coefficient_count, dtype=np.float64),
  )

  assert fixture.continuum.header.entity_block.incidence.values.shape == (1, 4)
  assert fixture.spring.header.entity_block.incidence.values.shape == (1, 2)
  assert fixture.loads.header.entity_block.incidence.values.shape == (2, 1)
  assert fixture.continuum.header.ports[0].gather.values.shape == (1, 8)
  assert fixture.spring.header.ports[0].gather.values.shape == (1, 2)
  assert fixture.loads.header.ports[0].gather.values.shape == (2, 1)

  assert fixture.continuum.header.owner is OperatorOwner.MODEL
  assert fixture.spring.header.owner is OperatorOwner.MODEL
  assert fixture.loads.header.owner is OperatorOwner.PROGRAM
  assert all(
    block.header.state_layout.width == 0 for block in fixture.prepared.operators
  )
  assert tuple(
    block.header.state_layout.entity_count for block in fixture.prepared.operators
  ) == (1, 1, 2)
  assert result.evaluated_block_ids == (
    "10-continuum",
    "20-spring",
    "30-point-load",
  )

  np.testing.assert_allclose(
    _jacobian_term(result, "10-continuum").values.values,
    _Q4_INTEGER_OPERATOR / 8.0,
    rtol=0.0,
    atol=4.0e-16,
  )
  np.testing.assert_array_equal(
    _jacobian_term(result, "20-spring").values.values,
    np.array([[2.0, -2.0], [-2.0, 2.0]], dtype=np.float64),
  )
  expected_residual = np.zeros(10, dtype=np.float64)
  expected_residual[8] = -1.0
  np.testing.assert_array_equal(result.residual.values, expected_residual)

  load_terms = result.residual_terms[-2:]
  assert [term.entity_id for term in load_terms] == [
    "load-quarter",
    "load-three-quarters",
  ]
  assert [term.source_id for term in load_terms] == [
    "program:load-quarter",
    "program:load-three-quarters",
  ]
  assert all(term.channel.balance_role is BalanceRole.EXTERNAL for term in load_terms)
  np.testing.assert_array_equal(load_terms[0].values.values, np.array([-0.25]))
  np.testing.assert_array_equal(load_terms[1].values.values, np.array([-0.75]))


def test_factory_outputs_are_detached_owning_and_read_only() -> None:
  fixture = _fixture()
  fixture.coordinate_input[:] = 99.0
  fixture.connectivity_input[:] = 4
  fixture.load_input[:] = 9.0

  np.testing.assert_array_equal(
    fixture.points.reference_coordinates.values,
    np.array(
      [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 0.0]],
      dtype=np.float64,
    ),
  )
  np.testing.assert_array_equal(
    fixture.continuum.header.entity_block.incidence.values,
    np.array([[0, 1, 2, 3]], dtype=np.int64),
  )
  np.testing.assert_array_equal(
    fixture.loads.payload.magnitudes.values,
    np.array([0.25, 0.75], dtype=np.float64),
  )
  assert fixture.points.reference_coordinates.values.flags.owndata
  assert fixture.continuum.header.entity_block.incidence.values.flags.owndata
  assert fixture.loads.payload.magnitudes.values.flags.owndata
  with pytest.raises(ValueError):
    fixture.points.reference_coordinates.values[0, 0] = 1.0
  with pytest.raises(ValueError):
    fixture.continuum.header.ports[0].gather.values[0, 0] = 1
  with pytest.raises(ValueError):
    fixture.loads.payload.magnitudes.values[0] = 1.0


def test_exact_constrained_solution_and_attributed_balance() -> None:
  fixture = _fixture()
  zero = np.zeros(fixture.space.coefficient_count, dtype=np.float64)
  zero_result = assemble(fixture.prepared, zero)
  applied_force = -zero_result.residual.values
  free = np.array([2, 3, 4, 5, 8], dtype=np.int64)
  reduced_operator = zero_result.jacobian.values[np.ix_(free, free)]
  assert np.all(np.linalg.eigvalsh(reduced_operator) > 0.0)

  solution = np.zeros(10, dtype=np.float64)
  solution[free] = np.linalg.solve(reduced_operator, applied_force[free])
  np.testing.assert_allclose(
    solution,
    _EXPECTED_SOLUTION,
    rtol=0.0,
    atol=4.0e-15,
  )

  result = assemble(fixture.prepared, solution)
  np.testing.assert_allclose(result.residual.values[free], 0.0, rtol=0.0, atol=2.0e-15)
  expected_reaction = np.zeros(10, dtype=np.float64)
  expected_reaction[0] = -1.0
  np.testing.assert_allclose(
    result.residual.values,
    expected_reaction,
    rtol=0.0,
    atol=2.0e-15,
  )
  np.testing.assert_allclose(
    _residual_term(result, "10-continuum", "cell-1").values.values,
    np.array([-1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    rtol=0.0,
    atol=2.0e-15,
  )
  np.testing.assert_array_equal(
    _residual_term(result, "20-spring", "link-1").values.values,
    np.array([-1.0, 1.0]),
  )

  applied_x = -sum(
    float(term.values.values[0])
    for term in result.residual_terms
    if term.channel.balance_role is BalanceRole.EXTERNAL
  )
  reaction_x = float(result.residual.values[0] + result.residual.values[6])
  assert applied_x == 1.0
  assert reaction_x == pytest.approx(-1.0, rel=0.0, abs=3.0e-15)
  assert applied_x + reaction_x == pytest.approx(0.0, rel=0.0, abs=3.0e-15)

  reconstructed = np.zeros(10, dtype=np.float64)
  for term in result.residual_terms:
    for dof, value in zip(
      term.target_dofs.values,
      term.values.values,
      strict=True,
    ):
      reconstructed[dof] += value
  np.testing.assert_allclose(
    reconstructed,
    result.residual.values,
    rtol=0.0,
    atol=2.0e-15,
  )

  reconstructed_jacobian = np.zeros((10, 10), dtype=np.float64)
  for term in result.jacobian_terms:
    for local_row, target_dof in enumerate(term.target_dofs.values):
      for local_column, source_dof in enumerate(term.source_dofs.values):
        reconstructed_jacobian[target_dof, source_dof] += term.values.values[
          local_row,
          local_column,
        ]
  np.testing.assert_allclose(
    reconstructed_jacobian,
    result.jacobian.values,
    rtol=0.0,
    atol=2.0e-15,
  )


def test_block_order_is_canonical_and_attribution_is_stable() -> None:
  forward = _fixture(reverse_model_blocks=False)
  reverse = _fixture(reverse_model_blocks=True)
  coefficients = _EXPECTED_SOLUTION.copy()
  forward_result = assemble(forward.prepared, coefficients)
  reverse_result = assemble(reverse.prepared, coefficients)

  np.testing.assert_array_equal(
    forward_result.residual.values,
    reverse_result.residual.values,
  )
  np.testing.assert_array_equal(
    forward_result.jacobian.values,
    reverse_result.jacobian.values,
  )
  assert forward_result.evaluated_block_ids == reverse_result.evaluated_block_ids
  assert [
    (term.operator_id, term.entity_id, term.source_id, term.channel.channel_id)
    for term in forward_result.residual_terms
  ] == [
    (term.operator_id, term.entity_id, term.source_id, term.channel.channel_id)
    for term in reverse_result.residual_terms
  ]
  assert [
    (term.operator_id, term.entity_id, term.source_id, term.channel.channel_id)
    for term in forward_result.jacobian_terms
  ] == [
    (term.operator_id, term.entity_id, term.source_id, term.channel.channel_id)
    for term in reverse_result.jacobian_terms
  ]


def test_second_continuum_topology_changes_descriptor_instance_only() -> None:
  fixture = _fixture(include_triangle=True)
  assert fixture.triangle is not None
  assert type(fixture.triangle.payload) is type(fixture.continuum.payload)
  assert fixture.triangle.evaluator.evaluate is fixture.continuum.evaluator.evaluate
  assert (
    fixture.triangle.header.implementation_id
    == fixture.continuum.header.implementation_id
  )
  assert fixture.triangle.header.descriptor_id != fixture.continuum.header.descriptor_id
  assert fixture.triangle.header.entity_block.incidence.values.shape == (1, 3)
  assert fixture.triangle.header.ports[0].gather.values.shape == (1, 6)

  result = assemble(
    fixture.prepared,
    np.zeros(fixture.space.coefficient_count, dtype=np.float64),
  )
  assert result.evaluated_block_ids == (
    "10-continuum",
    "15-triangle",
    "20-spring",
    "30-point-load",
  )
  np.testing.assert_allclose(
    _jacobian_term(result, "15-triangle").values.values,
    _T3_INTEGER_OPERATOR / 4.0,
    rtol=0.0,
    atol=2.0e-16,
  )


def test_invalid_continuum_orientation_fails_at_factory_with_source() -> None:
  coordinates = np.array(
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    dtype=np.float64,
  )
  points = compile_point_entities(
    block_id="points",
    entity_ids=("n1", "n2", "n3", "n4"),
    source_ids=("mesh:n1", "mesh:n2", "mesh:n3", "mesh:n4"),
    reference_coordinates=coordinates,
  )
  space = compile_displacement_space(space_id="u", support=points)
  with pytest.raises(
    ValueError,
    match="invalid positive continuum geometry for cell-clockwise@mesh:clockwise",
  ):
    compile_continuum_operator(
      block_id="clockwise",
      entity_ids=("cell-clockwise",),
      source_ids=("mesh:clockwise",),
      connectivity=np.array([[0, 3, 2, 1]], dtype=np.int64),
      space=space,
      descriptor=quad4_continuum_descriptor(),
      youngs_modulus=1.0,
      poisson_ratio=0.0,
    )
