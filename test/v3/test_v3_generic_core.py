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
  AttributedTerm,
  BalanceRole,
  ContinuumPayload,
  DirectionalSpringPayload,
  DiscreteSpace,
  JacobianChannel,
  OperatorBlock,
  PointEntityBlock,
  PointLoadPayload,
  PreparedExecution,
  ResidualChannel,
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


def _fixture(
  *,
  reverse_model_blocks: bool = False,
  include_triangle: bool = False,
) -> _Fixture:
  coordinates = np.array(
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 0.0]],
    dtype=np.float64,
  )
  point_ids = tuple(f"node-{index}" for index in range(1, 6))
  points = compile_point_entities(
    entity_ids=point_ids,
    source_ids=tuple(f"mesh:{point_id}" for point_id in point_ids),
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
  model_operators = (continuum, spring)
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
  coordinates[:] = 99.0
  connectivity[:] = 4
  load_values[:] = 9.0
  return _Fixture(
    points=points,
    space=space,
    continuum=continuum,
    spring=spring,
    loads=loads,
    triangle=triangle,
    prepared=prepared,
  )


def _residual_term(
  result: AssemblyResult,
  operator_id: str,
  entity_id: str,
) -> AttributedTerm[ResidualChannel]:
  return next(
    term
    for term in result.residual_terms
    if term.operator_id == operator_id and term.entity_id == entity_id
  )


def _jacobian_term(
  result: AssemblyResult,
  operator_id: str,
) -> AttributedTerm[JacobianChannel]:
  return next(term for term in result.jacobian_terms if term.operator_id == operator_id)


def _assert_close(
  actual: np.ndarray, expected: np.ndarray | float, atol: float
) -> None:
  np.testing.assert_allclose(actual, expected, rtol=0.0, atol=atol)


def test_reused_constitutive_kernel_has_independent_literal_check() -> None:
  np.testing.assert_array_equal(
    plane_stress_matrix(1.0, 0.0),
    np.diag(np.array([1.0, 1.0, 0.5], dtype=np.float64)),
  )


def test_three_unlike_blocks_share_typed_boundary_without_padding() -> None:
  fixture = _fixture()
  result = assemble(
    fixture.prepared,
    np.zeros(len(fixture.space.coefficient_ids), dtype=np.float64),
  )

  blocks = fixture.continuum, fixture.spring, fixture.loads
  assert tuple(block.entity_block.incidence.values.shape for block in blocks) == (
    (1, 4),
    (1, 2),
    (2, 1),
  )
  assert tuple(block.ports[0].gather.values.shape for block in blocks) == (
    (1, 8),
    (1, 2),
    (2, 1),
  )
  assert all(block.local_state_width == 0 for block in fixture.prepared.operators)
  assert tuple(block.block_id for block in fixture.prepared.operators) == (
    "10-continuum",
    "20-spring",
    "30-point-load",
  )

  _assert_close(
    _jacobian_term(result, "10-continuum").values.values,
    _Q4_INTEGER_OPERATOR / 8.0,
    4.0e-16,
  )
  np.testing.assert_array_equal(
    _jacobian_term(result, "20-spring").values.values,
    np.array([[2.0, -2.0], [-2.0, 2.0]], dtype=np.float64),
  )
  expected_residual = np.zeros(10, dtype=np.float64)
  expected_residual[8] = -1.0
  np.testing.assert_array_equal(result.residual.values, expected_residual)

  load_terms = result.residual_terms[-2:]
  assert tuple((term.entity_id, term.source_id) for term in load_terms) == (
    ("load-quarter", "program:load-quarter"),
    ("load-three-quarters", "program:load-three-quarters"),
  )
  assert all(term.channel.balance_role is BalanceRole.EXTERNAL for term in load_terms)
  np.testing.assert_array_equal(load_terms[0].values.values, np.array([-0.25]))
  np.testing.assert_array_equal(load_terms[1].values.values, np.array([-0.75]))


def test_factory_outputs_are_detached_owning_and_read_only() -> None:
  fixture = _fixture()
  compiled_arrays = (
    fixture.points.reference_coordinates.values,
    fixture.continuum.entity_block.incidence.values,
    fixture.loads.payload.magnitudes.values,
  )
  expected_arrays = (
    np.array([[0, 0], [1, 0], [1, 1], [0, 1], [2, 0]], dtype=np.float64),
    np.array([[0, 1, 2, 3]], dtype=np.int64),
    np.array([0.25, 0.75], dtype=np.float64),
  )
  for actual, expected in zip(compiled_arrays, expected_arrays, strict=True):
    np.testing.assert_array_equal(actual, expected)
    assert actual.flags.owndata and actual.dtype.metadata is None
  with pytest.raises(ValueError):
    fixture.points.reference_coordinates.values[0, 0] = 1.0
  with pytest.raises(ValueError):
    fixture.continuum.ports[0].gather.values[0, 0] = 1
  with pytest.raises(ValueError):
    fixture.loads.payload.magnitudes.values[0] = 1.0


def test_exact_constrained_solution_and_attributed_balance() -> None:
  fixture = _fixture()
  zero = np.zeros(len(fixture.space.coefficient_ids), dtype=np.float64)
  zero_result = assemble(fixture.prepared, zero)
  applied_force = -zero_result.residual.values
  free = np.array([2, 3, 4, 5, 8], dtype=np.int64)
  reduced_operator = zero_result.jacobian.values[np.ix_(free, free)]
  assert np.all(np.linalg.eigvalsh(reduced_operator) > 0.0)

  solution = np.zeros(10, dtype=np.float64)
  solution[free] = np.linalg.solve(reduced_operator, applied_force[free])
  _assert_close(
    solution,
    _EXPECTED_SOLUTION,
    4.0e-15,
  )

  result = assemble(fixture.prepared, solution)
  _assert_close(result.residual.values[free], 0.0, 2.0e-15)
  expected_reaction = np.zeros(10, dtype=np.float64)
  expected_reaction[0] = -1.0
  _assert_close(
    result.residual.values,
    expected_reaction,
    2.0e-15,
  )
  _assert_close(
    _residual_term(result, "10-continuum", "cell-1").values.values,
    np.array([-1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    2.0e-15,
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
    np.add.at(reconstructed, term.port_dofs[0].values, term.values.values)
  _assert_close(
    reconstructed,
    result.residual.values,
    2.0e-15,
  )

  reconstructed_jacobian = np.zeros((10, 10), dtype=np.float64)
  for term in result.jacobian_terms:
    rows, columns = (dofs.values for dofs in term.port_dofs)
    reconstructed_jacobian[np.ix_(rows, columns)] += term.values.values
  _assert_close(
    reconstructed_jacobian,
    result.jacobian.values,
    2.0e-15,
  )


def test_block_order_is_canonical_and_attribution_is_stable() -> None:
  forward = _fixture(reverse_model_blocks=False)
  reverse = _fixture(reverse_model_blocks=True)
  coefficients = _EXPECTED_SOLUTION.copy()
  forward_result = assemble(forward.prepared, coefficients)
  reverse_result = assemble(reverse.prepared, coefficients)

  for name in ("residual", "jacobian"):
    np.testing.assert_array_equal(
      getattr(forward_result, name).values,
      getattr(reverse_result, name).values,
    )
  assert tuple(block.block_id for block in forward.prepared.operators) == tuple(
    block.block_id for block in reverse.prepared.operators
  )
  forward_terms = (*forward_result.residual_terms, *forward_result.jacobian_terms)
  reverse_terms = (*reverse_result.residual_terms, *reverse_result.jacobian_terms)
  assert [
    (term.operator_id, term.entity_id, term.source_id, term.channel.channel_id)
    for term in forward_terms
  ] == [
    (term.operator_id, term.entity_id, term.source_id, term.channel.channel_id)
    for term in reverse_terms
  ]


def test_second_continuum_topology_changes_descriptor_instance_only() -> None:
  fixture = _fixture(include_triangle=True)
  assert fixture.triangle is not None
  assert fixture.triangle.evaluator is fixture.continuum.evaluator
  assert fixture.triangle.entity_block.incidence.values.shape == (1, 3)
  assert fixture.triangle.ports[0].gather.values.shape == (1, 6)

  result = assemble(
    fixture.prepared,
    np.zeros(len(fixture.space.coefficient_ids), dtype=np.float64),
  )
  assert tuple(block.block_id for block in fixture.prepared.operators) == (
    "10-continuum",
    "15-triangle",
    "20-spring",
    "30-point-load",
  )
  _assert_close(
    _jacobian_term(result, "15-triangle").values.values,
    _T3_INTEGER_OPERATOR / 4.0,
    2.0e-16,
  )


def test_invalid_continuum_orientation_fails_at_factory_with_source() -> None:
  fixture = _fixture()
  with pytest.raises(
    ValueError,
    match="invalid positive continuum geometry for cell-clockwise@mesh:clockwise",
  ):
    compile_continuum_operator(
      block_id="clockwise",
      entity_ids=("cell-clockwise",),
      source_ids=("mesh:clockwise",),
      connectivity=np.array([[0, 3, 2, 1]], dtype=np.int64),
      space=fixture.space,
      descriptor=quad4_continuum_descriptor(),
      youngs_modulus=1.0,
      poisson_ratio=0.0,
    )
