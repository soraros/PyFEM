# SPDX-License-Identifier: MIT

"""Direct compiler proof for the unexported generic Q8 system boundary."""

from __future__ import annotations

import sys
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

import pyfem.v3.compile.system as system_compiler
from pyfem.v3.compile.continuum import (
  Q8_FORMULATION_KEY,
  Q8_MATERIAL_KEY,
  Q8_QUADRATURE_KEY,
  Q8_TOPOLOGY_KEY,
  Q8ContinuumOperator,
  q8_descriptor_metadata,
  q8_reference_registry,
)
from pyfem.v3.compile.diagnostics import ModelCompilationError
from pyfem.v3.compile.system import compile_discrete_spaces, compile_system
from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.operator import (
  BalanceRole,
  ChannelRequest,
  OperatorEvaluationInput,
  OperatorHeader,
  ProgramSignalInput,
  SignalDerivativeInput,
  StateLifetime,
)
from pyfem.v3.model.registry import RegistryDescriptor
from pyfem.v3.model.system import CompiledSystem, PointEntityBlock
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

_UNIT_COORDINATES = (
  (0.0, 0.0),
  (0.5, 0.0),
  (1.0, 0.0),
  (1.0, 0.5),
  (1.0, 1.0),
  (0.5, 1.0),
  (0.0, 1.0),
  (0.0, 0.5),
)


def _source(label: str) -> SourceContext:
  return SourceContext(source=label)


def _model(
  *,
  scale: float = 1.0,
  offset: tuple[float, float] = (0.0, 0.0),
  local_order: tuple[int, ...] = tuple(range(8)),
  coordinates: tuple[tuple[float, float], ...] | None = None,
  cell_id: str | int = "cell-1",
) -> ModelSpec:
  raw_coordinates = _UNIT_COORDINATES if coordinates is None else coordinates
  nodes = tuple(
    NodeSpec(
      id=index + 1,
      coordinates=(
        offset[0] + scale * point[0],
        offset[1] + scale * point[1],
      ),
      source=_source(f"node-source-{index + 1}"),
    )
    for index, point in enumerate(raw_coordinates)
  )
  cell = CellSpec(
    id=cell_id,
    node_ids=tuple(nodes[index].id for index in local_order),
    source=_source("cell-source"),
  )
  block = CellBlockSpec(
    id="cells",
    reference_topology="quadrilateral",
    topological_dimension=2,
    embedding_dimension=2,
    geometry_interpolation="serendipity-quad8",
    cells=(cell,),
    source=_source("block-source"),
  )
  field = FieldSpec(
    id="displacement",
    components=("x", "y"),
    location="node",
    source=_source("field-source"),
  )
  material = MaterialSpec(
    id="elastic",
    model="plane-stress-linear-elastic",
    parameters=(
      MaterialParameterSpec("youngs_modulus", 1.0, _source("material:E")),
      MaterialParameterSpec("poisson_ratio", 0.0, _source("material:nu")),
    ),
    source=_source("material-source"),
  )
  region = RegionSpec(
    id="domain",
    cell_refs=(CellRef(block.id, cell.id),),
    field_ids=(field.id,),
    material_id=material.id,
    formulation="small-strain-continuum",
    quadrature="gauss-3x3",
    source=_source("region-source"),
  )
  return ModelSpec(
    mesh=MeshSpec(nodes=(nodes), cell_blocks=(block,), source=_source("mesh-source")),
    fields=(field,),
    materials=(material,),
    regions=(region,),
    source=_source("model-source"),
  )


def _operator(model: ModelSpec | None = None) -> Q8ContinuumOperator:
  system = compile_system(_model() if model is None else model, q8_reference_registry())
  operator = system.operators[0]
  assert isinstance(operator, Q8ContinuumOperator)
  return operator


def _inputs(
  operator: Q8ContinuumOperator,
  values: np.ndarray | None = None,
) -> OperatorEvaluationInput:
  if values is None:
    values = np.zeros((1, 16), dtype=np.float64)
  return OperatorEvaluationInput(
    port_values=(FinalizedArray(values, dtype=np.float64),),
    accepted_state=FinalizedArray(np.empty((1, 0)), dtype=np.float64),
    signals=(),
    request=ChannelRequest(
      tuple(item.channel_id for item in operator.header.residual_channels),
      tuple(item.channel_id for item in operator.header.jacobian_channels),
    ),
  )


def _independent_shapes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  xi = points[:, 0]
  eta = points[:, 1]
  values = np.stack(
    (
      -Fraction(1, 4) * (1 - xi) * (1 - eta) * (1 + xi + eta),
      Fraction(1, 2) * (1 - xi) * (1 + xi) * (1 - eta),
      -Fraction(1, 4) * (1 + xi) * (1 - eta) * (1 - xi + eta),
      Fraction(1, 2) * (1 + xi) * (1 + eta) * (1 - eta),
      -Fraction(1, 4) * (1 + xi) * (1 + eta) * (1 - xi - eta),
      Fraction(1, 2) * (1 - xi) * (1 + xi) * (1 + eta),
      -Fraction(1, 4) * (1 - xi) * (1 + eta) * (1 + xi - eta),
      Fraction(1, 2) * (1 - xi) * (1 + eta) * (1 - eta),
    ),
    axis=1,
  ).astype(np.float64)
  dxi = np.stack(
    (
      -Fraction(1, 4) * (-1 + eta) * (2 * xi + eta),
      xi * (-1 + eta),
      Fraction(1, 4) * (-1 + eta) * (-2 * xi + eta),
      -Fraction(1, 2) * (1 + eta) * (-1 + eta),
      Fraction(1, 4) * (1 + eta) * (2 * xi + eta),
      -xi * (1 + eta),
      -Fraction(1, 4) * (1 + eta) * (-2 * xi + eta),
      Fraction(1, 2) * (1 + eta) * (-1 + eta),
    ),
    axis=1,
  ).astype(np.float64)
  deta = np.stack(
    (
      -Fraction(1, 4) * (-1 + xi) * (xi + 2 * eta),
      Fraction(1, 2) * (1 + xi) * (-1 + xi),
      Fraction(1, 4) * (1 + xi) * (-xi + 2 * eta),
      -eta * (1 + xi),
      Fraction(1, 4) * (1 + xi) * (xi + 2 * eta),
      -Fraction(1, 2) * (1 + xi) * (-1 + xi),
      -Fraction(1, 4) * (-1 + xi) * (-xi + 2 * eta),
      eta * (-1 + xi),
    ),
    axis=1,
  ).astype(np.float64)
  return values, np.stack((dxi, deta), axis=2)


def _center_quadrature(order: int) -> tuple[np.ndarray, np.ndarray]:
  del order
  return np.zeros((9, 2), dtype=np.float64), np.full(9, 4.0 / 9.0)


def _padded_interpolation(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  values = np.zeros((len(points), 8), dtype=np.float64)
  values[:, :4] = 0.25
  return values, np.zeros((len(points), 8, 2), dtype=np.float64)


def _zero_kinematics(gradients: np.ndarray) -> np.ndarray:
  return np.zeros((*gradients.shape[:2], 3, 2 * gradients.shape[2]))


def _zero_constitutive(youngs_modulus: float, poisson_ratio: float) -> np.ndarray:
  del youngs_modulus, poisson_ratio
  return np.zeros((3, 3), dtype=np.float64)


def test_direct_q8_system_compilation_uses_generic_spaces_ports_and_channels(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  calls = 0
  original = system_compiler.normalize_model_spec

  def counted(spec: ModelSpec) -> ModelSpec:
    nonlocal calls
    calls += 1
    return original(spec)

  monkeypatch.setattr(system_compiler, "normalize_model_spec", counted)
  system = system_compiler.compile_system(_model(), q8_reference_registry())
  assert calls == 1
  assert not hasattr(system, "space")
  assert tuple(space.space_id for space in system.spaces) == ("displacement",)
  np.testing.assert_array_equal(
    system.spaces[0].coefficient_map.values,
    np.arange(16, dtype=np.int64).reshape(8, 2),
  )
  operator = system.operators[0]
  header = operator.header
  assert header.ports[0].space_id == system.spaces[0].space_id
  assert header.signal_ports == ()
  np.testing.assert_array_equal(
    header.ports[0].coefficient_map.values,
    np.arange(16, dtype=np.int64).reshape(1, 16),
  )
  assert header.residual_channels[0].balance_role is BalanceRole.INTERNAL
  assert header.jacobian_channels[0].source_port_id == "displacement"
  assert header.state_layout.slots == ()
  assert header.state_layout.row_shape == (1, 0)
  assert header.state_layout.lifetime is StateLifetime.ACCEPTED_TRIAL
  np.testing.assert_array_equal(header.state_layout.entity_offsets.values, [0, 0])
  signal = ProgramSignalInput(
    port_id="amplitude",
    values=FinalizedArray([2.0], dtype=np.float64),
    derivatives=(
      SignalDerivativeInput(
        "time",
        FinalizedArray([3.0], dtype=np.float64),
      ),
    ),
  )
  assert signal.derivatives[0].coordinate_id == "time"
  result = operator.evaluate(_inputs(operator))
  assert result.residual_values[0].values.shape == (1, 16)
  assert result.jacobian_values[0].values.shape == (1, 16, 16)
  assert result.trial_state.values.shape == (1, 0)


def test_q8_compiler_matches_hard_coded_shape_gradient_and_quadrature_oracle() -> None:
  operator = _operator()
  payload = operator.payload
  abscissa = 0.7745966692414834
  expected_points = np.array(
    [
      [-abscissa, -abscissa],
      [-abscissa, 0.0],
      [-abscissa, abscissa],
      [0.0, -abscissa],
      [0.0, 0.0],
      [0.0, abscissa],
      [abscissa, -abscissa],
      [abscissa, 0.0],
      [abscissa, abscissa],
    ],
    dtype=np.float64,
  )
  fifth = float(Fraction(5, 9))
  eighth = float(Fraction(8, 9))
  expected_weights = np.array(
    [
      fifth * fifth,
      fifth * eighth,
      fifth * fifth,
      eighth * fifth,
      eighth * eighth,
      eighth * fifth,
      fifth * fifth,
      fifth * eighth,
      fifth * fifth,
    ],
    dtype=np.float64,
  )
  np.testing.assert_allclose(
    payload.quadrature_points.values,
    expected_points,
    rtol=0.0,
    atol=2.0e-16,
  )
  np.testing.assert_allclose(
    payload.quadrature_weights.values,
    expected_weights,
    rtol=0.0,
    atol=1.1e-15,
  )
  expected_values, expected_gradients = _independent_shapes(expected_points)
  np.testing.assert_allclose(
    payload.shape_values.values,
    expected_values,
    rtol=0.0,
    atol=3.0e-16,
  )
  np.testing.assert_allclose(
    payload.parent_gradients.values,
    expected_gradients,
    rtol=0.0,
    atol=3.0e-16,
  )
  center_values, center_gradients = (
    compile_system(_model(), q8_reference_registry())
    .registry_snapshot.resolve(*Q8_TOPOLOGY_KEY)
    .binding(np.array([[0.0, 0.0]], dtype=np.float64))
  )
  np.testing.assert_array_equal(
    center_values,
    [[-0.25, 0.5, -0.25, 0.5, -0.25, 0.5, -0.25, 0.5]],
  )
  np.testing.assert_array_equal(
    center_gradients,
    [
      [
        [0.0, 0.0],
        [0.0, -0.5],
        [0.0, 0.0],
        [0.5, 0.0],
        [0.0, 0.0],
        [0.0, 0.5],
        [0.0, 0.0],
        [-0.5, 0.0],
      ]
    ],
  )


@pytest.mark.parametrize(
  ("key", "binding", "code"),
  [
    (Q8_QUADRATURE_KEY, _center_quadrature, "invalid-quadrature-binding-output"),
    (Q8_TOPOLOGY_KEY, _padded_interpolation, "invalid-topology-binding-output"),
    (
      Q8_FORMULATION_KEY,
      _zero_kinematics,
      "incompatible-formulation-binding-output",
    ),
    (
      Q8_MATERIAL_KEY,
      _zero_constitutive,
      "incompatible-material-binding-output",
    ),
  ],
)
def test_same_identity_spoofed_registry_bindings_fail_at_compile_boundary(
  key: tuple[str, str],
  binding: object,
  code: str,
) -> None:
  registry = q8_reference_registry()
  reference = registry[key]
  registry[key] = RegistryDescriptor(
    kind=reference.kind,
    name=reference.name,
    version=reference.version,
    implementation_id=reference.implementation_id,
    metadata=q8_descriptor_metadata(*key),
    binding=binding,
  )
  with pytest.raises(ModelCompilationError, match=code):
    compile_system(_model(), registry)


def test_multiple_spaces_have_disjoint_native_coefficient_maps() -> None:
  points = compile_system(_model(), q8_reference_registry()).point_blocks[0]
  mechanical = FieldSpec("mechanical", ("x", "y"), "node")
  thermal = FieldSpec("thermal", ("temperature",), "node")
  nonstandard = FieldSpec("ordered", ("y", "x", "rotation"), "node")
  cases = (
    ((mechanical,), ((0, 16),)),
    ((thermal,), ((0, 8),)),
    ((mechanical, thermal), ((0, 16), (16, 24))),
    ((nonstandard,), ((0, 24),)),
  )
  for fields, expected_ranges in cases:
    spaces = compile_discrete_spaces(points, fields)
    assert tuple(space.coefficient_range for space in spaces) == expected_ranges
    allocated = [
      set(int(value) for value in space.coefficient_map.values.flat) for space in spaces
    ]
    assert all(
      allocated[left].isdisjoint(allocated[right])
      for left in range(len(allocated))
      for right in range(left + 1, len(allocated))
    )
  ordered = compile_discrete_spaces(points, (nonstandard,))[0]
  assert ordered.components == ("y", "x", "rotation")
  assert ordered.coefficient_ids[:3] == (
    ("ordered", 1, "y"),
    ("ordered", 1, "x"),
    ("ordered", 1, "rotation"),
  )
  unsupported = FieldSpec(
    "cell-temperature",
    ("temperature",),
    "cell",
    source=_source("cell-field-source"),
  )
  with pytest.raises(ModelCompilationError, match="unsupported-space-support") as error:
    compile_discrete_spaces(points, (unsupported,))
  assert "cell-field-source" in str(error.value)
  authored = _model()
  with pytest.raises(ModelCompilationError, match="unsupported-space-support"):
    compile_system(
      replace(authored, fields=(*authored.fields, unsupported)),
      q8_reference_registry(),
    )


def test_compiled_system_owns_metadata_free_arrays_identity_and_attribution() -> None:
  authored = _model()
  reordered = replace(
    authored,
    mesh=replace(authored.mesh, nodes=tuple(reversed(authored.mesh.nodes))),
  )
  registry = q8_reference_registry()
  first = compile_system(authored, registry)
  registry.clear()
  second = compile_system(reordered, q8_reference_registry())
  assert first.content_fingerprint == second.content_fingerprint
  assert first.instance_id != second.instance_id
  assert first.registry_snapshot.resolve(*Q8_TOPOLOGY_KEY).implementation_id == (
    "pyfem-v3-serendipity-quad8-v1"
  )
  first_operator = first.operators[0]
  second_operator = second.operators[0]
  assert isinstance(first_operator, Q8ContinuumOperator)
  assert isinstance(second_operator, Q8ContinuumOperator)
  arrays = (
    first.point_blocks[0].reference_coordinates.values,
    first.entity_blocks[0].incidence.values,
    first.spaces[0].coefficient_map.values,
    first_operator.header.ports[0].coefficient_map.values,
    first_operator.payload.shape_values.values,
    first_operator.payload.normalized_strain_displacement.values,
  )
  for array in arrays:
    assert array.flags.owndata and not array.flags.writeable
    assert array.dtype.metadata is None
  assert not np.shares_memory(
    first_operator.payload.shape_values.values,
    second_operator.payload.shape_values.values,
  )
  assert first.source_for("cell", ("cells", "cell-1")).source == "cell-source"
  assert first.source_for("operator", ("cells", "domain")).source == "region-source"
  assert (
    first.source_for("material_parameter", ("elastic", "youngs_modulus")).source
    == "material:E"
  )
  with pytest.raises(KeyError, match="exact semantic identity"):
    first.source_for("node", True)

  material = authored.materials[0]
  changed_parameter = replace(
    material.parameters[0],
    source=_source("changed-material:E"),
  )
  changed_material = replace(
    material,
    parameters=(changed_parameter, material.parameters[1]),
  )
  changed_source = compile_system(
    replace(authored, materials=(changed_material,)),
    q8_reference_registry(),
  )
  assert changed_source.content_fingerprint != first.content_fingerprint
  for carrier in (
    CompiledSystem,
    PointEntityBlock,
    OperatorHeader,
    Q8ContinuumOperator,
  ):
    with pytest.raises(TypeError, match="constructed only by their compiler"):
      carrier()


def test_q8_geometry_classification_is_scale_and_translation_stable() -> None:
  models = (
    _model(scale=1.0),
    _model(scale=2.0),
    _model(scale=1.0e-200),
    _model(scale=1.0e200),
    _model(scale=1.0e90, offset=(1.0e100, -1.0e100)),
  )
  operators = tuple(_operator(model) for model in models)
  baseline = operators[0].evaluate(_inputs(operators[0])).jacobian_values[0].values
  for operator in operators[1:]:
    result = operator.evaluate(_inputs(operator))
    np.testing.assert_allclose(
      result.jacobian_values[0].values,
      baseline,
      rtol=0.0,
      atol=3.0e-15,
    )
  unit, doubled = operators[:2]
  np.testing.assert_allclose(
    doubled.payload.physical_gradients().values,
    0.5 * unit.payload.physical_gradients().values,
    rtol=0.0,
    atol=2.0e-15,
  )
  np.testing.assert_allclose(
    doubled.payload.physical_integration_weights().values,
    4.0 * unit.payload.physical_integration_weights().values,
    rtol=0.0,
    atol=2.0e-15,
  )
  affine_values = np.array(
    [[x, 0.0] for x, _ in _UNIT_COORDINATES], dtype=np.float64
  ).reshape(16)
  strains = np.einsum(
    "pai,i->pa",
    unit.payload.physical_strain_displacement().values[0],
    affine_values,
  )
  np.testing.assert_allclose(
    strains,
    np.tile([1.0, 0.0, 0.0], (9, 1)),
    rtol=0.0,
    atol=8.0e-16,
  )


@pytest.mark.parametrize(
  ("model", "code"),
  [
    (
      _model(local_order=(0, 7, 6, 5, 4, 3, 2, 1)),
      "inverted-reference-geometry",
    ),
    (
      _model(local_order=(0, 1, 4, 3, 2, 5, 6, 7)),
      "sign-changing-reference-geometry",
    ),
    (
      _model(
        coordinates=tuple((x, y * 1.0e-14) for x, y in _UNIT_COORDINATES),
        cell_id=10**5000,
      ),
      "near-singular-reference-geometry",
    ),
  ],
)
def test_q8_invalid_orientation_and_relative_singularity_fail_at_compile_boundary(
  model: ModelSpec,
  code: str,
) -> None:
  with pytest.raises(ModelCompilationError, match=code) as captured:
    compile_system(model, q8_reference_registry())
  rendered = str(captured.value)
  assert "cell-source" in rendered
  assert len(rendered) < 8192
