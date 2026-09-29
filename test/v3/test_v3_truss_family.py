# SPDX-License-Identifier: MIT

"""Direct compiler proof for the truss operator family behind the generic seam."""

from __future__ import annotations

import copy
import pickle
import sys
from dataclasses import replace

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

import pyfem.v3.compile.truss as truss_builder
from pyfem.v3.compile.continuum import (
  Q8ContinuumOperator,
  q8_reference_registry,
)
from pyfem.v3.compile.diagnostics import ModelCompilationError
from pyfem.v3.compile.system import compile_system
from pyfem.v3.compile.truss import (
  TRUSS_FORMULATION_KEY,
  TRUSS_MATERIAL_KEY,
  TRUSS_TOPOLOGY_KEY,
  TrussOperator,
  TrussPayload,
  truss_descriptor_metadata,
  truss_reference_registry,
)
from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.operator import (
  BalanceRole,
  ChannelRequest,
  OperatorEvaluationInput,
  ProgramSignalInput,
  StateLifetime,
)
from pyfem.v3.model.registry import RegistryDescriptor, RegistryKey
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
  normalize_model_spec,
)

_HUGE_INTEGER_ID = 10**5000
_MAX_HUGE_DIAGNOSTIC_LENGTH = 8192


def _source(label: str) -> SourceContext:
  return SourceContext(source=label)


def _model(
  *,
  coordinates: tuple[tuple[float, float], ...] = (
    (-10.0, 0.0),
    (10.0, 0.0),
    (0.0, 0.5),
  ),
  cells: tuple[tuple[str | int, tuple[int, ...]], ...] = (
    ("cell-1", (1, 3)),
    ("cell-2", (2, 3)),
  ),
  cell_refs: tuple[tuple[str, str | int], ...] | None = None,
  block_fields: dict[str, object] | None = None,
  field_components: tuple[str, ...] = ("x", "y"),
  field_location: str = "node",
  extra_fields: tuple[FieldSpec, ...] = (),
  parameters: tuple[tuple[str, object], ...] = (
    ("youngs_modulus", 5.0e6),
    ("area", 1.0),
  ),
  material_model: str = "uniaxial-linear-elastic",
  formulation: str = "total-lagrangian-truss",
  quadrature: str = "none",
) -> ModelSpec:
  nodes = tuple(
    NodeSpec(
      id=index,
      coordinates=point,
      source=_source(f"node:{index}"),
    )
    for index, point in enumerate(coordinates, start=1)
  )
  cell_specs = tuple(
    CellSpec(id=cell_id, node_ids=node_ids, source=_source(f"cell:{position}"))
    for position, (cell_id, node_ids) in enumerate(cells, start=1)
  )
  block_values = {
    "id": "bars",
    "reference_topology": "line",
    "topological_dimension": 1,
    "embedding_dimension": 2,
    "geometry_interpolation": "line2",
    "cells": cell_specs,
    "source": _source("cell-block"),
  }
  if block_fields is not None:
    block_values.update(block_fields)
  block = CellBlockSpec(**block_values)  # type: ignore[arg-type]
  field = FieldSpec(
    id="displacement",
    components=field_components,
    location=field_location,
    source=_source("field"),
  )
  material = MaterialSpec(
    id="steel",
    model=material_model,
    parameters=tuple(
      MaterialParameterSpec(name, value, _source(f"material:{name}"))
      for name, value in parameters
    ),
    source=_source("material"),
  )
  if cell_refs is None:
    cell_refs = tuple(("bars", cell_id) for cell_id, _ in cells)
  region = RegionSpec(
    id="domain",
    cell_refs=tuple(CellRef(block_id, cell_id) for block_id, cell_id in cell_refs),
    field_ids=(field.id,),
    material_id=material.id,
    formulation=formulation,
    quadrature=quadrature,
    source=_source("region"),
  )
  return ModelSpec(
    mesh=MeshSpec(nodes=nodes, cell_blocks=(block,), source=_source("mesh")),
    fields=(field, *extra_fields),
    materials=(material,),
    regions=(region,),
    source=_source("model"),
  )


def _q8_model(*, formulation: str = "small-strain-continuum") -> ModelSpec:
  coordinates = (
    (0.0, 0.0),
    (0.5, 0.0),
    (1.0, 0.0),
    (1.0, 0.5),
    (1.0, 1.0),
    (0.5, 1.0),
    (0.0, 1.0),
    (0.0, 0.5),
  )
  nodes = tuple(
    NodeSpec(id=index, coordinates=point, source=_source(f"node:{index}"))
    for index, point in enumerate(coordinates, start=1)
  )
  cell = CellSpec(id="cell", node_ids=tuple(range(1, 9)), source=_source("cell"))
  block = CellBlockSpec(
    id="q8-cells",
    reference_topology="quadrilateral",
    topological_dimension=2,
    embedding_dimension=2,
    geometry_interpolation="serendipity-quad8",
    cells=(cell,),
    source=_source("cell-block"),
  )
  field = FieldSpec(
    id="displacement",
    components=("x", "y"),
    location="node",
    source=_source("field"),
  )
  material = MaterialSpec(
    id="elastic",
    model="plane-stress-linear-elastic",
    parameters=(
      MaterialParameterSpec("youngs_modulus", 210.0e9, _source("material:E")),
      MaterialParameterSpec("poisson_ratio", 0.3, _source("material:nu")),
    ),
    source=_source("material"),
  )
  region = RegionSpec(
    id="domain",
    cell_refs=(CellRef("q8-cells", "cell"),),
    field_ids=(field.id,),
    material_id=material.id,
    formulation=formulation,
    quadrature="gauss-3x3",
    source=_source("region"),
  )
  return ModelSpec(
    mesh=MeshSpec(nodes=nodes, cell_blocks=(block,), source=_source("mesh")),
    fields=(field,),
    materials=(material,),
    regions=(region,),
    source=_source("model"),
  )


def _operator(model: ModelSpec | None = None) -> TrussOperator:
  system = compile_system(
    _model() if model is None else model,
    truss_reference_registry(),
  )
  operator = system.operators[0]
  assert isinstance(operator, TrussOperator)
  return operator


def _inputs(
  operator: TrussOperator,
  values: np.ndarray | None = None,
) -> OperatorEvaluationInput:
  if values is None:
    values = np.zeros((2, 4), dtype=np.float64)
  return OperatorEvaluationInput(
    port_values=(FinalizedArray(values, dtype=np.float64),),
    accepted_state=FinalizedArray(
      np.empty(operator.header.state_layout.row_shape),
      dtype=np.float64,
    ),
    signals=(),
    request=ChannelRequest(
      tuple(item.channel_id for item in operator.header.residual_channels),
      tuple(item.channel_id for item in operator.header.jacobian_channels),
    ),
  )


def _descriptor_with(
  key: RegistryKey,
  *,
  implementation_id: str = "changed-implementation",
  metadata: dict[str, object] | None = None,
  binding: object | None = None,
) -> RegistryDescriptor:
  reference = truss_reference_registry()[key]
  return RegistryDescriptor(
    kind=key[0],
    name=key[1],
    version=reference.version,
    implementation_id=implementation_id,
    metadata=truss_descriptor_metadata(*key) if metadata is None else metadata,
    binding=reference.binding if binding is None else binding,
  )


def _pickle_round_trip(value: object) -> object:
  return pickle.loads(pickle.dumps(value))


def test_dispatch_routes_each_family_and_preserves_landed_diagnostics() -> None:
  truss_system = compile_system(_model(), truss_reference_registry())
  assert isinstance(truss_system.operators[0], TrussOperator)
  continuum_system = compile_system(_q8_model(), q8_reference_registry())
  assert isinstance(continuum_system.operators[0], Q8ContinuumOperator)

  with pytest.raises(ModelCompilationError, match="incompatible-cell-block"):
    compile_system(_model(formulation="mystery"), truss_reference_registry())
  with pytest.raises(ModelCompilationError, match="incompatible-cell-block"):
    compile_system(
      _model(formulation="small-strain-continuum"),
      truss_reference_registry(),
    )
  with pytest.raises(ModelCompilationError, match="incompatible-formulation"):
    compile_system(_q8_model(formulation="mystery"), q8_reference_registry())

  truss_shaped_q8 = _q8_model(formulation="total-lagrangian-truss")
  with pytest.raises(ModelCompilationError, match="incompatible-cell-block"):
    compile_system(truss_shaped_q8, truss_reference_registry())


def test_unselected_family_descriptors_do_not_enter_compiled_identity() -> None:
  combined = {**q8_reference_registry(), **truss_reference_registry()}
  with_combined = compile_system(_model(), combined)
  with_truss_only = compile_system(_model(), truss_reference_registry())

  assert with_combined.content_fingerprint == with_truss_only.content_fingerprint
  assert len(with_combined.registry_snapshot.descriptors) == 3
  continuum = compile_system(_q8_model(), combined)
  assert (
    continuum.content_fingerprint
    == compile_system(
      _q8_model(),
      q8_reference_registry(),
    ).content_fingerprint
  )


def test_truss_system_compilation_uses_generic_spaces_ports_and_channels() -> None:
  system = compile_system(_model(), truss_reference_registry())
  assert tuple(space.space_id for space in system.spaces) == ("displacement",)
  np.testing.assert_array_equal(
    system.spaces[0].coefficient_map.values,
    np.arange(6, dtype=np.int64).reshape(3, 2),
  )
  assert len(system.entity_blocks) == 1
  np.testing.assert_array_equal(
    system.entity_blocks[0].incidence.values,
    [[0, 2], [1, 2]],
  )
  operator = system.operators[0]
  header = operator.header
  assert header.block_id == ("bars", "domain")
  assert header.entity_block_id == "bars"
  assert header.ports[0].space_id == system.spaces[0].space_id
  assert header.signal_ports == ()
  np.testing.assert_array_equal(
    header.ports[0].coefficient_map.values,
    [[0, 1, 4, 5], [2, 3, 4, 5]],
  )
  residual_channel = header.residual_channels[0]
  assert residual_channel.channel_id == "internal-force"
  assert residual_channel.balance_role is BalanceRole.INTERNAL
  assert not residual_channel.linear
  jacobian_channel = header.jacobian_channels[0]
  assert jacobian_channel.channel_id == "tangent"
  assert jacobian_channel.residual_channel_id == "internal-force"
  assert jacobian_channel.source_port_id == "displacement"
  assert not jacobian_channel.linear
  assert jacobian_channel.symmetric
  assert header.state_layout.slots == ()
  assert header.state_layout.row_shape == (2, 0)
  assert header.state_layout.lifetime is StateLifetime.ACCEPTED_TRIAL
  np.testing.assert_array_equal(
    header.state_layout.entity_offsets.values,
    [0, 0, 0],
  )
  result = operator.evaluate(_inputs(operator))
  assert result.residual_values[0].values.shape == (2, 4)
  assert result.jacobian_values[0].values.shape == (2, 4, 4)
  assert result.trial_state.values.shape == (2, 0)


def test_truss_compiler_matches_reference_geometry_and_parameters() -> None:
  system = compile_system(_model(), truss_reference_registry())
  operator = system.operators[0]
  assert isinstance(operator, TrussOperator)
  payload = operator.payload
  length = float(np.hypot(10.0, 0.5))
  np.testing.assert_allclose(
    payload.element_lengths.values,
    [length, length],
    rtol=0.0,
    atol=1.0e-15,
  )
  cos_alpha = 10.0 / length
  sin_alpha = 0.5 / length
  expected = np.array(
    [
      [cos_alpha, sin_alpha, 0.0, 0.0],
      [-sin_alpha, cos_alpha, 0.0, 0.0],
      [0.0, 0.0, cos_alpha, sin_alpha],
      [0.0, 0.0, -sin_alpha, cos_alpha],
    ],
    dtype=np.float64,
  )
  np.testing.assert_allclose(
    payload.element_rotations.values[0],
    expected,
    rtol=0.0,
    atol=1.0e-15,
  )
  reflected = payload.element_rotations.values[1]
  np.testing.assert_allclose(
    reflected[:2, :2],
    [[-cos_alpha, sin_alpha], [-sin_alpha, -cos_alpha]],
    rtol=0.0,
    atol=1.0e-15,
  )
  np.testing.assert_array_equal(payload.constitutive.values, [[5.0e6]])
  np.testing.assert_array_equal(payload.material_parameters.values, [[5.0e6, 1.0]])
  np.testing.assert_array_equal(
    payload.geometric_template.values,
    [
      [1.0, 0.0, -1.0, 0.0],
      [0.0, 1.0, 0.0, -1.0],
      [-1.0, 0.0, 1.0, 0.0],
      [0.0, -1.0, 0.0, 1.0],
    ],
  )
  snapshot = system.registry_snapshot
  identities = {
    (item.kind, item.name): item.implementation_id
    for item in operator.header.implementations
  }
  assert identities == {
    TRUSS_TOPOLOGY_KEY: "pyfem-v3-line2-geometry-v1",
    TRUSS_FORMULATION_KEY: "pyfem-v3-total-lagrangian-truss-v1",
    TRUSS_MATERIAL_KEY: "pyfem-v3-uniaxial-linear-elastic-v1",
  }
  registry = truss_reference_registry()
  assert (
    snapshot.resolve(*TRUSS_TOPOLOGY_KEY).binding
    is registry[TRUSS_TOPOLOGY_KEY].binding
  )
  registry.clear()
  assert snapshot.resolve(*TRUSS_MATERIAL_KEY).implementation_id == (
    "pyfem-v3-uniaxial-linear-elastic-v1"
  )


def test_truss_compiled_system_owns_identity_and_attribution() -> None:
  authored = _model()
  reordered = replace(
    authored,
    mesh=replace(authored.mesh, nodes=tuple(reversed(authored.mesh.nodes))),
  )
  registry = truss_reference_registry()
  first = compile_system(authored, registry)
  registry.clear()
  second = compile_system(reordered, truss_reference_registry())
  assert first.content_fingerprint == second.content_fingerprint
  assert first.instance_id != second.instance_id
  first_operator = first.operators[0]
  second_operator = second.operators[0]
  assert isinstance(first_operator, TrussOperator)
  assert isinstance(second_operator, TrussOperator)
  arrays = (
    first.point_blocks[0].reference_coordinates.values,
    first.entity_blocks[0].incidence.values,
    first.spaces[0].coefficient_map.values,
    first_operator.header.ports[0].coefficient_map.values,
    first_operator.payload.element_lengths.values,
    first_operator.payload.element_rotations.values,
  )
  for array in arrays:
    assert array.flags.owndata and not array.flags.writeable
    assert array.dtype.metadata is None
  assert not np.shares_memory(
    first_operator.payload.element_rotations.values,
    second_operator.payload.element_rotations.values,
  )
  assert first.source_for("cell", ("bars", "cell-1")).source == "cell:1"
  assert first.source_for("operator", ("bars", "domain")).source == "region"
  assert first.source_for("material_parameter", ("steel", "area")).source == (
    "material:area"
  )

  changed = replace(
    authored,
    materials=(
      replace(
        authored.materials[0],
        parameters=(
          replace(authored.materials[0].parameters[0], value=6.0e6),
          authored.materials[0].parameters[1],
        ),
      ),
    ),
  )
  assert (
    compile_system(changed, truss_reference_registry()).content_fingerprint
    != first.content_fingerprint
  )
  for carrier in (TrussOperator, TrussPayload):
    with pytest.raises(TypeError, match="constructed only by their compiler"):
      carrier()
  evaluation = first_operator.evaluate(_inputs(first_operator))
  trusted = (
    first,
    first.point_blocks[0],
    first.entity_blocks[0],
    first.spaces[0],
    first_operator,
    first_operator.payload,
    first_operator.header,
    first_operator.header.ports[0],
    first_operator.header.residual_channels[0],
    first_operator.header.jacobian_channels[0],
    first_operator.header.state_layout,
    evaluation,
  )
  for value in trusted:
    for reconstruct in (copy.copy, copy.deepcopy, _pickle_round_trip):
      with pytest.raises(TypeError, match="cannot be reconstructed"):
        reconstruct(value)


def test_caller_mutation_after_compile_cannot_change_any_compiled_meaning() -> None:
  authored = _model()
  compiled = compile_system(authored, truss_reference_registry())
  fingerprint = compiled.content_fingerprint
  lengths = compiled.operators[0].payload.element_lengths.values.copy()

  object.__setattr__(authored.mesh.nodes[0], "coordinates", (99.0, 99.0))
  object.__setattr__(authored.mesh.cell_blocks[0].cells[0], "node_ids", ())

  assert compiled.content_fingerprint == fingerprint
  operator = compiled.operators[0]
  assert isinstance(operator, TrussOperator)
  np.testing.assert_array_equal(operator.payload.element_lengths.values, lengths)


@pytest.mark.parametrize(
  ("model", "code"),
  [
    (
      _model(
        coordinates=((1.0, 1.0), (1.0, 1.0)),
        cells=(("cell-1", (1, 2)),),
      ),
      "degenerate-truss-geometry",
    ),
    (
      _model(
        coordinates=((-1.0e308, 0.0), (1.0e308, 0.0)),
        cells=(("cell-1", (1, 2)),),
      ),
      "degenerate-truss-geometry",
    ),
    (
      _model(
        coordinates=((1.0e10, 0.0), (1.0e10 + 1.0e-4, 0.0)),
        cells=(("cell-1", (1, 2)),),
      ),
      "degenerate-truss-geometry",
    ),
    (
      _model(cells=(("cell-1", (1, 2, 3)),)),
      "invalid-truss-arity",
    ),
    (
      _model(block_fields={"reference_topology": "triangle"}),
      "incompatible-cell-block",
    ),
    (
      _model(block_fields={"topological_dimension": 2}),
      "incompatible-cell-block",
    ),
    (
      _model(
        coordinates=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        block_fields={"embedding_dimension": 3},
      ),
      "incompatible-cell-block",
    ),
    (
      _model(block_fields={"geometry_interpolation": "line3"}),
      "incompatible-cell-block",
    ),
    (
      _model(field_components=("y", "x")),
      "incompatible-field-signature",
    ),
    (
      _model(field_components=("x", "y", "z")),
      "incompatible-field-signature",
    ),
    (
      _model(quadrature="gauss-1"),
      "incompatible-quadrature",
    ),
    (
      _model(material_model="plane-stress-linear-elastic"),
      "incompatible-material-model",
    ),
    (
      _model(parameters=(("youngs_modulus", 5.0e6),)),
      "invalid-material-parameter-schema",
    ),
    (
      _model(
        parameters=(
          ("youngs_modulus", 5.0e6),
          ("area", 1.0),
          ("density", 1.0),
        ),
      ),
      "invalid-material-parameter-schema",
    ),
    (
      _model(parameters=(("youngs_modulus", True), ("area", 1.0))),
      "invalid-material-parameter-type",
    ),
    (
      _model(parameters=(("youngs_modulus", 10**5000), ("area", 1.0))),
      "invalid-material-parameter-value",
    ),
    (
      _model(parameters=(("youngs_modulus", 0.0), ("area", 1.0))),
      "invalid-youngs-modulus",
    ),
    (
      _model(parameters=(("youngs_modulus", 5.0e6), ("area", -1.0))),
      "invalid-cross-section-area",
    ),
    (
      _model(
        extra_fields=(
          FieldSpec("temperature", ("temperature",), "node", _source("field:t")),
        ),
      ),
      "unsupported-truss-declaration-count",
    ),
    (
      _model(cell_refs=(("bars", "cell-1"),)),
      "incomplete-cell-membership",
    ),
  ],
)
def test_truss_incompatible_slices_fail_with_coded_diagnostics(
  model: ModelSpec,
  code: str,
) -> None:
  with pytest.raises(ModelCompilationError, match=code):
    compile_system(model, truss_reference_registry())


def test_truss_multiple_cell_membership_is_rejected() -> None:
  authored = _model()
  duplicate = replace(
    authored.regions[0],
    id="duplicate",
    source=_source("duplicate"),
  )
  with pytest.raises(ModelCompilationError, match="multiple-cell-membership"):
    compile_system(
      replace(authored, regions=(*authored.regions, duplicate)),
      truss_reference_registry(),
    )


def test_truss_formulation_check_holds_for_direct_builder_calls() -> None:
  normalized = normalize_model_spec(_model(formulation="small-strain-truss"))
  with pytest.raises(ModelCompilationError, match="incompatible-formulation"):
    truss_builder.select_model(normalized)


def test_non_node_field_location_is_rejected_before_carrier_allocation() -> None:
  authored = _model(field_location="cell")
  with pytest.raises(ModelCompilationError, match="unsupported-space-support"):
    compile_system(authored, truss_reference_registry())


def test_extreme_stiffness_and_length_scales_fail_closed() -> None:
  huge_modulus = _model(
    coordinates=((0.0, 0.0), (1.0e200, 0.0)),
    cells=(("cell-1", (1, 2)),),
    parameters=(("youngs_modulus", 1.0e200), ("area", 1.0)),
  )
  with pytest.raises(ModelCompilationError, match="non-finite-element-operator"):
    compile_system(huge_modulus, truss_reference_registry())
  unrepresentable = _model(
    parameters=(("youngs_modulus", 1.0e200), ("area", 1.0e200)),
  )
  with pytest.raises(ModelCompilationError, match="unrepresentable-material-law"):
    compile_system(unrepresentable, truss_reference_registry())
  tiny = _model(
    coordinates=((0.0, 0.0), (1.0e-320, 0.0)),
    cells=(("cell-1", (1, 2)),),
  )
  with pytest.raises(ModelCompilationError, match="degenerate-truss-geometry"):
    compile_system(tiny, truss_reference_registry())


@pytest.mark.parametrize(
  "missing_key",
  [TRUSS_TOPOLOGY_KEY, TRUSS_FORMULATION_KEY, TRUSS_MATERIAL_KEY],
)
def test_each_missing_registry_key_fails_closed(missing_key: RegistryKey) -> None:
  registry = truss_reference_registry()
  registry.pop(missing_key)
  with pytest.raises(ModelCompilationError, match="registry-capture-failed"):
    compile_system(_model(), registry)


@pytest.mark.parametrize(
  "key",
  [TRUSS_TOPOLOGY_KEY, TRUSS_FORMULATION_KEY, TRUSS_MATERIAL_KEY],
)
def test_descriptor_names_without_compatible_metadata_are_rejected(
  key: RegistryKey,
) -> None:
  registry = truss_reference_registry()
  metadata = truss_descriptor_metadata(*key)
  metadata["incompatible_change"] = True
  registry[key] = _descriptor_with(key, metadata=metadata)
  with pytest.raises(ModelCompilationError, match="incompatible-registry-descriptor"):
    compile_system(_model(), registry)


def _zero_geometry(coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  return np.zeros(len(coordinates)), np.zeros((len(coordinates), 2, 2))


def _short_geometry(coordinates: np.ndarray) -> np.ndarray:
  return np.zeros(len(coordinates))


def _raising_geometry(coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  del coordinates
  msg = "hostile geometry binding"
  raise RuntimeError(msg)


def _zero_kinematics(
  local_states: np.ndarray,
  lengths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  return np.zeros(len(local_states)), np.zeros((len(local_states), 4))


def _raising_kinematics(
  local_states: np.ndarray,
  lengths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  del local_states, lengths
  msg = "hostile kinematics binding"
  raise RuntimeError(msg)


def _wide_material(youngs_modulus: float, area: float) -> np.ndarray:
  del youngs_modulus, area
  return np.zeros((2, 2))


def _doubled_material(youngs_modulus: float, area: float) -> np.ndarray:
  return np.array([[2.0 * youngs_modulus * area]])


def _raising_material(youngs_modulus: float, area: float) -> np.ndarray:
  del youngs_modulus, area
  msg = "hostile material binding"
  raise RuntimeError(msg)


@pytest.mark.parametrize(
  ("key", "binding", "code"),
  [
    (TRUSS_TOPOLOGY_KEY, _zero_geometry, "incompatible-topology-binding-output"),
    (TRUSS_TOPOLOGY_KEY, _short_geometry, "invalid-topology-binding-output"),
    (TRUSS_TOPOLOGY_KEY, _raising_geometry, "topology-binding-failed"),
    (
      TRUSS_FORMULATION_KEY,
      _zero_kinematics,
      "incompatible-formulation-binding-output",
    ),
    (TRUSS_FORMULATION_KEY, _raising_kinematics, "formulation-binding-failed"),
    (TRUSS_MATERIAL_KEY, _wide_material, "invalid-material-binding-output"),
    (TRUSS_MATERIAL_KEY, _doubled_material, "incompatible-material-binding-output"),
    (TRUSS_MATERIAL_KEY, _raising_material, "material-binding-failed"),
  ],
)
def test_same_identity_spoofed_registry_bindings_fail_at_compile_boundary(
  key: RegistryKey,
  binding: object,
  code: str,
) -> None:
  registry = truss_reference_registry()
  registry[key] = _descriptor_with(key, binding=binding)
  with pytest.raises(ModelCompilationError, match=code):
    compile_system(_model(), registry)


def test_huge_integer_geometry_diagnostic_is_bounded_under_digit_limit() -> None:
  authored = _model(
    coordinates=((1.0, 1.0), (1.0, 1.0)),
    cells=((_HUGE_INTEGER_ID, (1, 2)),),
  )
  with pytest.raises(ModelCompilationError) as captured:
    compile_system(authored, truss_reference_registry())

  rendered = str(captured.value)
  assert len(rendered) < _MAX_HUGE_DIAGNOSTIC_LENGTH
  assert "<int sign=+ bits=" in rendered
  assert "degenerate-truss-geometry" in rendered


def test_truss_evaluation_validates_its_exact_inputs() -> None:
  operator = _operator()
  good = _inputs(operator)
  result = operator.evaluate(good)
  assert result.trial_state.values.shape == (2, 0)

  with pytest.raises(TypeError, match="exact immutable evaluation input"):
    operator.evaluate(object())  # type: ignore[arg-type]
  with pytest.raises(TypeError, match="exactly one displacement port batch"):
    operator.evaluate(replace(good, port_values=()))
  wrong_shape = OperatorEvaluationInput(
    port_values=(FinalizedArray(np.zeros((2, 8)), dtype=np.float64),),
    accepted_state=good.accepted_state,
    signals=(),
    request=good.request,
  )
  with pytest.raises(TypeError, match="finite metadata-free float64 batch"):
    operator.evaluate(wrong_shape)
  non_finite = OperatorEvaluationInput(
    port_values=(FinalizedArray(np.full((2, 4), np.nan), dtype=np.float64),),
    accepted_state=good.accepted_state,
    signals=(),
    request=good.request,
  )
  with pytest.raises(TypeError, match="finite metadata-free float64 batch"):
    operator.evaluate(non_finite)
  wrong_state = OperatorEvaluationInput(
    port_values=good.port_values,
    accepted_state=FinalizedArray(np.empty((2, 1)), dtype=np.float64),
    signals=(),
    request=good.request,
  )
  with pytest.raises(TypeError, match="zero-width state layout"):
    operator.evaluate(wrong_state)
  signaled = OperatorEvaluationInput(
    port_values=good.port_values,
    accepted_state=good.accepted_state,
    signals=(
      ProgramSignalInput(
        port_id="amplitude",
        values=FinalizedArray([1.0], dtype=np.float64),
        derivatives=(),
      ),
    ),
    request=good.request,
  )
  with pytest.raises(ValueError, match="does not accept program signal inputs"):
    operator.evaluate(signaled)
  duplicated = OperatorEvaluationInput(
    port_values=good.port_values,
    accepted_state=good.accepted_state,
    signals=(),
    request=ChannelRequest(("internal-force", "internal-force"), ()),
  )
  with pytest.raises(ValueError, match="unavailable or duplicate channel"):
    operator.evaluate(duplicated)
  unavailable = OperatorEvaluationInput(
    port_values=good.port_values,
    accepted_state=good.accepted_state,
    signals=(),
    request=ChannelRequest((), ("material-tangent",)),
  )
  with pytest.raises(ValueError, match="unavailable or duplicate channel"):
    operator.evaluate(unavailable)


def test_truss_evaluation_channel_gating_and_determinism() -> None:
  operator = _operator()
  values = np.array(
    [[0.02, -0.01, 0.3, -0.4], [0.05, 0.02, 0.3, -0.4]],
    dtype=np.float64,
  )
  residual_only = OperatorEvaluationInput(
    port_values=(FinalizedArray(values, dtype=np.float64),),
    accepted_state=FinalizedArray(np.empty((2, 0)), dtype=np.float64),
    signals=(),
    request=ChannelRequest(("internal-force",), ()),
  )
  result = operator.evaluate(residual_only)
  assert len(result.residual_values) == 1
  assert result.jacobian_values == ()
  jacobian_only = OperatorEvaluationInput(
    port_values=(FinalizedArray(values, dtype=np.float64),),
    accepted_state=FinalizedArray(np.empty((2, 0)), dtype=np.float64),
    signals=(),
    request=ChannelRequest((), ("tangent",)),
  )
  second = operator.evaluate(jacobian_only)
  assert second.residual_values == ()
  assert len(second.jacobian_values) == 1
  repeat = operator.evaluate(jacobian_only)
  np.testing.assert_array_equal(
    second.jacobian_values[0].values,
    repeat.jacobian_values[0].values,
  )
  tangent = second.jacobian_values[0].values
  np.testing.assert_allclose(
    tangent,
    tangent.transpose(0, 2, 1),
    rtol=0.0,
    atol=1.0e-8,
  )


def test_truss_evaluation_fails_closed_on_unrepresentable_response() -> None:
  operator = _operator()
  values = np.array(
    [[0.0, 0.0, 1.0e200, 0.0], [0.0, 0.0, 1.0e200, 0.0]],
    dtype=np.float64,
  )
  inputs = OperatorEvaluationInput(
    port_values=(FinalizedArray(values, dtype=np.float64),),
    accepted_state=FinalizedArray(np.empty((2, 0)), dtype=np.float64),
    signals=(),
    request=ChannelRequest(("internal-force",), ("tangent",)),
  )
  with pytest.raises(ValueError, match="not representable as finite float64"):
    operator.evaluate(inputs)
