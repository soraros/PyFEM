# SPDX-License-Identifier: MIT

"""Correctness matrix for the frozen normalized Q8 model compiler."""

from __future__ import annotations

import sys
from dataclasses import FrozenInstanceError, fields, replace

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from pyfem.v3.compile import (
  Q8_FORMULATION_KEY,
  Q8_MATERIAL_KEY,
  Q8_QUADRATURE_KEY,
  Q8_TOPOLOGY_KEY,
  ModelCompilationError,
  ModelCompilationPolicy,
  compile_model,
  q8_descriptor_metadata,
  q8_reference_registry,
)
from pyfem.v3.fem.kinematics import strain_displacement
from pyfem.v3.fem.quadrature import gauss_tensor_product_2d
from pyfem.v3.fem.shapes import serendipity_quad8
from pyfem.v3.materials.plane_stress import plane_stress_matrix
from pyfem.v3.model import CompiledModel, FinalizedArray, RegistryDescriptor
from pyfem.v3.spec import (
  CellBlockSpec,
  CellRef,
  CellSpec,
  FieldSpec,
  MaterialParameterSpec,
  MaterialSpec,
  MeshSpec,
  ModelSpec,
  ModelSpecValidationError,
  NodeSpec,
  RegionSpec,
  SourceContext,
  normalize_model_spec,
)

_HUGE_INTEGER_ID = 10**5000
_MAX_HUGE_DIAGNOSTIC_LENGTH = 8192


def _source(label: str) -> SourceContext:
  return SourceContext(source=label)


def _material(
  *,
  material_id: str | int = "steel",
  model: str = "plane-stress-linear-elastic",
  parameters: tuple[MaterialParameterSpec, ...] | None = None,
) -> MaterialSpec:
  if parameters is None:
    parameters = (
      MaterialParameterSpec(
        name="youngs_modulus",
        value=210.0e9,
        source=_source("material:E"),
      ),
      MaterialParameterSpec(
        name="poisson_ratio",
        value=0.3,
        source=_source("material:nu"),
      ),
    )
  return MaterialSpec(
    id=material_id,
    model=model,
    parameters=parameters,
    source=_source("material"),
  )


def _field(field_id: str | int = "displacement") -> FieldSpec:
  return FieldSpec(
    id=field_id,
    components=("x", "y"),
    location="node",
    source=_source("field"),
  )


def _two_cell_model() -> ModelSpec:
  coordinates = (
    (0.0, 0.0),
    (0.5, 0.0),
    (1.0, 0.0),
    (1.0, 0.5),
    (1.0, 1.0),
    (0.5, 1.0),
    (0.0, 1.0),
    (0.0, 0.5),
    (1.5, 0.0),
    (2.0, 0.0),
    (2.0, 0.5),
    (2.0, 1.0),
    (1.5, 1.0),
  )
  nodes = tuple(
    NodeSpec(
      id=index,
      coordinates=point,
      source=_source(f"node:{index}"),
    )
    for index, point in enumerate(coordinates, start=1)
  )
  cells = (
    CellSpec(
      id=101,
      node_ids=(1, 2, 3, 4, 5, 6, 7, 8),
      source=_source("cell:101"),
    ),
    CellSpec(
      id=102,
      node_ids=(3, 9, 10, 11, 12, 13, 5, 4),
      source=_source("cell:102"),
    ),
  )
  block = CellBlockSpec(
    id="q8-cells",
    reference_topology="quadrilateral",
    topological_dimension=2,
    embedding_dimension=2,
    geometry_interpolation="serendipity-quad8",
    cells=cells,
    source=_source("cell-block"),
  )
  field = _field()
  material = _material()
  region = RegionSpec(
    id="domain",
    cell_refs=tuple(CellRef(block_id=block.id, cell_id=cell.id) for cell in cells),
    field_ids=(field.id,),
    material_id=material.id,
    formulation="small-strain-continuum",
    quadrature="gauss-3x3",
    source=_source("region"),
  )
  return ModelSpec(
    mesh=MeshSpec(
      nodes=nodes,
      cell_blocks=(block,),
      source=_source("mesh"),
    ),
    fields=(field,),
    materials=(material,),
    regions=(region,),
    source=_source("model"),
  )


def _one_cell_model(
  *,
  scale: float = 1.0,
  cell_id: str | int = 101,
  node_ids: tuple[str | int, ...] | None = None,
  local_order: tuple[int, ...] = tuple(range(8)),
  coordinates: tuple[tuple[float, float], ...] | None = None,
) -> ModelSpec:
  if node_ids is None:
    node_ids = tuple(range(1, 9))
  if coordinates is None:
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
    NodeSpec(
      id=node_id,
      coordinates=(point[0] * scale, point[1] * scale),
      source=_source(f"node-source-{index}"),
    )
    for index, (node_id, point) in enumerate(zip(node_ids, coordinates, strict=True))
  )
  cell = CellSpec(
    id=cell_id,
    node_ids=tuple(node_ids[index] for index in local_order),
    source=_source("cell-source"),
  )
  block = CellBlockSpec(
    id="block",
    reference_topology="quadrilateral",
    topological_dimension=2,
    embedding_dimension=2,
    geometry_interpolation="serendipity-quad8",
    cells=(cell,),
    source=_source("block-source"),
  )
  field = _field()
  material = _material()
  region = RegionSpec(
    id="region",
    cell_refs=(CellRef(block_id=block.id, cell_id=cell.id),),
    field_ids=(field.id,),
    material_id=material.id,
    formulation="small-strain-continuum",
    quadrature="gauss-3x3",
    source=_source("region-source"),
  )
  return ModelSpec(
    mesh=MeshSpec(
      nodes=nodes,
      cell_blocks=(block,),
      source=_source("mesh-source"),
    ),
    fields=(field,),
    materials=(material,),
    regions=(region,),
    source=_source("model-source"),
  )


def _replace_block(model: ModelSpec, block: CellBlockSpec) -> ModelSpec:
  mesh = replace(model.mesh, cell_blocks=(block,))
  return replace(model, mesh=mesh)


def _replace_material(model: ModelSpec, material: MaterialSpec) -> ModelSpec:
  region = replace(model.regions[0], material_id=material.id)
  return replace(model, materials=(material,), regions=(region,))


def _descriptor_with(
  key: tuple[str, str],
  *,
  implementation_id: str = "changed-implementation",
  metadata: dict[str, object] | None = None,
  binding: object | None = None,
) -> RegistryDescriptor:
  reference = q8_reference_registry()[key]
  return RegistryDescriptor(
    kind=key[0],
    name=key[1],
    version=reference.version,
    implementation_id=implementation_id,
    metadata=q8_descriptor_metadata(*key) if metadata is None else metadata,
    binding=reference.binding if binding is None else binding,
  )


def _compiled_arrays(model: CompiledModel) -> tuple[FinalizedArray, ...]:
  block = model.domain_blocks[0]
  return (
    model.mesh.coordinates,
    model.mesh.cell_blocks[0].connectivity,
    model.dofs.node_component_dofs,
    block.dof_map,
    block.quadrature_points,
    block.quadrature_weights,
    block.shape_values,
    block.parent_gradients,
    block.material_parameters,
  )


def test_successful_compile_owns_canonical_recipe_and_explicit_maps() -> None:
  model = compile_model(_two_cell_model(), q8_reference_registry())
  block = model.domain_blocks[0]

  assert model is model
  assert model.mesh.node_ids == tuple(range(1, 14))
  assert model.mesh.cell_blocks[0].cell_ids == (101, 102)
  np.testing.assert_array_equal(
    block.connectivity.values,
    [
      [0, 1, 2, 3, 4, 5, 6, 7],
      [2, 8, 9, 10, 11, 12, 4, 3],
    ],
  )
  assert block.dof_map.values.shape == (2, 16)
  np.testing.assert_array_equal(
    block.dof_map.values[0],
    np.arange(16, dtype=np.int64),
  )
  np.testing.assert_array_equal(
    block.dof_map.values[1],
    [4, 5, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 8, 9, 6, 7],
  )
  assert block.connectivity is model.mesh.cell_blocks[0].connectivity
  assert block.dof_map is model.assembly_topology.block_recipes[0].dof_map
  assert block.measure_convention == "per-unit-out-of-plane-thickness"
  assert block.source_material_id == "steel"
  assert not hasattr(block, "section")
  assert model.assembly_topology.block_recipes[0].coupling == (
    "full-element-local-dof-clique"
  )

  for array in _compiled_arrays(model):
    assert array.values.flags.owndata
    assert array.values.flags.c_contiguous
    assert not array.values.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
      array.values.flat[0] = 0

  with pytest.raises(FrozenInstanceError):
    model.domain_blocks = ()


def test_state_layout_contains_sizes_but_no_evolving_values() -> None:
  model = compile_model(_two_cell_model(), q8_reference_registry())
  layout = model.physical_state_layout
  block_layout = layout.block_states[0]

  assert layout.global_primary_size == 26
  assert layout.evolving_value_count == 0
  assert block_layout.material_history_shape == (2, 9, 1, 0)
  assert block_layout.formulation_history_shape == (2, 0)
  assert block_layout.material_history_width == 0
  assert block_layout.formulation_history_width == 0
  for field in fields(layout):
    value = getattr(layout, field.name)
    assert not isinstance(value, (np.ndarray, FinalizedArray))


def test_capabilities_are_derived_from_the_frozen_block_meaning() -> None:
  model = compile_model(_two_cell_model(), q8_reference_registry())
  capabilities = model.capabilities

  assert capabilities.response_class == "linear-elastic"
  assert capabilities.fixed_model_coupling
  assert capabilities.tangent_class == "symmetric-constant-material"
  assert capabilities.tangent_is_symmetric
  assert capabilities.tangent_is_constant
  assert capabilities.conservative_internal_contribution
  assert not capabilities.state_dependent
  assert not capabilities.has_storage
  assert not capabilities.has_mass
  assert not capabilities.has_damping
  assert not capabilities.restart_history_required
  assert capabilities.contribution_channels == (
    "internal-force",
    "material-tangent",
  )


def test_entity_and_source_maps_cover_every_stable_identity() -> None:
  model = compile_model(_two_cell_model(), q8_reference_registry())

  assert model.entity_index.lookup("node", 1).dense_index == 0
  assert model.entity_index.lookup("cell", ("q8-cells", 102)).local_index == 1
  assert model.entity_index.lookup("field", "displacement").dense_index == 0
  assert model.entity_index.lookup("region", "domain").block_index == 0
  assert (
    model.entity_index.lookup(
      "domain_block",
      ("q8-cells", "domain"),
    ).block_index
    == 0
  )
  assert (
    model.entity_index.lookup(
      "dof",
      (13, "displacement", "y"),
    ).dense_index
    == 25
  )
  assert (
    model.entity_index.lookup(
      "integration_point",
      ("q8-cells", 102, 8),
    ).dense_index
    == 17
  )
  assert model.source_map.lookup("node", 1).source == "node:1"
  assert (
    model.source_map.lookup(
      "integration_point",
      ("q8-cells", 102, 8),
    ).source
    == "cell:102"
  )
  with pytest.raises(KeyError, match="exact semantic ID"):
    model.entity_index.lookup("node", "1")


def test_reference_registry_captures_exact_math_bindings_and_meaning() -> None:
  registry = q8_reference_registry()
  model = compile_model(_two_cell_model(), registry)
  snapshot = model.registry_snapshot

  assert snapshot.resolve(*Q8_TOPOLOGY_KEY).binding is serendipity_quad8
  assert snapshot.resolve(*Q8_QUADRATURE_KEY).binding is gauss_tensor_product_2d
  assert snapshot.resolve(*Q8_FORMULATION_KEY).binding is strain_displacement
  assert snapshot.resolve(*Q8_MATERIAL_KEY).binding is plane_stress_matrix
  assert tuple(descriptor.key for descriptor in snapshot.descriptors) == tuple(
    sorted(
      (
        Q8_TOPOLOGY_KEY,
        Q8_QUADRATURE_KEY,
        Q8_FORMULATION_KEY,
        Q8_MATERIAL_KEY,
      )
    )
  )
  assert model.domain_blocks[0].material_parameter_names == (
    "youngs_modulus",
    "poisson_ratio",
  )
  np.testing.assert_array_equal(
    model.domain_blocks[0].material_parameters.values,
    [[210.0e9, 0.3]],
  )

  registry.clear()
  assert snapshot.resolve(*Q8_TOPOLOGY_KEY).binding is serendipity_quad8


def test_shipped_q8_binding_matches_hard_coded_local_convention_oracle() -> None:
  declared_parent_nodes = np.array(
    [
      [-1.0, -1.0],
      [0.0, -1.0],
      [1.0, -1.0],
      [1.0, 0.0],
      [1.0, 1.0],
      [0.0, 1.0],
      [-1.0, 1.0],
      [-1.0, 0.0],
    ],
    dtype=np.float64,
  )
  metadata = q8_descriptor_metadata(*Q8_TOPOLOGY_KEY)
  assert metadata["local_node_parent_coordinates"] == declared_parent_nodes.tolist()

  binding = q8_reference_registry()[Q8_TOPOLOGY_KEY].binding
  nodal_shape_values, nodal_parent_gradients = binding(declared_parent_nodes)
  np.testing.assert_array_equal(nodal_shape_values, np.eye(8, dtype=np.float64))
  assert nodal_parent_gradients.shape == (8, 8, 2)

  center_shape_values, center_parent_gradients = binding(
    np.array([[0.0, 0.0]], dtype=np.float64)
  )
  expected_center_shape_values = np.array(
    [[-0.25, 0.5, -0.25, 0.5, -0.25, 0.5, -0.25, 0.5]],
    dtype=np.float64,
  )
  expected_center_parent_gradients = np.array(
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
    dtype=np.float64,
  )
  np.testing.assert_array_equal(
    center_shape_values,
    expected_center_shape_values,
  )
  np.testing.assert_array_equal(
    center_parent_gradients,
    expected_center_parent_gradients,
  )


def test_caller_mutation_after_compile_cannot_change_any_compiled_meaning() -> None:
  normalized = normalize_model_spec(_two_cell_model())
  registry = q8_reference_registry()
  compiled = compile_model(normalized, registry)
  fingerprint = compiled.content_fingerprint
  coordinates = compiled.mesh.coordinates.values.copy()
  connectivity = compiled.domain_blocks[0].connectivity.values.copy()

  object.__setattr__(normalized.mesh.nodes[0], "coordinates", (99.0, 99.0))
  object.__setattr__(normalized.mesh.cell_blocks[0].cells[0], "node_ids", ())
  registry.clear()

  assert compiled.content_fingerprint == fingerprint
  np.testing.assert_array_equal(compiled.mesh.coordinates.values, coordinates)
  np.testing.assert_array_equal(
    compiled.domain_blocks[0].connectivity.values,
    connectivity,
  )


def test_descriptor_output_arrays_are_detached_before_publication() -> None:
  caller_points, caller_weights = gauss_tensor_product_2d(3)

  def caller_binding(order: int) -> tuple[np.ndarray, np.ndarray]:
    assert order == 3
    return caller_points, caller_weights

  registry = q8_reference_registry()
  registry[Q8_QUADRATURE_KEY] = _descriptor_with(
    Q8_QUADRATURE_KEY,
    binding=caller_binding,
  )
  compiled = compile_model(_one_cell_model(), registry)
  expected_points = compiled.domain_blocks[0].quadrature_points.values.copy()
  expected_weights = compiled.domain_blocks[0].quadrature_weights.values.copy()

  caller_points.fill(99.0)
  caller_weights.fill(99.0)

  np.testing.assert_array_equal(
    compiled.domain_blocks[0].quadrature_points.values,
    expected_points,
  )
  np.testing.assert_array_equal(
    compiled.domain_blocks[0].quadrature_weights.values,
    expected_weights,
  )


def test_canonical_permutations_preserve_content_not_live_identity() -> None:
  authored = _two_cell_model()
  block = authored.mesh.cell_blocks[0]
  region = authored.regions[0]
  material = authored.materials[0]
  reordered = replace(
    authored,
    mesh=replace(
      authored.mesh,
      nodes=tuple(reversed(authored.mesh.nodes)),
      cell_blocks=(replace(block, cells=tuple(reversed(block.cells))),),
    ),
    materials=(replace(material, parameters=tuple(reversed(material.parameters))),),
    regions=(replace(region, cell_refs=tuple(reversed(region.cell_refs))),),
  )

  first = compile_model(authored, q8_reference_registry())
  second = compile_model(reordered, q8_reference_registry())

  assert first.content_fingerprint == second.content_fingerprint
  assert first.provenance.manifest.to_bytes() == second.provenance.manifest.to_bytes()
  assert first.instance_id != second.instance_id
  assert first is not second
  assert (first == second) is False
  for first_array, second_array in zip(
    _compiled_arrays(first),
    _compiled_arrays(second),
    strict=True,
  ):
    np.testing.assert_array_equal(first_array.values, second_array.values)
    assert not np.shares_memory(first_array.values, second_array.values)


def test_unselected_registry_entries_do_not_enter_compiled_identity() -> None:
  registry = q8_reference_registry()
  extra = RegistryDescriptor(
    kind="output",
    name="unused",
    version="1",
    implementation_id="unused-v1",
    metadata={},
    binding=lambda value: value,
  )
  registry[extra.key] = extra
  with_extra = compile_model(_two_cell_model(), registry)
  without_extra = compile_model(_two_cell_model(), q8_reference_registry())

  assert with_extra.content_fingerprint == without_extra.content_fingerprint
  assert len(with_extra.registry_snapshot.descriptors) == 4


def test_semantic_source_policy_and_descriptor_changes_change_fingerprint() -> None:
  authored = _two_cell_model()
  baseline = compile_model(authored, q8_reference_registry())

  changed_node = replace(authored.mesh.nodes[0], coordinates=(-0.1, 0.0))
  semantic_change = replace(
    authored,
    mesh=replace(
      authored.mesh,
      nodes=(changed_node, *authored.mesh.nodes[1:]),
    ),
  )
  changed_source_node = replace(
    authored.mesh.nodes[0],
    source=_source("changed-node-source"),
  )
  source_change = replace(
    authored,
    mesh=replace(
      authored.mesh,
      nodes=(changed_source_node, *authored.mesh.nodes[1:]),
    ),
  )
  registry = q8_reference_registry()
  registry[Q8_MATERIAL_KEY] = _descriptor_with(
    Q8_MATERIAL_KEY,
    implementation_id="plane-stress-compatible-v2",
  )

  variants = (
    compile_model(semantic_change, q8_reference_registry()),
    compile_model(source_change, q8_reference_registry()),
    compile_model(
      authored,
      q8_reference_registry(),
      policy=ModelCompilationPolicy(dense_index_dtype="int32"),
    ),
    compile_model(
      authored,
      q8_reference_registry(),
      policy=ModelCompilationPolicy(geometry_relative_tolerance=1.0e-10),
    ),
    compile_model(authored, registry),
  )
  assert all(
    item.content_fingerprint != baseline.content_fingerprint for item in variants
  )


def test_mixed_string_integer_ids_compile_without_coercion() -> None:
  node_ids: tuple[str | int, ...] = (
    "node-1",
    2,
    "node-3",
    4,
    "node-5",
    6,
    "node-7",
    8,
  )
  authored = _one_cell_model(
    node_ids=node_ids,
    cell_id=_HUGE_INTEGER_ID,
  )
  compiled = compile_model(authored, q8_reference_registry())

  assert set(compiled.mesh.node_ids) == set(node_ids)
  assert compiled.entity_index.lookup("node", 2).dense_index is not None
  assert compiled.entity_index.lookup("node", "node-1").dense_index is not None
  assert (
    compiled.entity_index.lookup(
      "cell",
      ("block", _HUGE_INTEGER_ID),
    ).dense_index
    == 0
  )


@pytest.mark.parametrize("scale", [1.0e-200, 1.0e200])
def test_reference_geometry_is_valid_under_extreme_uniform_scaling(
  scale: float,
) -> None:
  compiled = compile_model(
    _one_cell_model(scale=scale),
    q8_reference_registry(),
  )
  assert compiled.domain_blocks[0].connectivity.values.shape == (1, 8)


def test_reference_geometry_classification_is_translation_invariant() -> None:
  offset = 1.0e100
  length = 1.0e90
  unit_coordinates = (
    (0.0, 0.0),
    (0.5, 0.0),
    (1.0, 0.0),
    (1.0, 0.5),
    (1.0, 1.0),
    (0.5, 1.0),
    (0.0, 1.0),
    (0.0, 0.5),
  )
  translated = tuple(
    (offset + length * x, -offset + length * y) for x, y in unit_coordinates
  )

  compiled = compile_model(
    _one_cell_model(coordinates=translated),
    q8_reference_registry(),
  )

  assert compiled.domain_blocks[0].connectivity.values.shape == (1, 8)


@pytest.mark.parametrize(
  ("local_order", "code"),
  [
    ((0, 7, 6, 5, 4, 3, 2, 1), "inverted-reference-geometry"),
    ((0, 1, 4, 3, 2, 5, 6, 7), "sign-changing-reference-geometry"),
  ],
)
def test_inverted_and_sign_changing_reference_geometry_are_rejected(
  local_order: tuple[int, ...],
  code: str,
) -> None:
  with pytest.raises(ModelCompilationError, match=code):
    compile_model(
      _one_cell_model(local_order=local_order),
      q8_reference_registry(),
    )


def test_scale_relative_near_singular_reference_geometry_is_rejected() -> None:
  coordinates = (
    (0.0, 0.0),
    (0.5, 0.0),
    (1.0, 0.0),
    (1.0, 0.5e-14),
    (1.0, 1.0e-14),
    (0.5, 1.0e-14),
    (0.0, 1.0e-14),
    (0.0, 0.5e-14),
  )
  with pytest.raises(ModelCompilationError, match="near-singular-reference"):
    compile_model(
      _one_cell_model(coordinates=coordinates),
      q8_reference_registry(),
    )


def test_huge_integer_geometry_diagnostic_is_bounded_under_digit_limit() -> None:
  authored = _one_cell_model(
    cell_id=_HUGE_INTEGER_ID,
    local_order=(0, 7, 6, 5, 4, 3, 2, 1),
  )
  with pytest.raises(ModelCompilationError) as captured:
    compile_model(authored, q8_reference_registry())

  rendered = str(captured.value)
  assert len(rendered) < _MAX_HUGE_DIAGNOSTIC_LENGTH
  assert "<int sign=+ bits=" in rendered
  assert "inverted-reference-geometry" in rendered


def test_missing_and_multiple_cross_region_membership_are_rejected() -> None:
  authored = _two_cell_model()
  region = authored.regions[0]
  missing = replace(
    authored,
    regions=(replace(region, cell_refs=(region.cell_refs[0],)),),
  )
  duplicate_region = replace(region, id="duplicate", source=_source("duplicate"))
  multiple = replace(authored, regions=(region, duplicate_region))

  with pytest.raises(ModelCompilationError, match="incomplete-cell-membership"):
    compile_model(missing, q8_reference_registry())
  with pytest.raises(ModelCompilationError, match="multiple-cell-membership"):
    compile_model(multiple, q8_reference_registry())


@pytest.mark.parametrize(
  ("replacement", "expected"),
  [
    ({"reference_topology": "triangle"}, "incompatible-cell-block"),
    ({"topological_dimension": 1}, "incompatible-cell-block"),
    ({"geometry_interpolation": "quad8-by-name"}, "incompatible-cell-block"),
  ],
)
def test_wrong_topology_dimension_and_interpolation_are_rejected(
  replacement: dict[str, object],
  expected: str,
) -> None:
  authored = _one_cell_model()
  block = replace(authored.mesh.cell_blocks[0], **replacement)
  with pytest.raises(ModelCompilationError, match=expected):
    compile_model(_replace_block(authored, block), q8_reference_registry())


def test_wrong_q8_arity_is_rejected() -> None:
  authored = _one_cell_model()
  block = authored.mesh.cell_blocks[0]
  cell = replace(block.cells[0], node_ids=block.cells[0].node_ids[:-1])
  with pytest.raises(ModelCompilationError, match="invalid-q8-arity"):
    compile_model(
      _replace_block(authored, replace(block, cells=(cell,))),
      q8_reference_registry(),
    )


@pytest.mark.parametrize(
  "missing_key",
  [Q8_TOPOLOGY_KEY, Q8_QUADRATURE_KEY, Q8_FORMULATION_KEY, Q8_MATERIAL_KEY],
)
def test_each_missing_registry_key_fails_closed(
  missing_key: tuple[str, str],
) -> None:
  registry = q8_reference_registry()
  registry.pop(missing_key)
  with pytest.raises(ModelCompilationError, match="registry-capture-failed"):
    compile_model(_one_cell_model(), registry)


@pytest.mark.parametrize(
  "key",
  [Q8_TOPOLOGY_KEY, Q8_QUADRATURE_KEY, Q8_FORMULATION_KEY, Q8_MATERIAL_KEY],
)
def test_descriptor_names_without_compatible_metadata_are_rejected(
  key: tuple[str, str],
) -> None:
  registry = q8_reference_registry()
  metadata = q8_descriptor_metadata(*key)
  metadata["incompatible_change"] = True
  registry[key] = _descriptor_with(key, metadata=metadata)

  with pytest.raises(
    ModelCompilationError,
    match="incompatible-registry-descriptor",
  ):
    compile_model(_one_cell_model(), registry)


def _invalid_quadrature_binding(
  order: int,
) -> tuple[np.ndarray, np.ndarray]:
  del order
  return np.zeros((8, 2)), np.ones(8)


def _invalid_topology_binding(
  points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  return np.zeros((points.shape[0], 8)), np.zeros((points.shape[0], 8, 2))


@pytest.mark.parametrize(
  ("key", "binding", "expected"),
  [
    (
      Q8_QUADRATURE_KEY,
      _invalid_quadrature_binding,
      "invalid-quadrature-binding-output",
    ),
    (
      Q8_TOPOLOGY_KEY,
      _invalid_topology_binding,
      "invalid-topology-binding-output",
    ),
  ],
)
def test_malformed_descriptor_binding_outputs_fail_as_compilation_errors(
  key: tuple[str, str],
  binding: object,
  expected: str,
) -> None:
  registry = q8_reference_registry()
  registry[key] = _descriptor_with(key, binding=binding)
  with pytest.raises(ModelCompilationError, match=expected):
    compile_model(_one_cell_model(), registry)


@pytest.mark.parametrize(
  ("parameters", "expected"),
  [
    (
      (MaterialParameterSpec("youngs_modulus", 1.0),),
      "invalid-material-parameter-schema",
    ),
    (
      (
        MaterialParameterSpec("youngs_modulus", 1.0),
        MaterialParameterSpec("poisson_ratio", 0.3),
        MaterialParameterSpec("density", 1.0),
      ),
      "invalid-material-parameter-schema",
    ),
    (
      (
        MaterialParameterSpec("youngs_modulus", 0.0),
        MaterialParameterSpec("poisson_ratio", 0.3),
      ),
      "invalid-youngs-modulus",
    ),
    (
      (
        MaterialParameterSpec("youngs_modulus", 1.0),
        MaterialParameterSpec("poisson_ratio", 0.5),
      ),
      "invalid-poisson-ratio",
    ),
    (
      (
        MaterialParameterSpec("youngs_modulus", True),
        MaterialParameterSpec("poisson_ratio", 0.3),
      ),
      "invalid-material-parameter-type",
    ),
  ],
)
def test_missing_extra_and_out_of_domain_material_parameters_fail(
  parameters: tuple[MaterialParameterSpec, ...],
  expected: str,
) -> None:
  authored = _one_cell_model()
  material = _material(parameters=parameters)
  with pytest.raises(ModelCompilationError, match=expected):
    compile_model(
      _replace_material(authored, material),
      q8_reference_registry(),
    )


@pytest.mark.parametrize(
  ("youngs_modulus", "poisson_ratio"),
  [(210, 0.3), (210.0, 0)],
)
def test_integer_material_parameters_compile_to_float64(
  youngs_modulus: int | float,
  poisson_ratio: int | float,
) -> None:
  authored = _one_cell_model()
  material = _material(
    parameters=(
      MaterialParameterSpec("youngs_modulus", youngs_modulus),
      MaterialParameterSpec("poisson_ratio", poisson_ratio),
    )
  )

  compiled = compile_model(
    _replace_material(authored, material),
    q8_reference_registry(),
  )

  parameters = compiled.domain_blocks[0].material_parameters.values
  assert parameters.dtype == np.dtype(np.float64)
  np.testing.assert_array_equal(
    parameters,
    [[float(youngs_modulus), float(poisson_ratio)]],
  )


def test_material_integer_too_large_for_float64_fails_deterministically() -> None:
  authored = _one_cell_model()
  material = _material(
    parameters=(
      MaterialParameterSpec("youngs_modulus", _HUGE_INTEGER_ID),
      MaterialParameterSpec("poisson_ratio", 0),
    )
  )

  with pytest.raises(
    ModelCompilationError,
    match="invalid-material-parameter-value",
  ) as captured:
    compile_model(
      _replace_material(authored, material),
      q8_reference_registry(),
    )

  assert len(str(captured.value)) < _MAX_HUGE_DIAGNOSTIC_LENGTH
  assert "youngs_modulus cannot be represented as finite float64" in str(captured.value)


def test_nonfinite_material_and_coordinate_values_remain_normalization_errors() -> None:
  authored = _one_cell_model()
  nonfinite_material = _material(
    parameters=(
      MaterialParameterSpec("youngs_modulus", float("inf")),
      MaterialParameterSpec("poisson_ratio", 0.3),
    )
  )
  node = replace(authored.mesh.nodes[0], coordinates=(float("nan"), 0.0))
  nonfinite_coordinates = replace(
    authored,
    mesh=replace(authored.mesh, nodes=(node, *authored.mesh.nodes[1:])),
  )

  with pytest.raises(ModelSpecValidationError, match="finite plain scalars"):
    compile_model(
      _replace_material(authored, nonfinite_material),
      q8_reference_registry(),
    )
  with pytest.raises(ModelSpecValidationError, match="invalid-coordinate"):
    compile_model(nonfinite_coordinates, q8_reference_registry())


def test_wrong_field_material_and_region_contracts_fail_before_carrier() -> None:
  authored = _one_cell_model()
  wrong_field = replace(authored.fields[0], components=("y", "x"))
  wrong_material = replace(authored.materials[0], model="plane-strain-linear-elastic")
  wrong_quadrature = replace(authored.regions[0], quadrature="gauss-2x2")

  with pytest.raises(ModelCompilationError, match="incompatible-field"):
    compile_model(replace(authored, fields=(wrong_field,)), q8_reference_registry())
  with pytest.raises(ModelCompilationError, match="incompatible-material-model"):
    compile_model(
      _replace_material(authored, wrong_material),
      q8_reference_registry(),
    )
  with pytest.raises(ModelCompilationError, match="incompatible-quadrature"):
    compile_model(
      replace(authored, regions=(wrong_quadrature,)),
      q8_reference_registry(),
    )


def test_dense_index_capacity_is_checked_before_conversion() -> None:
  authored = _one_cell_model()
  extra_nodes = tuple(
    NodeSpec(
      id=f"unused-{index}",
      coordinates=(float(index), 2.0),
      source=_source(f"unused-source-{index}"),
    )
    for index in range(57)
  )
  oversized = replace(
    authored,
    mesh=replace(authored.mesh, nodes=(*authored.mesh.nodes, *extra_nodes)),
  )
  with pytest.raises(ModelCompilationError, match="dense-index-overflow"):
    compile_model(
      oversized,
      q8_reference_registry(),
      policy=ModelCompilationPolicy(dense_index_dtype="int8"),
    )


def test_malformed_exact_spec_is_rejected_by_normalization_boundary() -> None:
  normalized = normalize_model_spec(_one_cell_model())
  object.__setattr__(normalized.mesh, "cell_blocks", list(normalized.mesh.cell_blocks))

  with pytest.raises(ModelSpecValidationError, match="exactly tuple"):
    compile_model(normalized, q8_reference_registry())
