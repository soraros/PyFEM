# SPDX-License-Identifier: MIT

"""Dangerous-case contracts for authored and normalized model specifications."""

from __future__ import annotations

import math
import sys
from dataclasses import replace

import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

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


class _MutableDuck:
  def __init__(self) -> None:
    self.id = "duck"
    self.coordinates = [0.0, 0.0]
    self.source = SourceContext(source="untrusted-child")


class _MissingAttributes:
  pass


class _NodeSpecSubclass(NodeSpec):
  pass


class _MutableHashStr(str):
  def __new__(cls, value: str) -> _MutableHashStr:
    instance = super().__new__(cls, value)
    instance.hash_calls = 0
    instance.hash_value = 1
    instance.strip_calls = 0
    return instance

  def __hash__(self) -> int:
    self.hash_calls += 1
    return self.hash_value

  def strip(self, chars: str | None = None) -> str:
    self.strip_calls += 1
    return self


class _UnhashableStr(str):
  __hash__ = None

  def __new__(cls, value: str) -> _UnhashableStr:
    instance = super().__new__(cls, value)
    instance.strip_calls = 0
    return instance

  def strip(self, chars: str | None = None) -> str:
    self.strip_calls += 1
    return self


class _ReprBombStr(str):
  def __new__(cls, value: str) -> _ReprBombStr:
    instance = super().__new__(cls, value)
    instance.repr_calls = 0
    instance.strip_calls = 0
    return instance

  def __repr__(self) -> str:
    self.repr_calls += 1
    msg = "hostile repr invoked"
    raise RuntimeError(msg)

  def strip(self, chars: str | None = None) -> str:
    self.strip_calls += 1
    return self


class _ComparisonBombInt(int):
  def __new__(cls, value: int) -> _ComparisonBombInt:
    instance = super().__new__(cls, value)
    instance.comparison_calls = 0
    return instance

  def __eq__(self, other: object) -> bool:
    self.comparison_calls += 1
    msg = "hostile comparison invoked"
    raise RuntimeError(msg)

  def __ge__(self, other: object) -> bool:
    self.comparison_calls += 1
    msg = "hostile comparison invoked"
    raise RuntimeError(msg)

  def __gt__(self, other: object) -> bool:
    self.comparison_calls += 1
    msg = "hostile comparison invoked"
    raise RuntimeError(msg)

  def __le__(self, other: object) -> bool:
    self.comparison_calls += 1
    msg = "hostile comparison invoked"
    raise RuntimeError(msg)

  def __lt__(self, other: object) -> bool:
    self.comparison_calls += 1
    msg = "hostile comparison invoked"
    raise RuntimeError(msg)


def _source(label: str) -> SourceContext:
  return SourceContext(source=label)


def _forge_slot(value: object, name: str, replacement: object) -> object:
  object.__setattr__(value, name, replacement)
  return value


def _nodes_2d() -> tuple[NodeSpec, ...]:
  return (
    NodeSpec(id=1, coordinates=(0.0, 0.0), source=_source("nodes:1")),
    NodeSpec(id=2, coordinates=(1.0, 0.0), source=_source("nodes:2")),
    NodeSpec(id=3, coordinates=(1.0, 1.0), source=_source("nodes:3")),
    NodeSpec(id=4, coordinates=(0.0, 1.0), source=_source("nodes:4")),
  )


def _quad_block() -> CellBlockSpec:
  return CellBlockSpec(
    id="quad-cells",
    reference_topology="quadrilateral",
    topological_dimension=2,
    embedding_dimension=2,
    geometry_interpolation="quad4",
    cells=(
      CellSpec(
        id=10,
        node_ids=(1, 2, 3, 4),
        source=_source("cells:10"),
      ),
    ),
    source=_source("blocks:quad-cells"),
  )


def _field() -> FieldSpec:
  return FieldSpec(
    id="displacement",
    components=("x", "y"),
    location="node",
    source=_source("fields:displacement"),
  )


def _material() -> MaterialSpec:
  return MaterialSpec(
    id="solid",
    model="linear-elastic",
    parameters=(
      MaterialParameterSpec(
        name="youngs_modulus",
        value=210.0e9,
        source=_source("materials:solid:E"),
      ),
    ),
    source=_source("materials:solid"),
  )


def _region() -> RegionSpec:
  return RegionSpec(
    id="domain",
    cell_refs=(CellRef(block_id="quad-cells", cell_id=10),),
    field_ids=("displacement",),
    material_id="solid",
    formulation="small-strain-continuum",
    quadrature="gauss-2x2",
    source=_source("regions:domain"),
  )


def _valid_model() -> ModelSpec:
  return ModelSpec(
    mesh=MeshSpec(
      nodes=_nodes_2d(),
      cell_blocks=(_quad_block(),),
      source=_source("mesh"),
    ),
    fields=(_field(),),
    materials=(_material(),),
    regions=(_region(),),
    source=_source("model"),
  )


def _model_with_nested_value(family: str, value: object) -> ModelSpec:
  base = _valid_model()
  if family == "mesh":
    return replace(base, mesh=value)
  if family == "node":
    return replace(base, mesh=replace(base.mesh, nodes=(value,)))
  if family == "cell-block":
    return replace(base, mesh=replace(base.mesh, cell_blocks=(value,)))
  if family == "cell":
    block = replace(_quad_block(), cells=(value,))
    return replace(base, mesh=replace(base.mesh, cell_blocks=(block,)))
  if family == "field":
    return replace(base, fields=(value,))
  if family == "material":
    return replace(base, materials=(value,))
  if family == "material-parameter":
    material = replace(_material(), parameters=(value,))
    return replace(base, materials=(material,))
  if family == "region":
    return replace(base, regions=(value,))
  if family == "cell-ref":
    region = replace(_region(), cell_refs=(value,))
    return replace(base, regions=(region,))
  msg = f"unknown nested family {family!r}"
  raise AssertionError(msg)


def _diagnostic_codes(error: ModelSpecValidationError) -> tuple[str, ...]:
  return tuple(diagnostic.code for diagnostic in error.diagnostics)


def test_tet4_volume_and_quad4_surface_in_3d_do_not_collide_by_node_count() -> None:
  nodes = tuple(
    NodeSpec(id=node_id, coordinates=coordinates)
    for node_id, coordinates in enumerate(
      (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (2.0, 0.0, 0.0),
        (3.0, 0.0, 0.0),
        (3.0, 1.0, 0.0),
        (2.0, 1.0, 0.0),
      ),
      start=1,
    )
  )
  tet_block = CellBlockSpec(
    id="tet-volume",
    reference_topology="tetrahedron",
    topological_dimension=3,
    embedding_dimension=3,
    geometry_interpolation="tet4",
    cells=(CellSpec(id="tet-1", node_ids=(1, 2, 3, 4)),),
  )
  quad_block = CellBlockSpec(
    id="quad-surface",
    reference_topology="quadrilateral",
    topological_dimension=2,
    embedding_dimension=3,
    geometry_interpolation="quad4",
    cells=(CellSpec(id="quad-1", node_ids=(5, 6, 7, 8)),),
  )
  model = ModelSpec(
    mesh=MeshSpec(nodes=nodes, cell_blocks=(tet_block, quad_block)),
    fields=(
      FieldSpec(
        id="displacement",
        components=("x", "y", "z"),
        location="node",
      ),
    ),
    materials=(_material(),),
    regions=(
      RegionSpec(
        id="volume",
        cell_refs=(CellRef(block_id="tet-volume", cell_id="tet-1"),),
        field_ids=("displacement",),
        material_id="solid",
        formulation="small-strain-volume",
        quadrature="tet-1",
      ),
      RegionSpec(
        id="surface",
        cell_refs=(CellRef(block_id="quad-surface", cell_id="quad-1"),),
        field_ids=("displacement",),
        material_id="solid",
        formulation="surface-physics",
        quadrature="quad-2x2",
      ),
    ),
  )

  normalized = normalize_model_spec(model)
  tet, quad = normalized.mesh.cell_blocks

  assert len(tet.cells[0].node_ids) == len(quad.cells[0].node_ids) == 4
  assert (tet.reference_topology, tet.topological_dimension) == (
    "tetrahedron",
    3,
  )
  assert (quad.reference_topology, quad.topological_dimension) == (
    "quadrilateral",
    2,
  )
  assert tet.embedding_dimension == quad.embedding_dimension == 3


def test_mixed_quad4_tri3_connectivity_is_unpadded_and_has_no_sentinel_nodes() -> None:
  tri_block = CellBlockSpec(
    id="tri-cells",
    reference_topology="triangle",
    topological_dimension=2,
    embedding_dimension=2,
    geometry_interpolation="tri3",
    cells=(CellSpec(id=20, node_ids=(1, 2, 3)),),
  )
  model = replace(
    _valid_model(),
    mesh=MeshSpec(
      nodes=_nodes_2d(),
      cell_blocks=(_quad_block(), tri_block),
    ),
    regions=(
      _region(),
      replace(
        _region(),
        id="triangle-domain",
        cell_refs=(CellRef(block_id="tri-cells", cell_id=20),),
      ),
    ),
  )

  normalized = normalize_model_spec(model)
  quad_connectivity = normalized.mesh.cell_blocks[0].cells[0].node_ids
  tri_connectivity = normalized.mesh.cell_blocks[1].cells[0].node_ids

  assert len(quad_connectivity) == 4
  assert len(tri_connectivity) == 3
  assert -1 not in quad_connectivity
  assert -1 not in tri_connectivity


def test_normalized_model_is_owned_and_caller_mutation_cannot_change_it() -> None:
  coordinates = [0.0, 0.0]
  connectivity = [1, 2, 3, 4]
  components = ["x", "y"]
  parameter_value = [1.0, 2.0]
  nodes = [
    NodeSpec(id=1, coordinates=coordinates),
    *_nodes_2d()[1:],
  ]
  cells = [CellSpec(id=10, node_ids=connectivity)]
  blocks = [replace(_quad_block(), cells=cells)]
  fields = [replace(_field(), components=components)]
  parameters = [
    MaterialParameterSpec(name="table", value=parameter_value),
  ]
  materials = [replace(_material(), parameters=parameters)]
  cell_refs = [CellRef(block_id="quad-cells", cell_id=10)]
  field_ids = ["displacement"]
  regions = [replace(_region(), cell_refs=cell_refs, field_ids=field_ids)]
  model = ModelSpec(
    mesh=MeshSpec(nodes=nodes, cell_blocks=blocks),
    fields=fields,
    materials=materials,
    regions=regions,
  )

  normalized = normalize_model_spec(model)
  coordinates[0] = 99.0
  connectivity[0] = 99
  components.append("mutated")
  parameter_value.append(99.0)
  nodes.clear()
  cells.clear()
  blocks.clear()
  fields.clear()
  parameters.clear()
  materials.clear()
  cell_refs.clear()
  field_ids.clear()
  regions.clear()

  assert normalized.mesh.nodes[0].coordinates == (0.0, 0.0)
  assert normalized.mesh.cell_blocks[0].cells[0].node_ids == (1, 2, 3, 4)
  assert normalized.fields[0].components == ("x", "y")
  assert normalized.materials[0].parameters[0].value == (1.0, 2.0)
  assert normalized.regions[0].field_ids == ("displacement",)


def test_normalized_tree_is_recursively_exact_new_and_caller_detached() -> None:
  parameter = MaterialParameterSpec(
    name=" table ",
    value=(" label ", (1, 2.0, True)),
    source=_source("materials:solid:table"),
  )
  model = replace(
    _valid_model(),
    materials=(replace(_material(), parameters=(parameter,)),),
  )
  caller_mesh = model.mesh
  caller_node = caller_mesh.nodes[0]
  caller_block = caller_mesh.cell_blocks[0]
  caller_cell = caller_block.cells[0]
  caller_field = model.fields[0]
  caller_material = model.materials[0]
  caller_parameter = caller_material.parameters[0]
  caller_region = model.regions[0]
  caller_cell_ref = caller_region.cell_refs[0]

  normalized = normalize_model_spec(model)

  assert type(normalized) is ModelSpec
  assert type(normalized.mesh) is MeshSpec
  assert type(normalized.mesh.nodes[0]) is NodeSpec
  assert type(normalized.mesh.cell_blocks[0]) is CellBlockSpec
  assert type(normalized.mesh.cell_blocks[0].cells[0]) is CellSpec
  assert type(normalized.fields[0]) is FieldSpec
  assert type(normalized.materials[0]) is MaterialSpec
  assert type(normalized.materials[0].parameters[0]) is MaterialParameterSpec
  assert type(normalized.regions[0]) is RegionSpec
  assert type(normalized.regions[0].cell_refs[0]) is CellRef
  assert normalized is not model
  assert normalized.mesh is not caller_mesh
  assert normalized.mesh.nodes[0] is not caller_node
  assert normalized.mesh.cell_blocks[0] is not caller_block
  assert normalized.mesh.cell_blocks[0].cells[0] is not caller_cell
  assert normalized.fields[0] is not caller_field
  assert normalized.materials[0] is not caller_material
  assert normalized.materials[0].parameters[0] is not caller_parameter
  assert normalized.regions[0] is not caller_region
  assert normalized.regions[0].cell_refs[0] is not caller_cell_ref

  source_pairs = (
    (normalized.source, model.source),
    (normalized.mesh.source, caller_mesh.source),
    (normalized.mesh.nodes[0].source, caller_node.source),
    (normalized.mesh.cell_blocks[0].source, caller_block.source),
    (normalized.mesh.cell_blocks[0].cells[0].source, caller_cell.source),
    (normalized.fields[0].source, caller_field.source),
    (normalized.materials[0].source, caller_material.source),
    (
      normalized.materials[0].parameters[0].source,
      caller_parameter.source,
    ),
    (normalized.regions[0].source, caller_region.source),
  )
  assert all(type(owned) is SourceContext for owned, _ in source_pairs)
  assert all(owned is not caller for owned, caller in source_pairs)

  container_pairs = (
    (normalized.mesh.nodes, caller_mesh.nodes),
    (normalized.mesh.cell_blocks, caller_mesh.cell_blocks),
    (normalized.mesh.nodes[0].coordinates, caller_node.coordinates),
    (normalized.mesh.cell_blocks[0].cells, caller_block.cells),
    (normalized.mesh.cell_blocks[0].cells[0].node_ids, caller_cell.node_ids),
    (normalized.fields, model.fields),
    (normalized.fields[0].components, caller_field.components),
    (normalized.materials, model.materials),
    (normalized.materials[0].parameters, caller_material.parameters),
    (normalized.regions, model.regions),
    (normalized.regions[0].cell_refs, caller_region.cell_refs),
    (normalized.regions[0].field_ids, caller_region.field_ids),
  )
  assert all(type(owned) is tuple for owned, _ in container_pairs)
  assert all(owned is not caller for owned, caller in container_pairs)
  owned_parameter = normalized.materials[0].parameters[0].value
  assert type(owned_parameter) is tuple
  assert type(owned_parameter[1]) is tuple
  assert owned_parameter is not caller_parameter.value
  assert owned_parameter[1] is not caller_parameter.value[1]

  assert type(normalized.mesh.nodes[0].id) is int
  assert all(type(value) is float for value in normalized.mesh.nodes[0].coordinates)
  assert type(normalized.mesh.cell_blocks[0].reference_topology) is str
  assert type(normalized.mesh.cell_blocks[0].topological_dimension) is int
  assert type(normalized.fields[0].id) is str
  assert all(type(value) is str for value in normalized.fields[0].components)
  assert normalized.materials[0].parameters[0].name == "table"
  assert owned_parameter == ("label", (1, 2.0, True))
  assert tuple(type(value) for value in owned_parameter[1]) == (int, float, bool)

  object.__setattr__(model.source, "source", "mutated:model")
  object.__setattr__(caller_mesh, "nodes", ())
  object.__setattr__(caller_node, "coordinates", (99.0, 99.0))
  object.__setattr__(caller_block, "reference_topology", "mutated-topology")
  object.__setattr__(caller_cell, "node_ids", (99,))
  object.__setattr__(caller_field, "components", ("mutated",))
  object.__setattr__(caller_material, "model", "mutated-material")
  object.__setattr__(caller_parameter, "value", (99,))
  object.__setattr__(caller_region, "field_ids", ("mutated",))
  object.__setattr__(caller_cell_ref, "block_id", "mutated-block")

  assert normalized.source.source == "model"
  assert len(normalized.mesh.nodes) == 4
  assert normalized.mesh.nodes[0].coordinates == (0.0, 0.0)
  assert normalized.mesh.cell_blocks[0].reference_topology == "quadrilateral"
  assert normalized.mesh.cell_blocks[0].cells[0].node_ids == (1, 2, 3, 4)
  assert normalized.fields[0].components == ("x", "y")
  assert normalized.materials[0].model == "linear-elastic"
  assert normalized.materials[0].parameters[0].value == (
    "label",
    (1, 2.0, True),
  )
  assert normalized.regions[0].field_ids == ("displacement",)
  assert normalized.regions[0].cell_refs[0].block_id == "quad-cells"


def test_mutable_hash_string_subclass_rejects_without_hostile_calls() -> None:
  hostile = _MutableHashStr("mutable-node")
  node = NodeSpec(
    id=hostile,
    coordinates=(0.0, 0.0),
    source=_source("nodes:mutable-hash"),
  )
  model = _valid_model()
  object.__setattr__(model.mesh, "nodes", (node, *model.mesh.nodes[1:]))

  with pytest.raises(ModelSpecValidationError) as caught:
    normalize_model_spec(model)

  assert _diagnostic_codes(caught.value) == ("invalid-node-id",)
  assert caught.value.diagnostics[0].source == _source("nodes:mutable-hash")
  assert hostile.strip_calls == 0
  assert hostile.hash_calls == 0


def test_unhashable_string_subclass_rejects_without_strip_or_lookup() -> None:
  hostile = _UnhashableStr("unhashable-node")
  node = NodeSpec(
    id=hostile,
    coordinates=(0.0, 0.0),
    source=_source("nodes:unhashable"),
  )
  model = _valid_model()
  object.__setattr__(model.mesh, "nodes", (node, *model.mesh.nodes[1:]))

  with pytest.raises(ModelSpecValidationError) as caught:
    normalize_model_spec(model)

  assert _diagnostic_codes(caught.value) == ("invalid-node-id",)
  assert caught.value.diagnostics[0].source == _source("nodes:unhashable")
  assert hostile.strip_calls == 0


def test_repr_bomb_string_subclass_rejects_without_rendering() -> None:
  hostile = _ReprBombStr("duplicate")
  first = NodeSpec(id="duplicate", coordinates=(2.0, 0.0))
  second = NodeSpec(
    id=hostile,
    coordinates=(3.0, 0.0),
    source=_source("nodes:repr-bomb"),
  )
  model = _valid_model()
  object.__setattr__(model.mesh, "nodes", (*model.mesh.nodes, first, second))

  with pytest.raises(ModelSpecValidationError) as caught:
    normalize_model_spec(model)

  assert _diagnostic_codes(caught.value) == ("invalid-node-id",)
  assert caught.value.diagnostics[0].source == _source("nodes:repr-bomb")
  assert hostile.strip_calls == 0
  assert hostile.repr_calls == 0


def test_comparison_bomb_integer_subclass_rejects_without_comparison() -> None:
  hostile = _ComparisonBombInt(2)
  block = replace(
    _quad_block(),
    topological_dimension=hostile,
    source=_source("blocks:comparison-bomb"),
  )
  model = _valid_model()
  object.__setattr__(model.mesh, "cell_blocks", (block,))

  with pytest.raises(ModelSpecValidationError) as caught:
    normalize_model_spec(model)

  assert _diagnostic_codes(caught.value) == ("invalid-topological-dimension",)
  assert caught.value.diagnostics[0].source == _source("blocks:comparison-bomb")
  assert hostile.comparison_calls == 0


def test_hostile_text_and_source_scalars_reject_at_trusted_parent() -> None:
  hostile_text = _ReprBombStr("quadrilateral")
  hostile_source = _ReprBombStr("nodes:hostile-source")
  block = replace(_quad_block(), reference_topology=hostile_text)
  node = _valid_model().mesh.nodes[0]
  object.__setattr__(node.source, "source", hostile_source)
  model = _valid_model()
  object.__setattr__(model.mesh, "nodes", (node, *model.mesh.nodes[1:]))
  object.__setattr__(model.mesh, "cell_blocks", (block,))

  with pytest.raises(ModelSpecValidationError) as caught:
    normalize_model_spec(model)

  assert _diagnostic_codes(caught.value) == (
    "invalid-source-context-value",
    "invalid-reference-topology",
  )
  assert caught.value.diagnostics[0].source == _source("mesh")
  assert caught.value.diagnostics[1].source == _source("blocks:quad-cells")
  assert hostile_text.strip_calls == 0
  assert hostile_text.repr_calls == 0
  assert hostile_source.strip_calls == 0
  assert hostile_source.repr_calls == 0


def test_source_context_integer_subclass_rejects_without_comparison() -> None:
  hostile_line = _ComparisonBombInt(7)
  node = _valid_model().mesh.nodes[0]
  object.__setattr__(
    node,
    "source",
    SourceContext(source="nodes:hostile-line", line=hostile_line),
  )
  model = _valid_model()
  object.__setattr__(model.mesh, "nodes", (node, *model.mesh.nodes[1:]))

  with pytest.raises(ModelSpecValidationError) as caught:
    normalize_model_spec(model)

  assert _diagnostic_codes(caught.value) == ("invalid-source-context-value",)
  assert caught.value.diagnostics[0].source == _source("mesh")
  assert hostile_line.comparison_calls == 0


def test_exact_uninitialized_specs_never_escape_raw_slot_errors() -> None:
  uninitialized_model = object.__new__(ModelSpec)
  mesh_model = _valid_model()
  object.__setattr__(mesh_model, "mesh", object.__new__(MeshSpec))
  node_model = _valid_model()
  object.__setattr__(
    node_model.mesh,
    "nodes",
    (object.__new__(NodeSpec), *node_model.mesh.nodes[1:]),
  )
  source_model = _valid_model()
  object.__setattr__(
    source_model.mesh.nodes[0],
    "source",
    object.__new__(SourceContext),
  )
  cases = (
    (uninitialized_model, "<python>"),
    (mesh_model, "model"),
    (node_model, "mesh"),
    (source_model, "mesh"),
  )

  for model, expected_source in cases:
    with pytest.raises(ModelSpecValidationError) as first:
      normalize_model_spec(model)
    with pytest.raises(ModelSpecValidationError) as second:
      normalize_model_spec(model)

    assert first.value.diagnostics == second.value.diagnostics
    assert first.value.diagnostics[0].source == _source(expected_source)
    assert str(first.value) == str(second.value)


def test_wrong_slot_and_collection_shapes_reject_before_iteration() -> None:
  model_fields = _valid_model()
  _forge_slot(model_fields, "fields", object())
  model_materials = _valid_model()
  _forge_slot(model_materials, "materials", [])
  mesh_nodes = _valid_model()
  _forge_slot(mesh_nodes.mesh, "nodes", object())
  mesh_blocks = _valid_model()
  _forge_slot(mesh_blocks.mesh, "cell_blocks", [])
  node_coordinates = _valid_model()
  _forge_slot(node_coordinates.mesh.nodes[0], "coordinates", [0.0, 0.0])
  block_cells = _valid_model()
  _forge_slot(block_cells.mesh.cell_blocks[0], "cells", object())
  cell_nodes = _valid_model()
  _forge_slot(
    cell_nodes.mesh.cell_blocks[0].cells[0],
    "node_ids",
    [1, 2, 3, 4],
  )
  field_components = _valid_model()
  _forge_slot(field_components.fields[0], "components", object())
  material_parameters = _valid_model()
  _forge_slot(material_parameters.materials[0], "parameters", [])
  parameter_value = _valid_model()
  _forge_slot(parameter_value.materials[0].parameters[0], "value", [1.0])
  region_cells = _valid_model()
  _forge_slot(region_cells.regions[0], "cell_refs", object())
  region_fields = _valid_model()
  _forge_slot(region_fields.regions[0], "field_ids", ["displacement"])
  cases = (
    (model_fields, "invalid-model-spec-value", "model"),
    (model_materials, "invalid-model-spec-value", "model"),
    (mesh_nodes, "invalid-mesh-spec-value", "mesh"),
    (mesh_blocks, "invalid-mesh-spec-value", "mesh"),
    (node_coordinates, "invalid-node-spec-value", "nodes:1"),
    (block_cells, "invalid-cell-block-spec-value", "blocks:quad-cells"),
    (cell_nodes, "invalid-cell-spec-value", "cells:10"),
    (field_components, "invalid-field-spec-value", "fields:displacement"),
    (
      material_parameters,
      "invalid-material-spec-value",
      "materials:solid",
    ),
    (
      parameter_value,
      "invalid-material-parameter-value",
      "materials:solid:E",
    ),
    (region_cells, "invalid-region-spec-value", "regions:domain"),
    (region_fields, "invalid-region-spec-value", "regions:domain"),
  )

  for model, expected_code, expected_source in cases:
    with pytest.raises(ModelSpecValidationError) as first:
      normalize_model_spec(model)
    with pytest.raises(ModelSpecValidationError) as second:
      normalize_model_spec(model)

    assert _diagnostic_codes(first.value) == (expected_code,)
    assert first.value.diagnostics == second.value.diagnostics
    assert first.value.diagnostics[0].source == _source(expected_source)


def test_authored_constructors_defer_malformed_values_to_normalization() -> None:
  base = _valid_model()
  bad_model_container = ModelSpec(
    mesh=base.mesh,
    fields=object(),
    materials=base.materials,
    regions=base.regions,
    source=_source("model:bad-container"),
  )
  bad_node = NodeSpec(
    id=1,
    coordinates=object(),
    source=_source("nodes:bad-container"),
  )
  bad_node_model = replace(
    base,
    mesh=replace(base.mesh, nodes=(bad_node, *base.mesh.nodes[1:])),
  )
  cyclic_value: list[object] = []
  cyclic_value.append(cyclic_value)
  cyclic_parameter = MaterialParameterSpec(
    name="cyclic",
    value=cyclic_value,
    source=_source("materials:solid:cyclic"),
  )
  cyclic_model = replace(
    base,
    materials=(replace(_material(), parameters=(cyclic_parameter,)),),
  )
  cases = (
    (bad_model_container, "invalid-model-spec-value", "model:bad-container"),
    (bad_node_model, "invalid-node-spec-value", "nodes:bad-container"),
    (
      cyclic_model,
      "invalid-material-parameter-value",
      "materials:solid:cyclic",
    ),
  )

  for model, expected_code, expected_source in cases:
    with pytest.raises(ModelSpecValidationError) as caught:
      normalize_model_spec(model)

    assert _diagnostic_codes(caught.value) == (expected_code,)
    assert caught.value.diagnostics[0].source == _source(expected_source)


def test_foreign_nested_objects_fail_before_child_attribute_access() -> None:
  cases = (
    ("mesh", "invalid-mesh-spec-type", "model"),
    ("node", "invalid-node-spec-type", "mesh"),
    ("cell-block", "invalid-cell-block-spec-type", "mesh"),
    ("cell", "invalid-cell-spec-type", "blocks:quad-cells"),
    ("field", "invalid-field-spec-type", "model"),
    ("material", "invalid-material-spec-type", "model"),
    (
      "material-parameter",
      "invalid-material-parameter-spec-type",
      "materials:solid",
    ),
    ("region", "invalid-region-spec-type", "model"),
    ("cell-ref", "invalid-cell-ref-type", "regions:domain"),
  )

  for family, expected_code, expected_source in cases:
    for value in (_MutableDuck(), _MissingAttributes()):
      model = _model_with_nested_value(family, value)
      with pytest.raises(ModelSpecValidationError) as first:
        normalize_model_spec(model)
      with pytest.raises(ModelSpecValidationError) as second:
        normalize_model_spec(model)

      assert _diagnostic_codes(first.value) == (expected_code,)
      assert first.value.diagnostics == second.value.diagnostics
      assert first.value.diagnostics[0].source == _source(expected_source)


def test_nested_subclasses_are_not_canonical_spec_values() -> None:
  subclass_node = _NodeSpecSubclass(id=1, coordinates=(0.0, 0.0))
  model = _model_with_nested_value("node", subclass_node)

  with pytest.raises(ModelSpecValidationError) as caught:
    normalize_model_spec(model)

  assert _diagnostic_codes(caught.value) == ("invalid-node-spec-type",)
  assert caught.value.diagnostics[0].source == _source("mesh")


def test_foreign_nested_diagnostic_order_is_stable() -> None:
  base = _valid_model()
  model = replace(
    base,
    fields=(_MutableDuck(), _MissingAttributes()),
    materials=(_MutableDuck(),),
    regions=(_MissingAttributes(),),
  )

  with pytest.raises(ModelSpecValidationError) as first:
    normalize_model_spec(model)
  with pytest.raises(ModelSpecValidationError) as second:
    normalize_model_spec(model)

  assert _diagnostic_codes(first.value) == (
    "invalid-field-spec-type",
    "invalid-field-spec-type",
    "invalid-material-spec-type",
    "invalid-region-spec-type",
  )
  assert first.value.diagnostics == second.value.diagnostics


def test_malformed_source_context_uses_nearest_trusted_parent() -> None:
  base = _valid_model()
  malformed_source = SourceContext(source=[])
  node = replace(
    base.mesh.nodes[0],
    source=malformed_source,
  )
  model = replace(
    base,
    mesh=replace(base.mesh, nodes=(node, *base.mesh.nodes[1:])),
  )

  with pytest.raises(ModelSpecValidationError) as caught:
    normalize_model_spec(model)

  assert _diagnostic_codes(caught.value) == ("invalid-source-context-value",)
  assert caught.value.diagnostics[0].source == _source("mesh")


def test_duplicate_ids_fail_deterministically_with_source_context() -> None:
  base = _valid_model()
  cases = (
    (
      replace(
        base,
        mesh=replace(
          base.mesh,
          nodes=(
            *base.mesh.nodes,
            NodeSpec(
              id=1,
              coordinates=(2.0, 2.0),
              source=_source("nodes:duplicate"),
            ),
          ),
        ),
      ),
      "duplicate-node-id",
      "nodes:duplicate",
    ),
    (
      replace(
        base,
        mesh=replace(
          base.mesh,
          cell_blocks=(
            *base.mesh.cell_blocks,
            replace(
              _quad_block(),
              source=_source("blocks:duplicate"),
            ),
          ),
        ),
      ),
      "duplicate-cell-block-id",
      "blocks:duplicate",
    ),
    (
      replace(
        base,
        mesh=replace(
          base.mesh,
          cell_blocks=(
            replace(
              _quad_block(),
              cells=(
                *_quad_block().cells,
                replace(
                  _quad_block().cells[0],
                  source=_source("cells:duplicate"),
                ),
              ),
            ),
          ),
        ),
      ),
      "duplicate-cell-id",
      "cells:duplicate",
    ),
    (
      replace(
        base,
        fields=(
          *base.fields,
          replace(_field(), source=_source("fields:duplicate")),
        ),
      ),
      "duplicate-field-id",
      "fields:duplicate",
    ),
    (
      replace(
        base,
        materials=(
          *base.materials,
          replace(_material(), source=_source("materials:duplicate")),
        ),
      ),
      "duplicate-material-id",
      "materials:duplicate",
    ),
    (
      replace(
        base,
        regions=(
          *base.regions,
          replace(_region(), source=_source("regions:duplicate")),
        ),
      ),
      "duplicate-region-id",
      "regions:duplicate",
    ),
  )

  for model, expected_code, expected_source in cases:
    with pytest.raises(ModelSpecValidationError) as first:
      normalize_model_spec(model)
    with pytest.raises(ModelSpecValidationError) as second:
      normalize_model_spec(model)

    assert str(first.value) == str(second.value)
    assert first.value.diagnostics == second.value.diagnostics
    assert expected_code in _diagnostic_codes(first.value)
    assert expected_source in str(first.value)


def test_invalid_ids_fail_deterministically_with_source_context() -> None:
  base = _valid_model()
  model = replace(
    base,
    mesh=replace(
      base.mesh,
      nodes=(
        *base.mesh.nodes,
        NodeSpec(
          id="  ",
          coordinates=(2.0, 2.0),
          source=_source("nodes:invalid-id"),
        ),
      ),
    ),
  )

  with pytest.raises(ModelSpecValidationError) as first:
    normalize_model_spec(model)
  with pytest.raises(ModelSpecValidationError) as second:
    normalize_model_spec(model)

  assert _diagnostic_codes(first.value) == ("invalid-node-id",)
  assert first.value.diagnostics == second.value.diagnostics
  assert "nodes:invalid-id" in str(first.value)


def test_bad_connectivity_fails_deterministically_with_source_context() -> None:
  base = _valid_model()
  malformed = replace(
    base.mesh.cell_blocks[0],
    cells=(
      CellSpec(
        id=10,
        node_ids=(1, 2, 3, 4),
        source=_source("cells:10"),
      ),
      CellSpec(
        id=11,
        node_ids=(1, 2, 3),
        source=_source("cells:bad-arity"),
      ),
      CellSpec(
        id=12,
        node_ids=(1, 2, 3, 99),
        source=_source("cells:unknown-node"),
      ),
    ),
  )
  model = replace(base, mesh=replace(base.mesh, cell_blocks=(malformed,)))

  with pytest.raises(ModelSpecValidationError) as caught:
    normalize_model_spec(model)

  assert _diagnostic_codes(caught.value) == (
    "connectivity-arity",
    "unknown-node-reference",
  )
  assert "cells:bad-arity" in str(caught.value)
  assert "cells:unknown-node" in str(caught.value)


def test_empty_connectivity_fails_with_source_context() -> None:
  base = _valid_model()
  empty_block = replace(
    _quad_block(),
    cells=(
      CellSpec(
        id=10,
        node_ids=(),
        source=_source("cells:empty-connectivity"),
      ),
    ),
  )
  model = replace(
    base,
    mesh=replace(base.mesh, cell_blocks=(empty_block,)),
  )

  with pytest.raises(ModelSpecValidationError) as caught:
    normalize_model_spec(model)

  assert _diagnostic_codes(caught.value) == ("empty-connectivity",)
  assert "cells:empty-connectivity" in str(caught.value)


def test_invalid_region_references_fail_deterministically_with_source_context() -> None:
  base = _valid_model()
  cases = (
    (
      replace(
        _region(),
        cell_refs=(CellRef(block_id="missing", cell_id=10),),
        source=_source("regions:unknown-block"),
      ),
      "unknown-cell-block-reference",
      "regions:unknown-block",
    ),
    (
      replace(
        _region(),
        cell_refs=(CellRef(block_id="quad-cells", cell_id=99),),
        source=_source("regions:unknown-cell"),
      ),
      "unknown-cell-reference",
      "regions:unknown-cell",
    ),
    (
      replace(
        _region(),
        field_ids=("missing",),
        source=_source("regions:unknown-field"),
      ),
      "unknown-field-reference",
      "regions:unknown-field",
    ),
    (
      replace(
        _region(),
        material_id="missing",
        source=_source("regions:unknown-material"),
      ),
      "unknown-material-reference",
      "regions:unknown-material",
    ),
  )

  for region, expected_code, expected_source in cases:
    model = replace(base, regions=(region,))
    with pytest.raises(ModelSpecValidationError) as caught:
      normalize_model_spec(model)

    assert _diagnostic_codes(caught.value) == (expected_code,)
    assert expected_source in str(caught.value)


def test_unhashable_runtime_reference_ids_fail_with_source_context() -> None:
  base = _valid_model()
  unhashable_node = replace(
    base,
    mesh=replace(
      base.mesh,
      cell_blocks=(
        replace(
          _quad_block(),
          cells=(
            CellSpec(
              id=10,
              node_ids=([1], 2, 3, 4),
              source=_source("cells:unhashable-node"),
            ),
          ),
        ),
      ),
    ),
  )
  cases = (
    (
      unhashable_node,
      "invalid-node-reference",
      "cells:unhashable-node",
    ),
    (
      replace(
        base,
        regions=(
          replace(
            _region(),
            cell_refs=(CellRef(block_id=["quad-cells"], cell_id=10),),
            source=_source("regions:unhashable-block"),
          ),
        ),
      ),
      "invalid-cell-block-reference",
      "regions:unhashable-block",
    ),
    (
      replace(
        base,
        regions=(
          replace(
            _region(),
            cell_refs=(CellRef(block_id="quad-cells", cell_id=[10]),),
            source=_source("regions:unhashable-cell"),
          ),
        ),
      ),
      "invalid-cell-reference",
      "regions:unhashable-cell",
    ),
    (
      replace(
        base,
        regions=(
          replace(
            _region(),
            field_ids=(["displacement"],),
            source=_source("regions:unhashable-field"),
          ),
        ),
      ),
      "invalid-field-reference",
      "regions:unhashable-field",
    ),
  )

  for model, expected_code, expected_source in cases:
    with pytest.raises(ModelSpecValidationError) as first:
      normalize_model_spec(model)
    with pytest.raises(ModelSpecValidationError) as second:
      normalize_model_spec(model)

    assert _diagnostic_codes(first.value) == (expected_code,)
    assert first.value.diagnostics == second.value.diagnostics
    assert expected_source in str(first.value)


def test_topology_embedding_collisions_fail_deterministically_with_source_context() -> (
  None
):
  base = _valid_model()
  invalid_topology = replace(
    _quad_block(),
    topological_dimension=3,
    embedding_dimension=2,
    source=_source("blocks:topology-over-embedding"),
  )
  invalid_embedding = replace(
    _quad_block(),
    embedding_dimension=3,
    source=_source("blocks:coordinate-embedding-mismatch"),
  )
  topology_collision = replace(
    _quad_block(),
    id="edge-cells",
    topological_dimension=1,
    geometry_interpolation="line4",
    cells=(CellSpec(id=20, node_ids=(1, 2, 3, 4)),),
    source=_source("blocks:topology-name-collision"),
  )
  cases = (
    (
      replace(base, mesh=replace(base.mesh, cell_blocks=(invalid_topology,))),
      "invalid-topology-embedding",
      "blocks:topology-over-embedding",
    ),
    (
      replace(base, mesh=replace(base.mesh, cell_blocks=(invalid_embedding,))),
      "embedding-dimension-mismatch",
      "blocks:coordinate-embedding-mismatch",
    ),
    (
      replace(
        base,
        mesh=replace(
          base.mesh,
          cell_blocks=(_quad_block(), topology_collision),
        ),
      ),
      "reference-topology-collision",
      "blocks:topology-name-collision",
    ),
  )

  for model, expected_code, expected_source in cases:
    with pytest.raises(ModelSpecValidationError) as caught:
      normalize_model_spec(model)

    assert expected_code in _diagnostic_codes(caught.value)
    assert expected_source in str(caught.value)


def test_interpolation_arity_compatibility_is_registry_owned_in_either_order() -> None:
  base = _valid_model()
  short_block = replace(
    _quad_block(),
    id="short-quad-cells",
    cells=(CellSpec(id=20, node_ids=(1, 2, 3)),),
    source=_source("blocks:short-quad-cells"),
  )

  for blocks, expected_arities in (
    ((_quad_block(), short_block), (4, 3)),
    ((short_block, _quad_block()), (3, 4)),
  ):
    model = replace(base, mesh=replace(base.mesh, cell_blocks=blocks))
    normalized = normalize_model_spec(model)

    assert (
      tuple(len(block.cells[0].node_ids) for block in normalized.mesh.cell_blocks)
      == expected_arities
    )


def test_cell_identity_is_block_local_even_when_ids_match() -> None:
  base = _valid_model()
  triangle = CellBlockSpec(
    id="tri-cells",
    reference_topology="triangle",
    topological_dimension=2,
    embedding_dimension=2,
    geometry_interpolation="tri3",
    cells=(
      CellSpec(
        id=10,
        node_ids=(1, 2, 3),
        source=_source("cells:tri:10"),
      ),
    ),
    source=_source("blocks:tri-cells"),
  )
  region = replace(
    _region(),
    cell_refs=(
      CellRef(block_id="quad-cells", cell_id=10),
      CellRef(block_id="tri-cells", cell_id=10),
    ),
  )
  model = replace(
    base,
    mesh=replace(base.mesh, cell_blocks=(_quad_block(), triangle)),
    regions=(region,),
  )

  normalized = normalize_model_spec(model)

  assert normalized.regions[0].cell_refs == (
    CellRef(block_id="quad-cells", cell_id=10),
    CellRef(block_id="tri-cells", cell_id=10),
  )


def test_ordinary_multiple_defect_diagnostic_order_remains_stable() -> None:
  base = _valid_model()
  duplicate_node = NodeSpec(
    id=1,
    coordinates=(math.nan, 0.0),
    source=_source("nodes:duplicate-invalid"),
  )
  bad_cell = CellSpec(
    id=11,
    node_ids=(1, 2, 99),
    source=_source("cells:bad"),
  )
  block = replace(_quad_block(), cells=(*_quad_block().cells, bad_cell))
  field = replace(_field(), components=("x", "x"))
  material = replace(
    _material(),
    parameters=(
      MaterialParameterSpec(name="p", value=1.0),
      MaterialParameterSpec(name="p", value=2.0),
    ),
  )
  region = replace(
    _region(),
    field_ids=("displacement", "missing"),
    material_id="missing",
  )
  model = replace(
    base,
    mesh=replace(
      base.mesh,
      nodes=(*base.mesh.nodes, duplicate_node),
      cell_blocks=(block,),
    ),
    fields=(field,),
    materials=(material,),
    regions=(region,),
  )

  with pytest.raises(ModelSpecValidationError) as first:
    normalize_model_spec(model)
  with pytest.raises(ModelSpecValidationError) as second:
    normalize_model_spec(model)

  assert _diagnostic_codes(first.value) == (
    "duplicate-node-id",
    "invalid-coordinate",
    "connectivity-arity",
    "unknown-node-reference",
    "duplicate-field-component",
    "duplicate-material-parameter-name",
    "unknown-field-reference",
    "unknown-material-reference",
  )
  assert first.value.diagnostics == second.value.diagnostics


def test_zero_dimensional_point_topology_is_valid_in_positive_embedding() -> None:
  base = _valid_model()
  point_block = replace(
    _quad_block(),
    id="point-cells",
    reference_topology="point",
    topological_dimension=0,
    geometry_interpolation="point1",
    cells=(CellSpec(id=20, node_ids=(1,)),),
    source=_source("blocks:point-cells"),
  )
  point_region = replace(
    _region(),
    cell_refs=(CellRef(block_id="point-cells", cell_id=20),),
  )
  model = replace(
    base,
    mesh=replace(base.mesh, cell_blocks=(point_block,)),
    regions=(point_region,),
  )

  normalized = normalize_model_spec(model)

  assert normalized.mesh.cell_blocks[0].topological_dimension == 0
  assert normalized.mesh.cell_blocks[0].embedding_dimension == 2


@pytest.mark.parametrize("dimension", (-1, False, 1.5))
def test_invalid_topological_dimensions_still_fail(dimension: object) -> None:
  base = _valid_model()
  invalid_block = replace(
    _quad_block(),
    topological_dimension=dimension,
    source=_source("blocks:invalid-topological-dimension"),
  )
  model = replace(
    base,
    mesh=replace(base.mesh, cell_blocks=(invalid_block,)),
  )

  with pytest.raises(ModelSpecValidationError) as caught:
    normalize_model_spec(model)

  assert _diagnostic_codes(caught.value) == ("invalid-topological-dimension",)
  assert "blocks:invalid-topological-dimension" in str(caught.value)


@pytest.mark.parametrize("dimension", (0, -1, False, 1.5))
def test_embedding_dimension_must_remain_positive(dimension: object) -> None:
  base = _valid_model()
  invalid_block = replace(
    _quad_block(),
    embedding_dimension=dimension,
    source=_source("blocks:invalid-embedding-dimension"),
  )
  model = replace(
    base,
    mesh=replace(base.mesh, cell_blocks=(invalid_block,)),
  )

  with pytest.raises(ModelSpecValidationError) as caught:
    normalize_model_spec(model)

  assert _diagnostic_codes(caught.value) == ("invalid-embedding-dimension",)
  assert "blocks:invalid-embedding-dimension" in str(caught.value)
