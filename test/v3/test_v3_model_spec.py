# SPDX-License-Identifier: MIT

"""Dangerous-case contracts for authored and normalized model specifications."""

from __future__ import annotations

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


def _source(label: str) -> SourceContext:
  return SourceContext(source=label)


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
  interpolation_arity_collision = replace(
    _quad_block(),
    id="short-quad-cells",
    cells=(CellSpec(id=20, node_ids=(1, 2, 3)),),
    source=_source("blocks:interpolation-arity-collision"),
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
    (
      replace(
        base,
        mesh=replace(
          base.mesh,
          cell_blocks=(_quad_block(), interpolation_arity_collision),
        ),
      ),
      "geometry-interpolation-collision",
      "blocks:interpolation-arity-collision",
    ),
  )

  for model, expected_code, expected_source in cases:
    with pytest.raises(ModelSpecValidationError) as caught:
      normalize_model_spec(model)

    assert expected_code in _diagnostic_codes(caught.value)
    assert expected_source in str(caught.value)
