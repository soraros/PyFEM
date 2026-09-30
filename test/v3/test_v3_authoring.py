# SPDX-License-Identifier: MIT

"""Acceptance battery for the plain-Python authoring layer (M17 K3).

Covers the mesh/material/model builders, registry composition with field-level
metadata-mismatch diagnostics, byte-exact equivalence between layer-produced
and hand-written specs, and the student persona: a new material authored
end-to-end through the layer and verified against an analytic oracle.
"""

from __future__ import annotations

import sys
from dataclasses import replace

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from pyfem.v3 import authoring
from pyfem.v3.authoring.diagnostics import diff_manifest_fields
from pyfem.v3.compile.continuum import (
  Q8_MATERIAL_KEY,
  Q8_TOPOLOGY_KEY,
  q8_descriptor_metadata,
  q8_reference_registry,
)
from pyfem.v3.compile.diagnostics import ModelCompilationError
from pyfem.v3.compile.system import compile_system
from pyfem.v3.compile.truss import (
  TRUSS_MATERIAL_KEY,
  TRUSS_TOPOLOGY_KEY,
  truss_descriptor_metadata,
  truss_reference_registry,
)
from pyfem.v3.model.provenance import CanonicalManifest
from pyfem.v3.model.registry import RegistryDescriptor
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

_PATCH_COORDINATES = (
  (0.0, 0.0),
  (0.5, 0.0),
  (1.0, 0.0),
  (0.0, 0.5),
  (1.0, 0.5),
  (0.0, 1.0),
  (0.5, 1.0),
  (1.0, 1.0),
)
_PATCH_CONNECTIVITY = (1, 2, 3, 5, 8, 7, 6, 4)


def _source(label: str) -> SourceContext:
  return SourceContext(source=label)


# --- mesh helpers -----------------------------------------------------------


def test_quad8_patch_default_is_the_unit_square_teaching_mesh() -> None:
  mesh = authoring.quad8_patch()
  assert tuple(node.id for node in mesh.nodes) == (1, 2, 3, 4, 5, 6, 7, 8)
  for node, expected in zip(mesh.nodes, _PATCH_COORDINATES, strict=True):
    assert node.coordinates == expected
    assert node.source == _source(f"authoring.quad8_patch:node:{node.id}")
  (block,) = mesh.cell_blocks
  assert block.id == "cells"
  assert block.reference_topology == "quadrilateral"
  assert block.topological_dimension == 2
  assert block.embedding_dimension == 2
  assert block.geometry_interpolation == "serendipity-quad8"
  (cell,) = block.cells
  assert cell.id == "cell-1"
  assert cell.node_ids == _PATCH_CONNECTIVITY
  assert mesh.source == _source("authoring.quad8_patch")


def test_quad8_patch_grid_counts_and_shared_edges() -> None:
  mesh = authoring.quad8_patch(2, 1, width=2.0, height=1.0)
  assert len(mesh.nodes) == (2 * 2 + 1) * (2 * 1 + 1) - 2
  (block,) = mesh.cell_blocks
  assert tuple(cell.id for cell in block.cells) == ("cell-1", "cell-2")
  first, second = block.cells
  assert first.node_ids == (1, 2, 3, 7, 11, 10, 9, 6)
  assert second.node_ids == (3, 4, 5, 8, 13, 12, 11, 7)
  shared = {3, 7, 11}
  assert shared.issubset(first.node_ids)
  assert shared.issubset(second.node_ids)
  coordinates = {node.id: node.coordinates for node in mesh.nodes}
  assert coordinates[13] == (2.0, 1.0)
  assert coordinates[7] == (1.0, 0.5)


def test_quad8_patch_rejects_degenerate_grids() -> None:
  with pytest.raises(ValueError, match="nx"):
    authoring.quad8_patch(0, 1)
  with pytest.raises(ValueError, match="ny"):
    authoring.quad8_patch(1, -2)
  with pytest.raises(ValueError, match="width"):
    authoring.quad8_patch(1, 1, width=0.0)
  with pytest.raises(ValueError, match="height"):
    authoring.quad8_patch(1, 1, height=float("nan"))
  with pytest.raises(TypeError, match="coordinate pair"):
    authoring.quad8_patch(1, 1, origin=(0.0, 0.0, 0.0))


def test_line2_and_quad8_mesh_accept_plain_sequences_and_mappings() -> None:
  mesh = authoring.line2_mesh(
    [(-10.0, 0.0), (10.0, 0.0), (0.0, 0.5)],
    [(1, 3), (2, 3)],
    block_id="bars",
  )
  assert tuple(node.id for node in mesh.nodes) == (1, 2, 3)
  (block,) = mesh.cell_blocks
  assert block.id == "bars"
  assert block.reference_topology == "line"
  assert block.topological_dimension == 1
  assert block.geometry_interpolation == "line2"
  assert tuple(cell.id for cell in block.cells) == ("cell-1", "cell-2")
  assert tuple(cell.node_ids for cell in block.cells) == ((1, 3), (2, 3))

  mapped = authoring.quad8_mesh(
    {index: point for index, point in enumerate(_PATCH_COORDINATES, start=1)},
    {"quad": _PATCH_CONNECTIVITY},
  )
  (mapped_block,) = mapped.cell_blocks
  assert mapped_block.cells[0].id == "quad"
  assert mapped_block.cells[0].node_ids == _PATCH_CONNECTIVITY


def test_mesh_helpers_validate_arity_and_values() -> None:
  with pytest.raises(ValueError, match="exactly 2 nodes"):
    authoring.line2_mesh([(0.0, 0.0), (1.0, 0.0)], [(1, 2, 3)])
  with pytest.raises(ValueError, match="exactly 8 nodes"):
    authoring.quad8_mesh(list(_PATCH_COORDINATES), [(1, 2, 3)])
  with pytest.raises(ValueError, match="at least one node"):
    authoring.line2_mesh([], [(1, 2)])
  with pytest.raises(TypeError, match="coordinate pair"):
    authoring.line2_mesh([(0.0, 0.0, 0.0)], [])
  with pytest.raises(ValueError, match="finite"):
    authoring.line2_mesh([(0.0, float("inf")), (1.0, 0.0)], [(1, 2)])
  with pytest.raises(TypeError, match="node id"):
    authoring.line2_mesh({None: (0.0, 0.0), 2: (1.0, 0.0)}, [(None, 2)])


# --- material parameter slices ----------------------------------------------


def test_linear_elastic_authors_the_qualified_parameter_schema() -> None:
  material = authoring.linear_elastic(210.0e9, 0.3)
  assert material.id == "material"
  assert material.model == "plane-stress-linear-elastic"
  assert material.parameters == (
    MaterialParameterSpec(
      "youngs_modulus",
      210.0e9,
      _source("authoring.linear_elastic:youngs_modulus"),
    ),
    MaterialParameterSpec(
      "poisson_ratio",
      0.3,
      _source("authoring.linear_elastic:poisson_ratio"),
    ),
  )
  assert material.source == _source("authoring.linear_elastic")

  integral = authoring.linear_elastic(210000, 0, id="steel")
  assert integral.id == "steel"
  assert integral.parameters[0].value == 210000.0
  assert type(integral.parameters[0].value) is float


def test_material_helpers_reject_non_numeric_and_non_finite_values() -> None:
  with pytest.raises(TypeError, match="exact number"):
    authoring.linear_elastic(True, 0.3)
  with pytest.raises(TypeError, match="exact number"):
    authoring.linear_elastic(210.0e9, "0.3")
  with pytest.raises(ValueError, match="finite"):
    authoring.linear_elastic(float("nan"), 0.3)
  with pytest.raises(TypeError, match="material id"):
    authoring.linear_elastic(1.0, 0.0, id=None)
  with pytest.raises(ValueError, match="finite"):
    authoring.uniaxial_elastic(1.0, float("inf"))

  material = authoring.uniaxial_elastic(5.0e6, 1.0, id="steel")
  assert material.model == "uniaxial-linear-elastic"
  assert tuple(parameter.name for parameter in material.parameters) == (
    "youngs_modulus",
    "area",
  )


# --- model builders ----------------------------------------------------------


def test_small_strain_continuum_composes_the_qualified_slice() -> None:
  mesh = authoring.quad8_patch()
  material = authoring.linear_elastic(1.0, 0.0, id="elastic")
  model = authoring.small_strain_continuum(mesh, material=material)
  assert model.mesh is mesh
  assert model.materials == (material,)
  (field,) = model.fields
  assert field.id == "displacement"
  assert field.components == ("x", "y")
  assert field.location == "node"
  assert field.source == _source("authoring.small_strain_continuum:field")
  (region,) = model.regions
  assert region.id == "domain"
  assert region.cell_refs == (CellRef("cells", "cell-1"),)
  assert region.field_ids == ("displacement",)
  assert region.material_id == "elastic"
  assert region.formulation == "small-strain-continuum"
  assert region.quadrature == "gauss-3x3"
  assert model.source == _source("authoring.small_strain_continuum")


def test_truss_composes_the_qualified_slice() -> None:
  mesh = authoring.line2_mesh([(0.0, 0.0), (1.0, 0.0)], [(1, 2)])
  material = authoring.uniaxial_elastic(2.0e6, 0.5)
  model = authoring.truss(mesh, material=material)
  (region,) = model.regions
  assert region.formulation == "total-lagrangian-truss"
  assert region.quadrature == "none"
  assert region.cell_refs == (CellRef("cells", "cell-1"),)


def test_model_builders_reject_mismatched_families_with_direct_messages() -> None:
  line_mesh = authoring.line2_mesh([(0.0, 0.0), (1.0, 0.0)], [(1, 2)])
  with pytest.raises(ValueError, match="quad8_patch or quad8_mesh"):
    authoring.small_strain_continuum(
      line_mesh,
      material=authoring.linear_elastic(1.0, 0.0),
    )
  quad_mesh = authoring.quad8_patch()
  with pytest.raises(ValueError, match="line2_mesh"):
    authoring.truss(quad_mesh, material=authoring.uniaxial_elastic(1.0, 1.0))
  with pytest.raises(ValueError, match="linear_elastic"):
    authoring.small_strain_continuum(
      quad_mesh,
      material=authoring.uniaxial_elastic(1.0, 1.0),
    )
  with pytest.raises(ValueError, match="uniaxial_elastic"):
    authoring.truss(line_mesh, material=authoring.linear_elastic(1.0, 0.0))
  with pytest.raises(TypeError, match="exact MeshSpec"):
    authoring.small_strain_continuum(
      "not-a-mesh",
      material=authoring.linear_elastic(1.0, 0.0),
    )


# --- registry authors and composition ----------------------------------------


def _plane_stress_like_law(
  youngs_modulus: float,
  poisson_ratio: float,
) -> np.ndarray:
  c = youngs_modulus / (1.0 - poisson_ratio**2)
  shear = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
  return np.array(
    [
      [c, c * poisson_ratio, 0.0],
      [c * poisson_ratio, c, 0.0],
      [0.0, 0.0, shear],
    ],
  )


def _spoofed_descriptor(
  key: tuple[str, str],
  metadata: dict[str, object],
  *,
  binding: object,
) -> RegistryDescriptor:
  return RegistryDescriptor(
    kind=key[0],
    name=key[1],
    version="1",
    implementation_id="spoofed-for-diagnostics",
    metadata=metadata,
    binding=binding,  # type: ignore[arg-type]
  )


def test_law_descriptors_pin_the_convention_so_users_never_copy_metadata() -> None:
  law = authoring.plane_stress_law(
    _plane_stress_like_law,
    implementation_id="student-law-v1",
  )
  assert law.key == Q8_MATERIAL_KEY
  assert law.version == "1"
  assert law.implementation_id == "student-law-v1"
  assert (
    law.metadata.to_bytes()
    == CanonicalManifest(q8_descriptor_metadata(*Q8_MATERIAL_KEY)).to_bytes()
  )

  truss_law = authoring.uniaxial_law(
    lambda youngs_modulus, area: np.array([[youngs_modulus * area]]),
    implementation_id="student-truss-law-v1",
    version="2",
  )
  assert truss_law.key == TRUSS_MATERIAL_KEY
  assert truss_law.version == "2"
  assert (
    truss_law.metadata.to_bytes()
    == CanonicalManifest(truss_descriptor_metadata(*TRUSS_MATERIAL_KEY)).to_bytes()
  )


def test_layer_registries_are_byte_identical_to_reference_registries() -> None:
  for key, descriptor in q8_reference_registry().items():
    layer_descriptor = authoring.q8_registry()[key]
    assert layer_descriptor.manifest.to_bytes() == descriptor.manifest.to_bytes()
    assert layer_descriptor.metadata.to_bytes() == descriptor.metadata.to_bytes()
  for key, descriptor in truss_reference_registry().items():
    layer_descriptor = authoring.truss_registry()[key]
    assert layer_descriptor.manifest.to_bytes() == descriptor.manifest.to_bytes()
    assert layer_descriptor.metadata.to_bytes() == descriptor.metadata.to_bytes()


def test_registry_replacement_must_keep_the_convention_key() -> None:
  foreign = RegistryDescriptor(
    kind="material",
    name="my-own-material-name",
    version="1",
    implementation_id="foreign",
    metadata=q8_descriptor_metadata(*Q8_MATERIAL_KEY),
    binding=_plane_stress_like_law,
  )
  with pytest.raises(ValueError, match="does not match the qualified convention key"):
    authoring.q8_registry(material=foreign)
  with pytest.raises(TypeError, match="exact RegistryDescriptor"):
    authoring.q8_registry(material=_plane_stress_like_law)  # type: ignore[arg-type]


# --- field-level metadata-mismatch diagnostics --------------------------------


def test_q8_metadata_mismatch_names_every_disagreeing_field() -> None:
  metadata = q8_descriptor_metadata(*Q8_MATERIAL_KEY)
  metadata["stress_state"] = "plane-strain"
  metadata["parameter_names"] = ["youngs_modulus", "nu"]
  metadata["parameter_dtype"] = 1.0
  del metadata["tangent_class"]
  metadata["density"] = 7850.0
  spoofed = _spoofed_descriptor(
    Q8_MATERIAL_KEY,
    metadata,
    binding=_plane_stress_like_law,
  )

  model = authoring.small_strain_continuum(
    authoring.quad8_patch(),
    material=authoring.linear_elastic(1.0, 0.0),
  )
  registry = authoring.q8_registry()
  registry[Q8_MATERIAL_KEY] = spoofed
  with pytest.raises(ModelCompilationError) as captured:
    authoring.compile(model, registry)
  (diagnostic,) = captured.value.diagnostics
  assert diagnostic.code == "incompatible-registry-descriptor"
  assert "('material', 'plane-stress-linear-elastic')" in diagnostic.message
  assert (
    "field 'stress_state': expected 'plane-stress', authored 'plane-strain'"
    in diagnostic.message
  )
  assert (
    "field 'parameter_names[1]': expected 'poisson_ratio', authored 'nu'"
    in diagnostic.message
  )
  assert (
    "field 'parameter_dtype': expected 'float64', authored 1.0" in diagnostic.message
  )
  assert "missing field 'tangent_class'" in diagnostic.message
  assert "unexpected field 'density'" in diagnostic.message
  assert diagnostic.source == _source("authoring.linear_elastic")
  assert "authoring.linear_elastic: [incompatible-registry-descriptor]" in str(
    captured.value
  )


def test_metadata_mismatch_is_eager_when_composing_a_registry() -> None:
  metadata = q8_descriptor_metadata(*Q8_MATERIAL_KEY)
  metadata["stress_state"] = "plane-strain"
  spoofed = _spoofed_descriptor(
    Q8_MATERIAL_KEY,
    metadata,
    binding=_plane_stress_like_law,
  )
  with pytest.raises(ModelCompilationError) as captured:
    authoring.q8_registry(material=spoofed)
  (diagnostic,) = captured.value.diagnostics
  assert diagnostic.code == "incompatible-registry-descriptor"
  assert "field 'stress_state'" in diagnostic.message
  assert diagnostic.source == _source("authoring.q8_registry:material")


def test_truss_metadata_mismatch_diffs_nested_and_typed_fields() -> None:
  metadata = truss_descriptor_metadata(*TRUSS_TOPOLOGY_KEY)
  metadata["embedding_dimension"] = 3
  metadata["local_node_parent_coordinates"] = [[-1.0], [0.0]]
  spoofed = _spoofed_descriptor(
    TRUSS_TOPOLOGY_KEY,
    metadata,
    binding=truss_reference_registry()[TRUSS_TOPOLOGY_KEY].binding,
  )
  registry = authoring.truss_registry()
  registry[TRUSS_TOPOLOGY_KEY] = spoofed
  model = authoring.truss(
    authoring.line2_mesh([(0.0, 0.0), (1.0, 0.0)], [(1, 2)]),
    material=authoring.uniaxial_elastic(2.0e6, 0.5),
  )
  with pytest.raises(ModelCompilationError) as captured:
    authoring.compile(model, registry)
  (diagnostic,) = captured.value.diagnostics
  assert diagnostic.code == "incompatible-registry-descriptor"
  assert "field 'embedding_dimension': expected 2, authored 3" in diagnostic.message
  assert (
    "field 'local_node_parent_coordinates[1][0]': expected 1.0, authored 0.0"
    in diagnostic.message
  )
  assert diagnostic.source == _source("authoring.line2_mesh:block")


def test_manifest_diff_reports_sequence_length_and_handles_undecodable() -> None:
  metadata = q8_descriptor_metadata(*Q8_TOPOLOGY_KEY)
  metadata["parent_coordinates"] = ["xi"]
  (line,) = diff_manifest_fields(
    CanonicalManifest(q8_descriptor_metadata(*Q8_TOPOLOGY_KEY)),
    CanonicalManifest(metadata),
  )
  assert line == "field 'parent_coordinates': expected 2 entries, authored 1"


def test_landed_diagnostics_pass_through_unchanged() -> None:
  model = authoring.small_strain_continuum(
    authoring.quad8_patch(),
    material=authoring.linear_elastic(1.0, 0.0),
  )
  missing = authoring.q8_registry()
  del missing[Q8_MATERIAL_KEY]
  with pytest.raises(ModelCompilationError, match="registry-capture-failed"):
    authoring.compile(model, missing)

  def doubled_law(youngs_modulus: float, poisson_ratio: float) -> np.ndarray:
    return 2.0 * _plane_stress_like_law(youngs_modulus, poisson_ratio)

  with pytest.raises(ModelCompilationError, match="incompatible-material-binding"):
    authoring.compile(
      model,
      authoring.q8_registry(
        material=authoring.plane_stress_law(
          doubled_law,
          implementation_id="doubled-law",
        )
      ),
    )

  def wide_law(youngs_modulus: float, poisson_ratio: float) -> np.ndarray:
    del youngs_modulus, poisson_ratio
    return np.zeros((4, 4))

  with pytest.raises(ModelCompilationError, match="invalid-material-binding-output"):
    authoring.compile(
      model,
      authoring.q8_registry(
        material=authoring.plane_stress_law(
          wide_law,
          implementation_id="wide-law",
        )
      ),
    )


def test_check_registry_defers_specs_outside_the_landed_families() -> None:
  model = authoring.small_strain_continuum(
    authoring.quad8_patch(),
    material=authoring.linear_elastic(1.0, 0.0),
  )
  foreign = replace(
    model,
    regions=(replace(model.regions[0], formulation="mystery-formulation"),),
  )
  authoring.check_registry(foreign, authoring.q8_registry())
  with pytest.raises(ModelCompilationError, match="incompatible-formulation"):
    authoring.compile(foreign)
  with pytest.raises(TypeError, match="exact ModelSpec"):
    authoring.check_registry("not-a-model", authoring.q8_registry())  # type: ignore[arg-type]


# --- byte-exact equivalence with hand-written specs ---------------------------


def _handwritten_q8_model() -> ModelSpec:
  """The patch-test model written directly against the spec contracts.

  Labels deliberately match the authoring layer's deterministic source scheme,
  so the two values must compare equal field by field.
  """
  nodes = tuple(
    NodeSpec(
      id=index,
      coordinates=point,
      source=_source(f"authoring.quad8_patch:node:{index}"),
    )
    for index, point in enumerate(_PATCH_COORDINATES, start=1)
  )
  cell = CellSpec(
    id="cell-1",
    node_ids=_PATCH_CONNECTIVITY,
    source=_source("authoring.quad8_patch:cell:cell-1"),
  )
  block = CellBlockSpec(
    id="cells",
    reference_topology="quadrilateral",
    topological_dimension=2,
    embedding_dimension=2,
    geometry_interpolation="serendipity-quad8",
    cells=(cell,),
    source=_source("authoring.quad8_patch:block"),
  )
  mesh = MeshSpec(
    nodes=nodes,
    cell_blocks=(block,),
    source=_source("authoring.quad8_patch"),
  )
  field = FieldSpec(
    id="displacement",
    components=("x", "y"),
    location="node",
    source=_source("authoring.small_strain_continuum:field"),
  )
  material = MaterialSpec(
    id="elastic",
    model="plane-stress-linear-elastic",
    parameters=(
      MaterialParameterSpec(
        "youngs_modulus",
        1.0,
        _source("authoring.linear_elastic:youngs_modulus"),
      ),
      MaterialParameterSpec(
        "poisson_ratio",
        0.0,
        _source("authoring.linear_elastic:poisson_ratio"),
      ),
    ),
    source=_source("authoring.linear_elastic"),
  )
  region = RegionSpec(
    id="domain",
    cell_refs=(CellRef("cells", "cell-1"),),
    field_ids=("displacement",),
    material_id="elastic",
    formulation="small-strain-continuum",
    quadrature="gauss-3x3",
    source=_source("authoring.small_strain_continuum:region"),
  )
  return ModelSpec(
    mesh=mesh,
    fields=(field,),
    materials=(material,),
    regions=(region,),
    source=_source("authoring.small_strain_continuum"),
  )


def _handwritten_truss_model() -> ModelSpec:
  nodes = tuple(
    NodeSpec(
      id=index,
      coordinates=point,
      source=_source(f"authoring.line2_mesh:node:{index}"),
    )
    for index, point in enumerate(((-10.0, 0.0), (10.0, 0.0), (0.0, 0.5)), 1)
  )
  cells = tuple(
    CellSpec(
      id=cell_id,
      node_ids=node_ids,
      source=_source(f"authoring.line2_mesh:cell:{cell_id}"),
    )
    for cell_id, node_ids in (("cell-1", (1, 3)), ("cell-2", (2, 3)))
  )
  block = CellBlockSpec(
    id="bars",
    reference_topology="line",
    topological_dimension=1,
    embedding_dimension=2,
    geometry_interpolation="line2",
    cells=cells,
    source=_source("authoring.line2_mesh:block"),
  )
  mesh = MeshSpec(
    nodes=nodes,
    cell_blocks=(block,),
    source=_source("authoring.line2_mesh"),
  )
  field = FieldSpec(
    id="displacement",
    components=("x", "y"),
    location="node",
    source=_source("authoring.truss:field"),
  )
  material = MaterialSpec(
    id="steel",
    model="uniaxial-linear-elastic",
    parameters=(
      MaterialParameterSpec(
        "youngs_modulus",
        5.0e6,
        _source("authoring.uniaxial_elastic:youngs_modulus"),
      ),
      MaterialParameterSpec(
        "area",
        1.0,
        _source("authoring.uniaxial_elastic:area"),
      ),
    ),
    source=_source("authoring.uniaxial_elastic"),
  )
  region = RegionSpec(
    id="domain",
    cell_refs=(CellRef("bars", "cell-1"), CellRef("bars", "cell-2")),
    field_ids=("displacement",),
    material_id="steel",
    formulation="total-lagrangian-truss",
    quadrature="none",
    source=_source("authoring.truss:region"),
  )
  return ModelSpec(
    mesh=mesh,
    fields=(field,),
    materials=(material,),
    regions=(region,),
    source=_source("authoring.truss"),
  )


def test_q8_layer_output_is_byte_identical_to_handwritten_spec() -> None:
  layer_model = authoring.small_strain_continuum(
    authoring.quad8_patch(),
    material=authoring.linear_elastic(1.0, 0.0, id="elastic"),
  )
  hand_model = _handwritten_q8_model()
  assert layer_model == hand_model

  layer_system = authoring.compile(layer_model)
  hand_system = compile_system(hand_model, q8_reference_registry())
  assert layer_system.content_fingerprint == hand_system.content_fingerprint
  assert (
    layer_system.provenance.manifest.to_bytes()
    == hand_system.provenance.manifest.to_bytes()
  )
  assert (
    layer_system.registry_snapshot.manifest.to_bytes()
    == hand_system.registry_snapshot.manifest.to_bytes()
  )
  assert (
    layer_system.operators[0].content_manifest.to_bytes()
    == hand_system.operators[0].content_manifest.to_bytes()
  )

  repeat = authoring.compile(layer_model)
  assert repeat.content_fingerprint == layer_system.content_fingerprint
  assert repeat.instance_id != layer_system.instance_id
  assert layer_system.source_for("node", 1).source == "authoring.quad8_patch:node:1"
  assert (
    layer_system.source_for("material_parameter", ("elastic", "poisson_ratio")).source
    == "authoring.linear_elastic:poisson_ratio"
  )


def test_truss_layer_output_is_byte_identical_to_handwritten_spec() -> None:
  layer_model = authoring.truss(
    authoring.line2_mesh(
      [(-10.0, 0.0), (10.0, 0.0), (0.0, 0.5)],
      {"cell-1": (1, 3), "cell-2": (2, 3)},
      block_id="bars",
    ),
    material=authoring.uniaxial_elastic(5.0e6, 1.0, id="steel"),
  )
  hand_model = _handwritten_truss_model()
  assert layer_model == hand_model

  layer_system = authoring.compile(layer_model)
  hand_system = compile_system(hand_model, truss_reference_registry())
  assert layer_system.content_fingerprint == hand_system.content_fingerprint
  assert (
    layer_system.provenance.manifest.to_bytes()
    == hand_system.provenance.manifest.to_bytes()
  )
  assert (
    layer_system.registry_snapshot.manifest.to_bytes()
    == hand_system.registry_snapshot.manifest.to_bytes()
  )
  assert (
    layer_system.operators[0].content_manifest.to_bytes()
    == hand_system.operators[0].content_manifest.to_bytes()
  )


# --- student persona: a new material through the layer -------------------------


def student_plane_stress_law(
  youngs_modulus: float,
  poisson_ratio: float,
) -> np.ndarray:
  """Plane-stress constitutive matrix, transcribed from the lecture notes."""
  c = youngs_modulus / (1.0 - poisson_ratio**2)
  shear = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
  return np.array(
    [
      [c, c * poisson_ratio, 0.0],
      [c * poisson_ratio, c, 0.0],
      [0.0, 0.0, shear],
    ]
  )


def test_student_authors_a_new_material_end_to_end() -> None:
  """Persona: the whole flow touches only the authoring layer and NumPy.

  The material path is ~10 lines of user code: the law function above plus
  one descriptor line and one registry line below.
  """
  youngs_modulus, poisson_ratio = 210.0e9, 0.3
  mesh = authoring.quad8_patch()
  model = authoring.small_strain_continuum(
    mesh,
    material=authoring.linear_elastic(youngs_modulus, poisson_ratio, id="steel"),
  )
  law = authoring.plane_stress_law(
    student_plane_stress_law,
    implementation_id="student-plane-stress-2026",
  )
  system = authoring.compile(model, authoring.q8_registry(material=law))

  strain = 1.0e-3
  displacements = {node.id: (strain * node.coordinates[0], 0.0) for node in mesh.nodes}
  student_result = authoring.evaluate(system, displacements)

  # The student's law compiles bitwise-identical to the in-tree reference law.
  reference_result = authoring.evaluate(authoring.compile(model), displacements)
  np.testing.assert_array_equal(
    student_result.jacobian_values[0].values,
    reference_result.jacobian_values[0].values,
  )
  np.testing.assert_array_equal(
    student_result.residual_values[0].values,
    reference_result.residual_values[0].values,
  )

  # Numerical oracle: uniaxial strain gives the constant stress state
  # sigma = D @ (strain, 0, 0); the internal force vector of an exact linear
  # field equals the consistent nodal loads of that stress state, with the
  # serendipity edge fractions (1/6, 2/3, 1/6) per unit edge length.
  c = youngs_modulus / (1.0 - poisson_ratio**2)
  sigma_xx = c * strain
  sigma_yy = poisson_ratio * c * strain
  expected = np.array(
    [
      [-sigma_xx / 6.0, -sigma_yy / 6.0],
      [0.0, -2.0 * sigma_yy / 3.0],
      [sigma_xx / 6.0, -sigma_yy / 6.0],
      [2.0 * sigma_xx / 3.0, 0.0],
      [sigma_xx / 6.0, sigma_yy / 6.0],
      [0.0, 2.0 * sigma_yy / 3.0],
      [-sigma_xx / 6.0, sigma_yy / 6.0],
      [-2.0 * sigma_xx / 3.0, 0.0],
    ]
  )
  np.testing.assert_allclose(
    student_result.residual_values[0].values[0],
    expected.ravel(),
    rtol=1.0e-11,
    atol=1.0e-3,
  )
  # Equilibrium: the patch is self-equilibrated under the prescribed field.
  np.testing.assert_allclose(
    student_result.residual_values[0].values.sum(axis=1),
    0.0,
    rtol=0.0,
    atol=1.0e-4,
  )


# --- springs -------------------------------------------------------------------


def _q8_system() -> object:
  return authoring.compile(
    authoring.small_strain_continuum(
      authoring.quad8_patch(),
      material=authoring.linear_elastic(1.0, 0.0),
    )
  )


def test_damage_envelope_spring_compose_and_evaluate_oracle() -> None:
  system = authoring.damage_envelope_spring(
    _q8_system(),
    nodes=(1, 2),
    stiffness=2.0,
    critical_extension=2.0,
    max_increment=0.5,
  )
  assert len(system.operators) == 2
  layout = system.operators[1].header.state_layout
  assert layout.row_shape == (2, 1)
  assert layout.slots[0].name == "max_extension"

  result = authoring.evaluate(system, {1: (0.3, 0.0)}, operator=1)
  # Landed oracle from the state-transaction battery: kappa = 0.3 grows the
  # envelope to omega = 0.15, so the secant force is 0.85 * 2.0 * 0.3.
  np.testing.assert_allclose(
    result.residual_values[0].values[0],
    [0.51, 0.0],
    rtol=0.0,
    atol=1.0e-15,
  )
  np.testing.assert_allclose(
    result.jacobian_values[0].values[0],
    [[1.4, 0.0], [0.0, 1.7]],
    rtol=0.0,
    atol=1.0e-15,
  )
  np.testing.assert_array_equal(result.trial_state.values, [[0.3], [0.0]])


def student_spring_kernel(
  displacements: np.ndarray,
  accepted_rows: np.ndarray,
  parameters: np.ndarray,
) -> authoring.SpringKernelResult:
  """Plain linear spring that remembers its largest extension."""
  (stiffness,) = parameters
  trial = np.maximum(
    accepted_rows[:, 0],
    np.linalg.norm(displacements, axis=1),
  )
  return authoring.SpringKernelResult(
    force=stiffness * displacements,
    tangent=stiffness * np.tile(np.eye(2), (len(displacements), 1, 1)),
    trial_rows=trial[:, None],
    status=authoring.EvaluationStatus.OK,
  )


def test_student_spring_kernel_compiles_and_runs_with_state() -> None:
  system = authoring.spring(
    _q8_system(),
    nodes={"spring-a": 1, "spring-b": 2},
    kernel=student_spring_kernel,
    state=(("max_extension", 1),),
    parameters=(3.0,),
    name="linear-spring-with-memory",
    implementation_id="student-spring-v1",
  )
  operator = system.operators[1]
  assert operator.header.state_layout.row_shape == (2, 1)
  result = authoring.evaluate(
    system,
    {1: (0.5, 0.0), 2: (0.0, -0.25)},
    operator=1,
  )
  np.testing.assert_array_equal(
    result.residual_values[0].values,
    [[1.5, 0.0], [0.0, -0.75]],
  )
  np.testing.assert_array_equal(
    result.jacobian_values[0].values,
    3.0 * np.tile(np.eye(2), (2, 1, 1)),
  )
  np.testing.assert_array_equal(result.trial_state.values, [[0.5], [0.25]])


def test_spring_authoring_rejects_bad_inputs_early() -> None:
  base = _q8_system()
  with pytest.raises(ModelCompilationError, match="unknown-spring-support-node"):
    authoring.spring(
      base,
      nodes=(99,),
      kernel=student_spring_kernel,
      parameters=(1.0,),
      name="linear-spring-with-memory",
      implementation_id="student-spring-v1",
    )
  with pytest.raises(ValueError, match="at least one support node"):
    authoring.spring(
      base,
      nodes=(),
      kernel=student_spring_kernel,
      parameters=(1.0,),
      name="linear-spring-with-memory",
      implementation_id="student-spring-v1",
    )
  with pytest.raises(TypeError, match="kernel must be callable"):
    authoring.spring(
      base,
      nodes=(1,),
      kernel=3.0,  # type: ignore[arg-type]
      parameters=(1.0,),
      name="linear-spring-with-memory",
      implementation_id="student-spring-v1",
    )
  with pytest.raises(ValueError, match="positive finite"):
    authoring.damage_envelope_spring(
      base,
      nodes=(1,),
      stiffness=-1.0,
      critical_extension=1.0,
      max_increment=0.5,
    )
  with pytest.raises(ModelCompilationError, match="unknown-spring-space"):
    authoring.damage_envelope_spring(
      base,
      nodes=(1,),
      stiffness=1.0,
      critical_extension=1.0,
      max_increment=0.5,
      space_id="thermal",
    )


# --- evaluate helpers -----------------------------------------------------------


def test_trial_vector_maps_plain_displacements_onto_coefficients() -> None:
  system = _q8_system()
  values = authoring.trial_vector(system, {3: (1.0, -1.0)})
  assert values.shape == (system.coefficient_count,)
  coefficient_map = system.spaces[0].coefficient_map.values
  assert values[coefficient_map[2, 0]] == 1.0
  assert values[coefficient_map[2, 1]] == -1.0
  assert np.count_nonzero(values) == 2
  np.testing.assert_array_equal(authoring.trial_vector(system), np.zeros(16))

  with pytest.raises(ValueError, match="unknown node"):
    authoring.trial_vector(system, {99: (0.0, 0.0)})
  with pytest.raises(ValueError, match="exactly 2 components"):
    authoring.trial_vector(system, {1: (1.0,)})
  with pytest.raises(TypeError, match="map node ids"):
    authoring.trial_vector(system, [(1, (0.0, 0.0))])  # type: ignore[arg-type]
  with pytest.raises(ValueError, match="finite"):
    authoring.trial_vector(system, {1: (float("nan"), 0.0)})


def test_evaluate_zero_state_and_input_validation() -> None:
  system = _q8_system()
  result = authoring.evaluate(system)
  np.testing.assert_array_equal(
    result.residual_values[0].values,
    np.zeros((1, 16)),
  )
  assert result.jacobian_values[0].values.shape == (1, 16, 16)

  by_vector = authoring.evaluate(system, np.zeros(16, dtype=np.float64))
  np.testing.assert_array_equal(
    by_vector.residual_values[0].values,
    result.residual_values[0].values,
  )
  with pytest.raises(ValueError, match="index into"):
    authoring.evaluate(system, operator=5)
  with pytest.raises(ValueError, match="float64 array of shape"):
    authoring.evaluate(system, np.zeros(4, dtype=np.float64))
