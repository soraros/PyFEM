"""Model builders: compose authored meshes and materials into model specs.

Each builder emits exactly the ``ModelSpec`` slice the corresponding in-tree
builder module consumes: one displacement field, one region covering every
cell of the mesh, and the family's qualified formulation and quadrature
conventions. Structural mismatches (wrong mesh family, wrong material model)
fail here with direct authoring-time messages; the full coded semantic checks
still run at compile time.
"""

from __future__ import annotations

from pyfem.v3.spec.diagnostics import SourceContext
from pyfem.v3.spec.model import (
  CellBlockSpec,
  CellRef,
  FieldSpec,
  MaterialSpec,
  MeshSpec,
  ModelSpec,
  RegionSpec,
  SpecId,
)


def _source(label: str) -> SourceContext:
  return SourceContext(source=label)


def _semantic_id(value: object, *, label: str) -> SpecId:
  if type(value) is str and value:
    return value
  if type(value) is int:
    return value
  msg = f"{label} must be a non-empty string or integer id"
  raise TypeError(msg)


def _single_block(mesh: object, *, builder: str) -> CellBlockSpec:
  if type(mesh) is not MeshSpec:
    msg = f"{builder} requires an exact MeshSpec from the authoring mesh helpers"
    raise TypeError(msg)
  if len(mesh.cell_blocks) != 1:
    msg = (
      f"{builder} requires a mesh with exactly one cell block, "
      f"got {len(mesh.cell_blocks)}"
    )
    raise ValueError(msg)
  return mesh.cell_blocks[0]


def _require_material(
  material: object,
  *,
  models: tuple[str, ...],
  helper: str,
) -> MaterialSpec:
  if type(material) is not MaterialSpec:
    msg = f"material must be an exact MaterialSpec from {helper}"
    raise TypeError(msg)
  if material.model not in models:
    quoted = " or ".join(repr(model) for model in models)
    msg = (
      f"this family requires a {quoted} material (use {helper}), "
      f"got model {material.model!r}"
    )
    raise ValueError(msg)
  return material


def _require_geometry(
  block: CellBlockSpec,
  *,
  reference_topology: str,
  topological_dimension: int,
  embedding_dimension: int,
  geometry_interpolation: str,
  builder: str,
  helper: str,
) -> None:
  actual = (
    block.reference_topology,
    block.topological_dimension,
    block.embedding_dimension,
    block.geometry_interpolation,
  )
  expected = (
    reference_topology,
    topological_dimension,
    embedding_dimension,
    geometry_interpolation,
  )
  if actual != expected:
    msg = (
      f"{builder} requires a {geometry_interpolation} mesh (use {helper}), got "
      f"reference_topology={block.reference_topology!r}, "
      f"topological_dimension={block.topological_dimension!r}, "
      f"embedding_dimension={block.embedding_dimension!r}, "
      f"geometry_interpolation={block.geometry_interpolation!r}"
    )
    raise ValueError(msg)


def _model(
  mesh: MeshSpec,
  *,
  material: MaterialSpec,
  field_id: SpecId,
  region_id: SpecId,
  formulation: str,
  quadrature: str,
  source: str,
) -> ModelSpec:
  block = mesh.cell_blocks[0]
  field = FieldSpec(
    id=field_id,
    components=("x", "y"),
    location="node",
    source=_source(f"{source}:field"),
  )
  region = RegionSpec(
    id=region_id,
    cell_refs=tuple(CellRef(block.id, cell.id) for cell in block.cells),
    field_ids=(field.id,),
    material_id=material.id,
    formulation=formulation,
    quadrature=quadrature,
    source=_source(f"{source}:region"),
  )
  return ModelSpec(
    mesh=mesh,
    fields=(field,),
    materials=(material,),
    regions=(region,),
    source=_source(source),
  )


def small_strain_continuum(
  mesh: MeshSpec,
  *,
  material: MaterialSpec,
  field_id: SpecId = "displacement",
  region_id: SpecId = "domain",
  source: str = "authoring.small_strain_continuum",
) -> ModelSpec:
  """Compose a serendipity-quad8 mesh and continuum material into a model.

  The region covers every cell of the mesh with the qualified
  ``small-strain-continuum`` formulation on a ``gauss-3x3`` quadrature — one
  readable call for the classic teaching continuum. The material is the
  elastic ``linear_elastic(E, nu)`` slice or a stateful slice
  (``plasticity(E, nu, syield, hard)``, ``damage(E, nu, kappa0, kappac,
  k)``, ``prony_viscoelasticity(E, nu, Einf, n, tau_first, tau_last)``); a
  stateful law authors exactly like an elastic one.
  """
  block = _single_block(mesh, builder="small_strain_continuum")
  _require_geometry(
    block,
    reference_topology="quadrilateral",
    topological_dimension=2,
    embedding_dimension=2,
    geometry_interpolation="serendipity-quad8",
    builder="small_strain_continuum",
    helper="quad8_patch or quad8_mesh",
  )
  selected = _require_material(
    material,
    models=(
      "plane-stress-linear-elastic",
      "isotropic-hardening-plasticity",
      "plane-strain-damage",
      "prony-viscoelasticity",
    ),
    helper=(
      "linear_elastic(E, nu), plasticity(E, nu, syield, hard), "
      "damage(E, nu, kappa0, kappac, k), or "
      "prony_viscoelasticity(E, nu, Einf, n, tau_first, tau_last)"
    ),
  )
  return _model(
    mesh,
    material=selected,
    field_id=_semantic_id(field_id, label="field id"),
    region_id=_semantic_id(region_id, label="region id"),
    formulation="small-strain-continuum",
    quadrature="gauss-3x3",
    source=source,
  )


def truss(
  mesh: MeshSpec,
  *,
  material: MaterialSpec,
  field_id: SpecId = "displacement",
  region_id: SpecId = "domain",
  source: str = "authoring.truss",
) -> ModelSpec:
  """Compose a line2 mesh and uniaxial material into a truss model.

  The region covers every cell of the mesh with the qualified
  ``total-lagrangian-truss`` formulation (closed-form integration, no
  quadrature rule).
  """
  block = _single_block(mesh, builder="truss")
  _require_geometry(
    block,
    reference_topology="line",
    topological_dimension=1,
    embedding_dimension=2,
    geometry_interpolation="line2",
    builder="truss",
    helper="line2_mesh",
  )
  selected = _require_material(
    material,
    models=("uniaxial-linear-elastic",),
    helper="uniaxial_elastic(E, area)",
  )
  return _model(
    mesh,
    material=selected,
    field_id=_semantic_id(field_id, label="field id"),
    region_id=_semantic_id(region_id, label="region id"),
    formulation="total-lagrangian-truss",
    quadrature="none",
    source=source,
  )
