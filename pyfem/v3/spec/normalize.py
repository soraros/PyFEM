"""Normalization and source-aware validation for authored model values."""

from __future__ import annotations

import math

from pyfem.v3.spec.diagnostics import (
  ModelSpecValidationError,
  SourceContext,
  SpecDiagnostic,
)
from pyfem.v3.spec.model import (
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
  SpecId,
)


class _Validator:
  def __init__(self) -> None:
    self.diagnostics: list[SpecDiagnostic] = []

  def error(self, code: str, message: str, source: SourceContext) -> None:
    self.diagnostics.append(SpecDiagnostic(code=code, message=message, source=source))

  def text(
    self,
    value: object,
    *,
    code: str,
    label: str,
    source: SourceContext,
  ) -> bool:
    if isinstance(value, str) and value:
      return True
    self.error(code, f"{label} must be a non-empty string", source)
    return False

  def identifier(
    self,
    value: SpecId,
    source: SourceContext,
    *,
    kind: str,
    registry: dict[SpecId, SourceContext],
  ) -> bool:
    if not _valid_id(value):
      label = kind.replace("-", " ")
      self.error(
        f"invalid-{kind}-id",
        f"{label} ID must be a non-empty string or integer",
        source,
      )
      return False
    first = registry.get(value)
    if first is not None:
      label = kind.replace("-", " ")
      self.error(
        f"duplicate-{kind}-id",
        f"duplicate {label} ID {value!r}; first declared at {first.render()}",
        source,
      )
      return False
    registry[value] = source
    return True


def _valid_id(value: object) -> bool:
  return (
    isinstance(value, str)
    and bool(value)
    or isinstance(value, int)
    and not isinstance(value, bool)
  )


def _positive_int(value: object) -> bool:
  return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _non_negative_int(value: object) -> bool:
  return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_exact_type(
  value: object,
  expected: type[object],
  *,
  code: str,
  label: str,
  source: SourceContext,
  validator: _Validator,
) -> bool:
  if type(value) is expected:
    return True
  validator.error(
    code,
    f"{label} must be exactly {expected.__name__}; got {type(value).__name__}",
    source,
  )
  return False


def _trusted_source(
  value: object,
  *,
  label: str,
  fallback: SourceContext,
  validator: _Validator,
) -> SourceContext:
  if type(value) is not SourceContext:
    validator.error(
      "invalid-source-context-type",
      f"{label} source must be exactly SourceContext; got {type(value).__name__}",
      fallback,
    )
    return fallback
  if (
    type(value.source) is not str
    or (value.line is not None and type(value.line) is not int)
    or (value.column is not None and type(value.column) is not int)
  ):
    validator.error(
      "invalid-source-context-value",
      f"{label} source context must contain plain string/integer values",
      fallback,
    )
    return fallback
  return value


def _validate_canonical_mesh(
  mesh: MeshSpec,
  model_source: SourceContext,
  validator: _Validator,
) -> None:
  mesh_source = _trusted_source(
    mesh.source,
    label="mesh",
    fallback=model_source,
    validator=validator,
  )
  for index, node in enumerate(mesh.nodes):
    if _is_exact_type(
      node,
      NodeSpec,
      code="invalid-node-spec-type",
      label=f"mesh node {index}",
      source=mesh_source,
      validator=validator,
    ):
      _trusted_source(
        node.source,
        label=f"mesh node {index}",
        fallback=mesh_source,
        validator=validator,
      )
  for block_index, block in enumerate(mesh.cell_blocks):
    if not _is_exact_type(
      block,
      CellBlockSpec,
      code="invalid-cell-block-spec-type",
      label=f"mesh cell block {block_index}",
      source=mesh_source,
      validator=validator,
    ):
      continue
    block_source = _trusted_source(
      block.source,
      label=f"mesh cell block {block_index}",
      fallback=mesh_source,
      validator=validator,
    )
    for cell_index, cell in enumerate(block.cells):
      if _is_exact_type(
        cell,
        CellSpec,
        code="invalid-cell-spec-type",
        label=f"cell block {block_index} cell {cell_index}",
        source=block_source,
        validator=validator,
      ):
        _trusted_source(
          cell.source,
          label=f"cell block {block_index} cell {cell_index}",
          fallback=block_source,
          validator=validator,
        )


def _validate_canonical_fields(
  spec: ModelSpec,
  model_source: SourceContext,
  validator: _Validator,
) -> None:
  for index, field in enumerate(spec.fields):
    if _is_exact_type(
      field,
      FieldSpec,
      code="invalid-field-spec-type",
      label=f"model field {index}",
      source=model_source,
      validator=validator,
    ):
      _trusted_source(
        field.source,
        label=f"model field {index}",
        fallback=model_source,
        validator=validator,
      )


def _validate_canonical_materials(
  spec: ModelSpec,
  model_source: SourceContext,
  validator: _Validator,
) -> None:
  for material_index, material in enumerate(spec.materials):
    if not _is_exact_type(
      material,
      MaterialSpec,
      code="invalid-material-spec-type",
      label=f"model material {material_index}",
      source=model_source,
      validator=validator,
    ):
      continue
    material_source = _trusted_source(
      material.source,
      label=f"model material {material_index}",
      fallback=model_source,
      validator=validator,
    )
    for parameter_index, parameter in enumerate(material.parameters):
      if _is_exact_type(
        parameter,
        MaterialParameterSpec,
        code="invalid-material-parameter-spec-type",
        label=(f"model material {material_index} parameter {parameter_index}"),
        source=material_source,
        validator=validator,
      ):
        _trusted_source(
          parameter.source,
          label=(f"model material {material_index} parameter {parameter_index}"),
          fallback=material_source,
          validator=validator,
        )


def _validate_canonical_regions(
  spec: ModelSpec,
  model_source: SourceContext,
  validator: _Validator,
) -> None:
  for region_index, region in enumerate(spec.regions):
    if not _is_exact_type(
      region,
      RegionSpec,
      code="invalid-region-spec-type",
      label=f"model region {region_index}",
      source=model_source,
      validator=validator,
    ):
      continue
    region_source = _trusted_source(
      region.source,
      label=f"model region {region_index}",
      fallback=model_source,
      validator=validator,
    )
    for cell_ref_index, cell_ref in enumerate(region.cell_refs):
      _is_exact_type(
        cell_ref,
        CellRef,
        code="invalid-cell-ref-type",
        label=f"model region {region_index} cell reference {cell_ref_index}",
        source=region_source,
        validator=validator,
      )


def _validate_canonical_tree(spec: object, validator: _Validator) -> bool:
  fallback = SourceContext()
  if not _is_exact_type(
    spec,
    ModelSpec,
    code="invalid-model-spec-type",
    label="model",
    source=fallback,
    validator=validator,
  ):
    return False
  model_source = _trusted_source(
    spec.source,
    label="model",
    fallback=fallback,
    validator=validator,
  )
  if _is_exact_type(
    spec.mesh,
    MeshSpec,
    code="invalid-mesh-spec-type",
    label="model mesh",
    source=model_source,
    validator=validator,
  ):
    _validate_canonical_mesh(spec.mesh, model_source, validator)
  _validate_canonical_fields(spec, model_source, validator)
  _validate_canonical_materials(spec, model_source, validator)
  _validate_canonical_regions(spec, model_source, validator)
  return not validator.diagnostics


def _validate_nodes(
  mesh: MeshSpec,
  validator: _Validator,
) -> tuple[set[SpecId], int | None]:
  sources: dict[SpecId, SourceContext] = {}
  dimension: int | None = None
  if not mesh.nodes:
    validator.error(
      "empty-node-set", "mesh must declare at least one node", mesh.source
    )
  for node in mesh.nodes:
    validator.identifier(node.id, node.source, kind="node", registry=sources)
    if not node.coordinates:
      validator.error(
        "empty-node-coordinates",
        f"node {node.id!r} has no reference coordinates",
        node.source,
      )
    elif dimension is None:
      dimension = len(node.coordinates)
    elif len(node.coordinates) != dimension:
      validator.error(
        "coordinate-dimension-mismatch",
        f"node {node.id!r} has {len(node.coordinates)} coordinates; "
        f"expected {dimension}",
        node.source,
      )
    if any(
      not isinstance(value, float) or not math.isfinite(value)
      for value in node.coordinates
    ):
      validator.error(
        "invalid-coordinate",
        f"node {node.id!r} has a non-finite or non-numeric coordinate",
        node.source,
      )
  return set(sources), dimension


def _validate_connectivity(
  block: CellBlockSpec,
  cell: CellSpec,
  observed_arity: int,
  node_ids: set[SpecId],
  validator: _Validator,
) -> None:
  if not cell.node_ids:
    validator.error(
      "empty-connectivity",
      f"cell {cell.id!r} contains no node IDs",
      cell.source,
    )
  if len(cell.node_ids) != observed_arity:
    validator.error(
      "connectivity-arity",
      f"cell {cell.id!r} has {len(cell.node_ids)} nodes; "
      f"block {block.id!r} first observed arity {observed_arity}",
      cell.source,
    )
  seen: set[SpecId] = set()
  for node_id in cell.node_ids:
    if not _valid_id(node_id):
      validator.error(
        "invalid-node-reference",
        f"cell {cell.id!r} contains an invalid node ID",
        cell.source,
      )
      continue
    if node_id in seen:
      validator.error(
        "duplicate-node-reference",
        f"cell {cell.id!r} references node {node_id!r} more than once",
        cell.source,
      )
    elif node_id not in node_ids:
      validator.error(
        "unknown-node-reference",
        f"cell {cell.id!r} references unknown node {node_id!r}",
        cell.source,
      )
    seen.add(node_id)


def _validate_block_metadata(
  block: CellBlockSpec,
  coordinate_dimension: int | None,
  topology_dimensions: dict[str, tuple[int, SourceContext]],
  validator: _Validator,
) -> None:
  topology_ok = validator.text(
    block.reference_topology,
    code="invalid-reference-topology",
    label="reference topology",
    source=block.source,
  )
  validator.text(
    block.geometry_interpolation,
    code="invalid-geometry-interpolation",
    label="geometry interpolation",
    source=block.source,
  )
  topology_dimension_ok = _non_negative_int(block.topological_dimension)
  embedding_dimension_ok = _positive_int(block.embedding_dimension)
  if not topology_dimension_ok:
    validator.error(
      "invalid-topological-dimension",
      "topological dimension must be a non-negative integer",
      block.source,
    )
  if not embedding_dimension_ok:
    validator.error(
      "invalid-embedding-dimension",
      "embedding dimension must be a positive integer",
      block.source,
    )
  if (
    topology_dimension_ok
    and embedding_dimension_ok
    and block.topological_dimension > block.embedding_dimension
  ):
    validator.error(
      "invalid-topology-embedding",
      f"topological dimension {block.topological_dimension} exceeds "
      f"embedding dimension {block.embedding_dimension}",
      block.source,
    )
  if (
    coordinate_dimension is not None
    and embedding_dimension_ok
    and block.embedding_dimension != coordinate_dimension
  ):
    validator.error(
      "embedding-dimension-mismatch",
      f"block embedding dimension {block.embedding_dimension} does not match "
      f"mesh coordinate dimension {coordinate_dimension}",
      block.source,
    )
  if topology_ok and topology_dimension_ok:
    first = topology_dimensions.setdefault(
      block.reference_topology,
      (block.topological_dimension, block.source),
    )
    if first[0] != block.topological_dimension:
      validator.error(
        "reference-topology-collision",
        f"topology {block.reference_topology!r} has dimension {first[0]} at "
        f"{first[1].render()} and {block.topological_dimension} here",
        block.source,
      )


def _validate_blocks(
  mesh: MeshSpec,
  node_ids: set[SpecId],
  coordinate_dimension: int | None,
  validator: _Validator,
) -> dict[SpecId, set[SpecId]]:
  block_sources: dict[SpecId, SourceContext] = {}
  cells_by_block: dict[SpecId, set[SpecId]] = {}
  topology_dimensions: dict[str, tuple[int, SourceContext]] = {}
  if not mesh.cell_blocks:
    validator.error(
      "empty-cell-block-set",
      "mesh must declare at least one cell block",
      mesh.source,
    )
  for block in mesh.cell_blocks:
    unique_block = validator.identifier(
      block.id,
      block.source,
      kind="cell-block",
      registry=block_sources,
    )
    _validate_block_metadata(
      block,
      coordinate_dimension,
      topology_dimensions,
      validator,
    )
    if not block.cells:
      validator.error(
        "empty-cell-block",
        f"cell block {block.id!r} must contain at least one cell",
        block.source,
      )
    observed_arity = len(block.cells[0].node_ids) if block.cells else None
    cell_sources: dict[SpecId, SourceContext] = {}
    for cell in block.cells:
      validator.identifier(
        cell.id,
        cell.source,
        kind="cell",
        registry=cell_sources,
      )
      if observed_arity is not None:
        _validate_connectivity(
          block,
          cell,
          observed_arity,
          node_ids,
          validator,
        )
    if unique_block:
      cells_by_block[block.id] = set(cell_sources)
  return cells_by_block


def _validate_fields(
  fields: tuple[FieldSpec, ...],
  model_source: SourceContext,
  validator: _Validator,
) -> set[SpecId]:
  sources: dict[SpecId, SourceContext] = {}
  if not fields:
    validator.error(
      "empty-field-set",
      "model must declare at least one field",
      model_source,
    )
  for field in fields:
    validator.identifier(field.id, field.source, kind="field", registry=sources)
    validator.text(
      field.location,
      code="invalid-field-location",
      label="field location",
      source=field.source,
    )
    if not field.components:
      validator.error(
        "empty-field-components",
        f"field {field.id!r} must declare at least one component",
        field.source,
      )
    seen: set[str] = set()
    for component in field.components:
      if validator.text(
        component,
        code="invalid-field-component",
        label="field component",
        source=field.source,
      ):
        if component in seen:
          validator.error(
            "duplicate-field-component",
            f"field {field.id!r} repeats component {component!r}",
            field.source,
          )
        seen.add(component)
  return set(sources)


def _parameter_value_is_valid(value: object) -> bool:
  if isinstance(value, tuple):
    return all(_parameter_value_is_valid(item) for item in value)
  if isinstance(value, str):
    return bool(value)
  if isinstance(value, bool | int):
    return True
  return isinstance(value, float) and math.isfinite(value)


def _validate_materials(
  materials: tuple[MaterialSpec, ...],
  model_source: SourceContext,
  validator: _Validator,
) -> set[SpecId]:
  sources: dict[SpecId, SourceContext] = {}
  if not materials:
    validator.error(
      "empty-material-set",
      "model must declare at least one material",
      model_source,
    )
  for material in materials:
    validator.identifier(
      material.id,
      material.source,
      kind="material",
      registry=sources,
    )
    validator.text(
      material.model,
      code="invalid-material-model",
      label="material model",
      source=material.source,
    )
    parameter_sources: dict[str, SourceContext] = {}
    for parameter in material.parameters:
      if validator.text(
        parameter.name,
        code="invalid-material-parameter-name",
        label="material parameter name",
        source=parameter.source,
      ):
        first = parameter_sources.get(parameter.name)
        if first is not None:
          validator.error(
            "duplicate-material-parameter-name",
            f"material {material.id!r} repeats parameter {parameter.name!r}; "
            f"first declared at {first.render()}",
            parameter.source,
          )
        else:
          parameter_sources[parameter.name] = parameter.source
      if not _parameter_value_is_valid(parameter.value):
        validator.error(
          "invalid-material-parameter-value",
          f"parameter {parameter.name!r} must contain finite plain scalars or tuples",
          parameter.source,
        )
  return set(sources)


def _validate_cell_refs(
  region: RegionSpec,
  cells_by_block: dict[SpecId, set[SpecId]],
  validator: _Validator,
) -> None:
  if not region.cell_refs:
    validator.error(
      "empty-region-cell-selection",
      f"region {region.id!r} must select at least one cell",
      region.source,
    )
  seen: set[tuple[SpecId, SpecId]] = set()
  for cell_ref in region.cell_refs:
    if not _valid_id(cell_ref.block_id):
      validator.error(
        "invalid-cell-block-reference",
        f"region {region.id!r} contains an invalid cell block ID",
        region.source,
      )
      continue
    if not _valid_id(cell_ref.cell_id):
      validator.error(
        "invalid-cell-reference",
        f"region {region.id!r} contains an invalid cell ID",
        region.source,
      )
      continue
    key = (cell_ref.block_id, cell_ref.cell_id)
    if key in seen:
      validator.error(
        "duplicate-cell-reference",
        f"region {region.id!r} repeats cell reference {key!r}",
        region.source,
      )
    elif cell_ref.block_id not in cells_by_block:
      validator.error(
        "unknown-cell-block-reference",
        f"region {region.id!r} references unknown block {cell_ref.block_id!r}",
        region.source,
      )
    elif cell_ref.cell_id not in cells_by_block[cell_ref.block_id]:
      validator.error(
        "unknown-cell-reference",
        f"region {region.id!r} references unknown cell {cell_ref.cell_id!r} "
        f"in block {cell_ref.block_id!r}",
        region.source,
      )
    seen.add(key)


def _validate_field_refs(
  region: RegionSpec,
  field_ids: set[SpecId],
  validator: _Validator,
) -> None:
  if not region.field_ids:
    validator.error(
      "empty-region-field-signature",
      f"region {region.id!r} must reference at least one field",
      region.source,
    )
  seen: set[SpecId] = set()
  for field_id in region.field_ids:
    if not _valid_id(field_id):
      validator.error(
        "invalid-field-reference",
        f"region {region.id!r} contains an invalid field ID",
        region.source,
      )
      continue
    if field_id in seen:
      validator.error(
        "duplicate-field-reference",
        f"region {region.id!r} repeats field {field_id!r}",
        region.source,
      )
    elif field_id not in field_ids:
      validator.error(
        "unknown-field-reference",
        f"region {region.id!r} references unknown field {field_id!r}",
        region.source,
      )
    seen.add(field_id)


def _validate_regions(
  regions: tuple[RegionSpec, ...],
  model_source: SourceContext,
  *,
  cells_by_block: dict[SpecId, set[SpecId]],
  field_ids: set[SpecId],
  material_ids: set[SpecId],
  validator: _Validator,
) -> None:
  sources: dict[SpecId, SourceContext] = {}
  if not regions:
    validator.error(
      "empty-region-set",
      "model must declare at least one region",
      model_source,
    )
  for region in regions:
    validator.identifier(region.id, region.source, kind="region", registry=sources)
    validator.text(
      region.formulation,
      code="invalid-formulation",
      label="region formulation",
      source=region.source,
    )
    validator.text(
      region.quadrature,
      code="invalid-quadrature",
      label="region quadrature",
      source=region.source,
    )
    _validate_cell_refs(region, cells_by_block, validator)
    _validate_field_refs(region, field_ids, validator)
    if not _valid_id(region.material_id):
      validator.error(
        "invalid-material-reference",
        f"region {region.id!r} contains an invalid material ID",
        region.source,
      )
    elif region.material_id not in material_ids:
      validator.error(
        "unknown-material-reference",
        f"region {region.id!r} references unknown material {region.material_id!r}",
        region.source,
      )


def normalize_model_spec(spec: ModelSpec) -> ModelSpec:
  """Validate and return the canonical, caller-independent model value."""
  validator = _Validator()
  if not _validate_canonical_tree(spec, validator):
    raise ModelSpecValidationError(validator.diagnostics)
  node_ids, coordinate_dimension = _validate_nodes(spec.mesh, validator)
  cells_by_block = _validate_blocks(
    spec.mesh,
    node_ids,
    coordinate_dimension,
    validator,
  )
  field_ids = _validate_fields(spec.fields, spec.source, validator)
  material_ids = _validate_materials(spec.materials, spec.source, validator)
  _validate_regions(
    spec.regions,
    spec.source,
    cells_by_block=cells_by_block,
    field_ids=field_ids,
    material_ids=material_ids,
    validator=validator,
  )
  if validator.diagnostics:
    raise ModelSpecValidationError(validator.diagnostics)
  return spec
