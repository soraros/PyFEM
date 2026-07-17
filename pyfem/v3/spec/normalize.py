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
    if type(value) is str and value:
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
  return type(value) is str and bool(value) or type(value) is int


def _positive_int(value: object) -> bool:
  return type(value) is int and value > 0


def _non_negative_int(value: object) -> bool:
  return type(value) is int and value >= 0


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
    f"{label} must be exactly {expected.__name__}",
    source,
  )
  return False


_MISSING = object()
_INVALID = object()


def _read_slot(value: object, name: str) -> object:
  try:
    return object.__getattribute__(value, name)
  except AttributeError:
    return _MISSING


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
      f"{label} source must be exactly SourceContext",
      fallback,
    )
    return fallback
  source = _read_slot(value, "source")
  line = _read_slot(value, "line")
  column = _read_slot(value, "column")
  if (
    source is _MISSING
    or line is _MISSING
    or column is _MISSING
    or type(source) is not str
    or (line is not None and type(line) is not int)
    or (column is not None and type(column) is not int)
  ):
    validator.error(
      "invalid-source-context-value",
      f"{label} source context must contain plain string/integer values",
      fallback,
    )
    return fallback
  return SourceContext(source=source, line=line, column=column)


def _spec_source(
  value: object,
  *,
  label: str,
  fallback: SourceContext,
  validator: _Validator,
) -> SourceContext:
  source = _read_slot(value, "source")
  if source is _MISSING:
    validator.error(
      "invalid-source-context-value",
      f"{label} source context must initialize every canonical slot",
      fallback,
    )
    return fallback
  return _trusted_source(
    source,
    label=label,
    fallback=fallback,
    validator=validator,
  )


def _required_slots(
  value: object,
  names: tuple[str, ...],
  *,
  code: str,
  label: str,
  source: SourceContext,
  validator: _Validator,
) -> tuple[object, ...] | None:
  values = tuple(_read_slot(value, name) for name in names)
  if any(item is _MISSING for item in values):
    validator.error(
      code,
      f"{label} must initialize every canonical slot",
      source,
    )
    return None
  return values


def _exact_tuple(
  value: object,
  *,
  code: str,
  label: str,
  source: SourceContext,
  validator: _Validator,
) -> tuple[object, ...] | None:
  if type(value) is tuple:
    return value
  validator.error(code, f"{label} must be exactly tuple", source)
  return None


def _canonical_id(
  value: object,
  *,
  code: str,
  label: str,
  source: SourceContext,
  validator: _Validator,
) -> object:
  if type(value) is str:
    return str.strip(value)
  if type(value) is int:
    return value
  validator.error(code, f"{label} must be a plain string or integer", source)
  return _INVALID


def _canonical_text(
  value: object,
  *,
  code: str,
  label: str,
  source: SourceContext,
  validator: _Validator,
) -> object:
  if type(value) is str:
    return str.strip(value)
  validator.error(code, f"{label} must be a plain string", source)
  return _INVALID


def _canonical_int(
  value: object,
  *,
  code: str,
  label: str,
  source: SourceContext,
  validator: _Validator,
) -> object:
  if type(value) is int:
    return value
  validator.error(code, f"{label} must be a plain integer", source)
  return _INVALID


def _canonical_number(
  value: object,
  *,
  source: SourceContext,
  validator: _Validator,
) -> object:
  if type(value) is float:
    return value
  if type(value) is int:
    try:
      return float(value)
    except OverflowError:
      pass
  validator.error(
    "invalid-coordinate",
    "node coordinate must be a finite plain integer or float",
    source,
  )
  return _INVALID


def _canonical_parameter_value(
  value: object,
  *,
  source: SourceContext,
  validator: _Validator,
) -> object:
  if type(value) is str:
    return str.strip(value)
  if type(value) is bool or type(value) is int or type(value) is float:
    return value
  if type(value) is not tuple:
    validator.error(
      "invalid-material-parameter-value",
      "material parameter value must contain plain scalars or exact tuples",
      source,
    )
    return _INVALID

  stack: list[tuple[tuple[object, ...], int, list[object]]] = [(value, 0, [])]
  valid = True
  result: object = _INVALID
  while stack:
    container, index, built = stack[-1]
    if index == len(container):
      completed = tuple(built)
      stack.pop()
      if stack:
        stack[-1][2].append(completed)
      else:
        result = completed
      continue

    item = container[index]
    stack[-1] = (container, index + 1, built)
    if type(item) is tuple:
      stack.append((item, 0, []))
    elif type(item) is str:
      built.append(str.strip(item))
    elif type(item) is bool or type(item) is int or type(item) is float:
      built.append(item)
    else:
      validator.error(
        "invalid-material-parameter-value",
        "material parameter value must contain plain scalars or exact tuples",
        source,
      )
      valid = False
  return result if valid else _INVALID


def _preflight_node(
  value: object,
  index: int,
  mesh_source: SourceContext,
  validator: _Validator,
) -> NodeSpec | None:
  label = f"mesh node {index}"
  if not _is_exact_type(
    value,
    NodeSpec,
    code="invalid-node-spec-type",
    label=label,
    source=mesh_source,
    validator=validator,
  ):
    return None
  source = _spec_source(
    value,
    label=label,
    fallback=mesh_source,
    validator=validator,
  )
  slots = _required_slots(
    value,
    ("id", "coordinates"),
    code="invalid-node-spec-value",
    label=label,
    source=source,
    validator=validator,
  )
  if slots is None:
    return None
  raw_id, raw_coordinates = slots
  node_id = _canonical_id(
    raw_id,
    code="invalid-node-id",
    label="node ID",
    source=source,
    validator=validator,
  )
  coordinates = _exact_tuple(
    raw_coordinates,
    code="invalid-node-spec-value",
    label="node coordinates",
    source=source,
    validator=validator,
  )
  canonical_coordinates: list[object] = []
  valid = node_id is not _INVALID and coordinates is not None
  if coordinates is not None:
    for coordinate in coordinates:
      canonical = _canonical_number(
        coordinate,
        source=source,
        validator=validator,
      )
      if canonical is _INVALID:
        valid = False
      else:
        canonical_coordinates.append(canonical)
  if not valid:
    return None
  return NodeSpec(
    id=node_id,
    coordinates=tuple(canonical_coordinates),
    source=source,
  )


def _preflight_cell(
  value: object,
  *,
  block_index: int,
  cell_index: int,
  block_source: SourceContext,
  validator: _Validator,
) -> CellSpec | None:
  label = f"cell block {block_index} cell {cell_index}"
  if not _is_exact_type(
    value,
    CellSpec,
    code="invalid-cell-spec-type",
    label=label,
    source=block_source,
    validator=validator,
  ):
    return None
  source = _spec_source(
    value,
    label=label,
    fallback=block_source,
    validator=validator,
  )
  slots = _required_slots(
    value,
    ("id", "node_ids"),
    code="invalid-cell-spec-value",
    label=label,
    source=source,
    validator=validator,
  )
  if slots is None:
    return None
  raw_id, raw_node_ids = slots
  cell_id = _canonical_id(
    raw_id,
    code="invalid-cell-id",
    label="cell ID",
    source=source,
    validator=validator,
  )
  node_ids = _exact_tuple(
    raw_node_ids,
    code="invalid-cell-spec-value",
    label="cell node IDs",
    source=source,
    validator=validator,
  )
  canonical_node_ids: list[object] = []
  valid = cell_id is not _INVALID and node_ids is not None
  if node_ids is not None:
    for node_id in node_ids:
      canonical = _canonical_id(
        node_id,
        code="invalid-node-reference",
        label="cell node ID",
        source=source,
        validator=validator,
      )
      if canonical is _INVALID:
        valid = False
      else:
        canonical_node_ids.append(canonical)
  if not valid:
    return None
  return CellSpec(
    id=cell_id,
    node_ids=tuple(canonical_node_ids),
    source=source,
  )


def _preflight_cell_block(
  value: object,
  index: int,
  mesh_source: SourceContext,
  validator: _Validator,
) -> CellBlockSpec | None:
  label = f"mesh cell block {index}"
  if not _is_exact_type(
    value,
    CellBlockSpec,
    code="invalid-cell-block-spec-type",
    label=label,
    source=mesh_source,
    validator=validator,
  ):
    return None
  source = _spec_source(
    value,
    label=label,
    fallback=mesh_source,
    validator=validator,
  )
  slots = _required_slots(
    value,
    (
      "id",
      "reference_topology",
      "topological_dimension",
      "embedding_dimension",
      "geometry_interpolation",
      "cells",
    ),
    code="invalid-cell-block-spec-value",
    label=label,
    source=source,
    validator=validator,
  )
  if slots is None:
    return None
  (
    raw_id,
    raw_topology,
    raw_topological_dimension,
    raw_embedding_dimension,
    raw_interpolation,
    raw_cells,
  ) = slots
  block_id = _canonical_id(
    raw_id,
    code="invalid-cell-block-id",
    label="cell block ID",
    source=source,
    validator=validator,
  )
  topology = _canonical_text(
    raw_topology,
    code="invalid-reference-topology",
    label="reference topology",
    source=source,
    validator=validator,
  )
  topological_dimension = _canonical_int(
    raw_topological_dimension,
    code="invalid-topological-dimension",
    label="topological dimension",
    source=source,
    validator=validator,
  )
  embedding_dimension = _canonical_int(
    raw_embedding_dimension,
    code="invalid-embedding-dimension",
    label="embedding dimension",
    source=source,
    validator=validator,
  )
  interpolation = _canonical_text(
    raw_interpolation,
    code="invalid-geometry-interpolation",
    label="geometry interpolation",
    source=source,
    validator=validator,
  )
  cells = _exact_tuple(
    raw_cells,
    code="invalid-cell-block-spec-value",
    label="cell block cells",
    source=source,
    validator=validator,
  )
  canonical_cells: list[CellSpec] = []
  valid = (
    all(
      item is not _INVALID
      for item in (
        block_id,
        topology,
        topological_dimension,
        embedding_dimension,
        interpolation,
      )
    )
    and cells is not None
  )
  if cells is not None:
    for cell_index, cell in enumerate(cells):
      canonical = _preflight_cell(
        cell,
        block_index=index,
        cell_index=cell_index,
        block_source=source,
        validator=validator,
      )
      if canonical is None:
        valid = False
      else:
        canonical_cells.append(canonical)
  if not valid:
    return None
  return CellBlockSpec(
    id=block_id,
    reference_topology=topology,
    topological_dimension=topological_dimension,
    embedding_dimension=embedding_dimension,
    geometry_interpolation=interpolation,
    cells=tuple(canonical_cells),
    source=source,
  )


def _preflight_mesh(
  value: object,
  model_source: SourceContext,
  validator: _Validator,
) -> MeshSpec | None:
  if not _is_exact_type(
    value,
    MeshSpec,
    code="invalid-mesh-spec-type",
    label="model mesh",
    source=model_source,
    validator=validator,
  ):
    return None
  source = _spec_source(
    value,
    label="mesh",
    fallback=model_source,
    validator=validator,
  )
  slots = _required_slots(
    value,
    ("nodes", "cell_blocks"),
    code="invalid-mesh-spec-value",
    label="mesh",
    source=source,
    validator=validator,
  )
  if slots is None:
    return None
  raw_nodes, raw_cell_blocks = slots
  nodes = _exact_tuple(
    raw_nodes,
    code="invalid-mesh-spec-value",
    label="mesh nodes",
    source=source,
    validator=validator,
  )
  cell_blocks = _exact_tuple(
    raw_cell_blocks,
    code="invalid-mesh-spec-value",
    label="mesh cell blocks",
    source=source,
    validator=validator,
  )
  canonical_nodes: list[NodeSpec] = []
  canonical_blocks: list[CellBlockSpec] = []
  valid = nodes is not None and cell_blocks is not None
  if nodes is not None:
    for index, node in enumerate(nodes):
      canonical = _preflight_node(node, index, source, validator)
      if canonical is None:
        valid = False
      else:
        canonical_nodes.append(canonical)
  if cell_blocks is not None:
    for index, block in enumerate(cell_blocks):
      canonical = _preflight_cell_block(block, index, source, validator)
      if canonical is None:
        valid = False
      else:
        canonical_blocks.append(canonical)
  if not valid:
    return None
  return MeshSpec(
    nodes=tuple(canonical_nodes),
    cell_blocks=tuple(canonical_blocks),
    source=source,
  )


def _preflight_field(
  value: object,
  index: int,
  model_source: SourceContext,
  validator: _Validator,
) -> FieldSpec | None:
  label = f"model field {index}"
  if not _is_exact_type(
    value,
    FieldSpec,
    code="invalid-field-spec-type",
    label=label,
    source=model_source,
    validator=validator,
  ):
    return None
  source = _spec_source(
    value,
    label=label,
    fallback=model_source,
    validator=validator,
  )
  slots = _required_slots(
    value,
    ("id", "components", "location"),
    code="invalid-field-spec-value",
    label=label,
    source=source,
    validator=validator,
  )
  if slots is None:
    return None
  raw_id, raw_components, raw_location = slots
  field_id = _canonical_id(
    raw_id,
    code="invalid-field-id",
    label="field ID",
    source=source,
    validator=validator,
  )
  components = _exact_tuple(
    raw_components,
    code="invalid-field-spec-value",
    label="field components",
    source=source,
    validator=validator,
  )
  location = _canonical_text(
    raw_location,
    code="invalid-field-location",
    label="field location",
    source=source,
    validator=validator,
  )
  canonical_components: list[object] = []
  valid = field_id is not _INVALID and components is not None
  valid = valid and location is not _INVALID
  if components is not None:
    for component in components:
      canonical = _canonical_text(
        component,
        code="invalid-field-component",
        label="field component",
        source=source,
        validator=validator,
      )
      if canonical is _INVALID:
        valid = False
      else:
        canonical_components.append(canonical)
  if not valid:
    return None
  return FieldSpec(
    id=field_id,
    components=tuple(canonical_components),
    location=location,
    source=source,
  )


def _preflight_material_parameter(
  value: object,
  *,
  material_index: int,
  parameter_index: int,
  material_source: SourceContext,
  validator: _Validator,
) -> MaterialParameterSpec | None:
  label = f"model material {material_index} parameter {parameter_index}"
  if not _is_exact_type(
    value,
    MaterialParameterSpec,
    code="invalid-material-parameter-spec-type",
    label=label,
    source=material_source,
    validator=validator,
  ):
    return None
  source = _spec_source(
    value,
    label=label,
    fallback=material_source,
    validator=validator,
  )
  slots = _required_slots(
    value,
    ("name", "value"),
    code="invalid-material-parameter-spec-value",
    label=label,
    source=source,
    validator=validator,
  )
  if slots is None:
    return None
  raw_name, raw_value = slots
  name = _canonical_text(
    raw_name,
    code="invalid-material-parameter-name",
    label="material parameter name",
    source=source,
    validator=validator,
  )
  parameter_value = _canonical_parameter_value(
    raw_value,
    source=source,
    validator=validator,
  )
  if name is _INVALID or parameter_value is _INVALID:
    return None
  return MaterialParameterSpec(
    name=name,
    value=parameter_value,
    source=source,
  )


def _preflight_material(
  value: object,
  index: int,
  model_source: SourceContext,
  validator: _Validator,
) -> MaterialSpec | None:
  label = f"model material {index}"
  if not _is_exact_type(
    value,
    MaterialSpec,
    code="invalid-material-spec-type",
    label=label,
    source=model_source,
    validator=validator,
  ):
    return None
  source = _spec_source(
    value,
    label=label,
    fallback=model_source,
    validator=validator,
  )
  slots = _required_slots(
    value,
    ("id", "model", "parameters"),
    code="invalid-material-spec-value",
    label=label,
    source=source,
    validator=validator,
  )
  if slots is None:
    return None
  raw_id, raw_model, raw_parameters = slots
  material_id = _canonical_id(
    raw_id,
    code="invalid-material-id",
    label="material ID",
    source=source,
    validator=validator,
  )
  model = _canonical_text(
    raw_model,
    code="invalid-material-model",
    label="material model",
    source=source,
    validator=validator,
  )
  parameters = _exact_tuple(
    raw_parameters,
    code="invalid-material-spec-value",
    label="material parameters",
    source=source,
    validator=validator,
  )
  canonical_parameters: list[MaterialParameterSpec] = []
  valid = material_id is not _INVALID and model is not _INVALID
  valid = valid and parameters is not None
  if parameters is not None:
    for parameter_index, parameter in enumerate(parameters):
      canonical = _preflight_material_parameter(
        parameter,
        material_index=index,
        parameter_index=parameter_index,
        material_source=source,
        validator=validator,
      )
      if canonical is None:
        valid = False
      else:
        canonical_parameters.append(canonical)
  if not valid:
    return None
  return MaterialSpec(
    id=material_id,
    model=model,
    parameters=tuple(canonical_parameters),
    source=source,
  )


def _preflight_cell_ref(
  value: object,
  *,
  region_index: int,
  cell_ref_index: int,
  region_source: SourceContext,
  validator: _Validator,
) -> CellRef | None:
  label = f"model region {region_index} cell reference {cell_ref_index}"
  if not _is_exact_type(
    value,
    CellRef,
    code="invalid-cell-ref-type",
    label=label,
    source=region_source,
    validator=validator,
  ):
    return None
  slots = _required_slots(
    value,
    ("block_id", "cell_id"),
    code="invalid-cell-ref-value",
    label=label,
    source=region_source,
    validator=validator,
  )
  if slots is None:
    return None
  raw_block_id, raw_cell_id = slots
  block_id = _canonical_id(
    raw_block_id,
    code="invalid-cell-block-reference",
    label="cell reference block ID",
    source=region_source,
    validator=validator,
  )
  cell_id = _canonical_id(
    raw_cell_id,
    code="invalid-cell-reference",
    label="cell reference cell ID",
    source=region_source,
    validator=validator,
  )
  if block_id is _INVALID or cell_id is _INVALID:
    return None
  return CellRef(block_id=block_id, cell_id=cell_id)


def _preflight_region(
  value: object,
  index: int,
  model_source: SourceContext,
  validator: _Validator,
) -> RegionSpec | None:
  label = f"model region {index}"
  if not _is_exact_type(
    value,
    RegionSpec,
    code="invalid-region-spec-type",
    label=label,
    source=model_source,
    validator=validator,
  ):
    return None
  source = _spec_source(
    value,
    label=label,
    fallback=model_source,
    validator=validator,
  )
  slots = _required_slots(
    value,
    (
      "id",
      "cell_refs",
      "field_ids",
      "material_id",
      "formulation",
      "quadrature",
    ),
    code="invalid-region-spec-value",
    label=label,
    source=source,
    validator=validator,
  )
  if slots is None:
    return None
  (
    raw_id,
    raw_cell_refs,
    raw_field_ids,
    raw_material_id,
    raw_formulation,
    raw_quadrature,
  ) = slots
  region_id = _canonical_id(
    raw_id,
    code="invalid-region-id",
    label="region ID",
    source=source,
    validator=validator,
  )
  cell_refs = _exact_tuple(
    raw_cell_refs,
    code="invalid-region-spec-value",
    label="region cell references",
    source=source,
    validator=validator,
  )
  field_ids = _exact_tuple(
    raw_field_ids,
    code="invalid-region-spec-value",
    label="region field IDs",
    source=source,
    validator=validator,
  )
  material_id = _canonical_id(
    raw_material_id,
    code="invalid-material-reference",
    label="region material ID",
    source=source,
    validator=validator,
  )
  formulation = _canonical_text(
    raw_formulation,
    code="invalid-formulation",
    label="region formulation",
    source=source,
    validator=validator,
  )
  quadrature = _canonical_text(
    raw_quadrature,
    code="invalid-quadrature",
    label="region quadrature",
    source=source,
    validator=validator,
  )
  canonical_refs: list[CellRef] = []
  canonical_field_ids: list[object] = []
  valid = all(
    item is not _INVALID for item in (region_id, material_id, formulation, quadrature)
  )
  valid = valid and cell_refs is not None and field_ids is not None
  if cell_refs is not None:
    for cell_ref_index, cell_ref in enumerate(cell_refs):
      canonical = _preflight_cell_ref(
        cell_ref,
        region_index=index,
        cell_ref_index=cell_ref_index,
        region_source=source,
        validator=validator,
      )
      if canonical is None:
        valid = False
      else:
        canonical_refs.append(canonical)
  if field_ids is not None:
    for field_id in field_ids:
      canonical = _canonical_id(
        field_id,
        code="invalid-field-reference",
        label="region field ID",
        source=source,
        validator=validator,
      )
      if canonical is _INVALID:
        valid = False
      else:
        canonical_field_ids.append(canonical)
  if not valid:
    return None
  return RegionSpec(
    id=region_id,
    cell_refs=tuple(canonical_refs),
    field_ids=tuple(canonical_field_ids),
    material_id=material_id,
    formulation=formulation,
    quadrature=quadrature,
    source=source,
  )


def _preflight_model(spec: object, validator: _Validator) -> ModelSpec | None:
  fallback = SourceContext()
  if not _is_exact_type(
    spec,
    ModelSpec,
    code="invalid-model-spec-type",
    label="model",
    source=fallback,
    validator=validator,
  ):
    return None
  source = _spec_source(
    spec,
    label="model",
    fallback=fallback,
    validator=validator,
  )
  slots = _required_slots(
    spec,
    ("mesh", "fields", "materials", "regions"),
    code="invalid-model-spec-value",
    label="model",
    source=source,
    validator=validator,
  )
  if slots is None:
    return None
  raw_mesh, raw_fields, raw_materials, raw_regions = slots
  mesh = _preflight_mesh(raw_mesh, source, validator)
  fields = _exact_tuple(
    raw_fields,
    code="invalid-model-spec-value",
    label="model fields",
    source=source,
    validator=validator,
  )
  materials = _exact_tuple(
    raw_materials,
    code="invalid-model-spec-value",
    label="model materials",
    source=source,
    validator=validator,
  )
  regions = _exact_tuple(
    raw_regions,
    code="invalid-model-spec-value",
    label="model regions",
    source=source,
    validator=validator,
  )
  canonical_fields: list[FieldSpec] = []
  canonical_materials: list[MaterialSpec] = []
  canonical_regions: list[RegionSpec] = []
  valid = mesh is not None
  valid = valid and fields is not None and materials is not None and regions is not None
  if fields is not None:
    for index, field in enumerate(fields):
      canonical = _preflight_field(field, index, source, validator)
      if canonical is None:
        valid = False
      else:
        canonical_fields.append(canonical)
  if materials is not None:
    for index, material in enumerate(materials):
      canonical = _preflight_material(material, index, source, validator)
      if canonical is None:
        valid = False
      else:
        canonical_materials.append(canonical)
  if regions is not None:
    for index, region in enumerate(regions):
      canonical = _preflight_region(region, index, source, validator)
      if canonical is None:
        valid = False
      else:
        canonical_regions.append(canonical)
  if not valid:
    return None
  return ModelSpec(
    mesh=mesh,
    fields=tuple(canonical_fields),
    materials=tuple(canonical_materials),
    regions=tuple(canonical_regions),
    source=source,
  )


def _copy_source(source: SourceContext) -> SourceContext:
  return SourceContext(
    source=source.source,
    line=source.line,
    column=source.column,
  )


def _copy_parameter_value(value: object) -> object:
  if type(value) is not tuple:
    return value
  stack: list[tuple[tuple[object, ...], int, list[object]]] = [(value, 0, [])]
  result: object = value
  while stack:
    container, index, built = stack[-1]
    if index == len(container):
      completed = tuple(built)
      stack.pop()
      if stack:
        stack[-1][2].append(completed)
      else:
        result = completed
      continue
    item = container[index]
    stack[-1] = (container, index + 1, built)
    if type(item) is tuple:
      stack.append((item, 0, []))
    else:
      built.append(item)
  return result


def _copy_node(node: NodeSpec) -> NodeSpec:
  return NodeSpec(
    id=node.id,
    coordinates=tuple(value for value in node.coordinates),
    source=_copy_source(node.source),
  )


def _copy_cell(cell: CellSpec) -> CellSpec:
  return CellSpec(
    id=cell.id,
    node_ids=tuple(value for value in cell.node_ids),
    source=_copy_source(cell.source),
  )


def _copy_cell_block(block: CellBlockSpec) -> CellBlockSpec:
  return CellBlockSpec(
    id=block.id,
    reference_topology=block.reference_topology,
    topological_dimension=block.topological_dimension,
    embedding_dimension=block.embedding_dimension,
    geometry_interpolation=block.geometry_interpolation,
    cells=tuple(_copy_cell(cell) for cell in block.cells),
    source=_copy_source(block.source),
  )


def _copy_mesh(mesh: MeshSpec) -> MeshSpec:
  return MeshSpec(
    nodes=tuple(_copy_node(node) for node in mesh.nodes),
    cell_blocks=tuple(_copy_cell_block(block) for block in mesh.cell_blocks),
    source=_copy_source(mesh.source),
  )


def _copy_field(field: FieldSpec) -> FieldSpec:
  return FieldSpec(
    id=field.id,
    components=tuple(value for value in field.components),
    location=field.location,
    source=_copy_source(field.source),
  )


def _copy_parameter(parameter: MaterialParameterSpec) -> MaterialParameterSpec:
  return MaterialParameterSpec(
    name=parameter.name,
    value=_copy_parameter_value(parameter.value),
    source=_copy_source(parameter.source),
  )


def _copy_material(material: MaterialSpec) -> MaterialSpec:
  return MaterialSpec(
    id=material.id,
    model=material.model,
    parameters=tuple(_copy_parameter(item) for item in material.parameters),
    source=_copy_source(material.source),
  )


def _copy_cell_ref(cell_ref: CellRef) -> CellRef:
  return CellRef(block_id=cell_ref.block_id, cell_id=cell_ref.cell_id)


def _copy_region(region: RegionSpec) -> RegionSpec:
  return RegionSpec(
    id=region.id,
    cell_refs=tuple(_copy_cell_ref(item) for item in region.cell_refs),
    field_ids=tuple(value for value in region.field_ids),
    material_id=region.material_id,
    formulation=region.formulation,
    quadrature=region.quadrature,
    source=_copy_source(region.source),
  )


def _reconstruct_model(spec: ModelSpec) -> ModelSpec:
  return ModelSpec(
    mesh=_copy_mesh(spec.mesh),
    fields=tuple(_copy_field(field) for field in spec.fields),
    materials=tuple(_copy_material(material) for material in spec.materials),
    regions=tuple(_copy_region(region) for region in spec.regions),
    source=_copy_source(spec.source),
  )


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
      type(value) is not float or not math.isfinite(value) for value in node.coordinates
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
  if type(value) is tuple:
    pending = list(value)
    while pending:
      item = pending.pop()
      if type(item) is tuple:
        pending.extend(item)
      elif type(item) is str:
        if not item:
          return False
      elif type(item) is bool or type(item) is int:
        continue
      elif type(item) is not float or not math.isfinite(item):
        return False
    return True
  if type(value) is str:
    return bool(value)
  if type(value) is bool or type(value) is int:
    return True
  return type(value) is float and math.isfinite(value)


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
  snapshot = _preflight_model(spec, validator)
  if validator.diagnostics or snapshot is None:
    raise ModelSpecValidationError(validator.diagnostics)
  canonical = _reconstruct_model(snapshot)
  node_ids, coordinate_dimension = _validate_nodes(canonical.mesh, validator)
  cells_by_block = _validate_blocks(
    canonical.mesh,
    node_ids,
    coordinate_dimension,
    validator,
  )
  field_ids = _validate_fields(
    canonical.fields,
    canonical.source,
    validator,
  )
  material_ids = _validate_materials(
    canonical.materials,
    canonical.source,
    validator,
  )
  _validate_regions(
    canonical.regions,
    canonical.source,
    cells_by_block=cells_by_block,
    field_ids=field_ids,
    material_ids=material_ids,
    validator=validator,
  )
  if validator.diagnostics:
    raise ModelSpecValidationError(validator.diagnostics)
  return canonical
