"""Direct normalized-model compiler for the generic compiled-system boundary."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pyfem.v3.compile import continuum as _continuum_builder
from pyfem.v3.compile.diagnostics import (
  ModelCompilationDiagnostic,
  ModelCompilationError,
)
from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.identity import InstanceId
from pyfem.v3.model.provenance import CanonicalManifest, ContentFingerprint
from pyfem.v3.model.registry import RegistryDescriptor, RegistryKey
from pyfem.v3.model.system import (
  CompiledSource,
  CompiledSystem,
  DiscreteSpace,
  PointEntityBlock,
  SourceAttribution,
  SystemProvenance,
)
from pyfem.v3.spec.diagnostics import SourceContext
from pyfem.v3.spec.model import FieldSpec, ModelSpec, SpecId
from pyfem.v3.spec.normalize import normalize_model_spec

COMPILED_SYSTEM_MANIFEST_SCHEMA = "pyfem-v3-compiled-system-v1"
_SUPPORTED_INDEX_DTYPES = ("int8", "int16", "int32", "int64")


def _new[ValueT](cls: type[ValueT], /, **fields: object) -> ValueT:
  value = object.__new__(cls)
  for name, field in fields.items():
    object.__setattr__(value, name, field)
  return value


@dataclass(frozen=True, slots=True)
class SystemCompilationPolicy:
  """Numeric conventions entering compiled-system content identity."""

  dense_index_dtype: str = "int64"
  geometry_relative_tolerance: float = 1.0e-12

  def __post_init__(self) -> None:
    if self.dense_index_dtype not in _SUPPORTED_INDEX_DTYPES:
      msg = "dense index dtype must be int8, int16, int32, or int64"
      raise ValueError(msg)
    tolerance = self.geometry_relative_tolerance
    if type(tolerance) is not float or not math.isfinite(tolerance):
      msg = "geometry relative tolerance must be a finite exact float"
      raise ValueError(msg)
    if tolerance <= 0.0 or tolerance >= 1.0:
      msg = "geometry relative tolerance must lie strictly between zero and one"
      raise ValueError(msg)


def _fail(code: str, message: str, source: SourceContext) -> None:
  raise ModelCompilationError(
    (ModelCompilationDiagnostic(code=code, message=message, source=source),)
  )


def _sort_key(value: SpecId) -> tuple[int, object]:
  return (0, value) if type(value) is int else (1, value)


def _source(value: SourceContext) -> CompiledSource:
  return _new(
    CompiledSource,
    source=value.source,
    line=value.line,
    column=value.column,
  )


def _source_manifest(source: CompiledSource) -> dict[str, object]:
  return {"source": source.source, "line": source.line, "column": source.column}


def _validated_policy(
  policy: SystemCompilationPolicy | None,
  source: SourceContext,
) -> SystemCompilationPolicy:
  if policy is None:
    return SystemCompilationPolicy()
  if type(policy) is not SystemCompilationPolicy:
    _fail(
      "invalid-compilation-policy",
      "compiler policy must be an exact SystemCompilationPolicy",
      source,
    )
  try:
    return SystemCompilationPolicy(
      dense_index_dtype=object.__getattribute__(policy, "dense_index_dtype"),
      geometry_relative_tolerance=object.__getattribute__(
        policy,
        "geometry_relative_tolerance",
      ),
    )
  except (AttributeError, TypeError, ValueError):
    _fail(
      "invalid-compilation-policy",
      "compiler policy has malformed numeric conventions",
      source,
    )


def _index_dtype(
  *,
  policy: SystemCompilationPolicy,
  point_count: int,
  coefficient_count: int,
  cell_count: int,
  source: SourceContext,
) -> np.dtype:
  dtype = np.dtype(policy.dense_index_dtype)
  required_max = max(point_count - 1, coefficient_count - 1, cell_count - 1, 0)
  if required_max > int(np.iinfo(dtype).max):
    _fail(
      "dense-index-overflow",
      "dense index dtype cannot represent the compiled entities and coefficients",
      source,
    )
  return dtype


def compile_discrete_spaces(
  point_block: PointEntityBlock,
  fields: tuple[FieldSpec, ...],
  *,
  index_dtype: np.dtype = np.dtype(np.int64),
) -> tuple[DiscreteSpace, ...]:
  """Allocate ordered disjoint native coefficient maps for explicit fields.

  This structural helper consumes already-normalized declarations. It performs no
  normalization and is intentionally not re-exported as public API.
  """
  ordered = tuple(sorted(fields, key=lambda item: _sort_key(item.id)))
  offset = 0
  spaces: list[DiscreteSpace] = []
  point_count = len(point_block.entity_ids)
  for field in ordered:
    if field.location != "node":
      _fail(
        "unsupported-space-support",
        "this G1 compiler allocates only truthful node-supported spaces",
        field.source,
      )
    component_count = len(field.components)
    count = point_count * component_count
    coefficient_map = np.arange(offset, offset + count, dtype=index_dtype).reshape(
      point_count,
      component_count,
    )
    coefficient_ids = tuple(
      (field.id, point_id, component)
      for point_id in point_block.entity_ids
      for component in field.components
    )
    spaces.append(
      _new(
        DiscreteSpace,
        space_id=field.id,
        support_block_id=point_block.block_id,
        basis_id="nodal-lagrange",
        components=field.components,
        coefficient_ids=coefficient_ids,
        coefficient_map=FinalizedArray(coefficient_map, dtype=index_dtype),
        coefficient_range=(offset, offset + count),
      )
    )
    offset += count
  return tuple(spaces)


def _attribution(
  spec: ModelSpec,
  *,
  operator_block_id: tuple[SpecId, SpecId],
) -> tuple[SourceAttribution, ...]:
  records: list[SourceAttribution] = []

  def add(kind: str, semantic_id: object, source: SourceContext) -> None:
    records.append(
      _new(
        SourceAttribution,
        kind=kind,
        semantic_id=semantic_id,
        source=_source(source),
      )
    )

  add("model", "model", spec.source)
  add("mesh", "mesh", spec.mesh.source)
  for node in sorted(spec.mesh.nodes, key=lambda item: _sort_key(item.id)):
    add("node", node.id, node.source)
  for field in sorted(spec.fields, key=lambda item: _sort_key(item.id)):
    add("space", field.id, field.source)
    for component in field.components:
      add("space_component", (field.id, component), field.source)
  for block in sorted(spec.mesh.cell_blocks, key=lambda item: _sort_key(item.id)):
    add("entity_block", block.id, block.source)
    for cell in sorted(block.cells, key=lambda item: _sort_key(item.id)):
      add("cell", (block.id, cell.id), cell.source)
  for material in sorted(spec.materials, key=lambda item: _sort_key(item.id)):
    add("material", material.id, material.source)
    for parameter in sorted(material.parameters, key=lambda item: item.name):
      add(
        "material_parameter",
        (material.id, parameter.name),
        parameter.source,
      )
  for region in sorted(spec.regions, key=lambda item: _sort_key(item.id)):
    add("region", region.id, region.source)
  add("operator", operator_block_id, spec.regions[0].source)
  return tuple(records)


def compile_system(
  spec: ModelSpec,
  registry: dict[RegistryKey, RegistryDescriptor],
  *,
  policy: SystemCompilationPolicy | None = None,
) -> CompiledSystem:
  """Normalize once and compile directly to the unexported generic system."""
  normalized = normalize_model_spec(spec)
  selected_policy = _validated_policy(policy, normalized.source)
  selection = _continuum_builder.select_model(normalized)
  snapshot = _continuum_builder.capture_registry(registry, selection)

  nodes = tuple(sorted(normalized.mesh.nodes, key=lambda item: _sort_key(item.id)))
  total_coefficients = sum(
    len(nodes) * len(field.components) for field in normalized.fields
  )
  index_dtype = _index_dtype(
    policy=selected_policy,
    point_count=len(nodes),
    coefficient_count=total_coefficients,
    cell_count=len(selection.cells),
    source=normalized.mesh.source,
  )
  coordinates = FinalizedArray(
    [node.coordinates for node in nodes],
    dtype=np.float64,
  )
  point_block = _new(
    PointEntityBlock,
    block_id="nodes",
    entity_ids=tuple(node.id for node in nodes),
    sources=tuple(_source(node.source) for node in nodes),
    reference_coordinates=coordinates,
  )
  spaces = compile_discrete_spaces(
    point_block,
    normalized.fields,
    index_dtype=index_dtype,
  )
  selected_space = next(
    space for space in spaces if space.space_id == selection.field.id
  )
  node_dense = {node.id: index for index, node in enumerate(nodes)}
  entity_block, operator = _continuum_builder.compile_operator(
    selection,
    coordinates=coordinates,
    node_dense=node_dense,
    space=selected_space,
    snapshot=snapshot,
    index_dtype=index_dtype,
    geometry_relative_tolerance=selected_policy.geometry_relative_tolerance,
  )
  attributions = _attribution(
    normalized,
    operator_block_id=operator.header.block_id,
  )
  manifest = CanonicalManifest(
    {
      "schema": COMPILED_SYSTEM_MANIFEST_SCHEMA,
      "numeric_policy": {
        "floating_dtype": np.dtype(np.float64).str,
        "dense_index_dtype": index_dtype.str,
        "geometry_relative_tolerance": selected_policy.geometry_relative_tolerance,
      },
      "registry_snapshot": snapshot.manifest,
      "points": {
        "block_id": point_block.block_id,
        "entity_ids": point_block.entity_ids,
        "reference_coordinates": point_block.reference_coordinates.values,
      },
      "entity_blocks": [
        {
          "block_id": entity_block.block_id,
          "entity_ids": entity_block.entity_ids,
          "incidence": entity_block.incidence.values,
        }
      ],
      "spaces": [
        {
          "space_id": space.space_id,
          "support_block_id": space.support_block_id,
          "basis_id": space.basis_id,
          "components": space.components,
          "coefficient_ids": space.coefficient_ids,
          "coefficient_map": space.coefficient_map.values,
          "coefficient_range": space.coefficient_range,
        }
        for space in spaces
      ],
      "operators": [operator.content_manifest],
      "source_attribution": [
        {
          "kind": record.kind,
          "semantic_id": record.semantic_id,
          "source": _source_manifest(record.source),
        }
        for record in attributions
      ],
    }
  )
  provenance = _new(
    SystemProvenance,
    schema=COMPILED_SYSTEM_MANIFEST_SCHEMA,
    manifest=manifest,
    registry_fingerprint=snapshot.fingerprint,
    floating_dtype=np.dtype(np.float64).str,
    dense_index_dtype=index_dtype.str,
    geometry_relative_tolerance=selected_policy.geometry_relative_tolerance,
  )
  return _new(
    CompiledSystem,
    instance_id=InstanceId(),
    content_fingerprint=ContentFingerprint.from_manifest(manifest),
    provenance=provenance,
    registry_snapshot=snapshot,
    point_blocks=(point_block,),
    entity_blocks=(entity_block,),
    spaces=spaces,
    operators=(operator,),
    source_attribution=attributions,
  )
