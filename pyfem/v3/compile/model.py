"""Compile the frozen normalized Q8 plane-stress model slice."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NoReturn, final

import numpy as np

from pyfem.v3.compile.contracts import (
  Q8_FORMULATION_KEY,
  Q8_MATERIAL_KEY,
  Q8_QUADRATURE_KEY,
  Q8_REQUIRED_REGISTRY_KEYS,
  Q8_TOPOLOGY_KEY,
  q8_descriptor_metadata,
)
from pyfem.v3.compile.diagnostics import (
  ModelCompilationDiagnostic,
  ModelCompilationError,
)
from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.compiled import (
  BlockStateLayout,
  CompiledCellBlock,
  CompiledMesh,
  CompiledModel,
  CompiledSource,
  DescriptorIdentity,
  DofPlan,
  DomainBlock,
  ElementCouplingRecipe,
  EntityIndex,
  EntityRecord,
  IntegrationLayout,
  ModelAssemblyTopology,
  ModelCapabilities,
  ModelProvenance,
  PhysicalStateLayout,
  PrimaryFieldLayout,
  SourceMap,
  SourceRecord,
)
from pyfem.v3.model.identity import InstanceId
from pyfem.v3.model.provenance import CanonicalManifest, ContentFingerprint
from pyfem.v3.model.registry import (
  RegistryDescriptor,
  RegistryKey,
  RegistrySnapshot,
)
from pyfem.v3.spec.diagnostics import SourceContext, _render_diagnostic_value
from pyfem.v3.spec.model import (
  CellBlockSpec,
  CellSpec,
  FieldSpec,
  MaterialSpec,
  ModelSpec,
  NodeSpec,
  RegionSpec,
  SpecId,
)
from pyfem.v3.spec.normalize import normalize_model_spec

COMPILED_MODEL_MANIFEST_SCHEMA = "pyfem-v3-compiled-model-q8-v1"
_FLOATING_DTYPE = np.dtype(np.float64)
_SUPPORTED_INDEX_DTYPES = ("int8", "int16", "int32", "int64")
_MATERIAL_PARAMETER_NAMES = ("youngs_modulus", "poisson_ratio")


@final
@dataclass(frozen=True, slots=True)
class ModelCompilationPolicy:
  """Explicit numeric policy included in compiled-model content identity."""

  dense_index_dtype: str = "int64"
  geometry_relative_tolerance: float = 1.0e-12

  def __post_init__(self) -> None:
    if (
      type(self.dense_index_dtype) is not str
      or self.dense_index_dtype not in _SUPPORTED_INDEX_DTYPES
    ):
      msg = "dense index dtype must be one of int8, int16, int32, or int64"
      raise ValueError(msg)
    tolerance = self.geometry_relative_tolerance
    if type(tolerance) is not float or not math.isfinite(tolerance):
      msg = "geometry relative tolerance must be a finite exact float"
      raise ValueError(msg)
    if tolerance <= 0.0 or tolerance >= 1.0:
      msg = "geometry relative tolerance must lie strictly between zero and one"
      raise ValueError(msg)

  def __init_subclass__(cls, **kwargs: object) -> None:
    del kwargs
    msg = "ModelCompilationPolicy is runtime-final and cannot be subclassed"
    raise TypeError(msg)


def _fail(code: str, message: str, source: SourceContext) -> NoReturn:
  raise ModelCompilationError(
    (ModelCompilationDiagnostic(code=code, message=message, source=source),)
  )


def _semantic_sort_key(
  value: SpecId | tuple[SpecId, ...],
) -> tuple[int, object]:
  if type(value) is int:
    return 0, value
  if type(value) is str:
    return 1, value
  return 2, tuple(_semantic_sort_key(item) for item in value)


def _copy_source(source: SourceContext) -> CompiledSource:
  return CompiledSource(
    source=source.source,
    line=source.line,
    column=source.column,
  )


def _source_manifest(source: CompiledSource) -> dict[str, object]:
  return {
    "column": source.column,
    "line": source.line,
    "source": source.source,
  }


def _validated_policy(
  value: ModelCompilationPolicy | None,
  source: SourceContext,
) -> ModelCompilationPolicy:
  if value is None:
    return ModelCompilationPolicy()
  if type(value) is not ModelCompilationPolicy:
    _fail(
      "invalid-compilation-policy",
      "compiler policy must be an exact ModelCompilationPolicy",
      source,
    )
  try:
    dense_index_dtype = object.__getattribute__(value, "dense_index_dtype")
    geometry_tolerance = object.__getattribute__(
      value,
      "geometry_relative_tolerance",
    )
  except AttributeError:
    _fail(
      "invalid-compilation-policy",
      "compiler policy must initialize every canonical field",
      source,
    )
  if (
    type(dense_index_dtype) is not str
    or dense_index_dtype not in _SUPPORTED_INDEX_DTYPES
    or type(geometry_tolerance) is not float
    or not math.isfinite(geometry_tolerance)
    or geometry_tolerance <= 0.0
    or geometry_tolerance >= 1.0
  ):
    _fail(
      "invalid-compilation-policy",
      "compiler policy contains an unsupported index dtype or geometry tolerance",
      source,
    )
  return ModelCompilationPolicy(
    dense_index_dtype=dense_index_dtype,
    geometry_relative_tolerance=geometry_tolerance,
  )


def _validate_total_membership(spec: ModelSpec) -> None:
  cells: dict[tuple[SpecId, SpecId], CellSpec] = {}
  for block in spec.mesh.cell_blocks:
    for cell in block.cells:
      cells[(block.id, cell.id)] = cell

  counts = {key: 0 for key in cells}
  for region in spec.regions:
    for cell_ref in region.cell_refs:
      key = (cell_ref.block_id, cell_ref.cell_id)
      if key in counts:
        counts[key] += 1

  for key in sorted(counts, key=_semantic_sort_key):
    count = counts[key]
    cell = cells[key]
    rendered = _render_diagnostic_value(key)
    if count == 0:
      _fail(
        "incomplete-cell-membership",
        f"source cell {rendered} does not belong to a compiled region",
        cell.source,
      )
    if count > 1:
      _fail(
        "multiple-cell-membership",
        f"source cell {rendered} belongs to more than one compiled region",
        cell.source,
      )


def _validate_supported_slice(
  spec: ModelSpec,
) -> tuple[CellBlockSpec, FieldSpec, MaterialSpec, RegionSpec]:
  _validate_total_membership(spec)
  if len(spec.mesh.cell_blocks) != 1:
    _fail(
      "unsupported-cell-block-count",
      "the frozen Q8 compiler requires exactly one cell block",
      spec.mesh.source,
    )
  if len(spec.fields) != 1:
    _fail(
      "unsupported-field-count",
      "the frozen Q8 compiler requires exactly one node field",
      spec.source,
    )
  if len(spec.materials) != 1:
    _fail(
      "unsupported-material-count",
      "the frozen Q8 compiler requires exactly one material",
      spec.source,
    )
  if len(spec.regions) != 1:
    _fail(
      "unsupported-region-count",
      "the frozen Q8 compiler requires exactly one region",
      spec.source,
    )

  block = spec.mesh.cell_blocks[0]
  field = spec.fields[0]
  material = spec.materials[0]
  region = spec.regions[0]

  if (
    block.reference_topology != "quadrilateral"
    or block.topological_dimension != 2
    or block.embedding_dimension != 2
    or block.geometry_interpolation != "serendipity-quad8"
  ):
    _fail(
      "incompatible-cell-block",
      "the supported cell block is an explicit 2D quadrilateral using "
      "serendipity-quad8 geometry interpolation",
      block.source,
    )
  for cell in block.cells:
    if len(cell.node_ids) != 8:
      _fail(
        "invalid-q8-arity",
        f"Q8 cell {_render_diagnostic_value(cell.id)} must reference exactly "
        "eight nodes in the frozen local order",
        cell.source,
      )

  if field.location != "node" or field.components != ("x", "y"):
    _fail(
      "incompatible-field-signature",
      "the supported field is node-located with physical components ('x', 'y')",
      field.source,
    )
  if region.field_ids != (field.id,):
    _fail(
      "incompatible-region-field-signature",
      "the supported region must reference exactly the one declared node field",
      region.source,
    )
  if region.material_id != material.id:
    _fail(
      "incompatible-region-material",
      "the supported region must reference exactly the one declared material",
      region.source,
    )
  if region.formulation != Q8_FORMULATION_KEY[1]:
    _fail(
      "incompatible-formulation",
      f"the supported formulation is {Q8_FORMULATION_KEY[1]!r}",
      region.source,
    )
  if region.quadrature != Q8_QUADRATURE_KEY[1]:
    _fail(
      "incompatible-quadrature",
      f"the supported quadrature is {Q8_QUADRATURE_KEY[1]!r}",
      region.source,
    )
  if material.model != Q8_MATERIAL_KEY[1]:
    _fail(
      "incompatible-material-model",
      f"the supported material model is {Q8_MATERIAL_KEY[1]!r}",
      material.source,
    )
  return block, field, material, region


def _capture_registry(
  registry: dict[RegistryKey, RegistryDescriptor],
  source: SourceContext,
) -> RegistrySnapshot:
  try:
    return RegistrySnapshot.capture(
      registry,
      required=Q8_REQUIRED_REGISTRY_KEYS,
    )
  except (KeyError, TypeError, ValueError):
    _fail(
      "registry-capture-failed",
      "the injected registry could not capture exactly the four required Q8 "
      "descriptor keys",
      source,
    )


def _resolve_descriptors(
  snapshot: RegistrySnapshot,
  *,
  block_source: SourceContext,
  region_source: SourceContext,
  material_source: SourceContext,
) -> dict[RegistryKey, RegistryDescriptor]:
  descriptor_sources = {
    Q8_TOPOLOGY_KEY: block_source,
    Q8_QUADRATURE_KEY: region_source,
    Q8_FORMULATION_KEY: region_source,
    Q8_MATERIAL_KEY: material_source,
  }
  descriptors: dict[RegistryKey, RegistryDescriptor] = {}
  for key in Q8_REQUIRED_REGISTRY_KEYS:
    source = descriptor_sources[key]
    try:
      descriptor = snapshot.resolve(*key)
      expected = CanonicalManifest(q8_descriptor_metadata(*key))
      compatible = descriptor.metadata.to_bytes() == expected.to_bytes()
    except (KeyError, TypeError, ValueError):
      _fail(
        "malformed-registry-descriptor",
        f"captured descriptor {key!r} is malformed",
        source,
      )
    if not compatible:
      _fail(
        "incompatible-registry-descriptor",
        f"descriptor {key!r} does not carry the frozen canonical Q8 metadata",
        source,
      )
    descriptors[key] = descriptor
  return descriptors


def _float_binding_array(
  value: object,
  *,
  shape: tuple[int, ...],
  code: str,
  label: str,
  source: SourceContext,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
  if (
    type(value) is not np.ndarray
    or value.shape != shape
    or value.dtype.kind not in "iuf"
  ):
    _fail(
      code,
      f"{label} binding must return a plain numeric array with shape {shape!r}",
      source,
    )
  try:
    captured = np.array(
      value,
      dtype=np.float64,
      order="C",
      copy=True,
      subok=False,
    )
  except (OverflowError, TypeError, ValueError):
    _fail(code, f"{label} binding output cannot be represented as float64", source)
  if not bool(np.isfinite(captured).all()):
    _fail(code, f"{label} binding output must be finite", source)
  return captured


def _evaluate_descriptor_recipes(
  descriptors: dict[RegistryKey, RegistryDescriptor],
  *,
  block_source: SourceContext,
  region_source: SourceContext,
) -> tuple[
  np.ndarray[tuple[int, ...], np.dtype[np.float64]],
  np.ndarray[tuple[int, ...], np.dtype[np.float64]],
  np.ndarray[tuple[int, ...], np.dtype[np.float64]],
  np.ndarray[tuple[int, ...], np.dtype[np.float64]],
]:
  quadrature_binding = descriptors[Q8_QUADRATURE_KEY].binding
  try:
    quadrature_result = quadrature_binding(3)
  except Exception:
    _fail(
      "quadrature-binding-failed",
      "the captured gauss-3x3 binding failed for its frozen order argument",
      region_source,
    )
  if type(quadrature_result) is not tuple or len(quadrature_result) != 2:
    _fail(
      "invalid-quadrature-binding-output",
      "the gauss-3x3 binding must return exactly (points, weights)",
      region_source,
    )
  points = _float_binding_array(
    quadrature_result[0],
    shape=(9, 2),
    code="invalid-quadrature-binding-output",
    label="gauss-3x3 points",
    source=region_source,
  )
  weights = _float_binding_array(
    quadrature_result[1],
    shape=(9,),
    code="invalid-quadrature-binding-output",
    label="gauss-3x3 weights",
    source=region_source,
  )
  if (
    bool(np.any(weights <= 0.0))
    or bool(np.any(np.abs(points) > 1.0))
    or not math.isclose(float(weights.sum()), 4.0, rel_tol=1.0e-14, abs_tol=1.0e-14)
  ):
    _fail(
      "invalid-quadrature-binding-output",
      "the gauss-3x3 recipe must contain positive in-domain weights summing to four",
      region_source,
    )

  topology_binding = descriptors[Q8_TOPOLOGY_KEY].binding
  try:
    topology_result = topology_binding(np.array(points, copy=True))
  except Exception:
    _fail(
      "topology-binding-failed",
      "the captured serendipity-quad8 binding failed at the frozen quadrature points",
      block_source,
    )
  if type(topology_result) is not tuple or len(topology_result) != 2:
    _fail(
      "invalid-topology-binding-output",
      "the serendipity-quad8 binding must return exactly (values, gradients)",
      block_source,
    )
  shape_values = _float_binding_array(
    topology_result[0],
    shape=(9, 8),
    code="invalid-topology-binding-output",
    label="serendipity-quad8 shape values",
    source=block_source,
  )
  parent_gradients = _float_binding_array(
    topology_result[1],
    shape=(9, 8, 2),
    code="invalid-topology-binding-output",
    label="serendipity-quad8 parent gradients",
    source=block_source,
  )
  if not bool(
    np.allclose(shape_values.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-12)
  ) or not bool(np.allclose(parent_gradients.sum(axis=1), 0.0, rtol=0.0, atol=1.0e-12)):
    _fail(
      "invalid-topology-binding-output",
      "the serendipity-quad8 recipe violates partition or gradient completeness",
      block_source,
    )
  return points, weights, shape_values, parent_gradients


def _material_float64_scalar(
  value: object,
  *,
  name: str,
  source: SourceContext,
) -> float:
  if type(value) is not int and type(value) is not float:
    _fail(
      "invalid-material-parameter-type",
      f"{name} must be an exact integer or float scalar, excluding bool",
      source,
    )
  try:
    converted = float(value)
  except OverflowError:
    _fail(
      "invalid-material-parameter-value",
      f"{name} cannot be represented as finite float64",
      source,
    )
  if not math.isfinite(converted):
    _fail(
      "invalid-material-parameter-value",
      f"{name} cannot be represented as finite float64",
      source,
    )
  return converted


def _material_parameters(material: MaterialSpec) -> tuple[float, float]:
  by_name = {parameter.name: parameter for parameter in material.parameters}
  if tuple(sorted(by_name)) != tuple(sorted(_MATERIAL_PARAMETER_NAMES)):
    _fail(
      "invalid-material-parameter-schema",
      "plane-stress-linear-elastic requires exactly youngs_modulus and poisson_ratio",
      material.source,
    )
  youngs_parameter = by_name["youngs_modulus"]
  poisson_parameter = by_name["poisson_ratio"]
  youngs_modulus = _material_float64_scalar(
    youngs_parameter.value,
    name="youngs_modulus",
    source=youngs_parameter.source,
  )
  poisson_ratio = _material_float64_scalar(
    poisson_parameter.value,
    name="poisson_ratio",
    source=poisson_parameter.source,
  )
  if youngs_modulus <= 0.0:
    _fail(
      "invalid-youngs-modulus",
      "youngs_modulus must be strictly positive",
      youngs_parameter.source,
    )
  if not -1.0 < poisson_ratio < 0.5:
    _fail(
      "invalid-poisson-ratio",
      "poisson_ratio must lie strictly between -1 and 0.5",
      poisson_parameter.source,
    )
  return youngs_modulus, poisson_ratio


def _check_index_capacity(
  *,
  node_count: int,
  cell_count: int,
  point_count: int,
  policy: ModelCompilationPolicy,
  source: SourceContext,
) -> np.dtype[np.signedinteger]:
  dtype = np.dtype(policy.dense_index_dtype)
  limit = int(np.iinfo(dtype).max)
  required_max = max(
    node_count - 1,
    cell_count - 1,
    node_count * 2 - 1,
    cell_count * point_count - 1,
    15,
  )
  if required_max > limit:
    _fail(
      "dense-index-overflow",
      f"dense index dtype {policy.dense_index_dtype!r} cannot represent the "
      "compiled node, cell, DOF, or integration layout",
      source,
    )
  return dtype


def _scaled_relative_coordinates(
  coordinates: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]] | None:
  with np.errstate(over="ignore", invalid="ignore", under="ignore"):
    relative = coordinates - coordinates[0]
  if not bool(np.isfinite(relative).all()):
    coordinate_scale = float(np.max(np.abs(coordinates)))
    if not math.isfinite(coordinate_scale) or coordinate_scale == 0.0:
      return None
    scaled = coordinates / coordinate_scale
    relative = scaled - scaled[0]
  cell_scale = float(np.max(np.abs(relative)))
  if not math.isfinite(cell_scale) or cell_scale == 0.0:
    return None
  normalized = relative / cell_scale
  if not bool(np.isfinite(normalized).all()):
    return None
  return normalized


def _audit_reference_geometry(
  *,
  coordinates: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
  connectivity: np.ndarray[tuple[int, ...], np.dtype[np.signedinteger]],
  parent_gradients: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
  cells: tuple[CellSpec, ...],
  policy: ModelCompilationPolicy,
) -> None:
  tolerance = policy.geometry_relative_tolerance
  for cell_index, cell in enumerate(cells):
    cell_coordinates = coordinates[connectivity[cell_index]]
    normalized = _scaled_relative_coordinates(cell_coordinates)
    rendered_id = _render_diagnostic_value(cell.id)
    if normalized is None:
      _fail(
        "non-finite-reference-geometry",
        f"Q8 cell {rendered_id} has a non-finite or zero-scale reference mapping",
        cell.source,
      )

    determinants: list[float] = []
    for point_index in range(parent_gradients.shape[0]):
      jacobian = normalized.T @ parent_gradients[point_index]
      determinant = float(
        jacobian[0, 0] * jacobian[1, 1] - jacobian[0, 1] * jacobian[1, 0]
      )
      norm_squared = float(np.sum(jacobian * jacobian))
      if (
        not math.isfinite(determinant)
        or not math.isfinite(norm_squared)
        or norm_squared == 0.0
      ):
        _fail(
          "non-finite-reference-geometry",
          f"Q8 cell {rendered_id} has a non-finite reference Jacobian",
          cell.source,
        )
      relative_determinant = abs(determinant)
      condition_ratio = relative_determinant / norm_squared
      if relative_determinant <= tolerance or condition_ratio <= tolerance:
        _fail(
          "near-singular-reference-geometry",
          f"Q8 cell {rendered_id} has a scale-relative near-singular reference "
          f"Jacobian at local quadrature point {point_index}",
          cell.source,
        )
      determinants.append(determinant)

    has_positive = any(value > 0.0 for value in determinants)
    has_negative = any(value < 0.0 for value in determinants)
    if has_positive and has_negative:
      _fail(
        "sign-changing-reference-geometry",
        f"Q8 cell {rendered_id} changes Jacobian orientation across quadrature points",
        cell.source,
      )
    if has_negative:
      _fail(
        "inverted-reference-geometry",
        f"Q8 cell {rendered_id} has uniformly negative reference orientation",
        cell.source,
      )


def _descriptor_identity(descriptor: RegistryDescriptor) -> DescriptorIdentity:
  return DescriptorIdentity(
    kind=descriptor.kind,
    name=descriptor.name,
    version=descriptor.version,
    implementation_id=descriptor.implementation_id,
  )


def _build_entity_and_source_maps(
  *,
  spec: ModelSpec,
  block: CellBlockSpec,
  field: FieldSpec,
  material: MaterialSpec,
  region: RegionSpec,
  nodes: tuple[NodeSpec, ...],
  cells: tuple[CellSpec, ...],
) -> tuple[EntityIndex, SourceMap]:
  entities: list[EntityRecord] = []
  sources: list[SourceRecord] = []

  def add(
    kind: str,
    semantic_id: SpecId | tuple[SpecId | int, ...],
    source: SourceContext,
    *,
    dense_index: int | None,
    block_index: int | None,
    local_index: int | None,
  ) -> None:
    entities.append(
      EntityRecord(
        kind=kind,
        semantic_id=semantic_id,
        dense_index=dense_index,
        block_index=block_index,
        local_index=local_index,
      )
    )
    sources.append(
      SourceRecord(
        kind=kind,
        semantic_id=semantic_id,
        source=_copy_source(source),
      )
    )

  add(
    "model",
    "model",
    spec.source,
    dense_index=0,
    block_index=None,
    local_index=None,
  )
  add(
    "mesh",
    "mesh",
    spec.mesh.source,
    dense_index=0,
    block_index=None,
    local_index=None,
  )
  for dense_index, node in enumerate(nodes):
    add(
      "node",
      node.id,
      node.source,
      dense_index=dense_index,
      block_index=None,
      local_index=dense_index,
    )
  add(
    "cell_block",
    block.id,
    block.source,
    dense_index=0,
    block_index=0,
    local_index=0,
  )
  for dense_index, cell in enumerate(cells):
    add(
      "cell",
      (block.id, cell.id),
      cell.source,
      dense_index=dense_index,
      block_index=0,
      local_index=dense_index,
    )
  add(
    "field",
    field.id,
    field.source,
    dense_index=0,
    block_index=None,
    local_index=0,
  )
  for component_index, component in enumerate(field.components):
    add(
      "field_component",
      (field.id, component),
      field.source,
      dense_index=component_index,
      block_index=None,
      local_index=component_index,
    )
  add(
    "material",
    material.id,
    material.source,
    dense_index=0,
    block_index=None,
    local_index=0,
  )
  for parameter_index, parameter_name in enumerate(_MATERIAL_PARAMETER_NAMES):
    parameter = next(
      item for item in material.parameters if item.name == parameter_name
    )
    add(
      "material_parameter",
      (material.id, parameter_name),
      parameter.source,
      dense_index=parameter_index,
      block_index=None,
      local_index=parameter_index,
    )
  add(
    "region",
    region.id,
    region.source,
    dense_index=0,
    block_index=0,
    local_index=0,
  )
  domain_block_id = (block.id, region.id)
  add(
    "domain_block",
    domain_block_id,
    region.source,
    dense_index=0,
    block_index=0,
    local_index=0,
  )

  for node_index, node in enumerate(nodes):
    for component_index, component in enumerate(field.components):
      dof_index = node_index * len(field.components) + component_index
      add(
        "dof",
        (node.id, field.id, component),
        node.source,
        dense_index=dof_index,
        block_index=None,
        local_index=dof_index,
      )
  for cell_index, cell in enumerate(cells):
    for point_index in range(9):
      integration_index = cell_index * 9 + point_index
      add(
        "integration_point",
        (block.id, cell.id, point_index),
        cell.source,
        dense_index=integration_index,
        block_index=0,
        local_index=integration_index,
      )
  return EntityIndex(tuple(entities)), SourceMap(tuple(sources))


def _model_manifest(
  *,
  policy: ModelCompilationPolicy,
  registry_snapshot: RegistrySnapshot,
  mesh: CompiledMesh,
  dofs: DofPlan,
  domain_block: DomainBlock,
  assembly_topology: ModelAssemblyTopology,
  physical_state_layout: PhysicalStateLayout,
  capabilities: ModelCapabilities,
  entity_index: EntityIndex,
  source_map: SourceMap,
) -> CanonicalManifest:
  cell_block = mesh.cell_blocks[0]
  return CanonicalManifest(
    {
      "schema": COMPILED_MODEL_MANIFEST_SCHEMA,
      "numeric_policy": {
        "dense_index_dtype": np.dtype(policy.dense_index_dtype).str,
        "floating_dtype": _FLOATING_DTYPE.str,
        "geometry_relative_tolerance": policy.geometry_relative_tolerance,
      },
      "registry_snapshot": registry_snapshot.manifest,
      "mesh": {
        "node_ids": mesh.node_ids,
        "coordinates": mesh.coordinates.values,
        "cell_blocks": [
          {
            "id": cell_block.id,
            "reference_topology": cell_block.reference_topology,
            "topological_dimension": cell_block.topological_dimension,
            "embedding_dimension": cell_block.embedding_dimension,
            "geometry_interpolation": cell_block.geometry_interpolation,
            "cell_ids": cell_block.cell_ids,
            "connectivity": cell_block.connectivity.values,
          }
        ],
      },
      "dofs": {
        "field_id": dofs.field_id,
        "components": dofs.components,
        "node_ids": dofs.node_ids,
        "node_component_dofs": dofs.node_component_dofs.values,
        "global_size": dofs.global_size,
      },
      "domain_blocks": [
        {
          "block_id": domain_block.block_id,
          "source_cell_block_id": domain_block.source_cell_block_id,
          "source_region_id": domain_block.source_region_id,
          "cell_ids": domain_block.cell_ids,
          "field_id": domain_block.field_id,
          "source_material_id": domain_block.source_material_id,
          "field_components": domain_block.field_components,
          "descriptors": [
            {
              "kind": descriptor.kind,
              "name": descriptor.name,
              "version": descriptor.version,
              "implementation_id": descriptor.implementation_id,
            }
            for descriptor in (
              domain_block.topology,
              domain_block.quadrature,
              domain_block.formulation,
              domain_block.material,
            )
          ],
          "connectivity": domain_block.connectivity.values,
          "dof_map": domain_block.dof_map.values,
          "quadrature_points": domain_block.quadrature_points.values,
          "quadrature_weights": domain_block.quadrature_weights.values,
          "shape_values": domain_block.shape_values.values,
          "parent_gradients": domain_block.parent_gradients.values,
          "material_parameter_names": domain_block.material_parameter_names,
          "material_parameters": domain_block.material_parameters.values,
          "integration_layout": {
            "points_per_element": (domain_block.integration_layout.points_per_element),
            "material_slots_per_point": (
              domain_block.integration_layout.material_slots_per_point
            ),
            "local_point_ids": domain_block.integration_layout.local_point_ids,
          },
          "kinematic_regime": domain_block.kinematic_regime,
          "strain_voigt_order": domain_block.strain_voigt_order,
          "shear_convention": domain_block.shear_convention,
          "measure_convention": domain_block.measure_convention,
        }
      ],
      "assembly_topology": {
        "fixed_model_coupling": assembly_topology.fixed_model_coupling,
        "block_recipes": [
          {
            "block_index": recipe.block_index,
            "local_dof_count": recipe.local_dof_count,
            "coupling": recipe.coupling,
            "dof_map": recipe.dof_map.values,
          }
          for recipe in assembly_topology.block_recipes
        ],
      },
      "physical_state_layout": {
        "global_primary_size": physical_state_layout.global_primary_size,
        "evolving_value_count": physical_state_layout.evolving_value_count,
        "primary_fields": [
          {
            "field_id": item.field_id,
            "components": item.components,
            "global_size": item.global_size,
          }
          for item in physical_state_layout.primary_fields
        ],
        "block_states": [
          {
            "block_index": item.block_index,
            "element_count": item.element_count,
            "integration_points_per_element": (item.integration_points_per_element),
            "material_slots_per_point": item.material_slots_per_point,
            "material_history_width": item.material_history_width,
            "formulation_history_width": item.formulation_history_width,
          }
          for item in physical_state_layout.block_states
        ],
      },
      "capabilities": {
        "response_class": capabilities.response_class,
        "fixed_model_coupling": capabilities.fixed_model_coupling,
        "tangent_class": capabilities.tangent_class,
        "tangent_is_symmetric": capabilities.tangent_is_symmetric,
        "tangent_is_constant": capabilities.tangent_is_constant,
        "conservative_internal_contribution": (
          capabilities.conservative_internal_contribution
        ),
        "state_dependent": capabilities.state_dependent,
        "has_storage": capabilities.has_storage,
        "has_mass": capabilities.has_mass,
        "has_damping": capabilities.has_damping,
        "restart_history_required": capabilities.restart_history_required,
        "contribution_channels": capabilities.contribution_channels,
      },
      "entity_index": [
        {
          "kind": record.kind,
          "semantic_id": record.semantic_id,
          "dense_index": record.dense_index,
          "block_index": record.block_index,
          "local_index": record.local_index,
        }
        for record in entity_index.records
      ],
      "source_map": [
        {
          "kind": record.kind,
          "semantic_id": record.semantic_id,
          "source": _source_manifest(record.source),
        }
        for record in source_map.records
      ],
    }
  )


def compile_model(
  spec: ModelSpec,
  registry: dict[RegistryKey, RegistryDescriptor],
  *,
  policy: ModelCompilationPolicy | None = None,
) -> CompiledModel:
  """Normalize and compile the one frozen Q8 plane-stress model recipe.

  Normalization errors deliberately remain ``ModelSpecValidationError``. Every
  covered compatibility, registry, index, or reference-geometry failure is
  converted to ``ModelCompilationError`` before a carrier can escape.
  """
  normalized = normalize_model_spec(spec)
  selected_policy = _validated_policy(policy, normalized.source)
  block, field, material, region = _validate_supported_slice(normalized)
  registry_snapshot = _capture_registry(registry, normalized.source)
  descriptors = _resolve_descriptors(
    registry_snapshot,
    block_source=block.source,
    region_source=region.source,
    material_source=material.source,
  )
  points, weights, shape_values, parent_gradients = _evaluate_descriptor_recipes(
    descriptors,
    block_source=block.source,
    region_source=region.source,
  )
  youngs_modulus, poisson_ratio = _material_parameters(material)

  nodes = tuple(
    sorted(
      normalized.mesh.nodes,
      key=lambda item: _semantic_sort_key(item.id),
    )
  )
  cells = tuple(sorted(block.cells, key=lambda item: _semantic_sort_key(item.id)))
  index_dtype = _check_index_capacity(
    node_count=len(nodes),
    cell_count=len(cells),
    point_count=9,
    policy=selected_policy,
    source=normalized.mesh.source,
  )
  node_dense = {node.id: index for index, node in enumerate(nodes)}
  connectivity_values = [
    [node_dense[node_id] for node_id in cell.node_ids] for cell in cells
  ]
  node_component_dof_values = [
    [node_index * 2, node_index * 2 + 1] for node_index in range(len(nodes))
  ]
  dof_map_values = [
    [
      node_index * 2 + component_index
      for node_index in connectivity
      for component_index in range(2)
    ]
    for connectivity in connectivity_values
  ]

  coordinates = FinalizedArray(
    [node.coordinates for node in nodes],
    dtype=np.float64,
  )
  connectivity = FinalizedArray(connectivity_values, dtype=index_dtype)
  node_component_dofs = FinalizedArray(
    node_component_dof_values,
    dtype=index_dtype,
  )
  dof_map = FinalizedArray(dof_map_values, dtype=index_dtype)
  quadrature_points = FinalizedArray(points, dtype=np.float64)
  quadrature_weights = FinalizedArray(weights, dtype=np.float64)
  compiled_shape_values = FinalizedArray(shape_values, dtype=np.float64)
  compiled_parent_gradients = FinalizedArray(parent_gradients, dtype=np.float64)
  material_parameters = FinalizedArray(
    [[youngs_modulus, poisson_ratio]],
    dtype=np.float64,
  )

  _audit_reference_geometry(
    coordinates=coordinates.values,
    connectivity=connectivity.values,
    parent_gradients=compiled_parent_gradients.values,
    cells=cells,
    policy=selected_policy,
  )

  compiled_cell_block = CompiledCellBlock(
    id=block.id,
    reference_topology=block.reference_topology,
    topological_dimension=block.topological_dimension,
    embedding_dimension=block.embedding_dimension,
    geometry_interpolation=block.geometry_interpolation,
    cell_ids=tuple(cell.id for cell in cells),
    connectivity=connectivity,
  )
  mesh = CompiledMesh(
    node_ids=tuple(node.id for node in nodes),
    coordinates=coordinates,
    cell_blocks=(compiled_cell_block,),
  )
  dofs = DofPlan(
    field_id=field.id,
    components=field.components,
    node_ids=mesh.node_ids,
    node_component_dofs=node_component_dofs,
    global_size=len(nodes) * 2,
  )
  integration_layout = IntegrationLayout(
    points_per_element=9,
    material_slots_per_point=1,
    local_point_ids=tuple(range(9)),
  )
  domain_block = DomainBlock(
    block_id=(block.id, region.id),
    source_cell_block_id=block.id,
    source_region_id=region.id,
    cell_ids=compiled_cell_block.cell_ids,
    field_id=field.id,
    source_material_id=material.id,
    field_components=field.components,
    topology=_descriptor_identity(descriptors[Q8_TOPOLOGY_KEY]),
    quadrature=_descriptor_identity(descriptors[Q8_QUADRATURE_KEY]),
    formulation=_descriptor_identity(descriptors[Q8_FORMULATION_KEY]),
    material=_descriptor_identity(descriptors[Q8_MATERIAL_KEY]),
    connectivity=connectivity,
    dof_map=dof_map,
    quadrature_points=quadrature_points,
    quadrature_weights=quadrature_weights,
    shape_values=compiled_shape_values,
    parent_gradients=compiled_parent_gradients,
    material_parameter_names=_MATERIAL_PARAMETER_NAMES,
    material_parameters=material_parameters,
    integration_layout=integration_layout,
    kinematic_regime="small-strain",
    strain_voigt_order=("xx", "yy", "xy"),
    shear_convention="engineering",
    measure_convention="per-unit-out-of-plane-thickness",
  )
  assembly_topology = ModelAssemblyTopology(
    block_recipes=(
      ElementCouplingRecipe(
        block_index=0,
        local_dof_count=16,
        coupling="full-element-local-dof-clique",
        dof_map=dof_map,
      ),
    ),
    fixed_model_coupling=True,
  )
  physical_state_layout = PhysicalStateLayout(
    primary_fields=(
      PrimaryFieldLayout(
        field_id=field.id,
        components=field.components,
        global_size=dofs.global_size,
      ),
    ),
    block_states=(
      BlockStateLayout(
        block_index=0,
        element_count=len(cells),
        integration_points_per_element=9,
        material_slots_per_point=1,
        material_history_width=0,
        formulation_history_width=0,
      ),
    ),
    global_primary_size=dofs.global_size,
    evolving_value_count=0,
  )
  capabilities = ModelCapabilities(
    response_class="linear-elastic",
    fixed_model_coupling=True,
    tangent_class="symmetric-constant-material",
    tangent_is_symmetric=True,
    tangent_is_constant=True,
    conservative_internal_contribution=True,
    state_dependent=False,
    has_storage=False,
    has_mass=False,
    has_damping=False,
    restart_history_required=False,
    contribution_channels=("internal-force", "material-tangent"),
  )
  entity_index, source_map = _build_entity_and_source_maps(
    spec=normalized,
    block=block,
    field=field,
    material=material,
    region=region,
    nodes=nodes,
    cells=cells,
  )
  manifest = _model_manifest(
    policy=selected_policy,
    registry_snapshot=registry_snapshot,
    mesh=mesh,
    dofs=dofs,
    domain_block=domain_block,
    assembly_topology=assembly_topology,
    physical_state_layout=physical_state_layout,
    capabilities=capabilities,
    entity_index=entity_index,
    source_map=source_map,
  )
  content_fingerprint = ContentFingerprint.from_manifest(manifest)
  provenance = ModelProvenance(
    schema=COMPILED_MODEL_MANIFEST_SCHEMA,
    manifest=manifest,
    registry_fingerprint=registry_snapshot.fingerprint,
    floating_dtype=_FLOATING_DTYPE.str,
    dense_index_dtype=index_dtype.str,
    geometry_relative_tolerance=selected_policy.geometry_relative_tolerance,
  )
  return CompiledModel(
    instance_id=InstanceId(),
    content_fingerprint=content_fingerprint,
    provenance=provenance,
    registry_snapshot=registry_snapshot,
    mesh=mesh,
    dofs=dofs,
    domain_blocks=(domain_block,),
    assembly_topology=assembly_topology,
    physical_state_layout=physical_state_layout,
    capabilities=capabilities,
    entity_index=entity_index,
    source_map=source_map,
  )
