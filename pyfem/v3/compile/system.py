"""Direct normalized-model compiler for the generic compiled-system boundary.

Region-routed families (continuum, truss) are selected from the normalized
spec; the point-spring family is declaration-routed: spring groups bind
researcher kernel callables a pure spec cannot carry, so they enter
compilation through ``compile_system``'s ``springs`` channel and compose
through the landed kernel-parameterized seam.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NoReturn, Protocol, cast

import numpy as np

from pyfem.v3.compile import continuum as _continuum_builder
from pyfem.v3.compile import spring as _spring_builder
from pyfem.v3.compile import truss as _truss_builder
from pyfem.v3.compile.diagnostics import (
  ModelCompilationDiagnostic,
  ModelCompilationError,
)
from pyfem.v3.compile.spring import SpringDeclaration, SpringOperator
from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.identity import InstanceId
from pyfem.v3.model.operator import CompiledOperator
from pyfem.v3.model.provenance import CanonicalManifest, ContentFingerprint
from pyfem.v3.model.registry import (
  RegistryDescriptor,
  RegistryKey,
  RegistrySnapshot,
)
from pyfem.v3.model.system import (
  CompiledSource,
  CompiledSystem,
  DiscreteSpace,
  IncidenceEntityBlock,
  PointEntityBlock,
  SourceAttribution,
  SystemProvenance,
)
from pyfem.v3.spec.diagnostics import SourceContext
from pyfem.v3.spec.model import CellSpec, FieldSpec, ModelSpec, NodeSpec, SpecId
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


class OperatorFamilySelection(Protocol):
  """Validated family-specific spec slice consumed by the generic compiler.

  A selection either is itself a single operator slice (the landed
  ``cells``/``field`` shape, e.g. the truss family) or carries a ``regions``
  tuple of per-region slices with ``cells``/``fields`` members (the continuum
  family); the compiler compiles one operator per slice.
  """

  cells: tuple[CellSpec, ...]
  field: FieldSpec


class OperatorFamilyBuilder(Protocol):
  """Builder module contract behind the operator-family dispatch seam."""

  @staticmethod
  def select_model(spec: ModelSpec) -> OperatorFamilySelection:
    """Validate the authored slice after the sole normalization pass."""
    ...

  @staticmethod
  def capture_registry(
    registry: dict[RegistryKey, RegistryDescriptor],
    selection: OperatorFamilySelection,
  ) -> RegistrySnapshot:
    """Capture and validate exactly the implementations the family selected."""
    ...

  @staticmethod
  def compile_operator(
    selection: OperatorFamilySelection,
    *,
    coordinates: FinalizedArray,
    node_dense: dict[SpecId, int],
    space: DiscreteSpace,
    snapshot: RegistrySnapshot,
    index_dtype: np.dtype,
    geometry_relative_tolerance: float,
  ) -> tuple[IncidenceEntityBlock, CompiledOperator]:
    """Compile the family payload behind the open operator header.

    Single-field families receive the slice's one selected ``space``;
    multi-field families receive the system's ``spaces`` mapping keyed by
    field ID and bind one port per slice field.
    """
    ...


POINT_SPRING_FORMULATION_KEY: RegistryKey = ("formulation", "point-spring")


class SpringFamilyBuilder(Protocol):
  """Declaration-routed family contract behind the spring dispatch entry.

  A point-spring group binds a researcher kernel callable, which a pure
  ``ModelSpec`` cannot express, so the spring family enters compilation
  through ``compile_system``'s ``springs`` channel — one exact
  ``SpringDeclaration`` per authored spring group — instead of through
  region-formulation selection. This is the acceptance shape the wave-8
  converter emission targets: the parser emits the base ``ModelSpec`` for the
  region-routed groups plus one declaration per parsed spring group, and the
  system wiring composes every group in one compiled system.
  """

  @staticmethod
  def select_model(spec: ModelSpec) -> NoReturn:
    """Refuse region-routed selection with a coded diagnostic."""
    ...

  @staticmethod
  def select_declarations(
    declarations: object,
    source: SourceContext,
  ) -> tuple[SpringDeclaration, ...]:
    """Validate the authored spring groups alongside the normalized model."""
    ...

  @staticmethod
  def compile_group(
    system: CompiledSystem,
    declaration: SpringDeclaration,
  ) -> tuple[PointEntityBlock, SpringOperator]:
    """Compile one validated spring group against the composed system."""
    ...

  @staticmethod
  def compose(
    base: CompiledSystem,
    spring_block: PointEntityBlock,
    spring_operator: SpringOperator,
  ) -> CompiledSystem:
    """Compose one compiled spring group into the compiled system."""
    ...


class _PointSpringFamilyBuilder:
  """Spring family dispatch entry delegating to the landed spring seam.

  Every phase routes through the landed kernel-parameterized seam unchanged,
  so the wired path and the standalone
  ``compile_spring_operator``/``compose_system`` sequence produce identical
  operators, coded diagnostics, membership validation, and composed systems.
  """

  @staticmethod
  def select_model(spec: ModelSpec) -> NoReturn:
    region = next(
      (
        item
        for item in spec.regions
        if item.formulation == POINT_SPRING_FORMULATION_KEY[1]
      ),
      None,
    )
    _fail(
      "unsupported-spring-region",
      "point springs cannot compile from a spec region alone: a spring group "
      "binds a kernel callable, which a pure ModelSpec cannot carry; pass one "
      "SpringDeclaration per spring group via compile_system's springs channel",
      region.source if region is not None else spec.source,
    )

  @staticmethod
  def select_declarations(
    declarations: object,
    source: SourceContext,
  ) -> tuple[SpringDeclaration, ...]:
    if type(declarations) is not tuple or any(
      type(item) is not SpringDeclaration for item in declarations
    ):
      _fail(
        "invalid-spring-declarations",
        "spring groups must be an exact tuple of exact SpringDeclaration values",
        source,
      )
    return declarations

  @staticmethod
  def compile_group(
    system: CompiledSystem,
    declaration: SpringDeclaration,
  ) -> tuple[PointEntityBlock, SpringOperator]:
    return _spring_builder.compile_spring_operator(system, declaration)

  @staticmethod
  def compose(
    base: CompiledSystem,
    spring_block: PointEntityBlock,
    spring_operator: SpringOperator,
  ) -> CompiledSystem:
    return _spring_builder.compose_system(base, spring_block, spring_operator)


_OPERATOR_FAMILY_BUILDERS: tuple[
  tuple[str, OperatorFamilyBuilder | SpringFamilyBuilder],
  ...,
] = (
  (_continuum_builder.Q8_FORMULATION_KEY[1], _continuum_builder),
  (_continuum_builder.THERMAL_FORMULATION_KEY[1], _continuum_builder),
  (_continuum_builder.THERMO_FORMULATION_KEY[1], _continuum_builder),
  (_truss_builder.TRUSS_FORMULATION_KEY[1], _truss_builder),
  (POINT_SPRING_FORMULATION_KEY[1], _PointSpringFamilyBuilder),
)


def _operator_family_builder(
  spec: ModelSpec,
) -> OperatorFamilyBuilder | SpringFamilyBuilder:
  """Route the normalized spec to the builder owning its region formulation.

  Specs whose formulations no registered family claims fall back to the first
  builder so its landed coded diagnostics describe the mismatch. A region
  naming the spring formulation routes to the declaration-routed spring
  family, whose selection refuses with a coded diagnostic: spring groups need
  kernel declarations a pure spec cannot carry.
  """
  formulations = {region.formulation for region in spec.regions}
  for name, builder in _OPERATOR_FAMILY_BUILDERS:
    if name in formulations:
      return builder
  return _OPERATOR_FAMILY_BUILDERS[0][1]


def _operator_family(name: str) -> OperatorFamilyBuilder | SpringFamilyBuilder:
  """Return the family builder registered under one exact dispatch name."""
  for family_name, builder in _OPERATOR_FAMILY_BUILDERS:
    if family_name == name:
      return builder
  msg = f"no operator family builder is registered under {name!r}"
  raise KeyError(msg)


def _operator_slices(
  selection: OperatorFamilySelection,
) -> tuple[OperatorFamilySelection, ...]:
  """Return the per-region operator slices of a validated family selection."""
  regions = getattr(selection, "regions", None)
  if regions is None:
    return (selection,)
  return tuple(regions)


def _slice_fields(slice_: OperatorFamilySelection) -> tuple[FieldSpec, ...]:
  """Return the fields one operator slice binds, in port order."""
  fields = getattr(slice_, "fields", None)
  if fields is None:
    return (slice_.field,)
  return tuple(fields)


def _field_supports(
  spec: ModelSpec,
  slices: tuple[OperatorFamilySelection, ...],
  nodes: tuple[NodeSpec, ...],
) -> dict[SpecId, tuple[SpecId, ...]]:
  """Resolve each declared field's support in canonical point-block order.

  A field's support is the union of the nodes of the cells of every region
  referencing it; nodes outside all supports own no coefficients.
  """
  memberships: dict[SpecId, set[SpecId]] = {field.id: set() for field in spec.fields}
  for slice_ in slices:
    for field in _slice_fields(slice_):
      for cell in slice_.cells:
        memberships[field.id].update(cell.node_ids)
  return {
    field.id: tuple(node.id for node in nodes if node.id in memberships[field.id])
    for field in spec.fields
  }


def compile_discrete_spaces(
  point_block: PointEntityBlock,
  fields: tuple[FieldSpec, ...],
  *,
  supports: dict[SpecId, tuple[SpecId, ...]] | None = None,
  index_dtype: np.dtype = np.dtype(np.int64),
) -> tuple[DiscreteSpace, ...]:
  """Allocate ordered disjoint native coefficient maps for explicit fields.

  This structural helper consumes already-normalized declarations. It performs no
  normalization and is intentionally not re-exported as public API.

  Without ``supports`` every field covers the whole point block (the landed
  single-slice semantics). With ``supports`` each field covers exactly its
  listed support entities in point-block order, so partial-support fields own
  no coefficients outside their support: no ghost DOFs. The support of every
  allocated space remains recoverable from its ``coefficient_ids``.
  """
  ordered = tuple(sorted(fields, key=lambda item: _sort_key(item.id)))
  offset = 0
  spaces: list[DiscreteSpace] = []
  for field in ordered:
    if field.location != "node":
      _fail(
        "unsupported-space-support",
        "this G1 compiler allocates only truthful node-supported spaces",
        field.source,
      )
    if supports is None:
      support_ids: tuple[SpecId, ...] = point_block.entity_ids
    else:
      if type(supports) is not dict or type(supports.get(field.id)) is not tuple:
        _fail(
          "invalid-space-support",
          "field supports must be exact tuples of point entities keyed by field ID",
          field.source,
        )
      members = set(supports[field.id])
      support_ids = tuple(
        point_id for point_id in point_block.entity_ids if point_id in members
      )
      if len(support_ids) != len(members):
        _fail(
          "invalid-space-support",
          "field support references entities outside the point block",
          field.source,
        )
    component_count = len(field.components)
    count = len(support_ids) * component_count
    coefficient_map = np.arange(offset, offset + count, dtype=index_dtype).reshape(
      len(support_ids),
      component_count,
    )
    coefficient_ids = tuple(
      (field.id, point_id, component)
      for point_id in support_ids
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
  operator_sources: tuple[tuple[SpecId, SourceContext], ...],
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
  for block_id, source in operator_sources:
    add("operator", block_id, source)
  return tuple(records)


def compile_system(
  spec: ModelSpec,
  registry: dict[RegistryKey, RegistryDescriptor],
  *,
  policy: SystemCompilationPolicy | None = None,
  springs: tuple[SpringDeclaration, ...] = (),
) -> CompiledSystem:
  """Normalize once and compile directly to the unexported generic system.

  ``springs`` declares the point-spring groups of a mixed-family model: one
  exact ``SpringDeclaration`` per group, each binding its own kernel. Groups
  compile in order against the composed system through the landed
  kernel-parameterized seam, so the wired path reproduces the standalone
  ``compile_spring_operator``/``compose_system`` sequence exactly — same
  operators, state layouts, coded diagnostics, membership validation, and
  composed provenance. This is the acceptance shape prepared for the wave-8
  converter: it parses each legacy spring element group into one declaration
  and leaves all spring semantics to this channel; the base spec carries only
  the region-routed families. Without ``springs`` the compiled content is
  byte-identical to the region-routed path alone.
  """
  normalized = normalize_model_spec(spec)
  selected_policy = _validated_policy(policy, normalized.source)
  spring_family = cast(
    SpringFamilyBuilder,
    _operator_family(POINT_SPRING_FORMULATION_KEY[1]),
  )
  declarations = spring_family.select_declarations(springs, normalized.source)
  builder = _operator_family_builder(normalized)
  selection = builder.select_model(normalized)
  snapshot = builder.capture_registry(registry, selection)

  nodes = tuple(sorted(normalized.mesh.nodes, key=lambda item: _sort_key(item.id)))
  slices = _operator_slices(selection)
  supports: dict[SpecId, tuple[SpecId, ...]] | None = None
  if len(normalized.fields) != 1 or len(slices) != 1:
    supports = _field_supports(normalized, slices, nodes)
  total_coefficients = (
    sum(len(nodes) * len(field.components) for field in normalized.fields)
    if supports is None
    else sum(
      len(supports[field.id]) * len(field.components) for field in normalized.fields
    )
  )
  index_dtype = _index_dtype(
    policy=selected_policy,
    point_count=len(nodes),
    coefficient_count=total_coefficients,
    cell_count=max(len(slice_.cells) for slice_ in slices),
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
    supports=supports,
    index_dtype=index_dtype,
  )
  spaces_by_id = {space.space_id: space for space in spaces}
  node_dense = {node.id: index for index, node in enumerate(nodes)}
  entity_blocks: list[IncidenceEntityBlock] = []
  operators: list[CompiledOperator] = []
  operator_sources: list[tuple[SpecId, SourceContext]] = []
  for slice_ in slices:
    if hasattr(slice_, "fields"):
      entity_block, operator = builder.compile_operator(
        slice_,
        coordinates=coordinates,
        node_dense=node_dense,
        spaces=spaces_by_id,
        snapshot=snapshot,
        index_dtype=index_dtype,
        geometry_relative_tolerance=selected_policy.geometry_relative_tolerance,
      )
    else:
      entity_block, operator = builder.compile_operator(
        slice_,
        coordinates=coordinates,
        node_dense=node_dense,
        space=spaces_by_id[slice_.field.id],
        snapshot=snapshot,
        index_dtype=index_dtype,
        geometry_relative_tolerance=selected_policy.geometry_relative_tolerance,
      )
    entity_blocks.append(entity_block)
    operators.append(operator)
    operator_sources.append((operator.header.block_id, slice_.region.source))
  attributions = _attribution(
    normalized,
    operator_sources=tuple(operator_sources),
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
        for entity_block in entity_blocks
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
      "operators": [operator.content_manifest for operator in operators],
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
  system = _new(
    CompiledSystem,
    instance_id=InstanceId(),
    content_fingerprint=ContentFingerprint.from_manifest(manifest),
    provenance=provenance,
    registry_snapshot=snapshot,
    point_blocks=(point_block,),
    entity_blocks=tuple(entity_blocks),
    spaces=spaces,
    operators=tuple(operators),
    source_attribution=attributions,
  )
  for declaration in declarations:
    spring_block, spring_operator = spring_family.compile_group(
      system,
      declaration,
    )
    system = spring_family.compose(system, spring_block, spring_operator)
  return system
