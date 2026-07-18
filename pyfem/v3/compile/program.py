"""Compile and evaluate the canonical fixed affine finite-element program."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NoReturn
from uuid import UUID

import numpy as np

from pyfem.v3.compile.model import (
  COMPILED_MODEL_MANIFEST_SCHEMA,
  ModelCompilationPolicy,
  _model_manifest,
)
from pyfem.v3.compile.program_diagnostics import (
  ProgramCompilationDiagnostic,
  ProgramCompilationError,
  ProgramEvaluationDiagnostic,
  ProgramEvaluationError,
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
from pyfem.v3.model.program import (
  AffineConstraintPlan,
  CompiledProgram,
  NodalLoadPlan,
  ProgramAffineCoefficientWitness,
  ProgramAffineValueWitness,
  ProgramCapabilities,
  ProgramCoordinateWitness,
  ProgramDofWitness,
  ProgramEvaluation,
  ProgramLoadWitness,
  ProgramMeaningWitness,
  ProgramPrescribedWitness,
  ProgramProvenance,
  ProgramTieWitness,
)
from pyfem.v3.model.provenance import CanonicalManifest, ContentFingerprint
from pyfem.v3.model.registry import RegistrySnapshot, _validated_snapshot
from pyfem.v3.spec.diagnostics import SourceContext
from pyfem.v3.spec.normalize_program import normalize_program_spec
from pyfem.v3.spec.program import (
  AffineCoefficientSpec,
  AffineTieSpec,
  AffineValueSpec,
  DofRef,
  NodalLoadSpec,
  PrescribedDofSpec,
  ProgramCoordinateSpec,
  ProgramCoordinateValue,
  ProgramPoint,
  ProgramSpec,
)
from pyfem.v3.spec.program_diagnostics import (
  ProgramSpecValidationError,
  render_program_value,
)

COMPILED_PROGRAM_MANIFEST_SCHEMA = "pyfem-v3-compiled-program-affine-v1"
PROGRAM_REDUCTION_POLICY = (
  "canonical target-and-semantic-id order; math.fsum for declaration constants "
  "and each coordinate coefficient; binary64 coordinate products followed by "
  "math.fsum in canonical coordinate order"
)
_COORDINATE_KIND_ORDER = {"time": 0, "load": 1, "continuation": 2}
_COORDINATE_ORDER = ("time", "load", "continuation")
_INDEX_DTYPE = np.dtype(np.int64)
_FLOATING_DTYPE = np.dtype(np.float64)
_INT64_MAX = int(np.iinfo(np.int64).max)
_MISSING = object()
_PROGRAM_CAPABILITY_SLOTS = (
  "fixed_constraint_topology",
  "fixed_load_topology",
  "has_constraints",
  "has_nodal_loads",
  "has_coordinate_affine_prescribed",
  "has_coordinate_affine_nodal_loads",
  "state_dependent",
  "has_follower_loads",
  "has_interaction_tangent",
  "has_program_state",
  "contribution_channels",
)


class _BoundaryError(Exception):
  def __init__(self, code: str, message: str) -> None:
    self.code = code
    self.message = message
    super().__init__(message)


@dataclass(frozen=True, slots=True)
class _ResolvedConstraint:
  root_dof: int | None
  root_factor: float
  offset_constant: float
  offset_coefficients: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class _Tie:
  identifier: str | int
  slave: int
  master: int
  factor: float
  offset_constant: float
  offset_coefficients: tuple[float, ...]
  source: SourceContext


@dataclass(frozen=True, slots=True)
class _LoadRow:
  identifier: str | int
  dof_index: int
  constant: float
  coefficients: tuple[float, ...]
  source: SourceContext


@dataclass(frozen=True, slots=True)
class _DofLookup:
  field_id: str | int
  component_indices: dict[str, int]
  node_indices: dict[tuple[str, str | int], int]
  node_component_dofs: np.ndarray
  global_size: int


def _read_slot(value: object, name: str) -> object:
  try:
    return object.__getattribute__(value, name)
  except AttributeError:
    return _MISSING


def _required_slots(
  value: object,
  names: tuple[str, ...],
  *,
  label: str,
) -> tuple[object, ...]:
  values = tuple(_read_slot(value, name) for name in names)
  if any(item is _MISSING for item in values):
    raise _BoundaryError(
      "malformed-exact-carrier",
      f"{label} must initialize every canonical slot",
    )
  return values


def _compilation_fail(code: str, message: str, source: SourceContext) -> NoReturn:
  raise ProgramCompilationError(
    (ProgramCompilationDiagnostic(code=code, message=message, source=source),)
  )


def _evaluation_fail(code: str, message: str) -> NoReturn:
  raise ProgramEvaluationError(
    (
      ProgramEvaluationDiagnostic(
        code=code,
        message=message,
        source=SourceContext(source="<program-point>"),
      ),
    )
  )


def _semantic_id_key(value: str | int) -> tuple[int, object]:
  return (0, value) if type(value) is int else (1, value)


def _typed_id(value: str | int) -> tuple[str, str | int]:
  return ("int", value) if type(value) is int else ("str", value)


def _entity_key(kind: str, semantic_id: object) -> tuple[object, ...]:
  if type(semantic_id) is int:
    encoded: object = ("int", semantic_id)
  elif type(semantic_id) is str:
    encoded = ("str", semantic_id)
  elif type(semantic_id) is tuple:
    encoded_items: list[tuple[str, str | int]] = []
    for item in semantic_id:
      if type(item) is int:
        encoded_items.append(("int", item))
      elif type(item) is str:
        encoded_items.append(("str", item))
      else:
        raise _BoundaryError(
          "malformed-exact-carrier",
          "entity semantic ID tuples must be flat exact strings and integers",
        )
    encoded = ("tuple", tuple(encoded_items))
  else:
    raise _BoundaryError(
      "malformed-exact-carrier",
      "entity semantic IDs must contain exact strings, integers, or tuples",
    )
  return kind, encoded


def _validated_instance_id(value: object, *, label: str) -> InstanceId:
  if type(value) is not InstanceId:
    raise _BoundaryError(
      "malformed-exact-carrier",
      f"{label} must be an exact InstanceId",
    )
  token = _read_slot(value, "_token")
  if type(token) is not UUID:
    raise _BoundaryError(
      "malformed-exact-carrier",
      f"{label} contains a malformed opaque token",
    )
  return value


def _validated_fingerprint(
  value: object,
  *,
  label: str,
) -> ContentFingerprint:
  if type(value) is not ContentFingerprint:
    raise _BoundaryError(
      "malformed-exact-carrier",
      f"{label} must be an exact ContentFingerprint",
    )
  try:
    str(value)
  except (TypeError, ValueError) as exc:
    raise _BoundaryError(
      "malformed-exact-carrier",
      f"{label} contains a malformed digest",
    ) from exc
  return value


def _validated_manifest(value: object, *, label: str) -> CanonicalManifest:
  if type(value) is not CanonicalManifest:
    raise _BoundaryError(
      "malformed-exact-carrier",
      f"{label} must be an exact CanonicalManifest",
    )
  try:
    value.to_bytes()
  except (TypeError, ValueError) as exc:
    raise _BoundaryError(
      "malformed-exact-carrier",
      f"{label} contains malformed canonical bytes",
    ) from exc
  return value


def _validated_finalized_array(
  value: object,
  *,
  dtype: np.dtype[np.generic] | None,
  shape: tuple[int, ...] | None,
  label: str,
  finite: bool = False,
) -> np.ndarray:
  if type(value) is not FinalizedArray:
    raise _BoundaryError(
      "malformed-exact-carrier",
      f"{label} must be an exact FinalizedArray",
    )
  values = _read_slot(value, "values")
  if type(values) is not np.ndarray:
    raise _BoundaryError(
      "malformed-exact-carrier",
      f"{label} must own an exact plain ndarray",
    )
  if dtype is not None and values.dtype != dtype:
    raise _BoundaryError(
      "malformed-exact-carrier",
      f"{label} has the wrong exact dtype",
    )
  if shape is not None and values.shape != shape:
    raise _BoundaryError(
      "malformed-exact-carrier",
      f"{label} has the wrong exact shape",
    )
  if (
    not values.flags.owndata or not values.flags.c_contiguous or values.flags.writeable
  ):
    raise _BoundaryError(
      "malformed-exact-carrier",
      f"{label} must be owning, C-contiguous, and read-only",
    )
  if finite and not bool(np.isfinite(values).all()):
    raise _BoundaryError(
      "malformed-exact-carrier",
      f"{label} must contain only finite float64 values",
    )
  return values


def _validated_entity_index(value: object) -> EntityIndex:
  if type(value) is not EntityIndex:
    raise _BoundaryError(
      "malformed-exact-carrier",
      "entity index must be an exact EntityIndex",
    )
  (records,) = _required_slots(value, ("records",), label="entity index")
  if type(records) is not tuple:
    raise _BoundaryError(
      "malformed-exact-carrier",
      "entity index records must be an exact tuple",
    )
  seen: set[tuple[object, ...]] = set()
  for record in records:
    if type(record) is not EntityRecord:
      raise _BoundaryError(
        "malformed-exact-carrier",
        "entity index records must be exact EntityRecord values",
      )
    kind, semantic_id, dense_index, block_index, local_index = _required_slots(
      record,
      ("kind", "semantic_id", "dense_index", "block_index", "local_index"),
      label="entity record",
    )
    if type(kind) is not str or not kind:
      raise _BoundaryError(
        "malformed-exact-carrier",
        "entity record kind must be a non-empty exact string",
      )
    for index_value in (dense_index, block_index, local_index):
      if index_value is not None and (type(index_value) is not int or index_value < 0):
        raise _BoundaryError(
          "malformed-exact-carrier",
          "entity record indices must be non-negative exact integers or None",
        )
    key = _entity_key(kind, semantic_id)
    if key in seen:
      raise _BoundaryError(
        "malformed-exact-carrier",
        "entity index contains a duplicate exact semantic identity",
      )
    seen.add(key)
  return value


def _validated_source_map(value: object) -> SourceMap:
  if type(value) is not SourceMap:
    raise _BoundaryError(
      "malformed-exact-carrier",
      "source map must be an exact SourceMap",
    )
  (records,) = _required_slots(value, ("records",), label="source map")
  if type(records) is not tuple:
    raise _BoundaryError(
      "malformed-exact-carrier",
      "source map records must be an exact tuple",
    )
  seen: set[tuple[object, ...]] = set()
  for record in records:
    if type(record) is not SourceRecord:
      raise _BoundaryError(
        "malformed-exact-carrier",
        "source map records must be exact SourceRecord values",
      )
    kind, semantic_id, source = _required_slots(
      record,
      ("kind", "semantic_id", "source"),
      label="source record",
    )
    if type(kind) is not str or not kind or type(source) is not CompiledSource:
      raise _BoundaryError(
        "malformed-exact-carrier",
        "source records require an exact kind and CompiledSource",
      )
    source_name, line, column = _required_slots(
      source,
      ("source", "line", "column"),
      label="compiled source",
    )
    if (
      type(source_name) is not str
      or (line is not None and type(line) is not int)
      or (column is not None and type(column) is not int)
    ):
      raise _BoundaryError(
        "malformed-exact-carrier",
        "compiled source fields must be exact string/integer values",
      )
    key = _entity_key(kind, semantic_id)
    if key in seen:
      raise _BoundaryError(
        "malformed-exact-carrier",
        "source map contains a duplicate exact semantic identity",
      )
    seen.add(key)
  return value


def _validated_semantic_id(value: object, *, label: str) -> str | int:
  if type(value) is int or type(value) is str and bool(value):
    return value
  raise _BoundaryError(
    "malformed-exact-carrier",
    f"{label} must be an exact non-empty string or integer semantic ID",
  )


def _validated_semantic_ids(value: object, *, label: str) -> tuple[str | int, ...]:
  if type(value) is not tuple:
    raise _BoundaryError(
      "malformed-exact-carrier",
      f"{label} must use an exact tuple",
    )
  result = tuple(_validated_semantic_id(item, label=f"{label} item") for item in value)
  if len({_typed_id(item) for item in result}) != len(result):
    raise _BoundaryError(
      "malformed-exact-carrier",
      f"{label} must contain unique exact semantic IDs",
    )
  return result


def _validated_string_tuple(value: object, *, label: str) -> tuple[str, ...]:
  if type(value) is not tuple or any(
    type(item) is not str or not item for item in value
  ):
    raise _BoundaryError(
      "malformed-exact-carrier",
      f"{label} must contain non-empty exact strings in an exact tuple",
    )
  return value


def _validated_nonnegative_int(value: object, *, label: str) -> int:
  if type(value) is not int or value < 0:
    raise _BoundaryError(
      "malformed-exact-carrier",
      f"{label} must be a non-negative exact integer",
    )
  return value


def _validated_model_array(
  value: object,
  *,
  dtype: np.dtype[np.generic],
  label: str,
  dimensions: int | None = None,
) -> np.ndarray:
  values = _validated_finalized_array(
    value,
    dtype=dtype,
    shape=None,
    label=label,
    finite=dtype.kind == "f",
  )
  if dimensions is not None and values.ndim != dimensions:
    raise _BoundaryError(
      "malformed-exact-carrier",
      f"{label} has the wrong exact dimensionality",
    )
  return values


def _validated_registry_snapshot(value: object) -> RegistrySnapshot:
  if type(value) is not RegistrySnapshot:
    raise _BoundaryError(
      "malformed-exact-carrier",
      "compiled model registry snapshot must be an exact RegistrySnapshot",
    )
  try:
    _validated_snapshot(value)
  except (AttributeError, OverflowError, RecursionError, TypeError, ValueError) as exc:
    raise _BoundaryError(
      "malformed-exact-carrier",
      "compiled model registry snapshot is malformed",
    ) from exc
  return value


def _validated_mesh(
  value: object,
  *,
  index_dtype: np.dtype[np.generic],
) -> CompiledMesh:
  if type(value) is not CompiledMesh:
    raise _BoundaryError(
      "malformed-exact-carrier",
      "compiled model mesh must be an exact CompiledMesh",
    )
  node_ids, coordinates, cell_blocks = _required_slots(
    value,
    ("node_ids", "coordinates", "cell_blocks"),
    label="compiled mesh",
  )
  validated_node_ids = _validated_semantic_ids(node_ids, label="compiled mesh nodes")
  coordinate_values = _validated_model_array(
    coordinates,
    dtype=_FLOATING_DTYPE,
    label="compiled mesh coordinates",
    dimensions=2,
  )
  if coordinate_values.shape[0] != len(validated_node_ids):
    raise _BoundaryError(
      "malformed-exact-carrier",
      "compiled mesh coordinate count does not match its node IDs",
    )
  if type(cell_blocks) is not tuple or len(cell_blocks) != 1:
    raise _BoundaryError(
      "malformed-exact-carrier",
      "compiled Q8 mesh must contain one exact cell-block tuple",
    )
  block = cell_blocks[0]
  if type(block) is not CompiledCellBlock:
    raise _BoundaryError(
      "malformed-exact-carrier",
      "compiled mesh cell block must be an exact CompiledCellBlock",
    )
  (
    block_id,
    reference_topology,
    topological_dimension,
    embedding_dimension,
    geometry_interpolation,
    cell_ids,
    connectivity,
  ) = _required_slots(
    block,
    (
      "id",
      "reference_topology",
      "topological_dimension",
      "embedding_dimension",
      "geometry_interpolation",
      "cell_ids",
      "connectivity",
    ),
    label="compiled mesh cell block",
  )
  _validated_semantic_id(block_id, label="compiled cell-block ID")
  if any(
    type(item) is not str or not item
    for item in (reference_topology, geometry_interpolation)
  ):
    raise _BoundaryError(
      "malformed-exact-carrier",
      "compiled cell-block labels must be non-empty exact strings",
    )
  _validated_nonnegative_int(
    topological_dimension,
    label="compiled cell-block topological dimension",
  )
  _validated_nonnegative_int(
    embedding_dimension,
    label="compiled cell-block embedding dimension",
  )
  validated_cell_ids = _validated_semantic_ids(
    cell_ids,
    label="compiled cell IDs",
  )
  connectivity_values = _validated_model_array(
    connectivity,
    dtype=index_dtype,
    label="compiled mesh connectivity",
    dimensions=2,
  )
  if connectivity_values.shape[0] != len(validated_cell_ids):
    raise _BoundaryError(
      "malformed-exact-carrier",
      "compiled mesh connectivity count does not match its cell IDs",
    )
  return value


def _validated_descriptor_identity(value: object, *, label: str) -> None:
  if type(value) is not DescriptorIdentity:
    raise _BoundaryError(
      "malformed-exact-carrier",
      f"{label} must be an exact DescriptorIdentity",
    )
  slots = _required_slots(
    value,
    ("kind", "name", "version", "implementation_id"),
    label=label,
  )
  if any(type(item) is not str or not item for item in slots):
    raise _BoundaryError(
      "malformed-exact-carrier",
      f"{label} fields must be non-empty exact strings",
    )


def _validated_integration_layout(value: object) -> None:
  if type(value) is not IntegrationLayout:
    raise _BoundaryError(
      "malformed-exact-carrier",
      "compiled integration layout must be an exact IntegrationLayout",
    )
  points, slots_per_point, local_ids = _required_slots(
    value,
    ("points_per_element", "material_slots_per_point", "local_point_ids"),
    label="compiled integration layout",
  )
  _validated_nonnegative_int(points, label="integration points per element")
  _validated_nonnegative_int(slots_per_point, label="material slots per point")
  if type(local_ids) is not tuple or any(
    type(item) is not int or item < 0 for item in local_ids
  ):
    raise _BoundaryError(
      "malformed-exact-carrier",
      "compiled local integration-point IDs must be exact non-negative integers",
    )


def _validated_domain_block(
  value: object,
  *,
  index_dtype: np.dtype[np.generic],
) -> DomainBlock:
  if type(value) is not DomainBlock:
    raise _BoundaryError(
      "malformed-exact-carrier",
      "compiled domain block must be an exact DomainBlock",
    )
  names = (
    "block_id",
    "source_cell_block_id",
    "source_region_id",
    "cell_ids",
    "field_id",
    "source_material_id",
    "field_components",
    "topology",
    "quadrature",
    "formulation",
    "material",
    "connectivity",
    "dof_map",
    "quadrature_points",
    "quadrature_weights",
    "shape_values",
    "parent_gradients",
    "material_parameter_names",
    "material_parameters",
    "integration_layout",
    "kinematic_regime",
    "strain_voigt_order",
    "shear_convention",
    "measure_convention",
  )
  slots = _required_slots(value, names, label="compiled domain block")
  fields = dict(zip(names, slots, strict=True))
  block_id = fields["block_id"]
  if type(block_id) is not tuple or len(block_id) != 2:
    raise _BoundaryError(
      "malformed-exact-carrier",
      "compiled domain-block ID must be one exact semantic-ID pair",
    )
  for item in block_id:
    _validated_semantic_id(item, label="compiled domain-block ID item")
  for name in (
    "source_cell_block_id",
    "source_region_id",
    "field_id",
    "source_material_id",
  ):
    _validated_semantic_id(fields[name], label=f"compiled domain-block {name}")
  _validated_semantic_ids(fields["cell_ids"], label="compiled domain-block cells")
  _validated_string_tuple(
    fields["field_components"],
    label="compiled domain-block field components",
  )
  for name in ("topology", "quadrature", "formulation", "material"):
    _validated_descriptor_identity(
      fields[name],
      label=f"compiled domain-block {name} descriptor",
    )
  for name in ("connectivity", "dof_map"):
    _validated_model_array(
      fields[name],
      dtype=index_dtype,
      label=f"compiled domain-block {name}",
      dimensions=2,
    )
  for name in (
    "quadrature_points",
    "quadrature_weights",
    "shape_values",
    "parent_gradients",
    "material_parameters",
  ):
    _validated_model_array(
      fields[name],
      dtype=_FLOATING_DTYPE,
      label=f"compiled domain-block {name}",
    )
  _validated_string_tuple(
    fields["material_parameter_names"],
    label="compiled material parameter names",
  )
  _validated_integration_layout(fields["integration_layout"])
  _validated_string_tuple(
    fields["strain_voigt_order"],
    label="compiled strain Voigt order",
  )
  for name in ("kinematic_regime", "shear_convention", "measure_convention"):
    if type(fields[name]) is not str or not fields[name]:
      raise _BoundaryError(
        "malformed-exact-carrier",
        f"compiled domain-block {name} must be a non-empty exact string",
      )
  return value


def _validated_assembly_topology(
  value: object,
  *,
  index_dtype: np.dtype[np.generic],
) -> ModelAssemblyTopology:
  if type(value) is not ModelAssemblyTopology:
    raise _BoundaryError(
      "malformed-exact-carrier",
      "compiled assembly topology must be an exact ModelAssemblyTopology",
    )
  recipes, fixed = _required_slots(
    value,
    ("block_recipes", "fixed_model_coupling"),
    label="compiled assembly topology",
  )
  if type(recipes) is not tuple or type(fixed) is not bool:
    raise _BoundaryError(
      "malformed-exact-carrier",
      "compiled assembly topology has malformed exact fields",
    )
  for recipe in recipes:
    if type(recipe) is not ElementCouplingRecipe:
      raise _BoundaryError(
        "malformed-exact-carrier",
        "compiled coupling recipe must be an exact ElementCouplingRecipe",
      )
    block_index, local_count, coupling, dof_map = _required_slots(
      recipe,
      ("block_index", "local_dof_count", "coupling", "dof_map"),
      label="compiled coupling recipe",
    )
    _validated_nonnegative_int(block_index, label="coupling block index")
    _validated_nonnegative_int(local_count, label="coupling local DOF count")
    if type(coupling) is not str or not coupling:
      raise _BoundaryError(
        "malformed-exact-carrier",
        "compiled coupling label must be a non-empty exact string",
      )
    _validated_model_array(
      dof_map,
      dtype=index_dtype,
      label="compiled coupling DOF map",
      dimensions=2,
    )
  return value


def _validated_physical_state_layout(value: object) -> PhysicalStateLayout:
  if type(value) is not PhysicalStateLayout:
    raise _BoundaryError(
      "malformed-exact-carrier",
      "compiled state layout must be an exact PhysicalStateLayout",
    )
  primary_fields, block_states, global_size, evolving_count = _required_slots(
    value,
    (
      "primary_fields",
      "block_states",
      "global_primary_size",
      "evolving_value_count",
    ),
    label="compiled state layout",
  )
  if type(primary_fields) is not tuple or type(block_states) is not tuple:
    raise _BoundaryError(
      "malformed-exact-carrier",
      "compiled state-layout records must use exact tuples",
    )
  _validated_nonnegative_int(global_size, label="global primary size")
  _validated_nonnegative_int(evolving_count, label="evolving value count")
  for field in primary_fields:
    if type(field) is not PrimaryFieldLayout:
      raise _BoundaryError(
        "malformed-exact-carrier",
        "compiled primary field must be an exact PrimaryFieldLayout",
      )
    field_id, components, field_size = _required_slots(
      field,
      ("field_id", "components", "global_size"),
      label="compiled primary field",
    )
    _validated_semantic_id(field_id, label="compiled primary-field ID")
    _validated_string_tuple(components, label="compiled primary-field components")
    _validated_nonnegative_int(field_size, label="compiled primary-field size")
  state_names = (
    "block_index",
    "element_count",
    "integration_points_per_element",
    "material_slots_per_point",
    "material_history_width",
    "formulation_history_width",
  )
  for state in block_states:
    if type(state) is not BlockStateLayout:
      raise _BoundaryError(
        "malformed-exact-carrier",
        "compiled block state must be an exact BlockStateLayout",
      )
    state_values = _required_slots(state, state_names, label="compiled block state")
    for name, item in zip(state_names, state_values, strict=True):
      _validated_nonnegative_int(item, label=f"compiled block-state {name}")
  return value


def _validated_model_capabilities(value: object) -> ModelCapabilities:
  if type(value) is not ModelCapabilities:
    raise _BoundaryError(
      "malformed-exact-carrier",
      "compiled model capabilities must be an exact ModelCapabilities",
    )
  names = (
    "response_class",
    "fixed_model_coupling",
    "tangent_class",
    "tangent_is_symmetric",
    "tangent_is_constant",
    "conservative_internal_contribution",
    "state_dependent",
    "has_storage",
    "has_mass",
    "has_damping",
    "restart_history_required",
    "contribution_channels",
  )
  slots = _required_slots(value, names, label="compiled model capabilities")
  if (
    type(slots[0]) is not str
    or not slots[0]
    or type(slots[2]) is not str
    or not slots[2]
  ):
    raise _BoundaryError(
      "malformed-exact-carrier",
      "compiled model capability labels must be non-empty exact strings",
    )
  if any(type(item) is not bool for item in (*slots[1:2], *slots[3:-1])):
    raise _BoundaryError(
      "malformed-exact-carrier",
      "compiled model capability flags must be exact booleans",
    )
  _validated_string_tuple(
    slots[-1],
    label="compiled model contribution channels",
  )
  return value


def _validated_dof_plan(
  value: object,
  *,
  dense_index_dtype: np.dtype[np.generic] | None,
  label: str,
) -> DofPlan:
  if type(value) is not DofPlan:
    raise _BoundaryError(
      "malformed-exact-carrier",
      f"{label} must be an exact DofPlan",
    )
  field_id, components, node_ids, node_component_dofs, global_size = _required_slots(
    value,
    ("field_id", "components", "node_ids", "node_component_dofs", "global_size"),
    label=label,
  )
  _validated_semantic_id(field_id, label=f"{label} field ID")
  validated_components = _validated_string_tuple(
    components,
    label=f"{label} components",
  )
  if not validated_components or len(set(validated_components)) != len(
    validated_components
  ):
    raise _BoundaryError(
      "malformed-exact-carrier",
      f"{label} components must be non-empty and unique",
    )
  validated_nodes = _validated_semantic_ids(node_ids, label=f"{label} node IDs")
  expected_size = len(validated_nodes) * len(validated_components)
  if type(global_size) is not int or global_size != expected_size:
    raise _BoundaryError(
      "malformed-exact-carrier",
      f"{label} global DOF size does not match its semantic layout",
    )
  dof_values = _validated_finalized_array(
    node_component_dofs,
    dtype=None,
    shape=(len(validated_nodes), len(validated_components)),
    label=f"{label} node-component DOFs",
  )
  if dof_values.dtype.kind != "i":
    raise _BoundaryError(
      "malformed-exact-carrier",
      f"{label} node-component DOFs require a signed integer dtype",
    )
  if dense_index_dtype is not None and dof_values.dtype != dense_index_dtype:
    raise _BoundaryError(
      "malformed-exact-carrier",
      f"{label} indices do not match the recorded dense-index policy",
    )
  if global_size and global_size - 1 > int(np.iinfo(dof_values.dtype).max):
    raise _BoundaryError(
      "malformed-exact-carrier",
      f"{label} dense-index dtype cannot represent every global DOF index",
    )
  expected_indices = np.arange(global_size, dtype=dof_values.dtype).reshape(
    dof_values.shape
  )
  if not bool(np.array_equal(dof_values, expected_indices)):
    raise _BoundaryError(
      "malformed-exact-carrier",
      f"{label} node-component DOFs are not canonical dense indices",
    )
  return value


def _validated_model(value: object) -> CompiledModel:
  if type(value) is not CompiledModel:
    raise _BoundaryError(
      "invalid-compiled-model",
      "program compilation requires an exact CompiledModel",
    )
  (
    instance_id,
    content_fingerprint,
    provenance,
    registry_snapshot,
    mesh,
    dofs,
    domain_blocks,
    assembly_topology,
    physical_state_layout,
    capabilities,
    entity_index,
    source_map,
  ) = _required_slots(
    value,
    (
      "instance_id",
      "content_fingerprint",
      "provenance",
      "registry_snapshot",
      "mesh",
      "dofs",
      "domain_blocks",
      "assembly_topology",
      "physical_state_layout",
      "capabilities",
      "entity_index",
      "source_map",
    ),
    label="compiled model",
  )
  _validated_instance_id(instance_id, label="compiled model instance identity")
  fingerprint = _validated_fingerprint(
    content_fingerprint,
    label="compiled model content fingerprint",
  )
  if type(provenance) is not ModelProvenance:
    raise _BoundaryError(
      "malformed-exact-carrier",
      "compiled model provenance must be an exact ModelProvenance",
    )
  (
    schema,
    manifest,
    registry_fingerprint,
    floating_dtype,
    dense_index_dtype,
    geometry_tolerance,
  ) = _required_slots(
    provenance,
    (
      "schema",
      "manifest",
      "registry_fingerprint",
      "floating_dtype",
      "dense_index_dtype",
      "geometry_relative_tolerance",
    ),
    label="compiled model provenance",
  )
  if (
    type(schema) is not str
    or schema != COMPILED_MODEL_MANIFEST_SCHEMA
    or type(floating_dtype) is not str
    or floating_dtype != _FLOATING_DTYPE.str
    or type(dense_index_dtype) is not str
    or type(geometry_tolerance) is not float
    or not math.isfinite(geometry_tolerance)
    or geometry_tolerance <= 0.0
    or geometry_tolerance >= 1.0
  ):
    raise _BoundaryError(
      "malformed-exact-carrier",
      "compiled model provenance contains malformed numeric policy fields",
    )
  try:
    index_dtype = np.dtype(dense_index_dtype)
  except (TypeError, ValueError) as exc:
    raise _BoundaryError(
      "malformed-exact-carrier",
      "compiled model provenance contains an invalid dense-index dtype",
    ) from exc
  if index_dtype.str != dense_index_dtype or index_dtype.name not in (
    "int8",
    "int16",
    "int32",
    "int64",
  ):
    raise _BoundaryError(
      "malformed-exact-carrier",
      "compiled model provenance contains an unsupported dense-index dtype",
    )
  model_manifest = _validated_manifest(manifest, label="compiled model manifest")
  validated_registry_fingerprint = _validated_fingerprint(
    registry_fingerprint,
    label="compiled model registry fingerprint",
  )
  validated_registry = _validated_registry_snapshot(registry_snapshot)
  if validated_registry_fingerprint != validated_registry.fingerprint:
    raise _BoundaryError(
      "malformed-exact-carrier",
      "compiled model registry fingerprint does not match its snapshot",
    )
  try:
    expected_fingerprint = ContentFingerprint.from_manifest(model_manifest)
    fingerprint_matches = fingerprint == expected_fingerprint
  except (TypeError, ValueError) as exc:
    raise _BoundaryError(
      "malformed-exact-carrier",
      "compiled model content identity is malformed",
    ) from exc
  if not fingerprint_matches:
    raise _BoundaryError(
      "malformed-exact-carrier",
      "compiled model fingerprint does not match its provenance manifest",
    )

  validated_dofs = _validated_dof_plan(
    dofs,
    dense_index_dtype=index_dtype,
    label="compiled model DOF plan",
  )
  validated_mesh = _validated_mesh(mesh, index_dtype=index_dtype)
  if type(domain_blocks) is not tuple or len(domain_blocks) != 1:
    raise _BoundaryError(
      "malformed-exact-carrier",
      "compiled model domain blocks must be one exact tuple entry",
    )
  validated_domain_block = _validated_domain_block(
    domain_blocks[0],
    index_dtype=index_dtype,
  )
  validated_assembly = _validated_assembly_topology(
    assembly_topology,
    index_dtype=index_dtype,
  )
  validated_state_layout = _validated_physical_state_layout(physical_state_layout)
  validated_model_capabilities = _validated_model_capabilities(capabilities)

  validated_entities = _validated_entity_index(entity_index)
  validated_sources = _validated_source_map(source_map)
  entity_records = object.__getattribute__(validated_entities, "records")
  source_records = object.__getattribute__(validated_sources, "records")
  entities_by_key = {
    _entity_key(
      object.__getattribute__(record, "kind"),
      object.__getattribute__(record, "semantic_id"),
    ): record
    for record in entity_records
  }
  sources_by_key = {
    _entity_key(
      object.__getattribute__(record, "kind"),
      object.__getattribute__(record, "semantic_id"),
    ): record
    for record in source_records
  }
  for node_index, node_id in enumerate(validated_dofs.node_ids):
    for component_index, component in enumerate(validated_dofs.components):
      semantic_id = (node_id, validated_dofs.field_id, component)
      expected_index = node_index * len(validated_dofs.components) + component_index
      key = _entity_key("dof", semantic_id)
      matching_entity = entities_by_key.get(key)
      matching_source = sources_by_key.get(key)
      if (
        matching_entity is None
        or object.__getattribute__(matching_entity, "dense_index") != expected_index
        or matching_source is None
      ):
        raise _BoundaryError(
          "malformed-exact-carrier",
          "compiled model semantic DOF entity/source maps are inconsistent",
        )
  policy = ModelCompilationPolicy(
    dense_index_dtype=index_dtype.name,
    geometry_relative_tolerance=geometry_tolerance,
  )
  try:
    expected_model_manifest = _model_manifest(
      policy=policy,
      registry_snapshot=validated_registry,
      mesh=validated_mesh,
      dofs=validated_dofs,
      domain_block=validated_domain_block,
      assembly_topology=validated_assembly,
      physical_state_layout=validated_state_layout,
      capabilities=validated_model_capabilities,
      entity_index=validated_entities,
      source_map=validated_sources,
    )
    visible_model_matches = (
      expected_model_manifest.to_bytes() == model_manifest.to_bytes()
    )
  except (AttributeError, OverflowError, RecursionError, TypeError, ValueError) as exc:
    raise _BoundaryError(
      "malformed-exact-carrier",
      "compiled model visible meaning cannot form its canonical manifest",
    ) from exc
  if not visible_model_matches:
    raise _BoundaryError(
      "malformed-exact-carrier",
      "compiled model visible meaning does not match its provenance manifest",
    )
  return value


def _dof_lookup_from_plan(dofs: DofPlan) -> _DofLookup:
  return _DofLookup(
    field_id=dofs.field_id,
    component_indices={
      component: index for index, component in enumerate(dofs.components)
    },
    node_indices={
      _typed_id(node_id): index for index, node_id in enumerate(dofs.node_ids)
    },
    node_component_dofs=dofs.node_component_dofs.values,
    global_size=dofs.global_size,
  )


def _dof_lookup(model: CompiledModel) -> _DofLookup:
  return _dof_lookup_from_plan(model.dofs)


def _source_manifest(source: SourceContext | CompiledSource) -> dict[str, object]:
  return {
    "source": object.__getattribute__(source, "source"),
    "line": object.__getattribute__(source, "line"),
    "column": object.__getattribute__(source, "column"),
  }


def _dof_manifest(value: DofRef) -> dict[str, object]:
  return {
    "node_id": value.node_id,
    "field_id": value.field_id,
    "component": value.component,
  }


def _affine_manifest(value: AffineValueSpec) -> dict[str, object]:
  return {
    "constant": value.constant,
    "coefficients": [
      {
        "coordinate": item.coordinate,
        "coefficient": item.coefficient,
        "source": _source_manifest(item.source),
      }
      for item in value.coefficients
    ],
    "source": _source_manifest(value.source),
  }


def _normalized_program_manifest(spec: ProgramSpec) -> CanonicalManifest:
  constraints: list[dict[str, object]] = []
  for constraint in spec.constraints:
    if type(constraint) is PrescribedDofSpec:
      constraints.append(
        {
          "kind": "prescribed-dof",
          "id": constraint.id,
          "target": _dof_manifest(constraint.target),
          "value": _affine_manifest(constraint.value),
          "source": _source_manifest(constraint.source),
        }
      )
    else:
      constraints.append(
        {
          "kind": "one-master-affine-tie",
          "id": constraint.id,
          "slave": _dof_manifest(constraint.slave),
          "master": _dof_manifest(constraint.master),
          "factor": constraint.factor,
          "offset": _affine_manifest(constraint.offset),
          "source": _source_manifest(constraint.source),
        }
      )
  return CanonicalManifest(
    {
      "coordinates": [
        {
          "name": item.name,
          "kind": item.kind,
          "source": _source_manifest(item.source),
        }
        for item in spec.coordinates
      ],
      "constraints": constraints,
      "loads": [
        {
          "id": item.id,
          "target": _dof_manifest(item.target),
          "value": _affine_manifest(item.value),
          "source": _source_manifest(item.source),
        }
        for item in spec.loads
      ],
      "source": _source_manifest(spec.source),
    }
  )


def _entity_index_manifest(entity_index: EntityIndex) -> list[dict[str, object]]:
  return [
    {
      "kind": record.kind,
      "semantic_id": record.semantic_id,
      "dense_index": record.dense_index,
      "block_index": record.block_index,
      "local_index": record.local_index,
    }
    for record in entity_index.records
  ]


def _source_map_manifest(source_map: SourceMap) -> list[dict[str, object]]:
  return [
    {
      "kind": record.kind,
      "semantic_id": record.semantic_id,
      "source": _source_manifest(record.source),
    }
    for record in source_map.records
  ]


def _capabilities_manifest(
  capabilities: ProgramCapabilities,
) -> dict[str, object]:
  return {
    "fixed_constraint_topology": capabilities.fixed_constraint_topology,
    "fixed_load_topology": capabilities.fixed_load_topology,
    "has_constraints": capabilities.has_constraints,
    "has_nodal_loads": capabilities.has_nodal_loads,
    "has_coordinate_affine_prescribed": (capabilities.has_coordinate_affine_prescribed),
    "has_coordinate_affine_nodal_loads": (
      capabilities.has_coordinate_affine_nodal_loads
    ),
    "state_dependent": capabilities.state_dependent,
    "has_follower_loads": capabilities.has_follower_loads,
    "has_interaction_tangent": capabilities.has_interaction_tangent,
    "has_program_state": capabilities.has_program_state,
    "contribution_channels": capabilities.contribution_channels,
  }


def _program_manifest(
  *,
  normalized_manifest: CanonicalManifest,
  model_fingerprint: ContentFingerprint,
  coordinate_names: tuple[str, ...],
  coordinate_kinds: tuple[str, ...],
  constraint_plan: AffineConstraintPlan,
  nodal_load_plan: NodalLoadPlan,
  capabilities: ProgramCapabilities,
  entity_index: EntityIndex,
  source_map: SourceMap,
) -> CanonicalManifest:
  return CanonicalManifest(
    {
      "schema": COMPILED_PROGRAM_MANIFEST_SCHEMA,
      "normalized_program": normalized_manifest,
      "compatible_model_content": {
        "algorithm": model_fingerprint.algorithm,
        "digest": model_fingerprint.digest,
      },
      "numeric_policy": {
        "index_dtype": _INDEX_DTYPE.str,
        "floating_dtype": _FLOATING_DTYPE.str,
        "coordinate_order": _COORDINATE_ORDER,
        "reduction_policy": PROGRAM_REDUCTION_POLICY,
      },
      "coordinate_schema": {
        "names": coordinate_names,
        "kinds": coordinate_kinds,
      },
      "constraint_plan": {
        "full_dof_count": constraint_plan.full_dof_count,
        "reduced_dof_count": constraint_plan.reduced_dof_count,
        "free_dofs": constraint_plan.free_dofs.values,
        "row_offsets": constraint_plan.row_offsets.values,
        "column_indices": constraint_plan.column_indices.values,
        "coefficients": constraint_plan.coefficients.values,
        "offset_constant": constraint_plan.offset_constant.values,
        "offset_coordinate_coefficients": (
          constraint_plan.offset_coordinate_coefficients.values
        ),
        "shapes": {
          "free_dofs": constraint_plan.free_dofs.values.shape,
          "row_offsets": constraint_plan.row_offsets.values.shape,
          "column_indices": constraint_plan.column_indices.values.shape,
          "coefficients": constraint_plan.coefficients.values.shape,
          "offset_constant": constraint_plan.offset_constant.values.shape,
          "offset_coordinate_coefficients": (
            constraint_plan.offset_coordinate_coefficients.values.shape
          ),
        },
      },
      "nodal_load_plan": {
        "load_ids": nodal_load_plan.load_ids,
        "dof_indices": nodal_load_plan.dof_indices.values,
        "constant_values": nodal_load_plan.constant_values.values,
        "coordinate_coefficients": nodal_load_plan.coordinate_coefficients.values,
        "shapes": {
          "dof_indices": nodal_load_plan.dof_indices.values.shape,
          "constant_values": nodal_load_plan.constant_values.values.shape,
          "coordinate_coefficients": (
            nodal_load_plan.coordinate_coefficients.values.shape
          ),
        },
      },
      "capabilities": _capabilities_manifest(capabilities),
      "entity_index": _entity_index_manifest(entity_index),
      "source_map": _source_map_manifest(source_map),
    }
  )


def _affine_parts(
  value: AffineValueSpec,
  coordinate_indices: dict[str, int],
) -> tuple[float, tuple[float, ...]]:
  coefficients = [0.0] * len(coordinate_indices)
  for item in value.coefficients:
    coefficients[coordinate_indices[item.coordinate]] = item.coefficient
  return value.constant, tuple(coefficients)


def _resolve_dof(
  reference: DofRef,
  lookup: _DofLookup,
  *,
  source: SourceContext,
) -> int:
  node_key = _typed_id(reference.node_id)
  node_index = lookup.node_indices.get(node_key)
  if node_index is None:
    _compilation_fail(
      "unknown-program-node",
      "program DOF reference names an unknown exact model node ID",
      source,
    )
  if type(reference.field_id) is not type(lookup.field_id) or (
    reference.field_id != lookup.field_id
  ):
    _compilation_fail(
      "unknown-program-field",
      "program DOF reference names an unknown exact model field ID",
      source,
    )
  component_index = lookup.component_indices.get(reference.component)
  if component_index is None:
    _compilation_fail(
      "unknown-program-component",
      "program DOF reference names an unknown exact model field component",
      source,
    )
  return int(lookup.node_component_dofs[node_index, component_index])


def _finite_product(
  left: float,
  right: float,
  *,
  source: SourceContext,
  label: str,
) -> float:
  result = left * right
  if not math.isfinite(result):
    _compilation_fail(
      "nonfinite-affine-composition",
      f"{label} produced a nonfinite binary64 intermediate",
      source,
    )
  return result


def _finite_two_term_sum(
  first: float,
  second: float,
  *,
  source: SourceContext,
  label: str,
) -> float:
  try:
    result = math.fsum((first, second))
  except OverflowError:
    _compilation_fail(
      "nonfinite-affine-composition",
      f"{label} overflowed deterministic binary64 addition",
      source,
    )
  if not math.isfinite(result):
    _compilation_fail(
      "nonfinite-affine-composition",
      f"{label} produced a nonfinite deterministic sum",
      source,
    )
  return result


def _check_int64_count(value: int, *, label: str, source: SourceContext) -> None:
  if type(value) is not int or value < 0 or value > _INT64_MAX:
    _compilation_fail(
      "program-index-overflow",
      f"{label} cannot be represented by the frozen int64 program policy",
      source,
    )


def _compile_constraints(
  spec: ProgramSpec,
  lookup: _DofLookup,
  coordinate_indices: dict[str, int],
) -> AffineConstraintPlan:
  full_count = lookup.global_size
  _check_int64_count(full_count, label="full DOF count", source=spec.source)
  _check_int64_count(full_count + 1, label="CSR row-offset count", source=spec.source)

  prescribed: dict[int, _ResolvedConstraint] = {}
  ties: dict[int, _Tie] = {}
  for constraint in spec.constraints:
    if type(constraint) is PrescribedDofSpec:
      dof = _resolve_dof(constraint.target, lookup, source=constraint.source)
      if dof in prescribed:
        _compilation_fail(
          "duplicate-prescribed-dof",
          "more than one constraint prescribes the same semantic DOF",
          constraint.source,
        )
      if dof in ties:
        _compilation_fail(
          "prescribed-slave-conflict",
          "one semantic DOF cannot be both prescribed and an affine slave",
          constraint.source,
        )
      constant, coefficients = _affine_parts(
        constraint.value,
        coordinate_indices,
      )
      prescribed[dof] = _ResolvedConstraint(
        root_dof=None,
        root_factor=0.0,
        offset_constant=constant,
        offset_coefficients=coefficients,
      )
      continue

    slave = _resolve_dof(constraint.slave, lookup, source=constraint.source)
    master = _resolve_dof(constraint.master, lookup, source=constraint.source)
    if slave == master:
      _compilation_fail(
        "self-affine-tie",
        "an affine tie slave and master must be different semantic DOFs",
        constraint.source,
      )
    if slave in ties:
      _compilation_fail(
        "duplicate-affine-slave",
        "more than one affine tie owns the same dependent DOF",
        constraint.source,
      )
    if slave in prescribed:
      _compilation_fail(
        "prescribed-slave-conflict",
        "one semantic DOF cannot be both prescribed and an affine slave",
        constraint.source,
      )
    constant, coefficients = _affine_parts(constraint.offset, coordinate_indices)
    ties[slave] = _Tie(
      identifier=constraint.id,
      slave=slave,
      master=master,
      factor=constraint.factor,
      offset_constant=constant,
      offset_coefficients=coefficients,
      source=constraint.source,
    )

  resolved: dict[int, _ResolvedConstraint] = dict(prescribed)
  visit_state: dict[int, int] = {}
  zeros = tuple(0.0 for _ in coordinate_indices)
  for start in sorted(ties):
    if visit_state.get(start) == 2:
      continue
    stack: list[tuple[int, bool]] = [(start, False)]
    while stack:
      dof, exiting = stack.pop()
      if exiting:
        tie = ties[dof]
        master = tie.master
        if master in resolved:
          master_value = resolved[master]
        elif master in ties:
          master_value = resolved[master]
        else:
          master_value = _ResolvedConstraint(
            root_dof=master,
            root_factor=1.0,
            offset_constant=0.0,
            offset_coefficients=zeros,
          )
        root_factor = 0.0
        if master_value.root_dof is not None:
          root_factor = _finite_product(
            tie.factor,
            master_value.root_factor,
            source=tie.source,
            label="affine tie root factor",
          )
        scaled_constant = _finite_product(
          tie.factor,
          master_value.offset_constant,
          source=tie.source,
          label="affine tie offset constant",
        )
        offset_constant = _finite_two_term_sum(
          scaled_constant,
          tie.offset_constant,
          source=tie.source,
          label="affine tie offset constant",
        )
        offset_coefficients: list[float] = []
        for master_coefficient, own_coefficient in zip(
          master_value.offset_coefficients,
          tie.offset_coefficients,
          strict=True,
        ):
          scaled = _finite_product(
            tie.factor,
            master_coefficient,
            source=tie.source,
            label="affine tie coordinate coefficient",
          )
          offset_coefficients.append(
            _finite_two_term_sum(
              scaled,
              own_coefficient,
              source=tie.source,
              label="affine tie coordinate coefficient",
            )
          )
        resolved[dof] = _ResolvedConstraint(
          root_dof=master_value.root_dof,
          root_factor=root_factor,
          offset_constant=offset_constant,
          offset_coefficients=tuple(offset_coefficients),
        )
        visit_state[dof] = 2
        continue

      state = visit_state.get(dof, 0)
      if state == 2:
        continue
      if state == 1:
        _compilation_fail(
          "cyclic-affine-constraints",
          "affine tie declarations contain a dependency cycle",
          ties[dof].source,
        )
      visit_state[dof] = 1
      stack.append((dof, True))
      master = ties[dof].master
      if master in ties:
        master_state = visit_state.get(master, 0)
        if master_state == 1:
          _compilation_fail(
            "cyclic-affine-constraints",
            "affine tie declarations contain a dependency cycle",
            ties[dof].source,
          )
        if master_state != 2:
          stack.append((master, False))

  free_dofs = tuple(
    dof for dof in range(full_count) if dof not in prescribed and dof not in ties
  )
  reduced_count = len(free_dofs)
  _check_int64_count(reduced_count, label="reduced DOF count", source=spec.source)
  free_columns = {dof: column for column, dof in enumerate(free_dofs)}
  row_offsets = [0]
  column_indices: list[int] = []
  coefficients: list[float] = []
  offset_constant: list[float] = []
  offset_coordinate_coefficients: list[tuple[float, ...]] = []
  for dof in range(full_count):
    if dof in free_columns:
      column_indices.append(free_columns[dof])
      coefficients.append(1.0)
      offset_constant.append(0.0)
      offset_coordinate_coefficients.append(zeros)
    else:
      value = resolved[dof]
      if value.root_dof is not None:
        column_indices.append(free_columns[value.root_dof])
        coefficients.append(value.root_factor)
      offset_constant.append(value.offset_constant)
      offset_coordinate_coefficients.append(value.offset_coefficients)
    row_offsets.append(len(column_indices))

  nonzero_count = len(column_indices)
  _check_int64_count(nonzero_count, label="CSR nonzero count", source=spec.source)
  return AffineConstraintPlan(
    full_dof_count=full_count,
    reduced_dof_count=reduced_count,
    free_dofs=FinalizedArray(free_dofs, dtype=np.int64),
    row_offsets=FinalizedArray(row_offsets, dtype=np.int64),
    column_indices=FinalizedArray(column_indices, dtype=np.int64),
    coefficients=FinalizedArray(coefficients, dtype=np.float64),
    offset_constant=FinalizedArray(offset_constant, dtype=np.float64),
    offset_coordinate_coefficients=FinalizedArray(
      np.asarray(offset_coordinate_coefficients, dtype=np.float64).reshape(
        full_count,
        len(coordinate_indices),
      ),
      dtype=np.float64,
    ),
  )


def _compile_loads(
  spec: ProgramSpec,
  lookup: _DofLookup,
  coordinate_indices: dict[str, int],
) -> NodalLoadPlan:
  rows: list[_LoadRow] = []
  for load in spec.loads:
    dof_index = _resolve_dof(load.target, lookup, source=load.source)
    constant, coefficients = _affine_parts(load.value, coordinate_indices)
    rows.append(
      _LoadRow(
        identifier=load.id,
        dof_index=dof_index,
        constant=constant,
        coefficients=coefficients,
        source=load.source,
      )
    )
  rows.sort(key=lambda item: (item.dof_index, _semantic_id_key(item.identifier)))
  _check_int64_count(len(rows), label="nodal load count", source=spec.source)
  coefficient_values = np.asarray(
    [row.coefficients for row in rows],
    dtype=np.float64,
  ).reshape(len(rows), len(coordinate_indices))
  return NodalLoadPlan(
    load_ids=tuple(row.identifier for row in rows),
    dof_indices=FinalizedArray(
      [row.dof_index for row in rows],
      dtype=np.int64,
    ),
    constant_values=FinalizedArray(
      [row.constant for row in rows],
      dtype=np.float64,
    ),
    coordinate_coefficients=FinalizedArray(
      coefficient_values,
      dtype=np.float64,
    ),
  )


def _compiled_source(source: SourceContext) -> CompiledSource:
  return CompiledSource(
    source=source.source,
    line=source.line,
    column=source.column,
  )


def _witness_dof(value: DofRef) -> ProgramDofWitness:
  return ProgramDofWitness(
    node_id=value.node_id,
    field_id=value.field_id,
    component=value.component,
  )


def _witness_affine(value: AffineValueSpec) -> ProgramAffineValueWitness:
  return ProgramAffineValueWitness(
    constant=value.constant,
    coefficients=tuple(
      ProgramAffineCoefficientWitness(
        coordinate=item.coordinate,
        coefficient=item.coefficient,
        source=_compiled_source(item.source),
      )
      for item in value.coefficients
    ),
    source=_compiled_source(value.source),
  )


def _build_program_witness(
  spec: ProgramSpec,
  dofs: DofPlan,
) -> ProgramMeaningWitness:
  constraints: list[ProgramPrescribedWitness | ProgramTieWitness] = []
  for constraint in spec.constraints:
    if type(constraint) is PrescribedDofSpec:
      constraints.append(
        ProgramPrescribedWitness(
          id=constraint.id,
          target=_witness_dof(constraint.target),
          value=_witness_affine(constraint.value),
          source=_compiled_source(constraint.source),
        )
      )
    else:
      constraints.append(
        ProgramTieWitness(
          id=constraint.id,
          slave=_witness_dof(constraint.slave),
          master=_witness_dof(constraint.master),
          factor=constraint.factor,
          offset=_witness_affine(constraint.offset),
          source=_compiled_source(constraint.source),
        )
      )
  return ProgramMeaningWitness(
    coordinates=tuple(
      ProgramCoordinateWitness(
        name=item.name,
        kind=item.kind,
        source=_compiled_source(item.source),
      )
      for item in spec.coordinates
    ),
    constraints=tuple(constraints),
    loads=tuple(
      ProgramLoadWitness(
        id=item.id,
        target=_witness_dof(item.target),
        value=_witness_affine(item.value),
        source=_compiled_source(item.source),
      )
      for item in spec.loads
    ),
    source=_compiled_source(spec.source),
    model_dofs=DofPlan(
      field_id=dofs.field_id,
      components=tuple(dofs.components),
      node_ids=tuple(dofs.node_ids),
      node_component_dofs=FinalizedArray(
        dofs.node_component_dofs.values,
        dtype=dofs.node_component_dofs.values.dtype,
      ),
      global_size=dofs.global_size,
    ),
  )


def _build_program_maps(
  spec: ProgramSpec,
  nodal_load_plan: NodalLoadPlan,
) -> tuple[EntityIndex, SourceMap]:
  entities: list[EntityRecord] = [
    EntityRecord(
      kind="program",
      semantic_id="program",
      dense_index=0,
      block_index=None,
      local_index=None,
    )
  ]
  sources: list[SourceRecord] = [
    SourceRecord(
      kind="program",
      semantic_id="program",
      source=_compiled_source(spec.source),
    )
  ]
  for index, coordinate in enumerate(spec.coordinates):
    entities.append(
      EntityRecord(
        kind="program_coordinate",
        semantic_id=coordinate.name,
        dense_index=index,
        block_index=None,
        local_index=index,
      )
    )
    sources.append(
      SourceRecord(
        kind="program_coordinate",
        semantic_id=coordinate.name,
        source=_compiled_source(coordinate.source),
      )
    )
  for index, constraint in enumerate(spec.constraints):
    entities.append(
      EntityRecord(
        kind="program_constraint",
        semantic_id=constraint.id,
        dense_index=index,
        block_index=None,
        local_index=index,
      )
    )
    sources.append(
      SourceRecord(
        kind="program_constraint",
        semantic_id=constraint.id,
        source=_compiled_source(constraint.source),
      )
    )
  loads_by_typed_id = {_typed_id(load.id): load for load in spec.loads}
  for index, load_id in enumerate(nodal_load_plan.load_ids):
    load = loads_by_typed_id[_typed_id(load_id)]
    entities.append(
      EntityRecord(
        kind="program_load",
        semantic_id=load_id,
        dense_index=index,
        block_index=None,
        local_index=index,
      )
    )
    sources.append(
      SourceRecord(
        kind="program_load",
        semantic_id=load_id,
        source=_compiled_source(load.source),
      )
    )
  return EntityIndex(tuple(entities)), SourceMap(tuple(sources))


def _derived_capabilities(
  constraint_plan: AffineConstraintPlan,
  nodal_load_plan: NodalLoadPlan,
) -> ProgramCapabilities:
  has_constraints = constraint_plan.reduced_dof_count != constraint_plan.full_dof_count
  has_nodal_loads = bool(nodal_load_plan.load_ids)
  prescribed_affine = bool(
    np.any(constraint_plan.offset_coordinate_coefficients.values != 0.0)
  )
  load_affine = bool(np.any(nodal_load_plan.coordinate_coefficients.values != 0.0))
  channels: list[str] = []
  if has_constraints:
    channels.append("prescribed-offset")
  if has_nodal_loads:
    channels.append("external-nodal-force")
  return ProgramCapabilities(
    fixed_constraint_topology=True,
    fixed_load_topology=True,
    has_constraints=has_constraints,
    has_nodal_loads=has_nodal_loads,
    has_coordinate_affine_prescribed=prescribed_affine,
    has_coordinate_affine_nodal_loads=load_affine,
    state_dependent=False,
    has_follower_loads=False,
    has_interaction_tangent=False,
    has_program_state=False,
    contribution_channels=tuple(channels),
  )


def compile_program(model: CompiledModel, spec: ProgramSpec) -> CompiledProgram:
  """Compile a normalized affine program against one exact compiled model."""
  normalized = normalize_program_spec(spec)
  try:
    validated_model = _validated_model(model)
  except _BoundaryError as error:
    _compilation_fail(
      error.code,
      error.message,
      normalized.source,
    )

  coordinate_names = tuple(item.name for item in normalized.coordinates)
  coordinate_kinds = tuple(item.kind for item in normalized.coordinates)
  coordinate_indices = {name: index for index, name in enumerate(coordinate_names)}
  dof_lookup = _dof_lookup(validated_model)
  constraint_plan = _compile_constraints(
    normalized,
    dof_lookup,
    coordinate_indices,
  )
  nodal_load_plan = _compile_loads(
    normalized,
    dof_lookup,
    coordinate_indices,
  )
  meaning_witness = _build_program_witness(normalized, validated_model.dofs)
  capabilities = _derived_capabilities(constraint_plan, nodal_load_plan)
  entity_index, source_map = _build_program_maps(normalized, nodal_load_plan)
  normalized_manifest = _normalized_program_manifest(normalized)
  manifest = _program_manifest(
    normalized_manifest=normalized_manifest,
    model_fingerprint=validated_model.content_fingerprint,
    coordinate_names=coordinate_names,
    coordinate_kinds=coordinate_kinds,
    constraint_plan=constraint_plan,
    nodal_load_plan=nodal_load_plan,
    capabilities=capabilities,
    entity_index=entity_index,
    source_map=source_map,
  )
  provenance = ProgramProvenance(
    schema=COMPILED_PROGRAM_MANIFEST_SCHEMA,
    manifest=manifest,
    normalized_program_manifest=normalized_manifest,
    floating_dtype=_FLOATING_DTYPE.str,
    index_dtype=_INDEX_DTYPE.str,
    coordinate_order=_COORDINATE_ORDER,
    reduction_policy=PROGRAM_REDUCTION_POLICY,
  )
  return CompiledProgram(
    instance_id=InstanceId(),
    content_fingerprint=ContentFingerprint.from_manifest(manifest),
    provenance=provenance,
    compatible_model_instance_id=validated_model.instance_id,
    compatible_model_content_fingerprint=validated_model.content_fingerprint,
    coordinate_names=coordinate_names,
    coordinate_kinds=coordinate_kinds,
    meaning_witness=meaning_witness,
    constraint_plan=constraint_plan,
    nodal_load_plan=nodal_load_plan,
    capabilities=capabilities,
    entity_index=entity_index,
    source_map=source_map,
  )


def _validated_capabilities(value: object) -> ProgramCapabilities:
  if type(value) is not ProgramCapabilities:
    raise _BoundaryError(
      "malformed-compiled-program",
      "program capabilities must be an exact ProgramCapabilities",
    )
  slots = _required_slots(
    value,
    _PROGRAM_CAPABILITY_SLOTS,
    label="program capabilities",
  )
  if any(type(item) is not bool for item in slots[:-1]):
    raise _BoundaryError(
      "malformed-compiled-program",
      "program capability flags must be exact booleans",
    )
  channels = slots[-1]
  if (
    type(channels) is not tuple
    or any(type(item) is not str or not item for item in channels)
    or len(set(channels)) != len(channels)
  ):
    raise _BoundaryError(
      "malformed-compiled-program",
      "program contribution channels must be unique exact strings",
    )
  if (
    not value.fixed_constraint_topology
    or not value.fixed_load_topology
    or value.state_dependent
    or value.has_follower_loads
    or value.has_interaction_tangent
    or value.has_program_state
  ):
    raise _BoundaryError(
      "malformed-compiled-program",
      "program capabilities contradict the frozen affine slice",
    )
  return value


def _validated_constraint_plan(
  value: object,
  *,
  coordinate_count: int,
) -> AffineConstraintPlan:
  if type(value) is not AffineConstraintPlan:
    raise _BoundaryError(
      "malformed-compiled-program",
      "constraint plan must be an exact AffineConstraintPlan",
    )
  (
    full_count,
    reduced_count,
    free_dofs,
    row_offsets,
    column_indices,
    coefficients,
    offset_constant,
    offset_coordinate_coefficients,
  ) = _required_slots(
    value,
    (
      "full_dof_count",
      "reduced_dof_count",
      "free_dofs",
      "row_offsets",
      "column_indices",
      "coefficients",
      "offset_constant",
      "offset_coordinate_coefficients",
    ),
    label="affine constraint plan",
  )
  if (
    type(full_count) is not int
    or type(reduced_count) is not int
    or full_count < 0
    or reduced_count < 0
    or full_count > _INT64_MAX
    or full_count + 1 > _INT64_MAX
    or reduced_count > full_count
  ):
    raise _BoundaryError(
      "malformed-compiled-program",
      "constraint-plan counts violate the frozen int64 policy",
    )
  free = _validated_finalized_array(
    free_dofs,
    dtype=_INDEX_DTYPE,
    shape=(reduced_count,),
    label="constraint-plan free DOFs",
  )
  offsets = _validated_finalized_array(
    row_offsets,
    dtype=_INDEX_DTYPE,
    shape=(full_count + 1,),
    label="constraint-plan row offsets",
  )
  if offsets.size == 0 or int(offsets[0]) != 0:
    raise _BoundaryError(
      "malformed-compiled-program",
      "constraint-plan row offsets must begin at zero",
    )
  nonzero_count = int(offsets[-1])
  if nonzero_count < 0 or nonzero_count > _INT64_MAX:
    raise _BoundaryError(
      "malformed-compiled-program",
      "constraint-plan terminal row offset is invalid",
    )
  columns = _validated_finalized_array(
    column_indices,
    dtype=_INDEX_DTYPE,
    shape=(nonzero_count,),
    label="constraint-plan column indices",
  )
  values = _validated_finalized_array(
    coefficients,
    dtype=_FLOATING_DTYPE,
    shape=(nonzero_count,),
    label="constraint-plan coefficients",
    finite=True,
  )
  constants = _validated_finalized_array(
    offset_constant,
    dtype=_FLOATING_DTYPE,
    shape=(full_count,),
    label="constraint-plan offset constants",
    finite=True,
  )
  coordinate_coefficients = _validated_finalized_array(
    offset_coordinate_coefficients,
    dtype=_FLOATING_DTYPE,
    shape=(full_count, coordinate_count),
    label="constraint-plan coordinate coefficients",
    finite=True,
  )
  differences = np.diff(offsets)
  if bool(np.any(differences < 0)) or bool(np.any(differences > 1)):
    raise _BoundaryError(
      "malformed-compiled-program",
      "each canonical constraint-plan row must contain zero or one entry",
    )
  previous_free_dof = -1
  free_rows: set[int] = set()
  for raw_dof in free:
    dof = int(raw_dof)
    if dof < 0 or dof >= full_count or dof <= previous_free_dof:
      raise _BoundaryError(
        "malformed-compiled-program",
        "constraint-plan free DOFs must be strictly increasing in range",
      )
    previous_free_dof = dof
    free_rows.add(dof)
  if nonzero_count and (
    reduced_count == 0 or int(columns.min()) < 0 or int(columns.max()) >= reduced_count
  ):
    raise _BoundaryError(
      "malformed-compiled-program",
      "constraint-plan column indices are outside the reduced space",
    )
  for column, dof in enumerate(free):
    row = int(dof)
    begin = int(offsets[row])
    end = int(offsets[row + 1])
    if (
      end - begin != 1
      or int(columns[begin]) != column
      or float(values[begin]) != 1.0
      or float(constants[row]) != 0.0
      or bool(np.any(coordinate_coefficients[row] != 0.0))
    ):
      raise _BoundaryError(
        "malformed-compiled-program",
        "independent constraint-plan rows must be exact identity rows",
      )
  for row in range(full_count):
    begin = int(offsets[row])
    end = int(offsets[row + 1])
    if row not in free_rows and end - begin == 1 and float(values[begin]) == 0.0:
      raise _BoundaryError(
        "malformed-compiled-program",
        "dependent one-entry constraint-plan rows require a nonzero coefficient",
      )
  return value


def _validated_load_plan(
  value: object,
  *,
  full_count: int,
  coordinate_count: int,
) -> NodalLoadPlan:
  if type(value) is not NodalLoadPlan:
    raise _BoundaryError(
      "malformed-compiled-program",
      "nodal-load plan must be an exact NodalLoadPlan",
    )
  load_ids, dof_indices, constant_values, coordinate_coefficients = _required_slots(
    value,
    ("load_ids", "dof_indices", "constant_values", "coordinate_coefficients"),
    label="nodal-load plan",
  )
  if type(load_ids) is not tuple:
    raise _BoundaryError(
      "malformed-compiled-program",
      "nodal-load IDs must use an exact tuple",
    )
  seen_ids: set[tuple[str, str | int]] = set()
  for identifier in load_ids:
    if not (type(identifier) is int or type(identifier) is str and bool(identifier)):
      raise _BoundaryError(
        "malformed-compiled-program",
        "nodal-load IDs must be exact semantic IDs",
      )
    key = _typed_id(identifier)
    if key in seen_ids:
      raise _BoundaryError(
        "malformed-compiled-program",
        "nodal-load plan contains duplicate semantic IDs",
      )
    seen_ids.add(key)
  load_count = len(load_ids)
  indices = _validated_finalized_array(
    dof_indices,
    dtype=_INDEX_DTYPE,
    shape=(load_count,),
    label="nodal-load DOF indices",
  )
  _validated_finalized_array(
    constant_values,
    dtype=_FLOATING_DTYPE,
    shape=(load_count,),
    label="nodal-load constants",
    finite=True,
  )
  _validated_finalized_array(
    coordinate_coefficients,
    dtype=_FLOATING_DTYPE,
    shape=(load_count, coordinate_count),
    label="nodal-load coordinate coefficients",
    finite=True,
  )
  if load_count and (int(indices.min()) < 0 or int(indices.max()) >= full_count):
    raise _BoundaryError(
      "malformed-compiled-program",
      "nodal-load DOF indices are outside the full model space",
    )
  previous: tuple[int, tuple[int, object]] | None = None
  for dof_index, identifier in zip(indices, load_ids, strict=True):
    key = int(dof_index), _semantic_id_key(identifier)
    if previous is not None and key < previous:
      raise _BoundaryError(
        "malformed-compiled-program",
        "nodal-load rows are not in canonical target/semantic-ID order",
      )
    previous = key
  return value


def _validate_separate_program_storage(
  constraint_plan: AffineConstraintPlan,
  nodal_load_plan: NodalLoadPlan,
  witness_dofs: DofPlan,
) -> None:
  numeric_fields = (
    ("constraint free DOFs", constraint_plan.free_dofs),
    ("constraint row offsets", constraint_plan.row_offsets),
    ("constraint column indices", constraint_plan.column_indices),
    ("constraint coefficients", constraint_plan.coefficients),
    ("constraint offset constants", constraint_plan.offset_constant),
    (
      "constraint offset coordinate coefficients",
      constraint_plan.offset_coordinate_coefficients,
    ),
    ("nodal-load DOF indices", nodal_load_plan.dof_indices),
    ("nodal-load constants", nodal_load_plan.constant_values),
    (
      "nodal-load coordinate coefficients",
      nodal_load_plan.coordinate_coefficients,
    ),
    ("program meaning witness DOF indices", witness_dofs.node_component_dofs),
  )
  for index, (left_label, left) in enumerate(numeric_fields):
    for right_label, right in numeric_fields[index + 1 :]:
      if (
        left is right
        or left.values is right.values
        or np.shares_memory(left.values, right.values)
      ):
        raise _BoundaryError(
          "malformed-compiled-program",
          f"{left_label} and {right_label} require separate finalized storage",
        )


def _validate_program_map_entry(
  entity: EntityRecord,
  source: SourceRecord,
  *,
  kind: str,
  semantic_id: str | int,
  dense_index: int,
  local_index: int | None,
) -> None:
  if (
    entity.kind != kind
    or _entity_key(entity.kind, entity.semantic_id) != _entity_key(kind, semantic_id)
    or entity.dense_index != dense_index
    or entity.block_index is not None
    or entity.local_index != local_index
    or source.kind != kind
    or _entity_key(source.kind, source.semantic_id) != _entity_key(kind, semantic_id)
  ):
    raise _BoundaryError(
      "malformed-compiled-program",
      "program entity/source maps do not match visible coordinate and plan meaning",
    )


def _validate_program_maps(
  entity_index: EntityIndex,
  source_map: SourceMap,
  *,
  coordinate_names: tuple[str, ...],
  constraint_count: int,
  load_ids: tuple[str | int, ...],
) -> None:
  entities = entity_index.records
  sources = source_map.records
  expected_count = 1 + len(coordinate_names) + constraint_count + len(load_ids)
  if len(entities) != expected_count or len(sources) != expected_count:
    raise _BoundaryError(
      "malformed-compiled-program",
      "program entity/source maps do not match visible coordinate and plan meaning",
    )

  _validate_program_map_entry(
    entities[0],
    sources[0],
    kind="program",
    semantic_id="program",
    dense_index=0,
    local_index=None,
  )
  position = 1
  for index, name in enumerate(coordinate_names):
    _validate_program_map_entry(
      entities[position],
      sources[position],
      kind="program_coordinate",
      semantic_id=name,
      dense_index=index,
      local_index=index,
    )
    position += 1

  previous_constraint_id: tuple[int, object] | None = None
  for index in range(constraint_count):
    identifier = entities[position].semantic_id
    if not (type(identifier) is int or type(identifier) is str and bool(identifier)):
      raise _BoundaryError(
        "malformed-compiled-program",
        "program constraint map IDs must be exact semantic IDs",
      )
    identifier_key = _semantic_id_key(identifier)
    if previous_constraint_id is not None and identifier_key <= previous_constraint_id:
      raise _BoundaryError(
        "malformed-compiled-program",
        "program constraint map IDs are not in canonical semantic-ID order",
      )
    _validate_program_map_entry(
      entities[position],
      sources[position],
      kind="program_constraint",
      semantic_id=identifier,
      dense_index=index,
      local_index=index,
    )
    previous_constraint_id = identifier_key
    position += 1

  for index, identifier in enumerate(load_ids):
    _validate_program_map_entry(
      entities[position],
      sources[position],
      kind="program_load",
      semantic_id=identifier,
      dense_index=index,
      local_index=index,
    )
    position += 1


def _validate_derived_program_meaning(
  capabilities: ProgramCapabilities,
  constraint_plan: AffineConstraintPlan,
  nodal_load_plan: NodalLoadPlan,
  entity_index: EntityIndex,
  source_map: SourceMap,
  *,
  coordinate_names: tuple[str, ...],
) -> None:
  expected_capabilities = _derived_capabilities(
    constraint_plan,
    nodal_load_plan,
  )
  if any(
    object.__getattribute__(capabilities, name)
    != object.__getattribute__(expected_capabilities, name)
    for name in _PROGRAM_CAPABILITY_SLOTS
  ):
    raise _BoundaryError(
      "malformed-compiled-program",
      "program capabilities do not match visible constraint and load plans",
    )
  _validate_program_maps(
    entity_index,
    source_map,
    coordinate_names=coordinate_names,
    constraint_count=(
      constraint_plan.full_dof_count - constraint_plan.reduced_dof_count
    ),
    load_ids=nodal_load_plan.load_ids,
  )


def _witness_source_context(value: object, *, label: str) -> SourceContext:
  if type(value) is not CompiledSource:
    raise _BoundaryError(
      "malformed-compiled-program",
      f"{label} must be an exact CompiledSource",
    )
  source, line, column = _required_slots(
    value,
    ("source", "line", "column"),
    label=label,
  )
  if (
    type(source) is not str
    or (line is not None and type(line) is not int)
    or (column is not None and type(column) is not int)
  ):
    raise _BoundaryError(
      "malformed-compiled-program",
      f"{label} contains malformed exact source fields",
    )
  return SourceContext(source=source, line=line, column=column)


def _witness_float(value: object, *, label: str, nonzero: bool = False) -> float:
  if type(value) is not float or not math.isfinite(value) or (nonzero and value == 0.0):
    raise _BoundaryError(
      "malformed-compiled-program",
      f"{label} must be a finite exact float",
    )
  return value


def _witness_dof_ref(value: object, *, label: str) -> DofRef:
  if type(value) is not ProgramDofWitness:
    raise _BoundaryError(
      "malformed-compiled-program",
      f"{label} must be an exact ProgramDofWitness",
    )
  node_id, field_id, component = _required_slots(
    value,
    ("node_id", "field_id", "component"),
    label=label,
  )
  _validated_semantic_id(node_id, label=f"{label} node ID")
  _validated_semantic_id(field_id, label=f"{label} field ID")
  if type(component) is not str or not component:
    raise _BoundaryError(
      "malformed-compiled-program",
      f"{label} component must be a non-empty exact string",
    )
  return DofRef(node_id=node_id, field_id=field_id, component=component)


def _witness_affine_value(value: object, *, label: str) -> AffineValueSpec:
  if type(value) is not ProgramAffineValueWitness:
    raise _BoundaryError(
      "malformed-compiled-program",
      f"{label} must be an exact ProgramAffineValueWitness",
    )
  constant, coefficients, source = _required_slots(
    value,
    ("constant", "coefficients", "source"),
    label=label,
  )
  validated_constant = _witness_float(constant, label=f"{label} constant")
  validated_source = _witness_source_context(source, label=f"{label} source")
  if type(coefficients) is not tuple:
    raise _BoundaryError(
      "malformed-compiled-program",
      f"{label} coefficients must use an exact tuple",
    )
  authored_coefficients: list[AffineCoefficientSpec] = []
  for coefficient in coefficients:
    if type(coefficient) is not ProgramAffineCoefficientWitness:
      raise _BoundaryError(
        "malformed-compiled-program",
        f"{label} coefficient must be an exact ProgramAffineCoefficientWitness",
      )
    coordinate, numeric_value, coefficient_source = _required_slots(
      coefficient,
      ("coordinate", "coefficient", "source"),
      label=f"{label} coefficient",
    )
    if type(coordinate) is not str or not coordinate:
      raise _BoundaryError(
        "malformed-compiled-program",
        f"{label} coefficient coordinate must be a non-empty exact string",
      )
    authored_coefficients.append(
      AffineCoefficientSpec(
        coordinate=coordinate,
        coefficient=_witness_float(
          numeric_value,
          label=f"{label} coefficient value",
        ),
        source=_witness_source_context(
          coefficient_source,
          label=f"{label} coefficient source",
        ),
      )
    )
  return AffineValueSpec(
    constant=validated_constant,
    coefficients=tuple(authored_coefficients),
    source=validated_source,
  )


def _program_spec_from_witness(
  value: object,
) -> tuple[ProgramSpec, DofPlan]:
  if type(value) is not ProgramMeaningWitness:
    raise _BoundaryError(
      "malformed-compiled-program",
      "program meaning witness must be an exact ProgramMeaningWitness",
    )
  coordinates, constraints, loads, source, model_dofs = _required_slots(
    value,
    ("coordinates", "constraints", "loads", "source", "model_dofs"),
    label="program meaning witness",
  )
  if any(type(items) is not tuple for items in (coordinates, constraints, loads)):
    raise _BoundaryError(
      "malformed-compiled-program",
      "program meaning witness declarations must use exact tuples",
    )
  authored_coordinates: list[ProgramCoordinateSpec] = []
  for coordinate in coordinates:
    if type(coordinate) is not ProgramCoordinateWitness:
      raise _BoundaryError(
        "malformed-compiled-program",
        "program coordinate witness must be exact",
      )
    name, kind, coordinate_source = _required_slots(
      coordinate,
      ("name", "kind", "source"),
      label="program coordinate witness",
    )
    if type(name) is not str or not name or type(kind) is not str or not kind:
      raise _BoundaryError(
        "malformed-compiled-program",
        "program coordinate witness names and kinds must be exact strings",
      )
    authored_coordinates.append(
      ProgramCoordinateSpec(
        name=name,
        kind=kind,
        source=_witness_source_context(
          coordinate_source,
          label="program coordinate witness source",
        ),
      )
    )
  authored_constraints: list[PrescribedDofSpec | AffineTieSpec] = []
  for constraint in constraints:
    if type(constraint) is ProgramPrescribedWitness:
      identifier, target, affine, constraint_source = _required_slots(
        constraint,
        ("id", "target", "value", "source"),
        label="program prescribed witness",
      )
      _validated_semantic_id(identifier, label="program prescribed witness ID")
      authored_constraints.append(
        PrescribedDofSpec(
          id=identifier,
          target=_witness_dof_ref(target, label="program prescribed target"),
          value=_witness_affine_value(
            affine,
            label="program prescribed affine value",
          ),
          source=_witness_source_context(
            constraint_source,
            label="program prescribed witness source",
          ),
        )
      )
      continue
    if type(constraint) is not ProgramTieWitness:
      raise _BoundaryError(
        "malformed-compiled-program",
        "program constraint witness must be prescribed or one-master affine",
      )
    identifier, slave, master, factor, offset, constraint_source = _required_slots(
      constraint,
      ("id", "slave", "master", "factor", "offset", "source"),
      label="program tie witness",
    )
    _validated_semantic_id(identifier, label="program tie witness ID")
    authored_constraints.append(
      AffineTieSpec(
        id=identifier,
        slave=_witness_dof_ref(slave, label="program tie slave"),
        master=_witness_dof_ref(master, label="program tie master"),
        factor=_witness_float(
          factor,
          label="program tie factor",
          nonzero=True,
        ),
        offset=_witness_affine_value(offset, label="program tie offset"),
        source=_witness_source_context(
          constraint_source,
          label="program tie witness source",
        ),
      )
    )
  authored_loads: list[NodalLoadSpec] = []
  for load in loads:
    if type(load) is not ProgramLoadWitness:
      raise _BoundaryError(
        "malformed-compiled-program",
        "program load witness must be an exact ProgramLoadWitness",
      )
    identifier, target, affine, load_source = _required_slots(
      load,
      ("id", "target", "value", "source"),
      label="program load witness",
    )
    _validated_semantic_id(identifier, label="program load witness ID")
    authored_loads.append(
      NodalLoadSpec(
        id=identifier,
        target=_witness_dof_ref(target, label="program load target"),
        value=_witness_affine_value(affine, label="program load affine value"),
        source=_witness_source_context(
          load_source,
          label="program load witness source",
        ),
      )
    )
  authored = ProgramSpec(
    coordinates=tuple(authored_coordinates),
    constraints=tuple(authored_constraints),
    loads=tuple(authored_loads),
    source=_witness_source_context(source, label="program meaning witness source"),
  )
  validated_dofs = _validated_dof_plan(
    model_dofs,
    dense_index_dtype=None,
    label="program meaning witness DOF plan",
  )
  try:
    normalized = normalize_program_spec(authored)
  except ProgramSpecValidationError as exc:
    raise _BoundaryError(
      "malformed-compiled-program",
      "program meaning witness violates the normalized authored vocabulary",
    ) from exc
  if (
    _normalized_program_manifest(authored).to_bytes()
    != _normalized_program_manifest(normalized).to_bytes()
  ):
    raise _BoundaryError(
      "malformed-compiled-program",
      "program meaning witness is not in canonical normalized order",
    )
  return normalized, validated_dofs


def _same_finalized_values(left: FinalizedArray, right: FinalizedArray) -> bool:
  return (
    left.values.dtype == right.values.dtype
    and left.values.shape == right.values.shape
    and left.values.tobytes(order="C") == right.values.tobytes(order="C")
  )


def _same_constraint_plan(
  left: AffineConstraintPlan,
  right: AffineConstraintPlan,
) -> bool:
  return (
    left.full_dof_count == right.full_dof_count
    and left.reduced_dof_count == right.reduced_dof_count
    and _same_finalized_values(left.free_dofs, right.free_dofs)
    and _same_finalized_values(left.row_offsets, right.row_offsets)
    and _same_finalized_values(left.column_indices, right.column_indices)
    and _same_finalized_values(left.coefficients, right.coefficients)
    and _same_finalized_values(left.offset_constant, right.offset_constant)
    and _same_finalized_values(
      left.offset_coordinate_coefficients,
      right.offset_coordinate_coefficients,
    )
  )


def _same_load_plan(left: NodalLoadPlan, right: NodalLoadPlan) -> bool:
  return (
    left.load_ids == right.load_ids
    and _same_finalized_values(left.dof_indices, right.dof_indices)
    and _same_finalized_values(left.constant_values, right.constant_values)
    and _same_finalized_values(
      left.coordinate_coefficients,
      right.coordinate_coefficients,
    )
  )


def _same_program_maps(
  left_entities: EntityIndex,
  left_sources: SourceMap,
  right_entities: EntityIndex,
  right_sources: SourceMap,
) -> bool:
  if len(left_entities.records) != len(right_entities.records) or len(
    left_sources.records
  ) != len(right_sources.records):
    return False
  for left, right in zip(
    left_entities.records,
    right_entities.records,
    strict=True,
  ):
    if (
      left.kind != right.kind
      or _entity_key(left.kind, left.semantic_id)
      != _entity_key(right.kind, right.semantic_id)
      or left.dense_index != right.dense_index
      or left.block_index != right.block_index
      or left.local_index != right.local_index
    ):
      return False
  for left, right in zip(left_sources.records, right_sources.records, strict=True):
    if (
      left.kind != right.kind
      or _entity_key(left.kind, left.semantic_id)
      != _entity_key(right.kind, right.semantic_id)
      or left.source.source != right.source.source
      or left.source.line != right.source.line
      or left.source.column != right.source.column
    ):
      return False
  return True


def _validate_program_witness_correspondence(
  witness: object,
  *,
  normalized_manifest: CanonicalManifest,
  coordinate_names: tuple[str, ...],
  coordinate_kinds: tuple[str, ...],
  constraint_plan: AffineConstraintPlan,
  nodal_load_plan: NodalLoadPlan,
  entity_index: EntityIndex,
  source_map: SourceMap,
) -> DofPlan:
  normalized, witness_dofs = _program_spec_from_witness(witness)
  witness_manifest = _normalized_program_manifest(normalized)
  if witness_manifest.to_bytes() != normalized_manifest.to_bytes():
    raise _BoundaryError(
      "malformed-compiled-program",
      "program meaning witness does not match normalized program provenance",
    )
  expected_names = tuple(item.name for item in normalized.coordinates)
  expected_kinds = tuple(item.kind for item in normalized.coordinates)
  if coordinate_names != expected_names or coordinate_kinds != expected_kinds:
    raise _BoundaryError(
      "malformed-compiled-program",
      "visible program coordinate schema does not match normalized meaning",
    )
  coordinate_indices = {name: index for index, name in enumerate(expected_names)}
  lookup = _dof_lookup_from_plan(witness_dofs)
  try:
    expected_constraints = _compile_constraints(
      normalized,
      lookup,
      coordinate_indices,
    )
    expected_loads = _compile_loads(
      normalized,
      lookup,
      coordinate_indices,
    )
  except ProgramCompilationError as exc:
    raise _BoundaryError(
      "malformed-compiled-program",
      "program meaning witness cannot reproduce canonical visible plans",
    ) from exc
  if not _same_constraint_plan(constraint_plan, expected_constraints):
    raise _BoundaryError(
      "malformed-compiled-program",
      "visible affine constraint plan does not match normalized program meaning",
    )
  if not _same_load_plan(nodal_load_plan, expected_loads):
    raise _BoundaryError(
      "malformed-compiled-program",
      "visible nodal-load plan does not match normalized program meaning",
    )
  expected_entities, expected_sources = _build_program_maps(
    normalized,
    expected_loads,
  )
  if not _same_program_maps(
    entity_index,
    source_map,
    expected_entities,
    expected_sources,
  ):
    raise _BoundaryError(
      "malformed-compiled-program",
      "visible program entity/source maps do not match normalized program meaning",
    )
  return witness_dofs


def _validated_program(value: object) -> CompiledProgram:
  if type(value) is not CompiledProgram:
    raise _BoundaryError(
      "invalid-compiled-program",
      "evaluation requires an exact CompiledProgram",
    )
  (
    instance_id,
    content_fingerprint,
    provenance,
    model_instance_id,
    model_fingerprint,
    coordinate_names,
    coordinate_kinds,
    meaning_witness,
    constraint_plan,
    nodal_load_plan,
    capabilities,
    entity_index,
    source_map,
  ) = _required_slots(
    value,
    (
      "instance_id",
      "content_fingerprint",
      "provenance",
      "compatible_model_instance_id",
      "compatible_model_content_fingerprint",
      "coordinate_names",
      "coordinate_kinds",
      "meaning_witness",
      "constraint_plan",
      "nodal_load_plan",
      "capabilities",
      "entity_index",
      "source_map",
    ),
    label="compiled program",
  )
  _validated_instance_id(instance_id, label="compiled program instance identity")
  fingerprint = _validated_fingerprint(
    content_fingerprint,
    label="compiled program content fingerprint",
  )
  _validated_instance_id(model_instance_id, label="compatible model instance identity")
  compatible_fingerprint = _validated_fingerprint(
    model_fingerprint,
    label="compatible model content fingerprint",
  )
  if (
    type(coordinate_names) is not tuple
    or type(coordinate_kinds) is not tuple
    or len(coordinate_names) != len(coordinate_kinds)
  ):
    raise _BoundaryError(
      "malformed-compiled-program",
      "compiled coordinate names and kinds must be equal-length exact tuples",
    )
  previous_coordinate: tuple[int, str] | None = None
  seen_names: set[str] = set()
  for name, kind in zip(coordinate_names, coordinate_kinds, strict=True):
    if (
      type(name) is not str
      or not name
      or type(kind) is not str
      or kind not in _COORDINATE_KIND_ORDER
      or (name == "time") != (kind == "time")
      or name in seen_names
    ):
      raise _BoundaryError(
        "malformed-compiled-program",
        "compiled coordinate schema violates the frozen exact vocabulary",
      )
    key = _COORDINATE_KIND_ORDER[kind], name
    if previous_coordinate is not None and key <= previous_coordinate:
      raise _BoundaryError(
        "malformed-compiled-program",
        "compiled coordinates are not in canonical kind/name order",
      )
    previous_coordinate = key
    seen_names.add(name)

  if type(provenance) is not ProgramProvenance:
    raise _BoundaryError(
      "malformed-compiled-program",
      "program provenance must be an exact ProgramProvenance",
    )
  (
    schema,
    manifest,
    normalized_manifest,
    floating_dtype,
    index_dtype,
    coordinate_order,
    reduction_policy,
  ) = _required_slots(
    provenance,
    (
      "schema",
      "manifest",
      "normalized_program_manifest",
      "floating_dtype",
      "index_dtype",
      "coordinate_order",
      "reduction_policy",
    ),
    label="program provenance",
  )
  if (
    type(schema) is not str
    or schema != COMPILED_PROGRAM_MANIFEST_SCHEMA
    or type(floating_dtype) is not str
    or floating_dtype != _FLOATING_DTYPE.str
    or type(index_dtype) is not str
    or index_dtype != _INDEX_DTYPE.str
    or type(coordinate_order) is not tuple
    or coordinate_order != _COORDINATE_ORDER
    or type(reduction_policy) is not str
    or reduction_policy != PROGRAM_REDUCTION_POLICY
  ):
    raise _BoundaryError(
      "malformed-compiled-program",
      "program provenance does not match the frozen numeric/reduction policy",
    )
  validated_manifest = _validated_manifest(manifest, label="program manifest")
  validated_normalized_manifest = _validated_manifest(
    normalized_manifest,
    label="normalized program manifest",
  )
  validated_constraint_plan = _validated_constraint_plan(
    constraint_plan,
    coordinate_count=len(coordinate_names),
  )
  validated_load_plan = _validated_load_plan(
    nodal_load_plan,
    full_count=validated_constraint_plan.full_dof_count,
    coordinate_count=len(coordinate_names),
  )
  validated_capabilities = _validated_capabilities(capabilities)
  validated_entities = _validated_entity_index(entity_index)
  validated_sources = _validated_source_map(source_map)
  witness_dofs = _validate_program_witness_correspondence(
    meaning_witness,
    normalized_manifest=validated_normalized_manifest,
    coordinate_names=coordinate_names,
    coordinate_kinds=coordinate_kinds,
    constraint_plan=validated_constraint_plan,
    nodal_load_plan=validated_load_plan,
    entity_index=validated_entities,
    source_map=validated_sources,
  )
  _validate_separate_program_storage(
    validated_constraint_plan,
    validated_load_plan,
    witness_dofs,
  )
  _validate_derived_program_meaning(
    validated_capabilities,
    validated_constraint_plan,
    validated_load_plan,
    validated_entities,
    validated_sources,
    coordinate_names=coordinate_names,
  )
  expected_manifest = _program_manifest(
    normalized_manifest=validated_normalized_manifest,
    model_fingerprint=compatible_fingerprint,
    coordinate_names=coordinate_names,
    coordinate_kinds=coordinate_kinds,
    constraint_plan=validated_constraint_plan,
    nodal_load_plan=validated_load_plan,
    capabilities=validated_capabilities,
    entity_index=validated_entities,
    source_map=validated_sources,
  )
  try:
    manifest_matches = validated_manifest.to_bytes() == expected_manifest.to_bytes()
    fingerprint_matches = fingerprint == ContentFingerprint.from_manifest(
      expected_manifest
    )
  except (TypeError, ValueError) as exc:
    raise _BoundaryError(
      "malformed-compiled-program",
      "compiled program content identity is malformed",
    ) from exc
  if not manifest_matches or not fingerprint_matches:
    raise _BoundaryError(
      "malformed-compiled-program",
      "compiled program fields do not match their canonical content identity",
    )
  return value


def _validated_point(
  point: object,
  coordinate_names: tuple[str, ...],
) -> tuple[float, ...]:
  diagnostics: list[ProgramEvaluationDiagnostic] = []
  source = SourceContext(source="<program-point>")
  if type(point) is not ProgramPoint:
    diagnostics.append(
      ProgramEvaluationDiagnostic(
        code="invalid-program-point-type",
        message="program point must be exactly ProgramPoint",
        source=source,
      )
    )
    raise ProgramEvaluationError(diagnostics)
  try:
    (raw_values,) = _required_slots(point, ("values",), label="program point")
  except _BoundaryError:
    diagnostics.append(
      ProgramEvaluationDiagnostic(
        code="invalid-program-point-value",
        message="program point must initialize every canonical slot",
        source=source,
      )
    )
    raise ProgramEvaluationError(diagnostics) from None
  if type(raw_values) is not tuple:
    diagnostics.append(
      ProgramEvaluationDiagnostic(
        code="invalid-program-point-value",
        message="program point values must be an exact tuple",
        source=source,
      )
    )
    raise ProgramEvaluationError(diagnostics)

  by_name: dict[str, float] = {}
  declared = set(coordinate_names)
  for index, item in enumerate(raw_values):
    if type(item) is not ProgramCoordinateValue:
      diagnostics.append(
        ProgramEvaluationDiagnostic(
          code="invalid-program-coordinate-value-type",
          message=f"program point entry {render_program_value(index)} must be "
          "exactly ProgramCoordinateValue",
          source=source,
        )
      )
      continue
    slots = tuple(_read_slot(item, name) for name in ("name", "value"))
    if any(value is _MISSING for value in slots):
      diagnostics.append(
        ProgramEvaluationDiagnostic(
          code="invalid-program-coordinate-value",
          message="program coordinate value must initialize every canonical slot",
          source=source,
        )
      )
      continue
    name, raw_value = slots
    if type(name) is not str or not name:
      diagnostics.append(
        ProgramEvaluationDiagnostic(
          code="invalid-program-point-name",
          message="program point coordinate name must be a non-empty exact string",
          source=source,
        )
      )
      continue
    if name in by_name:
      diagnostics.append(
        ProgramEvaluationDiagnostic(
          code="duplicate-program-point-coordinate",
          message="program point repeats a coordinate name",
          source=source,
        )
      )
      continue
    if name not in declared:
      diagnostics.append(
        ProgramEvaluationDiagnostic(
          code="extra-program-point-coordinate",
          message="program point contains an undeclared coordinate name",
          source=source,
        )
      )
      continue
    if type(raw_value) is not int and type(raw_value) is not float:
      diagnostics.append(
        ProgramEvaluationDiagnostic(
          code="invalid-program-point-coordinate",
          message="program coordinate value must be a finite plain integer or float",
          source=source,
        )
      )
      continue
    try:
      converted = float(raw_value)
    except OverflowError:
      converted = math.inf
    if not math.isfinite(converted):
      diagnostics.append(
        ProgramEvaluationDiagnostic(
          code="invalid-program-point-coordinate",
          message="program coordinate value cannot be represented as finite float64",
          source=source,
        )
      )
      continue
    by_name[name] = converted
  missing = tuple(name for name in coordinate_names if name not in by_name)
  if missing:
    diagnostics.append(
      ProgramEvaluationDiagnostic(
        code="missing-program-point-coordinate",
        message="program point does not bind every declared coordinate exactly once",
        source=source,
      )
    )
  if diagnostics:
    raise ProgramEvaluationError(diagnostics)
  return tuple(by_name[name] for name in coordinate_names)


def _evaluation_product(left: float, right: float, *, label: str) -> float:
  result = left * right
  if not math.isfinite(result):
    _evaluation_fail(
      "nonfinite-program-evaluation",
      f"{label} produced a nonfinite binary64 product",
    )
  return result


def _evaluation_fsum(values: tuple[float, ...], *, label: str) -> float:
  try:
    result = math.fsum(values)
  except OverflowError:
    _evaluation_fail(
      "nonfinite-program-evaluation",
      f"{label} overflowed deterministic binary64 accumulation",
    )
  if not math.isfinite(result):
    _evaluation_fail(
      "nonfinite-program-evaluation",
      f"{label} produced a nonfinite deterministic sum",
    )
  return result


def evaluate_program(
  program: CompiledProgram,
  point: ProgramPoint,
) -> ProgramEvaluation:
  """Bind one exact program point into immutable full-space affine values."""
  try:
    validated = _validated_program(program)
  except _BoundaryError as error:
    _evaluation_fail(error.code, error.message)
  coordinate_values = _validated_point(point, validated.coordinate_names)
  constraint_plan = validated.constraint_plan
  full_count = constraint_plan.full_dof_count
  coordinate_count = len(coordinate_values)
  offset_constants = constraint_plan.offset_constant.values
  offset_coefficients = constraint_plan.offset_coordinate_coefficients.values
  prescribed_offsets = np.empty(full_count, dtype=np.float64)
  for dof in range(full_count):
    products = tuple(
      _evaluation_product(
        float(offset_coefficients[dof, coordinate]),
        coordinate_values[coordinate],
        label="prescribed offset",
      )
      for coordinate in range(coordinate_count)
    )
    prescribed_offsets[dof] = _evaluation_fsum(
      (float(offset_constants[dof]), *products),
      label="prescribed offset",
    )

  load_plan = validated.nodal_load_plan
  load_indices = load_plan.dof_indices.values
  load_constants = load_plan.constant_values.values
  load_coefficients = load_plan.coordinate_coefficients.values
  nodal_force = np.zeros(full_count, dtype=np.float64)
  nodal_derivatives = np.zeros(
    (full_count, coordinate_count),
    dtype=np.float64,
  )
  row = 0
  while row < len(load_indices):
    target = int(load_indices[row])
    end = row + 1
    while end < len(load_indices) and int(load_indices[end]) == target:
      end += 1
    constant = _evaluation_fsum(
      tuple(float(value) for value in load_constants[row:end]),
      label="nodal-load constant",
    )
    reduced_coefficients: list[float] = []
    for coordinate in range(coordinate_count):
      reduced = _evaluation_fsum(
        tuple(float(value) for value in load_coefficients[row:end, coordinate]),
        label="nodal-load coordinate coefficient",
      )
      reduced_coefficients.append(reduced)
      nodal_derivatives[target, coordinate] = reduced
    products = tuple(
      _evaluation_product(
        reduced_coefficients[coordinate],
        coordinate_values[coordinate],
        label="nodal force",
      )
      for coordinate in range(coordinate_count)
    )
    nodal_force[target] = _evaluation_fsum(
      (constant, *products),
      label="nodal force",
    )
    row = end

  return ProgramEvaluation(
    program_instance_id=validated.instance_id,
    program_content_fingerprint=validated.content_fingerprint,
    compatible_model_instance_id=validated.compatible_model_instance_id,
    compatible_model_content_fingerprint=(
      validated.compatible_model_content_fingerprint
    ),
    coordinate_names=tuple(name for name in validated.coordinate_names),
    coordinate_values=FinalizedArray(coordinate_values, dtype=np.float64),
    prescribed_offsets=FinalizedArray(prescribed_offsets, dtype=np.float64),
    prescribed_offset_derivatives=FinalizedArray(
      offset_coefficients,
      dtype=np.float64,
    ),
    nodal_force=FinalizedArray(nodal_force, dtype=np.float64),
    nodal_force_derivatives=FinalizedArray(
      nodal_derivatives,
      dtype=np.float64,
    ),
  )
