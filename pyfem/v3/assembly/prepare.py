"""Prepare immutable attributed COO topology for the frozen Q8 slice."""

from __future__ import annotations

import math
from typing import NoReturn

import numpy as np

from pyfem.v3.assembly.contracts import (
  AssemblyPlanProvenance,
  DomainCooPlan,
  LinearStaticContributionRequest,
  NodalVectorContributionPlan,
  PreparedAssemblyPlan,
)
from pyfem.v3.assembly.diagnostics import (
  AssemblyPreparationDiagnostic,
  AssemblyPreparationError,
)
from pyfem.v3.compile.contracts import (
  Q8_FORMULATION_KEY,
  Q8_MATERIAL_KEY,
  Q8_QUADRATURE_KEY,
  Q8_TOPOLOGY_KEY,
  q8_descriptor_metadata,
)
from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.compiled import CompiledModel, CompiledSource, DomainBlock
from pyfem.v3.model.identity import IdentityMismatchError, require_same_instance
from pyfem.v3.model.program import CompiledProgram
from pyfem.v3.model.provenance import CanonicalManifest, ContentFingerprint
from pyfem.v3.spec.diagnostics import SourceContext

PREPARED_ASSEMBLY_PLAN_MANIFEST_SCHEMA = "pyfem-v3-prepared-assembly-plan-q8-v1"
ASSEMBLY_REDUCTION_POLICY = (
  "raw order is block, canonical cell, local row, local column; reduced raw "
  "products use (left * full entry) * right; duplicate groups and vector "
  "components use math.fsum in ascending raw/full-DOF source order"
)
ASSEMBLY_GEOMETRY_POLICY = (
  "translation-free scalar-normalized Q8 geometry; positive det(J_hat) and "
  "det(J_hat)/sum(J_hat**2) must exceed the compiled relative tolerance"
)

_INDEX_DTYPE = np.dtype(np.int64)
_FLOATING_DTYPE = np.dtype(np.float64)
_INT64_MAX = int(np.iinfo(np.int64).max)
_INTP_MAX = int(np.iinfo(np.intp).max)
_LOCAL_DOF_COUNT = 16
_RAW_ENTRIES_PER_CELL = _LOCAL_DOF_COUNT * _LOCAL_DOF_COUNT
_PREPARATION_SOURCE = SourceContext(source="<assembly-preparation>")
_EVALUATION_SOURCE = SourceContext(source="<assembly-evaluation>")
_MISSING = object()


class _AssemblyBoundaryError(Exception):
  def __init__(self, code: str, message: str, source: SourceContext) -> None:
    self.code = code
    self.message = message
    self.source = source
    super().__init__(message)


def _raise_boundary(
  code: str,
  message: str,
  source: SourceContext,
) -> NoReturn:
  raise _AssemblyBoundaryError(code, message, source)


def _read_slot(value: object, name: str) -> object:
  try:
    return object.__getattribute__(value, name)
  except AttributeError:
    return _MISSING


def _copy_compiled_source(value: CompiledSource) -> SourceContext:
  return SourceContext(source=value.source, line=value.line, column=value.column)


def _model_source(
  model: CompiledModel,
  kind: str,
  semantic_id: object,
  *,
  fallback: SourceContext,
) -> SourceContext:
  try:
    source = model.source_map.lookup(kind, semantic_id)
  except (KeyError, TypeError, ValueError):
    return fallback
  return _copy_compiled_source(source)


def _program_source(
  program: CompiledProgram,
  *,
  fallback: SourceContext,
) -> SourceContext:
  try:
    source = program.source_map.lookup("program", "program")
  except (KeyError, TypeError, ValueError):
    return fallback
  return _copy_compiled_source(source)


def _validated_array_values(
  value: object,
  *,
  dtype: np.dtype[np.generic],
  shape: tuple[int, ...],
  label: str,
  source: SourceContext,
  finite: bool = False,
) -> np.ndarray:
  if type(value) is not FinalizedArray:
    _raise_boundary(
      "malformed-assembly-carrier",
      f"{label} must be an exact FinalizedArray",
      source,
    )
  values = _read_slot(value, "values")
  if (
    type(values) is not np.ndarray
    or values.dtype != dtype
    or values.dtype.metadata is not None
    or values.shape != shape
  ):
    _raise_boundary(
      "malformed-assembly-carrier",
      f"{label} has the wrong canonical metadata-free ndarray dtype or shape",
      source,
    )
  if (
    not values.flags.owndata or not values.flags.c_contiguous or values.flags.writeable
  ):
    _raise_boundary(
      "malformed-assembly-carrier",
      f"{label} must be owning, C-contiguous, and read-only",
      source,
    )
  if finite and not bool(np.isfinite(values).all()):
    _raise_boundary(
      "malformed-assembly-carrier",
      f"{label} must contain only finite float64 values",
      source,
    )
  return values


def _checked_count(value: int, *, label: str, source: SourceContext) -> int:
  if type(value) is not int or value < 0:
    _raise_boundary(
      "invalid-assembly-count",
      f"{label} must be a non-negative exact integer",
      source,
    )
  if value > _INT64_MAX or value > _INTP_MAX:
    _raise_boundary(
      "assembly-count-overflow",
      f"{label} exceeds the exact int64 or platform allocation policy",
      source,
    )
  return value


def _checked_product(
  left: int,
  right: int,
  *,
  label: str,
  source: SourceContext,
) -> int:
  checked_left = _checked_count(left, label=label, source=source)
  checked_right = _checked_count(right, label=label, source=source)
  if checked_left and checked_right > min(_INT64_MAX, _INTP_MAX) // checked_left:
    _raise_boundary(
      "assembly-count-overflow",
      f"{label} exceeds the exact int64 or platform allocation policy",
      source,
    )
  return checked_left * checked_right


def _check_allocation(
  count: int,
  dtype: np.dtype[np.generic],
  *,
  label: str,
  source: SourceContext,
) -> None:
  _checked_product(
    count,
    int(dtype.itemsize),
    label=f"{label} allocation",
    source=source,
  )


def _validate_model_program_compatibility(
  model: CompiledModel,
  program: CompiledProgram,
  *,
  source: SourceContext,
) -> None:
  require_same_instance(
    model.instance_id,
    program.compatible_model_instance_id,
    context="assembly model/program composition",
  )
  if model.content_fingerprint != program.compatible_model_content_fingerprint:
    _raise_boundary(
      "model-program-content-mismatch",
      "compiled program content compatibility does not match the compiled model",
      source,
    )
  if program.constraint_plan.full_dof_count != model.dofs.global_size:
    _raise_boundary(
      "model-program-dof-mismatch",
      "compiled program full-space size does not match the compiled model",
      source,
    )


def _descriptor_source(
  model: CompiledModel,
  block: DomainBlock,
  key: tuple[str, str],
  *,
  fallback: SourceContext,
) -> SourceContext:
  if key == Q8_TOPOLOGY_KEY:
    return _model_source(
      model,
      "cell_block",
      block.source_cell_block_id,
      fallback=fallback,
    )
  if key == Q8_MATERIAL_KEY:
    return _model_source(
      model,
      "material",
      block.source_material_id,
      fallback=fallback,
    )
  return _model_source(
    model,
    "region",
    block.source_region_id,
    fallback=fallback,
  )


def _validate_descriptors(
  model: CompiledModel,
  block: DomainBlock,
  *,
  fallback: SourceContext,
) -> None:
  identities = (
    (Q8_TOPOLOGY_KEY, block.topology),
    (Q8_QUADRATURE_KEY, block.quadrature),
    (Q8_FORMULATION_KEY, block.formulation),
    (Q8_MATERIAL_KEY, block.material),
  )
  for key, identity in identities:
    source = _descriptor_source(model, block, key, fallback=fallback)
    try:
      descriptor = model.registry_snapshot.resolve(*key)
      expected_metadata = CanonicalManifest(q8_descriptor_metadata(*key))
      metadata_matches = descriptor.metadata.to_bytes() == expected_metadata.to_bytes()
    except (
      AttributeError,
      KeyError,
      OverflowError,
      RecursionError,
      TypeError,
      ValueError,
    ):
      _raise_boundary(
        "malformed-registry-descriptor",
        "the captured Q8 registry descriptor is malformed",
        source,
      )
    if not metadata_matches:
      _raise_boundary(
        "incompatible-registry-descriptor",
        "the captured descriptor does not carry the frozen Q8 metadata",
        source,
      )
    if (
      identity.kind != descriptor.kind
      or identity.name != descriptor.name
      or identity.version != descriptor.version
      or identity.implementation_id != descriptor.implementation_id
    ):
      _raise_boundary(
        "descriptor-identity-mismatch",
        "compiled block descriptor identity does not match the captured registry",
        source,
      )


def _validate_unique_model_storage(
  arrays: tuple[tuple[str, FinalizedArray], ...],
  *,
  source: SourceContext,
) -> None:
  for index, (left_label, left) in enumerate(arrays):
    for right_label, right in arrays[index + 1 :]:
      if (
        left is right
        or left.values is right.values
        or np.shares_memory(left.values, right.values)
      ):
        _raise_boundary(
          "malformed-compiled-model",
          f"compiled {left_label} and {right_label} require separate storage",
          source,
        )


def _validate_q8_recipe(
  model: CompiledModel,
  *,
  fallback: SourceContext,
) -> DomainBlock:
  block = model.domain_blocks[0]
  block_source = _model_source(
    model,
    "domain_block",
    block.block_id,
    fallback=fallback,
  )
  _validate_descriptors(model, block, fallback=fallback)

  node_count = len(model.mesh.node_ids)
  cell_count = len(block.cell_ids)
  full_count = model.dofs.global_size
  index_dtype = np.dtype(model.provenance.dense_index_dtype)
  coordinates = _validated_array_values(
    model.mesh.coordinates,
    dtype=_FLOATING_DTYPE,
    shape=(node_count, 2),
    label="compiled mesh coordinates",
    source=block_source,
    finite=True,
  )
  connectivity = _validated_array_values(
    block.connectivity,
    dtype=index_dtype,
    shape=(cell_count, 8),
    label="compiled Q8 connectivity",
    source=block_source,
  )
  dof_map = _validated_array_values(
    block.dof_map,
    dtype=index_dtype,
    shape=(cell_count, _LOCAL_DOF_COUNT),
    label="compiled Q8 DOF map",
    source=block_source,
  )
  points = _validated_array_values(
    block.quadrature_points,
    dtype=_FLOATING_DTYPE,
    shape=(9, 2),
    label="compiled Q8 quadrature points",
    source=block_source,
    finite=True,
  )
  weights = _validated_array_values(
    block.quadrature_weights,
    dtype=_FLOATING_DTYPE,
    shape=(9,),
    label="compiled Q8 quadrature weights",
    source=block_source,
    finite=True,
  )
  shape_values = _validated_array_values(
    block.shape_values,
    dtype=_FLOATING_DTYPE,
    shape=(9, 8),
    label="compiled Q8 shape values",
    source=block_source,
    finite=True,
  )
  parent_gradients = _validated_array_values(
    block.parent_gradients,
    dtype=_FLOATING_DTYPE,
    shape=(9, 8, 2),
    label="compiled Q8 parent gradients",
    source=block_source,
    finite=True,
  )
  parameters = _validated_array_values(
    block.material_parameters,
    dtype=_FLOATING_DTYPE,
    shape=(1, 2),
    label="compiled Q8 material parameters",
    source=block_source,
    finite=True,
  )

  cell_block = model.mesh.cell_blocks[0]
  recipes = model.assembly_topology.block_recipes
  if type(recipes) is not tuple or len(recipes) != 1:
    _raise_boundary(
      "incompatible-q8-recipe",
      "compiled model must contain exactly one Q8 coupling recipe",
      block_source,
    )
  recipe = recipes[0]
  if (
    cell_count <= 0
    or cell_block.cell_ids != block.cell_ids
    or cell_block.connectivity is not block.connectivity
    or recipe.block_index != 0
    or recipe.local_dof_count != _LOCAL_DOF_COUNT
    or recipe.coupling != "full-element-local-dof-clique"
    or recipe.dof_map is not block.dof_map
    or block.source_cell_block_id != cell_block.id
    or cell_block.reference_topology != "quadrilateral"
    or cell_block.topological_dimension != 2
    or cell_block.embedding_dimension != 2
    or cell_block.geometry_interpolation != "serendipity-quad8"
    or block.field_id != model.dofs.field_id
    or model.dofs.components != ("x", "y")
    or block.field_components != ("x", "y")
    or block.material_parameter_names != ("youngs_modulus", "poisson_ratio")
    or block.integration_layout.points_per_element != 9
    or block.integration_layout.material_slots_per_point != 1
    or block.integration_layout.local_point_ids != tuple(range(9))
    or block.kinematic_regime != "small-strain"
    or block.strain_voigt_order != ("xx", "yy", "xy")
    or block.shear_convention != "engineering"
    or block.measure_convention != "per-unit-out-of-plane-thickness"
  ):
    _raise_boundary(
      "incompatible-q8-recipe",
      "compiled model does not carry the exact frozen Q8 domain recipe",
      block_source,
    )
  if (
    model.capabilities.response_class != "linear-elastic"
    or not model.capabilities.fixed_model_coupling
    or model.capabilities.tangent_class != "symmetric-constant-material"
    or not model.capabilities.tangent_is_symmetric
    or not model.capabilities.tangent_is_constant
    or not model.capabilities.conservative_internal_contribution
    or model.capabilities.state_dependent
    or model.capabilities.has_storage
    or model.capabilities.has_mass
    or model.capabilities.has_damping
    or model.capabilities.restart_history_required
    or model.capabilities.contribution_channels
    != ("internal-force", "material-tangent")
    or model.physical_state_layout.global_primary_size != full_count
    or model.physical_state_layout.evolving_value_count != 0
  ):
    _raise_boundary(
      "unsupported-model-capabilities",
      "compiled model capabilities contradict constant linear Q8 assembly",
      block_source,
    )
  if connectivity.size and (
    int(connectivity.min()) < 0 or int(connectivity.max()) >= node_count
  ):
    _raise_boundary(
      "invalid-q8-connectivity",
      "compiled Q8 connectivity references an out-of-range dense node",
      block_source,
    )
  node_component_dofs = model.dofs.node_component_dofs.values
  for cell_index in range(cell_count):
    for local_node in range(8):
      node_index = int(connectivity[cell_index, local_node])
      for component in range(2):
        local_dof = local_node * 2 + component
        if int(dof_map[cell_index, local_dof]) != int(
          node_component_dofs[node_index, component]
        ):
          _raise_boundary(
            "invalid-q8-dof-map",
            "compiled Q8 DOF map does not match connectivity and field layout",
            block_source,
          )
  if dof_map.size and (int(dof_map.min()) < 0 or int(dof_map.max()) >= full_count):
    _raise_boundary(
      "invalid-q8-dof-map",
      "compiled Q8 DOF map references an out-of-range full DOF",
      block_source,
    )
  if (
    bool(np.any(np.absolute(points) > 1.0))
    or bool(np.any(weights <= 0.0))
    or not math.isclose(
      math.fsum(float(item) for item in weights),
      4.0,
      rel_tol=1.0e-14,
      abs_tol=1.0e-14,
    )
    or not bool(np.allclose(shape_values.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-12))
    or not bool(np.allclose(parent_gradients.sum(axis=1), 0.0, rtol=0.0, atol=1.0e-12))
  ):
    _raise_boundary(
      "invalid-q8-reference-recipe",
      "compiled quadrature or shape recipe violates the frozen Q8 checks",
      block_source,
    )
  youngs_modulus = float(parameters[0, 0])
  poisson_ratio = float(parameters[0, 1])
  if youngs_modulus <= 0.0 or not -1.0 < poisson_ratio < 0.5:
    _raise_boundary(
      "invalid-q8-material-parameters",
      "compiled Q8 material parameters violate the plane-stress domain",
      block_source,
    )
  primary_fields = model.physical_state_layout.primary_fields
  block_states = model.physical_state_layout.block_states
  if (
    type(primary_fields) is not tuple
    or len(primary_fields) != 1
    or primary_fields[0].field_id != model.dofs.field_id
    or primary_fields[0].components != ("x", "y")
    or primary_fields[0].global_size != full_count
    or type(block_states) is not tuple
    or len(block_states) != 1
    or block_states[0].block_index != 0
    or block_states[0].element_count != cell_count
    or block_states[0].integration_points_per_element != 9
    or block_states[0].material_slots_per_point != 1
    or block_states[0].material_history_width != 0
    or block_states[0].formulation_history_width != 0
  ):
    _raise_boundary(
      "unsupported-model-state-layout",
      "compiled model state layout contradicts the stateless linear Q8 slice",
      block_source,
    )
  _validate_unique_model_storage(
    (
      ("coordinates", model.mesh.coordinates),
      ("connectivity", block.connectivity),
      ("DOF map", block.dof_map),
      ("quadrature points", block.quadrature_points),
      ("quadrature weights", block.quadrature_weights),
      ("shape values", block.shape_values),
      ("parent gradients", block.parent_gradients),
      ("material parameters", block.material_parameters),
    ),
    source=block_source,
  )
  del coordinates
  return block


def _validate_program_capabilities(
  program: CompiledProgram,
  *,
  source: SourceContext,
) -> None:
  capabilities = program.capabilities
  if (
    not capabilities.fixed_constraint_topology
    or not capabilities.fixed_load_topology
    or capabilities.state_dependent
    or capabilities.has_follower_loads
    or capabilities.has_interaction_tangent
    or capabilities.has_program_state
  ):
    _raise_boundary(
      "unsupported-program-contribution",
      "compiled program requires an unsupported operator or state contribution",
      source,
    )


def _preflight_plan_counts(
  model: CompiledModel,
  program: CompiledProgram,
  block: DomainBlock,
  *,
  source: SourceContext,
) -> None:
  full_count = _checked_count(
    model.dofs.global_size,
    label="full DOF count",
    source=source,
  )
  reduced_count = _checked_count(
    program.constraint_plan.reduced_dof_count,
    label="reduced DOF count",
    source=source,
  )
  cell_count = _checked_count(
    len(block.cell_ids),
    label="Q8 cell count",
    source=source,
  )
  raw_count = _checked_product(
    cell_count,
    _RAW_ENTRIES_PER_CELL,
    label="raw Q8 operator entry count",
    source=source,
  )
  _checked_count(
    full_count + 1,
    label="terminal full-row offset count",
    source=source,
  )
  _checked_product(
    full_count,
    full_count,
    label="full operator shape product",
    source=source,
  )
  _checked_product(
    reduced_count,
    reduced_count,
    label="reduced operator shape product",
    source=source,
  )
  for dtype, label in (
    (_INDEX_DTYPE, "raw assembly topology"),
    (_FLOATING_DTYPE, "raw assembly values"),
  ):
    _check_allocation(raw_count, dtype, label=label, source=source)


def _build_vector_plan(
  program: CompiledProgram,
  *,
  source: SourceContext,
) -> tuple[NodalVectorContributionPlan, np.ndarray, np.ndarray]:
  constraint = program.constraint_plan
  full_count = _checked_count(
    constraint.full_dof_count,
    label="full DOF count",
    source=source,
  )
  reduced_count = _checked_count(
    constraint.reduced_dof_count,
    label="reduced DOF count",
    source=source,
  )
  _check_allocation(
    full_count,
    _INDEX_DTYPE,
    label="full-to-reduced index map",
    source=source,
  )
  _check_allocation(
    full_count,
    _FLOATING_DTYPE,
    label="full-to-reduced coefficient map",
    source=source,
  )
  full_to_reduced = np.full(full_count, -1, dtype=np.int64)
  full_coefficients = np.zeros(full_count, dtype=np.float64)
  retained_full: list[int] = []
  retained_reduced: list[int] = []
  retained_coefficients: list[float] = []
  row_offsets = constraint.row_offsets.values
  column_indices = constraint.column_indices.values
  coefficients = constraint.coefficients.values
  for full_dof in range(full_count):
    begin = int(row_offsets[full_dof])
    end = int(row_offsets[full_dof + 1])
    if end == begin:
      continue
    reduced_dof = int(column_indices[begin])
    coefficient = float(coefficients[begin])
    if (
      end - begin != 1
      or reduced_dof < 0
      or reduced_dof >= reduced_count
      or not math.isfinite(coefficient)
      or coefficient == 0.0
    ):
      _raise_boundary(
        "invalid-affine-reduction-map",
        "compiled affine reduction row violates the zero-or-one-entry policy",
        source,
      )
    full_to_reduced[full_dof] = reduced_dof
    full_coefficients[full_dof] = coefficient
    retained_full.append(full_dof)
    retained_reduced.append(reduced_dof)
    retained_coefficients.append(coefficient)
  retained_count = _checked_count(
    len(retained_full),
    label="retained full-to-reduced entry count",
    source=source,
  )
  for dtype, label in (
    (_INDEX_DTYPE, "retained full DOFs"),
    (_INDEX_DTYPE, "retained reduced DOFs"),
    (_FLOATING_DTYPE, "retained coefficients"),
  ):
    _check_allocation(retained_count, dtype, label=label, source=source)
  return (
    NodalVectorContributionPlan(
      full_dof_count=full_count,
      reduced_dof_count=reduced_count,
      full_dof_indices=FinalizedArray(retained_full, dtype=np.int64),
      reduced_dof_indices=FinalizedArray(retained_reduced, dtype=np.int64),
      coefficients=FinalizedArray(retained_coefficients, dtype=np.float64),
    ),
    full_to_reduced,
    full_coefficients,
  )


def _canonical_pairs(
  rows: np.ndarray,
  columns: np.ndarray,
  *,
  source: SourceContext,
  label: str,
) -> tuple[list[int], list[int], np.ndarray]:
  raw_count = len(rows)
  pairs = sorted(
    {(int(rows[index]), int(columns[index])) for index in range(raw_count)}
  )
  canonical_count = _checked_count(
    len(pairs),
    label=f"{label} canonical pair count",
    source=source,
  )
  _check_allocation(
    canonical_count,
    _INDEX_DTYPE,
    label=f"{label} canonical rows",
    source=source,
  )
  _check_allocation(
    raw_count,
    _INDEX_DTYPE,
    label=f"{label} raw-to-canonical map",
    source=source,
  )
  pair_indices = {pair: index for index, pair in enumerate(pairs)}
  raw_to_canonical = np.empty(raw_count, dtype=np.int64)
  for index in range(raw_count):
    raw_to_canonical[index] = pair_indices[(int(rows[index]), int(columns[index]))]
  return (
    [pair[0] for pair in pairs],
    [pair[1] for pair in pairs],
    raw_to_canonical,
  )


def _build_domain_coo_plan(
  model: CompiledModel,
  program: CompiledProgram,
  block: DomainBlock,
  full_to_reduced: np.ndarray,
  full_coefficients: np.ndarray,
  *,
  source: SourceContext,
) -> DomainCooPlan:
  cell_count = _checked_count(
    len(block.cell_ids),
    label="Q8 cell count",
    source=source,
  )
  raw_count = _checked_product(
    cell_count,
    _RAW_ENTRIES_PER_CELL,
    label="raw Q8 operator entry count",
    source=source,
  )
  for label in (
    "full raw rows",
    "full raw columns",
    "raw block attribution",
    "raw cell attribution",
    "raw local-row attribution",
    "raw local-column attribution",
  ):
    _check_allocation(raw_count, _INDEX_DTYPE, label=label, source=source)

  raw_rows = np.empty(raw_count, dtype=np.int64)
  raw_columns = np.empty(raw_count, dtype=np.int64)
  block_indices = np.zeros(raw_count, dtype=np.int64)
  cell_indices = np.empty(raw_count, dtype=np.int64)
  local_rows = np.empty(raw_count, dtype=np.int64)
  local_columns = np.empty(raw_count, dtype=np.int64)
  dof_map = block.dof_map.values
  raw_index = 0
  for cell_index in range(cell_count):
    for local_row in range(_LOCAL_DOF_COUNT):
      full_row = int(dof_map[cell_index, local_row])
      for local_column in range(_LOCAL_DOF_COUNT):
        raw_rows[raw_index] = full_row
        raw_columns[raw_index] = int(dof_map[cell_index, local_column])
        cell_indices[raw_index] = cell_index
        local_rows[raw_index] = local_row
        local_columns[raw_index] = local_column
        raw_index += 1

  full_rows, full_columns, full_raw_map = _canonical_pairs(
    raw_rows,
    raw_columns,
    source=source,
    label="full operator",
  )

  reduced_raw_count = 0
  for index in range(raw_count):
    full_row = int(raw_rows[index])
    full_column = int(raw_columns[index])
    if full_to_reduced[full_row] >= 0 and full_to_reduced[full_column] >= 0:
      reduced_raw_count += 1
  reduced_raw_count = _checked_count(
    reduced_raw_count,
    label="reduced raw operator entry count",
    source=source,
  )
  for dtype, label in (
    (_INDEX_DTYPE, "reduced raw source indices"),
    (_INDEX_DTYPE, "reduced raw rows"),
    (_INDEX_DTYPE, "reduced raw columns"),
    (_FLOATING_DTYPE, "reduced raw left factors"),
    (_FLOATING_DTYPE, "reduced raw right factors"),
  ):
    _check_allocation(reduced_raw_count, dtype, label=label, source=source)
  reduced_source_indices = np.empty(reduced_raw_count, dtype=np.int64)
  reduced_raw_rows = np.empty(reduced_raw_count, dtype=np.int64)
  reduced_raw_columns = np.empty(reduced_raw_count, dtype=np.int64)
  reduced_left = np.empty(reduced_raw_count, dtype=np.float64)
  reduced_right = np.empty(reduced_raw_count, dtype=np.float64)
  reduced_index = 0
  for index in range(raw_count):
    full_row = int(raw_rows[index])
    full_column = int(raw_columns[index])
    reduced_row = int(full_to_reduced[full_row])
    reduced_column = int(full_to_reduced[full_column])
    if reduced_row < 0 or reduced_column < 0:
      continue
    reduced_source_indices[reduced_index] = index
    reduced_raw_rows[reduced_index] = reduced_row
    reduced_raw_columns[reduced_index] = reduced_column
    reduced_left[reduced_index] = full_coefficients[full_row]
    reduced_right[reduced_index] = full_coefficients[full_column]
    reduced_index += 1

  reduced_rows, reduced_columns, reduced_raw_map = _canonical_pairs(
    reduced_raw_rows,
    reduced_raw_columns,
    source=source,
    label="reduced operator",
  )
  full_count = model.dofs.global_size
  reduced_count = program.constraint_plan.reduced_dof_count
  return DomainCooPlan(
    full_shape=(full_count, full_count),
    full_raw_row_indices=FinalizedArray(raw_rows, dtype=np.int64),
    full_raw_column_indices=FinalizedArray(raw_columns, dtype=np.int64),
    block_indices=FinalizedArray(block_indices, dtype=np.int64),
    cell_indices=FinalizedArray(cell_indices, dtype=np.int64),
    local_row_indices=FinalizedArray(local_rows, dtype=np.int64),
    local_column_indices=FinalizedArray(local_columns, dtype=np.int64),
    full_row_indices=FinalizedArray(full_rows, dtype=np.int64),
    full_column_indices=FinalizedArray(full_columns, dtype=np.int64),
    full_raw_to_canonical=FinalizedArray(full_raw_map, dtype=np.int64),
    reduced_shape=(reduced_count, reduced_count),
    reduced_raw_source_indices=FinalizedArray(
      reduced_source_indices,
      dtype=np.int64,
    ),
    reduced_raw_row_indices=FinalizedArray(reduced_raw_rows, dtype=np.int64),
    reduced_raw_column_indices=FinalizedArray(reduced_raw_columns, dtype=np.int64),
    reduced_raw_left_factors=FinalizedArray(reduced_left, dtype=np.float64),
    reduced_raw_right_factors=FinalizedArray(reduced_right, dtype=np.float64),
    reduced_row_indices=FinalizedArray(reduced_rows, dtype=np.int64),
    reduced_column_indices=FinalizedArray(reduced_columns, dtype=np.int64),
    reduced_raw_to_canonical=FinalizedArray(reduced_raw_map, dtype=np.int64),
  )


def _plan_manifest(
  *,
  model: CompiledModel,
  program: CompiledProgram,
  request: LinearStaticContributionRequest,
  domain: DomainCooPlan,
  vector: NodalVectorContributionPlan,
  program_operator_recipes: tuple[str, ...],
) -> CanonicalManifest:
  del request
  block = model.domain_blocks[0]
  return CanonicalManifest(
    {
      "schema": PREPARED_ASSEMBLY_PLAN_MANIFEST_SCHEMA,
      "request": "linear-static-contributions-v1",
      "model": {
        "content_fingerprint": str(model.content_fingerprint),
        "manifest": model.provenance.manifest,
        "registry_snapshot": model.registry_snapshot.manifest,
      },
      "program": {
        "content_fingerprint": str(program.content_fingerprint),
        "manifest": program.provenance.manifest,
      },
      "numeric_policy": {
        "floating_dtype": _FLOATING_DTYPE.str,
        "index_dtype": _INDEX_DTYPE.str,
        "reduction_policy": ASSEMBLY_REDUCTION_POLICY,
        "geometry_policy": ASSEMBLY_GEOMETRY_POLICY,
      },
      "domain_attribution": {
        "block_ids": (block.block_id,),
        "cell_ids": block.cell_ids,
      },
      "domain_coo": {
        "full_shape": domain.full_shape,
        "full_raw_rows": domain.full_raw_row_indices.values,
        "full_raw_columns": domain.full_raw_column_indices.values,
        "block_indices": domain.block_indices.values,
        "cell_indices": domain.cell_indices.values,
        "local_rows": domain.local_row_indices.values,
        "local_columns": domain.local_column_indices.values,
        "full_rows": domain.full_row_indices.values,
        "full_columns": domain.full_column_indices.values,
        "full_raw_to_canonical": domain.full_raw_to_canonical.values,
        "reduced_shape": domain.reduced_shape,
        "reduced_raw_source_indices": domain.reduced_raw_source_indices.values,
        "reduced_raw_rows": domain.reduced_raw_row_indices.values,
        "reduced_raw_columns": domain.reduced_raw_column_indices.values,
        "reduced_raw_left_factors": domain.reduced_raw_left_factors.values,
        "reduced_raw_right_factors": domain.reduced_raw_right_factors.values,
        "reduced_rows": domain.reduced_row_indices.values,
        "reduced_columns": domain.reduced_column_indices.values,
        "reduced_raw_to_canonical": domain.reduced_raw_to_canonical.values,
      },
      "nodal_vector": {
        "full_dof_count": vector.full_dof_count,
        "reduced_dof_count": vector.reduced_dof_count,
        "full_dof_indices": vector.full_dof_indices.values,
        "reduced_dof_indices": vector.reduced_dof_indices.values,
        "coefficients": vector.coefficients.values,
      },
      "program_operator_recipes": program_operator_recipes,
    }
  )


def _validate_plan_compatibility(
  value: object,
  model: CompiledModel,
  program: CompiledProgram,
  *,
  source: SourceContext = _EVALUATION_SOURCE,
) -> PreparedAssemblyPlan:
  if type(value) is not PreparedAssemblyPlan:
    _raise_boundary(
      "invalid-prepared-assembly-plan",
      "assembly evaluation requires an exact PreparedAssemblyPlan",
      source,
    )
  require_same_instance(
    model.instance_id,
    value.compatible_model_instance_id,
    context="assembly plan/model composition",
  )
  require_same_instance(
    program.instance_id,
    value.compatible_program_instance_id,
    context="assembly plan/program composition",
  )
  if (
    value.compatible_model_content_fingerprint != model.content_fingerprint
    or value.compatible_program_content_fingerprint != program.content_fingerprint
  ):
    _raise_boundary(
      "assembly-plan-content-mismatch",
      "prepared plan content compatibility does not match its model/program inputs",
      source,
    )
  if type(value.request) is not LinearStaticContributionRequest:
    _raise_boundary(
      "malformed-assembly-plan",
      "prepared plan request must be the exact zero-field linear marker",
      source,
    )
  recipes = value.program_operator_recipes
  if type(recipes) is not tuple or recipes:
    _raise_boundary(
      "unsupported-program-contribution",
      "prepared plan program-operator recipe set must be exactly empty",
      source,
    )
  _validate_program_capabilities(program, source=source)
  return value


def _prepare(
  model: object,
  program: object,
  request: object,
) -> PreparedAssemblyPlan:
  if type(request) is not LinearStaticContributionRequest:
    _raise_boundary(
      "invalid-assembly-request",
      "assembly request must be exactly LinearStaticContributionRequest",
      _PREPARATION_SOURCE,
    )
  if type(model) is not CompiledModel:
    _raise_boundary(
      "invalid-compiled-model",
      "assembly preparation requires an exact CompiledModel",
      _PREPARATION_SOURCE,
    )
  if type(program) is not CompiledProgram:
    _raise_boundary(
      "invalid-compiled-program",
      "assembly preparation requires an exact CompiledProgram",
      _PREPARATION_SOURCE,
    )
  validated_model = model
  validated_program = program
  program_source = _program_source(
    validated_program,
    fallback=_PREPARATION_SOURCE,
  )
  try:
    _validate_model_program_compatibility(
      validated_model,
      validated_program,
      source=program_source,
    )
  except IdentityMismatchError:
    _raise_boundary(
      "model-program-live-identity-mismatch",
      "assembly preparation requires exact live model compatibility",
      program_source,
    )
  block = _validate_q8_recipe(validated_model, fallback=_PREPARATION_SOURCE)
  _validate_program_capabilities(validated_program, source=program_source)
  _preflight_plan_counts(
    validated_model,
    validated_program,
    block,
    source=program_source,
  )
  vector_plan, full_to_reduced, full_coefficients = _build_vector_plan(
    validated_program,
    source=program_source,
  )
  domain_plan = _build_domain_coo_plan(
    validated_model,
    validated_program,
    block,
    full_to_reduced,
    full_coefficients,
    source=program_source,
  )
  recipes: tuple[str, ...] = ()
  manifest = _plan_manifest(
    model=validated_model,
    program=validated_program,
    request=request,
    domain=domain_plan,
    vector=vector_plan,
    program_operator_recipes=recipes,
  )
  return PreparedAssemblyPlan(
    content_fingerprint=ContentFingerprint.from_manifest(manifest),
    provenance=AssemblyPlanProvenance(
      schema=PREPARED_ASSEMBLY_PLAN_MANIFEST_SCHEMA,
      manifest=manifest,
      floating_dtype=_FLOATING_DTYPE.str,
      index_dtype=_INDEX_DTYPE.str,
      reduction_policy=ASSEMBLY_REDUCTION_POLICY,
      geometry_policy=ASSEMBLY_GEOMETRY_POLICY,
    ),
    compatible_model_instance_id=validated_model.instance_id,
    compatible_model_content_fingerprint=validated_model.content_fingerprint,
    compatible_program_instance_id=validated_program.instance_id,
    compatible_program_content_fingerprint=validated_program.content_fingerprint,
    request=request,
    domain_coo_plan=domain_plan,
    nodal_vector_plan=vector_plan,
    program_operator_recipes=recipes,
  )


def prepare_assembly_plan(
  model: CompiledModel,
  program: CompiledProgram,
  request: LinearStaticContributionRequest,
) -> PreparedAssemblyPlan:
  """Compose exact validated model/program meaning into immutable COO topology."""
  try:
    return _prepare(model, program, request)
  except AssemblyPreparationError:
    raise
  except IdentityMismatchError:
    raise AssemblyPreparationError(
      (
        AssemblyPreparationDiagnostic(
          code="model-program-live-identity-mismatch",
          message="assembly preparation requires exact live model compatibility",
          source=_PREPARATION_SOURCE,
        ),
      )
    ) from None
  except _AssemblyBoundaryError as error:
    raise AssemblyPreparationError(
      (
        AssemblyPreparationDiagnostic(
          code=error.code,
          message=error.message,
          source=error.source,
        ),
      )
    ) from None
  except Exception:
    raise AssemblyPreparationError(
      (
        AssemblyPreparationDiagnostic(
          code="assembly-preparation-failed",
          message="assembly preparation failed before a plan could escape",
          source=_PREPARATION_SOURCE,
        ),
      )
    ) from None
