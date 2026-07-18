"""Prepare immutable attributed COO topology for the frozen Q8 slice."""

from __future__ import annotations

import math
from typing import NoReturn
from uuid import UUID

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
from pyfem.v3.compile.program import (
  _BoundaryError,
  _validated_model,
  _validated_program,
)
from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.compiled import CompiledModel, CompiledSource, DomainBlock
from pyfem.v3.model.identity import (
  IdentityMismatchError,
  InstanceId,
  require_same_instance,
)
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


def _required_slots(
  value: object,
  names: tuple[str, ...],
  *,
  label: str,
  source: SourceContext,
) -> tuple[object, ...]:
  values = tuple(_read_slot(value, name) for name in names)
  if any(item is _MISSING for item in values):
    _raise_boundary(
      "malformed-assembly-carrier",
      f"{label} must initialize every canonical slot",
      source,
    )
  return values


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


def _validated_fingerprint(
  value: object,
  *,
  label: str,
  source: SourceContext,
) -> ContentFingerprint:
  if type(value) is not ContentFingerprint:
    _raise_boundary(
      "malformed-assembly-carrier",
      f"{label} must be an exact ContentFingerprint",
      source,
    )
  try:
    str(value)
  except (TypeError, ValueError):
    _raise_boundary(
      "malformed-assembly-carrier",
      f"{label} contains a malformed digest",
      source,
    )
  return value


def _validated_manifest(
  value: object,
  *,
  label: str,
  source: SourceContext,
) -> CanonicalManifest:
  if type(value) is not CanonicalManifest:
    _raise_boundary(
      "malformed-assembly-carrier",
      f"{label} must be an exact CanonicalManifest",
      source,
    )
  try:
    value.to_bytes()
  except (OverflowError, RecursionError, TypeError, ValueError):
    _raise_boundary(
      "malformed-assembly-carrier",
      f"{label} contains malformed canonical bytes",
      source,
    )
  return value


def _validated_instance_id(
  value: object,
  *,
  label: str,
  source: SourceContext,
) -> InstanceId:
  if type(value) is not InstanceId:
    _raise_boundary(
      "malformed-assembly-carrier",
      f"{label} must be an exact InstanceId",
      source,
    )
  token = _read_slot(value, "_token")
  if token is _MISSING or type(token) is not UUID:
    _raise_boundary(
      "malformed-assembly-carrier",
      f"{label} contains a malformed opaque token",
      source,
    )
  return value


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


def _validated_binding_array(
  value: object,
  *,
  shape: tuple[int, ...],
  code: str,
  label: str,
  source: SourceContext,
) -> np.ndarray:
  if (
    type(value) is not np.ndarray
    or value.ndim != len(shape)
    or value.shape != shape
    or value.dtype != _FLOATING_DTYPE
    or value.dtype.metadata is not None
  ):
    _raise_boundary(
      code,
      f"{label} must return one exact metadata-free float64 ndarray of shape {shape!r}",
      source,
    )
  if not bool(np.isfinite(value).all()):
    _raise_boundary(
      code,
      f"{label} must return only finite float64 values",
      source,
    )
  return value


def _binding_values_match_compiled(
  binding_values: np.ndarray,
  compiled_values: np.ndarray,
) -> bool:
  return binding_values.tobytes(order="C") == compiled_values.tobytes(order="C")


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


def _validate_q8_binding_correspondence(
  model: CompiledModel,
  block: DomainBlock,
  *,
  fallback: SourceContext,
) -> None:
  quadrature_source = _descriptor_source(
    model,
    block,
    Q8_QUADRATURE_KEY,
    fallback=fallback,
  )
  try:
    quadrature_binding = model.registry_snapshot.resolve(*Q8_QUADRATURE_KEY).binding
    quadrature_result = quadrature_binding(3)
  except Exception:
    _raise_boundary(
      "quadrature-binding-failed",
      "the captured gauss-3x3 binding failed during assembly preparation",
      quadrature_source,
    )
  if type(quadrature_result) is not tuple or len(quadrature_result) != 2:
    _raise_boundary(
      "invalid-quadrature-binding-output",
      "the captured gauss-3x3 binding must return exactly (points, weights)",
      quadrature_source,
    )
  points = _validated_binding_array(
    quadrature_result[0],
    shape=(9, 2),
    code="invalid-quadrature-binding-output",
    label="captured gauss-3x3 points",
    source=quadrature_source,
  )
  weights = _validated_binding_array(
    quadrature_result[1],
    shape=(9,),
    code="invalid-quadrature-binding-output",
    label="captured gauss-3x3 weights",
    source=quadrature_source,
  )
  if not _binding_values_match_compiled(
    points,
    block.quadrature_points.values,
  ) or not _binding_values_match_compiled(
    weights,
    block.quadrature_weights.values,
  ):
    _raise_boundary(
      "compiled-quadrature-recipe-mismatch",
      "compiled Q8 quadrature carriers do not exactly match the captured binding",
      quadrature_source,
    )

  topology_source = _descriptor_source(
    model,
    block,
    Q8_TOPOLOGY_KEY,
    fallback=fallback,
  )
  try:
    topology_binding = model.registry_snapshot.resolve(*Q8_TOPOLOGY_KEY).binding
    topology_result = topology_binding(
      np.array(points, dtype=np.float64, order="C", copy=True, subok=False)
    )
  except Exception:
    _raise_boundary(
      "topology-binding-failed",
      "the captured serendipity-quad8 binding failed during assembly preparation",
      topology_source,
    )
  if type(topology_result) is not tuple or len(topology_result) != 2:
    _raise_boundary(
      "invalid-topology-binding-output",
      "the captured serendipity-quad8 binding must return exactly (values, gradients)",
      topology_source,
    )
  shape_values = _validated_binding_array(
    topology_result[0],
    shape=(9, 8),
    code="invalid-topology-binding-output",
    label="captured serendipity-quad8 shape values",
    source=topology_source,
  )
  parent_gradients = _validated_binding_array(
    topology_result[1],
    shape=(9, 8, 2),
    code="invalid-topology-binding-output",
    label="captured serendipity-quad8 parent gradients",
    source=topology_source,
  )
  if not _binding_values_match_compiled(
    shape_values,
    block.shape_values.values,
  ) or not _binding_values_match_compiled(
    parent_gradients,
    block.parent_gradients.values,
  ):
    _raise_boundary(
      "compiled-topology-recipe-mismatch",
      "compiled Q8 shape carriers do not exactly match the captured binding",
      topology_source,
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


def _validate_shape(
  value: object,
  *,
  expected: tuple[int, int],
  label: str,
  source: SourceContext,
) -> tuple[int, int]:
  if (
    type(value) is not tuple
    or len(value) != 2
    or any(type(item) is not int or item < 0 for item in value)
    or value != expected
  ):
    _raise_boundary(
      "malformed-assembly-plan",
      f"{label} does not match the compatible vector-space shape",
      source,
    )
  _checked_product(value[0], value[1], label=f"{label} product", source=source)
  return value


def _validate_separate_plan_storage(
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
          "malformed-assembly-plan",
          f"{left_label} and {right_label} require separate finalized storage",
          source,
        )


def _validate_plan_storage_detached_from_inputs(
  arrays: tuple[tuple[str, FinalizedArray], ...],
  model: CompiledModel,
  program: CompiledProgram,
  block: DomainBlock,
  *,
  source: SourceContext,
) -> None:
  constraint = program.constraint_plan
  loads = program.nodal_load_plan
  input_arrays = (
    ("model coordinates", model.mesh.coordinates),
    ("model cell connectivity", model.mesh.cell_blocks[0].connectivity),
    ("model node DOFs", model.dofs.node_component_dofs),
    ("domain connectivity", block.connectivity),
    ("domain DOF map", block.dof_map),
    ("domain quadrature points", block.quadrature_points),
    ("domain quadrature weights", block.quadrature_weights),
    ("domain shape values", block.shape_values),
    ("domain parent gradients", block.parent_gradients),
    ("domain material parameters", block.material_parameters),
    ("model coupling DOF map", model.assembly_topology.block_recipes[0].dof_map),
    ("program free DOFs", constraint.free_dofs),
    ("program row offsets", constraint.row_offsets),
    ("program column indices", constraint.column_indices),
    ("program reduction coefficients", constraint.coefficients),
    ("program offset constants", constraint.offset_constant),
    (
      "program offset-coordinate coefficients",
      constraint.offset_coordinate_coefficients,
    ),
    ("program load DOFs", loads.dof_indices),
    ("program load constants", loads.constant_values),
    ("program load-coordinate coefficients", loads.coordinate_coefficients),
  )
  for plan_label, plan_array in arrays:
    for input_label, input_array in input_arrays:
      if (
        plan_array is input_array
        or plan_array.values is input_array.values
        or np.shares_memory(plan_array.values, input_array.values)
      ):
        _raise_boundary(
          "malformed-assembly-plan",
          f"prepared {plan_label} cannot reuse {input_label} storage",
          source,
        )


def _validate_vector_plan(
  value: object,
  program: CompiledProgram,
  *,
  source: SourceContext,
) -> tuple[NodalVectorContributionPlan, np.ndarray, np.ndarray]:
  if type(value) is not NodalVectorContributionPlan:
    _raise_boundary(
      "malformed-assembly-plan",
      "nodal vector plan must be an exact NodalVectorContributionPlan",
      source,
    )
  full_count, reduced_count, full_dofs, reduced_dofs, coefficients = _required_slots(
    value,
    (
      "full_dof_count",
      "reduced_dof_count",
      "full_dof_indices",
      "reduced_dof_indices",
      "coefficients",
    ),
    label="nodal vector plan",
    source=source,
  )
  if (
    type(full_count) is not int
    or type(reduced_count) is not int
    or full_count != program.constraint_plan.full_dof_count
    or reduced_count != program.constraint_plan.reduced_dof_count
  ):
    _raise_boundary(
      "malformed-assembly-plan",
      "nodal vector plan counts do not match the affine constraint plan",
      source,
    )
  row_offsets = program.constraint_plan.row_offsets.values
  retained_count = int(row_offsets[-1])
  _checked_count(
    full_count,
    label="nodal vector full DOF count",
    source=source,
  )
  _checked_count(
    retained_count,
    label="nodal vector retained entry count",
    source=source,
  )
  _check_allocation(
    full_count,
    _INDEX_DTYPE,
    label="validated full-to-reduced index map",
    source=source,
  )
  _check_allocation(
    full_count,
    _FLOATING_DTYPE,
    label="validated full-to-reduced coefficient map",
    source=source,
  )
  full_values = _validated_array_values(
    full_dofs,
    dtype=_INDEX_DTYPE,
    shape=(retained_count,),
    label="nodal vector full DOFs",
    source=source,
  )
  reduced_values = _validated_array_values(
    reduced_dofs,
    dtype=_INDEX_DTYPE,
    shape=(retained_count,),
    label="nodal vector reduced DOFs",
    source=source,
  )
  coefficient_values = _validated_array_values(
    coefficients,
    dtype=_FLOATING_DTYPE,
    shape=(retained_count,),
    label="nodal vector coefficients",
    source=source,
    finite=True,
  )
  full_to_reduced = np.full(full_count, -1, dtype=np.int64)
  full_coefficients = np.zeros(full_count, dtype=np.float64)
  position = 0
  constraint_columns = program.constraint_plan.column_indices.values
  constraint_coefficients = program.constraint_plan.coefficients.values
  for full_dof in range(full_count):
    begin = int(row_offsets[full_dof])
    end = int(row_offsets[full_dof + 1])
    if end == begin:
      continue
    expected_reduced = int(constraint_columns[begin])
    expected_coefficient = float(constraint_coefficients[begin])
    if (
      end - begin != 1
      or position >= retained_count
      or int(full_values[position]) != full_dof
      or int(reduced_values[position]) != expected_reduced
      or float(coefficient_values[position]) != expected_coefficient
      or expected_coefficient == 0.0
    ):
      _raise_boundary(
        "malformed-assembly-plan",
        "nodal vector plan does not exactly lower the affine constraint rows",
        source,
      )
    full_to_reduced[full_dof] = expected_reduced
    full_coefficients[full_dof] = expected_coefficient
    position += 1
  if position != retained_count:
    _raise_boundary(
      "malformed-assembly-plan",
      "nodal vector plan retains an unexpected full-space row",
      source,
    )
  return value, full_to_reduced, full_coefficients


def _validate_canonical_mapping(
  raw_rows: np.ndarray,
  raw_columns: np.ndarray,
  canonical_rows: np.ndarray,
  canonical_columns: np.ndarray,
  raw_to_canonical: np.ndarray,
  *,
  size: int,
  label: str,
  source: SourceContext,
) -> None:
  canonical_count = _checked_count(
    len(canonical_rows),
    label=f"{label} canonical entry count",
    source=source,
  )
  if len(canonical_columns) != canonical_count:
    _raise_boundary(
      "malformed-assembly-plan",
      f"{label} canonical row/column counts differ",
      source,
    )
  previous: tuple[int, int] | None = None
  for index in range(canonical_count):
    pair = int(canonical_rows[index]), int(canonical_columns[index])
    if (
      pair[0] < 0
      or pair[0] >= size
      or pair[1] < 0
      or pair[1] >= size
      or previous is not None
      and pair <= previous
    ):
      _raise_boundary(
        "malformed-assembly-plan",
        f"{label} canonical pairs are not unique lexicographic in-range entries",
        source,
      )
    previous = pair
  _check_allocation(
    canonical_count,
    np.dtype(np.bool_),
    label=f"{label} canonical coverage",
    source=source,
  )
  seen = np.zeros(canonical_count, dtype=np.bool_)
  for raw_index in range(len(raw_rows)):
    canonical_index = int(raw_to_canonical[raw_index])
    if (
      canonical_index < 0
      or canonical_index >= canonical_count
      or int(canonical_rows[canonical_index]) != int(raw_rows[raw_index])
      or int(canonical_columns[canonical_index]) != int(raw_columns[raw_index])
    ):
      _raise_boundary(
        "malformed-assembly-plan",
        f"{label} raw-to-canonical map contradicts its attributed pair",
        source,
      )
    seen[canonical_index] = True
  if canonical_count and not bool(seen.all()):
    _raise_boundary(
      "malformed-assembly-plan",
      f"{label} canonical topology contains an unreferenced pair",
      source,
    )


def _validate_domain_plan(
  value: object,
  model: CompiledModel,
  program: CompiledProgram,
  block: DomainBlock,
  full_to_reduced: np.ndarray,
  full_coefficients: np.ndarray,
  *,
  source: SourceContext,
) -> DomainCooPlan:
  if type(value) is not DomainCooPlan:
    _raise_boundary(
      "malformed-assembly-plan",
      "domain COO plan must be an exact DomainCooPlan",
      source,
    )
  names = (
    "full_shape",
    "full_raw_row_indices",
    "full_raw_column_indices",
    "block_indices",
    "cell_indices",
    "local_row_indices",
    "local_column_indices",
    "full_row_indices",
    "full_column_indices",
    "full_raw_to_canonical",
    "reduced_shape",
    "reduced_raw_source_indices",
    "reduced_raw_row_indices",
    "reduced_raw_column_indices",
    "reduced_raw_left_factors",
    "reduced_raw_right_factors",
    "reduced_row_indices",
    "reduced_column_indices",
    "reduced_raw_to_canonical",
  )
  slots = _required_slots(
    value,
    names,
    label="domain COO plan",
    source=source,
  )
  fields = dict(zip(names, slots, strict=True))
  full_count = model.dofs.global_size
  reduced_count = program.constraint_plan.reduced_dof_count
  _validate_shape(
    fields["full_shape"],
    expected=(full_count, full_count),
    label="full operator shape",
    source=source,
  )
  _validate_shape(
    fields["reduced_shape"],
    expected=(reduced_count, reduced_count),
    label="reduced operator shape",
    source=source,
  )
  raw_count = _checked_product(
    len(block.cell_ids),
    _RAW_ENTRIES_PER_CELL,
    label="raw Q8 operator entry count",
    source=source,
  )
  integer_raw_names = (
    "full_raw_row_indices",
    "full_raw_column_indices",
    "block_indices",
    "cell_indices",
    "local_row_indices",
    "local_column_indices",
    "full_raw_to_canonical",
  )
  raw_arrays = {
    name: _validated_array_values(
      fields[name],
      dtype=_INDEX_DTYPE,
      shape=(raw_count,),
      label=name.replace("_", " "),
      source=source,
    )
    for name in integer_raw_names
  }
  full_rows_carrier = fields["full_row_indices"]
  full_columns_carrier = fields["full_column_indices"]
  if type(full_rows_carrier) is not FinalizedArray:
    _raise_boundary(
      "malformed-assembly-plan",
      "full canonical rows must be an exact FinalizedArray",
      source,
    )
  full_row_values = _read_slot(full_rows_carrier, "values")
  if type(full_row_values) is not np.ndarray:
    _raise_boundary(
      "malformed-assembly-plan",
      "full canonical rows must own an exact plain ndarray",
      source,
    )
  full_canonical_count = _checked_count(
    len(full_row_values),
    label="full canonical entry count",
    source=source,
  )
  full_rows = _validated_array_values(
    full_rows_carrier,
    dtype=_INDEX_DTYPE,
    shape=(full_canonical_count,),
    label="full canonical rows",
    source=source,
  )
  full_columns = _validated_array_values(
    full_columns_carrier,
    dtype=_INDEX_DTYPE,
    shape=(full_canonical_count,),
    label="full canonical columns",
    source=source,
  )
  dof_map = block.dof_map.values
  for raw_index in range(raw_count):
    cell_index = raw_index // _RAW_ENTRIES_PER_CELL
    local_flat = raw_index % _RAW_ENTRIES_PER_CELL
    local_row = local_flat // _LOCAL_DOF_COUNT
    local_column = local_flat % _LOCAL_DOF_COUNT
    if (
      int(raw_arrays["block_indices"][raw_index]) != 0
      or int(raw_arrays["cell_indices"][raw_index]) != cell_index
      or int(raw_arrays["local_row_indices"][raw_index]) != local_row
      or int(raw_arrays["local_column_indices"][raw_index]) != local_column
      or int(raw_arrays["full_raw_row_indices"][raw_index])
      != int(dof_map[cell_index, local_row])
      or int(raw_arrays["full_raw_column_indices"][raw_index])
      != int(dof_map[cell_index, local_column])
    ):
      _raise_boundary(
        "malformed-assembly-plan",
        "full raw COO topology or attribution violates canonical Q8 order",
        source,
      )
  _validate_canonical_mapping(
    raw_arrays["full_raw_row_indices"],
    raw_arrays["full_raw_column_indices"],
    full_rows,
    full_columns,
    raw_arrays["full_raw_to_canonical"],
    size=full_count,
    label="full operator",
    source=source,
  )

  source_carrier = fields["reduced_raw_source_indices"]
  if type(source_carrier) is not FinalizedArray:
    _raise_boundary(
      "malformed-assembly-plan",
      "reduced raw source indices must be an exact FinalizedArray",
      source,
    )
  reduced_source_values = _read_slot(source_carrier, "values")
  if type(reduced_source_values) is not np.ndarray:
    _raise_boundary(
      "malformed-assembly-plan",
      "reduced raw source indices must own an exact plain ndarray",
      source,
    )
  reduced_raw_count = _checked_count(
    len(reduced_source_values),
    label="reduced raw entry count",
    source=source,
  )
  reduced_sources = _validated_array_values(
    source_carrier,
    dtype=_INDEX_DTYPE,
    shape=(reduced_raw_count,),
    label="reduced raw source indices",
    source=source,
  )
  reduced_raw_rows = _validated_array_values(
    fields["reduced_raw_row_indices"],
    dtype=_INDEX_DTYPE,
    shape=(reduced_raw_count,),
    label="reduced raw rows",
    source=source,
  )
  reduced_raw_columns = _validated_array_values(
    fields["reduced_raw_column_indices"],
    dtype=_INDEX_DTYPE,
    shape=(reduced_raw_count,),
    label="reduced raw columns",
    source=source,
  )
  reduced_left = _validated_array_values(
    fields["reduced_raw_left_factors"],
    dtype=_FLOATING_DTYPE,
    shape=(reduced_raw_count,),
    label="reduced raw left factors",
    source=source,
    finite=True,
  )
  reduced_right = _validated_array_values(
    fields["reduced_raw_right_factors"],
    dtype=_FLOATING_DTYPE,
    shape=(reduced_raw_count,),
    label="reduced raw right factors",
    source=source,
    finite=True,
  )
  expected_position = 0
  for raw_index in range(raw_count):
    full_row = int(raw_arrays["full_raw_row_indices"][raw_index])
    full_column = int(raw_arrays["full_raw_column_indices"][raw_index])
    reduced_row = int(full_to_reduced[full_row])
    reduced_column = int(full_to_reduced[full_column])
    if reduced_row < 0 or reduced_column < 0:
      continue
    if (
      expected_position >= reduced_raw_count
      or int(reduced_sources[expected_position]) != raw_index
      or int(reduced_raw_rows[expected_position]) != reduced_row
      or int(reduced_raw_columns[expected_position]) != reduced_column
      or float(reduced_left[expected_position]) != full_coefficients[full_row]
      or float(reduced_right[expected_position]) != full_coefficients[full_column]
    ):
      _raise_boundary(
        "malformed-assembly-plan",
        "reduced raw COO topology does not exactly lower full raw topology",
        source,
      )
    expected_position += 1
  if expected_position != reduced_raw_count:
    _raise_boundary(
      "malformed-assembly-plan",
      "reduced raw COO topology retains an unexpected source entry",
      source,
    )

  reduced_rows_carrier = fields["reduced_row_indices"]
  if type(reduced_rows_carrier) is not FinalizedArray:
    _raise_boundary(
      "malformed-assembly-plan",
      "reduced canonical rows must be an exact FinalizedArray",
      source,
    )
  reduced_row_values = _read_slot(reduced_rows_carrier, "values")
  if type(reduced_row_values) is not np.ndarray:
    _raise_boundary(
      "malformed-assembly-plan",
      "reduced canonical rows must own an exact plain ndarray",
      source,
    )
  reduced_canonical_count = _checked_count(
    len(reduced_row_values),
    label="reduced canonical entry count",
    source=source,
  )
  reduced_rows = _validated_array_values(
    reduced_rows_carrier,
    dtype=_INDEX_DTYPE,
    shape=(reduced_canonical_count,),
    label="reduced canonical rows",
    source=source,
  )
  reduced_columns = _validated_array_values(
    fields["reduced_column_indices"],
    dtype=_INDEX_DTYPE,
    shape=(reduced_canonical_count,),
    label="reduced canonical columns",
    source=source,
  )
  reduced_raw_map = _validated_array_values(
    fields["reduced_raw_to_canonical"],
    dtype=_INDEX_DTYPE,
    shape=(reduced_raw_count,),
    label="reduced raw-to-canonical map",
    source=source,
  )
  _validate_canonical_mapping(
    reduced_raw_rows,
    reduced_raw_columns,
    reduced_rows,
    reduced_columns,
    reduced_raw_map,
    size=reduced_count,
    label="reduced operator",
    source=source,
  )
  _validate_separate_plan_storage(
    tuple(
      (name.replace("_", " "), fields[name])
      for name in names
      if name not in ("full_shape", "reduced_shape")
    ),
    source=source,
  )
  return value


def _validated_prepared_plan(
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
  names = (
    "content_fingerprint",
    "provenance",
    "compatible_model_instance_id",
    "compatible_model_content_fingerprint",
    "compatible_program_instance_id",
    "compatible_program_content_fingerprint",
    "request",
    "domain_coo_plan",
    "nodal_vector_plan",
    "program_operator_recipes",
  )
  slots = _required_slots(
    value,
    names,
    label="prepared assembly plan",
    source=source,
  )
  fields = dict(zip(names, slots, strict=True))
  fingerprint = _validated_fingerprint(
    fields["content_fingerprint"],
    label="prepared-plan content fingerprint",
    source=source,
  )
  model_instance = _validated_instance_id(
    fields["compatible_model_instance_id"],
    label="prepared-plan model instance identity",
    source=source,
  )
  program_instance = _validated_instance_id(
    fields["compatible_program_instance_id"],
    label="prepared-plan program instance identity",
    source=source,
  )
  model_fingerprint = _validated_fingerprint(
    fields["compatible_model_content_fingerprint"],
    label="prepared-plan model content fingerprint",
    source=source,
  )
  program_fingerprint = _validated_fingerprint(
    fields["compatible_program_content_fingerprint"],
    label="prepared-plan program content fingerprint",
    source=source,
  )
  require_same_instance(
    model.instance_id,
    model_instance,
    context="assembly plan/model composition",
  )
  require_same_instance(
    program.instance_id,
    program_instance,
    context="assembly plan/program composition",
  )
  if (
    model_fingerprint != model.content_fingerprint
    or program_fingerprint != program.content_fingerprint
  ):
    _raise_boundary(
      "assembly-plan-content-mismatch",
      "prepared plan content compatibility does not match its model/program inputs",
      source,
    )
  request = fields["request"]
  if type(request) is not LinearStaticContributionRequest:
    _raise_boundary(
      "malformed-assembly-plan",
      "prepared plan request must be the exact zero-field linear marker",
      source,
    )
  recipes = fields["program_operator_recipes"]
  if type(recipes) is not tuple or recipes:
    _raise_boundary(
      "unsupported-program-contribution",
      "prepared plan program-operator recipe set must be exactly empty",
      source,
    )
  block = _validate_q8_recipe(model, fallback=source)
  _validate_program_capabilities(program, source=source)
  _preflight_plan_counts(model, program, block, source=source)
  vector, full_to_reduced, full_coefficients = _validate_vector_plan(
    fields["nodal_vector_plan"],
    program,
    source=source,
  )
  domain = _validate_domain_plan(
    fields["domain_coo_plan"],
    model,
    program,
    block,
    full_to_reduced,
    full_coefficients,
    source=source,
  )
  plan_arrays = (
    ("vector full DOFs", vector.full_dof_indices),
    ("vector reduced DOFs", vector.reduced_dof_indices),
    ("vector coefficients", vector.coefficients),
    ("domain full raw rows", domain.full_raw_row_indices),
    ("domain full raw columns", domain.full_raw_column_indices),
    ("domain block attribution", domain.block_indices),
    ("domain cell attribution", domain.cell_indices),
    ("domain local-row attribution", domain.local_row_indices),
    ("domain local-column attribution", domain.local_column_indices),
    ("domain full canonical rows", domain.full_row_indices),
    ("domain full canonical columns", domain.full_column_indices),
    ("domain full raw map", domain.full_raw_to_canonical),
    ("domain reduced raw sources", domain.reduced_raw_source_indices),
    ("domain reduced raw rows", domain.reduced_raw_row_indices),
    ("domain reduced raw columns", domain.reduced_raw_column_indices),
    ("domain reduced left factors", domain.reduced_raw_left_factors),
    ("domain reduced right factors", domain.reduced_raw_right_factors),
    ("domain reduced canonical rows", domain.reduced_row_indices),
    ("domain reduced canonical columns", domain.reduced_column_indices),
    ("domain reduced raw map", domain.reduced_raw_to_canonical),
  )
  _validate_separate_plan_storage(plan_arrays, source=source)
  _validate_plan_storage_detached_from_inputs(
    plan_arrays,
    model,
    program,
    block,
    source=source,
  )
  provenance = fields["provenance"]
  if type(provenance) is not AssemblyPlanProvenance:
    _raise_boundary(
      "malformed-assembly-plan",
      "prepared plan provenance must be an exact AssemblyPlanProvenance",
      source,
    )
  (
    schema,
    manifest,
    floating_dtype,
    index_dtype,
    reduction_policy,
    geometry_policy,
  ) = _required_slots(
    provenance,
    (
      "schema",
      "manifest",
      "floating_dtype",
      "index_dtype",
      "reduction_policy",
      "geometry_policy",
    ),
    label="prepared plan provenance",
    source=source,
  )
  if (
    type(schema) is not str
    or schema != PREPARED_ASSEMBLY_PLAN_MANIFEST_SCHEMA
    or type(floating_dtype) is not str
    or floating_dtype != _FLOATING_DTYPE.str
    or type(index_dtype) is not str
    or index_dtype != _INDEX_DTYPE.str
    or type(reduction_policy) is not str
    or reduction_policy != ASSEMBLY_REDUCTION_POLICY
    or type(geometry_policy) is not str
    or geometry_policy != ASSEMBLY_GEOMETRY_POLICY
  ):
    _raise_boundary(
      "malformed-assembly-plan",
      "prepared plan provenance violates the frozen assembly policies",
      source,
    )
  validated_manifest = _validated_manifest(
    manifest,
    label="prepared-plan manifest",
    source=source,
  )
  expected_manifest = _plan_manifest(
    model=model,
    program=program,
    request=request,
    domain=domain,
    vector=vector,
    program_operator_recipes=recipes,
  )
  try:
    manifest_matches = validated_manifest.to_bytes() == expected_manifest.to_bytes()
    fingerprint_matches = fingerprint == ContentFingerprint.from_manifest(
      expected_manifest
    )
  except (OverflowError, RecursionError, TypeError, ValueError):
    _raise_boundary(
      "malformed-assembly-plan",
      "prepared plan cannot form its canonical content identity",
      source,
    )
  if not manifest_matches or not fingerprint_matches:
    _raise_boundary(
      "malformed-assembly-plan",
      "prepared plan fields do not match their canonical content identity",
      source,
    )
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
  validated_model = _validated_model(model)
  validated_program = _validated_program(program)
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
  _validate_q8_binding_correspondence(
    validated_model,
    block,
    fallback=_PREPARATION_SOURCE,
  )
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
  plan = PreparedAssemblyPlan(
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
  return _validated_prepared_plan(
    plan,
    validated_model,
    validated_program,
    source=_PREPARATION_SOURCE,
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
  except _BoundaryError as error:
    raise AssemblyPreparationError(
      (
        AssemblyPreparationDiagnostic(
          code=error.code,
          message=error.message,
          source=_PREPARATION_SOURCE,
        ),
      )
    ) from None
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
