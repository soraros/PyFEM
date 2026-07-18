"""Correctness-first reference evaluation of linear Q8 contributions."""

from __future__ import annotations

import math
from typing import NoReturn

import numpy as np

from pyfem.v3.assembly.contracts import (
  CanonicalCooOperator,
  LinearStaticContributions,
  PreparedAssemblyPlan,
)
from pyfem.v3.assembly.diagnostics import (
  AssemblyEvaluationDiagnostic,
  AssemblyEvaluationError,
)
from pyfem.v3.assembly.prepare import (
  _EVALUATION_SOURCE,
  _FLOATING_DTYPE,
  _LOCAL_DOF_COUNT,
  _AssemblyBoundaryError,
  _check_allocation,
  _checked_count,
  _checked_product,
  _model_source,
  _program_source,
  _raise_boundary,
  _validate_model_program_compatibility,
  _validated_array_values,
  _validated_fingerprint,
  _validated_instance_id,
  _validated_prepared_plan,
)
from pyfem.v3.compile.contracts import Q8_FORMULATION_KEY, Q8_MATERIAL_KEY
from pyfem.v3.compile.program import (
  _BoundaryError,
  _validated_model,
  _validated_program,
  evaluate_program,
)
from pyfem.v3.compile.program_diagnostics import (
  ProgramEvaluationDiagnostic,
  ProgramEvaluationError,
)
from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.compiled import CompiledModel, DomainBlock
from pyfem.v3.model.identity import IdentityMismatchError, require_same_instance
from pyfem.v3.model.program import CompiledProgram, ProgramEvaluation
from pyfem.v3.spec.diagnostics import SourceContext
from pyfem.v3.spec.program import ProgramPoint

_QUADRATURE_POINT_COUNT = 9


def _evaluation_boundary(
  code: str,
  message: str,
  source: SourceContext,
) -> NoReturn:
  _raise_boundary(code, message, source)


def _validate_binding_array(
  value: object,
  *,
  shape: tuple[int, ...],
  label: str,
  source: SourceContext,
) -> np.ndarray:
  if (
    type(value) is not np.ndarray
    or value.dtype != _FLOATING_DTYPE
    or value.dtype.metadata is not None
    or value.shape != shape
    or not bool(np.isfinite(value).all())
  ):
    _evaluation_boundary(
      "invalid-assembly-binding-output",
      f"{label} must return one exact finite metadata-free float64 ndarray "
      "of the frozen shape",
      source,
    )
  return value


def _safe_product(
  left: float,
  right: float,
  *,
  label: str,
  source: SourceContext,
) -> float:
  result = left * right
  if not math.isfinite(result):
    _evaluation_boundary(
      "nonfinite-assembly-evaluation",
      f"{label} produced a nonfinite binary64 product",
      source,
    )
  return result


def _safe_difference(
  left: float,
  right: float,
  *,
  label: str,
  source: SourceContext,
) -> float:
  result = left - right
  if not math.isfinite(result):
    _evaluation_boundary(
      "nonfinite-assembly-evaluation",
      f"{label} produced a nonfinite binary64 difference",
      source,
    )
  return result


def _safe_fsum(
  values: list[float] | tuple[float, ...],
  *,
  label: str,
  source: SourceContext,
) -> float:
  try:
    result = math.fsum(values)
  except OverflowError:
    _evaluation_boundary(
      "nonfinite-assembly-evaluation",
      f"{label} overflowed deterministic binary64 accumulation",
      source,
    )
  if not math.isfinite(result):
    _evaluation_boundary(
      "nonfinite-assembly-evaluation",
      f"{label} produced a nonfinite deterministic sum",
      source,
    )
  return result


def _normalized_cell_coordinates(
  coordinates: np.ndarray,
  *,
  source: SourceContext,
) -> np.ndarray:
  with np.errstate(over="ignore", invalid="ignore", under="ignore"):
    relative = coordinates - coordinates[0]
  if not bool(np.isfinite(relative).all()):
    coordinate_scale = float(np.max(np.absolute(coordinates)))
    if not math.isfinite(coordinate_scale) or coordinate_scale == 0.0:
      _evaluation_boundary(
        "nonfinite-reference-geometry",
        "Q8 cell has a nonfinite or zero-scale reference mapping",
        source,
      )
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
      scaled = coordinates / coordinate_scale
      relative = scaled - scaled[0]
  cell_scale = float(np.max(np.absolute(relative)))
  if not math.isfinite(cell_scale) or cell_scale == 0.0:
    _evaluation_boundary(
      "nonfinite-reference-geometry",
      "Q8 cell has a nonfinite or zero-scale reference mapping",
      source,
    )
  with np.errstate(over="ignore", invalid="ignore", under="ignore"):
    normalized = relative / cell_scale
  if not bool(np.isfinite(normalized).all()):
    _evaluation_boundary(
      "nonfinite-reference-geometry",
      "Q8 cell has a nonfinite normalized reference mapping",
      source,
    )
  return normalized


def _normalized_gradients_and_determinants(
  normalized_coordinates: np.ndarray,
  parent_gradients: np.ndarray,
  *,
  tolerance: float,
  source: SourceContext,
) -> tuple[np.ndarray, np.ndarray]:
  gradients = np.empty((_QUADRATURE_POINT_COUNT, 8, 2), dtype=np.float64)
  determinants = np.empty(_QUADRATURE_POINT_COUNT, dtype=np.float64)
  norm_squares = np.empty(_QUADRATURE_POINT_COUNT, dtype=np.float64)
  for point in range(_QUADRATURE_POINT_COUNT):
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
      jacobian = normalized_coordinates.T @ parent_gradients[point]
      determinant = float(
        jacobian[0, 0] * jacobian[1, 1] - jacobian[0, 1] * jacobian[1, 0]
      )
      norm_square = math.fsum(
        float(jacobian[row, column] * jacobian[row, column])
        for row in range(2)
        for column in range(2)
      )
    if (
      not math.isfinite(determinant)
      or not math.isfinite(norm_square)
      or norm_square == 0.0
    ):
      _evaluation_boundary(
        "nonfinite-reference-geometry",
        "Q8 cell has a nonfinite or zero-norm normalized Jacobian",
        source,
      )
    determinants[point] = determinant
    norm_squares[point] = norm_square

  has_positive = bool(np.any(determinants > 0.0))
  has_negative = bool(np.any(determinants < 0.0))
  if has_positive and has_negative:
    _evaluation_boundary(
      "sign-changing-reference-geometry",
      "Q8 cell changes Jacobian orientation across quadrature points",
      source,
    )
  if has_negative:
    _evaluation_boundary(
      "inverted-reference-geometry",
      "Q8 cell has negative reference orientation",
      source,
    )
  for point in range(_QUADRATURE_POINT_COUNT):
    determinant = float(determinants[point])
    condition_ratio = determinant / float(norm_squares[point])
    if determinant <= tolerance or condition_ratio <= tolerance:
      _evaluation_boundary(
        "near-singular-reference-geometry",
        "Q8 cell has a scale-relative near-singular reference Jacobian",
        source,
      )
    jacobian = normalized_coordinates.T @ parent_gradients[point]
    inverse = np.array(
      (
        (jacobian[1, 1], -jacobian[0, 1]),
        (-jacobian[1, 0], jacobian[0, 0]),
      ),
      dtype=np.float64,
    )
    inverse /= determinant
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
      gradients[point] = parent_gradients[point] @ inverse
    if not bool(np.isfinite(gradients[point]).all()):
      _evaluation_boundary(
        "nonfinite-reference-geometry",
        "Q8 cell normalized physical gradients are nonfinite",
        source,
      )
  return gradients, determinants


def _material_tangent(
  model: CompiledModel,
  block: DomainBlock,
  *,
  source: SourceContext,
) -> np.ndarray:
  try:
    binding = model.registry_snapshot.resolve(*Q8_MATERIAL_KEY).binding
    parameters = block.material_parameters.values[0]
    value = binding(float(parameters[0]), float(parameters[1]))
  except Exception:
    _evaluation_boundary(
      "material-binding-failed",
      "captured plane-stress material binding failed for compiled parameters",
      source,
    )
  tangent = _validate_binding_array(
    value,
    shape=(3, 3),
    label="plane-stress material binding",
    source=source,
  )
  if not bool(np.array_equal(tangent, tangent.T)):
    _evaluation_boundary(
      "nonsymmetric-material-binding",
      "captured material binding contradicts its exact symmetric tangent meaning",
      source,
    )
  return tangent


def _element_operator(
  model: CompiledModel,
  block: DomainBlock,
  cell_index: int,
  material_tangent: np.ndarray,
  *,
  source: SourceContext,
  formulation_source: SourceContext,
) -> np.ndarray:
  connectivity = block.connectivity.values[cell_index]
  coordinates = np.array(
    model.mesh.coordinates.values[connectivity],
    dtype=np.float64,
    order="C",
    copy=True,
  )
  normalized = _normalized_cell_coordinates(coordinates, source=source)
  gradients, determinants = _normalized_gradients_and_determinants(
    normalized,
    block.parent_gradients.values,
    tolerance=model.provenance.geometry_relative_tolerance,
    source=source,
  )
  try:
    formulation = model.registry_snapshot.resolve(*Q8_FORMULATION_KEY).binding
    raw_kinematics = formulation(gradients.reshape(1, 9, 8, 2))
  except Exception:
    _evaluation_boundary(
      "formulation-binding-failed",
      "captured small-strain formulation binding failed for normalized gradients",
      formulation_source,
    )
  kinematics = _validate_binding_array(
    raw_kinematics,
    shape=(1, 9, 3, _LOCAL_DOF_COUNT),
    label="small-strain formulation binding",
    source=formulation_source,
  )[0]
  point_operators = np.empty(
    (_QUADRATURE_POINT_COUNT, _LOCAL_DOF_COUNT, _LOCAL_DOF_COUNT),
    dtype=np.float64,
  )
  for point in range(_QUADRATURE_POINT_COUNT):
    b_matrix = kinematics[point]
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
      unweighted = b_matrix.T @ material_tangent @ b_matrix
      weighted = (float(block.quadrature_weights.values[point]) * unweighted) * float(
        determinants[point]
      )
    if not bool(np.isfinite(weighted).all()):
      _evaluation_boundary(
        "nonfinite-assembly-integration",
        "Q8 material-tangent integration produced a nonfinite point operator",
        source,
      )
    operator_scale = float(np.max(np.absolute(weighted)))
    symmetry_tolerance = 64.0 * float(np.finfo(np.float64).eps) * operator_scale
    if not bool(
      np.allclose(
        weighted,
        weighted.T,
        rtol=0.0,
        atol=symmetry_tolerance,
      )
    ):
      _evaluation_boundary(
        "nonsymmetric-element-operator",
        "computed Q8 operator contradicts the declared symmetric tangent meaning",
        source,
      )
    point_operators[point] = weighted

  element = np.empty((_LOCAL_DOF_COUNT, _LOCAL_DOF_COUNT), dtype=np.float64)
  for row in range(_LOCAL_DOF_COUNT):
    for column in range(_LOCAL_DOF_COUNT):
      element[row, column] = _safe_fsum(
        [
          float(point_operators[point, row, column])
          for point in range(_QUADRATURE_POINT_COUNT)
        ],
        label="Q8 point integration",
        source=source,
      )
  return element


def _full_raw_operator_values(
  model: CompiledModel,
  block: DomainBlock,
  *,
  source: SourceContext,
) -> np.ndarray:
  cell_count = _checked_count(
    len(block.cell_ids),
    label="Q8 cell count",
    source=source,
  )
  raw_count = _checked_product(
    cell_count,
    _LOCAL_DOF_COUNT * _LOCAL_DOF_COUNT,
    label="raw Q8 operator value count",
    source=source,
  )
  _check_allocation(
    raw_count,
    _FLOATING_DTYPE,
    label="raw Q8 operator values",
    source=source,
  )
  raw_values = np.empty(raw_count, dtype=np.float64)
  material_source = _model_source(
    model,
    "material",
    block.source_material_id,
    fallback=source,
  )
  material_tangent = _material_tangent(
    model,
    block,
    source=material_source,
  )
  formulation_source = _model_source(
    model,
    "region",
    block.source_region_id,
    fallback=source,
  )
  for cell_index, cell_id in enumerate(block.cell_ids):
    cell_source = _model_source(
      model,
      "cell",
      (block.source_cell_block_id, cell_id),
      fallback=source,
    )
    element = _element_operator(
      model,
      block,
      cell_index,
      material_tangent,
      source=cell_source,
      formulation_source=formulation_source,
    )
    begin = cell_index * _LOCAL_DOF_COUNT * _LOCAL_DOF_COUNT
    end = begin + _LOCAL_DOF_COUNT * _LOCAL_DOF_COUNT
    raw_values[begin:end] = element.reshape(-1)
  return raw_values


def _coalesce(
  raw_values: np.ndarray,
  raw_to_canonical: np.ndarray,
  canonical_count: int,
  *,
  label: str,
  source: SourceContext,
) -> np.ndarray:
  canonical_count = _checked_count(
    canonical_count,
    label=f"{label} canonical value count",
    source=source,
  )
  _check_allocation(
    canonical_count,
    _FLOATING_DTYPE,
    label=f"{label} canonical values",
    source=source,
  )
  groups: list[list[float]] = [[] for _ in range(canonical_count)]
  for raw_index in range(len(raw_values)):
    groups[int(raw_to_canonical[raw_index])].append(float(raw_values[raw_index]))
  canonical = np.empty(canonical_count, dtype=np.float64)
  for index, group in enumerate(groups):
    canonical[index] = _safe_fsum(group, label=label, source=source)
  return canonical


def _reduced_raw_values(
  full_raw_values: np.ndarray,
  plan: PreparedAssemblyPlan,
  *,
  source: SourceContext,
) -> np.ndarray:
  domain = plan.domain_coo_plan
  source_indices = domain.reduced_raw_source_indices.values
  left_factors = domain.reduced_raw_left_factors.values
  right_factors = domain.reduced_raw_right_factors.values
  reduced_count = _checked_count(
    len(source_indices),
    label="reduced raw operator value count",
    source=source,
  )
  _check_allocation(
    reduced_count,
    _FLOATING_DTYPE,
    label="reduced raw operator values",
    source=source,
  )
  reduced = np.empty(reduced_count, dtype=np.float64)
  for index in range(len(source_indices)):
    first = _safe_product(
      float(left_factors[index]),
      float(full_raw_values[int(source_indices[index])]),
      label="left affine operator reduction",
      source=source,
    )
    reduced[index] = _safe_product(
      first,
      float(right_factors[index]),
      label="right affine operator reduction",
      source=source,
    )
  return reduced


def _raw_matvec(
  raw_rows: np.ndarray,
  raw_columns: np.ndarray,
  raw_values: np.ndarray,
  vector: np.ndarray,
  output_count: int,
  *,
  label: str,
  source: SourceContext,
) -> np.ndarray:
  output_count = _checked_count(
    output_count,
    label=f"{label} output count",
    source=source,
  )
  _check_allocation(
    output_count,
    _FLOATING_DTYPE,
    label=f"{label} output",
    source=source,
  )
  groups: list[list[float]] = [[] for _ in range(output_count)]
  for raw_index in range(len(raw_values)):
    row = int(raw_rows[raw_index])
    column = int(raw_columns[raw_index])
    groups[row].append(
      _safe_product(
        float(raw_values[raw_index]),
        float(vector[column]),
        label=label,
        source=source,
      )
    )
  result = np.empty(output_count, dtype=np.float64)
  for index, group in enumerate(groups):
    result[index] = _safe_fsum(group, label=label, source=source)
  return result


def _raw_matmat(
  raw_rows: np.ndarray,
  raw_columns: np.ndarray,
  raw_values: np.ndarray,
  values: np.ndarray,
  output_count: int,
  *,
  label: str,
  source: SourceContext,
) -> np.ndarray:
  coordinate_count = values.shape[1]
  output_size = _checked_product(
    output_count,
    coordinate_count,
    label=f"{label} output count",
    source=source,
  )
  _check_allocation(
    output_size,
    _FLOATING_DTYPE,
    label=f"{label} output",
    source=source,
  )
  result = np.empty((output_count, coordinate_count), dtype=np.float64)
  for coordinate in range(coordinate_count):
    result[:, coordinate] = _raw_matvec(
      raw_rows,
      raw_columns,
      raw_values,
      values[:, coordinate],
      output_count,
      label=label,
      source=source,
    )
  return result


def _reduce_vector(
  full_values: np.ndarray,
  plan: PreparedAssemblyPlan,
  *,
  label: str,
  source: SourceContext,
) -> np.ndarray:
  vector_plan = plan.nodal_vector_plan
  reduced_count = _checked_count(
    vector_plan.reduced_dof_count,
    label=f"{label} reduced output count",
    source=source,
  )
  _check_allocation(
    reduced_count,
    _FLOATING_DTYPE,
    label=f"{label} output",
    source=source,
  )
  groups: list[list[float]] = [[] for _ in range(reduced_count)]
  full_dofs = vector_plan.full_dof_indices.values
  reduced_dofs = vector_plan.reduced_dof_indices.values
  coefficients = vector_plan.coefficients.values
  for index in range(len(full_dofs)):
    reduced = int(reduced_dofs[index])
    groups[reduced].append(
      _safe_product(
        float(coefficients[index]),
        float(full_values[int(full_dofs[index])]),
        label=label,
        source=source,
      )
    )
  result = np.empty(reduced_count, dtype=np.float64)
  for index, group in enumerate(groups):
    result[index] = _safe_fsum(group, label=label, source=source)
  return result


def _reduce_matrix(
  full_values: np.ndarray,
  plan: PreparedAssemblyPlan,
  *,
  label: str,
  source: SourceContext,
) -> np.ndarray:
  coordinate_count = full_values.shape[1]
  reduced_count = plan.nodal_vector_plan.reduced_dof_count
  output_size = _checked_product(
    reduced_count,
    coordinate_count,
    label=f"{label} output count",
    source=source,
  )
  _check_allocation(
    output_size,
    _FLOATING_DTYPE,
    label=f"{label} output",
    source=source,
  )
  result = np.empty((reduced_count, coordinate_count), dtype=np.float64)
  for coordinate in range(coordinate_count):
    result[:, coordinate] = _reduce_vector(
      full_values[:, coordinate],
      plan,
      label=label,
      source=source,
    )
  return result


def _difference_vector(
  left: np.ndarray,
  right: np.ndarray,
  *,
  label: str,
  source: SourceContext,
) -> np.ndarray:
  output_size = _checked_product(
    left.shape[0],
    1,
    label=f"{label} output count",
    source=source,
  )
  _check_allocation(
    output_size,
    _FLOATING_DTYPE,
    label=f"{label} output",
    source=source,
  )
  result = np.empty(left.shape, dtype=np.float64)
  for index in range(len(left)):
    result[index] = _safe_difference(
      float(left[index]),
      float(right[index]),
      label=label,
      source=source,
    )
  return result


def _difference_matrix(
  left: np.ndarray,
  right: np.ndarray,
  *,
  label: str,
  source: SourceContext,
) -> np.ndarray:
  output_size = _checked_product(
    left.shape[0],
    left.shape[1],
    label=f"{label} output count",
    source=source,
  )
  _check_allocation(
    output_size,
    _FLOATING_DTYPE,
    label=f"{label} output",
    source=source,
  )
  result = np.empty(left.shape, dtype=np.float64)
  for row in range(left.shape[0]):
    for column in range(left.shape[1]):
      result[row, column] = _safe_difference(
        float(left[row, column]),
        float(right[row, column]),
        label=label,
        source=source,
      )
  return result


def _validate_program_evaluation(
  value: object,
  model: CompiledModel,
  program: CompiledProgram,
  *,
  source: SourceContext,
) -> ProgramEvaluation:
  if type(value) is not ProgramEvaluation:
    _evaluation_boundary(
      "malformed-program-evaluation",
      "P1-A evaluation must return an exact ProgramEvaluation",
      source,
    )
  names = (
    "program_instance_id",
    "program_content_fingerprint",
    "compatible_model_instance_id",
    "compatible_model_content_fingerprint",
    "coordinate_names",
    "coordinate_values",
    "prescribed_offsets",
    "prescribed_offset_derivatives",
    "nodal_force",
    "nodal_force_derivatives",
  )
  try:
    slots = tuple(object.__getattribute__(value, name) for name in names)
  except AttributeError:
    _evaluation_boundary(
      "malformed-program-evaluation",
      "P1-A evaluation must initialize every canonical slot",
      source,
    )
  fields = dict(zip(names, slots, strict=True))
  program_instance = _validated_instance_id(
    fields["program_instance_id"],
    label="program-evaluation program instance identity",
    source=source,
  )
  model_instance = _validated_instance_id(
    fields["compatible_model_instance_id"],
    label="program-evaluation model instance identity",
    source=source,
  )
  program_fingerprint = _validated_fingerprint(
    fields["program_content_fingerprint"],
    label="program-evaluation program content fingerprint",
    source=source,
  )
  model_fingerprint = _validated_fingerprint(
    fields["compatible_model_content_fingerprint"],
    label="program-evaluation model content fingerprint",
    source=source,
  )
  require_same_instance(
    program.instance_id,
    program_instance,
    context="assembly program evaluation",
  )
  require_same_instance(
    model.instance_id,
    model_instance,
    context="assembly model evaluation",
  )
  coordinate_names = fields["coordinate_names"]
  if type(coordinate_names) is not tuple or any(
    type(item) is not str for item in coordinate_names
  ):
    _evaluation_boundary(
      "malformed-program-evaluation",
      "P1-A evaluation coordinate names must be an exact string tuple",
      source,
    )
  if (
    program_fingerprint != program.content_fingerprint
    or model_fingerprint != model.content_fingerprint
    or coordinate_names != program.coordinate_names
  ):
    _evaluation_boundary(
      "program-evaluation-compatibility-mismatch",
      "P1-A evaluation identities or coordinate schema do not match assembly",
      source,
    )
  full_count = model.dofs.global_size
  coordinate_count = len(program.coordinate_names)
  arrays = (
    (
      "coordinate values",
      fields["coordinate_values"],
      (coordinate_count,),
    ),
    ("prescribed offsets", fields["prescribed_offsets"], (full_count,)),
    (
      "prescribed offset derivatives",
      fields["prescribed_offset_derivatives"],
      (full_count, coordinate_count),
    ),
    ("nodal force", fields["nodal_force"], (full_count,)),
    (
      "nodal force derivatives",
      fields["nodal_force_derivatives"],
      (full_count, coordinate_count),
    ),
  )
  validated: list[tuple[str, FinalizedArray, np.ndarray]] = []
  for label, carrier, shape in arrays:
    values = _validated_array_values(
      carrier,
      dtype=_FLOATING_DTYPE,
      shape=shape,
      label=f"program-evaluation {label}",
      source=source,
      finite=True,
    )
    validated.append((label, carrier, values))
  for index, (left_label, left, left_values) in enumerate(validated):
    for right_label, right, right_values in validated[index + 1 :]:
      if (
        left is right
        or left_values is right_values
        or np.shares_memory(left_values, right_values)
      ):
        _evaluation_boundary(
          "malformed-program-evaluation",
          f"program-evaluation {left_label} and {right_label} require separate storage",
          source,
        )
  return value


def _translate_program_evaluation_error(error: ProgramEvaluationError) -> NoReturn:
  translated: list[AssemblyEvaluationDiagnostic] = []
  try:
    diagnostics = error.diagnostics
  except AttributeError:
    diagnostics = ()
  for diagnostic in diagnostics:
    if type(diagnostic) is not ProgramEvaluationDiagnostic:
      translated = []
      break
    translated.append(
      AssemblyEvaluationDiagnostic(
        code=diagnostic.code,
        message=diagnostic.message,
        source=diagnostic.source,
      )
    )
  if not translated:
    translated.append(
      AssemblyEvaluationDiagnostic(
        code="program-evaluation-failed",
        message="P1-A program evaluation failed at the assembly boundary",
        source=_EVALUATION_SOURCE,
      )
    )
  raise AssemblyEvaluationError(tuple(translated)) from None


def _finalized(value: np.ndarray) -> FinalizedArray:
  return FinalizedArray(value, dtype=np.float64)


def _validate_fresh_result_storage(
  arrays: tuple[tuple[str, FinalizedArray], ...],
) -> None:
  for index, (left_label, left) in enumerate(arrays):
    for right_label, right in arrays[index + 1 :]:
      if (
        left is right
        or left.values is right.values
        or np.shares_memory(left.values, right.values)
      ):
        _evaluation_boundary(
          "assembly-value-alias",
          f"assembly {left_label} and {right_label} require fresh separate storage",
          _EVALUATION_SOURCE,
        )


def _evaluate(
  model: object,
  program: object,
  plan: object,
  program_point: object,
) -> LinearStaticContributions:
  validated_model = _validated_model(model)
  validated_program = _validated_program(program)
  program_source = _program_source(
    validated_program,
    fallback=_EVALUATION_SOURCE,
  )
  try:
    _validate_model_program_compatibility(
      validated_model,
      validated_program,
      source=program_source,
    )
  except IdentityMismatchError:
    _evaluation_boundary(
      "model-program-live-identity-mismatch",
      "assembly evaluation requires exact live model compatibility",
      program_source,
    )
  validated_plan = _validated_prepared_plan(
    plan,
    validated_model,
    validated_program,
    source=_EVALUATION_SOURCE,
  )
  try:
    raw_program_evaluation = evaluate_program(validated_program, program_point)
  except ProgramEvaluationError as error:
    _translate_program_evaluation_error(error)
  evaluation = _validate_program_evaluation(
    raw_program_evaluation,
    validated_model,
    validated_program,
    source=program_source,
  )

  block = validated_model.domain_blocks[0]
  domain_plan = validated_plan.domain_coo_plan
  full_raw_values = _full_raw_operator_values(
    validated_model,
    block,
    source=_EVALUATION_SOURCE,
  )
  full_values = _coalesce(
    full_raw_values,
    domain_plan.full_raw_to_canonical.values,
    len(domain_plan.full_row_indices.values),
    label="full operator coalescing",
    source=_EVALUATION_SOURCE,
  )
  reduced_raw_values = _reduced_raw_values(
    full_raw_values,
    validated_plan,
    source=_EVALUATION_SOURCE,
  )
  reduced_values = _coalesce(
    reduced_raw_values,
    domain_plan.reduced_raw_to_canonical.values,
    len(domain_plan.reduced_row_indices.values),
    label="reduced operator coalescing",
    source=_EVALUATION_SOURCE,
  )

  full_count = validated_model.dofs.global_size
  coordinate_count = len(validated_program.coordinate_names)
  _checked_product(
    full_count,
    coordinate_count,
    label="full assembly derivative count",
    source=_EVALUATION_SOURCE,
  )
  raw_rows = domain_plan.full_raw_row_indices.values
  raw_columns = domain_plan.full_raw_column_indices.values
  offset_internal = _raw_matvec(
    raw_rows,
    raw_columns,
    full_raw_values,
    evaluation.prescribed_offsets.values,
    full_count,
    label="full affine-offset internal force",
    source=_EVALUATION_SOURCE,
  )
  offset_internal_derivatives = _raw_matmat(
    raw_rows,
    raw_columns,
    full_raw_values,
    evaluation.prescribed_offset_derivatives.values,
    full_count,
    label="full affine-offset internal-force derivative",
    source=_EVALUATION_SOURCE,
  )
  full_rhs = _difference_vector(
    evaluation.nodal_force.values,
    offset_internal,
    label="full offset-corrected RHS",
    source=_EVALUATION_SOURCE,
  )
  full_rhs_derivatives = _difference_matrix(
    evaluation.nodal_force_derivatives.values,
    offset_internal_derivatives,
    label="full offset-corrected RHS derivative",
    source=_EVALUATION_SOURCE,
  )
  reduced_external = _reduce_vector(
    evaluation.nodal_force.values,
    validated_plan,
    label="reduced external force",
    source=_EVALUATION_SOURCE,
  )
  reduced_external_derivatives = _reduce_matrix(
    evaluation.nodal_force_derivatives.values,
    validated_plan,
    label="reduced external-force derivative",
    source=_EVALUATION_SOURCE,
  )
  reduced_offset_internal = _reduce_vector(
    offset_internal,
    validated_plan,
    label="reduced affine-offset internal force",
    source=_EVALUATION_SOURCE,
  )
  reduced_offset_internal_derivatives = _reduce_matrix(
    offset_internal_derivatives,
    validated_plan,
    label="reduced affine-offset internal-force derivative",
    source=_EVALUATION_SOURCE,
  )
  reduced_rhs = _difference_vector(
    reduced_external,
    reduced_offset_internal,
    label="reduced RHS",
    source=_EVALUATION_SOURCE,
  )
  reduced_rhs_derivatives = _difference_matrix(
    reduced_external_derivatives,
    reduced_offset_internal_derivatives,
    label="reduced RHS derivative",
    source=_EVALUATION_SOURCE,
  )

  full_raw_carrier = _finalized(full_raw_values)
  reduced_raw_carrier = _finalized(reduced_raw_values)
  full_operator_values = _finalized(full_values)
  reduced_operator_values = _finalized(reduced_values)
  offset_internal_carrier = _finalized(offset_internal)
  offset_internal_derivative_carrier = _finalized(offset_internal_derivatives)
  full_rhs_carrier = _finalized(full_rhs)
  full_rhs_derivative_carrier = _finalized(full_rhs_derivatives)
  reduced_external_carrier = _finalized(reduced_external)
  reduced_external_derivative_carrier = _finalized(reduced_external_derivatives)
  reduced_offset_carrier = _finalized(reduced_offset_internal)
  reduced_offset_derivative_carrier = _finalized(reduced_offset_internal_derivatives)
  reduced_rhs_carrier = _finalized(reduced_rhs)
  reduced_rhs_derivative_carrier = _finalized(reduced_rhs_derivatives)
  _validate_fresh_result_storage(
    (
      ("full raw operator values", full_raw_carrier),
      ("reduced raw operator values", reduced_raw_carrier),
      ("full canonical operator values", full_operator_values),
      ("reduced canonical operator values", reduced_operator_values),
      ("full offset internal force", offset_internal_carrier),
      (
        "full offset internal derivatives",
        offset_internal_derivative_carrier,
      ),
      ("full RHS", full_rhs_carrier),
      ("full RHS derivatives", full_rhs_derivative_carrier),
      ("reduced external force", reduced_external_carrier),
      (
        "reduced external derivatives",
        reduced_external_derivative_carrier,
      ),
      ("reduced offset internal force", reduced_offset_carrier),
      (
        "reduced offset internal derivatives",
        reduced_offset_derivative_carrier,
      ),
      ("reduced RHS", reduced_rhs_carrier),
      ("reduced RHS derivatives", reduced_rhs_derivative_carrier),
    )
  )
  return LinearStaticContributions(
    program_evaluation=evaluation,
    plan_content_fingerprint=validated_plan.content_fingerprint,
    full_operator=CanonicalCooOperator(
      shape=domain_plan.full_shape,
      row_indices=domain_plan.full_row_indices,
      column_indices=domain_plan.full_column_indices,
      values=full_operator_values,
    ),
    reduced_operator=CanonicalCooOperator(
      shape=domain_plan.reduced_shape,
      row_indices=domain_plan.reduced_row_indices,
      column_indices=domain_plan.reduced_column_indices,
      values=reduced_operator_values,
    ),
    full_raw_operator_values=full_raw_carrier,
    reduced_raw_operator_values=reduced_raw_carrier,
    full_affine_offset_internal_force=offset_internal_carrier,
    full_affine_offset_internal_force_derivatives=(offset_internal_derivative_carrier),
    full_offset_corrected_rhs=full_rhs_carrier,
    full_offset_corrected_rhs_derivatives=full_rhs_derivative_carrier,
    reduced_external_force=reduced_external_carrier,
    reduced_external_force_derivatives=reduced_external_derivative_carrier,
    reduced_affine_offset_internal_force=reduced_offset_carrier,
    reduced_affine_offset_internal_force_derivatives=(
      reduced_offset_derivative_carrier
    ),
    reduced_rhs=reduced_rhs_carrier,
    reduced_rhs_derivatives=reduced_rhs_derivative_carrier,
  )


def assemble_reference_linear(
  model: CompiledModel,
  program: CompiledProgram,
  plan: PreparedAssemblyPlan,
  program_point: ProgramPoint,
) -> LinearStaticContributions:
  """Evaluate one point into exact full and affine-reduced Q8 contributions."""
  try:
    return _evaluate(model, program, plan, program_point)
  except AssemblyEvaluationError:
    raise
  except _BoundaryError as error:
    raise AssemblyEvaluationError(
      (
        AssemblyEvaluationDiagnostic(
          code=error.code,
          message=error.message,
          source=_EVALUATION_SOURCE,
        ),
      )
    ) from None
  except IdentityMismatchError:
    raise AssemblyEvaluationError(
      (
        AssemblyEvaluationDiagnostic(
          code="assembly-live-identity-mismatch",
          message="assembly evaluation requires exact live input compatibility",
          source=_EVALUATION_SOURCE,
        ),
      )
    ) from None
  except _AssemblyBoundaryError as error:
    raise AssemblyEvaluationError(
      (
        AssemblyEvaluationDiagnostic(
          code=error.code,
          message=error.message,
          source=error.source,
        ),
      )
    ) from None
  except Exception:
    raise AssemblyEvaluationError(
      (
        AssemblyEvaluationDiagnostic(
          code="assembly-evaluation-failed",
          message="reference assembly evaluation failed before values could escape",
          source=_EVALUATION_SOURCE,
        ),
      )
    ) from None
