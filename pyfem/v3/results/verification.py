"""Record-only and fresh verification for the frozen linear Q8 solution."""

from __future__ import annotations

import math
import sys
from typing import NoReturn

import numpy as np

from pyfem.v3.assembly import (
  AssemblyEvaluationError,
  AssemblyPreparationError,
  CanonicalCooOperator,
  LinearStaticContributionRequest,
  LinearStaticContributions,
  PreparedAssemblyPlan,
  assemble_reference_linear,
  prepare_assembly_plan,
)
from pyfem.v3.compile.program import (
  _BoundaryError,
  _validated_model,
  _validated_program,
)
from pyfem.v3.model import (
  EVOLUTION_STATE_SCHEMA,
  PHYSICAL_STATE_SCHEMA,
  PROGRAM_HISTORY_SCHEMA,
  CanonicalManifest,
  CommittedAnalysisState,
  CompiledModel,
  CompiledProgram,
  ContentFingerprint,
  EvolutionState,
  FinalizedArray,
  InstanceId,
  PhysicalState,
  ProgramEvaluation,
  ProgramHistory,
  StateGeneration,
  require_generation_successor,
  require_same_generation,
  require_same_instance,
)
from pyfem.v3.model.program import (
  ProgramAffineValueWitness,
  ProgramDofWitness,
  ProgramPrescribedWitness,
  ProgramTieWitness,
)
from pyfem.v3.results.contracts import (
  LinearBalanceLedger,
  VerificationCheck,
  VerificationReport,
)
from pyfem.v3.results.diagnostics import (
  SolutionVerificationDiagnostic,
  SolutionVerificationError,
)
from pyfem.v3.spec import ProgramCoordinateValue, ProgramPoint
from pyfem.v3.spec.program_diagnostics import render_program_value

_FLOAT64 = np.dtype(np.float64)
_INT64 = np.dtype(np.int64)
_MAX_FLOAT = sys.float_info.max


def _malformed(code: str, message: str) -> NoReturn:
  raise SolutionVerificationError((SolutionVerificationDiagnostic(code, message),))


def _manifest_equal(left: object, right: object) -> bool:
  if type(left) is not CanonicalManifest or type(right) is not CanonicalManifest:
    return False
  try:
    return left.to_bytes() == right.to_bytes()
  except (AttributeError, OverflowError, RecursionError, TypeError, ValueError):
    return False


def _fingerprint_matches_manifest(
  fingerprint: object,
  manifest: object,
) -> bool:
  if (
    type(fingerprint) is not ContentFingerprint
    or type(manifest) is not CanonicalManifest
  ):
    return False
  try:
    return fingerprint == ContentFingerprint.from_manifest(manifest)
  except (AttributeError, OverflowError, RecursionError, TypeError, ValueError):
    return False


def _same_fingerprint(left: object, right: object) -> bool:
  if type(left) is not ContentFingerprint or type(right) is not ContentFingerprint:
    return False
  try:
    return left == right
  except (AttributeError, TypeError, ValueError):
    return False


def _require_instance(expected: object, actual: object, label: str) -> None:
  try:
    require_same_instance(expected, actual, context=label)
  except (AttributeError, TypeError, ValueError):
    _malformed("solution-identity-mismatch", f"{label} has a foreign live identity")


def _require_generation(expected: object, actual: object, label: str) -> None:
  try:
    require_same_generation(expected, actual, context=label)
  except (AttributeError, TypeError, ValueError):
    _malformed(
      "solution-generation-mismatch",
      f"{label} has a foreign accepted-state generation",
    )


def _require_successor(base: object, candidate: object, label: str) -> None:
  try:
    require_generation_successor(base, candidate, context=label)
  except (AttributeError, TypeError, ValueError):
    _malformed(
      "solution-generation-mismatch",
      f"{label} is not the exact next accepted-state generation",
    )


def _array(
  value: object,
  *,
  dtype: np.dtype[np.generic],
  shape: tuple[int, ...],
  label: str,
  finite: bool = True,
) -> np.ndarray:
  if type(value) is not FinalizedArray:
    _malformed("malformed-solution-array", f"{label} is not a FinalizedArray")
  try:
    array = value.values
  except AttributeError:
    _malformed("malformed-solution-array", f"{label} has an uninitialized value")
  if (
    type(array) is not np.ndarray
    or array.dtype != dtype
    or array.dtype.metadata is not None
    or array.shape != shape
    or array.flags.writeable
    or not array.flags.c_contiguous
    or not array.flags.owndata
    or array.base is not None
  ):
    _malformed(
      "malformed-solution-array",
      f"{label} has the wrong dtype, shape, ownership, or mutability",
    )
  if finite and not bool(np.isfinite(array).all()):
    _malformed("nonfinite-solution-array", f"{label} contains nonfinite values")
  return array


def _final_float(value: np.ndarray | list[float] | tuple[float, ...]) -> FinalizedArray:
  return FinalizedArray(value, dtype=np.float64)


def _final_int(value: np.ndarray | list[int] | tuple[int, ...]) -> FinalizedArray:
  return FinalizedArray(value, dtype=np.int64)


def clone_program_evaluation(value: ProgramEvaluation) -> ProgramEvaluation:
  """Detach every array in one already validated program evaluation."""
  if type(value) is not ProgramEvaluation:
    _malformed(
      "malformed-program-evaluation",
      "program evaluation must be exactly ProgramEvaluation",
    )
  return ProgramEvaluation(
    program_instance_id=value.program_instance_id,
    program_content_fingerprint=value.program_content_fingerprint,
    compatible_model_instance_id=value.compatible_model_instance_id,
    compatible_model_content_fingerprint=(value.compatible_model_content_fingerprint),
    coordinate_names=tuple(value.coordinate_names),
    coordinate_values=_final_float(value.coordinate_values.values),
    prescribed_offsets=_final_float(value.prescribed_offsets.values),
    prescribed_offset_derivatives=_final_float(
      value.prescribed_offset_derivatives.values
    ),
    nodal_force=_final_float(value.nodal_force.values),
    nodal_force_derivatives=_final_float(value.nodal_force_derivatives.values),
  )


def clone_balance_ledger(value: LinearBalanceLedger) -> LinearBalanceLedger:
  """Copy a ledger into storage disjoint from its trial owner."""
  if type(value) is not LinearBalanceLedger:
    _malformed("malformed-balance-ledger", "balance ledger has a foreign type")
  return LinearBalanceLedger(
    ledger_id=value.ledger_id,
    prepared_instance_id=value.prepared_instance_id,
    model_instance_id=value.model_instance_id,
    model_content_fingerprint=value.model_content_fingerprint,
    program_instance_id=value.program_instance_id,
    program_content_fingerprint=value.program_content_fingerprint,
    plan_content_fingerprint=value.plan_content_fingerprint,
    request_manifest=value.request_manifest,
    transaction_id=value.transaction_id,
    trial_id=value.trial_id,
    base_generation=value.base_generation,
    candidate_generation=value.candidate_generation,
    program_evaluation=clone_program_evaluation(value.program_evaluation),
    reduced_coordinates=_final_float(value.reduced_coordinates.values),
    full_primary_values=_final_float(value.full_primary_values.values),
    reduced_primary_image=_final_float(value.reduced_primary_image.values),
    prescribed_offsets=_final_float(value.prescribed_offsets.values),
    external_force=_final_float(value.external_force.values),
    internal_force=_final_float(value.internal_force.values),
    full_residual=_final_float(value.full_residual.values),
    reduced_rhs=_final_float(value.reduced_rhs.values),
    reduced_internal_force=_final_float(value.reduced_internal_force.values),
    reduced_residual=_final_float(value.reduced_residual.values),
    constraint_force=_final_float(value.constraint_force.values),
    balance=_final_float(value.balance.values),
    direct_reaction_dof_indices=_final_int(value.direct_reaction_dof_indices.values),
    direct_reactions=_final_float(value.direct_reactions.values),
    constraint_ids=tuple(value.constraint_ids),
    constraint_violation=_final_float(value.constraint_violation.values),
    constraint_row_scales=_final_float(value.constraint_row_scales.values),
    constraint_work=value.constraint_work,
    external_work=value.external_work,
    internal_work=value.internal_work,
    full_force_scale=value.full_force_scale,
    reduced_force_scale=value.reduced_force_scale,
    reconstruction_scale=value.reconstruction_scale,
    work_scale=value.work_scale,
    full_operator_infinity_norm=value.full_operator_infinity_norm,
    reduced_operator_infinity_norm=value.reduced_operator_infinity_norm,
    prolongation_transpose_infinity_norm=(value.prolongation_transpose_infinity_norm),
    verification_tolerance=value.verification_tolerance,
  )


def dense_operator(operator: CanonicalCooOperator) -> np.ndarray:
  """Materialize one already validated canonical COO operator."""
  shape = operator.shape
  dense = np.zeros(shape, dtype=np.float64)
  dense[operator.row_indices.values, operator.column_indices.values] = (
    operator.values.values
  )
  return dense


def prolongation(plan: PreparedAssemblyPlan) -> np.ndarray:
  """Materialize the exact full-to-reduced affine prolongation."""
  vector = plan.nodal_vector_plan
  matrix = np.zeros(
    (vector.full_dof_count, vector.reduced_dof_count),
    dtype=np.float64,
  )
  matrix[vector.full_dof_indices.values, vector.reduced_dof_indices.values] = (
    vector.coefficients.values
  )
  return matrix


def _bounded_product(left: float, right: float) -> float:
  if left == 0.0 or right == 0.0:
    return 0.0
  product = left * right
  if math.isfinite(product):
    return abs(product)
  return _MAX_FLOAT


def _infinity_norm(value: np.ndarray) -> float:
  if value.size == 0:
    return 0.0
  if value.ndim == 1:
    return float(np.max(np.abs(value)))
  if value.ndim != 2:
    _malformed(
      "malformed-verification-value", "infinity norm requires a vector or matrix"
    )
  largest = 0.0
  for row in value:
    row_max = float(np.max(np.abs(row))) if row.size else 0.0
    if row_max == 0.0:
      total = 0.0
    else:
      scaled = math.fsum(float(abs(item) / row_max) for item in row)
      total = _bounded_product(row_max, scaled)
    largest = max(largest, total)
  return largest


def _difference_norm(left: np.ndarray, right: np.ndarray) -> float:
  with np.errstate(over="ignore", invalid="ignore"):
    difference = left - right
  if bool(np.isnan(difference).any()):
    return math.inf
  return _infinity_norm(difference)


def _dot(left: np.ndarray, right: np.ndarray, label: str) -> float:
  with np.errstate(over="ignore", invalid="ignore"):
    value = float(np.dot(left, right))
  if not math.isfinite(value):
    _malformed("nonfinite-verification-value", f"{label} is nonfinite")
  return value


def _strict_ratio_greater(value: float, scale: float, threshold: float) -> bool:
  if value <= 0.0 or scale <= 0.0:
    return False
  value_fraction, value_exponent = math.frexp(value)
  scale_fraction, scale_exponent = math.frexp(scale)
  ratio_fraction, normalization_exponent = math.frexp(value_fraction / scale_fraction)
  ratio_exponent = value_exponent - scale_exponent + normalization_exponent
  threshold_fraction, threshold_exponent = math.frexp(threshold)
  if ratio_exponent != threshold_exponent:
    return ratio_exponent > threshold_exponent
  return ratio_fraction > threshold_fraction


def _ratio(error: float, scale: float) -> float:
  if error == 0.0:
    return 0.0
  if scale == 0.0:
    return math.inf
  if math.isinf(error):
    return math.inf
  error_fraction, error_exponent = math.frexp(error)
  scale_fraction, scale_exponent = math.frexp(scale)
  exponent = error_exponent - scale_exponent
  fraction = error_fraction / scale_fraction
  if exponent > 1023:
    return math.inf
  if exponent < -1074:
    return 0.0
  try:
    return math.ldexp(fraction, exponent)
  except OverflowError:
    return math.inf


def verification_check(
  name: str,
  error: float,
  scale: float,
  tolerance: float,
) -> VerificationCheck:
  """Build one no-unit-floor, exact-zero-aware normalized check."""
  if error < 0.0 or math.isnan(error) or scale < 0.0 or math.isnan(scale):
    _malformed(
      "malformed-verification-value",
      f"{name} has an invalid error or normalization scale",
    )
  normalized = _ratio(error, scale)
  passed = error == 0.0 if scale == 0.0 else normalized <= tolerance
  return VerificationCheck(
    name=name,
    passed=passed,
    error=error,
    scale=scale,
    normalized_error=normalized,
    tolerance=tolerance,
  )


def _boolean_check(
  name: str,
  passed: bool,
  tolerance: float,
) -> VerificationCheck:
  return verification_check(name, 0.0 if passed else 1.0, 0.0, tolerance)


def _dof_index(model: CompiledModel, witness: ProgramDofWitness) -> int:
  dofs = model.dofs
  try:
    node_index = next(
      index
      for index, node_id in enumerate(dofs.node_ids)
      if type(node_id) is type(witness.node_id) and node_id == witness.node_id
    )
    component_index = dofs.components.index(witness.component)
  except (StopIteration, ValueError):
    _malformed(
      "malformed-program-witness",
      "program witness references a DOF absent from its exact model",
    )
  return int(dofs.node_component_dofs.values[node_index, component_index])


def _coordinate_map(evaluation: ProgramEvaluation) -> dict[str, float]:
  return {
    name: float(value)
    for name, value in zip(
      evaluation.coordinate_names,
      evaluation.coordinate_values.values,
      strict=True,
    )
  }


def _affine_terms(
  value: ProgramAffineValueWitness,
  coordinates: dict[str, float],
) -> tuple[float, tuple[float, ...]]:
  terms = tuple(
    float(item.coefficient) * coordinates[item.coordinate]
    for item in value.coefficients
  )
  try:
    evaluated = math.fsum((float(value.constant), *terms))
  except (OverflowError, ValueError):
    _malformed(
      "nonfinite-constraint-evaluation",
      "an authored constraint row cannot be evaluated finitely",
    )
  if not math.isfinite(evaluated) or any(not math.isfinite(item) for item in terms):
    _malformed(
      "nonfinite-constraint-evaluation",
      "an authored constraint row cannot be evaluated finitely",
    )
  return evaluated, terms


def constraint_evidence(
  model: CompiledModel,
  program: CompiledProgram,
  evaluation: ProgramEvaluation,
  primary: np.ndarray,
) -> tuple[tuple[str | int, ...], np.ndarray, np.ndarray, np.ndarray]:
  """Evaluate authored constraint rows and exact direct-prescription DOFs."""
  coordinates = _coordinate_map(evaluation)
  identifiers: list[str | int] = []
  violations: list[float] = []
  scales: list[float] = []
  direct_dofs: list[int] = []
  for constraint in program.meaning_witness.constraints:
    identifiers.append(constraint.id)
    if type(constraint) is ProgramPrescribedWitness:
      target_dof = _dof_index(model, constraint.target)
      target = float(primary[target_dof])
      evaluated, terms = _affine_terms(constraint.value, coordinates)
      violation = target - evaluated
      row_scale = max(
        abs(target),
        abs(evaluated),
        abs(float(constraint.value.constant)),
        *(abs(item) for item in terms),
      )
      direct_dofs.append(target_dof)
    elif type(constraint) is ProgramTieWitness:
      slave_dof = _dof_index(model, constraint.slave)
      master_dof = _dof_index(model, constraint.master)
      target = float(primary[slave_dof])
      master_term = float(constraint.factor) * float(primary[master_dof])
      offset, terms = _affine_terms(constraint.offset, coordinates)
      evaluated = master_term + offset
      violation = target - evaluated
      row_scale = max(
        abs(target),
        abs(evaluated),
        abs(master_term),
        abs(offset),
        abs(float(constraint.offset.constant)),
        *(abs(item) for item in terms),
      )
    else:
      _malformed(
        "malformed-program-witness",
        "program witness contains a foreign constraint carrier",
      )
    if not math.isfinite(violation) or not math.isfinite(row_scale):
      _malformed(
        "nonfinite-constraint-evaluation",
        "an authored constraint row produced nonfinite evidence",
      )
    violations.append(violation)
    scales.append(row_scale)
  return (
    tuple(identifiers),
    np.asarray(violations, dtype=np.float64),
    np.asarray(scales, dtype=np.float64),
    np.asarray(direct_dofs, dtype=np.int64),
  )


def build_balance_ledger(
  *,
  model: CompiledModel,
  program: CompiledProgram,
  plan: PreparedAssemblyPlan,
  contributions: LinearStaticContributions,
  request_manifest: CanonicalManifest,
  prepared_instance_id: InstanceId,
  transaction_id: InstanceId,
  trial_id: InstanceId,
  base_generation: StateGeneration,
  candidate_generation: StateGeneration,
  reduced_coordinates: np.ndarray,
  primary_values: np.ndarray,
  ledger_id: InstanceId | None = None,
) -> LinearBalanceLedger:
  """Derive the complete frozen balance ledger from original audit operators."""
  full_operator = dense_operator(contributions.full_operator)
  reduced_operator = dense_operator(contributions.reduced_operator)
  projection = prolongation(plan)
  offsets = contributions.program_evaluation.prescribed_offsets.values
  reduced_image = projection @ reduced_coordinates
  external = contributions.program_evaluation.nodal_force.values
  internal = full_operator @ primary_values
  full_residual = external - internal
  reduced_rhs = contributions.reduced_rhs.values
  reduced_internal = reduced_operator @ reduced_coordinates
  reduced_residual = reduced_rhs - reduced_internal
  constraint_force = internal - external
  balance = external + constraint_force - internal
  constraint_ids, violations, row_scales, direct_dofs = constraint_evidence(
    model,
    program,
    contributions.program_evaluation,
    primary_values,
  )
  direct_reactions = constraint_force[direct_dofs]

  full_operator_norm = _infinity_norm(full_operator)
  reduced_operator_norm = _infinity_norm(reduced_operator)
  projection_transpose_norm = _infinity_norm(projection.T)
  primary_norm = _infinity_norm(primary_values)
  offset_norm = _infinity_norm(offsets)
  full_scale = max(
    _infinity_norm(external),
    _infinity_norm(internal),
    _infinity_norm(constraint_force),
    _bounded_product(full_operator_norm, primary_norm),
  )
  reduced_scale = max(
    _infinity_norm(reduced_rhs),
    _infinity_norm(reduced_internal),
    _bounded_product(projection_transpose_norm, full_scale),
  )
  reconstruction_scale = max(
    primary_norm,
    _infinity_norm(reduced_image),
    offset_norm,
  )
  constraint_work = _dot(constraint_force, primary_values, "constraint work")
  external_work = _dot(external, primary_values, "external work")
  internal_work = _dot(internal, primary_values, "internal work")
  work_scale = max(
    abs(external_work),
    abs(internal_work),
    abs(constraint_work),
    _bounded_product(full_scale, max(primary_norm, offset_norm)),
  )
  return LinearBalanceLedger(
    ledger_id=InstanceId() if ledger_id is None else ledger_id,
    prepared_instance_id=prepared_instance_id,
    model_instance_id=model.instance_id,
    model_content_fingerprint=model.content_fingerprint,
    program_instance_id=program.instance_id,
    program_content_fingerprint=program.content_fingerprint,
    plan_content_fingerprint=plan.content_fingerprint,
    request_manifest=request_manifest,
    transaction_id=transaction_id,
    trial_id=trial_id,
    base_generation=base_generation,
    candidate_generation=candidate_generation,
    program_evaluation=clone_program_evaluation(contributions.program_evaluation),
    reduced_coordinates=_final_float(reduced_coordinates),
    full_primary_values=_final_float(primary_values),
    reduced_primary_image=_final_float(reduced_image),
    prescribed_offsets=_final_float(offsets),
    external_force=_final_float(external),
    internal_force=_final_float(internal),
    full_residual=_final_float(full_residual),
    reduced_rhs=_final_float(reduced_rhs),
    reduced_internal_force=_final_float(reduced_internal),
    reduced_residual=_final_float(reduced_residual),
    constraint_force=_final_float(constraint_force),
    balance=_final_float(balance),
    direct_reaction_dof_indices=_final_int(direct_dofs),
    direct_reactions=_final_float(direct_reactions),
    constraint_ids=constraint_ids,
    constraint_violation=_final_float(violations),
    constraint_row_scales=_final_float(row_scales),
    constraint_work=constraint_work,
    external_work=external_work,
    internal_work=internal_work,
    full_force_scale=full_scale,
    reduced_force_scale=reduced_scale,
    reconstruction_scale=reconstruction_scale,
    work_scale=work_scale,
    full_operator_infinity_norm=full_operator_norm,
    reduced_operator_infinity_norm=reduced_operator_norm,
    prolongation_transpose_infinity_norm=projection_transpose_norm,
    verification_tolerance=1.0e-12,
  )


def _point_from_evaluation(evaluation: ProgramEvaluation) -> ProgramPoint:
  return ProgramPoint(
    tuple(
      ProgramCoordinateValue(name, float(value))
      for name, value in zip(
        evaluation.coordinate_names,
        evaluation.coordinate_values.values,
        strict=True,
      )
    )
  )


def _fresh_program_evaluation_checks(
  retained: ProgramEvaluation,
  fresh: ProgramEvaluation,
  tolerance: float,
) -> tuple[VerificationCheck, ...]:
  checks = [
    _boolean_check(
      "program_coordinate_names",
      retained.coordinate_names == fresh.coordinate_names,
      tolerance,
    )
  ]
  fields = (
    ("program_coordinate_values", retained.coordinate_values, fresh.coordinate_values),
    (
      "program_prescribed_offsets",
      retained.prescribed_offsets,
      fresh.prescribed_offsets,
    ),
    (
      "program_prescribed_offset_derivatives",
      retained.prescribed_offset_derivatives,
      fresh.prescribed_offset_derivatives,
    ),
    ("program_nodal_force", retained.nodal_force, fresh.nodal_force),
    (
      "program_nodal_force_derivatives",
      retained.nodal_force_derivatives,
      fresh.nodal_force_derivatives,
    ),
  )
  for name, retained_array, fresh_array in fields:
    checks.append(
      _boolean_check(
        name,
        bool(np.array_equal(retained_array.values, fresh_array.values)),
        tolerance,
      )
    )
  return tuple(checks)


def _fresh_backend_checks(
  convergence: object,
  operator: np.ndarray,
  reduced_residual: np.ndarray,
  reduced_force_scale: float,
  tolerance: float,
) -> tuple[VerificationCheck, ...]:
  from pyfem.v3.analysis.contracts import (
    LINEAR_STATIC_BACKEND_POLICY,
    LINEAR_STATIC_CHOLESKY_PIVOT_RATIO,
    LINEAR_STATIC_SYMMETRY_EPSILON_FACTOR,
    LINEAR_STATIC_VERIFICATION_TOLERANCE,
    LinearConvergenceRecord,
  )

  if type(convergence) is not LinearConvergenceRecord:
    _malformed(
      "malformed-convergence-record",
      "fresh verification requires the complete exact convergence record",
    )
  checks: list[VerificationCheck] = [
    _boolean_check(
      "backend_policy",
      convergence.backend_policy == LINEAR_STATIC_BACKEND_POLICY,
      tolerance,
    ),
    _boolean_check(
      "backend_convergence_record",
      convergence.converged is True
      and convergence.iteration_count == 1
      and convergence.verification_tolerance == LINEAR_STATIC_VERIFICATION_TOLERANCE,
      tolerance,
    ),
  ]
  residual_norm = _infinity_norm(reduced_residual)
  checks.append(
    verification_check(
      "backend_reduced_residual_norm",
      abs(convergence.reduced_residual_norm - residual_norm),
      reduced_force_scale,
      tolerance,
    )
  )
  reduced_count = operator.shape[0]
  if reduced_count == 0:
    checks.extend(
      (
        _boolean_check(
          "backend_zero_free_bypass",
          convergence.factorization_bypassed is True
          and convergence.factorization_performed is False
          and convergence.factorization_reused is False,
          tolerance,
        ),
        verification_check(
          "backend_operator_norm",
          abs(convergence.operator_infinity_norm),
          0.0,
          tolerance,
        ),
        _boolean_check(
          "backend_minimum_pivot",
          convergence.minimum_unscaled_pivot is None,
          tolerance,
        ),
      )
    )
    return tuple(checks)

  maximum = float(np.max(np.abs(operator)))
  asymmetry = float(np.max(np.abs(operator - operator.T)))
  symmetry_bound = (
    0.0
    if maximum == 0.0
    else LINEAR_STATIC_SYMMETRY_EPSILON_FACTOR * np.finfo(np.float64).eps * maximum
  )
  checks.append(
    _boolean_check(
      "backend_symmetry_admission",
      asymmetry <= symmetry_bound,
      tolerance,
    )
  )
  solve_operator = 0.5 * (operator + operator.T)
  solve_finite = bool(np.isfinite(solve_operator).all())
  operator_scale = _infinity_norm(solve_operator) if solve_finite else math.inf
  checks.extend(
    (
      _boolean_check(
        "backend_solver_projection",
        solve_finite and bool(np.array_equal(solve_operator, solve_operator.T)),
        tolerance,
      ),
      verification_check(
        "backend_operator_norm",
        abs(convergence.operator_infinity_norm - operator_scale),
        max(convergence.operator_infinity_norm, operator_scale),
        tolerance,
      ),
      _boolean_check(
        "backend_factorization_mode",
        convergence.factorization_bypassed is False
        and (convergence.factorization_performed != convergence.factorization_reused),
        tolerance,
      ),
    )
  )
  factor: np.ndarray | None = None
  if solve_finite and math.isfinite(operator_scale) and operator_scale > 0.0:
    try:
      factor = np.linalg.cholesky(solve_operator)
    except np.linalg.LinAlgError:
      factor = None
  checks.append(_boolean_check("backend_cholesky", factor is not None, tolerance))
  if factor is None:
    checks.extend(
      (
        _boolean_check("backend_pivot_policy", False, tolerance),
        verification_check(
          "backend_minimum_pivot",
          math.inf,
          operator_scale if math.isfinite(operator_scale) else 0.0,
          tolerance,
        ),
      )
    )
    return tuple(checks)

  pivots = np.square(np.diag(factor))
  minimum_pivot = float(np.min(pivots))
  pivot_policy_passed = bool(np.isfinite(pivots).all()) and all(
    _strict_ratio_greater(
      float(pivot),
      operator_scale,
      LINEAR_STATIC_CHOLESKY_PIVOT_RATIO,
    )
    for pivot in pivots
  )
  checks.extend(
    (
      _boolean_check("backend_pivot_policy", pivot_policy_passed, tolerance),
      verification_check(
        "backend_minimum_pivot",
        abs(convergence.minimum_unscaled_pivot - minimum_pivot)
        if type(convergence.minimum_unscaled_pivot) is float
        else math.inf,
        minimum_pivot,
        tolerance,
      ),
    )
  )
  return tuple(checks)


def _fresh_checks(
  retained: LinearBalanceLedger,
  fresh: LinearBalanceLedger,
  primary: np.ndarray,
  projected_constraint_force: np.ndarray,
  operator: np.ndarray,
  convergence: object,
) -> tuple[VerificationCheck, ...]:
  tolerance = fresh.verification_tolerance
  checks: list[VerificationCheck] = []
  checks.extend(
    _fresh_program_evaluation_checks(
      retained.program_evaluation,
      fresh.program_evaluation,
      tolerance,
    )
  )
  checks.extend(
    _fresh_backend_checks(
      convergence,
      operator,
      fresh.reduced_residual.values,
      fresh.reduced_force_scale,
      tolerance,
    )
  )
  checks.append(
    verification_check(
      "field_reconstruction",
      _difference_norm(
        primary,
        fresh.reduced_primary_image.values + fresh.prescribed_offsets.values,
      ),
      fresh.reconstruction_scale,
      tolerance,
    )
  )
  for index, identifier in enumerate(fresh.constraint_ids):
    checks.append(
      verification_check(
        f"constraint[{render_program_value(identifier)}]",
        abs(float(fresh.constraint_violation.values[index])),
        float(fresh.constraint_row_scales.values[index]),
        tolerance,
      )
    )
  checks.extend(
    (
      verification_check(
        "reduced_equilibrium",
        _infinity_norm(fresh.reduced_residual.values),
        fresh.reduced_force_scale,
        tolerance,
      ),
      verification_check(
        "projected_constraint_force",
        _infinity_norm(projected_constraint_force),
        fresh.reduced_force_scale,
        tolerance,
      ),
      verification_check(
        "full_balance",
        _infinity_norm(fresh.balance.values),
        fresh.full_force_scale,
        tolerance,
      ),
      verification_check(
        "internal_force_record",
        _difference_norm(retained.internal_force.values, fresh.internal_force.values),
        fresh.full_force_scale,
        tolerance,
      ),
      verification_check(
        "external_force_record",
        _difference_norm(retained.external_force.values, fresh.external_force.values),
        fresh.full_force_scale,
        tolerance,
      ),
      verification_check(
        "constraint_force_record",
        _difference_norm(
          retained.constraint_force.values,
          fresh.constraint_force.values,
        ),
        fresh.full_force_scale,
        tolerance,
      ),
      verification_check(
        "full_residual_record",
        _difference_norm(retained.full_residual.values, fresh.full_residual.values),
        fresh.full_force_scale,
        tolerance,
      ),
      verification_check(
        "reduced_residual_record",
        _difference_norm(
          retained.reduced_residual.values,
          fresh.reduced_residual.values,
        ),
        fresh.reduced_force_scale,
        tolerance,
      ),
      verification_check(
        "reduced_coordinates_record",
        _difference_norm(
          retained.reduced_coordinates.values,
          fresh.reduced_coordinates.values,
        ),
        fresh.reconstruction_scale,
        tolerance,
      ),
      verification_check(
        "reduced_primary_image_record",
        _difference_norm(
          retained.reduced_primary_image.values,
          fresh.reduced_primary_image.values,
        ),
        fresh.reconstruction_scale,
        tolerance,
      ),
      verification_check(
        "prescribed_offsets_record",
        _difference_norm(
          retained.prescribed_offsets.values,
          fresh.prescribed_offsets.values,
        ),
        fresh.reconstruction_scale,
        tolerance,
      ),
      verification_check(
        "reduced_rhs_record",
        _difference_norm(retained.reduced_rhs.values, fresh.reduced_rhs.values),
        fresh.reduced_force_scale,
        tolerance,
      ),
      verification_check(
        "reduced_internal_force_record",
        _difference_norm(
          retained.reduced_internal_force.values,
          fresh.reduced_internal_force.values,
        ),
        fresh.reduced_force_scale,
        tolerance,
      ),
      verification_check(
        "balance_record",
        _difference_norm(retained.balance.values, fresh.balance.values),
        fresh.full_force_scale,
        tolerance,
      ),
      verification_check(
        "direct_reactions",
        _difference_norm(
          retained.direct_reactions.values, fresh.direct_reactions.values
        ),
        fresh.full_force_scale,
        tolerance,
      ),
      verification_check(
        "direct_reaction_dof_indices",
        0.0
        if np.array_equal(
          retained.direct_reaction_dof_indices.values,
          fresh.direct_reaction_dof_indices.values,
        )
        else 1.0,
        0.0,
        tolerance,
      ),
      verification_check(
        "constraint_violation_record",
        _difference_norm(
          retained.constraint_violation.values,
          fresh.constraint_violation.values,
        ),
        _infinity_norm(fresh.constraint_row_scales.values),
        tolerance,
      ),
      verification_check(
        "constraint_row_scales_record",
        _difference_norm(
          retained.constraint_row_scales.values,
          fresh.constraint_row_scales.values,
        ),
        _infinity_norm(fresh.constraint_row_scales.values),
        tolerance,
      ),
      verification_check(
        "full_force_scale_record",
        abs(retained.full_force_scale - fresh.full_force_scale),
        fresh.full_force_scale,
        tolerance,
      ),
      verification_check(
        "reduced_force_scale_record",
        abs(retained.reduced_force_scale - fresh.reduced_force_scale),
        fresh.reduced_force_scale,
        tolerance,
      ),
      verification_check(
        "reconstruction_scale_record",
        abs(retained.reconstruction_scale - fresh.reconstruction_scale),
        fresh.reconstruction_scale,
        tolerance,
      ),
      verification_check(
        "work_scale_record",
        abs(retained.work_scale - fresh.work_scale),
        fresh.work_scale,
        tolerance,
      ),
      verification_check(
        "full_operator_norm_record",
        abs(retained.full_operator_infinity_norm - fresh.full_operator_infinity_norm),
        fresh.full_operator_infinity_norm,
        tolerance,
      ),
      verification_check(
        "reduced_operator_norm_record",
        abs(
          retained.reduced_operator_infinity_norm - fresh.reduced_operator_infinity_norm
        ),
        fresh.reduced_operator_infinity_norm,
        tolerance,
      ),
      verification_check(
        "prolongation_transpose_norm_record",
        abs(
          retained.prolongation_transpose_infinity_norm
          - fresh.prolongation_transpose_infinity_norm
        ),
        fresh.prolongation_transpose_infinity_norm,
        tolerance,
      ),
      verification_check(
        "constraint_work",
        abs(retained.constraint_work - fresh.constraint_work),
        fresh.work_scale,
        tolerance,
      ),
      verification_check(
        "external_work",
        abs(retained.external_work - fresh.external_work),
        fresh.work_scale,
        tolerance,
      ),
      verification_check(
        "internal_work",
        abs(retained.internal_work - fresh.internal_work),
        fresh.work_scale,
        tolerance,
      ),
      verification_check(
        "work_balance",
        abs(fresh.external_work + fresh.constraint_work - fresh.internal_work),
        fresh.work_scale,
        tolerance,
      ),
    )
  )
  return tuple(checks)


def fresh_verify(
  *,
  model: CompiledModel,
  program: CompiledProgram,
  request_manifest: CanonicalManifest,
  prepared_instance_id: InstanceId,
  plan_content_fingerprint: ContentFingerprint,
  transaction_id: InstanceId,
  trial_id: InstanceId,
  base_generation: StateGeneration,
  candidate_generation: StateGeneration,
  state: CommittedAnalysisState,
  retained_ledger: LinearBalanceLedger,
  convergence: object,
) -> VerificationReport:
  """Prepare and evaluate P1-B anew without reading any solve workspace."""
  try:
    fresh_plan = prepare_assembly_plan(
      model,
      program,
      LinearStaticContributionRequest(),
    )
  except AssemblyPreparationError:
    _malformed(
      "fresh-plan-preparation-failed",
      "retained compiled carriers cannot produce a fresh exact assembly plan",
    )
  if not _same_fingerprint(fresh_plan.content_fingerprint, plan_content_fingerprint):
    _malformed(
      "solution-plan-mismatch",
      "fresh assembly plan does not match retained solution provenance",
    )
  point = _point_from_evaluation(retained_ledger.program_evaluation)
  try:
    fresh_contributions = assemble_reference_linear(
      model,
      program,
      fresh_plan,
      point,
    )
  except AssemblyEvaluationError:
    _malformed(
      "fresh-assembly-evaluation-failed",
      "retained compiled carriers cannot produce a fresh exact evaluation",
    )
  primary = state.physical.primary_values.values
  free_dofs = program.constraint_plan.free_dofs.values
  reduced_coordinates = np.asarray(primary[free_dofs], dtype=np.float64)
  fresh_ledger = build_balance_ledger(
    model=model,
    program=program,
    plan=fresh_plan,
    contributions=fresh_contributions,
    request_manifest=request_manifest,
    prepared_instance_id=prepared_instance_id,
    transaction_id=transaction_id,
    trial_id=trial_id,
    base_generation=base_generation,
    candidate_generation=candidate_generation,
    reduced_coordinates=reduced_coordinates,
    primary_values=primary,
    ledger_id=retained_ledger.ledger_id,
  )
  projected_constraint_force = prolongation(fresh_plan).T @ (
    fresh_ledger.constraint_force.values
  )
  fresh_operator = dense_operator(fresh_contributions.reduced_operator)
  checks = _fresh_checks(
    retained_ledger,
    fresh_ledger,
    primary,
    projected_constraint_force,
    fresh_operator,
    convergence,
  )
  return VerificationReport(
    passed=all(item.passed for item in checks),
    level="fresh-linear-equilibrium",
    checks=checks,
  )


def _validate_program_evaluation(
  evaluation: object,
  model: CompiledModel,
  program: CompiledProgram,
  full_count: int,
) -> tuple[np.ndarray, ...]:
  if type(evaluation) is not ProgramEvaluation:
    _malformed(
      "malformed-program-evaluation",
      "retained program evaluation has a foreign type",
    )
  _require_instance(
    program.instance_id, evaluation.program_instance_id, "program evaluation"
  )
  _require_instance(
    model.instance_id,
    evaluation.compatible_model_instance_id,
    "program evaluation model",
  )
  if not _same_fingerprint(
    program.content_fingerprint,
    evaluation.program_content_fingerprint,
  ) or not _same_fingerprint(
    model.content_fingerprint,
    evaluation.compatible_model_content_fingerprint,
  ):
    _malformed(
      "solution-fingerprint-mismatch",
      "program evaluation fingerprints do not match the retained owners",
    )
  coordinate_count = len(program.coordinate_names)
  if (
    type(evaluation.coordinate_names) is not tuple
    or evaluation.coordinate_names != program.coordinate_names
  ):
    _malformed(
      "malformed-program-evaluation",
      "program evaluation coordinate order is not exact",
    )
  return (
    _array(
      evaluation.coordinate_values,
      dtype=_FLOAT64,
      shape=(coordinate_count,),
      label="program coordinate values",
    ),
    _array(
      evaluation.prescribed_offsets,
      dtype=_FLOAT64,
      shape=(full_count,),
      label="program prescribed offsets",
    ),
    _array(
      evaluation.prescribed_offset_derivatives,
      dtype=_FLOAT64,
      shape=(full_count, coordinate_count),
      label="program prescribed-offset derivatives",
    ),
    _array(
      evaluation.nodal_force,
      dtype=_FLOAT64,
      shape=(full_count,),
      label="program nodal force",
    ),
    _array(
      evaluation.nodal_force_derivatives,
      dtype=_FLOAT64,
      shape=(full_count, coordinate_count),
      label="program nodal-force derivatives",
    ),
  )


def _program_evaluations_equal(
  left: ProgramEvaluation, right: ProgramEvaluation
) -> bool:
  """Compare bound meaning exactly while keeping storage ownership separate."""
  try:
    require_same_instance(left.program_instance_id, right.program_instance_id)
    require_same_instance(
      left.compatible_model_instance_id,
      right.compatible_model_instance_id,
    )
    return (
      _same_fingerprint(
        left.program_content_fingerprint,
        right.program_content_fingerprint,
      )
      and _same_fingerprint(
        left.compatible_model_content_fingerprint,
        right.compatible_model_content_fingerprint,
      )
      and left.coordinate_names == right.coordinate_names
      and np.array_equal(left.coordinate_values.values, right.coordinate_values.values)
      and np.array_equal(
        left.prescribed_offsets.values,
        right.prescribed_offsets.values,
      )
      and np.array_equal(
        left.prescribed_offset_derivatives.values,
        right.prescribed_offset_derivatives.values,
      )
      and np.array_equal(left.nodal_force.values, right.nodal_force.values)
      and np.array_equal(
        left.nodal_force_derivatives.values,
        right.nodal_force_derivatives.values,
      )
    )
  except (AttributeError, TypeError, ValueError):
    return False


def validate_state_record(
  *,
  state: object,
  model: CompiledModel,
  program: CompiledProgram,
  prepared_instance_id: InstanceId,
  request_manifest: CanonicalManifest,
) -> tuple[np.ndarray, ...]:
  """Validate one exact committed state and return its authoritative arrays."""
  if type(state) is not CommittedAnalysisState:
    _malformed("malformed-committed-state", "committed state has a foreign type")
  _require_instance(prepared_instance_id, state.prepared_instance_id, "committed state")
  if type(state.generation) is not StateGeneration:
    _malformed("malformed-state-generation", "committed generation has a foreign type")
  if type(state.physical) is not PhysicalState:
    _malformed("malformed-physical-state", "physical state has a foreign type")
  physical = state.physical
  _require_instance(
    model.instance_id, physical.model_instance_id, "physical state model"
  )
  if not _same_fingerprint(
    model.content_fingerprint, physical.model_content_fingerprint
  ):
    _malformed(
      "solution-fingerprint-mismatch",
      "physical-state fingerprint does not match the retained model",
    )
  if physical.schema != PHYSICAL_STATE_SCHEMA:
    _malformed("malformed-physical-state", "physical-state schema is not exact")
  _require_generation(state.generation, physical.generation, "physical state")
  full_count = model.physical_state_layout.global_primary_size
  arrays: list[np.ndarray] = [
    _array(
      physical.primary_values,
      dtype=_FLOAT64,
      shape=(full_count,),
      label="physical primary values",
    )
  ]
  if (
    type(physical.material_histories) is not tuple
    or len(physical.material_histories) != len(model.physical_state_layout.block_states)
    or type(physical.formulation_histories) is not tuple
    or len(physical.formulation_histories)
    != len(model.physical_state_layout.block_states)
  ):
    _malformed("malformed-physical-state", "local-history block ownership is not exact")
  for index, layout in enumerate(model.physical_state_layout.block_states):
    arrays.append(
      _array(
        physical.material_histories[index],
        dtype=_FLOAT64,
        shape=layout.material_history_shape,
        label=f"material history block {index}",
      )
    )
    arrays.append(
      _array(
        physical.formulation_histories[index],
        dtype=_FLOAT64,
        shape=layout.formulation_history_shape,
        label=f"formulation history block {index}",
      )
    )

  if type(state.evolution) is not EvolutionState:
    _malformed("malformed-evolution-state", "evolution state has a foreign type")
  evolution = state.evolution
  _require_instance(
    prepared_instance_id, evolution.prepared_instance_id, "evolution state"
  )
  if evolution.schema != EVOLUTION_STATE_SCHEMA or not _manifest_equal(
    evolution.request_manifest,
    request_manifest,
  ):
    _malformed(
      "malformed-evolution-state", "evolution schema or request manifest is not exact"
    )
  _require_generation(state.generation, evolution.generation, "evolution state")
  expected_fields = tuple(
    item.field_id for item in model.physical_state_layout.primary_fields
  )
  if evolution.algebraic_field_ids != expected_fields:
    _malformed(
      "malformed-evolution-state", "algebraic field classification is not exact"
    )
  if (
    type(evolution.accepted_step_index) is not int or evolution.accepted_step_index < 0
  ):
    _malformed("malformed-evolution-state", "accepted step index is invalid")
  arrays.extend(
    _validate_program_evaluation(
      evolution.program_evaluation, model, program, full_count
    )
  )
  arrays.extend(
    (
      _array(
        evolution.predictor,
        dtype=_FLOAT64,
        shape=(full_count,),
        label="evolution predictor",
      ),
      _array(
        evolution.actual_increment,
        dtype=_FLOAT64,
        shape=(full_count,),
        label="evolution actual increment",
      ),
    )
  )
  if type(state.program_history) is not ProgramHistory:
    _malformed("malformed-program-history", "program history has a foreign type")
  history = state.program_history
  _require_instance(program.instance_id, history.program_instance_id, "program history")
  if (
    not _same_fingerprint(
      program.content_fingerprint, history.program_content_fingerprint
    )
    or history.schema != PROGRAM_HISTORY_SCHEMA
    or type(history.entries) is not tuple
    or history.entries != ()
  ):
    _malformed(
      "malformed-program-history", "Phase 1 program history is not exactly empty"
    )
  _require_generation(state.generation, history.generation, "program history")
  return tuple(arrays)


def _validate_ledger_arrays(
  ledger: LinearBalanceLedger,
  *,
  full_count: int,
  reduced_count: int,
  direct_count: int,
  constraint_count: int,
) -> tuple[np.ndarray, ...]:
  return (
    _array(
      ledger.reduced_coordinates,
      dtype=_FLOAT64,
      shape=(reduced_count,),
      label="reduced coordinates",
    ),
    _array(
      ledger.full_primary_values,
      dtype=_FLOAT64,
      shape=(full_count,),
      label="ledger primary values",
    ),
    _array(
      ledger.reduced_primary_image,
      dtype=_FLOAT64,
      shape=(full_count,),
      label="reduced primary image",
    ),
    _array(
      ledger.prescribed_offsets,
      dtype=_FLOAT64,
      shape=(full_count,),
      label="ledger prescribed offsets",
    ),
    _array(
      ledger.external_force, dtype=_FLOAT64, shape=(full_count,), label="external force"
    ),
    _array(
      ledger.internal_force, dtype=_FLOAT64, shape=(full_count,), label="internal force"
    ),
    _array(
      ledger.full_residual, dtype=_FLOAT64, shape=(full_count,), label="full residual"
    ),
    _array(
      ledger.reduced_rhs, dtype=_FLOAT64, shape=(reduced_count,), label="reduced RHS"
    ),
    _array(
      ledger.reduced_internal_force,
      dtype=_FLOAT64,
      shape=(reduced_count,),
      label="reduced internal force",
    ),
    _array(
      ledger.reduced_residual,
      dtype=_FLOAT64,
      shape=(reduced_count,),
      label="reduced residual",
    ),
    _array(
      ledger.constraint_force,
      dtype=_FLOAT64,
      shape=(full_count,),
      label="constraint force",
    ),
    _array(ledger.balance, dtype=_FLOAT64, shape=(full_count,), label="full balance"),
    _array(
      ledger.direct_reaction_dof_indices,
      dtype=_INT64,
      shape=(direct_count,),
      label="direct-reaction DOF indices",
      finite=False,
    ),
    _array(
      ledger.direct_reactions,
      dtype=_FLOAT64,
      shape=(direct_count,),
      label="direct reactions",
    ),
    _array(
      ledger.constraint_violation,
      dtype=_FLOAT64,
      shape=(constraint_count,),
      label="constraint violation",
    ),
    _array(
      ledger.constraint_row_scales,
      dtype=_FLOAT64,
      shape=(constraint_count,),
      label="constraint row scales",
    ),
  )


def _verify_record_data(
  *,
  model: object,
  program: object,
  request: object,
  request_manifest: object,
  prepared_instance_id: object,
  plan_content_fingerprint: object,
  state: object,
  transition: object,
  convergence: object,
  ledger: object,
) -> VerificationReport:
  """Validate retained structure, provenance, storage, and ledger consistency."""
  from pyfem.v3.analysis.contracts import (
    LINEAR_STATIC_VERIFICATION_TOLERANCE,
    AcceptedTransition,
    LinearConvergenceRecord,
    LinearPredictor,
    LinearStatic,
    StepTransaction,
    linear_static_request_manifest,
  )

  if type(model) is not CompiledModel or type(program) is not CompiledProgram:
    _malformed(
      "malformed-solution-owner", "solution model/program carriers are not exact"
    )
  try:
    _validated_model(model)
    _validated_program(program)
  except _BoundaryError:
    _malformed(
      "malformed-solution-owner",
      "solution model/program carriers fail complete exact validation",
    )
  if type(request) is not LinearStatic:
    _malformed(
      "malformed-solution-request", "solution request is not exactly LinearStatic"
    )
  expected_manifest = linear_static_request_manifest(request)
  if not _manifest_equal(request_manifest, expected_manifest):
    _malformed("malformed-solution-request", "solution request manifest is not exact")
  if type(prepared_instance_id) is not InstanceId:
    _malformed("malformed-solution-owner", "prepared-analysis identity is malformed")
  if type(plan_content_fingerprint) is not ContentFingerprint:
    _malformed("malformed-solution-owner", "plan fingerprint is malformed")
  if not _fingerprint_matches_manifest(
    model.content_fingerprint, model.provenance.manifest
  ):
    _malformed("malformed-solution-owner", "model manifest and fingerprint disagree")
  if not _fingerprint_matches_manifest(
    program.content_fingerprint, program.provenance.manifest
  ):
    _malformed("malformed-solution-owner", "program manifest and fingerprint disagree")
  _require_instance(
    model.instance_id, program.compatible_model_instance_id, "solution model/program"
  )
  if not _same_fingerprint(
    model.content_fingerprint,
    program.compatible_model_content_fingerprint,
  ):
    _malformed(
      "solution-fingerprint-mismatch", "program is not content-compatible with model"
    )

  state_arrays = validate_state_record(
    state=state,
    model=model,
    program=program,
    prepared_instance_id=prepared_instance_id,
    request_manifest=request_manifest,
  )
  if type(transition) is not AcceptedTransition:
    _malformed("malformed-transition-record", "accepted transition has a foreign type")
  if type(transition.transaction) is not StepTransaction:
    _malformed(
      "malformed-transition-record",
      "accepted transition does not retain its exact StepTransaction",
    )
  transaction = transition.transaction
  if type(transaction.transaction_id) is not InstanceId:
    _malformed(
      "malformed-transition-record",
      "accepted transaction identity is malformed",
    )
  _require_instance(
    transaction.transaction_id,
    transition.transaction_id,
    "accepted transaction identity",
  )
  _require_instance(
    prepared_instance_id,
    transaction.prepared_instance_id,
    "accepted transaction prepared analysis",
  )
  _require_instance(
    model.instance_id,
    transaction.model_instance_id,
    "accepted transaction model",
  )
  _require_instance(
    program.instance_id,
    transaction.program_instance_id,
    "accepted transaction program",
  )
  if (
    not _same_fingerprint(
      model.content_fingerprint,
      transaction.model_content_fingerprint,
    )
    or not _same_fingerprint(
      program.content_fingerprint,
      transaction.program_content_fingerprint,
    )
    or not _same_fingerprint(
      plan_content_fingerprint,
      transaction.plan_content_fingerprint,
    )
    or not _manifest_equal(request_manifest, transaction.request_manifest)
  ):
    _malformed(
      "malformed-transition-record",
      "accepted transaction content provenance is not exact",
    )
  if (
    type(transaction.retry) is not int
    or transaction.retry != 0
    or type(transaction.cutback) is not int
    or transaction.cutback != 0
  ):
    _malformed(
      "malformed-transition-record",
      "Phase 1 transaction retry and cutback identities must be exactly zero",
    )
  if type(transaction.predictor) is not LinearPredictor:
    _malformed(
      "malformed-transition-record",
      "accepted transaction predictor has a foreign type",
    )
  _require_instance(
    prepared_instance_id, transition.prepared_instance_id, "accepted transition"
  )
  if type(transition.trial_id) is not InstanceId:
    _malformed("malformed-transition-record", "accepted trial identity is malformed")
  _require_generation(
    transaction.base_generation,
    transition.base_generation,
    "accepted transaction base",
  )
  transaction_state_arrays = validate_state_record(
    state=transaction.base_state,
    model=model,
    program=program,
    prepared_instance_id=prepared_instance_id,
    request_manifest=request_manifest,
  )
  if (
    transaction.base_state.generation.ordinal != 0
    or transaction.base_state.evolution.accepted_step_index != 0
  ):
    _malformed(
      "malformed-transition-record",
      "Phase 1 accepted transaction must retain an initialized generation-zero base",
    )
  _require_generation(
    transaction.base_generation,
    transaction.base_state.generation,
    "accepted transaction base state",
  )
  _require_generation(
    state.generation, transition.candidate_generation, "accepted transition candidate"
  )
  if (
    type(transition.committed_state) is not CommittedAnalysisState
    or transition.committed_state is not state
  ):
    _malformed(
      "malformed-transition-record",
      "accepted transition does not own the exact final state",
    )
  _require_generation(
    state.generation, transition.committed_state.generation, "accepted transition state"
  )
  _require_successor(
    transition.base_generation, transition.candidate_generation, "accepted transition"
  )
  if transition.accepted is not True:
    _malformed("malformed-transition-record", "solution transition was not accepted")
  full_count = model.dofs.global_size
  predictor = _array(
    transaction.predictor.full_increment,
    dtype=_FLOAT64,
    shape=(full_count,),
    label="transaction linear predictor",
  )
  transaction_evaluation_arrays = _validate_program_evaluation(
    transaction.target_evaluation,
    model,
    program,
    full_count,
  )
  base_primary = _array(
    transition.base_primary_values,
    dtype=_FLOAT64,
    shape=(full_count,),
    label="transition base primary values",
  )
  actual_increment = _array(
    transition.actual_increment,
    dtype=_FLOAT64,
    shape=(full_count,),
    label="transition actual increment",
  )
  transition_evaluation_arrays = _validate_program_evaluation(
    transition.target_evaluation,
    model,
    program,
    full_count,
  )
  if not _program_evaluations_equal(
    transaction.target_evaluation,
    transition.target_evaluation,
  ):
    _malformed(
      "malformed-transition-record",
      "transaction and accepted target evaluations disagree",
    )
  expected_predictor = (
    transaction.target_evaluation.prescribed_offsets.values
    - transaction.base_state.physical.primary_values.values
  )
  if (
    _difference_norm(predictor, expected_predictor) != 0.0
    or _difference_norm(predictor, state.evolution.predictor.values) != 0.0
  ):
    _malformed(
      "malformed-transition-record",
      "transaction and evolution predictors are not exact",
    )
  if (
    _difference_norm(
      base_primary,
      transaction.base_state.physical.primary_values.values,
    )
    != 0.0
  ):
    _malformed(
      "malformed-transition-record",
      "retained base primary values disagree with the exact transaction base",
    )
  if (
    _difference_norm(
      base_primary + actual_increment, state.physical.primary_values.values
    )
    != 0.0
  ):
    _malformed(
      "malformed-transition-record",
      "accepted increment does not reconstruct final fields",
    )
  if state.evolution.accepted_step_index != 1:
    _malformed(
      "malformed-transition-record",
      "direct Phase 1 solve must accept exactly step index one",
    )
  if _difference_norm(actual_increment, state.evolution.actual_increment.values) != 0.0:
    _malformed(
      "malformed-transition-record", "state and transition increments disagree"
    )

  if type(convergence) is not LinearConvergenceRecord:
    _malformed("malformed-convergence-record", "convergence record has a foreign type")
  if (
    convergence.converged is not True
    or type(convergence.iteration_count) is not int
    or convergence.iteration_count != 1
    or convergence.verification_tolerance != LINEAR_STATIC_VERIFICATION_TOLERANCE
    or any(
      type(flag) is not bool
      for flag in (
        convergence.factorization_performed,
        convergence.factorization_reused,
        convergence.factorization_bypassed,
      )
    )
    or sum(
      (
        convergence.factorization_performed,
        convergence.factorization_reused,
        convergence.factorization_bypassed,
      )
    )
    != 1
  ):
    _malformed(
      "malformed-convergence-record", "linear convergence record is inconsistent"
    )
  scalar_values = (
    convergence.operator_infinity_norm,
    convergence.reduced_residual_norm,
    convergence.verification_tolerance,
  )
  if any(
    type(value) is not float or not math.isfinite(value) or value < 0.0
    for value in scalar_values
  ):
    _malformed("malformed-convergence-record", "convergence scalars are invalid")
  if convergence.minimum_unscaled_pivot is not None and (
    type(convergence.minimum_unscaled_pivot) is not float
    or not math.isfinite(convergence.minimum_unscaled_pivot)
    or convergence.minimum_unscaled_pivot <= 0.0
  ):
    _malformed("malformed-convergence-record", "Cholesky pivot record is invalid")
  if type(convergence.backend_policy) is not str or not convergence.backend_policy:
    _malformed("malformed-convergence-record", "backend policy record is invalid")

  if type(ledger) is not LinearBalanceLedger:
    _malformed("malformed-balance-ledger", "balance ledger has a foreign type")
  if type(ledger.ledger_id) is not InstanceId:
    _malformed("malformed-balance-ledger", "balance ledger identity is malformed")
  _require_instance(prepared_instance_id, ledger.prepared_instance_id, "balance ledger")
  _require_instance(model.instance_id, ledger.model_instance_id, "balance ledger model")
  _require_instance(
    program.instance_id, ledger.program_instance_id, "balance ledger program"
  )
  _require_instance(
    transition.transaction_id, ledger.transaction_id, "balance ledger transaction"
  )
  _require_instance(transition.trial_id, ledger.trial_id, "balance ledger trial")
  _require_generation(
    transition.base_generation, ledger.base_generation, "balance ledger base"
  )
  _require_generation(
    transition.candidate_generation,
    ledger.candidate_generation,
    "balance ledger candidate",
  )
  if (
    not _same_fingerprint(model.content_fingerprint, ledger.model_content_fingerprint)
    or not _same_fingerprint(
      program.content_fingerprint, ledger.program_content_fingerprint
    )
    or not _same_fingerprint(plan_content_fingerprint, ledger.plan_content_fingerprint)
    or not _manifest_equal(request_manifest, ledger.request_manifest)
  ):
    _malformed(
      "solution-fingerprint-mismatch", "balance-ledger provenance is not exact"
    )
  ledger_evaluation_arrays = _validate_program_evaluation(
    ledger.program_evaluation,
    model,
    program,
    full_count,
  )
  if not _program_evaluations_equal(
    transition.target_evaluation,
    state.evolution.program_evaluation,
  ) or not _program_evaluations_equal(
    transition.target_evaluation,
    ledger.program_evaluation,
  ):
    _malformed(
      "malformed-balance-ledger",
      "retained transaction, state, and ledger program evaluations disagree",
    )
  direct_count = len(
    tuple(
      item
      for item in program.meaning_witness.constraints
      if type(item) is ProgramPrescribedWitness
    )
  )
  constraint_count = len(program.meaning_witness.constraints)
  reduced_count = program.constraint_plan.reduced_dof_count
  ledger_arrays = _validate_ledger_arrays(
    ledger,
    full_count=full_count,
    reduced_count=reduced_count,
    direct_count=direct_count,
    constraint_count=constraint_count,
  )
  if ledger.constraint_ids != tuple(
    item.id for item in program.meaning_witness.constraints
  ):
    _malformed("malformed-balance-ledger", "constraint row identities are not exact")
  direct_indices = ledger.direct_reaction_dof_indices.values
  if bool(np.any(direct_indices < 0)) or bool(np.any(direct_indices >= full_count)):
    _malformed("malformed-balance-ledger", "direct-reaction indices are out of range")
  scalar_ledger_values = (
    ledger.constraint_work,
    ledger.external_work,
    ledger.internal_work,
    ledger.full_force_scale,
    ledger.reduced_force_scale,
    ledger.reconstruction_scale,
    ledger.work_scale,
    ledger.full_operator_infinity_norm,
    ledger.reduced_operator_infinity_norm,
    ledger.prolongation_transpose_infinity_norm,
    ledger.verification_tolerance,
  )
  if any(
    type(value) is not float or not math.isfinite(value)
    for value in scalar_ledger_values
  ):
    _malformed(
      "malformed-balance-ledger", "ledger scalar evidence is nonfinite or mistyped"
    )
  if any(value < 0.0 for value in scalar_ledger_values[3:]):
    _malformed("malformed-balance-ledger", "ledger normalization is negative")
  if ledger.verification_tolerance != LINEAR_STATIC_VERIFICATION_TOLERANCE:
    _malformed(
      "malformed-balance-ledger", "ledger tolerance is not the frozen request tolerance"
    )
  if reduced_count == 0:
    if (
      convergence.factorization_bypassed is not True
      or convergence.minimum_unscaled_pivot is not None
      or convergence.operator_infinity_norm != 0.0
    ):
      _malformed(
        "malformed-convergence-record",
        "zero-free solve must carry the exact factorization-bypass record",
      )
  elif convergence.factorization_bypassed or convergence.minimum_unscaled_pivot is None:
    _malformed(
      "malformed-convergence-record",
      "nonempty reduced solve must carry a Cholesky pivot record",
    )

  record_external_work = _dot(
    ledger.external_force.values,
    ledger.full_primary_values.values,
    "record external work",
  )
  record_internal_work = _dot(
    ledger.internal_force.values,
    ledger.full_primary_values.values,
    "record internal work",
  )
  record_constraint_work = _dot(
    ledger.constraint_force.values,
    ledger.full_primary_values.values,
    "record constraint work",
  )
  primary_norm = _infinity_norm(ledger.full_primary_values.values)
  offset_norm = _infinity_norm(ledger.prescribed_offsets.values)
  expected_full_scale = max(
    _infinity_norm(ledger.external_force.values),
    _infinity_norm(ledger.internal_force.values),
    _infinity_norm(ledger.constraint_force.values),
    _bounded_product(ledger.full_operator_infinity_norm, primary_norm),
  )
  expected_reduced_scale = max(
    _infinity_norm(ledger.reduced_rhs.values),
    _infinity_norm(ledger.reduced_internal_force.values),
    _bounded_product(
      ledger.prolongation_transpose_infinity_norm,
      expected_full_scale,
    ),
  )
  expected_reconstruction_scale = max(
    primary_norm,
    _infinity_norm(ledger.reduced_primary_image.values),
    offset_norm,
  )
  expected_work_scale = max(
    abs(record_external_work),
    abs(record_internal_work),
    abs(record_constraint_work),
    _bounded_product(expected_full_scale, max(primary_norm, offset_norm)),
  )

  authoritative_arrays = (
    *transaction_state_arrays,
    predictor,
    *transaction_evaluation_arrays,
    *state_arrays,
    base_primary,
    actual_increment,
    *transition_evaluation_arrays,
    *ledger_evaluation_arrays,
    *ledger_arrays,
  )
  for index, left in enumerate(authoritative_arrays):
    for right in authoritative_arrays[index + 1 :]:
      if left is right or np.shares_memory(left, right):
        _malformed(
          "aliased-solution-storage",
          "published state, transition, evaluation, and ledger arrays must "
          "be storage-disjoint",
        )

  record_checks = (
    verification_check(
      "record_primary_values",
      _difference_norm(
        ledger.full_primary_values.values, state.physical.primary_values.values
      ),
      ledger.reconstruction_scale,
      ledger.verification_tolerance,
    ),
    _boolean_check(
      "record_program_prescribed_offsets",
      bool(
        np.array_equal(
          ledger.prescribed_offsets.values,
          ledger.program_evaluation.prescribed_offsets.values,
        )
      ),
      ledger.verification_tolerance,
    ),
    _boolean_check(
      "record_program_nodal_force",
      bool(
        np.array_equal(
          ledger.external_force.values,
          ledger.program_evaluation.nodal_force.values,
        )
      ),
      ledger.verification_tolerance,
    ),
    verification_check(
      "record_full_residual",
      _difference_norm(
        ledger.full_residual.values,
        ledger.external_force.values - ledger.internal_force.values,
      ),
      ledger.full_force_scale,
      ledger.verification_tolerance,
    ),
    verification_check(
      "record_constraint_force",
      _difference_norm(
        ledger.constraint_force.values,
        -ledger.full_residual.values,
      ),
      ledger.full_force_scale,
      ledger.verification_tolerance,
    ),
    verification_check(
      "record_full_balance",
      _difference_norm(
        ledger.balance.values,
        ledger.external_force.values
        + ledger.constraint_force.values
        - ledger.internal_force.values,
      ),
      ledger.full_force_scale,
      ledger.verification_tolerance,
    ),
    verification_check(
      "record_reduced_residual",
      _difference_norm(
        ledger.reduced_residual.values,
        ledger.reduced_rhs.values - ledger.reduced_internal_force.values,
      ),
      ledger.reduced_force_scale,
      ledger.verification_tolerance,
    ),
    verification_check(
      "record_direct_reactions",
      _difference_norm(
        ledger.direct_reactions.values,
        ledger.constraint_force.values[ledger.direct_reaction_dof_indices.values],
      ),
      ledger.full_force_scale,
      ledger.verification_tolerance,
    ),
    verification_check(
      "record_external_work",
      abs(ledger.external_work - record_external_work),
      ledger.work_scale,
      ledger.verification_tolerance,
    ),
    verification_check(
      "record_internal_work",
      abs(ledger.internal_work - record_internal_work),
      ledger.work_scale,
      ledger.verification_tolerance,
    ),
    verification_check(
      "record_constraint_work",
      abs(ledger.constraint_work - record_constraint_work),
      ledger.work_scale,
      ledger.verification_tolerance,
    ),
    verification_check(
      "record_full_force_scale",
      abs(ledger.full_force_scale - expected_full_scale),
      expected_full_scale,
      ledger.verification_tolerance,
    ),
    verification_check(
      "record_reduced_force_scale",
      abs(ledger.reduced_force_scale - expected_reduced_scale),
      expected_reduced_scale,
      ledger.verification_tolerance,
    ),
    verification_check(
      "record_reconstruction_scale",
      abs(ledger.reconstruction_scale - expected_reconstruction_scale),
      expected_reconstruction_scale,
      ledger.verification_tolerance,
    ),
    verification_check(
      "record_work_scale",
      abs(ledger.work_scale - expected_work_scale),
      expected_work_scale,
      ledger.verification_tolerance,
    ),
    verification_check(
      "record_convergence_residual",
      abs(
        convergence.reduced_residual_norm
        - _infinity_norm(ledger.reduced_residual.values)
      ),
      ledger.reduced_force_scale,
      ledger.verification_tolerance,
    ),
    verification_check(
      "record_work_balance",
      abs(ledger.external_work + ledger.constraint_work - ledger.internal_work),
      ledger.work_scale,
      ledger.verification_tolerance,
    ),
  )
  return VerificationReport(
    passed=all(item.passed for item in record_checks),
    level="record-only",
    checks=record_checks,
  )


def verify_record_data(
  *,
  model: object,
  program: object,
  request: object,
  request_manifest: object,
  prepared_instance_id: object,
  plan_content_fingerprint: object,
  state: object,
  transition: object,
  convergence: object,
  ledger: object,
) -> VerificationReport:
  """Provide one total structured boundary around retained-result validation."""
  try:
    return _verify_record_data(
      model=model,
      program=program,
      request=request,
      request_manifest=request_manifest,
      prepared_instance_id=prepared_instance_id,
      plan_content_fingerprint=plan_content_fingerprint,
      state=state,
      transition=transition,
      convergence=convergence,
      ledger=ledger,
    )
  except SolutionVerificationError:
    raise
  except (
    AttributeError,
    IndexError,
    KeyError,
    OverflowError,
    RecursionError,
    TypeError,
    ValueError,
  ) as error:
    raise SolutionVerificationError(
      (
        SolutionVerificationDiagnostic(
          "malformed-solution-record",
          "retained result is partially initialized or structurally malformed",
        ),
      )
    ) from error
