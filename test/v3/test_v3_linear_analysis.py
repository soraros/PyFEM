# SPDX-License-Identifier: MIT

"""Correctness matrix for the public verified Q8 linear-static flow."""

from __future__ import annotations

import math
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from fractions import Fraction
from threading import Barrier, Event, current_thread, main_thread
from typing import NoReturn
from uuid import UUID

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from pyfem.v3.analysis import (
  AcceptedTransition,
  LinearConvergenceRecord,
  LinearPredictor,
  StepTransaction,
)
from pyfem.v3.api import (
  AnalysisPreparationError,
  AnalysisSolveError,
  LinearStatic,
  PreparedAnalysis,
  Solution,
  SolutionVerificationError,
  StateTransactionError,
  prepare_analysis,
  solve,
)
from pyfem.v3.assembly import (
  CanonicalCooOperator,
  LinearStaticContributionRequest,
  LinearStaticContributions,
  assemble_reference_linear,
  prepare_assembly_plan,
)
from pyfem.v3.compile import compile_model, compile_program, q8_reference_registry
from pyfem.v3.model import (
  CommittedAnalysisState,
  CompiledModel,
  CompiledProgram,
  EvolutionState,
  FinalizedArray,
  InstanceId,
  PhysicalState,
  ProgramEvaluation,
  ProgramHistory,
  StateGeneration,
)
from pyfem.v3.results.contracts import LinearBalanceLedger
from pyfem.v3.results.verification import validate_state_record
from pyfem.v3.spec import (
  AffineCoefficientSpec,
  AffineTieSpec,
  AffineValueSpec,
  CellBlockSpec,
  CellRef,
  CellSpec,
  DofRef,
  FieldSpec,
  MaterialParameterSpec,
  MaterialSpec,
  MeshSpec,
  ModelSpec,
  NodalLoadSpec,
  NodeSpec,
  PrescribedDofSpec,
  ProgramCoordinateSpec,
  ProgramCoordinateValue,
  ProgramPoint,
  ProgramSpec,
  RegionSpec,
  SourceContext,
)

_UNIT_COORDINATES = (
  (0.0, 0.0),
  (0.5, 0.0),
  (1.0, 0.0),
  (1.0, 0.5),
  (1.0, 1.0),
  (0.5, 1.0),
  (0.0, 1.0),
  (0.0, 0.5),
)

# Literal independent oracle M for K=M/360. Production assembly and solve do not
# manufacture this fixture or the rational displacement below.
_Q8_INTEGER_OPERATOR = np.array(
  [
    [312, 85, -308, -100, 146, 15, -104, -20, 138, 35, -172, -20, 124, -15, -136, 20],
    [85, 312, 20, -136, -15, 124, -20, -172, 35, 138, -20, -104, 15, 146, -100, -308],
    [-308, 20, 736, 0, -308, -20, 0, -80, -172, -20, 224, 0, -172, 20, 0, 80],
    [-100, -136, 0, 512, 100, -136, -80, 0, -20, -104, 0, -32, 20, -104, 80, 0],
    [146, -15, -308, 100, 312, -85, -136, -20, 124, 15, -172, 20, 138, -35, -104, 20],
    [15, 124, -20, -136, -85, 312, 100, -308, -15, 146, 20, -104, -35, 138, 20, -172],
    [-104, -20, 0, -80, -136, 100, 512, 0, -136, -100, 0, 80, -104, 20, -32, 0],
    [-20, -172, -80, 0, -20, -308, 0, 736, 20, -308, 80, 0, 20, -172, 0, 224],
    [138, 35, -172, -20, 124, -15, -136, 20, 312, 85, -308, -100, 146, 15, -104, -20],
    [35, 138, -20, -104, 15, 146, -100, -308, 85, 312, 20, -136, -15, 124, -20, -172],
    [-172, -20, 224, 0, -172, 20, 0, 80, -308, 20, 736, 0, -308, -20, 0, -80],
    [-20, -104, 0, -32, 20, -104, 80, 0, -100, -136, 0, 512, 100, -136, -80, 0],
    [124, 15, -172, 20, 138, -35, -104, 20, 146, -15, -308, 100, 312, -85, -136, -20],
    [-15, 146, 20, -104, -35, 138, 20, -172, 15, 124, -20, -136, -85, 312, 100, -308],
    [-136, -100, 0, 80, -104, 20, -32, 0, -104, -20, 0, -80, -136, 100, 512, 0],
    [20, -308, 80, 0, 20, -172, 0, 224, -20, -172, -80, 0, -20, -308, 0, 736],
  ],
  dtype=np.float64,
)

_RATIONAL_DISPLACEMENT = np.array(
  [
    0,
    0,
    29 / 12,
    67 / 24,
    29 / 6,
    35 / 6,
    -23 / 24,
    33 / 4,
    -7,
    32 / 3,
    -77 / 12,
    27 / 8,
    -35 / 6,
    -7 / 6,
    -37 / 24,
    -7 / 12,
  ],
  dtype=np.float64,
)


def _source(label: str) -> SourceContext:
  return SourceContext(source=label)


def _dof(node_id: int, component: str) -> DofRef:
  return DofRef(node_id, "displacement", component)


def _affine(
  constant: float = 0.0,
  *coefficients: tuple[str, float],
) -> AffineValueSpec:
  return AffineValueSpec(
    constant,
    tuple(
      AffineCoefficientSpec(name, value, _source(f"coefficient:{name}"))
      for name, value in coefficients
    ),
    _source("affine"),
  )


def _model_spec(*, youngs_modulus: float = 1.0) -> ModelSpec:
  nodes = tuple(
    NodeSpec(index, point, _source(f"node:{index}"))
    for index, point in enumerate(_UNIT_COORDINATES, start=1)
  )
  cell = CellSpec("cell", tuple(range(1, 9)), _source("cell"))
  block = CellBlockSpec(
    "block",
    "quadrilateral",
    2,
    2,
    "serendipity-quad8",
    (cell,),
    _source("block"),
  )
  field = FieldSpec(
    "displacement",
    ("x", "y"),
    "node",
    _source("field"),
  )
  material = MaterialSpec(
    "material",
    "plane-stress-linear-elastic",
    (
      MaterialParameterSpec(
        "youngs_modulus",
        youngs_modulus,
        _source("material:E"),
      ),
      MaterialParameterSpec(
        "poisson_ratio",
        0.0,
        _source("material:nu"),
      ),
    ),
    _source("material"),
  )
  region = RegionSpec(
    "region",
    (CellRef("block", "cell"),),
    ("displacement",),
    "material",
    "small-strain-continuum",
    "gauss-3x3",
    _source("region"),
  )
  return ModelSpec(
    MeshSpec(nodes, (block,), _source("mesh")),
    (field,),
    (material,),
    (region,),
    _source("model"),
  )


def _rational_program(
  *,
  split_load: bool = False,
  constrained_load: float = 0.0,
  load_value: float = 1.0,
) -> ProgramSpec:
  loads = (
    (
      NodalLoadSpec(
        "load-a",
        _dof(5, "y"),
        _affine(0.25 * load_value),
        _source("load-a"),
      ),
      NodalLoadSpec(
        "load-b",
        _dof(5, "y"),
        _affine(0.75 * load_value),
        _source("load-b"),
      ),
    )
    if split_load
    else (
      NodalLoadSpec(
        "load",
        _dof(5, "y"),
        _affine(load_value),
        _source("load"),
      ),
    )
  )
  if constrained_load != 0.0:
    loads = (
      *loads,
      NodalLoadSpec(
        "constrained-load",
        _dof(1, "x"),
        _affine(constrained_load),
        _source("constrained-load"),
      ),
    )
  return ProgramSpec(
    coordinates=(ProgramCoordinateSpec("lambda", "load", _source("lambda")),),
    constraints=(
      PrescribedDofSpec("fix-x", _dof(1, "x"), _affine(), _source("fix-x")),
      PrescribedDofSpec("fix-y", _dof(1, "y"), _affine(), _source("fix-y")),
      AffineTieSpec(
        "tie",
        _dof(3, "y"),
        _dof(3, "x"),
        1.0,
        _affine(0.0, ("lambda", 1.0)),
        _source("tie"),
      ),
    ),
    loads=loads,
    source=_source("program"),
  )


def _point(value: float) -> ProgramPoint:
  return ProgramPoint((ProgramCoordinateValue("lambda", value),))


def _prepared(
  program_spec: ProgramSpec | None = None,
  *,
  youngs_modulus: float = 1.0,
) -> tuple[CompiledModel, CompiledProgram, PreparedAnalysis]:
  model = compile_model(
    _model_spec(youngs_modulus=youngs_modulus),
    q8_reference_registry(),
  )
  program = compile_program(
    model,
    _rational_program() if program_spec is None else program_spec,
  )
  return model, program, prepare_analysis(model, program, LinearStatic())


def _solve_rational() -> tuple[
  CompiledModel,
  CompiledProgram,
  PreparedAnalysis,
  CommittedAnalysisState,
  Solution,
]:
  model, program, analysis = _prepared()
  initial = analysis.initialize(point=_point(0.0))
  solution = analysis.solve(initial=initial, point=_point(1.0))
  return model, program, analysis, initial, solution


def _shares_any(left: tuple[np.ndarray, ...], right: tuple[np.ndarray, ...]) -> bool:
  return any(np.shares_memory(a, b) for a in left for b in right)


def _state_arrays(solution: Solution) -> tuple[np.ndarray, ...]:
  state = solution.state
  return (
    state.physical.primary_values.values,
    *(item.values for item in state.physical.material_histories),
    *(item.values for item in state.physical.formulation_histories),
    state.evolution.predictor.values,
    state.evolution.actual_increment.values,
    state.evolution.program_evaluation.coordinate_values.values,
    state.evolution.program_evaluation.prescribed_offsets.values,
    state.evolution.program_evaluation.prescribed_offset_derivatives.values,
    state.evolution.program_evaluation.nodal_force.values,
    state.evolution.program_evaluation.nodal_force_derivatives.values,
  )


def _ledger_arrays(solution: Solution) -> tuple[np.ndarray, ...]:
  ledger = solution.ledger
  return tuple(
    item.values
    for item in (
      ledger.reduced_coordinates,
      ledger.full_primary_values,
      ledger.reduced_primary_image,
      ledger.prescribed_offsets,
      ledger.external_force,
      ledger.internal_force,
      ledger.full_residual,
      ledger.reduced_rhs,
      ledger.reduced_internal_force,
      ledger.reduced_residual,
      ledger.constraint_force,
      ledger.balance,
      ledger.direct_reaction_dof_indices,
      ledger.direct_reactions,
      ledger.constraint_violation,
      ledger.constraint_row_scales,
    )
  )


def _coherent_primary_copy(
  solution: Solution,
  index: int,
  increment: float,
) -> Solution:
  primary = np.array(solution.state.physical.primary_values.values, copy=True)
  primary[index] += increment
  base = solution.transition.base_primary_values.values
  actual_increment = primary - base
  physical = replace(
    solution.state.physical,
    primary_values=FinalizedArray(primary, dtype=np.float64),
  )
  evolution = replace(
    solution.state.evolution,
    actual_increment=FinalizedArray(actual_increment, dtype=np.float64),
  )
  state = replace(solution.state, physical=physical, evolution=evolution)
  transition = replace(
    solution.transition,
    actual_increment=FinalizedArray(actual_increment, dtype=np.float64),
    committed_state=state,
  )
  ledger = replace(
    solution.ledger,
    full_primary_values=FinalizedArray(primary, dtype=np.float64),
    external_work=float(solution.ledger.external_force.values @ primary),
    internal_work=float(solution.ledger.internal_force.values @ primary),
    constraint_work=float(solution.ledger.constraint_force.values @ primary),
  )
  return replace(solution, state=state, transition=transition, ledger=ledger)


def _evaluation_copy(
  evaluation: object,
  *,
  field: str,
  values: np.ndarray,
) -> object:
  arrays = {
    name: FinalizedArray(
      values
      if name == field
      else np.array(getattr(evaluation, name).values, copy=True),
      dtype=np.float64,
    )
    for name in (
      "coordinate_values",
      "prescribed_offsets",
      "prescribed_offset_derivatives",
      "nodal_force",
      "nodal_force_derivatives",
    )
  }
  return replace(evaluation, **arrays)


def _coherent_target_evaluation_copy(
  solution: Solution,
  *,
  field: str,
  values: np.ndarray,
  ledger_changes: dict[str, object] | None = None,
) -> Solution:
  transaction = replace(
    solution.transition.transaction,
    target_evaluation=_evaluation_copy(
      solution.transition.transaction.target_evaluation,
      field=field,
      values=values,
    ),
  )
  evolution = replace(
    solution.state.evolution,
    program_evaluation=_evaluation_copy(
      solution.state.evolution.program_evaluation,
      field=field,
      values=values,
    ),
  )
  state = replace(solution.state, evolution=evolution)
  transition = replace(
    solution.transition,
    transaction=transaction,
    target_evaluation=_evaluation_copy(
      solution.transition.target_evaluation,
      field=field,
      values=values,
    ),
    committed_state=state,
  )
  ledger = replace(
    solution.ledger,
    program_evaluation=_evaluation_copy(
      solution.ledger.program_evaluation,
      field=field,
      values=values,
    ),
    **({} if ledger_changes is None else ledger_changes),
  )
  return replace(solution, state=state, transition=transition, ledger=ledger)


def _operator_values(
  operator: CanonicalCooOperator,
  dense: np.ndarray,
) -> FinalizedArray:
  return FinalizedArray(
    dense[operator.row_indices.values, operator.column_indices.values],
    dtype=np.float64,
  )


def test_rational_one_cell_solution_ledgers_and_rigid_derivative() -> None:
  model, _, _, initial, solution = _solve_rational()
  assert initial.generation.ordinal == 0
  assert solution.state.generation.ordinal == 1
  np.testing.assert_allclose(
    solution.primary_values.values,
    _RATIONAL_DISPLACEMENT,
    rtol=0.0,
    atol=4.0e-14,
  )
  expected_internal = np.array(
    [-1, 0, 0, 0, 1, -1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    dtype=np.float64,
  )
  expected_residual = np.array(
    [1, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    dtype=np.float64,
  )
  expected_constraint = np.array(
    [-1, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    dtype=np.float64,
  )
  np.testing.assert_allclose(
    solution.ledger.internal_force.values,
    expected_internal,
    rtol=0.0,
    atol=1.0e-14,
  )
  np.testing.assert_allclose(
    solution.ledger.full_residual.values,
    expected_residual,
    rtol=0.0,
    atol=1.0e-14,
  )
  np.testing.assert_allclose(
    solution.ledger.constraint_force.values,
    expected_constraint,
    rtol=0.0,
    atol=1.0e-14,
  )
  np.testing.assert_allclose(
    solution.ledger.direct_reactions.values,
    (-1.0, 0.0),
    rtol=0.0,
    atol=5.0e-15,
  )
  assert solution.ledger.constraint_work == pytest.approx(-1.0, abs=6.0e-14)
  assert solution.ledger.external_work == pytest.approx(32 / 3, abs=5.0e-14)
  assert solution.ledger.internal_work == pytest.approx(29 / 3, abs=8.0e-14)
  assert (
    solution.ledger.external_work + solution.ledger.constraint_work
    == pytest.approx(
      solution.ledger.internal_work,
      abs=8.0e-14,
    )
  )
  np.testing.assert_allclose(
    solution.ledger.reduced_residual.values,
    0.0,
    rtol=0.0,
    atol=7.0e-15,
  )

  rotation = np.array(
    [component for x, y in _UNIT_COORDINATES for component in (-y, x)],
    dtype=np.float64,
  )
  np.testing.assert_array_equal(_Q8_INTEGER_OPERATOR @ rotation, np.zeros(16))
  assert float(solution.ledger.constraint_force.values @ rotation) == pytest.approx(
    -1.0,
    abs=5.0e-15,
  )
  total_force = (
    solution.ledger.external_force.values + solution.ledger.constraint_force.values
  ).reshape(8, 2)
  np.testing.assert_allclose(total_force.sum(axis=0), 0.0, rtol=0.0, atol=1.0e-14)
  coordinates = model.mesh.coordinates.values
  moment = np.sum(
    coordinates[:, 0] * total_force[:, 1] - coordinates[:, 1] * total_force[:, 0]
  )
  assert float(moment) == pytest.approx(0.0, abs=1.0e-14)
  assert solution.verify_record().passed
  assert solution.verify().passed


def test_zero_load_nonzero_rigid_offset_and_fully_prescribed_bypass() -> None:
  constraints = tuple(
    PrescribedDofSpec(
      f"u-{node}-{component}",
      _dof(node, component),
      _affine(2.0 if component == "x" else -3.0),
      _source(f"u-{node}-{component}"),
    )
    for node in range(1, 9)
    for component in ("x", "y")
  )
  model, program, analysis = _prepared(ProgramSpec(constraints=constraints))
  del model, program
  initial = analysis.initialize(point=ProgramPoint())
  solution = analysis.solve(initial=initial, point=ProgramPoint())
  expected = np.tile((2.0, -3.0), 8)
  np.testing.assert_array_equal(initial.physical.primary_values.values, expected)
  np.testing.assert_array_equal(solution.primary_values.values, expected)
  assert solution.ledger.reduced_coordinates.values.shape == (0,)
  assert solution.convergence.factorization_bypassed
  statistics = analysis.workspace_statistics()
  assert statistics.factorization_count == 0
  assert statistics.factorization_reuse_count == 0
  assert statistics.zero_free_bypass_count == 1
  report = solution.verify()
  assert report.passed
  assert report.check("backend_zero_free_bypass").passed
  assert report.check("backend_minimum_pivot").passed


def test_additive_and_constrained_dof_loads_keep_direct_reaction_semantics() -> None:
  _, _, analysis = _prepared(_rational_program(split_load=True, constrained_load=2.0))
  solution = analysis.solve(initial_point=_point(0.0), point=_point(1.0))
  assert solution.ledger.external_force.values[8 + 1] == 1.0
  assert solution.ledger.external_force.values[0] == 2.0
  np.testing.assert_allclose(
    solution.primary_values.values,
    _RATIONAL_DISPLACEMENT,
    rtol=0.0,
    atol=4.0e-14,
  )
  np.testing.assert_allclose(
    solution.ledger.direct_reactions.values,
    (-3.0, 0.0),
    rtol=0.0,
    atol=5.0e-15,
  )
  assert not hasattr(solution.ledger, "mpc_multipliers")


def test_affine_mpc_chain_with_nonzero_offsets_solves_and_verifies() -> None:
  program_spec = ProgramSpec(
    coordinates=(ProgramCoordinateSpec("lambda", "load", _source("lambda")),),
    constraints=(
      PrescribedDofSpec("fix-x", _dof(1, "x"), _affine(), _source("fix-x")),
      PrescribedDofSpec("fix-y", _dof(1, "y"), _affine(), _source("fix-y")),
      AffineTieSpec(
        "first",
        _dof(2, "y"),
        _dof(2, "x"),
        2.0,
        _affine(1.0, ("lambda", 0.5)),
        _source("first"),
      ),
      AffineTieSpec(
        "second",
        _dof(3, "y"),
        _dof(2, "y"),
        -0.5,
        _affine(0.25, ("lambda", 1.0)),
        _source("second"),
      ),
    ),
    loads=(NodalLoadSpec("load", _dof(5, "y"), _affine(1.0), _source("load")),),
  )
  _, _, analysis = _prepared(program_spec)
  solution = analysis.solve(initial_point=_point(0.0), point=_point(2.0))
  np.testing.assert_allclose(
    solution.ledger.constraint_violation.values,
    0.0,
    rtol=0.0,
    atol=2.0e-14,
  )
  assert solution.verify().passed


def test_identity_reduction_rank_13_rejects_even_zero_rhs_without_acceptance() -> None:
  model = compile_model(_model_spec(), q8_reference_registry())
  program = compile_program(model, ProgramSpec())
  analysis = prepare_analysis(model, program, LinearStatic())
  initial = analysis.initialize(point=ProgramPoint())
  transaction = analysis.begin_step(initial=initial, point=ProgramPoint())
  with pytest.raises(AnalysisSolveError, match="cholesky|near-singular"):
    analysis.evaluate_trial(transaction)
  assert initial.generation.ordinal == 0
  assert analysis.workspace_statistics().factorization_count == 0
  second = analysis.begin_step(initial=initial, point=ProgramPoint())
  assert second.base_state is initial


@pytest.mark.parametrize(
  ("kind", "expected"),
  [
    ("nonsymmetric", "asymmetry"),
    ("indefinite", "cholesky"),
    ("near-singular", "near-singular"),
    ("nonfinite", "nonfinite"),
    ("malformed", "malformed"),
  ],
)
def test_reduced_operator_failures_are_inert(
  monkeypatch: pytest.MonkeyPatch,
  kind: str,
  expected: str,
) -> None:
  import pyfem.v3.analysis.linear as linear_module

  _, _, analysis = _prepared()
  initial = analysis.initialize(point=_point(0.0))
  original = linear_module.assemble_reference_linear

  def altered(*args: object, **kwargs: object) -> LinearStaticContributions:
    result = original(*args, **kwargs)
    operator = result.reduced_operator
    dense = np.zeros(operator.shape, dtype=np.float64)
    dense[operator.row_indices.values, operator.column_indices.values] = (
      operator.values.values
    )
    if kind == "nonsymmetric":
      dense[0, 1] += 1.0e-6
    elif kind == "indefinite":
      dense = np.eye(dense.shape[0])
      dense[-1, -1] = -1.0
    elif kind == "near-singular":
      dense = np.eye(dense.shape[0])
      dense[-1, -1] = 1.0e-14
    elif kind == "nonfinite":
      dense[0, 0] = np.nan
    elif kind == "malformed":
      malformed = replace(operator, shape=(operator.shape[0] - 1,) * 2)
      return replace(result, reduced_operator=malformed)
    changed = replace(operator, values=_operator_values(operator, dense))
    return replace(result, reduced_operator=changed)

  monkeypatch.setattr(linear_module, "assemble_reference_linear", altered)
  transaction = analysis.begin_step(initial=initial, point=_point(1.0))
  with pytest.raises(AnalysisSolveError, match=expected):
    analysis.evaluate_trial(transaction)
  assert initial.generation.ordinal == 0
  assert initial.evolution.accepted_step_index == 0


def test_64_eps_admission_projects_only_private_solver_operator(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  import pyfem.v3.analysis.linear as linear_module

  _, _, analysis = _prepared()
  original = linear_module.assemble_reference_linear

  def within_bound(*args: object, **kwargs: object) -> LinearStaticContributions:
    result = original(*args, **kwargs)
    operator = result.reduced_operator
    dense = np.zeros(operator.shape, dtype=np.float64)
    dense[operator.row_indices.values, operator.column_indices.values] = (
      operator.values.values
    )
    delta = 8.0 * np.finfo(np.float64).eps * float(np.max(np.abs(dense)))
    dense[0, 1] += delta
    dense[1, 0] -= delta
    return replace(
      result,
      reduced_operator=replace(
        operator,
        values=_operator_values(operator, dense),
      ),
    )

  monkeypatch.setattr(linear_module, "assemble_reference_linear", within_bound)
  solution = analysis.solve(initial_point=_point(0.0), point=_point(1.0))
  workspace = analysis._workspace
  assert not np.array_equal(workspace.audit_operator, workspace.audit_operator.T)
  np.testing.assert_array_equal(workspace.solve_operator, workspace.solve_operator.T)
  assert solution.verify().passed


def test_constant_operator_drift_fails_closed_without_replacing_cache(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  import pyfem.v3.analysis.linear as linear_module

  _, _, analysis = _prepared()
  first = analysis.solve(initial_point=_point(0.0), point=_point(1.0))
  cached = np.array(analysis._workspace.audit_operator, copy=True)
  original = linear_module.assemble_reference_linear

  def drifted(*args: object, **kwargs: object) -> LinearStaticContributions:
    result = original(*args, **kwargs)
    operator = result.reduced_operator
    values = np.array(operator.values.values, copy=True)
    values[0] = np.nextafter(values[0], np.inf)
    return replace(
      result,
      reduced_operator=replace(
        operator,
        values=FinalizedArray(values, dtype=np.float64),
      ),
    )

  monkeypatch.setattr(linear_module, "assemble_reference_linear", drifted)
  with pytest.raises(AnalysisSolveError, match="constant tangent changed"):
    analysis.solve(initial_point=_point(0.0), point=_point(0.5))
  np.testing.assert_array_equal(analysis._workspace.audit_operator, cached)
  assert first.verify().passed
  assert analysis.workspace_statistics().factorization_count == 1


def test_content_equal_live_distinct_owners_reject_foreign_state_and_solution() -> None:
  specification = _model_spec()
  program_specification = _rational_program()
  model_a = compile_model(specification, q8_reference_registry())
  model_b = compile_model(specification, q8_reference_registry())
  assert model_a.content_fingerprint == model_b.content_fingerprint
  assert model_a.instance_id is not model_b.instance_id
  program_a = compile_program(model_a, program_specification)
  program_b = compile_program(model_b, program_specification)
  analysis_a = prepare_analysis(model_a, program_a, LinearStatic())
  analysis_b = prepare_analysis(model_b, program_b, LinearStatic())
  assert analysis_a.instance_id is not analysis_b.instance_id
  assert analysis_a._workspace is not analysis_b._workspace
  initial_a = analysis_a.initialize(point=_point(0.0))
  with pytest.raises(StateTransactionError, match="foreign|exact immutable"):
    analysis_b.begin_step(initial=initial_a, point=_point(1.0))
  solution = analysis_a.solve(initial=initial_a, point=_point(1.0))
  foreign_solution = replace(solution, model=model_b, program=program_b)
  with pytest.raises(SolutionVerificationError, match="identity|fingerprint"):
    foreign_solution.verify_record()


def test_exact_request_and_prepared_construction_boundaries() -> None:
  model = compile_model(_model_spec(), q8_reference_registry())
  program = compile_program(model, _rational_program())
  with pytest.raises(AnalysisPreparationError, match="exactly LinearStatic"):
    prepare_analysis(model, program, object())
  with pytest.raises(AnalysisPreparationError, match="prepare_analysis"):
    from pyfem.v3.analysis import PreparedAnalysis

    PreparedAnalysis(
      model,
      program,
      LinearStatic(),
      object(),
      _token=object(),
    )

  analysis = prepare_analysis(model, program, LinearStatic())
  with pytest.raises(AttributeError, match="immutable"):
    analysis.request = LinearStatic()
  object.__setattr__(analysis, "request", object())
  assert analysis._workspace.evaluation_count == 0
  with pytest.raises(StateTransactionError, match="malformed-prepared-analysis"):
    analysis.initialize(point=_point(0.0))


def test_mismatched_live_plan_fails_before_workspace_use() -> None:
  _, _, first = _prepared()
  _, _, second = _prepared()
  object.__setattr__(first, "assembly_plan", second.assembly_plan)
  initial = second.initialize(point=_point(0.0))
  with pytest.raises(StateTransactionError, match="malformed-prepared-analysis"):
    first.begin_step(initial=initial, point=_point(1.0))
  assert first._workspace.evaluation_count == 0


def test_dense_workspace_capacity_fails_before_analysis_allocation(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  import pyfem.v3.analysis.linear as linear_module
  from pyfem.v3.analysis.contracts import (
    LINEAR_STATIC_WORKSPACE_BUDGET_SCOPE,
    linear_static_request_manifest,
  )

  assert linear_module._checked_workspace_bytes(2895) == 268331760
  assert 8 * (4 * 2896 * 2896 + 6 * 2896) == 268517120
  with pytest.raises(AnalysisPreparationError, match="256 MiB"):
    linear_module._checked_workspace_bytes(2896)
  manifest = linear_static_request_manifest(LinearStatic()).to_bytes()
  assert LINEAR_STATIC_WORKSPACE_BUDGET_SCOPE.encode() in manifest
  assert b"explicit-scalar-cholesky" in manifest
  assert b"explicit-forward-back-substitution" in manifest

  model = compile_model(_model_spec(), q8_reference_registry())
  program = compile_program(model, _rational_program())
  plan = prepare_assembly_plan(
    model,
    program,
    LinearStaticContributionRequest(),
  )
  oversized_domain = replace(
    plan.domain_coo_plan,
    reduced_shape=(2896, 2896),
  )
  oversized = replace(plan, domain_coo_plan=oversized_domain)
  monkeypatch.setattr(
    linear_module,
    "prepare_assembly_plan",
    lambda *_args: oversized,
  )
  with pytest.raises(AnalysisPreparationError, match="256 MiB"):
    prepare_analysis(model, program, LinearStatic())


def _exact_ratio_greater(value: float, scale: float, threshold: float) -> bool:
  return Fraction(value) > Fraction(scale) * Fraction(threshold)


def test_strict_pivot_ratio_exact_matrix_across_binary_binades() -> None:
  from pyfem.v3.analysis.contracts import LINEAR_STATIC_CHOLESKY_PIVOT_RATIO
  from pyfem.v3.analysis.numerics import _strict_ratio_greater

  threshold = LINEAR_STATIC_CHOLESKY_PIVOT_RATIO
  for exponent in (-900, 0, 900):
    scale = math.ldexp(1.0, exponent)
    boundary = scale * threshold
    assert not _strict_ratio_greater(
      float(np.nextafter(boundary, 0.0)),
      scale,
      threshold,
    )
    assert not _strict_ratio_greater(boundary, scale, threshold)
    assert _strict_ratio_greater(
      float(np.nextafter(boundary, math.inf)),
      scale,
      threshold,
    )
    assert _strict_ratio_greater(scale * 1.5e-12, scale, threshold)

  non_power_scales = (
    float.fromhex("0x1.79eaa73e7f552p-202"),
    float.fromhex("0x1.004189374bc6ap+0"),
    1.5,
    math.ldexp(1.75, 700),
  )
  for scale in non_power_scales:
    rounded_boundary = scale * threshold
    values = (
      float(np.nextafter(rounded_boundary, 0.0)),
      rounded_boundary,
      float(np.nextafter(rounded_boundary, math.inf)),
    )
    expected = tuple(_exact_ratio_greater(value, scale, threshold) for value in values)
    assert any(expected)
    assert not all(expected)
    assert (
      tuple(_strict_ratio_greater(value, scale, threshold) for value in values)
      == expected
    )

  concrete_scale = float.fromhex("0x1.79eaa73e7f552p-202")
  concrete_pivot = float.fromhex("0x1.9f8611fbd4c2cp-242")
  assert _exact_ratio_greater(concrete_pivot, concrete_scale, threshold)
  assert not concrete_pivot > threshold * concrete_scale
  assert _strict_ratio_greater(concrete_pivot, concrete_scale, threshold)


@pytest.mark.parametrize(
  ("scale_hex", "pivot_hex"),
  [
    ("0x1.79eaa73e7f552p-202", "0x1.9f8611fbd4c2cp-242"),
    ("0x1.004189374bc6ap+0", "0x1.19c1a6d15a3e6p-40"),
  ],
)
def test_concrete_non_power_of_two_spd_solve(
  scale_hex: str,
  pivot_hex: str,
) -> None:
  model, program, analysis = _prepared()
  contributions = assemble_reference_linear(
    model,
    program,
    analysis.assembly_plan,
    _point(1.0),
  )
  reduced_count = analysis.workspace_statistics().reduced_dof_count
  scale = float.fromhex(scale_hex)
  pivot = float.fromhex(pivot_hex)
  dense = np.eye(reduced_count, dtype=np.float64) * scale
  dense[-1, -1] = pivot
  operator = contributions.reduced_operator
  changed = replace(
    contributions,
    reduced_operator=replace(operator, values=_operator_values(operator, dense)),
    reduced_rhs=FinalizedArray(np.zeros(reduced_count), dtype=np.float64),
  )
  coordinates, convergence = analysis._solve_reduced(changed)
  np.testing.assert_array_equal(coordinates, 0.0)
  assert convergence.operator_infinity_norm == scale
  assert convergence.minimum_unscaled_pivot == pivot


def test_explicit_factor_and_triangular_solve_own_only_declared_storage(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  import pyfem.v3.analysis.linear as linear_module
  from pyfem.v3.analysis.numerics import _explicit_cholesky

  operator = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)
  factorization = _explicit_cholesky(operator)
  assert factorization is not None
  factor, pivots = factorization
  rhs = np.array([1.0, 2.0], dtype=np.float64)
  forward, coordinates = linear_module._bounded_cholesky_solve(
    factor,
    rhs,
  )
  for array in (operator, factor, pivots, rhs, forward, coordinates):
    assert array.flags.owndata
    assert array.base is None
  declared = (operator, factor, pivots, rhs, forward, coordinates)
  for index, left in enumerate(declared):
    for right in declared[index + 1 :]:
      assert not np.shares_memory(left, right)
  np.testing.assert_allclose(
    operator @ coordinates,
    np.array([1.0, 2.0]),
    rtol=0.0,
    atol=2.0e-15,
  )

  def forbidden(*_args: object, **_kwargs: object) -> None:
    msg = "opaque NumPy factorization/solve is forbidden"
    raise AssertionError(msg)

  monkeypatch.setattr(np.linalg, "solve", forbidden)
  monkeypatch.setattr(np.linalg, "cholesky", forbidden)
  _, _, analysis = _prepared()
  solution = analysis.solve(initial_point=_point(0.0), point=_point(1.0))
  assert solution.verify().passed


@pytest.mark.parametrize(
  ("factor", "expected"),
  [
    (np.array([[0.0]], dtype=np.float64), "positive finite diagonals"),
    (
      np.array([[float.fromhex("0x0.0000000000001p-1022")]], dtype=np.float64),
      "nonfinite value",
    ),
  ],
)
def test_explicit_triangular_solve_fails_closed(
  factor: np.ndarray,
  expected: str,
) -> None:
  import pyfem.v3.analysis.linear as linear_module

  with pytest.raises(AnalysisSolveError, match=expected):
    linear_module._bounded_cholesky_solve(
      factor,
      np.ones(factor.shape[0], dtype=np.float64),
    )


def test_factorization_reuse_and_all_published_storage_is_disjoint() -> None:
  _, _, analysis = _prepared()
  first = analysis.solve(initial_point=_point(0.0), point=_point(1.0))
  first_snapshot = np.array(first.primary_values.values, copy=True)
  second = analysis.solve(initial_point=_point(0.0), point=_point(0.5))
  statistics = analysis.workspace_statistics()
  assert statistics.factorization_count == 1
  assert statistics.factorization_reuse_count == 1
  assert not hasattr(first.convergence, "factorization_performed")
  assert not hasattr(second.convergence, "factorization_reused")
  retained = (
    analysis._workspace.audit_operator,
    analysis._workspace.solve_operator,
    analysis._workspace.factor,
  )
  assert all(array is not None for array in retained)
  assert all(
    array.flags.owndata and array.base is None and not array.flags.writeable
    for array in retained
    if array is not None
  )
  assert sum(array.nbytes for array in retained if array is not None) == 3 * 13 * 13 * 8
  assert statistics.checked_workspace_bytes == 6032
  np.testing.assert_array_equal(first.primary_values.values, first_snapshot)
  assert not _shares_any(_state_arrays(first), _state_arrays(second))
  assert not _shares_any(_ledger_arrays(first), _ledger_arrays(second))
  assert not _shares_any(_state_arrays(first), _ledger_arrays(first))
  for array in (*_state_arrays(first), *_ledger_arrays(first)):
    assert not array.flags.writeable


def test_exactly_once_acceptance_siblings_double_accept_discard_and_forgery() -> None:
  _, _, analysis = _prepared()
  initial = analysis.initialize(point=_point(0.0))
  first_transaction = analysis.begin_step(initial=initial, point=_point(1.0))
  sibling_transaction = analysis.begin_step(initial=initial, point=_point(0.5))
  first_trial = analysis.evaluate_trial(first_transaction)
  sibling_trial = analysis.evaluate_trial(sibling_transaction)
  solution = analysis.accept(first_trial)
  assert solution.state.generation.ordinal == 1
  with pytest.raises(StateTransactionError, match="closed"):
    analysis.accept(first_trial)
  with pytest.raises(StateTransactionError, match="stale|consumed"):
    analysis.accept(sibling_trial)

  fresh = analysis.initialize(point=_point(0.0))
  transaction = analysis.begin_step(initial=fresh, point=_point(1.0))
  trial = analysis.evaluate_trial(transaction)
  before = np.array(fresh.physical.primary_values.values, copy=True)
  analysis.discard(trial)
  with pytest.raises(StateTransactionError, match="discarded"):
    analysis.accept(trial)
  np.testing.assert_array_equal(fresh.physical.primary_values.values, before)

  another = analysis.initialize(point=_point(0.0))
  transaction = analysis.begin_step(initial=another, point=_point(1.0))
  trial = analysis.evaluate_trial(transaction)
  forged = replace(trial, candidate_generation=StateGeneration.initial())
  with pytest.raises(StateTransactionError, match="foreign|forged"):
    analysis.accept(forged)
  assert another.generation.ordinal == 0


def test_discard_and_accept_ordering_is_atomic(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  import pyfem.v3.analysis.linear as linear_module

  _, _, analysis = _prepared()
  initial = analysis.initialize(point=_point(0.0))
  transaction = analysis.begin_step(initial=initial, point=_point(1.0))
  trial = analysis.evaluate_trial(transaction)
  entered = Event()
  released = Event()
  original = linear_module.fresh_verify

  def blocked_fresh_verify(**kwargs: object) -> object:
    entered.set()
    if not released.wait(timeout=10.0):
      msg = "acceptance barrier timed out"
      raise AssertionError(msg)
    return original(**kwargs)

  monkeypatch.setattr(linear_module, "fresh_verify", blocked_fresh_verify)
  with ThreadPoolExecutor(max_workers=1) as executor:
    future = executor.submit(analysis.accept, trial)
    assert entered.wait(timeout=10.0)
    analysis.discard(trial)
    released.set()
    with pytest.raises(StateTransactionError, match="discarded"):
      future.result(timeout=10.0)
  assert not analysis._consumed_generations
  analysis.begin_step(initial=initial, point=_point(0.5))


def test_accept_wins_racing_discard_and_only_one_concurrent_discard_succeeds(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  import pyfem.v3.analysis.linear as linear_module

  original = linear_module.PreparedAnalysis._registered_trial
  _, _, accepted_first = _prepared()
  accepted_initial = accepted_first.initialize(point=_point(0.0))
  accepted_transaction = accepted_first.begin_step(
    initial=accepted_initial,
    point=_point(1.0),
  )
  accepted_trial = accepted_first.evaluate_trial(accepted_transaction)
  entered = Event()
  released = Event()

  def blocked_registered_trial(
    owner: PreparedAnalysis,
    candidate: object,
  ) -> tuple[object, object]:
    registered = original(owner, candidate)
    if owner is accepted_first and current_thread() is not main_thread():
      entered.set()
      if not released.wait(timeout=10.0):
        msg = "discard barrier timed out"
        raise AssertionError(msg)
    return registered

  monkeypatch.setattr(
    linear_module.PreparedAnalysis,
    "_registered_trial",
    blocked_registered_trial,
  )
  with ThreadPoolExecutor(max_workers=1) as executor:
    discard = executor.submit(accepted_first.discard, accepted_trial)
    assert entered.wait(timeout=10.0)
    solution = accepted_first.accept(accepted_trial)
    released.set()
    with pytest.raises(StateTransactionError, match="closed"):
      discard.result(timeout=10.0)
  assert solution.state.generation.ordinal == 1
  assert len(accepted_first._consumed_generations) == 1

  monkeypatch.setattr(
    linear_module.PreparedAnalysis,
    "_registered_trial",
    original,
  )
  _, _, concurrent = _prepared()
  concurrent_initial = concurrent.initialize(point=_point(0.0))
  concurrent_transaction = concurrent.begin_step(
    initial=concurrent_initial,
    point=_point(1.0),
  )
  concurrent_trial = concurrent.evaluate_trial(concurrent_transaction)
  barrier = Barrier(2)

  def synchronized_registered_trial(
    owner: PreparedAnalysis,
    candidate: object,
  ) -> tuple[object, object]:
    registered = original(owner, candidate)
    if owner is concurrent:
      barrier.wait(timeout=10.0)
    return registered

  monkeypatch.setattr(
    linear_module.PreparedAnalysis,
    "_registered_trial",
    synchronized_registered_trial,
  )
  with ThreadPoolExecutor(max_workers=2) as executor:
    discards = tuple(
      executor.submit(concurrent.discard, concurrent_trial) for _ in range(2)
    )
    successes = 0
    failures: list[str] = []
    for discard in discards:
      try:
        discard.result(timeout=10.0)
        successes += 1
      except StateTransactionError as error:
        failures.append(str(error))
  assert successes == 1
  assert len(failures) == 1
  assert "already discarded" in failures[0]
  assert not concurrent._consumed_generations
  concurrent.begin_step(initial=concurrent_initial, point=_point(0.5))


@pytest.mark.parametrize(
  "corrupt",
  ["identity", "generation", "convergence", "ledger", "storage"],
)
def test_verify_record_rejects_structural_corruption(corrupt: str) -> None:
  _, _, _, _, solution = _solve_rational()
  if corrupt == "identity":
    changed = replace(solution, prepared_instance_id=InstanceId())
  elif corrupt == "generation":
    transition = replace(
      solution.transition,
      candidate_generation=StateGeneration.initial(),
    )
    changed = replace(solution, transition=transition)
  elif corrupt == "convergence":
    changed = replace(
      solution,
      convergence=replace(solution.convergence, converged=False),
    )
  elif corrupt == "ledger":
    changed = replace(
      solution,
      ledger=replace(solution.ledger, transaction_id=InstanceId()),
    )
  else:
    aliased = object.__new__(FinalizedArray)
    object.__setattr__(
      aliased,
      "values",
      solution.state.physical.primary_values.values,
    )
    changed = replace(
      solution,
      ledger=replace(solution.ledger, full_primary_values=aliased),
    )
  with pytest.raises(SolutionVerificationError):
    changed.verify_record()


def test_result_arrays_require_owned_and_disjoint_storage() -> None:
  _, _, _, _, solution = _solve_rational()
  writable_base = np.array(solution.ledger.full_primary_values.values, copy=True)
  read_only_view = writable_base.view()
  read_only_view.setflags(write=False)
  viewed = object.__new__(FinalizedArray)
  object.__setattr__(viewed, "values", read_only_view)
  viewed_solution = replace(
    solution,
    ledger=replace(solution.ledger, full_primary_values=viewed),
  )
  with pytest.raises(SolutionVerificationError, match="ownership"):
    viewed_solution.verify_record()
  writable_base[0] = 123.0

  shared_empty = (
    solution.transition.transaction.base_state.physical.formulation_histories[0]
  )
  physical = replace(
    solution.state.physical,
    formulation_histories=(shared_empty,),
  )
  state = replace(solution.state, physical=physical)
  transition = replace(solution.transition, committed_state=state)
  empty_alias = replace(solution, state=state, transition=transition)
  assert shared_empty.values.shape == (1, 0)
  assert not np.shares_memory(shared_empty.values, shared_empty.values)
  with pytest.raises(SolutionVerificationError, match="aliased"):
    empty_alias.verify_record()


_COMPARISON_CALLS: list[str] = []


class _StringSubclass(str):
  pass


class _ComparisonBomb(str):
  def __eq__(self, other: object) -> bool:
    del other
    _COMPARISON_CALLS.append("eq")
    msg = "comparison bomb must not be invoked"
    raise RuntimeError(msg)

  def __ne__(self, other: object) -> bool:
    del other
    _COMPARISON_CALLS.append("ne")
    msg = "comparison bomb must not be invoked"
    raise RuntimeError(msg)

  __hash__ = str.__hash__


class _TupleSubclass(tuple[object, ...]):
  pass


class _TupleBomb(tuple[object, ...]):
  def __iter__(self) -> NoReturn:
    _COMPARISON_CALLS.append("tuple-iter")
    msg = "tuple iteration bomb must not be invoked"
    raise RuntimeError(msg)

  def __len__(self) -> int:
    _COMPARISON_CALLS.append("tuple-len")
    msg = "tuple length bomb must not be invoked"
    raise RuntimeError(msg)

  def __eq__(self, other: object) -> bool:
    del other
    _COMPARISON_CALLS.append("tuple-eq")
    msg = "tuple comparison bomb must not be invoked"
    raise RuntimeError(msg)

  def __ne__(self, other: object) -> bool:
    del other
    _COMPARISON_CALLS.append("tuple-ne")
    msg = "tuple comparison bomb must not be invoked"
    raise RuntimeError(msg)


class _IntSubclass(int):
  pass


class _IntBomb(int):
  def __eq__(self, other: object) -> bool:
    del other
    _COMPARISON_CALLS.append("int-eq")
    msg = "integer comparison bomb must not be invoked"
    raise RuntimeError(msg)

  def __lt__(self, other: object) -> bool:
    del other
    _COMPARISON_CALLS.append("int-lt")
    msg = "integer comparison bomb must not be invoked"
    raise RuntimeError(msg)

  def __ne__(self, other: object) -> bool:
    del other
    _COMPARISON_CALLS.append("int-ne")
    msg = "integer comparison bomb must not be invoked"
    raise RuntimeError(msg)

  __hash__ = int.__hash__


class _FloatSubclass(float):
  pass


class _FloatBomb(float):
  def __eq__(self, other: object) -> bool:
    del other
    _COMPARISON_CALLS.append("float-eq")
    msg = "float comparison bomb must not be invoked"
    raise RuntimeError(msg)

  def __lt__(self, other: object) -> bool:
    del other
    _COMPARISON_CALLS.append("float-lt")
    msg = "float comparison bomb must not be invoked"
    raise RuntimeError(msg)

  def __ne__(self, other: object) -> bool:
    del other
    _COMPARISON_CALLS.append("float-ne")
    msg = "float comparison bomb must not be invoked"
    raise RuntimeError(msg)

  __hash__ = float.__hash__


class _BoolBomb:
  def __bool__(self) -> bool:
    _COMPARISON_CALLS.append("bool")
    msg = "boolean conversion bomb must not be invoked"
    raise RuntimeError(msg)


class _UUIDSubclass(UUID):
  pass


class _UUIDBomb(UUID):
  def __eq__(self, other: object) -> bool:
    del other
    _COMPARISON_CALLS.append("uuid-eq")
    msg = "UUID comparison bomb must not be invoked"
    raise RuntimeError(msg)

  def __ne__(self, other: object) -> bool:
    del other
    _COMPARISON_CALLS.append("uuid-ne")
    msg = "UUID comparison bomb must not be invoked"
    raise RuntimeError(msg)

  __hash__ = UUID.__hash__


def _with_state(solution: Solution, state: CommittedAnalysisState) -> Solution:
  return replace(
    solution,
    state=state,
    transition=replace(solution.transition, committed_state=state),
  )


def _assert_structured_without_comparison(solution: Solution) -> None:
  _COMPARISON_CALLS.clear()
  with pytest.raises(SolutionVerificationError):
    solution.verify_record()
  assert not _COMPARISON_CALLS


@pytest.mark.parametrize(
  "field",
  [
    "physical_schema",
    "evolution_schema",
    "program_history_schema",
    "algebraic_field_id",
    "coordinate_name",
    "constraint_id",
    "backend_policy",
  ],
)
@pytest.mark.parametrize("carrier", [_StringSubclass, _ComparisonBomb])
def test_result_scalar_preflight_rejects_subclasses_without_comparison(
  field: str,
  carrier: type[str],
) -> None:
  _, _, _, _, solution = _solve_rational()
  state = solution.state
  if field == "physical_schema":
    state = replace(
      state,
      physical=replace(
        state.physical,
        schema=carrier(state.physical.schema),
      ),
    )
  elif field == "evolution_schema":
    state = replace(
      state,
      evolution=replace(
        state.evolution,
        schema=carrier(state.evolution.schema),
      ),
    )
  elif field == "program_history_schema":
    state = replace(
      state,
      program_history=replace(
        state.program_history,
        schema=carrier(state.program_history.schema),
      ),
    )
  elif field == "algebraic_field_id":
    identifiers = state.evolution.algebraic_field_ids
    state = replace(
      state,
      evolution=replace(
        state.evolution,
        algebraic_field_ids=(carrier(str(identifiers[0])), *identifiers[1:]),
      ),
    )
  elif field == "coordinate_name":
    evaluation = state.evolution.program_evaluation
    coordinate_names = evaluation.coordinate_names
    state = replace(
      state,
      evolution=replace(
        state.evolution,
        program_evaluation=replace(
          evaluation,
          coordinate_names=(
            carrier(coordinate_names[0]),
            *coordinate_names[1:],
          ),
        ),
      ),
    )
  elif field == "constraint_id":
    constraint_ids = solution.ledger.constraint_ids
    changed = replace(
      solution,
      ledger=replace(
        solution.ledger,
        constraint_ids=(carrier(str(constraint_ids[0])), *constraint_ids[1:]),
      ),
    )
  else:
    changed = replace(
      solution,
      convergence=replace(
        solution.convergence,
        backend_policy=carrier(solution.convergence.backend_policy),
      ),
    )

  if field not in {"constraint_id", "backend_policy"}:
    changed = replace(
      solution,
      state=state,
      transition=replace(solution.transition, committed_state=state),
    )
  _COMPARISON_CALLS.clear()
  with pytest.raises(SolutionVerificationError):
    changed.verify_record()
  assert not _COMPARISON_CALLS


@pytest.mark.parametrize("carrier", [_TupleSubclass, _TupleBomb])
def test_result_tuple_preflight_rejects_outer_subclasses_without_operations(
  carrier: type[tuple[object, ...]],
) -> None:
  _, _, _, _, solution = _solve_rational()
  for field in (
    "coordinate_names",
    "algebraic_field_ids",
    "constraint_ids",
    "material_histories",
    "formulation_histories",
    "history_entries",
  ):
    state = solution.state
    if field == "coordinate_names":
      evaluation = state.evolution.program_evaluation
      state = replace(
        state,
        evolution=replace(
          state.evolution,
          program_evaluation=replace(
            evaluation,
            coordinate_names=carrier(evaluation.coordinate_names),
          ),
        ),
      )
      changed = _with_state(solution, state)
    elif field == "algebraic_field_ids":
      state = replace(
        state,
        evolution=replace(
          state.evolution,
          algebraic_field_ids=carrier(state.evolution.algebraic_field_ids),
        ),
      )
      changed = _with_state(solution, state)
    elif field == "constraint_ids":
      changed = replace(
        solution,
        ledger=replace(
          solution.ledger,
          constraint_ids=carrier(solution.ledger.constraint_ids),
        ),
      )
    elif field == "material_histories":
      state = replace(
        state,
        physical=replace(
          state.physical,
          material_histories=carrier(state.physical.material_histories),
        ),
      )
      changed = _with_state(solution, state)
    elif field == "formulation_histories":
      state = replace(
        state,
        physical=replace(
          state.physical,
          formulation_histories=carrier(state.physical.formulation_histories),
        ),
      )
      changed = _with_state(solution, state)
    else:
      state = replace(
        state,
        program_history=replace(
          state.program_history,
          entries=carrier(state.program_history.entries),
        ),
      )
      changed = _with_state(solution, state)
    _assert_structured_without_comparison(changed)


@pytest.mark.parametrize("carrier", [_IntSubclass, _IntBomb])
def test_result_integer_preflight_rejects_subclasses_without_comparison(
  carrier: type[int],
) -> None:
  _, _, _, _, solution = _solve_rational()
  for field in (
    "algebraic_field_id",
    "constraint_id",
    "accepted_step_index",
    "retry",
    "cutback",
    "iteration_count",
    "generation_ordinal",
  ):
    state = solution.state
    if field == "algebraic_field_id":
      identifiers = state.evolution.algebraic_field_ids
      state = replace(
        state,
        evolution=replace(
          state.evolution,
          algebraic_field_ids=(carrier(1), *identifiers[1:]),
        ),
      )
      changed = _with_state(solution, state)
    elif field == "constraint_id":
      identifiers = solution.ledger.constraint_ids
      changed = replace(
        solution,
        ledger=replace(
          solution.ledger,
          constraint_ids=(carrier(1), *identifiers[1:]),
        ),
      )
    elif field == "accepted_step_index":
      state = replace(
        state,
        evolution=replace(
          state.evolution,
          accepted_step_index=carrier(state.evolution.accepted_step_index),
        ),
      )
      changed = _with_state(solution, state)
    elif field in {"retry", "cutback"}:
      transaction = replace(
        solution.transition.transaction,
        **{
          field: carrier(getattr(solution.transition.transaction, field)),
        },
      )
      changed = replace(
        solution,
        transition=replace(solution.transition, transaction=transaction),
      )
    elif field == "iteration_count":
      changed = replace(
        solution,
        convergence=replace(
          solution.convergence,
          iteration_count=carrier(solution.convergence.iteration_count),
        ),
      )
    else:
      generation = object.__new__(StateGeneration)
      object.__setattr__(generation, "_lineage", state.generation._lineage)
      object.__setattr__(
        generation,
        "ordinal",
        carrier(state.generation.ordinal),
      )
      state = replace(state, generation=generation)
      changed = _with_state(solution, state)
    _assert_structured_without_comparison(changed)


@pytest.mark.parametrize("carrier", [_FloatSubclass, _FloatBomb])
def test_result_float_preflight_rejects_subclasses_without_comparison(
  carrier: type[float],
) -> None:
  _, _, _, _, solution = _solve_rational()
  convergence_fields = (
    "operator_infinity_norm",
    "minimum_unscaled_pivot",
    "reduced_residual_norm",
    "verification_tolerance",
  )
  ledger_fields = (
    "constraint_work",
    "external_work",
    "internal_work",
    "full_force_scale",
    "reduced_force_scale",
    "reconstruction_scale",
    "work_scale",
    "full_operator_infinity_norm",
    "reduced_operator_infinity_norm",
    "prolongation_transpose_infinity_norm",
    "verification_tolerance",
  )
  for field in convergence_fields:
    value = getattr(solution.convergence, field)
    assert type(value) is float
    changed = replace(
      solution,
      convergence=replace(
        solution.convergence,
        **{field: carrier(value)},
      ),
    )
    _assert_structured_without_comparison(changed)
  for field in ledger_fields:
    value = getattr(solution.ledger, field)
    assert type(value) is float
    changed = replace(
      solution,
      ledger=replace(solution.ledger, **{field: carrier(value)}),
    )
    _assert_structured_without_comparison(changed)


@pytest.mark.parametrize("kind", ["numpy", "bomb"])
def test_result_boolean_preflight_never_invokes_conversion(kind: str) -> None:
  _, _, _, _, solution = _solve_rational()

  def changed_value(value: bool) -> object:
    return np.bool_(value) if kind == "numpy" else _BoolBomb()

  for field in ("accepted", "converged", "factorization_bypassed"):
    if field == "accepted":
      changed = replace(
        solution,
        transition=replace(
          solution.transition,
          accepted=changed_value(solution.transition.accepted),
        ),
      )
    else:
      changed = replace(
        solution,
        convergence=replace(
          solution.convergence,
          **{field: changed_value(getattr(solution.convergence, field))},
        ),
      )
    _assert_structured_without_comparison(changed)


@pytest.mark.parametrize("carrier", [_UUIDSubclass, _UUIDBomb])
def test_result_uuid_preflight_rejects_subclasses_without_comparison(
  carrier: type[UUID],
) -> None:
  _, _, _, _, solution = _solve_rational()
  token = solution.prepared_instance_id._token
  foreign_token = carrier(str(token))
  identity = object.__new__(InstanceId)
  object.__setattr__(identity, "_token", foreign_token)
  _assert_structured_without_comparison(
    replace(solution, prepared_instance_id=identity)
  )

  generation = object.__new__(StateGeneration)
  object.__setattr__(
    generation,
    "_lineage",
    carrier(str(solution.state.generation._lineage)),
  )
  object.__setattr__(generation, "ordinal", solution.state.generation.ordinal)
  _assert_structured_without_comparison(
    _with_state(solution, replace(solution.state, generation=generation))
  )


def test_result_uuid_preflight_rejects_partial_and_noncanonical_integer() -> None:
  _, _, _, _, solution = _solve_rational()
  partial_uuid = object.__new__(UUID)
  integer_bomb_uuid = object.__new__(UUID)
  object.__setattr__(integer_bomb_uuid, "int", _IntBomb(1))
  negative_uuid = object.__new__(UUID)
  object.__setattr__(negative_uuid, "int", -1)
  oversized_uuid = object.__new__(UUID)
  object.__setattr__(oversized_uuid, "int", 1 << 128)
  for token in (partial_uuid, integer_bomb_uuid, negative_uuid, oversized_uuid):
    identity = object.__new__(InstanceId)
    object.__setattr__(identity, "_token", token)
    _assert_structured_without_comparison(
      replace(solution, prepared_instance_id=identity)
    )

  partial_ledger_id = object.__new__(InstanceId)
  _assert_structured_without_comparison(
    replace(
      solution,
      ledger=replace(solution.ledger, ledger_id=partial_ledger_id),
    )
  )


@pytest.mark.parametrize(
  "partial",
  [
    "solution",
    "committed",
    "physical",
    "evolution",
    "program_history",
    "evaluation",
    "transition",
    "transaction",
    "predictor",
    "convergence",
    "ledger",
    "identity",
    "generation",
    "array",
  ],
)
def test_result_boundary_totalizes_partially_initialized_exact_carriers(
  partial: str,
) -> None:
  _, _, _, _, solution = _solve_rational()
  if partial == "solution":
    changed = object.__new__(Solution)
  elif partial == "identity":
    changed = replace(
      solution,
      prepared_instance_id=object.__new__(InstanceId),
    )
  elif partial == "generation":
    changed = replace(
      solution,
      transition=replace(
        solution.transition,
        candidate_generation=object.__new__(StateGeneration),
      ),
    )
  elif partial == "committed":
    changed = _with_state(solution, object.__new__(CommittedAnalysisState))
  elif partial == "physical":
    changed = _with_state(
      solution,
      replace(
        solution.state,
        physical=object.__new__(PhysicalState),
      ),
    )
  elif partial == "evolution":
    changed = _with_state(
      solution,
      replace(
        solution.state,
        evolution=object.__new__(EvolutionState),
      ),
    )
  elif partial == "program_history":
    changed = _with_state(
      solution,
      replace(
        solution.state,
        program_history=object.__new__(ProgramHistory),
      ),
    )
  elif partial == "evaluation":
    changed = _with_state(
      solution,
      replace(
        solution.state,
        evolution=replace(
          solution.state.evolution,
          program_evaluation=object.__new__(ProgramEvaluation),
        ),
      ),
    )
  elif partial == "transition":
    changed = replace(
      solution,
      transition=object.__new__(AcceptedTransition),
    )
  elif partial == "transaction":
    changed = replace(
      solution,
      transition=replace(
        solution.transition,
        transaction=object.__new__(StepTransaction),
      ),
    )
  elif partial == "predictor":
    changed = replace(
      solution,
      transition=replace(
        solution.transition,
        transaction=replace(
          solution.transition.transaction,
          predictor=object.__new__(LinearPredictor),
        ),
      ),
    )
  elif partial == "convergence":
    changed = replace(
      solution,
      convergence=object.__new__(LinearConvergenceRecord),
    )
  elif partial == "ledger":
    changed = replace(solution, ledger=object.__new__(LinearBalanceLedger))
  else:
    changed = replace(
      solution,
      ledger=replace(
        solution.ledger,
        balance=object.__new__(FinalizedArray),
      ),
    )
  _assert_structured_without_comparison(changed)


def test_validate_state_record_totalizes_partial_exact_committed_state() -> None:
  model, program, _, _, solution = _solve_rational()
  _COMPARISON_CALLS.clear()
  with pytest.raises(SolutionVerificationError):
    validate_state_record(
      state=object.__new__(CommittedAnalysisState),
      model=model,
      program=program,
      prepared_instance_id=solution.prepared_instance_id,
      request_manifest=solution.request_manifest,
    )
  assert not _COMPARISON_CALLS


def test_verify_record_rejects_internally_inconsistent_work() -> None:
  _, _, _, _, solution = _solve_rational()
  changed = replace(
    solution,
    ledger=replace(
      solution.ledger,
      external_work=solution.ledger.external_work + 1.0,
    ),
  )
  report = changed.verify_record()
  assert not report.passed
  assert not report.check("record_external_work").passed


def test_fresh_verification_recomputes_complete_backend_evidence() -> None:
  _, _, analysis, _, solution = _solve_rational()
  reused = analysis.solve(initial_point=_point(0.0), point=_point(0.5))
  statistics = analysis.workspace_statistics()
  assert statistics.factorization_count == 1
  assert statistics.factorization_reuse_count == 1
  assert not hasattr(reused.convergence, "factorization_performed")
  assert not hasattr(reused.convergence, "factorization_reused")
  assert reused.verify().passed

  for field, check_name in (
    ("operator_infinity_norm", "backend_operator_norm"),
    ("minimum_unscaled_pivot", "backend_minimum_pivot"),
  ):
    value = getattr(reused.convergence, field)
    assert type(value) is float
    changed = replace(
      reused,
      convergence=replace(
        reused.convergence,
        **{field: float(np.nextafter(value, math.inf))},
      ),
    )
    assert changed.verify_record().passed
    report = changed.verify()
    assert not report.passed
    assert not report.check(check_name).passed

  policy_changed = replace(
    reused,
    convergence=replace(
      reused.convergence,
      backend_policy="changed-backend-policy",
    ),
  )
  assert policy_changed.verify_record().passed
  policy_report = policy_changed.verify()
  assert not policy_report.passed
  assert not policy_report.check("backend_policy").passed
  assert policy_report.check("backend_convergence_record").passed
  assert policy_report.check("backend_symmetry_admission").passed
  assert policy_report.check("backend_solver_projection").passed
  assert policy_report.check("backend_factorization_bypass").passed
  assert policy_report.check("backend_cholesky").passed
  assert policy_report.check("backend_pivot_policy").passed

  residual = reused.convergence.reduced_residual_norm
  residual_changed = replace(
    reused,
    convergence=replace(
      reused.convergence,
      reduced_residual_norm=float(np.nextafter(residual, math.inf)),
    ),
  )
  record = residual_changed.verify_record()
  assert not record.passed
  assert not record.check("record_convergence_residual").passed
  assert not residual_changed.verify().passed


def test_coherent_perturbed_field_passes_record_but_fails_fresh_exactly() -> None:
  _, _, _, _, solution = _solve_rational()
  perturbed = _coherent_primary_copy(solution, 2, 1.0e-6)
  assert perturbed.verify_record().passed
  report = perturbed.verify()
  assert not report.passed
  residual = report.check("reduced_equilibrium")
  assert residual.error == pytest.approx(23 / 11250000, rel=2.0e-9, abs=1.0e-15)
  assert residual.error == pytest.approx(2.0444444444444444e-6, rel=2.0e-9)


def test_tiny_scale_wrong_field_has_no_unit_floor() -> None:
  _, _, analysis = _prepared(
    _rational_program(load_value=1.0e-200),
    youngs_modulus=1.0e-200,
  )
  solution = analysis.solve(initial_point=_point(0.0), point=_point(1.0))
  wrong = _coherent_primary_copy(solution, 2, 1.0)
  assert wrong.verify_record().passed
  report = wrong.verify()
  assert not report.passed
  check = report.check("reduced_equilibrium")
  assert check.scale < 1.0e-190
  assert check.normalized_error > 1.0e-3


def test_fresh_verification_catches_constraint_and_coherent_balance_changes() -> None:
  _, _, _, _, solution = _solve_rational()
  broken_constraint = _coherent_primary_copy(solution, 5, 1.0e-6)
  assert broken_constraint.verify_record().passed
  constraint_report = broken_constraint.verify()
  assert not constraint_report.passed
  assert not constraint_report.check('constraint["tie"]').passed

  ledger = solution.ledger
  external = np.array(ledger.external_force.values, copy=True)
  external[2] += 1.0e-6
  residual = external - ledger.internal_force.values
  constraint = -residual
  balance = external + constraint - ledger.internal_force.values
  external_work = float(external @ solution.primary_values.values)
  constraint_work = float(constraint @ solution.primary_values.values)
  coherent_ledger = replace(
    ledger,
    external_force=FinalizedArray(external, dtype=np.float64),
    full_residual=FinalizedArray(residual, dtype=np.float64),
    constraint_force=FinalizedArray(constraint, dtype=np.float64),
    balance=FinalizedArray(balance, dtype=np.float64),
    direct_reactions=FinalizedArray(
      constraint[ledger.direct_reaction_dof_indices.values],
      dtype=np.float64,
    ),
    external_work=external_work,
    constraint_work=constraint_work,
  )
  coherent = _coherent_target_evaluation_copy(
    replace(solution, ledger=coherent_ledger),
    field="nodal_force",
    values=external,
  )
  assert coherent.verify_record().passed
  report = coherent.verify()
  assert not report.passed
  assert not report.check("program_nodal_force").passed
  assert not report.check("external_force_record").passed
  assert not report.check("constraint_force_record").passed


@pytest.mark.parametrize(
  ("ledger_field", "evaluation_field", "check_name"),
  [
    (
      "prescribed_offsets",
      "prescribed_offsets",
      "record_program_prescribed_offsets",
    ),
    ("external_force", "nodal_force", "record_program_nodal_force"),
  ],
)
def test_record_exactly_links_program_evaluation_to_ledger_values(
  ledger_field: str,
  evaluation_field: str,
  check_name: str,
) -> None:
  _, _, _, _, solution = _solve_rational()
  values = np.array(getattr(solution.ledger, ledger_field).values, copy=True)
  values[0] = np.nextafter(values[0], math.inf)
  changed = replace(
    solution,
    ledger=replace(
      solution.ledger,
      **{ledger_field: FinalizedArray(values, dtype=np.float64)},
    ),
  )
  report = changed.verify_record()
  assert not report.passed
  assert not report.check(check_name).passed
  np.testing.assert_array_equal(
    getattr(changed.ledger.program_evaluation, evaluation_field).values,
    getattr(solution.ledger.program_evaluation, evaluation_field).values,
  )


@pytest.mark.parametrize(
  ("field", "check_name"),
  [
    (
      "prescribed_offset_derivatives",
      "program_prescribed_offset_derivatives",
    ),
    ("nodal_force_derivatives", "program_nodal_force_derivatives"),
  ],
)
def test_fresh_verification_compares_program_derivative_meaning(
  field: str,
  check_name: str,
) -> None:
  _, _, _, _, solution = _solve_rational()
  values = np.array(
    getattr(solution.ledger.program_evaluation, field).values,
    copy=True,
  )
  values.flat[0] = np.nextafter(values.flat[0], math.inf)
  changed = _coherent_target_evaluation_copy(
    solution,
    field=field,
    values=values,
  )
  assert changed.verify_record().passed
  report = changed.verify()
  assert not report.passed
  assert not report.check(check_name).passed


def test_poisoned_solve_cache_cannot_affect_fresh_solution_verification() -> None:
  _, _, analysis, _, solution = _solve_rational()
  factor = analysis._workspace.factor
  assert factor is not None
  factor.setflags(write=True)
  factor[:] = np.nan
  factor.setflags(write=False)
  before = analysis.workspace_statistics()
  assert solution.verify().passed
  assert analysis.workspace_statistics() == before
  with pytest.raises(AnalysisSolveError, match="cache"):
    analysis.solve(initial_point=_point(0.0), point=_point(0.5))


def test_patch_test8_analytical_field_is_primary_new_flow_oracle() -> None:
  values = tuple(
    (
      1.0e-3 * x + 5.0e-4 * y,
      5.0e-4 * x + 1.0e-3 * y,
    )
    for x, y in _UNIT_COORDINATES
  )
  constraints = tuple(
    PrescribedDofSpec(
      f"patch-{node}-{component}",
      _dof(node, component),
      _affine(values[node - 1][component_index]),
      _source(f"patch-{node}-{component}"),
    )
    for node in range(1, 9)
    for component_index, component in enumerate(("x", "y"))
  )
  _, _, analysis = _prepared(ProgramSpec(constraints=constraints))
  solution = analysis.solve(
    initial_point=ProgramPoint(),
    point=ProgramPoint(),
  )
  expected = np.asarray(values, dtype=np.float64).reshape(-1)
  np.testing.assert_allclose(
    solution.primary_values.values,
    expected,
    rtol=0.0,
    atol=1.0e-12,
  )
  assert solution.verify().passed


def test_one_shot_uses_same_verified_transition_path() -> None:
  solution = solve(
    _model_spec(),
    _rational_program(),
    LinearStatic(),
    registry=q8_reference_registry(),
    initial_point=_point(0.0),
    point=_point(1.0),
  )
  np.testing.assert_allclose(
    solution.primary_values.values,
    _RATIONAL_DISPLACEMENT,
    rtol=0.0,
    atol=4.0e-14,
  )
  assert solution.transition.accepted
  assert solution.transition.base_generation.ordinal == 0
  assert solution.transition.candidate_generation.ordinal == 1
  assert solution.verify().passed


def test_huge_exact_constraint_id_remains_bounded_under_digit_limit() -> None:
  huge_identifier = 10**5000
  base = _rational_program()
  constraints = (
    replace(base.constraints[0], id=huge_identifier),
    *base.constraints[1:],
  )
  _, _, analysis = _prepared(replace(base, constraints=constraints))
  solution = analysis.solve(initial_point=_point(0.0), point=_point(1.0))
  report = solution.verify()
  assert report.passed
  assert solution.ledger.constraint_ids[0] == huge_identifier
  assert max(len(item.name) for item in report.checks) < 2000
