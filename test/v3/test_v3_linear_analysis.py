# SPDX-License-Identifier: MIT

"""Correctness matrix for the public verified Q8 linear-static flow."""

from __future__ import annotations

import sys
from dataclasses import replace

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

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
  prepare_assembly_plan,
)
from pyfem.v3.compile import compile_model, compile_program, q8_reference_registry
from pyfem.v3.model import (
  CommittedAnalysisState,
  CompiledModel,
  CompiledProgram,
  FinalizedArray,
  InstanceId,
  StateGeneration,
)
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
  assert solution.verify().passed


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

  model = compile_model(_model_spec(), q8_reference_registry())
  program = compile_program(model, _rational_program())
  plan = prepare_assembly_plan(
    model,
    program,
    LinearStaticContributionRequest(),
  )
  oversized_domain = replace(
    plan.domain_coo_plan,
    reduced_shape=(4000, 4000),
  )
  oversized = replace(plan, domain_coo_plan=oversized_domain)
  monkeypatch.setattr(
    linear_module,
    "prepare_assembly_plan",
    lambda *_args: oversized,
  )
  with pytest.raises(AnalysisPreparationError, match="256 MiB"):
    prepare_analysis(model, program, LinearStatic())


def test_factorization_reuse_and_all_published_storage_is_disjoint() -> None:
  _, _, analysis = _prepared()
  first = analysis.solve(initial_point=_point(0.0), point=_point(1.0))
  first_snapshot = np.array(first.primary_values.values, copy=True)
  second = analysis.solve(initial_point=_point(0.0), point=_point(0.5))
  statistics = analysis.workspace_statistics()
  assert statistics.factorization_count == 1
  assert statistics.factorization_reuse_count == 1
  assert first.convergence.factorization_performed
  assert second.convergence.factorization_reused
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
  coherent = replace(solution, ledger=coherent_ledger)
  assert coherent.verify_record().passed
  report = coherent.verify()
  assert not report.passed
  assert not report.check("external_force_record").passed
  assert not report.check("constraint_force_record").passed


def test_poisoned_solve_cache_cannot_affect_fresh_solution_verification() -> None:
  _, _, analysis, _, solution = _solve_rational()
  factor = analysis._workspace.factor
  assert factor is not None
  factor.setflags(write=True)
  factor[:] = np.nan
  factor.setflags(write=False)
  assert solution.verify().passed
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
