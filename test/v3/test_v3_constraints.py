# SPDX-License-Identifier: MIT

"""Compiled affine coordinate maps: compilation, algebra, signals, reactions."""

from __future__ import annotations

import copy
import math
import pickle
import sys

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from scipy.sparse import csr_matrix

from pyfem.v3.compile.continuum import q8_reference_registry
from pyfem.v3.compile.system import compile_system
from pyfem.v3.constraints import (
  CONSTRAINT_MAP_MANIFEST_SCHEMA,
  AffineOffsetEvaluation,
  CompiledConstraintMap,
  ConstraintEvaluationError,
  ConstraintMapError,
  admissible_increment,
  compile_constraint_map,
  evaluate_offsets,
  full_coefficients,
  periodic_ties,
  reaction_forces,
  reduce_residual,
  reduce_tangent,
  require_compatible_system,
)
from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.identity import IdentityMismatchError
from pyfem.v3.model.system import CompiledSystem
from pyfem.v3.spec import (
  CellBlockSpec,
  CellRef,
  CellSpec,
  FieldSpec,
  MaterialParameterSpec,
  MaterialSpec,
  MeshSpec,
  ModelSpec,
  NodeSpec,
  RegionSpec,
  SourceContext,
)
from pyfem.v3.spec.program import (
  AffineCoefficientSpec,
  AffineTieSpec,
  AffineValueSpec,
  DofRef,
  PrescribedDofSpec,
  ProgramCoordinateSpec,
  ProgramCoordinateValue,
  ProgramPoint,
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


def _source(label: str) -> SourceContext:
  return SourceContext(source=label)


def _model() -> ModelSpec:
  nodes = tuple(
    NodeSpec(
      id=index + 1,
      coordinates=point,
      source=_source(f"node-source-{index + 1}"),
    )
    for index, point in enumerate(_UNIT_COORDINATES)
  )
  cell = CellSpec(
    id="cell-1",
    node_ids=tuple(node.id for node in nodes),
    source=_source("cell-source"),
  )
  block = CellBlockSpec(
    id="cells",
    reference_topology="quadrilateral",
    topological_dimension=2,
    embedding_dimension=2,
    geometry_interpolation="serendipity-quad8",
    cells=(cell,),
    source=_source("block-source"),
  )
  field = FieldSpec(
    id="displacement",
    components=("x", "y"),
    location="node",
    source=_source("field-source"),
  )
  material = MaterialSpec(
    id="elastic",
    model="plane-stress-linear-elastic",
    parameters=(
      MaterialParameterSpec("youngs_modulus", 1.0, _source("material:E")),
      MaterialParameterSpec("poisson_ratio", 0.25, _source("material:nu")),
    ),
    source=_source("material-source"),
  )
  region = RegionSpec(
    id="domain",
    cell_refs=(CellRef(block.id, cell.id),),
    field_ids=(field.id,),
    material_id=material.id,
    formulation="small-strain-continuum",
    quadrature="gauss-3x3",
    source=_source("region-source"),
  )
  return ModelSpec(
    mesh=MeshSpec(nodes=nodes, cell_blocks=(block,), source=_source("mesh-source")),
    fields=(field,),
    materials=(material,),
    regions=(region,),
    source=_source("model-source"),
  )


def _system() -> CompiledSystem:
  return compile_system(_model(), q8_reference_registry())


def _dof(node_id: int, component: str) -> DofRef:
  return DofRef(node_id=node_id, field_id="displacement", component=component)


def _prescribed(
  identifier: str,
  node_id: int,
  component: str,
  value: AffineValueSpec,
) -> PrescribedDofSpec:
  return PrescribedDofSpec(
    id=identifier,
    target=_dof(node_id, component),
    value=value,
    source=_source(f"constraint:{identifier}"),
  )


def _tie(
  identifier: str,
  slave: DofRef,
  master: DofRef,
  factor: float = 1.0,
  offset: AffineValueSpec | None = None,
) -> AffineTieSpec:
  return AffineTieSpec(
    id=identifier,
    slave=slave,
    master=master,
    factor=factor,
    offset=AffineValueSpec() if offset is None else offset,
    source=_source(f"constraint:{identifier}"),
  )


def _lam() -> ProgramCoordinateSpec:
  return ProgramCoordinateSpec(
    name="lam", kind="load", source=_source("coordinate:lam")
  )


def _point(*entries: tuple[str, float]) -> ProgramPoint:
  return ProgramPoint(
    values=tuple(
      ProgramCoordinateValue(name=name, value=value) for name, value in entries
    )
  )


def _diagnostic_codes(
  error: ConstraintMapError | ConstraintEvaluationError,
) -> tuple[str, ...]:
  return tuple(item.code for item in error.diagnostics)


# --- compile-time coded diagnostics ----------------------------------------


def test_pure_tie_cycle_is_coded_diagnostic_never_a_hang() -> None:
  """The agent-b2 pack-path infinite loop is a compile-time coded error."""
  ties = (
    _tie("a", _dof(1, "x"), _dof(2, "x"), factor=1.0),
    _tie("b", _dof(2, "x"), _dof(1, "x"), factor=1.0),
  )
  with pytest.raises(ConstraintMapError) as info:
    compile_constraint_map(_system(), constraints=ties)
  assert _diagnostic_codes(info.value) == ("cyclic-affine-constraints",)
  assert "constraint:b" in str(info.value)


def test_self_loop_tie_is_coded_diagnostic_never_a_hang() -> None:
  ties = (_tie("self", _dof(1, "x"), _dof(1, "x"), factor=2.0),)
  with pytest.raises(ConstraintMapError) as info:
    compile_constraint_map(_system(), constraints=ties)
  assert _diagnostic_codes(info.value) == ("self-affine-tie",)
  assert "constraint:self" in str(info.value)


def test_three_tie_cycle_is_coded_diagnostic() -> None:
  ties = (
    _tie("a", _dof(1, "x"), _dof(2, "x")),
    _tie("b", _dof(2, "x"), _dof(3, "x")),
    _tie("c", _dof(3, "x"), _dof(1, "x")),
  )
  with pytest.raises(ConstraintMapError) as info:
    compile_constraint_map(_system(), constraints=ties)
  assert _diagnostic_codes(info.value) == ("cyclic-affine-constraints",)


def test_cycle_through_resolved_chain_is_coded_diagnostic() -> None:
  constraints = (
    _prescribed("fix", 4, "x", AffineValueSpec(constant=1.0)),
    _tie("a", _dof(1, "x"), _dof(2, "x")),
    _tie("b", _dof(2, "x"), _dof(3, "x")),
    _tie("c", _dof(3, "x"), _dof(2, "x")),
  )
  with pytest.raises(ConstraintMapError) as info:
    compile_constraint_map(_system(), constraints=constraints)
  assert _diagnostic_codes(info.value) == ("cyclic-affine-constraints",)


def test_duplicate_prescribed_dof_is_rejected() -> None:
  constraints = (
    _prescribed("first", 1, "x", AffineValueSpec(constant=1.0)),
    _prescribed("second", 1, "x", AffineValueSpec(constant=2.0)),
  )
  with pytest.raises(ConstraintMapError) as info:
    compile_constraint_map(_system(), constraints=constraints)
  assert _diagnostic_codes(info.value) == ("duplicate-prescribed-dof",)


def test_prescribed_slave_conflict_is_rejected_in_both_orders() -> None:
  prescribed_first = (
    _prescribed("fix", 1, "x", AffineValueSpec(constant=1.0)),
    _tie("tie", _dof(1, "x"), _dof(2, "x")),
  )
  tie_first = tuple(reversed(prescribed_first))
  for constraints in (prescribed_first, tie_first):
    with pytest.raises(ConstraintMapError) as info:
      compile_constraint_map(_system(), constraints=constraints)
    assert _diagnostic_codes(info.value) == ("prescribed-slave-conflict",)


def test_duplicate_affine_slave_is_rejected() -> None:
  constraints = (
    _tie("first", _dof(1, "x"), _dof(2, "x")),
    _tie("second", _dof(1, "x"), _dof(3, "x")),
  )
  with pytest.raises(ConstraintMapError) as info:
    compile_constraint_map(_system(), constraints=constraints)
  assert _diagnostic_codes(info.value) == ("duplicate-affine-slave",)


def test_unknown_field_is_rejected() -> None:
  constraints = (
    PrescribedDofSpec(
      id="bad-space",
      target=DofRef(node_id=1, field_id="temperature", component="x"),
      value=AffineValueSpec(),
      source=_source("constraint:bad-space"),
    ),
  )
  with pytest.raises(ConstraintMapError) as info:
    compile_constraint_map(_system(), constraints=constraints)
  assert _diagnostic_codes(info.value) == ("unknown-constraint-space",)
  assert "constraint:bad-space" in str(info.value)


def test_unknown_node_is_rejected() -> None:
  constraints = (_prescribed("bad-node", 99, "x", AffineValueSpec()),)
  with pytest.raises(ConstraintMapError) as info:
    compile_constraint_map(_system(), constraints=constraints)
  assert _diagnostic_codes(info.value) == ("unknown-constraint-node",)


def test_unknown_component_is_rejected() -> None:
  constraints = (_prescribed("bad-component", 1, "z", AffineValueSpec()),)
  with pytest.raises(ConstraintMapError) as info:
    compile_constraint_map(_system(), constraints=constraints)
  assert _diagnostic_codes(info.value) == ("unknown-constraint-component",)


def test_undeclared_coordinate_in_offset_is_rejected() -> None:
  constraints = (
    _prescribed(
      "bad-coordinate",
      1,
      "x",
      AffineValueSpec(
        coefficients=(AffineCoefficientSpec(coordinate="ghost", coefficient=1.0),)
      ),
    ),
  )
  with pytest.raises(ConstraintMapError) as info:
    compile_constraint_map(_system(), constraints=constraints)
  assert _diagnostic_codes(info.value) == ("unknown-program-coordinate",)


def test_duplicate_coordinate_declaration_is_rejected() -> None:
  coordinates = (
    ProgramCoordinateSpec(name="lam", kind="load", source=_source("first")),
    ProgramCoordinateSpec(name="lam", kind="time", source=_source("second")),
  )
  with pytest.raises(ConstraintMapError) as info:
    compile_constraint_map(_system(), coordinates=coordinates)
  assert _diagnostic_codes(info.value) == ("duplicate-program-coordinate",)


def test_invalid_coordinate_kind_is_rejected() -> None:
  coordinates = (
    ProgramCoordinateSpec(name="lam", kind="mood", source=_source("kind")),
  )
  with pytest.raises(ConstraintMapError) as info:
    compile_constraint_map(_system(), coordinates=coordinates)
  assert _diagnostic_codes(info.value) == ("invalid-program-coordinate",)


def test_duplicate_coefficient_within_one_value_is_rejected() -> None:
  constraints = (
    _prescribed(
      "dup-coefficient",
      1,
      "x",
      AffineValueSpec(
        coefficients=(
          AffineCoefficientSpec(coordinate="lam", coefficient=1.0),
          AffineCoefficientSpec(coordinate="lam", coefficient=2.0),
        )
      ),
    ),
  )
  with pytest.raises(ConstraintMapError) as info:
    compile_constraint_map(_system(), constraints=constraints, coordinates=(_lam(),))
  assert _diagnostic_codes(info.value) == ("duplicate-coordinate-coefficient",)


def test_non_finite_values_are_rejected() -> None:
  cases = (
    _prescribed("c", 1, "x", AffineValueSpec(constant=math.inf)),
    _prescribed(
      "k",
      1,
      "x",
      AffineValueSpec(
        coefficients=(AffineCoefficientSpec(coordinate="lam", coefficient=math.nan),)
      ),
    ),
    _tie("f", _dof(1, "x"), _dof(2, "x"), factor=math.inf),
    _tie(
      "o",
      _dof(1, "x"),
      _dof(2, "x"),
      factor=1.0,
      offset=AffineValueSpec(constant=-math.inf),
    ),
  )
  for constraint in cases:
    with pytest.raises(ConstraintMapError) as info:
      compile_constraint_map(
        _system(),
        constraints=(constraint,),
        coordinates=(_lam(),),
      )
    assert _diagnostic_codes(info.value) == ("non-finite-constraint-value",)


def test_compilation_type_contracts() -> None:
  with pytest.raises(TypeError):
    compile_constraint_map("not-a-system")  # type: ignore[arg-type]
  with pytest.raises(TypeError):
    compile_constraint_map(_system(), constraints=(_lam(),))  # type: ignore[arg-type]
  with pytest.raises(TypeError):
    compile_constraint_map(_system(), coordinates=[_lam()])  # type: ignore[arg-type]


# --- map algebra against hand-computed references ---------------------------

_MIXED_CONSTRAINTS = (
  _prescribed("fix-x", 1, "x", AffineValueSpec(constant=0.25)),
  _prescribed(
    "fix-y",
    1,
    "y",
    AffineValueSpec(
      constant=-0.5,
      coefficients=(AffineCoefficientSpec(coordinate="lam", coefficient=0.5),),
    ),
  ),
  _tie(
    "tie-a",
    _dof(3, "x"),
    _dof(2, "x"),
    factor=2.0,
    offset=AffineValueSpec(
      constant=0.1,
      coefficients=(AffineCoefficientSpec(coordinate="lam", coefficient=0.25),),
    ),
  ),
  _tie(
    "tie-b",
    _dof(4, "x"),
    _dof(3, "x"),
    factor=-1.0,
    offset=AffineValueSpec(constant=0.05),
  ),
  _tie(
    "tie-c",
    _dof(5, "x"),
    _dof(1, "x"),
    factor=3.0,
    offset=AffineValueSpec(constant=0.02),
  ),
)
_MIXED_CONSTRAINED = (0, 1, 4, 6, 8)
_MIXED_FREE = tuple(dof for dof in range(16) if dof not in _MIXED_CONSTRAINED)


def _mixed_map() -> CompiledConstraintMap:
  return compile_constraint_map(
    _system(),
    constraints=_MIXED_CONSTRAINTS,
    coordinates=(_lam(),),
  )


def _expected_mixed_dense() -> np.ndarray:
  dense = np.zeros((16, len(_MIXED_FREE)), dtype=np.float64)
  for column, dof in enumerate(_MIXED_FREE):
    dense[dof, column] = 1.0
  dense[4, 0] = 2.0
  dense[6, 0] = -2.0
  return dense


def _expected_mixed_offset(lam: float) -> np.ndarray:
  offset = np.zeros(16, dtype=np.float64)
  offset[0] = 0.25
  offset[1] = -0.5 + 0.5 * lam
  offset[4] = 0.1 + 0.25 * lam
  offset[6] = -0.05 - 0.25 * lam
  offset[8] = 0.02 + 3.0 * 0.25
  return offset


def test_compiled_plan_matches_hand_computed_map() -> None:
  constraint_map = _mixed_map()
  assert constraint_map.full_dof_count == 16
  assert constraint_map.reduced_dof_count == len(_MIXED_FREE)
  np.testing.assert_array_equal(
    constraint_map.free_dofs.values,
    np.asarray(_MIXED_FREE, dtype=np.int64),
  )
  np.testing.assert_array_equal(
    constraint_map.constrained_dofs.values,
    np.asarray(_MIXED_CONSTRAINED, dtype=np.int64),
  )
  prolongation = constraint_map.prolongation()
  assert prolongation.shape == (16, len(_MIXED_FREE))
  np.testing.assert_allclose(
    prolongation.toarray(),
    _expected_mixed_dense(),
    rtol=0.0,
    atol=1e-15,
  )


def test_chain_resolution_composes_factors_and_offsets() -> None:
  constraint_map = _mixed_map()
  evaluation = evaluate_offsets(constraint_map, _point(("lam", 2.0)))
  np.testing.assert_allclose(
    evaluation.offsets.values,
    _expected_mixed_offset(2.0),
    rtol=0.0,
    atol=1e-15,
  )
  derivatives = np.zeros(16, dtype=np.float64)
  derivatives[[1, 4, 6]] = (0.5, 0.25, -0.25)
  np.testing.assert_allclose(
    evaluation.derivatives.values[:, 0],
    derivatives,
    rtol=0.0,
    atol=1e-15,
  )


def test_signal_rescaling_reuses_one_compiled_map() -> None:
  constraint_map = _mixed_map()
  for lam in (0.0, 0.37, 1.0, -2.5):
    evaluation = evaluate_offsets(constraint_map, _point(("lam", lam)))
    np.testing.assert_allclose(
      evaluation.offsets.values,
      _expected_mixed_offset(lam),
      rtol=1e-14,
      atol=1e-15,
    )
    assert type(evaluation) is AffineOffsetEvaluation
    assert evaluation.coordinate_names == ("lam",)
    np.testing.assert_array_equal(
      evaluation.coordinate_values.values,
      np.asarray([lam], dtype=np.float64),
    )


def test_full_coefficients_and_admissible_increment() -> None:
  constraint_map = _mixed_map()
  q = np.linspace(-1.0, 1.0, constraint_map.reduced_dof_count)
  lam = 0.75
  full = full_coefficients(constraint_map, q, _point(("lam", lam)))
  expected = _expected_mixed_dense() @ q + _expected_mixed_offset(lam)
  np.testing.assert_allclose(full.values, expected, rtol=1e-14, atol=1e-15)
  # Prescribed rows of P are empty: those entries equal u_bar exactly.
  prescribed_rows = (0, 1, 8)
  np.testing.assert_allclose(
    full.values[list(prescribed_rows)],
    _expected_mixed_offset(lam)[list(prescribed_rows)],
    rtol=0.0,
    atol=1e-15,
  )
  increment = admissible_increment(constraint_map, q)
  np.testing.assert_allclose(
    increment.values,
    _expected_mixed_dense() @ q,
    rtol=1e-14,
    atol=1e-15,
  )
  # Kinematic admissibility: prescribed rows carry no increment.
  assert np.all(increment.values[list(prescribed_rows)] == 0.0)


def test_evaluation_binding_failures_are_coded() -> None:
  constraint_map = _mixed_map()
  cases = (
    ("missing-program-coordinate", _point()),
    ("unknown-program-coordinate", _point(("ghost", 1.0))),
    (
      "duplicate-program-coordinate",
      _point(("lam", 1.0), ("lam", 2.0)),
    ),
    ("non-finite-coordinate-value", _point(("lam", math.nan))),
  )
  for code, point in cases:
    with pytest.raises(ConstraintEvaluationError) as info:
      evaluate_offsets(constraint_map, point)
    assert _diagnostic_codes(info.value) == (code,)


def test_evaluation_type_contracts() -> None:
  constraint_map = _mixed_map()
  with pytest.raises(TypeError):
    evaluate_offsets(constraint_map, {"lam": 1.0})  # type: ignore[arg-type]
  with pytest.raises(TypeError):
    evaluate_offsets("not-a-map", _point(("lam", 1.0)))  # type: ignore[arg-type]
  with pytest.raises(ValueError, match="length 16"):
    reaction_forces(constraint_map, np.zeros(4, dtype=np.float64))
  with pytest.raises(TypeError):
    reduce_tangent(constraint_map, np.eye(16))  # type: ignore[arg-type]
  with pytest.raises(ValueError, match="shape"):
    reduce_tangent(constraint_map, csr_matrix(np.eye(4)))


# --- reduced channels and reaction observation -------------------------------


def test_reduce_residual_matches_dense_projection() -> None:
  constraint_map = _mixed_map()
  rng = np.random.default_rng(20260929)
  residual = rng.standard_normal(constraint_map.full_dof_count)
  reduced = reduce_residual(constraint_map, residual)
  np.testing.assert_allclose(
    reduced.values,
    _expected_mixed_dense().T @ residual,
    rtol=1e-13,
    atol=1e-14,
  )


def test_reduce_tangent_matches_dense_projection_and_stays_sparse() -> None:
  constraint_map = _mixed_map()
  rng = np.random.default_rng(20260929)
  factor = rng.standard_normal((16, 16))
  tangent = csr_matrix(factor.T @ factor + np.eye(16))
  reduced = reduce_tangent(constraint_map, tangent)
  assert type(reduced) is csr_matrix
  assert reduced.shape == (len(_MIXED_FREE), len(_MIXED_FREE))
  np.testing.assert_allclose(
    reduced.toarray(),
    _expected_mixed_dense().T @ tangent.toarray() @ _expected_mixed_dense(),
    rtol=1e-12,
    atol=1e-12,
  )


def test_reaction_observation_from_full_residual() -> None:
  constraint_map = _mixed_map()
  residual = np.arange(1.0, 17.0, dtype=np.float64)
  reactions = reaction_forces(constraint_map, residual)
  expected = np.zeros(16, dtype=np.float64)
  expected[list(_MIXED_CONSTRAINED)] = residual[list(_MIXED_CONSTRAINED)]
  np.testing.assert_array_equal(reactions.values, expected)
  assert np.all(reactions.values[list(_MIXED_FREE)] == 0.0)


def test_full_residual_reactions_and_reduced_equilibrium_agree() -> None:
  """A full residual in the constraint-force basis has zero reduced channel."""
  constraint_map = _mixed_map()
  residual = np.zeros(16, dtype=np.float64)
  residual[list(_MIXED_CONSTRAINED)] = (1.0, 2.0, 3.0, 4.0, 5.0)
  # Reduced equilibrium fixes the free master row: r_2 = -(2*r_4 - 2*r_6).
  residual[2] = -(2.0 * residual[4] - 2.0 * residual[6])
  np.testing.assert_allclose(
    reduce_residual(constraint_map, residual).values,
    np.zeros(constraint_map.reduced_dof_count),
    rtol=0.0,
    atol=1e-14,
  )
  reactions = reaction_forces(constraint_map, residual)
  np.testing.assert_array_equal(
    reactions.values[list(_MIXED_CONSTRAINED)],
    residual[list(_MIXED_CONSTRAINED)],
  )


# --- edge cases ---------------------------------------------------------------


def test_empty_declarations_compile_the_identity_map() -> None:
  constraint_map = compile_constraint_map(_system())
  assert constraint_map.full_dof_count == 16
  assert constraint_map.reduced_dof_count == 16
  np.testing.assert_array_equal(
    constraint_map.prolongation().toarray(),
    np.eye(16),
  )
  evaluation = evaluate_offsets(constraint_map, ProgramPoint())
  np.testing.assert_array_equal(
    evaluation.offsets.values,
    np.zeros(16),
  )
  assert evaluation.derivatives.values.shape == (16, 0)
  residual = np.arange(16.0)
  np.testing.assert_array_equal(
    reduce_residual(constraint_map, residual).values,
    residual,
  )
  assert not reaction_forces(constraint_map, residual).values.any()


def test_fully_prescribed_system_has_explicit_zero_reduced_space() -> None:
  constraints = tuple(
    _prescribed(
      f"fix-{dof}",
      dof // 2 + 1,
      ("x", "y")[dof % 2],
      AffineValueSpec(constant=float(dof)),
    )
    for dof in range(16)
  )
  constraint_map = compile_constraint_map(_system(), constraints=constraints)
  assert constraint_map.reduced_dof_count == 0
  assert constraint_map.prolongation().shape == (16, 0)
  full = full_coefficients(constraint_map, np.empty(0), ProgramPoint())
  np.testing.assert_array_equal(full.values, np.arange(16.0))
  assert reduce_residual(constraint_map, np.ones(16)).values.shape == (0,)
  tangent = reduce_tangent(constraint_map, csr_matrix(np.eye(16)))
  assert tangent.shape == (0, 0)
  reactions = reaction_forces(constraint_map, np.arange(16.0))
  np.testing.assert_array_equal(reactions.values, np.arange(16.0))


# --- artifact identity, immutability, and provenance --------------------------


def test_compiled_map_is_compiler_constructed_and_immutable() -> None:
  constraint_map = _mixed_map()
  with pytest.raises(TypeError):
    CompiledConstraintMap()
  with pytest.raises(TypeError):
    copy.copy(constraint_map)
  with pytest.raises(TypeError):
    copy.deepcopy(constraint_map)
  with pytest.raises(TypeError):
    pickle.dumps(constraint_map)
  for array in (
    constraint_map.free_dofs,
    constraint_map.row_offsets,
    constraint_map.column_indices,
    constraint_map.coefficients,
    constraint_map.offset_constant,
    constraint_map.offset_coordinate_coefficients,
  ):
    assert type(array) is FinalizedArray
    assert not array.values.flags.writeable


def test_prolongation_hands_out_fresh_wrappers() -> None:
  constraint_map = _mixed_map()
  first = constraint_map.prolongation()
  second = constraint_map.prolongation()
  assert first is not second
  first.data[:] = 0.0
  np.testing.assert_allclose(
    second.toarray(),
    _expected_mixed_dense(),
    rtol=0.0,
    atol=1e-15,
  )
  np.testing.assert_allclose(
    constraint_map.prolongation().toarray(),
    _expected_mixed_dense(),
    rtol=0.0,
    atol=1e-15,
  )


def test_content_identity_tracks_declarations_not_instances() -> None:
  first = compile_constraint_map(
    _system(),
    constraints=_MIXED_CONSTRAINTS,
    coordinates=(_lam(),),
  )
  second = compile_constraint_map(
    _system(),
    constraints=_MIXED_CONSTRAINTS,
    coordinates=(_lam(),),
  )
  assert first.instance_id != second.instance_id
  assert first.content_fingerprint == second.content_fingerprint
  assert first.provenance.schema == CONSTRAINT_MAP_MANIFEST_SCHEMA
  assert first.provenance.coordinate_names == ("lam",)
  assert first.provenance.coordinate_kinds == ("load",)
  changed = compile_constraint_map(
    _system(),
    constraints=(*_MIXED_CONSTRAINTS, _prescribed("extra", 8, "y", AffineValueSpec())),
    coordinates=(_lam(),),
  )
  assert changed.content_fingerprint != first.content_fingerprint


def test_system_composition_requires_the_exact_live_system() -> None:
  system = _system()
  constraint_map = compile_constraint_map(
    system,
    constraints=_MIXED_CONSTRAINTS,
    coordinates=(_lam(),),
  )
  require_compatible_system(constraint_map, system)
  foreign = _system()
  with pytest.raises(IdentityMismatchError):
    require_compatible_system(constraint_map, foreign)
  with pytest.raises(TypeError):
    require_compatible_system(constraint_map, "not-a-system")  # type: ignore[arg-type]


# --- periodic pairings as researcher-declared data ----------------------------


def test_periodic_ties_expand_pair_major_then_component() -> None:
  offset = AffineValueSpec(
    coefficients=(AffineCoefficientSpec(coordinate="ex", coefficient=0.24),)
  )
  ties = periodic_ties(
    primary_node_ids=(1, 8, 7),
    image_node_ids=(3, 4, 5),
    field_id="displacement",
    components=("x", "y"),
    offsets=(offset, AffineValueSpec()),
    id_prefix="rve-x",
    source=_source("periodic"),
  )
  assert len(ties) == 6
  assert tuple(tie.id for tie in ties) == (
    "rve-x:0:x",
    "rve-x:0:y",
    "rve-x:1:x",
    "rve-x:1:y",
    "rve-x:2:x",
    "rve-x:2:y",
  )
  first = ties[0]
  assert first.slave.node_id == 3
  assert first.master.node_id == 1
  assert first.factor == 1.0
  assert first.offset is offset
  assert ties[1].offset.constant == 0.0
  constraint_map = compile_constraint_map(
    _system(),
    constraints=ties,
    coordinates=(
      ProgramCoordinateSpec(name="ex", kind="continuation", source=_source("ex")),
    ),
  )
  evaluation = evaluate_offsets(constraint_map, _point(("ex", 1.0e-3)))
  offset_matrix = evaluation.offsets.values.reshape(8, 2)
  np.testing.assert_allclose(
    offset_matrix[[2, 3, 4], 0],
    np.full(3, 0.24e-3),
    rtol=0.0,
    atol=1e-15,
  )
  assert np.all(offset_matrix[:, 1] == 0.0)


def test_periodic_ties_validation() -> None:
  base = {
    "primary_node_ids": (1, 8, 7),
    "image_node_ids": (3, 4, 5),
    "field_id": "displacement",
    "components": ("x", "y"),
  }
  with pytest.raises(ValueError, match="equally many"):
    periodic_ties(**{**base, "image_node_ids": (3, 4)})  # type: ignore[arg-type]
  with pytest.raises(ValueError, match="repeat a primary"):
    periodic_ties(**{**base, "primary_node_ids": (1, 1, 7)})  # type: ignore[arg-type]
  with pytest.raises(ValueError, match="repeat an image"):
    periodic_ties(**{**base, "image_node_ids": (3, 3, 5)})  # type: ignore[arg-type]
  with pytest.raises(ValueError, match="both primary and image"):
    periodic_ties(**{**base, "image_node_ids": (1, 4, 5)})  # type: ignore[arg-type]
  with pytest.raises(ValueError, match="at least one node pair"):
    periodic_ties(**{**base, "primary_node_ids": (), "image_node_ids": ()})  # type: ignore[arg-type]
  with pytest.raises(ValueError, match="at least one field component"):
    periodic_ties(**{**base, "components": ()})  # type: ignore[arg-type]
  with pytest.raises(ValueError, match="one-to-one"):
    periodic_ties(**{**base, "offsets": (AffineValueSpec(),)})  # type: ignore[arg-type]
  with pytest.raises(ValueError, match="finite"):
    periodic_ties(**{**base, "factor": math.nan})  # type: ignore[arg-type]
  with pytest.raises(TypeError):
    periodic_ties(**{**base, "offsets": (0.1, 0.2)})  # type: ignore[arg-type]
  with pytest.raises(TypeError):
    periodic_ties(**{**base, "id_prefix": ""})
