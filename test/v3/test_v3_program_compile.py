# SPDX-License-Identifier: MIT

"""Correctness matrix for the canonical affine program compiler."""

from __future__ import annotations

import math
import sys
from dataclasses import replace

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from pyfem.v3.compile import (
  ProgramCompilationError,
  ProgramEvaluationError,
  compile_model,
  compile_program,
  evaluate_program,
  q8_reference_registry,
)
from pyfem.v3.model import (
  CompiledModel,
  CompiledProgram,
  ContentFingerprint,
  EntityIndex,
  FinalizedArray,
  ProgramCapabilities,
  SourceMap,
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
  ProgramSpecValidationError,
  RegionSpec,
  SourceContext,
  normalize_program_spec,
)

_HUGE_INTEGER_ID = 10**5000
_HUGE_INTEGER_DECIMAL = "1" + "0" * 5000
_MAX_HUGE_DIAGNOSTIC_LENGTH = 8192


class _ExplosiveForeign:
  calls = 0

  def __getattribute__(self, name: str) -> object:
    del name
    type(self).calls += 1
    raise AssertionError("foreign attribute access executed")


class _ExplosiveString(str):
  calls = 0

  def __new__(cls, value: str) -> _ExplosiveString:
    return super().__new__(cls, value)

  def __hash__(self) -> int:
    type(self).calls += 1
    raise AssertionError("foreign hash executed")

  def __repr__(self) -> str:
    type(self).calls += 1
    raise AssertionError("foreign repr executed")

  def strip(self, chars: str | None = None) -> str:
    del chars
    type(self).calls += 1
    raise AssertionError("foreign strip executed")


def _source(label: str, *, line: int | None = None) -> SourceContext:
  return SourceContext(source=label, line=line)


def _model_spec(
  *,
  node_ids: tuple[str | int, ...] = tuple(range(1, 9)),
) -> ModelSpec:
  coordinates = (
    (0.0, 0.0),
    (0.5, 0.0),
    (1.0, 0.0),
    (1.0, 0.5),
    (1.0, 1.0),
    (0.5, 1.0),
    (0.0, 1.0),
    (0.0, 0.5),
  )
  nodes = tuple(
    NodeSpec(
      id=node_id,
      coordinates=point,
      source=_source(f"node:{index}"),
    )
    for index, (node_id, point) in enumerate(
      zip(node_ids, coordinates, strict=True),
      start=1,
    )
  )
  cell = CellSpec(
    id="cell",
    node_ids=node_ids,
    source=_source("cell"),
  )
  block = CellBlockSpec(
    id="block",
    reference_topology="quadrilateral",
    topological_dimension=2,
    embedding_dimension=2,
    geometry_interpolation="serendipity-quad8",
    cells=(cell,),
    source=_source("block"),
  )
  field = FieldSpec(
    id="displacement",
    components=("x", "y"),
    location="node",
    source=_source("field"),
  )
  material = MaterialSpec(
    id="steel",
    model="plane-stress-linear-elastic",
    parameters=(
      MaterialParameterSpec(
        name="youngs_modulus",
        value=210.0e9,
        source=_source("material:E"),
      ),
      MaterialParameterSpec(
        name="poisson_ratio",
        value=0.3,
        source=_source("material:nu"),
      ),
    ),
    source=_source("material"),
  )
  region = RegionSpec(
    id="region",
    cell_refs=(CellRef(block_id="block", cell_id="cell"),),
    field_ids=("displacement",),
    material_id="steel",
    formulation="small-strain-continuum",
    quadrature="gauss-3x3",
    source=_source("region"),
  )
  return ModelSpec(
    mesh=MeshSpec(nodes=nodes, cell_blocks=(block,), source=_source("mesh")),
    fields=(field,),
    materials=(material,),
    regions=(region,),
    source=_source("model"),
  )


def _compiled_model(
  *,
  node_ids: tuple[str | int, ...] = tuple(range(1, 9)),
) -> CompiledModel:
  return compile_model(_model_spec(node_ids=node_ids), q8_reference_registry())


def _dof(
  node_id: str | int,
  component: str,
  *,
  field_id: str | int = "displacement",
) -> DofRef:
  return DofRef(node_id=node_id, field_id=field_id, component=component)


def _affine(
  constant: int | float = 0.0,
  *coefficients: tuple[str, int | float],
  source: SourceContext | None = None,
) -> AffineValueSpec:
  if source is None:
    source = _source("affine")
  return AffineValueSpec(
    constant=constant,
    coefficients=tuple(
      AffineCoefficientSpec(
        coordinate=name,
        coefficient=value,
        source=_source(f"coefficient:{name}"),
      )
      for name, value in coefficients
    ),
    source=source,
  )


def _lambda_coordinate() -> ProgramCoordinateSpec:
  return ProgramCoordinateSpec(
    name="lambda",
    kind="load",
    source=_source("coordinate:lambda"),
  )


def _oracle_program() -> ProgramSpec:
  root = _dof(1, "x")
  first_slave = _dof(1, "y")
  second_slave = _dof(2, "x")
  return ProgramSpec(
    coordinates=(_lambda_coordinate(),),
    constraints=(
      AffineTieSpec(
        id="first-tie",
        slave=first_slave,
        master=root,
        factor=2.0,
        offset=_affine(3.0, ("lambda", 4.0)),
        source=_source("constraint:first"),
      ),
      AffineTieSpec(
        id="second-tie",
        slave=second_slave,
        master=first_slave,
        factor=-0.5,
        offset=_affine(1.0, ("lambda", -1.0)),
        source=_source("constraint:second"),
      ),
    ),
    loads=(
      NodalLoadSpec(
        id=10,
        target=second_slave,
        value=_affine(5.0, ("lambda", 2.0)),
        source=_source("load:10"),
      ),
      NodalLoadSpec(
        id="z-load",
        target=second_slave,
        value=_affine(-1.0, ("lambda", 3.0)),
        source=_source("load:z"),
      ),
    ),
    source=_source("program"),
  )


def _reidentified_program(
  compiled: CompiledProgram,
  *,
  capabilities: ProgramCapabilities | None = None,
  entity_index: EntityIndex | None = None,
  source_map: SourceMap | None = None,
) -> CompiledProgram:
  import pyfem.v3.compile.program as program_compiler

  selected_capabilities = (
    compiled.capabilities if capabilities is None else capabilities
  )
  selected_entity_index = (
    compiled.entity_index if entity_index is None else entity_index
  )
  selected_source_map = compiled.source_map if source_map is None else source_map
  manifest = program_compiler._program_manifest(
    normalized_manifest=compiled.provenance.normalized_program_manifest,
    model_fingerprint=compiled.compatible_model_content_fingerprint,
    coordinate_names=compiled.coordinate_names,
    coordinate_kinds=compiled.coordinate_kinds,
    constraint_plan=compiled.constraint_plan,
    nodal_load_plan=compiled.nodal_load_plan,
    capabilities=selected_capabilities,
    entity_index=selected_entity_index,
    source_map=selected_source_map,
  )
  return replace(
    compiled,
    content_fingerprint=ContentFingerprint.from_manifest(manifest),
    provenance=replace(compiled.provenance, manifest=manifest),
    capabilities=selected_capabilities,
    entity_index=selected_entity_index,
    source_map=selected_source_map,
  )


def _assert_finalized(array: FinalizedArray, dtype: np.dtype) -> None:
  assert type(array) is FinalizedArray
  assert type(array.values) is np.ndarray
  assert array.values.dtype == dtype
  assert array.values.flags.owndata
  assert array.values.flags.c_contiguous
  assert not array.values.flags.writeable


def test_normalization_is_canonical_exact_and_recursively_caller_detached() -> None:
  time = ProgramCoordinateSpec(" time ", " time ", _source("coordinate:time"))
  load = ProgramCoordinateSpec(" lambda ", " load ", _source("coordinate:load"))
  continuation = ProgramCoordinateSpec(
    " arc ",
    " continuation ",
    _source("coordinate:continuation"),
  )
  coefficient_list = [
    AffineCoefficientSpec("arc", 2, _source("coefficient:arc")),
    AffineCoefficientSpec("lambda", 3, _source("coefficient:lambda")),
  ]
  affine = AffineValueSpec(1, coefficient_list, _source("affine:owned"))
  target = _dof(1, "x")
  prescribed = PrescribedDofSpec(
    " constraint ",
    target,
    affine,
    _source("constraint"),
  )
  load_spec = NodalLoadSpec(" load ", target, affine, _source("load"))
  coordinate_list = [continuation, load, time]
  constraint_list = [prescribed]
  load_list = [load_spec]
  authored = ProgramSpec(
    coordinate_list,
    constraint_list,
    load_list,
    _source("program"),
  )

  normalized = normalize_program_spec(authored)
  coordinate_list.clear()
  constraint_list.clear()
  load_list.clear()
  coefficient_list.clear()
  object.__setattr__(time.source, "source", "mutated-time")
  object.__setattr__(affine, "constant", 99.0)
  object.__setattr__(target, "component", "mutated")
  object.__setattr__(prescribed, "id", "mutated")

  assert type(normalized) is ProgramSpec
  assert normalized is not authored
  assert tuple(item.name for item in normalized.coordinates) == (
    "time",
    "lambda",
    "arc",
  )
  assert tuple(item.kind for item in normalized.coordinates) == (
    "time",
    "load",
    "continuation",
  )
  owned_constraint = normalized.constraints[0]
  assert type(owned_constraint) is PrescribedDofSpec
  assert owned_constraint is not prescribed
  assert owned_constraint.target is not target
  assert owned_constraint.value is not affine
  assert owned_constraint.id == "constraint"
  assert owned_constraint.target == _dof(1, "x")
  assert owned_constraint.value.constant == 1.0
  assert tuple(item.coordinate for item in owned_constraint.value.coefficients) == (
    "lambda",
    "arc",
  )
  assert all(
    owned is not caller
    for owned, caller in zip(
      normalized.coordinates,
      (time, load, continuation),
      strict=True,
    )
  )
  assert normalized.coordinates[0].source.source == "coordinate:time"
  assert normalized.loads[0] is not load_spec
  assert normalized.loads[0].value is not affine
  assert normalized.source is not authored.source


def test_normalization_rejects_foreign_children_before_attribute_access() -> None:
  foreign = _ExplosiveForeign()
  _ExplosiveForeign.calls = 0
  spec = ProgramSpec(coordinates=(foreign,))

  with pytest.raises(ProgramSpecValidationError) as captured:
    normalize_program_spec(spec)

  assert tuple(item.code for item in captured.value.diagnostics) == (
    "invalid-program-coordinate-type",
  )
  assert _ExplosiveForeign.calls == 0


def test_normalization_rejects_string_subclasses_without_custom_hooks() -> None:
  custom = _ExplosiveString("lambda")
  _ExplosiveString.calls = 0
  spec = ProgramSpec(coordinates=(ProgramCoordinateSpec(custom, "load"),))

  with pytest.raises(ProgramSpecValidationError) as captured:
    normalize_program_spec(spec)

  assert tuple(item.code for item in captured.value.diagnostics) == (
    "invalid-program-coordinate-name",
  )
  assert _ExplosiveString.calls == 0


def test_uninitialized_and_wrong_container_specs_have_stable_total_diagnostics() -> (
  None
):
  uninitialized = object.__new__(ProgramSpec)
  wrong_container = ProgramSpec()
  object.__setattr__(wrong_container, "constraints", [])

  for value, code in (
    (uninitialized, "invalid-program-source-value"),
    (wrong_container, "invalid-program-spec-value"),
  ):
    with pytest.raises(ProgramSpecValidationError) as first:
      normalize_program_spec(value)
    with pytest.raises(ProgramSpecValidationError) as second:
      normalize_program_spec(value)
    assert first.value.diagnostics == second.value.diagnostics
    assert first.value.diagnostics[0].code == code
    assert str(first.value) == str(second.value)


def test_program_diagnostics_bound_long_names_and_source_labels() -> None:
  long_name = "coordinate-" + "n" * 20_000
  duplicate = ProgramSpec(
    coordinates=(
      ProgramCoordinateSpec(long_name, "load", _source("first")),
      ProgramCoordinateSpec(long_name, "load", _source("second")),
    )
  )
  with pytest.raises(ProgramSpecValidationError) as first:
    normalize_program_spec(duplicate)
  with pytest.raises(ProgramSpecValidationError) as second:
    normalize_program_spec(duplicate)

  assert str(first.value) == str(second.value)
  assert len(str(first.value)) < _MAX_HUGE_DIAGNOSTIC_LENGTH
  assert long_name not in str(first.value)
  assert "...<truncated>" in str(first.value)

  long_source = "source-" + "s" * 20_000
  invalid_reference = ProgramSpec(
    loads=(
      NodalLoadSpec(
        "load",
        _dof("missing", "x"),
        _affine(1.0),
        SourceContext(source=long_source),
      ),
    )
  )
  with pytest.raises(ProgramCompilationError) as captured:
    compile_program(_compiled_model(), invalid_reference)
  assert len(str(captured.value)) < _MAX_HUGE_DIAGNOSTIC_LENGTH
  assert long_source not in str(captured.value)
  assert "...<truncated>" in str(captured.value)


def test_program_diagnostics_escape_control_characters() -> None:
  name = "lambda\n\t\x00\r"
  first_source = SourceContext(source="first\n\t\x00\r")
  second_source = SourceContext(source="second\n\t\x00\r")
  spec = ProgramSpec(
    coordinates=(
      ProgramCoordinateSpec(name, "load", first_source),
      ProgramCoordinateSpec(name, "load", second_source),
    )
  )

  with pytest.raises(ProgramSpecValidationError) as captured:
    normalize_program_spec(spec)

  rendered = str(captured.value)
  assert len(rendered) < _MAX_HUGE_DIAGNOSTIC_LENGTH
  assert "\n" not in rendered
  assert "\t" not in rendered
  assert "\x00" not in rendered
  assert "\r" not in rendered
  assert "\\n" in rendered
  assert "\\t" in rendered
  assert "\\u0000" in rendered
  assert "\\r" in rendered


@pytest.mark.parametrize(
  ("spec", "code"),
  [
    (
      ProgramSpec(coordinates=(ProgramCoordinateSpec("time", "load"),)),
      "invalid-time-coordinate-reservation",
    ),
    (
      ProgramSpec(
        coordinates=(
          ProgramCoordinateSpec("lambda", "load"),
          ProgramCoordinateSpec("lambda", "continuation"),
        )
      ),
      "duplicate-program-coordinate-name",
    ),
    (
      ProgramSpec(
        coordinates=(_lambda_coordinate(),),
        loads=(
          NodalLoadSpec(
            "load",
            _dof(1, "x"),
            _affine(0.0, ("missing", 1.0)),
          ),
        ),
      ),
      "unknown-affine-coordinate",
    ),
    (
      ProgramSpec(
        coordinates=(_lambda_coordinate(),),
        loads=(
          NodalLoadSpec(
            "load",
            _dof(1, "x"),
            AffineValueSpec(
              0.0,
              (
                AffineCoefficientSpec("lambda", 1.0),
                AffineCoefficientSpec("lambda", 2.0),
              ),
            ),
          ),
        ),
      ),
      "duplicate-affine-coordinate",
    ),
    (
      ProgramSpec(
        constraints=(
          AffineTieSpec(
            "tie",
            _dof(1, "y"),
            _dof(1, "x"),
            0.0,
            _affine(),
          ),
        )
      ),
      "zero-affine-tie-factor",
    ),
  ],
)
def test_coordinate_and_affine_schema_failures_are_deterministic(
  spec: ProgramSpec,
  code: str,
) -> None:
  with pytest.raises(ProgramSpecValidationError) as captured:
    normalize_program_spec(spec)
  assert code in tuple(item.code for item in captured.value.diagnostics)


@pytest.mark.parametrize(
  "value",
  [
    True,
    np.float64(1.0),
    math.inf,
    math.nan,
    pytest.param(_HUGE_INTEGER_ID, id="float64-overflowing-int"),
  ],
)
def test_affine_numbers_require_finite_exact_python_float64_inputs(
  value: object,
) -> None:
  spec = ProgramSpec(
    constraints=(PrescribedDofSpec("fixed", _dof(1, "x"), _affine(value)),)
  )
  with pytest.raises(ProgramSpecValidationError, match="affine constant"):
    normalize_program_spec(spec)


def test_horizon_chain_and_additive_load_oracle_is_literal() -> None:
  compiled = compile_program(_compiled_model(), _oracle_program())
  plan = compiled.constraint_plan
  loads = compiled.nodal_load_plan

  np.testing.assert_array_equal(
    plan.free_dofs.values,
    [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
  )
  np.testing.assert_array_equal(
    plan.row_offsets.values,
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
  )
  np.testing.assert_array_equal(
    plan.column_indices.values,
    [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
  )
  np.testing.assert_array_equal(
    plan.coefficients.values,
    [1.0, 2.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
  )
  np.testing.assert_array_equal(
    plan.offset_constant.values,
    [0.0, 3.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  )
  np.testing.assert_array_equal(
    plan.offset_coordinate_coefficients.values,
    [
      [0.0],
      [4.0],
      [-3.0],
      [0.0],
      [0.0],
      [0.0],
      [0.0],
      [0.0],
      [0.0],
      [0.0],
      [0.0],
      [0.0],
      [0.0],
      [0.0],
      [0.0],
      [0.0],
    ],
  )
  assert loads.load_ids == (10, "z-load")
  np.testing.assert_array_equal(loads.dof_indices.values, [2, 2])
  np.testing.assert_array_equal(loads.constant_values.values, [5.0, -1.0])
  np.testing.assert_array_equal(loads.coordinate_coefficients.values, [[2.0], [3.0]])

  evaluation = evaluate_program(
    compiled,
    ProgramPoint((ProgramCoordinateValue("lambda", 2.0),)),
  )
  np.testing.assert_array_equal(
    evaluation.prescribed_offsets.values,
    [0.0, 11.0, -6.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  )
  np.testing.assert_array_equal(
    evaluation.prescribed_offset_derivatives.values,
    [
      [0.0],
      [4.0],
      [-3.0],
      [0.0],
      [0.0],
      [0.0],
      [0.0],
      [0.0],
      [0.0],
      [0.0],
      [0.0],
      [0.0],
      [0.0],
      [0.0],
      [0.0],
      [0.0],
    ],
  )
  np.testing.assert_array_equal(
    evaluation.nodal_force.values,
    [0.0, 0.0, 14.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  )
  np.testing.assert_array_equal(
    evaluation.nodal_force_derivatives.values,
    [
      [0.0],
      [0.0],
      [5.0],
      [0.0],
      [0.0],
      [0.0],
      [0.0],
      [0.0],
      [0.0],
      [0.0],
      [0.0],
      [0.0],
      [0.0],
      [0.0],
      [0.0],
      [0.0],
    ],
  )


def test_no_constraint_program_has_canonical_int64_identity_plan() -> None:
  compiled = compile_program(_compiled_model(), ProgramSpec())
  plan = compiled.constraint_plan

  assert plan.full_dof_count == plan.reduced_dof_count == 16
  np.testing.assert_array_equal(plan.free_dofs.values, np.arange(16))
  np.testing.assert_array_equal(plan.row_offsets.values, np.arange(17))
  np.testing.assert_array_equal(plan.column_indices.values, np.arange(16))
  np.testing.assert_array_equal(plan.coefficients.values, np.ones(16))
  np.testing.assert_array_equal(plan.offset_constant.values, np.zeros(16))
  assert plan.offset_coordinate_coefficients.values.shape == (16, 0)
  for array in (
    plan.free_dofs,
    plan.row_offsets,
    plan.column_indices,
    compiled.nodal_load_plan.dof_indices,
  ):
    _assert_finalized(array, np.dtype(np.int64))
  for array in (
    plan.coefficients,
    plan.offset_constant,
    plan.offset_coordinate_coefficients,
    compiled.nodal_load_plan.constant_values,
    compiled.nodal_load_plan.coordinate_coefficients,
  ):
    _assert_finalized(array, np.dtype(np.float64))


def test_program_compilation_checks_int64_counts_before_plan_allocation(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  import pyfem.v3.compile.program as program_compiler

  monkeypatch.setattr(program_compiler, "_INT64_MAX", 15)
  with pytest.raises(ProgramCompilationError) as captured:
    compile_program(_compiled_model(), ProgramSpec())

  assert captured.value.diagnostics[0].code == "program-index-overflow"
  assert "full DOF count" in captured.value.diagnostics[0].message


def test_fully_prescribed_program_has_explicit_zero_free_dof_plan() -> None:
  constraints = tuple(
    PrescribedDofSpec(
      identifier,
      _dof(node_id, component),
      _affine(float(identifier)),
    )
    for identifier, (node_id, component) in enumerate(
      (
        (1, "x"),
        (1, "y"),
        (2, "x"),
        (2, "y"),
        (3, "x"),
        (3, "y"),
        (4, "x"),
        (4, "y"),
        (5, "x"),
        (5, "y"),
        (6, "x"),
        (6, "y"),
        (7, "x"),
        (7, "y"),
        (8, "x"),
        (8, "y"),
      )
    )
  )
  compiled = compile_program(_compiled_model(), ProgramSpec(constraints=constraints))
  plan = compiled.constraint_plan

  assert plan.reduced_dof_count == 0
  assert plan.free_dofs.values.shape == (0,)
  np.testing.assert_array_equal(plan.row_offsets.values, np.zeros(17, dtype=np.int64))
  assert plan.column_indices.values.shape == (0,)
  assert plan.coefficients.values.shape == (0,)
  np.testing.assert_array_equal(plan.offset_constant.values, np.arange(16))


def test_chain_through_prescribed_root_composes_negative_factor_and_derivative() -> (
  None
):
  spec = ProgramSpec(
    coordinates=(_lambda_coordinate(),),
    constraints=(
      PrescribedDofSpec(
        "root",
        _dof(1, "x"),
        _affine(2.0, ("lambda", 1.0)),
      ),
      AffineTieSpec(
        "slave",
        _dof(1, "y"),
        _dof(1, "x"),
        -2.0,
        _affine(1.0, ("lambda", -1.0)),
      ),
    ),
  )
  compiled = compile_program(_compiled_model(), spec)
  plan = compiled.constraint_plan

  np.testing.assert_array_equal(plan.row_offsets.values[:3], [0, 0, 0])
  np.testing.assert_array_equal(plan.offset_constant.values[:2], [2.0, -3.0])
  np.testing.assert_array_equal(
    plan.offset_coordinate_coefficients.values[:2],
    [[1.0], [-3.0]],
  )
  evaluation = evaluate_program(
    compiled,
    ProgramPoint((ProgramCoordinateValue("lambda", 4.0),)),
  )
  np.testing.assert_array_equal(evaluation.prescribed_offsets.values[:2], [6.0, -15.0])


def test_multiple_slaves_may_share_one_free_master() -> None:
  spec = ProgramSpec(
    constraints=(
      AffineTieSpec("a", _dof(1, "y"), _dof(1, "x"), 2.0, _affine()),
      AffineTieSpec("b", _dof(2, "x"), _dof(1, "x"), -3.0, _affine()),
    )
  )
  plan = compile_program(_compiled_model(), spec).constraint_plan
  np.testing.assert_array_equal(plan.column_indices.values[:3], [0, 0, 0])
  np.testing.assert_array_equal(plan.coefficients.values[:3], [1.0, 2.0, -3.0])


@pytest.mark.parametrize(
  ("constraints", "code"),
  [
    (
      (AffineTieSpec("self", _dof(1, "x"), _dof(1, "x"), 1.0, _affine()),),
      "self-affine-tie",
    ),
    (
      (
        AffineTieSpec("a", _dof(1, "y"), _dof(1, "x"), 1.0, _affine()),
        AffineTieSpec("b", _dof(1, "y"), _dof(2, "x"), 1.0, _affine()),
      ),
      "duplicate-affine-slave",
    ),
    (
      (
        PrescribedDofSpec("a", _dof(1, "x"), _affine()),
        PrescribedDofSpec("b", _dof(1, "x"), _affine(1.0)),
      ),
      "duplicate-prescribed-dof",
    ),
    (
      (
        PrescribedDofSpec("a", _dof(1, "y"), _affine()),
        AffineTieSpec("b", _dof(1, "y"), _dof(1, "x"), 1.0, _affine()),
      ),
      "prescribed-slave-conflict",
    ),
    (
      (
        AffineTieSpec("a", _dof(1, "x"), _dof(1, "y"), 1.0, _affine()),
        AffineTieSpec("b", _dof(1, "y"), _dof(1, "x"), 1.0, _affine()),
      ),
      "cyclic-affine-constraints",
    ),
    (
      (
        AffineTieSpec("a", _dof(1, "x"), _dof(1, "y"), 1.0, _affine()),
        AffineTieSpec("b", _dof(1, "y"), _dof(2, "x"), 1.0, _affine()),
        AffineTieSpec("c", _dof(2, "x"), _dof(1, "x"), 1.0, _affine()),
      ),
      "cyclic-affine-constraints",
    ),
  ],
)
def test_conflicting_and_cyclic_constraint_graphs_fail_closed(
  constraints: tuple[object, ...],
  code: str,
) -> None:
  with pytest.raises(ProgramCompilationError, match=code):
    compile_program(_compiled_model(), ProgramSpec(constraints=constraints))


@pytest.mark.parametrize(
  "reference",
  [
    _dof(999, "x"),
    _dof(1, "x", field_id="missing"),
    _dof(1, "missing"),
  ],
)
def test_semantic_dof_resolution_rejects_unknown_node_field_and_component(
  reference: DofRef,
) -> None:
  spec = ProgramSpec(
    loads=(NodalLoadSpec("load", reference, _affine(1.0), _source("bad-ref")),)
  )
  with pytest.raises(ProgramCompilationError) as captured:
    compile_program(_compiled_model(), spec)
  assert captured.value.diagnostics[0].source == _source("bad-ref")
  assert captured.value.diagnostics[0].code in {
    "unknown-program-node",
    "unknown-program-field",
    "unknown-program-component",
  }


def test_structured_point_binds_all_coordinates_in_canonical_order() -> None:
  spec = ProgramSpec(
    coordinates=(
      ProgramCoordinateSpec("arc", "continuation"),
      ProgramCoordinateSpec("lambda", "load"),
      ProgramCoordinateSpec("time", "time"),
    ),
    constraints=(
      PrescribedDofSpec(
        "fixed",
        _dof(1, "x"),
        _affine(1.0, ("arc", 2.0), ("time", 3.0), ("lambda", 4.0)),
      ),
    ),
    loads=(
      NodalLoadSpec(
        "load",
        _dof(1, "x"),
        _affine(5.0, ("arc", 6.0), ("time", 7.0), ("lambda", 8.0)),
      ),
    ),
  )
  compiled = compile_program(_compiled_model(), spec)
  evaluation = evaluate_program(
    compiled,
    ProgramPoint(
      (
        ProgramCoordinateValue("arc", 3.0),
        ProgramCoordinateValue("time", 1.0),
        ProgramCoordinateValue("lambda", 2.0),
      )
    ),
  )

  assert compiled.coordinate_names == ("time", "lambda", "arc")
  assert evaluation.coordinate_names == ("time", "lambda", "arc")
  np.testing.assert_array_equal(evaluation.coordinate_values.values, [1.0, 2.0, 3.0])
  assert evaluation.prescribed_offsets.values[0] == 18.0
  np.testing.assert_array_equal(
    evaluation.prescribed_offset_derivatives.values[0],
    [3.0, 4.0, 2.0],
  )
  assert evaluation.nodal_force.values[0] == 46.0
  np.testing.assert_array_equal(
    evaluation.nodal_force_derivatives.values[0],
    [7.0, 8.0, 6.0],
  )


@pytest.mark.parametrize(
  ("point", "code"),
  [
    (ProgramPoint(()), "missing-program-point-coordinate"),
    (
      ProgramPoint(
        (
          ProgramCoordinateValue("lambda", 1.0),
          ProgramCoordinateValue("extra", 2.0),
        )
      ),
      "extra-program-point-coordinate",
    ),
    (
      ProgramPoint(
        (
          ProgramCoordinateValue("lambda", 1.0),
          ProgramCoordinateValue("lambda", 2.0),
        )
      ),
      "duplicate-program-point-coordinate",
    ),
    (
      ProgramPoint((ProgramCoordinateValue("lambda", True),)),
      "invalid-program-point-coordinate",
    ),
    (
      ProgramPoint((ProgramCoordinateValue("lambda", math.inf),)),
      "invalid-program-point-coordinate",
    ),
    (
      ProgramPoint((ProgramCoordinateValue("lambda", _HUGE_INTEGER_ID),)),
      "invalid-program-point-coordinate",
    ),
  ],
)
def test_program_point_failures_are_total_and_deterministic(
  point: ProgramPoint,
  code: str,
) -> None:
  compiled = compile_program(
    _compiled_model(),
    ProgramSpec(coordinates=(_lambda_coordinate(),)),
  )
  with pytest.raises(ProgramEvaluationError) as first:
    evaluate_program(compiled, point)
  with pytest.raises(ProgramEvaluationError) as second:
    evaluate_program(compiled, point)
  assert code in tuple(item.code for item in first.value.diagnostics)
  assert first.value.diagnostics == second.value.diagnostics


def test_uninitialized_exact_program_point_never_leaks_slot_error() -> None:
  compiled = compile_program(_compiled_model(), ProgramSpec())
  point = object.__new__(ProgramPoint)
  with pytest.raises(ProgramEvaluationError, match="initialize every canonical slot"):
    evaluate_program(compiled, point)


def test_semantically_unordered_declarations_preserve_content_identity() -> None:
  forward = _oracle_program()
  reverse = ProgramSpec(
    coordinates=tuple(reversed(forward.coordinates)),
    constraints=tuple(reversed(forward.constraints)),
    loads=tuple(reversed(forward.loads)),
    source=forward.source,
  )
  model = _compiled_model()
  first = compile_program(model, forward)
  second = compile_program(model, reverse)

  assert first.instance_id != second.instance_id
  assert first.content_fingerprint == second.content_fingerprint
  assert first.provenance.manifest.to_bytes() == second.provenance.manifest.to_bytes()
  np.testing.assert_array_equal(
    first.constraint_plan.coefficients.values,
    second.constraint_plan.coefficients.values,
  )
  np.testing.assert_array_equal(
    first.nodal_load_plan.constant_values.values,
    second.nodal_load_plan.constant_values.values,
  )


def test_distinct_content_equivalent_models_keep_live_compatibility_strict() -> None:
  first_model = _compiled_model()
  second_model = _compiled_model()
  first = compile_program(first_model, _oracle_program())
  second = compile_program(second_model, _oracle_program())

  assert first_model.instance_id != second_model.instance_id
  assert first_model.content_fingerprint == second_model.content_fingerprint
  assert first.compatible_model_instance_id == first_model.instance_id
  assert second.compatible_model_instance_id == second_model.instance_id
  assert first.compatible_model_instance_id != second.compatible_model_instance_id
  assert (
    first.compatible_model_content_fingerprint
    == second.compatible_model_content_fingerprint
  )
  assert first.content_fingerprint == second.content_fingerprint
  assert first.instance_id != second.instance_id


def test_semantic_and_source_changes_change_program_content_identity() -> None:
  baseline_spec = _oracle_program()
  model = _compiled_model()
  baseline = compile_program(model, baseline_spec)
  first_load = baseline_spec.loads[0]
  variants = (
    replace(
      baseline_spec,
      loads=(replace(first_load, target=_dof(2, "y")), *baseline_spec.loads[1:]),
    ),
    replace(
      baseline_spec,
      loads=(
        replace(first_load, value=_affine(5.0, ("lambda", 2.5))),
        *baseline_spec.loads[1:],
      ),
    ),
    replace(
      baseline_spec,
      loads=(
        replace(first_load, source=_source("load:changed-source")),
        *baseline_spec.loads[1:],
      ),
    ),
  )
  assert all(
    compile_program(model, variant).content_fingerprint != baseline.content_fingerprint
    for variant in variants
  )


def test_program_entity_and_source_maps_retain_exact_declaration_identity() -> None:
  compiled = compile_program(_compiled_model(), _oracle_program())

  assert compiled.entity_index.lookup("program_coordinate", "lambda").dense_index == 0
  assert (
    compiled.entity_index.lookup(
      "program_constraint",
      "first-tie",
    ).dense_index
    == 0
  )
  assert compiled.entity_index.lookup("program_load", 10).dense_index == 0
  assert compiled.entity_index.lookup("program_load", "z-load").dense_index == 1
  assert compiled.source_map.lookup("program", "program").source == "program"
  assert (
    compiled.source_map.lookup("program_constraint", "second-tie").source
    == "constraint:second"
  )
  assert compiled.source_map.lookup("program_load", 10).source == "load:10"
  with pytest.raises(KeyError, match="exact semantic ID"):
    compiled.entity_index.lookup("program_load", "10")


def test_capabilities_are_conservative_and_derived_from_actual_plans() -> None:
  empty = compile_program(_compiled_model(), ProgramSpec()).capabilities
  populated = compile_program(_compiled_model(), _oracle_program()).capabilities

  assert empty.fixed_constraint_topology
  assert empty.fixed_load_topology
  assert not empty.has_constraints
  assert not empty.has_nodal_loads
  assert empty.contribution_channels == ()
  assert populated.has_constraints
  assert populated.has_nodal_loads
  assert populated.has_coordinate_affine_prescribed
  assert populated.has_coordinate_affine_nodal_loads
  assert not populated.state_dependent
  assert not populated.has_follower_loads
  assert not populated.has_interaction_tangent
  assert not populated.has_program_state
  assert populated.contribution_channels == (
    "prescribed-offset",
    "external-nodal-force",
  )


def test_recomputed_identity_cannot_hide_inconsistent_derived_program_meaning() -> None:
  compiled = compile_program(_compiled_model(), _oracle_program())
  understated = replace(
    compiled.capabilities,
    has_constraints=False,
    has_nodal_loads=False,
    has_coordinate_affine_prescribed=False,
    has_coordinate_affine_nodal_loads=False,
    contribution_channels=(),
  )
  inconsistent_capabilities = _reidentified_program(
    compiled,
    capabilities=understated,
  )
  with pytest.raises(ProgramEvaluationError, match="capabilities do not match"):
    evaluate_program(
      inconsistent_capabilities,
      ProgramPoint((ProgramCoordinateValue("lambda", 1.0),)),
    )

  inconsistent_maps = _reidentified_program(
    compiled,
    entity_index=EntityIndex(()),
    source_map=SourceMap(()),
  )
  with pytest.raises(ProgramEvaluationError, match="entity/source maps do not match"):
    evaluate_program(
      inconsistent_maps,
      ProgramPoint((ProgramCoordinateValue("lambda", 1.0),)),
    )


def test_malformed_exact_compiled_models_fail_before_program_carrier() -> None:
  malformed = object.__new__(CompiledModel)
  with pytest.raises(ProgramCompilationError, match="initialize every canonical slot"):
    compile_program(malformed, ProgramSpec())

  complete = _compiled_model()
  partial = object.__new__(CompiledModel)
  for name in (
    "instance_id",
    "content_fingerprint",
    "provenance",
    "dofs",
    "entity_index",
    "source_map",
  ):
    object.__setattr__(partial, name, object.__getattribute__(complete, name))
  with pytest.raises(ProgramCompilationError, match="every canonical slot"):
    compile_program(partial, ProgramSpec())

  altered = _compiled_model()
  object.__setattr__(altered.dofs, "global_size", 15)
  with pytest.raises(ProgramCompilationError, match="global DOF size"):
    compile_program(altered, ProgramSpec())


@pytest.mark.parametrize(
  "slot",
  [
    "registry_snapshot",
    "mesh",
    "domain_blocks",
    "assembly_topology",
    "physical_state_layout",
    "capabilities",
  ],
)
def test_compiled_model_validates_every_complete_carrier_slot(slot: str) -> None:
  model = _compiled_model()
  object.__setattr__(model, slot, None)

  with pytest.raises(ProgramCompilationError, match="malformed-exact-carrier"):
    compile_program(model, ProgramSpec())


def test_compiled_model_visible_dof_meaning_must_match_retained_provenance() -> None:
  model = _compiled_model()
  changed_field = "altered-displacement"
  object.__setattr__(model.dofs, "field_id", changed_field)
  object.__setattr__(
    model.entity_index,
    "records",
    tuple(
      replace(
        record,
        semantic_id=(record.semantic_id[0], changed_field, record.semantic_id[2]),
      )
      if record.kind == "dof"
      else record
      for record in model.entity_index.records
    ),
  )
  object.__setattr__(
    model.source_map,
    "records",
    tuple(
      replace(
        record,
        semantic_id=(record.semantic_id[0], changed_field, record.semantic_id[2]),
      )
      if record.kind == "dof"
      else record
      for record in model.source_map.records
    ),
  )

  with pytest.raises(ProgramCompilationError, match="visible meaning does not match"):
    compile_program(model, ProgramSpec())


def test_malformed_model_dof_array_and_entity_map_fail_closed() -> None:
  altered_array = _compiled_model()
  object.__setattr__(
    altered_array.dofs,
    "node_component_dofs",
    FinalizedArray(np.arange(16).reshape(8, 2)[:, ::-1], dtype=np.int64),
  )
  with pytest.raises(ProgramCompilationError, match="not canonical dense indices"):
    compile_program(altered_array, ProgramSpec())

  altered_policy = _compiled_model()
  object.__setattr__(altered_policy.provenance, "dense_index_dtype", "<i2")
  with pytest.raises(ProgramCompilationError, match="dense-index policy"):
    compile_program(altered_policy, ProgramSpec())

  insufficient_capacity = _compiled_model()
  node_ids = tuple(range(65))
  wrapped_dofs = np.arange(130, dtype=np.int16).astype(np.int8).reshape(65, 2)
  object.__setattr__(insufficient_capacity.dofs, "node_ids", node_ids)
  object.__setattr__(insufficient_capacity.dofs, "global_size", 130)
  object.__setattr__(
    insufficient_capacity.dofs,
    "node_component_dofs",
    FinalizedArray(wrapped_dofs, dtype=np.int8),
  )
  object.__setattr__(
    insufficient_capacity.provenance,
    "dense_index_dtype",
    np.dtype(np.int8).str,
  )
  with pytest.raises(ProgramCompilationError, match="cannot represent every"):
    compile_program(insufficient_capacity, ProgramSpec())

  altered_map = _compiled_model()
  object.__setattr__(altered_map.entity_index, "records", ())
  with pytest.raises(ProgramCompilationError, match="entity/source maps"):
    compile_program(altered_map, ProgramSpec())


def test_malformed_exact_compiled_programs_fail_before_evaluation_values() -> None:
  malformed = object.__new__(CompiledProgram)
  with pytest.raises(ProgramEvaluationError, match="initialize every canonical slot"):
    evaluate_program(malformed, ProgramPoint())

  altered = compile_program(_compiled_model(), ProgramSpec())
  object.__setattr__(
    altered.constraint_plan,
    "row_offsets",
    FinalizedArray([0], dtype=np.int64),
  )
  with pytest.raises(ProgramEvaluationError, match="wrong exact shape"):
    evaluate_program(altered, ProgramPoint())


def test_malformed_program_meaning_witness_fails_before_child_use() -> None:
  missing_witness = compile_program(_compiled_model(), ProgramSpec())
  object.__setattr__(missing_witness, "meaning_witness", None)
  with pytest.raises(ProgramEvaluationError, match="exact ProgramMeaningWitness"):
    evaluate_program(missing_witness, ProgramPoint())

  missing_dofs = compile_program(_compiled_model(), ProgramSpec())
  object.__setattr__(missing_dofs.meaning_witness, "model_dofs", None)
  with pytest.raises(ProgramEvaluationError, match="exact DofPlan"):
    evaluate_program(missing_dofs, ProgramPoint())

  malformed_coordinate = compile_program(_compiled_model(), _oracle_program())
  object.__setattr__(malformed_coordinate.meaning_witness, "coordinates", (None,))
  with pytest.raises(ProgramEvaluationError, match="coordinate witness must be exact"):
    evaluate_program(malformed_coordinate, ProgramPoint())


def test_deeply_nested_entity_ids_fail_without_recursive_leakage() -> None:
  nested_id: object = "leaf"
  for _ in range(1_500):
    nested_id = (nested_id,)

  model = _compiled_model()
  model_records = model.entity_index.records
  malformed_model = replace(
    model,
    entity_index=EntityIndex(
      (
        replace(model_records[0], semantic_id=nested_id),
        *model_records[1:],
      )
    ),
  )
  with pytest.raises(ProgramCompilationError, match="must be flat"):
    compile_program(malformed_model, ProgramSpec())

  compiled = compile_program(_compiled_model(), ProgramSpec())
  program_records = compiled.entity_index.records
  malformed_program = replace(
    compiled,
    entity_index=EntityIndex(
      (
        replace(program_records[0], semantic_id=nested_id),
        *program_records[1:],
      )
    ),
  )
  with pytest.raises(ProgramEvaluationError, match="must be flat"):
    evaluate_program(malformed_program, ProgramPoint())


def test_compiled_program_detects_writeable_or_content_drifted_arrays() -> None:
  writeable = compile_program(_compiled_model(), ProgramSpec())
  writeable.constraint_plan.free_dofs.values.setflags(write=True)
  with pytest.raises(ProgramEvaluationError, match="read-only"):
    evaluate_program(writeable, ProgramPoint())

  drifted = compile_program(_compiled_model(), ProgramSpec())
  object.__setattr__(drifted.constraint_plan, "full_dof_count", 15)
  with pytest.raises(ProgramEvaluationError, match="constraint-plan counts"):
    evaluate_program(drifted, ProgramPoint())


def test_compiled_program_checks_free_dof_range_without_int64_wrap() -> None:
  compiled = compile_program(_compiled_model(), ProgramSpec())
  int64 = np.iinfo(np.int64)
  wrapped_order = [
    int64.max,
    int64.min,
    *(int64.min + offset for offset in range(1, 15)),
  ]
  object.__setattr__(
    compiled.constraint_plan,
    "free_dofs",
    FinalizedArray(wrapped_order, dtype=np.int64),
  )

  with pytest.raises(ProgramEvaluationError, match="strictly increasing in range"):
    evaluate_program(compiled, ProgramPoint())


def test_compiled_program_requires_separate_numeric_field_storage() -> None:
  compiled = compile_program(_compiled_model(), ProgramSpec())
  object.__setattr__(
    compiled.constraint_plan,
    "column_indices",
    compiled.constraint_plan.free_dofs,
  )

  with pytest.raises(ProgramEvaluationError, match="separate finalized storage"):
    evaluate_program(compiled, ProgramPoint())


@pytest.mark.parametrize(
  ("coefficient", "message"),
  [
    (0.0, "require a nonzero coefficient"),
    (np.nextafter(0.0, 1.0), "does not match normalized program meaning"),
    (np.nextafter(0.0, -1.0), "does not match normalized program meaning"),
  ],
)
def test_dependent_constraint_rows_reject_zero_and_reidentified_nearby_drift(
  coefficient: float,
  message: str,
) -> None:
  compiled = compile_program(_compiled_model(), _oracle_program())
  coefficients = compiled.constraint_plan.coefficients.values.copy()
  coefficients[1] = coefficient
  altered = replace(
    compiled,
    constraint_plan=replace(
      compiled.constraint_plan,
      coefficients=FinalizedArray(coefficients, dtype=np.float64),
    ),
  )
  reidentified = _reidentified_program(altered)

  with pytest.raises(ProgramEvaluationError, match=message):
    evaluate_program(
      reidentified,
      ProgramPoint((ProgramCoordinateValue("lambda", 1.0),)),
    )


def test_reidentified_declaration_ids_and_sources_match_normalized_meaning() -> None:
  compiled = compile_program(_compiled_model(), _oracle_program())
  renamed_constraint_entities = EntityIndex(
    tuple(
      replace(record, semantic_id="a-tie")
      if record.kind == "program_constraint" and record.semantic_id == "first-tie"
      else record
      for record in compiled.entity_index.records
    )
  )
  renamed_constraint_sources = SourceMap(
    tuple(
      replace(record, semantic_id="a-tie")
      if record.kind == "program_constraint" and record.semantic_id == "first-tie"
      else record
      for record in compiled.source_map.records
    )
  )
  renamed_constraint = _reidentified_program(
    replace(
      compiled,
      entity_index=renamed_constraint_entities,
      source_map=renamed_constraint_sources,
    )
  )

  renamed_load_plan = replace(
    compiled.nodal_load_plan,
    load_ids=(9, "z-load"),
  )
  renamed_load_entities = EntityIndex(
    tuple(
      replace(record, semantic_id=9)
      if record.kind == "program_load" and record.semantic_id == 10
      else record
      for record in compiled.entity_index.records
    )
  )
  renamed_load_sources = SourceMap(
    tuple(
      replace(record, semantic_id=9)
      if record.kind == "program_load" and record.semantic_id == 10
      else record
      for record in compiled.source_map.records
    )
  )
  renamed_load = _reidentified_program(
    replace(
      compiled,
      nodal_load_plan=renamed_load_plan,
      entity_index=renamed_load_entities,
      source_map=renamed_load_sources,
    )
  )

  changed_constraint_source = _reidentified_program(
    replace(
      compiled,
      source_map=SourceMap(
        tuple(
          replace(record, source=replace(record.source, source="changed-constraint"))
          if record.kind == "program_constraint" and record.semantic_id == "first-tie"
          else record
          for record in compiled.source_map.records
        )
      ),
    )
  )
  changed_load_source = _reidentified_program(
    replace(
      compiled,
      source_map=SourceMap(
        tuple(
          replace(record, source=replace(record.source, source="changed-load"))
          if record.kind == "program_load" and record.semantic_id == 10
          else record
          for record in compiled.source_map.records
        )
      ),
    )
  )

  for altered in (
    renamed_constraint,
    renamed_load,
    changed_constraint_source,
    changed_load_source,
  ):
    with pytest.raises(
      ProgramEvaluationError,
      match="not match normalized program meaning",
    ):
      evaluate_program(
        altered,
        ProgramPoint((ProgramCoordinateValue("lambda", 1.0),)),
      )


def test_reidentified_visible_coordinate_plan_offset_load_and_target_drift_fails() -> (
  None
):
  compiled = compile_program(_compiled_model(), _oracle_program())

  renamed_coordinate_entities = EntityIndex(
    tuple(
      replace(record, semantic_id="mu")
      if record.kind == "program_coordinate"
      else record
      for record in compiled.entity_index.records
    )
  )
  renamed_coordinate_sources = SourceMap(
    tuple(
      replace(record, semantic_id="mu")
      if record.kind == "program_coordinate"
      else record
      for record in compiled.source_map.records
    )
  )
  coordinate_drift = _reidentified_program(
    replace(
      compiled,
      coordinate_names=("mu",),
      entity_index=renamed_coordinate_entities,
      source_map=renamed_coordinate_sources,
    )
  )

  offsets = compiled.constraint_plan.offset_constant.values.copy()
  offsets[1] += 0.25
  offset_drift = _reidentified_program(
    replace(
      compiled,
      constraint_plan=replace(
        compiled.constraint_plan,
        offset_constant=FinalizedArray(offsets, dtype=np.float64),
      ),
    )
  )

  load_values = compiled.nodal_load_plan.constant_values.values.copy()
  load_values[0] += 0.25
  load_drift = _reidentified_program(
    replace(
      compiled,
      nodal_load_plan=replace(
        compiled.nodal_load_plan,
        constant_values=FinalizedArray(load_values, dtype=np.float64),
      ),
    )
  )

  load_targets = compiled.nodal_load_plan.dof_indices.values.copy()
  load_targets[:] = 3
  target_drift = _reidentified_program(
    replace(
      compiled,
      nodal_load_plan=replace(
        compiled.nodal_load_plan,
        dof_indices=FinalizedArray(load_targets, dtype=np.int64),
      ),
    )
  )

  cases = (
    (coordinate_drift, "coordinate schema"),
    (offset_drift, "affine constraint plan"),
    (load_drift, "nodal-load plan"),
    (target_drift, "nodal-load plan"),
  )
  for altered, message in cases:
    with pytest.raises(ProgramEvaluationError, match=message):
      evaluate_program(
        altered,
        ProgramPoint((ProgramCoordinateValue("lambda", 1.0),)),
      )


def test_compilation_rejects_nonfinite_chain_intermediate() -> None:
  spec = ProgramSpec(
    constraints=(
      PrescribedDofSpec("root", _dof(1, "x"), _affine(1.0e308)),
      AffineTieSpec(
        "slave",
        _dof(1, "y"),
        _dof(1, "x"),
        1.0e308,
        _affine(),
        _source("overflowing-tie"),
      ),
    )
  )
  with pytest.raises(ProgramCompilationError, match="nonfinite-affine-composition"):
    compile_program(_compiled_model(), spec)


def test_evaluation_rejects_nonfinite_affine_product_and_load_reduction() -> None:
  product_program = compile_program(
    _compiled_model(),
    ProgramSpec(
      coordinates=(_lambda_coordinate(),),
      constraints=(
        PrescribedDofSpec(
          "fixed",
          _dof(1, "x"),
          _affine(0.0, ("lambda", 1.0e308)),
        ),
      ),
    ),
  )
  with pytest.raises(ProgramEvaluationError, match="nonfinite-program-evaluation"):
    evaluate_program(
      product_program,
      ProgramPoint((ProgramCoordinateValue("lambda", 2.0),)),
    )

  sum_program = compile_program(
    _compiled_model(),
    ProgramSpec(
      loads=(
        NodalLoadSpec("a", _dof(1, "x"), _affine(1.0e308)),
        NodalLoadSpec("b", _dof(1, "x"), _affine(1.0e308)),
      )
    ),
  )
  with pytest.raises(ProgramEvaluationError, match="nonfinite-program-evaluation"):
    evaluate_program(sum_program, ProgramPoint())


def test_huge_exact_ids_and_sources_are_supported_under_digit_limit() -> None:
  limit_before = sys.get_int_max_str_digits()
  node_ids = tuple(_HUGE_INTEGER_ID + index for index in range(8))
  model = _compiled_model(node_ids=node_ids)
  source = _source("program:huge", line=_HUGE_INTEGER_ID)
  spec = ProgramSpec(
    constraints=(
      PrescribedDofSpec(
        _HUGE_INTEGER_ID + 100,
        _dof(node_ids[0], "x"),
        _affine(1.0),
        source,
      ),
    ),
    loads=(
      NodalLoadSpec(
        _HUGE_INTEGER_ID + 101,
        _dof(node_ids[1], "y"),
        _affine(2.0),
        source,
      ),
    ),
    source=source,
  )

  compiled = compile_program(model, spec)
  assert len(compiled.content_fingerprint.digest) == 64
  assert (
    compiled.entity_index.lookup(
      "program_constraint",
      _HUGE_INTEGER_ID + 100,
    ).dense_index
    == 0
  )
  assert sys.get_int_max_str_digits() == limit_before


def test_huge_source_diagnostic_is_bounded_and_does_not_change_digit_limit() -> None:
  limit_before = sys.get_int_max_str_digits()
  source = _source("program:huge-error", line=_HUGE_INTEGER_ID)
  spec = ProgramSpec(
    loads=(
      NodalLoadSpec(
        _HUGE_INTEGER_ID,
        _dof(_HUGE_INTEGER_ID, "x"),
        _affine(1.0),
        source,
      ),
    ),
    source=source,
  )
  with pytest.raises(ProgramCompilationError) as captured:
    compile_program(_compiled_model(), spec)
  rendered = str(captured.value)
  assert len(rendered) < _MAX_HUGE_DIAGNOSTIC_LENGTH
  assert _HUGE_INTEGER_DECIMAL not in rendered
  assert "<int sign=+" in rendered
  assert sys.get_int_max_str_digits() == limit_before


def test_caller_mutation_after_normalization_compilation_and_evaluation_isolated() -> (
  None
):
  authored = _oracle_program()
  normalized = normalize_program_spec(authored)
  compiled = compile_program(_compiled_model(), normalized)
  fingerprint = compiled.content_fingerprint
  coefficients = compiled.constraint_plan.coefficients.values.copy()
  point_value = ProgramCoordinateValue("lambda", 2.0)
  point = ProgramPoint((point_value,))
  evaluation = evaluate_program(compiled, point)
  offsets = evaluation.prescribed_offsets.values.copy()

  object.__setattr__(authored.constraints[0], "factor", 99.0)
  object.__setattr__(normalized.constraints[0], "factor", 77.0)
  object.__setattr__(normalized.source, "source", "mutated-normalized")
  object.__setattr__(point_value, "value", 99.0)
  object.__setattr__(point, "values", ())

  assert compiled.content_fingerprint == fingerprint
  np.testing.assert_array_equal(
    compiled.constraint_plan.coefficients.values,
    coefficients,
  )
  np.testing.assert_array_equal(evaluation.prescribed_offsets.values, offsets)


def test_every_compiled_and_evaluated_numeric_value_is_separately_finalized() -> None:
  compiled = compile_program(_compiled_model(), _oracle_program())
  evaluation = evaluate_program(
    compiled,
    ProgramPoint((ProgramCoordinateValue("lambda", 2.0),)),
  )
  arrays = (
    compiled.meaning_witness.model_dofs.node_component_dofs,
    compiled.constraint_plan.free_dofs,
    compiled.constraint_plan.row_offsets,
    compiled.constraint_plan.column_indices,
    compiled.constraint_plan.coefficients,
    compiled.constraint_plan.offset_constant,
    compiled.constraint_plan.offset_coordinate_coefficients,
    compiled.nodal_load_plan.dof_indices,
    compiled.nodal_load_plan.constant_values,
    compiled.nodal_load_plan.coordinate_coefficients,
    evaluation.coordinate_values,
    evaluation.prescribed_offsets,
    evaluation.prescribed_offset_derivatives,
    evaluation.nodal_force,
    evaluation.nodal_force_derivatives,
  )
  for array in arrays:
    assert array.values.flags.owndata
    assert not array.values.flags.writeable
  for index, first in enumerate(arrays):
    for second in arrays[index + 1 :]:
      assert not np.shares_memory(first.values, second.values)


def test_constant_displacement_only_and_zero_load_programs_are_explicitly_valid() -> (
  None
):
  displacement_only = compile_program(
    _compiled_model(),
    ProgramSpec(constraints=(PrescribedDofSpec("fixed", _dof(1, "x"), _affine(0.0)),)),
  )
  zero_load = compile_program(
    _compiled_model(),
    ProgramSpec(loads=(NodalLoadSpec("zero", _dof(1, "x"), _affine(0.0)),)),
  )

  first = evaluate_program(displacement_only, ProgramPoint())
  second = evaluate_program(zero_load, ProgramPoint())
  np.testing.assert_array_equal(first.nodal_force.values, np.zeros(16))
  np.testing.assert_array_equal(second.nodal_force.values, np.zeros(16))
  assert displacement_only.nodal_load_plan.load_ids == ()
  assert zero_load.nodal_load_plan.load_ids == ("zero",)
