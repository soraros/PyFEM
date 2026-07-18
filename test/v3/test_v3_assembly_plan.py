# SPDX-License-Identifier: MIT

"""Correctness matrix for immutable reference linear Q8 assembly."""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import fields, replace

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

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
from pyfem.v3.compile import (
  ModelCompilationPolicy,
  ProgramEvaluationError,
  compile_model,
  compile_program,
  evaluate_program,
  q8_descriptor_metadata,
  q8_reference_registry,
)
from pyfem.v3.compile.contracts import (
  Q8_FORMULATION_KEY,
  Q8_MATERIAL_KEY,
  Q8_QUADRATURE_KEY,
  Q8_TOPOLOGY_KEY,
)
from pyfem.v3.model import (
  CanonicalManifest,
  CompiledMesh,
  CompiledModel,
  CompiledProgram,
  ContentFingerprint,
  DomainBlock,
  FinalizedArray,
  RegistryDescriptor,
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


def _source(label: str) -> SourceContext:
  return SourceContext(source=label)


def _material(
  *,
  youngs_modulus: float = 1.0,
  poisson_ratio: float = 0.0,
) -> MaterialSpec:
  return MaterialSpec(
    id="material",
    model="plane-stress-linear-elastic",
    parameters=(
      MaterialParameterSpec(
        "youngs_modulus",
        youngs_modulus,
        _source("material:E"),
      ),
      MaterialParameterSpec(
        "poisson_ratio",
        poisson_ratio,
        _source("material:nu"),
      ),
    ),
    source=_source("material"),
  )


def _one_cell_model_spec(
  *,
  scale: float = 1.0,
  coordinates: tuple[tuple[float, float], ...] = _UNIT_COORDINATES,
  cell_id: str | int = "cell",
) -> ModelSpec:
  nodes = tuple(
    NodeSpec(
      id=index,
      coordinates=(scale * point[0], scale * point[1]),
      source=_source(f"node:{index}"),
    )
    for index, point in enumerate(coordinates, start=1)
  )
  cell = CellSpec(
    id=cell_id,
    node_ids=tuple(range(1, 9)),
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
  material = _material()
  region = RegionSpec(
    id="region",
    cell_refs=(CellRef(block.id, cell.id),),
    field_ids=(field.id,),
    material_id=material.id,
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


def _two_cell_model_spec(*, reverse_cells: bool = False) -> ModelSpec:
  coordinates = (
    *_UNIT_COORDINATES,
    (1.5, 0.0),
    (2.0, 0.0),
    (2.0, 0.5),
    (2.0, 1.0),
    (1.5, 1.0),
  )
  nodes = tuple(
    NodeSpec(index, point, _source(f"node:{index}"))
    for index, point in enumerate(coordinates, start=1)
  )
  canonical_cells = (
    CellSpec(101, (1, 2, 3, 4, 5, 6, 7, 8), _source("cell:101")),
    CellSpec(102, (3, 9, 10, 11, 12, 13, 5, 4), _source("cell:102")),
  )
  cells = canonical_cells[::-1] if reverse_cells else canonical_cells
  block = CellBlockSpec(
    "block",
    "quadrilateral",
    2,
    2,
    "serendipity-quad8",
    cells,
    _source("block"),
  )
  field = FieldSpec("displacement", ("x", "y"), "node", _source("field"))
  material = _material()
  region = RegionSpec(
    "region",
    tuple(CellRef("block", cell.id) for cell in cells),
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


def _oracle_program_spec() -> ProgramSpec:
  root = _dof(1, "x")
  first_slave = _dof(1, "y")
  second_slave = _dof(2, "x")
  return ProgramSpec(
    coordinates=(ProgramCoordinateSpec("lambda", "load", _source("lambda")),),
    constraints=(
      AffineTieSpec(
        "first",
        first_slave,
        root,
        2.0,
        _affine(3.0, ("lambda", 4.0)),
        _source("constraint:first"),
      ),
      AffineTieSpec(
        "second",
        second_slave,
        first_slave,
        -0.5,
        _affine(1.0, ("lambda", -1.0)),
        _source("constraint:second"),
      ),
    ),
    loads=(
      NodalLoadSpec(
        10,
        second_slave,
        _affine(5.0, ("lambda", 2.0)),
        _source("load:10"),
      ),
      NodalLoadSpec(
        "z-load",
        second_slave,
        _affine(-1.0, ("lambda", 3.0)),
        _source("load:z"),
      ),
    ),
    source=_source("program"),
  )


def _compiled_pair(
  model_spec: ModelSpec,
  program_spec: ProgramSpec = ProgramSpec(),
  *,
  policy: ModelCompilationPolicy | None = None,
  registry: dict[tuple[str, str], RegistryDescriptor] | None = None,
) -> tuple[CompiledModel, CompiledProgram, PreparedAssemblyPlan]:
  selected_registry = q8_reference_registry() if registry is None else registry
  model = compile_model(model_spec, selected_registry, policy=policy)
  program = compile_program(model, program_spec)
  plan = prepare_assembly_plan(
    model,
    program,
    LinearStaticContributionRequest(),
  )
  return model, program, plan


def _evaluate_at(
  model: CompiledModel,
  program: CompiledProgram,
  plan: PreparedAssemblyPlan,
  *,
  coordinate: float | None = None,
) -> LinearStaticContributions:
  point = (
    ProgramPoint()
    if coordinate is None
    else ProgramPoint((ProgramCoordinateValue("lambda", coordinate),))
  )
  return assemble_reference_linear(model, program, plan, point)


def _dense(operator: CanonicalCooOperator) -> np.ndarray:
  dense = np.zeros(operator.shape, dtype=np.float64)
  dense[operator.row_indices.values, operator.column_indices.values] = (
    operator.values.values
  )
  return dense


def _literal_prolongation() -> np.ndarray:
  prolongation = np.zeros((16, 14), dtype=np.float64)
  prolongation[0, 0] = 1.0
  prolongation[1, 0] = 2.0
  prolongation[2, 0] = -1.0
  for full_dof in range(3, 16):
    prolongation[full_dof, full_dof - 2] = 1.0
  return prolongation


def _replace_material(model: ModelSpec, material: MaterialSpec) -> ModelSpec:
  region = replace(model.regions[0], material_id=material.id)
  return replace(model, materials=(material,), regions=(region,))


def _reidentified_coordinates(
  model: CompiledModel,
  coordinates: np.ndarray,
) -> CompiledModel:
  mesh = replace(
    model.mesh,
    coordinates=FinalizedArray(coordinates, dtype=np.float64),
  )
  return _reidentified_model(model, mesh=mesh)


def _reidentified_model(
  model: CompiledModel,
  *,
  mesh: CompiledMesh | None = None,
  domain_block: DomainBlock | None = None,
) -> CompiledModel:
  import pyfem.v3.compile.model as model_compiler

  selected_mesh = model.mesh if mesh is None else mesh
  selected_domain_block = (
    model.domain_blocks[0] if domain_block is None else domain_block
  )
  policy = ModelCompilationPolicy(
    dense_index_dtype=np.dtype(model.provenance.dense_index_dtype).name,
    geometry_relative_tolerance=model.provenance.geometry_relative_tolerance,
  )
  manifest = model_compiler._model_manifest(
    policy=policy,
    registry_snapshot=model.registry_snapshot,
    mesh=selected_mesh,
    dofs=model.dofs,
    domain_block=selected_domain_block,
    assembly_topology=model.assembly_topology,
    physical_state_layout=model.physical_state_layout,
    capabilities=model.capabilities,
    entity_index=model.entity_index,
    source_map=model.source_map,
  )
  return replace(
    model,
    mesh=selected_mesh,
    domain_blocks=(selected_domain_block,),
    content_fingerprint=ContentFingerprint.from_manifest(manifest),
    provenance=replace(model.provenance, manifest=manifest),
  )


def _replace_registry_binding(
  registry: dict[tuple[str, str], RegistryDescriptor],
  key: tuple[str, str],
  binding: Callable[..., object],
  *,
  implementation_id: str,
) -> None:
  original = registry[key]
  registry[key] = RegistryDescriptor(
    kind=key[0],
    name=key[1],
    version=original.version,
    implementation_id=implementation_id,
    metadata=q8_descriptor_metadata(*key),
    binding=binding,
  )


def test_literal_q8_operator_and_affine_rhs_oracles() -> None:
  model, program, plan = _compiled_pair(
    _one_cell_model_spec(),
    _oracle_program_spec(),
  )
  result = _evaluate_at(model, program, plan, coordinate=2.0)
  assert not hasattr(plan, "instance_id")
  assert plan.program_operator_recipes == ()
  assert result.plan_content_fingerprint is plan.content_fingerprint
  assert tuple(field.name for field in fields(result)) == (
    "program_evaluation",
    "plan_content_fingerprint",
    "full_operator",
    "reduced_operator",
    "full_raw_operator_values",
    "reduced_raw_operator_values",
    "full_affine_offset_internal_force",
    "full_affine_offset_internal_force_derivatives",
    "full_offset_corrected_rhs",
    "full_offset_corrected_rhs_derivatives",
    "reduced_external_force",
    "reduced_external_force_derivatives",
    "reduced_affine_offset_internal_force",
    "reduced_affine_offset_internal_force_derivatives",
    "reduced_rhs",
    "reduced_rhs_derivatives",
  )
  expected_full = _Q8_INTEGER_OPERATOR / 360.0
  np.testing.assert_allclose(
    _dense(result.full_operator),
    expected_full,
    rtol=0.0,
    atol=2.0e-14,
  )
  np.testing.assert_allclose(
    result.full_raw_operator_values.values.reshape(16, 16),
    expected_full,
    rtol=0.0,
    atol=2.0e-14,
  )

  prolongation = _literal_prolongation()
  np.testing.assert_allclose(
    _dense(result.reduced_operator),
    prolongation.T @ expected_full @ prolongation,
    rtol=0.0,
    atol=5.0e-13,
  )
  evaluation = result.program_evaluation
  expected_offset_internal = expected_full @ evaluation.prescribed_offsets.values
  expected_offset_derivatives = (
    expected_full @ evaluation.prescribed_offset_derivatives.values
  )
  expected_full_rhs = evaluation.nodal_force.values - expected_offset_internal
  expected_full_rhs_derivatives = (
    evaluation.nodal_force_derivatives.values - expected_offset_derivatives
  )
  np.testing.assert_allclose(
    result.full_affine_offset_internal_force.values,
    expected_offset_internal,
    rtol=0.0,
    atol=5.0e-13,
  )
  np.testing.assert_allclose(
    result.full_affine_offset_internal_force_derivatives.values,
    expected_offset_derivatives,
    rtol=0.0,
    atol=5.0e-13,
  )
  np.testing.assert_allclose(
    result.full_offset_corrected_rhs.values,
    expected_full_rhs,
    rtol=0.0,
    atol=5.0e-13,
  )
  np.testing.assert_allclose(
    result.full_offset_corrected_rhs_derivatives.values,
    expected_full_rhs_derivatives,
    rtol=0.0,
    atol=5.0e-13,
  )
  np.testing.assert_allclose(
    result.reduced_external_force.values,
    prolongation.T @ evaluation.nodal_force.values,
    rtol=0.0,
    atol=5.0e-13,
  )
  np.testing.assert_allclose(
    result.reduced_external_force_derivatives.values,
    prolongation.T @ evaluation.nodal_force_derivatives.values,
    rtol=0.0,
    atol=5.0e-13,
  )
  np.testing.assert_allclose(
    result.reduced_affine_offset_internal_force.values,
    prolongation.T @ expected_offset_internal,
    rtol=0.0,
    atol=5.0e-13,
  )
  np.testing.assert_allclose(
    result.reduced_affine_offset_internal_force_derivatives.values,
    prolongation.T @ expected_offset_derivatives,
    rtol=0.0,
    atol=5.0e-13,
  )
  expected_rhs = np.array(
    [
      -3829 / 72,
      187 / 45,
      -1837 / 360,
      -83 / 20,
      11 / 18,
      343 / 90,
      -167 / 40,
      -206 / 45,
      419 / 90,
      143 / 45,
      -1283 / 360,
      -41 / 10,
      55 / 18,
      977 / 90,
    ],
    dtype=np.float64,
  )
  expected_derivative = np.array(
    [
      -946 / 45,
      68 / 45,
      -12 / 5,
      -139 / 90,
      2 / 9,
      56 / 45,
      -82 / 45,
      -17 / 10,
      94 / 45,
      52 / 45,
      -8 / 5,
      -131 / 90,
      10 / 9,
      184 / 45,
    ],
    dtype=np.float64,
  )
  np.testing.assert_allclose(
    result.reduced_rhs.values,
    expected_rhs,
    rtol=0.0,
    atol=5.0e-13,
  )
  np.testing.assert_allclose(
    result.reduced_rhs_derivatives.values[:, 0],
    expected_derivative,
    rtol=0.0,
    atol=5.0e-13,
  )
  assert result.program_evaluation.prescribed_offsets.values[1] == 11.0
  assert result.program_evaluation.prescribed_offsets.values[2] == -6.5
  assert result.program_evaluation.nodal_force.values[2] == 14.0
  assert result.program_evaluation.prescribed_offset_derivatives.values[1, 0] == 4.0
  assert result.program_evaluation.prescribed_offset_derivatives.values[2, 0] == -3.0
  assert result.program_evaluation.nodal_force_derivatives.values[2, 0] == 5.0

  np.testing.assert_array_equal(_Q8_INTEGER_OPERATOR, _Q8_INTEGER_OPERATOR.T)
  np.testing.assert_array_equal(_Q8_INTEGER_OPERATOR.sum(axis=1), np.zeros(16))
  translation_x = np.tile((1.0, 0.0), 8)
  translation_y = np.tile((0.0, 1.0), 8)
  rotation = np.array(
    [component for x, y in _UNIT_COORDINATES for component in (-y, x)]
  )
  np.testing.assert_array_equal(_Q8_INTEGER_OPERATOR @ translation_x, np.zeros(16))
  np.testing.assert_array_equal(_Q8_INTEGER_OPERATOR @ translation_y, np.zeros(16))
  np.testing.assert_array_equal(_Q8_INTEGER_OPERATOR @ rotation, np.zeros(16))


@pytest.mark.parametrize("scale", [1.0e-200, 1.0, 1.0e200])
def test_scale_normalized_geometry_preserves_literal_operator(scale: float) -> None:
  model, program, plan = _compiled_pair(_one_cell_model_spec(scale=scale))
  result = _evaluate_at(model, program, plan)
  np.testing.assert_allclose(
    _dense(result.full_operator),
    _Q8_INTEGER_OPERATOR / 360.0,
    rtol=0.0,
    atol=2.0e-14,
  )


def test_two_cell_raw_attribution_and_canonical_duplicate_coalescing() -> None:
  model, program, plan = _compiled_pair(_two_cell_model_spec())
  result = _evaluate_at(model, program, plan)
  reversed_model, reversed_program, reversed_plan = _compiled_pair(
    _two_cell_model_spec(reverse_cells=True)
  )
  reversed_result = _evaluate_at(reversed_model, reversed_program, reversed_plan)
  domain = plan.domain_coo_plan
  assert domain.full_raw_row_indices.values.shape == (512,)
  assert domain.full_row_indices.values.shape == (476,)
  assert 512 - 476 == 36
  np.testing.assert_array_equal(domain.block_indices.values, np.zeros(512))
  np.testing.assert_array_equal(domain.cell_indices.values[:256], np.zeros(256))
  np.testing.assert_array_equal(domain.cell_indices.values[256:], np.ones(256))
  np.testing.assert_array_equal(
    domain.local_row_indices.values,
    np.tile(np.repeat(np.arange(16), 16), 2),
  )
  np.testing.assert_array_equal(
    domain.local_column_indices.values,
    np.tile(np.arange(16), 32),
  )
  canonical_pairs = list(
    zip(
      domain.full_row_indices.values.tolist(),
      domain.full_column_indices.values.tolist(),
      strict=True,
    )
  )
  assert canonical_pairs == sorted(set(canonical_pairs))

  expected = np.zeros((26, 26), dtype=np.float64)
  local = _Q8_INTEGER_OPERATOR / 360.0
  for dofs in model.domain_blocks[0].dof_map.values:
    for local_row, full_row in enumerate(dofs):
      for local_column, full_column in enumerate(dofs):
        expected[int(full_row), int(full_column)] += local[local_row, local_column]
  np.testing.assert_allclose(
    _dense(result.full_operator),
    expected,
    rtol=0.0,
    atol=5.0e-14,
  )
  assert reversed_model.domain_blocks[0].cell_ids == model.domain_blocks[0].cell_ids
  np.testing.assert_array_equal(
    reversed_plan.domain_coo_plan.full_raw_row_indices.values,
    domain.full_raw_row_indices.values,
  )
  np.testing.assert_array_equal(
    reversed_plan.domain_coo_plan.full_raw_column_indices.values,
    domain.full_raw_column_indices.values,
  )
  np.testing.assert_allclose(
    reversed_result.full_raw_operator_values.values,
    result.full_raw_operator_values.values,
    rtol=0.0,
    atol=0.0,
  )


def test_identity_reduction_and_narrow_model_indices_lower_to_int64() -> None:
  model, program, plan = _compiled_pair(
    _one_cell_model_spec(),
    policy=ModelCompilationPolicy(dense_index_dtype="int8"),
  )
  result = _evaluate_at(model, program, plan)
  domain = plan.domain_coo_plan
  assert model.domain_blocks[0].dof_map.values.dtype == np.dtype(np.int8)
  for array in (
    domain.full_raw_row_indices,
    domain.full_raw_column_indices,
    domain.full_row_indices,
    domain.full_column_indices,
    domain.reduced_raw_source_indices,
    domain.reduced_raw_row_indices,
    domain.reduced_raw_column_indices,
    domain.reduced_row_indices,
    domain.reduced_column_indices,
  ):
    assert array.values.dtype == np.dtype(np.int64)
  np.testing.assert_array_equal(
    domain.reduced_raw_source_indices.values,
    np.arange(256),
  )
  np.testing.assert_array_equal(
    domain.reduced_raw_left_factors.values,
    np.ones(256),
  )
  np.testing.assert_allclose(
    _dense(result.reduced_operator),
    _dense(result.full_operator),
    rtol=0.0,
    atol=0.0,
  )
  assert result.full_affine_offset_internal_force_derivatives.values.shape == (16, 0)
  assert result.full_offset_corrected_rhs_derivatives.values.shape == (16, 0)
  assert result.reduced_external_force_derivatives.values.shape == (16, 0)
  assert result.reduced_affine_offset_internal_force_derivatives.values.shape == (16, 0)
  assert result.reduced_rhs_derivatives.values.shape == (16, 0)


def test_prescribed_rows_are_removed_and_fully_prescribed_shapes_are_exact() -> None:
  one_constraint = ProgramSpec(
    constraints=(
      PrescribedDofSpec("fixed", _dof(1, "x"), _affine(2.0), _source("fixed")),
    )
  )
  model, program, plan = _compiled_pair(_one_cell_model_spec(), one_constraint)
  result = _evaluate_at(model, program, plan)
  reduced_sources = plan.domain_coo_plan.reduced_raw_source_indices.values
  assert 0 not in plan.domain_coo_plan.full_raw_row_indices.values[reduced_sources]
  assert 0 not in plan.domain_coo_plan.full_raw_column_indices.values[reduced_sources]
  assert 0 not in plan.nodal_vector_plan.full_dof_indices.values
  assert result.reduced_operator.shape == (15, 15)

  constraints = tuple(
    PrescribedDofSpec(
      index,
      _dof(node, component),
      _affine(float(index)),
      _source(f"fixed:{index}"),
    )
    for index, (node, component) in enumerate(
      pair for node in range(1, 9) for pair in ((node, "x"), (node, "y"))
    )
  )
  full_model, full_program, full_plan = _compiled_pair(
    _one_cell_model_spec(),
    ProgramSpec(constraints=constraints),
  )
  full_result = _evaluate_at(full_model, full_program, full_plan)
  assert full_result.reduced_operator.shape == (0, 0)
  assert full_result.reduced_operator.row_indices.values.shape == (0,)
  assert full_result.reduced_operator.column_indices.values.shape == (0,)
  assert full_result.reduced_operator.values.values.shape == (0,)
  assert full_result.reduced_raw_operator_values.values.shape == (0,)
  assert full_result.reduced_external_force.values.shape == (0,)
  assert full_result.reduced_external_force_derivatives.values.shape == (0, 0)
  assert full_result.reduced_affine_offset_internal_force.values.shape == (0,)
  assert full_result.reduced_affine_offset_internal_force_derivatives.values.shape == (
    0,
    0,
  )
  assert full_result.reduced_rhs.values.shape == (0,)
  assert full_result.reduced_rhs_derivatives.values.shape == (0, 0)


def test_load_on_prescribed_dof_survives_full_force_and_reduces_to_zero() -> None:
  target = _dof(1, "x")
  spec = ProgramSpec(
    constraints=(PrescribedDofSpec("fixed", target, _affine(), _source("fixed")),),
    loads=(NodalLoadSpec("load", target, _affine(7.0), _source("load")),),
  )
  model, program, plan = _compiled_pair(_one_cell_model_spec(), spec)
  result = _evaluate_at(model, program, plan)
  assert result.program_evaluation.nodal_force.values[0] == 7.0
  np.testing.assert_array_equal(
    result.reduced_external_force.values,
    np.zeros(15),
  )


def test_repeated_evaluation_reuses_topology_but_all_values_are_fresh() -> None:
  model, program, plan = _compiled_pair(
    _one_cell_model_spec(),
    _oracle_program_spec(),
  )
  first = _evaluate_at(model, program, plan, coordinate=1.0)
  first_rhs = first.reduced_rhs.values.copy()
  second = _evaluate_at(model, program, plan, coordinate=2.0)
  assert first.full_operator.row_indices is plan.domain_coo_plan.full_row_indices
  assert second.full_operator.row_indices is plan.domain_coo_plan.full_row_indices
  assert first.reduced_operator.column_indices is (
    plan.domain_coo_plan.reduced_column_indices
  )
  assert second.reduced_operator.column_indices is (
    plan.domain_coo_plan.reduced_column_indices
  )
  first_arrays = [
    item
    for field in fields(first)
    if isinstance((item := getattr(first, field.name)), FinalizedArray)
  ] + [first.full_operator.values, first.reduced_operator.values]
  second_arrays = [
    item
    for field in fields(second)
    if isinstance((item := getattr(second, field.name)), FinalizedArray)
  ] + [second.full_operator.values, second.reduced_operator.values]
  for left in first_arrays:
    for right in second_arrays:
      assert left is not right
      assert not np.shares_memory(left.values, right.values)
  input_value_arrays = [
    item
    for owner in (
      first.program_evaluation,
      plan.domain_coo_plan,
      plan.nodal_vector_plan,
    )
    for field in fields(owner)
    if isinstance((item := getattr(owner, field.name)), FinalizedArray)
  ]
  for derived in first_arrays:
    for input_value in input_value_arrays:
      assert derived is not input_value
      assert not np.shares_memory(derived.values, input_value.values)
  np.testing.assert_array_equal(first.reduced_rhs.values, first_rhs)
  assert not np.array_equal(first.reduced_rhs.values, second.reduced_rhs.values)


def test_plan_content_is_repeatable_but_live_compatibility_is_strict() -> None:
  model_a, program_a, plan_a = _compiled_pair(_one_cell_model_spec())
  repeated = prepare_assembly_plan(
    model_a,
    program_a,
    LinearStaticContributionRequest(),
  )
  assert repeated.content_fingerprint == plan_a.content_fingerprint

  model_b, program_b, plan_b = _compiled_pair(_one_cell_model_spec())
  assert plan_b.content_fingerprint == plan_a.content_fingerprint
  with pytest.raises(AssemblyPreparationError, match="live-identity"):
    prepare_assembly_plan(
      model_a,
      program_b,
      LinearStaticContributionRequest(),
    )
  with pytest.raises(AssemblyEvaluationError, match="live-identity"):
    assemble_reference_linear(model_b, program_b, plan_a, ProgramPoint())


def test_huge_exact_cell_ids_remain_total_in_plan_identity_and_evaluation() -> None:
  huge_cell_id = 10**5000
  model, program, plan = _compiled_pair(_one_cell_model_spec(cell_id=huge_cell_id))
  result = _evaluate_at(model, program, plan)
  assert model.domain_blocks[0].cell_ids == (huge_cell_id,)
  assert result.plan_content_fingerprint is plan.content_fingerprint
  assert result.full_raw_operator_values.values.shape == (256,)


def test_exact_request_and_malformed_plan_storage_fail_at_assembly_boundary() -> None:
  class _RequestSubclass(LinearStaticContributionRequest):
    pass

  model = compile_model(_one_cell_model_spec(), q8_reference_registry())
  program = compile_program(model, ProgramSpec())
  with pytest.raises(AssemblyPreparationError, match="invalid-assembly-request"):
    prepare_assembly_plan(model, program, _RequestSubclass())

  plan = prepare_assembly_plan(
    model,
    program,
    LinearStaticContributionRequest(),
  )
  plan.domain_coo_plan.full_raw_row_indices.values.setflags(write=True)
  with pytest.raises(AssemblyEvaluationError, match="malformed-assembly-carrier"):
    assemble_reference_linear(model, program, plan, ProgramPoint())

  clean_plan = prepare_assembly_plan(
    model,
    program,
    LinearStaticContributionRequest(),
  )
  aliased_domain = replace(
    clean_plan.domain_coo_plan,
    full_raw_column_indices=clean_plan.domain_coo_plan.full_raw_row_indices,
  )
  aliased_plan = replace(clean_plan, domain_coo_plan=aliased_domain)
  with pytest.raises(AssemblyEvaluationError, match="malformed-assembly-plan"):
    assemble_reference_linear(model, program, aliased_plan, ProgramPoint())

  reused_vector = replace(
    clean_plan.nodal_vector_plan,
    coefficients=program.constraint_plan.coefficients,
  )
  reused_plan = replace(clean_plan, nodal_vector_plan=reused_vector)
  with pytest.raises(AssemblyEvaluationError, match="cannot reuse"):
    assemble_reference_linear(model, program, reused_plan, ProgramPoint())

  missing_plan = object.__new__(PreparedAssemblyPlan)
  object.__setattr__(
    missing_plan,
    "content_fingerprint",
    clean_plan.content_fingerprint,
  )
  with pytest.raises(AssemblyEvaluationError, match="canonical slot"):
    assemble_reference_linear(model, program, missing_plan, ProgramPoint())

  altered_provenance = replace(
    clean_plan.provenance,
    manifest=CanonicalManifest({"schema": "forged-plan"}),
  )
  altered_manifest_plan = replace(clean_plan, provenance=altered_provenance)
  with pytest.raises(AssemblyEvaluationError, match="content identity"):
    assemble_reference_linear(model, program, altered_manifest_plan, ProgramPoint())

  invalid_map_values = clean_plan.domain_coo_plan.full_raw_to_canonical.values.copy()
  invalid_map_values[0] = invalid_map_values[-1]
  invalid_map_domain = replace(
    clean_plan.domain_coo_plan,
    full_raw_to_canonical=FinalizedArray(invalid_map_values, dtype=np.int64),
  )
  invalid_map_plan = replace(clean_plan, domain_coo_plan=invalid_map_domain)
  with pytest.raises(AssemblyEvaluationError, match="raw-to-canonical"):
    assemble_reference_linear(model, program, invalid_map_plan, ProgramPoint())


def test_program_point_diagnostics_translate_without_losing_order_or_source() -> None:
  model, program, plan = _compiled_pair(
    _one_cell_model_spec(),
    _oracle_program_spec(),
  )
  point = ProgramPoint()
  with pytest.raises(ProgramEvaluationError) as original:
    evaluate_program(program, point)
  with pytest.raises(AssemblyEvaluationError) as translated:
    assemble_reference_linear(model, program, plan, point)
  assert [item.code for item in translated.value.diagnostics] == [
    item.code for item in original.value.diagnostics
  ]
  assert [item.message for item in translated.value.diagnostics] == [
    item.message for item in original.value.diagnostics
  ]
  assert [item.source for item in translated.value.diagnostics] == [
    item.source for item in original.value.diagnostics
  ]


@pytest.mark.parametrize(
  "field",
  ["topology", "quadrature", "formulation", "material"],
)
def test_preparation_rechecks_all_captured_descriptor_identities(field: str) -> None:
  model = compile_model(_one_cell_model_spec(), q8_reference_registry())
  block = model.domain_blocks[0]
  forged_identity = replace(
    getattr(block, field),
    implementation_id=f"forged-{field}-binding",
  )
  forged_block = replace(block, **{field: forged_identity})
  forged_model = _reidentified_model(model, domain_block=forged_block)
  program = compile_program(forged_model, ProgramSpec())
  with pytest.raises(AssemblyPreparationError, match="descriptor-identity-mismatch"):
    prepare_assembly_plan(
      forged_model,
      program,
      LinearStaticContributionRequest(),
    )


@pytest.mark.parametrize(
  ("field", "diagnostic_code"),
  [
    ("parent_gradients", "compiled-topology-recipe-mismatch"),
    ("quadrature_points", "compiled-quadrature-recipe-mismatch"),
  ],
)
def test_preparation_requires_exact_reidentified_compiled_recipe_correspondence(
  field: str,
  diagnostic_code: str,
) -> None:
  model = compile_model(_one_cell_model_spec(), q8_reference_registry())
  block = model.domain_blocks[0]
  if field == "parent_gradients":
    altered_values = 2.0 * block.parent_gradients.values
  else:
    altered_values = np.roll(block.quadrature_points.values, 1, axis=0)
  forged_block = replace(
    block,
    **{field: FinalizedArray(altered_values, dtype=np.float64)},
  )
  forged_model = _reidentified_model(model, domain_block=forged_block)
  assert forged_model.content_fingerprint != model.content_fingerprint
  assert forged_model.provenance.manifest is not model.provenance.manifest
  program = compile_program(forged_model, ProgramSpec())

  with pytest.raises(AssemblyPreparationError) as error:
    prepare_assembly_plan(
      forged_model,
      program,
      LinearStaticContributionRequest(),
    )
  assert [item.code for item in error.value.diagnostics] == [diagnostic_code]


def test_preparation_invokes_recipe_once_and_evaluation_only_value_bindings() -> None:
  reference = q8_reference_registry()
  calls = {key: 0 for key in reference}

  def wrapped(key: tuple[str, str]) -> Callable[..., object]:
    binding = reference[key].binding

    def invoke(*args: object) -> object:
      calls[key] += 1
      return binding(*args)

    return invoke

  registry = {
    key: RegistryDescriptor(
      kind=key[0],
      name=key[1],
      version=descriptor.version,
      implementation_id=descriptor.implementation_id,
      metadata=q8_descriptor_metadata(*key),
      binding=wrapped(key),
    )
    for key, descriptor in reference.items()
  }
  model = compile_model(_two_cell_model_spec(), registry)
  program = compile_program(model, ProgramSpec())
  calls.update(dict.fromkeys(calls, 0))
  plan = prepare_assembly_plan(
    model,
    program,
    LinearStaticContributionRequest(),
  )
  assert calls[Q8_TOPOLOGY_KEY] == 1
  assert calls[Q8_QUADRATURE_KEY] == 1
  assert calls[Q8_FORMULATION_KEY] == 0
  assert calls[Q8_MATERIAL_KEY] == 0
  _evaluate_at(model, program, plan)
  assert calls[Q8_TOPOLOGY_KEY] == 1
  assert calls[Q8_QUADRATURE_KEY] == 1
  assert calls[Q8_FORMULATION_KEY] == 2
  assert calls[Q8_MATERIAL_KEY] == 1
  _evaluate_at(model, program, plan)
  assert calls[Q8_TOPOLOGY_KEY] == 1
  assert calls[Q8_QUADRATURE_KEY] == 1
  assert calls[Q8_FORMULATION_KEY] == 4
  assert calls[Q8_MATERIAL_KEY] == 2


@pytest.mark.parametrize(
  ("key", "diagnostic_code"),
  [
    (Q8_QUADRATURE_KEY, "quadrature-binding-failed"),
    (Q8_TOPOLOGY_KEY, "topology-binding-failed"),
  ],
)
def test_preparation_translates_captured_recipe_binding_failures(
  key: tuple[str, str],
  diagnostic_code: str,
) -> None:
  registry = q8_reference_registry()
  reference_binding = registry[key].binding
  call_count = 0

  def fails_after_compilation(*args: object) -> object:
    nonlocal call_count
    call_count += 1
    if call_count == 1:
      return reference_binding(*args)
    raise RuntimeError("incidental preparation binding failure")

  _replace_registry_binding(
    registry,
    key,
    fails_after_compilation,
    implementation_id=f"stateful-{key[0]}-failure-for-assembly-test",
  )
  model = compile_model(_one_cell_model_spec(), registry)
  program = compile_program(model, ProgramSpec())
  with pytest.raises(AssemblyPreparationError) as error:
    prepare_assembly_plan(model, program, LinearStaticContributionRequest())
  assert [item.code for item in error.value.diagnostics] == [diagnostic_code]


@pytest.mark.parametrize(
  ("key", "diagnostic_code"),
  [
    (Q8_QUADRATURE_KEY, "invalid-quadrature-binding-output"),
    (Q8_TOPOLOGY_KEY, "invalid-topology-binding-output"),
  ],
)
def test_preparation_rejects_metadata_bearing_recipe_binding_outputs(
  key: tuple[str, str],
  diagnostic_code: str,
) -> None:
  registry = q8_reference_registry()
  reference_binding = registry[key].binding
  metadata_dtype = np.dtype(np.float64, metadata={"owner": "forged-binding"})
  call_count = 0

  def metadata_binding(*args: object) -> object:
    nonlocal call_count
    call_count += 1
    result = reference_binding(*args)
    assert type(result) is tuple
    if call_count == 1:
      return result
    return tuple(
      item.astype(metadata_dtype, order="C", copy=True, subok=False) for item in result
    )

  _replace_registry_binding(
    registry,
    key,
    metadata_binding,
    implementation_id=f"metadata-{key[0]}-output-for-assembly-test",
  )
  model = compile_model(_one_cell_model_spec(), registry)
  program = compile_program(model, ProgramSpec())
  with pytest.raises(AssemblyPreparationError) as error:
    prepare_assembly_plan(model, program, LinearStaticContributionRequest())
  assert call_count == 2
  assert [item.code for item in error.value.diagnostics] == [diagnostic_code]


@pytest.mark.parametrize("key", [Q8_FORMULATION_KEY, Q8_MATERIAL_KEY])
def test_evaluation_rejects_metadata_bearing_value_binding_outputs(
  key: tuple[str, str],
) -> None:
  registry = q8_reference_registry()
  reference_binding = registry[key].binding
  metadata_dtype = np.dtype(np.float64, metadata={"owner": "forged-binding"})

  def metadata_binding(*args: object) -> np.ndarray:
    return reference_binding(*args).astype(
      metadata_dtype,
      order="C",
      copy=True,
      subok=False,
    )

  _replace_registry_binding(
    registry,
    key,
    metadata_binding,
    implementation_id=f"metadata-{key[0]}-output-for-assembly-test",
  )
  model, program, plan = _compiled_pair(
    _one_cell_model_spec(),
    registry=registry,
  )
  with pytest.raises(AssemblyEvaluationError) as error:
    _evaluate_at(model, program, plan)
  assert [item.code for item in error.value.diagnostics] == [
    "invalid-assembly-binding-output"
  ]


@pytest.mark.parametrize("key", [Q8_FORMULATION_KEY, Q8_MATERIAL_KEY])
def test_wrong_binding_outputs_fail_through_evaluation_diagnostics(
  key: tuple[str, str],
) -> None:
  registry = q8_reference_registry()

  def invalid_binding(*args: object) -> np.ndarray:
    del args
    return np.zeros((1,), dtype=np.float32)

  original = registry[key]
  registry[key] = RegistryDescriptor(
    kind=key[0],
    name=key[1],
    version=original.version,
    implementation_id="invalid-binding-for-assembly-test",
    metadata=q8_descriptor_metadata(*key),
    binding=invalid_binding,
  )
  model, program, plan = _compiled_pair(
    _one_cell_model_spec(),
    registry=registry,
  )
  with pytest.raises(AssemblyEvaluationError, match="binding-output"):
    _evaluate_at(model, program, plan)


@pytest.mark.parametrize(
  ("key", "code"),
  [
    (Q8_FORMULATION_KEY, "formulation-binding-failed"),
    (Q8_MATERIAL_KEY, "material-binding-failed"),
  ],
)
def test_binding_failures_do_not_escape_raw_exceptions(
  key: tuple[str, str],
  code: str,
) -> None:
  registry = q8_reference_registry()

  def failing_binding(*args: object) -> object:
    del args
    raise RuntimeError("incidental binding failure")

  original = registry[key]
  registry[key] = RegistryDescriptor(
    kind=key[0],
    name=key[1],
    version=original.version,
    implementation_id="failing-binding-for-assembly-test",
    metadata=q8_descriptor_metadata(*key),
    binding=failing_binding,
  )
  model, program, plan = _compiled_pair(
    _one_cell_model_spec(),
    registry=registry,
  )
  with pytest.raises(AssemblyEvaluationError, match=code):
    _evaluate_at(model, program, plan)


@pytest.mark.parametrize(
  ("coordinates", "code"),
  [
    (
      tuple(_UNIT_COORDINATES[index] for index in (0, 7, 6, 5, 4, 3, 2, 1)),
      "inverted-reference-geometry",
    ),
    (
      tuple(_UNIT_COORDINATES[index] for index in (0, 1, 4, 3, 2, 5, 6, 7)),
      "sign-changing-reference-geometry",
    ),
    (
      (
        (0.0, 0.0),
        (0.5, 0.0),
        (1.0, 0.0),
        (1.0, 0.5e-14),
        (1.0, 1.0e-14),
        (0.5, 1.0e-14),
        (0.0, 1.0e-14),
        (0.0, 0.5e-14),
      ),
      "near-singular-reference-geometry",
    ),
    (
      (
        (0.0, 0.0),
        (0.5, 0.0),
        (1.0, 0.0),
        (1.0, 0.0),
        (1.0, 0.0),
        (0.5, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
      ),
      "near-singular-reference-geometry",
    ),
  ],
)
def test_evaluation_rechecks_orientation_and_relative_singularity(
  coordinates: tuple[tuple[float, float], ...],
  code: str,
) -> None:
  valid = compile_model(_one_cell_model_spec(), q8_reference_registry())
  forged = _reidentified_coordinates(valid, np.array(coordinates, dtype=np.float64))
  program = compile_program(forged, ProgramSpec())
  plan = prepare_assembly_plan(
    forged,
    program,
    LinearStaticContributionRequest(),
  )
  with pytest.raises(AssemblyEvaluationError, match=code):
    assemble_reference_linear(forged, program, plan, ProgramPoint())
