# SPDX-License-Identifier: MIT

"""Driver assembly plan: compiled artifact, values-only refill, affine loads."""

from __future__ import annotations

import sys

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from scipy.sparse import coo_matrix

from pyfem.v3.compile.continuum import q8_reference_registry
from pyfem.v3.compile.spring import (
  compile_spring_operator,
  compose_system,
  damage_envelope_declaration,
)
from pyfem.v3.compile.system import compile_system
from pyfem.v3.compile.truss import truss_reference_registry
from pyfem.v3.constraints import CompiledConstraintMap, compile_constraint_map
from pyfem.v3.driver import (
  DRIVER_ASSEMBLY_PLAN_MANIFEST_SCHEMA,
  DriverEvaluationError,
  DriverPreparationError,
  assemble_internal_force,
  compile_driver_plan,
  evaluate_loads,
  refill_tangent,
)
from pyfem.v3.fem.assembly import dedup_coo_segments
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
  AffineValueSpec,
  DofRef,
  NodalLoadSpec,
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


def _truss_model() -> ModelSpec:
  nodes = (
    NodeSpec(id=0, coordinates=(-10.0, 0.0), source=_source("n0")),
    NodeSpec(id=1, coordinates=(10.0, 0.0), source=_source("n1")),
    NodeSpec(id=2, coordinates=(0.0, 0.5), source=_source("n2")),
  )
  cells = (
    CellSpec(id="left", node_ids=(0, 2), source=_source("c0")),
    CellSpec(id="right", node_ids=(1, 2), source=_source("c1")),
  )
  block = CellBlockSpec(
    id="bars",
    reference_topology="line",
    topological_dimension=1,
    embedding_dimension=2,
    geometry_interpolation="line2",
    cells=cells,
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
    model="uniaxial-linear-elastic",
    parameters=(
      MaterialParameterSpec("youngs_modulus", 5.0e6),
      MaterialParameterSpec("area", 1.0),
    ),
    source=_source("material"),
  )
  region = RegionSpec(
    id="domain",
    cell_refs=(CellRef("bars", "left"), CellRef("bars", "right")),
    field_ids=("displacement",),
    material_id="steel",
    formulation="total-lagrangian-truss",
    quadrature="none",
    source=_source("region"),
  )
  return ModelSpec(
    mesh=MeshSpec(nodes=nodes, cell_blocks=(block,), source=_source("mesh")),
    fields=(field,),
    materials=(material,),
    regions=(region,),
    source=_source("model"),
  )


def _q8_model() -> ModelSpec:
  nodes = tuple(
    NodeSpec(id=index + 1, coordinates=point, source=_source(f"n{index + 1}"))
    for index, point in enumerate(_UNIT_COORDINATES)
  )
  cell = CellSpec(
    id="cell-1",
    node_ids=tuple(node.id for node in nodes),
    source=_source("cell"),
  )
  block = CellBlockSpec(
    id="cells",
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
    id="elastic",
    model="plane-stress-linear-elastic",
    parameters=(
      MaterialParameterSpec("youngs_modulus", 1.0e6),
      MaterialParameterSpec("poisson_ratio", 0.25),
    ),
    source=_source("material"),
  )
  region = RegionSpec(
    id="domain",
    cell_refs=(CellRef("cells", "cell-1"),),
    field_ids=("displacement",),
    material_id="elastic",
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


def _fixed_truss_map() -> tuple[CompiledSystem, CompiledConstraintMap]:
  system = compile_system(_truss_model(), truss_reference_registry())
  coordinate_map = compile_constraint_map(
    system,
    constraints=tuple(
      PrescribedDofSpec(
        id=f"fix-{node}-{component}",
        target=DofRef(
          node_id=node,
          field_id="displacement",
          component=component,
        ),
        value=AffineValueSpec(constant=0.0),
        source=_source(f"fix-{node}-{component}"),
      )
      for node in (0, 1)
      for component in ("x", "y")
    ),
    coordinates=(ProgramCoordinateSpec(name="load", kind="load"),),
  )
  return system, coordinate_map


def _apex_load(value: float = -100.0) -> tuple[NodalLoadSpec, ...]:
  return (
    NodalLoadSpec(
      id="apex",
      target=DofRef(node_id=2, field_id="displacement", component="y"),
      value=AffineValueSpec(
        coefficients=(AffineCoefficientSpec("load", value, _source("coef")),),
        source=_source("load"),
      ),
      source=_source("apex"),
    ),
  )


def _point(value: float) -> ProgramPoint:
  return ProgramPoint((ProgramCoordinateValue("load", value),))


def test_plan_binds_exact_identities_and_topology() -> None:
  system, coordinate_map = _fixed_truss_map()
  plan = compile_driver_plan(system, coordinate_map, _apex_load())
  assert plan.provenance.schema == DRIVER_ASSEMBLY_PLAN_MANIFEST_SCHEMA
  assert plan.compatible_system_instance_id == system.instance_id
  assert plan.compatible_system_content_fingerprint == system.content_fingerprint
  assert plan.compatible_map_instance_id == coordinate_map.instance_id
  assert plan.compatible_map_content_fingerprint == coordinate_map.content_fingerprint
  assert plan.full_dof_count == 6
  assert plan.reduced_dof_count == 2
  assert plan.coordinate_names == ("load",)
  assert plan.constant_tangent is False
  assert len(plan.operator_slices) == 1
  slice_ = plan.operator_slices[0]
  assert slice_.entity_count == 2
  assert slice_.element_dof_count == 4
  assert plan.coo_entry_count == 2 * 16
  np.testing.assert_array_equal(
    slice_.gather.values,
    system.operators[0].header.ports[0].coefficient_map.values,
  )


def test_plan_flags_linear_tangent_from_channel_metadata() -> None:
  system = compile_system(_q8_model(), q8_reference_registry())
  coordinate_map = compile_constraint_map(system)
  plan = compile_driver_plan(system, coordinate_map)
  assert plan.constant_tangent is True


def test_plan_requires_exact_types() -> None:
  system, coordinate_map = _fixed_truss_map()
  with pytest.raises(TypeError, match="exact CompiledSystem"):
    compile_driver_plan(object(), coordinate_map)  # type: ignore[arg-type]
  with pytest.raises(TypeError, match="exact CompiledConstraintMap"):
    compile_driver_plan(system, object())  # type: ignore[arg-type]
  with pytest.raises(TypeError, match="exact tuple of NodalLoadSpec"):
    compile_driver_plan(system, coordinate_map, loads=[object()])  # type: ignore[list-item]


def test_plan_rejects_foreign_map_instance() -> None:
  system, _coordinate_map = _fixed_truss_map()
  _other_system, other_map = _fixed_truss_map()
  with pytest.raises(IdentityMismatchError):
    compile_driver_plan(system, other_map)


def test_refill_matches_fresh_scipy_conversion_and_is_deterministic() -> None:
  system, coordinate_map = _fixed_truss_map()
  plan = compile_driver_plan(system, coordinate_map, _apex_load())
  rng = np.random.default_rng(20260930)
  for _ in range(8):
    batch = rng.normal(0.0, 1.0e3, (2, 4, 4))
    refilled = refill_tangent(plan, (batch,))
    reference = coo_matrix(
      (
        batch.reshape(-1),
        (plan.coo_row_indices.values, plan.coo_column_indices.values),
      ),
      shape=plan.csr_shape,
    ).tocsr()
    reference.sort_indices()
    np.testing.assert_array_equal(refilled.indptr, reference.indptr)
    np.testing.assert_array_equal(refilled.indices, reference.indices)
    np.testing.assert_allclose(
      refilled.toarray(),
      reference.toarray(),
      rtol=0.0,
      atol=1.0e-12,
    )
    again = refill_tangent(plan, (batch,))
    np.testing.assert_array_equal(refilled.toarray(), again.toarray())


def test_refill_matches_manual_element_loop() -> None:
  system, coordinate_map = _fixed_truss_map()
  plan = compile_driver_plan(system, coordinate_map, _apex_load())
  gather = plan.operator_slices[0].gather.values
  rng = np.random.default_rng(7)
  batch = rng.normal(0.0, 1.0, (2, 4, 4))
  expected = np.zeros(plan.csr_shape)
  for element in range(2):
    dofs = gather[element]
    expected[np.ix_(dofs, dofs)] += batch[element]
  np.testing.assert_array_equal(refill_tangent(plan, (batch,)).toarray(), expected)


def _q8_patch_model() -> ModelSpec:
  """A 2x2 patch of unit Q8 cells (5x5 nodes at half-unit spacing)."""

  def node_id(hx: int, hy: int) -> int:
    return hy * 5 + hx + 1

  nodes = tuple(
    NodeSpec(
      id=node_id(hx, hy),
      coordinates=(0.5 * hx, 0.5 * hy),
      source=_source(f"n{node_id(hx, hy)}"),
    )
    for hy in range(5)
    for hx in range(5)
  )
  cells = tuple(
    CellSpec(
      id=f"cell-{i}-{j}",
      node_ids=(
        node_id(2 * i, 2 * j),
        node_id(2 * i + 1, 2 * j),
        node_id(2 * i + 2, 2 * j),
        node_id(2 * i + 2, 2 * j + 1),
        node_id(2 * i + 2, 2 * j + 2),
        node_id(2 * i + 1, 2 * j + 2),
        node_id(2 * i, 2 * j + 2),
        node_id(2 * i, 2 * j + 1),
      ),
      source=_source(f"cell-{i}-{j}"),
    )
    for j in range(2)
    for i in range(2)
  )
  block = CellBlockSpec(
    id="cells",
    reference_topology="quadrilateral",
    topological_dimension=2,
    embedding_dimension=2,
    geometry_interpolation="serendipity-quad8",
    cells=cells,
    source=_source("block"),
  )
  field = FieldSpec(
    id="displacement",
    components=("x", "y"),
    location="node",
    source=_source("field"),
  )
  material = MaterialSpec(
    id="elastic",
    model="plane-stress-linear-elastic",
    parameters=(
      MaterialParameterSpec("youngs_modulus", 1.0e6),
      MaterialParameterSpec("poisson_ratio", 0.25),
    ),
    source=_source("material"),
  )
  region = RegionSpec(
    id="domain",
    cell_refs=tuple(CellRef("cells", cell.id) for cell in cells),
    field_ids=("displacement",),
    material_id="elastic",
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


def test_refill_sums_duplicates_in_strict_sequential_stream_order() -> None:
  """Refill reduces duplicate runs strictly left-to-right in stable order.

  The 2x2 Q8 patch's interior node is shared by four cells, so its
  (row, col) duplicate runs have length 4 — the class where
  ``np.add.reduceat``'s per-segment SIMD inner loop reassociates at ulp
  level in a platform-dependent way. The canonical order is the
  strict-sequential one (``fem.assembly.dedup_coo_segments``), pinned here
  bitwise against an explicit loop and against the shared public kernel,
  so the driver and production assembly paths reduce identically on every
  platform.
  """
  system = compile_system(_q8_patch_model(), q8_reference_registry())
  coordinate_map = compile_constraint_map(system)
  plan = compile_driver_plan(system, coordinate_map)
  (slice_,) = plan.operator_slices
  rng = np.random.default_rng(20261009)
  batch = rng.normal(
    0.0,
    1.0e3,
    (slice_.entity_count, slice_.element_dof_count, slice_.element_dof_count),
  )
  permutation = plan.csr_sort_permutation.values
  segment_offsets = plan.csr_segment_offsets.values
  run_lengths = np.diff(np.concatenate([segment_offsets, [plan.coo_entry_count]]))
  assert int(run_lengths.max()) >= 4  # the reduceat-divergent class is exercised
  coo_values = batch.reshape(-1)
  reference = np.empty(plan.csr_indices.values.shape[0], dtype=np.float64)
  for segment in range(segment_offsets.shape[0]):
    start = int(segment_offsets[segment])
    stop = (
      int(segment_offsets[segment + 1])
      if segment + 1 < segment_offsets.shape[0]
      else plan.coo_entry_count
    )
    acc = 0.0
    for j in range(start, stop):
      acc += coo_values[permutation[j]]
    reference[segment] = acc
  refilled = refill_tangent(plan, (batch,))
  np.testing.assert_array_equal(
    refilled.data.view(np.uint64), reference.view(np.uint64)
  )
  np.testing.assert_array_equal(
    refilled.data.view(np.uint64),
    dedup_coo_segments(coo_values, permutation, segment_offsets).view(np.uint64),
  )


def test_refill_validates_batches() -> None:
  system, coordinate_map = _fixed_truss_map()
  plan = compile_driver_plan(system, coordinate_map)
  with pytest.raises(TypeError, match="exact DriverAssemblyPlan"):
    refill_tangent(object(), ())  # type: ignore[arg-type]
  with pytest.raises(TypeError, match="one batch per operator"):
    refill_tangent(plan, ())
  with pytest.raises(TypeError, match="float64"):
    refill_tangent(plan, (np.zeros((2, 4, 4), dtype=np.float32),))
  with pytest.raises(TypeError, match="float64"):
    refill_tangent(plan, (np.zeros((2, 3, 3)),))


def test_internal_force_scatter_matches_manual_loop() -> None:
  system, coordinate_map = _fixed_truss_map()
  plan = compile_driver_plan(system, coordinate_map, _apex_load())
  gather = plan.operator_slices[0].gather.values
  rng = np.random.default_rng(11)
  batch = rng.normal(0.0, 1.0, (2, 4))
  expected = np.zeros(6)
  for element in range(2):
    expected[gather[element]] += batch[element]
  assembled = assemble_internal_force(plan, (batch,))
  np.testing.assert_array_equal(assembled, expected)
  again = assemble_internal_force(plan, (batch,))
  np.testing.assert_array_equal(assembled, again)
  with pytest.raises(TypeError, match="one batch per operator"):
    assemble_internal_force(plan, ())
  with pytest.raises(TypeError, match="float64"):
    assemble_internal_force(plan, (np.zeros((2, 3)),))


def test_multi_operator_plan_assembles_all_slices() -> None:
  base = compile_system(_q8_model(), q8_reference_registry())
  block, spring = compile_spring_operator(
    base,
    damage_envelope_declaration(
      block_id="damage-springs",
      space_id="displacement",
      spring_ids=("spring-2", "spring-3"),
      node_ids=(2, 3),
      stiffness=10.0,
      critical_extension=10.0,
      max_increment=1.0,
      source=_source("springs"),
    ),
  )
  system = compose_system(base, block, spring)
  coordinate_map = compile_constraint_map(system)
  plan = compile_driver_plan(system, coordinate_map)
  assert len(plan.operator_slices) == 2
  continuum_slice, spring_slice = plan.operator_slices
  assert spring_slice.entity_count == 2
  assert spring_slice.element_dof_count == 2
  assert plan.constant_tangent is False
  rng = np.random.default_rng(13)
  continuum_batch = rng.normal(0.0, 1.0, (1, 16, 16))
  spring_batch = rng.normal(0.0, 1.0, (2, 2, 2))
  expected = np.zeros(plan.csr_shape)
  continuum_gather = continuum_slice.gather.values
  expected[np.ix_(continuum_gather[0], continuum_gather[0])] += continuum_batch[0]
  for element in range(2):
    dofs = spring_slice.gather.values[element]
    expected[np.ix_(dofs, dofs)] += spring_batch[element]
  np.testing.assert_array_equal(
    refill_tangent(plan, (continuum_batch, spring_batch)).toarray(),
    expected,
  )
  residual_continuum = rng.normal(0.0, 1.0, (1, 16))
  residual_spring = rng.normal(0.0, 1.0, (2, 2))
  expected_vector = np.zeros(plan.full_dof_count)
  expected_vector[continuum_gather[0]] += residual_continuum[0]
  for element in range(2):
    expected_vector[spring_slice.gather.values[element]] += residual_spring[element]
  np.testing.assert_array_equal(
    assemble_internal_force(plan, (residual_continuum, residual_spring)),
    expected_vector,
  )


def test_loads_evaluate_affinely_and_accumulate() -> None:
  system, coordinate_map = _fixed_truss_map()
  loads = (
    NodalLoadSpec(
      id="apex-y",
      target=DofRef(node_id=2, field_id="displacement", component="y"),
      value=AffineValueSpec(
        coefficients=(AffineCoefficientSpec("load", -100.0),),
      ),
    ),
    NodalLoadSpec(
      id="apex-y-constant",
      target=DofRef(node_id=2, field_id="displacement", component="y"),
      value=AffineValueSpec(constant=-5.0),
    ),
    NodalLoadSpec(
      id="apex-x",
      target=DofRef(node_id=2, field_id="displacement", component="x"),
      value=AffineValueSpec(
        coefficients=(AffineCoefficientSpec("load", 3.0),),
      ),
    ),
  )
  plan = compile_driver_plan(system, coordinate_map, loads)
  assert plan.loads.load_count == 3
  at_half = evaluate_loads(plan.loads, _point(0.5)).values
  np.testing.assert_allclose(
    at_half,
    np.array([0.0, 0.0, 0.0, 0.0, 1.5, -55.0]),
    rtol=0.0,
    atol=1.0e-15,
  )
  at_zero = evaluate_loads(plan.loads, _point(0.0)).values
  np.testing.assert_allclose(
    at_zero,
    np.array([0.0, 0.0, 0.0, 0.0, 0.0, -5.0]),
    rtol=0.0,
    atol=1.0e-15,
  )


def test_load_compilation_reports_coded_failures() -> None:
  system, coordinate_map = _fixed_truss_map()
  with pytest.raises(DriverPreparationError, match="unknown-load-node"):
    compile_driver_plan(
      system,
      coordinate_map,
      (
        NodalLoadSpec(
          id="ghost",
          target=DofRef(node_id=99, field_id="displacement", component="y"),
          value=AffineValueSpec(constant=1.0),
        ),
      ),
    )
  with pytest.raises(DriverPreparationError, match="unknown-load-component"):
    compile_driver_plan(
      system,
      coordinate_map,
      (
        NodalLoadSpec(
          id="ghost",
          target=DofRef(node_id=2, field_id="displacement", component="z"),
          value=AffineValueSpec(constant=1.0),
        ),
      ),
    )
  with pytest.raises(DriverPreparationError, match="unknown-load-space"):
    compile_driver_plan(
      system,
      coordinate_map,
      (
        NodalLoadSpec(
          id="ghost",
          target=DofRef(node_id=2, field_id="temperature", component="x"),
          value=AffineValueSpec(constant=1.0),
        ),
      ),
    )
  with pytest.raises(DriverPreparationError, match="unknown-load-coordinate"):
    compile_driver_plan(
      system,
      coordinate_map,
      (
        NodalLoadSpec(
          id="ghost",
          target=DofRef(node_id=2, field_id="displacement", component="y"),
          value=AffineValueSpec(
            coefficients=(AffineCoefficientSpec("time", 1.0),),
          ),
        ),
      ),
    )
  with pytest.raises(DriverPreparationError, match="non-finite-load-value"):
    compile_driver_plan(
      system,
      coordinate_map,
      (
        NodalLoadSpec(
          id="ghost",
          target=DofRef(node_id=2, field_id="displacement", component="y"),
          value=AffineValueSpec(constant=float("inf")),
        ),
      ),
    )


def test_load_evaluation_validates_binding() -> None:
  system, coordinate_map = _fixed_truss_map()
  plan = compile_driver_plan(system, coordinate_map, _apex_load())
  with pytest.raises(DriverEvaluationError, match="missing-program-coordinate"):
    evaluate_loads(plan.loads, ProgramPoint())
  with pytest.raises(DriverEvaluationError, match="unknown-program-coordinate"):
    evaluate_loads(
      plan.loads,
      ProgramPoint(
        (
          ProgramCoordinateValue("load", 1.0),
          ProgramCoordinateValue("time", 0.0),
        )
      ),
    )
  with pytest.raises(DriverEvaluationError, match="duplicate-program-coordinate"):
    evaluate_loads(
      plan.loads,
      ProgramPoint(
        (
          ProgramCoordinateValue("load", 0.5),
          ProgramCoordinateValue("load", 0.6),
        )
      ),
    )
  with pytest.raises(DriverEvaluationError, match="non-finite-coordinate-value"):
    evaluate_loads(plan.loads, _point(float("nan")))
  with pytest.raises(TypeError, match="exact ProgramPoint"):
    evaluate_loads(plan.loads, object())  # type: ignore[arg-type]
  with pytest.raises(TypeError, match="exact CompiledLoadProgram"):
    evaluate_loads(object(), _point(0.0))  # type: ignore[arg-type]
