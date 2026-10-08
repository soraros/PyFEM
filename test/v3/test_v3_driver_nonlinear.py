# SPDX-License-Identifier: MIT

"""NonlinearStatic driver protocol: Newton loop, cutback, typed statuses, state.

The rollback oracles assert the D3 Case A discipline directly: a rejected or
failed attempt leaves the M12 owner's committed state byte-identical, and
cutback retries restart from the same committed generation (ordinals advance
exactly once per committed substep; rejections never advance them).

The budget-exhaust classification (SLOW_CONVERGENCE near-miss versus
NON_CONVERGENT failing) is pinned against the M48 finding's J2 cantilever
deck and the M50 snap-through guard deck.
"""

from __future__ import annotations

import sys
from collections.abc import Callable

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from pyfem.v3 import authoring
from pyfem.v3.compile.continuum import plasticity_reference_registry
from pyfem.v3.compile.spring import (
  SpringDeclaration,
  SpringKernel,
  SpringKernelResult,
  SpringOperator,
  compile_spring_operator,
  compose_system,
  damage_envelope_declaration,
)
from pyfem.v3.compile.system import compile_system
from pyfem.v3.compile.truss import truss_reference_registry
from pyfem.v3.constraints import (
  CompiledConstraintMap,
  ConstraintEvaluationError,
  compile_constraint_map,
)
from pyfem.v3.driver import (
  BudgetExhaustionTrend,
  DriverStatus,
  NonlinearStaticDriver,
  NonlinearStaticResult,
  NonlinearStaticSettings,
  SubstepStatus,
)
from pyfem.v3.model.operator import EvaluationStatus
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
from pyfem.v3.state import StateTransactionOwner

# Documented oracle tolerances for this module's closed-form and self-reference
# checks: converged Newton states at tolerance 1e-10 agree far below 1e-7.
_STATE_RTOL = 1.0e-7
_STATE_ATOL = 1.0e-9
_REACTION_ATOL = 1.0e-6


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


def _fixed_truss_map(system: CompiledSystem) -> CompiledConstraintMap:
  return compile_constraint_map(
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


def _apex_loads(factor: float = -1.0) -> tuple[NodalLoadSpec, ...]:
  return (
    NodalLoadSpec(
      id="apex",
      target=DofRef(node_id=2, field_id="displacement", component="y"),
      value=AffineValueSpec(
        coefficients=(AffineCoefficientSpec("load", factor, _source("coef")),),
        source=_source("load"),
      ),
      source=_source("apex"),
    ),
  )


def _points(*values: float) -> tuple[ProgramPoint, ...]:
  return tuple(
    ProgramPoint((ProgramCoordinateValue("load", value),)) for value in values
  )


def _truss_driver(
  settings: NonlinearStaticSettings | None = None,
  *,
  apex_factor: float = -1.0,
  with_damage_spring: float | None = None,
  scripted_kernel: SpringKernel | None = None,
) -> NonlinearStaticDriver:
  system = compile_system(_truss_model(), truss_reference_registry())
  if with_damage_spring is not None:
    block, spring = compile_spring_operator(
      system,
      damage_envelope_declaration(
        block_id="damage-springs",
        space_id="displacement",
        spring_ids=("apex-spring",),
        node_ids=(2,),
        stiffness=1.0e3,
        critical_extension=10.0,
        max_increment=with_damage_spring,
        source=_source("damage"),
      ),
    )
    system = compose_system(system, block, spring)
  if scripted_kernel is not None:
    block, spring = compile_spring_operator(
      system,
      SpringDeclaration(
        block_id="scripted-springs",
        space_id="displacement",
        spring_ids=("apex-scripted",),
        node_ids=(2,),
        state_schema="scripted-linear-spring-v1",
        state_slots=(),
        kernel_name="scripted-linear-spring",
        kernel_version="1",
        implementation_id="scripted-linear-spring-v1",
        parameters=(1.0e3,),
        kernel=scripted_kernel,
        source=_source("scripted"),
      ),
    )
    system = compose_system(system, block, spring)
  coordinate_map = _fixed_truss_map(system)
  loads = _apex_loads(factor=apex_factor)
  if settings is None:
    return NonlinearStaticDriver(system, coordinate_map, loads)
  return NonlinearStaticDriver(system, coordinate_map, loads, settings)


def _run_ramp(
  driver: NonlinearStaticDriver,
  *factors: float,
) -> NonlinearStaticResult:
  return driver.run(
    base_point=ProgramPoint((ProgramCoordinateValue("load", 0.0),)),
    target_points=_points(*factors),
  )


def _owner_snapshot(owner: StateTransactionOwner) -> tuple[object, ...]:
  return (
    owner.accepted_physical().values.tobytes(),
    owner.generation.ordinal,
    owner.history,
    tuple((block, owner.encode_state(block)) for block in owner.block_ids),
  )


def test_settings_validation() -> None:
  with pytest.raises(ValueError, match="tolerance"):
    NonlinearStaticSettings(tolerance=0.0)
  with pytest.raises(ValueError, match="tolerance"):
    NonlinearStaticSettings(tolerance=float("nan"))
  with pytest.raises(ValueError, match="max_iterations"):
    NonlinearStaticSettings(max_iterations=0)
  with pytest.raises(ValueError, match="max_cutbacks"):
    NonlinearStaticSettings(max_cutbacks=-1)
  with pytest.raises(ValueError, match="cutback_factor"):
    NonlinearStaticSettings(cutback_factor=1.0)
  with pytest.raises(ValueError, match="growth_factor"):
    NonlinearStaticSettings(growth_factor=0.5)
  with pytest.raises(ValueError, match="divergence_ratio"):
    NonlinearStaticSettings(divergence_ratio=1.0)
  with pytest.raises(ValueError, match="min_substep_size"):
    NonlinearStaticSettings(min_substep_size=0.0)
  with pytest.raises(ValueError, match="min_substep_size"):
    NonlinearStaticSettings(min_substep_size=1.0)


def test_driver_requires_exact_types() -> None:
  driver = _truss_driver()
  with pytest.raises(TypeError, match="exact CompiledSystem"):
    NonlinearStaticDriver(object(), driver.plan)  # type: ignore[arg-type]
  with pytest.raises(TypeError, match="exact CompiledConstraintMap"):
    NonlinearStaticDriver(
      compile_system(_truss_model(), truss_reference_registry()),
      object(),  # type: ignore[arg-type]
    )
  with pytest.raises(TypeError, match="exact NonlinearStaticSettings"):
    NonlinearStaticDriver(
      compile_system(_truss_model(), truss_reference_registry()),
      _fixed_truss_map(compile_system(_truss_model(), truss_reference_registry())),
      (),
      object(),  # type: ignore[arg-type]
    )


def test_run_validates_points() -> None:
  driver = _truss_driver()
  base = ProgramPoint((ProgramCoordinateValue("load", 0.0),))
  with pytest.raises(TypeError, match="exact tuple of ProgramPoint"):
    driver.run(base_point=base, target_points=[_points(1.0)[0]])  # type: ignore[list-item]
  with pytest.raises(TypeError, match="exact ProgramPoint"):
    driver.run(base_point=object(), target_points=_points(1.0))  # type: ignore[arg-type]
  with pytest.raises(ConstraintEvaluationError, match="missing-program-coordinate"):
    driver.run(base_point=ProgramPoint(), target_points=_points(1.0))
  with pytest.raises(ConstraintEvaluationError, match="missing-program-coordinate"):
    driver.run(base_point=base, target_points=(ProgramPoint(),))


def test_shallow_truss_ramp_commits_with_full_residual_observations() -> None:
  driver = _truss_driver()
  result = _run_ramp(driver, 25.0, 50.0, 75.0, 100.0)
  assert result.status is DriverStatus.COMPLETED
  assert result.failed_target_index is None
  assert [record.committed_ordinal for record in result.records] == [1, 2, 3, 4]
  assert result.final_generation.ordinal == 4
  assert result.initial_generation.ordinal == 0
  statistics = result.statistics
  assert statistics.committed_substep_count == 4
  assert statistics.rejected_substep_count == 0
  assert statistics.cutback_count == 0
  # The nonlinear truss factorizes per iteration and never reuses.
  assert statistics.factorization_reuse_count == 0
  assert statistics.factorization_count == statistics.tangent_refill_count
  assert statistics.factorization_count == statistics.linear_solve_count
  assert statistics.factorization_count > 4
  final = result.records[-1]
  assert final.observation is not None
  observation = final.observation
  # Reactions observed from the full residual balance the applied apex load.
  reactions = observation.reactions.values
  np.testing.assert_allclose(reactions[[1, 3]], [50.0, 50.0], rtol=0.0, atol=1.0e-6)
  np.testing.assert_allclose(reactions[[4, 5]], [0.0, 0.0], rtol=0.0, atol=0.0)
  # Fixed supports do no constraint work; the reduced residual is converged.
  assert observation.constraint_work == 0.0
  assert observation.reduced_residual_norm <= 1.0e-10 * 100.0
  assert observation.full_residual_norm == pytest.approx(
    float(np.linalg.norm(reactions))
  )
  # Every committed substep needed genuine Newton iteration (nonlinear path).
  assert all(len(record.iterations) >= 3 for record in result.records)
  apex = driver.owner.accepted_physical().values[4:]
  np.testing.assert_allclose(apex, [0.0, -0.0464125], rtol=1.0e-7, atol=1.0e-9)


def test_hyperelastic_final_state_is_path_independent() -> None:
  single = _truss_driver()
  single_result = _run_ramp(single, 100.0)
  stepped = _truss_driver()
  stepped_result = _run_ramp(stepped, 25.0, 50.0, 75.0, 100.0)
  assert single_result.status is DriverStatus.COMPLETED
  assert stepped_result.status is DriverStatus.COMPLETED
  np.testing.assert_allclose(
    single.owner.accepted_physical().values,
    stepped.owner.accepted_physical().values,
    rtol=_STATE_RTOL,
    atol=_STATE_ATOL,
  )


def test_linear_system_reuses_one_factorization_across_the_ramp() -> None:
  nodes = tuple(
    NodeSpec(id=index + 1, coordinates=point, source=_source(f"n{index + 1}"))
    for index, point in enumerate(
      (
        (0.0, 0.0),
        (0.5, 0.0),
        (1.0, 0.0),
        (1.0, 0.5),
        (1.0, 1.0),
        (0.5, 1.0),
        (0.0, 1.0),
        (0.0, 0.5),
      )
    )
  )
  cell = CellSpec(
    id="cell-1",
    node_ids=tuple(node.id for node in nodes),
    source=_source("cell"),
  )
  model = ModelSpec(
    mesh=MeshSpec(
      nodes=nodes,
      cell_blocks=(
        CellBlockSpec(
          id="cells",
          reference_topology="quadrilateral",
          topological_dimension=2,
          embedding_dimension=2,
          geometry_interpolation="serendipity-quad8",
          cells=(cell,),
          source=_source("block"),
        ),
      ),
      source=_source("mesh"),
    ),
    fields=(
      FieldSpec(
        id="displacement",
        components=("x", "y"),
        location="node",
        source=_source("field"),
      ),
    ),
    materials=(
      MaterialSpec(
        id="elastic",
        model="plane-stress-linear-elastic",
        parameters=(
          MaterialParameterSpec("youngs_modulus", 1.0e6),
          MaterialParameterSpec("poisson_ratio", 0.25),
        ),
        source=_source("material"),
      ),
    ),
    regions=(
      RegionSpec(
        id="domain",
        cell_refs=(CellRef("cells", "cell-1"),),
        field_ids=("displacement",),
        material_id="elastic",
        formulation="small-strain-continuum",
        quadrature="gauss-3x3",
        source=_source("region"),
      ),
    ),
    source=_source("model"),
  )
  from pyfem.v3.compile.continuum import q8_reference_registry

  system = compile_system(model, q8_reference_registry())
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
      for node in (1, 7, 8)
      for component in ("x", "y")
    ),
    coordinates=(ProgramCoordinateSpec(name="load", kind="load"),),
  )
  loads = tuple(
    NodalLoadSpec(
      id=f"load-{node}",
      target=DofRef(node_id=node, field_id="displacement", component="x"),
      value=AffineValueSpec(
        coefficients=(AffineCoefficientSpec("load", 1000.0),),
      ),
    )
    for node in (3, 4, 5)
  )
  driver = NonlinearStaticDriver(system, coordinate_map, loads)
  result = _run_ramp(driver, 0.25, 0.5, 0.75, 1.0)
  assert result.status is DriverStatus.COMPLETED
  statistics = result.statistics
  assert statistics.committed_substep_count == 4
  assert statistics.factorization_count == 1
  assert statistics.tangent_refill_count == 1
  assert statistics.factorization_reuse_count == 3
  assert statistics.linear_solve_count == 4
  # A second run over the same driver still reuses the same factorization.
  second = driver.run(
    base_point=_points(1.0)[0],
    target_points=_points(1.25),
  )
  assert second.status is DriverStatus.COMPLETED
  assert driver.statistics.factorization_count == 1
  assert driver.statistics.factorization_reuse_count == 4


def test_diverging_step_fails_typed_and_leaves_state_byte_identical() -> None:
  driver = _truss_driver(
    NonlinearStaticSettings(tolerance=1.0e-10, max_cutbacks=3),
  )
  first = _run_ramp(driver, 100.0)
  assert first.status is DriverStatus.COMPLETED
  snapshot = _owner_snapshot(driver.owner)
  failed = driver.run(
    base_point=_points(100.0)[0],
    target_points=_points(1.0e8),
  )
  assert failed.status is DriverStatus.STEP_FAILED
  assert failed.failed_target_index == 0
  assert failed.records[-1].status is SubstepStatus.FAILED
  assert failed.records[-1].committed_ordinal is None
  assert failed.final_generation.ordinal == snapshot[1]
  assert _owner_snapshot(driver.owner) == snapshot


def test_beyond_limit_load_fails_typed_at_the_substep_floor() -> None:
  # The reviewer-c5 finding's deck: with default deck settings the ramp
  # crosses the post-snap limit load (near load coordinate 2.399), and the
  # unguarded schedule spun forever — micro-commits asymptoting to the limit
  # each reset the consecutive cutback budget, so neither the iteration
  # budget nor max_cutbacks ever tripped. The substep floor types the
  # failure in bounded work instead.
  settings = NonlinearStaticSettings(tolerance=1.0e-3, max_iterations=10)
  driver = _truss_driver(settings, apex_factor=-100.0)
  result = _run_ramp(driver, 1.0, 2.0, 3.0)
  assert result.status is DriverStatus.STEP_FAILED
  assert result.failed_target_index == 2
  # Bounded work: the unguarded protocol burned tens of thousands of
  # evaluations per minute without terminating.
  assert result.statistics.evaluation_count < 2_000
  assert (
    len(result.records)
    == result.statistics.committed_substep_count
    + result.statistics.rejected_substep_count
  )
  assert result.statistics.rejected_substep_count == result.statistics.cutback_count + 1
  failed = result.records[-1]
  assert failed.status is SubstepStatus.FAILED
  assert failed.committed_ordinal is None
  assert len(failed.iterations) == settings.max_iterations
  committed = [
    record for record in result.records if record.status is SubstepStatus.COMMITTED
  ]
  assert [record.committed_ordinal for record in committed] == list(
    range(1, len(committed) + 1)
  )
  # The trajectory shows WHY: many micro-commits asymptote the third target
  # short of full progress, and the terminal attempt — rejected well inside
  # the consecutive cutback budget — is at or below the substep floor.
  assert len(committed) > 4
  last = committed[-1]
  assert last.target_index == 2
  assert last.progress < 1.0
  assert failed.cutback_level <= settings.max_cutbacks
  assert failed.progress - last.progress <= settings.min_substep_size
  # The terminal rejection never touched committed state: the final
  # generation is exactly the last micro-commit's.
  assert result.final_generation.ordinal == last.committed_ordinal


def test_singular_reduced_tangent_cuts_back_and_fails_typed() -> None:
  nodes = (
    NodeSpec(id=0, coordinates=(0.0, 0.0), source=_source("n0")),
    NodeSpec(id=1, coordinates=(2.0, 0.0), source=_source("n1")),
    NodeSpec(id=2, coordinates=(4.0, 0.0), source=_source("n2")),
  )
  model = ModelSpec(
    mesh=MeshSpec(
      nodes=nodes,
      cell_blocks=(
        CellBlockSpec(
          id="bars",
          reference_topology="line",
          topological_dimension=1,
          embedding_dimension=2,
          geometry_interpolation="line2",
          cells=(CellSpec(id="bar", node_ids=(0, 1), source=_source("c0")),),
          source=_source("block"),
        ),
      ),
      source=_source("mesh"),
    ),
    fields=(
      FieldSpec(
        id="displacement",
        components=("x", "y"),
        location="node",
        source=_source("field"),
      ),
    ),
    materials=(
      MaterialSpec(
        id="steel",
        model="uniaxial-linear-elastic",
        parameters=(
          MaterialParameterSpec("youngs_modulus", 5.0e6),
          MaterialParameterSpec("area", 1.0),
        ),
        source=_source("material"),
      ),
    ),
    regions=(
      RegionSpec(
        id="domain",
        cell_refs=(CellRef("bars", "bar"),),
        field_ids=("displacement",),
        material_id="steel",
        formulation="total-lagrangian-truss",
        quadrature="none",
        source=_source("region"),
      ),
    ),
    source=_source("model"),
  )
  system = compile_system(model, truss_reference_registry())
  # The isolated node 2 owns free DOFs with zero tangent rows: K_q is exactly
  # singular, so every factorization attempt fails and cutback exhausts.
  coordinate_map = compile_constraint_map(
    system,
    constraints=tuple(
      PrescribedDofSpec(
        id=f"fix-0-{component}",
        target=DofRef(node_id=0, field_id="displacement", component=component),
        value=AffineValueSpec(constant=0.0),
        source=_source(f"fix-0-{component}"),
      )
      for component in ("x", "y")
    ),
    coordinates=(ProgramCoordinateSpec(name="load", kind="load"),),
  )
  loads = (
    NodalLoadSpec(
      id="pull",
      target=DofRef(node_id=1, field_id="displacement", component="x"),
      value=AffineValueSpec(
        coefficients=(AffineCoefficientSpec("load", 100.0),),
      ),
    ),
  )
  driver = NonlinearStaticDriver(
    system,
    coordinate_map,
    loads,
    NonlinearStaticSettings(max_cutbacks=2),
  )
  snapshot = _owner_snapshot(driver.owner)
  result = _run_ramp(driver, 1.0)
  assert result.status is DriverStatus.STEP_FAILED
  assert result.statistics.rejected_substep_count == 3
  assert _owner_snapshot(driver.owner) == snapshot


def test_damage_spring_reject_step_drives_cutback_from_same_generation() -> None:
  limited = _truss_driver(with_damage_spring=0.005)
  result = _run_ramp(limited, 25.0, 50.0, 75.0, 100.0)
  assert result.status is DriverStatus.COMPLETED
  statistics = result.statistics
  assert statistics.cutback_count >= 2
  assert statistics.rejected_substep_count == statistics.cutback_count
  committed = [
    record for record in result.records if record.status is SubstepStatus.COMMITTED
  ]
  # Cutback retries never consume generations: ordinals advance once per
  # committed substep regardless of how many attempts were rejected.
  assert [record.committed_ordinal for record in committed] == list(
    range(1, len(committed) + 1)
  )
  assert result.final_generation.ordinal == len(committed)
  # Same physics without the increment cap agrees at converged accuracy.
  free = _truss_driver(with_damage_spring=10.0)
  free_result = _run_ramp(free, 25.0, 50.0, 75.0, 100.0)
  assert free_result.status is DriverStatus.COMPLETED
  assert free_result.statistics.cutback_count == 0
  np.testing.assert_allclose(
    limited.owner.accepted_physical().values,
    free.owner.accepted_physical().values,
    rtol=_STATE_RTOL,
    atol=_STATE_ATOL,
  )


def test_stateful_damage_rows_commit_and_survive_forced_failure() -> None:
  driver = _truss_driver(
    NonlinearStaticSettings(tolerance=1.0e-10, max_cutbacks=3),
    with_damage_spring=0.5,
  )
  result = _run_ramp(driver, 100.0)
  assert result.status is DriverStatus.COMPLETED
  (damage_block,) = [
    block for block in driver.owner.block_ids if block[0] == "damage-springs"
  ]
  rows = driver.owner.accepted_state(damage_block).values
  assert rows.shape == (1, 1)
  apex_drop = float(abs(driver.owner.accepted_physical().values[5]))
  np.testing.assert_allclose(rows[:, 0], [apex_drop], rtol=1.0e-12, atol=1.0e-14)
  snapshot = _owner_snapshot(driver.owner)
  failed = driver.run(
    base_point=_points(100.0)[0],
    target_points=_points(1.0e8),
  )
  assert failed.status is DriverStatus.STEP_FAILED
  assert _owner_snapshot(driver.owner) == snapshot


def _scripted_linear_spring(
  reject_at_runtime_calls: dict[int, EvaluationStatus],
) -> tuple[SpringKernel, list[int], Callable[[], int]]:
  """Linear spring scripting typed rejections by runtime kernel call index.

  The compile boundary probes the kernel (the virgin probe plus the seeded
  nonzero-state tangent probe), so the reject map keys only runtime calls:
  ``mark_runtime`` records the compile-time call count once driver
  construction — and with it compilation — has finished.
  """
  calls: list[int] = []
  runtime_start: list[int] = []

  def kernel(
    displacements: np.ndarray,
    accepted_rows: np.ndarray,
    parameters: np.ndarray,
  ) -> SpringKernelResult:
    calls.append(len(calls))
    stiffness = float(parameters[0])
    status = EvaluationStatus.OK
    if runtime_start:
      status = reject_at_runtime_calls.get(
        len(calls) - 1 - runtime_start[0],
        EvaluationStatus.OK,
      )
    if status is not EvaluationStatus.OK:
      return SpringKernelResult(
        force=np.zeros_like(displacements),
        tangent=np.zeros((len(displacements), 2, 2)),
        trial_rows=np.array(accepted_rows, copy=True),
        status=status,
      )
    entity_count = len(displacements)
    return SpringKernelResult(
      force=stiffness * displacements,
      tangent=np.broadcast_to(stiffness * np.eye(2), (entity_count, 2, 2)).copy(),
      trial_rows=np.array(accepted_rows, copy=True),
      status=EvaluationStatus.OK,
    )

  def mark_runtime() -> int:
    runtime_start.append(len(calls))
    return len(calls)

  return kernel, calls, mark_runtime


def test_operator_reject_iteration_retries_with_damped_iterate() -> None:
  # The scripted map keys runtime kernel calls; runtime call 1 is the second
  # driver evaluation (the first Newton correction of the first substep).
  kernel, calls, mark_runtime = _scripted_linear_spring(
    {1: EvaluationStatus.REJECT_ITERATION}
  )
  driver = _truss_driver(scripted_kernel=kernel)
  compile_calls = mark_runtime()
  result = _run_ramp(driver, 100.0)
  assert result.status is DriverStatus.COMPLETED
  first = result.records[0]
  statuses = [iteration.status for iteration in first.iterations]
  assert statuses[:2] == [EvaluationStatus.OK, EvaluationStatus.REJECT_ITERATION]
  assert first.iterations[1].residual_norm is None
  reference = _truss_driver(scripted_kernel=_scripted_linear_spring({})[0])
  reference_result = _run_ramp(reference, 100.0)
  assert reference_result.status is DriverStatus.COMPLETED
  np.testing.assert_allclose(
    driver.owner.accepted_physical().values,
    reference.owner.accepted_physical().values,
    rtol=_STATE_RTOL,
    atol=_STATE_ATOL,
  )
  # mark_runtime accounts for the compile-time probes; one kernel call per
  # driver evaluation afterwards.
  assert len(calls) == driver.statistics.evaluation_count + compile_calls


def test_operator_reject_step_cuts_back_and_recovers() -> None:
  kernel, _calls, mark_runtime = _scripted_linear_spring(
    {1: EvaluationStatus.REJECT_STEP}
  )
  driver = _truss_driver(scripted_kernel=kernel)
  mark_runtime()
  result = _run_ramp(driver, 100.0)
  assert result.status is DriverStatus.COMPLETED
  assert result.statistics.cutback_count == 1
  rejected_statuses = [iteration.status for iteration in result.records[0].iterations]
  assert EvaluationStatus.REJECT_STEP in rejected_statuses
  committed = [
    record for record in result.records if record.status is SubstepStatus.COMMITTED
  ]
  assert [record.committed_ordinal for record in committed] == [1, 2]


def test_first_evaluation_reject_iteration_escalates_to_cutback() -> None:
  # Rejecting the very first evaluation leaves no Newton direction to damp,
  # so the protocol cuts back instead of retrying in place.
  kernel, _calls, mark_runtime = _scripted_linear_spring(
    {0: EvaluationStatus.REJECT_ITERATION}
  )
  driver = _truss_driver(scripted_kernel=kernel)
  mark_runtime()
  result = _run_ramp(driver, 100.0)
  assert result.status is DriverStatus.COMPLETED
  assert result.statistics.cutback_count == 1


def test_operator_reject_step_budget_exhaustion_fails_typed() -> None:
  kernel, _calls, mark_runtime = _scripted_linear_spring(
    {index: EvaluationStatus.REJECT_STEP for index in range(11)}
  )
  driver = _truss_driver(
    NonlinearStaticSettings(max_cutbacks=2),
    scripted_kernel=kernel,
  )
  mark_runtime()
  snapshot = _owner_snapshot(driver.owner)
  result = _run_ramp(driver, 100.0)
  assert result.status is DriverStatus.STEP_FAILED
  assert result.failed_target_index == 0
  assert _owner_snapshot(driver.owner) == snapshot
  operator = driver.owner.system.operators[-1]
  assert isinstance(operator, SpringOperator)


def _j2_cantilever_driver(settings: NonlinearStaticSettings) -> NonlinearStaticDriver:
  """The M48 finding's deck: a load-controlled J2 cantilever on a Q8 patch."""
  mesh = authoring.quad8_patch(4, 1, width=4.0, height=1.0)
  material = authoring.plasticity(210000.0, 0.3, 250.0, 1000.0, id="steel")
  model = authoring.small_strain_continuum(mesh, material=material)
  system = compile_system(model, plasticity_reference_registry())
  left = tuple(node.id for node in mesh.nodes if node.coordinates[0] == 0.0)
  right = tuple(
    sorted(
      (node for node in mesh.nodes if node.coordinates[0] == 4.0),
      key=lambda node: node.coordinates[1],
    )
  )
  coordinate_map = compile_constraint_map(
    system,
    constraints=authoring.fixed(left, ("x", "y")),
    coordinates=(ProgramCoordinateSpec(name="load", kind="load"),),
  )
  loads = tuple(
    authoring.nodal_load(node.id, "y", fraction)
    for node, fraction in zip(right, (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0), strict=True)
  )
  return NonlinearStaticDriver(system, coordinate_map, loads, settings)


def test_budget_exhaustion_surfaces_slow_convergence_on_the_j2_near_miss() -> None:
  """The M48 finding: a slow-but-working Newton reads 'budget too small'.

  Re-based vehicle (M54): this is the finding's exact deck, but its
  33-iteration signature was the pre-fix J2 tangent's — the shear-tangent
  fix restores quadratic convergence, so the load-controlled cantilever
  now completes in 7 iterations at tolerance 1e-12 and the default budget
  no longer thrashes (the 294 records / 7270 evaluations were the tangent
  bug's amplification). Per M52's handoff note the near-miss vehicle
  re-bases the starved budget: at max_iterations=4 every full-size plastic
  attempt still needs 7 and exhausts its budget while contracting —
  SLOW_CONVERGENCE typed on every budget-exhausted record — while the
  elastic micro-substeps commit, until two consecutive exhausts fail the
  target typed in bounded work. The documented remedy completes the same
  solve in one substep.
  """
  settings = NonlinearStaticSettings(
    tolerance=1.0e-12, max_iterations=4, max_cutbacks=2
  )
  driver = _j2_cantilever_driver(settings)
  result = _run_ramp(driver, 20.0)
  assert result.status is DriverStatus.STEP_FAILED
  assert result.statistics.evaluation_count < 1_000  # measured: 40
  exhausted = [
    record for record in result.records if record.status is not SubstepStatus.COMMITTED
  ]
  assert len(exhausted) >= 3  # measured: 8 rejected/failed records
  for record in exhausted:
    observation = record.budget_exhaustion
    assert observation is not None
    assert observation.trend is BudgetExhaustionTrend.SLOW_CONVERGENCE
    assert observation.measured_step_count == settings.max_iterations
    # Chatter permits rare single-step upticks (2/4 decreasing measured at
    # worst); the worst net contraction over the budget is 0.13, pinned
    # with ~2x margin against the 0.5 classification bound.
    assert observation.decreasing_step_count >= observation.measured_step_count - 2
    assert observation.final_residual_norm <= 0.3 * observation.first_residual_norm
  committed = [
    record for record in result.records if record.status is SubstepStatus.COMMITTED
  ]
  assert all(record.budget_exhaustion is None for record in committed)
  # The typed failure leaves committed state at the last micro-commit.
  assert result.final_generation.ordinal == committed[-1].committed_ordinal

  # The documented remedy: the same deck at max_iterations=50 completes in
  # one substep — the budget, not the convergence, was the constraint.
  remedy = _j2_cantilever_driver(
    NonlinearStaticSettings(tolerance=1.0e-12, max_iterations=50)
  )
  remedy_result = _run_ramp(remedy, 20.0)
  assert remedy_result.status is DriverStatus.COMPLETED
  (record,) = remedy_result.records
  assert record.status is SubstepStatus.COMMITTED
  assert record.budget_exhaustion is None
  # 7 iterations measured on the reference platform — the quadratic count
  # the J2 shear fix restored (pre-fix: 33, linear); the bounds only need
  # to prove the re-based 4-iteration budget was binding.
  assert settings.max_iterations < len(record.iterations) <= 50
  assert remedy_result.statistics.evaluation_count == len(record.iterations)
  observation = record.observation
  assert observation is not None
  # Reactions observed from the full residual balance the applied traction.
  np.testing.assert_allclose(
    observation.reactions.values[1::2].sum(),
    -20.0,
    rtol=0.0,
    atol=_REACTION_ATOL,
  )

  # The finding's original settings: the default budget completes the deck
  # in the same quadratic substep — the cutback amplification the finding
  # measured left with the tangent bug.
  default = _j2_cantilever_driver(NonlinearStaticSettings(tolerance=1.0e-12))
  default_result = _run_ramp(default, 20.0)
  assert default_result.status is DriverStatus.COMPLETED
  (default_record,) = default_result.records
  assert default_record.status is SubstepStatus.COMMITTED
  assert default_record.budget_exhaustion is None
  assert len(default_record.iterations) <= 10  # 7 measured; pre-fix: 33


def test_snap_through_failures_type_non_convergent_at_budget_exhaustion() -> None:
  """The M50 guard deck: failing trajectories never type as near-misses.

  Past the limit load every budget-exhausted attempt oscillates instead of
  contracting, so the classification is NON_CONVERGENT on every rejected
  and failed record (59 measured), and the substep floor still trips the
  typed failure in bounded work — decisions identical to before the
  classification existed.
  """
  settings = NonlinearStaticSettings(tolerance=1.0e-3, max_iterations=10)
  driver = _truss_driver(settings, apex_factor=-100.0)
  result = _run_ramp(driver, 1.0, 2.0, 3.0)
  assert result.status is DriverStatus.STEP_FAILED
  assert result.failed_target_index == 2
  assert result.statistics.evaluation_count < 2_000  # measured: 640
  failing = [
    record for record in result.records if record.status is not SubstepStatus.COMMITTED
  ]
  assert len(failing) > 20  # measured: 59
  for record in failing:
    observation = record.budget_exhaustion
    assert observation is not None
    assert observation.trend is BudgetExhaustionTrend.NON_CONVERGENT
  failed = result.records[-1]
  assert failed.status is SubstepStatus.FAILED
  assert failed.budget_exhaustion is not None
  # The terminal attempt's residual GREW over its exhausted budget — the
  # exact opposite of a near-miss.
  assert (
    failed.budget_exhaustion.final_residual_norm
    > failed.budget_exhaustion.first_residual_norm
  )
  committed = [
    record for record in result.records if record.status is SubstepStatus.COMMITTED
  ]
  assert result.final_generation.ordinal == committed[-1].committed_ordinal


def test_divergence_trip_rejection_carries_no_budget_exhaustion() -> None:
  """Rejections with another typed cause never carry the observation."""
  driver = _truss_driver(
    NonlinearStaticSettings(tolerance=1.0e-10, max_cutbacks=3),
  )
  first = _run_ramp(driver, 100.0)
  assert first.status is DriverStatus.COMPLETED
  failed = driver.run(
    base_point=_points(100.0)[0],
    target_points=_points(1.0e8),
  )
  assert failed.status is DriverStatus.STEP_FAILED
  assert all(record.budget_exhaustion is None for record in failed.records)
