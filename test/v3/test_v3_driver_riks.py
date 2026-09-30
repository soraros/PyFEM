# SPDX-License-Identifier: MIT

"""Riks arc-length driver protocol: continuation baseline, cutback, counters.

The oracles pin the two-scope continuation discipline directly: the committed
baseline (lam, da_prev, dlam_prev, factor, total_factor, cycle) advances only
on commit paths, so a rejected or failed attempt leaves the M12 owner's
committed state AND the baseline byte-identical, and cutback retries restart
from the same committed generation with a shrunk predictor factor. Exact
float relations (predictor anchoring, delta_lam trails, factor formula,
damping halves) are asserted bitwise where the ported legacy arithmetic is a
single deterministic float expression; converged-state comparisons carry
tolerances. The scripted-spring tests key runtime kernel calls the same way
as the nonlinear-driver suite: the compile boundary probes the kernel, and
``mark_runtime`` anchors the runtime call count after driver construction.
"""

from __future__ import annotations

import dataclasses
import sys
from collections.abc import Callable

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from pyfem.v3.compile.spring import (
  SpringDeclaration,
  SpringKernelResult,
  compile_spring_operator,
  compose_system,
)
from pyfem.v3.compile.system import compile_system
from pyfem.v3.compile.truss import truss_reference_registry
from pyfem.v3.constraints import (
  CompiledConstraintMap,
  ConstraintEvaluationError,
  compile_constraint_map,
)
from pyfem.v3.driver import (
  ArcLengthContinuationState,
  ArcLengthResult,
  ArcLengthSettings,
  ArcLengthTermination,
  DriverStatus,
  RiksDriver,
  SubstepStatus,
)
from pyfem.v3.driver.diagnostics import DriverPreparationError
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
  AffineTieSpec,
  AffineValueSpec,
  DofRef,
  NodalLoadSpec,
  PrescribedDofSpec,
  ProgramCoordinateSpec,
  ProgramCoordinateValue,
  ProgramPoint,
)
from pyfem.v3.state import StateTransactionOwner

_BASE_POINT = ProgramPoint()


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


def _fixed_truss_map(
  system: CompiledSystem,
  *,
  extra_coordinates: tuple[ProgramCoordinateSpec, ...] = (),
) -> CompiledConstraintMap:
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
    coordinates=(
      ProgramCoordinateSpec(name="load", kind="load"),
      *extra_coordinates,
    ),
  )


def _apex_loads() -> tuple[NodalLoadSpec, ...]:
  return (
    NodalLoadSpec(
      id="apex",
      target=DofRef(node_id=2, field_id="displacement", component="y"),
      value=AffineValueSpec(
        coefficients=(AffineCoefficientSpec("load", -100.0, _source("coef")),),
        source=_source("load"),
      ),
      source=_source("apex"),
    ),
  )


def _truss_driver(settings: ArcLengthSettings | None = None) -> RiksDriver:
  system = compile_system(_truss_model(), truss_reference_registry())
  coordinate_map = _fixed_truss_map(system)
  if settings is None:
    return RiksDriver(system, coordinate_map, _apex_loads())
  return RiksDriver(system, coordinate_map, _apex_loads(), settings)


def _scripted_truss_driver(
  reject_at_runtime_calls: dict[int, EvaluationStatus],
  settings: ArcLengthSettings,
) -> tuple[RiksDriver, list[int], Callable[[], int]]:
  """Truss plus one scripted linear spring scripting typed rejections.

  The reject map keys runtime kernel calls (0-based, after the compile-time
  probes); each driver evaluation consumes exactly one runtime call.
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
      status=status,
    )

  def mark_runtime() -> int:
    runtime_start.append(len(calls))
    return len(calls)

  system = compile_system(_truss_model(), truss_reference_registry())
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
      kernel=kernel,
      source=_source("scripted"),
    ),
  )
  system = compose_system(system, block, spring)
  coordinate_map = _fixed_truss_map(system)
  driver = RiksDriver(system, coordinate_map, _apex_loads(), settings)
  return driver, calls, mark_runtime


def _owner_snapshot(owner: StateTransactionOwner) -> tuple[object, ...]:
  return (
    owner.accepted_physical().values.tobytes(),
    owner.generation.ordinal,
    owner.history,
    tuple((block, owner.encode_state(block)) for block in owner.block_ids),
  )


def _continuation_snapshot(state: ArcLengthContinuationState) -> tuple[object, ...]:
  return (
    state.lam,
    state.da_prev.values.tobytes(),
    state.dlam_prev,
    state.factor,
    state.total_factor,
    state.cycle,
  )


def _committed(result: ArcLengthResult) -> list:
  return [
    record for record in result.records if record.status is SubstepStatus.COMMITTED
  ]


def _settings(**overrides: object) -> ArcLengthSettings:
  values: dict[str, object] = {
    "tolerance": 1.0e-10,
    "max_iterations": 25,
    "fixed_step": True,
    "max_lam": 10.0,
  }
  values.update(overrides)
  return ArcLengthSettings(**values)  # type: ignore[arg-type]


def test_settings_validation() -> None:
  with pytest.raises(ValueError, match="tolerance"):
    ArcLengthSettings(tolerance=0.0)
  with pytest.raises(ValueError, match="tolerance"):
    ArcLengthSettings(tolerance=float("nan"))
  with pytest.raises(ValueError, match="max_iterations"):
    ArcLengthSettings(max_iterations=0)
  with pytest.raises(ValueError, match="optimal_iterations"):
    ArcLengthSettings(optimal_iterations=0)
  with pytest.raises(ValueError, match="fixed_step"):
    ArcLengthSettings(fixed_step=1)  # type: ignore[arg-type]
  with pytest.raises(ValueError, match="max_lam"):
    ArcLengthSettings(max_lam=0.0)
  with pytest.raises(ValueError, match="max_factor"):
    ArcLengthSettings(max_factor=float("inf"))
  with pytest.raises(ValueError, match="cycle_cap"):
    ArcLengthSettings(cycle_cap=0)
  with pytest.raises(ValueError, match="max_cutbacks"):
    ArcLengthSettings(max_cutbacks=-1)
  with pytest.raises(ValueError, match="cutback_factor"):
    ArcLengthSettings(cutback_factor=1.0)
  with pytest.raises(ValueError, match="divergence_ratio"):
    ArcLengthSettings(divergence_ratio=1.0)


def test_driver_requires_exact_types() -> None:
  driver = _truss_driver()
  with pytest.raises(TypeError, match="exact CompiledSystem"):
    RiksDriver(object(), driver.plan)  # type: ignore[arg-type]
  with pytest.raises(TypeError, match="exact CompiledConstraintMap"):
    RiksDriver(
      compile_system(_truss_model(), truss_reference_registry()),
      object(),  # type: ignore[arg-type]
    )
  with pytest.raises(TypeError, match="exact ArcLengthSettings"):
    RiksDriver(
      compile_system(_truss_model(), truss_reference_registry()),
      _fixed_truss_map(compile_system(_truss_model(), truss_reference_registry())),
      (),
      object(),  # type: ignore[arg-type]
    )


def test_driver_requires_unique_load_coordinate() -> None:
  system = compile_system(_truss_model(), truss_reference_registry())
  time_only = compile_constraint_map(
    system,
    constraints=(),
    coordinates=(ProgramCoordinateSpec(name="time", kind="time"),),
  )
  with pytest.raises(
    DriverPreparationError,
    match="load-parameter-coordinate-required",
  ):
    RiksDriver(system, time_only, _apex_loads())
  two_loads = compile_constraint_map(
    system,
    constraints=(),
    coordinates=(
      ProgramCoordinateSpec(name="load", kind="load"),
      ProgramCoordinateSpec(name="load2", kind="load"),
    ),
  )
  with pytest.raises(
    DriverPreparationError,
    match="load-parameter-coordinate-ambiguous",
  ):
    RiksDriver(system, two_loads, _apex_loads())


def test_run_validates_base_point() -> None:
  driver = _truss_driver()
  with pytest.raises(TypeError, match="exact ProgramPoint"):
    driver.run(base_point=object())  # type: ignore[arg-type]
  with pytest.raises(ValueError, match="must not bind the continuation load"):
    driver.run(
      base_point=ProgramPoint((ProgramCoordinateValue("load", 0.0),)),
    )
  system = compile_system(_truss_model(), truss_reference_registry())
  timed = RiksDriver(
    system,
    _fixed_truss_map(
      system,
      extra_coordinates=(ProgramCoordinateSpec(name="time", kind="time"),),
    ),
    _apex_loads(),
  )
  with pytest.raises(ConstraintEvaluationError, match="missing-program-coordinate"):
    timed.run(base_point=_BASE_POINT)


def test_fixed_step_run_pins_predictor_scaling_and_lam0_quirk() -> None:
  driver = _truss_driver(_settings())
  result = driver.run(base_point=_BASE_POINT)
  assert result.status is DriverStatus.COMPLETED
  assert result.termination_reason is ArcLengthTermination.LOAD_PARAMETER_LIMIT
  assert result.failed_cycle is None
  records = result.records
  assert all(record.status is SubstepStatus.COMMITTED for record in records)
  lam1 = records[0].lam
  # The ported lam0 = 1.0 quirk: cycle 1 anchors its predictor exactly at 1.0.
  assert records[0].iterations[0].load_parameter == 1.0
  # Cycle 1's dlam accumulator equals its committed lam bitwise (both are the
  # float 1.0 plus the same ddlam trail), so the fixed-step cycle-2 predictor
  # anchors exactly at lam1 + lam1.
  assert records[1].iterations[0].load_parameter == lam1 + lam1
  # Termination is checked after the commit: the trajectory ends beyond max_lam.
  assert result.final_continuation.lam > 10.0
  # fixed_step pins the predictor factor at its initial value.
  assert all(record.factor == 1.0 for record in records)
  # Every committed cycle applied at least one Newton correction (the legacy
  # loop never accepts the predictor state).
  assert all(len(record.iterations) >= 2 for record in records)
  # The committed lam is exactly the predictor lam plus the correction trail.
  for record in records:
    iterations = record.iterations
    trail = iterations[0].load_parameter
    for iteration in iterations[:-1]:
      assert iteration.delta_lam is not None
      trail += iteration.delta_lam
    assert trail == record.lam
    # The commit record's delta_lam is the correction that led to the
    # converged iterate — the same value the last OK record carried.
    assert iterations[-1].delta_lam == iterations[-2].delta_lam
  committed = _committed(result)
  assert [record.committed_ordinal for record in committed] == list(
    range(1, len(committed) + 1)
  )
  assert result.final_generation.ordinal == len(committed)
  assert result.initial_continuation.lam == 1.0
  assert result.initial_continuation.cycle == 0
  assert result.initial_continuation.dlam_prev == 1.0
  assert result.final_continuation.cycle == len(committed)
  # The final committed reactions equilibrate lam * fhat: symmetric supports
  # each carry half the apex load.
  observation = records[-1].observation
  assert observation is not None
  lam_final = records[-1].lam
  reactions = observation.reactions.values
  np.testing.assert_allclose(
    reactions[[1, 3]],
    [50.0 * lam_final, 50.0 * lam_final],
    rtol=1.0e-9,
    atol=1.0e-8,
  )
  assert float(reactions[0] + reactions[2]) == pytest.approx(0.0, abs=1.0e-8)


def test_adaptive_factor_formula_matches_legacy() -> None:
  # max_lam below the cycle-1 committed lam ends the run after one commit.
  driver = _truss_driver(_settings(fixed_step=False, max_lam=0.5))
  result = driver.run(base_point=_BASE_POINT)
  assert result.status is DriverStatus.COMPLETED
  assert result.termination_reason is ArcLengthTermination.LOAD_PARAMETER_LIMIT
  (record,) = _committed(result)
  corrections = len(record.iterations) - 1
  # Legacy RiksSolver: factor = 0.5^(0.25*(iiter - optiter)), accumulated
  # into totalFactor; the default max_factor never triggers the reset.
  expected = float(0.5 ** (0.25 * (corrections - 5)))
  assert record.factor == expected
  assert result.final_continuation.factor == expected
  assert result.final_continuation.total_factor == expected
  multi = _truss_driver(_settings(fixed_step=False))
  multi_result = multi.run(base_point=_BASE_POINT)
  factors = [record.factor for record in _committed(multi_result)]
  total = 1.0
  for factor in factors:
    total *= factor
  assert multi_result.final_continuation.total_factor == total


def test_total_factor_quirk_resets_factor_but_never_total() -> None:
  # The legacy quirk: totalFactor > maxFactor resets the factor to 1.0 but
  # never resets totalFactor itself. With max_factor below every adaptive
  # factor, each commit recomputes the factor and resets it, so the run
  # follows the fixed-step trajectory bitwise.
  reference = _truss_driver(_settings())
  reference_result = reference.run(base_point=_BASE_POINT)
  driver = _truss_driver(_settings(fixed_step=False, max_factor=0.9))
  result = driver.run(base_point=_BASE_POINT)
  assert result.status is DriverStatus.COMPLETED
  reference_lams = [record.lam for record in _committed(reference_result)]
  lams = [record.lam for record in _committed(result)]
  assert lams == reference_lams
  committed = _committed(result)
  assert all(record.factor == 1.0 for record in committed)
  assert result.final_continuation.factor == 1.0
  total = 1.0
  for record in committed:
    corrections = len(record.iterations) - 1
    total *= float(0.5 ** (0.25 * (corrections - 5)))
  assert result.final_continuation.total_factor == total
  assert result.final_continuation.total_factor > 0.9


def test_cycle_cap_terminates_after_commit_with_cycle_limit() -> None:
  driver = _truss_driver(_settings(cycle_cap=2))
  result = driver.run(base_point=_BASE_POINT)
  assert result.status is DriverStatus.COMPLETED
  assert result.termination_reason is ArcLengthTermination.CYCLE_LIMIT
  assert result.failed_cycle is None
  # Legacy parity: termination uses cycle > cap, so cap = 2 commits 3 cycles.
  assert [record.cycle for record in result.records] == [1, 2, 3]
  assert result.records[-1].status is SubstepStatus.COMMITTED
  assert result.final_continuation.cycle == 3
  assert result.final_continuation.lam <= 10.0


def test_iteration_budget_rejection_leaves_state_and_baseline_untouched() -> None:
  # max_iterations = 1 allows no commit (convergence is checked from the
  # second evaluation on, the legacy iterMax edge), so cycle 1 exhausts the
  # cutback budget without any commit.
  driver = _truss_driver(_settings(max_iterations=1, max_cutbacks=2))
  snapshot = _owner_snapshot(driver.owner)
  result = driver.run(base_point=_BASE_POINT)
  assert result.status is DriverStatus.STEP_FAILED
  assert result.termination_reason is None
  assert result.failed_cycle == 1
  assert _owner_snapshot(driver.owner) == snapshot
  assert result.final_generation.ordinal == 0
  # The continuation baseline never advanced: it is the initial state.
  assert _continuation_snapshot(result.final_continuation) == _continuation_snapshot(
    result.initial_continuation
  )
  assert result.final_continuation.lam == 1.0
  assert result.final_continuation.dlam_prev == 1.0
  assert result.final_continuation.factor == 1.0
  assert result.final_continuation.total_factor == 1.0
  assert result.final_continuation.cycle == 0
  assert not result.final_continuation.da_prev.values.any()
  statuses = [record.status for record in result.records]
  assert statuses == [
    SubstepStatus.REJECTED,
    SubstepStatus.REJECTED,
    SubstepStatus.FAILED,
  ]
  assert [record.cutback_level for record in result.records] == [1, 2, 3]
  assert result.statistics.cutback_count == 2
  assert result.statistics.rejected_substep_count == 3
  # Cycle-1 cutback shrinks the initial lam0 jump uniformly: the attempts
  # anchor at 1.0, 0.5, 0.25.
  predictor_lams = [record.iterations[0].load_parameter for record in result.records]
  assert predictor_lams == [1.0, 0.5, 0.25]


def test_singular_tangent_fails_typed_with_baseline_untouched() -> None:
  # An isolated free node owns zero tangent rows: every factorization fails,
  # so the cycle-1 predictor itself drives the cutback cascade to exhaustion.
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
  driver = RiksDriver(system, coordinate_map, loads, _settings(max_cutbacks=2))
  snapshot = _owner_snapshot(driver.owner)
  result = driver.run(base_point=_BASE_POINT)
  assert result.status is DriverStatus.STEP_FAILED
  assert result.failed_cycle == 1
  assert result.termination_reason is None
  assert result.statistics.rejected_substep_count == 3
  assert _owner_snapshot(driver.owner) == snapshot
  assert _continuation_snapshot(result.final_continuation) == _continuation_snapshot(
    result.initial_continuation
  )


def test_reject_step_cutback_retries_same_generation_and_baseline() -> None:
  settings = _settings(max_lam=3.0)
  reference, reference_calls, reference_mark = _scripted_truss_driver({}, settings)
  reference_mark()
  reference_result = reference.run(base_point=_BASE_POINT)
  assert reference_result.status is DriverStatus.COMPLETED
  # Cycle 2's first evaluation is the runtime call right after cycle 1's
  # (cycle 1 = predictor tangent evaluation plus its Newton evaluations).
  cycle2_first_call = len(reference_result.records[0].iterations) + 1
  reject_map = {cycle2_first_call: EvaluationStatus.REJECT_STEP}
  driver, _calls, mark_runtime = _scripted_truss_driver(reject_map, settings)
  mark_runtime()
  result = driver.run(base_point=_BASE_POINT)
  assert result.status is DriverStatus.COMPLETED
  assert result.statistics.cutback_count == 1
  assert result.statistics.rejected_substep_count == 1
  rejected = [
    record for record in result.records if record.status is SubstepStatus.REJECTED
  ]
  (rejected_record,) = rejected
  assert rejected_record.cycle == 2
  assert rejected_record.cutback_level == 1
  # The rejected attempt used the baseline factor; the retry shrank it by the
  # cutback factor, and its predictor anchored from the SAME committed
  # baseline: lam1 + 0.5 * dlam_prev with dlam_prev == lam1 (cycle-1 quirk).
  assert rejected_record.factor == 1.0
  lam1 = result.records[0].lam
  committed_cycle2 = result.records[2]
  assert committed_cycle2.status is SubstepStatus.COMMITTED
  assert committed_cycle2.cycle == 2
  assert committed_cycle2.cutback_level == 1
  assert committed_cycle2.iterations[0].load_parameter == lam1 + 0.5 * lam1
  # Rejections never consume generations: committed ordinals stay dense.
  committed = _committed(result)
  assert [record.committed_ordinal for record in committed] == list(
    range(1, len(committed) + 1)
  )
  assert result.final_generation.ordinal == len(committed)


def test_reject_step_exhaustion_leaves_state_and_baseline_byte_identical() -> None:
  settings = _settings(max_lam=0.5)
  reference, _reference_calls, reference_mark = _scripted_truss_driver({}, settings)
  reference_mark()
  reference_result = reference.run(base_point=_BASE_POINT)
  assert reference_result.status is DriverStatus.COMPLETED
  assert len(_committed(reference_result)) == 1
  reference_owner = _owner_snapshot(reference.owner)
  reference_baseline = _continuation_snapshot(reference_result.final_continuation)
  cycle2_first_call = len(reference_result.records[0].iterations) + 1
  # Reject every cycle-2 evaluation so the cutback budget (2) exhausts.
  reject_map = {
    cycle2_first_call + index: EvaluationStatus.REJECT_STEP for index in range(50)
  }
  driver, _calls, mark_runtime = _scripted_truss_driver(
    reject_map,
    _settings(max_lam=10.0, max_cutbacks=2),
  )
  mark_runtime()
  result = driver.run(base_point=_BASE_POINT)
  assert result.status is DriverStatus.STEP_FAILED
  assert result.termination_reason is None
  assert result.failed_cycle == 2
  statuses = [record.status for record in result.records]
  assert statuses == [
    SubstepStatus.COMMITTED,
    SubstepStatus.REJECTED,
    SubstepStatus.REJECTED,
    SubstepStatus.FAILED,
  ]
  assert [record.cutback_level for record in result.records] == [0, 1, 2, 3]
  assert result.statistics.cutback_count == 2
  assert result.statistics.rejected_substep_count == 3
  # The owner and the continuation baseline are byte-identical to the
  # reference run that stopped right after cycle 1.
  assert _owner_snapshot(driver.owner) == reference_owner
  assert _continuation_snapshot(result.final_continuation) == reference_baseline
  assert [record.committed_ordinal for record in _committed(result)] == [1]


def test_reject_iteration_damps_the_correction_in_q_and_lam() -> None:
  settings = _settings(max_lam=3.0)
  reference, _reference_calls, reference_mark = _scripted_truss_driver({}, settings)
  reference_mark()
  reference_result = reference.run(base_point=_BASE_POINT)
  assert reference_result.status is DriverStatus.COMPLETED
  cycle2_first_call = len(reference_result.records[0].iterations) + 1
  # Reject cycle 2's evaluation AFTER its first Newton correction: the retry
  # damps that correction by half in both q and lam from the same anchor.
  reject_map = {cycle2_first_call + 1: EvaluationStatus.REJECT_ITERATION}
  driver, _calls, mark_runtime = _scripted_truss_driver(reject_map, settings)
  mark_runtime()
  result = driver.run(base_point=_BASE_POINT)
  assert result.status is DriverStatus.COMPLETED
  assert result.statistics.cutback_count == 0
  committed = _committed(result)
  cycle2 = committed[1]
  iterations = cycle2.iterations
  assert iterations[1].status is EvaluationStatus.REJECT_ITERATION
  assert iterations[1].residual_norm is None
  assert iterations[1].delta_lam is None
  # The rejected trial was the anchor plus the full correction; the damped
  # retry is the anchor plus half of it, exactly.
  assert (
    iterations[1].load_parameter
    == iterations[0].load_parameter + iterations[0].delta_lam
  )
  damped = iterations[2]
  assert (
    damped.load_parameter
    == iterations[0].load_parameter + 0.5 * iterations[0].delta_lam
  )
  if len(iterations) == 3:
    # The damped trial itself converged: the commit record carries the
    # halved correction that led to it.
    assert damped.delta_lam == 0.5 * iterations[0].delta_lam
    assert damped.increment_norm == pytest.approx(
      0.5 * iterations[0].increment_norm,
      rel=1.0e-15,
    )
  assert [record.committed_ordinal for record in committed] == list(
    range(1, len(committed) + 1)
  )


def test_first_evaluation_reject_iteration_escalates_to_cutback() -> None:
  settings = _settings(max_lam=3.0)
  reference, _reference_calls, reference_mark = _scripted_truss_driver({}, settings)
  reference_mark()
  reference_result = reference.run(base_point=_BASE_POINT)
  assert reference_result.status is DriverStatus.COMPLETED
  cycle2_first_call = len(reference_result.records[0].iterations) + 1
  # Rejecting the predictor-state evaluation leaves no Newton direction to
  # damp, so the protocol cuts back instead of retrying in place.
  reject_map = {cycle2_first_call: EvaluationStatus.REJECT_ITERATION}
  driver, _calls, mark_runtime = _scripted_truss_driver(reject_map, settings)
  mark_runtime()
  result = driver.run(base_point=_BASE_POINT)
  assert result.status is DriverStatus.COMPLETED
  assert result.statistics.cutback_count == 1
  rejected = [
    record for record in result.records if record.status is SubstepStatus.REJECTED
  ]
  (rejected_record,) = rejected
  assert rejected_record.cycle == 2
  assert rejected_record.iterations[0].status is EvaluationStatus.REJECT_ITERATION
  committed = _committed(result)
  assert [record.committed_ordinal for record in committed] == list(
    range(1, len(committed) + 1)
  )


def test_counters_pin_one_refill_one_splu_two_solves_per_correction() -> None:
  driver = _truss_driver(_settings())
  result = driver.run(base_point=_BASE_POINT)
  assert result.status is DriverStatus.COMPLETED
  statistics = result.statistics
  corrections = sum(len(record.iterations) - 1 for record in result.records)
  evaluations = sum(len(record.iterations) for record in result.records)
  # Per Newton correction exactly one values-only refill, one splu, and two
  # back-substitutions; the cycle-1 predictor adds one of each (plus its
  # tangent evaluation round, which assembles no residual).
  assert statistics.tangent_refill_count == corrections + 1
  assert statistics.factorization_count == corrections + 1
  assert statistics.factorization_reuse_count == 0
  assert statistics.linear_solve_count == 2 * corrections + 1
  assert statistics.evaluation_count == evaluations + 1
  assert statistics.residual_assembly_count == evaluations
  assert statistics.committed_substep_count == len(result.records)
  assert statistics.rejected_substep_count == 0
  assert statistics.cutback_count == 0
  # The ported legacy loop never accepts the predictor state: at least one
  # correction per committed cycle.
  assert corrections >= len(result.records)


def _linear_q8_driver(settings: ArcLengthSettings) -> RiksDriver:
  from pyfem.v3.compile.continuum import q8_reference_registry

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
  return RiksDriver(system, coordinate_map, loads, settings)


def test_linear_system_reuses_one_factorization_across_the_run() -> None:
  driver = _linear_q8_driver(_settings(max_lam=2.0))
  result = driver.run(base_point=_BASE_POINT)
  assert result.status is DriverStatus.COMPLETED
  statistics = result.statistics
  corrections = sum(len(record.iterations) - 1 for record in result.records)
  # Every Jacobian channel is compiled linear: one refill and one splu cover
  # the whole run; every later factorization request reuses the cache.
  assert statistics.tangent_refill_count == 1
  assert statistics.factorization_count == 1
  assert statistics.factorization_reuse_count == corrections
  assert statistics.linear_solve_count == 2 * corrections + 1
  second = driver.run(base_point=_BASE_POINT)
  assert second.status is DriverStatus.COMPLETED
  assert driver.statistics.factorization_count == 1
  assert driver.statistics.tangent_refill_count == 1


def test_repeated_runs_are_bitwise_identical() -> None:
  settings = _settings(fixed_step=False, max_lam=3.0)
  first = _truss_driver(settings).run(base_point=_BASE_POINT)
  second = _truss_driver(settings).run(base_point=_BASE_POINT)
  assert first.status is DriverStatus.COMPLETED
  assert second.status is DriverStatus.COMPLETED
  assert [record.lam for record in first.records] == [
    record.lam for record in second.records
  ]
  assert [record.factor for record in first.records] == [
    record.factor for record in second.records
  ]
  assert first.final_continuation.total_factor == second.final_continuation.total_factor
  first_states = [
    record.committed_coefficients.values.tobytes() for record in _committed(first)
  ]
  second_states = [
    record.committed_coefficients.values.tobytes() for record in _committed(second)
  ]
  assert first_states == second_states
  assert dataclasses.astuple(first.statistics) == dataclasses.astuple(second.statistics)


def test_mpc_tie_run_converges_and_equilibrates() -> None:
  # MPC smoke test. The tie makes the prolongation mix free DOFs
  # (P.T @ P != I), so the reduced-space Riks metric differs from the legacy
  # full-space one — a documented difference, hence no legacy parity here.
  system = compile_system(_truss_model(), truss_reference_registry())
  coordinate_map = compile_constraint_map(
    system,
    constraints=(
      *(
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
      AffineTieSpec(
        id="tie-apex",
        slave=DofRef(node_id=2, field_id="displacement", component="x"),
        master=DofRef(node_id=2, field_id="displacement", component="y"),
        factor=1.0,
        offset=AffineValueSpec(constant=0.0),
        source=_source("tie"),
      ),
    ),
    coordinates=(ProgramCoordinateSpec(name="load", kind="load"),),
  )
  assert coordinate_map.reduced_dof_count == 1
  driver = RiksDriver(system, coordinate_map, _apex_loads(), _settings(max_lam=1.5))
  result = driver.run(base_point=_BASE_POINT)
  assert result.status is DriverStatus.COMPLETED
  assert result.termination_reason is ArcLengthTermination.LOAD_PARAMETER_LIMIT
  final = result.records[-1]
  coefficients = final.committed_coefficients
  assert coefficients is not None
  # The map enforces the tie exactly: slave and master share one reduced DOF.
  assert coefficients.values[4] == coefficients.values[5]
  observation = final.observation
  assert observation is not None
  reactions = observation.reactions.values
  # The honest invariant under a tie is reduced equilibrium (P.T @ r = 0):
  # the tie force flows through the free master DOF, so a plain support-sum
  # has no closed form. The supports still carry genuine reaction load.
  assert observation.reduced_residual_norm <= 1.0e-10 * (100.0 * final.lam)
  assert float(reactions[1] + reactions[3]) != 0.0
