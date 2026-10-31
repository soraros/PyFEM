"""FE2 nested-evaluation and nested-state memory benchmark pipelines."""

from __future__ import annotations

import tracemalloc
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from bench.gates import CheckResult, GateResult
from bench.workloads import FE2MemoryCase, FE2Workload
from pyfem.v3 import authoring
from pyfem.v3.compile.continuum import damage_reference_registry, q8_reference_registry
from pyfem.v3.compile.system import compile_system
from pyfem.v3.constraints import compile_constraint_map
from pyfem.v3.driver import (
  DriverStatus,
  NonlinearStaticDriver,
  NonlinearStaticSettings,
)
from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.operator import (
  ChannelRequest,
  EvaluationStatus,
  NestedEvaluationInput,
  OperatorEvaluationInput,
  evaluation_status,
)
from pyfem.v3.spec.program import (
  AffineCoefficientSpec,
  AffineValueSpec,
  DofRef,
  PrescribedDofSpec,
  ProgramCoordinateSpec,
  ProgramCoordinateValue,
  ProgramPoint,
)
from pyfem.v3.state import NestedBlockLayout, NestedStateLayout, NestedStateStore

if TYPE_CHECKING:
  from pyfem.v3.constraints import CompiledConstraintMap
  from pyfem.v3.model.operator import OperatorEvaluation
  from pyfem.v3.model.system import CompiledSystem

_E = 1000.0
_NU = 0.25
_KAPPA0 = 7.5e-4
_KAPPA_C = 2.5e-2
_DAMAGE_K = 1.0
_PRESCRIBED = 5.0e-4
_MEMORY_EVALUATIONS = 3


def _load_point(value: float) -> ProgramPoint:
  return ProgramPoint((ProgramCoordinateValue("load", value),))


@dataclass(frozen=True, slots=True)
class _FE2Deck:
  system: CompiledSystem
  coordinate_map: CompiledConstraintMap


def build_fe2_deck(workload: FE2Workload) -> _FE2Deck:
  """Compile the documented periodic-microcell FE2 cantilever."""
  micro_mesh = authoring.quad8_patch(2, 2, width=1.0, height=1.0)
  material = authoring.damage(_E, _NU, _KAPPA0, _KAPPA_C, _DAMAGE_K)
  micro_system = compile_system(
    authoring.small_strain_continuum(micro_mesh, material=material),
    damage_reference_registry(),
  )
  cell = authoring.periodic_cell(micro_mesh, width=1.0, height=1.0)
  micro_map = compile_constraint_map(
    micro_system, constraints=cell.constraints, coordinates=cell.coordinates
  )
  macro_mesh = authoring.quad8_patch(workload.nx, workload.ny, width=1.0, height=1.0)
  base = compile_system(
    authoring.small_strain_continuum(
      macro_mesh, material=authoring.linear_elastic(_E, _NU)
    ),
    q8_reference_registry(),
  )
  system = authoring.fe2_continuum(
    base,
    micro_system=micro_system,
    micro_coordinate_map=micro_map,
    homogenization=cell.homogenization_request(micro_system),
  )
  left = tuple(node.id for node in macro_mesh.nodes if abs(node.coordinates[0]) < 1e-12)
  right = tuple(
    node.id for node in macro_mesh.nodes if abs(node.coordinates[0] - 1.0) < 1e-12
  )
  constraints = list(authoring.fixed(left, components=("x", "y")))
  for node in right:
    constraints.append(
      PrescribedDofSpec(
        id=f"right-{node}",
        target=DofRef(node, "displacement", "x"),
        value=AffineValueSpec(
          coefficients=(AffineCoefficientSpec("load", _PRESCRIBED),)
        ),
      )
    )
  coordinate_map = compile_constraint_map(
    system,
    constraints=tuple(constraints),
    coordinates=(ProgramCoordinateSpec("load", "load"),),
  )
  return _FE2Deck(system, coordinate_map)


class FE2Pipeline:
  """One fresh FE2 system, its macro driver, and steady-state evaluation call."""

  def __init__(self, workload: FE2Workload) -> None:
    self.workload = workload
    deck = build_fe2_deck(workload)
    self.system = deck.system
    self.coordinate_map = deck.coordinate_map
    self.driver: NonlinearStaticDriver | None = None

  @property
  def logical_bytes(self) -> dict[str, int]:
    operator = self.system.operators[0]
    layout = operator.nested_store.layout
    per_ip = 8 * (
      layout.coefficient_count
      + sum(block.entity_count * block.row_width for block in layout.blocks)
    )
    nested = layout.ip_count * per_ip
    macro = operator.header.state_layout.entity_count * 3 * 8
    return {
      "accepted_nested": nested,
      "staged_nested": nested,
      "accepted_macro_strain": macro,
      "nested_evaluation_copy": 0,
    }

  def solve(self) -> object:
    """Run this pipeline's one fresh macro solve from its virgin state."""
    if self.driver is not None:
      msg = "an FE2 benchmark pipeline supports one fresh macro solve"
      raise RuntimeError(msg)
    self.driver = NonlinearStaticDriver(
      self.system,
      self.coordinate_map,
      settings=NonlinearStaticSettings(tolerance=1.0e-9),
    )
    result = self.driver.run(
      base_point=_load_point(0.0), target_points=(_load_point(1.0),)
    )
    self.result = result
    return result

  def nested_evaluation(self) -> OperatorEvaluation:
    """Evaluate homogenized outputs at the committed state, then reject the trial."""
    if self.driver is None:
      msg = "FE2 nested evaluation requires a solved pipeline"
      raise RuntimeError(msg)
    owner = self.driver.owner
    operator = self.system.operators[0]
    gather = operator.header.ports[0].coefficient_map.values
    store = operator.nested_store
    store.begin_trial()
    try:
      return operator.evaluate(
        OperatorEvaluationInput(
          port_values=(
            FinalizedArray(owner.accepted_physical().values[gather], dtype=np.float64),
          ),
          accepted_state=owner.accepted_state(operator.header.state_layout.block_id),
          signals=(),
          request=ChannelRequest(("internal-force",), ("material-tangent",)),
          nested_execution=NestedEvaluationInput(store),
        )
      )
    finally:
      if store.trial_open:
        store.reject_trial()


def solve_fresh_fe2(workload: FE2Workload) -> object:
  """Compile and solve a fresh FE2 case, including the virgin nested probe."""
  pipeline = FE2Pipeline(workload)
  return pipeline.solve()


def _uint64(values: np.ndarray) -> np.ndarray:
  return np.ascontiguousarray(values, dtype=np.float64).view(np.uint64)


def _identity_check(name: str, first: np.ndarray, second: np.ndarray) -> CheckResult:
  passed = first.shape == second.shape and bool(
    np.array_equal(_uint64(first), _uint64(second))
  )
  max_abs = (
    float(np.max(np.abs(first - second))) if first.shape == second.shape else np.inf
  )
  return CheckResult(
    name=name,
    passed=passed,
    max_rel_diff=0.0 if passed else np.inf,
    rtol=0.0,
    atol=0.0,
    detail=f"raw-uint64 identity; max_abs={max_abs:.3e}",
  )


def gate_fe2(reference: FE2Pipeline, candidate: FE2Pipeline) -> GateResult:
  """Apply the nested D2 race gate to committed states and homogenized outputs."""
  workload = candidate.workload.name
  reference_result = reference.result
  candidate_result = candidate.result
  status_passed = (
    reference_result.status is DriverStatus.COMPLETED
    and candidate_result.status is DriverStatus.COMPLETED
  )
  checks = [
    CheckResult(
      name="completed-macro-solves",
      passed=status_passed,
      max_rel_diff=0.0 if status_passed else np.inf,
      rtol=0.0,
      atol=0.0,
      detail=(
        f"reference={reference_result.status}, candidate={candidate_result.status}"
      ),
    )
  ]
  if not status_passed:
    return GateResult(workload, False, checks)
  reference_driver = reference.driver
  candidate_driver = candidate.driver
  assert reference_driver is not None and candidate_driver is not None
  reference_operator = reference.system.operators[0]
  candidate_operator = candidate.system.operators[0]
  block_id = reference_operator.header.state_layout.block_id
  checks.append(
    _identity_check(
      "committed-macro-physical",
      reference_driver.owner.accepted_physical().values,
      candidate_driver.owner.accepted_physical().values,
    )
  )
  checks.append(
    _identity_check(
      "committed-macro-strain",
      reference_driver.owner.accepted_state(block_id).values,
      candidate_driver.owner.accepted_state(block_id).values,
    )
  )
  nested_equal = (
    reference_operator.nested_store.encode() == candidate_operator.nested_store.encode()
  )
  checks.append(
    CheckResult(
      name="committed-nested-store",
      passed=nested_equal,
      max_rel_diff=0.0 if nested_equal else np.inf,
      rtol=0.0,
      atol=0.0,
      detail="versioned accepted-state payload byte identity",
    )
  )
  reference_evaluation = reference.nested_evaluation()
  candidate_evaluation = candidate.nested_evaluation()
  evaluation_passed = (
    evaluation_status(reference_evaluation) is EvaluationStatus.OK
    and evaluation_status(candidate_evaluation) is EvaluationStatus.OK
  )
  checks.append(
    CheckResult(
      name="nested-evaluation-status",
      passed=evaluation_passed,
      max_rel_diff=0.0 if evaluation_passed else np.inf,
      rtol=0.0,
      atol=0.0,
      detail="both rejected-trial homogenized evaluations completed",
    )
  )
  if evaluation_passed:
    for name, channel in (
      ("homogenized-internal-force", "residual_values"),
      ("homogenized-material-tangent", "jacobian_values"),
      ("trial-macro-strain", "trial_state"),
    ):
      first = getattr(reference_evaluation, channel)
      second = getattr(candidate_evaluation, channel)
      first_values = first.values if name == "trial-macro-strain" else first[0].values
      second_values = (
        second.values if name == "trial-macro-strain" else second[0].values
      )
      checks.append(_identity_check(name, first_values, second_values))
  return GateResult(workload, all(check.passed for check in checks), checks)


def _memory_layout(macro_ips: int) -> NestedStateLayout:
  return NestedStateLayout(
    schema="bench-fe2-memory-v1",
    coefficient_count=18,
    blocks=(
      NestedBlockLayout(
        schema="bench-fe2-memory-damage-v1",
        block_id="cells",
        entity_count=16,
        row_width=1,
      ),
    ),
    ip_count=macro_ips,
  )


type _MemoryState = dict[str, np.ndarray | int]


def _new_memory_state() -> _MemoryState:
  return {
    "coefficients": np.zeros(18, dtype=np.float64),
    "rows": np.zeros((16, 1), dtype=np.float64),
    "cycle": 0,
  }


class _LegacyMicroStates:
  """Legacy MicroModel.py:181/:241 deepcopy-per-evaluation/commit model."""

  def __init__(self, macro_ips: int) -> None:
    self.committed = [_new_memory_state() for _ in range(macro_ips)]
    self.trial = deepcopy(self.committed)

  def step(self) -> None:
    for evaluation in range(_MEMORY_EVALUATIONS):
      trial = deepcopy(self.committed)
      for ip_index, state in enumerate(trial):
        coefficients = state["coefficients"]
        rows = state["rows"]
        assert isinstance(coefficients, np.ndarray) and isinstance(rows, np.ndarray)
        coefficients[:] = evaluation + 0.125 * ip_index
        rows[:] = evaluation + 0.5 + 0.125 * ip_index
        state["cycle"] = int(state["cycle"]) + 1
      self.trial = trial
    self.committed = deepcopy(self.trial)


def _production_memory_step(store: NestedStateStore) -> None:
  store.begin_trial()
  try:
    for evaluation in range(_MEMORY_EVALUATIONS):
      for ip_index in range(store.ip_count):
        store.trial_coefficients(ip_index)[:] = evaluation + 0.125 * ip_index
        store.trial_block(ip_index, "cells")[:] = evaluation + 0.5 + 0.125 * ip_index
        if evaluation == _MEMORY_EVALUATIONS - 1:
          store.trial_complete(ip_index)
    store.commit_trial()
  finally:
    if store.trial_open:
      store.reject_trial()


def _traced_peak_bytes(step: object) -> int:
  tracemalloc.start()
  try:
    tracemalloc.reset_peak()
    assert callable(step)
    step()
    _current, peak = tracemalloc.get_traced_memory()
    return peak
  finally:
    tracemalloc.stop()


def _memory_arrays(states: list[_MemoryState]) -> tuple[np.ndarray, np.ndarray]:
  coefficients = np.array([state["coefficients"] for state in states])
  rows = np.array([state["rows"] for state in states])
  return coefficients, rows


def _memory_logical_bytes(case: FE2MemoryCase) -> dict[str, int]:
  volume = case.macro_ips * case.per_ip_bytes
  return {
    "macro_ip_count": case.macro_ips,
    "per_ip_bytes": case.per_ip_bytes,
    "accepted_or_committed": volume,
    "staged_or_trial": volume,
    "production_evaluation_copy": 0,
    "production_commit_copy": volume,
    "legacy_evaluation_copy": _MEMORY_EVALUATIONS * volume,
    "legacy_commit_copy": volume,
  }


def run_fe2_memory_case(case: FE2MemoryCase) -> tuple[object, object, GateResult]:
  """Measure the allocation/copy curve against the named legacy antipattern."""
  from bench.common import BenchRecord, measure_fixed

  store = NestedStateStore(_memory_layout(case.macro_ips))
  legacy = _LegacyMicroStates(case.macro_ips)
  store.begin_trial()
  coefficient_buffer = store.trial_coefficients(0).base
  block_buffer = store.trial_block(0, "cells").base
  store.reject_trial()
  production_stats = measure_fixed(
    lambda: _production_memory_step(store), warmup=1, reps=3
  )
  legacy_stats = measure_fixed(legacy.step, warmup=1, reps=3)
  production_stats.note = (
    "3 evaluations + 1 commit; preallocated scoped-sub-transaction buffers"
  )
  legacy_stats.note = (
    "3 evaluations + 1 commit; MicroModel.py:181/:241 deepcopy stand-in"
  )
  store.begin_trial()
  buffers_held = (
    store.trial_coefficients(0).base is coefficient_buffer
    and store.trial_block(0, "cells").base is block_buffer
  )
  store.reject_trial()
  snapshot = store.snapshot()
  legacy_coefficients, legacy_rows = _memory_arrays(legacy.committed)
  state_equal = bool(
    np.array_equal(snapshot.coefficients.values, legacy_coefficients)
    and np.array_equal(snapshot.blocks[0].values, legacy_rows)
  )
  production_peak = _traced_peak_bytes(lambda: _production_memory_step(store))
  legacy_peak = _traced_peak_bytes(legacy.step)
  checks = [
    CheckResult(
      name="scoped-buffer-object-identity",
      passed=buffers_held,
      max_rel_diff=0.0 if buffers_held else np.inf,
      rtol=0.0,
      atol=0.0,
      detail="coefficient and state-block base objects persist across committed steps",
    ),
    CheckResult(
      name="legacy-copy-model-state-parity",
      passed=state_equal,
      max_rel_diff=0.0 if state_equal else np.inf,
      rtol=0.0,
      atol=0.0,
      detail="production and deepcopy protocols commit identical float64 rows",
    ),
  ]
  gate = GateResult(case.workload, all(check.passed for check in checks), checks)
  logical = _memory_logical_bytes(case)
  production_logical = {
    **logical,
    "traced_step_peak": production_peak,
  }
  legacy_logical = {
    **logical,
    "traced_step_peak": legacy_peak,
  }
  production_record = BenchRecord(
    category="fe2-memory",
    workload=case.workload,
    side="v3",
    mode="warm",
    threads=1,
    n_elems=case.macro_ips,
    n_dofs=18 * case.macro_ips,
    metrics={"step": production_stats},
    logical_bytes=production_logical,
    correctness=gate.to_json(),
  )
  legacy_record = BenchRecord(
    category="fe2-memory",
    workload=case.workload,
    side="legacy",
    mode="warm",
    threads=1,
    n_elems=case.macro_ips,
    n_dofs=18 * case.macro_ips,
    metrics={"step": legacy_stats},
    logical_bytes=legacy_logical,
    correctness=gate.to_json(),
  )
  return production_record, legacy_record, gate
