# SPDX-License-Identifier: MIT

"""Nonzero-state tangent conformance battery for the spring compile boundary."""

from __future__ import annotations

import math
import sys

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

import pyfem.v3.compile.spring as spring_compile
from pyfem.v3.compile.continuum import q8_reference_registry
from pyfem.v3.compile.diagnostics import ModelCompilationError
from pyfem.v3.compile.spring import (
  SpringDeclaration,
  SpringKernel,
  SpringKernelResult,
  SpringStateSlot,
  compile_spring_operator,
  compose_system,
  damage_envelope_declaration,
  damage_envelope_kernel,
)
from pyfem.v3.compile.system import compile_system
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

_SLIP_BUDGET = {"iterations": 64}


def _source(label: str) -> SourceContext:
  return SourceContext(source=label)


def _q8_model() -> ModelSpec:
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
      MaterialParameterSpec("poisson_ratio", 0.0, _source("material:nu")),
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


def _base_system() -> CompiledSystem:
  return compile_system(_q8_model(), q8_reference_registry())


def _damage_declaration(max_increment: float) -> SpringDeclaration:
  return damage_envelope_declaration(
    block_id="damage-springs",
    space_id="displacement",
    spring_ids=("spring-1", "spring-2"),
    node_ids=(1, 2),
    stiffness=2.0,
    critical_extension=2.0,
    max_increment=max_increment,
    source=_source("damage-spring-source"),
  )


def researcher_slip_kernel(
  displacements: np.ndarray,
  accepted_rows: np.ndarray,
  parameters: np.ndarray,
) -> SpringKernelResult:
  """Researcher-authored elasto-slip spring with exponential gap hardening.

  Kept numerically identical to the landed kernel in the state-transaction
  battery: the slip update solves ``delta = m - gap * exp(hardening *
  (|s| + delta))`` by budgeted fixed-point iterations; exhausting the local
  budget is an expected outcome, reported as ``REJECT_ITERATION``.
  """
  stiffness, gap, hardening = parameters
  entity_count = len(displacements)
  force = np.zeros((entity_count, 2), dtype=np.float64)
  tangent = np.zeros((entity_count, 2, 2), dtype=np.float64)
  trial_rows = np.array(accepted_rows, dtype=np.float64, copy=True)
  for index in range(entity_count):
    predictor = displacements[index] - accepted_rows[index]
    magnitude = float(np.linalg.norm(predictor))
    slip_norm = float(np.linalg.norm(accepted_rows[index]))
    yield_gap = gap * math.exp(hardening * slip_norm)
    if magnitude <= yield_gap:
      force[index] = stiffness * predictor
      tangent[index] = stiffness * np.eye(2)
      continue
    direction = predictor / magnitude
    delta = 0.0
    converged = False
    for _ in range(_SLIP_BUDGET["iterations"]):
      updated = magnitude - gap * math.exp(hardening * (slip_norm + delta))
      if abs(updated - delta) <= 1.0e-13:
        delta = updated
        converged = True
        break
      delta = updated
    if not converged:
      return SpringKernelResult(
        force=np.zeros_like(displacements),
        tangent=np.zeros((entity_count, 2, 2), dtype=np.float64),
        trial_rows=np.array(accepted_rows, dtype=np.float64, copy=True),
        status=EvaluationStatus.REJECT_ITERATION,
      )
    trial_rows[index] = accepted_rows[index] + delta * direction
    force[index] = stiffness * (magnitude - delta) * direction
    hardening_slope = 1.0 / (
      1.0 + gap * hardening * math.exp(hardening * (slip_norm + delta))
    )
    tangent[index] = stiffness * (
      (magnitude - delta) / magnitude * (np.eye(2) - np.outer(direction, direction))
      + (1.0 - hardening_slope) * np.outer(direction, direction)
    )
  return SpringKernelResult(
    force=force,
    tangent=tangent,
    trial_rows=trial_rows,
    status=EvaluationStatus.OK,
  )


def _slip_declaration() -> SpringDeclaration:
  return SpringDeclaration(
    block_id="slip-springs",
    space_id="displacement",
    spring_ids=("slip-1",),
    node_ids=(3,),
    state_schema="research-slip-hardening-v1",
    state_slots=(SpringStateSlot("slip", 2),),
    kernel_name="exponential-hardening-slip-spring",
    kernel_version="1",
    implementation_id="research-slip-hardening-v1",
    parameters=(3.0, 0.2, 0.5),
    kernel=researcher_slip_kernel,
    source=_source("slip-spring-source"),
  )


def planted_nonzero_tangent_kernel(
  displacements: np.ndarray,
  accepted_rows: np.ndarray,
  parameters: np.ndarray,
) -> SpringKernelResult:
  """Planted violation from the X1 spike: the force is a consistent linear
  spring everywhere, but the tangent halves off the virgin state (1.5 I vs
  3 I), so the kernel passes a virgin-only probe and evaluates silently
  wrong at nonzero accepted state.
  """
  stiffness = float(parameters[0])
  entity_count = len(displacements)
  tangent_scale = stiffness if not bool(np.any(accepted_rows)) else 0.5 * stiffness
  return SpringKernelResult(
    force=stiffness * displacements,
    tangent=tangent_scale * np.tile(np.eye(2), (entity_count, 1, 1)),
    trial_rows=np.array(accepted_rows, copy=True),
    status=EvaluationStatus.OK,
  )


def _kernel_declaration(
  *,
  block_id: str,
  kernel: SpringKernel,
  name: str,
  source_label: str,
) -> SpringDeclaration:
  return SpringDeclaration(
    block_id=block_id,
    space_id="displacement",
    spring_ids=(f"{block_id}-1",),
    node_ids=(1,),
    state_schema=name,
    state_slots=(SpringStateSlot("marker", 1),),
    kernel_name=name,
    kernel_version="1",
    implementation_id=name,
    parameters=(3.0,),
    kernel=kernel,
    source=_source(source_label),
  )


def _planted_declaration() -> SpringDeclaration:
  return _kernel_declaration(
    block_id="planted-springs",
    kernel=planted_nonzero_tangent_kernel,
    name="planted-nonzero-tangent-v1",
    source_label="planted-kernel-source",
  )


def _accepted_probe_state_count(
  kernel: SpringKernel,
  parameters: tuple[float, ...],
  entity_count: int,
  row_width: int,
) -> int:
  """Replay the compile probe's seeded draws and count kernel-accepted states."""
  generator = np.random.default_rng(spring_compile._TANGENT_PROBE_SEED)
  values = np.array(parameters, dtype=np.float64)
  accepted = 0
  for _ in range(spring_compile._TANGENT_PROBE_STATE_COUNT):
    displacements = generator.standard_normal((entity_count, 2))
    accepted_rows = generator.standard_normal((entity_count, row_width))
    result = kernel(displacements, accepted_rows, values)
    accepted += result.status is EvaluationStatus.OK
  return accepted


def test_planted_kernel_with_tangent_wrong_only_at_nonzero_state_is_rejected() -> None:
  # The planted kernel passes the virgin probe (its tangent is correct there);
  # only the seeded nonzero-state tangent check can reject it.
  with pytest.raises(ModelCompilationError) as excinfo:
    compile_spring_operator(_base_system(), _planted_declaration())
  (diagnostic,) = excinfo.value.diagnostics
  assert diagnostic.code == "inconsistent-kernel-tangent"
  assert str(spring_compile._TANGENT_PROBE_SEED) in diagnostic.message
  assert diagnostic.source.source == "planted-kernel-source"


def test_nonzero_tangent_probe_diagnostic_replays_bit_for_bit() -> None:
  messages: list[str] = []
  for _ in range(2):
    with pytest.raises(ModelCompilationError) as excinfo:
      compile_spring_operator(_base_system(), _planted_declaration())
    messages.append(str(excinfo.value))
  assert messages[0] == messages[1]
  assert str(spring_compile._TANGENT_PROBE_SEED) in messages[0]


def test_landed_spring_kernels_pass_the_nonzero_tangent_probe() -> None:
  base = _base_system()
  block, operator = compile_spring_operator(base, _damage_declaration(5.0))
  system = compose_system(base, block, operator)
  slip_block, slip_operator = compile_spring_operator(system, _slip_declaration())
  compose_system(system, slip_block, slip_operator)
  # The probe verifies only states the kernel accepts; replaying the seeded
  # draws proves both landed kernels were finite-difference checked off the
  # virgin state at every probe state.
  assert (
    _accepted_probe_state_count(
      damage_envelope_kernel,
      (2.0, 2.0, 5.0),
      2,
      1,
    )
    == spring_compile._TANGENT_PROBE_STATE_COUNT
  )
  assert (
    _accepted_probe_state_count(
      researcher_slip_kernel,
      (3.0, 0.2, 0.5),
      1,
      2,
    )
    == spring_compile._TANGENT_PROBE_STATE_COUNT
  )


def test_tight_envelope_probe_states_are_skipped_not_failed() -> None:
  # The state-transaction battery's exact damage parameters reject every
  # seeded probe state with a typed status; typed rejections carry no channels
  # to verify, so compilation still succeeds.
  base = _base_system()
  block, operator = compile_spring_operator(base, _damage_declaration(0.5))
  compose_system(base, block, operator)
  assert (
    _accepted_probe_state_count(
      damage_envelope_kernel,
      (2.0, 2.0, 0.5),
      2,
      1,
    )
    == 0
  )


def test_typed_rejection_at_nonzero_probe_states_compiles_clean() -> None:
  def rejecting_kernel(
    displacements: np.ndarray,
    accepted_rows: np.ndarray,
    parameters: np.ndarray,
  ) -> SpringKernelResult:
    stiffness = float(parameters[0])
    entity_count = len(displacements)
    if bool(np.any(accepted_rows)):
      return SpringKernelResult(
        force=np.zeros_like(displacements),
        tangent=np.zeros((entity_count, 2, 2)),
        trial_rows=np.array(accepted_rows, copy=True),
        status=EvaluationStatus.REJECT_STEP,
      )
    return SpringKernelResult(
      force=stiffness * displacements,
      tangent=stiffness * np.tile(np.eye(2), (entity_count, 1, 1)),
      trial_rows=np.array(accepted_rows, copy=True),
      status=EvaluationStatus.OK,
    )

  base = _base_system()
  block, operator = compile_spring_operator(
    base,
    _kernel_declaration(
      block_id="rejecting-springs",
      kernel=rejecting_kernel,
      name="rejecting-kernel-v1",
      source_label="rejecting-kernel-source",
    ),
  )
  compose_system(base, block, operator)


def test_kernel_raising_at_a_nonzero_probe_state_fails_typed() -> None:
  def raising_kernel(
    displacements: np.ndarray,
    accepted_rows: np.ndarray,
    parameters: np.ndarray,
  ) -> SpringKernelResult:
    if bool(np.any(accepted_rows)):
      msg = "researcher failure off the virgin state"
      raise RuntimeError(msg)
    stiffness = float(parameters[0])
    entity_count = len(displacements)
    return SpringKernelResult(
      force=stiffness * displacements,
      tangent=stiffness * np.tile(np.eye(2), (entity_count, 1, 1)),
      trial_rows=np.array(accepted_rows, copy=True),
      status=EvaluationStatus.OK,
    )

  with pytest.raises(ModelCompilationError) as excinfo:
    compile_spring_operator(
      _base_system(),
      _kernel_declaration(
        block_id="raising-springs",
        kernel=raising_kernel,
        name="raising-kernel-v1",
        source_label="raising-kernel-source",
      ),
    )
  (diagnostic,) = excinfo.value.diagnostics
  assert diagnostic.code == "kernel-probe-failed"
  assert str(spring_compile._TANGENT_PROBE_SEED) in diagnostic.message


def test_kernel_mutating_inputs_at_a_nonzero_probe_state_fails_typed() -> None:
  def mutating_kernel(
    displacements: np.ndarray,
    accepted_rows: np.ndarray,
    parameters: np.ndarray,
  ) -> SpringKernelResult:
    if bool(np.any(accepted_rows)):
      accepted_rows[:] = 0.0
    stiffness = float(parameters[0])
    entity_count = len(displacements)
    return SpringKernelResult(
      force=stiffness * displacements,
      tangent=stiffness * np.tile(np.eye(2), (entity_count, 1, 1)),
      trial_rows=np.array(accepted_rows, copy=True),
      status=EvaluationStatus.OK,
    )

  with pytest.raises(ModelCompilationError, match="kernel-probe-failed"):
    compile_spring_operator(
      _base_system(),
      _kernel_declaration(
        block_id="mutating-springs",
        kernel=mutating_kernel,
        name="mutating-kernel-v1",
        source_label="mutating-kernel-source",
      ),
    )
