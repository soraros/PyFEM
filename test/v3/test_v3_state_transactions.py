# SPDX-License-Identifier: MIT

"""Proof C acceptance battery for the state transaction core."""

from __future__ import annotations

import copy
import math
import pickle
import sys
from dataclasses import replace

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from pyfem.v3.compile.continuum import (
  Q8ContinuumOperator,
  plasticity_reference_registry,
  q8_reference_registry,
)
from pyfem.v3.compile.diagnostics import ModelCompilationError
from pyfem.v3.compile.spring import (
  SpringDeclaration,
  SpringKernelResult,
  SpringOperator,
  SpringStateSlot,
  compile_spring_operator,
  compose_system,
  damage_envelope_declaration,
)
from pyfem.v3.compile.system import compile_system
from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.identity import IdentityMismatchError, require_generation_successor
from pyfem.v3.model.operator import (
  ChannelRequest,
  EvaluationStatus,
  OperatorEvaluationInput,
  evaluation_status,
)
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
from pyfem.v3.state import (
  STATE_ROW_CODEC_FORMAT,
  Float64StateRowCodec,
  StateCodecError,
  StateTransaction,
  StateTransactionOwner,
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
    mesh=MeshSpec(nodes=(nodes), cell_blocks=(block,), source=_source("mesh-source")),
    fields=(field,),
    materials=(material,),
    regions=(region,),
    source=_source("model-source"),
  )


def _damage_declaration() -> SpringDeclaration:
  return damage_envelope_declaration(
    block_id="damage-springs",
    space_id="displacement",
    spring_ids=("spring-1", "spring-2"),
    node_ids=(1, 2),
    stiffness=2.0,
    critical_extension=2.0,
    max_increment=0.5,
    source=_source("damage-spring-source"),
  )


def researcher_slip_kernel(
  displacements: np.ndarray,
  accepted_rows: np.ndarray,
  parameters: np.ndarray,
) -> SpringKernelResult:
  """Researcher-authored elasto-slip spring with exponential gap hardening.

  The slip update solves ``delta = m - gap * exp(hardening * (|s| + delta))``
  by budgeted fixed-point iterations; exhausting the local budget is an
  expected outcome, reported as ``REJECT_ITERATION`` so the driver can retry
  from the same accepted state with a larger budget.
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
        tangent=np.zeros((entity_count, 2, 2)),
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


def _base_system() -> CompiledSystem:
  return compile_system(_q8_model(), q8_reference_registry())


def _damage_system() -> CompiledSystem:
  base = _base_system()
  block, operator = compile_spring_operator(base, _damage_declaration())
  return compose_system(base, block, operator)


def _full_system() -> CompiledSystem:
  system = _damage_system()
  block, operator = compile_spring_operator(system, _slip_declaration())
  return compose_system(system, block, operator)


def _operators(system: CompiledSystem) -> tuple[object, SpringOperator, SpringOperator]:
  q8_operator, damage_operator, slip_operator = system.operators
  assert isinstance(q8_operator, Q8ContinuumOperator)
  assert isinstance(damage_operator, SpringOperator)
  assert isinstance(slip_operator, SpringOperator)
  return q8_operator, damage_operator, slip_operator


def _trial_vector(
  system: CompiledSystem,
  assignments: dict[int, tuple[float, float]],
) -> np.ndarray:
  values = np.zeros(system.coefficient_count, dtype=np.float64)
  space = system.spaces[0]
  nodes = system.point_blocks[0]
  for node_id, displacement in assignments.items():
    index = nodes.entity_ids.index(node_id)
    values[space.coefficient_map.values[index]] = displacement
  return values


def _request(*channel_ids: str) -> ChannelRequest:
  residual = tuple(channel for channel in channel_ids if "force" in channel)
  jacobian = tuple(channel for channel in channel_ids if "tangent" in channel)
  return ChannelRequest(residual, jacobian)


def _evaluate(
  operator: object,
  transaction: StateTransaction,
  trial_physical: np.ndarray,
  *channel_ids: str,
) -> object:
  header = operator.header
  batch = trial_physical[header.ports[0].coefficient_map.values]
  return operator.evaluate(
    OperatorEvaluationInput(
      port_values=(FinalizedArray(batch, dtype=np.float64),),
      accepted_state=transaction.accepted_state(header.state_layout.block_id),
      signals=(),
      request=_request(*channel_ids),
    )
  )


def _accepted_bytes(
  owner: StateTransactionOwner,
) -> tuple[int, bytes, tuple[tuple[object, bytes], ...]]:
  return (
    owner.generation.ordinal,
    owner.accepted_physical().values.tobytes(),
    tuple(
      (block_id, owner.accepted_state(block_id).values.tobytes())
      for block_id in owner.block_ids
    ),
  )


def test_spring_networks_coexist_with_zero_width_q8_in_one_compiled_system() -> None:
  base = _base_system()
  damage_block_entity, damage_op = compile_spring_operator(
    base,
    _damage_declaration(),
  )
  with_damage = compose_system(base, damage_block_entity, damage_op)
  slip_block_entity, slip_op = compile_spring_operator(
    with_damage,
    _slip_declaration(),
  )
  system = compose_system(with_damage, slip_block_entity, slip_op)
  q8_operator, damage_operator, slip_operator = _operators(system)
  assert system.operators[0] is base.operators[0]
  assert system.registry_snapshot is base.registry_snapshot
  assert system.content_fingerprint != base.content_fingerprint
  assert system.instance_id != base.instance_id
  assert tuple(space.space_id for space in system.spaces) == ("displacement",)
  assert system.coefficient_count == 16
  assert tuple(block.block_id for block in system.point_blocks) == (
    "nodes",
    "damage-springs",
    "slip-springs",
  )

  q8_layout = q8_operator.header.state_layout
  damage_layout = damage_operator.header.state_layout
  slip_layout = slip_operator.header.state_layout
  assert q8_layout.row_shape == (1, 0)
  assert damage_layout.row_shape == (2, 1)
  assert slip_layout.row_shape == (1, 2)
  assert damage_layout.slots[0].name == "max_extension"
  assert slip_layout.slots[0].name == "slip"
  np.testing.assert_array_equal(
    damage_layout.entity_offsets.values,
    [0, 1, 2],
  )
  np.testing.assert_array_equal(slip_layout.entity_offsets.values, [0, 2])
  np.testing.assert_array_equal(
    damage_operator.spring_block.reference_coordinates.values,
    [_UNIT_COORDINATES[0], _UNIT_COORDINATES[1]],
  )
  np.testing.assert_array_equal(
    damage_operator.header.ports[0].coefficient_map.values,
    [[0, 1], [2, 3]],
  )
  assert damage_operator.header.residual_channels[0].linear is False
  assert damage_operator.header.jacobian_channels[0].symmetric is True

  owner = StateTransactionOwner(system)
  assert owner.block_ids == (
    q8_layout.block_id,
    damage_layout.block_id,
    slip_layout.block_id,
  )
  assert owner.coefficient_count == 16
  assert owner.generation.ordinal == 0
  assert owner.history == ()

  zero_trial = np.zeros(16, dtype=np.float64)
  base_eval = base.operators[0].evaluate(
    OperatorEvaluationInput(
      port_values=(FinalizedArray(np.zeros((1, 16)), dtype=np.float64),),
      accepted_state=FinalizedArray(np.empty((1, 0)), dtype=np.float64),
      signals=(),
      request=_request("internal-force", "material-tangent"),
    )
  )
  transaction = owner.begin()
  composed_eval = _evaluate(
    q8_operator,
    transaction,
    zero_trial,
    "internal-force",
    "material-tangent",
  )
  transaction.reject()
  np.testing.assert_array_equal(
    composed_eval.jacobian_values[0].values,
    base_eval.jacobian_values[0].values,
  )

  with pytest.raises(TypeError, match="exact CompiledSystem"):
    StateTransactionOwner(base.operators[0])
  with pytest.raises(ValueError, match="duplicate state codec"):
    StateTransactionOwner(
      system,
      codecs=(
        Float64StateRowCodec("duplicate"),
        Float64StateRowCodec("duplicate"),
      ),
    )


def test_spring_compile_boundary_rejects_invalid_declarations() -> None:
  base = _base_system()
  unknown_space = damage_envelope_declaration(
    block_id="damage-springs",
    space_id="thermal",
    spring_ids=("spring-1",),
    node_ids=(1,),
    stiffness=2.0,
    critical_extension=2.0,
    max_increment=0.5,
    source=_source("damage-spring-source"),
  )
  with pytest.raises(ModelCompilationError, match="unknown-spring-space"):
    compile_spring_operator(base, unknown_space)
  unknown_node = damage_envelope_declaration(
    block_id="damage-springs",
    space_id="displacement",
    spring_ids=("spring-1",),
    node_ids=(99,),
    stiffness=2.0,
    critical_extension=2.0,
    max_increment=0.5,
    source=_source("damage-spring-source"),
  )
  with pytest.raises(ModelCompilationError, match="unknown-spring-support-node"):
    compile_spring_operator(base, unknown_node)

  def shape_breaking_kernel(
    displacements: np.ndarray,
    accepted_rows: np.ndarray,
    parameters: np.ndarray,
  ) -> SpringKernelResult:
    del accepted_rows, parameters
    return SpringKernelResult(
      force=np.zeros_like(displacements),
      tangent=np.zeros((len(displacements), 3, 3)),
      trial_rows=np.zeros((len(displacements), 1)),
      status=EvaluationStatus.OK,
    )

  broken = SpringDeclaration(
    block_id="damage-springs",
    space_id="displacement",
    spring_ids=("spring-1",),
    node_ids=(1,),
    state_schema="broken-kernel-v1",
    state_slots=(SpringStateSlot("kappa", 1),),
    kernel_name="shape-breaking-kernel",
    kernel_version="1",
    implementation_id="broken-kernel-v1",
    parameters=(1.0,),
    kernel=shape_breaking_kernel,
    source=_source("broken-kernel-source"),
  )
  with pytest.raises(TypeError, match="declared batched float64 shapes"):
    compile_spring_operator(base, broken)

  with pytest.raises(TypeError, match="paired unique spring and node id"):
    damage_envelope_declaration(
      block_id="damage-springs",
      space_id="displacement",
      spring_ids=("spring-1", "spring-1"),
      node_ids=(1, 2),
      stiffness=2.0,
      critical_extension=2.0,
      max_increment=0.5,
      source=_source("damage-spring-source"),
    )
  with pytest.raises(ValueError, match="positive finite exact float"):
    damage_envelope_declaration(
      block_id="damage-springs",
      space_id="displacement",
      spring_ids=("spring-1",),
      node_ids=(1,),
      stiffness=-2.0,
      critical_extension=2.0,
      max_increment=0.5,
      source=_source("damage-spring-source"),
    )


def test_compose_rejects_operator_compiled_against_a_foreign_system_instance() -> None:
  base = _base_system()
  block, operator = compile_spring_operator(base, _damage_declaration())

  model = _q8_model()
  shifted_model = replace(
    model,
    mesh=replace(
      model.mesh,
      nodes=(
        NodeSpec(id=0, coordinates=(2.0, 2.0), source=_source("node-source-0")),
        *model.mesh.nodes,
      ),
    ),
  )
  shifted = compile_system(shifted_model, q8_reference_registry())
  assert shifted.coefficient_count == 18
  assert shifted is not base
  with pytest.raises(IdentityMismatchError, match="spring system composition"):
    compose_system(shifted, block, operator)

  identical_content_foreign = _base_system()
  assert identical_content_foreign.content_fingerprint == base.content_fingerprint
  with pytest.raises(IdentityMismatchError, match="spring system composition"):
    compose_system(identical_content_foreign, block, operator)

  composed = compose_system(base, block, operator)
  assert composed.operators[1] is operator


def test_input_mutating_kernel_is_rejected_at_the_compile_boundary() -> None:
  def mutating_kernel(
    displacements: np.ndarray,
    accepted_rows: np.ndarray,
    parameters: np.ndarray,
  ) -> SpringKernelResult:
    del parameters
    accepted_rows[:] = 1.0
    return SpringKernelResult(
      force=np.zeros_like(displacements),
      tangent=np.zeros((len(displacements), 2, 2)),
      trial_rows=np.array(accepted_rows, copy=True),
      status=EvaluationStatus.OK,
    )

  declaration = SpringDeclaration(
    block_id="damage-springs",
    space_id="displacement",
    spring_ids=("spring-1",),
    node_ids=(1,),
    state_schema="mutating-kernel-v1",
    state_slots=(SpringStateSlot("kappa", 1),),
    kernel_name="input-mutating-kernel",
    kernel_version="1",
    implementation_id="mutating-kernel-v1",
    parameters=(1.0,),
    kernel=mutating_kernel,
    source=_source("mutating-kernel-source"),
  )
  with pytest.raises(ModelCompilationError, match="kernel-probe-failed"):
    compile_spring_operator(_base_system(), declaration)


def test_transaction_repeat_reject_accept_leaves_committed_state_identical() -> None:
  system = _full_system()
  q8_operator, damage_operator, _ = _operators(system)
  owner = StateTransactionOwner(system)
  damage_block = damage_operator.header.state_layout.block_id
  q8_block = q8_operator.header.state_layout.block_id
  before = _accepted_bytes(owner)
  trial_physical = _trial_vector(system, {1: (0.3, 0.0), 2: (0.0, 0.4)})

  first_transaction = owner.begin()
  first_damage = _evaluate(
    damage_operator,
    first_transaction,
    trial_physical,
    "spring-force",
    "spring-tangent",
  )
  first_q8 = _evaluate(
    q8_operator,
    first_transaction,
    trial_physical,
    "internal-force",
    "material-tangent",
  )
  assert evaluation_status(first_damage) is EvaluationStatus.OK
  first_transaction.stage_physical(trial_physical)
  first_transaction.stage_state(damage_block, first_damage.trial_state)
  first_transaction.stage_state(q8_block, first_q8.trial_state)
  first_transaction.reject()
  assert _accepted_bytes(owner) == before

  second_transaction = owner.begin()
  second_damage = _evaluate(
    damage_operator,
    second_transaction,
    trial_physical,
    "spring-force",
    "spring-tangent",
  )
  second_q8 = _evaluate(
    q8_operator,
    second_transaction,
    trial_physical,
    "internal-force",
    "material-tangent",
  )
  np.testing.assert_array_equal(
    second_damage.trial_state.values,
    first_damage.trial_state.values,
  )
  np.testing.assert_array_equal(
    second_damage.residual_values[0].values,
    first_damage.residual_values[0].values,
  )
  np.testing.assert_array_equal(
    second_q8.jacobian_values[0].values,
    first_q8.jacobian_values[0].values,
  )
  second_transaction.stage_physical(trial_physical)
  second_transaction.stage_state(damage_block, second_damage.trial_state)
  second_transaction.stage_state(q8_block, second_q8.trial_state)
  second_transaction.commit()

  assert owner.generation.ordinal == before[0] + 1
  np.testing.assert_array_equal(
    owner.accepted_physical().values,
    trial_physical,
  )
  np.testing.assert_array_equal(
    owner.accepted_state(damage_block).values,
    [[0.3], [0.4]],
  )
  assert owner.accepted_state(q8_block).values.shape == (1, 0)

  after_commit = _accepted_bytes(owner)
  third_transaction = owner.begin()
  third_transaction.reject()
  assert _accepted_bytes(owner) == after_commit


def test_published_accepted_snapshots_are_detached_and_immutable() -> None:
  system = _full_system()
  _, damage_operator, _ = _operators(system)
  owner = StateTransactionOwner(system)
  damage_block = damage_operator.header.state_layout.block_id

  first_transaction = owner.begin()
  published = first_transaction.accepted_state(damage_block)
  assert not published.values.flags.writeable
  assert not first_transaction.accepted_physical.values.flags.writeable
  first_transaction.stage_state(damage_block, np.array([[0.3], [0.4]]))
  first_transaction.stage_physical(_trial_vector(system, {1: (0.3, 0.0)}))
  first_transaction.commit()

  np.testing.assert_array_equal(published.values, np.zeros((2, 1)))
  second_transaction = owner.begin()
  refreshed = second_transaction.accepted_state(damage_block)
  np.testing.assert_array_equal(refreshed.values, [[0.3], [0.4]])
  assert not np.shares_memory(refreshed.values, published.values)
  second_transaction.reject()

  first_copy = owner.accepted_state(damage_block)
  second_copy = owner.accepted_state(damage_block)
  assert first_copy is not second_copy
  assert not np.shares_memory(first_copy.values, second_copy.values)


def test_typed_evaluation_outcomes_classify_expected_failures() -> None:
  system = _full_system()
  q8_operator, damage_operator, slip_operator = _operators(system)
  owner = StateTransactionOwner(system)
  damage_block = damage_operator.header.state_layout.block_id
  slip_block = slip_operator.header.state_layout.block_id

  transaction = owner.begin()
  snap_trial = _trial_vector(system, {1: (1.5, 0.0)})
  accepted_damage = transaction.accepted_state(damage_block)
  snapped = _evaluate(
    damage_operator,
    transaction,
    snap_trial,
    "spring-force",
    "spring-tangent",
  )
  assert evaluation_status(snapped) is EvaluationStatus.REJECT_STEP
  assert snapped.residual_values == ()
  assert snapped.jacobian_values == ()
  np.testing.assert_array_equal(
    snapped.trial_state.values,
    accepted_damage.values,
  )
  assert snapped.trial_state.values.tobytes() == accepted_damage.values.tobytes()

  slip_trial = _trial_vector(system, {3: (0.5, 0.0)})
  _SLIP_BUDGET["iterations"] = 2
  exhausted = _evaluate(
    slip_operator,
    transaction,
    slip_trial,
    "spring-force",
    "spring-tangent",
  )
  assert evaluation_status(exhausted) is EvaluationStatus.REJECT_ITERATION
  np.testing.assert_array_equal(
    exhausted.trial_state.values,
    transaction.accepted_state(slip_block).values,
  )
  _SLIP_BUDGET["iterations"] = 64
  retried = _evaluate(
    slip_operator,
    transaction,
    slip_trial,
    "spring-force",
    "spring-tangent",
  )
  assert evaluation_status(retried) is EvaluationStatus.OK
  transaction.reject()
  assert _accepted_bytes(owner)[0] == 0

  transaction = owner.begin()
  zero_trial = np.zeros(16, dtype=np.float64)
  q8_eval = _evaluate(
    q8_operator,
    transaction,
    zero_trial,
    "internal-force",
    "material-tangent",
  )
  assert evaluation_status(q8_eval) is EvaluationStatus.OK
  transaction.reject()
  with pytest.raises(TypeError, match="exact OperatorEvaluation"):
    evaluation_status(object())


def test_irreversible_damage_persists_across_commit() -> None:
  system = _damage_system()
  damage_operator = system.operators[1]
  assert isinstance(damage_operator, SpringOperator)
  owner = StateTransactionOwner(system)
  damage_block = damage_operator.header.state_layout.block_id

  load = _trial_vector(system, {1: (0.3, 0.0)})
  transaction = owner.begin()
  loaded = _evaluate(
    damage_operator,
    transaction,
    load,
    "spring-force",
    "spring-tangent",
  )
  np.testing.assert_allclose(
    loaded.residual_values[0].values[0],
    [0.51, 0.0],
    rtol=0.0,
    atol=1.0e-15,
  )
  np.testing.assert_allclose(
    loaded.jacobian_values[0].values[0],
    [[1.4, 0.0], [0.0, 1.7]],
    rtol=0.0,
    atol=1.0e-15,
  )
  transaction.stage_physical(load)
  transaction.stage_state(damage_block, loaded.trial_state)
  transaction.commit()

  unload = _trial_vector(system, {1: (0.1, 0.0)})
  transaction = owner.begin()
  unloaded = _evaluate(
    damage_operator,
    transaction,
    unload,
    "spring-force",
    "spring-tangent",
  )
  assert evaluation_status(unloaded) is EvaluationStatus.OK
  np.testing.assert_array_equal(unloaded.trial_state.values[0], [0.3])
  np.testing.assert_allclose(
    unloaded.residual_values[0].values[0],
    [0.17, 0.0],
    rtol=0.0,
    atol=1.0e-15,
  )
  np.testing.assert_allclose(
    unloaded.jacobian_values[0].values[0],
    [[1.7, 0.0], [0.0, 1.7]],
    rtol=0.0,
    atol=1.0e-15,
  )
  transaction.reject()

  virgin_owner = StateTransactionOwner(_damage_system())
  transaction = virgin_owner.begin()
  virgin = _evaluate(
    virgin_owner.system.operators[1],
    transaction,
    unload,
    "spring-force",
    "spring-tangent",
  )
  transaction.reject()
  # A virgin spring grows its envelope only to 0.1 here (omega = 0.05), while
  # the committed spring keeps the degraded omega = 0.15 from kappa = 0.3.
  np.testing.assert_allclose(
    virgin.residual_values[0].values[0],
    [0.19, 0.0],
    rtol=0.0,
    atol=1.0e-15,
  )


def test_state_codecs_round_trip_per_block_and_fail_closed() -> None:
  system = _full_system()
  _, damage_operator, slip_operator = _operators(system)
  owner = StateTransactionOwner(system)
  damage_block = damage_operator.header.state_layout.block_id
  slip_block = slip_operator.header.state_layout.block_id

  transaction = owner.begin()
  trial_physical = _trial_vector(system, {1: (0.3, 0.0), 3: (0.5, 0.0)})
  damaged = _evaluate(
    damage_operator,
    transaction,
    trial_physical,
    "spring-force",
    "spring-tangent",
  )
  slipped = _evaluate(
    slip_operator,
    transaction,
    trial_physical,
    "spring-force",
    "spring-tangent",
  )
  transaction.stage_physical(trial_physical)
  transaction.stage_state(damage_block, damaged.trial_state)
  transaction.stage_state(slip_block, slipped.trial_state)
  transaction.commit()

  for block_id in owner.block_ids:
    payload = owner.encode_state(block_id)
    assert STATE_ROW_CODEC_FORMAT.encode() in payload.split(b"\n", 1)[0]
    decoded = owner.decode_state(block_id, payload)
    assert decoded.values.tobytes() == owner.accepted_state(block_id).values.tobytes()
    assert not decoded.values.flags.writeable

  damage_layout = damage_operator.header.state_layout
  rows = owner.accepted_state(damage_block)
  foreign = Float64StateRowCodec("foreign-schema-v1")
  with pytest.raises(StateCodecError, match="content schema"):
    foreign.encode(damage_layout, rows)
  with pytest.raises(StateCodecError, match="content schema"):
    foreign.decode(damage_layout, owner.encode_state(damage_block))
  payload = owner.encode_state(damage_block)
  with pytest.raises(StateCodecError, match="header line"):
    owner.decode_state(damage_block, b"payload-without-a-header-line")
  with pytest.raises(StateCodecError, match="byte count"):
    owner.decode_state(damage_block, payload[:-4])
  header, _, body = payload.partition(b"\n")
  tampered = header.replace(b'"row_width":1', b'"row_width":2') + b"\n" + body
  with pytest.raises(StateCodecError, match="contradicts the compiled state layout"):
    owner.decode_state(damage_block, tampered)
  poisoned = header + b"\n" + np.array([[np.nan], [0.0]]).tobytes()
  with pytest.raises(StateCodecError, match="non-finite"):
    owner.decode_state(damage_block, poisoned)
  with pytest.raises(TypeError, match="exact bytes"):
    owner.decode_state(damage_block, "not-bytes")
  with pytest.raises(StateCodecError, match="finite"):
    owner.state_codec(damage_block).encode(
      damage_layout,
      FinalizedArray(np.array([[np.nan], [0.0]]), dtype=np.float64),
    )
  with pytest.raises(KeyError, match="no state block"):
    owner.encode_state("unknown-block")


def test_commit_is_atomic_and_unstaged_blocks_carry_over() -> None:
  system = _full_system()
  _, damage_operator, _ = _operators(system)
  owner = StateTransactionOwner(system)
  damage_block = damage_operator.header.state_layout.block_id
  generation_zero = owner.generation

  first_transaction = owner.begin()
  load = _trial_vector(system, {1: (0.3, 0.0)})
  loaded = _evaluate(
    damage_operator,
    first_transaction,
    load,
    "spring-force",
    "spring-tangent",
  )
  first_transaction.stage_physical(load)
  first_transaction.stage_state(damage_block, loaded.trial_state)
  first_transaction.commit()
  generation_one = owner.generation
  committed_rows = owner.accepted_state(damage_block).values.tobytes()

  second_transaction = owner.begin()
  follow_up = _trial_vector(system, {1: (0.35, 0.0)})
  second_transaction.stage_physical(follow_up)
  second_transaction.commit()
  assert owner.accepted_state(damage_block).values.tobytes() == committed_rows
  np.testing.assert_array_equal(owner.accepted_physical().values, follow_up)

  assert owner.generation.ordinal == 2
  require_generation_successor(generation_zero, generation_one)
  require_generation_successor(generation_one, owner.generation)
  first_record, second_record = owner.history
  assert first_record.ordinal == 1
  assert first_record.physical_staged is True
  assert damage_block in first_record.staged_block_ids
  assert second_record.ordinal == 2
  assert second_record.staged_block_ids == ()
  for record in owner.history:
    assert type(record.ordinal) is int
    assert type(record.physical_staged) is bool
    assert type(record.staged_block_ids) is tuple


def test_staging_and_lifecycle_violations_fail_closed() -> None:
  system = _damage_system()
  damage_operator = system.operators[1]
  owner = StateTransactionOwner(system)
  damage_block = damage_operator.header.state_layout.block_id
  before = _accepted_bytes(owner)

  transaction = owner.begin()
  with pytest.raises(RuntimeError, match="already open"):
    owner.begin()
  with pytest.raises(KeyError, match="no state block"):
    transaction.stage_state("unknown-block", np.zeros((2, 1)))
  with pytest.raises(ValueError, match="shape"):
    transaction.stage_state(damage_block, np.zeros((3, 1)))
  with pytest.raises(ValueError, match="finite"):
    transaction.stage_state(damage_block, np.array([[np.nan], [0.0]]))
  with pytest.raises(ValueError, match="shape"):
    transaction.stage_physical(np.zeros(15))
  with pytest.raises(TypeError, match="FinalizedArray or plain ndarray"):
    transaction.stage_physical([0.0] * 16)
  with pytest.raises(ValueError, match="at least one staged value"):
    transaction.commit()
  transaction.reject()
  assert _accepted_bytes(owner) == before

  with pytest.raises(RuntimeError, match="already consumed"):
    transaction.reject()
  with pytest.raises(RuntimeError, match="already consumed"):
    transaction.commit()
  with pytest.raises(RuntimeError, match="already consumed"):
    transaction.stage_physical(np.zeros(16))
  with pytest.raises(TypeError, match="issued only by"):
    StateTransaction(
      owner=owner,
      generation=owner.generation,
      accepted_physical=owner.accepted_physical(),
      accepted_blocks={},
    )


def test_researcher_kernel_authored_against_contract_runs_end_to_end() -> None:
  system = _full_system()
  _, _, slip_operator = _operators(system)
  owner = StateTransactionOwner(system)
  slip_block = slip_operator.header.state_layout.block_id
  before = _accepted_bytes(owner)

  load = _trial_vector(system, {3: (0.5, 0.0)})
  transaction = owner.begin()
  slipped = _evaluate(
    slip_operator,
    transaction,
    load,
    "spring-force",
    "spring-tangent",
  )
  assert evaluation_status(slipped) is EvaluationStatus.OK
  committed_slip = float(slipped.trial_state.values[0, 0])
  assert committed_slip > 0.0
  expected_force = 3.0 * (0.5 - committed_slip)
  np.testing.assert_allclose(
    slipped.residual_values[0].values[0],
    [expected_force, 0.0],
    rtol=1.0e-12,
    atol=1.0e-13,
  )
  transaction.stage_physical(load)
  transaction.stage_state(slip_block, slipped.trial_state)
  transaction.reject()
  assert _accepted_bytes(owner) == before

  transaction = owner.begin()
  slipped = _evaluate(
    slip_operator,
    transaction,
    load,
    "spring-force",
    "spring-tangent",
  )
  transaction.stage_physical(load)
  transaction.stage_state(slip_block, slipped.trial_state)
  transaction.commit()
  assert owner.generation.ordinal == 1

  unload = _trial_vector(system, {3: (0.1, 0.0)})
  transaction = owner.begin()
  unloaded = _evaluate(
    slip_operator,
    transaction,
    unload,
    "spring-force",
    "spring-tangent",
  )
  transaction.reject()
  predictor = 0.1 - committed_slip
  np.testing.assert_allclose(
    unloaded.residual_values[0].values[0],
    [3.0 * predictor, 0.0],
    rtol=1.0e-12,
    atol=1.0e-13,
  )
  np.testing.assert_array_equal(
    unloaded.trial_state.values,
    [[committed_slip, 0.0]],
  )

  payload = owner.encode_state(slip_block)
  decoded = owner.decode_state(slip_block, payload)
  assert decoded.values.tobytes() == owner.accepted_state(slip_block).values.tobytes()


def test_trusted_carriers_and_typed_evaluations_resist_reconstruction() -> None:
  system = _damage_system()
  damage_operator = system.operators[1]
  owner = StateTransactionOwner(system)
  transaction = owner.begin()
  evaluation = _evaluate(
    damage_operator,
    transaction,
    _trial_vector(system, {1: (0.1, 0.0)}),
    "spring-force",
    "spring-tangent",
  )
  transaction.reject()
  trusted = (
    system,
    damage_operator,
    damage_operator.payload,
    damage_operator.header,
    damage_operator.header.state_layout,
    damage_operator.header.state_layout.slots[0],
    damage_operator.header.ports[0],
    damage_operator.header.implementations[0],
    damage_operator.spring_block,
    evaluation,
  )
  for value in trusted:
    with pytest.raises(TypeError, match="cannot be reconstructed"):
      copy.copy(value)
    with pytest.raises(TypeError, match="cannot be reconstructed"):
      copy.deepcopy(value)
    with pytest.raises(TypeError, match="cannot be reconstructed"):
      pickle.loads(pickle.dumps(value))
  for carrier in (SpringOperator, type(damage_operator.header)):
    with pytest.raises(TypeError, match="constructed only by their compiler"):
      carrier()
  assert evaluation_status(evaluation) is EvaluationStatus.OK


def _plasticity_system() -> CompiledSystem:
  """One Q8 element bound to the first stateful law."""
  nodes = tuple(
    NodeSpec(
      id=index + 1,
      coordinates=point,
      source=_source(f"plastic-node-{index + 1}"),
    )
    for index, point in enumerate(_UNIT_COORDINATES)
  )
  cell = CellSpec(
    id="cell-1",
    node_ids=tuple(node.id for node in nodes),
    source=_source("plastic-cell"),
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
          source=_source("plastic-block"),
        ),
      ),
      source=_source("plastic-mesh"),
    ),
    fields=(
      FieldSpec(
        id="displacement",
        components=("x", "y"),
        location="node",
        source=_source("plastic-field"),
      ),
    ),
    materials=(
      MaterialSpec(
        id="steel",
        model="isotropic-hardening-plasticity",
        parameters=(
          MaterialParameterSpec("youngs_modulus", 210000.0),
          MaterialParameterSpec("poisson_ratio", 0.3),
          MaterialParameterSpec("initial_yield_stress", 250.0),
          MaterialParameterSpec("hardening_slope", 1000.0),
        ),
        source=_source("plastic-material"),
      ),
    ),
    regions=(
      RegionSpec(
        id="domain",
        cell_refs=(CellRef("cells", "cell-1"),),
        field_ids=("displacement",),
        material_id="steel",
        formulation="small-strain-continuum",
        quadrature="gauss-3x3",
        source=_source("plastic-region"),
      ),
    ),
    source=_source("plastic-model"),
  )
  return compile_system(model, plasticity_reference_registry())


def test_owner_seeds_accepted_rows_from_compiler_emitted_initial_rows() -> None:
  # Landed layouts carry no initial rows: the owner keeps zero-initializing.
  system = _damage_system()
  owner = StateTransactionOwner(system)
  for operator in system.operators:
    layout = operator.header.state_layout
    assert getattr(layout, "initial_rows", None) is None
    assert np.all(owner.accepted_state(layout.block_id).values == 0.0)

  # The first stateful law binds explicit (zero) initial rows and the owner
  # applies exactly those rows at construction (F3 G2).
  plastic = _plasticity_system()
  layout = plastic.operators[0].header.state_layout
  assert layout.initial_rows is not None
  assert layout.row_shape == (9, 19)
  owner = StateTransactionOwner(plastic)
  np.testing.assert_array_equal(
    owner.accepted_state(layout.block_id).values,
    layout.initial_rows.values,
  )
  # The emitted entity offsets stride one full row per integration point.
  np.testing.assert_array_equal(
    layout.entity_offsets.values,
    np.arange(10) * 19,
  )


def test_owner_rejects_malformed_initial_rows_and_annotations() -> None:
  system = _plasticity_system()
  layout = system.operators[0].header.state_layout
  object.__setattr__(layout, "initial_rows", np.zeros((9, 19)))
  with pytest.raises(TypeError, match="initial rows must be an exact FinalizedArray"):
    StateTransactionOwner(system)

  system = _plasticity_system()
  layout = system.operators[0].header.state_layout
  object.__setattr__(
    layout,
    "initial_rows",
    FinalizedArray(np.zeros((9, 18)), dtype=np.float64),
  )
  with pytest.raises(ValueError, match="initial rows must match the layout row shape"):
    StateTransactionOwner(system)

  system = _plasticity_system()
  layout = system.operators[0].header.state_layout
  object.__setattr__(
    layout,
    "initial_rows",
    FinalizedArray(np.full((9, 19), np.inf), dtype=np.float64),
  )
  with pytest.raises(ValueError, match="initial rows must be finite"):
    StateTransactionOwner(system)

  system = _plasticity_system()
  slot = system.operators[0].header.state_layout.slots[0]
  object.__setattr__(slot, "annotation", 3)
  with pytest.raises(
    TypeError, match="slot annotations must be non-empty exact strings"
  ):
    StateTransactionOwner(system)
