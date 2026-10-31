# SPDX-License-Identifier: MIT

"""Nested-state store, reference-slot, and macro-owner isolation battery.

Pins the FE² nested-state architecture contracts: the nested store is
addressable per macro integration point and evolves on per-point accepted
generations independently of the macro transaction generation; macro
transactions stage typed nested-state references (never payload copies);
trial commits advance exactly the completed points and rejects leave every
accepted buffer byte-identical; and neither side of the macro/nested
boundary can mutate the other's state.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from pyfem.v3.compile.continuum import q8_reference_registry
from pyfem.v3.compile.system import compile_system
from pyfem.v3.model.identity import (
  GenerationMismatchError,
  IdentityMismatchError,
  InstanceId,
  StateGeneration,
  require_same_generation,
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
  NestedBlockLayout,
  NestedStateLayout,
  NestedStateReference,
  NestedStateStore,
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


def _system() -> CompiledSystem:
  return compile_system(_q8_model(), q8_reference_registry())


def _micro_layout(
  ip_count: int = 3, schema: str = "test-nested-micro-v1"
) -> NestedStateLayout:
  return NestedStateLayout(
    schema=schema,
    coefficient_count=18,
    blocks=(
      NestedBlockLayout(
        schema="test-damage-v1",
        block_id="micro-cells",
        entity_count=16,
        row_width=1,
      ),
      NestedBlockLayout(
        schema="test-slip-v1",
        block_id=7,
        entity_count=4,
        row_width=2,
      ),
    ),
    ip_count=ip_count,
  )


def _child_layout() -> NestedStateLayout:
  return NestedStateLayout(
    schema="test-nested-child-v1",
    coefficient_count=4,
    blocks=(
      NestedBlockLayout(
        schema="test-child-damage-v1",
        block_id="child-cells",
        entity_count=6,
        row_width=2,
      ),
    ),
    ip_count=2,
  )


def _macro_layout() -> NestedStateLayout:
  child = _child_layout()
  return NestedStateLayout(
    schema="test-nested-macro-v1",
    coefficient_count=3,
    blocks=(
      NestedBlockLayout(
        schema="test-macro-slip-v1",
        block_id="slip-springs",
        entity_count=2,
        row_width=2,
      ),
    ),
    ip_count=3,
    children=(child, None, child),
  )


def _macro_store() -> NestedStateStore:
  layout = _macro_layout()
  child = layout.children[0]
  return NestedStateStore(
    layout,
    children=(NestedStateStore(child), None, NestedStateStore(child)),
  )


def _drive_micro_point(
  store: NestedStateStore,
  ip_index: int,
  *,
  base: float,
) -> None:
  store.write_trial_coefficients(
    ip_index,
    np.full(18, base, dtype=np.float64),
  )
  store.write_trial_block(
    ip_index,
    "micro-cells",
    np.full((16, 1), base + 0.5, dtype=np.float64),
  )
  store.write_trial_block(
    ip_index,
    7,
    np.full((4, 2), base + 0.25, dtype=np.float64),
  )
  store.trial_complete(ip_index)


def test_block_layout_validation_is_exact_and_fail_closed() -> None:
  with pytest.raises(ValueError, match="non-empty exact string"):
    NestedBlockLayout(schema="", block_id="b", entity_count=1, row_width=1)
  with pytest.raises(TypeError, match="exact string or integer"):
    NestedBlockLayout(schema="s", block_id=1.5, entity_count=1, row_width=1)
  with pytest.raises(ValueError, match="block id must be non-empty"):
    NestedBlockLayout(schema="s", block_id="", entity_count=1, row_width=1)
  with pytest.raises(ValueError, match="entity count"):
    NestedBlockLayout(schema="s", block_id="b", entity_count=-1, row_width=1)
  with pytest.raises(ValueError, match="entity count"):
    NestedBlockLayout(schema="s", block_id="b", entity_count=True, row_width=1)
  with pytest.raises(ValueError, match="row width"):
    NestedBlockLayout(schema="s", block_id="b", entity_count=1, row_width=-2)


def test_state_layout_validation_and_child_normalization() -> None:
  with pytest.raises(ValueError, match="non-empty exact string"):
    NestedStateLayout(schema="", coefficient_count=1, blocks=(), ip_count=1)
  with pytest.raises(ValueError, match="coefficient count"):
    NestedStateLayout(schema="s", coefficient_count=-1, blocks=(), ip_count=1)
  with pytest.raises(ValueError, match="point count"):
    NestedStateLayout(schema="s", coefficient_count=1, blocks=(), ip_count=-2)
  with pytest.raises(TypeError, match="exact NestedBlockLayout values"):
    NestedStateLayout(schema="s", coefficient_count=1, blocks=("x",), ip_count=1)
  duplicate = NestedBlockLayout(schema="s", block_id="b", entity_count=1, row_width=1)
  with pytest.raises(ValueError, match="unique within one level"):
    NestedStateLayout(
      schema="s",
      coefficient_count=1,
      blocks=(duplicate, duplicate),
      ip_count=1,
    )
  with pytest.raises(TypeError, match="length ip_count"):
    NestedStateLayout(
      schema="s",
      coefficient_count=1,
      blocks=(),
      ip_count=2,
      children=(None,),
    )
  with pytest.raises(TypeError, match="exact NestedStateLayout values or None"):
    NestedStateLayout(
      schema="s",
      coefficient_count=1,
      blocks=(),
      ip_count=1,
      children=("foreign",),
    )
  flat = NestedStateLayout(schema="s", coefficient_count=1, blocks=(), ip_count=4)
  assert flat.children == (None, None, None, None)


def test_reference_slots_are_exact_typed_metadata() -> None:
  generation = StateGeneration.initial()
  reference = NestedStateReference(
    store=InstanceId(), ip_index=2, generation=generation
  )
  assert reference.ip_index == 2
  assert reference.generation is generation
  with pytest.raises(TypeError, match="exact InstanceId"):
    NestedStateReference(store="store", ip_index=0, generation=generation)
  with pytest.raises(ValueError, match="non-negative exact int"):
    NestedStateReference(store=InstanceId(), ip_index=-1, generation=generation)
  with pytest.raises(TypeError, match="exact StateGeneration"):
    NestedStateReference(store=InstanceId(), ip_index=0, generation=object())


def test_store_construction_validates_child_bindings() -> None:
  layout = _macro_layout()
  with pytest.raises(TypeError, match="exact NestedStateLayout"):
    NestedStateStore("layout")
  with pytest.raises(ValueError, match="requires child stores"):
    NestedStateStore(layout)
  with pytest.raises(TypeError, match="length ip_count"):
    NestedStateStore(layout, children=(None, None))
  with pytest.raises(ValueError, match="where the layout declares none"):
    NestedStateStore(
      layout,
      children=(
        NestedStateStore(layout.children[0]),
        NestedStateStore(_child_layout()),
        NestedStateStore(layout.children[2]),
      ),
    )
  foreign_child = NestedStateLayout(
    schema="test-nested-child-v2",
    coefficient_count=4,
    blocks=(),
    ip_count=2,
  )
  with pytest.raises(ValueError, match="must equal the declared child layout"):
    NestedStateStore(
      layout,
      children=(
        NestedStateStore(foreign_child),
        None,
        NestedStateStore(layout.children[2]),
      ),
    )
  shared = NestedStateStore(layout.children[0])
  with pytest.raises(ValueError, match="distinct store instances"):
    NestedStateStore(layout, children=(shared, None, shared))


def test_virgin_store_is_zeroed_at_generation_zero() -> None:
  store = NestedStateStore(_micro_layout())
  assert store.ip_count == 3
  assert store.block_ids == ("micro-cells", 7)
  assert store.children == (None, None, None)
  assert not store.trial_open
  for index in range(3):
    assert store.generation(index).ordinal == 0
    np.testing.assert_array_equal(store.accepted_coefficients(index), np.zeros(18))
    np.testing.assert_array_equal(
      store.accepted_block(index, "micro-cells"), np.zeros((16, 1))
    )


def test_trial_operations_require_an_open_scope() -> None:
  store = NestedStateStore(_micro_layout())
  rows = np.zeros((16, 1))
  with pytest.raises(RuntimeError, match="open trial scope"):
    store.trial_coefficients(0)
  with pytest.raises(RuntimeError, match="open trial scope"):
    store.trial_block(0, "micro-cells")
  with pytest.raises(RuntimeError, match="open trial scope"):
    store.write_trial_coefficients(0, np.zeros(18))
  with pytest.raises(RuntimeError, match="open trial scope"):
    store.write_trial_block(0, "micro-cells", rows)
  with pytest.raises(RuntimeError, match="open trial scope"):
    store.trial_complete(0)
  with pytest.raises(RuntimeError, match="open trial scope"):
    store.trial_reference(0)
  with pytest.raises(RuntimeError, match="open trial scope"):
    store.commit_trial()
  with pytest.raises(RuntimeError, match="open trial scope"):
    store.reject_trial()
  store.begin_trial()
  with pytest.raises(RuntimeError, match="already open"):
    store.begin_trial()
  store.reject_trial()


def test_trial_staging_validation_is_fail_closed() -> None:
  store = NestedStateStore(_micro_layout())
  before = store.encode()
  store.begin_trial()
  with pytest.raises(ValueError, match="lies outside the store"):
    store.trial_coefficients(3)
  with pytest.raises(ValueError, match="lies outside the store"):
    store.write_trial_block(True, "micro-cells", np.zeros((16, 1)))
  with pytest.raises(KeyError, match="no state block"):
    store.trial_block(0, "unknown-block")
  with pytest.raises(ValueError, match="shape"):
    store.write_trial_coefficients(0, np.zeros(17))
  with pytest.raises(ValueError, match="shape"):
    store.write_trial_block(0, "micro-cells", np.zeros((15, 1)))
  with pytest.raises(ValueError, match="finite"):
    store.write_trial_coefficients(0, np.full(18, np.nan))
  with pytest.raises(TypeError, match="FinalizedArray or plain ndarray"):
    store.write_trial_coefficients(0, [0.0] * 18)
  with pytest.raises(RuntimeError, match="trial-written integration point"):
    store.trial_complete(1)
  with pytest.raises(RuntimeError, match="completed integration point trial"):
    store.trial_reference(0)
  store.reject_trial()
  assert store.encode() == before


def test_commit_advances_exactly_the_completed_points() -> None:
  store = NestedStateStore(_micro_layout())
  rooted = store.generations
  store.begin_trial()
  assert store.trial_open
  _drive_micro_point(store, 0, base=1.0)
  _drive_micro_point(store, 2, base=2.0)
  references = [store.trial_reference(0), store.trial_reference(2)]
  store.commit_trial()
  assert not store.trial_open
  assert store.generation(0).ordinal == 1
  assert store.generation(1).ordinal == 0
  assert store.generation(2).ordinal == 1
  require_same_generation(rooted[0].next_accepted(), store.generation(0))
  require_same_generation(rooted[1], store.generation(1))
  np.testing.assert_array_equal(store.accepted_coefficients(0), np.ones(18))
  np.testing.assert_array_equal(store.accepted_block(2, 7), np.full((4, 2), 2.25))
  np.testing.assert_array_equal(store.accepted_coefficients(1), np.zeros(18))
  for reference, index in zip(references, (0, 2), strict=True):
    assert reference.store == store.instance
    assert reference.ip_index == index
    require_same_generation(reference.generation, rooted[index])


def test_reject_leaves_accepted_state_and_generations_byte_identical() -> None:
  store = NestedStateStore(_micro_layout())
  store.begin_trial()
  _drive_micro_point(store, 1, base=3.0)
  store.commit_trial()
  before = store.encode()
  for _ in range(3):
    store.begin_trial()
    _drive_micro_point(store, 0, base=9.0)
    _drive_micro_point(store, 1, base=8.0)
    _drive_micro_point(store, 2, base=7.0)
    store.reject_trial()
    assert store.encode() == before
    assert not store.trial_open
  assert [generation.ordinal for generation in store.generations] == [0, 1, 0]


def test_incomplete_or_non_finite_trials_cannot_commit() -> None:
  store = NestedStateStore(_micro_layout())
  before = store.encode()
  store.begin_trial()
  _drive_micro_point(store, 0, base=1.0)
  store.write_trial_coefficients(1, np.ones(18))
  with pytest.raises(RuntimeError, match="completed trial at point 1"):
    store.commit_trial()
  assert store.trial_open
  assert store.encode() == before
  store.trial_coefficients(1)[:] = np.ones(18)
  store.trial_block(1, "micro-cells")[:] = 0.5
  store.trial_block(1, 7)[:] = np.nan
  store.trial_complete(1)
  with pytest.raises(ValueError, match="must be finite to commit"):
    store.commit_trial()
  assert store.trial_open
  assert store.encode() == before
  store.reject_trial()
  assert store.encode() == before


def test_accepted_views_stay_byte_stable_across_trial_writes_and_reject() -> None:
  store = NestedStateStore(_micro_layout())
  store.begin_trial()
  _drive_micro_point(store, 2, base=4.0)
  store.commit_trial()
  borrowed = store.accepted_coefficients(2)
  borrowed_block = store.accepted_block(2, "micro-cells")
  assert not borrowed.flags.writeable
  coefficients_before = borrowed.tobytes()
  block_before = borrowed_block.tobytes()
  store.begin_trial()
  _drive_micro_point(store, 2, base=8.0)
  _drive_micro_point(store, 0, base=1.0)
  # Trial writes stage into separate buffers: borrowed accepted views never
  # observe trial state.
  assert borrowed.tobytes() == coefficients_before
  assert borrowed_block.tobytes() == block_before
  store.reject_trial()
  assert borrowed.tobytes() == coefficients_before
  store.begin_trial()
  _drive_micro_point(store, 2, base=6.0)
  store.commit_trial()
  # Borrowed views are valid until the next commit; fresh views publish the
  # newly committed state.
  np.testing.assert_array_equal(store.accepted_coefficients(2), np.full(18, 6.0))
  np.testing.assert_array_equal(store.accepted_coefficients(0), np.zeros(18))


def test_trial_workspace_views_are_reused_across_scopes() -> None:
  store = NestedStateStore(_micro_layout())
  store.begin_trial()
  workspace = store.trial_coefficients(0)
  block_view = store.trial_block(0, "micro-cells")
  workspace[:] = 1.0
  block_view[:] = 2.0
  store.trial_complete(0)
  store.commit_trial()
  store.begin_trial()
  # Preallocated staged buffers are reused: no per-scope allocation.
  assert store.trial_coefficients(0).base is workspace.base
  assert store.trial_block(0, "micro-cells").base is block_view.base
  store.reject_trial()


def test_macro_trial_scope_cascades_across_levels() -> None:
  store = _macro_store()
  child_zero = store.children[0]
  child_two = store.children[2]
  store.begin_trial()
  assert child_zero.trial_open
  assert child_two.trial_open
  child_zero.write_trial_coefficients(0, np.arange(4, dtype=np.float64))
  child_zero.write_trial_block(0, "child-cells", np.ones((6, 2)))
  child_zero.trial_complete(0)
  store.write_trial_coefficients(2, np.array([1.0, 2.0, 3.0]))
  store.write_trial_block(2, "slip-springs", np.ones((2, 2)))
  store.trial_complete(2)
  store.commit_trial()
  assert [generation.ordinal for generation in store.generations] == [0, 0, 1]
  assert [generation.ordinal for generation in child_zero.generations] == [1, 0]
  assert [generation.ordinal for generation in child_two.generations] == [0, 0]
  before = store.encode()
  store.begin_trial()
  child_zero.write_trial_coefficients(1, np.full(4, 5.0))
  store.write_trial_coefficients(0, np.zeros(3))
  store.reject_trial()
  assert store.encode() == before
  assert not child_zero.trial_open


def test_child_scope_closed_out_from_under_the_macro_trial_fails_closed() -> None:
  store = _macro_store()
  child_zero = store.children[0]
  before = store.encode()
  store.begin_trial()
  child_zero.reject_trial()
  with pytest.raises(RuntimeError, match="closed out from under"):
    store.commit_trial()
  assert store.trial_open
  with pytest.raises(RuntimeError, match="closed out from under"):
    store.reject_trial()
  assert not store.trial_open
  assert store.encode() == before


def test_nested_commit_never_touches_the_macro_owner() -> None:
  system = _system()
  owner = StateTransactionOwner(system)
  store = NestedStateStore(_micro_layout())
  physical_before = owner.accepted_physical().values.tobytes()
  store.begin_trial()
  _drive_micro_point(store, 1, base=2.0)
  store.commit_trial()
  assert owner.accepted_physical().values.tobytes() == physical_before
  assert owner.generation.ordinal == 0
  assert owner.history == ()


def test_macro_owner_stages_typed_nested_references() -> None:
  system = _system()
  store = NestedStateStore(_micro_layout())
  owner = StateTransactionOwner(system, nested_stores=(store,))
  assert owner.nested_stores == (store,)
  transaction = owner.begin()
  store.begin_trial()
  _drive_micro_point(store, 0, base=1.0)
  _drive_micro_point(store, 2, base=2.0)
  references = (store.trial_reference(0), store.trial_reference(2))
  for reference in references:
    transaction.stage_nested(reference)
  assert transaction.nested_references == references
  transaction.stage_physical(np.zeros(system.coefficient_count))
  transaction.commit()
  store.commit_trial()
  record = owner.history[-1]
  assert record.nested_references == references
  assert type(record.nested_references) is tuple
  assert owner.generation.ordinal == 1


def test_macro_commit_and_reject_never_touch_nested_state() -> None:
  system = _system()
  store = NestedStateStore(_micro_layout())
  owner = StateTransactionOwner(system, nested_stores=(store,))
  store.begin_trial()
  _drive_micro_point(store, 1, base=1.5)
  store.commit_trial()
  nested_before = store.encode()
  transaction = owner.begin()
  transaction.stage_physical(np.ones(system.coefficient_count))
  store.begin_trial()
  _drive_micro_point(store, 1, base=4.0)
  transaction.stage_nested(store.trial_reference(1))
  transaction.commit()
  # The macro commit records the reference; nested state moves only when the
  # driver commits the nested trial.
  assert store.generation(1).ordinal == 1
  assert np.all(store.accepted_coefficients(1) == 1.5)
  store.commit_trial()
  assert store.generation(1).ordinal == 2
  nested_after = store.encode()
  transaction = owner.begin()
  transaction.stage_physical(np.ones(system.coefficient_count))
  store.begin_trial()
  _drive_micro_point(store, 0, base=9.0)
  transaction.stage_nested(store.trial_reference(0))
  transaction.reject()
  store.reject_trial()
  assert store.encode() == nested_after
  assert nested_after != nested_before


def test_stage_nested_is_fail_closed() -> None:
  system = _system()
  store = NestedStateStore(_micro_layout())
  foreign = NestedStateStore(_micro_layout())
  owner = StateTransactionOwner(system, nested_stores=(store,))
  transaction = owner.begin()
  generation = StateGeneration.initial()
  with pytest.raises(TypeError, match="exact NestedStateReference"):
    transaction.stage_nested("reference")
  with pytest.raises(IdentityMismatchError, match="bound to this transaction owner"):
    transaction.stage_nested(
      NestedStateReference(store=foreign.instance, ip_index=0, generation=generation)
    )
  with pytest.raises(ValueError, match="lies outside the bound nested store"):
    transaction.stage_nested(
      NestedStateReference(store=store.instance, ip_index=3, generation=generation)
    )
  transaction.stage_nested(
    NestedStateReference(store=store.instance, ip_index=1, generation=generation)
  )
  with pytest.raises(ValueError, match="duplicate nested-state reference"):
    transaction.stage_nested(
      NestedStateReference(store=store.instance, ip_index=1, generation=generation)
    )
  transaction.reject()
  assert transaction.nested_references == ()
  with pytest.raises(RuntimeError, match="already consumed"):
    transaction.stage_nested(
      NestedStateReference(store=store.instance, ip_index=0, generation=generation)
    )
  assert owner.history == ()


def test_nested_references_alone_do_not_make_a_macro_commit() -> None:
  system = _system()
  store = NestedStateStore(_micro_layout())
  owner = StateTransactionOwner(system, nested_stores=(store,))
  transaction = owner.begin()
  transaction.stage_nested(
    NestedStateReference(
      store=store.instance,
      ip_index=0,
      generation=StateGeneration.initial(),
    )
  )
  with pytest.raises(ValueError, match="at least one staged value"):
    transaction.commit()
  transaction.reject()
  assert owner.history == ()


def test_owner_nested_store_binding_validation() -> None:
  system = _system()
  store = NestedStateStore(_micro_layout())
  with pytest.raises(TypeError, match="exact NestedStateStore tuple"):
    StateTransactionOwner(system, nested_stores=[store])
  with pytest.raises(TypeError, match="exact NestedStateStore tuple"):
    StateTransactionOwner(system, nested_stores=(store, "foreign"))
  with pytest.raises(ValueError, match="distinct store instances"):
    StateTransactionOwner(system, nested_stores=(store, store))


def test_decode_gates_payload_generations_against_the_live_lineage() -> None:
  store = NestedStateStore(_micro_layout())
  store.begin_trial()
  _drive_micro_point(store, 0, base=2.0)
  store.commit_trial()
  payload = store.encode()
  restored = NestedStateStore.decode(store.layout, store.generations, payload)
  assert restored.encode() == payload
  require_same_generation(restored.generation(0), store.generation(0))
  shifted = list(store.generations)
  shifted[0] = shifted[0].next_accepted()
  with pytest.raises(GenerationMismatchError, match="does not match the live lineage"):
    NestedStateStore.decode(store.layout, tuple(shifted), payload)
  with pytest.raises(TypeError, match="one exact live generation per point"):
    NestedStateStore.decode(store.layout, store.generations[:2], payload)
  with pytest.raises(TypeError, match="exact tuple"):
    NestedStateStore.decode(store.layout, list(store.generations), payload)
