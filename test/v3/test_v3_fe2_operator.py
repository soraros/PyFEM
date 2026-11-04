# SPDX-License-Identifier: MIT

"""FE² continuum operator numerical, transaction, and cross-level failure battery."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from dataclasses import astuple, dataclass, replace

import numpy as np
import pytest
from numba import get_num_threads, set_num_threads

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from pyfem.v3 import authoring
from pyfem.v3.authoring.rve import PeriodicCell
from pyfem.v3.compile.continuum import damage_reference_registry, q8_reference_registry
from pyfem.v3.compile.fe2 import FE2_TANGENT_CLASS, FE2ContinuumOperator
from pyfem.v3.compile.system import compile_system
from pyfem.v3.constraints import compile_constraint_map
from pyfem.v3.driver import (
  ArcLengthSettings,
  DriverStatus,
  NonlinearStaticDriver,
  NonlinearStaticSettings,
  PreparedNestedExecution,
  RiksDriver,
  StepFailureClass,
)
from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.operator import (
  ChannelRequest,
  EvaluationStatus,
  NestedEvaluationInput,
  NestedExecutionDeclaration,
  NestedExecutionModality,
  NestedExecutionRecipe,
  OperatorEvaluation,
  OperatorEvaluationInput,
  evaluation_nested_references,
  evaluation_status,
  operator_nested_execution,
  operator_tangent_class,
)
from pyfem.v3.model.system import CompiledSystem
from pyfem.v3.spec.program import (
  AffineCoefficientSpec,
  AffineValueSpec,
  DofRef,
  PrescribedDofSpec,
  ProgramCoordinateSpec,
  ProgramCoordinateValue,
  ProgramPoint,
)
from pyfem.v3.state import StateTransactionOwner

_E, _NU, _KAPPA0, _K = 1000.0, 0.25, 7.5e-4, 1.0
_STRAIN_NAMES = ("eps11", "eps22", "gamma12")


@pytest.fixture
def _one_numba_thread() -> Iterator[None]:
  previous = get_num_threads()
  set_num_threads(1)
  try:
    yield
  finally:
    set_num_threads(previous)


@dataclass(frozen=True, slots=True)
class _Deck:
  system: CompiledSystem
  micro_system: CompiledSystem
  micro_map: object
  cell: PeriodicCell
  macro_mesh: object
  micro_mesh: object
  runtime: PreparedNestedExecution


@dataclass(frozen=True, slots=True)
class _Cantilever:
  deck: _Deck
  driver: NonlinearStaticDriver
  coordinate_map: object


def _new[ValueT](cls: type[ValueT], /, **fields: object) -> ValueT:
  value = object.__new__(cls)
  for name, field in fields.items():
    object.__setattr__(value, name, field)
  return value


def _point(strain: np.ndarray | tuple[float, float, float]) -> ProgramPoint:
  return ProgramPoint(
    tuple(
      ProgramCoordinateValue(name, float(value))
      for name, value in zip(_STRAIN_NAMES, strain, strict=True)
    )
  )


def _load_point(value: float) -> ProgramPoint:
  return ProgramPoint((ProgramCoordinateValue("load", float(value)),))


def _deck(
  *,
  kappa_c: float = 2.5e-2,
  elastic: bool = False,
  micro_settings: NonlinearStaticSettings | None = None,
  micro_loads: tuple = (),
) -> _Deck:
  micro_mesh = authoring.quad8_patch(2, 2, width=1.0, height=1.0)
  material = (
    authoring.linear_elastic(_E, _NU)
    if elastic
    else authoring.damage(_E, _NU, _KAPPA0, kappa_c, _K)
  )
  registry = q8_reference_registry() if elastic else damage_reference_registry()
  micro_system = compile_system(
    authoring.small_strain_continuum(micro_mesh, material=material), registry
  )
  cell = authoring.periodic_cell(micro_mesh, width=1.0, height=1.0)
  micro_map = compile_constraint_map(
    micro_system, constraints=cell.constraints, coordinates=cell.coordinates
  )
  macro_mesh = authoring.quad8_patch(1, 1, width=1.0, height=1.0)
  base = compile_system(
    authoring.small_strain_continuum(
      macro_mesh, material=authoring.linear_elastic(_E, _NU)
    ),
    q8_reference_registry(),
  )
  if micro_settings is None:
    micro_settings = NonlinearStaticSettings(nested_ip_thread_count=get_num_threads())
  arguments = {
    "micro_system": micro_system,
    "micro_coordinate_map": micro_map,
    "homogenization": cell.homogenization_request(micro_system),
    "micro_loads": micro_loads,
    "settings": micro_settings,
  }
  system = authoring.fe2_continuum(base, **arguments)
  declaration = operator_nested_execution(system.operators[0].header)
  assert declaration is not None
  runtime = PreparedNestedExecution(declaration.recipe)
  return _Deck(system, micro_system, micro_map, cell, macro_mesh, micro_mesh, runtime)


def _cantilever(
  *,
  prescribed: float = 5.0e-4,
  kappa_c: float = 2.5e-2,
  elastic: bool = False,
  micro_settings: NonlinearStaticSettings | None = None,
  macro_settings: NonlinearStaticSettings | None = None,
) -> _Cantilever:
  deck = _deck(kappa_c=kappa_c, elastic=elastic, micro_settings=micro_settings)
  mesh = deck.macro_mesh
  left = tuple(node.id for node in mesh.nodes if abs(node.coordinates[0]) < 1.0e-12)
  right = tuple(
    node.id for node in mesh.nodes if abs(node.coordinates[0] - 1.0) < 1.0e-12
  )
  constraints = list(authoring.fixed(left, components=("x", "y")))
  for node in right:
    constraints.append(
      PrescribedDofSpec(
        id=f"right-{node}",
        target=DofRef(node, "displacement", "x"),
        value=AffineValueSpec(
          coefficients=(AffineCoefficientSpec("load", prescribed),)
        ),
      )
    )
  coordinate_map = compile_constraint_map(
    deck.system,
    constraints=tuple(constraints),
    coordinates=(ProgramCoordinateSpec("load", "load"),),
  )
  driver = NonlinearStaticDriver(
    deck.system,
    coordinate_map,
    settings=macro_settings or NonlinearStaticSettings(tolerance=1.0e-9),
  )
  deck.runtime.close()
  runtime = driver.nested_executions[0]
  assert runtime is not None
  return _Cantilever(replace(deck, runtime=runtime), driver, coordinate_map)


def _evaluate_port(
  deck: _Deck,
  values: np.ndarray,
  accepted: FinalizedArray,
  *,
  runtime: PreparedNestedExecution | None = None,
) -> OperatorEvaluation:
  operator = deck.system.operators[0]
  runtime = deck.runtime if runtime is None else runtime
  store = runtime.store
  runtime.begin_trial()
  try:
    return operator.evaluate(
      OperatorEvaluationInput(
        port_values=(FinalizedArray(values, dtype=np.float64),),
        accepted_state=accepted,
        signals=(),
        request=ChannelRequest(("internal-force",), ("material-tangent",)),
        nested_execution=NestedEvaluationInput(runtime),
      )
    )
  finally:
    if store.trial_open:
      store.reject_trial()


def _seed_documented_ip_state(deck: _Deck, strain: np.ndarray) -> None:
  """Seed the measured converged IP displacement field with virgin material history."""
  store = deck.runtime.store
  displacements = {
    node.id: (
      float(strain[0] * node.coordinates[0]),
      float(strain[1] * node.coordinates[1]),
    )
    for node in deck.micro_mesh.nodes
  }
  coefficients = authoring.trial_vector(deck.micro_system, displacements)
  store.begin_trial()
  try:
    for ip_index in range(store.ip_count):
      store.write_trial_coefficients(ip_index, coefficients)
      for micro_operator, block in zip(
        deck.micro_system.operators, store.layout.blocks, strict=True
      ):
        layout = micro_operator.header.state_layout
        initial = getattr(layout, "initial_rows", None)
        rows = (
          np.zeros(layout.row_shape, dtype=np.float64)
          if initial is None
          else initial.values
        )
        store.write_trial_block(ip_index, block.block_id, rows)
      store.trial_complete(ip_index)
    store.commit_trial()
  finally:
    if store.trial_open:
      store.reject_trial()


def _stress_from_reactions(deck: _Deck, ip_index: int = 0) -> np.ndarray:
  point = _point(_documented_strain())
  result = deck.runtime.run_point(ip_index, base_point=point, target_points=(point,))
  assert result.status is DriverStatus.COMPLETED
  reactions = result.records[-1].observation.reactions.values
  request = deck.cell.homogenization_request(deck.micro_system)
  return np.array(
    [
      reduction.scale * float(reactions[list(reduction.dofs)].sum())
      for reduction in request.reductions
    ]
  )


def _documented_strain() -> np.ndarray:
  return np.array([2.0e-3, -2.0e-3 / 3.0, 0.0])


def test_header_declares_an_immutable_recipe_not_live_execution_state() -> None:
  deck = _deck(elastic=True)
  operator = deck.system.operators[0]
  declaration = operator_nested_execution(operator.header)
  assert type(declaration) is NestedExecutionDeclaration
  assert declaration.modality is NestedExecutionModality.DRIVER_PER_INTEGRATION_POINT
  recipe = declaration.recipe
  assert type(recipe) is NestedExecutionRecipe
  assert recipe.micro_system is deck.micro_system
  assert recipe.micro_coordinate_map is deck.micro_map
  assert recipe.layout.ip_count == 9
  assert recipe.settings.nested_ip_thread_count == deck.runtime.task_pool.thread_count
  other = _deck(elastic=True)
  assert recipe.provenance.to_bytes() == other.runtime.recipe.provenance.to_bytes()
  other.runtime.close()
  assert operator_tangent_class(operator.header) == FE2_TANGENT_CLASS
  assert not hasattr(operator, "drivers")
  assert not hasattr(operator, "nested_store")
  assert not hasattr(operator, "task_pool")

  store = deck.runtime.store
  assert not store.trial_open
  assert {generation.ordinal for generation in store.generations} == {1}
  assert all(
    statistics.evaluation_count == 0 for statistics in deck.runtime.driver_statistics
  )
  fresh = PreparedNestedExecution(recipe)
  assert fresh.recipe is recipe
  assert fresh.store is not store
  assert fresh.store.instance != store.instance
  assert tuple(astuple(item) for item in fresh.driver_statistics) == tuple(
    astuple(item) for item in deck.runtime.driver_statistics
  )
  fresh.close()

  direct_base = compile_system(
    authoring.small_strain_continuum(
      deck.macro_mesh, material=authoring.linear_elastic(_E, _NU)
    ),
    q8_reference_registry(),
  )
  direct_operator = direct_base.operators[0]
  assert operator_nested_execution(direct_operator.header) is None
  assert operator_tangent_class(direct_operator.header) is None
  direct = direct_operator.evaluate(
    OperatorEvaluationInput(
      port_values=(FinalizedArray(np.zeros((1, 16)), dtype=np.float64),),
      accepted_state=FinalizedArray(np.zeros((1, 0)), dtype=np.float64),
      signals=(),
      request=ChannelRequest(("internal-force",), ("material-tangent",)),
    )
  )
  assert evaluation_nested_references(direct) == ()
  assert evaluation_status(direct) is EvaluationStatus.OK
  # The untouched direct compiler retains deterministic content identity.
  again = compile_system(
    authoring.small_strain_continuum(
      deck.macro_mesh, material=authoring.linear_elastic(_E, _NU)
    ),
    q8_reference_registry(),
  )
  assert direct_base.content_fingerprint == again.content_fingerprint
  direct_manifest = direct_base.provenance.manifest.to_bytes()
  assert direct_manifest == again.provenance.manifest.to_bytes()


def test_micro_loads_are_part_of_fe2_content_identity() -> None:
  plain = _deck(elastic=True)
  center = next(
    node for node in plain.micro_mesh.nodes if node.coordinates == (0.5, 0.5)
  )
  load = authoring.nodal_load(center.id, "x", -10.0, coordinate="eps11")
  loaded = _deck(elastic=True, micro_loads=(load,))
  identical = _deck(elastic=True, micro_loads=(load,))

  plain_bytes = plain.system.provenance.manifest.to_bytes()
  loaded_bytes = loaded.system.provenance.manifest.to_bytes()
  identical_bytes = identical.system.provenance.manifest.to_bytes()
  assert plain_bytes != loaded_bytes
  plain_digest = plain.system.content_fingerprint.digest
  loaded_digest = loaded.system.content_fingerprint.digest
  assert plain_digest != loaded_digest
  assert loaded_bytes == identical_bytes
  assert loaded_digest == identical.system.content_fingerprint.digest
  assert (
    loaded.system.operators[0].content_manifest.to_bytes()
    == identical.system.operators[0].content_manifest.to_bytes()
  )
  assert plain.runtime.recipe.provenance.to_bytes() != (
    loaded.runtime.recipe.provenance.to_bytes()
  )
  assert loaded.runtime.recipe.provenance.to_bytes() == (
    identical.runtime.recipe.provenance.to_bytes()
  )


def test_macro_fe2_reported_ip_state_matches_legacy_sigma11() -> None:
  deck = _deck(kappa_c=2.5e-3)
  strain = _documented_strain()
  _seed_documented_ip_state(deck, strain)
  operator = deck.system.operators[0]
  accepted = FinalizedArray(np.zeros((9, 3)), dtype=np.float64)
  full = authoring.trial_vector(
    deck.system,
    {
      node.id: (
        float(strain[0] * node.coordinates[0]),
        float(strain[1] * node.coordinates[1]),
      )
      for node in deck.macro_mesh.nodes
    },
  )
  gather = operator.header.ports[0].coefficient_map.values
  evaluation = _evaluate_port(deck, full[gather], accepted)
  assert evaluation_status(evaluation) is EvaluationStatus.OK
  np.testing.assert_allclose(
    evaluation.trial_state.values,
    np.tile(strain, (9, 1)),
    rtol=0.0,
    atol=2.0e-18,
  )
  assert len(evaluation_nested_references(evaluation)) == 9
  # Rejecting the direct evaluation leaves the seeded accepted state intact;
  # observe the same point's solve-free boundary-reaction reduction next.
  store = deck.runtime.store
  store.begin_trial()
  try:
    stress = _stress_from_reactions(deck)
  finally:
    store.reject_trial()
  # Legacy macro_fe2.pro reports sigma11 = 0.2006 at eps = [0.002,
  # -0.0006666667, 0]. Its shared damage kernel gives 0.200569961 at that
  # state; the v3 boundary-reaction reduction differs only by summation-order
  # roundoff (measured 1.9e-10 abs). Pin 1e-7, and separately pin the
  # legacy four-significant-digit report within half its last printed unit.
  np.testing.assert_allclose(stress[0], 0.200569961, rtol=0.0, atol=1.0e-7)
  assert abs(stress[0] - 0.2006) <= 5.0e-5
  np.testing.assert_allclose(stress[1:], np.zeros(2), rtol=0.0, atol=1.0e-10)


def test_real_free_dof_micromodel_cantilever_runs_end_to_end() -> None:
  cantilever = _cantilever()
  assert cantilever.coordinate_map.reduced_dof_count == 7
  result = cantilever.driver.run(
    base_point=_load_point(0.0), target_points=(_load_point(1.0),)
  )
  assert result.status is DriverStatus.COMPLETED
  owner = cantilever.driver.owner
  deck = cantilever.deck
  operator = deck.system.operators[0]
  store = deck.runtime.store
  assert owner.generation.ordinal > 0
  assert len(owner.history[-1].nested_references) == 9
  assert not store.trial_open
  assert {generation.ordinal for generation in store.generations} == {
    1 + owner.generation.ordinal
  }
  macro_strain = owner.accepted_state(operator.header.state_layout.block_id).values
  assert float(np.max(np.abs(macro_strain))) > 4.0e-4
  assert any(np.any(block != 0.0) for block in store.snapshot().blocks)


def test_macro_tangent_is_fd_exact_at_nonzero_state_and_tamper_visible() -> None:
  cantilever = _cantilever(prescribed=2.0e-4, elastic=True)
  result = cantilever.driver.run(
    base_point=_load_point(0.0), target_points=(_load_point(1.0),)
  )
  assert result.status is DriverStatus.COMPLETED
  deck = cantilever.deck
  operator = deck.system.operators[0]
  owner = cantilever.driver.owner
  accepted = owner.accepted_state(operator.header.state_layout.block_id)
  assert float(np.max(np.abs(accepted.values))) > 0.0
  gather = operator.header.ports[0].coefficient_map.values
  base = owner.accepted_physical().values[gather]
  evaluation = _evaluate_port(deck, base, accepted)
  tangent = evaluation.jacobian_values[0].values[0]
  base_residual = evaluation.residual_values[0].values[0]
  h = 1.0e-7
  fd = np.empty_like(tangent)
  for axis in range(base.shape[1]):
    plus = np.array(base, copy=True)
    minus = np.array(base, copy=True)
    plus[0, axis] += h
    minus[0, axis] -= h
    residual_plus = _evaluate_port(deck, plus, accepted).residual_values[0].values[0]
    residual_minus = _evaluate_port(deck, minus, accepted).residual_values[0].values[0]
    fd[:, axis] = (residual_plus - residual_minus) / (2.0 * h)
  # Measured max abs deviation on this reference-platform run is below 3e-8
  # on an O(1e3) tangent (nested solve/reaction summation order); 2e-6 leaves
  # ample cross-platform headroom without admitting a tangent perturbation.
  np.testing.assert_allclose(tangent, fd, rtol=2.0e-8, atol=2.0e-6)
  tampered = np.array(tangent, copy=True)
  tampered[0, 0] += max(1.0, abs(tampered[0, 0])) * 1.0e-3
  assert float(np.max(np.abs(tampered - fd))) > 1.0e-3
  assert not np.allclose(tampered, fd, rtol=2.0e-8, atol=2.0e-6)
  assert np.array_equal(base_residual, evaluation.residual_values[0].values[0])


def test_fe2_commit_and_reject_isolate_macro_and_nested_bytes() -> None:
  deck = _deck(elastic=True)
  operator = deck.system.operators[0]
  store = deck.runtime.store
  block_id = operator.header.state_layout.block_id
  owner = StateTransactionOwner(deck.system, nested_stores=(store,))
  macro_before = (
    owner.accepted_physical().values.tobytes(),
    owner.accepted_state(block_id).values.tobytes(),
    owner.history,
  )
  nested_before = store.encode()

  store.begin_trial()
  coefficients = np.array(store.accepted_coefficients(0), copy=True) + 0.01
  store.write_trial_coefficients(0, coefficients)
  for block in store.layout.blocks:
    rows = np.array(store.accepted_block(0, block.block_id), copy=True) + 0.001
    store.write_trial_block(0, block.block_id, rows)
  store.trial_complete(0)
  store.commit_trial()
  nested_committed = store.encode()
  assert nested_committed != nested_before
  assert macro_before == (
    owner.accepted_physical().values.tobytes(),
    owner.accepted_state(block_id).values.tobytes(),
    owner.history,
  )

  store.begin_trial()
  store.write_trial_coefficients(0, coefficients + 1.0)
  for block in store.layout.blocks:
    store.write_trial_block(0, block.block_id, rows + 1.0)
  store.trial_complete(0)
  store.reject_trial()
  assert store.encode() == nested_committed
  assert macro_before[0] == owner.accepted_physical().values.tobytes()

  macro = owner.begin()
  macro.stage_physical(np.ones(deck.system.coefficient_count))
  macro.stage_state(block_id, np.ones(operator.header.state_layout.row_shape))
  macro.commit()
  assert store.encode() == nested_committed
  macro_after = owner.accepted_physical().values.tobytes()
  macro = owner.begin()
  macro.stage_physical(np.full(deck.system.coefficient_count, 2.0))
  macro.reject()
  assert owner.accepted_physical().values.tobytes() == macro_after
  assert store.encode() == nested_committed


def test_retry_iteration_discards_and_reopens_the_nested_scope(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  cantilever = _cantilever(prescribed=2.0e-4, elastic=True)
  deck = cantilever.deck
  operator = deck.system.operators[0]
  store = deck.runtime.store
  original = operator.evaluate
  calls = 0
  scopes = []

  def reject_second(
    _self: FE2ContinuumOperator, inputs: OperatorEvaluationInput
  ) -> OperatorEvaluation:
    nonlocal calls
    calls += 1
    evaluation = original(inputs)
    scopes.append(store.trial_scope_id)
    if calls == 2:
      return _new(
        OperatorEvaluation,
        residual_values=(),
        jacobian_values=(),
        trial_state=inputs.accepted_state,
        status=EvaluationStatus.REJECT_ITERATION,
        nested_references=(),
      )
    return evaluation

  monkeypatch.setattr(FE2ContinuumOperator, "evaluate", reject_second)
  result = cantilever.driver.run(
    base_point=_load_point(0.0), target_points=(_load_point(1.0),)
  )
  assert result.status is DriverStatus.COMPLETED
  assert calls >= 3
  assert scopes[0] != scopes[1]
  assert scopes[1] != scopes[2]
  assert not store.trial_open
  owner = cantilever.driver.owner
  assert len(owner.history[-1].nested_references) == 9
  assert {generation.ordinal for generation in store.generations} == {
    1 + owner.generation.ordinal
  }


def test_nested_nonconvergence_rejects_macro_without_state_resurrection(
  _one_numba_thread: None,
) -> None:
  # One Newton iteration cannot advance this nonzero micro deck. Its six
  # cutbacks plus terminal attempt must surface as one typed macro rejection.
  cantilever = _cantilever(
    kappa_c=2.5e-3,
    micro_settings=NonlinearStaticSettings(max_iterations=1),
    macro_settings=NonlinearStaticSettings(max_cutbacks=0),
  )
  deck = cantilever.deck
  operator = deck.system.operators[0]
  store = deck.runtime.store
  nested_before = store.encode()
  owner = cantilever.driver.owner
  physical_before = owner.accepted_physical().values.tobytes()
  result = cantilever.driver.run(
    base_point=_load_point(0.0), target_points=(_load_point(1.0),)
  )
  assert result.status is DriverStatus.STEP_FAILED
  (record,) = result.records
  assert record.failure_class is StepFailureClass.NESTED_NON_CONVERGENCE
  assert record.iterations[-1].status is EvaluationStatus.REJECT_STEP
  failure = record.nested_failure
  assert failure is not None
  assert failure.code == "nested-non-convergence"
  assert failure.operator_index == 0
  assert failure.ip_index == 0
  assert failure.failure_class is StepFailureClass.ITERATION_BUDGET_EXHAUSTED

  # Under the direct 1T schedule every micro attempt evaluates and factorizes
  # once. IP 0 fails after seven attempts; the other IPs are never run.
  assert astuple(record.nested_statistics[0])[2:] == (7, 7, 7, 0, 7)
  assert all(
    astuple(cost)[2:] == (0, 0, 0, 0, 0) for cost in record.nested_statistics[1:]
  )
  statistics = result.statistics
  assert (
    statistics.nested_iteration_count,
    statistics.nested_evaluation_count,
    statistics.nested_factorization_count,
    statistics.nested_factorization_reuse_count,
    statistics.nested_linear_solve_count,
  ) == (7, 7, 7, 0, 7)
  assert owner.generation.ordinal == 0
  assert owner.history == ()
  assert owner.accepted_physical().values.tobytes() == physical_before
  assert store.encode() == nested_before
  assert not store.trial_open

  # A fresh zero trial completes every point from accepted storage; no rejected
  # trial bytes can reappear through the reused scoped workspaces.
  restarted = cantilever.driver.run(
    base_point=_load_point(0.0), target_points=(_load_point(0.0),)
  )
  assert restarted.status is DriverStatus.COMPLETED
  snapshot = store.snapshot()
  assert not np.any(snapshot.coefficients.values)
  assert all(not np.any(block.values) for block in snapshot.blocks)
  assert not np.any(owner.accepted_state(operator.header.state_layout.block_id).values)


def test_nested_costs_match_hand_counted_elastic_work() -> None:
  cantilever = _cantilever(prescribed=2.0e-4, elastic=True)
  result = cantilever.driver.run(
    base_point=_load_point(0.0),
    target_points=(_load_point(1.0), _load_point(2.0)),
  )
  assert result.status is DriverStatus.COMPLETED
  assert len(result.records) == 2

  # Each macro substep takes two evaluations. Each one runs every elastic micro
  # driver through two Newton evaluations (one correction, one convergence).
  # Every fresh execution factorizes once per IP on its first run. Its constant
  # tangent then covers one Newton plus three homogenization back-substitutions
  # per run, so later runs contribute four reuses and no fresh factorization.
  for record_index, record in enumerate(result.records):
    assert len(record.iterations) == 2
    assert record.nested_failure is None
    assert len(record.nested_statistics) == 9
    expected = (4, 4, 1, 7, 8) if record_index == 0 else (4, 4, 0, 8, 8)
    for cost in record.nested_statistics:
      assert astuple(cost)[2:] == expected
      assert cost.factorization_count + cost.factorization_reuse_count == (
        cost.linear_solve_count
      )

  statistics = result.statistics
  assert (
    statistics.nested_iteration_count,
    statistics.nested_evaluation_count,
    statistics.nested_factorization_count,
    statistics.nested_factorization_reuse_count,
    statistics.nested_linear_solve_count,
  ) == (72, 72, 9, 135, 144)
  assert (
    statistics.factorization_count
    + statistics.factorization_reuse_count
    + (
      statistics.nested_factorization_count
      + statistics.nested_factorization_reuse_count
    )
    == statistics.linear_solve_count + statistics.nested_linear_solve_count
  )


def test_nested_accounting_is_primal_byte_transparent(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  control = _cantilever(prescribed=2.0e-4, elastic=True)
  with monkeypatch.context() as disabled:
    disabled.setattr(
      NonlinearStaticDriver,
      "_snapshot_nested_drivers",
      lambda _self: (),
    )
    disabled.setattr(
      NonlinearStaticDriver,
      "_collect_nested_activity",
      lambda _self, _cursors: ((), None),
    )
    plain = control.driver.run(
      base_point=_load_point(0.0), target_points=(_load_point(1.0),)
    )
  observed = _cantilever(prescribed=2.0e-4, elastic=True)
  counted = observed.driver.run(
    base_point=_load_point(0.0), target_points=(_load_point(1.0),)
  )
  assert plain.status is counted.status is DriverStatus.COMPLETED
  assert astuple(plain.statistics)[:9] == astuple(counted.statistics)[:9]
  assert astuple(plain.statistics)[9:] == (0, 0, 0, 0, 0)
  assert counted.statistics.nested_evaluation_count > 0

  for first, second in zip(plain.records, counted.records, strict=True):
    assert first.status is second.status
    assert first.progress == second.progress
    assert [
      (item.status, item.residual_norm, item.increment_norm)
      for item in first.iterations
    ] == [
      (item.status, item.residual_norm, item.increment_norm)
      for item in second.iterations
    ]
    assert first.observation is not None and second.observation is not None
    assert first.observation.reactions.values.tobytes() == (
      second.observation.reactions.values.tobytes()
    )
    assert first.observation.constraint_work == second.observation.constraint_work

  control_operator = control.deck.system.operators[0]
  observed_operator = observed.deck.system.operators[0]
  assert control.driver.owner.accepted_physical().values.tobytes() == (
    observed.driver.owner.accepted_physical().values.tobytes()
  )
  control_state = control.driver.owner.accepted_state(
    control_operator.header.state_layout.block_id
  )
  observed_state = observed.driver.owner.accepted_state(
    observed_operator.header.state_layout.block_id
  )
  assert control_state.values.tobytes() == observed_state.values.tobytes()
  assert control.deck.runtime.store.encode() == observed.deck.runtime.store.encode()


def _assert_raw_uint64_equal(first: np.ndarray, second: np.ndarray) -> None:
  assert first.shape == second.shape
  np.testing.assert_array_equal(
    np.ascontiguousarray(first, dtype=np.float64).view(np.uint64),
    np.ascontiguousarray(second, dtype=np.float64).view(np.uint64),
  )


def _evaluate_with_ip_outputs(
  deck: _Deck,
  values: np.ndarray,
  accepted: FinalizedArray,
) -> tuple[OperatorEvaluation, list[tuple[np.ndarray, np.ndarray] | None]]:
  evaluation = _evaluate_port(deck, values, accepted)
  stresses = deck.runtime.retained_buffer("fe2-stresses", (9, 3))
  tangents = deck.runtime.retained_buffer("fe2-tangents", (9, 3, 3))
  outputs = [
    (np.array(stresses[index], copy=True), np.array(tangents[index], copy=True))
    for index in range(9)
  ]
  return evaluation, outputs


def test_fe2_ip_pool_policy_defaults_to_serial_and_validates_width() -> None:
  deck = _deck(elastic=True, micro_settings=NonlinearStaticSettings())
  pool = deck.runtime.task_pool
  assert pool.thread_count == 1
  assert pool.worker_count == 1
  with pytest.raises(ValueError, match="nested IP thread count"):
    NonlinearStaticSettings(nested_ip_thread_count=0)


def test_ip_task_pool_matches_one_thread_raw_uint64_across_macro_steps() -> None:
  previous_threads = get_num_threads()
  try:
    set_num_threads(1)
    serial = _cantilever()
    serial_result = serial.driver.run(
      base_point=_load_point(0.0),
      target_points=(_load_point(0.5), _load_point(1.0)),
    )
    set_num_threads(2)
    parallel = _cantilever()
    parallel_result = parallel.driver.run(
      base_point=_load_point(0.0),
      target_points=(_load_point(0.5), _load_point(1.0)),
    )
    assert serial_result.status is parallel_result.status is DriverStatus.COMPLETED
    serial_operator = serial.deck.system.operators[0]
    parallel_operator = parallel.deck.system.operators[0]
    serial_pool = serial.deck.runtime.task_pool
    parallel_pool = parallel.deck.runtime.task_pool
    assert serial_pool.thread_count == 1
    assert parallel_pool.thread_count == 2
    assert parallel_pool.worker_count == 2
    assert parallel_pool.chunks == ((0, 5), (5, 9))
    assert (
      serial.deck.system.content_fingerprint.digest
      == parallel.deck.system.content_fingerprint.digest
    )
    block_id = serial_operator.header.state_layout.block_id
    _assert_raw_uint64_equal(
      serial.driver.owner.accepted_physical().values,
      parallel.driver.owner.accepted_physical().values,
    )
    _assert_raw_uint64_equal(
      serial.driver.owner.accepted_state(block_id).values,
      parallel.driver.owner.accepted_state(block_id).values,
    )
    assert serial.deck.runtime.store.encode() == parallel.deck.runtime.store.encode()

    set_num_threads(1)
    serial_gather = serial_operator.header.ports[0].coefficient_map.values
    serial_full = serial.driver.owner.accepted_physical().values
    serial_accepted = serial.driver.owner.accepted_state(block_id)
    serial_nested = serial.deck.runtime.store.encode()
    serial_evaluation, serial_outputs = _evaluate_with_ip_outputs(
      serial.deck, serial_full[serial_gather], serial_accepted
    )
    assert serial.deck.runtime.store.encode() == serial_nested

    set_num_threads(2)
    parallel_gather = parallel_operator.header.ports[0].coefficient_map.values
    parallel_full = parallel.driver.owner.accepted_physical().values
    parallel_accepted = parallel.driver.owner.accepted_state(block_id)
    parallel_nested = parallel.deck.runtime.store.encode()
    parallel_evaluation, parallel_outputs = _evaluate_with_ip_outputs(
      parallel.deck, parallel_full[parallel_gather], parallel_accepted
    )
    assert parallel.deck.runtime.store.encode() == parallel_nested
  finally:
    set_num_threads(previous_threads)

  assert evaluation_status(serial_evaluation) is EvaluationStatus.OK
  assert evaluation_status(parallel_evaluation) is EvaluationStatus.OK
  _assert_raw_uint64_equal(
    serial_evaluation.residual_values[0].values,
    parallel_evaluation.residual_values[0].values,
  )
  _assert_raw_uint64_equal(
    serial_evaluation.jacobian_values[0].values,
    parallel_evaluation.jacobian_values[0].values,
  )
  _assert_raw_uint64_equal(
    serial_evaluation.trial_state.values, parallel_evaluation.trial_state.values
  )
  assert len(serial_outputs) == len(parallel_outputs) == 9
  for serial_output, parallel_output in zip(
    serial_outputs, parallel_outputs, strict=True
  ):
    assert serial_output is not None and parallel_output is not None
    _assert_raw_uint64_equal(serial_output[0], parallel_output[0])
    _assert_raw_uint64_equal(serial_output[1], parallel_output[1])


def test_nested_staged_buffer_objects_survive_parallel_macro_steps() -> None:
  previous_threads = get_num_threads()
  try:
    set_num_threads(2)
    cantilever = _cantilever(prescribed=2.0e-4, elastic=True)
    deck = cantilever.deck
    store = deck.runtime.store
    block_id = store.layout.blocks[0].block_id
    store.begin_trial()
    coefficient_buffer = store.trial_coefficients(0).base
    block_buffer = store.trial_block(0, block_id).base
    store.reject_trial()
    stresses = deck.runtime.retained_buffer("fe2-stresses", (9, 3))
    tangents = deck.runtime.retained_buffer("fe2-tangents", (9, 3, 3))
    chunks = deck.runtime.task_pool.chunks
    result = cantilever.driver.run(
      base_point=_load_point(0.0),
      target_points=(_load_point(0.5), _load_point(1.0)),
    )
    assert result.status is DriverStatus.COMPLETED
    assert len(result.records) == 2
    store.begin_trial()
    assert store.trial_coefficients(0).base is coefficient_buffer
    assert store.trial_block(0, block_id).base is block_buffer
    store.reject_trial()
    assert deck.runtime.retained_buffer("fe2-stresses", (9, 3)) is stresses
    assert deck.runtime.retained_buffer("fe2-tangents", (9, 3, 3)) is tangents
    assert deck.runtime.task_pool.chunks == chunks
  finally:
    set_num_threads(previous_threads)


def test_independent_macro_executions_share_only_the_compiled_recipe() -> None:
  first = _cantilever(prescribed=2.0e-4, elastic=True)
  second = NonlinearStaticDriver(
    first.deck.system,
    first.coordinate_map,
    settings=NonlinearStaticSettings(tolerance=1.0e-9),
  )
  first_runtime = first.deck.runtime
  second_runtime = second.nested_executions[0]
  assert second_runtime is not None
  assert first_runtime.recipe is second_runtime.recipe
  assert first_runtime.store is not second_runtime.store
  assert first_runtime.task_pool is not second_runtime.task_pool
  block_id = first.deck.system.operators[0].header.state_layout.block_id

  def snapshot(
    driver: NonlinearStaticDriver,
    runtime: PreparedNestedExecution,
  ) -> tuple[object, ...]:
    return (
      driver.owner.accepted_physical().values.tobytes(),
      driver.owner.accepted_state(block_id).values.tobytes(),
      driver.owner.generation.ordinal,
      driver.owner.history,
      runtime.store.encode(),
      tuple(generation.ordinal for generation in runtime.store.generations),
      astuple(driver.statistics),
      tuple(astuple(item) for item in runtime.driver_statistics),
    )

  try:
    second_initial = snapshot(second, second_runtime)
    first_result = first.driver.run(
      base_point=_load_point(0.0), target_points=(_load_point(1.0),)
    )
    assert first_result.status is DriverStatus.COMPLETED
    first_physical = np.array(first.driver.owner.accepted_physical().values, copy=True)
    first_state = np.array(
      first.driver.owner.accepted_state(block_id).values, copy=True
    )
    first_nested = first_runtime.store.encode()
    first_statistics = astuple(first.driver.statistics)
    first_ip_statistics = tuple(
      astuple(item) for item in first_runtime.driver_statistics
    )
    assert snapshot(second, second_runtime) == second_initial

    advanced = first.driver.run(
      base_point=_load_point(1.0), target_points=(_load_point(2.0),)
    )
    assert advanced.status is DriverStatus.COMPLETED
    assert snapshot(second, second_runtime) == second_initial
    first_before_second = snapshot(first.driver, first_runtime)

    second_result = second.run(
      base_point=_load_point(0.0), target_points=(_load_point(1.0),)
    )
    assert second_result.status is DriverStatus.COMPLETED
    assert snapshot(first.driver, first_runtime) == first_before_second
    _assert_raw_uint64_equal(
      second.owner.accepted_physical().values,
      first_physical,
    )
    _assert_raw_uint64_equal(
      second.owner.accepted_state(block_id).values,
      first_state,
    )
    assert second_runtime.store.encode() == first_nested
    assert astuple(second.statistics) == first_statistics
    assert tuple(astuple(item) for item in second_runtime.driver_statistics) == (
      first_ip_statistics
    )
    assert second_result.records[-1].observation is not None
    assert first_result.records[-1].observation is not None
    _assert_raw_uint64_equal(
      second_result.records[-1].observation.reactions.values,
      first_result.records[-1].observation.reactions.values,
    )
  finally:
    first.driver.close()
    second.close()


def test_fresh_runtime_rebinding_and_cleanup_are_execution_local() -> None:
  deck = _deck(elastic=True)
  foreign = _deck(elastic=True)
  fresh = PreparedNestedExecution(deck.runtime.recipe)
  values = np.zeros((1, 16), dtype=np.float64)
  accepted = FinalizedArray(np.zeros((9, 3)), dtype=np.float64)
  try:
    control = _evaluate_port(deck, values, accepted)
    rebound = _evaluate_port(deck, values, accepted, runtime=fresh)
    _assert_raw_uint64_equal(
      control.residual_values[0].values,
      rebound.residual_values[0].values,
    )
    _assert_raw_uint64_equal(
      control.jacobian_values[0].values,
      rebound.jacobian_values[0].values,
    )
    _assert_raw_uint64_equal(control.trial_state.values, rebound.trial_state.values)
    with pytest.raises(ValueError, match="foreign recipe"):
      _evaluate_port(deck, values, accepted, runtime=foreign.runtime)

    fresh.begin_trial()
    buffer = fresh.retained_buffer("cleanup-probe", (2, 3))
    assert fresh.retained_buffer("cleanup-probe", (2, 3)) is buffer
    fresh.close()
    assert fresh.closed
    assert fresh.task_pool.closed
    assert not fresh.store.trial_open
    fresh.close()
    with pytest.raises(RuntimeError, match="pool is closed"):
      fresh.task_pool.run(lambda index: index)
    with pytest.raises(RuntimeError, match="execution is closed"):
      fresh.retained_buffer("cleanup-probe", (2, 3))
    with pytest.raises(RuntimeError, match="execution is closed"):
      fresh.run_point(
        0,
        base_point=_point((0.0, 0.0, 0.0)),
        target_points=(_point((0.0, 0.0, 0.0)),),
      )
  finally:
    deck.runtime.close()
    foreign.runtime.close()
    fresh.close()


def test_real_elastic_fe2_runs_through_riks_without_macro_owner_access(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  deck = _deck(elastic=True)
  deck.runtime.close()
  mesh = deck.macro_mesh
  left = tuple(node.id for node in mesh.nodes if abs(node.coordinates[0]) < 1.0e-12)
  right = tuple(
    node.id for node in mesh.nodes if abs(node.coordinates[0] - 1.0) < 1.0e-12
  )
  coordinate_map = compile_constraint_map(
    deck.system,
    constraints=authoring.fixed(left, components=("x", "y")),
    coordinates=(ProgramCoordinateSpec("load", "load"),),
  )
  loads = tuple(
    authoring.nodal_load(node, "x", 0.01, coordinate="load") for node in right
  )
  driver = RiksDriver(
    deck.system,
    coordinate_map,
    loads,
    ArcLengthSettings(tolerance=1.0e-9, fixed_step=True, max_lam=0.5),
  )
  runtime = driver.nested_executions[0]
  assert runtime is not None
  macro_owner = driver.owner
  nested_active = False
  original_evaluate = FE2ContinuumOperator.evaluate

  def guarded_evaluate(
    self: FE2ContinuumOperator,
    inputs: OperatorEvaluationInput,
  ) -> OperatorEvaluation:
    nonlocal nested_active
    assert inputs.nested_execution is not None
    assert inputs.nested_execution.runtime is runtime
    nested_active = True
    try:
      return original_evaluate(self, inputs)
    finally:
      nested_active = False

  def guard_owner_method(name: str) -> None:
    original = getattr(StateTransactionOwner, name)

    def guarded(self: StateTransactionOwner, *args: object, **kwargs: object) -> object:
      if nested_active and self is macro_owner:
        msg = f"nested execution accessed macro owner through {name}"
        raise AssertionError(msg)
      return original(self, *args, **kwargs)

    monkeypatch.setattr(StateTransactionOwner, name, guarded)

  for name in ("begin", "_commit", "_reject", "accepted_physical", "accepted_state"):
    guard_owner_method(name)
  monkeypatch.setattr(FE2ContinuumOperator, "evaluate", guarded_evaluate)
  try:
    result = driver.run(base_point=ProgramPoint())
    assert result.status is DriverStatus.COMPLETED
    committed = [
      record for record in result.records if record.status.value == "committed"
    ]
    assert len(committed) == 1
    record = committed[0]
    assert len(record.iterations) == 2
    assert record.nested_failure is None
    assert len(record.nested_statistics) == 9
    assert all(cost.evaluation_count > 0 for cost in record.nested_statistics)
    assert result.statistics.nested_evaluation_count == sum(
      cost.evaluation_count for cost in record.nested_statistics
    )
    assert record.observation is not None
    assert record.observation.reduced_residual_norm <= 1.0e-7
    assert record.committed_coefficients is not None
    assert bool(np.isfinite(record.committed_coefficients.values).all())
    assert not runtime.store.trial_open
    assert {generation.ordinal for generation in runtime.store.generations} == {
      1 + macro_owner.generation.ordinal
    }
    assert len(macro_owner.history[-1].nested_references) == 9
  finally:
    driver.close()
  assert runtime.closed
  assert runtime.task_pool.closed


def test_riks_nested_nonconvergence_is_typed_and_rolls_back() -> None:
  deck = _deck(
    elastic=True,
    micro_settings=NonlinearStaticSettings(max_iterations=1),
  )
  deck.runtime.close()
  mesh = deck.macro_mesh
  left = tuple(node.id for node in mesh.nodes if abs(node.coordinates[0]) < 1.0e-12)
  right = tuple(
    node.id for node in mesh.nodes if abs(node.coordinates[0] - 1.0) < 1.0e-12
  )
  coordinate_map = compile_constraint_map(
    deck.system,
    constraints=authoring.fixed(left, components=("x", "y")),
    coordinates=(ProgramCoordinateSpec("load", "load"),),
  )
  loads = tuple(
    authoring.nodal_load(node, "x", 0.01, coordinate="load") for node in right
  )
  driver = RiksDriver(
    deck.system,
    coordinate_map,
    loads,
    ArcLengthSettings(
      tolerance=1.0e-9,
      fixed_step=True,
      max_lam=10.0,
      max_cutbacks=0,
    ),
  )
  runtime = driver.nested_executions[0]
  assert runtime is not None
  nested_before = runtime.store.encode()
  physical_before = driver.owner.accepted_physical().values.tobytes()
  try:
    result = driver.run(base_point=ProgramPoint())
    assert result.status is DriverStatus.STEP_FAILED
    (record,) = result.records
    assert record.status.value == "failed"
    assert record.nested_failure is not None
    assert record.nested_failure.code == "nested-non-convergence"
    assert record.nested_failure.failure_class is (
      StepFailureClass.ITERATION_BUDGET_EXHAUSTED
    )
    assert record.nested_statistics[0].evaluation_count > 0
    assert result.statistics.nested_evaluation_count == sum(
      cost.evaluation_count for cost in record.nested_statistics
    )
    assert driver.owner.generation.ordinal == 0
    assert driver.owner.history == ()
    assert driver.owner.accepted_physical().values.tobytes() == physical_before
    assert runtime.store.encode() == nested_before
    assert not runtime.store.trial_open
  finally:
    driver.close()
