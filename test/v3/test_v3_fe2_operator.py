# SPDX-License-Identifier: MIT

"""FE² continuum operator numerical, transaction, and cross-level failure battery."""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from pyfem.v3 import authoring
from pyfem.v3.authoring.rve import PeriodicCell
from pyfem.v3.compile.continuum import damage_reference_registry, q8_reference_registry
from pyfem.v3.compile.fe2 import FE2_TANGENT_CLASS, FE2ContinuumOperator
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
  NestedExecutionDeclaration,
  NestedExecutionModality,
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


@dataclass(frozen=True, slots=True)
class _Deck:
  system: CompiledSystem
  micro_system: CompiledSystem
  micro_map: object
  cell: PeriodicCell
  macro_mesh: object
  micro_mesh: object


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
  arguments = {
    "micro_system": micro_system,
    "micro_coordinate_map": micro_map,
    "homogenization": cell.homogenization_request(micro_system),
  }
  if micro_settings is not None:
    arguments["settings"] = micro_settings
  system = authoring.fe2_continuum(base, **arguments)
  return _Deck(system, micro_system, micro_map, cell, macro_mesh, micro_mesh)


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
  return _Cantilever(deck, driver, coordinate_map)


def _evaluate_port(
  deck: _Deck,
  values: np.ndarray,
  accepted: FinalizedArray,
) -> OperatorEvaluation:
  operator = deck.system.operators[0]
  store = operator.nested_store
  store.begin_trial()
  try:
    return operator.evaluate(
      OperatorEvaluationInput(
        port_values=(FinalizedArray(values, dtype=np.float64),),
        accepted_state=accepted,
        signals=(),
        request=ChannelRequest(("internal-force",), ("material-tangent",)),
        nested_execution=NestedEvaluationInput(store),
      )
    )
  finally:
    if store.trial_open:
      store.reject_trial()


def _seed_documented_ip_state(deck: _Deck, strain: np.ndarray) -> None:
  """Seed the measured converged IP displacement field with virgin material history."""
  operator = deck.system.operators[0]
  store = operator.nested_store
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
  operator = deck.system.operators[0]
  point = _point(_documented_strain())
  result = operator.drivers[ip_index].run(base_point=point, target_points=(point,))
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


def test_header_additions_and_virgin_nested_probe_are_additive() -> None:
  deck = _deck(elastic=True)
  operator = deck.system.operators[0]
  declaration = operator_nested_execution(operator.header)
  assert type(declaration) is NestedExecutionDeclaration
  assert declaration.modality is NestedExecutionModality.DRIVER_PER_INTEGRATION_POINT
  assert declaration.store is operator.nested_store
  assert operator_tangent_class(operator.header) == FE2_TANGENT_CLASS
  assert len(operator.drivers) == operator.nested_store.ip_count == 9
  # The compile probe really evaluated every nested driver and then rejected:
  # only the one-time initial-state seeding generation remains accepted.
  assert not operator.nested_store.trial_open
  assert {generation.ordinal for generation in operator.nested_store.generations} == {1}
  assert all(driver.statistics.evaluation_count > 0 for driver in operator.drivers)

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
  store = operator.nested_store
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
  operator = cantilever.deck.system.operators[0]
  assert owner.generation.ordinal > 0
  assert len(owner.history[-1].nested_references) == 9
  assert not operator.nested_store.trial_open
  assert {generation.ordinal for generation in operator.nested_store.generations} == {
    1 + owner.generation.ordinal
  }
  macro_strain = owner.accepted_state(operator.header.state_layout.block_id).values
  assert float(np.max(np.abs(macro_strain))) > 4.0e-4
  assert any(np.any(block != 0.0) for block in operator.nested_store.snapshot().blocks)


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
  store = operator.nested_store
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
  operator = cantilever.deck.system.operators[0]
  store = operator.nested_store
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


def test_nested_nonconvergence_rejects_macro_without_state_resurrection() -> None:
  # No macro cutback is allowed: the first nested singular/nonconvergent
  # response must fail the target rather than committing a partial nested trial.
  cantilever = _cantilever(
    kappa_c=2.5e-3,
    macro_settings=NonlinearStaticSettings(max_cutbacks=0),
  )
  store = cantilever.deck.system.operators[0].nested_store
  nested_before = store.encode()
  owner = cantilever.driver.owner
  physical_before = owner.accepted_physical().values.tobytes()
  result = cantilever.driver.run(
    base_point=_load_point(0.0), target_points=(_load_point(1.0),)
  )
  assert result.status is DriverStatus.STEP_FAILED
  assert owner.generation.ordinal == 0
  assert owner.history == ()
  assert owner.accepted_physical().values.tobytes() == physical_before
  assert store.encode() == nested_before
  assert not store.trial_open
  assert any(
    iteration.status is EvaluationStatus.REJECT_STEP
    for record in result.records
    for iteration in record.iterations
  )
