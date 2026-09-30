# SPDX-License-Identifier: MIT

"""Transaction-level authoring ergonomics: begin/stage/commit and stepping.

Covers the plain-value program helpers (``fixed``/``nodal_load``), the
``StateOwner``/``StateTrial`` transaction facade (plain mappings and arrays in,
plain arrays and byte-exact snapshots out), the ``NonlinearStaticSession``
stepping loop over the landed driver, and the stateful student persona: an
authored memory spring stepped over multiple committed increments, verified
against an independent cubic-equilibrium oracle, with a forced cutback cascade
and a byte-identical rollback proof.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from pyfem.v3 import authoring
from pyfem.v3.authoring.transactions import _resolve_block_id
from pyfem.v3.driver import NonlinearStaticDriver, NonlinearStaticSettings
from pyfem.v3.model.system import CompiledSystem
from pyfem.v3.spec.program import (
  AffineCoefficientSpec,
  AffineTieSpec,
  AffineValueSpec,
  DofRef,
  NodalLoadSpec,
  ProgramCoordinateSpec,
  ProgramCoordinateValue,
  ProgramPoint,
)
from pyfem.v3.state import StateTransactionOwner

_BAR_LENGTH = 2.0
_YOUNGS_MODULUS = 5.0e6
_AREA = 1.0
_SPRING_STIFFNESS = 4.0e3
_EXTENSION_CAP = 1.0e-4
_LOAD_SCALE = 1200.0


def student_linear_spring(
  displacements: np.ndarray,
  accepted_rows: np.ndarray,
  parameters: np.ndarray,
) -> authoring.SpringKernelResult:
  """Plain linear spring that remembers its largest extension."""
  (stiffness,) = parameters
  trial = np.maximum(
    accepted_rows[:, 0],
    np.linalg.norm(displacements, axis=1),
  )
  return authoring.SpringKernelResult(
    force=stiffness * displacements,
    tangent=stiffness * np.tile(np.eye(2), (len(displacements), 1, 1)),
    trial_rows=trial[:, None],
    status=authoring.EvaluationStatus.OK,
  )


def student_memory_spring(
  displacements: np.ndarray,
  accepted_rows: np.ndarray,
  parameters: np.ndarray,
) -> authoring.SpringKernelResult:
  """Linear spring with a peak-extension memory capped per committed increment.

  Growing the envelope by more than ``max_increment`` in one substep reports
  ``REJECT_STEP`` — the student-law analog of the landed damage-envelope
  spring's increment cap, asking the driver to halve and retry.
  """
  stiffness, max_increment = parameters
  extension = np.linalg.norm(displacements, axis=1)
  previous = accepted_rows[:, 0]
  trial = np.maximum(previous, extension)
  if bool((trial > previous + max_increment).any()):
    return authoring.SpringKernelResult(
      force=np.zeros_like(displacements),
      tangent=np.zeros((len(displacements), 2, 2)),
      trial_rows=np.array(accepted_rows, copy=True),
      status=authoring.EvaluationStatus.REJECT_STEP,
    )
  return authoring.SpringKernelResult(
    force=stiffness * displacements,
    tangent=stiffness * np.tile(np.eye(2), (len(displacements), 1, 1)),
    trial_rows=trial[:, None],
    status=authoring.EvaluationStatus.OK,
  )


def _bar_system() -> CompiledSystem:
  mesh = authoring.line2_mesh([(0.0, 0.0), (_BAR_LENGTH, 0.0)], [(1, 2)])
  return authoring.compile(
    authoring.truss(
      mesh,
      material=authoring.uniaxial_elastic(_YOUNGS_MODULUS, _AREA),
    )
  )


def _spring_system(**overrides: object) -> CompiledSystem:
  options = {
    "nodes": {"tip": 2},
    "kernel": student_linear_spring,
    "state": (("max_extension", 1),),
    "parameters": (3.0,),
    "name": "linear-spring-with-memory",
    "implementation_id": "student-linear-spring-v1",
  }
  options.update(overrides)
  return authoring.spring(_bar_system(), **options)  # type: ignore[arg-type]


def _memory_session(
  settings: NonlinearStaticSettings | None = None,
) -> authoring.NonlinearStaticSession:
  system = _spring_system(
    kernel=student_memory_spring,
    parameters=(_SPRING_STIFFNESS, _EXTENSION_CAP),
    name="capped-memory-spring",
    implementation_id="student-memory-spring-v1",
  )
  return authoring.nonlinear_static(
    system,
    constraints=(
      authoring.fixed(nodes=(1,)) + authoring.fixed(nodes=(2,), components=("y",))
    ),
    loads=(authoring.nodal_load(2, "x", _LOAD_SCALE),),
    settings=settings,
  )


def _tip_extension_oracle(force: float, stiffness: float = _SPRING_STIFFNESS) -> float:
  """Independent 1-DOF equilibrium oracle for the bar-plus-spring persona.

  The landed truss binds uniaxial Green-Lagrange strain, so the bar internal
  force is the cubic ``EA * (e + e^2/2) * (1 + e)``; the ground spring adds
  ``k * L * e``. Bisection on the monotone branch solves ``u = L * e``
  without touching the driver's assembly.
  """

  def residual(e: float) -> float:
    return (
      _YOUNGS_MODULUS * _AREA * (e + 1.5 * e * e + 0.5 * e**3)
      + stiffness * _BAR_LENGTH * e
      - force
    )

  lo, hi = 0.0, 1.0
  for _ in range(200):
    mid = 0.5 * (lo + hi)
    if residual(mid) > 0.0:
      hi = mid
    else:
      lo = mid
  return _BAR_LENGTH * 0.5 * (lo + hi)


# --- program helpers ----------------------------------------------------------


def test_fixed_and_nodal_load_emit_landed_specs_with_labels() -> None:
  constraints = authoring.fixed(nodes=(1, "base")) + authoring.fixed(
    nodes=(2,),
    components=("y",),
  )
  assert tuple(spec.id for spec in constraints) == (
    "fixed-1-x",
    "fixed-1-y",
    "fixed-base-x",
    "fixed-base-y",
    "fixed-2-y",
  )
  first = constraints[0]
  assert first.target.node_id == 1
  assert first.target.field_id == "displacement"
  assert first.target.component == "x"
  assert first.value.constant == 0.0
  assert first.value.coefficients == ()
  assert first.source.source == "authoring.fixed:1:x"

  load = authoring.nodal_load(2, "x", -25.0)
  assert isinstance(load, NodalLoadSpec)
  assert load.id == "load-2-x"
  assert load.target.node_id == 2
  assert load.target.component == "x"
  (coefficient,) = load.value.coefficients
  assert coefficient.coordinate == "load"
  assert coefficient.coefficient == -25.0
  assert coefficient.source.source == "authoring.nodal_load:load-2-x:load"
  assert load.source.source == "authoring.nodal_load:load-2-x"

  named = authoring.nodal_load(
    "tip",
    "y",
    2,
    field_id="temperature",
    coordinate="time",
    id="heating",
  )
  assert named.id == "heating"
  assert named.target.field_id == "temperature"
  assert named.value.coefficients[0].coordinate == "time"
  assert named.value.coefficients[0].coefficient == 2.0


def test_program_helpers_validate_plain_inputs() -> None:
  with pytest.raises(ValueError, match="at least one node"):
    authoring.fixed(nodes=())
  with pytest.raises(ValueError, match="at least one component"):
    authoring.fixed(nodes=(1,), components=())
  with pytest.raises(ValueError, match="twice"):
    authoring.fixed(nodes=(1, 1), components=("x",))
  with pytest.raises(TypeError, match="sequence of node ids"):
    authoring.fixed(nodes=1)  # type: ignore[arg-type]
  with pytest.raises(TypeError, match="non-empty string or integer id"):
    authoring.fixed(nodes=(None,))  # type: ignore[arg-type]
  with pytest.raises(TypeError, match="non-empty exact strings"):
    authoring.fixed(nodes=(1,), components=("",))
  with pytest.raises(TypeError, match="non-empty string or integer id"):
    authoring.fixed(nodes=(1,), field_id=None)  # type: ignore[arg-type]

  with pytest.raises(TypeError, match="exact number"):
    authoring.nodal_load(2, "x", True)  # type: ignore[arg-type]
  with pytest.raises(ValueError, match="finite"):
    authoring.nodal_load(2, "x", float("inf"))
  with pytest.raises(TypeError, match="non-empty exact string"):
    authoring.nodal_load(2, "x", 1.0, coordinate="")
  with pytest.raises(TypeError, match="non-empty string or integer id"):
    authoring.nodal_load(None, "x", 1.0)  # type: ignore[arg-type]


# --- state owner and trial ergonomics ------------------------------------------


def test_manual_begin_stage_commit_cycle_advances_generations() -> None:
  """A driver-free stateful workflow runs through authoring calls only."""
  system = _spring_system()
  owner = authoring.state_owner(system)
  assert owner.ordinal == 0
  assert owner.system is system
  assert type(owner.owner) is StateTransactionOwner
  initial = owner.snapshot()
  np.testing.assert_array_equal(owner.accepted_state("springs"), np.zeros((1, 1)))

  trial = owner.begin()
  assert trial.ordinal == 0
  np.testing.assert_array_equal(trial.accepted_state("springs"), np.zeros((1, 1)))
  evaluation = authoring.evaluate(
    system,
    {2: (0.4, -0.3)},
    operator=1,
    accepted_state=trial.accepted_state("springs"),
  )
  trial.stage_displacements({2: (0.4, -0.3)})
  trial.stage_state("springs", evaluation.trial_state.values)
  trial.commit()

  assert owner.ordinal == 1
  np.testing.assert_array_equal(
    owner.accepted_coefficients(),
    authoring.trial_vector(system, {2: (0.4, -0.3)}),
  )
  np.testing.assert_array_equal(owner.accepted_state("springs"), [[0.5]])
  assert owner.snapshot() != initial
  assert type(owner.state_bytes("springs")) is bytes

  # The next trial is rooted at the committed generation, and a reject leaves
  # the accepted state byte-identical.
  trial = owner.begin()
  assert trial.ordinal == 1
  np.testing.assert_array_equal(trial.accepted_state("springs"), [[0.5]])
  snapshot = owner.snapshot()
  trial.stage_state("springs", [[9.0]])
  trial.reject()
  assert owner.snapshot() == snapshot
  assert owner.ordinal == 1
  assert len(owner.history) == 1


def test_trial_protocol_violations_are_typed_errors() -> None:
  owner = authoring.state_owner(_spring_system())
  trial = owner.begin()
  with pytest.raises(RuntimeError, match="already open"):
    owner.begin()
  with pytest.raises(ValueError, match="at least one staged value"):
    trial.commit()
  # The failed commit leaves the trial open; reject consumes it.
  trial.reject()
  with pytest.raises(RuntimeError, match="already consumed"):
    trial.reject()
  with pytest.raises(RuntimeError, match="already consumed"):
    trial.stage_state("springs", [[1.0]])

  trial = owner.begin()
  trial.stage_coefficients([0.0, 0.0, 0.0, 0.0])
  trial.commit()
  with pytest.raises(RuntimeError, match="already consumed"):
    trial.commit()
  assert owner.ordinal == 1


def test_staging_validates_plain_values() -> None:
  owner = authoring.state_owner(_spring_system())
  trial = owner.begin()
  with pytest.raises(ValueError, match="compiled float64 shape"):
    trial.stage_state("springs", [[1.0, 2.0]])
  with pytest.raises(ValueError, match="finite"):
    trial.stage_state("springs", [[float("nan")]])
  with pytest.raises(TypeError, match="sequence of numbers"):
    trial.stage_state("springs", [["a"]])
  with pytest.raises(TypeError, match="sequence of numbers"):
    trial.stage_state("springs", [[True]])
  with pytest.raises(TypeError, match="sequence of numbers"):
    trial.stage_state("springs", "not-rows")
  with pytest.raises(KeyError, match="no state block"):
    trial.stage_state("thermal", [[1.0]])
  with pytest.raises(ValueError, match="unknown node"):
    trial.stage_displacements({99: (0.0, 0.0)})
  with pytest.raises(ValueError, match="exactly 2 components"):
    trial.stage_displacements({2: (0.1,)})
  with pytest.raises(ValueError, match="compiled float64 shape"):
    trial.stage_coefficients([0.0])
  trial.reject()


def test_plain_block_names_resolve_to_namespaced_identities() -> None:
  system = _spring_system()
  owner = authoring.state_owner(system)
  assert owner.block_ids == (
    ("cells", "domain"),
    ("springs", "student-linear-spring-v1"),
  )
  # The authored plain name and the exact namespaced identity agree.
  np.testing.assert_array_equal(
    owner.accepted_state("springs"),
    owner.accepted_state(("springs", "student-linear-spring-v1")),
  )
  np.testing.assert_array_equal(owner.accepted_state("cells"), np.zeros((1, 0)))
  assert owner.state_bytes("springs") == owner.state_bytes(
    ("springs", "student-linear-spring-v1")
  )
  with pytest.raises(KeyError, match="no state block 'thermal'"):
    owner.accepted_state("thermal")


def test_block_resolution_names_ambiguous_heads() -> None:
  # Landed composers keep block heads unique, so the ambiguity guard is only
  # reachable on hand-assembled owners; pin its message at the unit level.
  blocks = (("springs", "a"), ("springs", "b"), ("cells", "domain"))
  with pytest.raises(ValueError, match="ambiguous"):
    _resolve_block_id(blocks, "springs")
  assert _resolve_block_id(blocks, ("springs", "a")) == ("springs", "a")
  assert _resolve_block_id(blocks, "cells") == ("cells", "domain")
  with pytest.raises(KeyError, match="no state block"):
    _resolve_block_id(blocks, "thermal")


def test_facade_constructors_require_the_landed_types() -> None:
  with pytest.raises(TypeError, match="exact StateTransactionOwner"):
    authoring.StateOwner(object())  # type: ignore[arg-type]
  with pytest.raises(TypeError, match="exact StateTransaction"):
    authoring.StateTrial(object(), object())  # type: ignore[arg-type]
  with pytest.raises(TypeError, match="exact NonlinearStaticDriver"):
    authoring.NonlinearStaticSession(object())  # type: ignore[arg-type]
  with pytest.raises(TypeError, match="exact CompiledSystem"):
    authoring.state_owner(object())  # type: ignore[arg-type]


# --- nonlinear static session ---------------------------------------------------


def test_nonlinear_static_validates_the_program_shape() -> None:
  system = _spring_system()
  with pytest.raises(TypeError, match="exact CompiledSystem"):
    authoring.nonlinear_static("not-a-system")  # type: ignore[arg-type]
  with pytest.raises(TypeError, match="sequence of names"):
    authoring.nonlinear_static(system, coordinates="load")  # type: ignore[arg-type]
  with pytest.raises(TypeError, match="non-empty names or ProgramCoordinateSpec"):
    authoring.nonlinear_static(system, coordinates=(3.0,))  # type: ignore[arg-type]
  with pytest.raises(ValueError, match="must be unique"):
    authoring.nonlinear_static(system, coordinates=("load", "load"))
  with pytest.raises(ValueError, match="undeclared coordinates"):
    authoring.nonlinear_static(
      system,
      loads=(authoring.nodal_load(2, "x", 1.0, coordinate="time"),),
    )
  tie = AffineTieSpec(
    id="tie",
    slave=DofRef(node_id=2, field_id="displacement", component="x"),
    master=DofRef(node_id=1, field_id="displacement", component="x"),
    factor=1.0,
    offset=AffineValueSpec(
      coefficients=(AffineCoefficientSpec("time", 1.0),),
    ),
  )
  with pytest.raises(ValueError, match="undeclared coordinates"):
    authoring.nonlinear_static(system, constraints=(tie,))
  with pytest.raises(TypeError, match="prescribed-DOF or affine-tie"):
    authoring.nonlinear_static(system, constraints="fixed")  # type: ignore[arg-type]
  with pytest.raises(TypeError, match="exact PrescribedDofSpec or AffineTieSpec"):
    authoring.nonlinear_static(system, constraints=(object(),))  # type: ignore[arg-type]
  with pytest.raises(TypeError, match="exact NodalLoadSpec"):
    authoring.nonlinear_static(system, loads=(object(),))  # type: ignore[arg-type]
  with pytest.raises(TypeError, match="exact NonlinearStaticSettings"):
    authoring.nonlinear_static(system, settings=object())  # type: ignore[arg-type]


def test_session_run_validates_points_and_passes_program_points_through() -> None:
  session = authoring.nonlinear_static(
    _spring_system(),
    constraints=authoring.fixed(nodes=(1,))
    + authoring.fixed(nodes=(2,), components=("y",)),
    loads=(authoring.nodal_load(2, "x", 10.0),),
  )
  assert session.coordinate_names == ("load",)
  with pytest.raises(ValueError, match="missing coordinates"):
    session.run({})
  with pytest.raises(ValueError, match="unknown coordinates"):
    session.run({"load": 0.0, "time": 1.0})
  with pytest.raises(TypeError, match="exact dict"):
    session.run("home")  # type: ignore[arg-type]
  with pytest.raises(ValueError, match="finite"):
    session.run({"load": float("nan")})

  # Landed ProgramPoint values pass through unchanged.
  result = session.run(
    ProgramPoint((ProgramCoordinateValue("load", 0.0),)),
    ProgramPoint((ProgramCoordinateValue("load", 1.0),)),
  )
  assert result.status is authoring.DriverStatus.COMPLETED
  oracle = _tip_extension_oracle(10.0, stiffness=3.0)
  np.testing.assert_allclose(
    session.accepted_coefficients(),
    authoring.trial_vector(session.system, {2: (oracle, 0.0)}),
    rtol=1.0e-9,
    atol=1.0e-12,
  )
  np.testing.assert_allclose(
    session.accepted_state("springs"),
    [[oracle]],
    rtol=1.0e-9,
    atol=1.0e-12,
  )


def test_session_delegates_state_reads_and_manual_transactions() -> None:
  system = _spring_system()
  session = authoring.nonlinear_static(
    system,
    constraints=authoring.fixed(nodes=(1,))
    + authoring.fixed(nodes=(2,), components=("y",)),
    loads=(authoring.nodal_load(2, "x", 10.0),),
  )
  assert type(session.driver) is NonlinearStaticDriver
  assert type(session.state.owner) is StateTransactionOwner
  assert session.system is system
  assert session.settings == NonlinearStaticSettings()
  assert session.ordinal == 0

  result = session.run({"load": 0.0}, {"load": 1.0})
  assert result.final_generation == session.generation
  assert session.ordinal == 1

  # A manual trial over the driver's owner observes the committed state, and a
  # reject leaves it byte-identical.
  trial = session.begin()
  assert trial.ordinal == 1
  np.testing.assert_allclose(
    trial.accepted_state("springs"),
    [[_tip_extension_oracle(10.0, stiffness=3.0)]],
    rtol=1.0e-9,
    atol=1.0e-12,
  )
  snapshot = session.snapshot()
  trial.reject()
  assert session.snapshot() == snapshot


def test_declared_coordinate_specs_pass_through() -> None:
  session = authoring.nonlinear_static(
    _spring_system(),
    constraints=authoring.fixed(nodes=(1,))
    + authoring.fixed(nodes=(2,), components=("y",)),
    loads=(authoring.nodal_load(2, "x", 5.0),),
    coordinates=(ProgramCoordinateSpec(name="load", kind="load"),),
  )
  assert session.coordinate_names == ("load",)
  result = session.run({"load": 0.0}, {"load": 1.0})
  assert result.status is authoring.DriverStatus.COMPLETED


def test_affine_ties_pass_through_with_declared_coordinates() -> None:
  """Landed affine ties compose with authored fixities in one program.

  The tie prescribes the tip motion ``u2x = u1x + 0.001 * load``; with node 1
  fixed, the ramp drives the spring deterministically, so the memory state is
  the exact prescribed extension.
  """
  tie = AffineTieSpec(
    id="mirror",
    slave=DofRef(node_id=2, field_id="displacement", component="x"),
    master=DofRef(node_id=1, field_id="displacement", component="x"),
    factor=1.0,
    offset=AffineValueSpec(
      coefficients=(AffineCoefficientSpec("load", 0.001),),
    ),
  )
  session = authoring.nonlinear_static(
    _spring_system(),
    constraints=(
      (tie,)
      + authoring.fixed(nodes=(1,))
      + authoring.fixed(nodes=(2,), components=("y",))
    ),
  )
  result = session.run({"load": 0.0}, {"load": 0.5}, {"load": 1.0})
  assert result.status is authoring.DriverStatus.COMPLETED
  np.testing.assert_allclose(
    session.accepted_coefficients(),
    authoring.trial_vector(session.system, {2: (0.001, 0.0)}),
    rtol=0.0,
    atol=1.0e-15,
  )
  np.testing.assert_allclose(
    session.accepted_state("springs"),
    [[0.001]],
    rtol=0.0,
    atol=1.0e-15,
  )


# --- student persona: a stateful spring stepped with cutback and rollback -------


def test_student_stateful_spring_steps_commits_and_rolls_back() -> None:
  """Persona: a stateful student law stepped entirely through authoring calls.

  The student authors a truss bar plus a linear ground spring that remembers
  its peak extension, caps the envelope growth per committed increment, and
  steps the load in two ramps. The cap forces REJECT_STEP cutbacks; every
  rejection retries from the same committed generation, so ordinals advance
  exactly once per committed substep. The converged state matches an
  independent bisection oracle of the 1-DOF cubic equilibrium, and a target
  beyond the cutback budget fails typed with byte-identical committed state.
  """
  session = _memory_session()

  half = session.run({"load": 0.0}, {"load": 0.5})
  assert half.status is authoring.DriverStatus.COMPLETED
  assert half.failed_target_index is None
  committed = [
    record
    for record in half.records
    if record.status is authoring.SubstepStatus.COMMITTED
  ]
  assert [record.committed_ordinal for record in committed] == [1, 2, 3, 4]
  assert [record.progress for record in committed] == [0.25, 0.5, 0.75, 1.0]
  rejected = [
    record
    for record in half.records
    if record.status is authoring.SubstepStatus.REJECTED
  ]
  assert len(rejected) == 4
  assert all(
    any(
      iteration.status is authoring.EvaluationStatus.REJECT_STEP
      for iteration in record.iterations
    )
    for record in rejected
  )
  assert half.statistics.cutback_count == 4
  oracle_half = _tip_extension_oracle(0.5 * _LOAD_SCALE)
  np.testing.assert_allclose(
    session.accepted_coefficients(),
    authoring.trial_vector(session.system, {2: (oracle_half, 0.0)}),
    rtol=1.0e-8,
    atol=1.0e-12,
  )
  # The memory state tracks the peak committed extension exactly.
  np.testing.assert_allclose(
    session.accepted_state("springs"),
    [[oracle_half]],
    rtol=1.0e-8,
    atol=1.0e-12,
  )

  # A second ramp continues the same lineage: state and generations accumulate.
  full = session.run({"load": 0.5}, {"load": 1.0})
  assert full.status is authoring.DriverStatus.COMPLETED
  committed = [
    record
    for record in full.records
    if record.status is authoring.SubstepStatus.COMMITTED
  ]
  assert [record.committed_ordinal for record in committed] == [5, 6, 7, 8]
  assert full.statistics.committed_substep_count == 8
  assert full.statistics.cutback_count == 8
  oracle_full = _tip_extension_oracle(_LOAD_SCALE)
  np.testing.assert_allclose(
    session.accepted_coefficients(),
    authoring.trial_vector(session.system, {2: (oracle_full, 0.0)}),
    rtol=1.0e-8,
    atol=1.0e-12,
  )
  np.testing.assert_allclose(
    session.accepted_state("springs"),
    [[oracle_full]],
    rtol=1.0e-8,
    atol=1.0e-12,
  )
  assert [record.ordinal for record in session.state.history] == list(range(1, 9))

  # Forced rollback: the far target needs halving deeper than the cutback
  # budget, so the run fails typed and the committed state is byte-identical.
  before = session.snapshot()
  failed = session.run({"load": 1.0}, {"load": 64.0})
  assert failed.status is authoring.DriverStatus.STEP_FAILED
  assert failed.failed_target_index == 0
  assert [record.status for record in failed.records] == [
    authoring.SubstepStatus.REJECTED
  ] * 6 + [authoring.SubstepStatus.FAILED]
  assert all(record.committed_ordinal is None for record in failed.records)
  assert all(
    any(
      iteration.status is authoring.EvaluationStatus.REJECT_STEP
      for iteration in record.iterations
    )
    for record in failed.records
  )
  assert session.snapshot() == before
  assert session.ordinal == 8
  np.testing.assert_allclose(
    session.accepted_state("springs"),
    [[oracle_full]],
    rtol=1.0e-8,
    atol=1.0e-12,
  )
