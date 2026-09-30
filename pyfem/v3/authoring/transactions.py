"""Transaction-level ergonomics over the M12 state owner and the M18 driver.

The landed state path is explicit in the jax fashion: an owner holds the
accepted state, a transaction stages one trial, and the driver schedules
Newton/cutback substeps over it. This module keeps that explicitness while
removing the plumbing: plain nodal-displacement mappings and arrays in, plain
arrays and bytes out. The facades hold no authoritative state and re-implement
no schedule logic — every read delegates to the owner's detached snapshots and
every mutation goes through the landed begin/stage/commit protocol.

A driver-free stateful workflow is a handful of calls::

    owner = state_owner(system)
    trial = owner.begin()
    result = evaluate(system, {1: (0.5, 0.0)}, operator=1,
                      accepted_state=trial.accepted_state("springs"))
    trial.stage_displacements({1: (0.5, 0.0)})
    trial.stage_state("springs", result.trial_state.values)
    trial.commit()

A stepped stateful analysis wraps the landed nonlinear driver::

    session = nonlinear_static(
      system,
      constraints=fixed(nodes=(1,)),
      loads=(nodal_load(2, "x", 100.0),),
    )
    result = session.run({"load": 0.0}, {"load": 0.5}, {"load": 1.0})
    before = session.snapshot()
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from pyfem.v3.authoring.evaluate import trial_vector
from pyfem.v3.constraints import compile_constraint_map
from pyfem.v3.driver import (
  NonlinearStaticDriver,
  NonlinearStaticResult,
  NonlinearStaticSettings,
)
from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.identity import StateGeneration
from pyfem.v3.model.operator import SemanticId, StateCodec
from pyfem.v3.model.system import CompiledSystem
from pyfem.v3.spec.diagnostics import SourceContext
from pyfem.v3.spec.program import (
  AffineTieSpec,
  NodalLoadSpec,
  PrescribedDofSpec,
  ProgramConstraintSpec,
  ProgramCoordinateSpec,
  ProgramCoordinateValue,
  ProgramPoint,
)
from pyfem.v3.state import GenerationRecord, StateTransaction, StateTransactionOwner


def _finite_number(value: object, *, label: str) -> float:
  if type(value) not in (int, float):
    msg = f"{label} must be an exact number"
    raise TypeError(msg)
  try:
    converted = float(value)
  except OverflowError:
    msg = f"{label} must fit finite float64"
    raise ValueError(msg) from None
  if not math.isfinite(converted):
    msg = f"{label} must be finite"
    raise ValueError(msg)
  return converted


def _float64_array(value: object, *, label: str) -> np.ndarray | FinalizedArray:
  if type(value) in (np.ndarray, FinalizedArray):
    return value
  if isinstance(value, Sequence) and not isinstance(value, str):
    try:
      raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
      msg = f"{label} must be an array or a sequence of numbers"
      raise TypeError(msg) from exc
    if raw.dtype.kind not in "iuf":
      msg = f"{label} must be an array or a sequence of numbers"
      raise TypeError(msg)
    return np.asarray(value, dtype=np.float64)
  msg = f"{label} must be an array or a sequence of numbers"
  raise TypeError(msg)


def _detached_copy(array: FinalizedArray) -> np.ndarray:
  return np.array(array.values, dtype=np.float64, order="C", copy=True)


def _resolve_block_id(
  block_ids: tuple[SemanticId, ...],
  ref: SemanticId,
) -> SemanticId:
  """Resolve a plain block reference to its bound namespaced identity.

  Exact identities pass through; a plain name resolves to the unique bound
  block whose namespaced head matches (``"springs"`` for
  ``("springs", <implementation>)``), so authors reference blocks by the ids
  they declared.
  """
  if ref in block_ids:
    return ref
  matches = tuple(
    block_id
    for block_id in block_ids
    if type(block_id) is tuple and len(block_id) > 0 and block_id[0] == ref
  )
  if len(matches) == 1:
    return matches[0]
  if matches:
    msg = f"state block reference {ref!r} is ambiguous across {list(matches)}"
    raise ValueError(msg)
  msg = f"the owner has no state block {ref!r}; available: {list(block_ids)}"
  raise KeyError(msg)


@dataclass(frozen=True, slots=True)
class StateSnapshot:
  """Byte-exact accepted-state capture: equality is byte-identity.

  Carries the accepted generation ordinal, the physical coefficient vector as
  raw bytes, each state block's codec-encoded payload, and the committed
  history metadata — never mutable buffers.
  """

  ordinal: int
  physical: bytes
  state: tuple[tuple[SemanticId, bytes], ...]
  history: tuple[GenerationRecord, ...]


class StateTrial:
  """One open transaction with plain-value staging ergonomics.

  Issued only by :meth:`StateOwner.begin`. Staging accepts plain nodal
  mappings and arrays; exactly one of ``commit`` or ``reject`` consumes the
  trial, exactly as the landed protocol requires.
  """

  __slots__ = ("_owner", "_transaction")

  def __init__(
    self,
    transaction: StateTransaction,
    owner: StateOwner,
  ) -> None:
    if type(transaction) is not StateTransaction:
      msg = "StateTrial wraps an exact StateTransaction"
      raise TypeError(msg)
    if type(owner) is not StateOwner:
      msg = "StateTrial requires the issuing StateOwner"
      raise TypeError(msg)
    self._transaction = transaction
    self._owner = owner

  @property
  def transaction(self) -> StateTransaction:
    """Return the wrapped landed transaction (the research escape hatch)."""
    return self._transaction

  @property
  def generation(self) -> StateGeneration:
    """Return the accepted generation this trial is rooted at."""
    return self._transaction.generation

  @property
  def ordinal(self) -> int:
    """Return the rooted accepted generation ordinal."""
    return self._transaction.generation.ordinal

  def accepted_coefficients(self) -> np.ndarray:
    """Return the accepted physical coefficients as a detached plain array."""
    return _detached_copy(self._transaction.accepted_physical)

  def accepted_state(self, block_id: SemanticId) -> np.ndarray:
    """Return one block's accepted state rows as a detached plain array."""
    resolved = _resolve_block_id(self._owner.block_ids, block_id)
    return _detached_copy(self._transaction.accepted_state(resolved))

  def stage_coefficients(self, values: object) -> None:
    """Stage the trial physical coefficient vector for commit."""
    self._transaction.stage_physical(
      _float64_array(values, label="trial physical coefficients")
    )

  def stage_displacements(
    self,
    displacements: Mapping[SemanticId, Sequence[float]] | None,
    *,
    space_id: SemanticId | None = None,
  ) -> None:
    """Stage trial physical coefficients from plain nodal displacements.

    The mapping is expanded onto the global coefficient vector exactly like
    :func:`pyfem.v3.authoring.trial_vector`: every unmentioned coefficient
    stages as zero.
    """
    self._transaction.stage_physical(
      trial_vector(self._owner.system, displacements, space_id=space_id)
    )

  def stage_state(self, block_id: SemanticId, rows: object) -> None:
    """Stage trial state rows of one operator block for commit."""
    resolved = _resolve_block_id(self._owner.block_ids, block_id)
    self._transaction.stage_state(
      resolved,
      _float64_array(rows, label="trial operator state rows"),
    )

  def commit(self) -> None:
    """Atomically commit every staged value as one generation transition."""
    self._transaction.commit()

  def reject(self) -> None:
    """Discard the trial, leaving committed state byte-identical."""
    self._transaction.reject()


class StateOwner:
  """Ergonomic facade over one landed :class:`StateTransactionOwner`.

  Holds no state of its own: reads delegate to the owner's detached read-only
  snapshots and mutations go through the landed begin/stage/commit protocol.
  """

  __slots__ = ("_owner",)

  def __init__(self, owner: StateTransactionOwner) -> None:
    if type(owner) is not StateTransactionOwner:
      msg = "StateOwner wraps an exact StateTransactionOwner"
      raise TypeError(msg)
    self._owner = owner

  @property
  def owner(self) -> StateTransactionOwner:
    """Return the wrapped landed owner (the research escape hatch)."""
    return self._owner

  @property
  def system(self) -> CompiledSystem:
    """Return the compiled system whose state the owner holds."""
    return self._owner.system

  @property
  def generation(self) -> StateGeneration:
    """Return the current accepted generation identity."""
    return self._owner.generation

  @property
  def ordinal(self) -> int:
    """Return the current accepted generation ordinal."""
    return self._owner.generation.ordinal

  @property
  def block_ids(self) -> tuple[SemanticId, ...]:
    """Return the bound operator state block identities."""
    return self._owner.block_ids

  @property
  def history(self) -> tuple[GenerationRecord, ...]:
    """Return the committed-generation metadata records; never state data."""
    return self._owner.history

  def accepted_coefficients(self) -> np.ndarray:
    """Return the accepted physical coefficients as a detached plain array."""
    return _detached_copy(self._owner.accepted_physical())

  def accepted_state(self, block_id: SemanticId) -> np.ndarray:
    """Return one block's accepted state rows as a detached plain array."""
    resolved = _resolve_block_id(self._owner.block_ids, block_id)
    return _detached_copy(self._owner.accepted_state(resolved))

  def state_bytes(self, block_id: SemanticId) -> bytes:
    """Return one block's accepted state rows codec-encoded as bytes."""
    return self._owner.encode_state(_resolve_block_id(self._owner.block_ids, block_id))

  def snapshot(self) -> StateSnapshot:
    """Capture the accepted state for byte-identity comparisons."""
    owner = self._owner
    return StateSnapshot(
      ordinal=owner.generation.ordinal,
      physical=owner.accepted_physical().values.tobytes(),
      state=tuple(
        (block_id, owner.encode_state(block_id)) for block_id in owner.block_ids
      ),
      history=owner.history,
    )

  def begin(self) -> StateTrial:
    """Open one trial transaction rooted at the current accepted generation."""
    return StateTrial(self._owner.begin(), self)


def state_owner(
  system: CompiledSystem,
  *,
  codecs: tuple[StateCodec, ...] = (),
) -> StateOwner:
  """Create an accepted-state owner facade for one compiled system."""
  return StateOwner(StateTransactionOwner(system, codecs=codecs))


class NonlinearStaticSession:
  """Stepping facade over one landed :class:`NonlinearStaticDriver`.

  ``run`` translates plain ``{coordinate: value}`` mappings into exact program
  points and forwards them to the driver's Newton/cutback schedule; state
  reads and manual transactions delegate to the driver-owned M12 owner through
  :class:`StateOwner`. The landed result records are returned unchanged.
  """

  __slots__ = ("_driver", "_state")

  def __init__(self, driver: NonlinearStaticDriver) -> None:
    if type(driver) is not NonlinearStaticDriver:
      msg = "NonlinearStaticSession wraps an exact NonlinearStaticDriver"
      raise TypeError(msg)
    self._driver = driver
    self._state = StateOwner(driver.owner)

  @property
  def driver(self) -> NonlinearStaticDriver:
    """Return the wrapped landed driver (the research escape hatch)."""
    return self._driver

  @property
  def state(self) -> StateOwner:
    """Return the accepted-state facade over the driver's owner."""
    return self._state

  @property
  def system(self) -> CompiledSystem:
    """Return the compiled system being stepped."""
    return self._driver.owner.system

  @property
  def settings(self) -> NonlinearStaticSettings:
    """Return the exact numeric policy."""
    return self._driver.settings

  @property
  def coordinate_names(self) -> tuple[str, ...]:
    """Return the declared program coordinate names, in compiled order."""
    return self._driver.plan.coordinate_names

  @property
  def generation(self) -> StateGeneration:
    """Return the current accepted generation identity."""
    return self._state.generation

  @property
  def ordinal(self) -> int:
    """Return the current accepted generation ordinal."""
    return self._state.ordinal

  def accepted_coefficients(self) -> np.ndarray:
    """Return the accepted physical coefficients as a detached plain array."""
    return self._state.accepted_coefficients()

  def accepted_state(self, block_id: SemanticId) -> np.ndarray:
    """Return one block's accepted state rows as a detached plain array."""
    return self._state.accepted_state(block_id)

  def state_bytes(self, block_id: SemanticId) -> bytes:
    """Return one block's accepted state rows codec-encoded as bytes."""
    return self._state.state_bytes(block_id)

  def snapshot(self) -> StateSnapshot:
    """Capture the accepted state for byte-identity comparisons."""
    return self._state.snapshot()

  def begin(self) -> StateTrial:
    """Open one manual trial transaction over the driver's owner."""
    return self._state.begin()

  def _program_point(self, value: object, *, label: str) -> ProgramPoint:
    if type(value) is ProgramPoint:
      return value
    if type(value) is not dict:
      msg = f"{label} must be an exact dict of coordinate values or a ProgramPoint"
      raise TypeError(msg)
    declared = self.coordinate_names
    missing = [name for name in declared if name not in value]
    unknown = [name for name in value if name not in declared]
    if missing or unknown:
      problems = []
      if missing:
        problems.append(f"missing coordinates {missing}")
      if unknown:
        problems.append(f"unknown coordinates {unknown}")
      msg = (
        f"{label} must bind exactly the declared coordinates "
        f"{list(declared)}: {'; '.join(problems)}"
      )
      raise ValueError(msg)
    return ProgramPoint(
      tuple(
        ProgramCoordinateValue(
          name,
          _finite_number(value[name], label=f"{label} coordinate {name!r}"),
        )
        for name in declared
      )
    )

  def run(
    self,
    base: dict[str, float] | ProgramPoint,
    *targets: dict[str, float] | ProgramPoint,
  ) -> NonlinearStaticResult:
    """Advance the committed state through the target-point schedule.

    ``base`` and each target bind every declared coordinate by name; the
    landed result (typed statuses, iteration trail, statistics, lineage) is
    returned unchanged.
    """
    return self._driver.run(
      base_point=self._program_point(base, label="base point"),
      target_points=tuple(
        self._program_point(target, label=f"target point {index}")
        for index, target in enumerate(targets)
      ),
    )


def _coordinate_declaration(
  value: object,
) -> ProgramCoordinateSpec:
  if type(value) is ProgramCoordinateSpec:
    return value
  if type(value) is str and value:
    return ProgramCoordinateSpec(
      name=value,
      kind="load",
      source=SourceContext(source=f"authoring.nonlinear_static:coordinate:{value}"),
    )
  msg = "program coordinates must be non-empty names or ProgramCoordinateSpec values"
  raise TypeError(msg)


def _constraint_tuple(values: object) -> tuple[ProgramConstraintSpec, ...]:
  if not isinstance(values, Sequence) or isinstance(values, str):
    msg = "constraints must be a sequence of prescribed-DOF or affine-tie specs"
    raise TypeError(msg)
  constraints: list[ProgramConstraintSpec] = []
  for value in values:
    if type(value) not in (PrescribedDofSpec, AffineTieSpec):
      msg = "constraints must be exact PrescribedDofSpec or AffineTieSpec values"
      raise TypeError(msg)
    constraints.append(value)
  return tuple(constraints)


def _load_tuple(values: object) -> tuple[NodalLoadSpec, ...]:
  if not isinstance(values, Sequence) or isinstance(values, str):
    msg = "loads must be a sequence of nodal load specs"
    raise TypeError(msg)
  loads: list[NodalLoadSpec] = []
  for value in values:
    if type(value) is not NodalLoadSpec:
      msg = "loads must be exact NodalLoadSpec values"
      raise TypeError(msg)
    loads.append(value)
  return tuple(loads)


def _referenced_coordinates(
  constraints: tuple[ProgramConstraintSpec, ...],
  loads: tuple[NodalLoadSpec, ...],
) -> set[str]:
  referenced: set[str] = set()
  for constraint in constraints:
    affine = (
      constraint.value if type(constraint) is PrescribedDofSpec else constraint.offset
    )
    referenced.update(coefficient.coordinate for coefficient in affine.coefficients)
  for load in loads:
    referenced.update(coefficient.coordinate for coefficient in load.value.coefficients)
  return referenced


def nonlinear_static(
  system: CompiledSystem,
  *,
  constraints: Sequence[ProgramConstraintSpec] = (),
  loads: Sequence[NodalLoadSpec] = (),
  coordinates: Sequence[str | ProgramCoordinateSpec] = ("load",),
  settings: NonlinearStaticSettings | None = None,
) -> NonlinearStaticSession:
  """Build a stepping session over a compiled system.

  ``constraints`` and ``loads`` accept the plain-authored declarations from
  :func:`pyfem.v3.authoring.fixed` and :func:`pyfem.v3.authoring.nodal_load`
  (or any landed spec values); ``coordinates`` names the declared program
  coordinates (default ``("load",)``, kind ``"load"``), and every coordinate
  referenced by a load or constraint must be declared. ``settings`` is the
  landed numeric policy, defaulted when omitted.
  """
  if type(system) is not CompiledSystem:
    msg = "nonlinear_static requires an exact CompiledSystem"
    raise TypeError(msg)
  if not isinstance(coordinates, Sequence) or isinstance(coordinates, str):
    msg = "coordinates must be a sequence of names or ProgramCoordinateSpec values"
    raise TypeError(msg)
  declared = tuple(_coordinate_declaration(value) for value in coordinates)
  names = [coordinate.name for coordinate in declared]
  if len(set(names)) != len(names):
    msg = f"program coordinates must be unique, declared {names}"
    raise ValueError(msg)
  constraint_tuple = _constraint_tuple(constraints)
  load_tuple = _load_tuple(loads)
  referenced = _referenced_coordinates(constraint_tuple, load_tuple)
  undeclared = sorted(referenced.difference(names))
  if undeclared:
    msg = f"loads or constraints reference undeclared coordinates {undeclared}"
    raise ValueError(msg)
  if settings is None:
    policy = NonlinearStaticSettings()
  elif type(settings) is NonlinearStaticSettings:
    policy = settings
  else:
    msg = "nonlinear_static settings must be exact NonlinearStaticSettings"
    raise TypeError(msg)
  coordinate_map = compile_constraint_map(
    system,
    constraints=constraint_tuple,
    coordinates=declared,
  )
  driver = NonlinearStaticDriver(system, coordinate_map, load_tuple, policy)
  return NonlinearStaticSession(driver)
