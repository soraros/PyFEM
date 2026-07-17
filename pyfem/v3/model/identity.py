"""Strict in-process identity and accepted-state generation primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID, uuid4


class IdentityMismatchError(ValueError):
  """Raised when live objects do not belong to the same compiled instance."""


class GenerationMismatchError(ValueError):
  """Raised when accepted-state generations are unrelated or unexpected."""


@dataclass(frozen=True, slots=True, init=False, repr=False)
class InstanceId:
  """Opaque, process-local identity for one live compiled owner.

  An instance ID deliberately carries no content meaning and is never a restart
  compatibility key. Constructing this type always creates a fresh identity.
  """

  _token: UUID = field(repr=False)

  def __init__(self) -> None:
    object.__setattr__(self, "_token", uuid4())

  def __repr__(self) -> str:
    return "InstanceId(<opaque>)"


@dataclass(frozen=True, slots=True, init=False)
class StateGeneration:
  """Monotonic accepted-state identity in an opaque persistent lineage.

  The lineage is intentionally distinct from ``InstanceId``: a later verified
  restore may preserve accepted-state provenance while creating a fresh live
  instance. This packet does not define that restore boundary.
  """

  _lineage: UUID = field(repr=False)
  ordinal: int

  def __init__(self) -> None:
    object.__setattr__(self, "_lineage", uuid4())
    object.__setattr__(self, "ordinal", 0)

  @classmethod
  def initial(cls) -> StateGeneration:
    """Create the initial identity for one accepted-state lineage."""
    return cls()

  @classmethod
  def _from_lineage(cls, lineage: UUID, ordinal: int) -> StateGeneration:
    generation = object.__new__(cls)
    object.__setattr__(generation, "_lineage", lineage)
    object.__setattr__(generation, "ordinal", ordinal)
    return generation

  def next_accepted(self) -> StateGeneration:
    """Return the identity for the next atomic accepted transition."""
    return self._from_lineage(self._lineage, self.ordinal + 1)


def require_same_instance(
  expected: object,
  actual: object,
  *,
  context: str = "live composition",
) -> None:
  """Require exact live instance identity, failing closed for foreign values."""
  if (
    not isinstance(expected, InstanceId)
    or not isinstance(actual, InstanceId)
    or expected != actual
  ):
    msg = f"{context} requires the exact same live instance"
    raise IdentityMismatchError(msg)


def require_same_generation(
  expected: object,
  actual: object,
  *,
  context: str = "accepted state",
) -> None:
  """Require one exact accepted-state generation identity."""
  if not isinstance(expected, StateGeneration) or not isinstance(
    actual,
    StateGeneration,
  ):
    msg = f"{context} requires an exact accepted-state generation"
    raise GenerationMismatchError(msg)
  if expected._lineage != actual._lineage or expected.ordinal != actual.ordinal:
    msg = f"{context} requires an exact accepted-state generation"
    raise GenerationMismatchError(msg)


def require_generation_successor(
  base: object,
  candidate: object,
  *,
  context: str = "state commit",
) -> None:
  """Require ``candidate`` to be the next generation in ``base``'s lineage."""
  if not isinstance(base, StateGeneration) or not isinstance(
    candidate,
    StateGeneration,
  ):
    msg = f"{context} requires a related next accepted-state generation"
    raise GenerationMismatchError(msg)
  if base._lineage != candidate._lineage or candidate.ordinal != base.ordinal + 1:
    msg = f"{context} requires a related next accepted-state generation"
    raise GenerationMismatchError(msg)
