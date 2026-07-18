"""Immutable authored values for a finite-element loading program."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pyfem.v3.spec.diagnostics import SourceContext

type ProgramCoordinateKind = Literal["time", "load", "continuation"]
type ProgramConstraintSpec = PrescribedDofSpec | AffineTieSpec


def _normalize_id(value: object) -> object:
  if type(value) is str:
    return str.strip(value)
  return value


def _normalize_text(value: object) -> object:
  if type(value) is str:
    return str.strip(value)
  return value


def _freeze_sequence(value: object) -> object:
  if type(value) is list or type(value) is tuple:
    return tuple(value)
  return value


@dataclass(frozen=True, slots=True)
class ProgramCoordinateSpec:
  """One declared structured coordinate of a compiled program."""

  name: str
  kind: ProgramCoordinateKind
  source: SourceContext = field(default_factory=SourceContext)

  def __post_init__(self) -> None:
    object.__setattr__(self, "name", _normalize_text(self.name))
    object.__setattr__(self, "kind", _normalize_text(self.kind))


@dataclass(frozen=True, slots=True)
class DofRef:
  """Authored semantic reference to one node field component."""

  node_id: str | int
  field_id: str | int
  component: str

  def __post_init__(self) -> None:
    object.__setattr__(self, "node_id", _normalize_id(self.node_id))
    object.__setattr__(self, "field_id", _normalize_id(self.field_id))
    object.__setattr__(self, "component", _normalize_text(self.component))


@dataclass(frozen=True, slots=True)
class AffineCoefficientSpec:
  """Coefficient multiplying one named program coordinate."""

  coordinate: str
  coefficient: int | float
  source: SourceContext = field(default_factory=SourceContext)

  def __post_init__(self) -> None:
    object.__setattr__(self, "coordinate", _normalize_text(self.coordinate))


@dataclass(frozen=True, slots=True)
class AffineValueSpec:
  """One constant plus named-coordinate affine coefficients."""

  constant: int | float = 0.0
  coefficients: tuple[AffineCoefficientSpec, ...] = ()
  source: SourceContext = field(default_factory=SourceContext)

  def __post_init__(self) -> None:
    object.__setattr__(self, "coefficients", _freeze_sequence(self.coefficients))


@dataclass(frozen=True, slots=True)
class PrescribedDofSpec:
  """Prescribed value for one semantic degree of freedom."""

  id: str | int
  target: DofRef
  value: AffineValueSpec
  source: SourceContext = field(default_factory=SourceContext)

  def __post_init__(self) -> None:
    object.__setattr__(self, "id", _normalize_id(self.id))


@dataclass(frozen=True, slots=True)
class AffineTieSpec:
  """One-master relation ``u_slave = factor*u_master + offset``."""

  id: str | int
  slave: DofRef
  master: DofRef
  factor: int | float
  offset: AffineValueSpec
  source: SourceContext = field(default_factory=SourceContext)

  def __post_init__(self) -> None:
    object.__setattr__(self, "id", _normalize_id(self.id))


@dataclass(frozen=True, slots=True)
class NodalLoadSpec:
  """One attributable full-space nodal-force declaration."""

  id: str | int
  target: DofRef
  value: AffineValueSpec
  source: SourceContext = field(default_factory=SourceContext)

  def __post_init__(self) -> None:
    object.__setattr__(self, "id", _normalize_id(self.id))


@dataclass(frozen=True, slots=True)
class ProgramSpec:
  """Authored constraints and nodal loads independent of solver state."""

  coordinates: tuple[ProgramCoordinateSpec, ...] = ()
  constraints: tuple[ProgramConstraintSpec, ...] = ()
  loads: tuple[NodalLoadSpec, ...] = ()
  source: SourceContext = field(default_factory=SourceContext)

  def __post_init__(self) -> None:
    object.__setattr__(self, "coordinates", _freeze_sequence(self.coordinates))
    object.__setattr__(self, "constraints", _freeze_sequence(self.constraints))
    object.__setattr__(self, "loads", _freeze_sequence(self.loads))


@dataclass(frozen=True, slots=True)
class ProgramCoordinateValue:
  """One named coordinate value at an evaluation point."""

  name: str
  value: int | float

  def __post_init__(self) -> None:
    object.__setattr__(self, "name", _normalize_text(self.name))


@dataclass(frozen=True, slots=True)
class ProgramPoint:
  """Exact complete binding for all declared program coordinates."""

  values: tuple[ProgramCoordinateValue, ...] = ()

  def __post_init__(self) -> None:
    object.__setattr__(self, "values", _freeze_sequence(self.values))
