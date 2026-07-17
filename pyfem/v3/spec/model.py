"""Immutable authored values for a normalized finite-element model."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from pyfem.v3.spec.diagnostics import SourceContext

type SpecId = str | int
type MaterialParameterValue = (
  str | bool | int | float | tuple[MaterialParameterValue, ...]
)


def _normalize_id(value: object) -> object:
  if type(value) is str:
    return str.strip(value)
  if type(value) is int:
    return value
  return value


def _normalize_text(value: object) -> object:
  if type(value) is str:
    return str.strip(value)
  return value


def _normalize_number(value: object) -> object:
  if type(value) is float:
    return value
  if type(value) is int:
    try:
      return float(value)
    except OverflowError:
      return value
  return value


def _normalize_sequence(
  value: object,
  normalize_item: Callable[[object], object],
) -> object:
  if type(value) is list or type(value) is tuple:
    return tuple(normalize_item(item) for item in value)
  return value


def _freeze_sequence(value: object) -> object:
  if type(value) is list or type(value) is tuple:
    return tuple(value)
  return value


def _freeze_parameter_value(value: object) -> object:
  """Own nested list/tuple parameter values as immutable tuples."""
  try:
    return _freeze_parameter_value_inner(value, set())
  except RecursionError:
    return value


def _freeze_parameter_value_inner(value: object, active: set[int]) -> object:
  if type(value) is list or type(value) is tuple:
    identity = id(value)
    if identity in active:
      return value
    active.add(identity)
    frozen = tuple(_freeze_parameter_value_inner(item, active) for item in value)
    active.remove(identity)
    return frozen
  if type(value) is str:
    return str.strip(value)
  if type(value) is bool or type(value) is int or type(value) is float:
    return value
  return value


@dataclass(frozen=True, slots=True)
class NodeSpec:
  """One source node and its reference coordinates."""

  id: SpecId
  coordinates: tuple[float, ...]
  source: SourceContext = field(default_factory=SourceContext)

  def __post_init__(self) -> None:
    object.__setattr__(self, "id", _normalize_id(self.id))
    object.__setattr__(
      self,
      "coordinates",
      _normalize_sequence(self.coordinates, _normalize_number),
    )


@dataclass(frozen=True, slots=True)
class CellSpec:
  """One source cell with block-local geometry connectivity."""

  id: SpecId
  node_ids: tuple[SpecId, ...]
  source: SourceContext = field(default_factory=SourceContext)

  def __post_init__(self) -> None:
    object.__setattr__(self, "id", _normalize_id(self.id))
    object.__setattr__(
      self,
      "node_ids",
      _normalize_sequence(self.node_ids, _normalize_id),
    )


@dataclass(frozen=True, slots=True)
class CellBlockSpec:
  """Cells sharing one explicit topology and geometry interpolation."""

  id: SpecId
  reference_topology: str
  topological_dimension: int
  embedding_dimension: int
  geometry_interpolation: str
  cells: tuple[CellSpec, ...]
  source: SourceContext = field(default_factory=SourceContext)

  def __post_init__(self) -> None:
    object.__setattr__(self, "id", _normalize_id(self.id))
    object.__setattr__(
      self,
      "reference_topology",
      _normalize_text(self.reference_topology),
    )
    object.__setattr__(
      self,
      "geometry_interpolation",
      _normalize_text(self.geometry_interpolation),
    )
    object.__setattr__(self, "cells", _freeze_sequence(self.cells))


@dataclass(frozen=True, slots=True)
class MeshSpec:
  """Authored nodes and several explicit, independently shaped cell blocks."""

  nodes: tuple[NodeSpec, ...]
  cell_blocks: tuple[CellBlockSpec, ...]
  source: SourceContext = field(default_factory=SourceContext)

  def __post_init__(self) -> None:
    object.__setattr__(self, "nodes", _freeze_sequence(self.nodes))
    object.__setattr__(self, "cell_blocks", _freeze_sequence(self.cell_blocks))


@dataclass(frozen=True, slots=True)
class FieldSpec:
  """Named field components at one explicit storage location."""

  id: SpecId
  components: tuple[str, ...]
  location: str
  source: SourceContext = field(default_factory=SourceContext)

  def __post_init__(self) -> None:
    object.__setattr__(self, "id", _normalize_id(self.id))
    object.__setattr__(
      self,
      "components",
      _normalize_sequence(self.components, _normalize_text),
    )
    object.__setattr__(self, "location", _normalize_text(self.location))


@dataclass(frozen=True, slots=True)
class MaterialParameterSpec:
  """One named immutable authored material parameter value."""

  name: str
  value: MaterialParameterValue
  source: SourceContext = field(default_factory=SourceContext)

  def __post_init__(self) -> None:
    object.__setattr__(self, "name", _normalize_text(self.name))
    object.__setattr__(self, "value", _freeze_parameter_value(self.value))


@dataclass(frozen=True, slots=True)
class MaterialSpec:
  """Explicit material descriptor and its authored parameters."""

  id: SpecId
  model: str
  parameters: tuple[MaterialParameterSpec, ...] = ()
  source: SourceContext = field(default_factory=SourceContext)

  def __post_init__(self) -> None:
    object.__setattr__(self, "id", _normalize_id(self.id))
    object.__setattr__(self, "model", _normalize_text(self.model))
    object.__setattr__(self, "parameters", _freeze_sequence(self.parameters))


@dataclass(frozen=True, slots=True)
class CellRef:
  """Unambiguous reference to a cell inside a named cell block."""

  block_id: SpecId
  cell_id: SpecId

  def __post_init__(self) -> None:
    object.__setattr__(self, "block_id", _normalize_id(self.block_id))
    object.__setattr__(self, "cell_id", _normalize_id(self.cell_id))


@dataclass(frozen=True, slots=True)
class RegionSpec:
  """Cell selection with explicit field, material, and kernel descriptors."""

  id: SpecId
  cell_refs: tuple[CellRef, ...]
  field_ids: tuple[SpecId, ...]
  material_id: SpecId
  formulation: str
  quadrature: str
  source: SourceContext = field(default_factory=SourceContext)

  def __post_init__(self) -> None:
    object.__setattr__(self, "id", _normalize_id(self.id))
    object.__setattr__(self, "cell_refs", _freeze_sequence(self.cell_refs))
    object.__setattr__(
      self,
      "field_ids",
      _normalize_sequence(self.field_ids, _normalize_id),
    )
    object.__setattr__(
      self,
      "material_id",
      _normalize_id(self.material_id),
    )
    object.__setattr__(self, "formulation", _normalize_text(self.formulation))
    object.__setattr__(self, "quadrature", _normalize_text(self.quadrature))


@dataclass(frozen=True, slots=True)
class ModelSpec:
  """Complete authored physical model intent, independent of any program."""

  mesh: MeshSpec
  fields: tuple[FieldSpec, ...]
  materials: tuple[MaterialSpec, ...]
  regions: tuple[RegionSpec, ...]
  source: SourceContext = field(default_factory=SourceContext)

  def __post_init__(self) -> None:
    object.__setattr__(self, "fields", _freeze_sequence(self.fields))
    object.__setattr__(self, "materials", _freeze_sequence(self.materials))
    object.__setattr__(self, "regions", _freeze_sequence(self.regions))
