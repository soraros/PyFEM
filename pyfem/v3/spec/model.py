"""Immutable authored values for a normalized finite-element model."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral, Real

from pyfem.v3.spec.diagnostics import SourceContext

type SpecId = str | int
type MaterialParameterValue = (
  str | bool | int | float | tuple[MaterialParameterValue, ...]
)


def _normalize_id(value: SpecId) -> SpecId:
  if isinstance(value, str):
    return value.strip()
  if isinstance(value, Integral) and not isinstance(value, bool):
    return int(value)
  return value


def _normalize_text(value: str) -> str:
  if isinstance(value, str):
    return value.strip()
  return value


def _normalize_number(value: object) -> object:
  if isinstance(value, Real) and not isinstance(value, bool):
    return float(value)
  return value


def _freeze_parameter_value(value: object) -> object:
  """Own nested list/tuple parameter values as immutable tuples."""
  if isinstance(value, list | tuple):
    return tuple(_freeze_parameter_value(item) for item in value)
  if isinstance(value, str):
    return value.strip()
  if isinstance(value, Integral) and not isinstance(value, bool):
    return int(value)
  if isinstance(value, Real) and not isinstance(value, bool):
    return float(value)
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
      tuple(_normalize_number(item) for item in self.coordinates),
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
      tuple(_normalize_id(item) for item in self.node_ids),
    )


@dataclass(frozen=True, slots=True)
class CellBlockSpec:
  """Cells sharing one explicit topology and geometry interpolation."""

  id: SpecId
  reference_topology: str
  topological_dimension: int
  embedding_dimension: int
  geometry_interpolation: str
  geometry_node_count: int
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
    object.__setattr__(self, "cells", tuple(self.cells))


@dataclass(frozen=True, slots=True)
class MeshSpec:
  """Authored nodes and several explicit, independently shaped cell blocks."""

  nodes: tuple[NodeSpec, ...]
  cell_blocks: tuple[CellBlockSpec, ...]
  source: SourceContext = field(default_factory=SourceContext)

  def __post_init__(self) -> None:
    object.__setattr__(self, "nodes", tuple(self.nodes))
    object.__setattr__(self, "cell_blocks", tuple(self.cell_blocks))


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
      tuple(_normalize_text(item) for item in self.components),
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
    object.__setattr__(self, "parameters", tuple(self.parameters))


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
    object.__setattr__(self, "cell_refs", tuple(self.cell_refs))
    object.__setattr__(
      self,
      "field_ids",
      tuple(_normalize_id(item) for item in self.field_ids),
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
    object.__setattr__(self, "fields", tuple(self.fields))
    object.__setattr__(self, "materials", tuple(self.materials))
    object.__setattr__(self, "regions", tuple(self.regions))
