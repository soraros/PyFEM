"""Immutable compiled-model carriers for the first explicit Q8 domain block."""

from __future__ import annotations

from dataclasses import dataclass

from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.identity import InstanceId
from pyfem.v3.model.provenance import CanonicalManifest, ContentFingerprint
from pyfem.v3.model.registry import RegistrySnapshot

type SemanticId = str | int
type EntitySemanticId = SemanticId | tuple[SemanticId | int, ...]


def _same_semantic_id(left: EntitySemanticId, right: object) -> bool:
  if type(left) is tuple:
    if type(right) is not tuple or len(left) != len(right):
      return False
    return all(
      _same_semantic_id(left_item, right_item)
      for left_item, right_item in zip(left, right, strict=True)
    )
  return type(left) is type(right) and left == right


@dataclass(frozen=True, slots=True, eq=False)
class CompiledSource:
  """Detached source location retained for diagnostics and provenance."""

  source: str
  line: int | None
  column: int | None


@dataclass(frozen=True, slots=True, eq=False)
class DescriptorIdentity:
  """Exact selected descriptor identity without duplicating its binding owner."""

  kind: str
  name: str
  version: str
  implementation_id: str


@dataclass(frozen=True, slots=True, eq=False)
class CompiledCellBlock:
  """Canonical dense connectivity for one explicit source cell block."""

  id: SemanticId
  reference_topology: str
  topological_dimension: int
  embedding_dimension: int
  geometry_interpolation: str
  cell_ids: tuple[SemanticId, ...]
  connectivity: FinalizedArray


@dataclass(frozen=True, slots=True, eq=False)
class CompiledMesh:
  """Canonical mesh coordinates and independently shaped cell blocks."""

  node_ids: tuple[SemanticId, ...]
  coordinates: FinalizedArray
  cell_blocks: tuple[CompiledCellBlock, ...]


@dataclass(frozen=True, slots=True, eq=False)
class DofPlan:
  """Explicit global indices for one node field and its declared components."""

  field_id: SemanticId
  components: tuple[str, ...]
  node_ids: tuple[SemanticId, ...]
  node_component_dofs: FinalizedArray
  global_size: int


@dataclass(frozen=True, slots=True, eq=False)
class IntegrationLayout:
  """Stable quadrature and material-slot sizes for one domain block."""

  points_per_element: int
  material_slots_per_point: int
  local_point_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True, eq=False)
class DomainBlock:
  """One homogeneous Q8 plane-stress contribution recipe."""

  block_id: tuple[SemanticId, SemanticId]
  source_cell_block_id: SemanticId
  source_region_id: SemanticId
  cell_ids: tuple[SemanticId, ...]
  field_id: SemanticId
  source_material_id: SemanticId
  field_components: tuple[str, ...]
  topology: DescriptorIdentity
  quadrature: DescriptorIdentity
  formulation: DescriptorIdentity
  material: DescriptorIdentity
  connectivity: FinalizedArray
  dof_map: FinalizedArray
  quadrature_points: FinalizedArray
  quadrature_weights: FinalizedArray
  shape_values: FinalizedArray
  parent_gradients: FinalizedArray
  material_parameter_names: tuple[str, ...]
  material_parameters: FinalizedArray
  integration_layout: IntegrationLayout
  kinematic_regime: str
  strain_voigt_order: tuple[str, ...]
  shear_convention: str
  measure_convention: str


@dataclass(frozen=True, slots=True, eq=False)
class ElementCouplingRecipe:
  """Backend-neutral declaration of one element-local dense coupling."""

  block_index: int
  local_dof_count: int
  coupling: str
  dof_map: FinalizedArray


@dataclass(frozen=True, slots=True, eq=False)
class ModelAssemblyTopology:
  """Model-owned coupling recipes, before any sparse backend is selected."""

  block_recipes: tuple[ElementCouplingRecipe, ...]
  fixed_model_coupling: bool


@dataclass(frozen=True, slots=True, eq=False)
class PrimaryFieldLayout:
  """Size-only layout for one model-owned primary field."""

  field_id: SemanticId
  components: tuple[str, ...]
  global_size: int


@dataclass(frozen=True, slots=True, eq=False)
class BlockStateLayout:
  """Size-only local-state schema for one domain block."""

  block_index: int
  element_count: int
  integration_points_per_element: int
  material_slots_per_point: int
  material_history_width: int
  formulation_history_width: int

  @property
  def material_history_shape(self) -> tuple[int, int, int, int]:
    """Return the future material-history shape without allocating values."""
    return (
      self.element_count,
      self.integration_points_per_element,
      self.material_slots_per_point,
      self.material_history_width,
    )

  @property
  def formulation_history_shape(self) -> tuple[int, int]:
    """Return the future formulation-history shape without allocating values."""
    return self.element_count, self.formulation_history_width


@dataclass(frozen=True, slots=True, eq=False)
class PhysicalStateLayout:
  """Model-owned state sizes; this carrier owns no evolving state values."""

  primary_fields: tuple[PrimaryFieldLayout, ...]
  block_states: tuple[BlockStateLayout, ...]
  global_primary_size: int
  evolving_value_count: int


@dataclass(frozen=True, slots=True, eq=False)
class ModelCapabilities:
  """Conservative guarantees derived from the compiled domain recipes."""

  response_class: str
  fixed_model_coupling: bool
  tangent_class: str
  tangent_is_symmetric: bool
  tangent_is_constant: bool
  conservative_internal_contribution: bool
  state_dependent: bool
  has_storage: bool
  has_mass: bool
  has_damping: bool
  restart_history_required: bool
  contribution_channels: tuple[str, ...]


@dataclass(frozen=True, slots=True, eq=False)
class EntityRecord:
  """Stable semantic identity plus a transient compiled execution locator."""

  kind: str
  semantic_id: EntitySemanticId
  dense_index: int | None
  block_index: int | None
  local_index: int | None


@dataclass(frozen=True, slots=True, eq=False)
class EntityIndex:
  """Tuple-backed semantic lookup independent of dictionary mutation."""

  records: tuple[EntityRecord, ...]

  def lookup(self, kind: str, semantic_id: object) -> EntityRecord:
    """Resolve one semantic entity or fail without guessing an ID coercion."""
    if type(kind) is not str:
      msg = "entity kind must be an exact string"
      raise TypeError(msg)
    for record in self.records:
      if record.kind == kind and _same_semantic_id(record.semantic_id, semantic_id):
        return record
    msg = f"compiled entity index has no {kind!r} record for that exact semantic ID"
    raise KeyError(msg)


@dataclass(frozen=True, slots=True, eq=False)
class SourceRecord:
  """Source location for one stable compiled semantic entity."""

  kind: str
  semantic_id: EntitySemanticId
  source: CompiledSource


@dataclass(frozen=True, slots=True, eq=False)
class SourceMap:
  """Detached tuple-backed source map for diagnostics and persisted meaning."""

  records: tuple[SourceRecord, ...]

  def lookup(self, kind: str, semantic_id: object) -> CompiledSource:
    """Resolve one exact semantic source identity."""
    if type(kind) is not str:
      msg = "source-map kind must be an exact string"
      raise TypeError(msg)
    for record in self.records:
      if record.kind == kind and _same_semantic_id(record.semantic_id, semantic_id):
        return record.source
    msg = f"compiled source map has no {kind!r} record for that exact semantic ID"
    raise KeyError(msg)


@dataclass(frozen=True, slots=True, eq=False)
class ModelProvenance:
  """Canonical content manifest and its explicit schema identity."""

  schema: str
  manifest: CanonicalManifest
  registry_fingerprint: ContentFingerprint
  floating_dtype: str
  dense_index_dtype: str
  geometry_relative_tolerance: float


@dataclass(frozen=True, slots=True, eq=False)
class CompiledModel:
  """Immutable reusable physical-discretization recipe."""

  instance_id: InstanceId
  content_fingerprint: ContentFingerprint
  provenance: ModelProvenance
  registry_snapshot: RegistrySnapshot
  mesh: CompiledMesh
  dofs: DofPlan
  domain_blocks: tuple[DomainBlock, ...]
  assembly_topology: ModelAssemblyTopology
  physical_state_layout: PhysicalStateLayout
  capabilities: ModelCapabilities
  entity_index: EntityIndex
  source_map: SourceMap
