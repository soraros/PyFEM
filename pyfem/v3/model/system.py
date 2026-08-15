"""Immutable generic compiled-system carriers."""

from __future__ import annotations

from dataclasses import dataclass

from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.identity import InstanceId
from pyfem.v3.model.operator import CompiledOperator, SemanticId
from pyfem.v3.model.provenance import CanonicalManifest, ContentFingerprint
from pyfem.v3.model.registry import RegistrySnapshot


@dataclass(frozen=True, slots=True, eq=False)
class CompiledSource:
  """Detached authored source location."""

  source: str
  line: int | None
  column: int | None


@dataclass(frozen=True, slots=True, eq=False)
class PointEntityBlock:
  """Canonical point support and immutable reference geometry."""

  block_id: SemanticId
  entity_ids: tuple[SemanticId, ...]
  sources: tuple[CompiledSource, ...]
  reference_coordinates: FinalizedArray


@dataclass(frozen=True, slots=True, eq=False)
class IncidenceEntityBlock:
  """Homogeneous entities with one native incidence width."""

  block_id: SemanticId
  entity_ids: tuple[SemanticId, ...]
  sources: tuple[CompiledSource, ...]
  incidence: FinalizedArray


@dataclass(frozen=True, slots=True, eq=False)
class DiscreteSpace:
  """One explicit field/basis space with its native coefficient allocation."""

  space_id: SemanticId
  support_block_id: SemanticId
  basis_id: str
  components: tuple[str, ...]
  coefficient_ids: tuple[SemanticId, ...]
  coefficient_map: FinalizedArray
  coefficient_range: tuple[int, int]

  @property
  def coefficient_count(self) -> int:
    """Return this space's exact active coefficient count."""
    return self.coefficient_range[1] - self.coefficient_range[0]


@dataclass(frozen=True, slots=True, eq=False)
class SourceAttribution:
  """Stable semantic entity-to-source record."""

  kind: str
  semantic_id: SemanticId
  source: CompiledSource


@dataclass(frozen=True, slots=True, eq=False)
class SystemProvenance:
  """Canonical system meaning and numeric convention."""

  schema: str
  manifest: CanonicalManifest
  registry_fingerprint: ContentFingerprint
  floating_dtype: str
  dense_index_dtype: str
  geometry_relative_tolerance: float


@dataclass(frozen=True, slots=True, eq=False)
class CompiledSystem:
  """Direct generic compilation of reusable physical discretization."""

  instance_id: InstanceId
  content_fingerprint: ContentFingerprint
  provenance: SystemProvenance
  registry_snapshot: RegistrySnapshot
  point_blocks: tuple[PointEntityBlock, ...]
  entity_blocks: tuple[IncidenceEntityBlock, ...]
  spaces: tuple[DiscreteSpace, ...]
  operators: tuple[CompiledOperator, ...]
  source_attribution: tuple[SourceAttribution, ...]

  @property
  def coefficient_count(self) -> int:
    """Return the total coefficient count across ordered disjoint spaces."""
    if not self.spaces:
      return 0
    return self.spaces[-1].coefficient_range[1]

  def source_for(self, kind: str, semantic_id: SemanticId) -> CompiledSource:
    """Resolve one exact stable semantic source identity."""
    for record in self.source_attribution:
      if record.kind == kind and record.semantic_id == semantic_id:
        return record.source
    msg = "compiled system has no source for that exact semantic identity"
    raise KeyError(msg)
