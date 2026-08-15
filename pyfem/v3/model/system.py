"""Immutable generic compiled-system carriers."""

from __future__ import annotations

from dataclasses import dataclass

from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.identity import InstanceId
from pyfem.v3.model.operator import (
  CompiledOperator,
  CompilerConstructed,
  SemanticId,
)
from pyfem.v3.model.provenance import CanonicalManifest, ContentFingerprint
from pyfem.v3.model.registry import RegistrySnapshot


@dataclass(frozen=True, slots=True, eq=False, init=False)
class CompiledSource(CompilerConstructed):
  source: str
  line: int | None
  column: int | None


@dataclass(frozen=True, slots=True, eq=False, init=False)
class PointEntityBlock(CompilerConstructed):
  block_id: SemanticId
  entity_ids: tuple[SemanticId, ...]
  sources: tuple[CompiledSource, ...]
  reference_coordinates: FinalizedArray


@dataclass(frozen=True, slots=True, eq=False, init=False)
class IncidenceEntityBlock(CompilerConstructed):
  block_id: SemanticId
  entity_ids: tuple[SemanticId, ...]
  sources: tuple[CompiledSource, ...]
  incidence: FinalizedArray


@dataclass(frozen=True, slots=True, eq=False, init=False)
class DiscreteSpace(CompilerConstructed):
  space_id: SemanticId
  support_block_id: SemanticId
  basis_id: str
  components: tuple[str, ...]
  coefficient_ids: tuple[SemanticId, ...]
  coefficient_map: FinalizedArray
  coefficient_range: tuple[int, int]

  @property
  def coefficient_count(self) -> int:
    return self.coefficient_range[1] - self.coefficient_range[0]


@dataclass(frozen=True, slots=True, eq=False, init=False)
class SourceAttribution(CompilerConstructed):
  kind: str
  semantic_id: SemanticId
  source: CompiledSource


@dataclass(frozen=True, slots=True, eq=False, init=False)
class SystemProvenance(CompilerConstructed):
  schema: str
  manifest: CanonicalManifest
  registry_fingerprint: ContentFingerprint
  floating_dtype: str
  dense_index_dtype: str
  geometry_relative_tolerance: float


@dataclass(frozen=True, slots=True, eq=False, init=False)
class CompiledSystem(CompilerConstructed):
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
    if not self.spaces:
      return 0
    return self.spaces[-1].coefficient_range[1]

  def source_for(self, kind: str, semantic_id: SemanticId) -> CompiledSource:
    if type(kind) is not str:
      raise TypeError("compiled source kind must be an exact str")
    for record in self.source_attribution:
      if record.kind == kind and _same_semantic_id(record.semantic_id, semantic_id):
        return record.source
    msg = "compiled system has no source for that exact semantic identity"
    raise KeyError(msg)


def _same_semantic_id(left: object, right: object) -> bool:
  if type(left) is not type(right):
    return False
  if type(left) is tuple:
    return len(left) == len(right) and all(
      _same_semantic_id(a, b) for a, b in zip(left, right, strict=True)
    )
  return bool(left == right)
