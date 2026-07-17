"""Compiler-owned identity, provenance, array, and registry primitives."""

from pyfem.v3.model.arrays import ArrayOrder, FinalizedArray, finalize_array
from pyfem.v3.model.identity import (
  GenerationMismatchError,
  IdentityMismatchError,
  InstanceId,
  StateGeneration,
  require_generation_successor,
  require_same_generation,
  require_same_instance,
)
from pyfem.v3.model.provenance import (
  CANONICAL_MANIFEST_FORMAT,
  CanonicalManifest,
  ContentFingerprint,
  UnorderedDeclarations,
)
from pyfem.v3.model.registry import (
  REGISTRY_MANIFEST_SCHEMA,
  RegistryDescriptor,
  RegistryKey,
  RegistrySnapshot,
)

__all__ = [
  "CANONICAL_MANIFEST_FORMAT",
  "REGISTRY_MANIFEST_SCHEMA",
  "ArrayOrder",
  "CanonicalManifest",
  "ContentFingerprint",
  "FinalizedArray",
  "GenerationMismatchError",
  "IdentityMismatchError",
  "InstanceId",
  "RegistryDescriptor",
  "RegistryKey",
  "RegistrySnapshot",
  "StateGeneration",
  "UnorderedDeclarations",
  "finalize_array",
  "require_generation_successor",
  "require_same_generation",
  "require_same_instance",
]
