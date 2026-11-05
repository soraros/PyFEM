"""Shared descriptor and diagnostic exports for generic compilers."""

from pyfem.v3.compile.contracts import (
  Q8_FORMULATION_KEY,
  Q8_MATERIAL_KEY,
  Q8_QUADRATURE_KEY,
  Q8_REQUIRED_REGISTRY_KEYS,
  Q8_TOPOLOGY_KEY,
  q8_descriptor_metadata,
  q8_reference_registry,
)
from pyfem.v3.compile.diagnostics import (
  ModelCompilationDiagnostic,
  ModelCompilationError,
)

__all__ = [
  "Q8_FORMULATION_KEY",
  "Q8_MATERIAL_KEY",
  "Q8_QUADRATURE_KEY",
  "Q8_REQUIRED_REGISTRY_KEYS",
  "Q8_TOPOLOGY_KEY",
  "ModelCompilationDiagnostic",
  "ModelCompilationError",
  "q8_descriptor_metadata",
  "q8_reference_registry",
]
