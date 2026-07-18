"""Internal normalized-model compiler for the frozen Q8 slice."""

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
from pyfem.v3.compile.model import (
  COMPILED_MODEL_MANIFEST_SCHEMA,
  ModelCompilationPolicy,
  compile_model,
)

__all__ = [
  "COMPILED_MODEL_MANIFEST_SCHEMA",
  "Q8_FORMULATION_KEY",
  "Q8_MATERIAL_KEY",
  "Q8_QUADRATURE_KEY",
  "Q8_REQUIRED_REGISTRY_KEYS",
  "Q8_TOPOLOGY_KEY",
  "ModelCompilationDiagnostic",
  "ModelCompilationError",
  "ModelCompilationPolicy",
  "compile_model",
  "q8_descriptor_metadata",
  "q8_reference_registry",
]
