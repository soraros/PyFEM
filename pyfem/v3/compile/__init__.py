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
from pyfem.v3.compile.program import (
  COMPILED_PROGRAM_MANIFEST_SCHEMA,
  PROGRAM_REDUCTION_POLICY,
  compile_program,
  evaluate_program,
)
from pyfem.v3.compile.program_diagnostics import (
  ProgramCompilationDiagnostic,
  ProgramCompilationError,
  ProgramEvaluationDiagnostic,
  ProgramEvaluationError,
)

__all__ = [
  "COMPILED_MODEL_MANIFEST_SCHEMA",
  "COMPILED_PROGRAM_MANIFEST_SCHEMA",
  "Q8_FORMULATION_KEY",
  "Q8_MATERIAL_KEY",
  "Q8_QUADRATURE_KEY",
  "Q8_REQUIRED_REGISTRY_KEYS",
  "Q8_TOPOLOGY_KEY",
  "ModelCompilationDiagnostic",
  "ModelCompilationError",
  "ModelCompilationPolicy",
  "PROGRAM_REDUCTION_POLICY",
  "ProgramCompilationDiagnostic",
  "ProgramCompilationError",
  "ProgramEvaluationDiagnostic",
  "ProgramEvaluationError",
  "compile_model",
  "compile_program",
  "evaluate_program",
  "q8_descriptor_metadata",
  "q8_reference_registry",
]
