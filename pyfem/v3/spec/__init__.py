"""Authored and normalized finite-element model specification contracts."""

from pyfem.v3.spec.diagnostics import (
  ModelSpecValidationError,
  SourceContext,
  SpecDiagnostic,
)
from pyfem.v3.spec.model import (
  CellBlockSpec,
  CellRef,
  CellSpec,
  FieldSpec,
  MaterialParameterSpec,
  MaterialParameterValue,
  MaterialSpec,
  MeshSpec,
  ModelSpec,
  NodeSpec,
  RegionSpec,
  SpecId,
)
from pyfem.v3.spec.normalize import normalize_model_spec
from pyfem.v3.spec.normalize_program import normalize_program_spec
from pyfem.v3.spec.program import (
  AffineCoefficientSpec,
  AffineTieSpec,
  AffineValueSpec,
  DofRef,
  NodalLoadSpec,
  PrescribedDofSpec,
  ProgramConstraintSpec,
  ProgramCoordinateKind,
  ProgramCoordinateSpec,
  ProgramCoordinateValue,
  ProgramPoint,
  ProgramSpec,
)
from pyfem.v3.spec.program_diagnostics import (
  ProgramSpecDiagnostic,
  ProgramSpecValidationError,
)

__all__ = [
  "CellBlockSpec",
  "CellRef",
  "CellSpec",
  "DofRef",
  "FieldSpec",
  "MaterialParameterSpec",
  "MaterialParameterValue",
  "MaterialSpec",
  "MeshSpec",
  "ModelSpec",
  "ModelSpecValidationError",
  "NodeSpec",
  "NodalLoadSpec",
  "PrescribedDofSpec",
  "ProgramConstraintSpec",
  "ProgramCoordinateKind",
  "ProgramCoordinateSpec",
  "ProgramCoordinateValue",
  "ProgramPoint",
  "ProgramSpec",
  "ProgramSpecDiagnostic",
  "ProgramSpecValidationError",
  "RegionSpec",
  "SourceContext",
  "SpecDiagnostic",
  "SpecId",
  "AffineCoefficientSpec",
  "AffineTieSpec",
  "AffineValueSpec",
  "normalize_model_spec",
  "normalize_program_spec",
]
