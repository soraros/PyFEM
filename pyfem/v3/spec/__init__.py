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

__all__ = [
  "CellBlockSpec",
  "CellRef",
  "CellSpec",
  "FieldSpec",
  "MaterialParameterSpec",
  "MaterialParameterValue",
  "MaterialSpec",
  "MeshSpec",
  "ModelSpec",
  "ModelSpecValidationError",
  "NodeSpec",
  "RegionSpec",
  "SourceContext",
  "SpecDiagnostic",
  "SpecId",
  "normalize_model_spec",
]
