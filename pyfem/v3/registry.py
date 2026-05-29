"""Explicit mapping from legacy ``.pro`` type strings to v3 implementations."""

from __future__ import annotations

from typing import Final

from pyfem.v3.types import GROUP_CONTINUUM, GROUP_SPRING, GROUP_TRUSS, ElementGroupSpec

# Legacy input names (Java-style) -> v3 modules (snake_case packages).
ELEMENT_TYPES: Final[dict[str, str]] = {
  "SmallStrainContinuum": "small_strain_continuum",
  "FiniteStrainContinuum": "finite_strain_continuum",
  "Truss": "truss",
  "Spring": "spring",
}

MATERIAL_TYPES: Final[dict[str, str]] = {
  "PlaneStress": "plane_stress",
  "PlaneStrain": "plane_strain",
  "Isotropic": "isotropic",
  "Truss": "truss",
  "Spring": "spring",
}

SOLVER_TYPES: Final[dict[str, str]] = {
  "LinearSolver": "linear",
  "NonlinearSolver": "nonlinear",
  "RiksSolver": "riks",
}


def _resolve(mapping: dict[str, str], legacy_name: str, kind: str) -> str:
  try:
    return mapping[legacy_name]
  except KeyError as exc:
    msg = f"Unknown {kind} {legacy_name!r}"
    raise ValueError(msg) from exc


def resolve_element_type(legacy_name: str) -> str:
  return _resolve(ELEMENT_TYPES, legacy_name, "element type")


def resolve_material_type(legacy_name: str) -> str:
  return _resolve(MATERIAL_TYPES, legacy_name, "material type")


def resolve_solver_type(legacy_name: str) -> str:
  return _resolve(SOLVER_TYPES, legacy_name, "solver type")


def group_kind_for(element_type: str) -> int:
  """Map legacy element type to assembly group kind."""
  if element_type == "Truss":
    return GROUP_TRUSS
  if element_type == "Spring":
    return GROUP_SPRING
  return GROUP_CONTINUUM


def group_props_for(spec: ElementGroupSpec) -> tuple[float, float]:
  """Pack group property array row for ``ProblemDefinition.group_props``."""
  if spec.element_type == "Truss":
    return float(spec.props[0]), float(spec.props[1])
  if spec.element_type == "Spring":
    return float(spec.props[0]), 0.0
  return 0.0, 0.0
