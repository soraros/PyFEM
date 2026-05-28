"""Explicit mapping from legacy ``.pro`` type strings to v3 implementations."""

from __future__ import annotations

from typing import Final

# Legacy input names (Java-style) -> v3 modules (snake_case packages).
ELEMENT_TYPES: Final[dict[str, str]] = {
  "SmallStrainContinuum": "small_strain_continuum",
}

MATERIAL_TYPES: Final[dict[str, str]] = {
  "PlaneStress": "plane_stress",
}

SOLVER_TYPES: Final[dict[str, str]] = {
  "LinearSolver": "linear",
}


def resolve_element_type(legacy_name: str) -> str:
  try:
    return ELEMENT_TYPES[legacy_name]
  except KeyError as exc:
    msg = f"Unknown element type {legacy_name!r}"
    raise ValueError(msg) from exc


def resolve_material_type(legacy_name: str) -> str:
  try:
    return MATERIAL_TYPES[legacy_name]
  except KeyError as exc:
    msg = f"Unknown material type {legacy_name!r}"
    raise ValueError(msg) from exc


def resolve_solver_type(legacy_name: str) -> str:
  try:
    return SOLVER_TYPES[legacy_name]
  except KeyError as exc:
    msg = f"Unknown solver type {legacy_name!r}"
    raise ValueError(msg) from exc
