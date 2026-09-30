"""Material parameter slices for the authoring layer.

These helpers author the ``MaterialSpec`` values the in-tree builders consume;
they pair with the registry-descriptor authors in
:mod:`pyfem.v3.authoring.registry`, which bind user-supplied law functions to
the same qualified conventions.
"""

from __future__ import annotations

import math

from pyfem.v3.spec.diagnostics import SourceContext
from pyfem.v3.spec.model import MaterialParameterSpec, MaterialSpec, SpecId


def _source(label: str) -> SourceContext:
  return SourceContext(source=label)


def _finite_scalar(value: object, *, label: str) -> float:
  if type(value) not in (int, float):
    msg = f"{label} must be an exact number"
    raise TypeError(msg)
  try:
    converted = float(value)
  except OverflowError:
    msg = f"{label} must fit finite float64"
    raise ValueError(msg) from None
  if not math.isfinite(converted):
    msg = f"{label} must be finite"
    raise ValueError(msg)
  return converted


def _material_id(value: object) -> SpecId:
  if type(value) is str and value:
    return value
  if type(value) is int:
    return value
  msg = "material id must be a non-empty string or integer id"
  raise TypeError(msg)


def linear_elastic(
  E: float,
  nu: float,
  *,
  id: SpecId = "material",
  source: str = "authoring.linear_elastic",
) -> MaterialSpec:
  """Author plane-stress linear-elastic parameters for a continuum region.

  ``E`` is Young's modulus and ``nu`` Poisson's ratio; they map onto the
  qualified ``youngs_modulus``/``poisson_ratio`` parameter convention. Domain
  checks (positive modulus, ``-1 < nu < 0.5``) run at compile time with coded
  diagnostics.
  """
  youngs_modulus = _finite_scalar(E, label="linear_elastic E")
  poisson_ratio = _finite_scalar(nu, label="linear_elastic nu")
  return MaterialSpec(
    id=_material_id(id),
    model="plane-stress-linear-elastic",
    parameters=(
      MaterialParameterSpec(
        "youngs_modulus",
        youngs_modulus,
        _source(f"{source}:youngs_modulus"),
      ),
      MaterialParameterSpec(
        "poisson_ratio",
        poisson_ratio,
        _source(f"{source}:poisson_ratio"),
      ),
    ),
    source=_source(source),
  )


def uniaxial_elastic(
  E: float,
  area: float,
  *,
  id: SpecId = "material",
  source: str = "authoring.uniaxial_elastic",
) -> MaterialSpec:
  """Author uniaxial linear-elastic section parameters for a truss region.

  ``E`` is Young's modulus and ``area`` the cross-section area; they map onto
  the qualified ``youngs_modulus``/``area`` parameter convention. Domain
  checks run at compile time with coded diagnostics.
  """
  youngs_modulus = _finite_scalar(E, label="uniaxial_elastic E")
  section_area = _finite_scalar(area, label="uniaxial_elastic area")
  return MaterialSpec(
    id=_material_id(id),
    model="uniaxial-linear-elastic",
    parameters=(
      MaterialParameterSpec(
        "youngs_modulus",
        youngs_modulus,
        _source(f"{source}:youngs_modulus"),
      ),
      MaterialParameterSpec(
        "area",
        section_area,
        _source(f"{source}:area"),
      ),
    ),
    source=_source(source),
  )
