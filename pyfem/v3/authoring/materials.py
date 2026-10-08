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


def plasticity(
  E: float,
  nu: float,
  syield: float,
  hard: float,
  *,
  id: SpecId = "material",
  source: str = "authoring.plasticity",
) -> MaterialSpec:
  """Author J2 isotropic-hardening plasticity parameters for a continuum region.

  ``E`` is Young's modulus, ``nu`` Poisson's ratio, ``syield`` the initial
  yield stress, and ``hard`` the linear hardening slope; they map onto the
  qualified ``youngs_modulus``/``poisson_ratio``/``initial_yield_stress``/
  ``hardening_slope`` parameter convention of the v2 stateful descriptor.
  Domain checks (positive modulus and yield stress, ``-1 < nu < 0.5``,
  non-negative slope) run at compile time with coded diagnostics.
  """
  youngs_modulus = _finite_scalar(E, label="plasticity E")
  poisson_ratio = _finite_scalar(nu, label="plasticity nu")
  initial_yield_stress = _finite_scalar(syield, label="plasticity syield")
  hardening_slope = _finite_scalar(hard, label="plasticity hard")
  return MaterialSpec(
    id=_material_id(id),
    model="isotropic-hardening-plasticity",
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
      MaterialParameterSpec(
        "initial_yield_stress",
        initial_yield_stress,
        _source(f"{source}:initial_yield_stress"),
      ),
      MaterialParameterSpec(
        "hardening_slope",
        hardening_slope,
        _source(f"{source}:hardening_slope"),
      ),
    ),
    source=_source(source),
  )


def damage(
  E: float,
  nu: float,
  kappa0: float,
  kappac: float,
  k: float,
  *,
  id: SpecId = "material",
  source: str = "authoring.damage",
) -> MaterialSpec:
  """Author plane-strain isotropic damage parameters for a continuum region.

  ``E`` is Young's modulus, ``nu`` Poisson's ratio, ``kappa0`` the damage
  onset threshold, ``kappac`` the failure equivalent strain, and ``k`` the
  de Vree compressive-to-tensile strength ratio; they map onto the qualified
  ``youngs_modulus``/``poisson_ratio``/``kappa_0``/``kappa_c``/
  ``strength_ratio`` parameter convention of the v2 stateful descriptor.
  Domain checks (positive modulus and thresholds, ``-1 < nu < 0.5``,
  ``kappa_c > kappa_0``) run at compile time with coded diagnostics.
  """
  youngs_modulus = _finite_scalar(E, label="damage E")
  poisson_ratio = _finite_scalar(nu, label="damage nu")
  kappa_0 = _finite_scalar(kappa0, label="damage kappa0")
  kappa_c = _finite_scalar(kappac, label="damage kappac")
  strength_ratio = _finite_scalar(k, label="damage k")
  return MaterialSpec(
    id=_material_id(id),
    model="plane-strain-damage",
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
      MaterialParameterSpec(
        "kappa_0",
        kappa_0,
        _source(f"{source}:kappa_0"),
      ),
      MaterialParameterSpec(
        "kappa_c",
        kappa_c,
        _source(f"{source}:kappa_c"),
      ),
      MaterialParameterSpec(
        "strength_ratio",
        strength_ratio,
        _source(f"{source}:strength_ratio"),
      ),
    ),
    source=_source(source),
  )


def prony_viscoelasticity(
  E: float,
  nu: float,
  Einf: float,
  n: float,
  tau_first: float,
  tau_last: float,
  *,
  id: SpecId = "material",
  source: str = "authoring.prony_viscoelasticity",
) -> MaterialSpec:
  """Author Prony-series viscoelasticity parameters for a continuum region.

  ``E`` is the instantaneous Young's modulus, ``nu`` Poisson's ratio,
  ``Einf`` the equilibrium (long-term) modulus, ``n`` the number of Maxwell
  terms, and ``tau_first``/``tau_last`` the endpoints of the logarithmically
  spaced relaxation times; they map onto the qualified
  ``youngs_modulus``/``poisson_ratio``/``equilibrium_modulus``/
  ``prony_term_count``/``relaxation_time_first``/``relaxation_time_last``
  parameter convention of the v2 stateful descriptor. Domain checks
  (``0 < Einf < E``, integral ``n >= 1``, positive spanning times) run at
  compile time with coded diagnostics. Time reaches the compiled law through
  its declared identity signal port: declare a ``time`` program coordinate
  and bind it per step — there is no solverStat-style channel.
  """
  youngs_modulus = _finite_scalar(E, label="prony_viscoelasticity E")
  poisson_ratio = _finite_scalar(nu, label="prony_viscoelasticity nu")
  equilibrium_modulus = _finite_scalar(Einf, label="prony_viscoelasticity Einf")
  prony_term_count = _finite_scalar(n, label="prony_viscoelasticity n")
  relaxation_time_first = _finite_scalar(
    tau_first, label="prony_viscoelasticity tau_first"
  )
  relaxation_time_last = _finite_scalar(
    tau_last, label="prony_viscoelasticity tau_last"
  )
  return MaterialSpec(
    id=_material_id(id),
    model="prony-viscoelasticity",
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
      MaterialParameterSpec(
        "equilibrium_modulus",
        equilibrium_modulus,
        _source(f"{source}:equilibrium_modulus"),
      ),
      MaterialParameterSpec(
        "prony_term_count",
        prony_term_count,
        _source(f"{source}:prony_term_count"),
      ),
      MaterialParameterSpec(
        "relaxation_time_first",
        relaxation_time_first,
        _source(f"{source}:relaxation_time_first"),
      ),
      MaterialParameterSpec(
        "relaxation_time_last",
        relaxation_time_last,
        _source(f"{source}:relaxation_time_last"),
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
