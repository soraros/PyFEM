"""Authoring entry for the FE² continuum operator family."""

from __future__ import annotations

from pyfem.v3.compile.fe2 import compile_fe2_continuum
from pyfem.v3.constraints.compile import CompiledConstraintMap
from pyfem.v3.driver import NonlinearStaticSettings
from pyfem.v3.driver.contracts import HomogenizationRequest
from pyfem.v3.model.system import CompiledSystem
from pyfem.v3.spec.diagnostics import SourceContext
from pyfem.v3.spec.program import NodalLoadSpec


def fe2_continuum(
  base: CompiledSystem,
  *,
  micro_system: CompiledSystem,
  micro_coordinate_map: CompiledConstraintMap,
  homogenization: HomogenizationRequest,
  micro_loads: tuple[NodalLoadSpec, ...] = (),
  settings: NonlinearStaticSettings = NonlinearStaticSettings(),
  implementation_id: str = "pyfem-v3-fe2-continuum-v1",
  source: str = "authoring.fe2_continuum",
) -> CompiledSystem:
  """Replace a direct Q8 macro operator with its compiled FE² response family."""
  if type(source) is not str or not source:
    msg = "FE2 authoring source must be a non-empty exact string"
    raise TypeError(msg)
  return compile_fe2_continuum(
    base,
    micro_system=micro_system,
    micro_coordinate_map=micro_coordinate_map,
    homogenization=homogenization,
    micro_loads=micro_loads,
    settings=settings,
    implementation_id=implementation_id,
    source=SourceContext(source=source),
  )
