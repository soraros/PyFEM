"""Authoring-layer compile entry: registry preflight plus the landed compiler."""

from __future__ import annotations

from pyfem.v3.authoring.registry import check_registry
from pyfem.v3.compile.continuum import (
  DAMAGE_MATERIAL_KEY,
  PLASTIC_MATERIAL_KEY,
  Q8_MATERIAL_KEY,
  damage_reference_registry,
  plasticity_reference_registry,
  q8_reference_registry,
)
from pyfem.v3.compile.diagnostics import (
  ModelCompilationDiagnostic,
  ModelCompilationError,
)
from pyfem.v3.compile.system import SystemCompilationPolicy, compile_system
from pyfem.v3.compile.truss import (
  TRUSS_FORMULATION_KEY,
  TRUSS_MATERIAL_KEY,
  truss_reference_registry,
)
from pyfem.v3.model.registry import RegistryDescriptor, RegistryKey
from pyfem.v3.model.system import CompiledSystem
from pyfem.v3.spec.model import ModelSpec


def _default_registry(model: ModelSpec) -> dict[RegistryKey, RegistryDescriptor]:
  defaults = {
    Q8_MATERIAL_KEY[1]: q8_reference_registry,
    TRUSS_MATERIAL_KEY[1]: truss_reference_registry,
    PLASTIC_MATERIAL_KEY[1]: plasticity_reference_registry,
    DAMAGE_MATERIAL_KEY[1]: damage_reference_registry,
  }
  candidates = [
    name
    for name in defaults
    if any(material.model == name for material in model.materials)
  ]
  if (
    any(region.formulation == TRUSS_FORMULATION_KEY[1] for region in model.regions)
    and TRUSS_MATERIAL_KEY[1] not in candidates
  ):
    candidates.append(TRUSS_MATERIAL_KEY[1])
  if len(candidates) > 1:
    raise ModelCompilationError(
      (
        ModelCompilationDiagnostic(
          code="ambiguous-default-registry",
          message=(
            "default registry selection is ambiguous for material families "
            f"{tuple(candidates)!r}; pass an explicit combined registry"
          ),
          source=model.source,
        ),
      )
    )
  if candidates:
    return defaults[candidates[0]]()
  # Unknown formulations and materials still reach the continuum builder so
  # its landed coded diagnostics describe the mismatch.
  return q8_reference_registry()


def compile(
  model: ModelSpec,
  registry: dict[RegistryKey, RegistryDescriptor] | None = None,
  *,
  policy: SystemCompilationPolicy | None = None,
) -> CompiledSystem:
  """Compile an authored model, preflighting registry metadata with field diffs.

  A single recognized material family selects its reference registry by default.
  Mixed material families, including a truss formulation competing with another
  material, require an explicit combined registry. Unknown formulations still
  fall back to the continuum builder for its coded diagnostics. Descriptor
  metadata mismatches are reported here with field-level diffs (which fields,
  expected versus authored values); every other failure keeps the landed coded
  diagnostics from the direct compiler, which runs unchanged after preflight.
  """
  if type(model) is not ModelSpec:
    msg = "authoring.compile requires an exact ModelSpec from the authoring builders"
    raise TypeError(msg)
  selected = _default_registry(model) if registry is None else registry
  check_registry(model, selected)
  if policy is None:
    return compile_system(model, selected)
  return compile_system(model, selected, policy=policy)
