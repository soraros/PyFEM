"""Authoring-layer compile entry: registry preflight plus the landed compiler."""

from __future__ import annotations

from pyfem.v3.authoring.registry import check_registry
from pyfem.v3.compile.continuum import (
  DAMAGE_MATERIAL_KEY,
  PLASTIC_MATERIAL_KEY,
  VISCOELASTIC_MATERIAL_KEY,
  damage_reference_registry,
  plasticity_reference_registry,
  q8_reference_registry,
  viscoelasticity_reference_registry,
)
from pyfem.v3.compile.system import SystemCompilationPolicy, compile_system
from pyfem.v3.compile.truss import TRUSS_FORMULATION_KEY, truss_reference_registry
from pyfem.v3.model.registry import RegistryDescriptor, RegistryKey
from pyfem.v3.model.system import CompiledSystem
from pyfem.v3.spec.model import ModelSpec


def _default_registry(model: ModelSpec) -> dict[RegistryKey, RegistryDescriptor]:
  # Mirrors the landed dispatch: unknown formulations fall back to the
  # continuum builder so its coded diagnostics describe the mismatch. A
  # material on the stateful seam selects its law's reference registry.
  formulations = {region.formulation for region in model.regions}
  if TRUSS_FORMULATION_KEY[1] in formulations:
    return truss_reference_registry()
  if any(material.model == PLASTIC_MATERIAL_KEY[1] for material in model.materials):
    return plasticity_reference_registry()
  if any(material.model == DAMAGE_MATERIAL_KEY[1] for material in model.materials):
    return damage_reference_registry()
  if any(
    material.model == VISCOELASTIC_MATERIAL_KEY[1] for material in model.materials
  ):
    return viscoelasticity_reference_registry()
  return q8_reference_registry()


def compile(
  model: ModelSpec,
  registry: dict[RegistryKey, RegistryDescriptor] | None = None,
  *,
  policy: SystemCompilationPolicy | None = None,
) -> CompiledSystem:
  """Compile an authored model, preflighting registry metadata with field diffs.

  ``registry`` defaults to the reference registry of the family the model's
  formulation selects. Descriptor metadata mismatches are reported here with
  field-level diffs (which fields, expected versus authored values); every
  other failure keeps the landed coded diagnostics from the direct compiler,
  which runs unchanged after the preflight.
  """
  if type(model) is not ModelSpec:
    msg = "authoring.compile requires an exact ModelSpec from the authoring builders"
    raise TypeError(msg)
  selected = _default_registry(model) if registry is None else registry
  check_registry(model, selected)
  if policy is None:
    return compile_system(model, selected)
  return compile_system(model, selected, policy=policy)
