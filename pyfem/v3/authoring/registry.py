"""Registry composition for the qualified Q8 and truss conventions.

A constitutive law is a plain Python function; the descriptor authors below
pin it to the qualified semantic convention (metadata the user never has to
copy), and the registry builders merge it with the in-tree reference
implementations. Replacement descriptors are validated eagerly, so a metadata
mismatch fails at authoring time with a field-level diff naming the exact
fields that disagree.
"""

from __future__ import annotations

from collections.abc import Callable

from pyfem.v3.authoring.diagnostics import raise_descriptor_mismatch
from pyfem.v3.compile.continuum import (
  PLASTIC_MATERIAL_KEY,
  Q8_FORMULATION_KEY,
  Q8_MATERIAL_KEY,
  Q8_QUADRATURE_KEY,
  Q8_TOPOLOGY_KEY,
  plasticity_reference_registry,
  q8_descriptor_metadata,
  q8_reference_registry,
)
from pyfem.v3.compile.contracts import StatefulContinuumBinding
from pyfem.v3.compile.truss import (
  TRUSS_FORMULATION_KEY,
  TRUSS_MATERIAL_KEY,
  TRUSS_TOPOLOGY_KEY,
  truss_descriptor_metadata,
  truss_reference_registry,
)
from pyfem.v3.materials.isotropic_hardening_plasticity import (
  isotropic_hardening_plasticity_metadata,
)
from pyfem.v3.model.provenance import CanonicalManifest
from pyfem.v3.model.registry import RegistryDescriptor, RegistryKey
from pyfem.v3.spec.diagnostics import SourceContext
from pyfem.v3.spec.model import ModelSpec


def plane_stress_law(
  binding: Callable[[float, float], object],
  *,
  implementation_id: str,
  version: str = "1",
) -> RegistryDescriptor:
  """Bind a plane-stress law function to the qualified Q8 material convention.

  ``binding`` receives ``(youngs_modulus, poisson_ratio)`` and returns the
  3x3 constitutive matrix in ``["xx", "yy", "xy"]`` Voigt order with the
  engineering shear convention. Compilation validates the returned matrix
  against the qualified plane-stress envelope, so equivalent formulations of
  the same law are accepted and contradictory ones are rejected with coded
  diagnostics.
  """
  return RegistryDescriptor(
    kind=Q8_MATERIAL_KEY[0],
    name=Q8_MATERIAL_KEY[1],
    version=version,
    implementation_id=implementation_id,
    metadata=q8_descriptor_metadata(*Q8_MATERIAL_KEY),
    binding=binding,
  )


def uniaxial_law(
  binding: Callable[[float, float], object],
  *,
  implementation_id: str,
  version: str = "1",
) -> RegistryDescriptor:
  """Bind a uniaxial section law function to the qualified truss convention.

  ``binding`` receives ``(youngs_modulus, area)`` and returns the 1x1 axial
  stiffness matrix. Compilation validates the returned value against the
  qualified uniaxial envelope.
  """
  return RegistryDescriptor(
    kind=TRUSS_MATERIAL_KEY[0],
    name=TRUSS_MATERIAL_KEY[1],
    version=version,
    implementation_id=implementation_id,
    metadata=truss_descriptor_metadata(*TRUSS_MATERIAL_KEY),
    binding=binding,
  )


def plasticity_law(
  binding: StatefulContinuumBinding,
  *,
  implementation_id: str,
  version: str = "1",
) -> RegistryDescriptor:
  """Bind a stateful J2 plasticity binding to the qualified material convention.

  ``binding`` implements the v2 stateful continuum binding protocol: a
  calibration call receiving ``(youngs_modulus, poisson_ratio,
  initial_yield_stress, hardening_slope)`` and returning the law's flat
  float64 calibration vector, a batched kernel over total 6-Voigt strains and
  accepted state rows, and an optional ``initial_state``. The descriptor pins
  the qualified isotropic-hardening-plasticity convention (metadata the user
  never has to copy); compilation re-declares the binding's metadata and
  validates it byte-wise against the captured descriptor, so equivalent
  kernels are accepted and contradictory ones rejected with coded
  diagnostics.
  """
  return RegistryDescriptor(
    kind=PLASTIC_MATERIAL_KEY[0],
    name=PLASTIC_MATERIAL_KEY[1],
    version=version,
    implementation_id=implementation_id,
    metadata=isotropic_hardening_plasticity_metadata(),
    binding=binding,
  )


def _replace(
  registry: dict[RegistryKey, RegistryDescriptor],
  replacements: tuple[tuple[RegistryKey, object], ...],
  *,
  family: str,
  metadata_for: Callable[[str, str], dict[str, object]],
  source: str,
) -> None:
  for key, replacement in replacements:
    if replacement is None:
      continue
    if type(replacement) is not RegistryDescriptor:
      msg = (
        f"{family} registry replacement for {key!r} must be an exact RegistryDescriptor"
      )
      raise TypeError(msg)
    if replacement.key != key:
      msg = (
        f"{family} registry replacement key {replacement.key!r} does not "
        f"match the qualified convention key {key!r}"
      )
      raise ValueError(msg)
    expected_metadata = metadata_for(*key)
    if (
      replacement.metadata.to_bytes() != CanonicalManifest(expected_metadata).to_bytes()
    ):
      raise_descriptor_mismatch(
        family=family,
        key=key,
        expected_metadata=expected_metadata,
        descriptor=replacement,
        source=SourceContext(source=f"{source}:{key[0]}"),
      )
    registry[key] = replacement


def q8_registry(
  *,
  topology: RegistryDescriptor | None = None,
  quadrature: RegistryDescriptor | None = None,
  formulation: RegistryDescriptor | None = None,
  material: RegistryDescriptor | None = None,
) -> dict[RegistryKey, RegistryDescriptor]:
  """Compose a Q8 registry: reference implementations plus your replacements.

  Replacements keep the qualified registry key and metadata convention; only
  the implementation (its id, version, and binding) is yours. A mismatching
  key or metadata field fails here with a field-level diff, not later with a
  bare compile error.
  """
  registry = q8_reference_registry()
  _replace(
    registry,
    (
      (Q8_TOPOLOGY_KEY, topology),
      (Q8_QUADRATURE_KEY, quadrature),
      (Q8_FORMULATION_KEY, formulation),
      (Q8_MATERIAL_KEY, material),
    ),
    family="Q8",
    metadata_for=q8_descriptor_metadata,
    source="authoring.q8_registry",
  )
  return registry


def truss_registry(
  *,
  topology: RegistryDescriptor | None = None,
  formulation: RegistryDescriptor | None = None,
  material: RegistryDescriptor | None = None,
) -> dict[RegistryKey, RegistryDescriptor]:
  """Compose a truss registry: reference implementations plus replacements.

  Mirrors :func:`q8_registry` for the qualified truss convention.
  """
  registry = truss_reference_registry()
  _replace(
    registry,
    (
      (TRUSS_TOPOLOGY_KEY, topology),
      (TRUSS_FORMULATION_KEY, formulation),
      (TRUSS_MATERIAL_KEY, material),
    ),
    family="truss",
    metadata_for=truss_descriptor_metadata,
    source="authoring.truss_registry",
  )
  return registry


def _continuum_descriptor_metadata(kind: str, name: str) -> dict[str, object]:
  """Resolve the qualified continuum metadata, including the stateful seam."""
  if (kind, name) == PLASTIC_MATERIAL_KEY:
    return isotropic_hardening_plasticity_metadata()
  return q8_descriptor_metadata(kind, name)


def plasticity_registry(
  *,
  topology: RegistryDescriptor | None = None,
  quadrature: RegistryDescriptor | None = None,
  formulation: RegistryDescriptor | None = None,
  material: RegistryDescriptor | None = None,
) -> dict[RegistryKey, RegistryDescriptor]:
  """Compose a plasticity registry: reference implementations plus replacements.

  Mirrors :func:`q8_registry` for the qualified stateful convention: the Q8
  reference implementations plus the first stateful law. Replacements keep
  the qualified registry key and metadata convention; only the implementation
  (its id, version, and binding) is yours. A mismatching key or metadata
  field fails here with a field-level diff, not later with a bare compile
  error.
  """
  registry = plasticity_reference_registry()
  _replace(
    registry,
    (
      (Q8_TOPOLOGY_KEY, topology),
      (Q8_QUADRATURE_KEY, quadrature),
      (Q8_FORMULATION_KEY, formulation),
      (PLASTIC_MATERIAL_KEY, material),
    ),
    family="plasticity",
    metadata_for=_continuum_descriptor_metadata,
    source="authoring.plasticity_registry",
  )
  return registry


def check_registry(
  spec: ModelSpec,
  registry: dict[RegistryKey, RegistryDescriptor],
) -> None:
  """Preflight descriptor metadata against the convention a model selects.

  For each descriptor the selected family requires, metadata is compared
  byte-wise exactly as the landed builders do — but a mismatch raises the
  same coded ``incompatible-registry-descriptor`` diagnostic with a
  field-level diff naming every disagreeing field, sourced to the offending
  spec element. Missing or malformed descriptors are left to the landed
  compiler diagnostics, which fire unchanged on delegation.
  """
  if type(spec) is not ModelSpec:
    msg = "check_registry requires an exact ModelSpec"
    raise TypeError(msg)
  if type(registry) is not dict:
    msg = "check_registry requires an exact registry dictionary"
    raise TypeError(msg)
  formulations = {region.formulation for region in spec.regions}
  if Q8_FORMULATION_KEY[1] in formulations:
    family = "Q8"
    metadata_for = _continuum_descriptor_metadata
  elif TRUSS_FORMULATION_KEY[1] in formulations:
    family = "truss"
    metadata_for = truss_descriptor_metadata
  else:
    return
  if (
    len(spec.mesh.cell_blocks) != 1
    or len(spec.materials) != 1
    or len(spec.regions) != 1
  ):
    return
  block = spec.mesh.cell_blocks[0]
  material = spec.materials[0]
  region = spec.regions[0]
  if family == "Q8":
    # The small-strain formulation is the open stateful seam: the compiler
    # selects the material descriptor by the spec's model name. Only the
    # pinned Q8 and plasticity conventions carry a qualified metadata
    # contract here; other stateful keys defer to the landed compiler's
    # binding re-declaration validation.
    material_key: RegistryKey = ("material", material.model)
    keys = (
      Q8_TOPOLOGY_KEY,
      Q8_QUADRATURE_KEY,
      Q8_FORMULATION_KEY,
      material_key,
    )
  else:
    keys = (TRUSS_TOPOLOGY_KEY, TRUSS_FORMULATION_KEY, TRUSS_MATERIAL_KEY)
  try:
    expected_by_key = {key: metadata_for(*key) for key in keys}
  except KeyError:
    return
  sources = {
    "topology": block.source,
    "quadrature": region.source,
    "formulation": region.source,
    "material": material.source,
  }
  for key in keys:
    descriptor = registry.get(key)
    if type(descriptor) is not RegistryDescriptor:
      continue
    try:
      expected = CanonicalManifest(expected_by_key[key])
      authored = descriptor.metadata.to_bytes()
    except (TypeError, ValueError):
      continue
    if authored != expected.to_bytes():
      raise_descriptor_mismatch(
        family=family,
        key=key,
        expected_metadata=expected_by_key[key],
        descriptor=descriptor,
        source=sources[key[0]],
      )
