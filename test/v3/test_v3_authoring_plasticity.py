# SPDX-License-Identifier: MIT

"""Stateful plasticity authoring: the v2 descriptor behind plain helpers.

Covers the ``plasticity(...)`` parameter slice, the registry authors
(``plasticity_law``/``plasticity_registry``) with field-level
metadata-mismatch diagnostics over the v2 descriptor fields, default-registry
selection for stateful models, and the stateful persona: a J2 law authored
end-to-end and stepped through the M27 transaction helpers, verified bitwise
against the M25 kernel oracle on the documented ramp.
"""

from __future__ import annotations

import sys
from dataclasses import replace

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from pyfem.v3 import authoring
from pyfem.v3.compile.continuum import (
  PLASTIC_MATERIAL_KEY,
  plasticity_reference_registry,
)
from pyfem.v3.compile.contracts import StatefulContinuumKernelResult
from pyfem.v3.compile.diagnostics import ModelCompilationError
from pyfem.v3.compile.system import compile_system
from pyfem.v3.materials.isotropic_hardening_plasticity import (
  ISOTROPIC_HARDENING_PLASTICITY_BINDING,
  isotropic_hardening_calibration,
  isotropic_hardening_initial_state,
  isotropic_hardening_plasticity_kernel,
  isotropic_hardening_plasticity_metadata,
)
from pyfem.v3.model.operator import OperatorStateLayout
from pyfem.v3.model.provenance import CanonicalManifest
from pyfem.v3.model.registry import RegistryDescriptor
from pyfem.v3.model.system import CompiledSystem
from pyfem.v3.spec import MaterialParameterSpec, MaterialSpec, ModelSpec, SourceContext
from pyfem.v3.spec.program import (
  AffineCoefficientSpec,
  AffineValueSpec,
  DofRef,
  PrescribedDofSpec,
  ProgramConstraintSpec,
)

# The documented M25 parity configuration (test_v3_stateful_plasticity.py).
_E = 210000.0
_NU = 0.3
_SYIELD = 250.0
_HARD = 1000.0
_LOAD_RAMP = (0.001, 0.002, 0.004)
# The quad8_patch node whose x displacement stays free, so every Newton
# iteration assembles and factorizes the tangent (node 7 sits at x = 0.5).
_FREE_NODE = 7


def _source(label: str) -> SourceContext:
  return SourceContext(source=label)


def _model(material: MaterialSpec | None = None) -> ModelSpec:
  selected = (
    material
    if material is not None
    else authoring.plasticity(_E, _NU, _SYIELD, _HARD, id="steel")
  )
  return authoring.small_strain_continuum(authoring.quad8_patch(), material=selected)


def _ramp_constraints() -> tuple[ProgramConstraintSpec, ...]:
  """Authored ``fixed`` y-fixities plus hand-written eps_xx ramp prescriptions.

  The documented mix-in idiom: helpers cover the plain fixities, and landed
  declarations prescribing ``u_x = x * load`` pass through alongside them.
  """
  mesh = authoring.quad8_patch()
  constraints: list[ProgramConstraintSpec] = list(
    authoring.fixed(nodes=tuple(node.id for node in mesh.nodes), components=("y",))
  )
  for node in mesh.nodes:
    if node.id == _FREE_NODE:
      continue
    constraints.append(
      PrescribedDofSpec(
        id=f"ux-{node.id}",
        target=DofRef(node_id=node.id, field_id="displacement", component="x"),
        value=AffineValueSpec(
          coefficients=(AffineCoefficientSpec("load", node.coordinates[0]),),
        ),
      )
    )
  return tuple(constraints)


def _committed_ip_strains(system: CompiledSystem, values: np.ndarray) -> np.ndarray:
  """Recompute the committed per-integration-point 6-Voigt strains.

  The M25 parity harness expression (test_v3_stateful_plasticity.py): the
  operator's own physical strain-displacement map applied to the committed
  coefficient vector.
  """
  operator = system.operators[0]
  payload = operator.payload
  b_matrix = (
    payload.normalized_strain_displacement.values
    / payload.geometry_scales.values[:, None, None, None]
  )
  gather = operator.header.ports[0].coefficient_map.values
  strain3 = np.einsum("epai,ei->epa", b_matrix, values[gather], optimize=True)
  strains = np.zeros((np.prod(strain3.shape[:2]), 6), dtype=np.float64)
  flat = strain3.reshape(-1, 3)
  strains[:, 0] = flat[:, 0]
  strains[:, 1] = flat[:, 1]
  strains[:, 5] = flat[:, 2]
  return strains


class _StudentJ2Binding:
  """The student's own J2 law: the v2 stateful binding protocol in plain methods."""

  def __call__(self, *parameters: float) -> np.ndarray:
    return isotropic_hardening_calibration(*parameters)

  def descriptor_metadata(self) -> dict[str, object]:
    return isotropic_hardening_plasticity_metadata()

  def kernel(
    self,
    strains: np.ndarray,
    accepted_rows: np.ndarray,
    calibration: np.ndarray,
  ) -> StatefulContinuumKernelResult:
    return isotropic_hardening_plasticity_kernel(strains, accepted_rows, calibration)

  def initial_state(
    self,
    parameters: tuple[float, ...],
    layout: OperatorStateLayout,
  ) -> np.ndarray:
    return isotropic_hardening_initial_state(parameters, layout)


def _spoofed_plasticity_descriptor(metadata: dict[str, object]) -> RegistryDescriptor:
  return RegistryDescriptor(
    kind=PLASTIC_MATERIAL_KEY[0],
    name=PLASTIC_MATERIAL_KEY[1],
    version="1",
    implementation_id="spoofed-for-diagnostics",
    metadata=metadata,
    binding=ISOTROPIC_HARDENING_PLASTICITY_BINDING,
  )


# --- material parameter slice ---------------------------------------------------


def test_plasticity_authors_the_qualified_stateful_parameter_schema() -> None:
  material = authoring.plasticity(210.0e3, 0.3, 250.0, 1000.0)
  assert material.id == "material"
  assert material.model == "isotropic-hardening-plasticity"
  assert material.parameters == (
    MaterialParameterSpec(
      "youngs_modulus",
      210.0e3,
      _source("authoring.plasticity:youngs_modulus"),
    ),
    MaterialParameterSpec(
      "poisson_ratio",
      0.3,
      _source("authoring.plasticity:poisson_ratio"),
    ),
    MaterialParameterSpec(
      "initial_yield_stress",
      250.0,
      _source("authoring.plasticity:initial_yield_stress"),
    ),
    MaterialParameterSpec(
      "hardening_slope",
      1000.0,
      _source("authoring.plasticity:hardening_slope"),
    ),
  )
  assert material.source == _source("authoring.plasticity")

  integral = authoring.plasticity(210000, 0, 250, 1000, id="steel")
  assert integral.id == "steel"
  assert integral.parameters[0].value == 210000.0
  assert type(integral.parameters[0].value) is float


def test_plasticity_rejects_non_numeric_and_non_finite_values() -> None:
  with pytest.raises(TypeError, match="exact number"):
    authoring.plasticity(True, 0.3, 250.0, 1000.0)  # type: ignore[arg-type]
  with pytest.raises(TypeError, match="exact number"):
    authoring.plasticity(210.0e3, 0.3, "250", 1000.0)  # type: ignore[arg-type]
  with pytest.raises(ValueError, match="finite"):
    authoring.plasticity(210.0e3, float("nan"), 250.0, 1000.0)
  with pytest.raises(ValueError, match="finite"):
    authoring.plasticity(210.0e3, 0.3, 250.0, float("inf"))
  with pytest.raises(TypeError, match="material id"):
    authoring.plasticity(1.0, 0.0, 1.0, 0.0, id=None)  # type: ignore[arg-type]


def test_small_strain_continuum_accepts_the_stateful_material_slice() -> None:
  mesh = authoring.quad8_patch()
  material = authoring.plasticity(1.0, 0.0, 1.0, 0.0, id="steel")
  model = authoring.small_strain_continuum(mesh, material=material)
  assert model.materials == (material,)
  (region,) = model.regions
  assert region.material_id == "steel"
  assert region.formulation == "small-strain-continuum"
  assert region.quadrature == "gauss-3x3"
  with pytest.raises(ValueError, match="plasticity"):
    authoring.small_strain_continuum(
      authoring.quad8_patch(),
      material=authoring.uniaxial_elastic(1.0, 1.0),
    )


# --- registry authors and composition --------------------------------------------


def test_plasticity_law_pins_the_convention_so_users_never_copy_metadata() -> None:
  law = authoring.plasticity_law(
    _StudentJ2Binding(),
    implementation_id="student-j2-v1",
  )
  assert law.key == PLASTIC_MATERIAL_KEY
  assert law.version == "1"
  assert law.implementation_id == "student-j2-v1"
  assert (
    law.metadata.to_bytes()
    == CanonicalManifest(isotropic_hardening_plasticity_metadata()).to_bytes()
  )


def test_plasticity_registry_is_byte_identical_to_the_reference_registry() -> None:
  for key, descriptor in plasticity_reference_registry().items():
    layer_descriptor = authoring.plasticity_registry()[key]
    assert layer_descriptor.manifest.to_bytes() == descriptor.manifest.to_bytes()
    assert layer_descriptor.metadata.to_bytes() == descriptor.metadata.to_bytes()


def test_plasticity_registry_replacement_must_keep_the_convention_key() -> None:
  foreign = RegistryDescriptor(
    kind="material",
    name="my-own-material-name",
    version="1",
    implementation_id="foreign",
    metadata=isotropic_hardening_plasticity_metadata(),
    binding=ISOTROPIC_HARDENING_PLASTICITY_BINDING,
  )
  with pytest.raises(ValueError, match="does not match the qualified convention key"):
    authoring.plasticity_registry(material=foreign)
  with pytest.raises(TypeError, match="exact RegistryDescriptor"):
    authoring.plasticity_registry(
      material=ISOTROPIC_HARDENING_PLASTICITY_BINDING,  # type: ignore[arg-type]
    )


def test_compile_defaults_to_the_plasticity_registry_for_stateful_models() -> None:
  model = _model()
  default = authoring.compile(model)
  explicit = authoring.compile(model, authoring.plasticity_registry())
  assert default.content_fingerprint == explicit.content_fingerprint
  reference = compile_system(model, plasticity_reference_registry())
  assert default.content_fingerprint == reference.content_fingerprint
  layout = default.operators[0].header.state_layout
  assert layout.schema == (
    "pyfem-v3-j2-isotropic-hardening-state-v1|sigma:6,epsilon_e:6,epsilon_p:6,kappa:1"
  )


# --- field-level metadata-mismatch diagnostics ------------------------------------


def test_plasticity_metadata_mismatch_names_the_v2_descriptor_fields() -> None:
  metadata = isotropic_hardening_plasticity_metadata()
  metadata["tangent_class"] = "algorithmic-nonsymmetric"
  metadata["state_slots"][3]["width"] = 2  # type: ignore[index]
  metadata["internal_voigt_order"] = ["xx", "yy", "zz", "xy", "yz", "zx"]
  del metadata["state_schema"]
  metadata["hardening_table"] = [[0.0, 250.0], [1.0, 1250.0]]
  spoofed = _spoofed_plasticity_descriptor(metadata)

  with pytest.raises(ModelCompilationError) as captured:
    authoring.plasticity_registry(material=spoofed)
  (diagnostic,) = captured.value.diagnostics
  assert diagnostic.code == "incompatible-registry-descriptor"
  assert "('material', 'isotropic-hardening-plasticity')" in diagnostic.message
  assert (
    "field 'tangent_class': expected 'algorithmic-symmetric', authored "
    "'algorithmic-nonsymmetric'" in diagnostic.message
  )
  assert "field 'state_slots[3].width': expected 1, authored 2" in diagnostic.message
  assert (
    "field 'internal_voigt_order[3]': expected 'yz', authored 'xy'"
    in diagnostic.message
  )
  assert "missing field 'state_schema'" in diagnostic.message
  assert "unexpected field 'hardening_table'" in diagnostic.message
  assert diagnostic.source == _source("authoring.plasticity_registry:material")


def test_plasticity_metadata_mismatch_fires_at_compile_with_field_diff() -> None:
  metadata = isotropic_hardening_plasticity_metadata()
  metadata["state_slots"][0]["name"] = "stress"  # type: ignore[index]
  registry = authoring.plasticity_registry()
  registry[PLASTIC_MATERIAL_KEY] = _spoofed_plasticity_descriptor(metadata)

  with pytest.raises(ModelCompilationError) as captured:
    authoring.compile(_model(), registry)
  (diagnostic,) = captured.value.diagnostics
  assert diagnostic.code == "incompatible-registry-descriptor"
  assert (
    "field 'state_slots[0].name': expected 'sigma', authored 'stress'"
    in diagnostic.message
  )
  assert diagnostic.source == _source("authoring.plasticity")


def test_landed_diagnostics_pass_through_unchanged() -> None:
  missing = authoring.plasticity_registry()
  del missing[PLASTIC_MATERIAL_KEY]
  with pytest.raises(ModelCompilationError, match="registry-capture-failed"):
    authoring.compile(_model(), missing)
  with pytest.raises(ModelCompilationError, match="material-binding-failed"):
    authoring.compile(_model(authoring.plasticity(_E, _NU, -1.0, _HARD)))


def test_check_registry_defers_stateful_keys_without_a_qualified_convention() -> None:
  model = _model()
  foreign = replace(
    model,
    materials=(replace(model.materials[0], model="research-law-v0"),),
  )
  authoring.check_registry(foreign, authoring.plasticity_registry())
  with pytest.raises(ModelCompilationError, match="registry-capture-failed"):
    authoring.compile(foreign, authoring.plasticity_registry())


# --- stateful persona: authored, stepped, oracle-checked ---------------------------


def test_persona_authors_and_steps_the_stateful_law_end_to_end() -> None:
  """Persona: plasticity() plus the M27 stepping helpers, oracle-checked.

  The stateful law authors exactly like an elastic one — one material call,
  one model call, one compile — then steps the documented M25 ramp through
  ``nonlinear_static``. The committed rows equal the batched M25 kernel
  stepped on the committed per-integration-point strain path, bitwise (the
  driver stages the trial rows of the converged iterate), and a manual
  begin/reject trial leaves the committed bytes identical.
  """
  model = _model()
  system = authoring.compile(model)
  session = authoring.nonlinear_static(system, constraints=_ramp_constraints())

  calibration = isotropic_hardening_calibration(_E, _NU, _SYIELD, _HARD)
  oracle_rows = np.zeros((9, 19))
  base = {"load": 0.0}
  for step, eps in enumerate(_LOAD_RAMP, 1):
    result = session.run(base, {"load": eps})
    assert result.status is authoring.DriverStatus.COMPLETED
    base = {"load": eps}
    oracle = isotropic_hardening_plasticity_kernel(
      _committed_ip_strains(session.system, session.accepted_coefficients()),
      oracle_rows,
      calibration,
    )
    assert oracle.status is authoring.EvaluationStatus.OK
    oracle_rows = oracle.trial_rows
    rows = session.accepted_state("cells")
    assert rows.shape == (9, 19)
    np.testing.assert_array_equal(rows, oracle_rows)
    assert session.ordinal == step
  assert rows[0, 18] > 0.0  # kappa: the ramp went plastic

  # A manual M27 trial observes the committed state; rejecting leaves the
  # accepted bytes identical.
  before = session.snapshot()
  trial = session.begin()
  assert trial.ordinal == 3
  np.testing.assert_array_equal(trial.accepted_state("cells"), oracle_rows)
  trial.reject()
  assert session.snapshot() == before


def test_student_law_binding_steps_bitwise_like_the_reference_law() -> None:
  """A binding behind plasticity_law compiles and steps like the reference.

  The student keeps the qualified convention (one descriptor line, one
  registry line) and supplies only the binding; stepping both systems over
  the documented ramp commits byte-identical state.
  """
  law = authoring.plasticity_law(
    _StudentJ2Binding(),
    implementation_id="student-j2-v1",
  )
  model = _model()
  student = authoring.nonlinear_static(
    authoring.compile(model, authoring.plasticity_registry(material=law)),
    constraints=_ramp_constraints(),
  )
  reference = authoring.nonlinear_static(
    authoring.compile(model),
    constraints=_ramp_constraints(),
  )
  base = {"load": 0.0}
  for eps in _LOAD_RAMP:
    target = {"load": eps}
    assert student.run(base, target).status is authoring.DriverStatus.COMPLETED
    assert reference.run(base, target).status is authoring.DriverStatus.COMPLETED
    base = target
  np.testing.assert_array_equal(
    student.accepted_state("cells"),
    reference.accepted_state("cells"),
  )
  np.testing.assert_array_equal(
    student.accepted_coefficients(),
    reference.accepted_coefficients(),
  )
