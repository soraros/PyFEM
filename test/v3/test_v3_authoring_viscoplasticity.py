# SPDX-License-Identifier: MIT

"""Stateful viscoplasticity authoring: the v2 descriptor behind plain helpers.

Covers the ``viscoplasticity(...)`` parameter slice, the registry authors
(``viscoplasticity_law``/``viscoplasticity_registry``) with field-level
metadata-mismatch diagnostics over the v2 descriptor fields — the 14-float
state slots and the ``signal_ports`` time port included — default-registry
selection for stateful models, and the stateful persona: a Perzyna-branded
law authored end-to-end and stepped through the M27 transaction helpers,
verified bitwise against the M68 kernel oracle on the documented schedule
(the constant-time supra-yield gate step included). Time reaches the law
exclusively through the declared identity program coordinate — there is no
solverStat-style channel. The law ships no derivative kernel, so its
qualified parameters read 'constant' in the M58 sensitivity diagnostics —
pinned literally.
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
  VISCOPLASTIC_MATERIAL_KEY,
  viscoplasticity_reference_registry,
)
from pyfem.v3.compile.contracts import (
  StatefulContinuumKernelResult,
  StatefulContinuumSignalDerivative,
  StatefulContinuumSignalInput,
)
from pyfem.v3.compile.diagnostics import ModelCompilationError
from pyfem.v3.compile.system import compile_system
from pyfem.v3.driver.diagnostics import DriverPreparationError
from pyfem.v3.materials.perzyna_viscoplasticity import (
  PERZYNA_VISCOPLASTICITY_BINDING,
  perzyna_viscoplasticity_calibration,
  perzyna_viscoplasticity_initial_state,
  perzyna_viscoplasticity_kernel,
  perzyna_viscoplasticity_metadata,
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
  ProgramCoordinateSpec,
)

# The documented M68 oracle configuration
# (examples/materials/viscoplasticity/bar_tension.pro).
_E = 2.0e5
_NU = 0.3
_SYIELD = 250.0
_HARD = 1000.0
_GAMMA = 1.0e-3
_N = 1.0
_ROW_WIDTH = 14
_KAPPA_COLUMN = 12
_TIME_COLUMN = 13
# The documented M68 driver schedule: the fourth step holds time constant
# while strain advances ABOVE YIELD, exercising the legacy ``dtime > 0`` gate
# branch end-to-end through the driver.
_SCHEDULE = (
  (0.05, 2.0e-4),
  (0.15, 5.0e-4),
  (0.40, 2.5e-3),
  (0.40, 4.0e-3),
  (1.00, 5.5e-3),
)
# The quad8_patch node whose x displacement stays free, so every Newton
# iteration assembles and factorizes the tangent (node 7 sits at x = 0.5).
_FREE_NODE = 7


def _source(label: str) -> SourceContext:
  return SourceContext(source=label)


def _model(material: MaterialSpec | None = None) -> ModelSpec:
  selected = (
    material
    if material is not None
    else authoring.viscoplasticity(_E, _NU, _SYIELD, _HARD, _GAMMA, _N, id="steel")
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


def _session(system: CompiledSystem) -> authoring.NonlinearStaticSession:
  """A stepping session declaring the schedule-owned ``time`` coordinate.

  The declared ``time`` coordinate is the whole time channel: the M29 plan
  binds it to the law's identity signal port per substep — there is no
  solverStat-style back channel to replace.
  """
  return authoring.nonlinear_static(
    system,
    constraints=_ramp_constraints(),
    coordinates=(ProgramCoordinateSpec(name="time", kind="time"), "load"),
  )


def _time_signal(time: float) -> StatefulContinuumSignalInput:
  return StatefulContinuumSignalInput(
    port_id="time",
    values=np.array([time], dtype=np.float64),
    derivatives=(
      StatefulContinuumSignalDerivative("time", np.array([1.0], dtype=np.float64)),
    ),
  )


def _committed_ip_strains(system: CompiledSystem, values: np.ndarray) -> np.ndarray:
  """Recompute the committed per-integration-point 6-Voigt strains.

  The M68 parity harness expression (test_v3_viscoplasticity.py): the
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


class _StudentVPBinding:
  """The student's own rate-gated J2 law: the v2 stateful binding protocol."""

  def __call__(self, *parameters: float) -> np.ndarray:
    return perzyna_viscoplasticity_calibration(*parameters)

  def descriptor_metadata(self) -> dict[str, object]:
    return perzyna_viscoplasticity_metadata()

  def kernel(
    self,
    strains: np.ndarray,
    accepted_rows: np.ndarray,
    calibration: np.ndarray,
    signals: tuple[StatefulContinuumSignalInput, ...],
  ) -> StatefulContinuumKernelResult:
    return perzyna_viscoplasticity_kernel(strains, accepted_rows, calibration, signals)

  def initial_state(
    self,
    parameters: tuple[float, ...],
    layout: OperatorStateLayout,
  ) -> np.ndarray:
    return perzyna_viscoplasticity_initial_state(parameters, layout)


def _spoofed_viscoplasticity_descriptor(
  metadata: dict[str, object],
) -> RegistryDescriptor:
  return RegistryDescriptor(
    kind=VISCOPLASTIC_MATERIAL_KEY[0],
    name=VISCOPLASTIC_MATERIAL_KEY[1],
    version="1",
    implementation_id="spoofed-for-diagnostics",
    metadata=metadata,
    binding=PERZYNA_VISCOPLASTICITY_BINDING,
  )


# --- material parameter slice ---------------------------------------------------


def test_viscoplasticity_authors_the_qualified_parameter_schema() -> None:
  material = authoring.viscoplasticity(2.0e5, 0.3, 250.0, 1000.0, 1.0e-3, 1.0)
  assert material.id == "material"
  assert material.model == "perzyna-viscoplasticity"
  assert material.parameters == (
    MaterialParameterSpec(
      "youngs_modulus",
      2.0e5,
      _source("authoring.viscoplasticity:youngs_modulus"),
    ),
    MaterialParameterSpec(
      "poisson_ratio",
      0.3,
      _source("authoring.viscoplasticity:poisson_ratio"),
    ),
    MaterialParameterSpec(
      "initial_yield_stress",
      250.0,
      _source("authoring.viscoplasticity:initial_yield_stress"),
    ),
    MaterialParameterSpec(
      "hardening_slope",
      1000.0,
      _source("authoring.viscoplasticity:hardening_slope"),
    ),
    MaterialParameterSpec(
      "fluidity",
      1.0e-3,
      _source("authoring.viscoplasticity:fluidity"),
    ),
    MaterialParameterSpec(
      "rate_exponent",
      1.0,
      _source("authoring.viscoplasticity:rate_exponent"),
    ),
  )
  assert material.source == _source("authoring.viscoplasticity")

  integral = authoring.viscoplasticity(200000, 0, 250, 1000, 1, 1, id="steel")
  assert integral.id == "steel"
  assert integral.parameters[0].value == 2.0e5
  assert type(integral.parameters[0].value) is float
  assert integral.parameters[5].value == 1.0
  assert type(integral.parameters[5].value) is float


def test_viscoplasticity_rejects_non_numeric_and_non_finite_values() -> None:
  with pytest.raises(TypeError, match="exact number"):
    authoring.viscoplasticity(True, 0.3, 250.0, 1000.0, 1.0e-3, 1.0)  # type: ignore[arg-type]
  with pytest.raises(TypeError, match="exact number"):
    authoring.viscoplasticity(2.0e5, 0.3, 250.0, 1000.0, 1.0e-3, "1")  # type: ignore[arg-type]
  with pytest.raises(ValueError, match="finite"):
    authoring.viscoplasticity(2.0e5, float("nan"), 250.0, 1000.0, 1.0e-3, 1.0)
  with pytest.raises(ValueError, match="finite"):
    authoring.viscoplasticity(2.0e5, 0.3, 250.0, 1000.0, 1.0e-3, float("inf"))
  with pytest.raises(TypeError, match="material id"):
    authoring.viscoplasticity(2.0e5, 0.3, 250.0, 1000.0, 1.0e-3, 1.0, id=None)  # type: ignore[arg-type]


def test_small_strain_continuum_accepts_the_viscoplastic_material_slice() -> None:
  mesh = authoring.quad8_patch()
  material = authoring.viscoplasticity(
    2.0e5, 0.3, 250.0, 1000.0, 1.0e-3, 1.0, id="steel"
  )
  model = authoring.small_strain_continuum(mesh, material=material)
  assert model.materials == (material,)
  (region,) = model.regions
  assert region.material_id == "steel"
  assert region.formulation == "small-strain-continuum"
  assert region.quadrature == "gauss-3x3"
  with pytest.raises(ValueError, match="viscoplasticity"):
    authoring.small_strain_continuum(
      authoring.quad8_patch(),
      material=authoring.uniaxial_elastic(1.0, 1.0),
    )


# --- registry authors and composition --------------------------------------------


def test_viscoplasticity_law_pins_the_convention() -> None:
  law = authoring.viscoplasticity_law(
    _StudentVPBinding(),
    implementation_id="student-vp-v1",
  )
  assert law.key == VISCOPLASTIC_MATERIAL_KEY
  assert law.version == "1"
  assert law.implementation_id == "student-vp-v1"
  assert (
    law.metadata.to_bytes()
    == CanonicalManifest(perzyna_viscoplasticity_metadata()).to_bytes()
  )


def test_viscoplasticity_registry_is_byte_identical_to_the_reference_registry() -> None:
  for key, descriptor in viscoplasticity_reference_registry().items():
    layer_descriptor = authoring.viscoplasticity_registry()[key]
    assert layer_descriptor.manifest.to_bytes() == descriptor.manifest.to_bytes()
    assert layer_descriptor.metadata.to_bytes() == descriptor.metadata.to_bytes()


def test_viscoplasticity_registry_replacement_must_keep_the_convention_key() -> None:
  foreign = RegistryDescriptor(
    kind="material",
    name="my-own-material-name",
    version="1",
    implementation_id="foreign",
    metadata=perzyna_viscoplasticity_metadata(),
    binding=PERZYNA_VISCOPLASTICITY_BINDING,
  )
  with pytest.raises(ValueError, match="does not match the qualified convention key"):
    authoring.viscoplasticity_registry(material=foreign)
  with pytest.raises(TypeError, match="exact RegistryDescriptor"):
    authoring.viscoplasticity_registry(
      material=PERZYNA_VISCOPLASTICITY_BINDING,  # type: ignore[arg-type]
    )


def test_compile_defaults_to_the_viscoplasticity_registry_for_stateful_models() -> None:
  model = _model()
  default = authoring.compile(model)
  explicit = authoring.compile(model, authoring.viscoplasticity_registry())
  assert default.content_fingerprint == explicit.content_fingerprint
  reference = compile_system(model, viscoplasticity_reference_registry())
  assert default.content_fingerprint == reference.content_fingerprint
  layout = default.operators[0].header.state_layout
  assert layout.schema == (
    "pyfem-v3-perzyna-viscoplastic-state-v1|epsilon_e:6,epsilon_p:6,kappa:1,time:1"
  )
  assert layout.initial_rows is not None
  assert np.all(layout.initial_rows.values == 0.0)
  (port,) = default.operators[0].header.signal_ports
  assert port.port_id == "time"
  assert port.signal_id == "time"


# --- field-level metadata-mismatch diagnostics ------------------------------------


def test_viscoplasticity_metadata_mismatch_names_the_v2_descriptor_fields() -> None:
  metadata = perzyna_viscoplasticity_metadata()
  metadata["tangent_class"] = "algorithmic-nonsymmetric"
  metadata["state_slots"][0]["width"] = 3  # type: ignore[index]
  metadata["signal_ports"][0]["signal_id"] = "temperature"  # type: ignore[index]
  del metadata["state_schema"]
  metadata["flow_table"] = [[1.0e-3, 1.0]]
  spoofed = _spoofed_viscoplasticity_descriptor(metadata)

  with pytest.raises(ModelCompilationError) as captured:
    authoring.viscoplasticity_registry(material=spoofed)
  (diagnostic,) = captured.value.diagnostics
  assert diagnostic.code == "incompatible-registry-descriptor"
  assert "('material', 'perzyna-viscoplasticity')" in diagnostic.message
  assert (
    "field 'tangent_class': expected 'algorithmic-symmetric', authored "
    "'algorithmic-nonsymmetric'" in diagnostic.message
  )
  assert "field 'state_slots[0].width': expected 6, authored 3" in diagnostic.message
  assert (
    "field 'signal_ports[0].signal_id': expected 'time', authored 'temperature'"
    in diagnostic.message
  )
  assert "missing field 'state_schema'" in diagnostic.message
  assert "unexpected field 'flow_table'" in diagnostic.message
  assert diagnostic.source == _source("authoring.viscoplasticity_registry:material")


def test_viscoplasticity_metadata_mismatch_fires_at_compile_with_field_diff() -> None:
  metadata = perzyna_viscoplasticity_metadata()
  metadata["state_slots"][2]["name"] = "history"  # type: ignore[index]
  registry = authoring.viscoplasticity_registry()
  registry[VISCOPLASTIC_MATERIAL_KEY] = _spoofed_viscoplasticity_descriptor(metadata)

  with pytest.raises(ModelCompilationError) as captured:
    authoring.compile(_model(), registry)
  (diagnostic,) = captured.value.diagnostics
  assert diagnostic.code == "incompatible-registry-descriptor"
  assert (
    "field 'state_slots[2].name': expected 'kappa', authored 'history'"
    in diagnostic.message
  )
  assert diagnostic.source == _source("authoring.viscoplasticity")


def test_landed_diagnostics_pass_through_unchanged() -> None:
  missing = authoring.viscoplasticity_registry()
  del missing[VISCOPLASTIC_MATERIAL_KEY]
  with pytest.raises(ModelCompilationError, match="registry-capture-failed"):
    authoring.compile(_model(), missing)
  with pytest.raises(ModelCompilationError, match="material-binding-failed"):
    authoring.compile(
      _model(authoring.viscoplasticity(_E, _NU, -1.0, _HARD, _GAMMA, _N))
    )


def test_check_registry_defers_stateful_keys_without_a_qualified_convention() -> None:
  model = _model()
  foreign = replace(
    model,
    materials=(replace(model.materials[0], model="research-law-v0"),),
  )
  authoring.check_registry(foreign, authoring.viscoplasticity_registry())
  with pytest.raises(ModelCompilationError, match="registry-capture-failed"):
    authoring.compile(foreign, authoring.viscoplasticity_registry())


# --- sensitivity surface: no derivative kernel reads 'constant' -------------------


def test_viscoplasticity_parameters_read_constant_in_the_sensitivity_surface() -> None:
  """The wave-13 law ships no derivative kernel: the M58 convention resolver.

  Every qualified parameter of the landed perzyna-viscoplasticity convention
  is baked into the compiled calibration, so a sensitivity request fails
  pre-substep with the parameterized-vs-constant diff, pinned literally.
  """
  session = _session(authoring.compile(_model()))
  with pytest.raises(DriverPreparationError) as captured:
    session.run(
      {"time": 0.0, "load": 0.0},
      {"time": 0.05, "load": 2.0e-4},
      sensitivities=("initial_yield_stress",),
    )
  (diagnostic,) = captured.value.diagnostics
  assert diagnostic.code == "unknown-sensitivity-parameter"
  assert (
    "field 'sensitivities.initial_yield_stress': expected {'parameter': "
    "'initial_yield_stress'}, authored 'constant'" in diagnostic.message
  )
  assert (
    "declared differentiable parameters: (); declared constant parameters: "
    "('youngs_modulus', 'poisson_ratio', 'initial_yield_stress', "
    "'hardening_slope', 'fluidity', 'rate_exponent')" in diagnostic.message
  )
  assert diagnostic.source == _source("authoring.run:sensitivities")
  assert session.driver.statistics.evaluation_count == 0
  assert session.ordinal == 0


# --- stateful persona: authored, stepped, oracle-checked ---------------------------


def test_persona_authors_and_steps_the_rate_gated_law_end_to_end() -> None:
  """Persona: viscoplasticity() plus the M27 helpers, oracle-checked.

  The signal-consuming law authors exactly like an elastic one — one
  material call, one model call, one compile — then steps the documented
  M68 schedule through ``nonlinear_static`` with the ``time`` program
  coordinate bound per step. The committed rows equal the batched M68
  kernel stepped on the committed per-integration-point strain path,
  bitwise (the driver stages the trial rows of the converged iterate); the
  ``time`` slot records each substep's bound time at every integration
  point — the identity port is the only time channel. The constant-time
  supra-yield fourth step freezes kappa (the documented gate). A manual
  begin/reject trial leaves the committed bytes identical.
  """
  model = _model()
  system = authoring.compile(model)
  session = _session(system)

  calibration = perzyna_viscoplasticity_calibration(_E, _NU, _SYIELD, _HARD, _GAMMA, _N)
  oracle_rows = np.zeros((9, _ROW_WIDTH))
  base = {"time": 0.0, "load": 0.0}
  kappa_after_plastic = None
  for step, (time, eps) in enumerate(_SCHEDULE, 1):
    result = session.run(base, {"time": time, "load": eps})
    assert result.status is authoring.DriverStatus.COMPLETED
    base = {"time": time, "load": eps}
    oracle = perzyna_viscoplasticity_kernel(
      _committed_ip_strains(session.system, session.accepted_coefficients()),
      oracle_rows,
      calibration,
      (_time_signal(time),),
    )
    assert oracle.status is authoring.EvaluationStatus.OK
    oracle_rows = oracle.trial_rows
    rows = session.accepted_state("cells")
    assert rows.shape == (9, _ROW_WIDTH)
    np.testing.assert_array_equal(rows, oracle_rows)
    np.testing.assert_array_equal(rows[:, _TIME_COLUMN], time)
    assert session.ordinal == step
    if step == 3:
      assert np.all(rows[:, _KAPPA_COLUMN] > 0.0)
      kappa_after_plastic = rows[:, _KAPPA_COLUMN].copy()
    if step == 4:
      # The constant-time supra-yield substep: the gate froze plastic flow.
      np.testing.assert_array_equal(rows[:, _KAPPA_COLUMN], kappa_after_plastic)
    if step == 5:
      assert np.all(rows[:, _KAPPA_COLUMN] > kappa_after_plastic)
  # Homogeneous strain: all nine integration points agree to solver tolerance.
  np.testing.assert_allclose(
    rows, np.broadcast_to(rows[0], rows.shape), rtol=1.0e-9, atol=1.0e-10
  )

  # A manual M27 trial observes the committed state; rejecting leaves the
  # accepted bytes identical.
  before = session.snapshot()
  trial = session.begin()
  assert trial.ordinal == 5
  np.testing.assert_array_equal(trial.accepted_state("cells"), oracle_rows)
  trial.reject()
  assert session.snapshot() == before


def test_dtime_gate_changes_the_response_through_the_authoring_path() -> None:
  """The same strain step with advancing versus held time flows differently.

  The positive control for the port wiring: schedule-owned time reaches the
  law through the declared ``time`` coordinate, so the held-time session's
  fourth step is gated elastic while the advancing session's flows (the
  documented M68 gate control, authored end-to-end).
  """
  advanced = _session(authoring.compile(_model()))
  held = _session(authoring.compile(_model()))
  base = {"time": 0.0, "load": 0.0}
  for session in (advanced, held):
    assert (
      session.run(base, {"time": 0.4, "load": 2.5e-3}).status
      is authoring.DriverStatus.COMPLETED
    )
  assert (
    advanced.run({"time": 0.4, "load": 2.5e-3}, {"time": 1.0, "load": 4.0e-3}).status
    is authoring.DriverStatus.COMPLETED
  )
  assert (
    held.run({"time": 0.4, "load": 2.5e-3}, {"time": 0.4, "load": 4.0e-3}).status
    is authoring.DriverStatus.COMPLETED
  )
  advanced_rows = advanced.accepted_state("cells")
  held_rows = held.accepted_state("cells")
  # The held-time run froze kappa; the advancing run flowed further.
  assert np.all(advanced_rows[:, _KAPPA_COLUMN] > held_rows[:, _KAPPA_COLUMN])
  assert advanced_rows.tobytes() != held_rows.tobytes()


def test_student_law_binding_steps_bitwise_like_the_reference_law() -> None:
  """A binding behind viscoplasticity_law steps like the reference.

  The student keeps the qualified convention (one descriptor line, one
  registry line) and supplies only the binding — the 14-float state layout
  and the time port included; stepping both systems over the documented
  schedule commits byte-identical state.
  """
  law = authoring.viscoplasticity_law(
    _StudentVPBinding(),
    implementation_id="student-vp-v1",
  )
  model = _model()
  student = _session(
    authoring.compile(model, authoring.viscoplasticity_registry(material=law))
  )
  reference = _session(authoring.compile(model))
  base = {"time": 0.0, "load": 0.0}
  for time, eps in _SCHEDULE:
    target = {"time": time, "load": eps}
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
