# SPDX-License-Identifier: MIT

"""Stateful skorohod-olevsky authoring: the v2 descriptor behind plain helpers.

Covers the ``skorohod_olevsky(...)`` parameter slice, the registry authors
(``skorohod_olevsky_law``/``skorohod_olevsky_registry``) with field-level
metadata-mismatch diagnostics over the v2 descriptor fields — the 14-float
state slots and the ``signal_ports`` time port included — default-registry
selection for stateful models, and the stateful persona: the explicit
viscous-sintering law authored end-to-end on its documented activated
constants and stepped through the M27 transaction helpers, verified bitwise
against the M68 kernel oracle (free sintering plus the pressure leg). Time
reaches the law exclusively through the declared identity program
coordinate — there is no solverStat-style channel. The law ships no
derivative kernel, so its qualified parameters read 'constant' in the M58
sensitivity diagnostics — pinned literally.
"""

from __future__ import annotations

import sys
from dataclasses import replace

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from pyfem.v3 import authoring
from pyfem.v3.compile.continuum import SOVS_MATERIAL_KEY, sovs_reference_registry
from pyfem.v3.compile.contracts import (
  StatefulContinuumKernelResult,
  StatefulContinuumSignalDerivative,
  StatefulContinuumSignalInput,
)
from pyfem.v3.compile.diagnostics import ModelCompilationError
from pyfem.v3.compile.system import compile_system
from pyfem.v3.driver.diagnostics import DriverPreparationError
from pyfem.v3.materials.skorohod_olevsky import (
  SKOROHOD_OLEVSKY_BINDING,
  skorohod_olevsky_calibration,
  skorohod_olevsky_initial_state,
  skorohod_olevsky_kernel,
  skorohod_olevsky_metadata,
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

# The documented M68 activated configuration (test_v3_sovs.py): the shipped
# decks are dormant (eta_ref ~ 1e28), so the battery owns the activated
# constants that exercise the viscous machinery.
_ETA0 = 1.0e10
_Q = 1.0
_T = 1600.0
_RHO0 = 0.6
_SIGMA_SINT = 1.0e6
_R = 8.314
_N_VOL = 2.0
_N_SHEAR = 1.0
_ROW_WIDTH = 14
_RHO_COLUMN = 12
_TIME_COLUMN = 13
# The documented M68 driver schedule: three free-sintering steps at zero
# strain, then the pressure-assisted leg.
_SCHEDULE = (
  (0.01, 0.0),
  (0.02, 0.0),
  (0.03, 0.0),
  (0.04, -2.0e-4),
  (0.05, -4.0e-4),
)
# The quad8_patch node whose x displacement stays free, so every Newton
# iteration assembles and factorizes the tangent (node 7 sits at x = 0.5).
_FREE_NODE = 7


def _source(label: str) -> SourceContext:
  return SourceContext(source=label)


def _material(**overrides: float) -> MaterialSpec:
  values = {
    "eta0": _ETA0,
    "Q": _Q,
    "T": _T,
    "rho0": _RHO0,
    "sigma_sint": _SIGMA_SINT,
    "R": _R,
    "n_vol": _N_VOL,
    "n_shear": _N_SHEAR,
  }
  values.update(overrides)
  return authoring.skorohod_olevsky(
    values["eta0"],
    values["Q"],
    values["T"],
    values["rho0"],
    values["sigma_sint"],
    values["R"],
    values["n_vol"],
    values["n_shear"],
    id="compact",
  )


def _model(material: MaterialSpec | None = None) -> ModelSpec:
  return authoring.small_strain_continuum(
    authoring.quad8_patch(),
    material=_material() if material is None else material,
  )


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
  solverStat-style back channel to replace. The tolerance is 1e-6 absolute:
  the law's hard-coded E = 100 GPa modulus puts assembled internal forces at
  ~1e7 on the pressure steps, above the default cancellation floor (the M68
  harness rationale); the bitwise oracle reads the committed strains either
  way.
  """
  return authoring.nonlinear_static(
    system,
    constraints=_ramp_constraints(),
    coordinates=(ProgramCoordinateSpec(name="time", kind="time"), "load"),
    settings=authoring.NonlinearStaticSettings(tolerance=1.0e-6),
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

  The M68 parity harness expression (test_v3_sovs.py): the operator's own
  physical strain-displacement map applied to the committed coefficient
  vector.
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


class _StudentSOVSBinding:
  """The student's own sintering law: the v2 stateful binding protocol."""

  def __call__(self, *parameters: float) -> np.ndarray:
    return skorohod_olevsky_calibration(*parameters)

  def descriptor_metadata(self) -> dict[str, object]:
    return skorohod_olevsky_metadata()

  def kernel(
    self,
    strains: np.ndarray,
    accepted_rows: np.ndarray,
    calibration: np.ndarray,
    signals: tuple[StatefulContinuumSignalInput, ...],
  ) -> StatefulContinuumKernelResult:
    return skorohod_olevsky_kernel(strains, accepted_rows, calibration, signals)

  def initial_state(
    self,
    parameters: tuple[float, ...],
    layout: OperatorStateLayout,
  ) -> np.ndarray:
    return skorohod_olevsky_initial_state(parameters, layout)


def _spoofed_skorohod_olevsky_descriptor(
  metadata: dict[str, object],
) -> RegistryDescriptor:
  return RegistryDescriptor(
    kind=SOVS_MATERIAL_KEY[0],
    name=SOVS_MATERIAL_KEY[1],
    version="1",
    implementation_id="spoofed-for-diagnostics",
    metadata=metadata,
    binding=SKOROHOD_OLEVSKY_BINDING,
  )


# --- material parameter slice ---------------------------------------------------


def test_skorohod_olevsky_authors_the_qualified_parameter_schema() -> None:
  material = authoring.skorohod_olevsky(
    1.0e10, 1.0, 1600.0, 0.6, 1.0e6, 8.314, 2.0, 1.0
  )
  assert material.id == "material"
  assert material.model == "skorohod-olevsky"
  assert material.parameters == (
    MaterialParameterSpec(
      "reference_viscosity",
      1.0e10,
      _source("authoring.skorohod_olevsky:reference_viscosity"),
    ),
    MaterialParameterSpec(
      "activation_energy",
      1.0,
      _source("authoring.skorohod_olevsky:activation_energy"),
    ),
    MaterialParameterSpec(
      "temperature",
      1600.0,
      _source("authoring.skorohod_olevsky:temperature"),
    ),
    MaterialParameterSpec(
      "initial_relative_density",
      0.6,
      _source("authoring.skorohod_olevsky:initial_relative_density"),
    ),
    MaterialParameterSpec(
      "sintering_stress",
      1.0e6,
      _source("authoring.skorohod_olevsky:sintering_stress"),
    ),
    MaterialParameterSpec(
      "gas_constant",
      8.314,
      _source("authoring.skorohod_olevsky:gas_constant"),
    ),
    MaterialParameterSpec(
      "viscosity_exponent_volumetric",
      2.0,
      _source("authoring.skorohod_olevsky:viscosity_exponent_volumetric"),
    ),
    MaterialParameterSpec(
      "viscosity_exponent_shear",
      1.0,
      _source("authoring.skorohod_olevsky:viscosity_exponent_shear"),
    ),
  )
  assert material.source == _source("authoring.skorohod_olevsky")

  integral = authoring.skorohod_olevsky(1, 1, 1600, 1, 1, 8, 2, 1, id="compact")
  assert integral.id == "compact"
  assert integral.parameters[0].value == 1.0
  assert type(integral.parameters[0].value) is float
  assert integral.parameters[7].value == 1.0
  assert type(integral.parameters[7].value) is float


def test_skorohod_olevsky_rejects_non_numeric_and_non_finite_values() -> None:
  with pytest.raises(TypeError, match="exact number"):
    authoring.skorohod_olevsky(True, 1.0, 1600.0, 0.6, 1.0e6, 8.314, 2.0, 1.0)  # type: ignore[arg-type]
  with pytest.raises(TypeError, match="exact number"):
    authoring.skorohod_olevsky(1.0e10, 1.0, 1600.0, 0.6, 1.0e6, 8.314, 2.0, "1")  # type: ignore[arg-type]
  with pytest.raises(ValueError, match="finite"):
    authoring.skorohod_olevsky(
      1.0e10, float("nan"), 1600.0, 0.6, 1.0e6, 8.314, 2.0, 1.0
    )
  with pytest.raises(ValueError, match="finite"):
    authoring.skorohod_olevsky(
      1.0e10, 1.0, 1600.0, 0.6, 1.0e6, 8.314, 2.0, float("inf")
    )
  with pytest.raises(TypeError, match="material id"):
    authoring.skorohod_olevsky(
      1.0e10,
      1.0,
      1600.0,
      0.6,
      1.0e6,
      8.314,
      2.0,
      1.0,
      id=None,  # type: ignore[arg-type]
    )


def test_small_strain_continuum_accepts_the_sintering_material_slice() -> None:
  mesh = authoring.quad8_patch()
  material = _material()
  model = authoring.small_strain_continuum(mesh, material=material)
  assert model.materials == (material,)
  (region,) = model.regions
  assert region.material_id == "compact"
  assert region.formulation == "small-strain-continuum"
  assert region.quadrature == "gauss-3x3"
  with pytest.raises(ValueError, match="skorohod_olevsky"):
    authoring.small_strain_continuum(
      authoring.quad8_patch(),
      material=authoring.uniaxial_elastic(1.0, 1.0),
    )


# --- registry authors and composition --------------------------------------------


def test_skorohod_olevsky_law_pins_the_convention() -> None:
  law = authoring.skorohod_olevsky_law(
    _StudentSOVSBinding(),
    implementation_id="student-sovs-v1",
  )
  assert law.key == SOVS_MATERIAL_KEY
  assert law.version == "1"
  assert law.implementation_id == "student-sovs-v1"
  assert (
    law.metadata.to_bytes() == CanonicalManifest(skorohod_olevsky_metadata()).to_bytes()
  )


def test_skorohod_olevsky_registry_byte_identical_to_the_reference() -> None:
  for key, descriptor in sovs_reference_registry().items():
    layer_descriptor = authoring.skorohod_olevsky_registry()[key]
    assert layer_descriptor.manifest.to_bytes() == descriptor.manifest.to_bytes()
    assert layer_descriptor.metadata.to_bytes() == descriptor.metadata.to_bytes()


def test_skorohod_olevsky_registry_replacement_must_keep_the_convention_key() -> None:
  foreign = RegistryDescriptor(
    kind="material",
    name="my-own-material-name",
    version="1",
    implementation_id="foreign",
    metadata=skorohod_olevsky_metadata(),
    binding=SKOROHOD_OLEVSKY_BINDING,
  )
  with pytest.raises(ValueError, match="does not match the qualified convention key"):
    authoring.skorohod_olevsky_registry(material=foreign)
  with pytest.raises(TypeError, match="exact RegistryDescriptor"):
    authoring.skorohod_olevsky_registry(
      material=SKOROHOD_OLEVSKY_BINDING,  # type: ignore[arg-type]
    )


def test_compile_defaults_to_the_sintering_registry_for_stateful_models() -> None:
  model = _model()
  default = authoring.compile(model)
  explicit = authoring.compile(model, authoring.skorohod_olevsky_registry())
  assert default.content_fingerprint == explicit.content_fingerprint
  reference = compile_system(model, sovs_reference_registry())
  assert default.content_fingerprint == reference.content_fingerprint
  layout = default.operators[0].header.state_layout
  assert layout.schema == (
    "pyfem-v3-skorohod-olevsky-state-v1|strain:6,strain_visc:6,rho:1,time:1"
  )
  # The family's first parameter-dependent initial state: rho = rho0 in
  # every row, bound at compile through the material slice alone.
  assert layout.initial_rows is not None
  assert np.all(layout.initial_rows.values[:, _RHO_COLUMN] == _RHO0)
  assert np.all(layout.initial_rows.values[:, :_RHO_COLUMN] == 0.0)
  (port,) = default.operators[0].header.signal_ports
  assert port.port_id == "time"
  assert port.signal_id == "time"


# --- field-level metadata-mismatch diagnostics ------------------------------------


def test_skorohod_olevsky_metadata_mismatch_names_the_v2_descriptor_fields() -> None:
  metadata = skorohod_olevsky_metadata()
  metadata["tangent_class"] = "algorithmic-nonsymmetric"
  metadata["state_slots"][0]["width"] = 3  # type: ignore[index]
  metadata["signal_ports"][0]["signal_id"] = "temperature"  # type: ignore[index]
  del metadata["state_schema"]
  metadata["sintering_table"] = [[0.6, 1.0e6]]
  spoofed = _spoofed_skorohod_olevsky_descriptor(metadata)

  with pytest.raises(ModelCompilationError) as captured:
    authoring.skorohod_olevsky_registry(material=spoofed)
  (diagnostic,) = captured.value.diagnostics
  assert diagnostic.code == "incompatible-registry-descriptor"
  assert "('material', 'skorohod-olevsky')" in diagnostic.message
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
  assert "unexpected field 'sintering_table'" in diagnostic.message
  assert diagnostic.source == _source("authoring.skorohod_olevsky_registry:material")


def test_skorohod_olevsky_metadata_mismatch_fires_at_compile_with_field_diff() -> None:
  metadata = skorohod_olevsky_metadata()
  metadata["state_slots"][2]["name"] = "density"  # type: ignore[index]
  registry = authoring.skorohod_olevsky_registry()
  registry[SOVS_MATERIAL_KEY] = _spoofed_skorohod_olevsky_descriptor(metadata)

  with pytest.raises(ModelCompilationError) as captured:
    authoring.compile(_model(), registry)
  (diagnostic,) = captured.value.diagnostics
  assert diagnostic.code == "incompatible-registry-descriptor"
  assert (
    "field 'state_slots[2].name': expected 'rho', authored 'density'"
    in diagnostic.message
  )
  assert diagnostic.source == _source("authoring.skorohod_olevsky")


def test_landed_diagnostics_pass_through_unchanged() -> None:
  missing = authoring.skorohod_olevsky_registry()
  del missing[SOVS_MATERIAL_KEY]
  with pytest.raises(ModelCompilationError, match="registry-capture-failed"):
    authoring.compile(_model(), missing)
  with pytest.raises(ModelCompilationError, match="material-binding-failed"):
    authoring.compile(_model(_material(rho0=1.5)))


def test_check_registry_defers_stateful_keys_without_a_qualified_convention() -> None:
  model = _model()
  foreign = replace(
    model,
    materials=(replace(model.materials[0], model="research-law-v0"),),
  )
  authoring.check_registry(foreign, authoring.skorohod_olevsky_registry())
  with pytest.raises(ModelCompilationError, match="registry-capture-failed"):
    authoring.compile(foreign, authoring.skorohod_olevsky_registry())


# --- sensitivity surface: no derivative kernel reads 'constant' -------------------


def test_skorohod_olevsky_parameters_read_constant_in_the_sensitivity_surface() -> None:
  """The wave-13 law ships no derivative kernel: the M58 convention resolver.

  Every qualified parameter of the landed skorohod-olevsky convention is
  baked into the compiled calibration, so a sensitivity request fails
  pre-substep with the parameterized-vs-constant diff, pinned literally.
  """
  session = _session(authoring.compile(_model()))
  with pytest.raises(DriverPreparationError) as captured:
    session.run(
      {"time": 0.0, "load": 0.0},
      {"time": 0.01, "load": 0.0},
      sensitivities=("sintering_stress",),
    )
  (diagnostic,) = captured.value.diagnostics
  assert diagnostic.code == "unknown-sensitivity-parameter"
  assert (
    "field 'sensitivities.sintering_stress': expected {'parameter': "
    "'sintering_stress'}, authored 'constant'" in diagnostic.message
  )
  assert (
    "declared differentiable parameters: (); declared constant parameters: "
    "('reference_viscosity', 'activation_energy', 'temperature', "
    "'initial_relative_density', 'sintering_stress', 'gas_constant', "
    "'viscosity_exponent_volumetric', 'viscosity_exponent_shear')" in diagnostic.message
  )
  assert diagnostic.source == _source("authoring.run:sensitivities")
  assert session.driver.statistics.evaluation_count == 0
  assert session.ordinal == 0


# --- stateful persona: authored, stepped, oracle-checked ---------------------------


def test_persona_authors_and_steps_the_sintering_law_end_to_end() -> None:
  """Persona: skorohod_olevsky() plus the M27 helpers, oracle-checked.

  The signal-consuming law authors exactly like an elastic one — one
  material call, one model call, one compile — then steps the documented
  M68 activated schedule (three free-sintering steps, then the pressure
  leg) through ``nonlinear_static`` with the ``time`` program coordinate
  bound per step. The committed rows equal the batched M68 kernel stepped
  on the committed per-integration-point strain path, bitwise (the driver
  stages the trial rows of the converged iterate); the ``time`` slot
  records each substep's bound time and ``rho`` densifies monotonically
  from ``rho0`` at every integration point. A manual begin/reject trial
  leaves the committed bytes identical.
  """
  model = _model()
  system = authoring.compile(model)
  session = _session(system)

  calibration = skorohod_olevsky_calibration(
    _ETA0, _Q, _T, _RHO0, _SIGMA_SINT, _R, _N_VOL, _N_SHEAR
  )
  oracle_rows = np.zeros((9, _ROW_WIDTH))
  oracle_rows[:, _RHO_COLUMN] = _RHO0
  base = {"time": 0.0, "load": 0.0}
  previous_rho = None
  for step, (time, eps) in enumerate(_SCHEDULE, 1):
    result = session.run(base, {"time": time, "load": eps})
    assert result.status is authoring.DriverStatus.COMPLETED
    base = {"time": time, "load": eps}
    oracle = skorohod_olevsky_kernel(
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
    assert np.all(rows[:, _RHO_COLUMN] >= _RHO0)
    if previous_rho is not None:
      assert np.all(rows[:, _RHO_COLUMN] > previous_rho)
    previous_rho = rows[:, _RHO_COLUMN]
    assert session.ordinal == step
  # Homogeneous state: all nine integration points agree to solver tolerance.
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


def test_time_schedule_changes_the_response_through_the_authoring_path() -> None:
  """A slower schedule densifies more at the same strain path.

  The positive control for the port wiring — SOVS is genuinely
  rate-dependent: schedule-owned time reaches the law through the declared
  ``time`` coordinate, so the slower session commits a materially denser
  state (the documented M68 control, authored end-to-end).
  """
  first = _session(authoring.compile(_model()))
  second = _session(authoring.compile(_model()))
  first_base = {"time": 0.0, "load": 0.0}
  second_base = {"time": 0.0, "load": 0.0}
  for first_time, second_time in ((0.01, 0.10), (0.02, 0.30), (0.03, 1.00)):
    assert (
      first.run(first_base, {"time": first_time, "load": 0.0}).status
      is authoring.DriverStatus.COMPLETED
    )
    assert (
      second.run(second_base, {"time": second_time, "load": 0.0}).status
      is authoring.DriverStatus.COMPLETED
    )
    first_base = {"time": first_time, "load": 0.0}
    second_base = {"time": second_time, "load": 0.0}
  first_rows = first.accepted_state("cells")
  second_rows = second.accepted_state("cells")
  assert np.all(second_rows[:, _RHO_COLUMN] > first_rows[:, _RHO_COLUMN])
  assert first_rows.tobytes() != second_rows.tobytes()


def test_student_law_binding_steps_bitwise_like_the_reference_law() -> None:
  """A binding behind skorohod_olevsky_law steps like the reference.

  The student keeps the qualified convention (one descriptor line, one
  registry line) and supplies only the binding — the 14-float state layout,
  the parameter-dependent rho0 initial rows, and the time port included;
  stepping both systems over the documented schedule commits byte-identical
  state.
  """
  law = authoring.skorohod_olevsky_law(
    _StudentSOVSBinding(),
    implementation_id="student-sovs-v1",
  )
  model = _model()
  student = _session(
    authoring.compile(model, authoring.skorohod_olevsky_registry(material=law))
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
