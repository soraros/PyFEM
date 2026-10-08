# SPDX-License-Identifier: MIT

"""Stateful viscoelasticity authoring: the v2 descriptor behind plain helpers.

Covers the ``prony_viscoelasticity(...)`` parameter slice, the registry
authors (``prony_viscoelasticity_law``/``viscoelasticity_registry``) with
field-level metadata-mismatch diagnostics over the v2 descriptor fields —
the parameterized ``eps_i`` state width and the ``signal_ports`` time port
included — default-registry selection for stateful models, and the stateful
persona: a Prony-series law authored end-to-end and stepped through the M27
transaction helpers, verified bitwise against the M49 kernel oracle on the
documented time schedule. Time reaches the law exclusively through the
declared identity program coordinate — there is no solverStat-style
channel.
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
  VISCOELASTIC_MATERIAL_KEY,
  viscoelasticity_reference_registry,
)
from pyfem.v3.compile.contracts import (
  StatefulContinuumKernelResult,
  StatefulContinuumSignalDerivative,
  StatefulContinuumSignalInput,
)
from pyfem.v3.compile.diagnostics import ModelCompilationError
from pyfem.v3.compile.system import compile_system
from pyfem.v3.materials.prony_viscoelasticity import (
  PRONY_VISCOELASTICITY_BINDING,
  prony_viscoelastic_initial_state,
  prony_viscoelasticity_calibration,
  prony_viscoelasticity_kernel,
  prony_viscoelasticity_metadata,
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

# The documented M49 oracle configuration
# (examples/materials/viscoelasticity/creep_test.pro).
_E = 1000.0
_NU = 0.3
_EINF = 100.0
_TERMS = 3
_T_FIRST = 0.1
_T_LAST = 10.0
_ROW_WIDTH = 6 * _TERMS + 13
_TIME_COLUMN = _ROW_WIDTH - 1
# The documented M49 driver schedule: a constant-time third step exercises
# the legacy ``dtime > 0`` guard branch end-to-end through the driver.
_SCHEDULE = (
  (0.05, 2.0e-4),
  (0.15, 5.0e-4),
  (0.15, 8.0e-4),
  (0.40, 1.1e-3),
  (1.00, 1.6e-3),
)
# The quad8_patch node whose x displacement stays free, so every Newton
# iteration assembles and factorizes the tangent (node 7 sits at x = 0.5).
_FREE_NODE = 7


def _source(label: str) -> SourceContext:
  return SourceContext(source=label)


def _model(
  material: MaterialSpec | None = None, terms: float = float(_TERMS)
) -> ModelSpec:
  selected = (
    material
    if material is not None
    else authoring.prony_viscoelasticity(
      _E, _NU, _EINF, terms, _T_FIRST, _T_LAST, id="polymer"
    )
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

  The M49 parity harness expression (test_v3_viscoelasticity.py): the
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


class _StudentPronyBinding:
  """The student's own Prony law: the v2 stateful binding protocol plainly."""

  def __call__(self, *parameters: float) -> np.ndarray:
    return prony_viscoelasticity_calibration(*parameters)

  def descriptor_metadata(self) -> dict[str, object]:
    return prony_viscoelasticity_metadata()

  def kernel(
    self,
    strains: np.ndarray,
    accepted_rows: np.ndarray,
    calibration: np.ndarray,
    signals: tuple[StatefulContinuumSignalInput, ...],
  ) -> StatefulContinuumKernelResult:
    return prony_viscoelasticity_kernel(strains, accepted_rows, calibration, signals)

  def initial_state(
    self,
    parameters: tuple[float, ...],
    layout: OperatorStateLayout,
  ) -> np.ndarray:
    return prony_viscoelastic_initial_state(parameters, layout)


def _spoofed_viscoelasticity_descriptor(
  metadata: dict[str, object],
) -> RegistryDescriptor:
  return RegistryDescriptor(
    kind=VISCOELASTIC_MATERIAL_KEY[0],
    name=VISCOELASTIC_MATERIAL_KEY[1],
    version="1",
    implementation_id="spoofed-for-diagnostics",
    metadata=metadata,
    binding=PRONY_VISCOELASTICITY_BINDING,
  )


# --- material parameter slice ---------------------------------------------------


def test_prony_viscoelasticity_authors_the_qualified_parameter_schema() -> None:
  material = authoring.prony_viscoelasticity(1000.0, 0.3, 100.0, 3, 0.1, 10.0)
  assert material.id == "material"
  assert material.model == "prony-viscoelasticity"
  assert material.parameters == (
    MaterialParameterSpec(
      "youngs_modulus",
      1000.0,
      _source("authoring.prony_viscoelasticity:youngs_modulus"),
    ),
    MaterialParameterSpec(
      "poisson_ratio",
      0.3,
      _source("authoring.prony_viscoelasticity:poisson_ratio"),
    ),
    MaterialParameterSpec(
      "equilibrium_modulus",
      100.0,
      _source("authoring.prony_viscoelasticity:equilibrium_modulus"),
    ),
    MaterialParameterSpec(
      "prony_term_count",
      3.0,
      _source("authoring.prony_viscoelasticity:prony_term_count"),
    ),
    MaterialParameterSpec(
      "relaxation_time_first",
      0.1,
      _source("authoring.prony_viscoelasticity:relaxation_time_first"),
    ),
    MaterialParameterSpec(
      "relaxation_time_last",
      10.0,
      _source("authoring.prony_viscoelasticity:relaxation_time_last"),
    ),
  )
  assert material.source == _source("authoring.prony_viscoelasticity")

  integral = authoring.prony_viscoelasticity(1000, 0, 100, 3, 1, 10, id="polymer")
  assert integral.id == "polymer"
  assert integral.parameters[0].value == 1000.0
  assert type(integral.parameters[0].value) is float
  assert integral.parameters[3].value == 3.0
  assert type(integral.parameters[3].value) is float


def test_prony_viscoelasticity_rejects_non_numeric_and_non_finite_values() -> None:
  with pytest.raises(TypeError, match="exact number"):
    authoring.prony_viscoelasticity(True, 0.3, 100.0, 3, 0.1, 10.0)  # type: ignore[arg-type]
  with pytest.raises(TypeError, match="exact number"):
    authoring.prony_viscoelasticity(1000.0, 0.3, 100.0, "3", 0.1, 10.0)  # type: ignore[arg-type]
  with pytest.raises(ValueError, match="finite"):
    authoring.prony_viscoelasticity(1000.0, float("nan"), 100.0, 3, 0.1, 10.0)
  with pytest.raises(ValueError, match="finite"):
    authoring.prony_viscoelasticity(1000.0, 0.3, 100.0, 3, 0.1, float("inf"))
  with pytest.raises(TypeError, match="material id"):
    authoring.prony_viscoelasticity(1.0, 0.0, 0.5, 1, 1.0, 2.0, id=None)  # type: ignore[arg-type]


def test_small_strain_continuum_accepts_the_viscoelastic_material_slice() -> None:
  mesh = authoring.quad8_patch()
  material = authoring.prony_viscoelasticity(2.0, 0.0, 1.0, 1, 1.0, 2.0, id="polymer")
  model = authoring.small_strain_continuum(mesh, material=material)
  assert model.materials == (material,)
  (region,) = model.regions
  assert region.material_id == "polymer"
  assert region.formulation == "small-strain-continuum"
  assert region.quadrature == "gauss-3x3"
  with pytest.raises(ValueError, match="prony_viscoelasticity"):
    authoring.small_strain_continuum(
      authoring.quad8_patch(),
      material=authoring.uniaxial_elastic(1.0, 1.0),
    )


# --- registry authors and composition --------------------------------------------


def test_prony_viscoelasticity_law_pins_the_convention() -> None:
  law = authoring.prony_viscoelasticity_law(
    _StudentPronyBinding(),
    implementation_id="student-prony-v1",
  )
  assert law.key == VISCOELASTIC_MATERIAL_KEY
  assert law.version == "1"
  assert law.implementation_id == "student-prony-v1"
  assert (
    law.metadata.to_bytes()
    == CanonicalManifest(prony_viscoelasticity_metadata()).to_bytes()
  )


def test_viscoelasticity_registry_is_byte_identical_to_the_reference_registry() -> None:
  for key, descriptor in viscoelasticity_reference_registry().items():
    layer_descriptor = authoring.viscoelasticity_registry()[key]
    assert layer_descriptor.manifest.to_bytes() == descriptor.manifest.to_bytes()
    assert layer_descriptor.metadata.to_bytes() == descriptor.metadata.to_bytes()


def test_viscoelasticity_registry_replacement_must_keep_the_convention_key() -> None:
  foreign = RegistryDescriptor(
    kind="material",
    name="my-own-material-name",
    version="1",
    implementation_id="foreign",
    metadata=prony_viscoelasticity_metadata(),
    binding=PRONY_VISCOELASTICITY_BINDING,
  )
  with pytest.raises(ValueError, match="does not match the qualified convention key"):
    authoring.viscoelasticity_registry(material=foreign)
  with pytest.raises(TypeError, match="exact RegistryDescriptor"):
    authoring.viscoelasticity_registry(
      material=PRONY_VISCOELASTICITY_BINDING,  # type: ignore[arg-type]
    )


def test_compile_defaults_to_the_viscoelasticity_registry_for_stateful_models() -> None:
  model = _model()
  default = authoring.compile(model)
  explicit = authoring.compile(model, authoring.viscoelasticity_registry())
  assert default.content_fingerprint == explicit.content_fingerprint
  reference = compile_system(model, viscoelasticity_reference_registry())
  assert default.content_fingerprint == reference.content_fingerprint
  layout = default.operators[0].header.state_layout
  assert layout.schema == (
    "pyfem-v3-prony-viscoelastic-state-v1|eps_i:18,sigma:6,epsilon:6,time:1"
  )
  (port,) = default.operators[0].header.signal_ports
  assert port.port_id == "time"
  assert port.signal_id == "time"


def test_parameterized_state_width_resolves_through_authoring() -> None:
  for terms, width in ((1, 19), (2, 25), (3, 31), (5, 43)):
    system = authoring.compile(_model(terms=float(terms)))
    layout = system.operators[0].header.state_layout
    assert layout.entity_count == 9
    assert layout.row_width == width == 6 * terms + 13
    assert layout.schema == (
      f"pyfem-v3-prony-viscoelastic-state-v1|eps_i:{6 * terms},sigma:6,epsilon:6,time:1"
    )


# --- field-level metadata-mismatch diagnostics ------------------------------------


def test_viscoelasticity_metadata_mismatch_names_the_parameterized_fields() -> None:
  metadata = prony_viscoelasticity_metadata()
  metadata["tangent_class"] = "algorithmic-nonsymmetric"
  metadata["state_slots"][0]["width"] = {"parameter": "prony_term_count", "scale": 3}  # type: ignore[index]
  metadata["signal_ports"][0]["signal_id"] = "temperature"  # type: ignore[index]
  del metadata["state_schema"]
  metadata["relaxation_table"] = [[0.1, 300.0], [10.0, 300.0]]
  spoofed = _spoofed_viscoelasticity_descriptor(metadata)

  with pytest.raises(ModelCompilationError) as captured:
    authoring.viscoelasticity_registry(material=spoofed)
  (diagnostic,) = captured.value.diagnostics
  assert diagnostic.code == "incompatible-registry-descriptor"
  assert "('material', 'prony-viscoelasticity')" in diagnostic.message
  assert (
    "field 'tangent_class': expected 'algorithmic-symmetric', authored "
    "'algorithmic-nonsymmetric'" in diagnostic.message
  )
  assert (
    "field 'state_slots[0].width.scale': expected 6, authored 3" in diagnostic.message
  )
  assert (
    "field 'signal_ports[0].signal_id': expected 'time', authored 'temperature'"
    in diagnostic.message
  )
  assert "missing field 'state_schema'" in diagnostic.message
  assert "unexpected field 'relaxation_table'" in diagnostic.message
  assert diagnostic.source == _source("authoring.viscoelasticity_registry:material")


def test_viscoelasticity_metadata_mismatch_fires_at_compile_with_field_diff() -> None:
  metadata = prony_viscoelasticity_metadata()
  metadata["state_slots"][0]["width"] = 18  # type: ignore[index]
  registry = authoring.viscoelasticity_registry()
  registry[VISCOELASTIC_MATERIAL_KEY] = _spoofed_viscoelasticity_descriptor(metadata)

  with pytest.raises(ModelCompilationError) as captured:
    authoring.compile(_model(), registry)
  (diagnostic,) = captured.value.diagnostics
  assert diagnostic.code == "incompatible-registry-descriptor"
  assert (
    "field 'state_slots[0].width': expected "
    "{'parameter': 'prony_term_count', 'scale': 6}, authored 18" in diagnostic.message
  )
  assert diagnostic.source == _source("authoring.prony_viscoelasticity")


def test_landed_diagnostics_pass_through_unchanged() -> None:
  missing = authoring.viscoelasticity_registry()
  del missing[VISCOELASTIC_MATERIAL_KEY]
  with pytest.raises(ModelCompilationError, match="registry-capture-failed"):
    authoring.compile(_model(), missing)
  with pytest.raises(ModelCompilationError, match="material-binding-failed"):
    authoring.compile(
      _model(authoring.prony_viscoelasticity(_E, _NU, _E, _TERMS, _T_FIRST, _T_LAST))
    )


def test_check_registry_defers_stateful_keys_without_a_qualified_convention() -> None:
  model = _model()
  foreign = replace(
    model,
    materials=(replace(model.materials[0], model="research-law-v0"),),
  )
  authoring.check_registry(foreign, authoring.viscoelasticity_registry())
  with pytest.raises(ModelCompilationError, match="registry-capture-failed"):
    authoring.compile(foreign, authoring.viscoelasticity_registry())


# --- stateful persona: authored, stepped, oracle-checked ---------------------------


def test_persona_authors_and_steps_the_signal_consuming_law_end_to_end() -> None:
  """Persona: prony_viscoelasticity() plus the M27 helpers, oracle-checked.

  The signal-consuming law authors exactly like an elastic one — one
  material call, one model call, one compile — then steps the documented
  M49 schedule through ``nonlinear_static`` with the ``time`` program
  coordinate bound per step. The committed rows equal the batched M49
  kernel stepped on the committed per-integration-point strain path,
  bitwise (the driver stages the trial rows of the converged iterate); the
  ``time`` slot records each substep's bound time at every integration
  point — the identity port is the only time channel. A manual begin/reject
  trial leaves the committed bytes identical.
  """
  model = _model()
  system = authoring.compile(model)
  session = _session(system)

  calibration = prony_viscoelasticity_calibration(
    _E, _NU, _EINF, float(_TERMS), _T_FIRST, _T_LAST
  )
  oracle_rows = np.zeros((9, _ROW_WIDTH))
  base = {"time": 0.0, "load": 0.0}
  for step, (time, eps) in enumerate(_SCHEDULE, 1):
    result = session.run(base, {"time": time, "load": eps})
    assert result.status is authoring.DriverStatus.COMPLETED
    base = {"time": time, "load": eps}
    oracle = prony_viscoelasticity_kernel(
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


def test_time_schedule_changes_the_response_through_the_authoring_path() -> None:
  """The same strain path under different time schedules relaxes differently.

  The positive control for the port wiring: schedule-owned time reaches the
  law through the declared ``time`` coordinate, so two sessions over the
  same prescribed strain ramp commit different state and coefficients.
  """
  first = _session(authoring.compile(_model()))
  second = _session(authoring.compile(_model()))
  first_base = {"time": 0.0, "load": 0.0}
  second_base = {"time": 0.0, "load": 0.0}
  for first_time, second_time, eps in (
    (0.05, 0.50, 2.0e-4),
    (0.15, 3.00, 5.0e-4),
    (0.40, 12.0, 1.1e-3),
  ):
    assert (
      first.run(first_base, {"time": first_time, "load": eps}).status
      is authoring.DriverStatus.COMPLETED
    )
    assert (
      second.run(second_base, {"time": second_time, "load": eps}).status
      is authoring.DriverStatus.COMPLETED
    )
    first_base = {"time": first_time, "load": eps}
    second_base = {"time": second_time, "load": eps}
  assert (
    first.accepted_state("cells").tobytes() != second.accepted_state("cells").tobytes()
  )
  assert (
    first.accepted_coefficients().tobytes() != second.accepted_coefficients().tobytes()
  )


def test_student_law_binding_steps_bitwise_like_the_reference_law() -> None:
  """A binding behind prony_viscoelasticity_law steps like the reference.

  The student keeps the qualified convention (one descriptor line, one
  registry line) and supplies only the binding — the parameterized state
  width and the time port included; stepping both systems over the
  documented schedule commits byte-identical state.
  """
  law = authoring.prony_viscoelasticity_law(
    _StudentPronyBinding(),
    implementation_id="student-prony-v1",
  )
  model = _model()
  student = _session(
    authoring.compile(model, authoring.viscoelasticity_registry(material=law))
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
