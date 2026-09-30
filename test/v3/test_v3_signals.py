# SPDX-License-Identifier: MIT

"""G3 signal ports end to end: declaration, compilation, forwarding, oracles.

The mock time-dependent law in this module is the C3 oracle witness: a linear
elastic law whose modulus is scaled by the bound program time,
``sigma = E * (1 + alpha * t) * eps`` with algorithmic tangent
``E * (1 + alpha * t) * I6`` and a one-float state row recording the bound
signal value. Because the response is linear in both strain and time, every
quantity is hand-derived:

- committed coefficients under a prescribed homogeneous strain ramp are the
  exact affine field ``u_x = eps * x, u_y = 0`` regardless of time;
- committed state rows equal the substep's bound time at every integration
  point;
- constraint reactions equal the independently assembled internal force
  ``E * (1 + alpha * t) * integral(B^T B) * u``;
- the signal response derivative is analytic:
  ``d(response)/dt = response * alpha / (1 + alpha * t)``, verified by finite
  difference through the compiled operator;
- the driver's forwarded derivative channels equal the finite difference of
  its own signal forwarding under coordinate perturbation (the identity
  binding's Kronecker delta).

A recording kernel proves the driver forwards the substep's bound point; a
signal-free variant of the same law driven under two different time schedules
yields byte-identical results, proving isolation: without declared ports no
schedule value can reach the law.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from pyfem.v3.compile.continuum import q8_reference_registry
from pyfem.v3.compile.contracts import (
  STATEFUL_MATERIAL_DESCRIPTOR_SCHEMA,
  MaterialSignalPort,
  StatefulContinuumKernelResult,
  StatefulContinuumSignalInput,
  resolve_material_signal_ports,
  validate_stateful_material_metadata,
)
from pyfem.v3.compile.diagnostics import ModelCompilationError
from pyfem.v3.compile.system import compile_system
from pyfem.v3.constraints import CompiledConstraintMap, compile_constraint_map
from pyfem.v3.driver import (
  DriverPreparationError,
  DriverStatus,
  NonlinearStaticDriver,
  evaluate_signals,
)
from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.operator import (
  ChannelRequest,
  CompiledOperator,
  EvaluationStatus,
  OperatorEvaluationInput,
  ProgramSignalInput,
  SignalDerivativeInput,
)
from pyfem.v3.model.provenance import CanonicalManifest
from pyfem.v3.model.registry import RegistryDescriptor, RegistryKey
from pyfem.v3.model.system import CompiledSystem
from pyfem.v3.spec import (
  CellBlockSpec,
  CellRef,
  CellSpec,
  FieldSpec,
  MaterialParameterSpec,
  MaterialSpec,
  MeshSpec,
  ModelSpec,
  NodeSpec,
  RegionSpec,
  SourceContext,
)
from pyfem.v3.spec.program import (
  AffineCoefficientSpec,
  AffineValueSpec,
  DofRef,
  PrescribedDofSpec,
  ProgramCoordinateSpec,
  ProgramCoordinateValue,
  ProgramPoint,
)

_UNIT_COORDINATES = (
  (0.0, 0.0),
  (0.5, 0.0),
  (1.0, 0.0),
  (1.0, 0.5),
  (1.0, 1.0),
  (0.5, 1.0),
  (0.0, 1.0),
  (0.0, 0.5),
)
_YOUNGS_MODULUS = 1000.0
_TIME_RATE = 0.5
_TIME_MODEL = "mock-time-scaled-elastic"
_PLAIN_MODEL = "mock-port-free-elastic"
_TIME_STATE_SCHEMA = "pyfem-v3-mock-time-scaled-state-v1"
_PLAIN_STATE_SCHEMA = "pyfem-v3-mock-port-free-state-v1"

type SignalLog = list[tuple[float, tuple[tuple[str, float], ...]]]


def _source(label: str) -> SourceContext:
  return SourceContext(source=label)


def _time_metadata() -> dict[str, object]:
  return {
    "schema": STATEFUL_MATERIAL_DESCRIPTOR_SCHEMA,
    "law": _TIME_MODEL,
    "stress_state": "plane-strain",
    "parameter_names": ["youngs_modulus", "time_rate"],
    "parameter_dtype": "float64",
    "stress_voigt_order": ["xx", "yy", "xy"],
    "strain_shear_convention": "engineering",
    "internal_voigt_order": ["xx", "yy", "zz", "yz", "zx", "xy"],
    "tangent_class": "algorithmic-symmetric",
    "state_schema": _TIME_STATE_SCHEMA,
    "state_slots": [
      {
        "name": "signal_time",
        "width": 1,
        "dtype": "float64",
        "lifetime": "accepted-trial",
      },
    ],
    "signal_ports": [
      {
        "port_id": "time",
        "signal_id": "time",
        "derivative_coordinate_ids": ["time", "eps"],
      },
    ],
  }


def _plain_metadata() -> dict[str, object]:
  metadata = _time_metadata()
  metadata["law"] = _PLAIN_MODEL
  metadata["parameter_names"] = ["youngs_modulus"]
  metadata["state_schema"] = _PLAIN_STATE_SCHEMA
  del metadata["signal_ports"]
  return metadata


def _time_kernel(
  strains: np.ndarray,
  accepted_rows: np.ndarray,
  calibration: np.ndarray,
  signals: tuple[StatefulContinuumSignalInput, ...],
  log: SignalLog,
) -> StatefulContinuumKernelResult:
  """Time-scaled linear law; records every bound signal it receives."""
  if len(signals) != 1 or signals[0].port_id != "time":
    msg = "time-scaled kernel requires exactly the declared time port"
    raise TypeError(msg)
  time = float(signals[0].values[0])
  log.append(
    (
      time,
      tuple(
        (derivative.coordinate_id, float(derivative.values[0]))
        for derivative in signals[0].derivatives
      ),
    )
  )
  modulus = calibration[0] * (1.0 + calibration[1] * time)
  trial_rows = np.array(accepted_rows, dtype=np.float64, copy=True)
  trial_rows[:, 0] = time
  return StatefulContinuumKernelResult(
    stresses=modulus * strains,
    tangents=np.broadcast_to(modulus * np.eye(6), (len(strains), 6, 6)).copy(),
    trial_rows=trial_rows,
    status=EvaluationStatus.OK,
  )


def _port_free_kernel(
  strains: np.ndarray,
  accepted_rows: np.ndarray,
  calibration: np.ndarray,
) -> StatefulContinuumKernelResult:
  """Signal-free linear echo: the three-argument call, state rows untouched."""
  modulus = float(calibration[0])
  return StatefulContinuumKernelResult(
    stresses=modulus * strains,
    tangents=np.broadcast_to(modulus * np.eye(6), (len(strains), 6, 6)).copy(),
    trial_rows=np.array(accepted_rows, dtype=np.float64, copy=True),
    status=EvaluationStatus.OK,
  )


@dataclass(frozen=True, slots=True, eq=False)
class _TimeScaledBinding:
  """Time-ported mock binding; ``log`` records every kernel signal receipt."""

  log: SignalLog

  def __call__(self, *parameters: float) -> np.ndarray:
    return np.array(parameters, dtype=np.float64)

  def descriptor_metadata(self) -> dict[str, object]:
    return _time_metadata()

  def kernel(
    self,
    strains: np.ndarray,
    accepted_rows: np.ndarray,
    calibration: np.ndarray,
    signals: tuple[StatefulContinuumSignalInput, ...],
  ) -> StatefulContinuumKernelResult:
    return _time_kernel(strains, accepted_rows, calibration, signals, self.log)


@dataclass(frozen=True, slots=True, eq=False)
class _ThreeArgTimeBinding:
  """Ported metadata with a three-argument kernel: must fail the virgin probe."""

  def __call__(self, *parameters: float) -> np.ndarray:
    return np.array(parameters, dtype=np.float64)

  def descriptor_metadata(self) -> dict[str, object]:
    return _time_metadata()

  def kernel(
    self,
    strains: np.ndarray,
    accepted_rows: np.ndarray,
    calibration: np.ndarray,
  ) -> StatefulContinuumKernelResult:
    return _port_free_kernel(strains, accepted_rows, calibration)


@dataclass(frozen=True, slots=True, eq=False)
class _CustomMetadataBinding:
  """Time kernel re-declaring caller-supplied metadata (byte-compare clean)."""

  metadata: dict[str, object]
  log: SignalLog

  def __call__(self, *parameters: float) -> np.ndarray:
    return np.array(parameters, dtype=np.float64)

  def descriptor_metadata(self) -> dict[str, object]:
    return self.metadata

  def kernel(
    self,
    strains: np.ndarray,
    accepted_rows: np.ndarray,
    calibration: np.ndarray,
    signals: tuple[StatefulContinuumSignalInput, ...],
  ) -> StatefulContinuumKernelResult:
    return _time_kernel(strains, accepted_rows, calibration, signals, self.log)


@dataclass(frozen=True, slots=True, eq=False)
class _PortFreeBinding:
  """Signal-free variant of the mock law (no declared ports)."""

  def __call__(self, *parameters: float) -> np.ndarray:
    return np.array(parameters, dtype=np.float64)

  def descriptor_metadata(self) -> dict[str, object]:
    return _plain_metadata()

  def kernel(
    self,
    strains: np.ndarray,
    accepted_rows: np.ndarray,
    calibration: np.ndarray,
  ) -> StatefulContinuumKernelResult:
    return _port_free_kernel(strains, accepted_rows, calibration)


def _registry(
  *,
  binding: object | None = None,
  metadata: dict[str, object] | None = None,
  with_ports: bool = True,
) -> dict[RegistryKey, RegistryDescriptor]:
  registry = dict(q8_reference_registry())
  if with_ports:
    descriptor = RegistryDescriptor(
      kind="material",
      name=_TIME_MODEL,
      version="1",
      implementation_id="pyfem-v3-test-time-scaled-v1",
      metadata=_time_metadata() if metadata is None else metadata,
      binding=_TimeScaledBinding(log=[]) if binding is None else binding,
    )
  else:
    descriptor = RegistryDescriptor(
      kind="material",
      name=_PLAIN_MODEL,
      version="1",
      implementation_id="pyfem-v3-test-port-free-v1",
      metadata=_plain_metadata() if metadata is None else metadata,
      binding=_PortFreeBinding() if binding is None else binding,
    )
  registry[descriptor.key] = descriptor
  return registry


def _model(*, with_ports: bool = True) -> ModelSpec:
  nodes = tuple(
    NodeSpec(
      id=index + 1,
      coordinates=point,
      source=_source(f"node-{index + 1}"),
    )
    for index, point in enumerate(_UNIT_COORDINATES)
  )
  cell = CellSpec(
    id="cell-1",
    node_ids=tuple(node.id for node in nodes),
    source=_source("cell"),
  )
  parameters: tuple[tuple[str, float], ...] = (("youngs_modulus", _YOUNGS_MODULUS),)
  if with_ports:
    parameters += (("time_rate", _TIME_RATE),)
  return ModelSpec(
    mesh=MeshSpec(
      nodes=nodes,
      cell_blocks=(
        CellBlockSpec(
          id="cells",
          reference_topology="quadrilateral",
          topological_dimension=2,
          embedding_dimension=2,
          geometry_interpolation="serendipity-quad8",
          cells=(cell,),
          source=_source("block"),
        ),
      ),
      source=_source("mesh"),
    ),
    fields=(
      FieldSpec(
        id="displacement",
        components=("x", "y"),
        location="node",
        source=_source("field"),
      ),
    ),
    materials=(
      MaterialSpec(
        id="material",
        model=_TIME_MODEL if with_ports else _PLAIN_MODEL,
        parameters=tuple(
          MaterialParameterSpec(name, value, _source(f"parameter:{name}"))
          for name, value in parameters
        ),
        source=_source("material"),
      ),
    ),
    regions=(
      RegionSpec(
        id="domain",
        cell_refs=(CellRef("cells", "cell-1"),),
        field_ids=("displacement",),
        material_id="material",
        formulation="small-strain-continuum",
        quadrature="gauss-3x3",
        source=_source("region"),
      ),
    ),
    source=_source("model"),
  )


def _compiled(*, with_ports: bool = True) -> CompiledSystem:
  return compile_system(_model(with_ports=with_ports), _registry(with_ports=with_ports))


def _strain_ramp_driver(
  system: CompiledSystem | None = None, *, with_ports: bool = True
) -> NonlinearStaticDriver:
  """One Q8 element under a prescribed affine eps_xx ramp, one free DOF.

  All nodal displacements follow the homogeneous strain field exactly
  (``u_x = eps * x``, ``u_y = 0``) except node 6's x component, which stays
  free so every Newton iteration assembles and factorizes the tangent. The
  program declares the schedule-owned ``time`` coordinate alongside ``eps``.
  """
  if system is None:
    system = _compiled(with_ports=with_ports)
  constraints = []
  for index, node in enumerate(_UNIT_COORDINATES):
    node_id = index + 1
    if node_id != 6:
      constraints.append(
        PrescribedDofSpec(
          id=f"ux-{node_id}",
          target=DofRef(
            node_id=node_id,
            field_id="displacement",
            component="x",
          ),
          value=AffineValueSpec(
            coefficients=(AffineCoefficientSpec("eps", node[0]),),
          ),
          source=_source(f"ux-{node_id}"),
        )
      )
    constraints.append(
      PrescribedDofSpec(
        id=f"uy-{node_id}",
        target=DofRef(node_id=node_id, field_id="displacement", component="y"),
        value=AffineValueSpec(constant=0.0),
        source=_source(f"uy-{node_id}"),
      )
    )
  coordinate_map = compile_constraint_map(
    system,
    constraints=tuple(constraints),
    coordinates=(
      ProgramCoordinateSpec(name="time", kind="time"),
      ProgramCoordinateSpec(name="eps", kind="load"),
    ),
  )
  return NonlinearStaticDriver(system, coordinate_map, ())


def _point(time: float, eps: float) -> ProgramPoint:
  return ProgramPoint(
    (
      ProgramCoordinateValue("time", time),
      ProgramCoordinateValue("eps", eps),
    )
  )


def _time_signal(
  time: float,
  derivatives: tuple[tuple[str, float], ...] = (("time", 1.0), ("eps", 0.0)),
) -> ProgramSignalInput:
  return ProgramSignalInput(
    port_id="time",
    values=FinalizedArray(np.array([time], dtype=np.float64), dtype=np.float64),
    derivatives=tuple(
      SignalDerivativeInput(
        coordinate,
        FinalizedArray(np.array([value], dtype=np.float64), dtype=np.float64),
      )
      for coordinate, value in derivatives
    ),
  )


def _evaluation_input(
  operator: CompiledOperator,
  *,
  eps: float,
  signals: tuple[ProgramSignalInput, ...],
) -> OperatorEvaluationInput:
  values = np.zeros(16, dtype=np.float64)
  values[0::2] = eps * np.array(
    [node[0] for node in _UNIT_COORDINATES], dtype=np.float64
  )
  return OperatorEvaluationInput(
    port_values=(FinalizedArray(values.reshape(1, 16), dtype=np.float64),),
    accepted_state=FinalizedArray(
      np.zeros(operator.header.state_layout.row_shape), dtype=np.float64
    ),
    signals=signals,
    request=ChannelRequest(("internal-force",), ("material-tangent",)),
  )


def _decode_node(node: object) -> object:
  if type(node) is not list or not node:
    return node
  tag = node[0]
  if tag == "mapping":
    return {pair[0]: _decode_node(pair[1]) for pair in node[1]}
  if tag == "sequence":
    return [_decode_node(item) for item in node[1]]
  if tag == "str":
    return node[1]
  return node


def _manifest_content(manifest: CanonicalManifest) -> dict[str, object]:
  payload = manifest.to_bytes().split(b"\n", 1)[1]
  node = json.loads(payload)
  if type(node) is not list or not node or node[0] != "mapping":
    msg = "compiled manifests must decode to a canonical mapping"
    raise TypeError(msg)
  return {pair[0]: pair[1] for pair in node[1]}


def test_signal_port_declaration_resolution_is_fail_closed() -> None:
  ports = resolve_material_signal_ports(_time_metadata()["signal_ports"])
  assert ports == (
    MaterialSignalPort(
      port_id="time",
      signal_id="time",
      derivative_coordinate_ids=("time", "eps"),
    ),
  )
  bare = resolve_material_signal_ports([{"port_id": "load", "signal_id": "load"}])
  assert bare[0].derivative_coordinate_ids == ()
  with pytest.raises(TypeError, match="exact list"):
    resolve_material_signal_ports({"port_id": "time"})
  with pytest.raises(ValueError, match="at least one port"):
    resolve_material_signal_ports([])
  with pytest.raises(TypeError, match="exact dictionaries"):
    resolve_material_signal_ports(["time"])
  with pytest.raises(ValueError, match="exactly port_id, signal_id"):
    resolve_material_signal_ports([{"port_id": "time"}])
  with pytest.raises(ValueError, match="exactly port_id, signal_id"):
    resolve_material_signal_ports(
      [{"port_id": "time", "signal_id": "time", "extra": 1}]
    )
  with pytest.raises(ValueError, match="unique"):
    resolve_material_signal_ports(
      [
        {"port_id": "time", "signal_id": "time"},
        {"port_id": "time", "signal_id": "eps"},
      ]
    )
  with pytest.raises(TypeError, match="non-empty exact string"):
    resolve_material_signal_ports([{"port_id": "", "signal_id": "time"}])
  with pytest.raises(TypeError, match="exact list"):
    resolve_material_signal_ports(
      [{"port_id": "time", "signal_id": "time", "derivative_coordinate_ids": "t"}]
    )
  with pytest.raises(ValueError, match="unique"):
    resolve_material_signal_ports(
      [
        {
          "port_id": "time",
          "signal_id": "time",
          "derivative_coordinate_ids": ["eps", "eps"],
        }
      ]
    )


def test_metadata_validation_accepts_only_the_optional_signal_field() -> None:
  metadata = _time_metadata()
  assert validate_stateful_material_metadata(metadata) is metadata
  plain = _plain_metadata()
  assert validate_stateful_material_metadata(plain) is plain
  with pytest.raises(ValueError, match="frozen v2 field set"):
    validate_stateful_material_metadata({**metadata, "extra": 1})
  missing = {key: value for key, value in metadata.items() if key != "state_slots"}
  with pytest.raises(ValueError, match="frozen v2 field set"):
    validate_stateful_material_metadata(missing)
  with pytest.raises(ValueError, match="unique"):
    validate_stateful_material_metadata(
      {
        **metadata,
        "signal_ports": [
          {"port_id": "time", "signal_id": "time"},
          {"port_id": "time", "signal_id": "eps"},
        ],
      }
    )


def test_compiler_emits_signal_ports_and_versions_content_identity() -> None:
  ported = _compiled().operators[0]
  header = ported.header
  assert len(header.signal_ports) == 1
  port = header.signal_ports[0]
  assert port.port_id == "time"
  assert port.signal_id == "time"
  assert port.derivative_coordinate_ids == ("time", "eps")
  assert header.state_layout.row_shape == (9, 1)
  assert [slot.name for slot in header.state_layout.slots] == ["signal_time"]
  ported_manifest = _manifest_content(ported.content_manifest)
  assert _decode_node(ported_manifest["signal_ports"]) == [
    {
      "port_id": "time",
      "signal_id": "time",
      "derivative_coordinate_ids": ["time", "eps"],
    }
  ]
  plain = _compiled(with_ports=False).operators[0]
  assert plain.header.signal_ports == ()
  assert "signal_ports" not in _manifest_content(plain.content_manifest)
  assert ported.content_manifest.to_bytes() != plain.content_manifest.to_bytes()
  ported_plan = _strain_ramp_driver().plan
  assert "signals" in _manifest_content(ported_plan.provenance.manifest)
  plain_plan = _strain_ramp_driver(with_ports=False).plan
  assert "signals" not in _manifest_content(plain_plan.provenance.manifest)
  assert ported_plan.content_fingerprint != plain_plan.content_fingerprint


def test_ported_descriptor_requires_the_four_argument_kernel() -> None:
  with pytest.raises(ModelCompilationError, match="kernel-probe-failed"):
    compile_system(
      _model(),
      _registry(binding=_ThreeArgTimeBinding()),
    )


def test_compiler_rejects_malformed_signal_port_descriptors() -> None:
  metadata = _time_metadata()
  metadata["signal_ports"] = [
    {"port_id": "time", "signal_id": "time"},
    {"port_id": "time", "signal_id": "eps"},
  ]
  with pytest.raises(ModelCompilationError, match="malformed-registry-descriptor"):
    compile_system(
      _model(),
      _registry(
        binding=_CustomMetadataBinding(metadata=metadata, log=[]),
        metadata=metadata,
      ),
    )


def test_plan_rejects_signal_ports_on_undeclared_coordinates() -> None:
  def load_only_map(system: CompiledSystem) -> CompiledConstraintMap:
    return compile_constraint_map(
      system,
      constraints=(
        PrescribedDofSpec(
          id="fix",
          target=DofRef(node_id=1, field_id="displacement", component="x"),
          value=AffineValueSpec(constant=0.0),
          source=_source("fix"),
        ),
      ),
      coordinates=(ProgramCoordinateSpec(name="eps", kind="load"),),
    )

  system = _compiled()
  with pytest.raises(DriverPreparationError) as signal_error:
    NonlinearStaticDriver(system, load_only_map(system), ())
  assert signal_error.value.diagnostics[0].code == "unknown-signal-coordinate"

  metadata = _time_metadata()
  metadata["signal_ports"] = [
    {
      "port_id": "time",
      "signal_id": "time",
      "derivative_coordinate_ids": ["shear"],
    }
  ]
  derivative_system = compile_system(
    _model(),
    _registry(
      binding=_CustomMetadataBinding(metadata=metadata, log=[]),
      metadata=metadata,
    ),
  )
  time_eps_map = compile_constraint_map(
    derivative_system,
    constraints=(
      PrescribedDofSpec(
        id="fix",
        target=DofRef(node_id=1, field_id="displacement", component="x"),
        value=AffineValueSpec(constant=0.0),
        source=_source("fix"),
      ),
    ),
    coordinates=(
      ProgramCoordinateSpec(name="time", kind="time"),
      ProgramCoordinateSpec(name="eps", kind="load"),
    ),
  )
  with pytest.raises(DriverPreparationError) as derivative_error:
    NonlinearStaticDriver(derivative_system, time_eps_map, ())
  assert (
    derivative_error.value.diagnostics[0].code == "unknown-signal-derivative-coordinate"
  )


def test_evaluate_signals_forwards_bound_point_and_fd_checked_derivatives() -> None:
  driver = _strain_ramp_driver()
  plan = driver.plan
  assert len(plan.signal_slices) == 1
  assert len(plan.signal_slices[0]) == 1
  point = _point(0.3, 0.002)
  forwarded = evaluate_signals(plan, point)
  assert len(forwarded) == 1 and len(forwarded[0]) == 1
  signal = forwarded[0][0]
  assert signal.port_id == "time"
  np.testing.assert_array_equal(signal.values.values, [0.3])
  assert tuple(item.coordinate_id for item in signal.derivatives) == ("time", "eps")
  delta = 1.0e-7
  for derivative in signal.derivatives:
    perturbed = ProgramPoint(
      tuple(
        ProgramCoordinateValue(
          item.name,
          item.value + delta if item.name == derivative.coordinate_id else item.value,
        )
        for item in point.values
      )
    )
    moved = evaluate_signals(plan, perturbed)[0][0].values.values[0]
    finite_difference = (moved - signal.values.values[0]) / delta
    np.testing.assert_allclose(
      derivative.values.values,
      [finite_difference],
      rtol=0.0,
      atol=1.0e-6,
    )
  np.testing.assert_array_equal(
    [item.values.values[0] for item in signal.derivatives], [1.0, 0.0]
  )


def test_evaluate_signals_never_touches_points_without_ports() -> None:
  plain_plan = _strain_ramp_driver(with_ports=False).plan
  assert plain_plan.signal_slices == ((),)
  forwarded = evaluate_signals(plain_plan, ProgramPoint(()))
  assert forwarded == ((),)


def test_ported_operator_binds_signals_exactly() -> None:
  operator = _compiled().operators[0]
  result = operator.evaluate(
    _evaluation_input(operator, eps=0.002, signals=(_time_signal(0.25),))
  )
  assert result.trial_state.values.shape == (9, 1)
  np.testing.assert_array_equal(result.trial_state.values, 0.25)
  with pytest.raises(ValueError, match="missing a declared program signal port"):
    operator.evaluate(_evaluation_input(operator, eps=0.002, signals=()))
  undeclared = ProgramSignalInput(
    port_id="temperature",
    values=FinalizedArray(np.array([1.0], dtype=np.float64), dtype=np.float64),
    derivatives=(),
  )
  with pytest.raises(ValueError, match="undeclared program signal port"):
    operator.evaluate(
      _evaluation_input(operator, eps=0.002, signals=(_time_signal(0.25), undeclared))
    )
  with pytest.raises(ValueError, match="duplicate program signal port"):
    operator.evaluate(
      _evaluation_input(
        operator, eps=0.002, signals=(_time_signal(0.25), _time_signal(0.5))
      )
    )
  with pytest.raises(ValueError, match="derivatives must match"):
    operator.evaluate(
      _evaluation_input(
        operator,
        eps=0.002,
        signals=(_time_signal(0.25, (("eps", 0.0), ("time", 1.0))),),
      )
    )
  with pytest.raises(ValueError, match="derivatives must match"):
    operator.evaluate(
      _evaluation_input(
        operator, eps=0.002, signals=(_time_signal(0.25, (("time", 1.0),)),)
      )
    )
  malformed = ProgramSignalInput(
    port_id="time",
    values=FinalizedArray(np.array([0.1, 0.2]), dtype=np.float64),
    derivatives=_time_signal(0.25).derivatives,
  )
  with pytest.raises(TypeError, match="float64 scalar"):
    operator.evaluate(_evaluation_input(operator, eps=0.002, signals=(malformed,)))
  nonfinite = ProgramSignalInput(
    port_id="time",
    values=FinalizedArray(np.array([np.nan]), dtype=np.float64),
    derivatives=_time_signal(0.25).derivatives,
  )
  with pytest.raises(TypeError, match="float64 scalar"):
    operator.evaluate(_evaluation_input(operator, eps=0.002, signals=(nonfinite,)))


def test_port_free_stateful_operator_rejects_every_signal() -> None:
  operator = _compiled(with_ports=False).operators[0]
  with pytest.raises(ValueError, match="does not accept program signal inputs"):
    operator.evaluate(
      _evaluation_input(operator, eps=0.002, signals=(_time_signal(0.25),))
    )


def test_operator_signal_response_matches_analytic_derivative_by_fd() -> None:
  operator = _compiled().operators[0]
  eps = 0.003
  time = 0.4
  # The law is linear in time, so the finite difference is analytic; the step
  # is chosen large enough that float rounding of the response is negligible.
  step = 1.0e-3
  base = operator.evaluate(
    _evaluation_input(operator, eps=eps, signals=(_time_signal(time),))
  )
  moved = operator.evaluate(
    _evaluation_input(operator, eps=eps, signals=(_time_signal(time + step),))
  )
  np.testing.assert_array_equal(base.trial_state.values, time)
  np.testing.assert_array_equal(moved.trial_state.values, time + step)
  # d(response)/dt = response * alpha / (1 + alpha*t), hand-derived from the law.
  analytic_rate = _TIME_RATE / (1.0 + _TIME_RATE * time)
  residual = base.residual_values[0].values
  residual_moved = moved.residual_values[0].values
  np.testing.assert_allclose(
    (residual_moved - residual) / step,
    residual * analytic_rate,
    rtol=1.0e-6,
    atol=1.0e-9,
  )
  tangent = base.jacobian_values[0].values
  tangent_moved = moved.jacobian_values[0].values
  np.testing.assert_allclose(
    (tangent_moved - tangent) / step,
    tangent * analytic_rate,
    rtol=1.0e-6,
    atol=1.0e-6,
  )


def _expected_internal_force(driver: NonlinearStaticDriver, time: float) -> np.ndarray:
  """Independently assembled internal force of the committed state.

  Recomputes the element force from the payload geometry with the law's
  hand-derived modulus ``E * (1 + alpha * t)``, mirroring the evaluation's
  arithmetic without touching the driver or kernel.
  """
  operator = driver.owner.system.operators[0]
  payload = operator.payload
  scales = payload.geometry_scales.values
  b_matrix = payload.normalized_strain_displacement.values / scales[:, None, None, None]
  weights = (
    payload.normalized_integration_weights.values * scales[:, None] * scales[:, None]
  )
  gather = operator.header.ports[0].coefficient_map.values
  committed = driver.owner.accepted_physical().values
  strain3 = np.einsum("epai,ei->epa", b_matrix, committed[gather], optimize=True)
  stress3 = _YOUNGS_MODULUS * (1.0 + _TIME_RATE * time) * strain3
  element_force = np.einsum(
    "ep,epa,epai->ei", weights, stress3, b_matrix, optimize=True
  )
  full = np.zeros(driver.owner.system.coefficient_count, dtype=np.float64)
  np.add.at(full, gather.reshape(-1), element_force.reshape(-1))
  return full


def test_time_law_driver_matches_hand_derived_schedule_oracle() -> None:
  binding = _TimeScaledBinding(log=[])
  system = compile_system(_model(), _registry(binding=binding))
  driver = _strain_ramp_driver(system)
  # The compile-time virgin probe legitimately evaluated the origin signal;
  # the driver schedule log starts from a clean slate.
  assert binding.log == [(0.0, (("time", 1.0), ("eps", 0.0)))]
  binding.log.clear()
  layout = driver.owner.system.operators[0].header.state_layout
  block_id = layout.block_id
  schedule = ((0.25, 0.001), (0.5, 0.002), (0.75, 0.003), (1.0, 0.004))
  base = _point(0.0, 0.0)
  for step, (time, eps) in enumerate(schedule, 1):
    result = driver.run(base_point=base, target_points=(_point(time, eps),))
    assert result.status is DriverStatus.COMPLETED
    assert result.statistics.rejected_substep_count == 0
    record = result.records[-1]
    assert record.observation is not None
    committed = driver.owner.accepted_physical().values
    expected = np.zeros(16, dtype=np.float64)
    expected[0::2] = eps * np.array(
      [node[0] for node in _UNIT_COORDINATES], dtype=np.float64
    )
    np.testing.assert_allclose(committed, expected, rtol=1.0e-8, atol=1.0e-11)
    rows = driver.owner.accepted_state(block_id).values
    assert rows.shape == (9, 1)
    np.testing.assert_array_equal(rows, time)
    reactions = record.observation.reactions.values
    constrained_dofs = np.flatnonzero(reactions)
    expected_force = _expected_internal_force(driver, time)
    np.testing.assert_allclose(
      reactions[constrained_dofs],
      expected_force[constrained_dofs],
      rtol=1.0e-9,
      atol=1.0e-9,
    )
    base = _point(time, eps)
    assert result.final_generation.ordinal == step
  statistics = result.statistics
  assert statistics.committed_substep_count == 4
  assert statistics.factorization_reuse_count == 0
  logged_times = {entry[0] for entry in binding.log}
  assert logged_times == {time for time, _ in schedule}
  for _, derivatives in binding.log:
    assert derivatives == (("time", 1.0), ("eps", 0.0))


def test_port_free_law_results_are_signal_independent() -> None:
  first = _strain_ramp_driver(with_ports=False)
  second = _strain_ramp_driver(with_ports=False)
  layout = first.owner.system.operators[0].header.state_layout
  block_id = layout.block_id
  eps_path = (0.001, 0.002, 0.004)
  first_base = _point(0.0, 0.0)
  second_base = _point(0.0, 0.0)
  for first_time, second_time, eps in (
    (0.0, 3.0, eps_path[0]),
    (0.0, 7.5, eps_path[1]),
    (0.0, 11.25, eps_path[2]),
  ):
    first_result = first.run(
      base_point=first_base, target_points=(_point(first_time, eps),)
    )
    second_result = second.run(
      base_point=second_base, target_points=(_point(second_time, eps),)
    )
    assert first_result.status is DriverStatus.COMPLETED
    assert second_result.status is DriverStatus.COMPLETED
    assert (
      first.owner.accepted_physical().values.tobytes()
      == second.owner.accepted_physical().values.tobytes()
    )
    assert (
      first.owner.accepted_state(block_id).values.tobytes()
      == second.owner.accepted_state(block_id).values.tobytes()
    )
    first_base = _point(first_time, eps)
    second_base = _point(second_time, eps)
