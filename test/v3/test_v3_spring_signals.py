# SPDX-License-Identifier: MIT

"""Signal forwarding on the point-entity seam (M66 task 1).

The spring operator seam now forwards declared scalar program signals to
signal-consuming kernels (the ``SignalSpringKernel`` four-argument form),
mirroring the landed stateful-continuum idiom: declarations without
``signal_ports`` compile byte-identical operators that reject every signal
input (the pre-M66 behavior, pinned here), while declared ports emit
``SignalPortBinding`` values the driver binds from the program point.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from pyfem.v3.compile.continuum import q8_reference_registry
from pyfem.v3.compile.diagnostics import ModelCompilationError
from pyfem.v3.compile.spring import (
  SpringDeclaration,
  SpringKernelResult,
  SpringOperator,
  SpringSignalInput,
  SpringSignalPort,
  compile_spring_operator,
  compose_system,
  damage_envelope_declaration,
)
from pyfem.v3.compile.system import compile_system
from pyfem.v3.compile.truss import truss_reference_registry
from pyfem.v3.constraints import compile_constraint_map
from pyfem.v3.driver import DriverStatus, NonlinearStaticDriver
from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.operator import (
  ChannelRequest,
  EvaluationStatus,
  OperatorEvaluation,
  OperatorEvaluationInput,
  ProgramSignalInput,
  SignalDerivativeInput,
  evaluation_status,
)
from pyfem.v3.model.provenance import CanonicalManifest
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
  NodalLoadSpec,
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

_LOAD_PORT = SpringSignalPort(port_id="load-factor", signal_id="load")


def _source(label: str) -> SourceContext:
  return SourceContext(source=label)


def _q8_model() -> ModelSpec:
  nodes = tuple(
    NodeSpec(
      id=index + 1,
      coordinates=point,
      source=_source(f"node-source-{index + 1}"),
    )
    for index, point in enumerate(_UNIT_COORDINATES)
  )
  cell = CellSpec(
    id="cell-1",
    node_ids=tuple(node.id for node in nodes),
    source=_source("cell-source"),
  )
  block = CellBlockSpec(
    id="cells",
    reference_topology="quadrilateral",
    topological_dimension=2,
    embedding_dimension=2,
    geometry_interpolation="serendipity-quad8",
    cells=(cell,),
    source=_source("block-source"),
  )
  field = FieldSpec(
    id="displacement",
    components=("x", "y"),
    location="node",
    source=_source("field-source"),
  )
  material = MaterialSpec(
    id="elastic",
    model="plane-stress-linear-elastic",
    parameters=(
      MaterialParameterSpec("youngs_modulus", 1.0, _source("material:E")),
      MaterialParameterSpec("poisson_ratio", 0.0, _source("material:nu")),
    ),
    source=_source("material-source"),
  )
  region = RegionSpec(
    id="domain",
    cell_refs=(CellRef(block.id, cell.id),),
    field_ids=(field.id,),
    material_id=material.id,
    formulation="small-strain-continuum",
    quadrature="gauss-3x3",
    source=_source("region-source"),
  )
  return ModelSpec(
    mesh=MeshSpec(nodes=nodes, cell_blocks=(block,), source=_source("mesh-source")),
    fields=(field,),
    materials=(material,),
    regions=(region,),
    source=_source("model-source"),
  )


def _base_system() -> CompiledSystem:
  return compile_system(_q8_model(), q8_reference_registry())


def modulated_spring_kernel(
  displacements: np.ndarray,
  accepted_rows: np.ndarray,
  parameters: np.ndarray,
  signals: tuple[SpringSignalInput, ...],
) -> SpringKernelResult:
  """Linear grounded spring whose stiffness scales with the bound signal."""
  (stiffness,) = parameters
  lam = float(signals[0].values[0])
  return SpringKernelResult(
    force=(lam * stiffness) * displacements,
    tangent=(lam * stiffness) * np.tile(np.eye(2), (len(displacements), 1, 1)),
    trial_rows=np.array(accepted_rows, copy=True),
    status=EvaluationStatus.OK,
  )


def _ported_declaration(**overrides: object) -> SpringDeclaration:
  fields: dict[str, object] = {
    "block_id": "modulated-springs",
    "space_id": "displacement",
    "spring_ids": ("modulated-1",),
    "node_ids": (3,),
    "state_schema": "modulated-spring-state-v1",
    "state_slots": (),
    "kernel_name": "signal-modulated-spring",
    "kernel_version": "1",
    "implementation_id": "modulated-spring-v1",
    "parameters": (100.0,),
    "kernel": modulated_spring_kernel,
    "source": _source("modulated-spring-source"),
    "signal_ports": (_LOAD_PORT,),
  }
  fields.update(overrides)
  return SpringDeclaration(**fields)  # type: ignore[arg-type]


def _signal_input(
  value: float,
  *,
  port_id: str = "load-factor",
  derivatives: tuple[tuple[str, float], ...] = (),
) -> ProgramSignalInput:
  return ProgramSignalInput(
    port_id=port_id,
    values=FinalizedArray(np.array([value], dtype=np.float64), dtype=np.float64),
    derivatives=tuple(
      SignalDerivativeInput(
        coordinate,
        FinalizedArray(np.array([entry], dtype=np.float64), dtype=np.float64),
      )
      for coordinate, entry in derivatives
    ),
  )


def _evaluate(
  operator: SpringOperator,
  signal_value: float | None,
  displacements: np.ndarray | None = None,
) -> OperatorEvaluation:
  layout = operator.header.state_layout
  batch = (
    np.zeros((layout.entity_count, 2), dtype=np.float64)
    if displacements is None
    else np.asarray(displacements, dtype=np.float64)
  )
  signals = () if signal_value is None else (_signal_input(signal_value),)
  return operator.evaluate(
    OperatorEvaluationInput(
      port_values=(FinalizedArray(batch, dtype=np.float64),),
      accepted_state=FinalizedArray(np.zeros(layout.row_shape), dtype=np.float64),
      signals=signals,
      request=ChannelRequest(
        ("spring-force",),
        ("spring-tangent",),
      ),
    )
  )


def _manifest_content(manifest: CanonicalManifest) -> dict[str, object]:
  payload = manifest.to_bytes().split(b"\n", 1)[1]
  node = json.loads(payload)
  if type(node) is not list or not node or node[0] != "mapping":
    msg = "compiled manifests must decode to a canonical mapping"
    raise TypeError(msg)
  return {pair[0]: pair[1] for pair in node[1]}


def test_signal_less_declaration_compiles_byte_identical_and_rejects_signals() -> None:
  declaration = damage_envelope_declaration(
    block_id="damage-springs",
    space_id="displacement",
    spring_ids=("spring-1", "spring-2"),
    node_ids=(1, 2),
    stiffness=2.0,
    critical_extension=2.0,
    max_increment=5.0,
    source=_source("damage-spring-source"),
  )
  assert declaration.signal_ports == ()
  first = compile_spring_operator(_base_system(), declaration)[1]
  second = compile_spring_operator(_base_system(), declaration)[1]
  # The no-signal path is byte-identical: same manifest bytes, no ports, and
  # no signal key versioned into the content.
  assert first.header.signal_ports == ()
  assert first.content_manifest.to_bytes() == second.content_manifest.to_bytes()
  assert "signal_ports" not in _manifest_content(first.content_manifest)
  with pytest.raises(ValueError, match="does not accept program signal inputs"):
    _evaluate(first, 0.5)


def test_ported_declaration_emits_bindings_and_versions_the_manifest() -> None:
  _, operator = compile_spring_operator(_base_system(), _ported_declaration())
  (port,) = operator.header.signal_ports
  assert port.port_id == "load-factor"
  assert port.signal_id == "load"
  assert port.derivative_coordinate_ids == ()
  content = _manifest_content(operator.content_manifest)
  assert "signal_ports" in content
  # The same network without the port declaration compiles a byte-distinct
  # operator with no signal key versioned into its manifest.
  _, unported = compile_spring_operator(
    _base_system(),
    _ported_declaration(
      kernel=lambda displacements, accepted_rows, parameters: SpringKernelResult(
        force=parameters[0] * displacements,
        tangent=parameters[0] * np.tile(np.eye(2), (len(displacements), 1, 1)),
        trial_rows=np.array(accepted_rows, copy=True),
        status=EvaluationStatus.OK,
      ),
      signal_ports=(),
    ),
  )
  assert unported.header.signal_ports == ()
  assert "signal_ports" not in _manifest_content(unported.content_manifest)
  assert operator.content_manifest.to_bytes() != unported.content_manifest.to_bytes()


def test_signal_evaluation_forwards_scalars_and_derivatives() -> None:
  received: list[tuple[float, tuple[tuple[str, float], ...]]] = []

  def recording_kernel(
    displacements: np.ndarray,
    accepted_rows: np.ndarray,
    parameters: np.ndarray,
    signals: tuple[SpringSignalInput, ...],
  ) -> SpringKernelResult:
    (signal,) = signals
    received.append(
      (
        float(signal.values[0]),
        tuple(
          (derivative.coordinate_id, float(derivative.values[0]))
          for derivative in signal.derivatives
        ),
      )
    )
    return modulated_spring_kernel(displacements, accepted_rows, parameters, signals)

  port = SpringSignalPort(
    port_id="load-factor",
    signal_id="load",
    derivative_coordinate_ids=("load", "time"),
  )
  _, operator = compile_spring_operator(
    _base_system(),
    _ported_declaration(kernel=recording_kernel, signal_ports=(port,)),
  )
  recorded_at_compile = len(received)
  assert recorded_at_compile > 0  # the compile probes consume the kernel too
  received.clear()
  displacements = np.array([[0.25, -0.5]], dtype=np.float64)
  signal = _signal_input(
    0.75,
    derivatives=(("load", 1.0), ("time", 0.0)),
  )
  evaluation = operator.evaluate(
    OperatorEvaluationInput(
      port_values=(FinalizedArray(displacements, dtype=np.float64),),
      accepted_state=FinalizedArray(
        np.zeros(operator.header.state_layout.row_shape), dtype=np.float64
      ),
      signals=(signal,),
      request=ChannelRequest(("spring-force",), ("spring-tangent",)),
    )
  )
  assert evaluation_status(evaluation) is EvaluationStatus.OK
  assert received == [(0.75, (("load", 1.0), ("time", 0.0)))]
  # Zero-copy forwarding: the kernel reads the evaluation's own array.
  np.testing.assert_array_equal(
    evaluation.residual_values[0].values,
    0.75 * 100.0 * displacements,
  )
  np.testing.assert_array_equal(
    evaluation.jacobian_values[0].values,
    (0.75 * 100.0) * np.eye(2)[None],
  )


def test_signal_evaluation_binding_is_fail_closed() -> None:
  _, operator = compile_spring_operator(_base_system(), _ported_declaration())
  layout = operator.header.state_layout
  state = FinalizedArray(np.zeros(layout.row_shape), dtype=np.float64)
  batch = FinalizedArray(np.zeros((layout.entity_count, 2)), dtype=np.float64)
  request = ChannelRequest(("spring-force",), ("spring-tangent",))

  def attempt(signals: tuple[ProgramSignalInput, ...]) -> OperatorEvaluation:
    return operator.evaluate(
      OperatorEvaluationInput(
        port_values=(batch,),
        accepted_state=state,
        signals=signals,
        request=request,
      )
    )

  with pytest.raises(ValueError, match="missing a declared program signal port"):
    attempt(())
  with pytest.raises(ValueError, match="undeclared program signal port"):
    attempt((_signal_input(0.5, port_id="foreign"),))
  with pytest.raises(ValueError, match="duplicate program signal port"):
    attempt((_signal_input(0.5), _signal_input(0.6)))
  mismatched = ProgramSignalInput(
    port_id="load-factor",
    values=FinalizedArray(np.array([0.5]), dtype=np.float64),
    derivatives=(
      SignalDerivativeInput(
        "load",
        FinalizedArray(np.array([1.0]), dtype=np.float64),
      ),
    ),
  )
  with pytest.raises(ValueError, match="match the declared coordinates"):
    attempt((mismatched,))
  nonscalar = ProgramSignalInput(
    port_id="load-factor",
    values=FinalizedArray(np.array([0.5, 0.6]), dtype=np.float64),
    derivatives=(),
  )
  with pytest.raises(TypeError, match="one-element float64"):
    attempt((nonscalar,))


def test_ported_declaration_requires_the_four_argument_kernel() -> None:
  def three_arg_kernel(
    displacements: np.ndarray,
    accepted_rows: np.ndarray,
    parameters: np.ndarray,
  ) -> SpringKernelResult:
    (stiffness,) = parameters
    return SpringKernelResult(
      force=stiffness * displacements,
      tangent=stiffness * np.tile(np.eye(2), (len(displacements), 1, 1)),
      trial_rows=np.array(accepted_rows, copy=True),
      status=EvaluationStatus.OK,
    )

  with pytest.raises(ModelCompilationError) as excinfo:
    compile_spring_operator(
      _base_system(),
      _ported_declaration(kernel=three_arg_kernel),
    )
  (diagnostic,) = excinfo.value.diagnostics
  assert diagnostic.code == "kernel-probe-failed"
  assert diagnostic.source.source == "modulated-spring-source"


def _truss_model() -> ModelSpec:
  """The ch.4 shallow-truss geometry (spring-system battery's model)."""
  nodes = (
    NodeSpec(id=0, coordinates=(-10.0, 0.0), source=_source("n0")),
    NodeSpec(id=1, coordinates=(10.0, 0.0), source=_source("n1")),
    NodeSpec(id=2, coordinates=(0.0, 0.5), source=_source("n2")),
  )
  cells = (
    CellSpec(id="left", node_ids=(0, 2), source=_source("c0")),
    CellSpec(id="right", node_ids=(1, 2), source=_source("c1")),
  )
  block = CellBlockSpec(
    id="bars",
    reference_topology="line",
    topological_dimension=1,
    embedding_dimension=2,
    geometry_interpolation="line2",
    cells=cells,
    source=_source("block"),
  )
  field = FieldSpec(
    id="displacement",
    components=("x", "y"),
    location="node",
    source=_source("field"),
  )
  material = MaterialSpec(
    id="steel",
    model="uniaxial-linear-elastic",
    parameters=(
      MaterialParameterSpec("youngs_modulus", 5.0e6),
      MaterialParameterSpec("area", 1.0),
    ),
    source=_source("material"),
  )
  region = RegionSpec(
    id="domain",
    cell_refs=(CellRef("bars", "left"), CellRef("bars", "right")),
    field_ids=(field.id,),
    material_id="steel",
    formulation="total-lagrangian-truss",
    quadrature="none",
    source=_source("region"),
  )
  return ModelSpec(
    mesh=MeshSpec(nodes=nodes, cell_blocks=(block,), source=_source("mesh")),
    fields=(field,),
    materials=(material,),
    regions=(region,),
    source=_source("model"),
  )


def _truss_driver(system: CompiledSystem) -> NonlinearStaticDriver:
  coordinate_map = compile_constraint_map(
    system,
    constraints=tuple(
      PrescribedDofSpec(
        id=f"fix-{node}-{component}",
        target=DofRef(
          node_id=node,
          field_id="displacement",
          component=component,
        ),
        value=AffineValueSpec(constant=0.0),
        source=_source(f"fix-{node}-{component}"),
      )
      for node in (0, 1)
      for component in ("x", "y")
    ),
    coordinates=(ProgramCoordinateSpec(name="load", kind="load"),),
  )
  loads = (
    NodalLoadSpec(
      id="apex",
      target=DofRef(node_id=2, field_id="displacement", component="y"),
      value=AffineValueSpec(
        coefficients=(AffineCoefficientSpec("load", -100.0, _source("coef")),),
        source=_source("load"),
      ),
      source=_source("apex"),
    ),
  )
  return NonlinearStaticDriver(system, coordinate_map, loads)


def _ramp(factors: tuple[float, ...]) -> tuple[ProgramPoint, ...]:
  return tuple(
    ProgramPoint((ProgramCoordinateValue("load", factor),)) for factor in factors
  )


def test_driver_forwards_the_load_coordinate_to_the_spring_kernel() -> None:
  seen: list[float] = []

  def recording_kernel(
    displacements: np.ndarray,
    accepted_rows: np.ndarray,
    parameters: np.ndarray,
    signals: tuple[SpringSignalInput, ...],
  ) -> SpringKernelResult:
    seen.append(float(signals[0].values[0]))
    return modulated_spring_kernel(displacements, accepted_rows, parameters, signals)

  base = compile_system(_truss_model(), truss_reference_registry())
  declaration = SpringDeclaration(
    block_id="apex-spring",
    space_id="displacement",
    spring_ids=("apex-support",),
    node_ids=(2,),
    state_schema="apex-modulated-spring-v1",
    state_slots=(),
    kernel_name="signal-modulated-spring",
    kernel_version="1",
    implementation_id="apex-modulated-spring-v1",
    parameters=(100.0,),
    kernel=recording_kernel,
    source=_source("apex-spring-source"),
    signal_ports=(_LOAD_PORT,),
  )
  block, operator = compile_spring_operator(base, declaration)
  system = compose_system(base, block, operator)
  assert "signals" in _manifest_content(_truss_driver(system).plan.provenance.manifest)
  seen.clear()
  driver = _truss_driver(system)
  result = driver.run(
    base_point=_ramp((0.0,))[0],
    target_points=_ramp((0.5, 1.0)),
  )
  assert result.status is DriverStatus.COMPLETED
  assert result.statistics.committed_substep_count == 2
  # Every evaluation of a substep binds that substep's fixed trial point.
  assert sorted(set(seen)) == [0.5, 1.0]
  state = driver.owner.accepted_physical().values
  apex = state[4:6]
  # The committed spring response at lam=1 is exactly lam*k*u_apex.
  spring_evaluation = _evaluate(operator, 1.0, apex[None, :])
  np.testing.assert_array_equal(
    spring_evaluation.residual_values[0].values,
    100.0 * apex[None, :],
  )
  # The signal genuinely drives the response: the same network with the
  # stiffness fixed at its full value equilibrates at a different apex state.
  fixed_base = compile_system(_truss_model(), truss_reference_registry())
  fixed_block, fixed_operator = compile_spring_operator(
    fixed_base,
    SpringDeclaration(
      block_id="apex-spring",
      space_id="displacement",
      spring_ids=("apex-support",),
      node_ids=(2,),
      state_schema="apex-fixed-spring-v1",
      state_slots=(),
      kernel_name="axial-linear-spring",
      kernel_version="1",
      implementation_id="apex-fixed-spring-v1",
      parameters=(100.0,),
      kernel=lambda displacements, accepted_rows, parameters: SpringKernelResult(
        force=parameters[0] * displacements,
        tangent=parameters[0] * np.tile(np.eye(2), (len(displacements), 1, 1)),
        trial_rows=np.array(accepted_rows, copy=True),
        status=EvaluationStatus.OK,
      ),
      source=_source("apex-spring-source"),
    ),
  )
  fixed_system = compose_system(fixed_base, fixed_block, fixed_operator)
  fixed_driver = _truss_driver(fixed_system)
  fixed_result = fixed_driver.run(
    base_point=_ramp((0.0,))[0],
    target_points=_ramp((0.5, 1.0)),
  )
  assert fixed_result.status is DriverStatus.COMPLETED
  # The trajectories separate at the first committed substep (lam=0.5), where
  # the modulated stiffness is half the fixed one; they rejoin at lam=1 by
  # construction of the two laws (hyperelastic substrate, path-independent).
  half_modulated = _truss_driver(system)
  half_modulated.run(base_point=_ramp((0.0,))[0], target_points=_ramp((0.5,)))
  half_fixed = _truss_driver(fixed_system)
  half_fixed.run(base_point=_ramp((0.0,))[0], target_points=_ramp((0.5,)))
  assert not np.allclose(
    half_modulated.owner.accepted_physical().values,
    half_fixed.owner.accepted_physical().values,
    rtol=1e-9,
    atol=1e-12,
  )
  np.testing.assert_allclose(
    driver.owner.accepted_physical().values,
    fixed_driver.owner.accepted_physical().values,
    rtol=1e-9,
    atol=1e-12,
  )
