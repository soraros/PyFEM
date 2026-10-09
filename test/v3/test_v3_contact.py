# SPDX-License-Identifier: MIT

"""Penalty contact operator battery (M66 task 2 and the unit legs of task 3).

Pins the compiled operator's header shape and manifest, the kernel's bitwise
parity with the legacy ``pyfem.models.Contact`` force law, and the conviction
pair for the exact symmetric tangent: the compile-time finite-difference probe
accepts the exact form and rejects legacy's appended ``penalty * n x n`` form
(measured inexact by exactly ``overlap/d``, the M62 survey's finding 5), and a
planted legacy-tangent kernel fails compilation with the coded diagnostic.
"""

from __future__ import annotations

import math
import sys

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

import pyfem.v3.compile.contact as contact_compile
from pyfem.v3.compile.contact import (
  ContactDeclaration,
  ContactKernelResult,
  ContactSignalInput,
  ContactSignalPort,
  PenaltyContactOperator,
  compile_contact_operator,
  compose_contact_system,
  penalty_disc_declaration,
  penalty_disc_kernel,
)
from pyfem.v3.compile.continuum import q8_reference_registry
from pyfem.v3.compile.contracts import continuum_reference_registry
from pyfem.v3.compile.diagnostics import ModelCompilationError
from pyfem.v3.compile.system import compile_system
from pyfem.v3.constraints import compile_constraint_map
from pyfem.v3.driver import DriverStatus, NonlinearStaticDriver
from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.identity import IdentityMismatchError
from pyfem.v3.model.operator import (
  ChannelRequest,
  EvaluationStatus,
  OperatorEvaluation,
  OperatorEvaluationInput,
  ProgramSignalInput,
  SignalDerivativeInput,
  evaluation_status,
)
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

_CENTRE = (0.5, 1.3)
_DIRECTION = (0.0, -0.1)
_RADIUS = 0.5
_PENALTY = 1.0e6
_PARAMETERS = np.array(
  (_PENALTY, _RADIUS, *_CENTRE, *_DIRECTION),
  dtype=np.float64,
)


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
      MaterialParameterSpec("youngs_modulus", 1.0e6, _source("material:E")),
      MaterialParameterSpec("poisson_ratio", 0.25, _source("material:nu")),
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


def _hex8_model() -> ModelSpec:
  nodes = tuple(
    NodeSpec(id=index, coordinates=point, source=_source(f"hex-node-{index}"))
    for index, point in enumerate(
      (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
        (0.0, 1.0, 1.0),
      )
    )
  )
  cell = CellSpec(
    id="cell-1",
    node_ids=tuple(node.id for node in nodes),
    source=_source("hex-cell"),
  )
  block = CellBlockSpec(
    id="cells",
    reference_topology="hexahedron",
    topological_dimension=3,
    embedding_dimension=3,
    geometry_interpolation="trilinear-hex8",
    cells=(cell,),
    source=_source("hex-block"),
  )
  field = FieldSpec(
    id="displacement",
    components=("x", "y", "z"),
    location="node",
    source=_source("hex-field"),
  )
  material = MaterialSpec(
    id="elastic",
    model="isotropic-linear-elastic",
    parameters=(
      MaterialParameterSpec("youngs_modulus", 1.0e6, _source("material:E")),
      MaterialParameterSpec("poisson_ratio", 0.25, _source("material:nu")),
    ),
    source=_source("hex-material"),
  )
  region = RegionSpec(
    id="domain",
    cell_refs=(CellRef(block.id, cell.id),),
    field_ids=(field.id,),
    material_id=material.id,
    formulation="small-strain-continuum",
    quadrature="gauss-2x2x2",
    source=_source("hex-region"),
  )
  return ModelSpec(
    mesh=MeshSpec(nodes=nodes, cell_blocks=(block,), source=_source("hex-mesh")),
    fields=(field,),
    materials=(material,),
    regions=(region,),
    source=_source("hex-model"),
  )


def _base_system() -> CompiledSystem:
  return compile_system(_q8_model(), q8_reference_registry())


def _declaration(**overrides: object) -> ContactDeclaration:
  fields: dict[str, object] = {
    "block_id": "obstacle-contact",
    "space_id": "displacement",
    "contact_ids": tuple(f"contact-{index + 1}" for index in range(8)),
    "node_ids": tuple(index + 1 for index in range(8)),
    "centre": _CENTRE,
    "direction": _DIRECTION,
    "radius": _RADIUS,
    "penalty": _PENALTY,
    "state_schema": "test-contact-state-v1",
    "kernel_name": "penalty-disc-contact",
    "kernel_version": "1",
    "implementation_id": "test-contact-v1",
    "kernel": penalty_disc_kernel,
    "source": _source("contact-source"),
    "signal_ports": (ContactSignalPort(port_id="load-factor", signal_id="load"),),
  }
  fields.update(overrides)
  return ContactDeclaration(**fields)  # type: ignore[arg-type]


def _compiled_operator() -> PenaltyContactOperator:
  return compile_contact_operator(_base_system(), _declaration())[1]


def _lam_signal(value: float) -> ContactSignalInput:
  return ContactSignalInput(
    port_id="load-factor",
    values=np.array([value], dtype=np.float64),
    derivatives=(),
  )


def _evaluate(
  operator: PenaltyContactOperator,
  displacements: np.ndarray,
  lam: float,
) -> OperatorEvaluation:
  layout = operator.header.state_layout
  return operator.evaluate(
    OperatorEvaluationInput(
      port_values=(FinalizedArray(displacements, dtype=np.float64),),
      accepted_state=FinalizedArray(np.zeros(layout.row_shape), dtype=np.float64),
      signals=(
        ProgramSignalInput(
          port_id="load-factor",
          values=FinalizedArray(np.array([lam], dtype=np.float64), dtype=np.float64),
          derivatives=(),
        ),
      ),
      request=ChannelRequest(("contact-force",), ("contact-tangent",)),
    )
  )


def _legacy_force_rows(
  positions: np.ndarray,
  centre: np.ndarray,
  radius: float,
  penalty: float,
) -> np.ndarray:
  """The legacy Contact force law, operation for operation, per node."""
  force = np.zeros_like(positions)
  for index in range(len(positions)):
    ds = positions[index] - centre
    dsnorm = np.linalg.norm(ds)
    overlap = radius - dsnorm
    if overlap > 0:
      normal = ds / dsnorm
      force[index] = -penalty * overlap * normal
  return force


def _probe_states(
  entity_count: int,
  centre: tuple[float, float],
  radius: float,
) -> tuple[np.ndarray, ...]:
  """Replay the compile probe's seeded obstacle-relative state placement."""
  generator = np.random.default_rng(contact_compile._TANGENT_PROBE_SEED)
  base_centre = np.array(centre, dtype=np.float64)
  states = []
  for band in contact_compile._TANGENT_PROBE_BANDS:
    angles = generator.uniform(0.0, 2.0 * math.pi, entity_count)
    distances = radius * generator.uniform(band[0], band[1], entity_count)
    states.append(
      base_centre[None, :]
      + distances[:, None] * np.stack((np.cos(angles), np.sin(angles)), axis=1)
    )
  return tuple(states)


def _central_difference_tangent(
  positions: np.ndarray,
  row_width: int = 0,
) -> np.ndarray:
  """Central FD of the landed kernel force along each position component."""
  rows = np.zeros((len(positions), row_width), dtype=np.float64)
  signal = _lam_signal(0.0)
  tangent = np.zeros((len(positions), 2, 2), dtype=np.float64)
  for entity_index in range(len(positions)):
    for component in range(2):
      step = contact_compile._TANGENT_PROBE_STEP * max(
        1.0,
        abs(float(positions[entity_index, component])),
      )
      plus = np.array(positions, copy=True)
      minus = np.array(positions, copy=True)
      plus[entity_index, component] += step
      minus[entity_index, component] -= step
      plus_force = penalty_disc_kernel(plus, rows, _PARAMETERS, (signal,)).force
      minus_force = penalty_disc_kernel(minus, rows, _PARAMETERS, (signal,)).force
      tangent[entity_index, :, component] = (
        plus_force[entity_index] - minus_force[entity_index]
      ) / (2.0 * step)
  return tangent


def _legacy_tangent(positions: np.ndarray) -> np.ndarray:
  """Legacy's appended tangent: penalty * n x n on engaged entities."""
  centre = np.array(_CENTRE, dtype=np.float64)
  tangent = np.zeros((len(positions), 2, 2), dtype=np.float64)
  for index, position in enumerate(positions):
    ds = position - centre
    dsnorm = np.linalg.norm(ds)
    if _RADIUS - dsnorm > 0:
      normal = ds / dsnorm
      tangent[index] = _PENALTY * np.outer(normal, normal)
  return tangent


def test_landed_declaration_helper_pins_the_law_metadata() -> None:
  declaration = penalty_disc_declaration(
    block_id="obstacle-contact",
    space_id="displacement",
    contact_ids=("contact-1",),
    node_ids=(1,),
    centre=_CENTRE,
    direction=_DIRECTION,
    radius=_RADIUS,
    penalty=_PENALTY,
    signal_port=ContactSignalPort(port_id="load-factor", signal_id="load"),
    source=_source("contact-source"),
  )
  assert declaration.kernel is penalty_disc_kernel
  assert declaration.state_schema == contact_compile.PENALTY_DISC_CONTACT_SCHEMA
  assert declaration.implementation_id == contact_compile.PENALTY_DISC_CONTACT_SCHEMA
  assert declaration.kernel_name == "penalty-disc-contact"
  assert declaration.signal_ports[0].signal_id == "load"
  with pytest.raises(TypeError, match="exactly one obstacle schedule signal port"):
    penalty_disc_declaration(
      block_id="obstacle-contact",
      space_id="displacement",
      contact_ids=("contact-1",),
      node_ids=(1,),
      centre=_CENTRE,
      direction=_DIRECTION,
      radius=_RADIUS,
      penalty=_PENALTY,
      signal_port="load",  # type: ignore[arg-type]
      source=_source("contact-source"),
    )
  # The helper's declaration compiles end-to-end on the base system.
  block, operator = compile_contact_operator(_base_system(), declaration)
  assert operator.header.entity_block_id == block.block_id == "obstacle-contact"


def test_compile_emits_the_declared_header_shape_and_manifest() -> None:
  block, operator = compile_contact_operator(_base_system(), _declaration())
  assert isinstance(operator, PenaltyContactOperator)
  header = operator.header
  assert header.block_id == ("obstacle-contact", "test-contact-state-v1")
  assert header.entity_block_id == "obstacle-contact"
  assert tuple(item.channel_id for item in header.residual_channels) == (
    "contact-force",
  )
  (jacobian,) = header.jacobian_channels
  assert jacobian.channel_id == "contact-tangent"
  assert jacobian.residual_channel_id == "contact-force"
  assert jacobian.linear is False
  assert jacobian.symmetric is True
  (port,) = header.ports
  assert port.port_id == "displacement"
  (signal_port,) = header.signal_ports
  assert signal_port.port_id == "load-factor"
  assert signal_port.signal_id == "load"
  assert signal_port.derivative_coordinate_ids == ()
  # Stateless: the active set is a pure function of the trial point.
  assert header.state_layout.row_shape == (8, 0)
  np.testing.assert_array_equal(
    block.reference_coordinates.values,
    np.array(_UNIT_COORDINATES, dtype=np.float64),
  )
  np.testing.assert_array_equal(
    operator.payload.parameters.values,
    _PARAMETERS,
  )
  manifest_bytes = operator.content_manifest.to_bytes()
  assert b"signal_ports" in manifest_bytes
  assert b"penalty-disc-contact" in manifest_bytes
  # Compilation is deterministic: identical declaration, identical bytes.
  _, twin = compile_contact_operator(_base_system(), _declaration())
  assert twin.content_manifest.to_bytes() == manifest_bytes


def test_kernel_force_is_bitwise_identical_to_the_legacy_force_law() -> None:
  generator = np.random.default_rng(20261009)
  rows = np.zeros((16, 0), dtype=np.float64)
  for _ in range(25):
    lam = float(generator.uniform(0.0, 1.0))
    offsets = generator.uniform(0.0, 1.5 * _RADIUS, (16, 2))
    signs = generator.choice((-1.0, 1.0), (16, 2))
    positions = np.array(_CENTRE)[None, :] + offsets * signs
    centre = np.array(_CENTRE) + lam * np.array(_DIRECTION)
    kernel_signal = _lam_signal(lam)
    result = penalty_disc_kernel(positions, rows, _PARAMETERS, (kernel_signal,))
    assert result.status is EvaluationStatus.OK
    legacy = _legacy_force_rows(positions, centre, _RADIUS, _PENALTY)
    np.testing.assert_array_equal(result.force, legacy)


def test_exact_tangent_matches_central_fd_at_the_probe_states() -> None:
  signal = _lam_signal(0.0)
  rows = np.zeros((8, 0), dtype=np.float64)
  measured: list[float] = []
  for positions in _probe_states(8, _CENTRE, _RADIUS):
    result = penalty_disc_kernel(positions, rows, _PARAMETERS, (signal,))
    fd = _central_difference_tangent(positions)
    for entity_index in range(len(positions)):
      column_scale = max(
        1.0,
        float(np.abs(result.tangent[entity_index]).max()),
        float(np.abs(fd[entity_index]).max()),
      )
      measured.append(
        float(np.abs(result.tangent[entity_index] - fd[entity_index]).max())
        / column_scale
      )
    # The exact tangent is bitwise symmetric.
    np.testing.assert_array_equal(
      result.tangent,
      np.swapaxes(result.tangent, 1, 2),
    )
  # Measured max relative FD deviation on this host: ~1e-9 (engaged bands
  # carry tangents up to 1e5 with h^2 truncation error ~1e-9 relative).
  assert max(measured) < 1.0e-6


def test_legacy_appended_tangent_fails_the_same_fd_by_overlap_over_d() -> None:
  signal = _lam_signal(0.0)
  rows = np.zeros((8, 0), dtype=np.float64)
  centre = np.array(_CENTRE, dtype=np.float64)
  for state_index, positions in enumerate(_probe_states(8, _CENTRE, _RADIUS)):
    fd = _central_difference_tangent(positions)
    legacy = _legacy_tangent(positions)
    for entity_index, position in enumerate(positions):
      ds = position - centre
      distance = float(np.linalg.norm(ds))
      overlap = _RADIUS - distance
      scale = max(
        1.0,
        float(np.abs(fd[entity_index]).max()),
        float(np.abs(legacy[entity_index]).max()),
      )
      mismatch = float(np.abs(legacy[entity_index] - fd[entity_index]).max()) / scale
      if overlap <= 0.0:
        # Disengaged: legacy's tangent is exactly zero, matching the FD.
        assert state_index == 1
        assert mismatch == 0.0
        continue
      # Engaged: the measured relative mismatch equals overlap/d exactly (the
      # survey's finding: 1.01% at overlap=0.01R), past the probe's 1e-4 gate
      # by 100x on the deep band and by 10x even on the boundary band.
      assert mismatch > 10.0 * contact_compile._TANGENT_PROBE_RTOL
      if state_index == 0:
        assert mismatch > 100.0 * contact_compile._TANGENT_PROBE_RTOL
      assert 0.8 * (overlap / distance) <= mismatch <= 1.25 * (overlap / distance)
      result = penalty_disc_kernel(
        positions[entity_index : entity_index + 1],
        rows[:1],
        _PARAMETERS,
        (signal,),
      )
      exact_mismatch = float(np.abs(result.tangent[0] - fd[entity_index]).max()) / scale
      assert exact_mismatch < 1.0e-6


def test_planted_legacy_tangent_kernel_fails_compilation() -> None:
  def legacy_tangent_kernel(
    current_positions: np.ndarray,
    accepted_rows: np.ndarray,
    parameters: np.ndarray,
    signals: tuple[ContactSignalInput, ...],
  ) -> ContactKernelResult:
    result = penalty_disc_kernel(current_positions, accepted_rows, parameters, signals)
    tangent = np.zeros_like(result.tangent)
    centre = parameters[2:4] + signals[0].values[0] * parameters[4:]
    for index, position in enumerate(current_positions):
      ds = position - centre
      dsnorm = np.linalg.norm(ds)
      if parameters[1] - dsnorm > 0:
        normal = ds / dsnorm
        tangent[index] = parameters[0] * np.outer(normal, normal)
    return ContactKernelResult(
      force=result.force,
      tangent=tangent,
      trial_rows=result.trial_rows,
      status=result.status,
    )

  with pytest.raises(ModelCompilationError) as excinfo:
    compile_contact_operator(_base_system(), _declaration(kernel=legacy_tangent_kernel))
  (diagnostic,) = excinfo.value.diagnostics
  assert diagnostic.code == "inconsistent-kernel-tangent"
  assert str(contact_compile._TANGENT_PROBE_SEED) in diagnostic.message
  assert diagnostic.source.source == "contact-source"


def test_boundary_state_is_continuous_and_the_disengaged_side_is_zero() -> None:
  signal = _lam_signal(0.0)
  rows = np.zeros((1, 0), dtype=np.float64)
  direction = np.array([1.0, 0.0])
  centre = np.array(_CENTRE, dtype=np.float64)
  for exponent in (1.0e-12, 1.0e-9, 1.0e-6, 1.0e-3):
    inside = (centre + (_RADIUS - exponent) * direction).reshape(1, 2)
    result = penalty_disc_kernel(inside, rows, _PARAMETERS, (signal,))
    # The force is axial, inward-negative, of magnitude penalty * overlap.
    # Position arithmetic rounds at the ulp level (~1e-16 here), which maps to
    # ~1e-10 in the force through the 1e6 penalty, so the check is absolute.
    assert result.force[0, 1] == 0.0
    assert result.force[0, 0] < 0.0
    np.testing.assert_allclose(
      -result.force[0, 0],
      _PENALTY * exponent,
      rtol=0.0,
      atol=1.0e-9,
    )
    outside = (centre + (_RADIUS + exponent) * direction).reshape(1, 2)
    result = penalty_disc_kernel(outside, rows, _PARAMETERS, (signal,))
    np.testing.assert_array_equal(result.force, np.zeros((1, 2)))
    np.testing.assert_array_equal(result.tangent, np.zeros((1, 2, 2)))
  # Exactly on the boundary the legacy strict inequality disengages.
  on_boundary = (centre + _RADIUS * direction).reshape(1, 2)
  result = penalty_disc_kernel(on_boundary, rows, _PARAMETERS, (signal,))
  np.testing.assert_array_equal(result.force, np.zeros((1, 2)))


def test_compile_is_fail_closed() -> None:
  base = _base_system()
  with pytest.raises(ModelCompilationError) as excinfo:
    compile_contact_operator(base, _declaration(space_id="thermal"))
  (diagnostic,) = excinfo.value.diagnostics
  assert diagnostic.code == "unknown-contact-space"

  with pytest.raises(ModelCompilationError) as excinfo:
    compile_contact_operator(
      base, _declaration(node_ids=tuple(range(1, 9))[:-1] + (99,))
    )
  (diagnostic,) = excinfo.value.diagnostics
  assert diagnostic.code == "unknown-contact-support-node"

  # The dimension-generic kernel is pinned to two-component spaces by a coded
  # rejection until a 3D oracle exists (the legacy sphere branch is dead).
  hex8 = compile_system(_hex8_model(), continuum_reference_registry())
  with pytest.raises(ModelCompilationError) as excinfo:
    compile_contact_operator(hex8, _declaration())
  (diagnostic,) = excinfo.value.diagnostics
  assert diagnostic.code == "unsupported-contact-space"
  assert "no three-dimensional oracle" in diagnostic.message

  def raising_kernel(
    current_positions: np.ndarray,
    accepted_rows: np.ndarray,
    parameters: np.ndarray,
    signals: tuple[ContactSignalInput, ...],
  ) -> ContactKernelResult:
    msg = "researcher failure at the virgin state"
    raise RuntimeError(msg)

  with pytest.raises(ModelCompilationError) as excinfo:
    compile_contact_operator(base, _declaration(kernel=raising_kernel))
  (diagnostic,) = excinfo.value.diagnostics
  assert diagnostic.code == "kernel-probe-failed"

  def rejecting_kernel(
    current_positions: np.ndarray,
    accepted_rows: np.ndarray,
    parameters: np.ndarray,
    signals: tuple[ContactSignalInput, ...],
  ) -> ContactKernelResult:
    return ContactKernelResult(
      force=np.zeros_like(current_positions),
      tangent=np.zeros((len(current_positions), 2, 2)),
      trial_rows=np.array(accepted_rows, copy=True),
      status=EvaluationStatus.REJECT_STEP,
    )

  with pytest.raises(ModelCompilationError) as excinfo:
    compile_contact_operator(base, _declaration(kernel=rejecting_kernel))
  (diagnostic,) = excinfo.value.diagnostics
  assert diagnostic.code == "invalid-kernel-probe"

  # The landed kernel binds exactly one signal: a signal-less declaration
  # fails the virgin probe fail-closed.
  with pytest.raises(ModelCompilationError) as excinfo:
    compile_contact_operator(base, _declaration(signal_ports=()))
  (diagnostic,) = excinfo.value.diagnostics
  assert diagnostic.code == "kernel-probe-failed"

  # Declaration validation legs.
  with pytest.raises(TypeError, match="positive finite exact float"):
    _declaration(radius=-1.0)
  with pytest.raises(TypeError, match="finite exact float pair"):
    _declaration(centre=(0.0, 0.0, 0.0))
  with pytest.raises(ValueError, match="signal port ids must be unique"):
    _declaration(
      signal_ports=(
        ContactSignalPort(port_id="lam", signal_id="load"),
        ContactSignalPort(port_id="lam", signal_id="load"),
      )
    )


def test_evaluation_is_fail_closed() -> None:
  operator = _compiled_operator()
  layout = operator.header.state_layout
  batch = FinalizedArray(np.zeros((8, 2)), dtype=np.float64)
  state = FinalizedArray(np.zeros(layout.row_shape), dtype=np.float64)
  signal = ProgramSignalInput(
    port_id="load-factor",
    values=FinalizedArray(np.array([0.5]), dtype=np.float64),
    derivatives=(),
  )
  request = ChannelRequest(("contact-force",), ("contact-tangent",))

  def attempt(
    *,
    port_values: tuple[FinalizedArray, ...] = (batch,),
    signals: tuple[ProgramSignalInput, ...] = (signal,),
    channel_request: ChannelRequest = request,
  ) -> OperatorEvaluation:
    return operator.evaluate(
      OperatorEvaluationInput(
        port_values=port_values,
        accepted_state=state,
        signals=signals,
        request=channel_request,
      )
    )

  with pytest.raises(TypeError, match="exactly one displacement port batch"):
    attempt(port_values=())
  with pytest.raises(TypeError, match="finite float64 batch"):
    attempt(port_values=(FinalizedArray(np.full((8, 2), np.nan), dtype=np.float64),))
  with pytest.raises(ValueError, match="unavailable or duplicate channel"):
    attempt(channel_request=ChannelRequest(("spring-force",), ()))
  with pytest.raises(ValueError, match="missing a declared program signal port"):
    attempt(signals=())
  with pytest.raises(ValueError, match="undeclared program signal port"):
    attempt(
      signals=(
        ProgramSignalInput(
          port_id="foreign",
          values=FinalizedArray(np.array([0.5]), dtype=np.float64),
          derivatives=(),
        ),
      )
    )
  with pytest.raises(ValueError, match="duplicate program signal port"):
    attempt(signals=(signal, signal))
  with pytest.raises(ValueError, match="match the declared coordinates"):
    attempt(
      signals=(
        ProgramSignalInput(
          port_id="load-factor",
          values=FinalizedArray(np.array([0.5]), dtype=np.float64),
          derivatives=(
            SignalDerivativeInput(
              "load",
              FinalizedArray(np.array([1.0]), dtype=np.float64),
            ),
          ),
        ),
      )
    )
  with pytest.raises(TypeError, match="one-element float64"):
    attempt(
      signals=(
        ProgramSignalInput(
          port_id="load-factor",
          values=FinalizedArray(np.array([0.5, 0.6]), dtype=np.float64),
          derivatives=(),
        ),
      )
    )


def test_compose_is_fail_closed_and_deterministic() -> None:
  base = _base_system()
  block, operator = compile_contact_operator(base, _declaration())
  composed = compose_contact_system(base, block, operator)
  assert tuple(type(item).__name__ for item in composed.operators) == (
    "Q8ContinuumOperator",
    "PenaltyContactOperator",
  )
  assert composed.point_blocks[-1].block_id == "obstacle-contact"
  # A second network with the same block id collides on the composed system.
  twin_block, twin_operator = compile_contact_operator(composed, _declaration())
  with pytest.raises(ValueError, match="collides"):
    compose_contact_system(composed, twin_block, twin_operator)
  # A foreign system's operator is rejected by instance identity.
  foreign_block, foreign_operator = compile_contact_operator(
    _base_system(),
    _declaration(),
  )
  with pytest.raises(IdentityMismatchError, match="same live instance"):
    compose_contact_system(base, foreign_block, foreign_operator)
  # The operator must bind the exact composed block object.
  other_block, other_operator = compile_contact_operator(base, _declaration())
  assert other_block is not block
  with pytest.raises(ValueError, match="exact composed contact block"):
    compose_contact_system(base, block, other_operator)


def _contact_driver(system: CompiledSystem) -> NonlinearStaticDriver:
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
      for node, component in ((1, "x"), (1, "y"), (2, "y"), (3, "y"))
    ),
    coordinates=(ProgramCoordinateSpec(name="load", kind="load"),),
  )
  return NonlinearStaticDriver(system, coordinate_map, ())


def _ramp(factors: tuple[float, ...]) -> tuple[ProgramPoint, ...]:
  return tuple(
    ProgramPoint((ProgramCoordinateValue("load", factor),)) for factor in factors
  )


def _contact_system(
  centre: tuple[float, float],
  direction: tuple[float, float],
) -> tuple[CompiledSystem, PenaltyContactOperator]:
  base = _base_system()
  block, operator = compile_contact_operator(
    base,
    _declaration(centre=centre, direction=direction),
  )
  return compose_contact_system(base, block, operator), operator


def test_driver_engages_and_disengages_contact_across_substeps() -> None:
  # Engage: the disc starts clear of the top edge by 0.1 and presses in.
  system, operator = _contact_system((0.5, 1.6), (0.0, -0.2))
  driver = _contact_driver(system)
  result = driver.run(
    base_point=_ramp((0.0,))[0],
    target_points=_ramp((0.5, 1.0)),
  )
  assert result.status is DriverStatus.COMPLETED
  assert result.statistics.committed_substep_count == 2
  state = driver.owner.accepted_physical().values
  # At lam=0.5 the disc bottom just touches the top edge: no contact force,
  # so the first substep commits the zero state; at lam=1.0 the top row is
  # engaged and pushed down.
  engaged = _evaluate(operator, state.reshape(8, 2), 1.0)
  assert evaluation_status(engaged) is EvaluationStatus.OK
  force = engaged.residual_values[0].values
  assert float(np.abs(force).max()) > 0.0
  assert float(state[11]) < 0.0  # node 6 (0.5, 1.0) moved down
  touching = _evaluate(operator, np.zeros((8, 2)), 0.5)
  np.testing.assert_array_equal(
    touching.residual_values[0].values,
    np.zeros((8, 2)),
  )

  # Disengage: the disc starts overlapping the top row and retreats past it;
  # with no external load the strip relaxes back to its reference state.
  retreat_system, retreat_operator = _contact_system((0.5, 1.2), (0.0, 0.4))
  retreat_driver = _contact_driver(retreat_system)
  retreat = retreat_driver.run(
    base_point=_ramp((0.0,))[0],
    target_points=_ramp((0.5, 1.0)),
  )
  assert retreat.status is DriverStatus.COMPLETED
  final_state = retreat_driver.owner.accepted_physical().values
  disengaged = _evaluate(retreat_operator, final_state.reshape(8, 2), 1.0)
  np.testing.assert_array_equal(
    disengaged.residual_values[0].values,
    np.zeros((8, 2)),
  )
  np.testing.assert_allclose(final_state, np.zeros(16), rtol=0.0, atol=1.0e-8)
