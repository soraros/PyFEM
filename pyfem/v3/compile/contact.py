"""Penalty contact compiler: node-vs-analytic-obstacle point operator.

This module ports the legacy ``pyfem.models.Contact`` capability onto the
point-entity seam: a frictionless penalty law between a declared set of
surface nodes and an analytic disc obstacle whose centre moves affine in a
bound program signal (``centre + lam * direction``), evaluated over the
declared node set with no contact search, exactly as legacy loops over all
nodes. The researcher-facing kernel receives the CURRENT node positions
(reference coordinates captured at compile plus the displacement port
values), the accepted state rows, the packed parameters, and the bound
signals.

The landed ``penalty_disc_kernel`` reproduces the legacy force law
``-penalty * overlap * n`` operation for operation, so converged solutions
match legacy; its tangent is the exact symmetric derivative
``penalty * (1 - radius/d) * I + penalty * (radius/d) * n x n`` of that force
— a documented improvement over legacy's appended ``penalty * n x n``, which
the M62 survey measured inexact by exactly ``overlap/d`` (1.01% at an overlap
of 0.01 radius). The exact tangent keeps ``symmetric=True`` honest; its
tangential block is negative while penetrating, which the driver's general
splu path handles without a definiteness assumption.

The operator is stateless: the active set is a pure function of the trial
point and the bound signal, so contact engage/disengage transitions across
substeps ride the driver's transaction discipline (trial evaluations stage
zero-width rows; commit/reject atomically advances or rewinds) with no state
slots and no observation channels.

The compile boundary mirrors the spring seam: a virgin-state probe over the
captured reference coordinates enforces array and status plumbing, and a
seeded finite-difference probe verifies the kernel tangent at engaged,
disengaged, and boundary-adjacent states placed deterministically around the
declared obstacle, clear of the non-differentiable kink at ``d = radius`` by
orders of magnitude more than the finite-difference step. The kernel is
dimension-generic, but compilation is pinned to two-component spaces by a
coded rejection until a 3D oracle exists (legacy's sphere branch is
unreachable through its ModelManager import — M62 finding 4). NOT-yet:
friction, finite sliding with contact search, Lagrange-multiplier or
augmented-Lagrange constraint enforcement, and gap/status observation
channels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NoReturn, Protocol

import numpy as np

from pyfem.v3.compile.diagnostics import (
  ModelCompilationDiagnostic,
  ModelCompilationError,
)
from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.identity import InstanceId, require_same_instance
from pyfem.v3.model.operator import (
  BalanceRole,
  ChannelRequest,
  CompilerConstructed,
  CouplingPolicy,
  EvaluationStatus,
  ImplementationIdentity,
  JacobianChannel,
  OperatorEvaluation,
  OperatorEvaluationInput,
  OperatorHeader,
  OperatorStateLayout,
  PortBinding,
  PortMode,
  ProgramSignalInput,
  ResidualChannel,
  SemanticId,
  SignalPortBinding,
  StateLifetime,
)
from pyfem.v3.model.provenance import CanonicalManifest, ContentFingerprint
from pyfem.v3.model.system import (
  CompiledSource,
  CompiledSystem,
  PointEntityBlock,
  SourceAttribution,
  SystemProvenance,
)
from pyfem.v3.spec.diagnostics import SourceContext

CONTACT_SYSTEM_EXTENSION_SCHEMA = "pyfem-v3-compiled-system-contact-extension-v1"
PENALTY_DISC_CONTACT_SCHEMA = "pyfem-v3-contact-penalty-disc-v1"
_FLOAT64_DTYPE = np.dtype(np.float64).str

# The tangent probe places seeded states on circles around the obstacle's base
# centre, one state per distance band (in units of the radius): engaged deep,
# disengaged, and engaged boundary-adjacent. The bands are part of the
# conformance contract: they keep every probed state clear of the
# non-differentiable kink at d = radius by far more than the finite-difference
# step, and the engaged bands sit where legacy's inexact tangent misses by
# overlap/d (one to eleven percent). The seed is recorded in every probe
# diagnostic so a failure replays bit-for-bit.
_TANGENT_PROBE_SEED = 20261009
_TANGENT_PROBE_BANDS = ((0.90, 0.99), (1.01, 1.10), (0.995, 0.999))
_TANGENT_PROBE_RTOL = 1.0e-4
_TANGENT_PROBE_STEP = float(np.cbrt(np.finfo(np.float64).eps))


def _new[ValueT](cls: type[ValueT], /, **fields: object) -> ValueT:
  value = object.__new__(cls)
  for name, field in fields.items():
    object.__setattr__(value, name, field)
  return value


def _fail(code: str, message: str, source: SourceContext) -> NoReturn:
  raise ModelCompilationError(
    (ModelCompilationDiagnostic(code=code, message=message, source=source),)
  )


def _source(value: SourceContext) -> CompiledSource:
  return _new(
    CompiledSource,
    source=value.source,
    line=value.line,
    column=value.column,
  )


def _signal_scalar(value: np.ndarray, label: str) -> None:
  if (
    type(value) is not np.ndarray
    or value.dtype != np.dtype(np.float64)
    or value.dtype.metadata is not None
    or value.shape != (1,)
    or not bool(np.isfinite(value).all())
  ):
    msg = f"contact {label} must be a plain finite one-element float64 array"
    raise TypeError(msg)


@dataclass(frozen=True, slots=True)
class ContactSignalPort:
  """One declared program-signal port of a contact network.

  ``port_id`` names the operator-local port evaluation inputs bind by;
  ``signal_id`` names the program signal the driver binds to the port, under
  the driver's two binding rules (identity binding when the id names a
  declared program coordinate, committed-increment derivation for a
  ``d<coordinate>`` id). ``derivative_coordinate_ids`` names the program
  coordinates whose ``d(signal)/d(coordinate)`` channels the driver forwards
  alongside the value.
  """

  port_id: str
  signal_id: str
  derivative_coordinate_ids: tuple[str, ...] = ()

  def __post_init__(self) -> None:
    if type(self.port_id) is not str or not self.port_id:
      msg = "contact signal port ids must be non-empty exact strings"
      raise TypeError(msg)
    if type(self.signal_id) is not str or not self.signal_id:
      msg = "contact signal port signal ids must be non-empty exact strings"
      raise TypeError(msg)
    if type(self.derivative_coordinate_ids) is not tuple or any(
      type(item) is not str or not item for item in self.derivative_coordinate_ids
    ):
      msg = "contact signal port derivative coordinates must be exact strings"
      raise TypeError(msg)
    if len(set(self.derivative_coordinate_ids)) != len(self.derivative_coordinate_ids):
      msg = "contact signal port derivative coordinates must be unique"
      raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ContactSignalDerivative:
  """One program-coordinate derivative channel of one bound signal port."""

  coordinate_id: str
  values: np.ndarray

  def __post_init__(self) -> None:
    if type(self.coordinate_id) is not str or not self.coordinate_id:
      msg = "contact signal derivative coordinates must be non-empty exact strings"
      raise TypeError(msg)
    _signal_scalar(self.values, "signal derivative values")


@dataclass(frozen=True, slots=True)
class ContactSignalInput:
  """One bound signal port forwarded to a contact kernel evaluation.

  The compiled operator builds these from validated ``ProgramSignalInput``
  values in declared port order; ``derivatives`` follows the port's declared
  ``derivative_coordinate_ids`` order exactly.
  """

  port_id: str
  values: np.ndarray
  derivatives: tuple[ContactSignalDerivative, ...]

  def __post_init__(self) -> None:
    if type(self.port_id) is not str or not self.port_id:
      msg = "contact signal input port ids must be non-empty exact strings"
      raise TypeError(msg)
    _signal_scalar(self.values, "signal values")
    if type(self.derivatives) is not tuple or any(
      type(item) is not ContactSignalDerivative for item in self.derivatives
    ):
      msg = "contact signal derivatives must be exact ContactSignalDerivative values"
      raise TypeError(msg)


@dataclass(frozen=True, slots=True)
class ContactKernelResult:
  """One batched local response of a contact law evaluation.

  ``force`` has shape ``(entity_count, dimension)`` and ``tangent``
  ``(entity_count, dimension, dimension)``; ``trial_rows`` has shape
  ``(entity_count, 0)`` — penalty contact is stateless, so the trial rows are
  always the accepted rows byte-equal. When ``status`` is not ``OK`` the
  operator discards the arrays and returns the accepted rows byte-equal, so
  kernels report expected numerical outcomes instead of raising.
  """

  force: np.ndarray
  tangent: np.ndarray
  trial_rows: np.ndarray
  status: EvaluationStatus


class ContactKernel(Protocol):
  """Researcher-authored batched law for node-vs-obstacle penalty contact."""

  def __call__(
    self,
    current_positions: np.ndarray,
    accepted_rows: np.ndarray,
    parameters: np.ndarray,
    signals: tuple[ContactSignalInput, ...],
  ) -> ContactKernelResult: ...


@dataclass(frozen=True, slots=True)
class ContactDeclaration:
  """Authored meaning for one penalty contact network against one obstacle.

  ``centre``, ``direction``, ``radius``, and ``penalty`` describe the analytic
  disc: at the bound signal value ``lam`` the obstacle centre sits at
  ``centre + lam * direction``. They are packed into the kernel's parameter
  array as ``(penalty, radius, *centre, *direction)`` and drive the seeded
  tangent probe's obstacle-relative state placement.
  """

  block_id: SemanticId
  space_id: SemanticId
  contact_ids: tuple[SemanticId, ...]
  node_ids: tuple[SemanticId, ...]
  centre: tuple[float, float]
  direction: tuple[float, float]
  radius: float
  penalty: float
  state_schema: str
  kernel_name: str
  kernel_version: str
  implementation_id: str
  kernel: ContactKernel
  source: SourceContext
  signal_ports: tuple[ContactSignalPort, ...] = ()

  def __post_init__(self) -> None:
    if type(self.block_id) not in (str, int, tuple):
      msg = "contact block id must be an exact semantic id"
      raise TypeError(msg)
    if (
      type(self.contact_ids) is not tuple
      or type(self.node_ids) is not tuple
      or not self.contact_ids
      or len(self.contact_ids) != len(self.node_ids)
      or len(set(self.contact_ids)) != len(self.contact_ids)
    ):
      msg = "contact declarations require paired unique contact and node id tuples"
      raise TypeError(msg)
    for label, pair in (("centre", self.centre), ("direction", self.direction)):
      if (
        type(pair) is not tuple
        or len(pair) != 2
        or any(type(value) is not float or not math.isfinite(value) for value in pair)
      ):
        msg = f"contact {label} must be a finite exact float pair"
        raise TypeError(msg)
    for label, value in (("radius", self.radius), ("penalty", self.penalty)):
      if type(value) is not float or not math.isfinite(value) or value <= 0.0:
        msg = f"contact {label} must be a positive finite exact float"
        raise TypeError(msg)
    if type(self.state_schema) is not str or not self.state_schema:
      msg = "contact state schema must be a non-empty exact string"
      raise TypeError(msg)
    for label in ("kernel_name", "kernel_version", "implementation_id"):
      value = getattr(self, label)
      if type(value) is not str or not value:
        msg = f"contact {label} must be a non-empty exact string"
        raise TypeError(msg)
    if not callable(self.kernel):
      msg = "contact kernel must be callable"
      raise TypeError(msg)
    if type(self.source) is not SourceContext:
      msg = "contact declarations require an exact SourceContext"
      raise TypeError(msg)
    if type(self.signal_ports) is not tuple or any(
      type(port) is not ContactSignalPort for port in self.signal_ports
    ):
      msg = "contact signal ports must be exact ContactSignalPort values"
      raise TypeError(msg)
    if len({port.port_id for port in self.signal_ports}) != len(self.signal_ports):
      msg = "contact signal port ids must be unique"
      raise ValueError(msg)


def penalty_disc_kernel(
  current_positions: np.ndarray,
  accepted_rows: np.ndarray,
  parameters: np.ndarray,
  signals: tuple[ContactSignalInput, ...],
) -> ContactKernelResult:
  """Frictionless penalty contact against a signal-driven disc obstacle.

  The single bound signal is the obstacle schedule coordinate ``lam``: the
  centre sits at ``centre + lam * direction``. Engaged entities
  (``overlap = radius - d > 0`` with ``d = |x - centre|``) carry exactly the
  legacy force law ``-penalty * overlap * n``, operation for operation, and
  the exact symmetric tangent of that force,
  ``penalty * (1 - radius/d) * I + penalty * (radius/d) * n x n``; disengaged
  entities carry zeros. The kernel is dimension-generic over the component
  axis; the compiler pins two-component spaces because no 3D oracle exists.
  """
  if len(signals) != 1:
    msg = "the penalty disc kernel binds exactly one obstacle schedule signal"
    raise TypeError(msg)
  dimension = current_positions.shape[1]
  penalty = parameters[0]
  radius = parameters[1]
  base_centre = parameters[2 : 2 + dimension]
  centre = base_centre + signals[0].values[0] * parameters[2 + dimension :]
  entity_count = len(current_positions)
  force = np.zeros((entity_count, dimension), dtype=np.float64)
  tangent = np.zeros((entity_count, dimension, dimension), dtype=np.float64)
  ds = current_positions - centre
  distance = np.sqrt(np.sum(ds * ds, axis=1))
  overlap = radius - distance
  engaged = overlap > 0.0
  if bool(engaged.any()):
    normal = ds[engaged] / distance[engaged, None]
    force[engaged] = (-penalty * overlap[engaged])[:, None] * normal
    tangent[engaged] = (penalty * (1.0 - radius / distance[engaged]))[
      :, None, None
    ] * np.eye(dimension)[None, :, :] + (penalty * (radius / distance[engaged]))[
      :, None, None
    ] * np.einsum("ei,ej->eij", normal, normal)
  return ContactKernelResult(
    force=force,
    tangent=tangent,
    trial_rows=np.array(accepted_rows, copy=True),
    status=EvaluationStatus.OK,
  )


def penalty_disc_declaration(
  *,
  block_id: SemanticId,
  space_id: SemanticId,
  contact_ids: tuple[SemanticId, ...],
  node_ids: tuple[SemanticId, ...],
  centre: tuple[float, float],
  direction: tuple[float, float],
  radius: float,
  penalty: float,
  signal_port: ContactSignalPort,
  source: SourceContext,
) -> ContactDeclaration:
  """Build a validated declaration for the landed penalty disc law."""
  if type(signal_port) is not ContactSignalPort:
    msg = "the penalty disc law binds exactly one obstacle schedule signal port"
    raise TypeError(msg)
  return ContactDeclaration(
    block_id=block_id,
    space_id=space_id,
    contact_ids=contact_ids,
    node_ids=node_ids,
    centre=centre,
    direction=direction,
    radius=radius,
    penalty=penalty,
    state_schema=PENALTY_DISC_CONTACT_SCHEMA,
    kernel_name="penalty-disc-contact",
    kernel_version="1",
    implementation_id=PENALTY_DISC_CONTACT_SCHEMA,
    kernel=penalty_disc_kernel,
    source=source,
    signal_ports=(signal_port,),
  )


def _validated_kernel_arrays(
  result: object,
  *,
  entity_count: int,
  row_width: int,
  dimension: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, EvaluationStatus]:
  if type(result) is not ContactKernelResult:
    msg = "contact kernels must return an exact ContactKernelResult"
    raise TypeError(msg)
  status = result.status
  if type(status) is not EvaluationStatus:
    msg = "contact kernel status must be an exact EvaluationStatus"
    raise TypeError(msg)
  arrays = (result.force, result.tangent, result.trial_rows)
  shapes = (
    (entity_count, dimension),
    (entity_count, dimension, dimension),
    (entity_count, row_width),
  )
  for array, shape in zip(arrays, shapes, strict=True):
    if (
      type(array) is not np.ndarray
      or array.dtype != np.dtype(np.float64)
      or array.dtype.metadata is not None
      or array.shape != shape
    ):
      msg = "contact kernel arrays must match the declared batched float64 shapes"
      raise TypeError(msg)
  if status is EvaluationStatus.OK and not all(
    bool(np.isfinite(array).all()) for array in arrays
  ):
    msg = "contact kernel arrays must be finite for a successful evaluation"
    raise TypeError(msg)
  return result.force, result.tangent, result.trial_rows, status


def _validated_contact_signals(
  ports: tuple[SignalPortBinding, ...],
  signals: tuple[ProgramSignalInput, ...],
) -> tuple[ContactSignalInput, ...]:
  """Bind evaluation signal inputs exactly onto the declared signal ports.

  Every declared port must be bound exactly once and every input must name a
  declared port; derivative channels must follow the port's declared
  ``derivative_coordinate_ids`` order. The forwarded carriers preserve
  declaration order so kernels read a deterministic layout.
  """
  by_port: dict[str, ProgramSignalInput] = {}
  for signal in signals:
    if signal.port_id in by_port:
      msg = "contact evaluation received a duplicate program signal port"
      raise ValueError(msg)
    by_port[signal.port_id] = signal
  declared = {port.port_id: port for port in ports}
  for port_id in by_port:
    if port_id not in declared:
      msg = "contact evaluation received an undeclared program signal port"
      raise ValueError(msg)
  kernel_signals: list[ContactSignalInput] = []
  for port in ports:
    signal = by_port.get(port.port_id)
    if signal is None:
      msg = "contact evaluation is missing a declared program signal port"
      raise ValueError(msg)
    if (
      tuple(item.coordinate_id for item in signal.derivatives)
      != port.derivative_coordinate_ids
    ):
      msg = "contact signal derivatives must match the declared coordinates"
      raise ValueError(msg)
    kernel_signals.append(
      ContactSignalInput(
        port_id=port.port_id,
        values=signal.values.values,
        derivatives=tuple(
          ContactSignalDerivative(
            coordinate_id=item.coordinate_id,
            values=item.values.values,
          )
          for item in signal.derivatives
        ),
      )
    )
  return tuple(kernel_signals)


def _probe_signals(
  ports: tuple[ContactSignalPort, ...],
) -> tuple[ContactSignalInput, ...]:
  """Zero-valued compile-probe signals with identity-style derivative deltas.

  Mirrors the landed stateful-continuum precedent: a zero signal with a
  Kronecker delta on the bound coordinate is a legal input under either
  driver binding rule, so a signal-consuming kernel sees its schedule origin
  at the compile boundary. The carriers are read-only, exactly as at runtime.
  """
  carriers: list[ContactSignalInput] = []
  for port in ports:
    values = np.zeros(1, dtype=np.float64)
    values.setflags(write=False)
    derivatives: list[ContactSignalDerivative] = []
    for coordinate in port.derivative_coordinate_ids:
      derivative = np.array(
        [1.0 if coordinate == port.signal_id else 0.0],
        dtype=np.float64,
      )
      derivative.setflags(write=False)
      derivatives.append(
        ContactSignalDerivative(coordinate_id=coordinate, values=derivative)
      )
    carriers.append(
      ContactSignalInput(
        port_id=port.port_id,
        values=values,
        derivatives=tuple(derivatives),
      )
    )
  return tuple(carriers)


def _probe_kernel_tangent(
  kernel: ContactKernel,
  parameters: np.ndarray,
  *,
  entity_count: int,
  row_width: int,
  centre: tuple[float, float],
  radius: float,
  signals: tuple[ContactSignalInput, ...],
  source: SourceContext,
) -> None:
  """Central-difference the kernel tangent around the declared obstacle.

  Every probed state places ``entity_count`` seeded positions on a circle
  whose radius sweeps one of the fixed bands — engaged deep, disengaged, and
  engaged boundary-adjacent — so a tangent wrong in any regime fails
  compilation with a coded diagnostic instead of evaluating silently wrong.
  States the kernel rejects with a typed status carry no channels to verify
  and are skipped. The draws come from one fixed-seed generator and the seed
  is recorded in every diagnostic, so a failure replays bit-for-bit.
  """
  generator = np.random.default_rng(_TANGENT_PROBE_SEED)
  base_centre = np.array(centre, dtype=np.float64)
  for state_index, band in enumerate(_TANGENT_PROBE_BANDS):
    angles = generator.uniform(0.0, 2.0 * math.pi, entity_count)
    distances = radius * generator.uniform(band[0], band[1], entity_count)
    positions = base_centre[None, :] + distances[:, None] * np.stack(
      (np.cos(angles), np.sin(angles)),
      axis=1,
    )
    accepted_rows = np.zeros((entity_count, row_width), dtype=np.float64)
    # The probe mirrors runtime input mutability exactly, exactly as at the
    # virgin state: every array handed to the kernel is read-only.
    positions.setflags(write=False)
    accepted_rows.setflags(write=False)
    try:
      probed = kernel(positions, accepted_rows, parameters, signals)
    except Exception:
      _fail(
        "kernel-probe-failed",
        "contact kernel failed its seeded nonzero-state compile probe "
        f"(seed {_TANGENT_PROBE_SEED}, state {state_index})",
        source,
      )
    _, tangent, _, status = _validated_kernel_arrays(
      probed,
      entity_count=entity_count,
      row_width=row_width,
      dimension=2,
    )
    if status is not EvaluationStatus.OK:
      continue
    for entity_index in range(entity_count):
      for component in range(2):
        step = _TANGENT_PROBE_STEP * max(
          1.0,
          abs(float(positions[entity_index, component])),
        )
        plus = np.array(positions, copy=True)
        minus = np.array(positions, copy=True)
        plus[entity_index, component] += step
        minus[entity_index, component] -= step
        plus.setflags(write=False)
        minus.setflags(write=False)
        try:
          plus_result = kernel(plus, accepted_rows, parameters, signals)
          minus_result = kernel(minus, accepted_rows, parameters, signals)
        except Exception:
          _fail(
            "kernel-probe-failed",
            "contact kernel failed its seeded nonzero-state compile probe "
            f"(seed {_TANGENT_PROBE_SEED}, state {state_index}, entity "
            f"{entity_index}, component {component})",
            source,
          )
        plus_force, _, _, plus_status = _validated_kernel_arrays(
          plus_result,
          entity_count=entity_count,
          row_width=row_width,
          dimension=2,
        )
        minus_force, _, _, minus_status = _validated_kernel_arrays(
          minus_result,
          entity_count=entity_count,
          row_width=row_width,
          dimension=2,
        )
        if (
          plus_status is not EvaluationStatus.OK
          or minus_status is not EvaluationStatus.OK
        ):
          continue
        difference = (plus_force[entity_index] - minus_force[entity_index]) / (
          2.0 * step
        )
        column = tangent[entity_index, :, component]
        scale = max(
          1.0,
          float(np.abs(column).max()),
          float(np.abs(difference).max()),
        )
        mismatch = float(np.abs(column - difference).max())
        if mismatch > _TANGENT_PROBE_RTOL * scale:
          _fail(
            "inconsistent-kernel-tangent",
            "contact kernel tangent contradicts a central finite difference of "
            "its force at a seeded nonzero accepted state (seed "
            f"{_TANGENT_PROBE_SEED}, state {state_index}, entity "
            f"{entity_index}, component {component}): kernel tangent column "
            f"{column.tolist()} vs finite difference {difference.tolist()}",
            source,
          )


@dataclass(frozen=True, slots=True, eq=False, init=False)
class ContactPayload(CompilerConstructed):
  parameters: FinalizedArray


@dataclass(frozen=True, slots=True, eq=False, init=False)
class PenaltyContactOperator(CompilerConstructed):
  """Compiled stateless penalty contact operator over a declared node set."""

  header: OperatorHeader
  contact_block: PointEntityBlock
  payload: ContactPayload
  content_manifest: CanonicalManifest
  kernel: ContactKernel
  system_instance: InstanceId

  def evaluate(
    self,
    inputs: OperatorEvaluationInput,
  ) -> OperatorEvaluation:
    """Evaluate the contact network from accepted state with typed outcomes."""
    if type(inputs) is not OperatorEvaluationInput:
      msg = "contact evaluation requires an exact immutable evaluation input"
      raise TypeError(msg)
    if type(inputs.port_values) is not tuple or len(inputs.port_values) != 1:
      msg = "contact evaluation requires exactly one displacement port batch"
      raise TypeError(msg)
    values = inputs.port_values[0].values
    expected = self.header.ports[0].coefficient_map.values.shape
    if (
      values.dtype != np.dtype(np.float64)
      or values.dtype.metadata is not None
      or values.shape != expected
      or not bool(np.isfinite(values).all())
    ):
      msg = "contact displacement port values must be a finite float64 batch"
      raise TypeError(msg)
    layout = self.header.state_layout
    accepted_state = inputs.accepted_state.values
    if (
      accepted_state.dtype != np.dtype(np.float64)
      or accepted_state.dtype.metadata is not None
      or accepted_state.shape != layout.row_shape
      or not bool(np.isfinite(accepted_state).all())
    ):
      msg = "contact accepted state must match the compiled state layout"
      raise TypeError(msg)
    signal_ports = self.header.signal_ports
    if signal_ports:
      kernel_signals = _validated_contact_signals(signal_ports, inputs.signals)
    else:
      if inputs.signals:
        msg = "contact model operator does not accept program signal inputs"
        raise ValueError(msg)
      kernel_signals = ()
    residual_ids = tuple(item.channel_id for item in self.header.residual_channels)
    jacobian_ids = tuple(item.channel_id for item in self.header.jacobian_channels)
    derivative_ids = tuple(
      item.channel_id for item in getattr(self.header, "derivative_channels", ())
    )
    request = inputs.request
    derivative_request = request.derivative_channel_ids
    if (
      type(request) is not ChannelRequest
      or len(set(request.residual_channel_ids)) != len(request.residual_channel_ids)
      or len(set(request.jacobian_channel_ids)) != len(request.jacobian_channel_ids)
      or len(set(derivative_request)) != len(derivative_request)
      or not set(request.residual_channel_ids).issubset(residual_ids)
      or not set(request.jacobian_channel_ids).issubset(jacobian_ids)
      or not set(derivative_request).issubset(derivative_ids)
    ):
      msg = "contact evaluation request contains an unavailable or duplicate channel"
      raise ValueError(msg)

    # Current positions = compile-time reference coordinates + displacements:
    # the legacy ``crd += state`` accumulation, one add per component.
    current_positions = np.array(
      self.contact_block.reference_coordinates.values + values,
      dtype=np.float64,
    )
    if not bool(np.isfinite(current_positions).all()):
      msg = "contact current positions must be finite float64 values"
      raise TypeError(msg)
    current_positions.setflags(write=False)
    force, tangent, trial_rows, status = _validated_kernel_arrays(
      self.kernel(
        current_positions,
        accepted_state,
        self.payload.parameters.values,
        kernel_signals,
      ),
      entity_count=layout.entity_count,
      row_width=layout.row_width,
      dimension=2,
    )
    if status is not EvaluationStatus.OK:
      return _new(
        OperatorEvaluation,
        residual_values=(),
        jacobian_values=(),
        trial_state=FinalizedArray(accepted_state, dtype=np.float64),
        status=status,
      )
    residual_values = (
      (FinalizedArray(force, dtype=np.float64),) if request.residual_channel_ids else ()
    )
    jacobian_values = (
      (FinalizedArray(tangent, dtype=np.float64),)
      if request.jacobian_channel_ids
      else ()
    )
    return _new(
      OperatorEvaluation,
      residual_values=residual_values,
      jacobian_values=jacobian_values,
      trial_state=FinalizedArray(trial_rows, dtype=np.float64),
      status=status,
    )


def compile_contact_operator(
  system: CompiledSystem,
  declaration: ContactDeclaration,
) -> tuple[PointEntityBlock, PenaltyContactOperator]:
  """Compile one authored contact network against an existing compiled system."""
  if type(system) is not CompiledSystem:
    msg = "contact compilation requires an exact CompiledSystem"
    raise TypeError(msg)
  if type(declaration) is not ContactDeclaration:
    msg = "contact compilation requires an exact ContactDeclaration"
    raise TypeError(msg)
  source = declaration.source
  space = next(
    (item for item in system.spaces if item.space_id == declaration.space_id),
    None,
  )
  if space is None:
    _fail(
      "unknown-contact-space",
      "contact network references a space the compiled system does not have",
      source,
    )
  if len(space.components) != 2:
    _fail(
      "unsupported-contact-space",
      "penalty contact requires a two-component displacement space: the "
      "kernel is dimension-generic, but no three-dimensional oracle exists "
      "(the legacy sphere branch is unreachable), so 3D compilation is "
      "rejected until one lands",
      source,
    )
  support = next(
    (item for item in system.point_blocks if item.block_id == space.support_block_id),
    None,
  )
  if support is None:
    _fail(
      "unknown-contact-support-block",
      "contact space support block is absent from the compiled system",
      source,
    )
  node_dense = {node_id: index for index, node_id in enumerate(support.entity_ids)}
  node_indices: list[int] = []
  for node_id in declaration.node_ids:
    index = node_dense.get(node_id)
    if index is None:
      _fail(
        "unknown-contact-support-node",
        "contact network references a support node the compiled system does not have",
        source,
      )
    node_indices.append(index)

  entity_count = len(declaration.contact_ids)
  row_width = 0
  parameters = FinalizedArray(
    (
      declaration.penalty,
      declaration.radius,
      *declaration.centre,
      *declaration.direction,
    ),
    dtype=np.float64,
  )
  probe_signals = _probe_signals(declaration.signal_ports)
  # The probe must mirror runtime input mutability exactly: evaluate hands the
  # kernel read-only arrays, so the probe does too, or an input-mutating kernel
  # would compile and fail untyped at first evaluation.
  probe_positions = np.array(
    support.reference_coordinates.values[node_indices],
    dtype=np.float64,
  )
  probe_rows = np.zeros((entity_count, row_width), dtype=np.float64)
  probe_positions.setflags(write=False)
  probe_rows.setflags(write=False)
  try:
    probe_result = declaration.kernel(
      probe_positions,
      probe_rows,
      parameters.values,
      probe_signals,
    )
  except Exception:
    _fail(
      "kernel-probe-failed",
      "contact kernel failed its virgin-state compile probe",
      source,
    )
  probe = _validated_kernel_arrays(
    probe_result,
    entity_count=entity_count,
    row_width=row_width,
    dimension=2,
  )
  if probe[3] is not EvaluationStatus.OK:
    _fail(
      "invalid-kernel-probe",
      "contact kernel must evaluate its virgin reference state successfully",
      source,
    )
  _probe_kernel_tangent(
    declaration.kernel,
    parameters.values,
    entity_count=entity_count,
    row_width=row_width,
    centre=declaration.centre,
    radius=declaration.radius,
    signals=probe_signals,
    source=source,
  )

  contact_block = _new(
    PointEntityBlock,
    block_id=declaration.block_id,
    entity_ids=declaration.contact_ids,
    sources=tuple(_source(source) for _ in declaration.contact_ids),
    reference_coordinates=FinalizedArray(
      support.reference_coordinates.values[node_indices],
      dtype=np.float64,
    ),
  )
  block_id = declaration.block_id, declaration.state_schema
  state_layout = _new(
    OperatorStateLayout,
    schema=declaration.state_schema,
    block_id=block_id,
    entity_count=entity_count,
    slots=(),
    entity_offsets=FinalizedArray(
      np.zeros(entity_count + 1, dtype=space.coefficient_map.values.dtype),
      dtype=space.coefficient_map.values.dtype,
    ),
    row_width=row_width,
    dtype=_FLOAT64_DTYPE,
    lifetime=StateLifetime.ACCEPTED_TRIAL,
  )
  port = _new(
    PortBinding,
    port_id="displacement",
    space_id=space.space_id,
    mode=PortMode.COEFFICIENTS,
    coefficient_map=FinalizedArray(
      space.coefficient_map.values[node_indices],
      dtype=space.coefficient_map.values.dtype,
    ),
  )
  residual_channel = _new(
    ResidualChannel,
    channel_id="contact-force",
    target_port_id=port.port_id,
    balance_role=BalanceRole.INTERNAL,
    linear=False,
  )
  jacobian_channel = _new(
    JacobianChannel,
    channel_id="contact-tangent",
    residual_channel_id=residual_channel.channel_id,
    target_port_id=port.port_id,
    source_port_id=port.port_id,
    balance_role=BalanceRole.INTERNAL,
    linear=False,
    symmetric=True,
  )
  header = _new(
    OperatorHeader,
    block_id=block_id,
    entity_block_id=contact_block.block_id,
    implementations=(
      _new(
        ImplementationIdentity,
        kind="constitutive-kernel",
        name=declaration.kernel_name,
        version=declaration.kernel_version,
        implementation_id=declaration.implementation_id,
      ),
    ),
    ports=(port,),
    signal_ports=tuple(
      _new(
        SignalPortBinding,
        port_id=signal_port.port_id,
        signal_id=signal_port.signal_id,
        derivative_coordinate_ids=signal_port.derivative_coordinate_ids,
      )
      for signal_port in declaration.signal_ports
    ),
    residual_channels=(residual_channel,),
    jacobian_channels=(jacobian_channel,),
    state_layout=state_layout,
    coupling_policy=CouplingPolicy.FIXED,
  )
  payload = _new(ContactPayload, parameters=parameters)
  manifest_content: dict[str, object] = {
    "block_id": block_id,
    "entity_block_id": contact_block.block_id,
    "entity_ids": contact_block.entity_ids,
    "support_node_ids": declaration.node_ids,
    "implementations": [
      {
        "kind": "constitutive-kernel",
        "name": declaration.kernel_name,
        "version": declaration.kernel_version,
        "implementation_id": declaration.implementation_id,
      }
    ],
    "obstacle": {
      "centre": declaration.centre,
      "direction": declaration.direction,
      "radius": declaration.radius,
      "penalty": declaration.penalty,
    },
    "port": {
      "port_id": port.port_id,
      "space_id": port.space_id,
      "coefficient_map": port.coefficient_map.values,
    },
    "channels": ["contact-force", "contact-tangent"],
    "state": {
      "schema": state_layout.schema,
      "row_width": row_width,
      "slots": [],
      "entity_offsets": state_layout.entity_offsets.values,
    },
    "payload": {"parameters": payload.parameters.values},
  }
  # Declarations without signal ports keep the key out of the manifest, the
  # landed signal-versioning precedent.
  if declaration.signal_ports:
    manifest_content["signal_ports"] = [
      {
        "port_id": signal_port.port_id,
        "signal_id": signal_port.signal_id,
        "derivative_coordinate_ids": list(signal_port.derivative_coordinate_ids),
      }
      for signal_port in declaration.signal_ports
    ]
  manifest = CanonicalManifest(manifest_content)
  operator = _new(
    PenaltyContactOperator,
    header=header,
    contact_block=contact_block,
    payload=payload,
    content_manifest=manifest,
    kernel=declaration.kernel,
    system_instance=system.instance_id,
  )
  return contact_block, operator


def compose_contact_system(
  base: CompiledSystem,
  contact_block: PointEntityBlock,
  contact_operator: PenaltyContactOperator,
) -> CompiledSystem:
  """Compose one compiled contact network into a base compiled system."""
  if type(base) is not CompiledSystem:
    msg = "contact composition requires an exact base CompiledSystem"
    raise TypeError(msg)
  if type(contact_block) is not PointEntityBlock:
    msg = "contact composition requires an exact contact PointEntityBlock"
    raise TypeError(msg)
  if type(contact_operator) is not PenaltyContactOperator:
    msg = "contact composition requires an exact PenaltyContactOperator"
    raise TypeError(msg)
  require_same_instance(
    contact_operator.system_instance,
    base.instance_id,
    context="contact system composition",
  )
  if contact_operator.contact_block is not contact_block:
    msg = "contact operator must bind the exact composed contact block"
    raise ValueError(msg)
  block_collision = any(
    block.block_id == contact_block.block_id
    for block in (*base.point_blocks, *base.entity_blocks)
  )
  if block_collision:
    msg = "contact block id collides with an existing compiled entity block"
    raise ValueError(msg)
  if any(
    operator.header.block_id == contact_operator.header.block_id
    for operator in base.operators
  ):
    msg = "contact operator block id collides with an existing compiled operator"
    raise ValueError(msg)
  space_ids = {space.space_id for space in base.spaces}
  if contact_operator.header.ports[0].space_id not in space_ids:
    msg = "contact operator port references a space outside the base system"
    raise ValueError(msg)

  contact_attribution = (
    _new(
      SourceAttribution,
      kind="entity_block",
      semantic_id=contact_block.block_id,
      source=contact_block.sources[0],
    ),
    *(
      _new(
        SourceAttribution,
        kind="contact",
        semantic_id=entity_id,
        source=source,
      )
      for entity_id, source in zip(
        contact_block.entity_ids,
        contact_block.sources,
        strict=True,
      )
    ),
  )
  attributions = (*base.source_attribution, *contact_attribution)
  manifest = CanonicalManifest(
    {
      "schema": CONTACT_SYSTEM_EXTENSION_SCHEMA,
      "base_system": base.provenance.manifest,
      "contact_point_block": {
        "block_id": contact_block.block_id,
        "entity_ids": contact_block.entity_ids,
        "reference_coordinates": contact_block.reference_coordinates.values,
      },
      "contact_operator": contact_operator.content_manifest,
      "source_attribution": [
        {
          "kind": record.kind,
          "semantic_id": record.semantic_id,
          "source": {
            "source": record.source.source,
            "line": record.source.line,
            "column": record.source.column,
          },
        }
        for record in contact_attribution
      ],
    }
  )
  provenance = _new(
    SystemProvenance,
    schema=CONTACT_SYSTEM_EXTENSION_SCHEMA,
    manifest=manifest,
    registry_fingerprint=base.provenance.registry_fingerprint,
    floating_dtype=base.provenance.floating_dtype,
    dense_index_dtype=base.provenance.dense_index_dtype,
    geometry_relative_tolerance=base.provenance.geometry_relative_tolerance,
  )
  return _new(
    CompiledSystem,
    instance_id=InstanceId(),
    content_fingerprint=ContentFingerprint.from_manifest(manifest),
    provenance=provenance,
    registry_snapshot=base.registry_snapshot,
    point_blocks=(*base.point_blocks, contact_block),
    entity_blocks=base.entity_blocks,
    spaces=base.spaces,
    operators=(*base.operators, contact_operator),
    source_attribution=attributions,
  )
