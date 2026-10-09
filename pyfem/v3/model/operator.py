"""Generic compiled finite-element operator contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.provenance import CanonicalManifest

type SemanticId = str | int | tuple[str | int, ...]


def _exact_tuple(value: object, item_type: type[object], label: str) -> None:
  if type(value) is not tuple or any(type(item) is not item_type for item in value):
    msg = f"{label} must be an exact immutable tuple of {item_type.__name__}"
    raise TypeError(msg)


class CompilerConstructed:
  __slots__ = ()

  def __init__(self, *_args: object, **_kwargs: object) -> None:
    msg = "trusted compiled carriers are constructed only by their compiler"
    raise TypeError(msg)

  def __copy__(self) -> None:
    self._deny_reconstruction()

  def __deepcopy__(self, _memo: object) -> None:
    self._deny_reconstruction()

  def __reduce_ex__(self, _protocol: int) -> None:
    self._deny_reconstruction()

  @staticmethod
  def _deny_reconstruction() -> None:
    msg = "trusted compiled carriers cannot be reconstructed"
    raise TypeError(msg)


class BalanceRole(Enum):
  INTERNAL = "internal"
  EXTERNAL = "external"
  CONSTRAINT = "constraint"


class PortMode(Enum):
  COEFFICIENTS = "coefficients"


class StateLifetime(Enum):
  ACCEPTED_TRIAL = "accepted-trial"


class EvaluationStatus(Enum):
  """Typed classification of expected operator evaluation outcomes.

  Expected numerical outcomes are reported through these status values, never
  through exceptions. ``OK`` marks a usable evaluation. ``REJECT_ITERATION``
  marks a trial the driver may retry from the same accepted state with a better
  iterate (for example a local solve that exhausted its iteration budget).
  ``REJECT_STEP`` marks a trial that cannot succeed at the current step size, so
  the schedule must cut back and restart from the same accepted generation.
  Whenever the status is not ``OK`` the returned trial state must be byte-equal
  to the accepted state and the channel values may be empty. Contract violations
  (wrong shapes, dtypes, or unknown channels) remain ``TypeError`` exceptions.
  """

  OK = "ok"
  REJECT_ITERATION = "reject-iteration"
  REJECT_STEP = "reject-step"


class CouplingPolicy(Enum):
  FIXED = "fixed"


@dataclass(frozen=True, slots=True, eq=False, init=False)
class ImplementationIdentity(CompilerConstructed):
  kind: str
  name: str
  version: str
  implementation_id: str


@dataclass(frozen=True, slots=True, eq=False, init=False)
class PortBinding(CompilerConstructed):
  port_id: str
  space_id: SemanticId
  mode: PortMode
  coefficient_map: FinalizedArray


@dataclass(frozen=True, slots=True, eq=False, init=False)
class SignalPortBinding(CompilerConstructed):
  """One declared program-signal port of a compiled operator.

  ``port_id`` is the operator-local name evaluation inputs bind by;
  ``signal_id`` names the program signal the driver binds to the port under
  one of two binding rules: identity binding, where the signal id exactly
  names a declared program coordinate, and committed-increment derivation,
  where a ``d<coordinate>`` signal id binds the base coordinate's increment
  over the previous committed point (``dtime`` derives the time increment of
  rate-form laws). ``derivative_coordinate_ids`` names the program coordinates
  whose ``d(signal)/d(coordinate)`` channels the driver forwards alongside the
  value — the Kronecker delta on the bound coordinate under identity binding
  and on the base coordinate under increment derivation. Operators with no
  declared ports reject every signal input, and an evaluation input naming an
  undeclared port is rejected fail-closed: schedule values reach operators
  exclusively through declared ports.
  """

  port_id: str
  signal_id: SemanticId
  derivative_coordinate_ids: tuple[SemanticId, ...]


@dataclass(frozen=True, slots=True, eq=False, init=False)
class ResidualChannel(CompilerConstructed):
  channel_id: str
  target_port_id: str
  balance_role: BalanceRole
  linear: bool


@dataclass(frozen=True, slots=True, eq=False, init=False)
class ParameterBinding(CompilerConstructed):
  """One declared differentiable parameter of a compiled operator.

  ``parameter_id`` keys the parameter at spec level — for the stateful
  continuum slice it is the material law's declared ``parameter_names``
  entry. The header declaration order fixes the stacking order of
  per-parameter derivative batches everywhere they appear (kernel results
  and evaluation derivative values alike).
  """

  parameter_id: SemanticId


@dataclass(frozen=True, slots=True, eq=False, init=False)
class ResidualDerivativeChannel(CompilerConstructed):
  """One declared ``d(residual)/d(parameter)`` channel of a compiled operator.

  ``channel_id`` is the operator-local name evaluation requests bind by;
  ``residual_channel_id`` names the residual channel being differentiated
  and ``parameter_id`` the declared ``ParameterBinding`` the derivative is
  taken with respect to. Derivative channel values follow the referenced
  residual channel's element-batch layout exactly, so assembly machinery
  built for residual values scatters them unchanged. The derivative is taken
  at the evaluation point with the accepted state held fixed — it is a pure
  function of the committed state, the port values, and the parameters, and
  never carries trial-state increments. Operators with no declared
  derivative channels reject every derivative request fail-closed.
  """

  channel_id: str
  residual_channel_id: str
  parameter_id: SemanticId


@dataclass(frozen=True, slots=True, eq=False, init=False)
class JacobianChannel(CompilerConstructed):
  channel_id: str
  residual_channel_id: str
  target_port_id: str
  source_port_id: str
  balance_role: BalanceRole
  linear: bool
  symmetric: bool


@dataclass(frozen=True, slots=True, eq=False, init=False)
class OperatorStateSlot(CompilerConstructed):
  """One named contiguous slice of an operator state row.

  ``annotation`` is optional slot metadata (for example ``envelope-max`` or
  ``monotone-nondecreasing``); compilers predating annotations leave it unset
  and consumers read it as ``None``.
  """

  name: str
  width: int
  dtype: str
  lifetime: StateLifetime
  annotation: str | None


@dataclass(frozen=True, slots=True, eq=False, init=False)
class OperatorStateLayout(CompilerConstructed):
  """The accepted-trial state-row ABI of one operator block.

  ``initial_rows`` is optional: when the compiler sets it, state owners seed
  the accepted buffer with those rows at construction instead of zeros. Layouts
  predating the field carry no ``initial_rows`` attribute and consumers read
  them as ``None`` (zero initialization).
  """

  schema: str
  block_id: SemanticId
  entity_count: int
  slots: tuple[OperatorStateSlot, ...]
  entity_offsets: FinalizedArray
  row_width: int
  dtype: str
  lifetime: StateLifetime
  initial_rows: FinalizedArray | None

  @property
  def row_shape(self) -> tuple[int, int]:
    return self.entity_count, self.row_width


@dataclass(frozen=True, slots=True, eq=False, init=False)
class OperatorHeader(CompilerConstructed):
  """The compiled channel and state declaration of one operator block.

  ``parameters`` and ``derivative_channels`` are optional: compilers
  predating the parameter-derivative channel (and compilers of channel-free
  operators) leave both unset, and consumers read them as empty tuples —
  the ``OperatorStateLayout.initial_rows`` precedent. A header carrying no
  derivative channels admits no derivative channel requests.
  """

  block_id: SemanticId
  entity_block_id: SemanticId
  implementations: tuple[ImplementationIdentity, ...]
  ports: tuple[PortBinding, ...]
  signal_ports: tuple[SignalPortBinding, ...]
  residual_channels: tuple[ResidualChannel, ...]
  jacobian_channels: tuple[JacobianChannel, ...]
  state_layout: OperatorStateLayout
  coupling_policy: CouplingPolicy
  parameters: tuple[ParameterBinding, ...]
  derivative_channels: tuple[ResidualDerivativeChannel, ...]


@dataclass(frozen=True, slots=True, eq=False)
class SignalDerivativeInput:
  """One ``d(signal)/d(coordinate_id)`` channel bound at the program point.

  The landed ABI revision's signals are scalar: ``values`` carries exactly one
  float64. A signal input's derivatives follow the declaring port's
  ``derivative_coordinate_ids`` order exactly.
  """

  coordinate_id: SemanticId
  values: FinalizedArray

  def __post_init__(self) -> None:
    if type(self.values) is not FinalizedArray:
      raise TypeError("signal derivative values must be an exact FinalizedArray")


@dataclass(frozen=True, slots=True, eq=False)
class ProgramSignalInput:
  """One bound program signal forwarded to an operator evaluation.

  ``port_id`` must name a declared ``SignalPortBinding`` of the operator;
  ``values`` carries the bound signal value at the current program point (one
  float64 in this scalar-signal ABI revision) and ``derivatives`` the declared
  coordinate-derivative channels.
  """

  port_id: str
  values: FinalizedArray
  derivatives: tuple[SignalDerivativeInput, ...]

  def __post_init__(self) -> None:
    if type(self.port_id) is not str or type(self.values) is not FinalizedArray:
      raise TypeError("program signal input requires an exact port and values")
    _exact_tuple(self.derivatives, SignalDerivativeInput, "signal derivatives")


@dataclass(frozen=True, slots=True, eq=False)
class ChannelRequest:
  """The per-evaluation channel selection of one operator call.

  ``derivative_channel_ids`` names the requested ``d(residual)/d(parameter)``
  channels (``ResidualDerivativeChannel`` declarations of the operator
  header); it defaults to empty, so callers predating the parameter
  derivative channel construct requests unchanged.
  """

  residual_channel_ids: tuple[str, ...]
  jacobian_channel_ids: tuple[str, ...]
  derivative_channel_ids: tuple[str, ...] = ()

  def __post_init__(self) -> None:
    _exact_tuple(self.residual_channel_ids, str, "residual channel IDs")
    _exact_tuple(self.jacobian_channel_ids, str, "Jacobian channel IDs")
    _exact_tuple(self.derivative_channel_ids, str, "derivative channel IDs")


@dataclass(frozen=True, slots=True, eq=False)
class OperatorEvaluationInput:
  port_values: tuple[FinalizedArray, ...]
  accepted_state: FinalizedArray
  signals: tuple[ProgramSignalInput, ...]
  request: ChannelRequest

  def __post_init__(self) -> None:
    _exact_tuple(self.port_values, FinalizedArray, "operator port values")
    _exact_tuple(self.signals, ProgramSignalInput, "program signal inputs")
    if (
      type(self.accepted_state) is not FinalizedArray
      or type(self.request) is not ChannelRequest
    ):
      raise TypeError("operator input requires exact state and channel request")


@dataclass(frozen=True, slots=True, eq=False, init=False)
class OperatorEvaluation(CompilerConstructed):
  """The typed result of one operator evaluation.

  ``derivative_values`` carries one ``FinalizedArray`` per requested
  derivative channel in the header's declaration order, mirroring
  ``residual_values`` semantics; each entry follows the referenced residual
  channel's element-batch layout. Evaluators predating the parameter
  derivative channel construct evaluations without the slot — read them as
  empty via ``evaluation_derivative_values`` (the ``status`` precedent).
  """

  residual_values: tuple[FinalizedArray, ...]
  jacobian_values: tuple[FinalizedArray, ...]
  trial_state: FinalizedArray
  status: EvaluationStatus
  derivative_values: tuple[FinalizedArray, ...]


def evaluation_status(evaluation: OperatorEvaluation) -> EvaluationStatus:
  """Return the typed outcome classification of one operator evaluation.

  Evaluators predating this contract (the landed zero-width Q8 slice) construct
  evaluations without a status slot; those evaluations are ``OK`` by
  construction. Any other foreign or malformed value fails closed.
  """
  if type(evaluation) is not OperatorEvaluation:
    msg = "evaluation status requires an exact OperatorEvaluation"
    raise TypeError(msg)
  status = getattr(evaluation, "status", EvaluationStatus.OK)
  if type(status) is not EvaluationStatus:
    msg = "operator evaluation status must be an exact EvaluationStatus"
    raise TypeError(msg)
  return status


def evaluation_derivative_values(
  evaluation: OperatorEvaluation,
) -> tuple[FinalizedArray, ...]:
  """Return the requested derivative channel values of one operator evaluation.

  Evaluators predating the parameter derivative channel construct evaluations
  without a ``derivative_values`` slot, and evaluations that served no
  derivative request carry an empty tuple; both read as empty by
  construction. Any other foreign or malformed value fails closed.
  """
  if type(evaluation) is not OperatorEvaluation:
    msg = "evaluation derivative values require an exact OperatorEvaluation"
    raise TypeError(msg)
  values = getattr(evaluation, "derivative_values", ())
  if type(values) is not tuple or any(
    type(item) is not FinalizedArray for item in values
  ):
    msg = "operator evaluation derivative values must be FinalizedArray tuples"
    raise TypeError(msg)
  return values


class StateCodec(Protocol):
  """Versioned per-block serialization contract for operator state rows.

  The codec schema matches the content schema recorded in the block's
  ``OperatorStateLayout.schema``. Encoding must be deterministic and decoding
  must fail closed on foreign schemas, dtypes, shapes, or malformed payloads;
  a decode of an encode must reproduce the state rows byte-exactly. This is the
  restart seed: opaque state rows are un-restartable without a versioned codec.
  """

  @property
  def schema(self) -> str: ...

  def encode(self, layout: OperatorStateLayout, rows: FinalizedArray) -> bytes: ...

  def decode(self, layout: OperatorStateLayout, payload: bytes) -> FinalizedArray: ...


class CompiledOperator(Protocol):
  @property
  def header(self) -> OperatorHeader: ...

  @property
  def content_manifest(self) -> CanonicalManifest: ...

  def evaluate(
    self,
    inputs: OperatorEvaluationInput,
  ) -> OperatorEvaluation: ...
