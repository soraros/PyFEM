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
  name: str
  width: int
  dtype: str
  lifetime: StateLifetime


@dataclass(frozen=True, slots=True, eq=False, init=False)
class OperatorStateLayout(CompilerConstructed):
  schema: str
  block_id: SemanticId
  entity_count: int
  slots: tuple[OperatorStateSlot, ...]
  entity_offsets: FinalizedArray
  row_width: int
  dtype: str
  lifetime: StateLifetime

  @property
  def row_shape(self) -> tuple[int, int]:
    return self.entity_count, self.row_width


@dataclass(frozen=True, slots=True, eq=False, init=False)
class OperatorHeader(CompilerConstructed):
  block_id: SemanticId
  entity_block_id: SemanticId
  implementations: tuple[ImplementationIdentity, ...]
  ports: tuple[PortBinding, ...]
  signal_ports: tuple[SignalPortBinding, ...]
  residual_channels: tuple[ResidualChannel, ...]
  jacobian_channels: tuple[JacobianChannel, ...]
  state_layout: OperatorStateLayout
  coupling_policy: CouplingPolicy


@dataclass(frozen=True, slots=True, eq=False)
class SignalDerivativeInput:
  coordinate_id: SemanticId
  values: FinalizedArray

  def __post_init__(self) -> None:
    if type(self.values) is not FinalizedArray:
      raise TypeError("signal derivative values must be an exact FinalizedArray")


@dataclass(frozen=True, slots=True, eq=False)
class ProgramSignalInput:
  port_id: str
  values: FinalizedArray
  derivatives: tuple[SignalDerivativeInput, ...]

  def __post_init__(self) -> None:
    if type(self.port_id) is not str or type(self.values) is not FinalizedArray:
      raise TypeError("program signal input requires an exact port and values")
    _exact_tuple(self.derivatives, SignalDerivativeInput, "signal derivatives")


@dataclass(frozen=True, slots=True, eq=False)
class ChannelRequest:
  residual_channel_ids: tuple[str, ...]
  jacobian_channel_ids: tuple[str, ...]

  def __post_init__(self) -> None:
    _exact_tuple(self.residual_channel_ids, str, "residual channel IDs")
    _exact_tuple(self.jacobian_channel_ids, str, "Jacobian channel IDs")


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
  residual_values: tuple[FinalizedArray, ...]
  jacobian_values: tuple[FinalizedArray, ...]
  trial_state: FinalizedArray
  status: EvaluationStatus


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
