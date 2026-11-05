"""Generic compiled finite-element operator contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
  from pyfem.v3.constraints.compile import CompiledConstraintMap
  from pyfem.v3.driver.contracts import HomogenizationRequest, NonlinearStaticSettings
  from pyfem.v3.driver.nested import PreparedNestedExecution
  from pyfem.v3.model.system import CompiledSystem
  from pyfem.v3.spec.program import NodalLoadSpec
  from pyfem.v3.state.nested import NestedStateLayout, NestedStateReference

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


class NestedExecutionModality(Enum):
  DRIVER_PER_INTEGRATION_POINT = "driver-per-integration-point"


@dataclass(frozen=True, slots=True, eq=False)
class NestedExecutionRecipe:
  """Immutable compiled inputs shared by any number of nested executions.

  The recipe carries numerical content and execution policy only. Accepted
  state, drivers, workspaces, and host workers belong to one prepared runtime.
  """

  micro_system: CompiledSystem
  micro_coordinate_map: CompiledConstraintMap
  micro_loads: tuple[NodalLoadSpec, ...]
  homogenization: HomogenizationRequest
  layout: NestedStateLayout
  settings: NonlinearStaticSettings
  provenance: CanonicalManifest

  def __post_init__(self) -> None:
    from pyfem.v3.constraints.compile import CompiledConstraintMap
    from pyfem.v3.driver.contracts import HomogenizationRequest, NonlinearStaticSettings
    from pyfem.v3.model.system import CompiledSystem
    from pyfem.v3.spec.program import NodalLoadSpec
    from pyfem.v3.state.nested import NestedStateLayout

    if type(self.micro_system) is not CompiledSystem:
      msg = "nested execution recipes require an exact micro CompiledSystem"
      raise TypeError(msg)
    if type(self.micro_coordinate_map) is not CompiledConstraintMap:
      msg = "nested execution recipes require an exact micro CompiledConstraintMap"
      raise TypeError(msg)
    _exact_tuple(self.micro_loads, NodalLoadSpec, "nested micro loads")
    if type(self.homogenization) is not HomogenizationRequest:
      msg = "nested execution recipes require an exact HomogenizationRequest"
      raise TypeError(msg)
    if type(self.layout) is not NestedStateLayout:
      msg = "nested execution recipes require an exact NestedStateLayout"
      raise TypeError(msg)
    if type(self.settings) is not NonlinearStaticSettings:
      msg = "nested execution recipes require exact NonlinearStaticSettings"
      raise TypeError(msg)
    if type(self.provenance) is not CanonicalManifest:
      msg = "nested execution recipes require an exact CanonicalManifest"
      raise TypeError(msg)
    if self.layout.coefficient_count != self.micro_system.coefficient_count:
      msg = "nested layout coefficient width does not match the micro system"
      raise ValueError(msg)
    if len(self.layout.blocks) != len(self.micro_system.operators):
      msg = "nested layout block count does not match the micro system"
      raise ValueError(msg)
    for operator, block in zip(
      self.micro_system.operators, self.layout.blocks, strict=True
    ):
      state = operator.header.state_layout
      if (
        block.schema != state.schema
        or block.entity_count != state.entity_count
        or block.row_width != state.row_width
      ):
        msg = "nested layout state blocks do not match the micro system"
        raise ValueError(msg)


@dataclass(frozen=True, slots=True, eq=False)
class NestedExecutionDeclaration:
  """The immutable nested-execution recipe declared by an operator header."""

  modality: NestedExecutionModality
  recipe: NestedExecutionRecipe

  def __post_init__(self) -> None:
    if type(self.modality) is not NestedExecutionModality:
      msg = "nested execution requires an exact NestedExecutionModality"
      raise TypeError(msg)
    if type(self.recipe) is not NestedExecutionRecipe:
      msg = "nested execution declarations require an exact NestedExecutionRecipe"
      raise TypeError(msg)


@dataclass(frozen=True, slots=True, eq=False)
class NestedEvaluationInput:
  """One prepared runtime; deliberately excludes the macro transaction owner."""

  runtime: PreparedNestedExecution

  def __post_init__(self) -> None:
    from pyfem.v3.driver.nested import PreparedNestedExecution

    if type(self.runtime) is not PreparedNestedExecution:
      msg = "nested evaluation requires an exact PreparedNestedExecution"
      raise TypeError(msg)


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

  ``annotation`` carries optional metadata such as ``envelope-max`` or
  ``monotone-nondecreasing``; unannotated slots emit ``None`` explicitly.
  """

  name: str
  width: int
  dtype: str
  lifetime: StateLifetime
  annotation: str | None


@dataclass(frozen=True, slots=True, eq=False, init=False)
class OperatorStateLayout(CompilerConstructed):
  """The accepted-trial state-row ABI of one operator block.

  When ``initial_rows`` is populated, state owners seed the accepted buffer
  with those rows at construction; ``None`` selects zero initialization.
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
  """The complete compiled channel and state declaration of one operator block.

  Channel-free compilers emit empty parameter and derivative tuples explicitly.
  Direct operators emit ``None`` for nested execution and tangent class.
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
  nested_execution: NestedExecutionDeclaration | None
  tangent_class: str | None


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
  header); its empty default selects a primal-only evaluation.
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
  nested_execution: NestedEvaluationInput | None = None

  def __post_init__(self) -> None:
    _exact_tuple(self.port_values, FinalizedArray, "operator port values")
    _exact_tuple(self.signals, ProgramSignalInput, "program signal inputs")
    if (
      type(self.accepted_state) is not FinalizedArray
      or type(self.request) is not ChannelRequest
    ):
      raise TypeError("operator input requires exact state and channel request")
    if (
      self.nested_execution is not None
      and type(self.nested_execution) is not NestedEvaluationInput
    ):
      msg = "operator input nested execution must be an exact input or None"
      raise TypeError(msg)


@dataclass(frozen=True, slots=True, eq=False, init=False)
class OperatorEvaluation(CompilerConstructed):
  """The complete typed result of one operator evaluation.

  ``derivative_values`` carries one ``FinalizedArray`` per requested
  derivative channel in the header's declaration order, mirroring
  ``residual_values`` semantics; each entry follows the referenced residual
  channel's element-batch layout. Evaluations without derivative or nested
  results emit empty tuples explicitly.
  """

  residual_values: tuple[FinalizedArray, ...]
  jacobian_values: tuple[FinalizedArray, ...]
  trial_state: FinalizedArray
  status: EvaluationStatus
  derivative_values: tuple[FinalizedArray, ...]
  nested_references: tuple[NestedStateReference, ...]


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
