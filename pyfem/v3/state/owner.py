"""Generation-scoped atomic transaction owner for compiled-system state.

The owner is the single point of correctness for accepted state: it holds the
mutable accepted buffers (the global physical coefficient vector plus one
contiguous state-row array per operator block, honoring the compiled
``OperatorStateLayout`` ABI) and exposes them to evaluations only as detached
read-only snapshots. Generations are metadata — a ``StateGeneration``
lineage/ordinal plus a history of small records — never copied data
structures. A rejected transaction discards its staged trial and leaves the
committed state byte-identical; a commit validates everything staged before
writing anything, then swings physical coefficients and per-block state rows
in one indivisible transition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.identity import IdentityMismatchError, StateGeneration
from pyfem.v3.model.operator import (
  OperatorStateLayout,
  OperatorStateSlot,
  SemanticId,
  StateCodec,
  StateLifetime,
)
from pyfem.v3.model.system import CompiledSystem
from pyfem.v3.state.codecs import Float64StateRowCodec
from pyfem.v3.state.nested import NestedStateReference, NestedStateStore

_TRANSACTION_SEAL = object()
_FLOAT64_DTYPE = np.dtype(np.float64).str


def _exact_int(value: object) -> bool:
  return type(value) is int


@dataclass(frozen=True, slots=True)
class GenerationRecord:
  """Metadata of one committed generation transition; carries no state data."""

  ordinal: int
  physical_staged: bool
  staged_block_ids: tuple[SemanticId, ...]
  nested_references: tuple[NestedStateReference, ...] = ()


def _validated_layout(value: object) -> OperatorStateLayout:
  if type(value) is not OperatorStateLayout:
    msg = "transaction owner bindings require an exact OperatorStateLayout"
    raise TypeError(msg)
  layout = value
  if type(layout.schema) is not str or not layout.schema:
    msg = "operator state layout schema must be a non-empty exact string"
    raise TypeError(msg)
  if not _exact_int(layout.entity_count) or layout.entity_count < 0:
    msg = "operator state layout entity count must be a non-negative exact int"
    raise TypeError(msg)
  if not _exact_int(layout.row_width) or layout.row_width < 0:
    msg = "operator state layout row width must be a non-negative exact int"
    raise TypeError(msg)
  if layout.dtype != _FLOAT64_DTYPE:
    msg = "transaction owner state rows must use the float64 layout dtype"
    raise TypeError(msg)
  if layout.lifetime is not StateLifetime.ACCEPTED_TRIAL:
    msg = "transaction owner state rows must use the accepted-trial lifetime"
    raise TypeError(msg)
  if type(layout.slots) is not tuple or any(
    type(slot) is not OperatorStateSlot for slot in layout.slots
  ):
    msg = "operator state layout slots must be exact OperatorStateSlot values"
    raise TypeError(msg)
  for slot in layout.slots:
    if (
      type(slot.name) is not str
      or not slot.name
      or not _exact_int(slot.width)
      or slot.width <= 0
      or slot.dtype != _FLOAT64_DTYPE
      or slot.lifetime is not StateLifetime.ACCEPTED_TRIAL
    ):
      msg = "operator state layout slots must be positive-width float64 rows"
      raise TypeError(msg)
    annotation = slot.annotation
    if annotation is not None and (type(annotation) is not str or not annotation):
      msg = "operator state slot annotations must be non-empty exact strings"
      raise TypeError(msg)
  if sum(slot.width for slot in layout.slots) != layout.row_width:
    msg = "operator state layout row width must equal the summed slot widths"
    raise TypeError(msg)
  if type(layout.entity_offsets) is not FinalizedArray:
    msg = "operator state layout entity offsets must be an exact FinalizedArray"
    raise TypeError(msg)
  offsets = layout.entity_offsets.values
  if (
    offsets.dtype.kind not in "iu"
    or offsets.dtype.metadata is not None
    or offsets.shape != (layout.entity_count + 1,)
    or int(offsets[0]) != 0
    or int(offsets[-1]) != layout.entity_count * layout.row_width
    or bool((offsets[1:] < offsets[:-1]).any())
  ):
    msg = "operator state layout entity offsets must bound contiguous rows"
    raise TypeError(msg)
  initial_rows = layout.initial_rows
  if initial_rows is not None:
    if type(initial_rows) is not FinalizedArray:
      msg = "operator state layout initial rows must be an exact FinalizedArray"
      raise TypeError(msg)
    rows = initial_rows.values
    if (
      rows.dtype != np.dtype(np.float64)
      or rows.dtype.metadata is not None
      or rows.shape != layout.row_shape
    ):
      msg = "operator state layout initial rows must match the layout row shape"
      raise ValueError(msg)
    if not bool(np.isfinite(rows).all()):
      msg = "operator state layout initial rows must be finite"
      raise ValueError(msg)
  return layout


@dataclass(slots=True)
class _BlockBinding:
  layout: OperatorStateLayout
  accepted: np.ndarray
  codec: StateCodec


def _detached(array: np.ndarray) -> FinalizedArray:
  return FinalizedArray(array, dtype=np.float64)


def _owned_rows(value: object, shape: tuple[int, int], label: str) -> np.ndarray:
  if type(value) is FinalizedArray:
    source = value.values
  elif type(value) is np.ndarray:
    source = value
  else:
    msg = f"{label} must be an exact FinalizedArray or plain ndarray"
    raise TypeError(msg)
  if (
    source.dtype != np.dtype(np.float64)
    or source.dtype.metadata is not None
    or source.shape != shape
  ):
    msg = f"{label} must match the compiled float64 shape {shape}"
    raise ValueError(msg)
  owned = np.array(source, dtype=np.float64, order="C", copy=True, subok=False)
  if not bool(np.isfinite(owned).all()):
    msg = f"{label} must be finite"
    raise ValueError(msg)
  return owned


class StateTransaction:
  """One open trial rooted at a published accepted generation.

  Transactions are issued only by ``StateTransactionOwner.begin``. Staged
  values are validated and ownership-copied at staging time, so a later
  mutation of driver-side arrays cannot corrupt a pending commit. Exactly one
  of ``commit`` or ``reject`` consumes the transaction.
  """

  __slots__ = (
    "_accepted_blocks",
    "_accepted_physical",
    "_closed",
    "_generation",
    "_owner",
    "_staged_blocks",
    "_staged_nested",
    "_staged_nested_keys",
    "_staged_physical",
  )

  def __init__(
    self,
    *,
    _seal: object = None,
    owner: StateTransactionOwner,
    generation: StateGeneration,
    accepted_physical: FinalizedArray,
    accepted_blocks: dict[SemanticId, FinalizedArray],
  ) -> None:
    if _seal is not _TRANSACTION_SEAL:
      msg = "state transactions are issued only by StateTransactionOwner.begin"
      raise TypeError(msg)
    self._owner = owner
    self._generation = generation
    self._accepted_physical = accepted_physical
    self._accepted_blocks = accepted_blocks
    self._staged_physical: np.ndarray | None = None
    self._staged_blocks: dict[SemanticId, np.ndarray] = {}
    self._staged_nested: list[NestedStateReference] = []
    self._staged_nested_keys: set[tuple[object, int]] = set()
    self._closed = False

  def _require_open(self) -> None:
    if self._closed:
      msg = "state transaction is already consumed"
      raise RuntimeError(msg)

  @property
  def generation(self) -> StateGeneration:
    """Return the accepted generation this trial is rooted at."""
    return self._generation

  @property
  def accepted_physical(self) -> FinalizedArray:
    """Return the detached read-only accepted physical coefficients."""
    return self._accepted_physical

  def accepted_state(self, block_id: SemanticId) -> FinalizedArray:
    """Return the detached read-only accepted state rows of one block."""
    return self._accepted_blocks[self._owner._block_key(block_id)]

  def stage_physical(self, values: FinalizedArray | np.ndarray) -> None:
    """Stage the trial global physical coefficient vector for commit."""
    self._require_open()
    self._staged_physical = _owned_rows(
      values,
      (self._owner.coefficient_count,),
      "trial physical coefficients",
    )

  def stage_state(
    self,
    block_id: SemanticId,
    rows: FinalizedArray | np.ndarray,
  ) -> None:
    """Stage trial state rows of one operator block for commit."""
    self._require_open()
    binding = self._owner._binding(block_id)
    self._staged_blocks[binding.layout.block_id] = _owned_rows(
      rows,
      binding.layout.row_shape,
      "trial operator state rows",
    )

  def stage_nested(self, reference: NestedStateReference) -> None:
    """Stage one macro integration point's nested-state reference for commit.

    The reference is a typed slot — the nested store's live instance
    identity, the integration-point index, and the rooted nested generation —
    never nested payload data. Staging validates the reference against the
    owner's bound nested stores; committing or rejecting cannot mutate nested
    state either way.
    """
    self._require_open()
    if type(reference) is not NestedStateReference:
      msg = "staged nested-state linkage must be an exact NestedStateReference"
      raise TypeError(msg)
    store = self._owner._nested_store_for(reference)
    if reference.ip_index >= store.ip_count:
      msg = (
        f"nested-state reference index {reference.ip_index} lies outside the "
        "bound nested store"
      )
      raise ValueError(msg)
    key = (reference.store, reference.ip_index)
    if key in self._staged_nested_keys:
      msg = f"duplicate nested-state reference for point {reference.ip_index}"
      raise ValueError(msg)
    self._staged_nested_keys.add(key)
    self._staged_nested.append(reference)

  @property
  def nested_references(self) -> tuple[NestedStateReference, ...]:
    """Return the nested-state references staged so far (typed slots only)."""
    return tuple(self._staged_nested)

  def commit(self) -> None:
    """Atomically commit every staged value as one generation transition."""
    self._require_open()
    self._owner._commit(self)

  def reject(self) -> None:
    """Discard the trial, leaving committed state byte-identical."""
    self._require_open()
    self._owner._reject(self)


class StateTransactionOwner:
  """The atomic accepted-state owner for one compiled system."""

  __slots__ = (
    "_blocks",
    "_generation",
    "_history",
    "_nested_block_ids",
    "_nested_point",
    "_nested_scope_id",
    "_nested_stores",
    "_open",
    "_physical",
    "_system",
  )

  def __init__(
    self,
    system: CompiledSystem,
    *,
    codecs: tuple[StateCodec, ...] = (),
    nested_stores: tuple[NestedStateStore, ...] = (),
  ) -> None:
    if type(system) is not CompiledSystem:
      msg = "transaction owner requires an exact CompiledSystem"
      raise TypeError(msg)
    if type(nested_stores) is not tuple or any(
      type(store) is not NestedStateStore for store in nested_stores
    ):
      msg = "transaction owner nested stores must be an exact NestedStateStore tuple"
      raise TypeError(msg)
    if len({store.instance for store in nested_stores}) != len(nested_stores):
      msg = "transaction owner nested stores must be distinct store instances"
      raise ValueError(msg)
    if type(codecs) is not tuple:
      msg = "transaction owner codecs must be an exact tuple"
      raise TypeError(msg)
    codec_schemas: list[str] = []
    for codec in codecs:
      schema = codec.schema
      if type(schema) is not str or not schema:
        msg = "state codecs must publish a non-empty exact schema string"
        raise TypeError(msg)
      if schema in codec_schemas:
        msg = f"duplicate state codec for schema {schema!r}"
        raise ValueError(msg)
      codec_schemas.append(schema)

    blocks: dict[SemanticId, _BlockBinding] = {}
    for operator in system.operators:
      layout = _validated_layout(operator.header.state_layout)
      block_id = layout.block_id
      if block_id in blocks:
        msg = f"duplicate operator state block id {block_id!r}"
        raise ValueError(msg)
      codec = next(
        (candidate for candidate in codecs if candidate.schema == layout.schema),
        None,
      )
      if codec is None:
        codec = Float64StateRowCodec(layout.schema)
      # Layouts carrying compiler-emitted initial rows seed the accepted buffer
      # with them; all other layouts keep zero initialization.
      initial_rows = layout.initial_rows
      accepted = (
        np.array(
          initial_rows.values,
          dtype=np.float64,
          order="C",
          copy=True,
          subok=False,
        )
        if initial_rows is not None
        else np.zeros(layout.row_shape, dtype=np.float64)
      )
      blocks[block_id] = _BlockBinding(
        layout=layout,
        accepted=accepted,
        codec=codec,
      )

    self._system = system
    self._physical = np.zeros(system.coefficient_count, dtype=np.float64)
    self._blocks = blocks
    self._nested_stores = nested_stores
    self._nested_point: tuple[NestedStateStore, int] | None = None
    self._nested_block_ids: dict[SemanticId, str | int] = {}
    self._nested_scope_id = -1
    self._generation = StateGeneration.initial()
    self._history: tuple[GenerationRecord, ...] = ()
    self._open: StateTransaction | None = None

  @classmethod
  def for_nested_point(
    cls,
    system: CompiledSystem,
    *,
    store: NestedStateStore,
    ip_index: int,
  ) -> StateTransactionOwner:
    """Create a driver owner backed by one point of an open nested trial scope."""
    if type(store) is not NestedStateStore:
      msg = "point-backed owners require an exact NestedStateStore"
      raise TypeError(msg)
    store.generation(ip_index)
    owner = cls(system)
    layout = store.layout
    if layout.coefficient_count != system.coefficient_count:
      msg = "nested point coefficient width does not match the compiled system"
      raise ValueError(msg)
    if len(layout.blocks) != len(owner._blocks):
      msg = "nested point state block count does not match the compiled system"
      raise ValueError(msg)
    block_ids: dict[SemanticId, str | int] = {}
    for (block_id, binding), nested_block in zip(
      owner._blocks.items(), layout.blocks, strict=True
    ):
      if (
        binding.layout.entity_count != nested_block.entity_count
        or binding.layout.row_width != nested_block.row_width
      ):
        msg = (
          f"nested point state block {block_id!r} does not match the compiled layout"
        )
        raise ValueError(msg)
      block_ids[block_id] = nested_block.block_id
    owner._nested_point = (store, ip_index)
    owner._nested_block_ids = block_ids
    owner._physical = np.empty(0, dtype=np.float64)
    for binding in owner._blocks.values():
      binding.accepted = np.empty((0, 0), dtype=np.float64)
    return owner

  @property
  def system(self) -> CompiledSystem:
    """Return the compiled system whose state this owner holds."""
    return self._system

  @property
  def generation(self) -> StateGeneration:
    """Return the current accepted generation identity."""
    return self._generation

  @property
  def history(self) -> tuple[GenerationRecord, ...]:
    """Return the committed-generation metadata records; never state data."""
    return self._history

  @property
  def coefficient_count(self) -> int:
    """Return the global physical coefficient vector length."""
    return self._system.coefficient_count

  @property
  def block_ids(self) -> tuple[SemanticId, ...]:
    """Return the bound operator state block identities."""
    return tuple(self._blocks)

  def _block_key(self, block_id: SemanticId) -> SemanticId:
    if block_id not in self._blocks:
      msg = f"transaction owner has no state block {block_id!r}"
      raise KeyError(msg)
    return block_id

  def _binding(self, block_id: SemanticId) -> _BlockBinding:
    return self._blocks[self._block_key(block_id)]

  @property
  def nested_stores(self) -> tuple[NestedStateStore, ...]:
    """Return the nested stores bound for reference validation."""
    return self._nested_stores

  def _nested_store_for(self, reference: NestedStateReference) -> NestedStateStore:
    for store in self._nested_stores:
      if store.instance == reference.store:
        return store
    msg = "nested-state staging requires a store bound to this transaction owner"
    raise IdentityMismatchError(msg)

  def state_layout(self, block_id: SemanticId) -> OperatorStateLayout:
    """Return the compiled state layout bound for one block."""
    return self._binding(block_id).layout

  def state_codec(self, block_id: SemanticId) -> StateCodec:
    """Return the versioned state codec bound for one block."""
    return self._binding(block_id).codec

  def _current_physical(self) -> np.ndarray:
    if self._nested_point is None:
      return self._physical
    store, ip_index = self._nested_point
    return store.current_coefficients(ip_index)

  def _current_state(self, block_id: SemanticId) -> np.ndarray:
    binding = self._binding(block_id)
    if self._nested_point is None:
      return binding.accepted
    store, ip_index = self._nested_point
    nested_id = self._nested_block_ids[binding.layout.block_id]
    return store.current_block(ip_index, nested_id)

  def accepted_physical(self) -> FinalizedArray:
    """Return a detached read-only copy of the accepted physical vector."""
    return _detached(self._current_physical())

  def accepted_state(self, block_id: SemanticId) -> FinalizedArray:
    """Return a detached read-only copy of one block's accepted state rows."""
    return _detached(self._current_state(block_id))

  def encode_state(self, block_id: SemanticId) -> bytes:
    """Encode one block's accepted state rows with its versioned codec."""
    binding = self._binding(block_id)
    rows = _detached(self._current_state(block_id))
    return binding.codec.encode(binding.layout, rows)

  def decode_state(self, block_id: SemanticId, payload: bytes) -> FinalizedArray:
    """Decode one block's state-row payload, failing closed on foreign data."""
    binding = self._binding(block_id)
    return binding.codec.decode(binding.layout, payload)

  def begin(self) -> StateTransaction:
    """Open one trial transaction rooted at the current accepted generation.

    The accepted state is published once per transaction as detached read-only
    snapshots, so evaluations stay pure from accepted state and a later commit
    or reject cannot retroactively mutate what an evaluation observed.
    """
    if self._open is not None:
      msg = "a state transaction is already open for this owner"
      raise RuntimeError(msg)
    if self._nested_point is not None:
      store, ip_index = self._nested_point
      scope_id = store.trial_scope_id
      if self._nested_scope_id != scope_id:
        self._generation = store.generation(ip_index)
        self._history = ()
        self._nested_scope_id = scope_id
    transaction = StateTransaction(
      _seal=_TRANSACTION_SEAL,
      owner=self,
      generation=self._generation,
      accepted_physical=_detached(self._current_physical()),
      accepted_blocks={
        block_id: _detached(self._current_state(block_id)) for block_id in self._blocks
      },
    )
    self._open = transaction
    return transaction

  def _commit(self, transaction: StateTransaction) -> None:
    if self._open is not transaction:
      msg = "only the open state transaction can commit"
      raise RuntimeError(msg)
    staged_physical = transaction._staged_physical
    staged_blocks = transaction._staged_blocks
    if staged_physical is None and not staged_blocks:
      msg = "a state transaction commit requires at least one staged value"
      raise ValueError(msg)
    # Every staged value was validated and ownership-copied at staging time.
    # A point-backed microstep writes only its store's trial buffers; a macro
    # owner writes its own accepted buffers and never writes nested payloads.
    if self._nested_point is not None:
      store, ip_index = self._nested_point
      if staged_physical is not None:
        store.write_trial_coefficients(ip_index, staged_physical)
      for block_id, rows in staged_blocks.items():
        store.write_trial_block(ip_index, self._nested_block_ids[block_id], rows)
    else:
      if staged_physical is not None:
        np.copyto(self._physical, staged_physical)
      for block_id, rows in staged_blocks.items():
        np.copyto(self._blocks[block_id].accepted, rows)
    self._generation = self._generation.next_accepted()
    self._history = (
      *self._history,
      GenerationRecord(
        ordinal=self._generation.ordinal,
        physical_staged=staged_physical is not None,
        staged_block_ids=tuple(staged_blocks),
        nested_references=tuple(transaction._staged_nested),
      ),
    )
    transaction._closed = True
    self._open = None

  def _reject(self, transaction: StateTransaction) -> None:
    if self._open is not transaction:
      msg = "only the open state transaction can be rejected"
      raise RuntimeError(msg)
    transaction._staged_physical = None
    transaction._staged_blocks = {}
    transaction._staged_nested = []
    transaction._staged_nested_keys = set()
    transaction._closed = True
    self._open = None
