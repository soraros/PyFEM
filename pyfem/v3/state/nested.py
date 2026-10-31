"""Per-integration-point nested-state stores, references, and composition codec.

This module is the FE² nested-state architecture: the state substrate for a
macro operator family whose host-side evaluator runs a nested problem per
macro integration point. Three rules, taken from the program survey, shape
everything below:

- Macro transactions stage per-integration-point nested-state REFERENCES
  (:class:`NestedStateReference` — a typed slot carrying the store's live
  instance identity, the integration-point index, and the rooted nested
  generation), never copies of nested payloads. Nested solves never touch
  the macro owner; the macro owner never touches nested buffers.
- The nested store is addressable per macro integration point and evolves on
  its own per-point accepted generations, independently of the macro
  transaction generation.
- Memory discipline is a hard contract: no per-evaluation copy of nested
  state. The legacy antipattern (``pyfem/materials/MicroModel.py:181``
  deep-copies the entire micro ``globdat`` per ``getStress``) is designed
  out: accepted state is read through zero-copy read-only views, trial state
  is written into preallocated staged buffers, and the only state copy in
  the protocol is the touched-set copy at trial commit.

Rollback is the scoped-sub-transaction candidate, chosen by measurement over
generation-gated ``restore_accepted`` (see the rollback battery in
``test/v3/test_v3_state_nested_rollback.py``): with one contiguous staged
buffer set reused across steps, a commit costs one touched-set copy and a
reject costs nothing, while the checkpoint candidate pays a full-store
snapshot every step plus a touched-set restore per reject. Measured on the
micro_rve per-point volume (272 B, 1024 points, 30 steps at 25 commits +
5 rejects): 6.96 MB copied here against 9.75 MB for restore_accepted and
25.1 MB for legacy deepcopy-per-evaluation; on a ~195 KB-per-point refined
RVE: 57.2 MB against 69.9 MB and 190.7 MB. The staged design also keeps
accepted-state reads valid throughout an open trial scope; the checkpoint
design mutates the accepted buffers in place and cannot.

Trial protocol, mirroring the owner's single-open-transaction discipline:

- ``begin_trial`` opens the one trial scope (recursively across child
  levels) and captures the per-point rooted generations.
- Evaluations read accepted views and write staged views (or use the
  validating ``write_trial_*`` staging calls); both mark the point touched.
- ``trial_complete`` marks a touched point's nested solve converged and its
  staged state complete. Completeness is whole-point: every buffer of the
  point (coefficients and every state block) must have been written in the
  current scope, so the preallocated staged buffers — reused across scopes —
  can never carry stale or rejected bytes into a commit. ``commit_trial``
  requires the touched and completed point sets to agree exactly, then
  validates finiteness, then copies the staged rows into the accepted
  buffers and advances exactly the touched points' generations. A nested
  non-convergence therefore forces a reject at the macro boundary: an
  uncompleted point can never commit.
- ``reject_trial`` discards the staged trial and leaves every accepted
  buffer and generation byte-identical.

The composition codec extends the per-level codec idiom recursively: each
level's payload is one canonical ASCII JSON header line (codec format,
content schema, coefficient width, the nested block schema registry, the
per-point generation ordinals, and per-point child payload lengths) followed
by the exact C-order float64 body — coefficients, then block rows, then the
framed child payloads in point order. Decoding never migrates: it fails
closed per level on any foreign, tampered, truncated, padded, or non-finite
payload, and :meth:`NestedStateStore.decode` additionally gates the payload
generation ordinals against the live lineage.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from uuid import uuid4

import numpy as np

from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.identity import (
  GenerationMismatchError,
  InstanceId,
  StateGeneration,
)
from pyfem.v3.state.codecs import StateCodecError

NESTED_CODEC_FORMAT = "pyfem-v3-state-nested-float64-v1"
_FLOAT64_DTYPE = np.dtype(np.float64).str


def _exact_int(value: object) -> bool:
  return type(value) is int


@dataclass(frozen=True, slots=True)
class NestedBlockLayout:
  """One nested operator state block of a nested-state level.

  ``schema`` is the block's content schema id (the macro composition header
  lists these ids as the nested schema registry); ``block_id`` the point-local
  block identity (exact string or integer — the JSON header grammar admits no
  other identifier forms); ``entity_count`` and ``row_width`` the per-point
  row geometry.
  """

  schema: str
  block_id: str | int
  entity_count: int
  row_width: int

  def __post_init__(self) -> None:
    if type(self.schema) is not str or not self.schema:
      msg = "nested block layout schema must be a non-empty exact string"
      raise ValueError(msg)
    if type(self.block_id) is str:
      if not self.block_id:
        msg = "nested block layout block id must be non-empty"
        raise ValueError(msg)
    elif type(self.block_id) is not int:
      msg = "nested block layout block id must be an exact string or integer"
      raise TypeError(msg)
    if not _exact_int(self.entity_count) or self.entity_count < 0:
      msg = "nested block layout entity count must be a non-negative exact int"
      raise ValueError(msg)
    if not _exact_int(self.row_width) or self.row_width < 0:
      msg = "nested block layout row width must be a non-negative exact int"
      raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class NestedStateLayout:
  """Codec authority and geometry of one nested-state level (recursive).

  Within one level every macro integration point shares the same nested
  coefficient width and block geometry. ``children`` pins, per point, the
  layout of the child level hosted at that point (or ``None``); it defaults
  to a flat level of ``ip_count`` childless points. A child layout makes the
  composition recursive: the point's payload embeds a full child-level
  payload, validated fail-closed at each level.
  """

  schema: str
  coefficient_count: int
  blocks: tuple[NestedBlockLayout, ...]
  ip_count: int
  children: tuple[NestedStateLayout | None, ...] | None = None

  def __post_init__(self) -> None:
    if type(self.schema) is not str or not self.schema:
      msg = "nested state layout schema must be a non-empty exact string"
      raise ValueError(msg)
    if not _exact_int(self.coefficient_count) or self.coefficient_count < 0:
      msg = "nested state layout coefficient count must be a non-negative exact int"
      raise ValueError(msg)
    if not _exact_int(self.ip_count) or self.ip_count < 0:
      msg = "nested state layout point count must be a non-negative exact int"
      raise ValueError(msg)
    if type(self.blocks) is not tuple or any(
      type(block) is not NestedBlockLayout for block in self.blocks
    ):
      msg = "nested state layout blocks must be exact NestedBlockLayout values"
      raise TypeError(msg)
    block_ids = [block.block_id for block in self.blocks]
    if len(set(block_ids)) != len(block_ids):
      msg = "nested state layout block ids must be unique within one level"
      raise ValueError(msg)
    children = self.children
    if children is None:
      object.__setattr__(self, "children", (None,) * self.ip_count)
      return
    if type(children) is not tuple or len(children) != self.ip_count:
      msg = "nested state layout children must be an exact tuple of length ip_count"
      raise TypeError(msg)
    if any(
      child is not None and type(child) is not NestedStateLayout for child in children
    ):
      msg = "children must be exact NestedStateLayout values or None"
      raise TypeError(msg)


@dataclass(frozen=True, slots=True)
class NestedStateReference:
  """Typed slot staging one point's nested-state linkage on a macro trial.

  Carries the nested store's live instance identity, the integration-point
  index, and the accepted nested generation the referenced trial is rooted
  at — never nested payload data. Staging, committing, or rejecting a macro
  transaction cannot mutate nested state; the reference is inert metadata.
  """

  store: InstanceId
  ip_index: int
  generation: StateGeneration

  def __post_init__(self) -> None:
    if type(self.store) is not InstanceId:
      msg = "nested state references require an exact InstanceId store identity"
      raise TypeError(msg)
    if not _exact_int(self.ip_index) or self.ip_index < 0:
      msg = "nested state reference index must be a non-negative exact int"
      raise ValueError(msg)
    if type(self.generation) is not StateGeneration:
      msg = "nested state references require an exact StateGeneration"
      raise TypeError(msg)


@dataclass(frozen=True, slots=True, eq=False)
class NestedStateSnapshot:
  """Immutable typed view of one nested level's per-point accepted state.

  ``coefficients`` is the ``(ip_count, coefficient_count)`` accepted nested
  physical coefficient array; ``blocks`` the per-block
  ``(ip_count, entity_count, row_width)`` accepted row arrays in layout
  order; ``generations`` the per-point accepted generation ordinals;
  ``children`` the per-point child snapshots (or ``None``).
  """

  coefficients: FinalizedArray
  blocks: tuple[FinalizedArray, ...]
  generations: tuple[int, ...]
  children: tuple[NestedStateSnapshot | None, ...]


def _validated_layout(codec_schema: str, layout: object) -> NestedStateLayout:
  if type(layout) is not NestedStateLayout:
    msg = "nested-state codecs require an exact NestedStateLayout"
    raise TypeError(msg)
  if layout.schema != codec_schema:
    msg = "nested-state codec schema does not match the nested layout schema"
    raise StateCodecError(msg)
  return layout


def _validated_rows(
  label: str,
  values: object,
  shape: tuple[int, ...],
) -> np.ndarray:
  if type(values) is not FinalizedArray:
    msg = f"nested-state {label} must be an exact FinalizedArray"
    raise TypeError(msg)
  array = values.values
  if (
    array.dtype != np.dtype(np.float64)
    or array.dtype.metadata is not None
    or array.shape != shape
  ):
    msg = f"nested-state {label} must match the layout shape {shape} and float64 dtype"
    raise StateCodecError(msg)
  if not bool(np.isfinite(array).all()):
    msg = f"nested-state {label} must be finite to enter a codec payload"
    raise StateCodecError(msg)
  return array


def _validated_snapshot(
  codec_schema: str,
  layout: NestedStateLayout,
  snapshot: object,
) -> NestedStateSnapshot:
  if type(snapshot) is not NestedStateSnapshot:
    msg = "nested-state encoding requires an exact NestedStateSnapshot"
    raise TypeError(msg)
  shape = (layout.ip_count, layout.coefficient_count)
  _validated_rows("coefficients", snapshot.coefficients, shape)
  if type(snapshot.blocks) is not tuple or len(snapshot.blocks) != len(layout.blocks):
    msg = "nested-state snapshot blocks must match the layout block registry"
    raise StateCodecError(msg)
  for block, rows in zip(layout.blocks, snapshot.blocks, strict=True):
    _validated_rows(
      f"block {block.block_id!r} rows",
      rows,
      (layout.ip_count, block.entity_count, block.row_width),
    )
  generations = snapshot.generations
  if (
    type(generations) is not tuple
    or len(generations) != layout.ip_count
    or any(not _exact_int(ordinal) or ordinal < 0 for ordinal in generations)
  ):
    msg = "nested-state snapshot generations must be non-negative exact ints per point"
    raise StateCodecError(msg)
  children = snapshot.children
  if type(children) is not tuple or len(children) != layout.ip_count:
    msg = "nested-state snapshot children must be an exact tuple of length ip_count"
    raise StateCodecError(msg)
  for child_layout, child in zip(layout.children, children, strict=True):
    if child_layout is None:
      if child is not None:
        msg = "nested-state snapshot carries a child where the layout declares none"
        raise StateCodecError(msg)
    else:
      _validated_snapshot(child_layout.schema, child_layout, child)
  return snapshot


def _header_blocks(layout: NestedStateLayout) -> list[dict[str, object]]:
  return [
    {
      "block_id": block.block_id,
      "entity_count": block.entity_count,
      "row_width": block.row_width,
      "schema": block.schema,
    }
    for block in layout.blocks
  ]


def _encode_level(
  codec_schema: str,
  layout: NestedStateLayout,
  snapshot: object,
) -> bytes:
  validated = _validated_snapshot(codec_schema, layout, snapshot)
  child_payloads: list[bytes | None] = [
    None
    if child_layout is None
    else _encode_level(child_layout.schema, child_layout, child)
    for child_layout, child in zip(layout.children, validated.children, strict=True)
  ]
  header = json.dumps(
    {
      "blocks": _header_blocks(layout),
      "child_bytes": [
        None if payload is None else len(payload) for payload in child_payloads
      ],
      "coefficient_count": layout.coefficient_count,
      "format": NESTED_CODEC_FORMAT,
      "generations": list(validated.generations),
      "schema": codec_schema,
    },
    ensure_ascii=True,
    separators=(",", ":"),
    sort_keys=True,
  ).encode("ascii")
  body = bytearray(validated.coefficients.values.tobytes(order="C"))
  for rows in validated.blocks:
    body += rows.values.tobytes(order="C")
  for payload in child_payloads:
    if payload is not None:
      body += payload
  return header + b"\n" + bytes(body)


def _decode_header_block(entry: object, block: NestedBlockLayout) -> None:
  if type(entry) is not dict or set(entry) != {
    "block_id",
    "entity_count",
    "row_width",
    "schema",
  }:
    msg = "nested-state payload block header contradicts the layout block registry"
    raise StateCodecError(msg)
  if (
    type(entry["block_id"]) is not type(block.block_id)
    or entry["block_id"] != block.block_id
  ):
    msg = "nested-state payload block id contradicts the layout block registry"
    raise StateCodecError(msg)
  if (
    entry["schema"] != block.schema
    or not _exact_int(entry["entity_count"])
    or entry["entity_count"] != block.entity_count
    or not _exact_int(entry["row_width"])
    or entry["row_width"] != block.row_width
  ):
    msg = "nested-state payload block header contradicts the layout block registry"
    raise StateCodecError(msg)


def _decode_level(
  codec_schema: str,
  layout: NestedStateLayout,
  payload: bytes,
) -> NestedStateSnapshot:
  header, separator, body = payload.partition(b"\n")
  if not separator:
    msg = "nested-state payload is missing its versioned header line"
    raise StateCodecError(msg)
  try:
    declared = json.loads(header.decode("ascii"))
  except (UnicodeDecodeError, json.JSONDecodeError) as exc:
    msg = "nested-state payload header is not canonical ASCII JSON"
    raise StateCodecError(msg) from exc
  if type(declared) is not dict or set(declared) != {
    "blocks",
    "child_bytes",
    "coefficient_count",
    "format",
    "generations",
    "schema",
  }:
    msg = "nested-state payload header contradicts the nested layout"
    raise StateCodecError(msg)
  if declared["format"] != NESTED_CODEC_FORMAT:
    msg = "nested-state payload codec format is foreign to this codec"
    raise StateCodecError(msg)
  if declared["schema"] != codec_schema:
    msg = "nested-state payload schema contradicts the nested layout schema"
    raise StateCodecError(msg)
  if (
    not _exact_int(declared["coefficient_count"])
    or declared["coefficient_count"] != layout.coefficient_count
  ):
    msg = "nested-state payload coefficient count contradicts the nested layout"
    raise StateCodecError(msg)
  declared_blocks = declared["blocks"]
  if type(declared_blocks) is not list or len(declared_blocks) != len(layout.blocks):
    msg = "nested-state payload block registry contradicts the nested layout"
    raise StateCodecError(msg)
  for entry, block in zip(declared_blocks, layout.blocks, strict=True):
    _decode_header_block(entry, block)
  generations = declared["generations"]
  if (
    type(generations) is not list
    or len(generations) != layout.ip_count
    or any(not _exact_int(ordinal) or ordinal < 0 for ordinal in generations)
  ):
    msg = "nested-state payload generations must be non-negative exact ints per point"
    raise StateCodecError(msg)
  child_bytes = declared["child_bytes"]
  if type(child_bytes) is not list or len(child_bytes) != layout.ip_count:
    msg = "nested-state payload child framing contradicts the nested layout"
    raise StateCodecError(msg)
  for child_layout, declared_length in zip(layout.children, child_bytes, strict=True):
    if child_layout is None:
      if declared_length is not None:
        msg = "nested-state payload carries a child where the layout declares none"
        raise StateCodecError(msg)
    elif not _exact_int(declared_length) or declared_length < 0:
      msg = "nested-state payload child lengths must be non-negative exact ints"
      raise StateCodecError(msg)
  fixed_bytes = (
    layout.ip_count
    * 8
    * (
      layout.coefficient_count
      + sum(block.entity_count * block.row_width for block in layout.blocks)
    )
  )
  child_total = sum(length for length in child_bytes if length is not None)
  if len(body) != fixed_bytes + child_total:
    msg = "nested-state payload byte count does not match the declared layout"
    raise StateCodecError(msg)
  offset = 0

  def take(shape: tuple[int, ...]) -> np.ndarray:
    nonlocal offset
    count = math.prod(shape)
    values = np.frombuffer(body[offset : offset + count * 8], dtype=np.float64).reshape(
      shape
    )
    offset += count * 8
    if not bool(np.isfinite(values).all()):
      msg = "nested-state payload carries non-finite values"
      raise StateCodecError(msg)
    return values

  coefficients = take((layout.ip_count, layout.coefficient_count))
  blocks = tuple(
    take((layout.ip_count, block.entity_count, block.row_width))
    for block in layout.blocks
  )
  children: list[NestedStateSnapshot | None] = []
  for child_layout, declared_length in zip(layout.children, child_bytes, strict=True):
    if child_layout is None:
      children.append(None)
      continue
    segment = body[offset : offset + declared_length]
    offset += declared_length
    children.append(_decode_level(child_layout.schema, child_layout, segment))
  return NestedStateSnapshot(
    coefficients=FinalizedArray(coefficients, dtype=np.float64),
    blocks=tuple(FinalizedArray(rows, dtype=np.float64) for rows in blocks),
    generations=tuple(generations),
    children=tuple(children),
  )


class NestedStateCodec:
  """Stock versioned codec for one nested-state composition level tree.

  Each level's payload is one canonical ASCII JSON header line recording the
  codec format, the level content schema, the coefficient width, the nested
  block schema registry, the per-point generation ordinals, and the per-point
  child payload lengths, followed by the exact C-order float64 body:
  coefficients, then per-block rows, then the framed child payloads in point
  order. All state floats live in the body for bitwise safety. Decoding fails
  closed per level on any foreign, tampered, truncated, padded, or
  non-finite content, and ``decode ∘ encode`` is byte-exact, including at
  zero point count, zero coefficient width, and zero row width.
  """

  def __init__(self, schema: str) -> None:
    if type(schema) is not str or not schema:
      msg = "nested-state codec schema must be a non-empty exact string"
      raise ValueError(msg)
    self._schema = schema

  @property
  def schema(self) -> str:
    return self._schema

  def encode(self, layout: NestedStateLayout, snapshot: NestedStateSnapshot) -> bytes:
    validated_layout = _validated_layout(self._schema, layout)
    return _encode_level(self._schema, validated_layout, snapshot)

  def decode(self, layout: NestedStateLayout, payload: bytes) -> NestedStateSnapshot:
    validated_layout = _validated_layout(self._schema, layout)
    if type(payload) is not bytes:
      msg = "nested-state payloads must be exact bytes"
      raise TypeError(msg)
    return _decode_level(self._schema, validated_layout, payload)


@dataclass(slots=True)
class _TrialScope:
  """One open nested trial scope: rooted generations plus progress masks.

  ``written`` tracks per-point buffer coverage within the scope — column 0
  is the coefficient buffer, column ``1 + slot`` the block in layout order.
  A point's trial is complete only when every buffer was written in the
  current scope, so staged-buffer reuse across scopes can never resurrect
  stale (or rejected) bytes through a partially written point.
  """

  rooted: tuple[StateGeneration, ...]
  touched: np.ndarray
  completed: np.ndarray
  written: np.ndarray


class NestedStateStore:
  """Accepted nested state of one composition level, addressable per point.

  Holds the per-point accepted nested coefficient rows and block row arrays
  (contiguous float64 buffers), the per-point accepted
  :class:`StateGeneration` lineages, and the bound child stores. Trial
  scopes stage writes into preallocated buffers reused across scopes, so the
  protocol performs no per-evaluation state copies and no steady-state
  allocation: accepted reads are zero-copy read-only views valid until the
  next ``commit_trial``, and a reject touches nothing.
  """

  __slots__ = (
    "_accepted_blocks",
    "_accepted_coefficients",
    "_block_index",
    "_children",
    "_generations",
    "_instance",
    "_layout",
    "_open",
    "_scope_serial",
    "_staged_blocks",
    "_staged_coefficients",
  )

  def __init__(
    self,
    layout: NestedStateLayout,
    *,
    children: tuple[NestedStateStore | None, ...] | None = None,
  ) -> None:
    if type(layout) is not NestedStateLayout:
      msg = "nested state stores require an exact NestedStateLayout"
      raise TypeError(msg)
    ip_count = layout.ip_count
    if children is None:
      if any(child is not None for child in layout.children):
        msg = "nested state store requires child stores where the layout declares them"
        raise ValueError(msg)
      children = (None,) * ip_count
    if type(children) is not tuple or len(children) != ip_count:
      msg = "nested state store children must be an exact tuple of length ip_count"
      raise TypeError(msg)
    for child_layout, child in zip(layout.children, children, strict=True):
      if child_layout is None:
        if child is not None:
          msg = "nested state store carries a child where the layout declares none"
          raise ValueError(msg)
      elif type(child) is not NestedStateStore or child.layout != child_layout:
        msg = "nested state store child layout must equal the declared child layout"
        raise ValueError(msg)
    bound = [child for child in children if child is not None]
    if any(bound.count(child) > 1 for child in bound):
      msg = "nested state store children must be distinct store instances"
      raise ValueError(msg)
    self._layout = layout
    self._instance = InstanceId()
    self._children = children
    self._accepted_coefficients = np.zeros(
      (ip_count, layout.coefficient_count), dtype=np.float64
    )
    self._accepted_blocks = tuple(
      np.zeros((ip_count, block.entity_count, block.row_width), dtype=np.float64)
      for block in layout.blocks
    )
    self._block_index = {
      block.block_id: index for index, block in enumerate(layout.blocks)
    }
    self._generations = [StateGeneration.initial() for _ in range(ip_count)]
    self._staged_coefficients: np.ndarray | None = None
    self._staged_blocks: tuple[np.ndarray, ...] | None = None
    self._open: _TrialScope | None = None
    self._scope_serial = 0

  @property
  def layout(self) -> NestedStateLayout:
    """Return the codec authority and geometry this store is bound to."""
    return self._layout

  @property
  def instance(self) -> InstanceId:
    """Return the live instance identity references bind to."""
    return self._instance

  @property
  def ip_count(self) -> int:
    """Return the number of addressed macro integration points."""
    return self._layout.ip_count

  @property
  def block_ids(self) -> tuple[str | int, ...]:
    """Return the block identities of this level, in layout order."""
    return tuple(block.block_id for block in self._layout.blocks)

  @property
  def children(self) -> tuple[NestedStateStore | None, ...]:
    """Return the bound per-point child stores (or ``None``)."""
    return self._children

  @property
  def trial_open(self) -> bool:
    """Return whether a trial scope is currently open on this store."""
    return self._open is not None

  @property
  def trial_scope_id(self) -> int:
    """Identify the open scope so point-backed owners can re-root their lineage."""
    self._require_trial()
    return self._scope_serial

  def current_coefficients(self, ip_index: int) -> np.ndarray:
    """Return one point's current scope coefficients as a read-only view."""
    scope = self._require_trial()
    index = self._ip_index(ip_index)
    source = (
      self._staged_coefficients if scope.touched[index] else self._accepted_coefficients
    )
    assert source is not None
    view = source[index]
    view.setflags(write=False)
    return view

  def current_block(self, ip_index: int, block_id: str | int) -> np.ndarray:
    """Return one point's current scope block rows as a read-only view."""
    scope = self._require_trial()
    index = self._ip_index(ip_index)
    block = self._block(block_id)
    staged = self._staged_blocks
    assert staged is not None
    source = staged[block] if scope.touched[index] else self._accepted_blocks[block]
    view = source[index]
    view.setflags(write=False)
    return view

  @property
  def generations(self) -> tuple[StateGeneration, ...]:
    """Return the per-point accepted generation identities."""
    return tuple(self._generations)

  def generation(self, ip_index: int) -> StateGeneration:
    """Return the accepted generation identity of one point."""
    return self._generations[self._ip_index(ip_index)]

  def _ip_index(self, ip_index: int) -> int:
    if not _exact_int(ip_index) or not 0 <= ip_index < self._layout.ip_count:
      msg = f"nested-state integration point index {ip_index!r} lies outside the store"
      raise ValueError(msg)
    return ip_index

  def _block(self, block_id: str | int) -> int:
    if block_id not in self._block_index:
      msg = f"nested state store has no state block {block_id!r}"
      raise KeyError(msg)
    return self._block_index[block_id]

  def _require_trial(self) -> _TrialScope:
    if self._open is None:
      msg = "nested-state trial operation requires an open trial scope"
      raise RuntimeError(msg)
    return self._open

  def _touch(self, scope: _TrialScope, ip_index: int) -> None:
    scope.touched[ip_index] = True
    scope.completed[ip_index] = False

  def accepted_coefficients(self, ip_index: int) -> np.ndarray:
    """Return a zero-copy read-only view of one point's accepted coefficients.

    The view is borrowed: it stays valid until the next ``commit_trial``,
    which is the only operation that ever mutates accepted buffers.
    """
    view = self._accepted_coefficients[self._ip_index(ip_index)]
    view.setflags(write=False)
    return view

  def accepted_block(self, ip_index: int, block_id: str | int) -> np.ndarray:
    """Return a zero-copy read-only view of one point's accepted block rows."""
    view = self._accepted_blocks[self._block(block_id)][self._ip_index(ip_index)]
    view.setflags(write=False)
    return view

  def begin_trial(self) -> None:
    """Open the one trial scope, recursively across child levels.

    Staged buffers are allocated on the first scope and reused thereafter;
    opening a scope copies nothing. The scope captures the per-point rooted
    generations the trial evolves from.
    """
    if self._open is not None:
      msg = "a nested-state trial scope is already open for this store"
      raise RuntimeError(msg)
    if self._staged_coefficients is None:
      self._staged_coefficients = np.empty_like(self._accepted_coefficients)
      self._staged_blocks = tuple(np.empty_like(rows) for rows in self._accepted_blocks)
    self._open = _TrialScope(
      rooted=tuple(self._generations),
      touched=np.zeros(self._layout.ip_count, dtype=bool),
      completed=np.zeros(self._layout.ip_count, dtype=bool),
      written=np.zeros(
        (self._layout.ip_count, 1 + len(self._layout.blocks)), dtype=bool
      ),
    )
    opened: list[NestedStateStore] = []
    try:
      for child in self._children:
        if child is not None:
          child.begin_trial()
          opened.append(child)
    except Exception:
      self._open = None
      for child in opened:
        child.reject_trial()
      raise
    self._scope_serial += 1

  def trial_coefficients(self, ip_index: int) -> np.ndarray:
    """Return the writable staged coefficient view of one point (zero-copy).

    Requesting the workspace marks the point touched and declares the
    coefficient buffer written for this scope: the caller must overwrite the
    view completely and report convergence through ``trial_complete`` before
    the point can commit.
    """
    scope = self._require_trial()
    index = self._ip_index(ip_index)
    self._touch(scope, index)
    scope.written[index, 0] = True
    return self._staged_coefficients[index]

  def trial_block(self, ip_index: int, block_id: str | int) -> np.ndarray:
    """Return the writable staged block-row view of one point (zero-copy).

    Requesting the workspace declares the block written for this scope; the
    caller must overwrite the view completely.
    """
    scope = self._require_trial()
    index = self._ip_index(ip_index)
    block = self._block(block_id)
    self._touch(scope, index)
    scope.written[index, 1 + block] = True
    return self._staged_blocks[block][index]

  def _stage_rows(
    self,
    label: str,
    values: object,
    shape: tuple[int, ...],
  ) -> np.ndarray:
    if type(values) is FinalizedArray:
      source = values.values
    elif type(values) is np.ndarray:
      source = values
    else:
      msg = f"{label} must be an exact FinalizedArray or plain ndarray"
      raise TypeError(msg)
    if (
      source.dtype != np.dtype(np.float64)
      or source.dtype.metadata is not None
      or source.shape != shape
    ):
      msg = f"{label} must match the layout shape {shape} and float64 dtype"
      raise ValueError(msg)
    if not bool(np.isfinite(source).all()):
      msg = f"{label} must be finite"
      raise ValueError(msg)
    return source

  def write_trial_coefficients(
    self,
    ip_index: int,
    values: FinalizedArray | np.ndarray,
  ) -> None:
    """Stage validated trial coefficients of one point."""
    scope = self._require_trial()
    index = self._ip_index(ip_index)
    source = self._stage_rows(
      "trial nested coefficients",
      values,
      (self._layout.coefficient_count,),
    )
    self._touch(scope, index)
    np.copyto(self._staged_coefficients[index], source)
    scope.written[index, 0] = True

  def write_trial_block(
    self,
    ip_index: int,
    block_id: str | int,
    values: FinalizedArray | np.ndarray,
  ) -> None:
    """Stage validated trial block rows of one point."""
    scope = self._require_trial()
    index = self._ip_index(ip_index)
    block = self._block(block_id)
    layout_block = self._layout.blocks[block]
    source = self._stage_rows(
      "trial nested state rows",
      values,
      (layout_block.entity_count, layout_block.row_width),
    )
    self._touch(scope, index)
    np.copyto(self._staged_blocks[block][index], source)
    scope.written[index, 1 + block] = True

  def trial_complete(self, ip_index: int) -> None:
    """Mark one touched point's nested solve converged and its trial complete.

    Completeness is whole-point: every buffer of the point — the coefficient
    buffer and every state block — must have been written in the current
    scope. A workspace request declares its buffer written (the caller owns
    the overwrite); the ``write_trial_*`` calls validate theirs. A partial
    point trial is a coded rejection naming the point and each unwritten
    buffer, so staged-buffer reuse across scopes can never resurrect stale
    or rejected bytes.
    """
    scope = self._require_trial()
    index = self._ip_index(ip_index)
    if not scope.touched[index]:
      msg = "nested-state trial completion requires a trial-written integration point"
      raise RuntimeError(msg)
    missing: list[str] = []
    if not scope.written[index, 0]:
      missing.append("coefficients")
    for slot, block in enumerate(self._layout.blocks):
      if not scope.written[index, 1 + slot]:
        missing.append(f"state block {block.block_id!r}")
    if missing:
      msg = (
        f"nested-state trial at point {index} is incomplete: "
        f"unwritten {', '.join(missing)} in this scope"
      )
      raise RuntimeError(msg)
    scope.completed[index] = True

  def trial_reference(self, ip_index: int) -> NestedStateReference:
    """Return the typed reference slot of one completed point's trial."""
    scope = self._require_trial()
    index = self._ip_index(ip_index)
    if not scope.completed[index]:
      msg = "nested-state trial references require a completed integration point trial"
      raise RuntimeError(msg)
    return NestedStateReference(
      store=self._instance,
      ip_index=index,
      generation=scope.rooted[index],
    )

  def _validate_trial(self, scope: _TrialScope) -> None:
    incomplete = scope.touched & ~scope.completed
    if bool(incomplete.any()):
      index = int(np.flatnonzero(incomplete)[0])
      msg = f"nested-state trial commit requires a completed trial at point {index}"
      raise RuntimeError(msg)
    if bool(scope.touched.any()):
      finite = np.isfinite(self._staged_coefficients).all(axis=1)
      for rows in self._staged_blocks:
        finite &= np.isfinite(rows).all(axis=(1, 2))
      bad = scope.touched & ~finite
      if bool(bad.any()):
        index = int(np.flatnonzero(bad)[0])
        msg = f"nested-state trial state must be finite to commit (point {index})"
        raise ValueError(msg)
    for child in self._children:
      if child is None:
        continue
      if child._open is None:
        msg = "nested-state trial scope closed out from under the macro transaction"
        raise RuntimeError(msg)
      child._validate_trial(child._open)

  def _commit_validated(self, scope: _TrialScope) -> None:
    # Validation already passed for the whole level tree; the writes below
    # cannot fail, so the transition is indivisible.
    touched = scope.touched
    if bool(touched.any()):
      np.copyto(
        self._accepted_coefficients,
        self._staged_coefficients,
        where=touched[:, None],
      )
      for accepted, staged in zip(
        self._accepted_blocks, self._staged_blocks, strict=True
      ):
        np.copyto(accepted, staged, where=touched[:, None, None])
      for index in np.flatnonzero(touched):
        self._generations[index] = self._generations[index].next_accepted()
    self._open = None
    for child in self._children:
      if child is not None:
        child._commit_validated(child._open)

  def validate_trial(self) -> None:
    """Validate the whole open trial tree before the macro owner commits."""
    self._validate_trial(self._require_trial())

  def commit_trial(self) -> None:
    """Atomically accept every completed point's trial, recursively.

    Validates the whole level tree before writing anything: the touched and
    completed point sets must agree exactly at every level, and staged trial
    state must be finite. Untouched points keep their accepted state and
    generation byte-identically.
    """
    scope = self._require_trial()
    self._validate_trial(scope)
    self._commit_validated(scope)

  def reject_trial(self) -> None:
    """Discard the trial, recursively, leaving accepted state byte-identical."""
    self._require_trial()
    breached = False
    for child in self._children:
      if child is None:
        continue
      if child._open is None:
        breached = True
      else:
        child.reject_trial()
    self._open = None
    if breached:
      msg = "nested-state trial scope closed out from under the macro transaction"
      raise RuntimeError(msg)

  def snapshot(self) -> NestedStateSnapshot:
    """Return the immutable typed view of the accepted state (detached copies)."""
    return NestedStateSnapshot(
      coefficients=FinalizedArray(self._accepted_coefficients, dtype=np.float64),
      blocks=tuple(
        FinalizedArray(rows, dtype=np.float64) for rows in self._accepted_blocks
      ),
      generations=tuple(generation.ordinal for generation in self._generations),
      children=tuple(
        None if child is None else child.snapshot() for child in self._children
      ),
    )

  def encode(self) -> bytes:
    """Encode the accepted state tree with the versioned composition codec."""
    codec = NestedStateCodec(self._layout.schema)
    return codec.encode(self._layout, self.snapshot())

  @classmethod
  def _restore(
    cls,
    layout: NestedStateLayout,
    snapshot: NestedStateSnapshot,
  ) -> NestedStateStore:
    # Child levels root at fresh lineages at the payload ordinals: accepted
    # generations are process-local lineage identities, and cross-process
    # lineage restore belongs to the checkpoint-bundle follow-up (the M45
    # boundary). Rooting at the payload ordinals keeps re-encode byte-exact.
    children = tuple(
      None if child_layout is None else cls._restore(child_layout, child)
      for child_layout, child in zip(layout.children, snapshot.children, strict=True)
    )
    store = cls(layout, children=children)
    np.copyto(store._accepted_coefficients, snapshot.coefficients.values)
    for accepted, rows in zip(store._accepted_blocks, snapshot.blocks, strict=True):
      np.copyto(accepted, rows.values)
    store._generations = [
      StateGeneration._from_lineage(uuid4(), ordinal)
      for ordinal in snapshot.generations
    ]
    return store

  @classmethod
  def decode(
    cls,
    layout: NestedStateLayout,
    generations: tuple[StateGeneration, ...],
    payload: bytes,
  ) -> NestedStateStore:
    """Decode one payload into a store rooted at the supplied live generations.

    Like the evolution store, the payload deliberately carries no lineage:
    the caller supplies the per-point live generations this payload was
    captured at. The payload's top-level generation ordinals must equal the
    supplied generations' ordinals exactly; a mismatch is a coded
    :class:`GenerationMismatchError` rejection. Decoding never migrates.
    """
    if type(layout) is not NestedStateLayout:
      msg = "nested state stores require an exact NestedStateLayout"
      raise TypeError(msg)
    codec = NestedStateCodec(layout.schema)
    snapshot = codec.decode(layout, payload)
    if type(generations) is not tuple:
      msg = "nested-state decode generations must be an exact tuple"
      raise TypeError(msg)
    if len(generations) != layout.ip_count or any(
      type(generation) is not StateGeneration for generation in generations
    ):
      msg = "nested-state decode requires one exact live generation per point"
      raise TypeError(msg)
    for ordinal, (generation, declared) in enumerate(
      zip(generations, snapshot.generations, strict=True)
    ):
      if generation.ordinal != declared:
        msg = (
          f"nested-state payload generation at point {ordinal} does not match "
          "the live lineage"
        )
        raise GenerationMismatchError(msg)
    children = tuple(
      None if child_layout is None else cls._restore(child_layout, child)
      for child_layout, child in zip(layout.children, snapshot.children, strict=True)
    )
    store = cls(layout, children=children)
    np.copyto(store._accepted_coefficients, snapshot.coefficients.values)
    for accepted, rows in zip(store._accepted_blocks, snapshot.blocks, strict=True):
      np.copyto(accepted, rows.values)
    store._generations = list(generations)
    return store
