"""Versioned primal state-row and request-owned total-derivative codecs."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.identity import StateGeneration
from pyfem.v3.model.operator import OperatorStateLayout, SemanticId

STATE_ROW_CODEC_FORMAT = "pyfem-v3-state-rows-float64-v1"
TOTAL_PARAMETER_DERIVATIVE_STATE_SCHEMA = "pyfem-v3-total-j2-path-derivatives-v1"
TOTAL_DERIVATIVE_CODEC_FORMAT = "pyfem-v3-total-derivative-bundle-float64-v1"
_FLOAT64_DTYPE = np.dtype(np.float64).str


class StateCodecError(ValueError):
  """Raised when a state-row codec cannot honor its closed payload contract."""


def _validated_layout(codec_schema: str, layout: object) -> OperatorStateLayout:
  if type(layout) is not OperatorStateLayout:
    msg = "state row codecs require an exact OperatorStateLayout"
    raise TypeError(msg)
  if layout.schema != codec_schema:
    msg = "state row codec schema does not match the state layout content schema"
    raise StateCodecError(msg)
  if layout.dtype != _FLOAT64_DTYPE:
    msg = "float64 state row codecs require a float64 state layout"
    raise StateCodecError(msg)
  return layout


def _validated_rows(layout: OperatorStateLayout, rows: object) -> np.ndarray:
  if type(rows) is not FinalizedArray:
    msg = "state row encoding requires an exact FinalizedArray"
    raise TypeError(msg)
  values = rows.values
  if (
    values.dtype != np.dtype(np.float64)
    or values.dtype.metadata is not None
    or values.shape != layout.row_shape
  ):
    msg = "state rows must match the compiled state layout row shape and dtype"
    raise StateCodecError(msg)
  if not bool(np.isfinite(values).all()):
    msg = "state rows must be finite to enter a versioned codec payload"
    raise StateCodecError(msg)
  return values


class Float64StateRowCodec:
  """Stock versioned codec for contiguous float64 operator state rows.

  The payload is one ASCII JSON header line recording the codec format, the
  content schema, the dtype, and the row shape, followed by the exact C-order
  row bytes. Decoding fails closed on any foreign or malformed content and
  reproduces the encoded rows byte-exactly, including at zero row width.
  """

  def __init__(self, schema: str) -> None:
    if type(schema) is not str or not schema:
      msg = "state codec schema must be a non-empty exact string"
      raise ValueError(msg)
    self._schema = schema

  @property
  def schema(self) -> str:
    return self._schema

  def encode(self, layout: OperatorStateLayout, rows: FinalizedArray) -> bytes:
    validated_layout = _validated_layout(self._schema, layout)
    values = _validated_rows(validated_layout, rows)
    header = json.dumps(
      {
        "dtype": _FLOAT64_DTYPE,
        "entity_count": validated_layout.entity_count,
        "format": STATE_ROW_CODEC_FORMAT,
        "row_width": validated_layout.row_width,
        "schema": self._schema,
      },
      ensure_ascii=True,
      separators=(",", ":"),
      sort_keys=True,
    ).encode("ascii")
    return header + b"\n" + values.tobytes(order="C")

  def decode(self, layout: OperatorStateLayout, payload: bytes) -> FinalizedArray:
    validated_layout = _validated_layout(self._schema, layout)
    if type(payload) is not bytes:
      msg = "state row payloads must be exact bytes"
      raise TypeError(msg)
    header, separator, body = payload.partition(b"\n")
    if not separator:
      msg = "state row payload is missing its versioned header line"
      raise StateCodecError(msg)
    try:
      declared = json.loads(header.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
      msg = "state row payload header is not canonical ASCII JSON"
      raise StateCodecError(msg) from exc
    expected = {
      "dtype": _FLOAT64_DTYPE,
      "entity_count": validated_layout.entity_count,
      "format": STATE_ROW_CODEC_FORMAT,
      "row_width": validated_layout.row_width,
      "schema": self._schema,
    }
    if type(declared) is not dict or declared != expected:
      msg = "state row payload header contradicts the compiled state layout"
      raise StateCodecError(msg)
    row_shape = validated_layout.row_shape
    if len(body) != row_shape[0] * row_shape[1] * 8:
      msg = "state row payload byte count does not match the declared row shape"
      raise StateCodecError(msg)
    values = np.frombuffer(body, dtype=np.dtype(np.float64)).reshape(row_shape)
    if not bool(np.isfinite(values).all()):
      msg = "state row payload carries non-finite values"
      raise StateCodecError(msg)
    return FinalizedArray(values, dtype=np.float64)


def _nonempty_string(value: object, label: str) -> str:
  if type(value) is not str or not value:
    msg = f"total derivative {label} must be a non-empty exact string"
    raise ValueError(msg)
  return value


def _nonnegative_int(value: object, label: str) -> int:
  if type(value) is not int or value < 0:
    msg = f"total derivative {label} must be a non-negative exact int"
    raise ValueError(msg)
  return value


def _fingerprint(value: object, label: str) -> str:
  if (
    type(value) is not str
    or len(value) != 64
    or any(character not in "0123456789abcdef" for character in value)
  ):
    msg = f"total derivative {label} must be a lowercase SHA-256 hex digest"
    raise ValueError(msg)
  return value


def _semantic_json(value: SemanticId) -> object:
  if type(value) is tuple:
    return [_semantic_json(item) for item in value]
  if type(value) is not str and type(value) is not int:
    msg = "total derivative block IDs must contain only exact strings and integers"
    raise ValueError(msg)
  return value


@dataclass(frozen=True, slots=True)
class TotalParameterDerivativeBlockLayout:
  """Persistable geometry of one operator block's derivative rows."""

  block_id: SemanticId
  state_schema: str
  entity_count: int
  row_width: int

  def __post_init__(self) -> None:
    _semantic_json(self.block_id)
    _nonempty_string(self.state_schema, "block state schema")
    _nonnegative_int(self.entity_count, "block entity count")
    _nonnegative_int(self.row_width, "block row width")


@dataclass(frozen=True, slots=True)
class TotalParameterDerivativeLayout:
  """Request-owned identity and array authority for total path derivatives."""

  schema: str
  path_abi_schema: str
  parameter_ids: tuple[str, ...]
  system_fingerprint: str
  map_fingerprint: str
  reduced_dof_count: int
  blocks: tuple[TotalParameterDerivativeBlockLayout, ...]

  def __post_init__(self) -> None:
    _nonempty_string(self.schema, "schema")
    _nonempty_string(self.path_abi_schema, "path ABI schema")
    if type(self.parameter_ids) is not tuple or not self.parameter_ids:
      msg = "total derivative parameter IDs must be a non-empty exact tuple"
      raise ValueError(msg)
    for parameter_id in self.parameter_ids:
      _nonempty_string(parameter_id, "parameter ID")
    if len(set(self.parameter_ids)) != len(self.parameter_ids):
      msg = "total derivative parameter IDs must be unique and ordered"
      raise ValueError(msg)
    _fingerprint(self.system_fingerprint, "system fingerprint")
    _fingerprint(self.map_fingerprint, "map fingerprint")
    _nonnegative_int(self.reduced_dof_count, "reduced DOF count")
    if (
      type(self.blocks) is not tuple
      or not self.blocks
      or any(
        type(block) is not TotalParameterDerivativeBlockLayout for block in self.blocks
      )
    ):
      msg = "total derivative blocks must be a non-empty exact block-layout tuple"
      raise ValueError(msg)
    block_ids = [block.block_id for block in self.blocks]
    if len(set(block_ids)) != len(block_ids):
      msg = "total derivative block IDs must be unique and ordered"
      raise ValueError(msg)


@dataclass(frozen=True, slots=True, eq=False)
class TotalParameterDerivativeBundle:
  """Reduced coefficient and complete operator-state derivatives at one generation."""

  coefficients: FinalizedArray
  state_derivatives: tuple[FinalizedArray, ...]
  generation: StateGeneration


def _validated_total_layout(
  codec_schema: str,
  layout: object,
) -> TotalParameterDerivativeLayout:
  if type(layout) is not TotalParameterDerivativeLayout:
    msg = "total derivative codecs require an exact TotalParameterDerivativeLayout"
    raise TypeError(msg)
  if layout.schema != codec_schema:
    msg = "total derivative codec schema does not match the derivative layout schema"
    raise StateCodecError(msg)
  return layout


def _validated_total_bundle(
  layout: TotalParameterDerivativeLayout,
  bundle: object,
) -> TotalParameterDerivativeBundle:
  if type(bundle) is not TotalParameterDerivativeBundle:
    msg = "total derivative encoding requires an exact TotalParameterDerivativeBundle"
    raise TypeError(msg)
  if type(bundle.generation) is not StateGeneration:
    msg = "total derivative bundles require an exact StateGeneration"
    raise TypeError(msg)
  if type(bundle.coefficients) is not FinalizedArray:
    msg = "total derivative coefficients must be an exact FinalizedArray"
    raise TypeError(msg)
  coefficient_values = bundle.coefficients.values
  coefficient_shape = (len(layout.parameter_ids), layout.reduced_dof_count)
  if (
    coefficient_values.dtype != np.dtype(np.float64)
    or coefficient_values.dtype.metadata is not None
    or coefficient_values.shape != coefficient_shape
  ):
    msg = "total derivative coefficients do not match the layout shape and dtype"
    raise StateCodecError(msg)
  if type(bundle.state_derivatives) is not tuple or len(
    bundle.state_derivatives
  ) != len(layout.blocks):
    msg = "total derivative state blocks must follow the layout block order"
    raise StateCodecError(msg)
  arrays = [coefficient_values]
  for block, rows in zip(layout.blocks, bundle.state_derivatives, strict=True):
    if type(rows) is not FinalizedArray:
      msg = "total derivative state rows must be exact FinalizedArray values"
      raise TypeError(msg)
    values = rows.values
    shape = (len(layout.parameter_ids), block.entity_count, block.row_width)
    if (
      values.dtype != np.dtype(np.float64)
      or values.dtype.metadata is not None
      or values.shape != shape
    ):
      msg = f"total derivative state block {block.block_id!r} has a foreign shape"
      raise StateCodecError(msg)
    arrays.append(values)
  if not all(bool(np.isfinite(values).all()) for values in arrays):
    msg = "total derivative bundles must contain only finite float64 values"
    raise StateCodecError(msg)
  return bundle


def _total_header(
  layout: TotalParameterDerivativeLayout,
  generation_ordinal: int,
) -> dict[str, object]:
  parameter_count = len(layout.parameter_ids)
  body_count = parameter_count * layout.reduced_dof_count + sum(
    parameter_count * block.entity_count * block.row_width for block in layout.blocks
  )
  return {
    "blocks": [
      {
        "block_id": _semantic_json(block.block_id),
        "entity_count": block.entity_count,
        "row_width": block.row_width,
        "state_schema": block.state_schema,
      }
      for block in layout.blocks
    ],
    "body_float64_count": body_count,
    "coefficient_shape": [parameter_count, layout.reduced_dof_count],
    "dtype": _FLOAT64_DTYPE,
    "format": TOTAL_DERIVATIVE_CODEC_FORMAT,
    "generation_ordinal": generation_ordinal,
    "map_fingerprint": layout.map_fingerprint,
    "order": "C",
    "parameter_ids": list(layout.parameter_ids),
    "path_abi_schema": layout.path_abi_schema,
    "reduced_dof_count": layout.reduced_dof_count,
    "schema": layout.schema,
    "state_shapes": [
      [parameter_count, block.entity_count, block.row_width] for block in layout.blocks
    ],
    "system_fingerprint": layout.system_fingerprint,
  }


def _canonical_header(header: dict[str, object]) -> bytes:
  return json.dumps(
    header,
    ensure_ascii=True,
    separators=(",", ":"),
    sort_keys=True,
  ).encode("ascii")


class TotalParameterDerivativeCodec:
  """Closed versioned codec for one complete total-derivative bundle."""

  def __init__(self, schema: str = TOTAL_PARAMETER_DERIVATIVE_STATE_SCHEMA) -> None:
    self._schema = _nonempty_string(schema, "codec schema")

  @property
  def schema(self) -> str:
    return self._schema

  def encode(
    self,
    layout: TotalParameterDerivativeLayout,
    bundle: TotalParameterDerivativeBundle,
  ) -> bytes:
    validated_layout = _validated_total_layout(self._schema, layout)
    validated = _validated_total_bundle(validated_layout, bundle)
    header = _canonical_header(
      _total_header(validated_layout, validated.generation.ordinal)
    )
    bodies = [validated.coefficients.values.tobytes(order="C")]
    bodies.extend(
      rows.values.tobytes(order="C") for rows in validated.state_derivatives
    )
    return header + b"\n" + b"".join(bodies)

  def decode(
    self,
    layout: TotalParameterDerivativeLayout,
    payload: bytes,
    generation: StateGeneration,
  ) -> TotalParameterDerivativeBundle:
    validated_layout = _validated_total_layout(self._schema, layout)
    if type(generation) is not StateGeneration:
      msg = "total derivative decoding requires an exact expected StateGeneration"
      raise TypeError(msg)
    if type(payload) is not bytes:
      msg = "total derivative payloads must be exact bytes"
      raise TypeError(msg)
    header, separator, body = payload.partition(b"\n")
    if not separator:
      msg = "total derivative payload is missing its versioned header line"
      raise StateCodecError(msg)
    try:
      declared = json.loads(header.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
      msg = "total derivative payload header is not canonical ASCII JSON"
      raise StateCodecError(msg) from exc
    expected = _total_header(validated_layout, generation.ordinal)
    if type(declared) is not dict or declared != expected:
      msg = "total derivative payload header contradicts the layout or generation"
      raise StateCodecError(msg)
    if header != _canonical_header(expected):
      msg = "total derivative payload header is not canonical ordered ASCII JSON"
      raise StateCodecError(msg)
    if len(body) != 8 * int(expected["body_float64_count"]):
      msg = "total derivative payload byte count does not match the declared shapes"
      raise StateCodecError(msg)
    values = np.frombuffer(body, dtype=np.dtype(np.float64))
    if not bool(np.isfinite(values).all()):
      msg = "total derivative payload carries non-finite values"
      raise StateCodecError(msg)
    parameter_count = len(validated_layout.parameter_ids)
    coefficient_size = parameter_count * validated_layout.reduced_dof_count
    coefficients = values[:coefficient_size].reshape(
      parameter_count,
      validated_layout.reduced_dof_count,
    )
    offset = coefficient_size
    state_derivatives: list[FinalizedArray] = []
    for block in validated_layout.blocks:
      shape = (parameter_count, block.entity_count, block.row_width)
      size = parameter_count * block.entity_count * block.row_width
      state_derivatives.append(
        FinalizedArray(values[offset : offset + size].reshape(shape), dtype=np.float64)
      )
      offset += size
    return TotalParameterDerivativeBundle(
      coefficients=FinalizedArray(coefficients, dtype=np.float64),
      state_derivatives=tuple(state_derivatives),
      generation=generation,
    )
