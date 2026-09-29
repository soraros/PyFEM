"""Versioned per-block state-row codecs for the transaction owner."""

from __future__ import annotations

import json

import numpy as np

from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.operator import OperatorStateLayout

STATE_ROW_CODEC_FORMAT = "pyfem-v3-state-rows-float64-v1"
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
