"""Versioned canonical semantic manifests and reproducible fingerprints.

The serialization boundary is deliberately narrow and explicit. Supported values
are ``None``, exact Python booleans/integers/finite floats/strings, equivalent NumPy
scalars no wider than binary64, exact plain numeric NumPy arrays without extended
floating dtypes, exact string-keyed dictionaries, exact ordered lists/tuples,
``UnorderedDeclarations``, and already captured ``CanonicalManifest`` values.

Mappings are ordered by key. Lists and tuples retain physical order and share one
sequence representation. Numeric arrays record shape, canonical little-endian
dtype, and C-order bytes. Unordered declarations must opt in explicitly and carry a
stable semantic ID. Sets, arbitrary objects, and callables are rejected rather
than guessed at or content-hashed.

Serialized bytes begin with ``pyfem-v3-semantic-manifest-v1`` plus a newline,
followed by compact ASCII JSON containing explicit type tags. Any change to those
tags, ordering rules, or numeric encoding requires a new format identifier; v1
never depends on ``repr()``, pickle, mapping insertion order, or callable source.
"""

from __future__ import annotations

import json
import math
from base64 import b64decode, b64encode
from binascii import Error as BinasciiError
from dataclasses import dataclass, field
from hashlib import sha256
from typing import ClassVar, Self, final

import numpy as np

from pyfem.v3.model.arrays import _dtype_has_metadata, _type_derives_from

CANONICAL_MANIFEST_FORMAT = "pyfem-v3-semantic-manifest-v1"
_MANIFEST_HEADER = f"{CANONICAL_MANIFEST_FORMAT}\n".encode()
_BINARY64_INFO = np.finfo(np.float64)
_NUMPY_BOOL_TYPES = (np.bool_,)
_NUMPY_INTEGER_TYPES = tuple(
  scalar_type
  for scalar_type in np.ScalarType
  if isinstance(scalar_type, type) and issubclass(scalar_type, np.integer)
)
_NUMPY_FLOAT_TYPES = tuple(
  scalar_type
  for scalar_type in np.ScalarType
  if isinstance(scalar_type, type) and issubclass(scalar_type, np.floating)
)

type _CanonicalNode = tuple[object, ...]


def _encode_node(node: _CanonicalNode) -> bytes:
  return json.dumps(
    node,
    ensure_ascii=True,
    separators=(",", ":"),
  ).encode()


def _canonical_float(value: float) -> _CanonicalNode:
  if not math.isfinite(value):
    msg = "semantic manifests require finite floating-point values"
    raise ValueError(msg)
  return ("float", value.hex())


def _floating_dtype_exceeds_binary64(dtype: np.dtype[np.generic]) -> bool:
  info = np.finfo(dtype)
  return (
    info.nmant > _BINARY64_INFO.nmant
    or info.minexp < _BINARY64_INFO.minexp
    or info.maxexp > _BINARY64_INFO.maxexp
  )


def _malformed_carrier() -> None:
  msg = "malformed exact canonical carrier payload"
  raise TypeError(msg)


def _validate_canonical_node(node: object) -> None:
  if type(node) is not tuple or not node or type(node[0]) is not str:
    _malformed_carrier()
  tag = node[0]
  if tag == "none":
    if len(node) != 1:
      _malformed_carrier()
    return
  if tag == "bool":
    if len(node) != 2 or type(node[1]) is not bool:
      _malformed_carrier()
    return
  if tag == "int":
    if len(node) != 2 or type(node[1]) is not str:
      _malformed_carrier()
    text = node[1]
    digits = text[1:] if text.startswith("-") else text
    if (
      not digits
      or not digits.isascii()
      or not digits.isdigit()
      or (len(digits) > 1 and digits[0] == "0")
      or text == "-0"
    ):
      _malformed_carrier()
    return
  if tag == "float":
    if len(node) != 2 or type(node[1]) is not str:
      _malformed_carrier()
    try:
      value = float.fromhex(node[1])
    except ValueError:
      _malformed_carrier()
    if not math.isfinite(value) or value.hex() != node[1]:
      _malformed_carrier()
    return
  if tag == "str":
    if len(node) != 2 or type(node[1]) is not str:
      _malformed_carrier()
    return
  if tag == "ndarray":
    if (
      len(node) != 4
      or type(node[1]) is not str
      or type(node[2]) is not tuple
      or type(node[3]) is not str
      or any(type(size) is not int or size < 0 for size in node[2])
    ):
      _malformed_carrier()
    try:
      dtype = np.dtype(node[1])
      payload = b64decode(node[3], validate=True)
    except (BinasciiError, TypeError, ValueError):
      _malformed_carrier()
    if (
      dtype.str != node[1]
      or _dtype_has_metadata(dtype)
      or dtype.hasobject
      or dtype.fields is not None
      or dtype.subdtype is not None
      or dtype.kind not in "biuf"
      or (dtype.kind == "f" and _floating_dtype_exceeds_binary64(dtype))
      or len(payload) != dtype.itemsize * math.prod(node[2])
    ):
      _malformed_carrier()
    return
  if tag == "captured-manifest":
    if len(node) != 2 or type(node[1]) is not str:
      _malformed_carrier()
    try:
      captured = b64decode(node[1], validate=True)
    except (BinasciiError, ValueError):
      _malformed_carrier()
    if not captured.startswith(_MANIFEST_HEADER):
      _malformed_carrier()
    _validate_manifest_payload(captured[len(_MANIFEST_HEADER) :])
    return
  if tag == "unordered-declarations":
    if (
      len(node) != 3
      or type(node[1]) is not str
      or not node[1]
      or type(node[2]) is not tuple
    ):
      _malformed_carrier()
    for item in node[2]:
      _validate_canonical_node(item)
    return
  if tag == "mapping":
    if len(node) != 2 or type(node[1]) is not tuple:
      _malformed_carrier()
    previous_key: str | None = None
    for pair in node[1]:
      if (
        type(pair) is not tuple
        or len(pair) != 2
        or type(pair[0]) is not str
        or (previous_key is not None and pair[0] <= previous_key)
      ):
        _malformed_carrier()
      previous_key = pair[0]
      _validate_canonical_node(pair[1])
    return
  if tag == "sequence":
    if len(node) != 2 or type(node[1]) is not tuple:
      _malformed_carrier()
    for item in node[1]:
      _validate_canonical_node(item)
    return
  _malformed_carrier()


def _json_to_canonical_node(value: object) -> object:
  if type(value) is list:
    return tuple(_json_to_canonical_node(item) for item in value)
  if type(value) in (str, int, bool) or value is None:
    return value
  _malformed_carrier()


def _validate_manifest_payload(payload: bytes) -> None:
  try:
    node = _json_to_canonical_node(json.loads(payload))
    _validate_canonical_node(node)
    if _encode_node(node) != payload:
      _malformed_carrier()
  except (RecursionError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
    _malformed_carrier()


def _validated_manifest_payload(value: object) -> bytes:
  if type(value) is not CanonicalManifest:
    msg = "canonical manifest reuse requires an exact CanonicalManifest"
    raise TypeError(msg)
  try:
    payload = object.__getattribute__(value, "_payload")
  except AttributeError:
    _malformed_carrier()
  if type(payload) is not bytes:
    _malformed_carrier()
  _validate_manifest_payload(payload)
  return payload


def _validated_unordered_payload(
  value: object,
) -> tuple[str, tuple[_CanonicalNode, ...]]:
  if type(value) is not UnorderedDeclarations:
    msg = "unordered declaration reuse requires an exact carrier"
    raise TypeError(msg)
  try:
    id_key = object.__getattribute__(value, "id_key")
    items = object.__getattribute__(value, "_items")
  except AttributeError:
    _malformed_carrier()
  if type(id_key) is not str or not id_key or type(items) is not tuple:
    _malformed_carrier()
  for item in items:
    _validate_canonical_node(item)
  return id_key, items


def _validated_fingerprint_digest(value: object) -> str:
  if type(value) is not ContentFingerprint:
    msg = "content fingerprint reuse requires an exact ContentFingerprint"
    raise TypeError(msg)
  try:
    digest = object.__getattribute__(value, "digest")
  except AttributeError:
    _malformed_carrier()
  if (
    type(digest) is not str
    or len(digest) != 64
    or any(character not in "0123456789abcdef" for character in digest)
  ):
    msg = "content fingerprint must be a lowercase SHA-256 hex digest"
    raise ValueError(msg)
  return digest


def _canonical_numpy_float(value: np.floating) -> _CanonicalNode:
  if _floating_dtype_exceeds_binary64(value.dtype):
    msg = "extended NumPy floating scalars exceeding binary64 are unsupported"
    raise TypeError(msg)
  return _canonical_float(float(value))


def _canonical_array(
  value: np.ndarray[tuple[int, ...], np.dtype[np.generic]],
) -> _CanonicalNode:
  if type(value) is not np.ndarray:
    msg = "semantic manifest ndarray subclasses require explicit lowering"
    raise TypeError(msg)
  dtype = value.dtype
  if _dtype_has_metadata(dtype):
    msg = "semantic manifest arrays cannot use dtype metadata"
    raise TypeError(msg)
  if dtype.hasobject or dtype.fields is not None or dtype.subdtype is not None:
    msg = "semantic manifest arrays require a plain numeric dtype"
    raise TypeError(msg)
  if dtype.kind not in "biuf":
    msg = "semantic manifest arrays support bool, integer, and real float dtypes"
    raise TypeError(msg)
  if dtype.kind == "f" and _floating_dtype_exceeds_binary64(dtype):
    msg = "semantic manifest arrays do not support extended floating dtypes"
    raise TypeError(msg)
  if dtype.kind == "f" and not bool(np.isfinite(value).all()):
    msg = "semantic manifest arrays require finite floating-point values"
    raise ValueError(msg)

  canonical_dtype = dtype.newbyteorder("<")
  canonical = np.array(
    value,
    dtype=canonical_dtype,
    order="C",
    copy=True,
    subok=False,
  )
  payload = b64encode(canonical.tobytes(order="C")).decode("ascii")
  return (
    "ndarray",
    canonical_dtype.str,
    tuple(int(size) for size in canonical.shape),
    payload,
  )


def _enter_container(value: object, active: set[int]) -> int:
  marker = id(value)
  if marker in active:
    msg = "semantic manifests cannot contain reference cycles"
    raise ValueError(msg)
  active.add(marker)
  return marker


def _canonicalize(value: object, active: set[int]) -> _CanonicalNode:
  value_type = type(value)
  if value is None:
    return ("none",)
  if value_type is bool:
    return ("bool", value)
  if value_type is int:
    return ("int", str(value))
  if value_type is float:
    return _canonical_float(value)
  if value_type is str:
    return ("str", value)
  if any(value_type is scalar_type for scalar_type in _NUMPY_BOOL_TYPES):
    return ("bool", bool(value))
  if any(value_type is scalar_type for scalar_type in _NUMPY_INTEGER_TYPES):
    return ("int", str(int(value)))
  if any(value_type is scalar_type for scalar_type in _NUMPY_FLOAT_TYPES):
    return _canonical_numpy_float(value)
  if _type_derives_from(value_type, np.generic):
    msg = "semantic manifests require exact supported NumPy scalar base types"
    raise TypeError(msg)
  if _type_derives_from(value_type, np.ndarray):
    return _canonical_array(value)
  if value_type is CanonicalManifest:
    payload = _validated_manifest_payload(value)
    encoded = b64encode(_MANIFEST_HEADER + payload).decode("ascii")
    return ("captured-manifest", encoded)
  if value_type is UnorderedDeclarations:
    id_key, items = _validated_unordered_payload(value)
    return (
      "unordered-declarations",
      id_key,
      items,
    )
  if value_type is dict:
    marker = _enter_container(value, active)
    try:
      items: list[tuple[str, _CanonicalNode]] = []
      for key, item in value.items():
        if type(key) is not str:
          msg = "semantic manifest mapping keys must be exact strings"
          raise TypeError(msg)
        items.append((key, _canonicalize(item, active)))
      items.sort(key=lambda pair: pair[0])
      return ("mapping", tuple(items))
    finally:
      active.remove(marker)
  if value_type is list or value_type is tuple:
    marker = _enter_container(value, active)
    try:
      return (
        "sequence",
        tuple(_canonicalize(item, active) for item in value),
      )
    finally:
      active.remove(marker)

  msg = (
    "unsupported semantic manifest value; normalize it to the documented "
    "canonical boundary"
  )
  raise TypeError(msg)


@final
@dataclass(frozen=True, slots=True, init=False)
class UnorderedDeclarations:
  """Detached declarations sorted by an explicit stable semantic ID field."""

  id_key: str
  _items: tuple[_CanonicalNode, ...] = field(repr=False)

  def __init__(
    self,
    declarations: list[dict[str, object]] | tuple[dict[str, object], ...],
    *,
    id_key: str = "id",
  ) -> None:
    if type(id_key) is not str:
      msg = "unordered declaration ID key must be an exact string"
      raise TypeError(msg)
    if not id_key:
      msg = "unordered declaration ID key cannot be empty"
      raise ValueError(msg)
    if type(declarations) is not list and type(declarations) is not tuple:
      msg = "unordered declarations require an exact list/tuple container"
      raise TypeError(msg)

    captured: list[tuple[bytes, _CanonicalNode]] = []
    identities: set[bytes] = set()
    for declaration in declarations:
      if type(declaration) is not dict:
        msg = "unordered declarations must be exact string-keyed dictionaries"
        raise TypeError(msg)
      if id_key not in declaration:
        msg = f"unordered declaration is missing semantic ID field {id_key!r}"
        raise ValueError(msg)
      identity = _canonicalize(declaration[id_key], set())
      identity_bytes = _encode_node(identity)
      if identity_bytes in identities:
        msg = f"duplicate unordered declaration semantic ID in {id_key!r}"
        raise ValueError(msg)
      identities.add(identity_bytes)
      captured.append((identity_bytes, _canonicalize(declaration, set())))

    captured.sort(key=lambda item: item[0])
    object.__setattr__(self, "id_key", id_key)
    object.__setattr__(self, "_items", tuple(item for _, item in captured))

  def __init_subclass__(cls, **kwargs: object) -> None:
    del kwargs
    msg = "UnorderedDeclarations is runtime-final and cannot be subclassed"
    raise TypeError(msg)


@final
@dataclass(frozen=True, slots=True, init=False)
class CanonicalManifest:
  """Detached immutable bytes at the v1 canonical semantic boundary."""

  _payload: bytes = field(repr=False)

  def __init__(self, value: object) -> None:
    object.__setattr__(self, "_payload", _encode_node(_canonicalize(value, set())))

  def __init_subclass__(cls, **kwargs: object) -> None:
    del kwargs
    msg = "CanonicalManifest is runtime-final and cannot be subclassed"
    raise TypeError(msg)

  @classmethod
  def capture(cls, value: object) -> Self:
    """Capture ``value`` immediately into immutable canonical bytes."""
    return cls(value)

  def to_bytes(self) -> bytes:
    """Return the versioned bytes used by the fingerprint algorithm."""
    return _MANIFEST_HEADER + _validated_manifest_payload(self)

  def __bytes__(self) -> bytes:
    return self.to_bytes()


@final
@dataclass(frozen=True, slots=True, eq=False)
class ContentFingerprint:
  """Persistable SHA-256 identity of one canonical semantic manifest."""

  digest: str

  algorithm: ClassVar[str] = "sha256"
  manifest_format: ClassVar[str] = CANONICAL_MANIFEST_FORMAT

  def __post_init__(self) -> None:
    _validated_fingerprint_digest(self)

  def __init_subclass__(cls, **kwargs: object) -> None:
    del kwargs
    msg = "ContentFingerprint is runtime-final and cannot be subclassed"
    raise TypeError(msg)

  @classmethod
  def from_manifest(cls, manifest: CanonicalManifest) -> Self:
    """Fingerprint an already captured canonical manifest."""
    payload = _validated_manifest_payload(manifest)
    return cls(sha256(_MANIFEST_HEADER + payload).hexdigest())

  @classmethod
  def capture(cls, value: object) -> Self:
    """Capture and fingerprint one semantic value in a single operation."""
    return cls.from_manifest(CanonicalManifest(value))

  def __eq__(self, other: object) -> bool:
    if type(other) is not ContentFingerprint:
      return False
    return _validated_fingerprint_digest(self) == _validated_fingerprint_digest(other)

  def __ne__(self, other: object) -> bool:
    return not self == other

  def __hash__(self) -> int:
    return hash((_validated_fingerprint_digest(self),))

  def __str__(self) -> str:
    return f"{self.algorithm}:{_validated_fingerprint_digest(self)}"
