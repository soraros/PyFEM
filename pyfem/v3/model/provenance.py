"""Versioned canonical semantic manifests and reproducible fingerprints.

The serialization boundary is deliberately narrow and explicit. Supported values
are ``None``, exact Python booleans/integers/finite floats/strings, equivalent NumPy
scalars no wider than binary64, exact plain numeric NumPy arrays without extended
floating dtypes, string-keyed mappings, ordered lists/tuples,
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
from base64 import b64encode
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from hashlib import sha256
from typing import ClassVar, Self

import numpy as np

CANONICAL_MANIFEST_FORMAT = "pyfem-v3-semantic-manifest-v1"
_MANIFEST_HEADER = f"{CANONICAL_MANIFEST_FORMAT}\n".encode()
_BINARY64_INFO = np.finfo(np.float64)

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
  if value is None:
    return ("none",)
  if type(value) is bool:
    return ("bool", value)
  if type(value) is int:
    return ("int", str(value))
  if type(value) is float:
    return _canonical_float(value)
  if type(value) is str:
    return ("str", value)
  if isinstance(value, np.bool_):
    return ("bool", bool(value))
  if isinstance(value, np.integer):
    return ("int", str(int(value)))
  if isinstance(value, np.floating):
    return _canonical_numpy_float(value)
  if isinstance(value, np.ndarray):
    return _canonical_array(value)
  if isinstance(value, CanonicalManifest):
    encoded = b64encode(value.to_bytes()).decode("ascii")
    return ("captured-manifest", encoded)
  if isinstance(value, UnorderedDeclarations):
    return (
      "unordered-declarations",
      value.id_key,
      value._items,  # noqa: SLF001 - private immutable canonical payload
    )
  if isinstance(value, Mapping):
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
  if isinstance(value, (list, tuple)):
    marker = _enter_container(value, active)
    try:
      return (
        "sequence",
        tuple(_canonicalize(item, active) for item in value),
      )
    finally:
      active.remove(marker)

  value_type = type(value).__qualname__
  msg = (
    f"unsupported semantic manifest value {value_type}; normalize it to the "
    "documented canonical boundary"
  )
  raise TypeError(msg)


@dataclass(frozen=True, slots=True, init=False)
class UnorderedDeclarations:
  """Detached declarations sorted by an explicit stable semantic ID field."""

  id_key: str
  _items: tuple[_CanonicalNode, ...] = field(repr=False)

  def __init__(
    self,
    declarations: Iterable[Mapping[str, object]],
    *,
    id_key: str = "id",
  ) -> None:
    if not id_key:
      msg = "unordered declaration ID key cannot be empty"
      raise ValueError(msg)

    captured: list[tuple[bytes, _CanonicalNode]] = []
    identities: set[bytes] = set()
    for declaration in declarations:
      if not isinstance(declaration, Mapping):
        msg = "unordered declarations must be string-keyed mappings"
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


@dataclass(frozen=True, slots=True, init=False)
class CanonicalManifest:
  """Detached immutable bytes at the v1 canonical semantic boundary."""

  _payload: bytes = field(repr=False)

  def __init__(self, value: object) -> None:
    object.__setattr__(self, "_payload", _encode_node(_canonicalize(value, set())))

  @classmethod
  def capture(cls, value: object) -> Self:
    """Capture ``value`` immediately into immutable canonical bytes."""
    return cls(value)

  def to_bytes(self) -> bytes:
    """Return the versioned bytes used by the fingerprint algorithm."""
    return _MANIFEST_HEADER + self._payload

  def __bytes__(self) -> bytes:
    return self.to_bytes()


@dataclass(frozen=True, slots=True)
class ContentFingerprint:
  """Persistable SHA-256 identity of one canonical semantic manifest."""

  digest: str

  algorithm: ClassVar[str] = "sha256"
  manifest_format: ClassVar[str] = CANONICAL_MANIFEST_FORMAT

  def __post_init__(self) -> None:
    if len(self.digest) != 64 or any(
      character not in "0123456789abcdef" for character in self.digest
    ):
      msg = "content fingerprint must be a lowercase SHA-256 hex digest"
      raise ValueError(msg)

  @classmethod
  def from_manifest(cls, manifest: CanonicalManifest) -> Self:
    """Fingerprint an already captured canonical manifest."""
    if not isinstance(manifest, CanonicalManifest):
      msg = "content fingerprints require a CanonicalManifest"
      raise TypeError(msg)
    return cls(sha256(manifest.to_bytes()).hexdigest())

  @classmethod
  def capture(cls, value: object) -> Self:
    """Capture and fingerprint one semantic value in a single operation."""
    return cls.from_manifest(CanonicalManifest(value))

  def __str__(self) -> str:
    return f"{self.algorithm}:{self.digest}"
