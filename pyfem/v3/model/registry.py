"""Immutable registry descriptors and exact binding snapshots."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Self, final

from pyfem.v3.model.provenance import CanonicalManifest, ContentFingerprint

type RegistryKey = tuple[str, str]

REGISTRY_MANIFEST_SCHEMA = "pyfem-v3-registry-snapshot-v1"


def _validate_label(value: object, field_name: str) -> str:
  if type(value) is not str or not value:
    msg = f"registry descriptor {field_name} must be a non-empty exact string"
    raise ValueError(msg)
  return value


def _validate_key(value: object) -> RegistryKey:
  if (
    type(value) is not tuple
    or len(value) != 2
    or type(value[0]) is not str
    or type(value[1]) is not str
    or not value[0]
    or not value[1]
  ):
    msg = "registry keys must be non-empty exact (kind, name) string tuples"
    raise TypeError(msg)
  return value


def _validated_descriptor(
  value: object,
) -> tuple[RegistryDescriptor, RegistryKey, CanonicalManifest]:
  if type(value) is not RegistryDescriptor:
    msg = "registry snapshots require exact RegistryDescriptor values"
    raise TypeError(msg)
  try:
    kind = object.__getattribute__(value, "kind")
    name = object.__getattribute__(value, "name")
    version = object.__getattribute__(value, "version")
    implementation_id = object.__getattribute__(value, "implementation_id")
    metadata = object.__getattribute__(value, "metadata")
    binding = object.__getattribute__(value, "binding")
    manifest = object.__getattribute__(value, "manifest")
  except AttributeError:
    msg = "registry snapshot received a malformed exact descriptor"
    raise TypeError(msg) from None

  validated_kind = _validate_label(kind, "kind")
  validated_name = _validate_label(name, "name")
  validated_version = _validate_label(version, "version")
  validated_implementation = _validate_label(
    implementation_id,
    "implementation_id",
  )
  if type(metadata) is not CanonicalManifest:
    msg = "registry descriptor metadata must be an exact CanonicalManifest"
    raise TypeError(msg)
  if not callable(binding):
    msg = "registry descriptor binding must be callable"
    raise TypeError(msg)
  if type(manifest) is not CanonicalManifest:
    msg = "registry descriptor manifest must be an exact CanonicalManifest"
    raise TypeError(msg)

  expected_manifest = CanonicalManifest(
    {
      "implementation_id": validated_implementation,
      "key": [validated_kind, validated_name],
      "metadata": metadata,
      "version": validated_version,
    }
  )
  if manifest.to_bytes() != expected_manifest.to_bytes():
    msg = "registry descriptor manifest does not match its validated fields"
    raise ValueError(msg)
  return value, (validated_kind, validated_name), manifest


@final
@dataclass(frozen=True, slots=True, eq=False, init=False)
class RegistryDescriptor:
  """One immutable explicit kernel binding and its recorded semantic meaning.

  ``implementation_id`` is supplied by the implementation owner and must change
  whenever behavior changes. The callable is captured by exact reference; it is
  intentionally not introspected or content-hashed.
  """

  kind: str
  name: str
  version: str
  implementation_id: str
  metadata: CanonicalManifest
  binding: Callable[..., object] = field(repr=False)
  manifest: CanonicalManifest = field(repr=False)

  def __init__(
    self,
    *,
    kind: str,
    name: str,
    version: str,
    implementation_id: str,
    metadata: dict[str, object],
    binding: Callable[..., object],
  ) -> None:
    validated_kind = _validate_label(kind, "kind")
    validated_name = _validate_label(name, "name")
    validated_version = _validate_label(version, "version")
    validated_implementation = _validate_label(
      implementation_id,
      "implementation_id",
    )
    if type(metadata) is not dict:
      msg = "registry descriptor metadata must be an exact dictionary"
      raise TypeError(msg)
    if not callable(binding):
      msg = "registry descriptor binding must be callable"
      raise TypeError(msg)

    captured_metadata = CanonicalManifest(metadata)
    manifest = CanonicalManifest(
      {
        "implementation_id": validated_implementation,
        "key": [validated_kind, validated_name],
        "metadata": captured_metadata,
        "version": validated_version,
      }
    )
    object.__setattr__(self, "kind", validated_kind)
    object.__setattr__(self, "name", validated_name)
    object.__setattr__(self, "version", validated_version)
    object.__setattr__(self, "implementation_id", validated_implementation)
    object.__setattr__(self, "metadata", captured_metadata)
    object.__setattr__(self, "binding", binding)
    object.__setattr__(self, "manifest", manifest)

  def __init_subclass__(cls, **kwargs: object) -> None:
    del kwargs
    msg = "RegistryDescriptor is runtime-final and cannot be subclassed"
    raise TypeError(msg)

  @property
  def key(self) -> RegistryKey:
    """Return the explicit registry lookup key."""
    return self.kind, self.name


@final
@dataclass(frozen=True, slots=True, eq=False, init=False)
class RegistrySnapshot:
  """Detached selected registry bindings plus their canonical manifest."""

  descriptors: tuple[RegistryDescriptor, ...]
  manifest: CanonicalManifest
  fingerprint: ContentFingerprint

  def __init__(
    self,
    source: dict[RegistryKey, RegistryDescriptor],
    *,
    required: list[RegistryKey] | tuple[RegistryKey, ...] | None = None,
  ) -> None:
    if type(source) is not dict:
      msg = "registry snapshot source must be an exact dictionary"
      raise TypeError(msg)

    source_keys = tuple(_validate_key(raw_key) for raw_key in source)

    if required is None:
      requested = source_keys
    else:
      if type(required) is not list and type(required) is not tuple:
        msg = "registry required keys must use an exact list/tuple container"
        raise TypeError(msg)
      requested = required

    validated_keys: list[RegistryKey] = []
    seen: set[RegistryKey] = set()
    for raw_key in requested:
      key = _validate_key(raw_key)
      if key in seen:
        msg = f"registry snapshot requested duplicate key {key!r}"
        raise ValueError(msg)
      seen.add(key)
      validated_keys.append(key)

    captured: list[RegistryDescriptor] = []
    for key in sorted(validated_keys):
      try:
        descriptor = source[key]
      except KeyError as exc:
        msg = f"registry snapshot is missing required key {key!r}"
        raise KeyError(msg) from exc
      descriptor, descriptor_key, _ = _validated_descriptor(descriptor)
      if descriptor_key != key:
        msg = f"registry mapping key {key!r} does not match its descriptor"
        raise ValueError(msg)
      captured.append(descriptor)

    descriptors = tuple(captured)
    manifest = CanonicalManifest(
      {
        "descriptors": [descriptor.manifest for descriptor in descriptors],
        "schema": REGISTRY_MANIFEST_SCHEMA,
      }
    )
    object.__setattr__(self, "descriptors", descriptors)
    object.__setattr__(self, "manifest", manifest)
    object.__setattr__(
      self,
      "fingerprint",
      ContentFingerprint.from_manifest(manifest),
    )

  def __init_subclass__(cls, **kwargs: object) -> None:
    del kwargs
    msg = "RegistrySnapshot is runtime-final and cannot be subclassed"
    raise TypeError(msg)

  @classmethod
  def capture(
    cls,
    source: dict[RegistryKey, RegistryDescriptor],
    *,
    required: list[RegistryKey] | tuple[RegistryKey, ...] | None = None,
  ) -> Self:
    """Capture selected bindings without retaining the source mapping."""
    return cls(source, required=required)

  def resolve(self, kind: str, name: str) -> RegistryDescriptor:
    """Return the exact descriptor captured for ``(kind, name)``."""
    key = _validate_key((kind, name))
    for descriptor in self.descriptors:
      if descriptor.key == key:
        return descriptor
    msg = f"registry snapshot has no descriptor for {key!r}"
    raise KeyError(msg)
