"""Immutable registry descriptors and exact binding snapshots."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Self

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
    not isinstance(value, tuple)
    or len(value) != 2
    or type(value[0]) is not str
    or type(value[1]) is not str
    or not value[0]
    or not value[1]
  ):
    msg = "registry keys must be non-empty exact (kind, name) string tuples"
    raise TypeError(msg)
  return value


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
    metadata: Mapping[str, object],
    binding: Callable[..., object],
  ) -> None:
    validated_kind = _validate_label(kind, "kind")
    validated_name = _validate_label(name, "name")
    validated_version = _validate_label(version, "version")
    validated_implementation = _validate_label(
      implementation_id,
      "implementation_id",
    )
    if not isinstance(metadata, Mapping):
      msg = "registry descriptor metadata must be a mapping"
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

  @property
  def key(self) -> RegistryKey:
    """Return the explicit registry lookup key."""
    return self.kind, self.name


@dataclass(frozen=True, slots=True, eq=False, init=False)
class RegistrySnapshot:
  """Detached selected registry bindings plus their canonical manifest."""

  descriptors: tuple[RegistryDescriptor, ...]
  manifest: CanonicalManifest
  fingerprint: ContentFingerprint

  def __init__(
    self,
    source: Mapping[RegistryKey, RegistryDescriptor],
    *,
    required: Iterable[RegistryKey] | None = None,
  ) -> None:
    if not isinstance(source, Mapping):
      msg = "registry snapshot source must be a mapping"
      raise TypeError(msg)

    if required is None:
      requested = tuple(source.keys())
    else:
      requested = tuple(required)

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
      if not isinstance(descriptor, RegistryDescriptor):
        msg = f"registry value for {key!r} is not a RegistryDescriptor"
        raise TypeError(msg)
      if descriptor.key != key:
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

  @classmethod
  def capture(
    cls,
    source: Mapping[RegistryKey, RegistryDescriptor],
    *,
    required: Iterable[RegistryKey] | None = None,
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
