# SPDX-License-Identifier: MIT

"""Dangerous cases for compiler-owned identity and provenance primitives."""

from __future__ import annotations

import sys
from copy import copy
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from pyfem.v3.model import (
  CanonicalManifest,
  ContentFingerprint,
  FinalizedArray,
  GenerationMismatchError,
  IdentityMismatchError,
  InstanceId,
  RegistryDescriptor,
  RegistrySnapshot,
  StateGeneration,
  UnorderedDeclarations,
  finalize_array,
  require_generation_successor,
  require_same_generation,
  require_same_instance,
)


def _original_kernel(value: float) -> float:
  return value + 1.0


def _replacement_kernel(value: float) -> float:
  return value - 1.0


class _SpoofedInstanceId(InstanceId):
  __slots__ = ()

  def __eq__(self, other: object) -> bool:
    return True

  def __ne__(self, other: object) -> bool:
    return False


class _SpoofedStateGeneration(StateGeneration):
  __slots__ = ()

  def __eq__(self, other: object) -> bool:
    return True

  def __ne__(self, other: object) -> bool:
    return False


def test_finalization_detaches_converts_contiguity_and_rejects_writes() -> None:
  source = np.arange(12, dtype=np.float32).reshape(3, 4)
  expected = source.astype(np.float64)

  finalized = finalize_array(source, dtype=np.float64, order="F")
  source.fill(-99.0)

  assert finalized.dtype == np.dtype(np.float64)
  assert finalized.flags.f_contiguous
  assert finalized.flags.owndata
  assert not finalized.flags.writeable
  assert not np.shares_memory(finalized, source)
  np.testing.assert_array_equal(finalized, expected)
  with pytest.raises(ValueError, match="read-only"):
    finalized[0, 0] = 10.0


def test_separate_finalizations_never_alias_matching_source_or_each_other() -> None:
  source = np.arange(12, dtype=np.float64).reshape(3, 4)

  first = finalize_array(source, dtype=np.float64, order="C")
  second = finalize_array(source, dtype=np.float64, order="C")
  source.fill(-1.0)

  assert first.flags.c_contiguous
  assert second.flags.c_contiguous
  assert first.flags.owndata
  assert second.flags.owndata
  assert not np.shares_memory(first, source)
  assert not np.shares_memory(second, source)
  assert not np.shares_memory(first, second)
  np.testing.assert_array_equal(first, np.arange(12).reshape(3, 4))
  np.testing.assert_array_equal(second, first)


def test_finalized_array_carrier_uses_identity_equality() -> None:
  source = np.array([1.0, 2.0], dtype=np.float64)

  first = FinalizedArray(source, dtype=np.float64)
  second = FinalizedArray(source, dtype=np.float64)
  source.fill(0.0)

  assert first == first
  assert (first == second) is False
  assert (first == np.array([1.0, 2.0])) is False
  assert (first != np.array([1.0, 2.0])) is True
  assert len({first, second}) == 2
  assert not np.shares_memory(first.values, second.values)
  np.testing.assert_array_equal(first.values, [1.0, 2.0])
  with pytest.raises(ValueError, match="read-only"):
    first.values[0] = 9.0


def test_finalization_rejects_object_arrays_that_cannot_detach_deeply() -> None:
  source = np.array([["mutable"]], dtype=object)

  with pytest.raises(TypeError, match="object dtype"):
    finalize_array(source, dtype=object)


def test_array_boundaries_reject_masked_array_semantics() -> None:
  data = np.array([1.0, 2.0], dtype=np.float64)
  first_mask = np.ma.array(data, mask=[False, True])
  second_mask = np.ma.array(data, mask=[True, False])

  assert not np.array_equal(first_mask.mask, second_mask.mask)
  for masked in (first_mask, second_mask):
    with pytest.raises(TypeError, match="ndarray subclasses"):
      CanonicalManifest({"values": masked})
    with pytest.raises(TypeError, match="exact plain ndarray"):
      finalize_array(masked, dtype=np.float64)


def test_plain_non_native_strided_arrays_retain_semantics() -> None:
  non_native_code = ">" if sys.byteorder == "little" else "<"
  non_native_dtype = np.dtype(np.float64).newbyteorder(non_native_code)
  storage = np.arange(24, dtype=np.float64).astype(non_native_dtype).reshape(4, 6)
  source = storage[:, ::2]
  expected = np.array(source, dtype=np.float64, order="C", copy=True)
  equivalent = np.ascontiguousarray(expected)

  assert type(source) is np.ndarray
  assert not source.flags.c_contiguous
  manifest = CanonicalManifest({"values": source})
  fingerprint = ContentFingerprint.from_manifest(manifest)
  assert fingerprint == ContentFingerprint.capture({"values": equivalent})

  finalized = finalize_array(source, dtype=np.float64, order="C")
  storage.fill(-1.0)

  assert type(finalized) is np.ndarray
  assert finalized.flags.c_contiguous
  assert finalized.flags.owndata
  assert not finalized.flags.writeable
  assert not np.shares_memory(finalized, source)
  np.testing.assert_array_equal(finalized, expected)
  assert ContentFingerprint.from_manifest(manifest) == fingerprint


def test_live_identity_is_distinct_from_equal_content_fingerprint() -> None:
  fingerprint_a = ContentFingerprint.capture(
    {"schema": "example-v1", "values": [1, 2, 3]}
  )
  fingerprint_b = ContentFingerprint.capture(
    {"values": [1, 2, 3], "schema": "example-v1"}
  )
  instance_a = InstanceId()
  instance_b = InstanceId()

  assert fingerprint_a == fingerprint_b
  assert instance_a != instance_b
  require_same_instance(instance_a, instance_a)
  with pytest.raises(IdentityMismatchError, match="exact same live instance"):
    require_same_instance(instance_a, instance_b)


def test_live_identity_helpers_reject_spoofed_subclasses_before_comparison() -> None:
  expected = InstanceId()
  logical_copy = copy(expected)
  spoofed = _SpoofedInstanceId()

  assert type(logical_copy) is InstanceId
  assert logical_copy is not expected
  assert logical_copy._token == expected._token
  assert logical_copy == expected
  assert spoofed._token != expected._token
  assert expected == spoofed
  require_same_instance(expected, logical_copy)
  with pytest.raises(IdentityMismatchError, match="exact same live instance"):
    require_same_instance(expected, spoofed)


def test_canonical_manifest_snapshots_arrays_and_nested_mappings() -> None:
  coordinates = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
  convention = {"length": "m", "stress": "Pa"}
  source: dict[str, object] = {
    "convention": convention,
    "coordinates": coordinates,
    "schema": "model-v1",
  }
  manifest = CanonicalManifest(source)
  fingerprint = ContentFingerprint.from_manifest(manifest)
  canonical_bytes = manifest.to_bytes()

  coordinates.fill(42.0)
  convention.clear()
  source.clear()

  assert manifest.to_bytes() == canonical_bytes
  assert ContentFingerprint.from_manifest(manifest) == fingerprint


def test_unordered_declarations_are_explicit_and_ordered_sequences_are_not() -> None:
  steel_parameters = np.array([210.0, 0.3], dtype=np.float64)
  steel: dict[str, object] = {"id": "steel", "parameters": steel_parameters}
  rubber: dict[str, object] = {"id": "rubber", "parameters": [1.0, 0.49]}
  declarations = [steel, rubber]
  captured_unordered = UnorderedDeclarations(declarations)
  captured_before_mutation = ContentFingerprint.capture(
    {"materials": captured_unordered, "schedule": ["ramp", "hold"]}
  )

  equivalent = ContentFingerprint.capture(
    {
      "materials": UnorderedDeclarations(
        [
          {"parameters": [1.0, 0.49], "id": "rubber"},
          {
            "parameters": np.array([210.0, 0.3], dtype=np.float64),
            "id": "steel",
          },
        ]
      ),
      "schedule": ["ramp", "hold"],
    }
  )

  steel_parameters.fill(-1.0)
  steel.clear()
  rubber.clear()
  declarations.clear()

  assert captured_before_mutation == equivalent
  assert (
    ContentFingerprint.capture(
      {"materials": captured_unordered, "schedule": ["ramp", "hold"]}
    )
    == equivalent
  )
  assert (
    ContentFingerprint.capture(
      {"materials": captured_unordered, "schedule": ["hold", "ramp"]}
    )
    != equivalent
  )
  assert (
    ContentFingerprint.capture(
      {
        "materials": UnorderedDeclarations(
          [
            {"id": "rubber", "parameters": [1.0, 0.49]},
            {"id": "steel", "parameters": [211.0, 0.3]},
          ]
        ),
        "schedule": ["ramp", "hold"],
      }
    )
    != equivalent
  )


def test_canonical_boundary_rejects_unsupported_values() -> None:
  with pytest.raises(TypeError, match="documented canonical boundary"):
    CanonicalManifest({"kernel": _original_kernel})
  with pytest.raises(ValueError, match="finite"):
    CanonicalManifest({"value": float("nan")})


def test_numpy_float_scalars_preserve_binary64_boundary_and_finiteness() -> None:
  for scalar_type in (np.float16, np.float32, np.float64):
    value = scalar_type(0.1)
    assert ContentFingerprint.capture(value) == ContentFingerprint.capture(float(value))
    with pytest.raises(ValueError, match="finite"):
      CanonicalManifest(scalar_type(np.nan))


def test_adjacent_extended_numpy_float_scalars_are_rejected_without_narrowing() -> None:
  extended = np.finfo(np.longdouble)
  binary64 = np.finfo(np.float64)
  if (
    extended.nmant <= binary64.nmant
    and extended.minexp >= binary64.minexp
    and extended.maxexp <= binary64.maxexp
  ):
    pytest.skip("np.longdouble does not exceed binary64 on this platform")

  first = np.longdouble(1.0)
  adjacent = np.nextafter(first, np.longdouble(2.0))
  assert first != adjacent
  assert float(first) == float(adjacent)

  for value in (first, adjacent):
    with pytest.raises(TypeError, match="exceeding binary64"):
      CanonicalManifest(value)
  with pytest.raises(TypeError, match="extended floating dtypes"):
    CanonicalManifest(np.array([first, adjacent], dtype=np.longdouble))


def test_unordered_declarations_reject_duplicate_ids() -> None:
  with pytest.raises(ValueError, match="duplicate"):
    UnorderedDeclarations([{"id": "same"}, {"id": "same"}])


def test_registry_snapshot_freezes_metadata_and_exact_binding() -> None:
  weights = np.array([1.0, 2.0], dtype=np.float64)
  components = ["xx", "yy", "xy"]
  metadata: dict[str, object] = {
    "components": components,
    "weights": weights,
  }
  original = RegistryDescriptor(
    kind="material",
    name="elastic",
    version="1",
    implementation_id="elastic-reference-sha256:1111",
    metadata=metadata,
    binding=_original_kernel,
  )
  replacement = RegistryDescriptor(
    kind="material",
    name="elastic",
    version="1",
    implementation_id="elastic-reference-sha256:2222",
    metadata={"components": ["xx"], "weights": [7.0]},
    binding=_replacement_kernel,
  )
  registry = {original.key: original}
  snapshot = RegistrySnapshot.capture(registry)
  frozen_manifest = snapshot.manifest.to_bytes()
  frozen_fingerprint = snapshot.fingerprint

  weights.fill(-1.0)
  components.clear()
  metadata.clear()
  registry[original.key] = replacement
  registry.clear()

  resolved = snapshot.resolve("material", "elastic")
  assert resolved is original
  assert resolved.binding(2.0) == 3.0
  assert snapshot.manifest.to_bytes() == frozen_manifest
  assert snapshot.fingerprint == frozen_fingerprint
  assert (
    RegistrySnapshot({replacement.key: replacement}).fingerprint != frozen_fingerprint
  )
  with pytest.raises(FrozenInstanceError):
    original.version = "2"


def test_registry_manifest_is_independent_of_source_mapping_order() -> None:
  material = RegistryDescriptor(
    kind="material",
    name="elastic",
    version="1",
    implementation_id="elastic-v1",
    metadata={},
    binding=_original_kernel,
  )
  formulation = RegistryDescriptor(
    kind="formulation",
    name="small-strain",
    version="3",
    implementation_id="small-strain-v3",
    metadata={},
    binding=_replacement_kernel,
  )

  forward = RegistrySnapshot({material.key: material, formulation.key: formulation})
  reverse = RegistrySnapshot({formulation.key: formulation, material.key: material})

  assert forward.fingerprint == reverse.fingerprint
  assert forward.manifest.to_bytes() == reverse.manifest.to_bytes()


def test_generation_helpers_fail_closed_for_foreign_values_and_lineages() -> None:
  owner = InstanceId()
  initial = StateGeneration.initial()
  accepted = initial.next_accepted()
  foreign = StateGeneration.initial().next_accepted()

  assert initial.ordinal == 0
  assert accepted.ordinal == 1
  require_same_generation(accepted, accepted)
  require_generation_successor(initial, accepted)

  with pytest.raises(GenerationMismatchError, match="exact accepted-state"):
    require_same_generation(accepted, foreign)
  with pytest.raises(GenerationMismatchError, match="related next"):
    require_generation_successor(initial, foreign)
  with pytest.raises(IdentityMismatchError, match="exact same live instance"):
    require_same_instance(owner, np.array([1, 2], dtype=np.int64))
  with pytest.raises(GenerationMismatchError, match="exact accepted-state"):
    require_same_generation(accepted, np.array([1, 2], dtype=np.int64))


def test_generation_helpers_reject_subclasses_with_spoofed_fields() -> None:
  initial = StateGeneration.initial()
  accepted = initial.next_accepted()
  spoofed = _SpoofedStateGeneration()
  object.__setattr__(spoofed, "_lineage", accepted._lineage)
  object.__setattr__(spoofed, "ordinal", accepted.ordinal)

  assert spoofed._lineage == accepted._lineage
  assert spoofed.ordinal == accepted.ordinal
  with pytest.raises(GenerationMismatchError, match="exact accepted-state"):
    require_same_generation(accepted, spoofed)
  with pytest.raises(GenerationMismatchError, match="related next"):
    require_generation_successor(initial, spoofed)
