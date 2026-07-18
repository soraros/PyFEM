# SPDX-License-Identifier: MIT

"""Edge cases for compiler-owned identity and provenance primitives."""

from __future__ import annotations

import sys
from base64 import b64encode
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


class _CustomInstanceId(InstanceId):
  __slots__ = ()

  def __eq__(self, other: object) -> bool:
    return True

  def __ne__(self, other: object) -> bool:
    return False


class _CustomStateGeneration(StateGeneration):
  __slots__ = ()

  def __eq__(self, other: object) -> bool:
    return True

  def __ne__(self, other: object) -> bool:
    return False


class _RaisingConversionFloat64(np.float64):
  calls = 0

  def __bool__(self) -> bool:
    type(self).calls += 1
    raise AssertionError("NumPy float subclass conversion executed")

  def __float__(self) -> float:
    type(self).calls += 1
    raise AssertionError("NumPy float subclass conversion executed")

  def __int__(self) -> int:
    type(self).calls += 1
    raise AssertionError("NumPy float subclass conversion executed")


class _RaisingConversionInt64(np.int64):
  calls = 0

  def __bool__(self) -> bool:
    type(self).calls += 1
    raise AssertionError("NumPy integer subclass conversion executed")

  def __float__(self) -> float:
    type(self).calls += 1
    raise AssertionError("NumPy integer subclass conversion executed")

  def __int__(self) -> int:
    type(self).calls += 1
    raise AssertionError("NumPy integer subclass conversion executed")


class _RaisingArray(np.ndarray):
  calls = 0

  def __array__(self, dtype: object = None, copy: object = None) -> object:
    del dtype, copy
    type(self).calls += 1
    raise AssertionError("ndarray subclass conversion executed")

  def __iter__(self) -> object:
    type(self).calls += 1
    raise AssertionError("ndarray subclass iteration executed")


class _RaisingMaskedArray(np.ma.MaskedArray):
  calls = 0

  def __array__(self, dtype: object = None, copy: object = None) -> object:
    del dtype, copy
    type(self).calls += 1
    raise AssertionError("masked-array conversion executed")

  def __iter__(self) -> object:
    type(self).calls += 1
    raise AssertionError("masked-array iteration executed")


class _RaisingDictionary(dict[object, object]):
  calls = 0

  def __iter__(self) -> object:
    type(self).calls += 1
    raise AssertionError("dictionary subclass iteration executed")

  def __getitem__(self, key: object) -> object:
    del key
    type(self).calls += 1
    raise AssertionError("dictionary subclass lookup executed")

  def __contains__(self, key: object) -> bool:
    del key
    type(self).calls += 1
    raise AssertionError("dictionary subclass membership executed")

  def items(self) -> object:
    type(self).calls += 1
    raise AssertionError("dictionary subclass items executed")

  def keys(self) -> object:
    type(self).calls += 1
    raise AssertionError("dictionary subclass keys executed")


class _RaisingList(list[object]):
  calls = 0

  def __iter__(self) -> object:
    type(self).calls += 1
    raise AssertionError("list subclass iteration executed")

  def __getitem__(self, key: int | slice) -> object:
    del key
    type(self).calls += 1
    raise AssertionError("list subclass lookup executed")


class _RaisingTuple(tuple[object, ...]):
  calls = 0
  armed = False

  def __iter__(self) -> object:
    if not type(self).armed:
      return tuple.__iter__(self)
    type(self).calls += 1
    raise AssertionError("tuple subclass iteration executed")

  def __getitem__(self, key: int | slice) -> object:
    if not type(self).armed:
      return tuple.__getitem__(self, key)
    type(self).calls += 1
    raise AssertionError("tuple subclass lookup executed")

  def __len__(self) -> int:
    if not type(self).armed:
      return tuple.__len__(self)
    type(self).calls += 1
    raise AssertionError("tuple subclass length executed")

  def __hash__(self) -> int:
    if not type(self).armed:
      return tuple.__hash__(self)
    type(self).calls += 1
    raise AssertionError("tuple subclass hash executed")


class _RaisingString(str):
  calls = 0

  def __bool__(self) -> bool:
    type(self).calls += 1
    raise AssertionError("string subclass truth executed")

  def __eq__(self, other: object) -> bool:
    del other
    type(self).calls += 1
    raise AssertionError("string subclass equality executed")

  def __hash__(self) -> int:
    type(self).calls += 1
    raise AssertionError("string subclass hash executed")

  def __repr__(self) -> str:
    type(self).calls += 1
    raise AssertionError("string subclass representation executed")


class _RaisingObjectConversion:
  calls = 0

  def __float__(self) -> float:
    type(self).calls += 1
    raise AssertionError("object-array element conversion executed")


class _RaisingForeignValue:
  calls = 0

  def __getattribute__(self, name: str) -> object:
    del name
    type(self).calls += 1
    raise AssertionError("foreign value attribute access executed")


def _assert_exact_bool(value: object, expected: bool) -> None:
  assert type(value) is bool
  assert value is expected


def _manifest_from_payload_for_test(payload: bytes) -> CanonicalManifest:
  manifest = object.__new__(CanonicalManifest)
  object.__setattr__(manifest, "_payload", payload)
  return manifest


def _single_registry_snapshot() -> RegistrySnapshot:
  descriptor = RegistryDescriptor(
    kind="material",
    name="elastic",
    version="1",
    implementation_id="elastic-v1",
    metadata={"components": ["xx", "yy"]},
    binding=_original_kernel,
  )
  return RegistrySnapshot({descriptor.key: descriptor})


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


def test_identity_helpers_reject_custom_subclasses_before_comparison() -> None:
  expected = InstanceId()
  logical_copy = copy(expected)
  custom_value = _CustomInstanceId()

  assert type(logical_copy) is InstanceId
  assert logical_copy is not expected
  assert logical_copy._token == expected._token
  assert logical_copy == expected
  assert custom_value._token != expected._token
  assert expected == custom_value
  require_same_instance(expected, logical_copy)
  with pytest.raises(IdentityMismatchError, match="exact same live instance"):
    require_same_instance(expected, custom_value)


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
  original_key = original.key
  registry = {original_key: original}
  snapshot = RegistrySnapshot.capture(registry)
  captured = snapshot.resolve(*original_key)
  frozen_manifest = snapshot.manifest.to_bytes()
  frozen_fingerprint = snapshot.fingerprint
  frozen_descriptor_manifest = captured.manifest.to_bytes()
  frozen_metadata = captured.metadata.to_bytes()

  assert captured is not original
  assert captured.binding is _original_kernel
  with pytest.raises(FrozenInstanceError):
    original.version = "2"

  weights.fill(-1.0)
  components.clear()
  metadata.clear()
  object.__setattr__(original, "kind", "changed-material")
  object.__setattr__(original, "name", "changed-elastic")
  object.__setattr__(original, "binding", _replacement_kernel)
  object.__setattr__(
    object.__getattribute__(original, "metadata"),
    "_payload",
    object.__getattribute__(replacement.metadata, "_payload"),
  )
  object.__setattr__(original, "manifest", replacement.manifest)
  registry[original_key] = replacement
  registry.clear()

  resolved = snapshot.resolve("material", "elastic")
  assert resolved is captured
  assert resolved.key == original_key
  assert resolved.binding is _original_kernel
  assert resolved.binding(2.0) == 3.0
  assert resolved.metadata.to_bytes() == frozen_metadata
  assert resolved.manifest.to_bytes() == frozen_descriptor_manifest
  assert snapshot.manifest.to_bytes() == frozen_manifest
  assert snapshot.fingerprint == frozen_fingerprint
  assert (
    RegistrySnapshot({replacement.key: replacement}).fingerprint != frozen_fingerprint
  )


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
  assert tuple(descriptor.key for descriptor in forward.descriptors) == (
    formulation.key,
    material.key,
  )
  assert forward.resolve(*formulation.key).binding is _replacement_kernel
  assert forward.resolve(*material.key).binding is _original_kernel

  source = {material.key: material, formulation.key: formulation}
  selected = RegistrySnapshot(source, required=[material.key])
  source[material.key] = formulation
  source.clear()
  assert selected.resolve(*material.key).binding is _original_kernel
  with pytest.raises(KeyError, match="has no descriptor"):
    selected.resolve(*formulation.key)


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


def test_generation_helpers_reject_subclasses_with_custom_value_fields() -> None:
  initial = StateGeneration.initial()
  accepted = initial.next_accepted()
  custom_value = _CustomStateGeneration()
  object.__setattr__(custom_value, "_lineage", accepted._lineage)
  object.__setattr__(custom_value, "ordinal", accepted.ordinal)

  assert custom_value._lineage == accepted._lineage
  assert custom_value.ordinal == accepted.ordinal
  with pytest.raises(GenerationMismatchError, match="exact accepted-state"):
    require_same_generation(accepted, custom_value)
  with pytest.raises(GenerationMismatchError, match="related next"):
    require_generation_successor(initial, custom_value)


def test_numpy_scalar_subclasses_reject_before_conversion_at_every_boundary() -> None:
  for scalar_type, raw_value in (
    (_RaisingConversionFloat64, 0.25),
    (_RaisingConversionInt64, 7),
  ):
    value = scalar_type(raw_value)
    scalar_type.calls = 0

    for manifest_value in (
      value,
      [value],
      (value,),
      {"nested": [value]},
    ):
      with pytest.raises(TypeError, match="exact supported NumPy scalar"):
        CanonicalManifest(manifest_value)

    for array_value in (value, [value], (value,), [[value]]):
      with pytest.raises(TypeError, match="exact NumPy scalar"):
        finalize_array(array_value, dtype=np.float64)

    assert scalar_type.calls == 0


def test_nested_ndarray_subclasses_reject_before_conversion_or_iteration() -> None:
  custom = np.arange(3.0).view(_RaisingArray)
  masked = np.ma.array([1.0, 2.0], mask=[False, True]).view(_RaisingMaskedArray)

  for value in (custom, masked):
    type(value).calls = 0
    for source in ([value], (value,), [[value]]):
      with pytest.raises(TypeError, match="exact plain ndarray"):
        finalize_array(source, dtype=np.float64)
    assert type(value).calls == 0


def test_target_dtype_rejects_foreign_forms_before_any_hook_executes() -> None:
  class _ForeignDTypeLike:
    calls = 0

    def __getattribute__(self, name: str) -> object:
      del name
      type(self).calls += 1
      raise AssertionError("foreign dtype attribute access executed")

    def __iter__(self) -> object:
      type(self).calls += 1
      raise AssertionError("foreign dtype iteration executed")

    def __str__(self) -> str:
      type(self).calls += 1
      raise AssertionError("foreign dtype string conversion executed")

  foreign = _ForeignDTypeLike()
  _ForeignDTypeLike.calls = 0

  with pytest.raises(TypeError, match="exact dtype string"):
    finalize_array([1.0], dtype=foreign)
  assert _ForeignDTypeLike.calls == 0

  dtype_string = _RaisingString("float64")
  _RaisingString.calls = 0
  with pytest.raises(TypeError, match="exact dtype string"):
    finalize_array([1.0], dtype=dtype_string)
  assert _RaisingString.calls == 0


def test_object_source_array_rejects_before_element_conversion() -> None:
  source = np.array([_RaisingObjectConversion()], dtype=object)
  _RaisingObjectConversion.calls = 0

  with pytest.raises(TypeError, match="array sources.*object dtype"):
    finalize_array(source, dtype=np.float64)
  assert _RaisingObjectConversion.calls == 0


def test_foreign_values_reject_without_consulting_overridden_class_hooks() -> None:
  value = _RaisingForeignValue()
  _RaisingForeignValue.calls = 0

  with pytest.raises(TypeError, match="exact scalars"):
    finalize_array(value, dtype=np.float64)
  with pytest.raises(TypeError, match="documented canonical boundary"):
    CanonicalManifest(value)
  assert _RaisingForeignValue.calls == 0


def test_dtype_metadata_rejects_without_fingerprint_collision_or_aliasing() -> None:
  first_nested = {"units": ["m"]}
  second_nested = {"units": ["s"]}
  first_dtype = np.dtype(np.float64, metadata={"semantic": first_nested})
  second_dtype = np.dtype(np.float64, metadata={"semantic": second_nested})
  first = np.array([1.0, 2.0], dtype=first_dtype)
  second = np.array([1.0, 2.0], dtype=second_dtype)

  assert first.dtype.str == second.dtype.str
  assert first.dtype.metadata is not None
  assert first.dtype.metadata["semantic"] is first_nested
  for value in (first, second):
    with pytest.raises(TypeError, match="dtype metadata"):
      CanonicalManifest(value)
    with pytest.raises(TypeError, match="dtype metadata"):
      finalize_array(value, dtype=np.float64)

  for target_dtype in (first_dtype, second_dtype):
    with pytest.raises(TypeError, match="dtype metadata"):
      finalize_array([1.0, 2.0], dtype=target_dtype)

  first_nested["units"].append("mutated")
  second_nested.clear()


def test_nested_dtype_metadata_rejects_but_plain_structured_values_detach() -> None:
  nested_values: list[str] = []
  metadata_leaf = np.dtype(np.float64, metadata={"nested": nested_values})
  structured_metadata = np.dtype([("value", metadata_leaf)])
  subdtype_metadata = np.dtype((metadata_leaf, (2,)))

  metadata_source = np.array([(1.0,)], dtype=structured_metadata)
  plain_structured = np.dtype([("value", np.float64)])
  with pytest.raises(TypeError, match="dtype metadata"):
    CanonicalManifest(metadata_source)
  with pytest.raises(TypeError, match="dtype metadata"):
    finalize_array(metadata_source, dtype=plain_structured)
  for target_dtype in (structured_metadata, subdtype_metadata):
    with pytest.raises(TypeError, match="dtype metadata"):
      finalize_array([1.0], dtype=target_dtype)

  owned_dtype = np.dtype([("value", "<f8"), ("indices", "<i4", (2,))])
  source = np.array([(1.5, [2, 3]), (4.5, [5, 6])], dtype=owned_dtype)
  expected = source.copy()
  finalized = finalize_array(source, dtype=owned_dtype)
  source["value"].fill(-1.0)
  source["indices"].fill(-1)

  assert finalized.dtype == owned_dtype
  assert finalized.flags.c_contiguous
  assert finalized.flags.owndata
  assert not finalized.flags.writeable
  assert not np.shares_memory(finalized, source)
  np.testing.assert_array_equal(finalized, expected)


def test_finalized_array_comparisons_are_scalar_identity_in_both_orders() -> None:
  source = np.array([1.0, 2.0])
  first = FinalizedArray(source, dtype=np.float64)
  second = FinalizedArray(source, dtype=np.float64)
  plain = np.array([1.0, 2.0])
  unrelated = object()

  for left, right, identical in (
    (first, first, True),
    (first, second, False),
    (second, first, False),
    (first, plain, False),
    (plain, first, False),
    (first, unrelated, False),
    (unrelated, first, False),
  ):
    _assert_exact_bool(left == right, identical)
    _assert_exact_bool(left != right, not identical)

  for left, right, identical in (
    (first, first, True),
    (first, second, False),
    (first, plain, False),
    (plain, first, False),
  ):
    _assert_exact_bool(np.equal(left, right), identical)
    _assert_exact_bool(np.not_equal(left, right), not identical)

  assert hash(first) == object.__hash__(first)


def test_finalized_array_comparison_ufunc_kwargs_fail_without_mutating_out() -> None:
  carrier = FinalizedArray([1.0], dtype=np.float64)
  plain = np.array([1.0])
  output = np.array(False)

  with pytest.raises(TypeError, match="do not accept keyword"):
    np.equal(plain, carrier, out=output)
  assert output.item() is False
  with pytest.raises(TypeError, match="do not accept keyword"):
    np.not_equal(carrier, plain, where=True)


@pytest.mark.parametrize(
  "base",
  [
    FinalizedArray,
    CanonicalManifest,
    UnorderedDeclarations,
    ContentFingerprint,
    RegistryDescriptor,
    RegistrySnapshot,
  ],
)
def test_semantic_carriers_are_runtime_final(base: type[object]) -> None:
  with pytest.raises(TypeError, match="runtime-final"):
    type(f"Attempted{base.__name__}Subclass", (base,), {})


def test_manifest_v1_bytes_remain_unchanged_for_exact_values() -> None:
  manifest = CanonicalManifest({"a": [True, np.int16(2), np.float32(0.5)], "z": None})

  assert manifest.to_bytes() == (
    b"pyfem-v3-semantic-manifest-v1\n"
    b'["mapping",[["a",["sequence",[["bool",true],["int","2"],'
    b'["float","0x1.0000000000000p-1"]]]],["z",["none"]]]]'
  )


def test_huge_exact_integers_capture_without_changing_global_digit_limit() -> None:
  limit_before = sys.get_int_max_str_digits()
  for ordinary in (
    -1_000_000_001,
    -1_000_000_000,
    -999_999_999,
    -1,
    0,
    1,
    999_999_999,
    1_000_000_000,
    1_000_000_001,
  ):
    expected = (
      b"pyfem-v3-semantic-manifest-v1\n" + b'["int","' + str(ordinary).encode() + b'"]'
    )
    assert CanonicalManifest(ordinary).to_bytes() == expected

  exponent = 5_000
  positive = 10**exponent
  negative = -positive
  positive_text = "1" + "0" * exponent
  negative_text = "-" + positive_text

  assert CanonicalManifest(positive).to_bytes() == (
    b"pyfem-v3-semantic-manifest-v1\n" + b'["int","' + positive_text.encode() + b'"]'
  )
  assert CanonicalManifest(negative).to_bytes() == (
    b"pyfem-v3-semantic-manifest-v1\n" + b'["int","' + negative_text.encode() + b'"]'
  )

  unordered = UnorderedDeclarations([{"id": positive, "nested": [negative]}])
  nested = CanonicalManifest({"declarations": unordered, "value": positive})
  fingerprint = ContentFingerprint.from_manifest(nested)
  descriptor = RegistryDescriptor(
    kind="material",
    name="huge-id",
    version="1",
    implementation_id="huge-id-v1",
    metadata={"negative": negative, "positive": positive},
    binding=_original_kernel,
  )
  snapshot = RegistrySnapshot({descriptor.key: descriptor})

  assert len(fingerprint.digest) == 64
  assert positive_text.encode() in descriptor.metadata.to_bytes()
  assert negative_text.encode() in descriptor.metadata.to_bytes()
  assert len(snapshot.fingerprint.digest) == 64
  assert sys.get_int_max_str_digits() == limit_before


def test_manual_float_text_rejects_as_malformed_carrier() -> None:
  overflow = _manifest_from_payload_for_test(b'["float","0x1p999999999"]')

  with pytest.raises(TypeError, match="malformed exact canonical carrier"):
    overflow.to_bytes()

  canonical_payload = b'["float","0x1.0000000000000p+0"]'
  canonical = _manifest_from_payload_for_test(canonical_payload)
  assert canonical.to_bytes() == (
    b"pyfem-v3-semantic-manifest-v1\n" + canonical_payload
  )


def test_manual_ndarray_payload_requires_finite_canonical_bytes() -> None:
  nan_bytes = np.array([np.nan], dtype="<f8").tobytes()
  nan_payload = b'["ndarray","<f8",[1],"' + b64encode(nan_bytes) + b'"]'
  with pytest.raises(TypeError, match="malformed exact canonical carrier"):
    _manifest_from_payload_for_test(nan_payload).to_bytes()

  invalid_bool_payload = b'["ndarray","|b1",[1],"' + b64encode(b"\x02") + b'"]'
  with pytest.raises(TypeError, match="malformed exact canonical carrier"):
    _manifest_from_payload_for_test(invalid_bool_payload).to_bytes()

  big_endian_bytes = np.array([1.5], dtype=">f8").tobytes()
  big_endian_payload = b'["ndarray",">f8",[1],"' + b64encode(big_endian_bytes) + b'"]'
  with pytest.raises(TypeError, match="malformed exact canonical carrier"):
    _manifest_from_payload_for_test(big_endian_payload).to_bytes()

  finite_bytes = np.array([-0.0, 1.5], dtype="<f8").tobytes()
  finite_payload = b'["ndarray","<f8",[2],"' + b64encode(finite_bytes) + b'"]'
  assert _manifest_from_payload_for_test(finite_payload).to_bytes() == (
    b"pyfem-v3-semantic-manifest-v1\n" + finite_payload
  )
  bool_payload = b'["ndarray","|b1",[2],"' + b64encode(b"\x00\x01") + b'"]'
  assert _manifest_from_payload_for_test(bool_payload).to_bytes() == (
    b"pyfem-v3-semantic-manifest-v1\n" + bool_payload
  )


def test_manual_zero_payload_shape_must_be_numpy_constructible() -> None:
  intp_max = int(np.iinfo(np.intp).max)
  too_large = intp_max + 1
  oversized_dimension = b'["ndarray","|i1",[' + str(too_large).encode() + b',0],""]'
  with pytest.raises(TypeError, match="malformed exact canonical carrier"):
    _manifest_from_payload_for_test(oversized_dimension).to_bytes()

  max_dims = int(np._core.multiarray.MAXDIMS)
  excessive_rank = b'["ndarray","|i1",[' + b",".join([b"0"] * (max_dims + 1)) + b'],""]'
  with pytest.raises(TypeError, match="malformed exact canonical carrier"):
    _manifest_from_payload_for_test(excessive_rank).to_bytes()

  excessive_nonzero_product = (
    b'["ndarray","|i1",['
    + str(intp_max).encode()
    + b","
    + str(intp_max).encode()
    + b',0],""]'
  )
  with pytest.raises(TypeError, match="malformed exact canonical carrier"):
    _manifest_from_payload_for_test(excessive_nonzero_product).to_bytes()

  excessive_itemsize_product = (
    b'["ndarray","<f8",[' + str(intp_max).encode() + b',0],""]'
  )
  with pytest.raises(TypeError, match="malformed exact canonical carrier"):
    _manifest_from_payload_for_test(excessive_itemsize_product).to_bytes()

  supported = np.empty((0,) * max_dims, dtype=np.int8)
  manifest = CanonicalManifest(supported)
  assert ContentFingerprint.from_manifest(manifest) == ContentFingerprint.capture(
    supported
  )

  supported_wide = np.empty((intp_max, 0), dtype=np.int8)
  wide_manifest = CanonicalManifest(supported_wide)
  assert wide_manifest.to_bytes() == CanonicalManifest(supported_wide).to_bytes()


def test_unordered_payload_requires_id_keyed_unique_strictly_sorted_mappings() -> None:
  non_mapping_payload = b'["unordered-declarations","id",[["int","1"]]]'
  with pytest.raises(TypeError, match="malformed exact canonical carrier"):
    _manifest_from_payload_for_test(non_mapping_payload).to_bytes()

  missing_id = object.__new__(UnorderedDeclarations)
  object.__setattr__(missing_id, "id_key", "id")
  object.__setattr__(
    missing_id,
    "_items",
    (("mapping", (("value", ("int", "1")),)),),
  )
  with pytest.raises(TypeError, match="malformed exact canonical carrier"):
    CanonicalManifest(missing_id)

  declaration_a = ("mapping", (("id", ("str", "a")),))
  declaration_b = ("mapping", (("id", ("str", "b")),))
  for invalid_items in (
    (declaration_a, declaration_a),
    (declaration_b, declaration_a),
  ):
    invalid = object.__new__(UnorderedDeclarations)
    object.__setattr__(invalid, "id_key", "id")
    object.__setattr__(invalid, "_items", invalid_items)
    with pytest.raises(TypeError, match="malformed exact canonical carrier"):
      CanonicalManifest(invalid)

  ordinary = UnorderedDeclarations([{"id": "b", "value": 2}, {"id": "a", "value": 1}])
  ordinary_manifest = CanonicalManifest(ordinary)
  ordinary_payload = ordinary_manifest.to_bytes().split(b"\n", maxsplit=1)[1]
  assert (
    _manifest_from_payload_for_test(ordinary_payload).to_bytes()
    == ordinary_manifest.to_bytes()
  )

  reused = object.__new__(UnorderedDeclarations)
  object.__setattr__(reused, "id_key", ordinary.id_key)
  object.__setattr__(
    reused,
    "_items",
    object.__getattribute__(ordinary, "_items"),
  )
  assert CanonicalManifest(reused).to_bytes() == ordinary_manifest.to_bytes()


def test_manifest_containers_reject_subclasses_without_executing_overrides() -> None:
  mapping = _RaisingDictionary({"value": 1})
  sequence = _RaisingList([1, 2])
  tuple_sequence = _RaisingTuple((1, 2))
  declarations = _RaisingList([{"id": "a"}])
  declaration = _RaisingDictionary({"id": "a"})

  for custom_type in (_RaisingDictionary, _RaisingList):
    custom_type.calls = 0
  _RaisingTuple.calls = 0
  _RaisingTuple.armed = True
  with pytest.raises(TypeError, match="documented canonical boundary"):
    CanonicalManifest(mapping)
  with pytest.raises(TypeError, match="documented canonical boundary"):
    CanonicalManifest({"values": sequence})
  try:
    with pytest.raises(TypeError, match="documented canonical boundary"):
      CanonicalManifest({"values": tuple_sequence})
  finally:
    _RaisingTuple.armed = False
  with pytest.raises(TypeError, match="exact list/tuple"):
    UnorderedDeclarations(declarations)
  with pytest.raises(TypeError, match="exact string-keyed"):
    UnorderedDeclarations([declaration])
  assert _RaisingDictionary.calls == 0
  assert _RaisingList.calls == 0
  assert _RaisingTuple.calls == 0


def test_invalid_unordered_id_key_rejects_before_declaration_lookup() -> None:
  declaration = _RaisingDictionary({"id": "a"})
  id_key = _RaisingString("id")
  _RaisingDictionary.calls = 0
  _RaisingString.calls = 0

  with pytest.raises(TypeError, match="exact string"):
    UnorderedDeclarations([declaration], id_key=id_key)
  assert _RaisingDictionary.calls == 0
  assert _RaisingString.calls == 0
  with pytest.raises(TypeError, match="exact string"):
    UnorderedDeclarations([{"id": "a"}], id_key=1)
  with pytest.raises(ValueError, match="cannot be empty"):
    UnorderedDeclarations([{"id": "a"}], id_key="")


def test_malformed_exact_canonical_carriers_fail_deterministically() -> None:
  uninitialized_manifest = object.__new__(CanonicalManifest)
  with pytest.raises(TypeError, match="malformed exact canonical carrier"):
    uninitialized_manifest.to_bytes()
  with pytest.raises(TypeError, match="malformed exact canonical carrier"):
    CanonicalManifest({"captured": uninitialized_manifest})
  with pytest.raises(TypeError, match="malformed exact canonical carrier"):
    ContentFingerprint.from_manifest(uninitialized_manifest)

  invalid_manifest = object.__new__(CanonicalManifest)
  object.__setattr__(invalid_manifest, "_payload", b"{}")
  with pytest.raises(TypeError, match="malformed exact canonical carrier"):
    invalid_manifest.to_bytes()

  malformed_unordered = object.__new__(UnorderedDeclarations)
  object.__setattr__(malformed_unordered, "id_key", "id")
  object.__setattr__(malformed_unordered, "_items", ("not-a-node",))
  with pytest.raises(TypeError, match="malformed exact canonical carrier"):
    CanonicalManifest(malformed_unordered)

  malformed_fingerprint = object.__new__(ContentFingerprint)
  with pytest.raises(TypeError, match="malformed exact canonical carrier"):
    str(malformed_fingerprint)


def test_registry_rejects_polymorphic_inputs_without_executing_overrides() -> None:
  descriptor = RegistryDescriptor(
    kind="material",
    name="elastic",
    version="1",
    implementation_id="elastic-v1",
    metadata={},
    binding=_original_kernel,
  )
  source_subclass = _RaisingDictionary({descriptor.key: descriptor})
  required_subclass = _RaisingList([descriptor.key])
  metadata_subclass = _RaisingDictionary({"value": 1})

  _RaisingDictionary.calls = 0
  _RaisingList.calls = 0
  with pytest.raises(TypeError, match="exact dictionary"):
    RegistrySnapshot(source_subclass)
  with pytest.raises(TypeError, match="exact list/tuple"):
    RegistrySnapshot({descriptor.key: descriptor}, required=required_subclass)
  with pytest.raises(TypeError, match="exact dictionary"):
    RegistryDescriptor(
      kind="material",
      name="other",
      version="1",
      implementation_id="other-v1",
      metadata=metadata_subclass,
      binding=_original_kernel,
    )
  assert _RaisingDictionary.calls == 0
  assert _RaisingList.calls == 0


def test_registry_key_subclasses_reject_before_lookup_iteration_or_hash() -> None:
  descriptor = RegistryDescriptor(
    kind="material",
    name="elastic",
    version="1",
    implementation_id="elastic-v1",
    metadata={},
    binding=_original_kernel,
  )
  tuple_key = _RaisingTuple(descriptor.key)
  source = {tuple_key: descriptor}
  _RaisingTuple.calls = 0
  _RaisingTuple.armed = True
  try:
    with pytest.raises(TypeError, match="registry keys"):
      RegistrySnapshot(source)
  finally:
    _RaisingTuple.armed = False
  assert _RaisingTuple.calls == 0

  string_key = _RaisingString("material")
  _RaisingString.calls = 0
  with pytest.raises(TypeError, match="registry keys"):
    RegistrySnapshot(
      {descriptor.key: descriptor},
      required=[(string_key, "elastic")],
    )
  assert _RaisingString.calls == 0


def test_registry_rejects_malformed_exact_descriptor_before_field_behavior() -> None:
  malformed = object.__new__(RegistryDescriptor)
  with pytest.raises(TypeError, match="malformed exact descriptor"):
    RegistrySnapshot({("material", "elastic"): malformed})


def test_registry_rejects_exact_descriptor_whose_manifest_no_longer_matches() -> None:
  descriptor = RegistryDescriptor(
    kind="material",
    name="elastic",
    version="1",
    implementation_id="elastic-v1",
    metadata={},
    binding=_original_kernel,
  )
  stable_manifest = descriptor.manifest.to_bytes()
  object.__setattr__(descriptor, "version", "2")

  assert descriptor.manifest.to_bytes() == stable_manifest
  with pytest.raises(ValueError, match="manifest does not match"):
    RegistrySnapshot({descriptor.key: descriptor})


def test_registry_descriptor_cannot_subclass_to_drift_key_or_binding() -> None:
  with pytest.raises(TypeError, match="runtime-final"):
    type(
      "DriftingRegistryDescriptor",
      (RegistryDescriptor,),
      {
        "binding": property(lambda self: _replacement_kernel),
        "key": property(lambda self: ("material", "drifted")),
      },
    )


@pytest.mark.parametrize(
  "field_name",
  ["descriptors", "manifest", "fingerprint", "_binding_references"],
)
def test_registry_snapshot_missing_fields_fail_deterministically(
  field_name: str,
) -> None:
  snapshot = _single_registry_snapshot()
  object.__delattr__(snapshot, field_name)

  with pytest.raises(TypeError, match="malformed exact snapshot"):
    snapshot.resolve("material", "elastic")


@pytest.mark.parametrize(
  ("field_name", "altered", "message"),
  [
    ("descriptors", [], "descriptors must be an exact tuple"),
    ("manifest", object(), "manifest must be an exact CanonicalManifest"),
    ("fingerprint", object(), "fingerprint must be an exact ContentFingerprint"),
    (
      "_binding_references",
      [],
      "binding references must be an exact tuple",
    ),
  ],
)
def test_registry_snapshot_altered_field_types_fail_deterministically(
  field_name: str,
  altered: object,
  message: str,
) -> None:
  snapshot = _single_registry_snapshot()
  object.__setattr__(snapshot, field_name, altered)

  with pytest.raises(TypeError, match=message):
    snapshot.resolve("material", "elastic")


@pytest.mark.parametrize(
  "field_name",
  [
    "kind",
    "name",
    "version",
    "implementation_id",
    "metadata",
    "binding",
    "manifest",
  ],
)
def test_registry_snapshot_missing_nested_descriptor_fields_fail_deterministically(
  field_name: str,
) -> None:
  snapshot = _single_registry_snapshot()
  object.__delattr__(snapshot.descriptors[0], field_name)

  with pytest.raises(TypeError, match="malformed exact descriptor"):
    snapshot.resolve("material", "elastic")


@pytest.mark.parametrize(
  ("field_name", "altered", "message"),
  [
    ("kind", "formulation", "descriptor manifest does not match"),
    (
      "metadata",
      CanonicalManifest({"components": ["zz"]}),
      "descriptor manifest does not match",
    ),
    (
      "manifest",
      CanonicalManifest({"changed": True}),
      "descriptor manifest does not match",
    ),
    ("binding", _replacement_kernel, "selected binding reference changed"),
  ],
)
def test_registry_snapshot_altered_nested_descriptor_fields_fail_deterministically(
  field_name: str,
  altered: object,
  message: str,
) -> None:
  snapshot = _single_registry_snapshot()
  object.__setattr__(snapshot.descriptors[0], field_name, altered)

  with pytest.raises(ValueError, match=message):
    snapshot.resolve("material", "elastic")


def test_registry_snapshot_rejects_noncanonical_order_and_duplicate_keys() -> None:
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
    version="1",
    implementation_id="small-strain-v1",
    metadata={},
    binding=_replacement_kernel,
  )
  source = {material.key: material, formulation.key: formulation}

  reversed_snapshot = RegistrySnapshot(source)
  object.__setattr__(
    reversed_snapshot,
    "descriptors",
    tuple(reversed(reversed_snapshot.descriptors)),
  )
  with pytest.raises(ValueError, match="not canonically ordered"):
    reversed_snapshot.resolve(*material.key)

  duplicate_snapshot = RegistrySnapshot(source)
  first = duplicate_snapshot.descriptors[0]
  object.__setattr__(duplicate_snapshot, "descriptors", (first, first))
  with pytest.raises(ValueError, match="duplicate descriptor key"):
    duplicate_snapshot.resolve(*first.key)


def test_registry_snapshot_revalidates_manifest_fingerprint_and_bindings() -> None:
  altered_manifest = _single_registry_snapshot()
  object.__setattr__(
    altered_manifest,
    "manifest",
    CanonicalManifest({"changed": True}),
  )
  with pytest.raises(ValueError, match="snapshot manifest does not match"):
    altered_manifest.resolve("material", "elastic")

  altered_fingerprint = _single_registry_snapshot()
  object.__setattr__(
    altered_fingerprint,
    "fingerprint",
    ContentFingerprint.capture({"changed": True}),
  )
  with pytest.raises(ValueError, match="fingerprint does not match"):
    altered_fingerprint.resolve("material", "elastic")

  altered_binding_reference = _single_registry_snapshot()
  object.__setattr__(
    altered_binding_reference,
    "_binding_references",
    (_replacement_kernel,),
  )
  with pytest.raises(ValueError, match="selected binding reference changed"):
    altered_binding_reference.resolve("material", "elastic")
