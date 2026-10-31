# SPDX-License-Identifier: MIT

"""Recursive nested-state composition codec: round-trip and fail-closed battery.

Mirrors the state-row and evolution codec batteries: the versioned
composition payload must round-trip byte-exactly at every level (virgin and
mid-run stores, heterogeneous two-level trees, and the zero-width edges),
and decoding must fail closed per level on any foreign, tampered, truncated,
padded, or non-finite payload — the codec never migrates.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.state import (
  NESTED_CODEC_FORMAT,
  NestedBlockLayout,
  NestedStateCodec,
  NestedStateLayout,
  NestedStateSnapshot,
  NestedStateStore,
  StateCodecError,
)

_MICRO_SCHEMA = "test-nested-micro-v1"
# An older version of the same schema family: foreign by the closed contract.
_OLDER_SCHEMA = "test-nested-micro-v0"
_OLDER_FORMAT = "pyfem-v3-state-nested-float64-v0"


def _micro_layout(
  ip_count: int = 3,
  schema: str = _MICRO_SCHEMA,
  coefficient_count: int = 18,
) -> NestedStateLayout:
  return NestedStateLayout(
    schema=schema,
    coefficient_count=coefficient_count,
    blocks=(
      NestedBlockLayout(
        schema="test-damage-v1",
        block_id="micro-cells",
        entity_count=16,
        row_width=1,
      ),
      NestedBlockLayout(
        schema="test-slip-v1",
        block_id=7,
        entity_count=4,
        row_width=2,
      ),
    ),
    ip_count=ip_count,
  )


def _two_level_layout() -> NestedStateLayout:
  child = NestedStateLayout(
    schema="test-nested-child-v1",
    coefficient_count=4,
    blocks=(
      NestedBlockLayout(
        schema="test-child-damage-v1",
        block_id="child-cells",
        entity_count=6,
        row_width=2,
      ),
    ),
    ip_count=2,
  )
  return NestedStateLayout(
    schema="test-nested-macro-v1",
    coefficient_count=3,
    blocks=(
      NestedBlockLayout(
        schema="test-macro-slip-v1",
        block_id="slip-springs",
        entity_count=2,
        row_width=2,
      ),
    ),
    ip_count=3,
    children=(child, None, child),
  )


def _rows(shape: tuple[int, ...], seed: float) -> FinalizedArray:
  values = seed + np.arange(math_prod(shape), dtype=np.float64).reshape(shape) * 0.25
  return FinalizedArray(values, dtype=np.float64)


def math_prod(shape: tuple[int, ...]) -> int:
  result = 1
  for extent in shape:
    result *= extent
  return result


def _snapshot(layout: NestedStateLayout, seed: float = 0.0) -> NestedStateSnapshot:
  children = tuple(
    None if child is None else _snapshot(child, seed + 100.0 + index)
    for index, child in enumerate(layout.children)
  )
  return NestedStateSnapshot(
    coefficients=_rows((layout.ip_count, layout.coefficient_count), seed),
    blocks=tuple(
      _rows((layout.ip_count, block.entity_count, block.row_width), seed + 10.0 + slot)
      for slot, block in enumerate(layout.blocks)
    ),
    generations=tuple(index * 2 + 1 for index in range(layout.ip_count)),
    children=children,
  )


def _codec(schema: str = _MICRO_SCHEMA) -> NestedStateCodec:
  return NestedStateCodec(schema)


def _assert_snapshots_bitwise(
  actual: NestedStateSnapshot,
  expected: NestedStateSnapshot,
) -> None:
  assert actual.coefficients.values.tobytes() == expected.coefficients.values.tobytes()
  assert len(actual.blocks) == len(expected.blocks)
  for actual_rows, expected_rows in zip(actual.blocks, expected.blocks, strict=True):
    assert actual_rows.values.tobytes() == expected_rows.values.tobytes()
  assert actual.generations == expected.generations
  assert len(actual.children) == len(expected.children)
  for actual_child, expected_child in zip(
    actual.children, expected.children, strict=True
  ):
    if expected_child is None:
      assert actual_child is None
    else:
      _assert_snapshots_bitwise(actual_child, expected_child)


def _tamper(payload: bytes, **changes: object) -> bytes:
  header, _, body = payload.partition(b"\n")
  declared = json.loads(header.decode("ascii"))
  declared.update(changes)
  tampered = json.dumps(
    declared,
    ensure_ascii=True,
    separators=(",", ":"),
    sort_keys=True,
  ).encode("ascii")
  return tampered + b"\n" + body


def test_leaf_snapshot_round_trips_byte_exact() -> None:
  layout = _micro_layout()
  codec = _codec()
  snapshot = _snapshot(layout)
  payload = codec.encode(layout, snapshot)
  header, _, body = payload.partition(b"\n")
  assert NESTED_CODEC_FORMAT.encode() in header
  assert _MICRO_SCHEMA.encode() in header
  assert b"test-damage-v1" in header and b"test-slip-v1" in header
  assert len(body) == 3 * 8 * (18 + 16 * 1 + 4 * 2)
  decoded = codec.decode(layout, payload)
  _assert_snapshots_bitwise(decoded, snapshot)
  assert codec.encode(layout, decoded) == payload


def test_zero_width_and_empty_levels_round_trip_byte_exact() -> None:
  cases = (
    NestedStateLayout(schema="z", coefficient_count=0, blocks=(), ip_count=0),
    NestedStateLayout(schema="z", coefficient_count=5, blocks=(), ip_count=2),
    NestedStateLayout(schema="z", coefficient_count=0, blocks=(), ip_count=3),
    NestedStateLayout(
      schema="z",
      coefficient_count=2,
      blocks=(
        NestedBlockLayout(schema="zb", block_id="b", entity_count=0, row_width=3),
      ),
      ip_count=2,
    ),
    NestedStateLayout(
      schema="z",
      coefficient_count=2,
      blocks=(
        NestedBlockLayout(schema="zb", block_id="b", entity_count=3, row_width=0),
      ),
      ip_count=2,
    ),
  )
  for layout in cases:
    codec = _codec("z")
    payload = codec.encode(layout, _snapshot(layout, seed=1.0))
    decoded = codec.decode(layout, payload)
    _assert_snapshots_bitwise(decoded, _snapshot(layout, seed=1.0))
    assert codec.encode(layout, decoded) == payload


def test_heterogeneous_two_level_round_trip_byte_exact() -> None:
  layout = _two_level_layout()
  codec = _codec("test-nested-macro-v1")
  snapshot = _snapshot(layout)
  payload = codec.encode(layout, snapshot)
  decoded = codec.decode(layout, payload)
  _assert_snapshots_bitwise(decoded, snapshot)
  assert codec.encode(layout, decoded) == payload
  # The composition header lists the nested schema registry of every level.
  header, _, body = payload.partition(b"\n")
  assert b"test-macro-slip-v1" in header
  child_codec = NestedStateCodec(layout.children[0].schema)
  child_zero = child_codec.encode(layout.children[0], snapshot.children[0])
  child_two = child_codec.encode(layout.children[2], snapshot.children[2])
  fixed_bytes = 3 * 8 * (3 + 2 * 2)
  assert body[fixed_bytes : fixed_bytes + len(child_zero)] == child_zero
  assert body[fixed_bytes + len(child_zero) :] == child_two
  assert b"test-child-damage-v1" in child_zero


def test_store_level_two_level_round_trip_preserves_all_generations() -> None:
  layout = _two_level_layout()
  child = layout.children[0]
  store = NestedStateStore(
    layout,
    children=(NestedStateStore(child), None, NestedStateStore(child)),
  )
  store.begin_trial()
  child_zero = store.children[0]
  child_zero.write_trial_coefficients(1, np.arange(4, dtype=np.float64))
  child_zero.write_trial_block(1, "child-cells", np.ones((6, 2)))
  child_zero.trial_complete(1)
  store.write_trial_coefficients(2, np.array([1.0, 2.0, 3.0]))
  store.write_trial_block(2, "slip-springs", np.full((2, 2), 0.25))
  store.trial_complete(2)
  store.commit_trial()
  payload = store.encode()
  restored = NestedStateStore.decode(layout, store.generations, payload)
  assert restored.encode() == payload
  # Child levels restore at fresh lineages rooted at the payload ordinals.
  assert [generation.ordinal for generation in restored.children[0].generations] == [
    0,
    1,
  ]
  assert restored.children[0] is not restored.children[2]
  again = NestedStateStore.decode(layout, restored.generations, restored.encode())
  assert again.encode() == payload


def test_header_is_canonical_json_with_exact_grammar() -> None:
  layout = NestedStateLayout(
    schema="grammar",
    coefficient_count=2,
    blocks=(NestedBlockLayout(schema="gb", block_id="b", entity_count=1, row_width=2),),
    ip_count=1,
  )
  payload = _codec("grammar").encode(layout, _snapshot(layout, seed=2.0))
  header, _, _ = payload.partition(b"\n")
  expected = json.dumps(
    {
      "blocks": [{"block_id": "b", "entity_count": 1, "row_width": 2, "schema": "gb"}],
      "child_bytes": [None],
      "coefficient_count": 2,
      "format": NESTED_CODEC_FORMAT,
      "generations": [1],
      "schema": "grammar",
    },
    ensure_ascii=True,
    separators=(",", ":"),
    sort_keys=True,
  ).encode("ascii")
  assert header == expected


def test_decode_rejects_foreign_structure_before_content() -> None:
  layout = _micro_layout()
  codec = _codec()
  payload = codec.encode(layout, _snapshot(layout))
  with pytest.raises(TypeError, match="exact bytes"):
    codec.decode(layout, bytearray(payload))
  with pytest.raises(StateCodecError, match="missing its versioned header line"):
    codec.decode(layout, b"no-header-line")
  with pytest.raises(StateCodecError, match="not canonical ASCII JSON"):
    codec.decode(layout, b"\xff\xfe\n" + payload.partition(b"\n")[2])
  with pytest.raises(StateCodecError, match="not canonical ASCII JSON"):
    codec.decode(layout, b"{not json}\n" + payload.partition(b"\n")[2])
  with pytest.raises(StateCodecError, match="contradicts the nested layout"):
    codec.decode(layout, b"[1,2,3]\n" + payload.partition(b"\n")[2])
  with pytest.raises(StateCodecError, match="contradicts the nested layout"):
    codec.decode(layout, _tamper(payload, extra="key"))
  with pytest.raises(StateCodecError, match="contradicts the nested layout"):
    codec.decode(layout, _tamper(payload, blocks=None))


def test_decode_rejects_foreign_format_schema_and_shape() -> None:
  layout = _micro_layout()
  codec = _codec()
  payload = codec.encode(layout, _snapshot(layout))
  with pytest.raises(StateCodecError, match="format is foreign"):
    codec.decode(layout, _tamper(payload, format=_OLDER_FORMAT))
  with pytest.raises(StateCodecError, match="schema contradicts"):
    codec.decode(layout, _tamper(payload, schema=_OLDER_SCHEMA))
  with pytest.raises(StateCodecError, match="coefficient count contradicts"):
    codec.decode(layout, _tamper(payload, coefficient_count=17))
  with pytest.raises(StateCodecError, match="coefficient count contradicts"):
    codec.decode(layout, _tamper(payload, coefficient_count="18"))
  with pytest.raises(StateCodecError, match="block registry contradicts"):
    codec.decode(layout, _tamper(payload, blocks=[]))
  with pytest.raises(StateCodecError, match="block header contradicts"):
    codec.decode(
      layout,
      _tamper(
        payload,
        blocks=[
          {"block_id": "micro-cells", "entity_count": 16, "row_width": 1},
          {"block_id": 7, "entity_count": 4, "row_width": 2, "schema": "test-slip-v1"},
        ],
      ),
    )
  wrong_block_schema = json.loads(payload.partition(b"\n")[0].decode("ascii"))["blocks"]
  pristine_blocks = [dict(entry) for entry in wrong_block_schema]
  wrong_block_schema[0] = {**wrong_block_schema[0], "schema": "test-damage-v0"}
  with pytest.raises(StateCodecError, match="block header contradicts"):
    codec.decode(layout, _tamper(payload, blocks=wrong_block_schema))
  wrong_block_id = pristine_blocks
  wrong_block_id[1]["block_id"] = "7"
  with pytest.raises(StateCodecError, match="block id contradicts"):
    codec.decode(layout, _tamper(payload, blocks=wrong_block_id))


def test_decode_rejects_malformed_generations_and_child_framing() -> None:
  layout = _two_level_layout()
  codec = _codec("test-nested-macro-v1")
  payload = codec.encode(layout, _snapshot(layout))
  with pytest.raises(StateCodecError, match="non-negative exact ints per point"):
    codec.decode(layout, _tamper(payload, generations=[1, 2]))
  with pytest.raises(StateCodecError, match="non-negative exact ints per point"):
    codec.decode(layout, _tamper(payload, generations=[1, -2, 3]))
  with pytest.raises(StateCodecError, match="non-negative exact ints per point"):
    codec.decode(layout, _tamper(payload, generations=[1, True, 3]))
  with pytest.raises(StateCodecError, match="non-negative exact ints per point"):
    codec.decode(layout, _tamper(payload, generations=[1, 2.0, 3]))
  with pytest.raises(StateCodecError, match="child framing contradicts"):
    codec.decode(layout, _tamper(payload, child_bytes=[None, None]))
  with pytest.raises(StateCodecError, match="child where the layout declares none"):
    codec.decode(layout, _tamper(payload, child_bytes=[0, 4, None]))
  declared = json.loads(payload.partition(b"\n")[0].decode("ascii"))
  missing_child = [declared["child_bytes"][0], None, None]
  with pytest.raises(StateCodecError, match="child lengths must be"):
    codec.decode(layout, _tamper(payload, child_bytes=missing_child))
  with pytest.raises(StateCodecError, match="child lengths must be"):
    codec.decode(layout, _tamper(payload, child_bytes=[-1, None, 0]))


def test_decode_rejects_truncated_padded_and_non_finite_bodies() -> None:
  layout = _micro_layout()
  codec = _codec()
  payload = codec.encode(layout, _snapshot(layout))
  header, _, body = payload.partition(b"\n")
  with pytest.raises(StateCodecError, match="byte count does not match"):
    codec.decode(layout, header + b"\n" + body[:-8])
  with pytest.raises(StateCodecError, match="byte count does not match"):
    codec.decode(layout, header + b"\n" + body + b"\x00" * 8)
  with pytest.raises(StateCodecError, match="byte count does not match"):
    codec.decode(layout, header + b"\n")
  poisoned = bytearray(body)
  poisoned[-8:] = np.float64(np.inf).tobytes()
  with pytest.raises(StateCodecError, match="non-finite values"):
    codec.decode(layout, header + b"\n" + bytes(poisoned))
  nan_coefficients = bytearray(body)
  nan_coefficients[:8] = np.float64(np.nan).tobytes()
  with pytest.raises(StateCodecError, match="non-finite values"):
    codec.decode(layout, header + b"\n" + bytes(nan_coefficients))


def test_decode_fails_closed_at_the_child_level() -> None:
  layout = _two_level_layout()
  codec = _codec("test-nested-macro-v1")
  snapshot = _snapshot(layout)
  payload = codec.encode(layout, snapshot)
  header, _, body = payload.partition(b"\n")
  child_codec = NestedStateCodec(layout.children[0].schema)
  child_payload = child_codec.encode(layout.children[0], snapshot.children[0])
  fixed_bytes = 3 * 8 * (3 + 2 * 2)
  assert body[fixed_bytes : fixed_bytes + len(child_payload)] == child_payload
  # A child payload carrying a foreign child schema is rejected by the child
  # level's own closed contract.
  tampered_child = _tamper(child_payload, schema="test-nested-child-v0")
  spliced = (
    body[:fixed_bytes] + tampered_child + body[fixed_bytes + len(child_payload) :]
  )
  declared = json.loads(header.decode("ascii"))
  declared["child_bytes"][0] = len(tampered_child)
  foreign_header = json.dumps(
    declared, ensure_ascii=True, separators=(",", ":"), sort_keys=True
  ).encode("ascii")
  with pytest.raises(StateCodecError, match="schema contradicts"):
    codec.decode(layout, foreign_header + b"\n" + spliced)
  # A declared child length that truncates the child payload fails at the
  # child level's byte-count check.
  truncated_child = child_payload[:-8]
  spliced = (
    body[:fixed_bytes] + truncated_child + body[fixed_bytes + len(child_payload) :]
  )
  declared["child_bytes"][0] = len(truncated_child)
  declared["child_bytes"][2] = len(child_payload)
  truncated_header = json.dumps(
    declared, ensure_ascii=True, separators=(",", ":"), sort_keys=True
  ).encode("ascii")
  with pytest.raises(StateCodecError, match="byte count does not match"):
    codec.decode(layout, truncated_header + b"\n" + spliced)
  # Non-finite child content is the child level's rejection.
  poisoned_child = bytearray(child_payload)
  poisoned_child[-8:] = np.float64(np.nan).tobytes()
  spliced = (
    body[:fixed_bytes]
    + bytes(poisoned_child)
    + body[fixed_bytes + len(child_payload) :]
  )
  with pytest.raises(StateCodecError, match="non-finite values"):
    codec.decode(layout, header + b"\n" + spliced)
  assert codec.encode(layout, codec.decode(layout, payload)) == payload


def test_failed_decodes_never_pollute_a_store() -> None:
  layout = _micro_layout()
  store = NestedStateStore(layout)
  store.begin_trial()
  store.write_trial_coefficients(0, np.ones(18))
  store.write_trial_block(0, "micro-cells", np.ones((16, 1)))
  store.write_trial_block(0, 7, np.ones((4, 2)))
  store.trial_complete(0)
  store.commit_trial()
  before = store.encode()
  header, _, body = before.partition(b"\n")
  for bad in (
    b"garbage",
    header + b"\n" + body[:-8],
    _tamper(before, schema=_OLDER_SCHEMA),
    _tamper(before, format=_OLDER_FORMAT),
  ):
    with pytest.raises((StateCodecError, TypeError)):
      NestedStateStore.decode(layout, store.generations, bad)
    assert store.encode() == before


def test_encode_validation_is_fail_closed() -> None:
  layout = _micro_layout()
  codec = _codec()
  snapshot = _snapshot(layout)
  with pytest.raises(TypeError, match="exact NestedStateLayout"):
    codec.encode("layout", snapshot)
  with pytest.raises(StateCodecError, match="does not match the nested layout schema"):
    _codec("other-schema").encode(layout, snapshot)
  with pytest.raises(TypeError, match="exact NestedStateSnapshot"):
    codec.encode(layout, "snapshot")
  with pytest.raises(StateCodecError, match="match the layout shape"):
    codec.encode(
      layout,
      NestedStateSnapshot(
        coefficients=_rows((2, 18), 0.0),
        blocks=snapshot.blocks,
        generations=(1, 2),
        children=(None, None),
      ),
    )
  with pytest.raises(StateCodecError, match="must be finite"):
    codec.encode(
      layout,
      NestedStateSnapshot(
        coefficients=FinalizedArray(np.full((3, 18), np.nan), dtype=np.float64),
        blocks=snapshot.blocks,
        generations=snapshot.generations,
        children=snapshot.children,
      ),
    )
  with pytest.raises(StateCodecError, match="block registry"):
    codec.encode(
      layout,
      NestedStateSnapshot(
        coefficients=snapshot.coefficients,
        blocks=(),
        generations=snapshot.generations,
        children=snapshot.children,
      ),
    )
  with pytest.raises(StateCodecError, match="non-negative exact ints per point"):
    codec.encode(
      layout,
      NestedStateSnapshot(
        coefficients=snapshot.coefficients,
        blocks=snapshot.blocks,
        generations=(1, -2, 3),
        children=snapshot.children,
      ),
    )
  two_level = _two_level_layout()
  macro_snapshot = _snapshot(two_level)
  with pytest.raises(StateCodecError, match="child where the layout declares none"):
    _codec("test-nested-macro-v1").encode(
      two_level,
      NestedStateSnapshot(
        coefficients=macro_snapshot.coefficients,
        blocks=macro_snapshot.blocks,
        generations=macro_snapshot.generations,
        children=(
          macro_snapshot.children[0],
          macro_snapshot.children[0],
          macro_snapshot.children[2],
        ),
      ),
    )
