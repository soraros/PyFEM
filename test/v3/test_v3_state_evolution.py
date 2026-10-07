# SPDX-License-Identifier: MIT

"""Continuation evolution state codec: round-trip, fail-closed, store battery.

Mirrors the state-row codec battery idiom
(test_v3_state_transactions.py:792-854): the versioned payload must round-trip
byte-exactly (virgin baseline, mid-run, and zero reduced width), and decoding
must fail closed on every foreign, tampered, truncated, padded, or non-finite
payload — including older versions of the same schema family. The store
battery pins the virgin legacy constants, advance/generation tracking, and
reject-path byte-identity: failed decode attempts never mutate a store.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.identity import StateGeneration, require_same_generation
from pyfem.v3.state import (
  CONTINUATION_EVOLUTION_STATE_SCHEMA,
  EVOLUTION_CODEC_FORMAT,
  ContinuationEvolutionCodec,
  ContinuationEvolutionLayout,
  ContinuationEvolutionState,
  ContinuationEvolutionStore,
  StateCodecError,
)

_FINGERPRINT_A = "0" * 63 + "a"
_FINGERPRINT_B = "0" * 63 + "b"
# An older version of the same schema family: foreign by the closed contract.
_OLDER_SCHEMA = "pyfem-v3-evolution-state-arc-length-riks-v0"
_OLDER_FORMAT = "pyfem-v3-evolution-continuation-float64-v0"


def _layout(
  reduced_dof_count: int = 3,
  map_fingerprint: str = _FINGERPRINT_A,
  schema: str = CONTINUATION_EVOLUTION_STATE_SCHEMA,
) -> ContinuationEvolutionLayout:
  return ContinuationEvolutionLayout(
    schema=schema,
    reduced_dof_count=reduced_dof_count,
    map_fingerprint=map_fingerprint,
  )


def _mid_run_store(
  layout: ContinuationEvolutionLayout,
  generation: StateGeneration,
) -> ContinuationEvolutionStore:
  store = ContinuationEvolutionStore.virgin(layout, generation)
  store.advance(
    lam=2.5,
    da_prev=np.array([0.1, -0.2, 0.3][: layout.reduced_dof_count], dtype=np.float64),
    dlam_prev=0.75,
    factor=float(0.5**0.25),
    total_factor=1.5,
    cycle=7,
    generation=generation.next_accepted(),
  )
  return store


def _assert_states_bitwise(
  actual: ContinuationEvolutionState,
  expected: ContinuationEvolutionState,
) -> None:
  assert actual.lam == expected.lam
  assert actual.dlam_prev == expected.dlam_prev
  assert actual.factor == expected.factor
  assert actual.total_factor == expected.total_factor
  assert actual.cycle == expected.cycle
  assert type(actual.cycle) is int
  assert actual.da_prev.values.tobytes() == expected.da_prev.values.tobytes()
  assert not actual.da_prev.values.flags.writeable


def test_virgin_store_encodes_legacy_constants_and_round_trips() -> None:
  layout = _layout()
  generation = StateGeneration.initial()
  store = ContinuationEvolutionStore.virgin(layout, generation)
  assert store.generation is generation
  snapshot = store.snapshot()
  # The generation-zero evolution state IS the legacy continuation baseline.
  assert snapshot.lam == 1.0
  assert snapshot.dlam_prev == 1.0
  assert snapshot.factor == 1.0
  assert snapshot.total_factor == 1.0
  assert snapshot.cycle == 0
  np.testing.assert_array_equal(snapshot.da_prev.values, np.zeros(3))

  payload = store.encode()
  header, _, body = payload.partition(b"\n")
  assert EVOLUTION_CODEC_FORMAT.encode() in header
  assert CONTINUATION_EVOLUTION_STATE_SCHEMA.encode() in header
  assert len(body) == 8 * (4 + 3)

  decoded = ContinuationEvolutionStore.decode(layout, generation, payload)
  _assert_states_bitwise(decoded.snapshot(), snapshot)
  assert decoded.encode() == payload
  require_same_generation(decoded.generation, generation)


def test_mid_run_state_round_trip_byte_exact() -> None:
  layout = _layout()
  generation = StateGeneration.initial()
  store = _mid_run_store(layout, generation)
  snapshot = store.snapshot()
  payload = store.encode()

  codec = ContinuationEvolutionCodec(CONTINUATION_EVOLUTION_STATE_SCHEMA)
  decoded = codec.decode(layout, payload)
  _assert_states_bitwise(decoded, snapshot)
  assert codec.encode(layout, decoded) == payload

  decoded_store = ContinuationEvolutionStore.decode(
    layout,
    store.generation,
    payload,
  )
  _assert_states_bitwise(decoded_store.snapshot(), snapshot)
  assert decoded_store.encode() == payload
  require_same_generation(decoded_store.generation, store.generation)
  assert decoded_store.generation.ordinal == 1


def test_zero_reduced_width_round_trip() -> None:
  layout = _layout(reduced_dof_count=0)
  generation = StateGeneration.initial()
  store = ContinuationEvolutionStore.virgin(layout, generation)
  store.advance(
    lam=3.25,
    da_prev=np.zeros(0, dtype=np.float64),
    dlam_prev=1.25,
    factor=1.0,
    total_factor=2.0,
    cycle=4,
    generation=generation.next_accepted(),
  )
  payload = store.encode()
  _, _, body = payload.partition(b"\n")
  assert len(body) == 8 * 4
  decoded = ContinuationEvolutionStore.decode(layout, store.generation, payload)
  _assert_states_bitwise(decoded.snapshot(), store.snapshot())
  assert decoded.snapshot().da_prev.values.shape == (0,)
  assert decoded.encode() == payload


def test_header_is_canonical_ascii_json() -> None:
  layout = _layout()
  store = _mid_run_store(layout, StateGeneration.initial())
  codec = ContinuationEvolutionCodec(CONTINUATION_EVOLUTION_STATE_SCHEMA)
  payload = store.encode()
  header, _, _ = payload.partition(b"\n")

  expected = json.dumps(
    {
      "cycle": 7,
      "dtype": "<f8",
      "format": EVOLUTION_CODEC_FORMAT,
      "map_fingerprint": _FINGERPRINT_A,
      "reduced_dof_count": 3,
      "schema": CONTINUATION_EVOLUTION_STATE_SCHEMA,
    },
    ensure_ascii=True,
    separators=(",", ":"),
    sort_keys=True,
  ).encode("ascii")
  assert header == expected
  assert codec.decode(layout, payload).cycle == 7


def test_fail_closed_on_foreign_schema_and_format() -> None:
  layout = _layout()
  store = _mid_run_store(layout, StateGeneration.initial())
  payload = store.encode()
  header, _, body = payload.partition(b"\n")

  codec = ContinuationEvolutionCodec(CONTINUATION_EVOLUTION_STATE_SCHEMA)
  foreign = ContinuationEvolutionCodec(_OLDER_SCHEMA)
  with pytest.raises(StateCodecError, match="does not match"):
    foreign.encode(layout, store.snapshot())
  with pytest.raises(StateCodecError, match="does not match"):
    foreign.decode(layout, payload)
  older_layout = _layout(schema=_OLDER_SCHEMA)
  with pytest.raises(StateCodecError, match="does not match"):
    codec.encode(older_layout, store.snapshot())

  tampered_schema = (
    header.replace(
      CONTINUATION_EVOLUTION_STATE_SCHEMA.encode(),
      _OLDER_SCHEMA.encode(),
    )
    + b"\n"
    + body
  )
  with pytest.raises(StateCodecError, match="contradicts"):
    codec.decode(layout, tampered_schema)
  tampered_format = (
    header.replace(EVOLUTION_CODEC_FORMAT.encode(), _OLDER_FORMAT.encode())
    + b"\n"
    + body
  )
  with pytest.raises(StateCodecError, match="contradicts"):
    codec.decode(layout, tampered_format)


def test_fail_closed_on_header_malformations() -> None:
  layout = _layout()
  codec = ContinuationEvolutionCodec(CONTINUATION_EVOLUTION_STATE_SCHEMA)
  payload = _mid_run_store(layout, StateGeneration.initial()).encode()
  header, _, body = payload.partition(b"\n")

  with pytest.raises(StateCodecError, match="header line"):
    codec.decode(layout, b"payload-without-a-header-line")
  with pytest.raises(StateCodecError, match="canonical ASCII JSON"):
    codec.decode(layout, b"not-json\n" + body)
  with pytest.raises(StateCodecError, match="canonical ASCII JSON"):
    codec.decode(layout, b"\xff\xfe\n" + body)
  with pytest.raises(StateCodecError, match="contradicts"):
    codec.decode(layout, b"[1, 2, 3]\n" + body)

  tampered_width = header.replace(b'"reduced_dof_count":3', b'"reduced_dof_count":4')
  with pytest.raises(StateCodecError, match="contradicts"):
    codec.decode(layout, tampered_width + b"\n" + body)
  tampered_fingerprint = header.replace(
    _FINGERPRINT_A.encode(),
    _FINGERPRINT_B.encode(),
  )
  with pytest.raises(StateCodecError, match="contradicts"):
    codec.decode(layout, tampered_fingerprint + b"\n" + body)
  tampered_dtype = header.replace(b'"<f8"', b'">f8"')
  with pytest.raises(StateCodecError, match="contradicts"):
    codec.decode(layout, tampered_dtype + b"\n" + body)
  extra_key = header[:-1] + b',"extra":1}'
  with pytest.raises(StateCodecError, match="contradicts"):
    codec.decode(layout, extra_key + b"\n" + body)
  missing_cycle = header.replace(b'"cycle":7,', b"")
  with pytest.raises(StateCodecError, match="contradicts"):
    codec.decode(layout, missing_cycle + b"\n" + body)

  float_cycle = header.replace(b'"cycle":7', b'"cycle":7.0')
  with pytest.raises(StateCodecError, match="cycle"):
    codec.decode(layout, float_cycle + b"\n" + body)
  negative_cycle = header.replace(b'"cycle":7', b'"cycle":-1')
  with pytest.raises(StateCodecError, match="cycle"):
    codec.decode(layout, negative_cycle + b"\n" + body)
  string_cycle = header.replace(b'"cycle":7', b'"cycle":"7"')
  with pytest.raises(StateCodecError, match="cycle"):
    codec.decode(layout, string_cycle + b"\n" + body)
  bool_cycle = header.replace(b'"cycle":7', b'"cycle":true')
  with pytest.raises(StateCodecError, match="cycle"):
    codec.decode(layout, bool_cycle + b"\n" + body)


def test_fail_closed_on_body_malformations() -> None:
  layout = _layout()
  codec = ContinuationEvolutionCodec(CONTINUATION_EVOLUTION_STATE_SCHEMA)
  payload = _mid_run_store(layout, StateGeneration.initial()).encode()
  header, _, body = payload.partition(b"\n")

  with pytest.raises(StateCodecError, match="byte count"):
    codec.decode(layout, payload[:-4])
  with pytest.raises(StateCodecError, match="byte count"):
    codec.decode(layout, payload + b"\x00" * 8)
  poisoned_nan = header + b"\n" + np.array([np.nan] * 7).tobytes()
  with pytest.raises(StateCodecError, match="non-finite"):
    codec.decode(layout, poisoned_nan)
  poisoned_inf = header + b"\n" + np.array([np.inf] * 7).tobytes()
  with pytest.raises(StateCodecError, match="non-finite"):
    codec.decode(layout, poisoned_inf)
  with pytest.raises(TypeError, match="exact bytes"):
    codec.decode(layout, "not-bytes")  # type: ignore[arg-type]


def test_encode_validation() -> None:
  layout = _layout()
  codec = ContinuationEvolutionCodec(CONTINUATION_EVOLUTION_STATE_SCHEMA)
  state = _mid_run_store(layout, StateGeneration.initial()).snapshot()

  with pytest.raises(TypeError, match="exact ContinuationEvolutionLayout"):
    codec.encode(object(), state)  # type: ignore[arg-type]
  with pytest.raises(TypeError, match="exact ContinuationEvolutionState"):
    codec.encode(layout, object())  # type: ignore[arg-type]

  def _variant(**overrides: object) -> ContinuationEvolutionState:
    values: dict[str, object] = {
      "lam": 2.5,
      "da_prev": FinalizedArray(np.array([0.1, -0.2, 0.3]), dtype=np.float64),
      "dlam_prev": 0.75,
      "factor": 1.0,
      "total_factor": 1.5,
      "cycle": 7,
    }
    values.update(overrides)
    return ContinuationEvolutionState(**values)  # type: ignore[arg-type]

  with pytest.raises(TypeError, match="exact floats"):
    codec.encode(layout, _variant(lam=1))
  with pytest.raises(TypeError, match="exact int"):
    codec.encode(layout, _variant(cycle=7.0))
  with pytest.raises(TypeError, match="exact FinalizedArray"):
    codec.encode(layout, _variant(da_prev=np.zeros(3)))
  with pytest.raises(StateCodecError, match="reduced DOF count"):
    codec.encode(
      layout,
      _variant(da_prev=FinalizedArray(np.zeros(4), dtype=np.float64)),
    )
  with pytest.raises(StateCodecError, match="reduced DOF count"):
    codec.encode(
      layout,
      _variant(da_prev=FinalizedArray(np.zeros(3, dtype=np.float32), dtype=np.float32)),
    )
  with pytest.raises(StateCodecError, match="finite"):
    codec.encode(layout, _variant(lam=float("nan")))
  with pytest.raises(StateCodecError, match="finite"):
    codec.encode(
      layout,
      _variant(da_prev=FinalizedArray(np.array([0.1, np.inf, 0.3]), dtype=np.float64)),
    )
  with pytest.raises(StateCodecError, match="non-negative"):
    codec.encode(layout, _variant(cycle=-1))
  with pytest.raises(ValueError, match="non-empty exact string"):
    ContinuationEvolutionCodec("")


def test_layout_validation() -> None:
  with pytest.raises(ValueError, match="non-empty exact string"):
    _layout(schema="")
  with pytest.raises(ValueError, match="non-empty exact string"):
    _layout(schema=1)  # type: ignore[arg-type]
  with pytest.raises(ValueError, match="non-negative exact int"):
    _layout(reduced_dof_count=-1)
  with pytest.raises(ValueError, match="non-negative exact int"):
    _layout(reduced_dof_count=True)
  with pytest.raises(ValueError, match="SHA-256 hex digest"):
    _layout(map_fingerprint="not-a-digest")
  with pytest.raises(ValueError, match="SHA-256 hex digest"):
    _layout(map_fingerprint="A" * 64)
  with pytest.raises(ValueError, match="SHA-256 hex digest"):
    _layout(map_fingerprint="0" * 63)

  layout = _layout()
  assert layout == _layout()
  assert layout != _layout(map_fingerprint=_FINGERPRINT_B)
  assert layout != _layout(reduced_dof_count=4)


def test_map_fingerprint_foreignness_fails_closed() -> None:
  layout_a = _layout()
  layout_b = _layout(map_fingerprint=_FINGERPRINT_B)
  codec = ContinuationEvolutionCodec(CONTINUATION_EVOLUTION_STATE_SCHEMA)
  payload = _mid_run_store(layout_a, StateGeneration.initial()).encode()
  # The same state under a foreign map is meaningless: reduced increments do
  # not survive a fingerprint change, so decoding fails closed.
  with pytest.raises(StateCodecError, match="contradicts"):
    codec.decode(layout_b, payload)
  with pytest.raises(StateCodecError, match="contradicts"):
    ContinuationEvolutionStore.decode(
      layout_b,
      StateGeneration.initial(),
      payload,
    )


def test_store_construction_and_advance() -> None:
  layout = _layout()
  codec = ContinuationEvolutionCodec(CONTINUATION_EVOLUTION_STATE_SCHEMA)
  generation = StateGeneration.initial()
  state = _mid_run_store(layout, generation).snapshot()

  with pytest.raises(TypeError, match="exact StateGeneration"):
    ContinuationEvolutionStore(layout, state, object())  # type: ignore[arg-type]
  with pytest.raises(TypeError, match="exact ContinuationEvolutionState"):
    ContinuationEvolutionStore(layout, object(), generation)  # type: ignore[arg-type]
  with pytest.raises(StateCodecError, match="does not match"):
    ContinuationEvolutionStore(_layout(schema=_OLDER_SCHEMA), state, generation)

  store = ContinuationEvolutionStore.virgin(layout, generation)
  virgin_payload = store.encode()
  next_generation = generation.next_accepted()
  store.advance(
    lam=4.5,
    da_prev=np.array([1.0, 2.0, 3.0]),
    dlam_prev=1.5,
    factor=0.9,
    total_factor=1.1,
    cycle=3,
    generation=next_generation,
  )
  snapshot = store.snapshot()
  assert snapshot.lam == 4.5
  assert snapshot.dlam_prev == 1.5
  assert snapshot.factor == 0.9
  assert snapshot.total_factor == 1.1
  assert snapshot.cycle == 3
  np.testing.assert_array_equal(snapshot.da_prev.values, [1.0, 2.0, 3.0])
  assert not snapshot.da_prev.values.flags.writeable
  assert store.generation is next_generation
  assert store.encode() != virgin_payload
  # The store encodes exactly what the codec encodes for the same state.
  assert store.encode() == codec.encode(layout, snapshot)


def test_failed_decodes_leave_the_store_byte_identical() -> None:
  layout = _layout()
  generation = StateGeneration.initial()
  store = _mid_run_store(layout, generation)
  payload_before = store.encode()
  header, _, body = payload_before.partition(b"\n")

  rejections = (
    b"payload-without-a-header-line",
    b"not-json\n" + body,
    payload_before[:-4],
    payload_before + b"\x00" * 8,
    header.replace(_FINGERPRINT_A.encode(), _FINGERPRINT_B.encode()) + b"\n" + body,
    header + b"\n" + np.array([np.nan] * 7).tobytes(),
  )
  for rejected in rejections:
    with pytest.raises(StateCodecError):
      ContinuationEvolutionStore.decode(layout, generation, rejected)
  with pytest.raises(TypeError, match="exact bytes"):
    ContinuationEvolutionStore.decode(layout, generation, "not-bytes")  # type: ignore[arg-type]
  assert store.encode() == payload_before
  require_same_generation(store.generation, generation.next_accepted())
