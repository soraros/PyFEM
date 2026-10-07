"""Request-owned continuation evolution state and its versioned codec.

The continuation evolution state is the promoted Scope-A continuation
baseline of the arc-length driver: the committed ``{lam, da_prev, dlam_prev,
factor, total_factor, cycle}`` of one run, request-owned per the design
contracts (the evolution layout belongs to the typed request, not the model).
``da_prev`` is stored in REDUCED coordinates exactly as the predictor
consumes it, and the layout pins the reduced DOF count together with the
constraint-map content fingerprint: reduced increments are meaningless under
a foreign map, and full-space storage was rejected because back-projection
through ``P`` is inexact under MPC — a restart must reproduce the trajectory
bitwise.

Two-level versioning, parallel to the owner's state-row codecs: the component
schema ``pyfem-v3-evolution-state-arc-length-riks-v1`` versions the field set
and its semantics (a field or semantics change means a new schema and a new
codec class), while :data:`EVOLUTION_CODEC_FORMAT` versions the byte layout.
Decoding NEVER migrates: it fails closed on any foreign, tampered, truncated,
padded, or non-finite payload — including older versions of the same schema
family — and ``decode ∘ encode`` is byte-exact, including at zero reduced
width and at the virgin baseline (the legacy constants ``lam = 1.0``,
``dlam_prev = 1.0``, ``factor = total_factor = 1.0``, ``cycle = 0``).

The store carries the live :class:`StateGeneration` its accepted state was
advanced at, so in-process resume is a live-generation identity check against
the transaction owner. Cross-process lineage restore is deliberately out of
scope here; it belongs to the checkpoint-bundle follow-up.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np

from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.identity import StateGeneration
from pyfem.v3.state.codecs import StateCodecError

CONTINUATION_EVOLUTION_STATE_SCHEMA = "pyfem-v3-evolution-state-arc-length-riks-v1"
EVOLUTION_CODEC_FORMAT = "pyfem-v3-evolution-continuation-float64-v1"
_FLOAT64_DTYPE = np.dtype(np.float64).str

# The virgin evolution state encodes the legacy continuation initial values
# (RiksSolver.__init__: lam = 1.0, the lazy Dlamprev = 1.0, and
# factor = totalFactor = 1.0) at cycle 0 — the generation-zero state of this
# schema, round-tripped like any other.
_VIRGIN_LOAD_PARAMETER = 1.0
_VIRGIN_INCREMENT = 1.0
_VIRGIN_FACTOR = 1.0


@dataclass(frozen=True, slots=True)
class ContinuationEvolutionLayout:
  """Codec authority for the continuation evolution state.

  ``schema`` is the exact content schema string; ``reduced_dof_count`` the
  reduced-space width of ``da_prev``; ``map_fingerprint`` the persistable
  SHA-256 digest of the constraint map the reduced coordinates live in
  (``CompiledConstraintMap.content_fingerprint.digest``).
  """

  schema: str
  reduced_dof_count: int
  map_fingerprint: str

  def __post_init__(self) -> None:
    if type(self.schema) is not str or not self.schema:
      msg = "continuation evolution layout schema must be a non-empty exact string"
      raise ValueError(msg)
    if type(self.reduced_dof_count) is not int or self.reduced_dof_count < 0:
      msg = (
        "continuation evolution layout reduced DOF count must be a "
        "non-negative exact int"
      )
      raise ValueError(msg)
    digest = self.map_fingerprint
    if (
      type(digest) is not str
      or len(digest) != 64
      or any(character not in "0123456789abcdef" for character in digest)
    ):
      msg = (
        "continuation evolution layout map fingerprint must be a lowercase "
        "SHA-256 hex digest"
      )
      raise ValueError(msg)


@dataclass(frozen=True, slots=True, eq=False)
class ContinuationEvolutionState:
  """One accepted continuation evolution snapshot (Scope A).

  ``lam`` is the committed load parameter; ``da_prev`` the committed total
  reduced displacement increment of the last committed cycle; ``dlam_prev``
  its load-parameter part; ``factor`` the predictor multiplier the next
  cycle's first attempt uses; ``total_factor`` the legacy ``totalFactor``
  accumulator (never reset, ported verbatim); ``cycle`` the number of
  committed cycles of the run.
  """

  lam: float
  da_prev: FinalizedArray
  dlam_prev: float
  factor: float
  total_factor: float
  cycle: int


def _validated_layout(
  codec_schema: str,
  layout: object,
) -> ContinuationEvolutionLayout:
  if type(layout) is not ContinuationEvolutionLayout:
    msg = "continuation evolution codecs require an exact ContinuationEvolutionLayout"
    raise TypeError(msg)
  if layout.schema != codec_schema:
    msg = (
      "continuation evolution codec schema does not match the evolution layout schema"
    )
    raise StateCodecError(msg)
  return layout


def _validated_state(
  layout: ContinuationEvolutionLayout,
  state: object,
) -> ContinuationEvolutionState:
  if type(state) is not ContinuationEvolutionState:
    msg = "continuation evolution encoding requires an exact ContinuationEvolutionState"
    raise TypeError(msg)
  for scalar in (state.lam, state.dlam_prev, state.factor, state.total_factor):
    if type(scalar) is not float:
      msg = "continuation evolution scalar fields must be exact floats"
      raise TypeError(msg)
  if type(state.cycle) is not int:
    msg = "continuation evolution cycle must be an exact int"
    raise TypeError(msg)
  if type(state.da_prev) is not FinalizedArray:
    msg = "continuation evolution da_prev must be an exact FinalizedArray"
    raise TypeError(msg)
  values = state.da_prev.values
  if (
    values.dtype != np.dtype(np.float64)
    or values.dtype.metadata is not None
    or values.shape != (layout.reduced_dof_count,)
  ):
    msg = (
      "continuation evolution da_prev must match the layout reduced DOF "
      "count and float64 dtype"
    )
    raise StateCodecError(msg)
  if not (
    math.isfinite(state.lam)
    and math.isfinite(state.dlam_prev)
    and math.isfinite(state.factor)
    and math.isfinite(state.total_factor)
    and bool(np.isfinite(values).all())
  ):
    msg = "continuation evolution state must be finite to enter a codec payload"
    raise StateCodecError(msg)
  if state.cycle < 0:
    msg = "continuation evolution cycle must be non-negative"
    raise StateCodecError(msg)
  return state


class ContinuationEvolutionCodec:
  """Stock versioned codec for the continuation evolution state.

  The payload is one canonical ASCII JSON header line recording the codec
  format, the content schema, the dtype, the reduced DOF count, the
  constraint-map fingerprint, and the cycle, followed by the exact C-order
  float64 bytes of ``[lam, dlam_prev, factor, total_factor, *da_prev]``. All
  floats live in the body for bitwise safety; ``cycle`` is an exact JSON int
  in the header. Decoding fails closed on any foreign or malformed content
  and reproduces the encoded state byte-exactly, including at zero reduced
  width.
  """

  def __init__(self, schema: str) -> None:
    if type(schema) is not str or not schema:
      msg = "continuation evolution codec schema must be a non-empty exact string"
      raise ValueError(msg)
    self._schema = schema

  @property
  def schema(self) -> str:
    return self._schema

  def encode(
    self,
    layout: ContinuationEvolutionLayout,
    state: ContinuationEvolutionState,
  ) -> bytes:
    validated_layout = _validated_layout(self._schema, layout)
    validated_state = _validated_state(validated_layout, state)
    header = json.dumps(
      {
        "cycle": validated_state.cycle,
        "dtype": _FLOAT64_DTYPE,
        "format": EVOLUTION_CODEC_FORMAT,
        "map_fingerprint": validated_layout.map_fingerprint,
        "reduced_dof_count": validated_layout.reduced_dof_count,
        "schema": self._schema,
      },
      ensure_ascii=True,
      separators=(",", ":"),
      sort_keys=True,
    ).encode("ascii")
    body = np.concatenate(
      (
        np.array(
          [
            validated_state.lam,
            validated_state.dlam_prev,
            validated_state.factor,
            validated_state.total_factor,
          ],
          dtype=np.float64,
        ),
        validated_state.da_prev.values,
      )
    ).tobytes(order="C")
    return header + b"\n" + body

  def decode(
    self,
    layout: ContinuationEvolutionLayout,
    payload: bytes,
  ) -> ContinuationEvolutionState:
    validated_layout = _validated_layout(self._schema, layout)
    if type(payload) is not bytes:
      msg = "continuation evolution payloads must be exact bytes"
      raise TypeError(msg)
    header, separator, body = payload.partition(b"\n")
    if not separator:
      msg = "continuation evolution payload is missing its versioned header line"
      raise StateCodecError(msg)
    try:
      declared = json.loads(header.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
      msg = "continuation evolution payload header is not canonical ASCII JSON"
      raise StateCodecError(msg) from exc
    expected = {
      "dtype": _FLOAT64_DTYPE,
      "format": EVOLUTION_CODEC_FORMAT,
      "map_fingerprint": validated_layout.map_fingerprint,
      "reduced_dof_count": validated_layout.reduced_dof_count,
      "schema": self._schema,
    }
    if (
      type(declared) is not dict
      or set(declared) != {"cycle", *expected}
      or {key: value for key, value in declared.items() if key != "cycle"} != expected
    ):
      msg = "continuation evolution payload header contradicts the evolution layout"
      raise StateCodecError(msg)
    cycle = declared["cycle"]
    if type(cycle) is not int or cycle < 0:
      msg = "continuation evolution payload cycle must be a non-negative exact int"
      raise StateCodecError(msg)
    if len(body) != 8 * (4 + validated_layout.reduced_dof_count):
      msg = "continuation evolution payload byte count does not match the layout"
      raise StateCodecError(msg)
    values = np.frombuffer(body, dtype=np.dtype(np.float64))
    if not bool(np.isfinite(values).all()):
      msg = "continuation evolution payload carries non-finite values"
      raise StateCodecError(msg)
    return ContinuationEvolutionState(
      lam=float(values[0]),
      da_prev=FinalizedArray(values[4:], dtype=np.float64),
      dlam_prev=float(values[1]),
      factor=float(values[2]),
      total_factor=float(values[3]),
      cycle=cycle,
    )


class ContinuationEvolutionStore:
  """Request-owned accepted continuation evolution state plus live generation.

  The store holds the mutable fields of ONE accepted
  :class:`ContinuationEvolutionState` together with the live
  :class:`StateGeneration` they were advanced at. It is written ONLY on the
  driver commit path via :meth:`advance`, immediately after the owner commit;
  reject paths never touch it, so a rejected or cut-back attempt leaves both
  the owner's committed state and this store byte-identical.
  """

  __slots__ = (
    "_codec",
    "_cycle",
    "_da_prev",
    "_dlam_prev",
    "_factor",
    "_generation",
    "_lam",
    "_layout",
    "_total_factor",
  )

  def __init__(
    self,
    layout: ContinuationEvolutionLayout,
    state: ContinuationEvolutionState,
    generation: StateGeneration,
  ) -> None:
    codec = ContinuationEvolutionCodec(CONTINUATION_EVOLUTION_STATE_SCHEMA)
    validated_layout = _validated_layout(codec.schema, layout)
    validated_state = _validated_state(validated_layout, state)
    if type(generation) is not StateGeneration:
      msg = "continuation evolution stores require an exact StateGeneration"
      raise TypeError(msg)
    self._layout = validated_layout
    self._codec = codec
    self._lam = validated_state.lam
    self._da_prev = np.array(
      validated_state.da_prev.values,
      dtype=np.float64,
      order="C",
      copy=True,
    )
    self._da_prev.setflags(write=False)
    self._dlam_prev = validated_state.dlam_prev
    self._factor = validated_state.factor
    self._total_factor = validated_state.total_factor
    self._cycle = validated_state.cycle
    self._generation = generation

  @classmethod
  def virgin(
    cls,
    layout: ContinuationEvolutionLayout,
    generation: StateGeneration,
  ) -> ContinuationEvolutionStore:
    """Create the generation-zero evolution store (the legacy constants)."""
    validated_layout = _validated_layout(CONTINUATION_EVOLUTION_STATE_SCHEMA, layout)
    state = ContinuationEvolutionState(
      lam=_VIRGIN_LOAD_PARAMETER,
      da_prev=FinalizedArray(
        np.zeros(validated_layout.reduced_dof_count, dtype=np.float64),
        dtype=np.float64,
      ),
      dlam_prev=_VIRGIN_INCREMENT,
      factor=_VIRGIN_FACTOR,
      total_factor=_VIRGIN_FACTOR,
      cycle=0,
    )
    return cls(validated_layout, state, generation)

  @classmethod
  def decode(
    cls,
    layout: ContinuationEvolutionLayout,
    generation: StateGeneration,
    payload: bytes,
  ) -> ContinuationEvolutionStore:
    """Decode one payload into a store rooted at the supplied live generation.

    The payload deliberately carries no generation: accepted-state
    generations are process-local lineage identities, so the caller supplies
    the live generation this payload was captured at. Decoding fails closed
    on any foreign or tampered payload and never migrates.
    """
    codec = ContinuationEvolutionCodec(CONTINUATION_EVOLUTION_STATE_SCHEMA)
    state = codec.decode(layout, payload)
    return cls(layout, state, generation)

  @property
  def layout(self) -> ContinuationEvolutionLayout:
    """Return the codec authority this store is bound to."""
    return self._layout

  @property
  def generation(self) -> StateGeneration:
    """Return the live generation the accepted state was advanced at."""
    return self._generation

  @property
  def lam(self) -> float:
    """Return the committed load parameter."""
    return self._lam

  @property
  def da_prev(self) -> np.ndarray:
    """Return the committed reduced increment (read-only)."""
    return self._da_prev

  @property
  def dlam_prev(self) -> float:
    """Return the committed load-parameter increment."""
    return self._dlam_prev

  @property
  def factor(self) -> float:
    """Return the predictor multiplier the next attempt uses."""
    return self._factor

  @property
  def total_factor(self) -> float:
    """Return the legacy ``totalFactor`` accumulator."""
    return self._total_factor

  @property
  def cycle(self) -> int:
    """Return the number of committed cycles of the run."""
    return self._cycle

  def snapshot(self) -> ContinuationEvolutionState:
    """Return the immutable typed view of the accepted evolution state."""
    return ContinuationEvolutionState(
      lam=self._lam,
      da_prev=FinalizedArray(
        np.array(self._da_prev, dtype=np.float64, order="C", copy=True),
        dtype=np.float64,
      ),
      dlam_prev=self._dlam_prev,
      factor=self._factor,
      total_factor=self._total_factor,
      cycle=self._cycle,
    )

  def advance(
    self,
    *,
    lam: float,
    da_prev: np.ndarray,
    dlam_prev: float,
    factor: float,
    total_factor: float,
    cycle: int,
    generation: StateGeneration,
  ) -> None:
    """Write one committed evolution state and its live generation.

    Infallible field writes by contract (the commit atomicity argument): the
    driver calls this only on the commit path, immediately after the owner
    commit, with values it computed itself — so the owner commit and the
    store advance cannot separate. Reject paths never call it.
    """
    da_prev.setflags(write=False)
    self._lam = lam
    self._da_prev = da_prev
    self._dlam_prev = dlam_prev
    self._factor = factor
    self._total_factor = total_factor
    self._cycle = cycle
    self._generation = generation

  def encode(self) -> bytes:
    """Encode the accepted evolution state with the versioned codec."""
    return self._codec.encode(self._layout, self.snapshot())
