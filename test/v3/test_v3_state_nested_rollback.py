# SPDX-License-Identifier: MIT

"""Nested-state rollback evidence: scoped sub-transactions vs restore_accepted.

The M74 survey left the nested-store rollback mechanism evidence-decided:
generation-gated ``restore_accepted`` (candidate A: snapshot every accepted
buffer at trial begin, mutate accepted in place, restore on reject gated on
the rooted generation identity) against scoped sub-transactions (candidate B:
preallocated staged buffers, commit copies the touched set, reject is free —
the production :class:`NestedStateStore`).

Both candidates are driven far enough to measure snapshot/restore cost on a
representative nested store. The reference volume is the micro_rve deck's
per-integration-point state (``examples/materials/micromodel/micro_rve.dat``:
9 nodes x 2 dofs = 18 nested coefficients; 4 Q4 elements x 2x2 Gauss = 16
integration points; v3 ``plane-strain-damage`` carries a 1-float kappa
history per point, pyfem/v3/compile/continuum.py:43) — 34 float64 values,
272 bytes per macro integration point. A refined 64x64-cell RVE (8450
coefficients + 16384 kappa rows, ~195 KB per point) supplies the scaled leg.

The legacy antipattern is named and measured: ``MicroModel.py:181``
deep-copies the entire per-point micro state on every ``getStress``
evaluation, so its copy traffic grows with the number of evaluations per
step. The production store's copy traffic is independent of the evaluation
count: the only state copy in the protocol is the touched-set copy at trial
commit. Byte counters are the deterministic evidence; wall-clock timings are
printed for the record (``pytest -s``) and cited in the commit body.
"""

from __future__ import annotations

import sys
import time
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from pyfem.v3.model.identity import (
  GenerationMismatchError,
  StateGeneration,
  require_same_generation,
)
from pyfem.v3.state import NestedBlockLayout, NestedStateLayout, NestedStateStore

if TYPE_CHECKING:
  from pyfem.v3.state.nested import _TrialScope

# micro_rve deck per-integration-point state volume (module docstring).
_MICRO_COEFFICIENTS = 18
_MICRO_IPS = 16
_MICRO_WIDTH = 1
# Refined 64x64-cell RVE per-point volume: 65x65 nodes x 2 dofs; 4096
# elements x 4 Gauss points.
_REFINED_COEFFICIENTS = 8450
_REFINED_IPS = 16384
_REFINED_WIDTH = 1


def _store_layout(
  coefficients: int,
  ips: int,
  width: int,
  macro_ips: int,
) -> NestedStateLayout:
  return NestedStateLayout(
    schema="test-nested-rollback-v1",
    coefficient_count=coefficients,
    blocks=(
      NestedBlockLayout(
        schema="test-rollback-damage-v1",
        block_id="cells",
        entity_count=ips,
        row_width=width,
      ),
    ),
    ip_count=macro_ips,
  )


def _per_ip_bytes(layout: NestedStateLayout) -> int:
  rows = sum(block.entity_count * block.row_width for block in layout.blocks)
  return 8 * (layout.coefficient_count + rows)


class _Meter:
  """Copy-byte and evaluation counters — the deterministic evidence."""

  def __init__(self) -> None:
    self.copied_bytes = 0
    self.evaluations = 0


class _MeteredStore(NestedStateStore):
  """Production store (candidate B) with the commit copy accounted."""

  __slots__ = ("meter",)

  def __init__(self, layout: NestedStateLayout, meter: _Meter) -> None:
    super().__init__(layout)
    self.meter = meter

  def _commit_validated(self, scope: _TrialScope) -> None:
    touched = int(scope.touched.sum())
    self.meter.copied_bytes += touched * _per_ip_bytes(self.layout)
    super()._commit_validated(scope)


class _CheckpointRestoreStore:
  """Candidate A prototype: generation-gated restore_accepted.

  ``begin_trial`` snapshots every accepted buffer up front (one full-store
  copy, paid whether the step commits or rejects) and captures the rooted
  per-point generations. Evaluations read the checkpoint (the restart state)
  and mutate the accepted buffers in place — so accepted-state reads are
  trial-polluted for the whole scope. ``commit_trial`` validates and
  advances the touched generations; ``reject_trial`` restores the touched
  points from the checkpoint, gated on the rooted generation identity.
  """

  def __init__(self, layout: NestedStateLayout, meter: _Meter) -> None:
    self._layout = layout
    self._meter = meter
    self._accepted_coefficients = np.zeros(
      (layout.ip_count, layout.coefficient_count), dtype=np.float64
    )
    self._accepted_blocks = tuple(
      np.zeros((layout.ip_count, block.entity_count, block.row_width), dtype=np.float64)
      for block in layout.blocks
    )
    # Preallocated checkpoint buffers: the fair best case for this candidate.
    self._checkpoint_coefficients = np.empty_like(self._accepted_coefficients)
    self._checkpoint_blocks = tuple(
      np.empty_like(rows) for rows in self._accepted_blocks
    )
    self._generations = [StateGeneration.initial() for _ in range(layout.ip_count)]
    self._rooted: tuple[StateGeneration, ...] = ()
    self._touched = np.zeros(layout.ip_count, dtype=bool)
    self._open = False

  @property
  def layout(self) -> NestedStateLayout:
    return self._layout

  @property
  def generations(self) -> tuple[StateGeneration, ...]:
    return tuple(self._generations)

  def begin_trial(self) -> None:
    assert not self._open
    np.copyto(self._checkpoint_coefficients, self._accepted_coefficients)
    for checkpoint, accepted in zip(self._checkpoint_blocks, self._accepted_blocks):
      np.copyto(checkpoint, accepted)
    self._meter.copied_bytes += self._layout.ip_count * _per_ip_bytes(self._layout)
    self._rooted = tuple(self._generations)
    self._touched.fill(False)
    self._open = True

  def trial_coefficients(self, ip_index: int) -> np.ndarray:
    assert self._open
    self._touched[ip_index] = True
    return self._accepted_coefficients[ip_index]

  def trial_block(self, ip_index: int, block_index: int = 0) -> np.ndarray:
    assert self._open
    self._touched[ip_index] = True
    return self._accepted_blocks[block_index][ip_index]

  def evaluation_coefficients(self, ip_index: int) -> np.ndarray:
    # Under this candidate the restart state lives in the checkpoint: the
    # accepted buffers are the trial area and must not be read mid-scope.
    assert self._open
    return self._checkpoint_coefficients[ip_index]

  def evaluation_block(self, ip_index: int, block_index: int = 0) -> np.ndarray:
    assert self._open
    return self._checkpoint_blocks[block_index][ip_index]

  def commit_trial(self) -> None:
    assert self._open
    finite = np.isfinite(self._accepted_coefficients).all(axis=1)
    for rows in self._accepted_blocks:
      finite &= np.isfinite(rows).all(axis=(1, 2))
    assert bool((finite | ~self._touched).all())
    for index in np.flatnonzero(self._touched):
      self._generations[index] = self._generations[index].next_accepted()
    self._open = False

  def _restore_accepted(self) -> None:
    # The generation gate: the store must not have moved under the checkpoint.
    for index in np.flatnonzero(self._touched):
      require_same_generation(self._rooted[index], self._generations[index])
    where = self._touched[:, None]
    np.copyto(self._accepted_coefficients, self._checkpoint_coefficients, where=where)
    for accepted, checkpoint in zip(self._accepted_blocks, self._checkpoint_blocks):
      np.copyto(accepted, checkpoint, where=self._touched[:, None, None])
    self._meter.copied_bytes += int(self._touched.sum()) * _per_ip_bytes(self._layout)

  def reject_trial(self) -> None:
    assert self._open
    self._restore_accepted()
    self._open = False

  def accepted_bytes(self) -> bytes:
    parts = [self._accepted_coefficients.tobytes(order="C")]
    parts.extend(rows.tobytes(order="C") for rows in self._accepted_blocks)
    parts.append(np.array([g.ordinal for g in self._generations]).tobytes())
    return b"".join(parts)


class _LegacyMicroState:
  """The legacy antipattern: MicroModel.py:181's deepcopy per evaluation.

  One embedded nested state per macro integration point; every evaluation
  deep-copies the point's entire committed state (the ``globdat`` stand-in:
  coefficient and state-row arrays plus scalar metadata), mutates the copy,
  and the macro commit installs the last copies.
  """

  def __init__(self, layout: NestedStateLayout, meter: _Meter) -> None:
    self._meter = meter
    self._layout = layout
    self._committed = {
      ip: {
        "props": {"tol": 1.0e-8, "iterMax": 30},
        "globdat": {
          "coefficients": np.zeros(layout.coefficient_count),
          "state": {
            block.block_id: np.zeros((block.entity_count, block.row_width))
            for block in layout.blocks
          },
          "cycle": 0,
        },
      }
      for ip in range(layout.ip_count)
    }
    self._pending: dict[int, dict] = {}

  def evaluate(self, ip_index: int, value: float) -> None:
    state = deepcopy(self._committed[ip_index])
    self._meter.copied_bytes += _per_ip_bytes(self._layout)
    self._meter.evaluations += 1
    globdat = state["globdat"]
    globdat["coefficients"][:] = value
    for rows in globdat["state"].values():
      rows[:] = value + 0.5
    globdat["cycle"] += 1
    self._pending[ip_index] = state

  def commit(self) -> None:
    for ip_index, state in self._pending.items():
      self._committed[ip_index] = state
    self._pending = {}

  def reject(self) -> None:
    self._pending = {}

  def accepted_values(self, ip_index: int) -> tuple[np.ndarray, np.ndarray]:
    globdat = self._committed[ip_index]["globdat"]
    return globdat["coefficients"], next(iter(globdat["state"].values()))


def _step_value(step: int, evaluation: int, ip_index: int) -> float:
  return 1000.0 * step + 0.125 * ip_index + 0.001 * evaluation


def _run_production_step(
  store: _MeteredStore,
  ips: np.ndarray,
  evaluations: int,
  step: int,
  *,
  commit: bool,
) -> None:
  store.meter.evaluations += len(ips) * evaluations
  store.begin_trial()
  for evaluation in range(evaluations):
    for ip_index in ips:
      value = _step_value(step, evaluation, int(ip_index))
      # Zero-copy protocol: read the accepted views, write the staged views.
      _ = store.accepted_coefficients(int(ip_index))
      _ = store.accepted_block(int(ip_index), "cells")
      store.trial_coefficients(int(ip_index))[:] = value
      store.trial_block(int(ip_index), "cells")[:] = value + 0.5
      store.trial_complete(int(ip_index))
  if commit:
    store.commit_trial()
  else:
    store.reject_trial()


def _run_production_step_held(
  store: _MeteredStore,
  ips: np.ndarray,
  evaluations: int,
  step: int,
  *,
  commit: bool,
) -> None:
  # The mission-3 evaluator pattern: workspace and accepted views are
  # acquired once per point per step and held across the step's evaluations.
  store.meter.evaluations += len(ips) * evaluations
  store.begin_trial()
  workspaces = []
  for ip_index in ips:
    _ = store.accepted_coefficients(int(ip_index))
    _ = store.accepted_block(int(ip_index), "cells")
    workspaces.append(
      (
        int(ip_index),
        store.trial_coefficients(int(ip_index)),
        store.trial_block(int(ip_index), "cells"),
      )
    )
  for evaluation in range(evaluations):
    for ip_index, coefficients, rows in workspaces:
      value = _step_value(step, evaluation, ip_index)
      coefficients[:] = value
      rows[:] = value + 0.5
  for ip_index, _, _ in workspaces:
    store.trial_complete(ip_index)
  if commit:
    store.commit_trial()
  else:
    store.reject_trial()


def _run_checkpoint_step(
  store: _CheckpointRestoreStore,
  ips: np.ndarray,
  evaluations: int,
  step: int,
  *,
  commit: bool,
) -> None:
  store.begin_trial()
  for evaluation in range(evaluations):
    for ip_index in ips:
      value = _step_value(step, evaluation, int(ip_index))
      _ = store.evaluation_coefficients(int(ip_index))
      _ = store.evaluation_block(int(ip_index))
      store.trial_coefficients(int(ip_index))[:] = value
      store.trial_block(int(ip_index))[:] = value + 0.5
  if commit:
    store.commit_trial()
  else:
    store.reject_trial()


def _run_legacy_step(
  store: _LegacyMicroState,
  ips: np.ndarray,
  evaluations: int,
  step: int,
  *,
  commit: bool,
) -> None:
  for evaluation in range(evaluations):
    for ip_index in ips:
      store.evaluate(int(ip_index), _step_value(step, evaluation, int(ip_index)))
  if commit:
    store.commit()
  else:
    store.reject()


def _schedule(steps: int) -> tuple[bool, ...]:
  # Deterministic commit/reject pattern: every fifth step rejects, and the
  # final step commits so end-state comparisons cover a committed tail.
  return tuple((index + 1) % 5 != 0 or index == steps - 1 for index in range(steps))


def _accepted_bytes(store: NestedStateStore) -> bytes:
  snapshot = store.snapshot()
  parts = [snapshot.coefficients.values.tobytes()]
  parts.extend(rows.values.tobytes() for rows in snapshot.blocks)
  parts.append(np.array(snapshot.generations).tobytes())
  return b"".join(parts)


def test_rollback_candidates_agree_bitwise_and_diverge_in_cost() -> None:
  layout = _store_layout(_MICRO_COEFFICIENTS, _MICRO_IPS, _MICRO_WIDTH, 64)
  schedule = _schedule(20)
  evaluations = 3
  meters = (_Meter(), _Meter(), _Meter())
  production = _MeteredStore(layout, meters[0])
  checkpoint = _CheckpointRestoreStore(layout, meters[1])
  legacy = _LegacyMicroState(layout, meters[2])
  ips = np.arange(layout.ip_count)

  for step, commit in enumerate(schedule):
    _run_production_step(production, ips, evaluations, step, commit=commit)
    _run_checkpoint_step(checkpoint, ips, evaluations, step, commit=commit)
    _run_legacy_step(legacy, ips, evaluations, step, commit=commit)
    assert _accepted_bytes(production) == checkpoint.accepted_bytes()
  commits = sum(1 for commit in schedule if commit)
  rejects = len(schedule) - commits
  assert rejects > 0
  volume = layout.ip_count * _per_ip_bytes(layout)

  # Candidate B (production): the only state copy in the whole run is the
  # touched-set copy at each commit — rejects copy nothing, evaluations copy
  # nothing, and the total is independent of the evaluation count.
  assert meters[0].copied_bytes == commits * volume
  # Candidate A (restore_accepted): one full-store snapshot per step whether
  # the step commits or rejects, plus a touched-set restore per reject.
  assert meters[1].copied_bytes == len(schedule) * volume + rejects * volume
  # Legacy: one full per-point deepcopy per evaluation (MicroModel.py:181).
  assert meters[2].copied_bytes == evaluations * len(schedule) * volume
  assert meters[2].evaluations == evaluations * len(schedule) * layout.ip_count
  # The evidence decides: B < A < legacy, strictly.
  assert meters[0].copied_bytes < meters[1].copied_bytes
  assert meters[1].copied_bytes < meters[2].copied_bytes

  # Bitwise agreement of the surviving state across all three protocols.
  last_committed = max(step for step, commit in enumerate(schedule) if commit)
  for ip_index in (0, 17, 63):
    expected = _step_value(last_committed, evaluations - 1, ip_index)
    np.testing.assert_array_equal(
      production.accepted_coefficients(ip_index), np.full(18, expected)
    )
    coefficients, rows = legacy.accepted_values(ip_index)
    np.testing.assert_array_equal(coefficients, np.full(18, expected))
    np.testing.assert_array_equal(rows, np.full((16, 1), expected + 0.5))
  assert [g.ordinal for g in production.generations] == [commits] * 64


def test_copy_traffic_is_evaluation_count_independent_unlike_legacy() -> None:
  layout = _store_layout(_MICRO_COEFFICIENTS, _MICRO_IPS, _MICRO_WIDTH, 32)
  schedule = _schedule(10)
  ips = np.arange(layout.ip_count)
  copied_by_k: dict[int, tuple[int, int]] = {}
  for evaluations in (2, 6):
    meters = (_Meter(), _Meter())
    production = _MeteredStore(layout, meters[0])
    legacy = _LegacyMicroState(layout, meters[1])
    for step, commit in enumerate(schedule):
      _run_production_step(production, ips, evaluations, step, commit=commit)
      _run_legacy_step(legacy, ips, evaluations, step, commit=commit)
    copied_by_k[evaluations] = (meters[0].copied_bytes, meters[1].copied_bytes)
  production_2, legacy_2 = copied_by_k[2]
  production_6, legacy_6 = copied_by_k[6]
  # The memory-discipline pin: tripling the per-step evaluation count moves
  # the production store's copy traffic not at all; legacy's triples.
  assert production_6 == production_2
  assert legacy_6 == 3 * legacy_2


def test_partial_touch_steps_still_snapshot_everything_under_restore() -> None:
  layout = _store_layout(_MICRO_COEFFICIENTS, _MICRO_IPS, _MICRO_WIDTH, 256)
  touched = np.arange(0, 256, 4)  # one point in four carries nested state
  schedule = _schedule(10)
  meters = (_Meter(), _Meter())
  production = _MeteredStore(layout, meters[0])
  checkpoint = _CheckpointRestoreStore(layout, meters[1])
  for step, commit in enumerate(schedule):
    _run_production_step(production, touched, 3, step, commit=commit)
    _run_checkpoint_step(checkpoint, touched, 3, step, commit=commit)
  commits = sum(1 for commit in schedule if commit)
  rejects = len(schedule) - commits
  touched_volume = len(touched) * _per_ip_bytes(layout)
  volume = layout.ip_count * _per_ip_bytes(layout)
  assert meters[0].copied_bytes == commits * touched_volume
  assert meters[1].copied_bytes == len(schedule) * volume + rejects * touched_volume
  assert meters[0].copied_bytes * 4 < meters[1].copied_bytes


def test_checkpoint_restore_is_gated_on_the_rooted_generation() -> None:
  layout = _store_layout(_MICRO_COEFFICIENTS, _MICRO_IPS, _MICRO_WIDTH, 4)
  meter = _Meter()
  store = _CheckpointRestoreStore(layout, meter)
  store.begin_trial()
  store.trial_coefficients(1)[:] = 1.0
  # A store that advanced under the open scope fails the generation gate.
  store._generations[1] = store._generations[1].next_accepted()
  with pytest.raises(GenerationMismatchError):
    store.reject_trial()


def test_rollback_correctness_under_both_candidates() -> None:
  layout = _store_layout(_MICRO_COEFFICIENTS, _MICRO_IPS, _MICRO_WIDTH, 8)
  schedule = _schedule(12)
  ips = np.arange(8)
  meters = (_Meter(), _Meter())
  production = _MeteredStore(layout, meters[0])
  checkpoint = _CheckpointRestoreStore(layout, meters[1])
  for step, commit in enumerate(schedule):
    before_production = _accepted_bytes(production)
    before_checkpoint = checkpoint.accepted_bytes()
    assert before_production == before_checkpoint
    _run_production_step(production, ips, 3, step, commit=commit)
    _run_checkpoint_step(checkpoint, ips, 3, step, commit=commit)
    if not commit:
      # Rejected trials leave every accepted byte and generation identical.
      assert _accepted_bytes(production) == before_production
      assert checkpoint.accepted_bytes() == before_checkpoint
  assert _accepted_bytes(production) == checkpoint.accepted_bytes()


def test_snapshot_restore_wall_clock_on_scaled_volumes() -> None:
  cases = (
    ("micro_rve-exact x1024 macro-IPs", _MICRO_COEFFICIENTS, _MICRO_IPS, 1024, 30),
    ("refined-rve x32 macro-IPs", _REFINED_COEFFICIENTS, _REFINED_IPS, 32, 10),
  )
  table: list[tuple[str, str, float, int]] = []
  for label, coefficients, ips, macro_ips, steps in cases:
    layout = _store_layout(coefficients, ips, _MICRO_WIDTH, macro_ips)
    schedule = _schedule(steps)
    all_ips = np.arange(macro_ips)
    for candidate, runner in (
      ("B scoped-sub-transaction", _run_production_step),
      ("B held-workspace", _run_production_step_held),
      ("A restore-accepted", _run_checkpoint_step),
      ("legacy deepcopy", _run_legacy_step),
    ):
      meter = _Meter()
      if candidate.startswith("B"):
        subject = _MeteredStore(layout, meter)
      elif candidate.startswith("A"):
        subject = _CheckpointRestoreStore(layout, meter)
      else:
        subject = _LegacyMicroState(layout, meter)
      start = time.perf_counter()
      for step, commit in enumerate(schedule):
        runner(subject, all_ips, 3, step, commit=commit)
      elapsed = time.perf_counter() - start
      table.append((label, candidate, elapsed, meter.copied_bytes))
  for label, candidate, elapsed, copied in table:
    print(f"{label:36s} {candidate:24s} {elapsed * 1e3:9.2f} ms {copied:12d} B")
