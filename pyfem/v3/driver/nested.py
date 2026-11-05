"""Per-execution nested runtimes and shared transaction-scope accounting.

A prepared runtime belongs to exactly one macro driver execution. Its accepted
and staged buffers, per-IP drivers and counters, response workspaces, and host
pool are never module or operator singletons. Other runtimes prepared from the
same recipe share only immutable compiled content. The macro owner receives
typed nested references, never ownership of nested buffers; the shared scope
helpers give nonlinear and Riks identical begin/reject/validate/commit behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from pyfem.v3.driver.contracts import (
  DriverActivitySnapshot,
  DriverStatistics,
  NestedFailure,
  NestedSolveStatistics,
)
from pyfem.v3.driver.task_pool import IntegrationPointTaskPool
from pyfem.v3.model.operator import NestedExecutionRecipe
from pyfem.v3.state.nested import NestedStateStore
from pyfem.v3.state.owner import StateTransactionOwner

if TYPE_CHECKING:
  from pyfem.v3.driver.contracts import NonlinearStaticResult
  from pyfem.v3.driver.nonlinear import NonlinearStaticDriver
  from pyfem.v3.spec.program import ProgramPoint


def _seed_store(recipe: NestedExecutionRecipe, store: NestedStateStore) -> None:
  store.begin_trial()
  try:
    for ip_index in range(store.ip_count):
      store.write_trial_coefficients(
        ip_index, np.zeros(recipe.micro_system.coefficient_count, dtype=np.float64)
      )
      for operator, block in zip(
        recipe.micro_system.operators, store.layout.blocks, strict=True
      ):
        layout = operator.header.state_layout
        initial_rows = layout.initial_rows
        rows = (
          np.zeros(layout.row_shape, dtype=np.float64)
          if initial_rows is None
          else initial_rows.values
        )
        store.write_trial_block(ip_index, block.block_id, rows)
      store.trial_complete(ip_index)
    store.commit_trial()
  except BaseException:
    if store.trial_open:
      store.reject_trial()
    raise


@dataclass(frozen=True, slots=True, eq=False)
class _NestedDriverCursor:
  driver: NonlinearStaticDriver
  snapshot: DriverActivitySnapshot


class PreparedNestedExecution:
  """Mutable nested state and solve resources owned by one macro execution.

  Construction reuses the exact compiled recipe without compiling the FE2
  operator again. The store, point-backed drivers, response buffers, and task
  pool are fresh and remain private to this runtime until ``close``.
  """

  __slots__ = (
    "_buffers",
    "_closed",
    "_drivers",
    "_recipe",
    "_store",
    "_task_pool",
  )

  def __init__(self, recipe: NestedExecutionRecipe) -> None:
    if type(recipe) is not NestedExecutionRecipe:
      msg = "prepared nested execution requires an exact NestedExecutionRecipe"
      raise TypeError(msg)
    from pyfem.v3.driver.nonlinear import NonlinearStaticDriver

    self._recipe = recipe
    self._closed = False
    self._buffers: dict[str, np.ndarray] = {}
    self._store = NestedStateStore(recipe.layout)
    _seed_store(recipe, self._store)
    self._drivers = tuple(
      NonlinearStaticDriver(
        recipe.micro_system,
        recipe.micro_coordinate_map,
        recipe.micro_loads,
        recipe.settings,
        owner=StateTransactionOwner.for_nested_point(
          recipe.micro_system,
          store=self._store,
          ip_index=ip_index,
        ),
      )
      for ip_index in range(self._store.ip_count)
    )
    self._task_pool = IntegrationPointTaskPool(
      self._store.ip_count, recipe.settings.nested_ip_thread_count
    )

  @property
  def recipe(self) -> NestedExecutionRecipe:
    return self._recipe

  @property
  def store(self) -> NestedStateStore:
    return self._store

  @property
  def task_pool(self) -> IntegrationPointTaskPool:
    return self._task_pool

  @property
  def closed(self) -> bool:
    return self._closed

  @property
  def driver_statistics(self) -> tuple[DriverStatistics, ...]:
    """Return independent per-IP counter snapshots without exposing drivers."""
    return tuple(driver.statistics for driver in self._drivers)

  def _require_active(self) -> None:
    if self._closed:
      msg = "prepared nested execution is closed"
      raise RuntimeError(msg)

  def retained_buffer(self, name: str, shape: tuple[int, ...]) -> np.ndarray:
    """Return one execution-local float64 response buffer, allocated only once."""
    self._require_active()
    if type(name) is not str or not name:
      msg = "retained nested buffer names must be non-empty exact strings"
      raise TypeError(msg)
    if type(shape) is not tuple or any(
      type(size) is not int or size < 0 for size in shape
    ):
      msg = "retained nested buffer shapes must be non-negative exact integer tuples"
      raise TypeError(msg)
    buffer = self._buffers.get(name)
    if buffer is None:
      buffer = np.empty(shape, dtype=np.float64)
      self._buffers[name] = buffer
    elif buffer.shape != shape:
      msg = f"retained nested buffer {name!r} was requested with a conflicting shape"
      raise ValueError(msg)
    return buffer

  def run_point(
    self,
    ip_index: int,
    *,
    base_point: ProgramPoint,
    target_points: tuple[ProgramPoint, ...],
  ) -> NonlinearStaticResult:
    """Run one point driver against this runtime's open nested trial scope."""
    self._require_active()
    if not self._store.trial_open:
      msg = "nested point execution requires an open nested trial scope"
      raise RuntimeError(msg)
    return self._drivers[ip_index].run(
      base_point=base_point,
      target_points=target_points,
      homogenization=self._recipe.homogenization,
    )

  def begin_trial(self) -> None:
    self._require_active()
    self._store.begin_trial()

  def reject_trial(self) -> None:
    self._require_active()
    self._store.reject_trial()

  def validate_trial(self) -> None:
    self._require_active()
    self._store.validate_trial()

  def commit_trial(self) -> None:
    self._require_active()
    self._store.commit_trial()

  def _snapshot_activity(self) -> tuple[_NestedDriverCursor, ...]:
    return tuple(
      _NestedDriverCursor(
        driver=driver,
        snapshot=driver._snapshot_nested_activity(),
      )
      for driver in self._drivers
    )

  def close(self) -> None:
    """Release host workers and retained response buffers; idempotent."""
    if self._closed:
      return
    try:
      if self._store.trial_open:
        self._store.reject_trial()
      for driver in self._drivers:
        driver.close()
    finally:
      self._task_pool.close()
      self._buffers.clear()
      self._closed = True

  def __enter__(self) -> PreparedNestedExecution:
    self._require_active()
    return self

  def __exit__(self, *_exc: object) -> None:
    self.close()

  def __del__(self) -> None:
    try:
      self.close()
    except Exception:
      pass


def prepare_nested_executions(
  recipes: tuple[NestedExecutionRecipe | None, ...],
) -> tuple[PreparedNestedExecution | None, ...]:
  """Prepare an aligned runtime sequence, cleaning up partial construction."""
  runtimes: list[PreparedNestedExecution | None] = []
  try:
    for recipe in recipes:
      runtimes.append(None if recipe is None else PreparedNestedExecution(recipe))
  except BaseException:
    for runtime in runtimes:
      if runtime is not None:
        runtime.close()
    raise
  return tuple(runtimes)


def close_nested_executions(
  runtimes: tuple[PreparedNestedExecution | None, ...],
) -> None:
  for runtime in runtimes:
    if runtime is not None:
      runtime.close()


def begin_nested_trial(
  runtimes: tuple[PreparedNestedExecution | None, ...],
) -> None:
  """Open every execution scope, rolling back a partially opened sequence."""
  opened: list[PreparedNestedExecution] = []
  try:
    for runtime in runtimes:
      if runtime is not None:
        runtime.begin_trial()
        opened.append(runtime)
  except BaseException:
    for runtime in reversed(opened):
      runtime.reject_trial()
    raise


def reject_nested_trial(
  runtimes: tuple[PreparedNestedExecution | None, ...],
) -> None:
  for runtime in runtimes:
    if runtime is not None and runtime.store.trial_open:
      runtime.reject_trial()


def validate_nested_trial(
  runtimes: tuple[PreparedNestedExecution | None, ...],
) -> None:
  for runtime in runtimes:
    if runtime is not None:
      runtime.validate_trial()


def commit_nested_trial(
  runtimes: tuple[PreparedNestedExecution | None, ...],
) -> None:
  for runtime in runtimes:
    if runtime is not None:
      runtime.commit_trial()


def snapshot_nested_activity(
  runtimes: tuple[PreparedNestedExecution | None, ...],
) -> tuple[tuple[_NestedDriverCursor, ...], ...]:
  return tuple(
    () if runtime is None else runtime._snapshot_activity() for runtime in runtimes
  )


def collect_nested_activity(
  cursors: tuple[tuple[_NestedDriverCursor, ...], ...],
) -> tuple[tuple[NestedSolveStatistics, ...], NestedFailure | None]:
  """Attribute per-IP work and the first typed failure in stable operator order."""
  statistics = []
  failure = None
  for operator_index, operator_cursors in enumerate(cursors):
    for ip_index, cursor in enumerate(operator_cursors):
      delta = cursor.driver._nested_activity_delta(cursor.snapshot)
      statistics.append(
        NestedSolveStatistics(
          operator_index=operator_index,
          ip_index=ip_index,
          iteration_count=delta.iteration_count,
          evaluation_count=delta.evaluation_count,
          factorization_count=delta.factorization_count,
          factorization_reuse_count=delta.factorization_reuse_count,
          linear_solve_count=delta.linear_solve_count,
        )
      )
      if failure is not None or delta.failure_class is None:
        continue
      failure = NestedFailure(
        code="nested-non-convergence",
        operator_index=operator_index,
        ip_index=ip_index,
        failure_class=delta.failure_class,
      )
  return tuple(statistics), failure
