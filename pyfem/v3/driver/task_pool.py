"""Deterministic host-side task scheduling over macro integration points."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, wait


class IntegrationPointTaskPool:
  """Run independent per-IP callbacks on fixed, contiguous scheduling chunks.

  The worker set and chunk boundaries persist across evaluations. Each callback
  writes only its own integration-point slot, and chunk results are observed in
  stable IP order. This is a host-side Python task pool, not a free-threaded
  Numba path. The one-thread case executes the same direct serial loop without
  executor overhead.
  """

  __slots__ = ("_chunks", "_closed", "_executor", "_ip_count", "_thread_count")

  def __init__(self, ip_count: int, thread_count: int) -> None:
    if type(ip_count) is not int or ip_count < 0:
      msg = "integration-point task pools require a non-negative exact IP count"
      raise ValueError(msg)
    if type(thread_count) is not int or thread_count < 1:
      msg = "integration-point task pools require a positive exact thread count"
      raise ValueError(msg)
    self._ip_count = ip_count
    self._thread_count = thread_count
    self._closed = False
    worker_count = min(ip_count, thread_count)
    chunks: list[tuple[int, int]] = []
    if worker_count:
      size, larger = divmod(ip_count, worker_count)
      start = 0
      for chunk_index in range(worker_count):
        stop = start + size + int(chunk_index < larger)
        chunks.append((start, stop))
        start = stop
    self._chunks = tuple(chunks)
    self._executor = (
      ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="pyfem-fe2-ip")
      if worker_count > 1
      else None
    )

  @property
  def thread_count(self) -> int:
    """Return the requested host worker count."""
    return self._thread_count

  @property
  def worker_count(self) -> int:
    """Return the number of non-empty deterministic scheduling chunks."""
    return len(self._chunks)

  @property
  def chunks(self) -> tuple[tuple[int, int], ...]:
    """Return the stable half-open IP ranges assigned to workers."""
    return self._chunks

  @property
  def closed(self) -> bool:
    """Return whether host workers have been shut down."""
    return self._closed

  def run[ResultT](self, evaluate: Callable[[int], ResultT]) -> tuple[ResultT, ...]:
    """Evaluate every IP exactly once and return results in stable IP order."""
    if self._closed:
      msg = "integration-point task pool is closed"
      raise RuntimeError(msg)
    executor = self._executor
    if executor is None:
      return tuple(evaluate(ip_index) for ip_index in range(self._ip_count))
    results: list[ResultT | None] = [None] * self._ip_count

    def run_chunk(bounds: tuple[int, int]) -> None:
      start, stop = bounds
      for ip_index in range(start, stop):
        results[ip_index] = evaluate(ip_index)

    futures = [executor.submit(run_chunk, chunk) for chunk in self._chunks]
    wait(futures)
    for future in futures:
      future.result()
    return tuple(results)

  def close(self) -> None:
    """Shut down host workers after outstanding tasks finish; idempotent."""
    if self._closed:
      return
    executor = self._executor
    if executor is not None:
      executor.shutdown(wait=True)
    self._closed = True

  def __enter__(self) -> IntegrationPointTaskPool:
    if self._closed:
      msg = "integration-point task pool is closed"
      raise RuntimeError(msg)
    return self

  def __exit__(self, *_exc: object) -> None:
    self.close()
