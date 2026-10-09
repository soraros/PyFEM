# SPDX-License-Identifier: MIT

"""Global stiffness COO assembly: v3 vs legacy on patch-test skims."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from _legacy_parity import legacy_stiffness_coo

from pyfem.v3 import load_problem
from pyfem.v3._prototype_assembly import assemble_loaded
from pyfem.v3.fem.assembly import (
  CooCsrPattern,
  _fill_stiffness_coo,
  _fill_stiffness_coo_parallel,
  _fill_stiffness_coo_serial,
  compile_csr_pattern,
  dedup_coo_values,
)
from pyfem.v3.fem.element import continuum_stiffness_batched
from pyfem.v3.mesh.refined_patch import build_uniform_q8_loaded

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
  "skim_name",
  [
    "patch_test8",
    "patch_test4",
    "patch_test3",
    "patch_test8_3d",
    "patch_test8_plane_strain",
    "patch_test8_mpc",
  ],
)
def test_assembly_coo_matches_legacy(skim_name: str) -> None:
  skim_pro = ROOT / "skims" / skim_name / "skim.pro"
  loaded = load_problem(skim_pro)
  v3 = assemble_loaded(loaded).stiffness.tocoo()
  legacy = legacy_stiffness_coo(skim_pro)

  assert v3.shape == legacy.shape
  np.testing.assert_array_equal(v3.row, legacy.row)
  np.testing.assert_array_equal(v3.col, legacy.col)
  np.testing.assert_allclose(v3.data, legacy.data, rtol=0.0, atol=1e-8)


def _patch_coo_buffers(
  nx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
  """Element-major COO buffers of one uniform Q8 patch via the dispatcher."""
  loaded = build_uniform_q8_loaded(nx, nx)
  problem = loaded.problem
  conn = problem.conn
  n_elems = conn.shape[0]
  n_dof = 2 * conn.shape[1]
  nodal_coords = problem.coords[conn]
  element_dofs = problem.global_dofs[conn].reshape(n_elems, n_dof)
  stiffness = continuum_stiffness_batched(nodal_coords, problem.constitutive)
  nnz = n_elems * n_dof * n_dof
  row = np.empty(nnz, dtype=np.int32)
  col = np.empty(nnz, dtype=np.int32)
  val = np.empty(nnz, dtype=np.float64)
  _fill_stiffness_coo(element_dofs, stiffness, row, col, val)
  return row, col, val, (problem.n_dofs, problem.n_dofs)


def test_fill_stiffness_coo_kernels_are_bitwise_identical() -> None:
  """Serial and parallel scatter kernels emit the same COO bytes."""
  loaded = build_uniform_q8_loaded(8, 8)
  problem = loaded.problem
  conn = problem.conn
  n_elems = conn.shape[0]
  n_dof = 2 * conn.shape[1]
  element_dofs = problem.global_dofs[conn].reshape(n_elems, n_dof)
  stiffness = continuum_stiffness_batched(
    problem.coords[conn],
    problem.constitutive,
  )
  buffers = []
  for kernel in (_fill_stiffness_coo_serial, _fill_stiffness_coo_parallel):
    nnz = n_elems * n_dof * n_dof
    row = np.empty(nnz, dtype=np.int32)
    col = np.empty(nnz, dtype=np.int32)
    val = np.empty(nnz, dtype=np.float64)
    kernel(element_dofs, stiffness, row, col, val)
    buffers.append((row, col, val))
  for left, right in zip(buffers[0], buffers[1], strict=True):
    np.testing.assert_array_equal(left.view(np.uint64), right.view(np.uint64))


def test_fill_stiffness_coo_bitwise_identical_across_thread_counts() -> None:
  """1T vs nT raw-uint64 identity of the parallel scatter (prange race canary).

  Each COO slot is written by exactly one element, so the dispatcher's
  parallel kernel must reproduce the 1T buffers bit for bit at any thread
  count. The 32x32 patch (1024 elements, 262144 entries) is above the
  dispatcher's parallel threshold.
  """
  import numba

  loaded = build_uniform_q8_loaded(32, 32)
  problem = loaded.problem
  conn = problem.conn
  n_elems = conn.shape[0]
  n_dof = 2 * conn.shape[1]
  element_dofs = problem.global_dofs[conn].reshape(n_elems, n_dof)
  stiffness = continuum_stiffness_batched(
    problem.coords[conn],
    problem.constitutive,
  )
  nnz = n_elems * n_dof * n_dof

  def scatter() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row = np.empty(nnz, dtype=np.int32)
    col = np.empty(nnz, dtype=np.int32)
    val = np.empty(nnz, dtype=np.float64)
    _fill_stiffness_coo(element_dofs, stiffness, row, col, val)
    return row, col, val

  previous = numba.get_num_threads()
  n_threads = max(2, min(16, previous))
  try:
    numba.set_num_threads(1)
    buffers_1t = scatter()
    numba.set_num_threads(n_threads)
    buffers_nt = scatter()
  finally:
    numba.set_num_threads(previous)
  for left, right in zip(buffers_1t, buffers_nt, strict=True):
    np.testing.assert_array_equal(left.view(np.uint64), right.view(np.uint64))


def _sequential_dedup_reference(val: np.ndarray, pattern: CooCsrPattern) -> np.ndarray:
  """Strict left-to-right duplicate sums in the pattern's stable stream order."""
  data = np.empty(pattern.indices.shape[0], dtype=np.float64)
  nnz = val.shape[0]
  n_segments = pattern.segment_offsets.shape[0]
  for s in range(n_segments):
    start = pattern.segment_offsets[s]
    stop = pattern.segment_offsets[s + 1] if s + 1 < n_segments else nnz
    acc = 0.0
    for j in range(start, stop):
      acc += val[pattern.permutation[j]]
    data[s] = acc
  return data


def test_compile_csr_pattern_matches_scipy_on_landed_meshes() -> None:
  """Compiled pattern (indptr/indices) is exactly scipy's canonical CSR one."""
  for skim_name in (
    "patch_test8",
    "patch_test4",
    "patch_test3",
    "patch_test8_3d",
    "patch_test8_plane_strain",
    "patch_test8_mpc",
  ):
    loaded = load_problem(ROOT / "skims" / skim_name / "skim.pro")
    coo = assemble_loaded(loaded).stiffness
    pattern = compile_csr_pattern(coo.row, coo.col, coo.shape)
    reference = coo.tocsr()
    np.testing.assert_array_equal(pattern.indptr, reference.indptr)
    np.testing.assert_array_equal(pattern.indices, reference.indices)


def test_compile_csr_pattern_handles_empty_stream() -> None:
  pattern = compile_csr_pattern(
    np.empty(0, dtype=np.int32),
    np.empty(0, dtype=np.int32),
    (7, 7),
  )
  np.testing.assert_array_equal(pattern.indptr, np.zeros(8, dtype=np.int32))
  assert pattern.indices.shape == (0,)
  assert pattern.permutation.shape == (0,)
  assert pattern.segment_offsets.shape == (0,)
  np.testing.assert_array_equal(dedup_coo_values(pattern, np.empty(0)), np.empty(0))


def test_dedup_coo_values_bitwise_match_strict_sequential_reference() -> None:
  """Dedup sums each duplicate run left-to-right in stable stream order.

  This is the canonical v3 accumulation order (the same one
  ``driver.plan.refill_tangent`` documents); it is thread-count- and
  platform-independent by construction (no reduction reassociation).
  """
  row, col, val, shape = _patch_coo_buffers(8)
  pattern = compile_csr_pattern(row, col, shape)
  np.testing.assert_array_equal(
    dedup_coo_values(pattern, val).view(np.uint64),
    _sequential_dedup_reference(val, pattern).view(np.uint64),
  )


def test_dedup_coo_values_bitwise_identical_across_thread_counts() -> None:
  """1T vs nT raw-uint64 identity of the parallel dedup (race canary).

  Each output slot is one disjoint segment, and the per-segment sum order
  is fixed by the pattern, so any thread count yields the same bits. The
  32x32 patch (262144 entries) is above the dispatcher's parallel
  threshold.
  """
  import numba

  row, col, val, shape = _patch_coo_buffers(32)
  pattern = compile_csr_pattern(row, col, shape)
  previous = numba.get_num_threads()
  n_threads = max(2, min(16, previous))
  try:
    numba.set_num_threads(1)
    data_1t = dedup_coo_values(pattern, val)
    numba.set_num_threads(n_threads)
    data_nt = dedup_coo_values(pattern, val)
  finally:
    numba.set_num_threads(previous)
  np.testing.assert_array_equal(data_1t.view(np.uint64), data_nt.view(np.uint64))


def test_dedup_coo_values_matches_scipy_within_order_tolerance() -> None:
  """Fused dedup agrees with scipy's ``tocsr`` at summation-order ulp level.

  scipy sums duplicate runs in its unstable per-row ``std::sort`` order
  (``csr_sort_indices`` compares columns only), so its bits are STL-order
  dependent and an exact match is not defined. The observed max abs
  deviation across the landed skims and the 2..64 uniform patches is
  9.313e-10 at entry magnitudes up to 2e7 — pure summation-order round-off.
  """
  loaded = build_uniform_q8_loaded(8, 8)
  coo = assemble_loaded(loaded).stiffness
  pattern = compile_csr_pattern(coo.row, coo.col, coo.shape)
  data = dedup_coo_values(pattern, coo.data)
  reference = coo.tocsr()
  np.testing.assert_allclose(data, reference.data, rtol=1e-12, atol=1e-8)
