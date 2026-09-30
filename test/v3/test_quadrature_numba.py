# SPDX-License-Identifier: MIT

"""Numba / JIT compatibility and correctness for v3 Gauss quadrature."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from scipy.special import roots_legendre

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

pytest.importorskip("numba")

from numba import njit
from numba.core.errors import TypingError
from numba.core.registry import CPUDispatcher

from pyfem.v3.fem.quadrature import (
  _meshgrid_2d,
  gauss_legendre_1d,
  gauss_tensor_product_2d,
  gauss_tria3,
)
from pyfem.v3.types import F64

# Q8 plane-stress rule; also spot-check a couple of tabulated orders.
_LITERAL_ORDERS = (2, 3, 4)
_Q8_ORDER = 3


def _reference_gauss_tensor_product_2d(order: int) -> tuple[F64, F64]:
  xi, wx = roots_legendre(order)
  eta, wy = roots_legendre(order)
  xi_grid, eta_grid = np.meshgrid(xi, eta, indexing="ij")
  weights = np.outer(wx, wy).ravel()
  points = np.column_stack((xi_grid.ravel(), eta_grid.ravel()))
  return points.astype(np.float64), weights.astype(np.float64)


@pytest.mark.parametrize("order", _LITERAL_ORDERS)
def test_gauss_legendre_1d_dispatcher_matches_scipy(order: int) -> None:
  """Top-level calls compile via Numba and match SciPy (runtime ``order`` ok)."""
  nodes, weights = gauss_legendre_1d(order)
  ref_nodes, ref_weights = roots_legendre(order)
  assert nodes.dtype == np.float64
  assert weights.dtype == np.float64
  np.testing.assert_allclose(nodes, ref_nodes, rtol=0, atol=1e-14)
  np.testing.assert_allclose(weights, ref_weights, rtol=0, atol=1e-14)


@pytest.mark.parametrize("order", _LITERAL_ORDERS)
def test_gauss_tensor_product_2d_dispatcher_matches_scipy(order: int) -> None:
  points, weights = gauss_tensor_product_2d(order)
  ref_points, ref_weights = _reference_gauss_tensor_product_2d(order)
  n = order * order
  assert points.shape == (n, 2)
  assert weights.shape == (n,)
  np.testing.assert_allclose(points, ref_points, rtol=0, atol=1e-14)
  np.testing.assert_allclose(weights, ref_weights, rtol=0, atol=1e-14)


def test_quadrature_exports_are_numba_dispatchers() -> None:
  assert isinstance(gauss_legendre_1d, CPUDispatcher)
  assert isinstance(gauss_tensor_product_2d, CPUDispatcher)
  assert isinstance(gauss_tria3, CPUDispatcher)


def test_gauss_tria3_order1_tabulated() -> None:
  points, weights = gauss_tria3(1)
  np.testing.assert_allclose(points, [[1.0 / 3.0, 1.0 / 3.0]], rtol=0, atol=0.0)
  np.testing.assert_allclose(weights, [0.5], rtol=0, atol=0.0)


# cache=False: probes call kernels from pyfem/v3/fem/quadrature.py, and numba's
# on-disk cache cannot invalidate cross-module callees (NUMBA_CACHING.md §5) —
# a cached probe would keep passing against stale quadrature code.
@njit(cache=False)
def _probe_gauss_tria3_order1() -> tuple[F64, F64]:
  return gauss_tria3(1)


def test_gauss_tria3_njit_literal_order_matches_tabulated() -> None:
  points, weights = _probe_gauss_tria3_order1()
  assert gauss_tria3.signatures
  np.testing.assert_allclose(points, [[1.0 / 3.0, 1.0 / 3.0]], rtol=0, atol=0.0)
  np.testing.assert_allclose(weights, [0.5], rtol=0, atol=0.0)


# cache=False: cross-module probe (NUMBA_CACHING.md §5; see first probe above).
@njit(cache=False)
def _probe_gauss_tria3_variable_in_loop() -> None:
  for order in range(1, 3):
    gauss_tria3(order)


def test_gauss_tria3_rejects_non_literal_order_in_njit_loop() -> None:
  with pytest.raises(TypingError, match="literal"):
    _probe_gauss_tria3_variable_in_loop()


# cache=False: cross-module probe (NUMBA_CACHING.md §5; see first probe above).
@njit(cache=False)
def _probe_gauss_legendre_1d_q8() -> tuple[F64, F64]:
  return gauss_legendre_1d(3)


# cache=False: cross-module probe (NUMBA_CACHING.md §5; see first probe above).
@njit(cache=False)
def _probe_gauss_tensor_product_2d_q8() -> tuple[F64, F64]:
  return gauss_tensor_product_2d(3)


# cache=False: cross-module probe (NUMBA_CACHING.md §5; see first probe above).
@njit(cache=False)
def _probe_gauss_tensor_product_2d_decomposed_q8() -> tuple[F64, F64]:
  xi, wx = gauss_legendre_1d(3)
  return _meshgrid_2d(xi, wx)


def test_gauss_legendre_1d_njit_literal_order_matches_scipy() -> None:
  nodes, weights = _probe_gauss_legendre_1d_q8()
  assert gauss_legendre_1d.signatures
  ref_nodes, ref_weights = roots_legendre(_Q8_ORDER)
  np.testing.assert_allclose(nodes, ref_nodes, rtol=0, atol=1e-14)
  np.testing.assert_allclose(weights, ref_weights, rtol=0, atol=1e-14)


def test_gauss_tensor_product_2d_njit_literal_order_matches_scipy() -> None:
  points, weights = _probe_gauss_tensor_product_2d_q8()
  assert gauss_tensor_product_2d.signatures
  ref_points, ref_weights = _reference_gauss_tensor_product_2d(_Q8_ORDER)
  np.testing.assert_allclose(points, ref_points, rtol=0, atol=1e-14)
  np.testing.assert_allclose(weights, ref_weights, rtol=0, atol=1e-14)


def test_gauss_tensor_product_2d_njit_matches_decomposed_path() -> None:
  """``gauss_tensor_product_2d(3)`` and 1D+``_meshgrid_2d`` agree inside ``@njit``."""
  combined = _probe_gauss_tensor_product_2d_q8()
  decomposed = _probe_gauss_tensor_product_2d_decomposed_q8()
  np.testing.assert_allclose(combined[0], decomposed[0], rtol=0, atol=0.0)
  np.testing.assert_allclose(combined[1], decomposed[1], rtol=0, atol=0.0)


def test_dispatcher_and_nested_njit_agree_for_q8() -> None:
  d_nodes, d_weights = gauss_legendre_1d(_Q8_ORDER)
  n_nodes, n_weights = _probe_gauss_legendre_1d_q8()
  np.testing.assert_allclose(d_nodes, n_nodes, rtol=0, atol=0.0)
  np.testing.assert_allclose(d_weights, n_weights, rtol=0, atol=0.0)

  d_points, d_weights_2d = gauss_tensor_product_2d(_Q8_ORDER)
  n_points, n_weights_2d = _probe_gauss_tensor_product_2d_q8()
  np.testing.assert_allclose(d_points, n_points, rtol=0, atol=0.0)
  np.testing.assert_allclose(d_weights_2d, n_weights_2d, rtol=0, atol=0.0)


# cache=False: cross-module probe (NUMBA_CACHING.md §5; see first probe above).
@njit(cache=False)
def _probe_gauss_legendre_1d_variable_in_loop() -> None:
  for order in range(2, 5):
    gauss_legendre_1d(order)


# cache=False: cross-module probe (NUMBA_CACHING.md §5; see first probe above).
@njit(cache=False)
def _probe_gauss_tensor_product_2d_variable_in_loop() -> None:
  for order in range(2, 5):
    gauss_tensor_product_2d(order)


def test_gauss_legendre_1d_rejects_non_literal_order_in_njit_loop() -> None:
  with pytest.raises(TypingError, match="literal"):
    _probe_gauss_legendre_1d_variable_in_loop()


def test_gauss_tensor_product_2d_rejects_non_literal_order_in_njit_loop() -> None:
  with pytest.raises(TypingError, match="literal"):
    _probe_gauss_tensor_product_2d_variable_in_loop()


# --- NUMBA_CACHING.md §5 canary self-test --------------------------------------

_PLANTED_CACHE_TRUE = """\
from numba import njit

from pyfem.v3.fem.quadrature import gauss_tria3


@njit(cache=True)
def planted_cached() -> tuple:
  return gauss_tria3(1)
"""

_PLANTED_UNDOCUMENTED = """\
from numba import njit

from pyfem.v3.fem.quadrature import gauss_tria3


@njit(cache=False)
def planted_undocumented() -> tuple:
  return gauss_tria3(1)
"""

_PLANTED_DOCUMENTED = """\
from numba import njit

from pyfem.v3.fem.quadrature import gauss_tria3


# cache=False: cross-module callee gauss_tria3 (NUMBA_CACHING.md §5).
@njit(cache=False)
def planted_documented() -> tuple:
  return gauss_tria3(1)
"""

_PLANTED_SAME_FILE = """\
from numba import njit

from pyfem.v3.types import GROUP_TRUSS


@njit(cache=True)
def planted_helper(x: float) -> float:
  return 2.0 * x


@njit(cache=True)
def planted_friendly(x: float) -> float:
  return planted_helper(x) + 1.0


def plain_python() -> int:
  return GROUP_TRUSS
"""


def test_cache_policy_canary_fires_on_planted_violation(
  numba_cache_policy_scanner: Callable[[Path], list[str]],
  tmp_path: Path,
) -> None:
  """The §5 scanner flags planted violations and passes clean/documented sites."""
  (tmp_path / "planted_cached.py").write_text(_PLANTED_CACHE_TRUE)
  (tmp_path / "planted_undocumented.py").write_text(_PLANTED_UNDOCUMENTED)
  (tmp_path / "planted_documented.py").write_text(_PLANTED_DOCUMENTED)
  (tmp_path / "planted_friendly.py").write_text(_PLANTED_SAME_FILE)
  violations = numba_cache_policy_scanner(tmp_path)
  assert len(violations) == 2
  assert any("planted_cached" in v and "cache=True" in v for v in violations)
  assert any("planted_undocumented" in v and "documented" in v for v in violations)
