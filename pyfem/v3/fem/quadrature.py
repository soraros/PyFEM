"""Gauss–Legendre quadrature on reference intervals and products."""

from __future__ import annotations

import numpy as np
from numba import literally, njit
from numba.core import types
from numba.core.errors import TypingError
from numba.extending import overload
from scipy.special import roots_legendre

from pyfem.v3.types import F64


def _gauss_legendre_1d_impl(order: int) -> tuple[F64, F64]:  # type: ignore
  pass


@overload(_gauss_legendre_1d_impl)
def _gauss_legendre_1d(order):
  if not isinstance(order, types.IntegerLiteral):
    raise TypingError("The 'order' argument must be a compile-time literal integer.")

  val = order.literal_value
  n, w = roots_legendre(val)
  n, w = n.astype(np.float64), w.astype(np.float64)

  def impl(order):
    return n, w

  return impl


@njit(cache=True)
def gauss_legendre_1d(order: int) -> tuple[F64, F64]:
  return _gauss_legendre_1d_impl(literally(order))


@njit(cache=True)
def _meshgrid_2d(xi: F64, wx: F64) -> tuple[F64, F64]:
  """Tensor product of identical 1D rules (Numba-safe, no ``meshgrid``)."""
  m = xi.shape[0]
  n = m * m
  points = np.empty((n, 2), dtype=np.float64)
  weights = np.empty(n, dtype=np.float64)
  for i in range(m):
    for j in range(m):
      k = i * m + j
      points[k, 0] = xi[i]
      points[k, 1] = xi[j]
      weights[k] = wx[i] * wx[j]
  return points, weights


@njit(cache=True)
def gauss_tensor_product_2d(order: int) -> tuple[F64, F64]:
  xi, wx = gauss_legendre_1d(order)
  return _meshgrid_2d(xi, wx)
