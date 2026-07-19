"""Private allocation-bounded numeric primitives for linear statics."""

from __future__ import annotations

import math

import numpy as np


def _strict_ratio_greater(value: float, scale: float, threshold: float) -> bool:
  """Compare positive binary64 ratios exactly without rounded division."""
  if (
    type(value) is not float
    or type(scale) is not float
    or type(threshold) is not float
    or not math.isfinite(value)
    or not math.isfinite(scale)
    or not math.isfinite(threshold)
    or value <= 0.0
    or scale <= 0.0
    or threshold <= 0.0
  ):
    return False
  value_numerator, value_denominator = value.as_integer_ratio()
  scale_numerator, scale_denominator = scale.as_integer_ratio()
  threshold_numerator, threshold_denominator = threshold.as_integer_ratio()
  return (
    value_numerator * scale_denominator * threshold_denominator
    > threshold_numerator * scale_numerator * value_denominator
  )


def _explicit_cholesky(
  operator: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
  """Factor one symmetric float64 matrix with one declared matrix output."""
  if (
    type(operator) is not np.ndarray
    or operator.dtype != np.dtype(np.float64)
    or operator.dtype.metadata is not None
    or operator.ndim != 2
    or operator.shape[0] != operator.shape[1]
  ):
    return None
  size = operator.shape[0]
  factor = np.zeros((size, size), dtype=np.float64)
  pivots = np.empty(size, dtype=np.float64)
  for row in range(size):
    for column in range(row + 1):
      product = 0.0
      for inner in range(column):
        product += float(factor[row, inner]) * float(factor[column, inner])
      remainder = float(operator[row, column]) - product
      if not math.isfinite(remainder):
        return None
      if row == column:
        if remainder <= 0.0:
          return None
        diagonal = math.sqrt(remainder)
        if not math.isfinite(diagonal) or diagonal <= 0.0:
          return None
        factor[row, column] = diagonal
        pivot = diagonal * diagonal
        if not math.isfinite(pivot) or pivot <= 0.0:
          return None
        pivots[row] = pivot
      else:
        quotient = remainder / float(factor[column, column])
        if not math.isfinite(quotient):
          return None
        factor[row, column] = quotient
  return factor, pivots
