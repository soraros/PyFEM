"""Compiler-owned NumPy array finalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray

type ArrayOrder = Literal["C", "F"]


def finalize_array(
  source: ArrayLike,
  *,
  dtype: DTypeLike,
  order: ArrayOrder = "C",
) -> NDArray[np.generic]:
  """Detach, convert, make contiguous, and publish a read-only base array.

  ``copy=True`` is unconditional: even an already matching NumPy input cannot be
  reused. Object dtypes are rejected because copying their pointer table would not
  detach the referenced Python objects. NumPy subclasses are rejected because
  their additional semantics cannot be preserved by this generic boundary.
  """
  if isinstance(source, np.ndarray) and type(source) is not np.ndarray:
    msg = "compiler-owned array finalization requires an exact plain ndarray"
    raise TypeError(msg)
  if order not in ("C", "F"):
    msg = "array order must be 'C' or 'F'"
    raise ValueError(msg)

  target_dtype = np.dtype(dtype)
  if target_dtype.hasobject:
    msg = "compiler-owned arrays cannot use an object dtype"
    raise TypeError(msg)

  owned = np.array(
    source,
    dtype=target_dtype,
    order=order,
    copy=True,
    subok=False,
  )
  owned.setflags(write=False)
  return owned


@dataclass(frozen=True, slots=True, eq=False, init=False)
class FinalizedArray:
  """Small array-owning carrier with identity, never elementwise, equality."""

  values: NDArray[np.generic]

  def __init__(
    self,
    source: ArrayLike,
    *,
    dtype: DTypeLike,
    order: ArrayOrder = "C",
  ) -> None:
    object.__setattr__(
      self,
      "values",
      finalize_array(source, dtype=dtype, order=order),
    )

  def __eq__(self, other: object) -> bool:
    return self is other

  def __ne__(self, other: object) -> bool:
    return self is not other

  __hash__ = object.__hash__
