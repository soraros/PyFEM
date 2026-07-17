"""Compiler-owned NumPy array finalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, final

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray

type ArrayOrder = Literal["C", "F"]

_PYTHON_ARRAY_SCALAR_TYPES = (bool, int, float, complex, str, bytes)
_NUMPY_ARRAY_SCALAR_TYPES = tuple(
  scalar_type
  for scalar_type in np.ScalarType
  if isinstance(scalar_type, type) and issubclass(scalar_type, np.generic)
)
_TARGET_DTYPE_SCALAR_TYPES = (
  bool,
  int,
  float,
  complex,
  str,
  bytes,
  object,
  *_NUMPY_ARRAY_SCALAR_TYPES,
)
_NUMPY_DTYPE_TYPES = tuple(
  {type(np.dtype(scalar_type)) for scalar_type in _TARGET_DTYPE_SCALAR_TYPES}
)


def _type_derives_from(value_type: type[object], parent: type[object]) -> bool:
  """Check a real type hierarchy without consulting an instance's hooks."""
  return any(
    candidate is parent for candidate in type.__getattribute__(value_type, "__mro__")
  )


def _dtype_has_metadata(dtype: np.dtype[np.generic]) -> bool:
  """Return whether ``dtype`` or any nested dtype carries metadata."""
  if dtype.metadata is not None:
    return True
  if dtype.fields is not None and any(
    _dtype_has_metadata(field[0]) for field in dtype.fields.values()
  ):
    return True
  if dtype.subdtype is not None:
    return _dtype_has_metadata(dtype.subdtype[0])
  return False


def _validate_source_dtype(dtype: np.dtype[np.generic]) -> None:
  if _dtype_has_metadata(dtype):
    msg = "compiler-owned array sources cannot use dtype metadata"
    raise TypeError(msg)
  if dtype.hasobject:
    msg = "compiler-owned array sources cannot use an object dtype"
    raise TypeError(msg)


def _capture_target_dtype(dtype: DTypeLike) -> np.dtype[np.generic]:
  """Admit only target dtype forms that cannot run foreign conversion hooks."""
  dtype_type = type(dtype)
  if any(dtype_type is trusted_type for trusted_type in _NUMPY_DTYPE_TYPES):
    return dtype
  if dtype_type is str or any(
    dtype is scalar_type for scalar_type in _TARGET_DTYPE_SCALAR_TYPES
  ):
    return np.dtype(dtype)
  msg = (
    "compiler-owned arrays require an exact dtype string, trusted scalar dtype "
    "class, or NumPy dtype"
  )
  raise TypeError(msg)


def _preflight_array_source(value: object, active: set[int]) -> None:
  """Validate the exact safe ArrayLike grammar without coercing ``value``."""
  value_type = type(value)
  if value_type is np.ndarray:
    _validate_source_dtype(value.dtype)
    return
  if any(value_type is scalar_type for scalar_type in _PYTHON_ARRAY_SCALAR_TYPES):
    return
  if any(value_type is scalar_type for scalar_type in _NUMPY_ARRAY_SCALAR_TYPES):
    _validate_source_dtype(value.dtype)
    return
  if value_type is list or value_type is tuple:
    marker = id(value)
    if marker in active:
      msg = "compiler-owned array sources cannot contain reference cycles"
      raise ValueError(msg)
    active.add(marker)
    try:
      for item in value:
        _preflight_array_source(item, active)
    finally:
      active.remove(marker)
    return
  if _type_derives_from(value_type, np.ndarray):
    msg = "compiler-owned array finalization requires an exact plain ndarray"
    raise TypeError(msg)
  if _type_derives_from(value_type, np.generic):
    msg = "compiler-owned array finalization requires exact NumPy scalar types"
    raise TypeError(msg)
  if _type_derives_from(value_type, list) or _type_derives_from(value_type, tuple):
    msg = "compiler-owned array finalization requires exact list/tuple containers"
    raise TypeError(msg)
  msg = (
    "compiler-owned array sources require exact scalars, list/tuple containers, "
    "or plain ndarrays"
  )
  raise TypeError(msg)


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
  if type(order) is not str or order not in ("C", "F"):
    msg = "array order must be 'C' or 'F'"
    raise ValueError(msg)

  target_dtype = _capture_target_dtype(dtype)
  if _dtype_has_metadata(target_dtype):
    msg = "compiler-owned arrays cannot use dtype metadata"
    raise TypeError(msg)
  if target_dtype.hasobject:
    msg = "compiler-owned arrays cannot use an object dtype"
    raise TypeError(msg)
  _preflight_array_source(source, set())

  owned = np.array(
    source,
    dtype=target_dtype,
    order=order,
    copy=True,
    subok=False,
  )
  owned.setflags(write=False)
  return owned


@final
@dataclass(frozen=True, slots=True, eq=False, init=False)
class FinalizedArray:
  """Small array-owning carrier with identity, never elementwise, equality."""

  __array_priority__ = 1000.0

  values: NDArray[np.generic]

  def __init_subclass__(cls, **kwargs: object) -> None:
    del kwargs
    msg = "FinalizedArray is runtime-final and cannot be subclassed"
    raise TypeError(msg)

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

  def __array_ufunc__(
    self,
    ufunc: np.ufunc,
    method: str,
    *inputs: object,
    **kwargs: object,
  ) -> object:
    if method != "__call__" or (ufunc is not np.equal and ufunc is not np.not_equal):
      return NotImplemented
    if len(inputs) != 2:
      return NotImplemented
    if kwargs:
      msg = "FinalizedArray comparison ufuncs do not accept keyword arguments"
      raise TypeError(msg)
    identical = inputs[0] is inputs[1]
    return identical if ufunc is np.equal else not identical

  __hash__ = object.__hash__
