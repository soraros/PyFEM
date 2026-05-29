"""2D element ↔ global coordinate transforms."""

from __future__ import annotations

import numpy as np
from numba import njit

from pyfem.v3.types import F64


@njit(cache=True)
def element_length_2d(el_coords: F64) -> float:
  dx = el_coords[1, 0] - el_coords[0, 0]
  dy = el_coords[1, 1] - el_coords[0, 1]
  return np.sqrt(dx * dx + dy * dy)


@njit(cache=True)
def rotation_matrix_2d(el_coords: F64) -> F64:
  """Rotation from global to element coordinates (2-node line element)."""
  length = element_length_2d(el_coords)
  cos_alpha = (el_coords[1, 0] - el_coords[0, 0]) / length
  sin_alpha = (el_coords[1, 1] - el_coords[0, 1]) / length
  rot = np.empty((2, 2), dtype=np.float64)
  rot[0, 0] = cos_alpha
  rot[0, 1] = sin_alpha
  rot[1, 0] = -sin_alpha
  rot[1, 1] = cos_alpha
  return rot


@njit(cache=True)
def to_element_vector_4(a: F64, el_coords: F64) -> F64:
  rot = rotation_matrix_2d(el_coords)
  out = np.empty(4, dtype=np.float64)
  for block in range(2):
    i0 = 2 * block
    out[i0] = rot[0, 0] * a[i0] + rot[0, 1] * a[i0 + 1]
    out[i0 + 1] = rot[1, 0] * a[i0] + rot[1, 1] * a[i0 + 1]
  return out


@njit(cache=True)
def local_to_global_4(k_bar: F64, f_bar: F64, el_coords: F64) -> tuple[F64, F64]:
  """Rotate local 4×4 stiffness and 4-vector force to global coordinates."""
  rot = rotation_matrix_2d(el_coords)
  k = np.zeros((4, 4), dtype=np.float64)
  f = np.empty(4, dtype=np.float64)
  for block in range(2):
    i0 = 2 * block
    f[i0] = rot[0, 0] * f_bar[i0] + rot[1, 0] * f_bar[i0 + 1]
    f[i0 + 1] = rot[0, 1] * f_bar[i0] + rot[1, 1] * f_bar[i0 + 1]
  for i_block in range(2):
    ir0 = 2 * i_block
    for j_block in range(2):
      jc0 = 2 * j_block
      for ii in range(2):
        for jj in range(2):
          s = 0.0
          for kk in range(2):
            for ll in range(2):
              s += rot[ii, kk] * k_bar[ir0 + kk, jc0 + ll] * rot[jj, ll]
          k[ir0 + ii, jc0 + jj] = s
  return k, f
