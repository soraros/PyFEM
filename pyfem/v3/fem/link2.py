"""2-node link elements: corotational truss and axial spring."""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from pyfem.v3 import types as _types
from pyfem.v3.types import F64

# Group-kind constants mirror pyfem.v3.types but are defined locally so the
# cache=True kernels below read same-file globals: numba freezes cross-file
# global reads at compile time and never invalidates them (NUMBA_CACHING.md
# §5). The import-time guard keeps the copies from drifting.
GROUP_TRUSS = 1
GROUP_SPRING = 2

if (GROUP_TRUSS, GROUP_SPRING) != (_types.GROUP_TRUSS, _types.GROUP_SPRING):
  msg = "link2 GROUP_* constants drifted from pyfem.v3.types"
  raise RuntimeError(msg)


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
              s += rot[kk, ii] * k_bar[ir0 + kk, jc0 + ll] * rot[ll, jj]
          k[ir0 + ii, jc0 + jj] = s
  return k, f


@njit(cache=True)
def _truss_local(
  a: F64,
  l0: float,
  youngs_modulus: float,
  area: float,
) -> tuple[F64, F64]:
  du = (a[2] - a[0]) / l0
  dv = (a[3] - a[1]) / l0
  epsilon = du + 0.5 * du * du + 0.5 * dv * dv
  sigma = youngs_modulus * epsilon
  bl = np.zeros(4, dtype=np.float64)
  bl[0] = (-1.0 / l0) * (1.0 + du)
  bl[1] = (-1.0 / l0) * dv
  bl[2] = -bl[0]
  bl[3] = -bl[1]
  kl = youngs_modulus * area * l0 * np.outer(bl, bl)
  coeff = sigma * area / l0
  knl = np.zeros((4, 4), dtype=np.float64)
  knl[0, 0] = coeff
  knl[0, 2] = -coeff
  knl[1, 1] = coeff
  knl[1, 3] = -coeff
  knl[2, 0] = -coeff
  knl[2, 2] = coeff
  knl[3, 1] = -coeff
  knl[3, 3] = coeff
  k_bar = kl + knl
  f_bar = l0 * sigma * area * bl
  return k_bar, f_bar


@njit(cache=True)
def _spring_local(a: F64, stiffness_k: float) -> tuple[F64, F64]:
  elong = a[2] - a[0]
  force = elong * stiffness_k
  k_bar = np.zeros((4, 4), dtype=np.float64)
  k_bar[0, 0] = stiffness_k
  k_bar[0, 2] = -stiffness_k
  k_bar[2, 0] = -stiffness_k
  k_bar[2, 2] = stiffness_k
  f_bar = np.array([-force, 0.0, force, 0.0], dtype=np.float64)
  return k_bar, f_bar


@njit(cache=True)
def link2_tangent_single(
  el_coords: F64,
  el_state: F64,
  group_kind: int,
  prop0: float,
  prop1: float,
) -> tuple[F64, F64]:
  """Single 2-node element tangent and internal force in global coordinates."""
  a = to_element_vector_4(el_state, el_coords)
  if group_kind == GROUP_TRUSS:
    k_bar, f_bar = _truss_local(a, element_length_2d(el_coords), prop0, prop1)
  elif group_kind == GROUP_SPRING:
    k_bar, f_bar = _spring_local(a, prop0)
  else:
    msg = f"Unsupported link group kind {group_kind}"
    raise ValueError(msg)
  return local_to_global_4(k_bar, f_bar, el_coords)


@njit(cache=True, parallel=True)
def link2_tangent_batched(
  nodal_coords: F64,
  element_state: F64,
  group_kind: int,
  prop0: float,
  prop1: float,
) -> tuple[F64, F64]:
  """Batched 2-node tangent and internal force (uniform kind and props per batch)."""
  n_elems = nodal_coords.shape[0]
  stiffness = np.zeros((n_elems, 4, 4), dtype=np.float64)
  fint = np.zeros((n_elems, 4), dtype=np.float64)
  for e in prange(n_elems):
    ke, fe = link2_tangent_single(
      nodal_coords[e],
      element_state[e],
      group_kind,
      prop0,
      prop1,
    )
    stiffness[e] = ke
    fint[e] = fe
  return stiffness, fint
