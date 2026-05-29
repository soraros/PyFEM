"""2-node link elements: corotational truss and axial spring."""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from pyfem.v3.fem.transforms import (
  element_length_2d,
  local_to_global_4,
  to_element_vector_4,
)
from pyfem.v3.types import F64, GROUP_SPRING, GROUP_TRUSS


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
