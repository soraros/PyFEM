"""Element-level operators."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numba import njit, prange

from pyfem.v3.fem.kinematics import physical_gradients, strain_displacement
from pyfem.v3.fem.quadrature import gauss_tensor_product_2d, gauss_tria3
from pyfem.v3.fem.shapes import bilinear_quad4, linear_tria3, serendipity_quad8
from pyfem.v3.types import F64


def _wrap_batched(
  batched_fn: Callable[[F64, F64], F64],
  nodal_coords: F64,
  constitutive: F64,
) -> F64:
  if nodal_coords.ndim == 2:
    return batched_fn(nodal_coords[None, ...], constitutive)[0]
  return batched_fn(nodal_coords, constitutive)


@njit(cache=True, parallel=True)
def _quad8_stiffness_from_coords_batched(nodal_coords: F64, constitutive: F64) -> F64:
  """Batched Q8 stiffness, ``nodal_coords`` shape ``(n_elems, 8, 2)``."""
  parent_pts, parent_w = gauss_tensor_product_2d(3)
  _, dN = serendipity_quad8(parent_pts)
  grad_n, det_j = physical_gradients(nodal_coords, dN)
  b = strain_displacement(grad_n)

  n_elems, n_gp = nodal_coords.shape[0], parent_w.shape[0]
  n_dof = b.shape[-1]
  stiffness = np.zeros((n_elems, n_dof, n_dof))

  for e in prange(n_elems):
    for p in range(n_gp):
      weight = parent_w[p] * abs(det_j[e, p])
      stiffness[e] += weight * (b[e, p].T @ constitutive @ b[e, p])

  return stiffness


@njit(cache=True, parallel=True)
def _quad4_stiffness_from_coords_batched(nodal_coords: F64, constitutive: F64) -> F64:
  """Batched Quad4 stiffness, ``nodal_coords`` shape ``(n_elems, 4, 2)``."""
  parent_pts, parent_w = gauss_tensor_product_2d(2)
  _, dN = bilinear_quad4(parent_pts)
  grad_n, det_j = physical_gradients(nodal_coords, dN)
  b = strain_displacement(grad_n)

  n_elems, n_gp = nodal_coords.shape[0], parent_w.shape[0]
  n_dof = b.shape[-1]
  stiffness = np.zeros((n_elems, n_dof, n_dof))

  for e in prange(n_elems):
    for p in range(n_gp):
      weight = parent_w[p] * abs(det_j[e, p])
      stiffness[e] += weight * (b[e, p].T @ constitutive @ b[e, p])

  return stiffness


@njit(cache=True, parallel=True)
def _tria3_stiffness_from_coords_batched(nodal_coords: F64, constitutive: F64) -> F64:
  """Batched Tria3 stiffness, ``nodal_coords`` shape ``(n_elems, 3, 2)``."""
  parent_pts, parent_w = gauss_tria3(1)
  _, dN = linear_tria3(parent_pts)
  grad_n, det_j = physical_gradients(nodal_coords, dN)
  b = strain_displacement(grad_n)

  n_elems, n_gp = nodal_coords.shape[0], parent_w.shape[0]
  n_dof = b.shape[-1]
  stiffness = np.zeros((n_elems, n_dof, n_dof))

  for e in prange(n_elems):
    for p in range(n_gp):
      weight = parent_w[p] * abs(det_j[e, p])
      stiffness[e] += weight * (b[e, p].T @ constitutive @ b[e, p])

  return stiffness


def quad8_plane_stress_stiffness(
  nodal_coords: F64,
  constitutive: F64,
) -> F64:
  r"""Element stiffness K_e = ∫_Ω Bᵀ C B dΩ (plane stress, Q8)."""
  return _wrap_batched(_quad8_stiffness_from_coords_batched, nodal_coords, constitutive)


def quad4_plane_stress_stiffness(
  nodal_coords: F64,
  constitutive: F64,
) -> F64:
  r"""Element stiffness K_e = ∫_Ω Bᵀ C B dΩ (plane stress, Quad4)."""
  return _wrap_batched(_quad4_stiffness_from_coords_batched, nodal_coords, constitutive)


def tria3_plane_stress_stiffness(
  nodal_coords: F64,
  constitutive: F64,
) -> F64:
  r"""Element stiffness K_e = ∫_Ω Bᵀ C B dΩ (plane stress, Tria3)."""
  return _wrap_batched(_tria3_stiffness_from_coords_batched, nodal_coords, constitutive)


# Backward-compatible alias for benchmarks and internal callers.
_stiffness_from_coords_batched = _quad8_stiffness_from_coords_batched
