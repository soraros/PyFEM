"""Element-level operators."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numba import njit, prange

from pyfem.v3.fem.kinematics import (
  physical_gradients,
  physical_gradients_3d,
  strain_displacement,
  strain_displacement_3d,
)
from pyfem.v3.fem.quadrature import (
  gauss_tensor_product_2d,
  gauss_tensor_product_3d,
  gauss_tet4,
  gauss_tria3,
)
from pyfem.v3.fem.shapes import (
  bilinear_quad4,
  linear_tet4,
  linear_tria3,
  serendipity_quad8,
  trilinear_hex8,
)
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
def _integrate_btcb_batched(
  parent_w: F64,
  det_j: F64,
  b: F64,
  constitutive: F64,
) -> F64:
  """Integrate ``B.T @ C @ B`` over Gauss points for a batched mesh."""
  n_elems = det_j.shape[0]
  n_gp = parent_w.shape[0]
  n_dof = b.shape[-1]
  stiffness = np.zeros((n_elems, n_dof, n_dof), dtype=np.float64)
  for e in prange(n_elems):
    for p in range(n_gp):
      weight = parent_w[p] * abs(det_j[e, p])
      stiffness[e] += weight * (b[e, p].T @ constitutive @ b[e, p])
  return stiffness


@njit(cache=True, parallel=True)
def _quad8_stiffness_from_coords_batched(nodal_coords: F64, constitutive: F64) -> F64:
  """Batched Q8 stiffness, ``nodal_coords`` shape ``(n_elems, 8, 2)``."""
  parent_pts, parent_w = gauss_tensor_product_2d(3)
  _, dN = serendipity_quad8(parent_pts)
  n_elems = nodal_coords.shape[0]
  n_gp = parent_w.shape[0]
  n_nodes = nodal_coords.shape[1]
  n_dof = 2 * n_nodes
  stiffness = np.zeros((n_elems, n_dof, n_dof), dtype=np.float64)

  for e in prange(n_elems):
    xt = nodal_coords[e].T
    for p in range(n_gp):
      jac = xt @ dN[p]
      det_j = abs(np.linalg.det(jac))
      grad_n = dN[p] @ np.linalg.inv(jac)
      b = np.zeros((3, n_dof))
      b[0, 0::2] = grad_n[:, 0]
      b[1, 1::2] = grad_n[:, 1]
      b[2, 0::2] = grad_n[:, 1]
      b[2, 1::2] = grad_n[:, 0]
      weight = parent_w[p] * det_j
      stiffness[e] += weight * (b.T @ constitutive @ b)

  return stiffness


@njit(cache=True, parallel=True)
def _quad4_stiffness_from_coords_batched(nodal_coords: F64, constitutive: F64) -> F64:
  parent_pts, parent_w = gauss_tensor_product_2d(2)
  _, dN = bilinear_quad4(parent_pts)
  grad_n, det_j = physical_gradients(nodal_coords, dN)
  b = strain_displacement(grad_n)
  return _integrate_btcb_batched(parent_w, det_j, b, constitutive)


@njit(cache=True, parallel=True)
def _tria3_stiffness_from_coords_batched(nodal_coords: F64, constitutive: F64) -> F64:
  parent_pts, parent_w = gauss_tria3(1)
  _, dN = linear_tria3(parent_pts)
  grad_n, det_j = physical_gradients(nodal_coords, dN)
  b = strain_displacement(grad_n)
  return _integrate_btcb_batched(parent_w, det_j, b, constitutive)


@njit(cache=True, parallel=True)
def _hex8_stiffness_from_coords_batched(nodal_coords: F64, constitutive: F64) -> F64:
  parent_pts, parent_w = gauss_tensor_product_3d(2)
  _, dN = trilinear_hex8(parent_pts)
  grad_n, det_j = physical_gradients_3d(nodal_coords, dN)
  b = strain_displacement_3d(grad_n)
  return _integrate_btcb_batched(parent_w, det_j, b, constitutive)


@njit(cache=True, parallel=True)
def _tet4_stiffness_from_coords_batched(nodal_coords: F64, constitutive: F64) -> F64:
  parent_pts, parent_w = gauss_tet4(1)
  _, dN = linear_tet4(parent_pts)
  grad_n, det_j = physical_gradients_3d(nodal_coords, dN)
  b = strain_displacement_3d(grad_n)
  return _integrate_btcb_batched(parent_w, det_j, b, constitutive)


def continuum_stiffness_batched(
  nodal_coords: F64,
  constitutive: F64,
) -> F64:
  """Dispatch batched small-strain continuum stiffness by mesh rank and nodes/elem."""
  spatial_dim = int(nodal_coords.shape[-1])
  n_nodes = int(nodal_coords.shape[-2])
  if spatial_dim == 2:
    if n_nodes == 8:
      return _quad8_stiffness_from_coords_batched(nodal_coords, constitutive)
    if n_nodes == 4:
      return _quad4_stiffness_from_coords_batched(nodal_coords, constitutive)
    if n_nodes == 3:
      return _tria3_stiffness_from_coords_batched(nodal_coords, constitutive)
  elif spatial_dim == 3:
    if n_nodes == 8:
      return _hex8_stiffness_from_coords_batched(nodal_coords, constitutive)
    if n_nodes == 4:
      return _tet4_stiffness_from_coords_batched(nodal_coords, constitutive)
  msg = (
    f"Unsupported element with {n_nodes} nodes per element "
    f"in {spatial_dim}D"
  )
  raise ValueError(msg)


def quad8_plane_stress_stiffness(nodal_coords: F64, constitutive: F64) -> F64:
  r"""Element stiffness K_e = ∫_Ω Bᵀ C B dΩ (plane stress, Q8)."""
  return _wrap_batched(_quad8_stiffness_from_coords_batched, nodal_coords, constitutive)


def quad4_plane_stress_stiffness(nodal_coords: F64, constitutive: F64) -> F64:
  r"""Element stiffness K_e = ∫_Ω Bᵀ C B dΩ (plane stress, Quad4)."""
  return _wrap_batched(_quad4_stiffness_from_coords_batched, nodal_coords, constitutive)


def tria3_plane_stress_stiffness(nodal_coords: F64, constitutive: F64) -> F64:
  r"""Element stiffness K_e = ∫_Ω Bᵀ C B dΩ (plane stress, Tria3)."""
  return _wrap_batched(_tria3_stiffness_from_coords_batched, nodal_coords, constitutive)


def hex8_stiffness(nodal_coords: F64, constitutive: F64) -> F64:
  r"""Element stiffness K_e = ∫_Ω Bᵀ C B dΩ (3D, Hex8)."""
  return _wrap_batched(_hex8_stiffness_from_coords_batched, nodal_coords, constitutive)


def tet4_stiffness(nodal_coords: F64, constitutive: F64) -> F64:
  r"""Element stiffness K_e = ∫_Ω Bᵀ C B dΩ (3D, Tet4)."""
  return _wrap_batched(_tet4_stiffness_from_coords_batched, nodal_coords, constitutive)


# Backward-compatible alias for benchmarks and internal callers.
_stiffness_from_coords_batched = _quad8_stiffness_from_coords_batched
