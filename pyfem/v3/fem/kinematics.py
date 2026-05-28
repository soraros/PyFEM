"""Kinematic operators (strain–displacement, etc.)."""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from pyfem.v3.types import F64


@njit(cache=True, parallel=True)
def physical_gradients(
  nodal_coords: F64,
  parent_gradients: F64,
) -> tuple[F64, F64]:
  """
  Map shape-function gradients to physical space (batched).

  Parameters
  ----------
  nodal_coords
      ``(n_elems, n_nodes, spatial_dim)``.
  parent_gradients
      ``(n_points, n_nodes, spatial_dim)`` ∂N/∂(xi, eta, ...).
  """
  n_elems = nodal_coords.shape[0]
  n_pts = parent_gradients.shape[0]
  n_nodes = nodal_coords.shape[1]
  jacobian = np.empty((n_elems, n_pts, 2, 2))
  grad_n = np.empty((n_elems, n_pts, n_nodes, 2))
  for e in prange(n_elems):
    xt = nodal_coords[e].T
    for p in range(n_pts):
      jac = xt @ parent_gradients[p]
      jacobian[e, p] = jac
      grad_n[e, p] = parent_gradients[p] @ np.linalg.inv(jac)
  return jacobian, grad_n


@njit(cache=True, parallel=True)
def strain_displacement(grad_n: F64) -> F64:
  """
  Plane 2D strain–displacement operator B.

  Parameters
  ----------
  grad_n
      ``(n_elems, n_points, n_nodes, 2)`` with columns ∂N/∂x, ∂N/∂y.

  Returns
  -------
  B
      ``(n_elems, n_points, 3, 2 * n_nodes)`` so ε = B @ u_e at each point.
  """
  n_elems, n_pts, n_nodes, _ = grad_n.shape
  n_dof = 2 * n_nodes
  b = np.zeros((n_elems, n_pts, 3, n_dof), dtype=np.float64)
  b[..., 0, 0::2] = grad_n[..., :, 0]
  b[..., 1, 1::2] = grad_n[..., :, 1]
  b[..., 2, 0::2] = grad_n[..., :, 1]
  b[..., 2, 1::2] = grad_n[..., :, 0]
  return b
