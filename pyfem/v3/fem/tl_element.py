"""Total Lagrangian element kernels."""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from pyfem.v3.fem.quadrature import gauss_tensor_product_2d
from pyfem.v3.fem.shapes import serendipity_quad8
from pyfem.v3.fem.tl_kinematics import (
  bnl_matrix_2d,
  deformation_gradient_2d,
  green_lagrange_strain_voigt_2d,
  pk2_stress_voigt_2d,
  stress_to_matrix_2d,
  tl_b_matrix_2d,
)
from pyfem.v3.types import F64


@njit(cache=True, parallel=True)
def quad8_tl_tangent_batched(
  nodal_coords: F64,
  element_state: F64,
  constitutive: F64,
) -> tuple[F64, F64]:
  """
  Batched Q8 TL tangent stiffness and element internal force.

  Parameters
  ----------
  nodal_coords
      ``(n_elems, 8, 2)`` reference coordinates.
  element_state
      ``(n_elems, 16)`` element displacements ``[u0, v0, u1, v1, ...]``.
  constitutive
      ``(3, 3)`` plane-stress tangent / elasticity matrix.
  """
  parent_pts, parent_w = gauss_tensor_product_2d(3)
  _, dN = serendipity_quad8(parent_pts)
  n_elems = nodal_coords.shape[0]
  n_gp = parent_w.shape[0]
  n_nodes = 8
  n_dof = 2 * n_nodes
  stiffness = np.zeros((n_elems, n_dof, n_dof), dtype=np.float64)
  fint = np.zeros((n_elems, n_dof), dtype=np.float64)

  for e in prange(n_elems):
    elstate = element_state[e]
    xt = nodal_coords[e].T
    ke = stiffness[e]
    fe = fint[e]
    for p in range(n_gp):
      jac = xt @ dN[p]
      det_j = abs(np.linalg.det(jac))
      grad_n = dN[p] @ np.linalg.inv(jac)
      weight = parent_w[p] * det_j

      f = deformation_gradient_2d(grad_n, elstate, n_nodes)
      strain = green_lagrange_strain_voigt_2d(f)
      sigma = pk2_stress_voigt_2d(constitutive, strain)
      b = tl_b_matrix_2d(grad_n, f, n_nodes, n_dof)
      bnl = bnl_matrix_2d(grad_n, n_nodes, n_dof)
      t = stress_to_matrix_2d(sigma)

      cb = constitutive @ b
      ke += weight * (b.T @ cb)
      ke += weight * (bnl.T @ (t @ bnl))
      fe += weight * (b.T @ sigma)

  return stiffness, fint
