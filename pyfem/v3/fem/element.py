"""Element-level operators."""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from pyfem.v3.fem.kinematics import physical_gradients, strain_displacement
from pyfem.v3.fem.quadrature import gauss_tensor_product_2d
from pyfem.v3.fem.shapes import serendipity_quad8
from pyfem.v3.types import F64


@njit(cache=True, parallel=True)
def _stiffness_from_coords_batched(nodal_coords: F64, constitutive: F64) -> F64:
  """Batched element stiffness, ``nodal_coords`` shape ``(n_elems, n_nodes, 2)``."""
  parent_pts, parent_w = gauss_tensor_product_2d(3)
  _, dN = serendipity_quad8(parent_pts)
  jacobian, grad_n = physical_gradients(nodal_coords, dN)
  b = strain_displacement(grad_n)

  n_elems, n_gp = nodal_coords.shape[0], parent_w.shape[0]
  n_dof = b.shape[-1]
  stiffness = np.zeros((n_elems, n_dof, n_dof))

  for e in prange(n_elems):
    for p in range(n_gp):
      jac = jacobian[e, p]
      weight = parent_w[p] * abs(np.linalg.det(jac))
      stiffness[e] += weight * (b[e, p].T @ constitutive @ b[e, p])

  return stiffness


def quad8_plane_stress_stiffness(
  nodal_coords: F64,
  constitutive: F64,
) -> F64:
  r"""Element stiffness K_e = ∫_Ω Bᵀ C B dΩ (plane stress)."""
  if nodal_coords.ndim == 2:
    return _stiffness_from_coords_batched(nodal_coords[None, ...], constitutive)[0]
  return _stiffness_from_coords_batched(nodal_coords, constitutive)
