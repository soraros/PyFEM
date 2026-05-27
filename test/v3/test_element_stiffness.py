# SPDX-License-Identifier: MIT

"""Q8 plane-stress stiffness: Numba batched vs NumPy reference."""

from __future__ import annotations

import sys

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

pytest.importorskip("numba")

from pyfem.v3 import load_problem
from pyfem.v3.fem.element import quad8_plane_stress_stiffness
from pyfem.v3.fem.quadrature import gauss_tensor_product_2d
from pyfem.v3.fem.shapes import serendipity_quad8
from pyfem.v3.materials.plane_stress import plane_stress_matrix

_STIFFNESS_ATOL = 1e-8


def _physical_gradients_numpy(
  nodal_coords: np.ndarray,
  parent_gradients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  single = nodal_coords.ndim == 2
  if single:
    nodal_coords = nodal_coords[None, ...]
  jacobian = np.einsum("...na,pnb->...pab", nodal_coords, parent_gradients)
  jacobian_inv = np.linalg.inv(jacobian)
  grad_n = np.einsum("...pnb,...pbc->...pnc", parent_gradients, jacobian_inv)
  return jacobian, grad_n


def _strain_displacement_numpy(grad_n: np.ndarray) -> np.ndarray:
  *lead, n_nodes, _ = grad_n.shape
  n_dof = 2 * n_nodes
  b = np.zeros((*lead, 3, n_dof), dtype=np.float64)
  b[..., 0, 0::2] = grad_n[..., :, 0]
  b[..., 1, 1::2] = grad_n[..., :, 1]
  b[..., 2, 0::2] = grad_n[..., :, 1]
  b[..., 2, 1::2] = grad_n[..., :, 0]
  return b


def _stiffness_numpy(nodal_coords: np.ndarray, constitutive: np.ndarray) -> np.ndarray:
  single = nodal_coords.ndim == 2
  if single:
    nodal_coords = nodal_coords[None, ...]

  parent_pts, parent_w = gauss_tensor_product_2d(3)
  _, dN_parent = serendipity_quad8(parent_pts)
  _jacobian, grad_n = _physical_gradients_numpy(nodal_coords, dN_parent)
  det_j = np.linalg.det(_jacobian)
  b = _strain_displacement_numpy(grad_n)

  measure = parent_w * np.abs(det_j)
  bt_c_b = np.einsum("...paj,ac,...pck->...pjk", b, constitutive, b)
  stiffness = np.einsum("...p,...pjk->...jk", measure, bt_c_b)
  return stiffness[0] if single else stiffness


def _make_coords(n_elems: int, rng: np.random.Generator) -> np.ndarray:
  base = np.array(
    [
      [0.0, 0.0],
      [1.0, 0.0],
      [1.0, 1.0],
      [0.0, 1.0],
      [0.5, 0.0],
      [1.0, 0.5],
      [0.5, 1.0],
      [0.0, 0.5],
    ],
    dtype=np.float64,
  )
  out = np.empty((n_elems, 8, 2), dtype=np.float64)
  for e in range(n_elems):
    out[e] = base + 0.02 * rng.standard_normal((8, 2))
  return out


@pytest.fixture(scope="module")
def constitutive() -> np.ndarray:
  return plane_stress_matrix(1.0e6, 0.25)


def test_quad8_stiffness_single_element_matches_numpy(
  constitutive: np.ndarray,
) -> None:
  rng = np.random.default_rng(0)
  coords = _make_coords(1, rng)[0]
  k_np = _stiffness_numpy(coords, constitutive)
  k_nb = quad8_plane_stress_stiffness(coords, constitutive)
  np.testing.assert_allclose(k_nb, k_np, rtol=0.0, atol=_STIFFNESS_ATOL)


def test_quad8_stiffness_patch_test8_batch_matches_numpy(
  constitutive: np.ndarray,
) -> None:
  loaded = load_problem("skims/patch_test8/problem.toml").problem
  coords = loaded.coords[loaded.conn]
  k_np = _stiffness_numpy(coords, constitutive)
  k_nb = quad8_plane_stress_stiffness(coords, constitutive)
  np.testing.assert_allclose(k_nb, k_np, rtol=0.0, atol=_STIFFNESS_ATOL)
