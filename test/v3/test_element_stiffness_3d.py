# SPDX-License-Identifier: MIT

"""3D element stiffness: Numba batched vs NumPy reference."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

pytest.importorskip("numba")

from pyfem.v3 import load_problem
from pyfem.v3.fem.element import hex8_stiffness, tet4_stiffness
from pyfem.v3.fem.quadrature import gauss_tensor_product_3d, gauss_tet4
from pyfem.v3.fem.shapes import linear_tet4, trilinear_hex8
from pyfem.v3.materials.isotropic import isotropic_matrix

ROOT = Path(__file__).resolve().parents[2]
_STIFFNESS_ATOL = 1e-8


def _physical_gradients_numpy_3d(
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


def _strain_displacement_numpy_3d(grad_n: np.ndarray) -> np.ndarray:
  *lead, n_nodes, _ = grad_n.shape
  n_dof = 3 * n_nodes
  b = np.zeros((*lead, 6, n_dof), dtype=np.float64)
  b[..., 0, 0::3] = grad_n[..., :, 0]
  b[..., 1, 1::3] = grad_n[..., :, 1]
  b[..., 2, 2::3] = grad_n[..., :, 2]
  b[..., 3, 1::3] = grad_n[..., :, 2]
  b[..., 3, 2::3] = grad_n[..., :, 1]
  b[..., 4, 0::3] = grad_n[..., :, 2]
  b[..., 4, 2::3] = grad_n[..., :, 0]
  b[..., 5, 0::3] = grad_n[..., :, 1]
  b[..., 5, 1::3] = grad_n[..., :, 0]
  return b


def _stiffness_numpy_3d(
  nodal_coords: np.ndarray,
  constitutive: np.ndarray,
  *,
  parent_pts: np.ndarray,
  parent_w: np.ndarray,
  shape_fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
  single = nodal_coords.ndim == 2
  if single:
    nodal_coords = nodal_coords[None, ...]

  _, dN_parent = shape_fn(parent_pts)
  _jacobian, grad_n = _physical_gradients_numpy_3d(nodal_coords, dN_parent)
  det_j = np.linalg.det(_jacobian)
  b = _strain_displacement_numpy_3d(grad_n)

  measure = parent_w * np.abs(det_j)
  bt_c_b = np.einsum("...paj,ac,...pck->...pjk", b, constitutive, b)
  stiffness = np.einsum("...p,...pjk->...jk", measure, bt_c_b)
  return stiffness[0] if single else stiffness


def _stiffness_numpy_hex8(
  nodal_coords: np.ndarray,
  constitutive: np.ndarray,
) -> np.ndarray:
  parent_pts, parent_w = gauss_tensor_product_3d(2)
  return _stiffness_numpy_3d(
    nodal_coords,
    constitutive,
    parent_pts=parent_pts,
    parent_w=parent_w,
    shape_fn=trilinear_hex8,
  )


def _stiffness_numpy_tet4(
  nodal_coords: np.ndarray,
  constitutive: np.ndarray,
) -> np.ndarray:
  parent_pts, parent_w = gauss_tet4(1)
  return _stiffness_numpy_3d(
    nodal_coords,
    constitutive,
    parent_pts=parent_pts,
    parent_w=parent_w,
    shape_fn=linear_tet4,
  )


def _make_coords_hex8(n_elems: int, rng: np.random.Generator) -> np.ndarray:
  base = np.array(
    [
      [0.0, 0.0, 0.0],
      [1.0, 0.0, 0.0],
      [1.0, 1.0, 0.0],
      [0.0, 1.0, 0.0],
      [0.0, 0.0, 1.0],
      [1.0, 0.0, 1.0],
      [1.0, 1.0, 1.0],
      [0.0, 1.0, 1.0],
    ],
    dtype=np.float64,
  )
  out = np.empty((n_elems, 8, 3), dtype=np.float64)
  for e in range(n_elems):
    out[e] = base + 0.02 * rng.standard_normal((8, 3))
  return out


def _make_coords_tet4(n_elems: int, rng: np.random.Generator) -> np.ndarray:
  base = np.array(
    [
      [0.0, 0.0, 0.0],
      [1.0, 0.0, 0.0],
      [0.0, 1.0, 0.0],
      [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
  )
  out = np.empty((n_elems, 4, 3), dtype=np.float64)
  for e in range(n_elems):
    out[e] = base + 0.02 * rng.standard_normal((4, 3))
  return out


@pytest.fixture(scope="module")
def constitutive_3d() -> np.ndarray:
  return isotropic_matrix(1.0e6, 0.25)


def test_hex8_stiffness_single_element_matches_numpy(
  constitutive_3d: np.ndarray,
) -> None:
  rng = np.random.default_rng(0)
  coords = _make_coords_hex8(1, rng)[0]
  k_np = _stiffness_numpy_hex8(coords, constitutive_3d)
  k_nb = hex8_stiffness(coords, constitutive_3d)
  np.testing.assert_allclose(k_nb, k_np, rtol=0.0, atol=_STIFFNESS_ATOL)


def test_hex8_stiffness_patch_test8_3d_batch_matches_numpy(
  constitutive_3d: np.ndarray,
) -> None:
  loaded = load_problem(ROOT / "skims/patch_test8_3d/problem.toml").problem
  coords = loaded.coords[loaded.conn]
  k_np = _stiffness_numpy_hex8(coords, constitutive_3d)
  k_nb = hex8_stiffness(coords, constitutive_3d)
  np.testing.assert_allclose(k_nb, k_np, rtol=0.0, atol=_STIFFNESS_ATOL)


def test_tet4_stiffness_single_element_matches_numpy(
  constitutive_3d: np.ndarray,
) -> None:
  rng = np.random.default_rng(1)
  coords = _make_coords_tet4(1, rng)[0]
  k_np = _stiffness_numpy_tet4(coords, constitutive_3d)
  k_nb = tet4_stiffness(coords, constitutive_3d)
  np.testing.assert_allclose(k_nb, k_np, rtol=0.0, atol=_STIFFNESS_ATOL)
