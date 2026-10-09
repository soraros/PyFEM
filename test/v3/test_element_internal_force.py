# SPDX-License-Identifier: MIT

"""Element internal-force assembly: the shared residual/derivative contraction.

``pyfem.v3.fem.element.continuum_internal_force_batched`` is the residual
map's element assembly, ∫ Bᵀ σ dΩ over a continuum batch. The stateful
continuum operator routes both its internal-force residual and every
requested per-parameter residual derivative column through this one
expression, so a derivative channel is assembled by exactly the engine its
residual map uses — the operator-level FD pins of
test_v3_param_derivatives.py therefore cover the contraction at the channel
level, and this file pins the contraction itself against an explicit
per-point accumulation reference (the test_element_stiffness.py idiom).
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

pytest.importorskip("numba")

from pyfem.v3.fem.element import continuum_internal_force_batched
from pyfem.v3.fem.quadrature import gauss_tensor_product_2d
from pyfem.v3.fem.shapes import serendipity_quad8

_FORCE_ATOL = 1.0e-8


def _unit_patch_b_matrix() -> tuple[np.ndarray, np.ndarray]:
  """B matrix and integration measures for a perturbed unit-patch Q8 batch."""
  base = np.array(
    [
      [0.0, 0.0],
      [0.5, 0.0],
      [1.0, 0.0],
      [1.0, 0.5],
      [1.0, 1.0],
      [0.5, 1.0],
      [0.0, 1.0],
      [0.0, 0.5],
    ]
  )
  rng = np.random.default_rng(20261009)
  batch = base[None, :, :] + 0.02 * rng.normal(size=(3, 8, 2))
  parent_pts, parent_w = gauss_tensor_product_2d(3)
  _, dN_parent = serendipity_quad8(parent_pts)
  jacobian = np.einsum("...na,pnb->...pab", batch, dN_parent)
  grad_n = np.einsum("...pnb,...pbc->...pnc", dN_parent, np.linalg.inv(jacobian))
  det_j = np.linalg.det(jacobian)
  n_dof = 16
  n_points = parent_w.shape[0]
  b_matrix = np.zeros((3, n_points, 3, n_dof))
  b_matrix[:, :, 0, 0::2] = grad_n[:, :, :, 0]
  b_matrix[:, :, 1, 1::2] = grad_n[:, :, :, 1]
  b_matrix[:, :, 2, 0::2] = grad_n[:, :, :, 1]
  b_matrix[:, :, 2, 1::2] = grad_n[:, :, :, 0]
  weights = parent_w[None, :] * np.abs(det_j)
  return b_matrix, weights


def _internal_force_reference(
  b_matrix: np.ndarray, weights: np.ndarray, stresses: np.ndarray
) -> np.ndarray:
  """Per-point accumulation reference: f_e += w_p * B_pᵀ σ_p."""
  n_elems, n_points, _, n_dof = b_matrix.shape
  force = np.zeros((n_elems, n_dof))
  for point in range(n_points):
    for elem in range(n_elems):
      force[elem] += weights[elem, point] * (
        b_matrix[elem, point].T @ stresses[elem, point]
      )
  return force


def test_continuum_internal_force_batched_matches_per_point_reference() -> None:
  b_matrix, weights = _unit_patch_b_matrix()
  rng = np.random.default_rng(7)
  stresses = rng.normal(size=(3, 9, 3))
  actual = continuum_internal_force_batched(b_matrix, weights, stresses)
  expected = _internal_force_reference(b_matrix, weights, stresses)
  # The two contraction orders agree at ulp level (~1e-15 observed here);
  # the bound mirrors the element-stiffness pin discipline.
  np.testing.assert_allclose(actual, expected, rtol=0.0, atol=_FORCE_ATOL)


def test_continuum_internal_force_batched_is_linear_in_the_stress() -> None:
  """The derivative-channel identity: assembly of dσ is d of the assembly."""
  b_matrix, weights = _unit_patch_b_matrix()
  rng = np.random.default_rng(11)
  base = rng.normal(size=(3, 9, 3))
  column = rng.normal(size=(3, 9, 3))
  step = 1.0e-7
  plus = continuum_internal_force_batched(b_matrix, weights, base + step * column)
  minus = continuum_internal_force_batched(b_matrix, weights, base - step * column)
  fd = (plus - minus) / (2.0 * step)
  exact = continuum_internal_force_batched(b_matrix, weights, column)
  # Linearity makes the FD of the assembled residual recover the assembled
  # column to rounding — the property the channel's FD verification relies on.
  np.testing.assert_allclose(fd, exact, rtol=0.0, atol=1.0e-8)
