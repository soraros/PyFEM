"""Element-level operators."""

from __future__ import annotations

from collections.abc import Callable
from functools import cache

import numpy as np
from numba import njit, prange

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

# The batched stiffness kernels below are one fused pass per element:
# Jacobian, closed-form small inverse, physical gradients, and the
# B.T @ C @ B contraction accumulate in registers with no per-point
# temporaries. This replaces the earlier form (``np.linalg.det/inv`` and
# small ``@`` products per Gauss point), which the M5/D2 compute model
# measured at ~10x above the fused form (28 vs 2.0 us/elem 1T on Q8).
# Quadrature and shape data are evaluated by the Python dispatch layer and
# arrive as arguments, so the kernels reference only NumPy and same-file
# symbols and cache=True is cache-friendly (NUMBA_CACHING.md §5).
#
# Numerical contract: the contraction reads every constitutive entry, so
# nonsymmetric tangents keep their meaning, and each element's accumulation
# is independent and reduction-free, so results are bitwise identical at any
# thread count. The fused operation order differs from both the predecessor
# kernels and the test-suite NumPy einsum reference at ulp level (~1e-14
# relative); test/v3/test_element_stiffness*.py pins the agreement at
# atol 1e-8.


@njit(cache=True, parallel=True)
def _btcb_2d_batched(
  parent_w: F64,
  parent_gradients: F64,
  nodal_coords: F64,
  constitutive: F64,
) -> F64:
  """Integrate ``B.T @ C @ B`` over a 2D continuum batch (3-Voigt stress).

  ``parent_gradients`` is ``(n_points, n_nodes, 2)`` ∂N/∂(xi, eta);
  ``nodal_coords`` is ``(n_elems, n_nodes, 2)``. Works for any 2D node
  count (Tria3, Quad4, Quad8).
  """
  n_elems = nodal_coords.shape[0]
  n_nodes = nodal_coords.shape[1]
  n_dof = 2 * n_nodes
  n_gp = parent_w.shape[0]
  c00 = constitutive[0, 0]
  c01 = constitutive[0, 1]
  c02 = constitutive[0, 2]
  c10 = constitutive[1, 0]
  c11 = constitutive[1, 1]
  c12 = constitutive[1, 2]
  c20 = constitutive[2, 0]
  c21 = constitutive[2, 1]
  c22 = constitutive[2, 2]
  stiffness = np.zeros((n_elems, n_dof, n_dof), dtype=np.float64)
  for e in prange(n_elems):
    scratch = np.empty((n_nodes, 8), dtype=np.float64)
    for p in range(n_gp):
      j00 = 0.0
      j01 = 0.0
      j10 = 0.0
      j11 = 0.0
      for n in range(n_nodes):
        x = nodal_coords[e, n, 0]
        y = nodal_coords[e, n, 1]
        dxi = parent_gradients[p, n, 0]
        deta = parent_gradients[p, n, 1]
        j00 += x * dxi
        j01 += x * deta
        j10 += y * dxi
        j11 += y * deta
      det_j = j00 * j11 - j01 * j10
      weight = parent_w[p] * abs(det_j)
      inv00 = j11 / det_j
      inv01 = -j01 / det_j
      inv10 = -j10 / det_j
      inv11 = j00 / det_j
      # Per-node physical gradients and weighted P = weight * B.T @ C rows.
      for n in range(n_nodes):
        dxi = parent_gradients[p, n, 0]
        deta = parent_gradients[p, n, 1]
        gx = dxi * inv00 + deta * inv10
        gy = dxi * inv01 + deta * inv11
        wx = gx * weight
        wy = gy * weight
        scratch[n, 0] = gx
        scratch[n, 1] = gy
        scratch[n, 2] = wx * c00 + wy * c20
        scratch[n, 3] = wx * c01 + wy * c21
        scratch[n, 4] = wx * c02 + wy * c22
        scratch[n, 5] = wy * c10 + wx * c20
        scratch[n, 6] = wy * c11 + wx * c21
        scratch[n, 7] = wy * c12 + wx * c22
      for i in range(n_nodes):
        p00 = scratch[i, 2]
        p01 = scratch[i, 3]
        p02 = scratch[i, 4]
        p10 = scratch[i, 5]
        p11 = scratch[i, 6]
        p12 = scratch[i, 7]
        for j in range(n_nodes):
          gxj = scratch[j, 0]
          gyj = scratch[j, 1]
          stiffness[e, 2 * i, 2 * j] += p00 * gxj + p02 * gyj
          stiffness[e, 2 * i, 2 * j + 1] += p01 * gyj + p02 * gxj
          stiffness[e, 2 * i + 1, 2 * j] += p10 * gxj + p12 * gyj
          stiffness[e, 2 * i + 1, 2 * j + 1] += p11 * gyj + p12 * gxj
  return stiffness


@njit(cache=True, parallel=True)
def _btcb_3d_batched(
  parent_w: F64,
  parent_gradients: F64,
  nodal_coords: F64,
  constitutive: F64,
) -> F64:
  """Integrate ``B.T @ C @ B`` over a 3D continuum batch (6-Voigt stress).

  ``parent_gradients`` is ``(n_points, n_nodes, 3)`` ∂N/∂(xi, eta, zeta);
  ``nodal_coords`` is ``(n_elems, n_nodes, 3)``. Works for any 3D node
  count (Tet4, Hex8).
  """
  n_elems = nodal_coords.shape[0]
  n_nodes = nodal_coords.shape[1]
  n_dof = 3 * n_nodes
  n_gp = parent_w.shape[0]
  stiffness = np.zeros((n_elems, n_dof, n_dof), dtype=np.float64)
  for e in prange(n_elems):
    scratch = np.empty((n_nodes, 21), dtype=np.float64)
    for p in range(n_gp):
      j00 = 0.0
      j01 = 0.0
      j02 = 0.0
      j10 = 0.0
      j11 = 0.0
      j12 = 0.0
      j20 = 0.0
      j21 = 0.0
      j22 = 0.0
      for n in range(n_nodes):
        x = nodal_coords[e, n, 0]
        y = nodal_coords[e, n, 1]
        z = nodal_coords[e, n, 2]
        dxi = parent_gradients[p, n, 0]
        deta = parent_gradients[p, n, 1]
        dzeta = parent_gradients[p, n, 2]
        j00 += x * dxi
        j01 += x * deta
        j02 += x * dzeta
        j10 += y * dxi
        j11 += y * deta
        j12 += y * dzeta
        j20 += z * dxi
        j21 += z * deta
        j22 += z * dzeta
      cof0 = j11 * j22 - j12 * j21
      cof1 = j12 * j20 - j10 * j22
      cof2 = j10 * j21 - j11 * j20
      det_j = j00 * cof0 + j01 * cof1 + j02 * cof2
      weight = parent_w[p] * abs(det_j)
      inv00 = cof0 / det_j
      inv01 = (j02 * j21 - j01 * j22) / det_j
      inv02 = (j01 * j12 - j02 * j11) / det_j
      inv10 = cof1 / det_j
      inv11 = (j00 * j22 - j02 * j20) / det_j
      inv12 = (j02 * j10 - j00 * j12) / det_j
      inv20 = cof2 / det_j
      inv21 = (j01 * j20 - j00 * j21) / det_j
      inv22 = (j00 * j11 - j01 * j10) / det_j
      # Per-node physical gradients and weighted P = weight * B.T @ C rows.
      for n in range(n_nodes):
        dxi = parent_gradients[p, n, 0]
        deta = parent_gradients[p, n, 1]
        dzeta = parent_gradients[p, n, 2]
        gx = dxi * inv00 + deta * inv10 + dzeta * inv20
        gy = dxi * inv01 + deta * inv11 + dzeta * inv21
        gz = dxi * inv02 + deta * inv12 + dzeta * inv22
        wx = gx * weight
        wy = gy * weight
        wz = gz * weight
        scratch[n, 0] = gx
        scratch[n, 1] = gy
        scratch[n, 2] = gz
        for q in range(6):
          scratch[n, 3 + q] = (
            wx * constitutive[0, q] + wz * constitutive[4, q] + wy * constitutive[5, q]
          )
          scratch[n, 9 + q] = (
            wy * constitutive[1, q] + wz * constitutive[3, q] + wx * constitutive[5, q]
          )
          scratch[n, 15 + q] = (
            wz * constitutive[2, q] + wy * constitutive[3, q] + wx * constitutive[4, q]
          )
      for i in range(n_nodes):
        for j in range(n_nodes):
          gxj = scratch[j, 0]
          gyj = scratch[j, 1]
          gzj = scratch[j, 2]
          for a in range(3):
            pa0 = scratch[i, 3 + 6 * a]
            pa1 = scratch[i, 3 + 6 * a + 1]
            pa2 = scratch[i, 3 + 6 * a + 2]
            pa3 = scratch[i, 3 + 6 * a + 3]
            pa4 = scratch[i, 3 + 6 * a + 4]
            pa5 = scratch[i, 3 + 6 * a + 5]
            stiffness[e, 3 * i + a, 3 * j] += pa0 * gxj + pa4 * gzj + pa5 * gyj
            stiffness[e, 3 * i + a, 3 * j + 1] += pa1 * gyj + pa3 * gzj + pa5 * gxj
            stiffness[e, 3 * i + a, 3 * j + 2] += pa2 * gzj + pa3 * gyj + pa4 * gxj
  return stiffness


# The rule helpers below return process-wide constant arrays (quadrature
# weights and parent gradients). They are memoized because the quadrature
# helpers' literal overloads re-resolve on every Python-side call (~14 ms
# for gauss_tensor_product_2d; inside an njit caller they fold to
# compile-time constants). The returned arrays are shared — callers must
# not mutate them.


@cache
def _quad8_rule() -> tuple[F64, F64]:
  parent_pts, parent_w = gauss_tensor_product_2d(3)
  _, parent_gradients = serendipity_quad8(parent_pts)
  return parent_w, parent_gradients


@cache
def _quad4_rule() -> tuple[F64, F64]:
  parent_pts, parent_w = gauss_tensor_product_2d(2)
  _, parent_gradients = bilinear_quad4(parent_pts)
  return parent_w, parent_gradients


@cache
def _tria3_rule() -> tuple[F64, F64]:
  parent_pts, parent_w = gauss_tria3(1)
  _, parent_gradients = linear_tria3(parent_pts)
  return parent_w, parent_gradients


@cache
def _hex8_rule() -> tuple[F64, F64]:
  parent_pts, parent_w = gauss_tensor_product_3d(2)
  _, parent_gradients = trilinear_hex8(parent_pts)
  return parent_w, parent_gradients


@cache
def _tet4_rule() -> tuple[F64, F64]:
  parent_pts, parent_w = gauss_tet4(1)
  _, parent_gradients = linear_tet4(parent_pts)
  return parent_w, parent_gradients


def _wrap_batched(
  batched_fn: Callable[[F64, F64, F64, F64], F64],
  rule: Callable[[], tuple[F64, F64]],
  nodal_coords: F64,
  constitutive: F64,
) -> F64:
  parent_w, parent_gradients = rule()
  if nodal_coords.ndim == 2:
    single = batched_fn(
      parent_w, parent_gradients, nodal_coords[None, ...], constitutive
    )
    return single[0]
  return batched_fn(parent_w, parent_gradients, nodal_coords, constitutive)


def continuum_stiffness_batched(
  nodal_coords: F64,
  constitutive: F64,
) -> F64:
  """Dispatch batched small-strain continuum stiffness by mesh rank and nodes/elem."""
  spatial_dim = int(nodal_coords.shape[-1])
  n_nodes = int(nodal_coords.shape[-2])
  if spatial_dim == 2:
    if n_nodes == 8:
      return _wrap_batched(_btcb_2d_batched, _quad8_rule, nodal_coords, constitutive)
    if n_nodes == 4:
      return _wrap_batched(_btcb_2d_batched, _quad4_rule, nodal_coords, constitutive)
    if n_nodes == 3:
      return _wrap_batched(_btcb_2d_batched, _tria3_rule, nodal_coords, constitutive)
  elif spatial_dim == 3:
    if n_nodes == 8:
      return _wrap_batched(_btcb_3d_batched, _hex8_rule, nodal_coords, constitutive)
    if n_nodes == 4:
      return _wrap_batched(_btcb_3d_batched, _tet4_rule, nodal_coords, constitutive)
  msg = f"Unsupported element with {n_nodes} nodes per element in {spatial_dim}D"
  raise ValueError(msg)


def quad8_plane_stress_stiffness(nodal_coords: F64, constitutive: F64) -> F64:
  r"""Element stiffness K_e = ∫_Ω Bᵀ C B dΩ (plane stress, Q8)."""
  return _wrap_batched(_btcb_2d_batched, _quad8_rule, nodal_coords, constitutive)


def quad4_plane_stress_stiffness(nodal_coords: F64, constitutive: F64) -> F64:
  r"""Element stiffness K_e = ∫_Ω Bᵀ C B dΩ (plane stress, Quad4)."""
  return _wrap_batched(_btcb_2d_batched, _quad4_rule, nodal_coords, constitutive)


def tria3_plane_stress_stiffness(nodal_coords: F64, constitutive: F64) -> F64:
  r"""Element stiffness K_e = ∫_Ω Bᵀ C B dΩ (plane stress, Tria3)."""
  return _wrap_batched(_btcb_2d_batched, _tria3_rule, nodal_coords, constitutive)


def hex8_stiffness(nodal_coords: F64, constitutive: F64) -> F64:
  r"""Element stiffness K_e = ∫_Ω Bᵀ C B dΩ (3D, Hex8)."""
  return _wrap_batched(_btcb_3d_batched, _hex8_rule, nodal_coords, constitutive)


def tet4_stiffness(nodal_coords: F64, constitutive: F64) -> F64:
  r"""Element stiffness K_e = ∫_Ω Bᵀ C B dΩ (3D, Tet4)."""
  return _wrap_batched(_btcb_3d_batched, _tet4_rule, nodal_coords, constitutive)


# Backward-compatible alias for benchmarks and internal callers.
_stiffness_from_coords_batched = quad8_plane_stress_stiffness
