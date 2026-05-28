"""Reference-element shape functions."""

from __future__ import annotations

import numpy as np
from numba import njit

from pyfem.v3.types import F64


@njit(cache=True)
def serendipity_quad8(parent_coords: F64) -> tuple[F64, F64]:
  """
  Serendipity 8-node quadrilateral on [-1, 1]².

  Parameters
  ----------
  parent_coords
      ``(n_points, 2)`` with columns ``(xi, eta)``.
  """
  xi = parent_coords[:, 0]
  eta = parent_coords[:, 1]

  n = np.stack(
    (
      -0.25 * (1 - xi) * (1 - eta) * (1 + xi + eta),
      0.5 * (1 - xi) * (1 + xi) * (1 - eta),
      -0.25 * (1 + xi) * (1 - eta) * (1 - xi + eta),
      0.5 * (1 + xi) * (1 + eta) * (1 - eta),
      -0.25 * (1 + xi) * (1 + eta) * (1 - xi - eta),
      0.5 * (1 - xi) * (1 + xi) * (1 + eta),
      -0.25 * (1 - xi) * (1 + eta) * (1 + xi - eta),
      0.5 * (1 - xi) * (1 + eta) * (1 - eta),
    ),
    axis=1,
  )

  dxi = np.stack(
    (
      -0.25 * (-1 + eta) * (2 * xi + eta),
      xi * (-1 + eta),
      0.25 * (-1 + eta) * (-2 * xi + eta),
      -0.5 * (1 + eta) * (-1 + eta),
      0.25 * (1 + eta) * (2 * xi + eta),
      -xi * (1 + eta),
      -0.25 * (1 + eta) * (-2 * xi + eta),
      0.5 * (1 + eta) * (-1 + eta),
    ),
    axis=1,
  )

  deta = np.stack(
    (
      -0.25 * (-1 + xi) * (xi + 2 * eta),
      0.5 * (1 + xi) * (-1 + xi),
      0.25 * (1 + xi) * (-xi + 2 * eta),
      -eta * (1 + xi),
      0.25 * (1 + xi) * (xi + 2 * eta),
      -0.5 * (1 + xi) * (-1 + xi),
      -0.25 * (-1 + xi) * (-xi + 2 * eta),
      eta * (-1 + xi),
    ),
    axis=1,
  )

  dN = np.stack((dxi, deta), axis=2)
  return n, dN


@njit(cache=True)
def bilinear_quad4(parent_coords: F64) -> tuple[F64, F64]:
  """
  Bilinear 4-node quadrilateral on [-1, 1]².

  Parameters
  ----------
  parent_coords
      ``(n_points, 2)`` with columns ``(xi, eta)``.
  """
  xi = parent_coords[:, 0]
  eta = parent_coords[:, 1]

  n = np.stack(
    (
      0.25 * (1.0 - xi) * (1.0 - eta),
      0.25 * (1.0 + xi) * (1.0 - eta),
      0.25 * (1.0 + xi) * (1.0 + eta),
      0.25 * (1.0 - xi) * (1.0 + eta),
    ),
    axis=1,
  )

  dxi = np.stack(
    (
      -0.25 * (1.0 - eta),
      0.25 * (1.0 - eta),
      0.25 * (1.0 + eta),
      -0.25 * (1.0 + eta),
    ),
    axis=1,
  )

  deta = np.stack(
    (
      -0.25 * (1.0 - xi),
      -0.25 * (1.0 + xi),
      0.25 * (1.0 + xi),
      0.25 * (1.0 - xi),
    ),
    axis=1,
  )

  dN = np.stack((dxi, deta), axis=2)
  return n, dN


@njit(cache=True)
def linear_tria3(parent_coords: F64) -> tuple[F64, F64]:
  """
  Linear 3-node triangle in area coordinates (ξ, η).

  Parameters
  ----------
  parent_coords
      ``(n_points, 2)`` with columns ``(xi, eta)``.
  """
  xi = parent_coords[:, 0]
  eta = parent_coords[:, 1]

  n = np.stack(
    (
      1.0 - xi - eta,
      xi,
      eta,
    ),
    axis=1,
  )

  dxi = np.stack(
    (
      -1.0 * np.ones(xi.shape[0]),
      np.ones(xi.shape[0]),
      np.zeros(xi.shape[0]),
    ),
    axis=1,
  )

  deta = np.stack(
    (
      -1.0 * np.ones(xi.shape[0]),
      np.zeros(xi.shape[0]),
      np.ones(xi.shape[0]),
    ),
    axis=1,
  )

  dN = np.stack((dxi, deta), axis=2)
  return n, dN
