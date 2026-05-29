"""Total Lagrangian kinematics (2D plane stress)."""

from __future__ import annotations

import numpy as np
from numba import njit

from pyfem.v3.types import F64


@njit(cache=True)
def deformation_gradient_2d(dphi: F64, elstate: F64, n_nodes: int) -> F64:
  """Deformation gradient F from reference shape gradients and element state."""
  f = np.eye(2)
  for i in range(n_nodes):
    for j in range(2):
      for k in range(2):
        f[j, k] += dphi[i, k] * elstate[2 * i + j]
  return f


@njit(cache=True)
def green_lagrange_strain_voigt_2d(f: F64) -> F64:
  """Green–Lagrange strain in Voigt form ``[E11, E22, 2*E12]``."""
  ft_f = f.T @ f
  strain = np.empty(3, dtype=np.float64)
  strain[0] = 0.5 * (ft_f[0, 0] - 1.0)
  strain[1] = 0.5 * (ft_f[1, 1] - 1.0)
  strain[2] = ft_f[0, 1]
  return strain


@njit(cache=True)
def pk2_stress_voigt_2d(constitutive: F64, strain: F64) -> F64:
  """Second Piola–Kirchhoff stress ``sigma = C @ strain``."""
  return constitutive @ strain


@njit(cache=True)
def tl_b_matrix_2d(dphi: F64, f: F64, n_nodes: int, n_dof: int) -> F64:
  """Material part of the TL strain–displacement operator."""
  b = np.zeros((3, n_dof), dtype=np.float64)
  for i in range(n_nodes):
    dp0 = dphi[i, 0]
    dp1 = dphi[i, 1]
    ii = 2 * i
    b[0, ii] = dp0 * f[0, 0]
    b[0, ii + 1] = dp0 * f[1, 0]
    b[1, ii] = dp1 * f[0, 1]
    b[1, ii + 1] = dp1 * f[1, 1]
    b[2, ii] = dp1 * f[0, 0] + dp0 * f[0, 1]
    b[2, ii + 1] = dp0 * f[1, 1] + dp1 * f[1, 0]
  return b


@njit(cache=True)
def bnl_matrix_2d(dphi: F64, n_nodes: int, n_dof: int) -> F64:
  """Geometric nonlinearity operator B_NL (2D)."""
  bnl = np.zeros((4, n_dof), dtype=np.float64)
  for i in range(n_nodes):
    dp0 = dphi[i, 0]
    dp1 = dphi[i, 1]
    ii = 2 * i
    bnl[0, ii] = dp0
    bnl[1, ii] = dp1
    bnl[2, ii + 1] = dp0
    bnl[3, ii + 1] = dp1
  return bnl


@njit(cache=True)
def stress_to_matrix_2d(stress: F64) -> F64:
  """Map Voigt PK2 stress to the 4×4 geometric stiffness block."""
  t = np.zeros((4, 4), dtype=np.float64)
  t[0, 0] = stress[0]
  t[1, 1] = stress[1]
  t[0, 1] = stress[2]
  t[1, 0] = stress[2]
  t[2, 2] = stress[0]
  t[2, 3] = stress[2]
  t[3, 2] = stress[2]
  t[3, 3] = stress[1]
  return t
