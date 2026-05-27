"""Finite-element primitives (vectorized)."""

from pyfem.v3.fem.assembly import assemble_stiffness_coo
from pyfem.v3.fem.element import quad8_plane_stress_stiffness
from pyfem.v3.fem.quadrature import gauss_tensor_product_2d

__all__ = [
  "assemble_stiffness_coo",
  "gauss_tensor_product_2d",
  "quad8_plane_stress_stiffness",
]
