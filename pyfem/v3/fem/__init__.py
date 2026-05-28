"""Finite-element primitives (vectorized)."""

from pyfem.v3.fem.assembly import (
  assemble_stiffness_coo,
  entries_per_elem,
  nodes_per_elem,
)
from pyfem.v3.fem.element import (
  quad4_plane_stress_stiffness,
  quad8_plane_stress_stiffness,
  tria3_plane_stress_stiffness,
)
from pyfem.v3.fem.quadrature import gauss_tensor_product_2d, gauss_tria3

__all__ = [
  "assemble_stiffness_coo",
  "entries_per_elem",
  "gauss_tensor_product_2d",
  "gauss_tria3",
  "nodes_per_elem",
  "quad4_plane_stress_stiffness",
  "quad8_plane_stress_stiffness",
  "tria3_plane_stress_stiffness",
]
