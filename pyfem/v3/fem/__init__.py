"""Finite-element primitives (vectorized)."""

from pyfem.v3.fem.assembly import (
  assemble_stiffness_coo,
  entries_per_elem,
  nodes_per_elem,
)
from pyfem.v3.fem.element import (
  hex8_stiffness,
  quad4_plane_stress_stiffness,
  quad8_plane_stress_stiffness,
  tet4_stiffness,
  tria3_plane_stress_stiffness,
)
from pyfem.v3.fem.quadrature import (
  gauss_tensor_product_2d,
  gauss_tensor_product_3d,
  gauss_tet4,
  gauss_tria3,
)

__all__ = [
  "assemble_stiffness_coo",
  "entries_per_elem",
  "gauss_tensor_product_2d",
  "gauss_tensor_product_3d",
  "gauss_tet4",
  "gauss_tria3",
  "hex8_stiffness",
  "nodes_per_elem",
  "quad4_plane_stress_stiffness",
  "quad8_plane_stress_stiffness",
  "tet4_stiffness",
  "tria3_plane_stress_stiffness",
]
