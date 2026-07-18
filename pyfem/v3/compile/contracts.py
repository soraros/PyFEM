"""Frozen descriptor meaning for the first injected Q8 registry."""

from __future__ import annotations

from pyfem.v3.fem.kinematics import strain_displacement
from pyfem.v3.fem.quadrature import gauss_tensor_product_2d
from pyfem.v3.fem.shapes import serendipity_quad8
from pyfem.v3.materials.plane_stress import plane_stress_matrix
from pyfem.v3.model.registry import RegistryDescriptor, RegistryKey

Q8_TOPOLOGY_KEY: RegistryKey = ("topology", "serendipity-quad8")
Q8_QUADRATURE_KEY: RegistryKey = ("quadrature", "gauss-3x3")
Q8_FORMULATION_KEY: RegistryKey = (
  "formulation",
  "small-strain-continuum",
)
Q8_MATERIAL_KEY: RegistryKey = (
  "material",
  "plane-stress-linear-elastic",
)
Q8_REQUIRED_REGISTRY_KEYS = (
  Q8_TOPOLOGY_KEY,
  Q8_QUADRATURE_KEY,
  Q8_FORMULATION_KEY,
  Q8_MATERIAL_KEY,
)


def q8_descriptor_metadata(kind: str, name: str) -> dict[str, object]:
  """Return detached canonical metadata for one supported descriptor key."""
  key = (kind, name)
  if key == Q8_TOPOLOGY_KEY:
    return {
      "schema": "pyfem-v3-topology-descriptor-v1",
      "reference_topology": "quadrilateral",
      "parent_dimension": 2,
      "embedding_dimension": 2,
      "node_count": 8,
      "parent_coordinates": ["xi", "eta"],
      "local_node_parent_coordinates": [
        [-1.0, -1.0],
        [0.0, -1.0],
        [1.0, -1.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [-1.0, 1.0],
        [-1.0, 0.0],
      ],
      "shape_value_layout": ["point", "node"],
      "parent_gradient_layout": ["point", "node", "parent_coordinate"],
    }
  if key == Q8_QUADRATURE_KEY:
    return {
      "schema": "pyfem-v3-quadrature-descriptor-v1",
      "family": "tensor-gauss-legendre",
      "parent_coordinates": ["xi", "eta"],
      "orders": [3, 3],
      "point_count": 9,
      "binding_arguments": [3],
    }
  if key == Q8_FORMULATION_KEY:
    return {
      "schema": "pyfem-v3-formulation-descriptor-v1",
      "field_quantity": "displacement",
      "field_location": "node",
      "field_components": ["x", "y"],
      "dofs_per_node": 2,
      "kinematic_regime": "small-strain",
      "strain_measure": "infinitesimal",
      "strain_voigt_order": ["xx", "yy", "xy"],
      "shear_convention": "engineering",
      "formulation_history_width": 0,
      "tangent_contribution": "material",
      "tangent_symmetry": "symmetric",
    }
  if key == Q8_MATERIAL_KEY:
    return {
      "schema": "pyfem-v3-material-descriptor-v1",
      "law": "linear-elastic",
      "stress_state": "plane-stress",
      "parameter_names": ["youngs_modulus", "poisson_ratio"],
      "parameter_dtype": "float64",
      "stress_voigt_order": ["xx", "yy", "xy"],
      "strain_shear_convention": "engineering",
      "material_history_width": 0,
      "tangent_class": "constant-symmetric",
    }
  msg = "no Q8 descriptor metadata contract exists for that exact registry key"
  raise KeyError(msg)


def q8_reference_registry() -> dict[RegistryKey, RegistryDescriptor]:
  """Build a fresh injectable registry from existing direct math functions."""
  descriptors = (
    RegistryDescriptor(
      kind=Q8_TOPOLOGY_KEY[0],
      name=Q8_TOPOLOGY_KEY[1],
      version="1",
      implementation_id="pyfem-v3-serendipity-quad8-v1",
      metadata=q8_descriptor_metadata(*Q8_TOPOLOGY_KEY),
      binding=serendipity_quad8,
    ),
    RegistryDescriptor(
      kind=Q8_QUADRATURE_KEY[0],
      name=Q8_QUADRATURE_KEY[1],
      version="1",
      implementation_id="pyfem-v3-gauss-tensor-product-2d-order-3-v1",
      metadata=q8_descriptor_metadata(*Q8_QUADRATURE_KEY),
      binding=gauss_tensor_product_2d,
    ),
    RegistryDescriptor(
      kind=Q8_FORMULATION_KEY[0],
      name=Q8_FORMULATION_KEY[1],
      version="1",
      implementation_id="pyfem-v3-small-strain-engineering-shear-v1",
      metadata=q8_descriptor_metadata(*Q8_FORMULATION_KEY),
      binding=strain_displacement,
    ),
    RegistryDescriptor(
      kind=Q8_MATERIAL_KEY[0],
      name=Q8_MATERIAL_KEY[1],
      version="1",
      implementation_id="pyfem-v3-plane-stress-linear-elastic-v1",
      metadata=q8_descriptor_metadata(*Q8_MATERIAL_KEY),
      binding=plane_stress_matrix,
    ),
  )
  return {descriptor.key: descriptor for descriptor in descriptors}
