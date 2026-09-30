"""Frozen descriptor meaning for the first injected Q8 registry.

The v1 descriptor set also covers the small-strain continuum breadth wave:
the ``linear-tria3``, ``bilinear-quad4``, and ``trilinear-hex8`` topologies
with their pinned quadrature rules, the three-dimensional small-strain
kinematics, and the ``plane-strain-linear-elastic`` and
``isotropic-linear-elastic`` (3D) linear laws — all mirroring the landed
plane-stress v1 idiom. :func:`continuum_reference_registry` assembles the
full v1 continuum set; snapshots capture exactly the keys a model selects,
so the superset leaves existing pinned snapshots byte-identical.

Also hosts the permanent stateful-material descriptor ABI (schema
``pyfem-v3-material-descriptor-v2``): typed per-entity state slots declared as
canonical metadata, their compile-time resolution into ``OperatorStateLayout``
emissions, and the researcher-facing binding protocol kernels and initial-state
bindings plug into. Descriptors carrying the v1 material schema remain
byte-identical; the v2 schema is a strict extension consumed only by the
generic stateful compiler path.

The v2 schema carries one optional extension field, ``signal_ports``: typed
program-signal port declarations (``port_id``, ``signal_id``,
``derivative_coordinate_ids``) resolved into ``SignalPortBinding`` emissions at
compile time. Descriptors without the field declare no ports and keep
byte-identical metadata, manifests, and kernel calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from pyfem.v3.fem.kinematics import strain_displacement, strain_displacement_3d
from pyfem.v3.fem.quadrature import (
  gauss_tensor_product_2d,
  gauss_tensor_product_3d,
  gauss_tria3,
)
from pyfem.v3.fem.shapes import (
  bilinear_quad4,
  linear_tria3,
  serendipity_quad8,
  trilinear_hex8,
)
from pyfem.v3.materials.isotropic import isotropic_matrix
from pyfem.v3.materials.plane_strain import plane_strain_matrix
from pyfem.v3.materials.plane_stress import plane_stress_matrix
from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.operator import (
  EvaluationStatus,
  OperatorStateLayout,
  OperatorStateSlot,
  SemanticId,
  StateLifetime,
)
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


TRIA3_TOPOLOGY_KEY: RegistryKey = ("topology", "linear-tria3")
QUAD4_TOPOLOGY_KEY: RegistryKey = ("topology", "bilinear-quad4")
HEX8_TOPOLOGY_KEY: RegistryKey = ("topology", "trilinear-hex8")
TRIA3_QUADRATURE_KEY: RegistryKey = ("quadrature", "gauss-tria3-1")
QUAD4_QUADRATURE_KEY: RegistryKey = ("quadrature", "gauss-2x2")
HEX8_QUADRATURE_KEY: RegistryKey = ("quadrature", "gauss-2x2x2")
CONTINUUM_3D_FORMULATION_KEY: RegistryKey = (
  "formulation",
  "small-strain-continuum-3d",
)
PLANE_STRAIN_MATERIAL_KEY: RegistryKey = (
  "material",
  "plane-strain-linear-elastic",
)
ISOTROPIC_MATERIAL_KEY: RegistryKey = (
  "material",
  "isotropic-linear-elastic",
)


def breadth_descriptor_metadata(kind: str, name: str) -> dict[str, object]:
  """Return detached canonical metadata for one continuum breadth descriptor."""
  key = (kind, name)
  if key == TRIA3_TOPOLOGY_KEY:
    return {
      "schema": "pyfem-v3-topology-descriptor-v1",
      "reference_topology": "triangle",
      "parent_dimension": 2,
      "embedding_dimension": 2,
      "node_count": 3,
      "parent_coordinates": ["xi", "eta"],
      "local_node_parent_coordinates": [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
      ],
      "shape_value_layout": ["point", "node"],
      "parent_gradient_layout": ["point", "node", "parent_coordinate"],
    }
  if key == QUAD4_TOPOLOGY_KEY:
    return {
      "schema": "pyfem-v3-topology-descriptor-v1",
      "reference_topology": "quadrilateral",
      "parent_dimension": 2,
      "embedding_dimension": 2,
      "node_count": 4,
      "parent_coordinates": ["xi", "eta"],
      "local_node_parent_coordinates": [
        [-1.0, -1.0],
        [1.0, -1.0],
        [1.0, 1.0],
        [-1.0, 1.0],
      ],
      "shape_value_layout": ["point", "node"],
      "parent_gradient_layout": ["point", "node", "parent_coordinate"],
    }
  if key == HEX8_TOPOLOGY_KEY:
    return {
      "schema": "pyfem-v3-topology-descriptor-v1",
      "reference_topology": "hexahedron",
      "parent_dimension": 3,
      "embedding_dimension": 3,
      "node_count": 8,
      "parent_coordinates": ["xi", "eta", "zeta"],
      "local_node_parent_coordinates": [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
      ],
      "shape_value_layout": ["point", "node"],
      "parent_gradient_layout": ["point", "node", "parent_coordinate"],
    }
  if key == TRIA3_QUADRATURE_KEY:
    return {
      "schema": "pyfem-v3-quadrature-descriptor-v1",
      "family": "gauss-triangle",
      "parent_coordinates": ["xi", "eta"],
      "orders": [1],
      "point_count": 1,
      "binding_arguments": [1],
    }
  if key == QUAD4_QUADRATURE_KEY:
    return {
      "schema": "pyfem-v3-quadrature-descriptor-v1",
      "family": "tensor-gauss-legendre",
      "parent_coordinates": ["xi", "eta"],
      "orders": [2, 2],
      "point_count": 4,
      "binding_arguments": [2],
    }
  if key == HEX8_QUADRATURE_KEY:
    return {
      "schema": "pyfem-v3-quadrature-descriptor-v1",
      "family": "tensor-gauss-legendre",
      "parent_coordinates": ["xi", "eta", "zeta"],
      "orders": [2, 2, 2],
      "point_count": 8,
      "binding_arguments": [2],
    }
  if key == CONTINUUM_3D_FORMULATION_KEY:
    return {
      "schema": "pyfem-v3-formulation-descriptor-v1",
      "field_quantity": "displacement",
      "field_location": "node",
      "field_components": ["x", "y", "z"],
      "dofs_per_node": 3,
      "kinematic_regime": "small-strain",
      "strain_measure": "infinitesimal",
      "strain_voigt_order": ["xx", "yy", "zz", "yz", "zx", "xy"],
      "shear_convention": "engineering",
      "formulation_history_width": 0,
      "tangent_contribution": "material",
      "tangent_symmetry": "symmetric",
    }
  if key == PLANE_STRAIN_MATERIAL_KEY:
    return {
      "schema": "pyfem-v3-material-descriptor-v1",
      "law": "linear-elastic",
      "stress_state": "plane-strain",
      "parameter_names": ["youngs_modulus", "poisson_ratio"],
      "parameter_dtype": "float64",
      "stress_voigt_order": ["xx", "yy", "xy"],
      "strain_shear_convention": "engineering",
      "material_history_width": 0,
      "tangent_class": "constant-symmetric",
    }
  if key == ISOTROPIC_MATERIAL_KEY:
    return {
      "schema": "pyfem-v3-material-descriptor-v1",
      "law": "linear-elastic",
      "stress_state": "three-dimensional",
      "parameter_names": ["youngs_modulus", "poisson_ratio"],
      "parameter_dtype": "float64",
      "stress_voigt_order": ["xx", "yy", "zz", "yz", "zx", "xy"],
      "strain_shear_convention": "engineering",
      "material_history_width": 0,
      "tangent_class": "constant-symmetric",
    }
  msg = "no continuum breadth descriptor metadata exists for that exact registry key"
  raise KeyError(msg)


def continuum_reference_registry() -> dict[RegistryKey, RegistryDescriptor]:
  """Build the full v1 small-strain continuum registry (all landed geometries).

  The superset carries every v1 continuum descriptor: the four topologies
  (serendipity-quad8, linear-tria3, bilinear-quad4, trilinear-hex8) with their
  pinned quadrature rules, the two-dimensional and three-dimensional
  small-strain kinematics, and the three linear laws (plane-stress,
  plane-strain, 3D isotropic). Snapshots capture exactly the keys a model
  selects, so injecting this registry for a pinned Q8 model captures the same
  descriptor bytes as the Q8-only registry.
  """
  bindings = {
    Q8_TOPOLOGY_KEY: ("pyfem-v3-serendipity-quad8-v1", serendipity_quad8),
    Q8_QUADRATURE_KEY: (
      "pyfem-v3-gauss-tensor-product-2d-order-3-v1",
      gauss_tensor_product_2d,
    ),
    Q8_FORMULATION_KEY: (
      "pyfem-v3-small-strain-engineering-shear-v1",
      strain_displacement,
    ),
    Q8_MATERIAL_KEY: (
      "pyfem-v3-plane-stress-linear-elastic-v1",
      plane_stress_matrix,
    ),
    TRIA3_TOPOLOGY_KEY: ("pyfem-v3-linear-tria3-v1", linear_tria3),
    TRIA3_QUADRATURE_KEY: ("pyfem-v3-gauss-tria3-order-1-v1", gauss_tria3),
    QUAD4_TOPOLOGY_KEY: ("pyfem-v3-bilinear-quad4-v1", bilinear_quad4),
    QUAD4_QUADRATURE_KEY: (
      "pyfem-v3-gauss-tensor-product-2d-order-2-v1",
      gauss_tensor_product_2d,
    ),
    HEX8_TOPOLOGY_KEY: ("pyfem-v3-trilinear-hex8-v1", trilinear_hex8),
    HEX8_QUADRATURE_KEY: (
      "pyfem-v3-gauss-tensor-product-3d-order-2-v1",
      gauss_tensor_product_3d,
    ),
    CONTINUUM_3D_FORMULATION_KEY: (
      "pyfem-v3-small-strain-engineering-shear-3d-v1",
      strain_displacement_3d,
    ),
    PLANE_STRAIN_MATERIAL_KEY: (
      "pyfem-v3-plane-strain-linear-elastic-v1",
      plane_strain_matrix,
    ),
    ISOTROPIC_MATERIAL_KEY: (
      "pyfem-v3-isotropic-linear-elastic-v1",
      isotropic_matrix,
    ),
  }
  descriptors = tuple(
    RegistryDescriptor(
      kind=key[0],
      name=key[1],
      version="1",
      implementation_id=implementation_id,
      metadata=(
        q8_descriptor_metadata(*key)
        if key in Q8_REQUIRED_REGISTRY_KEYS
        else breadth_descriptor_metadata(*key)
      ),
      binding=binding,
    )
    for key, (implementation_id, binding) in bindings.items()
  )
  return {descriptor.key: descriptor for descriptor in descriptors}


STATEFUL_MATERIAL_DESCRIPTOR_SCHEMA = "pyfem-v3-material-descriptor-v2"

_STATE_SLOT_DTYPES = ("float64",)
_STATE_SLOT_LIFETIMES = ("accepted-trial",)
_STATE_SLOT_ANNOTATIONS = ("envelope-max", "monotone-nondecreasing")
_STATE_SLOT_KEYS = frozenset(("name", "width", "dtype", "lifetime", "annotation"))
_STATE_SLOT_REQUIRED_KEYS = frozenset(("name", "width", "dtype", "lifetime"))
_STATEFUL_METADATA_KEYS = frozenset(
  (
    "schema",
    "law",
    "stress_state",
    "parameter_names",
    "parameter_dtype",
    "stress_voigt_order",
    "strain_shear_convention",
    "internal_voigt_order",
    "tangent_class",
    "state_schema",
    "state_slots",
  )
)
_STATEFUL_OPTIONAL_METADATA_KEYS = frozenset(("signal_ports",))
_STATEFUL_TANGENT_CLASS_FLAGS: dict[str, tuple[bool, bool]] = {
  "algorithmic-symmetric": (False, True),
  "algorithmic-nonsymmetric": (False, False),
  "secant-branch": (False, False),
}
_LIFETIME_ENUMS = {"accepted-trial": StateLifetime.ACCEPTED_TRIAL}
_FLOAT64_STR = np.dtype(np.float64).str
_SIGNAL_PORT_KEYS = frozenset(("port_id", "signal_id", "derivative_coordinate_ids"))
_SIGNAL_PORT_REQUIRED_KEYS = frozenset(("port_id", "signal_id"))


def _new[ValueT](cls: type[ValueT], /, **fields: object) -> ValueT:
  value = object.__new__(cls)
  for name, field in fields.items():
    object.__setattr__(value, name, field)
  return value


def _nonempty_exact_str(value: object, label: str) -> str:
  if type(value) is not str or not value:
    msg = f"stateful material {label} must be a non-empty exact string"
    raise TypeError(msg)
  return value


def _exact_str_list(value: object, label: str) -> list[str]:
  if (
    type(value) is not list
    or not value
    or any(type(item) is not str or not item for item in value)
  ):
    msg = f"stateful material {label} must be a non-empty list of exact strings"
    raise TypeError(msg)
  return value


@dataclass(frozen=True, slots=True)
class MaterialStateSlot:
  """One resolved descriptor state slot of the permanent material state ABI."""

  name: str
  width: int
  dtype: str
  lifetime: str
  annotation: str | None

  def __post_init__(self) -> None:
    _nonempty_exact_str(self.name, "state slot name")
    if type(self.width) is not int or self.width <= 0:
      msg = "material state slot widths must be positive exact integers"
      raise TypeError(msg)
    if _nonempty_exact_str(self.dtype, "state slot dtype") not in _STATE_SLOT_DTYPES:
      msg = f"material state slot dtypes must be one of {_STATE_SLOT_DTYPES}"
      raise ValueError(msg)
    if (
      _nonempty_exact_str(self.lifetime, "state slot lifetime")
      not in _STATE_SLOT_LIFETIMES
    ):
      msg = f"material state slot lifetimes must be one of {_STATE_SLOT_LIFETIMES}"
      raise ValueError(msg)
    if self.annotation is not None and (
      type(self.annotation) is not str or self.annotation not in _STATE_SLOT_ANNOTATIONS
    ):
      msg = f"material state slot annotations must be one of {_STATE_SLOT_ANNOTATIONS}"
      raise ValueError(msg)


def _resolve_slot_width(declaration: object, parameters: dict[str, float]) -> int:
  """Resolve one slot width declaration, possibly from validated parameters."""
  if type(declaration) is int:
    if declaration <= 0:
      msg = "material state slot widths must be positive exact integers"
      raise ValueError(msg)
    return declaration
  if type(declaration) is dict:
    keys = set(declaration)
    if not keys.issubset(("parameter", "scale")) or "parameter" not in keys:
      msg = "parameterized slot widths declare exactly parameter plus optional scale"
      raise ValueError(msg)
    parameter = _nonempty_exact_str(declaration["parameter"], "slot width parameter")
    scale = declaration.get("scale", 1)
    if type(scale) is not int or scale <= 0:
      msg = "parameterized slot width scales must be positive exact integers"
      raise TypeError(msg)
    if parameter not in parameters:
      msg = f"slot width references undeclared material parameter {parameter!r}"
      raise ValueError(msg)
    value = parameters[parameter]
    if not float(value).is_integer() or value < 1.0:
      msg = f"slot width parameter {parameter!r} must be a positive integer value"
      raise ValueError(msg)
    return scale * int(value)
  msg = "material state slot widths must be exact ints or parameter mappings"
  raise TypeError(msg)


def resolve_material_state_slots(
  slots: object,
  parameters: dict[str, float],
) -> tuple[MaterialStateSlot, ...]:
  """Resolve descriptor ``state_slots`` metadata into typed slots.

  Widths are either literal positive ints or ``{"parameter": name,
  "scale": int}`` mappings resolved against the validated material parameters
  at compile time, so parameter-dependent laws (a Crystal-style slip count or a
  ViscoElasticity-style term count) fit the ABI without field changes.
  """
  if type(slots) is not list and type(slots) is not tuple:
    msg = "stateful material state_slots must be an exact list of declarations"
    raise TypeError(msg)
  if not slots:
    msg = "stateful material state_slots must declare at least one slot"
    raise ValueError(msg)
  resolved: list[MaterialStateSlot] = []
  for item in slots:
    if type(item) is not dict:
      msg = "material state slot declarations must be exact dictionaries"
      raise TypeError(msg)
    keys = set(item)
    if not keys.issubset(_STATE_SLOT_KEYS) or not keys.issuperset(
      _STATE_SLOT_REQUIRED_KEYS
    ):
      msg = (
        "material state slot declarations carry exactly name, width, dtype, "
        "lifetime, and optional annotation"
      )
      raise ValueError(msg)
    resolved.append(
      MaterialStateSlot(
        name=_nonempty_exact_str(item["name"], "state slot name"),
        width=_resolve_slot_width(item["width"], parameters),
        dtype=_nonempty_exact_str(item["dtype"], "state slot dtype"),
        lifetime=_nonempty_exact_str(item["lifetime"], "state slot lifetime"),
        annotation=item.get("annotation"),
      )
    )
  names = [slot.name for slot in resolved]
  if len(set(names)) != len(names):
    msg = "material state slot names must be unique"
    raise ValueError(msg)
  return tuple(resolved)


@dataclass(frozen=True, slots=True)
class MaterialSignalPort:
  """One resolved descriptor signal port declaration of the material ABI.

  ``port_id`` names the operator-facing port emitted onto the compiled
  ``SignalPortBinding``; ``signal_id`` names the program signal the driver
  binds to the port (this ABI revision binds declared program coordinates by
  name); ``derivative_coordinate_ids`` names the program coordinates whose
  ``d(signal)/d(coordinate)`` channels the driver forwards alongside the value.
  """

  port_id: str
  signal_id: str
  derivative_coordinate_ids: tuple[str, ...]

  def __post_init__(self) -> None:
    _nonempty_exact_str(self.port_id, "signal port id")
    _nonempty_exact_str(self.signal_id, "signal id")
    if type(self.derivative_coordinate_ids) is not tuple or any(
      type(item) is not str or not item for item in self.derivative_coordinate_ids
    ):
      msg = "material signal port derivative coordinates must be exact strings"
      raise TypeError(msg)
    if len(set(self.derivative_coordinate_ids)) != len(self.derivative_coordinate_ids):
      msg = "material signal port derivative coordinates must be unique"
      raise ValueError(msg)


def _signal_derivative_coordinates(value: object) -> tuple[str, ...]:
  if type(value) is not list:
    msg = "material signal port derivative_coordinate_ids must be an exact list"
    raise TypeError(msg)
  coordinates = tuple(
    _nonempty_exact_str(item, "signal derivative coordinate") for item in value
  )
  if len(set(coordinates)) != len(coordinates):
    msg = "material signal port derivative coordinates must be unique"
    raise ValueError(msg)
  return coordinates


def resolve_material_signal_ports(ports: object) -> tuple[MaterialSignalPort, ...]:
  """Resolve descriptor ``signal_ports`` metadata into typed port declarations.

  Each declaration names one typed signal port the compiled operator accepts
  program signal inputs for. Ports are the only channel through which schedule
  values (time-like or load-like coordinates) may reach a law: a descriptor
  without ``signal_ports`` compiles an operator that rejects every signal, so
  no hidden-global back channel can form.
  """
  if type(ports) is not list:
    msg = "stateful material signal_ports must be an exact list of declarations"
    raise TypeError(msg)
  if not ports:
    msg = "stateful material signal_ports must declare at least one port"
    raise ValueError(msg)
  resolved: list[MaterialSignalPort] = []
  for item in ports:
    if type(item) is not dict:
      msg = "material signal port declarations must be exact dictionaries"
      raise TypeError(msg)
    keys = set(item)
    if not keys.issubset(_SIGNAL_PORT_KEYS) or not keys.issuperset(
      _SIGNAL_PORT_REQUIRED_KEYS
    ):
      msg = (
        "material signal port declarations carry exactly port_id, signal_id, "
        "and optional derivative_coordinate_ids"
      )
      raise ValueError(msg)
    resolved.append(
      MaterialSignalPort(
        port_id=_nonempty_exact_str(item["port_id"], "signal port id"),
        signal_id=_nonempty_exact_str(item["signal_id"], "signal id"),
        derivative_coordinate_ids=_signal_derivative_coordinates(
          item.get("derivative_coordinate_ids", [])
        ),
      )
    )
  port_ids = [port.port_id for port in resolved]
  if len(set(port_ids)) != len(port_ids):
    msg = "material signal port ids must be unique"
    raise ValueError(msg)
  return tuple(resolved)


def validate_stateful_material_metadata(metadata: object) -> dict[str, object]:
  """Validate the structure of one v2 stateful material metadata mapping."""
  if type(metadata) is not dict:
    msg = "stateful material metadata must be an exact dictionary"
    raise TypeError(msg)
  keys = set(metadata)
  if (
    not keys.issuperset(_STATEFUL_METADATA_KEYS)
    or (keys - _STATEFUL_METADATA_KEYS) - _STATEFUL_OPTIONAL_METADATA_KEYS
  ):
    msg = (
      "stateful material metadata carries exactly the frozen v2 field set "
      f"{tuple(sorted(_STATEFUL_METADATA_KEYS))} plus the optional extension "
      f"fields {tuple(sorted(_STATEFUL_OPTIONAL_METADATA_KEYS))}"
    )
    raise ValueError(msg)
  if metadata["schema"] != STATEFUL_MATERIAL_DESCRIPTOR_SCHEMA:
    msg = f"stateful material schema must be {STATEFUL_MATERIAL_DESCRIPTOR_SCHEMA!r}"
    raise ValueError(msg)
  _nonempty_exact_str(metadata["law"], "law")
  _nonempty_exact_str(metadata["stress_state"], "stress_state")
  _nonempty_exact_str(metadata["state_schema"], "state_schema")
  names = _exact_str_list(metadata["parameter_names"], "parameter_names")
  if len(set(names)) != len(names):
    msg = "stateful material parameter names must be unique"
    raise ValueError(msg)
  if metadata["parameter_dtype"] != "float64":
    msg = "stateful material parameter_dtype must be 'float64'"
    raise ValueError(msg)
  _exact_str_list(metadata["stress_voigt_order"], "stress_voigt_order")
  _exact_str_list(metadata["internal_voigt_order"], "internal_voigt_order")
  if metadata["strain_shear_convention"] != "engineering":
    msg = "stateful material strain_shear_convention must be 'engineering'"
    raise ValueError(msg)
  tangent_class = metadata["tangent_class"]
  if (
    type(tangent_class) is not str or tangent_class not in _STATEFUL_TANGENT_CLASS_FLAGS
  ):
    msg = (
      "stateful material tangent_class must be one of "
      f"{tuple(sorted(_STATEFUL_TANGENT_CLASS_FLAGS))}"
    )
    raise ValueError(msg)
  slots = metadata["state_slots"]
  if type(slots) is not list or not slots:
    msg = "stateful material state_slots must be a non-empty exact list"
    raise TypeError(msg)
  for item in slots:
    if type(item) is not dict:
      msg = "material state slot declarations must be exact dictionaries"
      raise TypeError(msg)
    keys = set(item)
    if not keys.issubset(_STATE_SLOT_KEYS) or not keys.issuperset(
      _STATE_SLOT_REQUIRED_KEYS
    ):
      msg = (
        "material state slot declarations carry exactly name, width, dtype, "
        "lifetime, and optional annotation"
      )
      raise ValueError(msg)
    _nonempty_exact_str(item["name"], "state slot name")
    _nonempty_exact_str(item["dtype"], "state slot dtype")
    _nonempty_exact_str(item["lifetime"], "state slot lifetime")
    width = item["width"]
    if type(width) is dict:
      sub_keys = set(width)
      if not sub_keys.issubset(("parameter", "scale")) or "parameter" not in sub_keys:
        msg = "parameterized slot widths declare exactly parameter plus optional scale"
        raise ValueError(msg)
      _nonempty_exact_str(width["parameter"], "slot width parameter")
      if "scale" in width and (type(width["scale"]) is not int or width["scale"] <= 0):
        msg = "parameterized slot width scales must be positive exact integers"
        raise TypeError(msg)
    elif type(width) is not int or width <= 0:
      msg = "material state slot widths must be positive exact integers"
      raise TypeError(msg)
    annotation = item.get("annotation")
    if annotation is not None and (
      type(annotation) is not str or annotation not in _STATE_SLOT_ANNOTATIONS
    ):
      msg = f"material state slot annotations must be one of {_STATE_SLOT_ANNOTATIONS}"
      raise ValueError(msg)
  if "signal_ports" in metadata:
    resolve_material_signal_ports(metadata["signal_ports"])
  return metadata


def stateful_tangent_channel_flags(tangent_class: object) -> tuple[bool, bool]:
  """Map one stateful tangent class onto ``(linear, symmetric)`` channel flags.

  Every stateful class is nonlinear: no v2 tangent class is ever linear, so a
  stateful Jacobian channel can never mark itself factorization-reusable.
  """
  if (
    type(tangent_class) is not str or tangent_class not in _STATEFUL_TANGENT_CLASS_FLAGS
  ):
    msg = (
      "stateful tangent class must be one of "
      f"{tuple(sorted(_STATEFUL_TANGENT_CLASS_FLAGS))}"
    )
    raise ValueError(msg)
  return _STATEFUL_TANGENT_CLASS_FLAGS[tangent_class]


def material_state_layout_schema(
  schema_prefix: str,
  slots: tuple[MaterialStateSlot, ...],
) -> str:
  """Compose the layout schema string versioning the resolved slot layout."""
  _nonempty_exact_str(schema_prefix, "state schema prefix")
  if (
    type(slots) is not tuple
    or not slots
    or any(type(slot) is not MaterialStateSlot for slot in slots)
  ):
    msg = "state layout schemas require a non-empty resolved slot tuple"
    raise TypeError(msg)
  encoded = ",".join(f"{slot.name}:{slot.width}" for slot in slots)
  return f"{schema_prefix}|{encoded}"


def validated_material_initial_rows(
  value: object,
  *,
  entity_count: int,
  row_width: int,
) -> np.ndarray | None:
  """Validate descriptor-bound initial state rows, or ``None`` for zero-init."""
  if value is None:
    return None
  if (
    type(value) is not np.ndarray
    or value.dtype != np.dtype(np.float64)
    or value.dtype.metadata is not None
    or value.shape != (entity_count, row_width)
  ):
    msg = (
      "initial state rows must be a plain float64 ndarray with shape "
      "(entity_count, row_width)"
    )
    raise TypeError(msg)
  if not bool(np.isfinite(value).all()):
    msg = "initial state rows must be finite"
    raise ValueError(msg)
  return np.array(value, dtype=np.float64, order="C", copy=True, subok=False)


def build_material_state_layout(
  *,
  schema_prefix: str,
  block_id: SemanticId,
  entity_count: int,
  slots: tuple[MaterialStateSlot, ...],
  initial_rows: np.ndarray | None,
  index_dtype: np.dtype,
) -> OperatorStateLayout:
  """Emit the compiled operator state layout for resolved descriptor slots.

  The entity axis is the flattened element x integration-point x material-slot
  axis of the block's integration layout; every entity owns one contiguous row
  of ``sum(widths)`` float64 values. ``initial_rows`` (when the descriptor
  binds an initial state) is validated and embedded for the state owner to
  apply at construction.
  """
  schema = material_state_layout_schema(schema_prefix, slots)
  if type(entity_count) is not int or entity_count <= 0:
    msg = "state layout entity count must be a positive exact int"
    raise TypeError(msg)
  if not isinstance(index_dtype, np.dtype) or index_dtype.kind not in "iu":
    msg = "state layout index dtype must be an integer NumPy dtype"
    raise TypeError(msg)
  row_width = sum(slot.width for slot in slots)
  if entity_count * row_width > int(np.iinfo(index_dtype).max):
    msg = "state layout size overflows the compiled index dtype"
    raise ValueError(msg)
  initial = validated_material_initial_rows(
    initial_rows,
    entity_count=entity_count,
    row_width=row_width,
  )
  operator_slots = tuple(
    _new(
      OperatorStateSlot,
      name=slot.name,
      width=slot.width,
      dtype=_FLOAT64_STR,
      lifetime=_LIFETIME_ENUMS[slot.lifetime],
      annotation=slot.annotation,
    )
    for slot in slots
  )
  offsets = np.arange(entity_count + 1, dtype=index_dtype) * row_width
  return _new(
    OperatorStateLayout,
    schema=schema,
    block_id=block_id,
    entity_count=entity_count,
    slots=operator_slots,
    entity_offsets=FinalizedArray(offsets, dtype=index_dtype),
    row_width=row_width,
    dtype=_FLOAT64_STR,
    lifetime=StateLifetime.ACCEPTED_TRIAL,
    initial_rows=(
      None if initial is None else FinalizedArray(initial, dtype=np.float64)
    ),
  )


@dataclass(frozen=True, slots=True)
class StatefulContinuumKernelResult:
  """One batched stateful law response over flattened state entities.

  ``stresses`` has shape ``(entity_count, 6)`` in the law's internal Voigt
  order, ``tangents`` has shape ``(entity_count, 6, 6)``, and ``trial_rows``
  has shape ``(entity_count, row_width)``. When ``status`` is not ``OK`` the
  operator discards the arrays and returns the accepted rows byte-equal.
  """

  stresses: np.ndarray
  tangents: np.ndarray
  trial_rows: np.ndarray
  status: EvaluationStatus

  def __post_init__(self) -> None:
    if type(self.status) is not EvaluationStatus:
      msg = "stateful kernel status must be an exact EvaluationStatus"
      raise TypeError(msg)
    for label, array in (
      ("stresses", self.stresses),
      ("tangents", self.tangents),
      ("trial_rows", self.trial_rows),
    ):
      if (
        type(array) is not np.ndarray
        or array.dtype != np.dtype(np.float64)
        or array.dtype.metadata is not None
      ):
        msg = f"stateful kernel {label} must be a plain float64 ndarray"
        raise TypeError(msg)


class StatefulContinuumKernel(Protocol):
  """Batched stateful law kernel: total strains and accepted rows in, trial out."""

  def __call__(
    self,
    strains: np.ndarray,
    accepted_rows: np.ndarray,
    calibration: np.ndarray,
  ) -> StatefulContinuumKernelResult: ...


def _plain_float64_scalar(value: object, label: str) -> None:
  if (
    type(value) is not np.ndarray
    or value.dtype != np.dtype(np.float64)
    or value.dtype.metadata is not None
    or value.shape != (1,)
  ):
    msg = f"stateful material {label} must be a plain one-element float64 array"
    raise TypeError(msg)


@dataclass(frozen=True, slots=True)
class StatefulContinuumSignalDerivative:
  """One program-coordinate derivative channel of one bound signal port.

  ``values`` holds ``d(signal)/d(coordinate_id)`` at the bound program point.
  This ABI revision's signals are scalar, so every values array carries exactly
  one float64.
  """

  coordinate_id: str
  values: np.ndarray

  def __post_init__(self) -> None:
    _nonempty_exact_str(self.coordinate_id, "signal derivative coordinate")
    _plain_float64_scalar(self.values, "signal derivative values")


@dataclass(frozen=True, slots=True)
class StatefulContinuumSignalInput:
  """One bound signal port forwarded to a stateful kernel evaluation.

  The compiled operator builds these from validated ``ProgramSignalInput``
  values in declared port order; ``derivatives`` follows the port's declared
  ``derivative_coordinate_ids`` order exactly. ``values`` carries the bound
  scalar signal value at the current program point.
  """

  port_id: str
  values: np.ndarray
  derivatives: tuple[StatefulContinuumSignalDerivative, ...]

  def __post_init__(self) -> None:
    _nonempty_exact_str(self.port_id, "signal port id")
    _plain_float64_scalar(self.values, "signal values")
    if type(self.derivatives) is not tuple or any(
      type(item) is not StatefulContinuumSignalDerivative for item in self.derivatives
    ):
      msg = (
        "stateful signal derivatives must be an exact tuple of "
        "StatefulContinuumSignalDerivative values"
      )
      raise TypeError(msg)


class StatefulContinuumSignalKernel(Protocol):
  """Batched stateful law kernel with bound signal ports (four-argument form).

  Descriptors declaring ``signal_ports`` bind kernels of this form: the fourth
  positional argument carries one ``StatefulContinuumSignalInput`` per declared
  port, in declaration order. Signal-free descriptors keep the three-argument
  ``StatefulContinuumKernel`` call byte-identical.
  """

  def __call__(
    self,
    strains: np.ndarray,
    accepted_rows: np.ndarray,
    calibration: np.ndarray,
    signals: tuple[StatefulContinuumSignalInput, ...],
  ) -> StatefulContinuumKernelResult: ...


class StatefulContinuumBinding(Protocol):
  """The permanent stateful-material binding ABI behind registry descriptors.

  ``__call__`` maps the validated parameter tuple (in ``parameter_names``
  order) onto the law's flat float64 calibration vector. ``kernel`` evaluates
  total engineering 6-Voigt strain batches plus accepted state rows into
  stresses, algorithmic tangents, and trial rows. ``descriptor_metadata``
  re-declares the descriptor's v2 metadata so the compiler can verify the
  binding against the captured canonical bytes. Bindings may omit
  ``initial_state``; laws without it keep zero-initialized state rows.
  Kernels must return the accepted rows byte-equal on a zero-strain
  evaluation over the initial state (the compiler probes this invariant), and
  expected numerical failures report typed statuses with byte-equal trial
  rows, never exceptions.

  A descriptor declaring the optional ``signal_ports`` field requires a kernel
  accepting the bound signal tuple as its fourth positional argument (the
  ``StatefulContinuumSignalKernel`` form); the compiler probes that form before
  the operator can escape. Signal values reach the law exclusively through
  this argument — there is no schedule back channel.
  """

  def __call__(self, *parameters: float) -> np.ndarray: ...

  def descriptor_metadata(self) -> dict[str, object]: ...

  def kernel(
    self,
    strains: np.ndarray,
    accepted_rows: np.ndarray,
    calibration: np.ndarray,
  ) -> StatefulContinuumKernelResult: ...

  def initial_state(
    self,
    parameters: tuple[float, ...],
    layout: OperatorStateLayout,
  ) -> np.ndarray: ...
