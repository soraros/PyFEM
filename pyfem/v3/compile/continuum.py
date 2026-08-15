"""Concrete direct compiler and evaluator for the qualified Q8 continuum slice."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NoReturn

import numpy as np

from pyfem.v3.compile.diagnostics import (
  ModelCompilationDiagnostic,
  ModelCompilationError,
)
from pyfem.v3.fem.kinematics import strain_displacement
from pyfem.v3.fem.quadrature import gauss_tensor_product_2d
from pyfem.v3.fem.shapes import serendipity_quad8
from pyfem.v3.materials.plane_stress import plane_stress_matrix
from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.operator import (
  BalanceRole,
  ChannelRequest,
  CompilerConstructed,
  CouplingPolicy,
  ImplementationIdentity,
  JacobianChannel,
  OperatorEvaluation,
  OperatorEvaluationInput,
  OperatorHeader,
  OperatorStateLayout,
  PortBinding,
  PortMode,
  ResidualChannel,
  StateLifetime,
)
from pyfem.v3.model.provenance import CanonicalManifest
from pyfem.v3.model.registry import (
  RegistryDescriptor,
  RegistryKey,
  RegistrySnapshot,
)
from pyfem.v3.model.system import CompiledSource, DiscreteSpace, IncidenceEntityBlock
from pyfem.v3.spec.diagnostics import SourceContext, render_diagnostic_value
from pyfem.v3.spec.model import (
  CellBlockSpec,
  CellSpec,
  FieldSpec,
  MaterialSpec,
  ModelSpec,
  RegionSpec,
  SpecId,
)

Q8_TOPOLOGY_KEY: RegistryKey = ("topology", "serendipity-quad8")
Q8_QUADRATURE_KEY: RegistryKey = ("quadrature", "gauss-3x3")
Q8_FORMULATION_KEY: RegistryKey = ("formulation", "small-strain-continuum")
Q8_MATERIAL_KEY: RegistryKey = ("material", "plane-stress-linear-elastic")
_REQUIRED_KEYS = (
  Q8_TOPOLOGY_KEY,
  Q8_QUADRATURE_KEY,
  Q8_FORMULATION_KEY,
  Q8_MATERIAL_KEY,
)
_PARAMETER_NAMES = ("youngs_modulus", "poisson_ratio")
_POINT_COUNT = 9
_NODE_COUNT = 8
_LOCAL_COEFFICIENT_COUNT = 16


def _new[ValueT](cls: type[ValueT], /, **fields: object) -> ValueT:
  value = object.__new__(cls)
  for name, field in fields.items():
    object.__setattr__(value, name, field)
  return value


def _fail(code: str, message: str, source: SourceContext) -> NoReturn:
  raise ModelCompilationError(
    (ModelCompilationDiagnostic(code=code, message=message, source=source),)
  )


def _sort_key(value: SpecId | tuple[SpecId, ...]) -> tuple[int, object]:
  if type(value) is int:
    return 0, value
  if type(value) is str:
    return 1, value
  return 2, tuple(_sort_key(item) for item in value)


def _source(value: SourceContext) -> CompiledSource:
  return _new(
    CompiledSource,
    source=value.source,
    line=value.line,
    column=value.column,
  )


@dataclass(frozen=True, slots=True, eq=False)
class ContinuumSelection:
  block: CellBlockSpec
  field: FieldSpec
  material: MaterialSpec
  region: RegionSpec
  cells: tuple[CellSpec, ...]


@dataclass(frozen=True, slots=True, eq=False, init=False)
class Q8ContinuumPayload(CompilerConstructed):
  quadrature_points: FinalizedArray
  quadrature_weights: FinalizedArray
  shape_values: FinalizedArray
  parent_gradients: FinalizedArray
  geometry_scales: FinalizedArray
  normalized_gradients: FinalizedArray
  normalized_strain_displacement: FinalizedArray
  normalized_integration_weights: FinalizedArray
  constitutive: FinalizedArray
  material_parameters: FinalizedArray

  def physical_gradients(self) -> FinalizedArray:
    values = (
      self.normalized_gradients.values
      / self.geometry_scales.values[:, None, None, None]
    )
    return FinalizedArray(values, dtype=np.float64)

  def physical_strain_displacement(self) -> FinalizedArray:
    values = (
      self.normalized_strain_displacement.values
      / self.geometry_scales.values[:, None, None, None]
    )
    return FinalizedArray(values, dtype=np.float64)

  def physical_integration_weights(self) -> FinalizedArray:
    scales = self.geometry_scales.values[:, None]
    values = self.normalized_integration_weights.values * scales * scales
    return FinalizedArray(values, dtype=np.float64)


@dataclass(frozen=True, slots=True, eq=False, init=False)
class Q8ContinuumOperator(CompilerConstructed):
  header: OperatorHeader
  entity_block: IncidenceEntityBlock
  payload: Q8ContinuumPayload
  content_manifest: CanonicalManifest

  def evaluate(
    self,
    inputs: OperatorEvaluationInput,
  ) -> OperatorEvaluation:
    """Evaluate local internal force and material tangent from compiled meaning."""
    if type(inputs) is not OperatorEvaluationInput:
      msg = "Q8 evaluation requires an exact immutable evaluation input"
      raise TypeError(msg)
    if type(inputs.port_values) is not tuple or len(inputs.port_values) != 1:
      msg = "Q8 evaluation requires exactly one displacement port batch"
      raise TypeError(msg)
    values = inputs.port_values[0].values
    expected = self.header.ports[0].coefficient_map.values.shape
    if (
      values.dtype != np.dtype(np.float64)
      or values.dtype.metadata is not None
      or values.shape != expected
      or not bool(np.isfinite(values).all())
    ):
      msg = "Q8 displacement port values must be a finite metadata-free float64 batch"
      raise TypeError(msg)
    layout = self.header.state_layout
    accepted_state = inputs.accepted_state.values
    if (
      accepted_state.dtype != np.dtype(np.float64)
      or accepted_state.dtype.metadata is not None
      or accepted_state.shape != layout.row_shape
      or not bool(np.isfinite(accepted_state).all())
    ):
      msg = "Q8 accepted state must match the compiled zero-width state layout"
      raise TypeError(msg)
    if inputs.signals or self.header.signal_ports:
      msg = "Q8 model operator does not accept program signal inputs"
      raise ValueError(msg)
    residual_ids = tuple(item.channel_id for item in self.header.residual_channels)
    jacobian_ids = tuple(item.channel_id for item in self.header.jacobian_channels)
    request = inputs.request
    if (
      type(request) is not ChannelRequest
      or len(set(request.residual_channel_ids)) != len(request.residual_channel_ids)
      or len(set(request.jacobian_channel_ids)) != len(request.jacobian_channel_ids)
      or not set(request.residual_channel_ids).issubset(residual_ids)
      or not set(request.jacobian_channel_ids).issubset(jacobian_ids)
    ):
      msg = "Q8 evaluation request contains an unavailable or duplicate channel"
      raise ValueError(msg)

    b_matrix = self.payload.normalized_strain_displacement.values
    weights = self.payload.normalized_integration_weights.values
    constitutive = self.payload.constitutive.values
    tangent = np.einsum(
      "ep,epai,ab,epbj->eij",
      weights,
      b_matrix,
      constitutive,
      b_matrix,
      optimize=True,
    )
    residual_values = ()
    if request.residual_channel_ids:
      residual = np.einsum("eij,ej->ei", tangent, values, optimize=True)
      residual_values = (FinalizedArray(residual, dtype=np.float64),)
    jacobian_values = (
      (FinalizedArray(tangent, dtype=np.float64),)
      if request.jacobian_channel_ids
      else ()
    )
    return _new(
      OperatorEvaluation,
      residual_values=residual_values,
      jacobian_values=jacobian_values,
      trial_state=FinalizedArray(accepted_state, dtype=np.float64),
    )


def q8_descriptor_metadata(kind: str, name: str) -> dict[str, object]:
  """Return the exact semantic metadata for one selected Q8 implementation."""
  key = kind, name
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
  msg = "no Q8 descriptor metadata exists for that exact registry key"
  raise KeyError(msg)


def q8_reference_registry() -> dict[RegistryKey, RegistryDescriptor]:
  """Build a fresh injectable registry for the qualified Q8 convention."""
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
  }
  descriptors = tuple(
    RegistryDescriptor(
      kind=key[0],
      name=key[1],
      version="1",
      implementation_id=implementation_id,
      metadata=q8_descriptor_metadata(*key),
      binding=binding,
    )
    for key, (implementation_id, binding) in bindings.items()
  )
  return {descriptor.key: descriptor for descriptor in descriptors}


def select_model(spec: ModelSpec) -> ContinuumSelection:
  """Validate the concrete authored slice after the sole normalization pass."""
  for field in spec.fields:
    if field.location != "node":
      _fail(
        "unsupported-space-support",
        "the direct Q8 slice cannot allocate a non-node field",
        field.source,
      )
  cells_by_key = {
    (block.id, cell.id): cell for block in spec.mesh.cell_blocks for cell in block.cells
  }
  counts = {key: 0 for key in cells_by_key}
  for region in spec.regions:
    for cell_ref in region.cell_refs:
      key = cell_ref.block_id, cell_ref.cell_id
      if key in counts:
        counts[key] += 1
  for key in sorted(counts, key=_sort_key):
    count = counts[key]
    cell = cells_by_key[key]
    rendered = render_diagnostic_value(key)
    if count == 0:
      _fail(
        "incomplete-cell-membership",
        f"source cell {rendered} does not belong to a compiled region",
        cell.source,
      )
    if count > 1:
      _fail(
        "multiple-cell-membership",
        f"source cell {rendered} belongs to more than one compiled region",
        cell.source,
      )
  if (
    len(spec.mesh.cell_blocks) != 1
    or len(spec.fields) != 1
    or len(spec.materials) != 1
    or len(spec.regions) != 1
  ):
    _fail(
      "unsupported-continuum-declaration-count",
      "the direct Q8 slice requires one active field, cell block, material, and region",
      spec.source,
    )
  block = spec.mesh.cell_blocks[0]
  material = spec.materials[0]
  region = spec.regions[0]
  fields = {field.id: field for field in spec.fields}
  if len(region.field_ids) != 1 or region.field_ids[0] not in fields:
    _fail(
      "incompatible-region-field-signature",
      "the Q8 region must reference exactly one declared field",
      region.source,
    )
  field = fields[region.field_ids[0]]
  if (
    block.reference_topology != "quadrilateral"
    or block.topological_dimension != 2
    or block.embedding_dimension != 2
    or block.geometry_interpolation != "serendipity-quad8"
  ):
    _fail(
      "incompatible-cell-block",
      "Q8 requires explicit two-dimensional quadrilateral serendipity geometry",
      block.source,
    )
  for cell in block.cells:
    if len(cell.node_ids) != _NODE_COUNT:
      _fail(
        "invalid-q8-arity",
        f"Q8 cell {render_diagnostic_value(cell.id)} must reference eight nodes",
        cell.source,
      )
  if field.location != "node" or field.components != ("x", "y"):
    _fail(
      "incompatible-field-signature",
      "Q8 requires one node field with physical components ('x', 'y')",
      field.source,
    )
  if region.material_id != material.id:
    _fail(
      "incompatible-region-material",
      "the Q8 region must reference the selected material",
      region.source,
    )
  if region.formulation != Q8_FORMULATION_KEY[1]:
    _fail("incompatible-formulation", "unsupported Q8 formulation", region.source)
  if region.quadrature != Q8_QUADRATURE_KEY[1]:
    _fail("incompatible-quadrature", "unsupported Q8 quadrature", region.source)
  if material.model != Q8_MATERIAL_KEY[1]:
    _fail("incompatible-material-model", "unsupported Q8 material", material.source)
  return ContinuumSelection(
    block=block,
    field=field,
    material=material,
    region=region,
    cells=tuple(sorted(block.cells, key=lambda item: _sort_key(item.id))),
  )


def capture_registry(
  registry: dict[RegistryKey, RegistryDescriptor],
  selection: ContinuumSelection,
) -> RegistrySnapshot:
  """Capture and validate exactly the implementations selected by this builder."""
  try:
    snapshot = RegistrySnapshot.capture(registry, required=_REQUIRED_KEYS)
  except (KeyError, TypeError, ValueError):
    _fail(
      "registry-capture-failed",
      "the injected registry could not capture the required Q8 implementations",
      selection.region.source,
    )
  sources = {
    Q8_TOPOLOGY_KEY: selection.block.source,
    Q8_QUADRATURE_KEY: selection.region.source,
    Q8_FORMULATION_KEY: selection.region.source,
    Q8_MATERIAL_KEY: selection.material.source,
  }
  for key in _REQUIRED_KEYS:
    try:
      descriptor = snapshot.resolve(*key)
      expected = CanonicalManifest(q8_descriptor_metadata(*key))
      compatible = descriptor.metadata.to_bytes() == expected.to_bytes()
    except (KeyError, TypeError, ValueError):
      _fail("malformed-registry-descriptor", "malformed Q8 descriptor", sources[key])
    if not compatible:
      _fail(
        "incompatible-registry-descriptor",
        "Q8 descriptor metadata does not match the qualified convention",
        sources[key],
      )
  return snapshot


def _binding_array(
  value: object,
  *,
  shape: tuple[int, ...],
  code: str,
  label: str,
  source: SourceContext,
) -> np.ndarray:
  if (
    type(value) is not np.ndarray
    or value.shape != shape
    or value.dtype.metadata is not None
    or value.dtype.kind not in "iuf"
  ):
    _fail(code, f"{label} must return a metadata-free numeric array {shape!r}", source)
  try:
    captured = np.array(value, dtype=np.float64, order="C", copy=True, subok=False)
  except (OverflowError, TypeError, ValueError):
    _fail(code, f"{label} cannot be represented as float64", source)
  if not bool(np.isfinite(captured).all()):
    _fail(code, f"{label} must contain finite values", source)
  return captured


def _corresponds(actual: np.ndarray, expected: np.ndarray) -> bool:
  scale = max(1.0, float(np.max(np.abs(expected))))
  tolerance = 8.0 * float(np.finfo(np.float64).eps) * scale
  return bool(np.allclose(actual, expected, rtol=0.0, atol=tolerance))


def _qualified_shapes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  xi, eta = points[:, 0], points[:, 1]
  values = np.stack(
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
  return values, np.stack((dxi, deta), axis=2)


def _recipes(
  snapshot: RegistrySnapshot,
  selection: ContinuumSelection,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  quadrature = snapshot.resolve(*Q8_QUADRATURE_KEY).binding
  try:
    raw_quadrature = quadrature(3)
  except Exception:
    _fail(
      "quadrature-binding-failed",
      "Q8 quadrature binding failed",
      selection.region.source,
    )
  if type(raw_quadrature) is not tuple or len(raw_quadrature) != 2:
    _fail(
      "invalid-quadrature-binding-output",
      "Q8 quadrature must return exactly points and weights",
      selection.region.source,
    )
  points = _binding_array(
    raw_quadrature[0],
    shape=(_POINT_COUNT, 2),
    code="invalid-quadrature-binding-output",
    label="Q8 quadrature points",
    source=selection.region.source,
  )
  weights = _binding_array(
    raw_quadrature[1],
    shape=(_POINT_COUNT,),
    code="invalid-quadrature-binding-output",
    label="Q8 quadrature weights",
    source=selection.region.source,
  )
  abscissa = math.sqrt(3.0 / 5.0)
  expected_points = np.array(
    [(x, y) for x in (-abscissa, 0.0, abscissa) for y in (-abscissa, 0.0, abscissa)],
    dtype=np.float64,
  )
  expected_weights = np.array(
    [
      x * y
      for x in (5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0)
      for y in (5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0)
    ],
    dtype=np.float64,
  )
  if (
    bool(np.any(weights <= 0.0))
    or bool(np.any(np.abs(points) > 1.0))
    or not math.isclose(float(weights.sum()), 4.0, rel_tol=1.0e-14, abs_tol=1.0e-14)
    or not _corresponds(points, expected_points)
    or not _corresponds(weights, expected_weights)
  ):
    _fail(
      "invalid-quadrature-binding-output",
      "Q8 quadrature requires positive in-domain weights summing to four",
      selection.region.source,
    )
  topology = snapshot.resolve(*Q8_TOPOLOGY_KEY).binding
  try:
    raw_topology = topology(np.array(points, copy=True))
  except Exception:
    _fail(
      "topology-binding-failed", "Q8 topology binding failed", selection.block.source
    )
  if type(raw_topology) is not tuple or len(raw_topology) != 2:
    _fail(
      "invalid-topology-binding-output",
      "Q8 topology must return exactly shape values and parent gradients",
      selection.block.source,
    )
  shape_values = _binding_array(
    raw_topology[0],
    shape=(_POINT_COUNT, _NODE_COUNT),
    code="invalid-topology-binding-output",
    label="Q8 shape values",
    source=selection.block.source,
  )
  parent_gradients = _binding_array(
    raw_topology[1],
    shape=(_POINT_COUNT, _NODE_COUNT, 2),
    code="invalid-topology-binding-output",
    label="Q8 parent gradients",
    source=selection.block.source,
  )
  expected_values, expected_gradients = _qualified_shapes(points)
  if (
    not bool(np.allclose(shape_values.sum(1), 1.0, rtol=0.0, atol=1.0e-12))
    or not bool(np.allclose(parent_gradients.sum(1), 0.0, rtol=0.0, atol=1.0e-12))
    or not _corresponds(shape_values, expected_values)
    or not _corresponds(parent_gradients, expected_gradients)
  ):
    _fail(
      "invalid-topology-binding-output",
      "Q8 topology violates partition or gradient completeness",
      selection.block.source,
    )
  return points, weights, shape_values, parent_gradients


def _parameters(selection: ContinuumSelection) -> tuple[float, float]:
  by_name = {item.name: item for item in selection.material.parameters}
  if tuple(sorted(by_name)) != tuple(sorted(_PARAMETER_NAMES)):
    _fail(
      "invalid-material-parameter-schema",
      "Q8 plane stress requires youngs_modulus and poisson_ratio",
      selection.material.source,
    )
  values: list[float] = []
  for name in _PARAMETER_NAMES:
    parameter = by_name[name]
    if type(parameter.value) is not int and type(parameter.value) is not float:
      _fail(
        "invalid-material-parameter-type",
        f"{name} must be an exact scalar",
        parameter.source,
      )
    try:
      value = float(parameter.value)
    except OverflowError:
      _fail(
        "invalid-material-parameter-value",
        f"{name} must fit finite float64",
        parameter.source,
      )
    if not math.isfinite(value):
      _fail(
        "invalid-material-parameter-value",
        f"{name} must fit finite float64",
        parameter.source,
      )
    values.append(value)
  youngs_modulus, poisson_ratio = values
  if youngs_modulus <= 0.0:
    _fail(
      "invalid-youngs-modulus",
      "youngs_modulus must be positive",
      by_name[_PARAMETER_NAMES[0]].source,
    )
  if not -1.0 < poisson_ratio < 0.5:
    _fail(
      "invalid-poisson-ratio",
      "poisson_ratio must lie between -1 and 0.5",
      by_name[_PARAMETER_NAMES[1]].source,
    )
  return youngs_modulus, poisson_ratio


def _normalized_coordinates(
  coordinates: np.ndarray, cell: CellSpec
) -> tuple[np.ndarray, float]:
  coordinate_scale = 1.0
  with np.errstate(over="ignore", invalid="ignore", under="ignore"):
    relative = coordinates - coordinates[0]
  if not bool(np.isfinite(relative).all()):
    coordinate_scale = float(np.max(np.abs(coordinates)))
    if not math.isfinite(coordinate_scale) or coordinate_scale == 0.0:
      _fail(
        "non-finite-reference-geometry",
        "Q8 has non-finite or zero-scale geometry",
        cell.source,
      )
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
      relative = coordinates / coordinate_scale - coordinates[0] / coordinate_scale
  cell_scale = float(np.max(np.abs(relative)))
  if not math.isfinite(cell_scale) or cell_scale == 0.0:
    _fail(
      "non-finite-reference-geometry",
      "Q8 has non-finite or zero-scale geometry",
      cell.source,
    )
  normalized = relative / cell_scale
  if not bool(np.isfinite(normalized).all()):
    _fail(
      "non-finite-reference-geometry",
      "Q8 normalized geometry is non-finite",
      cell.source,
    )
  physical_scale = coordinate_scale * cell_scale
  if not math.isfinite(physical_scale) or physical_scale == 0.0:
    _fail(
      "non-finite-reference-geometry", "Q8 physical scale is not finite", cell.source
    )
  return normalized, physical_scale


def _geometry(
  coordinates: np.ndarray,
  connectivity: np.ndarray,
  parent_gradients: np.ndarray,
  cells: tuple[CellSpec, ...],
  tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  gradients = np.empty((len(cells), _POINT_COUNT, _NODE_COUNT, 2), dtype=np.float64)
  determinants = np.empty((len(cells), _POINT_COUNT), dtype=np.float64)
  scales = np.empty(len(cells), dtype=np.float64)
  for cell_index, cell in enumerate(cells):
    normalized, scales[cell_index] = _normalized_coordinates(
      coordinates[connectivity[cell_index]], cell
    )
    norm_squares = np.empty(_POINT_COUNT, dtype=np.float64)
    for point_index in range(_POINT_COUNT):
      jacobian = normalized.T @ parent_gradients[point_index]
      determinant = float(
        jacobian[0, 0] * jacobian[1, 1] - jacobian[0, 1] * jacobian[1, 0]
      )
      norm_square = math.fsum(float(value * value) for value in jacobian.flat)
      if (
        not math.isfinite(determinant)
        or not math.isfinite(norm_square)
        or norm_square == 0.0
      ):
        _fail(
          "non-finite-reference-geometry",
          "Q8 has a non-finite reference Jacobian",
          cell.source,
        )
      determinants[cell_index, point_index] = determinant
      norm_squares[point_index] = norm_square
    signs = determinants[cell_index]
    if bool(np.any(signs > 0.0)) and bool(np.any(signs < 0.0)):
      _fail(
        "sign-changing-reference-geometry",
        "Q8 changes orientation across quadrature points",
        cell.source,
      )
    if bool(np.any(signs < 0.0)):
      _fail(
        "inverted-reference-geometry",
        "Q8 has negative reference orientation",
        cell.source,
      )
    for point_index in range(_POINT_COUNT):
      determinant = float(determinants[cell_index, point_index])
      if (
        determinant <= tolerance
        or determinant / float(norm_squares[point_index]) <= tolerance
      ):
        _fail(
          "near-singular-reference-geometry",
          f"Q8 cell {render_diagnostic_value(cell.id)} has a scale-relative "
          f"near-singular Jacobian at point {point_index}",
          cell.source,
        )
      jacobian = normalized.T @ parent_gradients[point_index]
      inverse = np.array(
        ((jacobian[1, 1], -jacobian[0, 1]), (-jacobian[1, 0], jacobian[0, 0])),
        dtype=np.float64,
      )
      inverse /= determinant
      gradients[cell_index, point_index] = parent_gradients[point_index] @ inverse
      if not bool(np.isfinite(gradients[cell_index, point_index]).all()):
        _fail(
          "non-finite-reference-geometry",
          "Q8 physical gradients are non-finite",
          cell.source,
        )
  return gradients, determinants, scales


def _identities(snapshot: RegistrySnapshot) -> tuple[ImplementationIdentity, ...]:
  return tuple(
    _new(
      ImplementationIdentity,
      kind=descriptor.kind,
      name=descriptor.name,
      version=descriptor.version,
      implementation_id=descriptor.implementation_id,
    )
    for descriptor in snapshot.descriptors
  )


def compile_operator(
  selection: ContinuumSelection,
  *,
  coordinates: FinalizedArray,
  node_dense: dict[SpecId, int],
  space: DiscreteSpace,
  snapshot: RegistrySnapshot,
  index_dtype: np.dtype,
  geometry_relative_tolerance: float,
) -> tuple[IncidenceEntityBlock, Q8ContinuumOperator]:
  """Compile the concrete payload and return it behind the open operator header."""
  connectivity_values = [
    [node_dense[node_id] for node_id in cell.node_ids] for cell in selection.cells
  ]
  connectivity = FinalizedArray(connectivity_values, dtype=index_dtype)
  entity_block_id = selection.block.id
  entity_block = _new(
    IncidenceEntityBlock,
    block_id=entity_block_id,
    entity_ids=tuple(cell.id for cell in selection.cells),
    sources=tuple(_source(cell.source) for cell in selection.cells),
    incidence=connectivity,
  )
  points, weights, shape_values, parent_gradients = _recipes(snapshot, selection)
  gradients, determinants, geometry_scales = _geometry(
    coordinates.values,
    connectivity.values,
    parent_gradients,
    selection.cells,
    geometry_relative_tolerance,
  )
  formulation = snapshot.resolve(*Q8_FORMULATION_KEY).binding
  try:
    raw_b_matrix = formulation(np.array(gradients, copy=True))
  except Exception:
    _fail(
      "formulation-binding-failed",
      "Q8 formulation binding failed",
      selection.region.source,
    )
  b_matrix = _binding_array(
    raw_b_matrix,
    shape=(len(selection.cells), _POINT_COUNT, 3, _LOCAL_COEFFICIENT_COUNT),
    code="invalid-formulation-binding-output",
    label="Q8 strain-displacement binding",
    source=selection.region.source,
  )
  expected_b = np.zeros_like(b_matrix)
  expected_b[..., 0, 0::2] = gradients[..., :, 0]
  expected_b[..., 1, 1::2] = gradients[..., :, 1]
  expected_b[..., 2, 0::2] = gradients[..., :, 1]
  expected_b[..., 2, 1::2] = gradients[..., :, 0]
  if not _corresponds(b_matrix, expected_b):
    _fail(
      "incompatible-formulation-binding-output",
      "Q8 formulation output contradicts the qualified engineering-shear map",
      selection.region.source,
    )
  youngs_modulus, poisson_ratio = _parameters(selection)
  material = snapshot.resolve(*Q8_MATERIAL_KEY).binding
  try:
    raw_constitutive = material(youngs_modulus, poisson_ratio)
  except Exception:
    _fail(
      "material-binding-failed", "Q8 material binding failed", selection.material.source
    )
  constitutive = _binding_array(
    raw_constitutive,
    shape=(3, 3),
    code="invalid-material-binding-output",
    label="Q8 material binding",
    source=selection.material.source,
  )
  modulus = youngs_modulus / (1.0 - poisson_ratio * poisson_ratio)
  expected_constitutive = np.array(
    [
      [modulus, modulus * poisson_ratio, 0.0],
      [modulus * poisson_ratio, modulus, 0.0],
      [0.0, 0.0, youngs_modulus / (2.0 * (1.0 + poisson_ratio))],
    ],
    dtype=np.float64,
  )
  if not bool(np.array_equal(constitutive, constitutive.T)):
    _fail(
      "nonsymmetric-material-binding",
      "Q8 material tangent must be symmetric",
      selection.material.source,
    )
  if not _corresponds(constitutive, expected_constitutive):
    _fail(
      "incompatible-material-binding-output",
      "Q8 material output contradicts the qualified plane-stress law",
      selection.material.source,
    )
  integration_weights = determinants * weights[None, :]
  tangent = np.einsum(
    "ep,epai,ab,epbj->eij",
    integration_weights,
    b_matrix,
    constitutive,
    b_matrix,
    optimize=True,
  )
  if not bool(np.isfinite(tangent).all()):
    _fail(
      "non-finite-element-operator",
      "Q8 element operator is non-finite",
      selection.region.source,
    )
  tangent_scale = float(np.max(np.abs(tangent)))
  symmetry_tolerance = 64.0 * float(np.finfo(np.float64).eps) * tangent_scale
  if not bool(
    np.allclose(
      tangent,
      tangent.transpose(0, 2, 1),
      rtol=0.0,
      atol=symmetry_tolerance,
    )
  ):
    _fail(
      "nonsymmetric-element-operator",
      "Q8 element operator is not symmetric",
      selection.region.source,
    )

  gather = FinalizedArray(
    space.coefficient_map.values[connectivity.values].reshape(len(selection.cells), -1),
    dtype=index_dtype,
  )
  block_id = selection.block.id, selection.region.id
  state_layout = _new(
    OperatorStateLayout,
    schema="pyfem-v3-operator-state-layout-v1",
    block_id=block_id,
    entity_count=len(selection.cells),
    slots=(),
    entity_offsets=FinalizedArray(
      np.zeros(len(selection.cells) + 1), dtype=index_dtype
    ),
    row_width=0,
    dtype=np.dtype(np.float64).str,
    lifetime=StateLifetime.ACCEPTED_TRIAL,
  )
  port = _new(
    PortBinding,
    port_id="displacement",
    space_id=space.space_id,
    mode=PortMode.COEFFICIENTS,
    coefficient_map=gather,
  )
  residual_channel = _new(
    ResidualChannel,
    channel_id="internal-force",
    target_port_id=port.port_id,
    balance_role=BalanceRole.INTERNAL,
    linear=True,
  )
  jacobian_channel = _new(
    JacobianChannel,
    channel_id="material-tangent",
    residual_channel_id="internal-force",
    target_port_id=port.port_id,
    source_port_id=port.port_id,
    balance_role=BalanceRole.INTERNAL,
    linear=True,
    symmetric=True,
  )
  header = _new(
    OperatorHeader,
    block_id=block_id,
    entity_block_id=entity_block_id,
    implementations=_identities(snapshot),
    ports=(port,),
    signal_ports=(),
    residual_channels=(residual_channel,),
    jacobian_channels=(jacobian_channel,),
    state_layout=state_layout,
    coupling_policy=CouplingPolicy.FIXED,
  )
  payload = _new(
    Q8ContinuumPayload,
    quadrature_points=FinalizedArray(points, dtype=np.float64),
    quadrature_weights=FinalizedArray(weights, dtype=np.float64),
    shape_values=FinalizedArray(shape_values, dtype=np.float64),
    parent_gradients=FinalizedArray(parent_gradients, dtype=np.float64),
    geometry_scales=FinalizedArray(geometry_scales, dtype=np.float64),
    normalized_gradients=FinalizedArray(gradients, dtype=np.float64),
    normalized_strain_displacement=FinalizedArray(b_matrix, dtype=np.float64),
    normalized_integration_weights=FinalizedArray(
      integration_weights, dtype=np.float64
    ),
    constitutive=FinalizedArray(constitutive, dtype=np.float64),
    material_parameters=FinalizedArray(
      [[youngs_modulus, poisson_ratio]], dtype=np.float64
    ),
  )
  manifest = CanonicalManifest(
    {
      "block_id": block_id,
      "entity_block_id": entity_block_id,
      "entity_ids": entity_block.entity_ids,
      "implementations": [
        {
          "kind": item.kind,
          "name": item.name,
          "version": item.version,
          "implementation_id": item.implementation_id,
        }
        for item in header.implementations
      ],
      "port": {
        "port_id": port.port_id,
        "space_id": port.space_id,
        "coefficient_map": port.coefficient_map.values,
      },
      "channels": ["internal-force", "material-tangent"],
      "state": {
        "schema": state_layout.schema,
        "row_width": 0,
        "entity_offsets": state_layout.entity_offsets.values,
      },
      "payload": {
        "quadrature_points": payload.quadrature_points.values,
        "quadrature_weights": payload.quadrature_weights.values,
        "shape_values": payload.shape_values.values,
        "parent_gradients": payload.parent_gradients.values,
        "geometry_scales": payload.geometry_scales.values,
        "normalized_gradients": payload.normalized_gradients.values,
        "normalized_strain_displacement": (
          payload.normalized_strain_displacement.values
        ),
        "normalized_integration_weights": (
          payload.normalized_integration_weights.values
        ),
        "constitutive": payload.constitutive.values,
        "material_parameters": payload.material_parameters.values,
      },
    }
  )
  return entity_block, _new(
    Q8ContinuumOperator,
    header=header,
    entity_block=entity_block,
    payload=payload,
    content_manifest=manifest,
  )
