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
  CouplingPolicy,
  ImplementationIdentity,
  JacobianChannel,
  OperatorEvaluation,
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
  return CompiledSource(value.source, value.line, value.column)


@dataclass(frozen=True, slots=True, eq=False)
class ContinuumSelection:
  """Validated authored declarations consumed only by this concrete builder."""

  block: CellBlockSpec
  field: FieldSpec
  material: MaterialSpec
  region: RegionSpec
  cells: tuple[CellSpec, ...]


@dataclass(frozen=True, slots=True, eq=False)
class Q8ContinuumPayload:
  """Builder-owned immutable numerical recipe for the concrete evaluator."""

  quadrature_points: FinalizedArray
  quadrature_weights: FinalizedArray
  shape_values: FinalizedArray
  parent_gradients: FinalizedArray
  physical_gradients: FinalizedArray
  strain_displacement: FinalizedArray
  integration_weights: FinalizedArray
  constitutive: FinalizedArray
  material_parameters: FinalizedArray


@dataclass(frozen=True, slots=True, eq=False)
class Q8ContinuumOperator:
  """Concrete payload/evaluator satisfying the open compiled-operator protocol."""

  header: OperatorHeader
  entity_block: IncidenceEntityBlock
  payload: Q8ContinuumPayload
  content_manifest: CanonicalManifest

  def evaluate(
    self,
    port_values: tuple[np.ndarray, ...],
    accepted_state: np.ndarray,
  ) -> OperatorEvaluation:
    """Evaluate local internal force and material tangent from compiled meaning."""
    if type(port_values) is not tuple or len(port_values) != 1:
      msg = "Q8 evaluation requires exactly one displacement port batch"
      raise TypeError(msg)
    values = port_values[0]
    expected = self.header.ports[0].coefficient_map.values.shape
    if (
      type(values) is not np.ndarray
      or values.dtype != np.dtype(np.float64)
      or values.dtype.metadata is not None
      or values.shape != expected
      or not bool(np.isfinite(values).all())
    ):
      msg = "Q8 displacement port values must be a finite metadata-free float64 batch"
      raise TypeError(msg)
    layout = self.header.state_layout
    if (
      type(accepted_state) is not np.ndarray
      or accepted_state.dtype != np.dtype(np.float64)
      or accepted_state.dtype.metadata is not None
      or accepted_state.shape != layout.row_shape
      or not bool(np.isfinite(accepted_state).all())
    ):
      msg = "Q8 accepted state must match the compiled zero-width state layout"
      raise TypeError(msg)

    b_matrix = self.payload.strain_displacement.values
    weights = self.payload.integration_weights.values
    constitutive = self.payload.constitutive.values
    tangent = np.einsum(
      "ep,epai,ab,epbj->eij",
      weights,
      b_matrix,
      constitutive,
      b_matrix,
      optimize=True,
    )
    residual = np.einsum("eij,ej->ei", tangent, values, optimize=True)
    return OperatorEvaluation(
      residual_values=(FinalizedArray(residual, dtype=np.float64),),
      jacobian_values=(FinalizedArray(tangent, dtype=np.float64),),
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
    or len(spec.materials) != 1
    or len(spec.regions) != 1
  ):
    _fail(
      "unsupported-continuum-declaration-count",
      "the direct Q8 slice requires one cell block, material, and region",
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
  if (
    bool(np.any(weights <= 0.0))
    or bool(np.any(np.abs(points) > 1.0))
    or not math.isclose(float(weights.sum()), 4.0, rel_tol=1.0e-14, abs_tol=1.0e-14)
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
  if not bool(
    np.allclose(shape_values.sum(1), 1.0, rtol=0.0, atol=1.0e-12)
  ) or not bool(np.allclose(parent_gradients.sum(1), 0.0, rtol=0.0, atol=1.0e-12)):
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


def _normalized_coordinates(coordinates: np.ndarray, cell: CellSpec) -> np.ndarray:
  with np.errstate(over="ignore", invalid="ignore", under="ignore"):
    relative = coordinates - coordinates[0]
  if not bool(np.isfinite(relative).all()):
    scale = float(np.max(np.abs(coordinates)))
    if not math.isfinite(scale) or scale == 0.0:
      _fail(
        "non-finite-reference-geometry",
        "Q8 has non-finite or zero-scale geometry",
        cell.source,
      )
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
      relative = coordinates / scale - coordinates[0] / scale
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
  return normalized


def _geometry(
  coordinates: np.ndarray,
  connectivity: np.ndarray,
  parent_gradients: np.ndarray,
  cells: tuple[CellSpec, ...],
  tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
  gradients = np.empty((len(cells), _POINT_COUNT, _NODE_COUNT, 2), dtype=np.float64)
  determinants = np.empty((len(cells), _POINT_COUNT), dtype=np.float64)
  for cell_index, cell in enumerate(cells):
    normalized = _normalized_coordinates(coordinates[connectivity[cell_index]], cell)
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
  return gradients, determinants


def _identities(snapshot: RegistrySnapshot) -> tuple[ImplementationIdentity, ...]:
  return tuple(
    ImplementationIdentity(
      descriptor.kind,
      descriptor.name,
      descriptor.version,
      descriptor.implementation_id,
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
  entity_block = IncidenceEntityBlock(
    block_id=entity_block_id,
    entity_ids=tuple(cell.id for cell in selection.cells),
    sources=tuple(_source(cell.source) for cell in selection.cells),
    incidence=connectivity,
  )
  points, weights, shape_values, parent_gradients = _recipes(snapshot, selection)
  gradients, determinants = _geometry(
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
  if not bool(np.array_equal(constitutive, constitutive.T)):
    _fail(
      "nonsymmetric-material-binding",
      "Q8 material tangent must be symmetric",
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
  state_layout = OperatorStateLayout(
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
  port = PortBinding("displacement", space.space_id, PortMode.COEFFICIENTS, gather)
  header = OperatorHeader(
    block_id=block_id,
    entity_block_id=entity_block_id,
    implementations=_identities(snapshot),
    ports=(port,),
    residual_channels=(
      ResidualChannel("internal-force", port.port_id, BalanceRole.INTERNAL, True),
    ),
    jacobian_channels=(
      JacobianChannel(
        "material-tangent",
        "internal-force",
        port.port_id,
        port.port_id,
        BalanceRole.INTERNAL,
        True,
        True,
      ),
    ),
    state_layout=state_layout,
    coupling_policy=CouplingPolicy.FIXED,
  )
  payload = Q8ContinuumPayload(
    quadrature_points=FinalizedArray(points, dtype=np.float64),
    quadrature_weights=FinalizedArray(weights, dtype=np.float64),
    shape_values=FinalizedArray(shape_values, dtype=np.float64),
    parent_gradients=FinalizedArray(parent_gradients, dtype=np.float64),
    physical_gradients=FinalizedArray(gradients, dtype=np.float64),
    strain_displacement=FinalizedArray(b_matrix, dtype=np.float64),
    integration_weights=FinalizedArray(integration_weights, dtype=np.float64),
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
        "physical_gradients": payload.physical_gradients.values,
        "strain_displacement": payload.strain_displacement.values,
        "integration_weights": payload.integration_weights.values,
        "constitutive": payload.constitutive.values,
        "material_parameters": payload.material_parameters.values,
      },
    }
  )
  return entity_block, Q8ContinuumOperator(header, entity_block, payload, manifest)
