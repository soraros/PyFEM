"""Concrete direct compiler and evaluator for the qualified truss slice."""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from typing import NoReturn

import numpy as np

from pyfem.v3.compile.diagnostics import (
  ModelCompilationDiagnostic,
  ModelCompilationError,
)
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

TRUSS_TOPOLOGY_KEY: RegistryKey = ("topology", "line2")
TRUSS_FORMULATION_KEY: RegistryKey = ("formulation", "total-lagrangian-truss")
TRUSS_MATERIAL_KEY: RegistryKey = ("material", "uniaxial-linear-elastic")
_REQUIRED_KEYS = (
  TRUSS_TOPOLOGY_KEY,
  TRUSS_FORMULATION_KEY,
  TRUSS_MATERIAL_KEY,
)
_PARAMETER_NAMES = ("youngs_modulus", "area")
_NODE_COUNT = 2
_MATERIAL_RELATIVE_TOLERANCE = 16.0 * float(np.finfo(np.float64).eps)


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
class TrussSelection:
  block: CellBlockSpec
  field: FieldSpec
  material: MaterialSpec
  region: RegionSpec
  cells: tuple[CellSpec, ...]


@dataclass(frozen=True, slots=True, eq=False, init=False)
class TrussPayload(CompilerConstructed):
  element_lengths: FinalizedArray
  element_rotations: FinalizedArray
  constitutive: FinalizedArray
  geometric_template: FinalizedArray
  material_parameters: FinalizedArray


def _evaluation_array(
  value: object,
  *,
  shape: tuple[int, ...],
  label: str,
) -> np.ndarray:
  if (
    type(value) is not np.ndarray
    or value.shape != shape
    or value.dtype.metadata is not None
    or value.dtype.kind not in "iuf"
  ):
    msg = f"{label} must return a metadata-free numeric array {shape!r}"
    raise TypeError(msg)
  captured = np.array(value, dtype=np.float64, order="C", copy=True, subok=False)
  if not bool(np.isfinite(captured).all()):
    msg = "truss evaluation response is not representable as finite float64"
    raise ValueError(msg)
  return captured


@dataclass(frozen=True, slots=True, eq=False, init=False)
class TrussOperator(CompilerConstructed):
  header: OperatorHeader
  entity_block: IncidenceEntityBlock
  payload: TrussPayload
  content_manifest: CanonicalManifest
  kinematics: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]

  def evaluate(
    self,
    inputs: OperatorEvaluationInput,
  ) -> OperatorEvaluation:
    """Evaluate local internal force and consistent tangent from compiled meaning."""
    if type(inputs) is not OperatorEvaluationInput:
      msg = "truss evaluation requires an exact immutable evaluation input"
      raise TypeError(msg)
    if type(inputs.port_values) is not tuple or len(inputs.port_values) != 1:
      msg = "truss evaluation requires exactly one displacement port batch"
      raise TypeError(msg)
    values = inputs.port_values[0].values
    expected = self.header.ports[0].coefficient_map.values.shape
    if (
      values.dtype != np.dtype(np.float64)
      or values.dtype.metadata is not None
      or values.shape != expected
      or not bool(np.isfinite(values).all())
    ):
      msg = "truss displacement values must be a finite metadata-free float64 batch"
      raise TypeError(msg)
    layout = self.header.state_layout
    accepted_state = inputs.accepted_state.values
    if (
      accepted_state.dtype != np.dtype(np.float64)
      or accepted_state.dtype.metadata is not None
      or accepted_state.shape != layout.row_shape
      or not bool(np.isfinite(accepted_state).all())
    ):
      msg = "truss accepted state must match the compiled zero-width state layout"
      raise TypeError(msg)
    if inputs.signals or self.header.signal_ports:
      msg = "truss model operator does not accept program signal inputs"
      raise ValueError(msg)
    residual_ids = tuple(item.channel_id for item in self.header.residual_channels)
    jacobian_ids = tuple(item.channel_id for item in self.header.jacobian_channels)
    derivative_ids = tuple(
      item.channel_id for item in getattr(self.header, "derivative_channels", ())
    )
    request = inputs.request
    derivative_request = request.derivative_channel_ids
    if (
      type(request) is not ChannelRequest
      or len(set(request.residual_channel_ids)) != len(request.residual_channel_ids)
      or len(set(request.jacobian_channel_ids)) != len(request.jacobian_channel_ids)
      or len(set(derivative_request)) != len(derivative_request)
      or not set(request.residual_channel_ids).issubset(residual_ids)
      or not set(request.jacobian_channel_ids).issubset(jacobian_ids)
      or not set(derivative_request).issubset(derivative_ids)
    ):
      msg = "truss evaluation request contains an unavailable or duplicate channel"
      raise ValueError(msg)

    rotations = self.payload.element_rotations.values
    lengths = self.payload.element_lengths.values
    local = np.einsum("eij,ej->ei", rotations, values, optimize=True)
    try:
      with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        raw_strains, raw_bl = self.kinematics(
          np.array(local, dtype=np.float64, order="C", copy=True),
          np.array(lengths, dtype=np.float64, order="C", copy=True),
        )
    except Exception:
      msg = "truss kinematics binding failed during evaluation"
      raise ValueError(msg) from None
    strains = _evaluation_array(
      raw_strains,
      shape=(len(lengths),),
      label="truss strain binding",
    )
    bl = _evaluation_array(
      raw_bl,
      shape=(len(lengths), _NODE_COUNT * 2),
      label="truss strain-displacement binding",
    )
    stiffness = float(self.payload.constitutive.values[0, 0])
    template = self.payload.geometric_template.values
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
      tangent_local = (stiffness * lengths)[:, None, None] * (
        bl[:, :, None] * bl[:, None, :]
      ) + (stiffness * strains / lengths)[:, None, None] * template[None, :, :]
      residual_local = (stiffness * lengths * strains)[:, None] * bl
      tangent = np.einsum(
        "eji,ejk,ekl->eil",
        rotations,
        tangent_local,
        rotations,
        optimize=True,
      )
      residual = np.einsum("eji,ej->ei", rotations, residual_local, optimize=True)
    if not bool(np.isfinite(tangent).all()) or not bool(np.isfinite(residual).all()):
      msg = "truss evaluation response is not representable as finite float64"
      raise ValueError(msg)
    residual_values = (
      (FinalizedArray(residual, dtype=np.float64),)
      if request.residual_channel_ids
      else ()
    )
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


def line2_geometry(coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """Reference two-node line geometry: lengths and global-to-local rotations."""
  delta = coordinates[:, 1, :] - coordinates[:, 0, :]
  lengths = np.hypot(delta[:, 0], delta[:, 1])
  rotations = np.empty((len(coordinates), 2, 2), dtype=np.float64)
  rotations[:, 0, 0] = delta[:, 0] / lengths
  rotations[:, 0, 1] = delta[:, 1] / lengths
  rotations[:, 1, 0] = -rotations[:, 0, 1]
  rotations[:, 1, 1] = rotations[:, 0, 0]
  return lengths, rotations


def total_lagrangian_truss(
  local_states: np.ndarray,
  lengths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  """Reference uniaxial Green-Lagrange kinematics: strains and BL rows."""
  axial = (local_states[:, 2] - local_states[:, 0]) / lengths
  transverse = (local_states[:, 3] - local_states[:, 1]) / lengths
  strains = axial + 0.5 * axial * axial + 0.5 * transverse * transverse
  bl = np.empty_like(local_states)
  bl[:, 0] = -(1.0 + axial) / lengths
  bl[:, 1] = -transverse / lengths
  bl[:, 2] = -bl[:, 0]
  bl[:, 3] = -bl[:, 1]
  return strains, bl


def uniaxial_linear_elastic(youngs_modulus: float, area: float) -> np.ndarray:
  """Reference uniaxial section law: axial stiffness as a 1x1 matrix."""
  return np.array([[youngs_modulus * area]], dtype=np.float64)


def truss_descriptor_metadata(kind: str, name: str) -> dict[str, object]:
  """Return the exact semantic metadata for one selected truss implementation."""
  key = kind, name
  if key == TRUSS_TOPOLOGY_KEY:
    return {
      "schema": "pyfem-v3-topology-descriptor-v1",
      "reference_topology": "line",
      "parent_dimension": 1,
      "embedding_dimension": 2,
      "node_count": 2,
      "parent_coordinates": ["xi"],
      "local_node_parent_coordinates": [[-1.0], [1.0]],
      "binding_outputs": ["element_lengths", "global_to_local_rotations"],
    }
  if key == TRUSS_FORMULATION_KEY:
    return {
      "schema": "pyfem-v3-formulation-descriptor-v1",
      "field_quantity": "displacement",
      "field_location": "node",
      "field_components": ["x", "y"],
      "dofs_per_node": 2,
      "kinematic_regime": "finite-displacement",
      "strain_measure": "green-lagrange-uniaxial",
      "reference_frame": "total-lagrangian",
      "formulation_history_width": 0,
      "tangent_contribution": "material-geometric",
      "tangent_symmetry": "symmetric",
    }
  if key == TRUSS_MATERIAL_KEY:
    return {
      "schema": "pyfem-v3-material-descriptor-v1",
      "law": "linear-elastic",
      "stress_state": "uniaxial",
      "parameter_names": ["youngs_modulus", "area"],
      "parameter_dtype": "float64",
      "material_history_width": 0,
      "tangent_class": "constant-symmetric",
    }
  msg = "no truss descriptor metadata exists for that exact registry key"
  raise KeyError(msg)


def truss_reference_registry() -> dict[RegistryKey, RegistryDescriptor]:
  """Build a fresh injectable registry for the qualified truss convention."""
  bindings = {
    TRUSS_TOPOLOGY_KEY: ("pyfem-v3-line2-geometry-v1", line2_geometry),
    TRUSS_FORMULATION_KEY: (
      "pyfem-v3-total-lagrangian-truss-v1",
      total_lagrangian_truss,
    ),
    TRUSS_MATERIAL_KEY: (
      "pyfem-v3-uniaxial-linear-elastic-v1",
      uniaxial_linear_elastic,
    ),
  }
  descriptors = tuple(
    RegistryDescriptor(
      kind=key[0],
      name=key[1],
      version="1",
      implementation_id=implementation_id,
      metadata=truss_descriptor_metadata(*key),
      binding=binding,
    )
    for key, (implementation_id, binding) in bindings.items()
  )
  return {descriptor.key: descriptor for descriptor in descriptors}


def select_model(spec: ModelSpec) -> TrussSelection:
  """Validate the concrete authored slice after the sole normalization pass."""
  for field in spec.fields:
    if field.location != "node":
      _fail(
        "unsupported-space-support",
        "the direct truss slice cannot allocate a non-node field",
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
      "unsupported-truss-declaration-count",
      "the direct truss slice requires one active field, cell block, material, "
      "and region",
      spec.source,
    )
  block = spec.mesh.cell_blocks[0]
  material = spec.materials[0]
  region = spec.regions[0]
  fields = {field.id: field for field in spec.fields}
  if len(region.field_ids) != 1 or region.field_ids[0] not in fields:
    _fail(
      "incompatible-region-field-signature",
      "the truss region must reference exactly one declared field",
      region.source,
    )
  field = fields[region.field_ids[0]]
  if (
    block.reference_topology != "line"
    or block.topological_dimension != 1
    or block.embedding_dimension != 2
    or block.geometry_interpolation != "line2"
  ):
    _fail(
      "incompatible-cell-block",
      "truss requires explicit one-dimensional two-node line geometry",
      block.source,
    )
  for cell in block.cells:
    if len(cell.node_ids) != _NODE_COUNT:
      _fail(
        "invalid-truss-arity",
        f"truss cell {render_diagnostic_value(cell.id)} must reference two nodes",
        cell.source,
      )
  if field.location != "node" or field.components != ("x", "y"):
    _fail(
      "incompatible-field-signature",
      "truss requires one node field with physical components ('x', 'y')",
      field.source,
    )
  if region.material_id != material.id:
    _fail(
      "incompatible-region-material",
      "the truss region must reference the selected material",
      region.source,
    )
  if region.formulation != TRUSS_FORMULATION_KEY[1]:
    _fail("incompatible-formulation", "unsupported truss formulation", region.source)
  if region.quadrature != "none":
    _fail(
      "incompatible-quadrature",
      "the direct truss slice integrates in closed form and requires 'none'",
      region.source,
    )
  if material.model != TRUSS_MATERIAL_KEY[1]:
    _fail("incompatible-material-model", "unsupported truss material", material.source)
  return TrussSelection(
    block=block,
    field=field,
    material=material,
    region=region,
    cells=tuple(sorted(block.cells, key=lambda item: _sort_key(item.id))),
  )


def capture_registry(
  registry: dict[RegistryKey, RegistryDescriptor],
  selection: TrussSelection,
) -> RegistrySnapshot:
  """Capture and validate exactly the implementations selected by this builder."""
  try:
    snapshot = RegistrySnapshot.capture(registry, required=_REQUIRED_KEYS)
  except (KeyError, TypeError, ValueError):
    _fail(
      "registry-capture-failed",
      "the injected registry could not capture the required truss implementations",
      selection.region.source,
    )
  sources = {
    TRUSS_TOPOLOGY_KEY: selection.block.source,
    TRUSS_FORMULATION_KEY: selection.region.source,
    TRUSS_MATERIAL_KEY: selection.material.source,
  }
  for key in _REQUIRED_KEYS:
    try:
      descriptor = snapshot.resolve(*key)
      expected = CanonicalManifest(truss_descriptor_metadata(*key))
      compatible = descriptor.metadata.to_bytes() == expected.to_bytes()
    except (KeyError, TypeError, ValueError):
      _fail(
        "malformed-registry-descriptor",
        "malformed truss descriptor",
        sources[key],
      )
    if not compatible:
      _fail(
        "incompatible-registry-descriptor",
        "truss descriptor metadata does not match the qualified convention",
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
  scale = float(np.max(np.abs(expected)))
  zero_tolerance = 8.0 * float(np.finfo(np.float64).eps) * scale
  ulps = 16.0 * np.abs(np.spacing(expected))
  tolerance = np.where(expected == 0.0, zero_tolerance, ulps)
  return bool(np.all(np.abs(actual - expected) <= tolerance))


def _parameters(selection: TrussSelection) -> tuple[float, float]:
  by_name = {item.name: item for item in selection.material.parameters}
  if tuple(sorted(by_name)) != tuple(sorted(_PARAMETER_NAMES)):
    _fail(
      "invalid-material-parameter-schema",
      "uniaxial truss material requires youngs_modulus and area",
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
  youngs_modulus, area = values
  if youngs_modulus <= 0.0:
    _fail(
      "invalid-youngs-modulus",
      "youngs_modulus must be positive",
      by_name[_PARAMETER_NAMES[0]].source,
    )
  if area <= 0.0:
    _fail(
      "invalid-cross-section-area",
      "area must be positive",
      by_name[_PARAMETER_NAMES[1]].source,
    )
  return youngs_modulus, area


def _geometry(
  coordinates: np.ndarray,
  cells: tuple[CellSpec, ...],
  tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
  lengths = np.empty(len(cells), dtype=np.float64)
  rotations = np.empty((len(cells), 2, 2), dtype=np.float64)
  for index, cell in enumerate(cells):
    with np.errstate(over="ignore", invalid="ignore", under="ignore", divide="ignore"):
      delta = coordinates[index, 1] - coordinates[index, 0]
      length = float(np.hypot(delta[0], delta[1]))
      scale = float(
        max(
          np.max(np.abs(coordinates[index, 0])),
          np.max(np.abs(coordinates[index, 1])),
        )
      )
      reciprocal = float(np.divide(1.0, length))
    if (
      not math.isfinite(length)
      or not math.isfinite(scale)
      or length <= tolerance * scale
      or not math.isfinite(reciprocal)
    ):
      _fail(
        "degenerate-truss-geometry",
        f"truss cell {render_diagnostic_value(cell.id)} has a non-finite, zero, "
        "or scale-unresolvable element length",
        cell.source,
      )
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
      cos_alpha = float(delta[0] / length)
      sin_alpha = float(delta[1] / length)
    if not math.isfinite(cos_alpha) or not math.isfinite(sin_alpha):
      _fail(
        "degenerate-truss-geometry",
        f"truss cell {render_diagnostic_value(cell.id)} has a non-finite "
        "element direction",
        cell.source,
      )
    lengths[index] = length
    rotations[index, 0, 0] = cos_alpha
    rotations[index, 0, 1] = sin_alpha
    rotations[index, 1, 0] = -sin_alpha
    rotations[index, 1, 1] = cos_alpha
  return lengths, rotations


def _kinematics_probes(lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  element_count = len(lengths)
  direction = np.array([1.0, -0.5, 2.0, 0.25], dtype=np.float64)
  probes = np.zeros((3 * element_count, 4), dtype=np.float64)
  for scale_index, factor in enumerate((0.05, 0.3), start=1):
    for element_index in range(element_count):
      probes[scale_index * element_count + element_index] = (
        factor * (element_index + 1) * lengths[element_index] * direction
      )
  return probes, np.concatenate((lengths, lengths, lengths))


def _expected_kinematics(
  local_states: np.ndarray,
  lengths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  axial = (local_states[:, 2] - local_states[:, 0]) / lengths
  transverse = (local_states[:, 3] - local_states[:, 1]) / lengths
  strains = axial + 0.5 * axial * axial + 0.5 * transverse * transverse
  bl = np.empty_like(local_states)
  bl[:, 0] = -(1.0 + axial) / lengths
  bl[:, 1] = -transverse / lengths
  bl[:, 2] = -bl[:, 0]
  bl[:, 3] = -bl[:, 1]
  return strains, bl


def _qualified_constitutive(
  youngs_modulus: float,
  area: float,
  source: SourceContext,
) -> tuple[np.ndarray, np.ndarray]:
  try:
    exact_value = float(Fraction.from_float(youngs_modulus) * Fraction.from_float(area))
    if not math.isfinite(exact_value):
      raise OverflowError
  except (ArithmeticError, OverflowError, ValueError):
    _fail(
      "unrepresentable-material-law",
      "truss axial stiffness cannot be represented as finite float64",
      source,
    )
  with np.errstate(over="ignore", invalid="ignore"):
    binary64_value = youngs_modulus * area
  if not math.isfinite(binary64_value):
    binary64_value = exact_value
  exact = np.array([[exact_value]], dtype=np.float64)
  binary64_route = np.array([[binary64_value]], dtype=np.float64)
  return exact, binary64_route


def _outside_relative_envelope(actual: float, first: float, second: float) -> bool:
  lower, upper = sorted((first, second))
  return not (
    lower <= actual <= upper
    or any(
      math.isclose(
        actual,
        boundary,
        rel_tol=_MATERIAL_RELATIVE_TOLERANCE,
        abs_tol=0.0,
      )
      for boundary in (lower, upper)
    )
  )


def _constitutive_corresponds(
  actual: np.ndarray,
  exact: np.ndarray,
  binary64_route: np.ndarray,
) -> bool:
  return not any(
    _outside_relative_envelope(float(value), float(qualified), float(route))
    for value, qualified, route in zip(
      actual.flat,
      exact.flat,
      binary64_route.flat,
      strict=True,
    )
  )


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
  selection: TrussSelection,
  *,
  coordinates: FinalizedArray,
  node_dense: dict[SpecId, int],
  space: DiscreteSpace,
  snapshot: RegistrySnapshot,
  index_dtype: np.dtype,
  geometry_relative_tolerance: float,
) -> tuple[IncidenceEntityBlock, TrussOperator]:
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
  cell_coordinates = np.array(
    coordinates.values[connectivity.values],
    dtype=np.float64,
    order="C",
    copy=True,
  )
  expected_lengths, expected_rotations = _geometry(
    cell_coordinates,
    selection.cells,
    geometry_relative_tolerance,
  )
  topology = snapshot.resolve(*TRUSS_TOPOLOGY_KEY).binding
  try:
    raw_geometry = topology(cell_coordinates)
  except Exception:
    _fail(
      "topology-binding-failed",
      "truss topology binding failed",
      selection.block.source,
    )
  if type(raw_geometry) is not tuple or len(raw_geometry) != 2:
    _fail(
      "invalid-topology-binding-output",
      "truss topology must return exactly lengths and rotations",
      selection.block.source,
    )
  lengths = _binding_array(
    raw_geometry[0],
    shape=(len(selection.cells),),
    code="invalid-topology-binding-output",
    label="truss element lengths",
    source=selection.block.source,
  )
  rotations = _binding_array(
    raw_geometry[1],
    shape=(len(selection.cells), 2, 2),
    code="invalid-topology-binding-output",
    label="truss element rotations",
    source=selection.block.source,
  )
  if not _corresponds(lengths, expected_lengths) or not _corresponds(
    rotations,
    expected_rotations,
  ):
    _fail(
      "incompatible-topology-binding-output",
      "truss topology output contradicts the qualified line geometry",
      selection.block.source,
    )
  probe_states, probe_lengths = _kinematics_probes(lengths)
  formulation = snapshot.resolve(*TRUSS_FORMULATION_KEY).binding
  try:
    raw_strains, raw_bl = formulation(probe_states, probe_lengths)
  except Exception:
    _fail(
      "formulation-binding-failed",
      "truss formulation binding failed",
      selection.region.source,
    )
  strains = _binding_array(
    raw_strains,
    shape=(len(probe_states),),
    code="invalid-formulation-binding-output",
    label="truss strain binding",
    source=selection.region.source,
  )
  bl = _binding_array(
    raw_bl,
    shape=(len(probe_states), _NODE_COUNT * 2),
    code="invalid-formulation-binding-output",
    label="truss strain-displacement binding",
    source=selection.region.source,
  )
  expected_strains, expected_bl = _expected_kinematics(probe_states, probe_lengths)
  if not _corresponds(strains, expected_strains) or not _corresponds(bl, expected_bl):
    _fail(
      "incompatible-formulation-binding-output",
      "truss formulation output contradicts the qualified Green-Lagrange map",
      selection.region.source,
    )
  youngs_modulus, area = _parameters(selection)
  exact_constitutive, binary64_route = _qualified_constitutive(
    youngs_modulus,
    area,
    selection.material.source,
  )
  material = snapshot.resolve(*TRUSS_MATERIAL_KEY).binding
  try:
    with warnings.catch_warnings():
      warnings.simplefilter("error", RuntimeWarning)
      raw_constitutive = material(youngs_modulus, area)
  except Exception:
    _fail(
      "material-binding-failed",
      "truss material binding failed",
      selection.material.source,
    )
  constitutive = _binding_array(
    raw_constitutive,
    shape=(1, 1),
    code="invalid-material-binding-output",
    label="truss material binding",
    source=selection.material.source,
  )
  if not _constitutive_corresponds(
    constitutive,
    exact_constitutive,
    binary64_route,
  ):
    _fail(
      "incompatible-material-binding-output",
      "truss material output contradicts the qualified uniaxial law",
      selection.material.source,
    )
  with np.errstate(over="ignore", invalid="ignore", under="ignore"):
    tangent_scale = constitutive[0, 0] * lengths
    geometric_scale = constitutive[0, 0] / lengths
  if not bool(np.isfinite(tangent_scale).all()) or not bool(
    np.isfinite(geometric_scale).all()
  ):
    _fail(
      "non-finite-element-operator",
      "truss element operator scale is non-finite",
      selection.region.source,
    )

  rotations4 = np.zeros((len(selection.cells), 4, 4), dtype=np.float64)
  rotations4[:, :2, :2] = rotations
  rotations4[:, 2:, 2:] = rotations
  geometric_template = np.array(
    [
      [1.0, 0.0, -1.0, 0.0],
      [0.0, 1.0, 0.0, -1.0],
      [-1.0, 0.0, 1.0, 0.0],
      [0.0, -1.0, 0.0, 1.0],
    ],
    dtype=np.float64,
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
      np.zeros(len(selection.cells) + 1),
      dtype=index_dtype,
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
    linear=False,
  )
  jacobian_channel = _new(
    JacobianChannel,
    channel_id="tangent",
    residual_channel_id="internal-force",
    target_port_id=port.port_id,
    source_port_id=port.port_id,
    balance_role=BalanceRole.INTERNAL,
    linear=False,
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
    TrussPayload,
    element_lengths=FinalizedArray(lengths, dtype=np.float64),
    element_rotations=FinalizedArray(rotations4, dtype=np.float64),
    constitutive=FinalizedArray(constitutive, dtype=np.float64),
    geometric_template=FinalizedArray(geometric_template, dtype=np.float64),
    material_parameters=FinalizedArray(
      [[youngs_modulus, area]],
      dtype=np.float64,
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
      "channels": ["internal-force", "tangent"],
      "state": {
        "schema": state_layout.schema,
        "row_width": 0,
        "entity_offsets": state_layout.entity_offsets.values,
      },
      "payload": {
        "element_lengths": payload.element_lengths.values,
        "element_rotations": payload.element_rotations.values,
        "constitutive": payload.constitutive.values,
        "geometric_template": payload.geometric_template.values,
        "material_parameters": payload.material_parameters.values,
      },
    }
  )
  return entity_block, _new(
    TrussOperator,
    header=header,
    entity_block=entity_block,
    payload=payload,
    content_manifest=manifest,
    kinematics=formulation,
  )
