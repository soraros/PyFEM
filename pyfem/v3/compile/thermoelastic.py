"""Fresh steady one-point T3 thermoelastic operator and compiler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NoReturn, Protocol

import numpy as np

from pyfem.v3.compile.contracts import (
  THERMOELASTIC_FORMULATION_KEY,
  THERMOELASTIC_MATERIAL_KEY,
  TRIA3_QUADRATURE_KEY,
  TRIA3_TOPOLOGY_KEY,
  continuum_reference_registry,
  thermoelastic_descriptor_metadata,
)
from pyfem.v3.compile.diagnostics import (
  ModelCompilationDiagnostic,
  ModelCompilationError,
)
from pyfem.v3.fem.kinematics import strain_displacement
from pyfem.v3.materials.plane_strain import plane_strain_matrix
from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.operator import (
  BalanceRole,
  ChannelRequest,
  CompilerConstructed,
  CouplingPolicy,
  EvaluationStatus,
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
from pyfem.v3.spec.diagnostics import SourceContext
from pyfem.v3.spec.model import (
  CellBlockSpec,
  CellSpec,
  FieldSpec,
  MaterialSpec,
  RegionSpec,
  SpecId,
)


def _new[ValueT](cls: type[ValueT], /, **fields: object) -> ValueT:
  value = object.__new__(cls)
  for name, field in fields.items():
    object.__setattr__(value, name, field)
  return value


def _fail(code: str, message: str, source: SourceContext) -> NoReturn:
  raise ModelCompilationError(
    (ModelCompilationDiagnostic(code=code, message=message, source=source),)
  )


def _source(value: SourceContext) -> CompiledSource:
  return _new(
    CompiledSource,
    source=value.source,
    line=value.line,
    column=value.column,
  )


class ThermoRegionSelection(Protocol):
  block: CellBlockSpec
  fields: tuple[FieldSpec, ...]
  material: MaterialSpec
  region: RegionSpec
  cells: tuple[CellSpec, ...]


@dataclass(frozen=True, slots=True, eq=False, init=False)
class T3ThermoElasticPayload(CompilerConstructed):
  shape_values: FinalizedArray
  physical_gradients: FinalizedArray
  integration_weights: FinalizedArray
  strain_displacement: FinalizedArray
  constitutive: FinalizedArray
  mechanical_tangent: FinalizedArray
  mechanical_thermal_tangent: FinalizedArray
  thermal_tangent: FinalizedArray
  thermal_expansion_force: FinalizedArray
  reference_temperature: float
  thermal_expansion: float
  conductivity: float


@dataclass(frozen=True, slots=True, eq=False, init=False)
class T3ThermoElasticOperator(CompilerConstructed):
  header: OperatorHeader
  entity_block: IncidenceEntityBlock
  payload: T3ThermoElasticPayload
  content_manifest: CanonicalManifest

  def evaluate(self, inputs: OperatorEvaluationInput) -> OperatorEvaluation:
    """Evaluate both steady residuals and all present rectangular Jacobians."""
    if type(inputs) is not OperatorEvaluationInput:
      msg = "T3 thermoelastic evaluation requires an exact evaluation input"
      raise TypeError(msg)
    if len(inputs.port_values) != 2:
      msg = "T3 thermoelastic evaluation requires displacement and temperature ports"
      raise TypeError(msg)
    displacement = inputs.port_values[0].values
    temperature = inputs.port_values[1].values
    entity_count = self.entity_block.incidence.values.shape[0]
    if (
      displacement.dtype != np.dtype(np.float64)
      or displacement.dtype.metadata is not None
      or displacement.shape != (entity_count, 6)
      or temperature.dtype != np.dtype(np.float64)
      or temperature.dtype.metadata is not None
      or temperature.shape != (entity_count, 3)
      or not bool(np.isfinite(displacement).all())
      or not bool(np.isfinite(temperature).all())
    ):
      msg = "T3 thermoelastic ports must be finite metadata-free float64 batches"
      raise TypeError(msg)
    layout = self.header.state_layout
    accepted = inputs.accepted_state.values
    if (
      accepted.dtype != np.dtype(np.float64)
      or accepted.dtype.metadata is not None
      or accepted.shape != layout.row_shape
      or not bool(np.isfinite(accepted).all())
    ):
      msg = "T3 thermoelastic accepted state must match its zero-width layout"
      raise TypeError(msg)
    if (
      inputs.signals or self.header.signal_ports or inputs.nested_execution is not None
    ):
      msg = "T3 thermoelastic evaluation accepts neither signals nor nested execution"
      raise ValueError(msg)
    request = inputs.request
    if type(request) is not ChannelRequest:
      msg = "T3 thermoelastic evaluation requires an exact channel request"
      raise TypeError(msg)
    residual_ids = tuple(item.channel_id for item in self.header.residual_channels)
    jacobian_ids = tuple(item.channel_id for item in self.header.jacobian_channels)
    if (
      len(set(request.residual_channel_ids)) != len(request.residual_channel_ids)
      or len(set(request.jacobian_channel_ids)) != len(request.jacobian_channel_ids)
      or request.derivative_channel_ids
      or not set(request.residual_channel_ids).issubset(residual_ids)
      or not set(request.jacobian_channel_ids).issubset(jacobian_ids)
    ):
      msg = "T3 thermoelastic request contains an unavailable or duplicate channel"
      raise ValueError(msg)

    payload = self.payload
    if request.residual_channel_ids:
      average_temperature = temperature @ payload.shape_values.values
      mechanical = (
        np.einsum(
          "eij,ej->ei",
          payload.mechanical_tangent.values,
          displacement,
        )
        + payload.thermal_expansion_force.values
        * (payload.reference_temperature - average_temperature)[:, None]
      )
      thermal = np.einsum(
        "eij,ej->ei",
        payload.thermal_tangent.values,
        temperature,
      )
      if not bool(np.isfinite(mechanical).all()) or not bool(
        np.isfinite(thermal).all()
      ):
        msg = "T3 thermoelastic residual evaluation left the finite float64 range"
        raise ValueError(msg)
    else:
      mechanical = np.empty((entity_count, 6), dtype=np.float64)
      thermal = np.empty((entity_count, 3), dtype=np.float64)
    residual_candidates = (mechanical, thermal)
    residual_values = tuple(
      FinalizedArray(
        np.array(value, dtype=np.float64, order="C", copy=True),
        dtype=np.float64,
      )
      for channel, value in zip(
        self.header.residual_channels,
        residual_candidates,
        strict=True,
      )
      if channel.channel_id in request.residual_channel_ids
    )
    jacobian_candidates = (
      payload.mechanical_tangent.values,
      payload.mechanical_thermal_tangent.values,
      payload.thermal_tangent.values,
    )
    jacobian_values = tuple(
      FinalizedArray(
        np.array(value, dtype=np.float64, order="C", copy=True),
        dtype=np.float64,
      )
      for channel, value in zip(
        self.header.jacobian_channels,
        jacobian_candidates,
        strict=True,
      )
      if channel.channel_id in request.jacobian_channel_ids
    )
    return _new(
      OperatorEvaluation,
      residual_values=residual_values,
      jacobian_values=jacobian_values,
      trial_state=FinalizedArray(
        np.array(accepted, dtype=np.float64, order="C", copy=True),
        dtype=np.float64,
      ),
      status=EvaluationStatus.OK,
      derivative_values=(),
      nested_references=(),
    )


def thermoelastic_reference_registry() -> dict[RegistryKey, RegistryDescriptor]:
  """Build the qualified T3 registry plus fresh coupled descriptors."""
  registry = continuum_reference_registry()
  descriptors = (
    RegistryDescriptor(
      kind=THERMOELASTIC_FORMULATION_KEY[0],
      name=THERMOELASTIC_FORMULATION_KEY[1],
      version="1",
      implementation_id="pyfem-v3-steady-thermoelastic-tria3-v1",
      metadata=thermoelastic_descriptor_metadata(*THERMOELASTIC_FORMULATION_KEY),
      binding=strain_displacement,
    ),
    RegistryDescriptor(
      kind=THERMOELASTIC_MATERIAL_KEY[0],
      name=THERMOELASTIC_MATERIAL_KEY[1],
      version="1",
      implementation_id="pyfem-v3-plane-strain-thermoelastic-v1",
      metadata=thermoelastic_descriptor_metadata(*THERMOELASTIC_MATERIAL_KEY),
      binding=plane_strain_matrix,
    ),
  )
  for descriptor in descriptors:
    registry[descriptor.key] = descriptor
  return registry


def _parameters(material: MaterialSpec) -> tuple[float, float, float, float, float]:
  expected = {
    "youngs_modulus",
    "poisson_ratio",
    "thermal_expansion",
    "conductivity",
    "reference_temperature",
  }
  values = {parameter.name: parameter.value for parameter in material.parameters}
  if set(values) != expected:
    _fail(
      "incompatible-material-parameters",
      "plane-strain thermoelastic requires youngs_modulus, poisson_ratio, "
      "thermal_expansion, conductivity, and reference_temperature",
      material.source,
    )
  converted: dict[str, float] = {}
  for name in sorted(expected):
    value = values[name]
    if type(value) is not int and type(value) is not float:
      _fail(
        "incompatible-material-parameters",
        f"thermoelastic parameter {name} must be an exact int or float",
        material.source,
      )
    converted[name] = float(value)
  youngs = converted["youngs_modulus"]
  poisson = converted["poisson_ratio"]
  expansion = converted["thermal_expansion"]
  conductivity = converted["conductivity"]
  reference = converted["reference_temperature"]
  if not all(np.isfinite(tuple(converted.values()))):
    _fail(
      "incompatible-material-parameters",
      "thermoelastic parameters must be finite",
      material.source,
    )
  if youngs <= 0.0 or not -1.0 < poisson < 0.5:
    _fail(
      "incompatible-material-parameters",
      "thermoelastic Young modulus must be positive and Poisson ratio in (-1, 0.5)",
      material.source,
    )
  if expansion < 0.0 or conductivity <= 0.0:
    _fail(
      "incompatible-material-parameters",
      "thermal_expansion must be nonnegative and conductivity positive",
      material.source,
    )
  return youngs, poisson, expansion, conductivity, reference


def _gather_map(
  space: DiscreteSpace,
  connectivity: np.ndarray,
  node_dense: dict[SpecId, int],
) -> np.ndarray:
  coefficient_map = space.coefficient_map.values
  if coefficient_map.shape[0] == len(node_dense):
    return coefficient_map[connectivity]
  component_count = len(space.components)
  support_rows = {
    node_dense[point_id]: row
    for row, point_id in enumerate(
      space.coefficient_ids[index][1]
      for index in range(0, len(space.coefficient_ids), component_count)
    )
  }
  rows = np.fromiter(
    (support_rows[int(dense)] for dense in connectivity.flat),
    dtype=connectivity.dtype,
    count=connectivity.size,
  ).reshape(connectivity.shape)
  return coefficient_map[rows]


def compile_t3_thermoelastic_operator(
  selection: ThermoRegionSelection,
  *,
  coordinates: FinalizedArray,
  node_dense: dict[SpecId, int],
  spaces: dict[SpecId, DiscreteSpace],
  snapshot: RegistrySnapshot,
  index_dtype: np.dtype,
  geometry_relative_tolerance: float,
) -> tuple[IncidenceEntityBlock, T3ThermoElasticOperator]:
  """Compile one homogeneous batch of steady one-point T3 elements."""
  connectivity_values = [
    [node_dense[node_id] for node_id in cell.node_ids] for cell in selection.cells
  ]
  connectivity = np.asarray(connectivity_values, dtype=index_dtype)
  entity_block = _new(
    IncidenceEntityBlock,
    block_id=selection.block.id,
    entity_ids=tuple(cell.id for cell in selection.cells),
    sources=tuple(_source(cell.source) for cell in selection.cells),
    incidence=FinalizedArray(connectivity, dtype=index_dtype),
  )
  topology = snapshot.resolve(*TRIA3_TOPOLOGY_KEY).binding
  quadrature = snapshot.resolve(*TRIA3_QUADRATURE_KEY).binding
  formulation = snapshot.resolve(*THERMOELASTIC_FORMULATION_KEY).binding
  material_binding = snapshot.resolve(*THERMOELASTIC_MATERIAL_KEY).binding
  points, weights = quadrature(1)
  shape_table, parent_gradients = topology(points)
  if (
    points.shape != (1, 2)
    or weights.shape != (1,)
    or shape_table.shape != (1, 3)
    or parent_gradients.shape != (1, 3, 2)
  ):
    _fail(
      "invalid-thermoelastic-recipe",
      "the T3 topology and one-point quadrature bindings returned invalid shapes",
      selection.region.source,
    )
  shape_values = np.array(shape_table[0], dtype=np.float64, copy=True)
  parent_gradient = np.array(parent_gradients[0], dtype=np.float64, copy=True)
  nodes = coordinates.values[connectivity]
  normalized = nodes - nodes[:, 0:1, :]
  jacobians = normalized.transpose(0, 2, 1) @ parent_gradient
  determinants = (
    jacobians[:, 0, 0] * jacobians[:, 1, 1] - jacobians[:, 0, 1] * jacobians[:, 1, 0]
  )
  norm_squares = np.sum(jacobians * jacobians, axis=(1, 2))
  if (
    not bool(np.isfinite(determinants).all())
    or not bool(np.isfinite(norm_squares).all())
    or np.any(norm_squares == 0.0)
    or np.any(determinants <= geometry_relative_tolerance)
    or np.any(determinants / norm_squares <= geometry_relative_tolerance)
  ):
    _fail(
      "near-singular-reference-geometry",
      "steady T3 thermoelastic requires positive scale-nonsingular reference geometry",
      selection.region.source,
    )
  inverses = np.empty_like(jacobians)
  inverses[:, 0, 0] = jacobians[:, 1, 1]
  inverses[:, 0, 1] = -jacobians[:, 0, 1]
  inverses[:, 1, 0] = -jacobians[:, 1, 0]
  inverses[:, 1, 1] = jacobians[:, 0, 0]
  inverses /= determinants[:, None, None]
  gradients = parent_gradient[None, :, :] @ inverses
  integration_weights = determinants * weights[0]
  raw_b = formulation(gradients[:, None, :, :])
  b_matrix = np.asarray(raw_b[:, 0], dtype=np.float64)
  expected_b = np.zeros((len(selection.cells), 3, 6), dtype=np.float64)
  expected_b[:, 0, 0::2] = gradients[:, :, 0]
  expected_b[:, 1, 1::2] = gradients[:, :, 1]
  expected_b[:, 2, 0::2] = gradients[:, :, 1]
  expected_b[:, 2, 1::2] = gradients[:, :, 0]
  if not np.array_equal(b_matrix, expected_b):
    _fail(
      "incompatible-formulation-binding-output",
      "the thermoelastic formulation binding contradicts engineering strain",
      selection.region.source,
    )
  youngs, poisson, expansion, conductivity, reference = _parameters(selection.material)
  constitutive = np.asarray(
    material_binding(youngs, poisson),
    dtype=np.float64,
  )
  lame = youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
  shear = youngs / (2.0 * (1.0 + poisson))
  expected_c = np.array(
    [
      [lame + 2.0 * shear, lame, 0.0],
      [lame, lame + 2.0 * shear, 0.0],
      [0.0, 0.0, shear],
    ],
    dtype=np.float64,
  )
  if constitutive.shape != (3, 3) or not np.allclose(
    constitutive,
    expected_c,
    rtol=2.0e-14,
    atol=2.0e-14,
  ):
    _fail(
      "incompatible-material-binding-output",
      "the thermoelastic material binding contradicts plane-strain elasticity",
      selection.material.source,
    )
  mechanical_tangent = integration_weights[:, None, None] * np.einsum(
    "eai,ab,ebj->eij",
    b_matrix,
    constitutive,
    b_matrix,
  )
  expansion_force = (
    integration_weights[:, None]
    * expansion
    * np.einsum("eai,ab,b->ei", b_matrix, constitutive, np.array([1.0, 1.0, 0.0]))
  )
  mechanical_thermal = -expansion_force[:, :, None] * shape_values[None, None, :]
  thermal_tangent = (
    integration_weights[:, None, None]
    * conductivity
    * np.einsum("eic,ejc->eij", gradients, gradients)
  )
  for label, array in (
    ("mechanical tangent", mechanical_tangent),
    ("thermal coupling", mechanical_thermal),
    ("thermal tangent", thermal_tangent),
  ):
    if not bool(np.isfinite(array).all()):
      _fail(
        "non-finite-element-operator",
        f"steady T3 {label} is non-finite",
        selection.region.source,
      )

  displacement_space = spaces[selection.fields[0].id]
  temperature_space = spaces[selection.fields[1].id]
  displacement_gather = _gather_map(displacement_space, connectivity, node_dense)
  if displacement_space.components == ("y", "x"):
    displacement_gather = displacement_gather[..., ::-1]
  displacement_gather = np.array(
    displacement_gather.reshape(len(selection.cells), 6),
    dtype=index_dtype,
    order="C",
    copy=True,
  )
  temperature_gather = np.array(
    _gather_map(temperature_space, connectivity, node_dense).reshape(
      len(selection.cells),
      3,
    ),
    dtype=index_dtype,
    order="C",
    copy=True,
  )
  displacement_port = _new(
    PortBinding,
    port_id="u",
    space_id=displacement_space.space_id,
    mode=PortMode.COEFFICIENTS,
    coefficient_map=FinalizedArray(displacement_gather, dtype=index_dtype),
  )
  temperature_port = _new(
    PortBinding,
    port_id="T",
    space_id=temperature_space.space_id,
    mode=PortMode.COEFFICIENTS,
    coefficient_map=FinalizedArray(temperature_gather, dtype=index_dtype),
  )
  mechanical_residual = _new(
    ResidualChannel,
    channel_id="internal-u",
    target_port_id="u",
    balance_role=BalanceRole.INTERNAL,
    linear=True,
  )
  thermal_residual = _new(
    ResidualChannel,
    channel_id="internal-T",
    target_port_id="T",
    balance_role=BalanceRole.INTERNAL,
    linear=True,
  )
  jacobians_by_name = (
    ("tangent-uu", mechanical_residual, displacement_port, displacement_port, True),
    ("tangent-uT", mechanical_residual, displacement_port, temperature_port, False),
    ("tangent-TT", thermal_residual, temperature_port, temperature_port, True),
  )
  jacobian_channels = tuple(
    _new(
      JacobianChannel,
      channel_id=channel_id,
      residual_channel_id=residual.channel_id,
      target_port_id=target.port_id,
      source_port_id=source.port_id,
      balance_role=BalanceRole.INTERNAL,
      linear=True,
      symmetric=symmetric,
    )
    for channel_id, residual, target, source, symmetric in jacobians_by_name
  )
  block_id = selection.block.id, selection.region.id
  state_layout = _new(
    OperatorStateLayout,
    schema="pyfem-v3-operator-state-layout-v1",
    block_id=block_id,
    entity_count=len(selection.cells),
    slots=(),
    entity_offsets=FinalizedArray(
      np.zeros(len(selection.cells) + 1, dtype=index_dtype),
      dtype=index_dtype,
    ),
    row_width=0,
    dtype=np.dtype(np.float64).str,
    lifetime=StateLifetime.ACCEPTED_TRIAL,
    initial_rows=None,
  )
  header = _new(
    OperatorHeader,
    block_id=block_id,
    entity_block_id=entity_block.block_id,
    implementations=tuple(
      _new(
        ImplementationIdentity,
        kind=descriptor.kind,
        name=descriptor.name,
        version=descriptor.version,
        implementation_id=descriptor.implementation_id,
      )
      for descriptor in snapshot.descriptors
    ),
    ports=(displacement_port, temperature_port),
    signal_ports=(),
    residual_channels=(mechanical_residual, thermal_residual),
    jacobian_channels=jacobian_channels,
    state_layout=state_layout,
    coupling_policy=CouplingPolicy.FIXED,
    parameters=(),
    derivative_channels=(),
    nested_execution=None,
    tangent_class=None,
  )
  payload = _new(
    T3ThermoElasticPayload,
    shape_values=FinalizedArray(shape_values, dtype=np.float64),
    physical_gradients=FinalizedArray(gradients, dtype=np.float64),
    integration_weights=FinalizedArray(integration_weights, dtype=np.float64),
    strain_displacement=FinalizedArray(b_matrix, dtype=np.float64),
    constitutive=FinalizedArray(constitutive, dtype=np.float64),
    mechanical_tangent=FinalizedArray(mechanical_tangent, dtype=np.float64),
    mechanical_thermal_tangent=FinalizedArray(mechanical_thermal, dtype=np.float64),
    thermal_tangent=FinalizedArray(thermal_tangent, dtype=np.float64),
    thermal_expansion_force=FinalizedArray(expansion_force, dtype=np.float64),
    reference_temperature=reference,
    thermal_expansion=expansion,
    conductivity=conductivity,
  )
  manifest = CanonicalManifest(
    {
      "schema": "pyfem-v3-steady-thermoelastic-tria3-v1",
      "block_id": block_id,
      "entity_block_id": entity_block.block_id,
      "entity_ids": entity_block.entity_ids,
      "ports": [
        {
          "port_id": port.port_id,
          "space_id": port.space_id,
          "coefficient_map": port.coefficient_map.values,
        }
        for port in header.ports
      ],
      "residual_channels": [
        {
          "channel_id": channel.channel_id,
          "target_port_id": channel.target_port_id,
          "balance_role": channel.balance_role.value,
          "linear": channel.linear,
        }
        for channel in header.residual_channels
      ],
      "jacobian_channels": [
        {
          "channel_id": channel.channel_id,
          "residual_channel_id": channel.residual_channel_id,
          "target_port_id": channel.target_port_id,
          "source_port_id": channel.source_port_id,
          "balance_role": channel.balance_role.value,
          "linear": channel.linear,
          "symmetric": channel.symmetric,
        }
        for channel in header.jacobian_channels
      ],
      "payload": {
        "shape_values": payload.shape_values.values,
        "physical_gradients": payload.physical_gradients.values,
        "integration_weights": payload.integration_weights.values,
        "strain_displacement": payload.strain_displacement.values,
        "constitutive": payload.constitutive.values,
        "mechanical_tangent": payload.mechanical_tangent.values,
        "mechanical_thermal_tangent": payload.mechanical_thermal_tangent.values,
        "thermal_tangent": payload.thermal_tangent.values,
        "thermal_expansion_force": payload.thermal_expansion_force.values,
        "reference_temperature": payload.reference_temperature,
        "thermal_expansion": payload.thermal_expansion,
        "conductivity": payload.conductivity,
      },
    }
  )
  return entity_block, _new(
    T3ThermoElasticOperator,
    header=header,
    entity_block=entity_block,
    payload=payload,
    content_manifest=manifest,
  )
