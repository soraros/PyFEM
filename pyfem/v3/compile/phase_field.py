"""Fresh Q4 AT2 spectral-split phase-field operator and compiler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NoReturn, Protocol

import numpy as np

from pyfem.v3.compile.contracts import (
  PHASE_FIELD_FORMULATION_KEY,
  PHASE_FIELD_MATERIAL_KEY,
  QUAD4_QUADRATURE_KEY,
  QUAD4_TOPOLOGY_KEY,
  continuum_reference_registry,
  phase_field_descriptor_metadata,
)
from pyfem.v3.compile.diagnostics import (
  ModelCompilationDiagnostic,
  ModelCompilationError,
)
from pyfem.v3.fem.kinematics import strain_displacement
from pyfem.v3.fem.phase_field import (
  PhaseFieldPreparationError,
  spectral_at2_response,
)
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
  OperatorStateSlot,
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

PHASE_FIELD_STATE_SCHEMA = "pyfem-v3-q4-at2-history-v1"
_POINT_COUNT = 4


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


class PhaseFieldRegionSelection(Protocol):
  block: CellBlockSpec
  fields: tuple[FieldSpec, ...]
  material: MaterialSpec
  region: RegionSpec
  cells: tuple[CellSpec, ...]


@dataclass(frozen=True, slots=True, eq=False, init=False)
class Q4AT2Payload(CompilerConstructed):
  shape_values: FinalizedArray
  physical_gradients: FinalizedArray
  physical_integration_weights: FinalizedArray
  strain_displacement: FinalizedArray
  constitutive: FinalizedArray
  lame_lambda: float
  shear_modulus: float
  critical_energy_release_rate: float
  length_scale: float
  residual_stiffness: float


@dataclass(frozen=True, slots=True, eq=False, init=False)
class Q4AT2PhaseFieldOperator(CompilerConstructed):
  header: OperatorHeader
  entity_block: IncidenceEntityBlock
  payload: Q4AT2Payload
  content_manifest: CanonicalManifest

  def evaluate(self, inputs: OperatorEvaluationInput) -> OperatorEvaluation:
    """Evaluate both residuals and all branch-consistent rectangular Jacobians."""
    if type(inputs) is not OperatorEvaluationInput:
      msg = "Q4 AT2 evaluation requires an exact evaluation input"
      raise TypeError(msg)
    if len(inputs.port_values) != 2:
      msg = "Q4 AT2 evaluation requires displacement and phase ports"
      raise TypeError(msg)
    displacement = inputs.port_values[0].values
    phase = inputs.port_values[1].values
    entity_count = self.entity_block.incidence.values.shape[0]
    if (
      displacement.dtype != np.dtype(np.float64)
      or displacement.dtype.metadata is not None
      or displacement.shape != (entity_count, 8)
      or phase.dtype != np.dtype(np.float64)
      or phase.dtype.metadata is not None
      or phase.shape != (entity_count, 4)
      or not bool(np.isfinite(displacement).all())
      or not bool(np.isfinite(phase).all())
    ):
      msg = "Q4 AT2 ports must be finite metadata-free float64 batches"
      raise TypeError(msg)
    layout = self.header.state_layout
    accepted = inputs.accepted_state.values
    if (
      accepted.dtype != np.dtype(np.float64)
      or accepted.dtype.metadata is not None
      or accepted.shape != layout.row_shape
      or not bool(np.isfinite(accepted).all())
      or bool(np.any(accepted < 0.0))
    ):
      msg = "Q4 AT2 accepted history must be finite, nonnegative, and match its layout"
      raise ValueError(msg)
    if (
      inputs.signals or self.header.signal_ports or inputs.nested_execution is not None
    ):
      msg = "Q4 AT2 evaluation accepts neither signals nor nested execution"
      raise ValueError(msg)
    request = inputs.request
    if type(request) is not ChannelRequest:
      msg = "Q4 AT2 evaluation requires an exact channel request"
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
      msg = "Q4 AT2 request contains an unavailable or duplicate channel"
      raise ValueError(msg)
    if bool(np.any(phase < 0.0)) or bool(np.any(phase > 1.0)):
      return _new(
        OperatorEvaluation,
        residual_values=(),
        jacobian_values=(),
        trial_state=FinalizedArray(
          np.array(accepted, dtype=np.float64, order="C", copy=True),
          dtype=np.float64,
        ),
        status=EvaluationStatus.REJECT_ITERATION,
        derivative_values=(),
        nested_references=(),
      )

    payload = self.payload
    b_matrix = payload.strain_displacement.values
    shape = payload.shape_values.values
    gradients = payload.physical_gradients.values
    weights = payload.physical_integration_weights.values
    strains = np.einsum("epai,ei->epa", b_matrix, displacement, optimize=True)
    response = spectral_at2_response(
      strains.reshape(-1, 3),
      payload.lame_lambda,
      payload.shear_modulus,
      tangent=bool(request.jacobian_channel_ids),
    )
    positive_energy = response.positive_energy.reshape(entity_count, _POINT_COUNT)
    accepted_history = accepted[:, 0].reshape(entity_count, _POINT_COUNT)
    if request.jacobian_channel_ids and bool(
      np.any(positive_energy == accepted_history)
    ):
      raise PhaseFieldPreparationError(
        "phase-field-history-kink",
        "branch-consistent AT2 derivatives are undefined when psi+ equals accepted H",
      )
    active = positive_energy > accepted_history
    trial = np.array(accepted, dtype=np.float64, order="C", copy=True)
    trial[:, 0] = np.maximum(accepted[:, 0], response.positive_energy)
    history = trial[:, 0].reshape(entity_count, _POINT_COUNT)
    phase_ip = np.einsum("pn,en->ep", shape, phase, optimize=True)
    phase_gradient = np.einsum("epni,en->epi", gradients, phase, optimize=True)
    degradation = (1.0 - phase_ip) ** 2 + payload.residual_stiffness
    positive_stress = response.positive_stress.reshape(entity_count, _POINT_COUNT, 3)
    negative_stress = response.negative_stress.reshape(entity_count, _POINT_COUNT, 3)
    stress = degradation[:, :, None] * positive_stress + negative_stress
    gc = payload.critical_energy_release_rate
    ell = payload.length_scale

    residual_candidates: tuple[np.ndarray | None, np.ndarray | None] = (None, None)
    if request.residual_channel_ids:
      mechanical = np.einsum(
        "ep,epai,epa->ei",
        weights,
        b_matrix,
        stress,
        optimize=True,
      )
      local_phase = -2.0 * (1.0 - phase_ip) * history + gc * phase_ip / ell
      phase_residual = np.einsum(
        "ep,pn->en", weights * local_phase, shape, optimize=True
      ) + gc * ell * np.einsum(
        "ep,epni,epi->en",
        weights,
        gradients,
        phase_gradient,
        optimize=True,
      )
      residual_candidates = mechanical, phase_residual

    jacobian_candidates: tuple[
      np.ndarray | None,
      np.ndarray | None,
      np.ndarray | None,
      np.ndarray | None,
    ] = (None, None, None, None)
    if request.jacobian_channel_ids:
      assert response.positive_tangent is not None
      assert response.negative_tangent is not None
      positive_tangent = response.positive_tangent.reshape(
        entity_count, _POINT_COUNT, 3, 3
      )
      negative_tangent = response.negative_tangent.reshape(
        entity_count, _POINT_COUNT, 3, 3
      )
      mechanical_tangent = degradation[:, :, None, None] * positive_tangent
      mechanical_tangent += negative_tangent
      tangent_uu = np.einsum(
        "ep,epai,epab,epbj->eij",
        weights,
        b_matrix,
        mechanical_tangent,
        b_matrix,
        optimize=True,
      )
      coupling_factor = -2.0 * weights * (1.0 - phase_ip)
      tangent_ud = np.einsum(
        "ep,epai,epa,pn->ein",
        coupling_factor,
        b_matrix,
        positive_stress,
        shape,
        optimize=True,
      )
      tangent_du = np.einsum(
        "ep,pn,epa,epai->eni",
        coupling_factor * active,
        shape,
        positive_stress,
        b_matrix,
        optimize=True,
      )
      phase_factor = weights * (2.0 * history + gc / ell)
      tangent_dd = np.einsum(
        "ep,pn,pm->enm", phase_factor, shape, shape, optimize=True
      ) + gc * ell * np.einsum(
        "ep,epni,epmi->enm",
        weights,
        gradients,
        gradients,
        optimize=True,
      )
      jacobian_candidates = tangent_uu, tangent_ud, tangent_du, tangent_dd

    blocks = (*residual_candidates, *jacobian_candidates)
    if any(
      block is not None and not bool(np.isfinite(block).all()) for block in blocks
    ):
      msg = "Q4 AT2 evaluation left the finite float64 range"
      raise ValueError(msg)
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
      trial_state=FinalizedArray(trial, dtype=np.float64),
      status=EvaluationStatus.OK,
      derivative_values=(),
      nested_references=(),
    )


def phase_field_reference_registry() -> dict[RegistryKey, RegistryDescriptor]:
  """Build the Q4 continuum registry plus fresh AT2 descriptors."""
  registry = continuum_reference_registry()
  descriptors = (
    RegistryDescriptor(
      kind=PHASE_FIELD_FORMULATION_KEY[0],
      name=PHASE_FIELD_FORMULATION_KEY[1],
      version="1",
      implementation_id="pyfem-v3-at2-spectral-split-quad4-v1",
      metadata=phase_field_descriptor_metadata(*PHASE_FIELD_FORMULATION_KEY),
      binding=strain_displacement,
    ),
    RegistryDescriptor(
      kind=PHASE_FIELD_MATERIAL_KEY[0],
      name=PHASE_FIELD_MATERIAL_KEY[1],
      version="1",
      implementation_id="pyfem-v3-plane-strain-at2-spectral-split-v1",
      metadata=phase_field_descriptor_metadata(*PHASE_FIELD_MATERIAL_KEY),
      binding=plane_strain_matrix,
    ),
  )
  for descriptor in descriptors:
    registry[descriptor.key] = descriptor
  return registry


def _parameters(material: MaterialSpec) -> tuple[float, float, float, float, float]:
  values = {parameter.name: parameter.value for parameter in material.parameters}
  if "initial_crack" in values or "initial_crack_state" in values:
    _fail(
      "unsupported-initial-crack-state",
      "the bounded Q4 AT2 formulation supports virgin zero history only; "
      "initial crack states require a separate formulation",
      material.source,
    )
  expected = {
    "youngs_modulus",
    "poisson_ratio",
    "critical_energy_release_rate",
    "length_scale",
    "residual_stiffness",
  }
  if set(values) != expected:
    _fail(
      "incompatible-material-parameters",
      "plane-strain AT2 requires youngs_modulus, poisson_ratio, "
      "critical_energy_release_rate, length_scale, and residual_stiffness",
      material.source,
    )
  converted: dict[str, float] = {}
  for name in sorted(expected):
    value = values[name]
    if type(value) is not int and type(value) is not float:
      _fail(
        "incompatible-material-parameters",
        f"AT2 parameter {name} must be an exact int or float",
        material.source,
      )
    converted[name] = float(value)
  if not all(np.isfinite(tuple(converted.values()))):
    _fail(
      "incompatible-material-parameters",
      "AT2 parameters must be finite",
      material.source,
    )
  youngs = converted["youngs_modulus"]
  poisson = converted["poisson_ratio"]
  gc = converted["critical_energy_release_rate"]
  ell = converted["length_scale"]
  eta = converted["residual_stiffness"]
  if youngs <= 0.0 or not -1.0 < poisson < 0.5:
    _fail(
      "incompatible-material-parameters",
      "AT2 Young modulus must be positive and Poisson ratio in (-1, 0.5)",
      material.source,
    )
  if gc <= 0.0 or ell <= 0.0 or eta <= 0.0:
    _fail(
      "incompatible-material-parameters",
      "AT2 fracture energy, length scale, and residual stiffness must be positive",
      material.source,
    )
  return youngs, poisson, gc, ell, eta


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


def _geometry(
  coordinates: np.ndarray,
  connectivity: np.ndarray,
  parent_gradients: np.ndarray,
  quadrature_weights: np.ndarray,
  tolerance: float,
  selection: PhaseFieldRegionSelection,
) -> tuple[np.ndarray, np.ndarray]:
  nodes = coordinates[connectivity]
  relative = nodes - nodes[:, 0:1, :]
  scales = np.max(np.abs(relative), axis=(1, 2))
  if not bool(np.isfinite(scales).all()) or bool(np.any(scales <= 0.0)):
    _fail(
      "non-finite-reference-geometry",
      "Q4 AT2 requires finite nonzero-scale reference geometry",
      selection.block.source,
    )
  normalized = relative / scales[:, None, None]
  jacobians = np.einsum("enk,pnj->epkj", normalized, parent_gradients, optimize=True)
  determinants = (
    jacobians[:, :, 0, 0] * jacobians[:, :, 1, 1]
    - jacobians[:, :, 0, 1] * jacobians[:, :, 1, 0]
  )
  norm_squares = np.sum(jacobians * jacobians, axis=(2, 3))
  if (
    not bool(np.isfinite(determinants).all())
    or not bool(np.isfinite(norm_squares).all())
    or bool(np.any(norm_squares == 0.0))
    or bool(np.any(determinants <= tolerance))
    or bool(np.any(determinants / norm_squares <= tolerance))
  ):
    _fail(
      "near-singular-reference-geometry",
      "Q4 AT2 requires positive scale-nonsingular reference geometry",
      selection.region.source,
    )
  inverses = np.empty_like(jacobians)
  inverses[:, :, 0, 0] = jacobians[:, :, 1, 1]
  inverses[:, :, 0, 1] = -jacobians[:, :, 0, 1]
  inverses[:, :, 1, 0] = -jacobians[:, :, 1, 0]
  inverses[:, :, 1, 1] = jacobians[:, :, 0, 0]
  inverses /= determinants[:, :, None, None]
  normalized_gradients = parent_gradients[None, :, :, :] @ inverses
  gradients = normalized_gradients / scales[:, None, None, None]
  integration_weights = (
    determinants * quadrature_weights[None, :] * scales[:, None] ** 2
  )
  if not bool(np.isfinite(gradients).all()) or not bool(
    np.isfinite(integration_weights).all()
  ):
    _fail(
      "non-finite-reference-geometry",
      "Q4 AT2 physical gradients or integration weights are non-finite",
      selection.region.source,
    )
  return gradients, integration_weights


def compile_q4_at2_phase_field_operator(
  selection: PhaseFieldRegionSelection,
  *,
  coordinates: FinalizedArray,
  node_dense: dict[SpecId, int],
  spaces: dict[SpecId, DiscreteSpace],
  snapshot: RegistrySnapshot,
  index_dtype: np.dtype,
  geometry_relative_tolerance: float,
) -> tuple[IncidenceEntityBlock, Q4AT2PhaseFieldOperator]:
  """Compile one homogeneous batch of Q4 AT2 spectral phase-field elements."""
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
  topology = snapshot.resolve(*QUAD4_TOPOLOGY_KEY).binding
  quadrature = snapshot.resolve(*QUAD4_QUADRATURE_KEY).binding
  formulation = snapshot.resolve(*PHASE_FIELD_FORMULATION_KEY).binding
  material_binding = snapshot.resolve(*PHASE_FIELD_MATERIAL_KEY).binding
  points, quadrature_weights = quadrature(2)
  shape_values, parent_gradients = topology(points)
  if (
    points.shape != (_POINT_COUNT, 2)
    or quadrature_weights.shape != (_POINT_COUNT,)
    or shape_values.shape != (_POINT_COUNT, 4)
    or parent_gradients.shape != (_POINT_COUNT, 4, 2)
    or bool(np.any(quadrature_weights <= 0.0))
    or not np.isclose(quadrature_weights.sum(), 4.0, rtol=0.0, atol=1.0e-14)
    or not np.allclose(shape_values.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-14)
    or not np.allclose(parent_gradients.sum(axis=1), 0.0, rtol=0.0, atol=1.0e-14)
  ):
    _fail(
      "invalid-phase-field-recipe",
      "the Q4 AT2 topology and 2x2 quadrature bindings are invalid",
      selection.region.source,
    )
  gradients, integration_weights = _geometry(
    coordinates.values,
    connectivity,
    parent_gradients,
    quadrature_weights,
    geometry_relative_tolerance,
    selection,
  )
  raw_b = formulation(np.array(gradients, dtype=np.float64, copy=True))
  b_matrix = np.asarray(raw_b, dtype=np.float64)
  expected_b = np.zeros((len(selection.cells), _POINT_COUNT, 3, 8), dtype=np.float64)
  expected_b[:, :, 0, 0::2] = gradients[:, :, :, 0]
  expected_b[:, :, 1, 1::2] = gradients[:, :, :, 1]
  expected_b[:, :, 2, 0::2] = gradients[:, :, :, 1]
  expected_b[:, :, 2, 1::2] = gradients[:, :, :, 0]
  if b_matrix.shape != expected_b.shape or not np.array_equal(b_matrix, expected_b):
    _fail(
      "incompatible-formulation-binding-output",
      "the AT2 formulation binding contradicts engineering strain",
      selection.region.source,
    )
  youngs, poisson, gc, ell, eta = _parameters(selection.material)
  constitutive = np.asarray(material_binding(youngs, poisson), dtype=np.float64)
  lame_lambda = youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
  shear_modulus = youngs / (2.0 * (1.0 + poisson))
  expected_c = np.array(
    [
      [lame_lambda + 2.0 * shear_modulus, lame_lambda, 0.0],
      [lame_lambda, lame_lambda + 2.0 * shear_modulus, 0.0],
      [0.0, 0.0, shear_modulus],
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
      "the AT2 material binding contradicts plane-strain elasticity",
      selection.material.source,
    )

  displacement_space = spaces[selection.fields[0].id]
  phase_space = spaces[selection.fields[1].id]
  displacement_gather = np.array(
    _gather_map(displacement_space, connectivity, node_dense).reshape(
      len(selection.cells), 8
    ),
    dtype=index_dtype,
    order="C",
    copy=True,
  )
  phase_gather = np.array(
    _gather_map(phase_space, connectivity, node_dense).reshape(len(selection.cells), 4),
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
  phase_port = _new(
    PortBinding,
    port_id="d",
    space_id=phase_space.space_id,
    mode=PortMode.COEFFICIENTS,
    coefficient_map=FinalizedArray(phase_gather, dtype=index_dtype),
  )
  mechanical_residual = _new(
    ResidualChannel,
    channel_id="internal-u",
    target_port_id="u",
    balance_role=BalanceRole.INTERNAL,
    linear=False,
  )
  phase_residual = _new(
    ResidualChannel,
    channel_id="internal-d",
    target_port_id="d",
    balance_role=BalanceRole.INTERNAL,
    linear=False,
  )
  jacobian_data = (
    ("tangent-uu", mechanical_residual, displacement_port, displacement_port, True),
    ("tangent-ud", mechanical_residual, displacement_port, phase_port, False),
    ("tangent-du", phase_residual, phase_port, displacement_port, False),
    ("tangent-dd", phase_residual, phase_port, phase_port, True),
  )
  jacobian_channels = tuple(
    _new(
      JacobianChannel,
      channel_id=channel_id,
      residual_channel_id=residual.channel_id,
      target_port_id=target.port_id,
      source_port_id=source.port_id,
      balance_role=BalanceRole.INTERNAL,
      linear=False,
      symmetric=symmetric,
    )
    for channel_id, residual, target, source, symmetric in jacobian_data
  )
  block_id = selection.block.id, selection.region.id
  state_count = len(selection.cells) * _POINT_COUNT
  history_slot = _new(
    OperatorStateSlot,
    name="history",
    width=1,
    dtype=np.dtype(np.float64).str,
    lifetime=StateLifetime.ACCEPTED_TRIAL,
    annotation="monotone-nondecreasing",
  )
  state_layout = _new(
    OperatorStateLayout,
    schema=PHASE_FIELD_STATE_SCHEMA,
    block_id=block_id,
    entity_count=state_count,
    slots=(history_slot,),
    entity_offsets=FinalizedArray(
      np.arange(state_count + 1, dtype=index_dtype), dtype=index_dtype
    ),
    row_width=1,
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
    ports=(displacement_port, phase_port),
    signal_ports=(),
    residual_channels=(mechanical_residual, phase_residual),
    jacobian_channels=jacobian_channels,
    state_layout=state_layout,
    coupling_policy=CouplingPolicy.FIXED,
    parameters=(),
    derivative_channels=(),
    nested_execution=None,
    tangent_class="branch-consistent-at2-spectral",
  )
  payload = _new(
    Q4AT2Payload,
    shape_values=FinalizedArray(shape_values, dtype=np.float64),
    physical_gradients=FinalizedArray(gradients, dtype=np.float64),
    physical_integration_weights=FinalizedArray(integration_weights, dtype=np.float64),
    strain_displacement=FinalizedArray(b_matrix, dtype=np.float64),
    constitutive=FinalizedArray(constitutive, dtype=np.float64),
    lame_lambda=lame_lambda,
    shear_modulus=shear_modulus,
    critical_energy_release_rate=gc,
    length_scale=ell,
    residual_stiffness=eta,
  )
  manifest = CanonicalManifest(
    {
      "schema": "pyfem-v3-at2-spectral-split-quad4-v1",
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
      "state_layout": {
        "schema": state_layout.schema,
        "entity_count": state_layout.entity_count,
        "row_width": state_layout.row_width,
        "slot_annotation": history_slot.annotation,
      },
      "payload": {
        "shape_values": payload.shape_values.values,
        "physical_gradients": payload.physical_gradients.values,
        "physical_integration_weights": payload.physical_integration_weights.values,
        "strain_displacement": payload.strain_displacement.values,
        "constitutive": payload.constitutive.values,
        "lame_lambda": payload.lame_lambda,
        "shear_modulus": payload.shear_modulus,
        "critical_energy_release_rate": payload.critical_energy_release_rate,
        "length_scale": payload.length_scale,
        "residual_stiffness": payload.residual_stiffness,
      },
    }
  )
  return entity_block, _new(
    Q4AT2PhaseFieldOperator,
    header=header,
    entity_block=entity_block,
    payload=payload,
    content_manifest=manifest,
  )
