"""Typed executable semantic-IR prototype for three unlike operators.

This module is intentionally isolated from the frozen Phase-1 carrier.  Factory
functions validate and detach inputs once; assembly trusts their immutable output
and dispatches only through evaluator bindings captured on homogeneous blocks.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from pyfem.v3.fem.shapes import bilinear_quad4, linear_tria3
from pyfem.v3.materials.plane_stress import plane_stress_matrix
from pyfem.v3.model.arrays import FinalizedArray

type FloatArray = NDArray[np.float64]
type IndexArray = NDArray[np.int64]
type ShapeBinding = Callable[[FloatArray], tuple[FloatArray, FloatArray]]


class EntityRole(Enum):
  """Geometric role of one native-width entity block."""

  CELL = "cell"
  LINK = "link"
  POINT_LOAD = "point-load"


class OperatorOwner(Enum):
  """Lifetime owner of a compiled operator block."""

  MODEL = "model"
  PROGRAM = "program"


class BalanceRole(Enum):
  """Attribution role of a residual contribution."""

  INTERNAL = "internal"
  EXTERNAL = "external"


@dataclass(frozen=True, slots=True, eq=False)
class PointEntityBlock:
  """Compiler-owned points and reference coordinates."""

  block_id: str
  entity_ids: tuple[str, ...]
  source_ids: tuple[str, ...]
  reference_coordinates: FinalizedArray


@dataclass(frozen=True, slots=True, eq=False)
class IncidenceEntityBlock:
  """Compiler-owned homogeneous entities with their native incidence width."""

  block_id: str
  role: EntityRole
  entity_ids: tuple[str, ...]
  source_ids: tuple[str, ...]
  incidence: FinalizedArray


@dataclass(frozen=True, slots=True, eq=False)
class DiscreteSpace:
  """One explicit coefficient space over a point support."""

  space_id: str
  support: PointEntityBlock
  components: tuple[str, ...]
  coefficient_ids: tuple[str, ...]
  coefficient_count: int


@dataclass(frozen=True, slots=True, eq=False)
class PortBinding:
  """A named operator port and its block-local gather map."""

  port_id: str
  space: DiscreteSpace
  gather: FinalizedArray


@dataclass(frozen=True, slots=True)
class ResidualChannel:
  """Typed attributed residual-vector channel."""

  channel_id: str
  target_port: int
  balance_role: BalanceRole
  multiplier: float


@dataclass(frozen=True, slots=True)
class JacobianChannel:
  """Typed derivative channel between explicit ports."""

  channel_id: str
  target_port: int
  source_port: int
  balance_role: BalanceRole
  multiplier: float
  symmetric: bool


@dataclass(frozen=True, slots=True)
class LocalStateLayout:
  """Block-local accepted/trial row metadata; width is zero in this proof."""

  schema_id: str
  entity_count: int
  width: int


@dataclass(frozen=True, slots=True, eq=False)
class OperatorHeader:
  """Small common header shared by every specialized operator payload."""

  block_id: str
  entity_block: IncidenceEntityBlock
  descriptor_id: str
  implementation_id: str
  owner: OperatorOwner
  ports: tuple[PortBinding, ...]
  residual_channels: tuple[ResidualChannel, ...]
  jacobian_channels: tuple[JacobianChannel, ...]
  state_layout: LocalStateLayout


@dataclass(frozen=True, slots=True, eq=False)
class ContinuumPayload:
  """Compiled plane-stress integration data for a homogeneous cell block."""

  strain_displacement: FinalizedArray
  integration_weights: FinalizedArray
  constitutive: FinalizedArray


@dataclass(frozen=True, slots=True, eq=False)
class DirectionalSpringPayload:
  """One scalar directional stiffness per link entity."""

  stiffness: FinalizedArray


@dataclass(frozen=True, slots=True, eq=False)
class PointLoadPayload:
  """One program-owned scalar load magnitude per point entity."""

  magnitudes: FinalizedArray


@dataclass(frozen=True, slots=True, eq=False)
class LocalEvaluation:
  """Channel-ordered local values returned through the generic boundary."""

  residual_batches: tuple[FinalizedArray, ...]
  jacobian_batches: tuple[FinalizedArray, ...]


@dataclass(frozen=True, slots=True, eq=False)
class EvaluatorBinding[PayloadT]:
  """Exact evaluator implementation captured by the compiler."""

  implementation_id: str
  evaluate: Callable[
    [PayloadT, tuple[FloatArray, ...]],
    LocalEvaluation,
  ]


@dataclass(frozen=True, slots=True, eq=False)
class OperatorBlock[PayloadT]:
  """One homogeneous header, specialized payload, and captured evaluator."""

  header: OperatorHeader
  payload: PayloadT
  evaluator: EvaluatorBinding[PayloadT]


type AnyOperatorBlock = (
  OperatorBlock[ContinuumPayload]
  | OperatorBlock[DirectionalSpringPayload]
  | OperatorBlock[PointLoadPayload]
)


@dataclass(frozen=True, slots=True, eq=False)
class ContinuumDescriptor:
  """Compiler input selecting one explicit interpolation/quadrature recipe."""

  descriptor_id: str
  implementation_id: str
  node_count: int
  quadrature_points: FinalizedArray
  quadrature_weights: FinalizedArray
  shape_binding: ShapeBinding


@dataclass(frozen=True, slots=True, eq=False)
class CompiledSystem:
  """Model-owned points, one space, and model operator blocks."""

  space: DiscreteSpace
  operators: tuple[AnyOperatorBlock, ...]


@dataclass(frozen=True, slots=True, eq=False)
class CompiledProgram:
  """Program-owned operators bound to the exact model space."""

  compatible_space: DiscreteSpace
  operators: tuple[OperatorBlock[PointLoadPayload], ...]


@dataclass(frozen=True, slots=True, eq=False)
class PreparedExecution:
  """Canonical generic schedule over model- and program-owned blocks."""

  space: DiscreteSpace
  operators: tuple[AnyOperatorBlock, ...]


@dataclass(frozen=True, slots=True, eq=False)
class AttributedResidual:
  """One signed residual contribution with exact compiled provenance."""

  operator_id: str
  channel: ResidualChannel
  entity_id: str
  source_id: str
  target_dofs: FinalizedArray
  values: FinalizedArray


@dataclass(frozen=True, slots=True, eq=False)
class AttributedJacobian:
  """One signed operator contribution with exact compiled provenance."""

  operator_id: str
  channel: JacobianChannel
  entity_id: str
  source_id: str
  target_dofs: FinalizedArray
  source_dofs: FinalizedArray
  values: FinalizedArray


@dataclass(frozen=True, slots=True, eq=False)
class AssemblyResult:
  """Additive dense proof result plus its complete attribution ledger."""

  residual: FinalizedArray
  jacobian: FinalizedArray
  residual_terms: tuple[AttributedResidual, ...]
  jacobian_terms: tuple[AttributedJacobian, ...]
  evaluated_block_ids: tuple[str, ...]


def _text(value: str, label: str) -> str:
  if type(value) is not str or not value:
    msg = f"{label} must be an exact nonempty string"
    raise TypeError(msg)
  return value


def _texts(values: tuple[str, ...], count: int, label: str) -> tuple[str, ...]:
  if type(values) is not tuple or len(values) != count:
    msg = f"{label} must be an exact tuple of length {count}"
    raise TypeError(msg)
  checked = tuple(_text(value, label) for value in values)
  if len(set(checked)) != len(checked):
    msg = f"{label} must be unique"
    raise ValueError(msg)
  return checked


def _owned_float(
  source: FloatArray,
  *,
  ndim: int,
  label: str,
) -> FinalizedArray:
  if (
    type(source) is not np.ndarray
    or source.dtype != np.dtype(np.float64)
    or source.dtype.metadata is not None
    or source.ndim != ndim
  ):
    msg = f"{label} must be an exact metadata-free float64 array of rank {ndim}"
    raise TypeError(msg)
  if not np.all(np.isfinite(source)):
    msg = f"{label} must contain finite values"
    raise ValueError(msg)
  return FinalizedArray(source, dtype=np.float64)


def _owned_indices(
  source: IndexArray,
  *,
  ndim: int,
  label: str,
) -> FinalizedArray:
  if (
    type(source) is not np.ndarray
    or source.dtype != np.dtype(np.int64)
    or source.dtype.metadata is not None
    or source.ndim != ndim
  ):
    msg = f"{label} must be an exact metadata-free int64 array of rank {ndim}"
    raise TypeError(msg)
  return FinalizedArray(source, dtype=np.int64)


def _finite_scalar(value: float, label: str) -> float:
  if type(value) is not float or not np.isfinite(value):
    msg = f"{label} must be an exact finite float"
    raise TypeError(msg)
  return value


def compile_point_entities(
  *,
  block_id: str,
  entity_ids: tuple[str, ...],
  source_ids: tuple[str, ...],
  reference_coordinates: FloatArray,
) -> PointEntityBlock:
  """Detach one point geometry block at the compiler boundary."""
  coordinates = _owned_float(
    reference_coordinates,
    ndim=2,
    label="reference coordinates",
  )
  if coordinates.values.shape[1] != 2:
    msg = "the prototype requires explicit two-dimensional point coordinates"
    raise ValueError(msg)
  count = int(coordinates.values.shape[0])
  return PointEntityBlock(
    block_id=_text(block_id, "point block ID"),
    entity_ids=_texts(entity_ids, count, "point entity IDs"),
    source_ids=_texts(source_ids, count, "point source IDs"),
    reference_coordinates=coordinates,
  )


def compile_displacement_space(
  *,
  space_id: str,
  support: PointEntityBlock,
  components: tuple[str, ...] = ("ux", "uy"),
) -> DiscreteSpace:
  """Compile one explicit displacement coefficient space."""
  component_ids = _texts(components, 2, "displacement components")
  coefficient_ids = tuple(
    f"{space_id}:{point_id}:{component}"
    for point_id in support.entity_ids
    for component in component_ids
  )
  return DiscreteSpace(
    space_id=_text(space_id, "space ID"),
    support=support,
    components=component_ids,
    coefficient_ids=coefficient_ids,
    coefficient_count=len(coefficient_ids),
  )


def quad4_continuum_descriptor() -> ContinuumDescriptor:
  """Create the explicit bilinear-Q4, two-by-two Gauss recipe."""
  abscissa = 1.0 / np.sqrt(3.0)
  points = np.array(
    [
      [-abscissa, -abscissa],
      [-abscissa, abscissa],
      [abscissa, -abscissa],
      [abscissa, abscissa],
    ],
    dtype=np.float64,
  )
  weights = np.ones(4, dtype=np.float64)
  return ContinuumDescriptor(
    descriptor_id="plane-stress-q4-gauss2x2-v1",
    implementation_id="continuum-small-strain-plane-stress-v1",
    node_count=4,
    quadrature_points=_owned_float(points, ndim=2, label="Q4 quadrature points"),
    quadrature_weights=_owned_float(weights, ndim=1, label="Q4 quadrature weights"),
    shape_binding=bilinear_quad4,
  )


def tria3_continuum_descriptor() -> ContinuumDescriptor:
  """Create the explicit linear-T3, centroid quadrature recipe."""
  points = np.array([[1.0 / 3.0, 1.0 / 3.0]], dtype=np.float64)
  weights = np.array([0.5], dtype=np.float64)
  return ContinuumDescriptor(
    descriptor_id="plane-stress-t3-centroid-v1",
    implementation_id="continuum-small-strain-plane-stress-v1",
    node_count=3,
    quadrature_points=_owned_float(points, ndim=2, label="T3 quadrature points"),
    quadrature_weights=_owned_float(weights, ndim=1, label="T3 quadrature weights"),
    shape_binding=linear_tria3,
  )


def _compile_entity_block(
  *,
  block_id: str,
  role: EntityRole,
  entity_ids: tuple[str, ...],
  source_ids: tuple[str, ...],
  incidence: IndexArray,
  width: int,
  point_count: int,
) -> IncidenceEntityBlock:
  owned_incidence = _owned_indices(incidence, ndim=2, label="entity incidence")
  if owned_incidence.values.shape[1] != width:
    msg = f"entity incidence width must be exactly {width}"
    raise ValueError(msg)
  count = int(owned_incidence.values.shape[0])
  if np.any(owned_incidence.values < 0) or np.any(
    owned_incidence.values >= point_count
  ):
    msg = "entity incidence contains an out-of-range point index"
    raise ValueError(msg)
  return IncidenceEntityBlock(
    block_id=_text(block_id, "entity block ID"),
    role=role,
    entity_ids=_texts(entity_ids, count, "entity IDs"),
    source_ids=_texts(source_ids, count, "entity source IDs"),
    incidence=owned_incidence,
  )


def _all_component_gather(
  space: DiscreteSpace,
  incidence: IndexArray,
) -> FinalizedArray:
  component_count = len(space.components)
  gather = (
    incidence[:, :, None] * component_count + np.arange(component_count, dtype=np.int64)
  ).reshape(incidence.shape[0], -1)
  return FinalizedArray(gather, dtype=np.int64)


def _one_component_gather(
  space: DiscreteSpace,
  incidence: IndexArray,
  component: str,
) -> FinalizedArray:
  try:
    component_index = space.components.index(component)
  except ValueError as error:
    msg = f"unknown displacement component {component!r}"
    raise ValueError(msg) from error
  gather = incidence * len(space.components) + component_index
  return FinalizedArray(gather, dtype=np.int64)


def _state_layout(block_id: str, entity_count: int) -> LocalStateLayout:
  return LocalStateLayout(
    schema_id=f"{block_id}:stateless-v1",
    entity_count=entity_count,
    width=0,
  )


def _evaluate_continuum(
  payload: ContinuumPayload,
  ports: tuple[FloatArray, ...],
) -> LocalEvaluation:
  b = payload.strain_displacement.values
  weights = payload.integration_weights.values
  constitutive = payload.constitutive.values
  stiffness = np.einsum(
    "ep,epai,ab,epbj->eij",
    weights,
    b,
    constitutive,
    b,
    optimize=True,
  )
  residual = np.einsum("eij,ej->ei", stiffness, ports[0], optimize=True)
  return LocalEvaluation(
    residual_batches=(_owned_float(residual, ndim=2, label="continuum residual"),),
    jacobian_batches=(_owned_float(stiffness, ndim=3, label="continuum Jacobian"),),
  )


def _evaluate_directional_spring(
  payload: DirectionalSpringPayload,
  ports: tuple[FloatArray, ...],
) -> LocalEvaluation:
  stiffness_values = payload.stiffness.values
  local_values = ports[0]
  extension = local_values[:, 1] - local_values[:, 0]
  residual = np.stack(
    (-stiffness_values * extension, stiffness_values * extension),
    axis=1,
  )
  jacobian = np.zeros((stiffness_values.size, 2, 2), dtype=np.float64)
  jacobian[:, 0, 0] = stiffness_values
  jacobian[:, 0, 1] = -stiffness_values
  jacobian[:, 1, 0] = -stiffness_values
  jacobian[:, 1, 1] = stiffness_values
  return LocalEvaluation(
    residual_batches=(_owned_float(residual, ndim=2, label="spring residual"),),
    jacobian_batches=(_owned_float(jacobian, ndim=3, label="spring Jacobian"),),
  )


def _evaluate_point_load(
  payload: PointLoadPayload,
  ports: tuple[FloatArray, ...],
) -> LocalEvaluation:
  del ports
  residual = payload.magnitudes.values[:, None]
  return LocalEvaluation(
    residual_batches=(_owned_float(residual, ndim=2, label="point-load residual"),),
    jacobian_batches=(),
  )


_CONTINUUM_EVALUATOR = EvaluatorBinding(
  implementation_id="continuum-small-strain-plane-stress-v1",
  evaluate=_evaluate_continuum,
)
_SPRING_EVALUATOR = EvaluatorBinding(
  implementation_id="directional-spring-linear-v1",
  evaluate=_evaluate_directional_spring,
)
_POINT_LOAD_EVALUATOR = EvaluatorBinding(
  implementation_id="point-load-dead-v1",
  evaluate=_evaluate_point_load,
)


def compile_continuum_operator(
  *,
  block_id: str,
  entity_ids: tuple[str, ...],
  source_ids: tuple[str, ...],
  connectivity: IndexArray,
  space: DiscreteSpace,
  descriptor: ContinuumDescriptor,
  youngs_modulus: float,
  poisson_ratio: float,
) -> OperatorBlock[ContinuumPayload]:
  """Compile a real signed-Jacobian 2D continuum operator block."""
  entity_block = _compile_entity_block(
    block_id=f"{block_id}:entities",
    role=EntityRole.CELL,
    entity_ids=entity_ids,
    source_ids=source_ids,
    incidence=connectivity,
    width=descriptor.node_count,
    point_count=len(space.support.entity_ids),
  )
  modulus = _finite_scalar(youngs_modulus, "Young's modulus")
  ratio = _finite_scalar(poisson_ratio, "Poisson ratio")
  if modulus <= 0.0 or not -1.0 < ratio < 0.5:
    msg = "plane-stress parameters require E > 0 and -1 < nu < 0.5"
    raise ValueError(msg)

  shapes, parent_gradients = descriptor.shape_binding(
    descriptor.quadrature_points.values
  )
  if (
    type(shapes) is not np.ndarray
    or type(parent_gradients) is not np.ndarray
    or shapes.dtype != np.dtype(np.float64)
    or parent_gradients.dtype != np.dtype(np.float64)
    or shapes.dtype.metadata is not None
    or parent_gradients.dtype.metadata is not None
    or shapes.shape
    != (descriptor.quadrature_weights.values.size, descriptor.node_count)
    or parent_gradients.shape
    != (descriptor.quadrature_weights.values.size, descriptor.node_count, 2)
    or not np.all(np.isfinite(shapes))
    or not np.all(np.isfinite(parent_gradients))
  ):
    msg = "continuum descriptor returned an invalid shape table"
    raise ValueError(msg)

  incidence = entity_block.incidence.values
  coordinates = space.support.reference_coordinates.values[incidence]
  entity_count = int(incidence.shape[0])
  point_count = int(descriptor.quadrature_weights.values.size)
  local_width = 2 * descriptor.node_count
  strain_displacement = np.zeros(
    (entity_count, point_count, 3, local_width),
    dtype=np.float64,
  )
  integration_weights = np.empty(
    (entity_count, point_count),
    dtype=np.float64,
  )
  relative_tolerance = 64.0 * np.finfo(np.float64).eps
  for entity_index in range(entity_count):
    for point_index in range(point_count):
      jacobian = coordinates[entity_index].T @ parent_gradients[point_index]
      determinant = float(np.linalg.det(jacobian))
      scale = float(np.sum(jacobian * jacobian))
      if (
        not np.isfinite(determinant)
        or not np.isfinite(scale)
        or determinant <= 0.0
        or scale <= 0.0
        or determinant / scale <= relative_tolerance
      ):
        entity = entity_block.entity_ids[entity_index]
        source = entity_block.source_ids[entity_index]
        msg = f"invalid positive continuum geometry for {entity}@{source}"
        raise ValueError(msg)
      gradients = parent_gradients[point_index] @ np.linalg.inv(jacobian)
      b = strain_displacement[entity_index, point_index]
      b[0, 0::2] = gradients[:, 0]
      b[1, 1::2] = gradients[:, 1]
      b[2, 0::2] = gradients[:, 1]
      b[2, 1::2] = gradients[:, 0]
      integration_weights[entity_index, point_index] = (
        descriptor.quadrature_weights.values[point_index] * determinant
      )

  constitutive = plane_stress_matrix(modulus, ratio)
  payload = ContinuumPayload(
    strain_displacement=_owned_float(
      strain_displacement,
      ndim=4,
      label="continuum strain-displacement table",
    ),
    integration_weights=_owned_float(
      integration_weights,
      ndim=2,
      label="continuum integration weights",
    ),
    constitutive=_owned_float(
      constitutive,
      ndim=2,
      label="continuum constitutive matrix",
    ),
  )
  gather = _all_component_gather(space, incidence)
  header = OperatorHeader(
    block_id=_text(block_id, "continuum operator ID"),
    entity_block=entity_block,
    descriptor_id=descriptor.descriptor_id,
    implementation_id=descriptor.implementation_id,
    owner=OperatorOwner.MODEL,
    ports=(PortBinding("displacement", space, gather),),
    residual_channels=(
      ResidualChannel(
        channel_id="internal-force",
        target_port=0,
        balance_role=BalanceRole.INTERNAL,
        multiplier=1.0,
      ),
    ),
    jacobian_channels=(
      JacobianChannel(
        channel_id="internal-force/d-displacement",
        target_port=0,
        source_port=0,
        balance_role=BalanceRole.INTERNAL,
        multiplier=1.0,
        symmetric=True,
      ),
    ),
    state_layout=_state_layout(block_id, entity_count),
  )
  return OperatorBlock(header, payload, _CONTINUUM_EVALUATOR)


def compile_directional_spring_operator(
  *,
  block_id: str,
  entity_ids: tuple[str, ...],
  source_ids: tuple[str, ...],
  connectivity: IndexArray,
  space: DiscreteSpace,
  component: str,
  stiffness: FloatArray,
) -> OperatorBlock[DirectionalSpringPayload]:
  """Compile a two-scalar directional spring block."""
  entity_block = _compile_entity_block(
    block_id=f"{block_id}:entities",
    role=EntityRole.LINK,
    entity_ids=entity_ids,
    source_ids=source_ids,
    incidence=connectivity,
    width=2,
    point_count=len(space.support.entity_ids),
  )
  stiffness_values = _owned_float(stiffness, ndim=1, label="spring stiffness")
  entity_count = len(entity_block.entity_ids)
  if stiffness_values.values.shape != (entity_count,) or np.any(
    stiffness_values.values <= 0.0
  ):
    msg = "spring stiffness must contain one positive value per link"
    raise ValueError(msg)
  gather = _one_component_gather(space, entity_block.incidence.values, component)
  payload = DirectionalSpringPayload(stiffness=stiffness_values)
  header = OperatorHeader(
    block_id=_text(block_id, "spring operator ID"),
    entity_block=entity_block,
    descriptor_id=f"directional-spring:{component}:v1",
    implementation_id=_SPRING_EVALUATOR.implementation_id,
    owner=OperatorOwner.MODEL,
    ports=(PortBinding("extension", space, gather),),
    residual_channels=(
      ResidualChannel(
        channel_id="internal-force",
        target_port=0,
        balance_role=BalanceRole.INTERNAL,
        multiplier=1.0,
      ),
    ),
    jacobian_channels=(
      JacobianChannel(
        channel_id="internal-force/d-extension",
        target_port=0,
        source_port=0,
        balance_role=BalanceRole.INTERNAL,
        multiplier=1.0,
        symmetric=True,
      ),
    ),
    state_layout=_state_layout(block_id, entity_count),
  )
  return OperatorBlock(header, payload, _SPRING_EVALUATOR)


def compile_point_load_operator(
  *,
  block_id: str,
  entity_ids: tuple[str, ...],
  source_ids: tuple[str, ...],
  point_incidence: IndexArray,
  space: DiscreteSpace,
  component: str,
  magnitudes: FloatArray,
) -> OperatorBlock[PointLoadPayload]:
  """Compile a program-owned additive point-load block."""
  entity_block = _compile_entity_block(
    block_id=f"{block_id}:entities",
    role=EntityRole.POINT_LOAD,
    entity_ids=entity_ids,
    source_ids=source_ids,
    incidence=point_incidence,
    width=1,
    point_count=len(space.support.entity_ids),
  )
  magnitude_values = _owned_float(magnitudes, ndim=1, label="load magnitudes")
  entity_count = len(entity_block.entity_ids)
  if magnitude_values.values.shape != (entity_count,):
    msg = "load magnitudes must contain one value per point-load entity"
    raise ValueError(msg)
  gather = _one_component_gather(
    space,
    entity_block.incidence.values,
    component,
  )
  payload = PointLoadPayload(magnitudes=magnitude_values)
  header = OperatorHeader(
    block_id=_text(block_id, "point-load operator ID"),
    entity_block=entity_block,
    descriptor_id=f"dead-point-load:{component}:v1",
    implementation_id=_POINT_LOAD_EVALUATOR.implementation_id,
    owner=OperatorOwner.PROGRAM,
    ports=(PortBinding("loaded-coefficient", space, gather),),
    residual_channels=(
      ResidualChannel(
        channel_id="external-force",
        target_port=0,
        balance_role=BalanceRole.EXTERNAL,
        multiplier=-1.0,
      ),
    ),
    jacobian_channels=(),
    state_layout=_state_layout(block_id, entity_count),
  )
  return OperatorBlock(header, payload, _POINT_LOAD_EVALUATOR)


def _canonical_operators(
  operators: tuple[AnyOperatorBlock, ...],
  space: DiscreteSpace,
) -> tuple[AnyOperatorBlock, ...]:
  if type(operators) is not tuple:
    msg = "operator blocks must be supplied as an exact tuple"
    raise TypeError(msg)
  for block in operators:
    if any(port.space is not space for port in block.header.ports):
      msg = "every operator port must bind the exact compiled space"
      raise ValueError(msg)
  ordered = tuple(sorted(operators, key=lambda block: block.header.block_id))
  ids = tuple(block.header.block_id for block in ordered)
  if len(set(ids)) != len(ids):
    msg = "operator block IDs must be unique"
    raise ValueError(msg)
  return ordered


def compile_system(
  *,
  space: DiscreteSpace,
  operators: tuple[
    OperatorBlock[ContinuumPayload] | OperatorBlock[DirectionalSpringPayload],
    ...,
  ],
) -> CompiledSystem:
  """Compile the model-owned half of the proof system."""
  canonical = _canonical_operators(operators, space)
  if any(block.header.owner is not OperatorOwner.MODEL for block in canonical):
    msg = "compiled-system operators must be model-owned"
    raise ValueError(msg)
  return CompiledSystem(space=space, operators=canonical)


def compile_program(
  *,
  space: DiscreteSpace,
  operators: tuple[OperatorBlock[PointLoadPayload], ...],
) -> CompiledProgram:
  """Compile the separately owned program half of the proof system."""
  canonical = _canonical_operators(operators, space)
  if any(block.header.owner is not OperatorOwner.PROGRAM for block in canonical):
    msg = "compiled-program operators must be program-owned"
    raise ValueError(msg)
  return CompiledProgram(compatible_space=space, operators=canonical)


def prepare_execution(
  system: CompiledSystem,
  program: CompiledProgram,
) -> PreparedExecution:
  """Join owners once and canonicalize one generic execution schedule."""
  if program.compatible_space is not system.space:
    msg = "compiled program is bound to a different live discrete space"
    raise ValueError(msg)
  operators = _canonical_operators(
    (*system.operators, *program.operators),
    system.space,
  )
  return PreparedExecution(space=system.space, operators=operators)


def assemble(
  prepared: PreparedExecution,
  coefficients: FloatArray,
) -> AssemblyResult:
  """Assemble all typed channels without inspecting operator kind or owner."""
  owned_coefficients = _owned_float(coefficients, ndim=1, label="coefficients")
  if owned_coefficients.values.shape != (prepared.space.coefficient_count,):
    msg = "coefficient vector does not match the prepared discrete space"
    raise ValueError(msg)

  residual = np.zeros(prepared.space.coefficient_count, dtype=np.float64)
  jacobian = np.zeros(
    (prepared.space.coefficient_count, prepared.space.coefficient_count),
    dtype=np.float64,
  )
  residual_terms: list[AttributedResidual] = []
  jacobian_terms: list[AttributedJacobian] = []
  evaluated_block_ids: list[str] = []

  for block in prepared.operators:
    header = block.header
    gathered = tuple(
      owned_coefficients.values[port.gather.values] for port in header.ports
    )
    evaluated = block.evaluator.evaluate(block.payload, gathered)
    evaluated_block_ids.append(header.block_id)

    for channel, batch in zip(
      header.residual_channels,
      evaluated.residual_batches,
      strict=True,
    ):
      target_gather = header.ports[channel.target_port].gather.values
      for entity_index, (entity_id, source_id) in enumerate(
        zip(
          header.entity_block.entity_ids,
          header.entity_block.source_ids,
          strict=True,
        )
      ):
        target_dofs = target_gather[entity_index]
        values = channel.multiplier * batch.values[entity_index]
        for local_index, target_dof in enumerate(target_dofs):
          residual[target_dof] += values[local_index]
        residual_terms.append(
          AttributedResidual(
            operator_id=header.block_id,
            channel=channel,
            entity_id=entity_id,
            source_id=source_id,
            target_dofs=FinalizedArray(target_dofs, dtype=np.int64),
            values=FinalizedArray(values, dtype=np.float64),
          )
        )

    for channel, batch in zip(
      header.jacobian_channels,
      evaluated.jacobian_batches,
      strict=True,
    ):
      target_gather = header.ports[channel.target_port].gather.values
      source_gather = header.ports[channel.source_port].gather.values
      for entity_index, (entity_id, source_id) in enumerate(
        zip(
          header.entity_block.entity_ids,
          header.entity_block.source_ids,
          strict=True,
        )
      ):
        target_dofs = target_gather[entity_index]
        source_dofs = source_gather[entity_index]
        values = channel.multiplier * batch.values[entity_index]
        for local_row, target_dof in enumerate(target_dofs):
          for local_column, source_dof in enumerate(source_dofs):
            jacobian[target_dof, source_dof] += values[
              local_row,
              local_column,
            ]
        jacobian_terms.append(
          AttributedJacobian(
            operator_id=header.block_id,
            channel=channel,
            entity_id=entity_id,
            source_id=source_id,
            target_dofs=FinalizedArray(target_dofs, dtype=np.int64),
            source_dofs=FinalizedArray(source_dofs, dtype=np.int64),
            values=FinalizedArray(values, dtype=np.float64),
          )
        )

  return AssemblyResult(
    residual=_owned_float(residual, ndim=1, label="assembled residual"),
    jacobian=_owned_float(jacobian, ndim=2, label="assembled Jacobian"),
    residual_terms=tuple(residual_terms),
    jacobian_terms=tuple(jacobian_terms),
    evaluated_block_ids=tuple(evaluated_block_ids),
  )
