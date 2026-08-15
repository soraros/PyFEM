"""Isolated executable proof of three unlike typed finite-element operators."""

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


class BalanceRole(Enum):
  INTERNAL = "internal"
  EXTERNAL = "external"


@dataclass(frozen=True, slots=True, eq=False)
class PointEntityBlock:
  entity_ids: tuple[str, ...]
  source_ids: tuple[str, ...]
  reference_coordinates: FinalizedArray


@dataclass(frozen=True, slots=True, eq=False)
class IncidenceEntityBlock:
  entity_ids: tuple[str, ...]
  source_ids: tuple[str, ...]
  incidence: FinalizedArray


@dataclass(frozen=True, slots=True, eq=False)
class DiscreteSpace:
  space_id: str
  support: PointEntityBlock
  components: tuple[str, ...]
  coefficient_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True, eq=False)
class PortBinding:
  port_id: str
  space: DiscreteSpace
  gather: FinalizedArray


@dataclass(frozen=True, slots=True)
class ResidualChannel:
  channel_id: str
  target_port: int
  balance_role: BalanceRole
  multiplier: float


@dataclass(frozen=True, slots=True)
class JacobianChannel:
  channel_id: str
  target_port: int
  source_port: int
  balance_role: BalanceRole
  multiplier: float
  symmetric: bool


@dataclass(frozen=True, slots=True, eq=False)
class ContinuumPayload:
  strain_displacement: FinalizedArray
  integration_weights: FinalizedArray
  constitutive: FinalizedArray


@dataclass(frozen=True, slots=True, eq=False)
class DirectionalSpringPayload:
  stiffness: FinalizedArray


@dataclass(frozen=True, slots=True, eq=False)
class PointLoadPayload:
  magnitudes: FinalizedArray


type LocalEvaluation = tuple[tuple[FloatArray, ...], tuple[FloatArray, ...]]


@dataclass(frozen=True, slots=True, eq=False)
class OperatorBlock[PayloadT]:
  block_id: str
  entity_block: IncidenceEntityBlock
  ports: tuple[PortBinding, ...]
  residual_channels: tuple[ResidualChannel, ...]
  jacobian_channels: tuple[JacobianChannel, ...]
  local_state_width: int
  payload: PayloadT
  evaluator: Callable[[PayloadT, tuple[FloatArray, ...]], LocalEvaluation]


type AnyOperatorBlock = (
  OperatorBlock[ContinuumPayload]
  | OperatorBlock[DirectionalSpringPayload]
  | OperatorBlock[PointLoadPayload]
)


@dataclass(frozen=True, slots=True, eq=False)
class ContinuumDescriptor:
  parent_gradients: FinalizedArray
  quadrature_weights: FinalizedArray


@dataclass(frozen=True, slots=True, eq=False)
class CompiledSystem:
  space: DiscreteSpace
  operators: tuple[AnyOperatorBlock, ...]


@dataclass(frozen=True, slots=True, eq=False)
class CompiledProgram:
  compatible_space: DiscreteSpace
  operators: tuple[OperatorBlock[PointLoadPayload], ...]


@dataclass(frozen=True, slots=True, eq=False)
class PreparedExecution:
  space: DiscreteSpace
  operators: tuple[AnyOperatorBlock, ...]


@dataclass(frozen=True, slots=True, eq=False)
class AttributedTerm[ChannelT]:
  operator_id: str
  channel: ChannelT
  entity_id: str
  source_id: str
  port_dofs: tuple[FinalizedArray, ...]
  values: FinalizedArray


@dataclass(frozen=True, slots=True, eq=False)
class AssemblyResult:
  residual: FinalizedArray
  jacobian: FinalizedArray
  residual_terms: tuple[AttributedTerm[ResidualChannel], ...]
  jacobian_terms: tuple[AttributedTerm[JacobianChannel], ...]


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


def _owned_array(
  source: FloatArray | IndexArray,
  *,
  dtype: np.dtype[np.float64] | np.dtype[np.int64],
  ndim: int,
  label: str,
  finite: bool = False,
) -> FinalizedArray:
  if (
    type(source) is not np.ndarray
    or source.dtype != dtype
    or source.dtype.metadata is not None
    or source.ndim != ndim
  ):
    msg = f"{label} must be an exact metadata-free {dtype.name} array of rank {ndim}"
    raise TypeError(msg)
  if finite and not np.all(np.isfinite(source)):
    msg = f"{label} must contain finite values"
    raise ValueError(msg)
  return FinalizedArray(source, dtype=dtype)


def _owned_float(source: FloatArray, *, ndim: int, label: str) -> FinalizedArray:
  return _owned_array(
    source,
    dtype=np.dtype(np.float64),
    ndim=ndim,
    label=label,
    finite=True,
  )


def _finite_scalar(value: float, label: str) -> float:
  if type(value) is not float or not np.isfinite(value):
    msg = f"{label} must be an exact finite float"
    raise TypeError(msg)
  return value


def compile_point_entities(
  *,
  entity_ids: tuple[str, ...],
  source_ids: tuple[str, ...],
  reference_coordinates: FloatArray,
) -> PointEntityBlock:
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
  )


def quad4_continuum_descriptor() -> ContinuumDescriptor:
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
  _, parent_gradients = bilinear_quad4(points)
  return ContinuumDescriptor(
    parent_gradients=_owned_float(
      parent_gradients,
      ndim=3,
      label="Q4 parent gradients",
    ),
    quadrature_weights=_owned_float(weights, ndim=1, label="Q4 quadrature weights"),
  )


def tria3_continuum_descriptor() -> ContinuumDescriptor:
  points = np.array([[1.0 / 3.0, 1.0 / 3.0]], dtype=np.float64)
  weights = np.array([0.5], dtype=np.float64)
  _, parent_gradients = linear_tria3(points)
  return ContinuumDescriptor(
    parent_gradients=_owned_float(
      parent_gradients,
      ndim=3,
      label="T3 parent gradients",
    ),
    quadrature_weights=_owned_float(weights, ndim=1, label="T3 quadrature weights"),
  )


def _compile_entity_block(
  *,
  entity_ids: tuple[str, ...],
  source_ids: tuple[str, ...],
  incidence: IndexArray,
  width: int,
  point_count: int,
) -> IncidenceEntityBlock:
  owned_incidence = _owned_array(
    incidence,
    dtype=np.dtype(np.int64),
    ndim=2,
    label="entity incidence",
  )
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
  return (residual,), (stiffness,)


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
  spring_matrix = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float64)
  jacobian = stiffness_values[:, None, None] * spring_matrix
  return (residual,), (jacobian,)


def _evaluate_point_load(
  payload: PointLoadPayload,
  ports: tuple[FloatArray, ...],
) -> LocalEvaluation:
  del ports
  residual = payload.magnitudes.values[:, None]
  return (residual,), ()


def _operator_block[PayloadT](
  *,
  block_id: str,
  entities: IncidenceEntityBlock,
  port: PortBinding,
  balance_role: BalanceRole,
  payload: PayloadT,
  evaluator: Callable[[PayloadT, tuple[FloatArray, ...]], LocalEvaluation],
  with_jacobian: bool,
) -> OperatorBlock[PayloadT]:
  channel_id = f"{balance_role.value}-force"
  multiplier = 1.0 if balance_role is BalanceRole.INTERNAL else -1.0
  jacobian_channels = (
    (
      JacobianChannel(
        channel_id=f"{channel_id}/d-{port.port_id}",
        target_port=0,
        source_port=0,
        balance_role=balance_role,
        multiplier=multiplier,
        symmetric=True,
      ),
    )
    if with_jacobian
    else ()
  )
  return OperatorBlock(
    block_id=_text(block_id, "operator ID"),
    entity_block=entities,
    ports=(port,),
    residual_channels=(ResidualChannel(channel_id, 0, balance_role, multiplier),),
    jacobian_channels=jacobian_channels,
    local_state_width=0,
    payload=payload,
    evaluator=evaluator,
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
  node_count = int(descriptor.parent_gradients.values.shape[1])
  entity_block = _compile_entity_block(
    entity_ids=entity_ids,
    source_ids=source_ids,
    incidence=connectivity,
    width=node_count,
    point_count=len(space.support.entity_ids),
  )
  modulus = _finite_scalar(youngs_modulus, "Young's modulus")
  ratio = _finite_scalar(poisson_ratio, "Poisson ratio")
  if modulus <= 0.0 or not -1.0 < ratio < 0.5:
    msg = "plane-stress parameters require E > 0 and -1 < nu < 0.5"
    raise ValueError(msg)

  incidence = entity_block.incidence.values
  coordinates = space.support.reference_coordinates.values[incidence]
  entity_count = int(incidence.shape[0])
  point_count = int(descriptor.quadrature_weights.values.size)
  jacobians = np.einsum(
    "eni,pnj->epij",
    coordinates,
    descriptor.parent_gradients.values,
    optimize=True,
  )
  determinants = np.linalg.det(jacobians)
  scales = np.sum(jacobians * jacobians, axis=(2, 3))
  invalid = (
    ~np.isfinite(determinants)
    | ~np.isfinite(scales)
    | (determinants <= 0.0)
    | (scales <= 0.0)
    | (determinants / scales <= 64.0 * np.finfo(np.float64).eps)
  )
  if np.any(invalid):
    entity_index = int(np.argwhere(invalid)[0, 0])
    entity = entity_block.entity_ids[entity_index]
    source = entity_block.source_ids[entity_index]
    msg = f"invalid positive continuum geometry for {entity}@{source}"
    raise ValueError(msg)

  gradients = np.einsum(
    "pnj,epjk->epnk",
    descriptor.parent_gradients.values,
    np.linalg.inv(jacobians),
    optimize=True,
  )
  strain_displacement = np.zeros(
    (entity_count, point_count, 3, 2 * node_count),
    dtype=np.float64,
  )
  strain_displacement[:, :, 0, 0::2] = gradients[:, :, :, 0]
  strain_displacement[:, :, 1, 1::2] = gradients[:, :, :, 1]
  strain_displacement[:, :, 2, 0::2] = gradients[:, :, :, 1]
  strain_displacement[:, :, 2, 1::2] = gradients[:, :, :, 0]
  integration_weights = determinants * descriptor.quadrature_weights.values

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
  return _operator_block(
    block_id=block_id,
    entities=entity_block,
    port=PortBinding("displacement", space, gather),
    balance_role=BalanceRole.INTERNAL,
    payload=payload,
    evaluator=_evaluate_continuum,
    with_jacobian=True,
  )


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
  entity_block = _compile_entity_block(
    entity_ids=entity_ids,
    source_ids=source_ids,
    incidence=connectivity,
    width=2,
    point_count=len(space.support.entity_ids),
  )
  stiffness_values = _owned_float(stiffness, ndim=1, label="spring stiffness")
  if stiffness_values.values.shape != (len(entity_block.entity_ids),) or np.any(
    stiffness_values.values <= 0.0
  ):
    msg = "spring stiffness must contain one positive value per link"
    raise ValueError(msg)
  gather = _one_component_gather(space, entity_block.incidence.values, component)
  payload = DirectionalSpringPayload(stiffness=stiffness_values)
  return _operator_block(
    block_id=block_id,
    entities=entity_block,
    port=PortBinding("extension", space, gather),
    balance_role=BalanceRole.INTERNAL,
    payload=payload,
    evaluator=_evaluate_directional_spring,
    with_jacobian=True,
  )


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
  entity_block = _compile_entity_block(
    entity_ids=entity_ids,
    source_ids=source_ids,
    incidence=point_incidence,
    width=1,
    point_count=len(space.support.entity_ids),
  )
  magnitude_values = _owned_float(magnitudes, ndim=1, label="load magnitudes")
  if magnitude_values.values.shape != (len(entity_block.entity_ids),):
    msg = "load magnitudes must contain one value per point-load entity"
    raise ValueError(msg)
  gather = _one_component_gather(
    space,
    entity_block.incidence.values,
    component,
  )
  payload = PointLoadPayload(magnitudes=magnitude_values)
  return _operator_block(
    block_id=block_id,
    entities=entity_block,
    port=PortBinding("loaded-coefficient", space, gather),
    balance_role=BalanceRole.EXTERNAL,
    payload=payload,
    evaluator=_evaluate_point_load,
    with_jacobian=False,
  )


def _canonical_operators(
  operators: tuple[AnyOperatorBlock, ...],
  space: DiscreteSpace,
) -> tuple[AnyOperatorBlock, ...]:
  if type(operators) is not tuple:
    msg = "operator blocks must be supplied as an exact tuple"
    raise TypeError(msg)
  for block in operators:
    if any(port.space is not space for port in block.ports):
      msg = "every operator port must bind the exact compiled space"
      raise ValueError(msg)
  ordered = tuple(sorted(operators, key=lambda block: block.block_id))
  ids = tuple(block.block_id for block in ordered)
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
  return CompiledSystem(space=space, operators=_canonical_operators(operators, space))


def compile_program(
  *,
  space: DiscreteSpace,
  operators: tuple[OperatorBlock[PointLoadPayload], ...],
) -> CompiledProgram:
  return CompiledProgram(
    compatible_space=space,
    operators=_canonical_operators(operators, space),
  )


def prepare_execution(
  system: CompiledSystem,
  program: CompiledProgram,
) -> PreparedExecution:
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
  owned_coefficients = _owned_float(coefficients, ndim=1, label="coefficients")
  coefficient_count = len(prepared.space.coefficient_ids)
  if owned_coefficients.values.shape != (coefficient_count,):
    msg = "coefficient vector does not match the prepared discrete space"
    raise ValueError(msg)

  residual = np.zeros(coefficient_count, dtype=np.float64)
  jacobian = np.zeros(
    (coefficient_count, coefficient_count),
    dtype=np.float64,
  )
  residual_terms: list[AttributedTerm[ResidualChannel]] = []
  jacobian_terms: list[AttributedTerm[JacobianChannel]] = []

  for block in prepared.operators:
    gathered = tuple(
      owned_coefficients.values[port.gather.values] for port in block.ports
    )
    residual_batches, jacobian_batches = block.evaluator(
      block.payload,
      gathered,
    )
    for channel, batch in zip(
      block.residual_channels,
      residual_batches,
      strict=True,
    ):
      target_gather = block.ports[channel.target_port].gather.values
      for entity_index, (entity_id, source_id) in enumerate(
        zip(
          block.entity_block.entity_ids,
          block.entity_block.source_ids,
          strict=True,
        )
      ):
        target_dofs = target_gather[entity_index]
        values = channel.multiplier * batch[entity_index]
        np.add.at(residual, target_dofs, values)
        residual_terms.append(
          AttributedTerm(
            operator_id=block.block_id,
            channel=channel,
            entity_id=entity_id,
            source_id=source_id,
            port_dofs=(FinalizedArray(target_dofs, dtype=np.int64),),
            values=FinalizedArray(values, dtype=np.float64),
          )
        )

    for channel, batch in zip(
      block.jacobian_channels,
      jacobian_batches,
      strict=True,
    ):
      target_gather = block.ports[channel.target_port].gather.values
      source_gather = block.ports[channel.source_port].gather.values
      for entity_index, (entity_id, source_id) in enumerate(
        zip(
          block.entity_block.entity_ids,
          block.entity_block.source_ids,
          strict=True,
        )
      ):
        target_dofs = target_gather[entity_index]
        source_dofs = source_gather[entity_index]
        values = channel.multiplier * batch[entity_index]
        np.add.at(
          jacobian,
          (target_dofs[:, None], source_dofs[None, :]),
          values,
        )
        jacobian_terms.append(
          AttributedTerm(
            operator_id=block.block_id,
            channel=channel,
            entity_id=entity_id,
            source_id=source_id,
            port_dofs=(
              FinalizedArray(target_dofs, dtype=np.int64),
              FinalizedArray(source_dofs, dtype=np.int64),
            ),
            values=FinalizedArray(values, dtype=np.float64),
          )
        )

  return AssemblyResult(
    residual=_owned_float(residual, ndim=1, label="assembled residual"),
    jacobian=_owned_float(jacobian, ndim=2, label="assembled Jacobian"),
    residual_terms=tuple(residual_terms),
    jacobian_terms=tuple(jacobian_terms),
  )
