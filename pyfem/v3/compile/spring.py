"""Kernel-parameterized point-spring compiler with operator-local state.

This module is the researcher-facing extension seam for stateful laws. A
researcher authors a plain batched kernel — displacements and accepted state
rows in, force, tangent, trial rows, and a typed evaluation status out — plus a
small declaration of state slots and parameters. The compiler owns every
trusted carrier, validates the kernel against its declared schema once at the
compile boundary, and the resulting operator coexists with the zero-width Q8
slice in one composed compiled system. Kernels never touch transactions,
codecs, or carrier construction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NoReturn, Protocol

import numpy as np

from pyfem.v3.compile.diagnostics import (
  ModelCompilationDiagnostic,
  ModelCompilationError,
)
from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.identity import InstanceId, require_same_instance
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
  SemanticId,
  StateLifetime,
)
from pyfem.v3.model.provenance import CanonicalManifest, ContentFingerprint
from pyfem.v3.model.system import (
  CompiledSource,
  CompiledSystem,
  PointEntityBlock,
  SourceAttribution,
  SystemProvenance,
)
from pyfem.v3.spec.diagnostics import SourceContext

SPRING_SYSTEM_EXTENSION_SCHEMA = "pyfem-v3-compiled-system-spring-extension-v1"
DAMAGE_ENVELOPE_STATE_SCHEMA = "pyfem-v3-spring-damage-envelope-v1"
_FLOAT64_DTYPE = np.dtype(np.float64).str


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


@dataclass(frozen=True, slots=True)
class SpringKernelResult:
  """One batched local response of a point-spring law evaluation.

  ``force`` has shape ``(entity_count, 2)``, ``tangent`` has shape
  ``(entity_count, 2, 2)``, and ``trial_rows`` has shape
  ``(entity_count, row_width)``. When ``status`` is not ``OK`` the operator
  discards the arrays and returns the accepted rows byte-equal, so kernels
  report expected numerical outcomes instead of raising.
  """

  force: np.ndarray
  tangent: np.ndarray
  trial_rows: np.ndarray
  status: EvaluationStatus


class SpringKernel(Protocol):
  """Researcher-authored batched constitutive kernel for point springs."""

  def __call__(
    self,
    displacements: np.ndarray,
    accepted_rows: np.ndarray,
    parameters: np.ndarray,
  ) -> SpringKernelResult: ...


@dataclass(frozen=True, slots=True)
class SpringStateSlot:
  """One named contiguous slice of a spring entity's state row."""

  name: str
  width: int

  def __post_init__(self) -> None:
    if type(self.name) is not str or not self.name:
      msg = "spring state slot names must be non-empty exact strings"
      raise TypeError(msg)
    if type(self.width) is not int or self.width <= 0:
      msg = "spring state slot widths must be positive exact integers"
      raise TypeError(msg)


@dataclass(frozen=True, slots=True)
class SpringDeclaration:
  """Authored meaning for one network of stateful point springs."""

  block_id: SemanticId
  space_id: SemanticId
  spring_ids: tuple[SemanticId, ...]
  node_ids: tuple[SemanticId, ...]
  state_schema: str
  state_slots: tuple[SpringStateSlot, ...]
  kernel_name: str
  kernel_version: str
  implementation_id: str
  parameters: tuple[float, ...]
  kernel: SpringKernel
  source: SourceContext

  def __post_init__(self) -> None:
    if type(self.block_id) not in (str, int, tuple):
      msg = "spring block id must be an exact semantic id"
      raise TypeError(msg)
    if (
      type(self.spring_ids) is not tuple
      or type(self.node_ids) is not tuple
      or not self.spring_ids
      or len(self.spring_ids) != len(self.node_ids)
      or len(set(self.spring_ids)) != len(self.spring_ids)
    ):
      msg = "spring declarations require paired unique spring and node id tuples"
      raise TypeError(msg)
    if type(self.state_schema) is not str or not self.state_schema:
      msg = "spring state schema must be a non-empty exact string"
      raise TypeError(msg)
    if type(self.state_slots) is not tuple or any(
      type(slot) is not SpringStateSlot for slot in self.state_slots
    ):
      msg = "spring state slots must be exact SpringStateSlot values"
      raise TypeError(msg)
    if len({slot.name for slot in self.state_slots}) != len(self.state_slots):
      msg = "spring state slot names must be unique"
      raise ValueError(msg)
    for label in ("kernel_name", "kernel_version", "implementation_id"):
      value = getattr(self, label)
      if type(value) is not str or not value:
        msg = f"spring {label} must be a non-empty exact string"
        raise TypeError(msg)
    if type(self.parameters) is not tuple or any(
      type(value) is not float or not math.isfinite(value) for value in self.parameters
    ):
      msg = "spring parameters must be finite exact floats"
      raise TypeError(msg)
    if not callable(self.kernel):
      msg = "spring kernel must be callable"
      raise TypeError(msg)
    if type(self.source) is not SourceContext:
      msg = "spring declarations require an exact SourceContext"
      raise TypeError(msg)


def damage_envelope_kernel(
  displacements: np.ndarray,
  accepted_rows: np.ndarray,
  parameters: np.ndarray,
) -> SpringKernelResult:
  """Isotropic damage-envelope spring with an irreversible max-extension row.

  State row ``[kappa]`` records the largest displacement magnitude ever
  accepted. Damage ``omega = min(kappa / critical_extension, 1)`` degrades the
  stiffness irreversibly; unloading keeps the degraded secant. A trial whose
  extension jump exceeds ``max_increment`` skips the envelope path and is
  classified ``REJECT_STEP`` so the schedule cuts back from the same accepted
  generation. At full damage the spring carries no force.
  """
  stiffness, critical_extension, max_increment = parameters
  kappa = accepted_rows[:, 0]
  radius = np.linalg.norm(displacements, axis=1)
  if bool((radius - kappa > max_increment).any()):
    return SpringKernelResult(
      force=np.zeros_like(displacements),
      tangent=np.zeros((len(displacements), 2, 2)),
      trial_rows=np.array(accepted_rows, copy=True),
      status=EvaluationStatus.REJECT_STEP,
    )
  trial_kappa = np.maximum(kappa, radius)
  damage = np.minimum(trial_kappa / critical_extension, 1.0)
  force = (1.0 - damage)[:, None] * stiffness * displacements
  tangent = ((1.0 - damage) * stiffness)[:, None, None] * np.eye(2)[None, :, :]
  growing = (radius > kappa) & (trial_kappa < critical_extension)
  safe_radius = np.where(growing, np.maximum(radius, 1.0e-300), 1.0)
  degradation = np.where(
    growing,
    stiffness / (critical_extension * safe_radius),
    0.0,
  )
  tangent = tangent - degradation[:, None, None] * np.einsum(
    "ei,ej->eij",
    displacements,
    displacements,
  )
  return SpringKernelResult(
    force=force,
    tangent=tangent,
    trial_rows=trial_kappa[:, None],
    status=EvaluationStatus.OK,
  )


def damage_envelope_declaration(
  *,
  block_id: SemanticId,
  space_id: SemanticId,
  spring_ids: tuple[SemanticId, ...],
  node_ids: tuple[SemanticId, ...],
  stiffness: float,
  critical_extension: float,
  max_increment: float,
  source: SourceContext,
) -> SpringDeclaration:
  """Build a validated declaration for the damage-envelope spring law."""
  for label, value in (
    ("stiffness", stiffness),
    ("critical_extension", critical_extension),
    ("max_increment", max_increment),
  ):
    if type(value) is not float or not math.isfinite(value) or value <= 0.0:
      msg = f"damage envelope {label} must be a positive finite exact float"
      raise ValueError(msg)
  return SpringDeclaration(
    block_id=block_id,
    space_id=space_id,
    spring_ids=spring_ids,
    node_ids=node_ids,
    state_schema=DAMAGE_ENVELOPE_STATE_SCHEMA,
    state_slots=(SpringStateSlot("max_extension", 1),),
    kernel_name="damage-envelope-spring",
    kernel_version="1",
    implementation_id=DAMAGE_ENVELOPE_STATE_SCHEMA,
    parameters=(stiffness, critical_extension, max_increment),
    kernel=damage_envelope_kernel,
    source=source,
  )


def _validated_kernel_arrays(
  result: object,
  *,
  entity_count: int,
  row_width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, EvaluationStatus]:
  if type(result) is not SpringKernelResult:
    msg = "spring kernels must return an exact SpringKernelResult"
    raise TypeError(msg)
  status = result.status
  if type(status) is not EvaluationStatus:
    msg = "spring kernel status must be an exact EvaluationStatus"
    raise TypeError(msg)
  arrays = (result.force, result.tangent, result.trial_rows)
  shapes = ((entity_count, 2), (entity_count, 2, 2), (entity_count, row_width))
  for array, shape in zip(arrays, shapes, strict=True):
    if (
      type(array) is not np.ndarray
      or array.dtype != np.dtype(np.float64)
      or array.dtype.metadata is not None
      or array.shape != shape
    ):
      msg = "spring kernel arrays must match the declared batched float64 shapes"
      raise TypeError(msg)
  if status is EvaluationStatus.OK and not all(
    bool(np.isfinite(array).all()) for array in arrays
  ):
    msg = "spring kernel arrays must be finite for a successful evaluation"
    raise TypeError(msg)
  return result.force, result.tangent, result.trial_rows, status


@dataclass(frozen=True, slots=True, eq=False, init=False)
class SpringPayload(CompilerConstructed):
  parameters: FinalizedArray


@dataclass(frozen=True, slots=True, eq=False, init=False)
class SpringOperator(CompilerConstructed):
  header: OperatorHeader
  spring_block: PointEntityBlock
  payload: SpringPayload
  content_manifest: CanonicalManifest
  kernel: SpringKernel
  system_instance: InstanceId

  def evaluate(
    self,
    inputs: OperatorEvaluationInput,
  ) -> OperatorEvaluation:
    """Evaluate the spring network from accepted state with typed outcomes."""
    if type(inputs) is not OperatorEvaluationInput:
      msg = "spring evaluation requires an exact immutable evaluation input"
      raise TypeError(msg)
    if type(inputs.port_values) is not tuple or len(inputs.port_values) != 1:
      msg = "spring evaluation requires exactly one displacement port batch"
      raise TypeError(msg)
    values = inputs.port_values[0].values
    expected = self.header.ports[0].coefficient_map.values.shape
    if (
      values.dtype != np.dtype(np.float64)
      or values.dtype.metadata is not None
      or values.shape != expected
      or not bool(np.isfinite(values).all())
    ):
      msg = "spring displacement port values must be a finite float64 batch"
      raise TypeError(msg)
    layout = self.header.state_layout
    accepted_state = inputs.accepted_state.values
    if (
      accepted_state.dtype != np.dtype(np.float64)
      or accepted_state.dtype.metadata is not None
      or accepted_state.shape != layout.row_shape
      or not bool(np.isfinite(accepted_state).all())
    ):
      msg = "spring accepted state must match the compiled state layout"
      raise TypeError(msg)
    if inputs.signals or self.header.signal_ports:
      msg = "spring model operator does not accept program signal inputs"
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
      msg = "spring evaluation request contains an unavailable or duplicate channel"
      raise ValueError(msg)

    force, tangent, trial_rows, status = _validated_kernel_arrays(
      self.kernel(values, accepted_state, self.payload.parameters.values),
      entity_count=layout.entity_count,
      row_width=layout.row_width,
    )
    if status is not EvaluationStatus.OK:
      return _new(
        OperatorEvaluation,
        residual_values=(),
        jacobian_values=(),
        trial_state=FinalizedArray(accepted_state, dtype=np.float64),
        status=status,
      )
    residual_values = (
      (FinalizedArray(force, dtype=np.float64),) if request.residual_channel_ids else ()
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
      trial_state=FinalizedArray(trial_rows, dtype=np.float64),
      status=status,
    )


def compile_spring_operator(
  system: CompiledSystem,
  declaration: SpringDeclaration,
) -> tuple[PointEntityBlock, SpringOperator]:
  """Compile one authored spring network against an existing compiled system."""
  if type(system) is not CompiledSystem:
    msg = "spring compilation requires an exact CompiledSystem"
    raise TypeError(msg)
  if type(declaration) is not SpringDeclaration:
    msg = "spring compilation requires an exact SpringDeclaration"
    raise TypeError(msg)
  source = declaration.source
  space = next(
    (item for item in system.spaces if item.space_id == declaration.space_id),
    None,
  )
  if space is None:
    _fail(
      "unknown-spring-space",
      "spring network references a space the compiled system does not have",
      source,
    )
  if len(space.components) != 2:
    _fail(
      "unsupported-spring-space",
      "point springs require a two-component displacement space",
      source,
    )
  support = next(
    (item for item in system.point_blocks if item.block_id == space.support_block_id),
    None,
  )
  if support is None:
    _fail(
      "unknown-spring-support-block",
      "spring space support block is absent from the compiled system",
      source,
    )
  node_dense = {node_id: index for index, node_id in enumerate(support.entity_ids)}
  node_indices: list[int] = []
  for node_id in declaration.node_ids:
    index = node_dense.get(node_id)
    if index is None:
      _fail(
        "unknown-spring-support-node",
        "spring network references a support node the compiled system does not have",
        source,
      )
    node_indices.append(index)

  entity_count = len(declaration.spring_ids)
  row_width = sum(slot.width for slot in declaration.state_slots)
  parameters = FinalizedArray(declaration.parameters, dtype=np.float64)
  # The probe must mirror runtime input mutability exactly: evaluate hands the
  # kernel read-only arrays, so the probe does too, or an input-mutating kernel
  # would compile and fail untyped at first evaluation.
  probe_displacements = np.zeros((entity_count, 2), dtype=np.float64)
  probe_rows = np.zeros((entity_count, row_width), dtype=np.float64)
  probe_displacements.setflags(write=False)
  probe_rows.setflags(write=False)
  try:
    probe_result = declaration.kernel(
      probe_displacements,
      probe_rows,
      parameters.values,
    )
  except Exception:
    _fail(
      "kernel-probe-failed",
      "spring kernel failed its virgin-state compile probe",
      source,
    )
  probe = _validated_kernel_arrays(
    probe_result,
    entity_count=entity_count,
    row_width=row_width,
  )
  if probe[3] is not EvaluationStatus.OK:
    _fail(
      "invalid-kernel-probe",
      "spring kernel must evaluate its virgin zero state successfully",
      source,
    )

  spring_block = _new(
    PointEntityBlock,
    block_id=declaration.block_id,
    entity_ids=declaration.spring_ids,
    sources=tuple(_source(source) for _ in declaration.spring_ids),
    reference_coordinates=FinalizedArray(
      support.reference_coordinates.values[node_indices],
      dtype=np.float64,
    ),
  )
  block_id = declaration.block_id, declaration.state_schema
  state_layout = _new(
    OperatorStateLayout,
    schema=declaration.state_schema,
    block_id=block_id,
    entity_count=entity_count,
    slots=tuple(
      _new(
        OperatorStateSlot,
        name=slot.name,
        width=slot.width,
        dtype=_FLOAT64_DTYPE,
        lifetime=StateLifetime.ACCEPTED_TRIAL,
      )
      for slot in declaration.state_slots
    ),
    entity_offsets=FinalizedArray(
      np.arange(entity_count + 1, dtype=space.coefficient_map.values.dtype) * row_width,
      dtype=space.coefficient_map.values.dtype,
    ),
    row_width=row_width,
    dtype=_FLOAT64_DTYPE,
    lifetime=StateLifetime.ACCEPTED_TRIAL,
  )
  port = _new(
    PortBinding,
    port_id="displacement",
    space_id=space.space_id,
    mode=PortMode.COEFFICIENTS,
    coefficient_map=FinalizedArray(
      space.coefficient_map.values[node_indices],
      dtype=space.coefficient_map.values.dtype,
    ),
  )
  residual_channel = _new(
    ResidualChannel,
    channel_id="spring-force",
    target_port_id=port.port_id,
    balance_role=BalanceRole.INTERNAL,
    linear=False,
  )
  jacobian_channel = _new(
    JacobianChannel,
    channel_id="spring-tangent",
    residual_channel_id=residual_channel.channel_id,
    target_port_id=port.port_id,
    source_port_id=port.port_id,
    balance_role=BalanceRole.INTERNAL,
    linear=False,
    symmetric=True,
  )
  header = _new(
    OperatorHeader,
    block_id=block_id,
    entity_block_id=spring_block.block_id,
    implementations=(
      _new(
        ImplementationIdentity,
        kind="constitutive-kernel",
        name=declaration.kernel_name,
        version=declaration.kernel_version,
        implementation_id=declaration.implementation_id,
      ),
    ),
    ports=(port,),
    signal_ports=(),
    residual_channels=(residual_channel,),
    jacobian_channels=(jacobian_channel,),
    state_layout=state_layout,
    coupling_policy=CouplingPolicy.FIXED,
  )
  payload = _new(SpringPayload, parameters=parameters)
  manifest = CanonicalManifest(
    {
      "block_id": block_id,
      "entity_block_id": spring_block.block_id,
      "entity_ids": spring_block.entity_ids,
      "support_node_ids": declaration.node_ids,
      "implementations": [
        {
          "kind": "constitutive-kernel",
          "name": declaration.kernel_name,
          "version": declaration.kernel_version,
          "implementation_id": declaration.implementation_id,
        }
      ],
      "port": {
        "port_id": port.port_id,
        "space_id": port.space_id,
        "coefficient_map": port.coefficient_map.values,
      },
      "channels": ["spring-force", "spring-tangent"],
      "state": {
        "schema": state_layout.schema,
        "row_width": row_width,
        "slots": [
          {"name": slot.name, "width": slot.width} for slot in declaration.state_slots
        ],
        "entity_offsets": state_layout.entity_offsets.values,
      },
      "payload": {"parameters": payload.parameters.values},
    }
  )
  operator = _new(
    SpringOperator,
    header=header,
    spring_block=spring_block,
    payload=payload,
    content_manifest=manifest,
    kernel=declaration.kernel,
    system_instance=system.instance_id,
  )
  return spring_block, operator


def compose_system(
  base: CompiledSystem,
  spring_block: PointEntityBlock,
  spring_operator: SpringOperator,
) -> CompiledSystem:
  """Compose one compiled spring network into a base compiled system."""
  if type(base) is not CompiledSystem:
    msg = "spring composition requires an exact base CompiledSystem"
    raise TypeError(msg)
  if type(spring_block) is not PointEntityBlock:
    msg = "spring composition requires an exact spring PointEntityBlock"
    raise TypeError(msg)
  if type(spring_operator) is not SpringOperator:
    msg = "spring composition requires an exact SpringOperator"
    raise TypeError(msg)
  require_same_instance(
    spring_operator.system_instance,
    base.instance_id,
    context="spring system composition",
  )
  if spring_operator.spring_block is not spring_block:
    msg = "spring operator must bind the exact composed spring block"
    raise ValueError(msg)
  if any(block.block_id == spring_block.block_id for block in base.point_blocks) or any(
    block.block_id == spring_block.block_id for block in base.entity_blocks
  ):
    msg = "spring block id collides with an existing compiled entity block"
    raise ValueError(msg)
  if any(
    operator.header.block_id == spring_operator.header.block_id
    for operator in base.operators
  ):
    msg = "spring operator block id collides with an existing compiled operator"
    raise ValueError(msg)
  space_ids = {space.space_id for space in base.spaces}
  if spring_operator.header.ports[0].space_id not in space_ids:
    msg = "spring operator port references a space outside the base system"
    raise ValueError(msg)

  spring_attribution = (
    _new(
      SourceAttribution,
      kind="entity_block",
      semantic_id=spring_block.block_id,
      source=spring_block.sources[0],
    ),
    *(
      _new(
        SourceAttribution,
        kind="spring",
        semantic_id=entity_id,
        source=source,
      )
      for entity_id, source in zip(
        spring_block.entity_ids,
        spring_block.sources,
        strict=True,
      )
    ),
  )
  attributions = (*base.source_attribution, *spring_attribution)
  manifest = CanonicalManifest(
    {
      "schema": SPRING_SYSTEM_EXTENSION_SCHEMA,
      "base_system": base.provenance.manifest,
      "spring_point_block": {
        "block_id": spring_block.block_id,
        "entity_ids": spring_block.entity_ids,
        "reference_coordinates": spring_block.reference_coordinates.values,
      },
      "spring_operator": spring_operator.content_manifest,
      "source_attribution": [
        {
          "kind": record.kind,
          "semantic_id": record.semantic_id,
          "source": {
            "source": record.source.source,
            "line": record.source.line,
            "column": record.source.column,
          },
        }
        for record in spring_attribution
      ],
    }
  )
  provenance = _new(
    SystemProvenance,
    schema=SPRING_SYSTEM_EXTENSION_SCHEMA,
    manifest=manifest,
    registry_fingerprint=base.provenance.registry_fingerprint,
    floating_dtype=base.provenance.floating_dtype,
    dense_index_dtype=base.provenance.dense_index_dtype,
    geometry_relative_tolerance=base.provenance.geometry_relative_tolerance,
  )
  return _new(
    CompiledSystem,
    instance_id=InstanceId(),
    content_fingerprint=ContentFingerprint.from_manifest(manifest),
    provenance=provenance,
    registry_snapshot=base.registry_snapshot,
    point_blocks=(*base.point_blocks, spring_block),
    entity_blocks=base.entity_blocks,
    spaces=base.spaces,
    operators=(*base.operators, spring_operator),
    source_attribution=attributions,
  )
