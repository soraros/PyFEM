"""FE² continuum operator: orchestrated nested response per macro integration point."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyfem.v3.compile.continuum import Q8ContinuumOperator, Q8ContinuumPayload
from pyfem.v3.compile.diagnostics import (
  ModelCompilationDiagnostic,
  ModelCompilationError,
)
from pyfem.v3.constraints.compile import CompiledConstraintMap
from pyfem.v3.driver import (
  DriverStatus,
  NonlinearStaticDriver,
  NonlinearStaticSettings,
  SubstepStatus,
)
from pyfem.v3.driver.contracts import HomogenizationRequest
from pyfem.v3.driver.diagnostics import DriverEvaluationError
from pyfem.v3.fem.element import continuum_internal_force_batched
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
  NestedEvaluationInput,
  NestedExecutionDeclaration,
  NestedExecutionModality,
  OperatorEvaluation,
  OperatorEvaluationInput,
  OperatorHeader,
  OperatorStateLayout,
  OperatorStateSlot,
  ResidualChannel,
  StateLifetime,
  evaluation_nested_references,
  evaluation_status,
)
from pyfem.v3.model.provenance import CanonicalManifest, ContentFingerprint
from pyfem.v3.model.system import CompiledSystem, SystemProvenance
from pyfem.v3.spec.diagnostics import SourceContext
from pyfem.v3.spec.program import (
  NodalLoadSpec,
  ProgramCoordinateValue,
  ProgramPoint,
)
from pyfem.v3.state import (
  NestedBlockLayout,
  NestedStateLayout,
  NestedStateStore,
  StateTransactionOwner,
)

FE2_SYSTEM_SCHEMA = "pyfem-v3-fe2-continuum-system-v1"
FE2_STATE_SCHEMA = "pyfem-v3-fe2-macro-strain-state-v1"
FE2_TANGENT_CLASS = "homogenized-condensation"
_STRAIN_COORDINATES = ("eps11", "eps22", "gamma12")
_STRESS_COMPONENTS = ("sigma11", "sigma22", "sigma12")
_POINT_COUNT = 9


def _new[ValueT](cls: type[ValueT], /, **fields: object) -> ValueT:
  value = object.__new__(cls)
  for name, field in fields.items():
    object.__setattr__(value, name, field)
  return value


def _fail(code: str, message: str, source: SourceContext) -> None:
  raise ModelCompilationError(
    (ModelCompilationDiagnostic(code=code, message=message, source=source),)
  )


def _point(strain: np.ndarray) -> ProgramPoint:
  return ProgramPoint(
    tuple(
      ProgramCoordinateValue(name, float(value))
      for name, value in zip(_STRAIN_COORDINATES, strain, strict=True)
    )
  )


def _nested_layout(system: CompiledSystem, ip_count: int) -> NestedStateLayout:
  return NestedStateLayout(
    schema=f"pyfem-v3-fe2-nested-{system.content_fingerprint.digest[:16]}",
    coefficient_count=system.coefficient_count,
    blocks=tuple(
      NestedBlockLayout(
        schema=operator.header.state_layout.schema,
        block_id=f"micro-{index}",
        entity_count=operator.header.state_layout.entity_count,
        row_width=operator.header.state_layout.row_width,
      )
      for index, operator in enumerate(system.operators)
    ),
    ip_count=ip_count,
  )


def _seed_store(system: CompiledSystem, store: NestedStateStore) -> None:
  store.begin_trial()
  try:
    for ip_index in range(store.ip_count):
      store.write_trial_coefficients(
        ip_index, np.zeros(system.coefficient_count, dtype=np.float64)
      )
      for operator, block in zip(system.operators, store.layout.blocks, strict=True):
        layout = operator.header.state_layout
        initial_rows = getattr(layout, "initial_rows", None)
        rows = (
          np.zeros(layout.row_shape, dtype=np.float64)
          if initial_rows is None
          else initial_rows.values
        )
        store.write_trial_block(ip_index, block.block_id, rows)
      store.trial_complete(ip_index)
    store.commit_trial()
  except BaseException:
    if store.trial_open:
      store.reject_trial()
    raise


@dataclass(frozen=True, slots=True, eq=False, init=False)
class FE2ContinuumOperator(CompilerConstructed):
  """Macro Q8 geometry with one host-side nested driver per integration point."""

  header: OperatorHeader
  entity_block: object
  payload: Q8ContinuumPayload
  content_manifest: CanonicalManifest
  drivers: tuple[NonlinearStaticDriver, ...]
  nested_store: NestedStateStore
  homogenization: HomogenizationRequest
  base_system_instance: InstanceId

  def _rejected(
    self,
    accepted_state: FinalizedArray,
    status: EvaluationStatus = EvaluationStatus.REJECT_STEP,
  ) -> OperatorEvaluation:
    return _new(
      OperatorEvaluation,
      residual_values=(),
      jacobian_values=(),
      trial_state=accepted_state,
      status=status,
      nested_references=(),
    )

  def evaluate(self, inputs: OperatorEvaluationInput) -> OperatorEvaluation:
    """Solve each micro problem from accepted state and homogenize its response."""
    if type(inputs) is not OperatorEvaluationInput:
      msg = "FE2 evaluation requires an exact immutable evaluation input"
      raise TypeError(msg)
    if type(inputs.port_values) is not tuple or len(inputs.port_values) != 1:
      msg = "FE2 evaluation requires exactly one displacement port batch"
      raise TypeError(msg)
    values = inputs.port_values[0].values
    expected = self.header.ports[0].coefficient_map.values.shape
    if (
      values.dtype != np.dtype(np.float64)
      or values.dtype.metadata is not None
      or values.shape != expected
      or not bool(np.isfinite(values).all())
    ):
      msg = "FE2 displacement port values must be a finite metadata-free float64 batch"
      raise TypeError(msg)
    accepted = inputs.accepted_state
    layout = self.header.state_layout
    if (
      accepted.values.dtype != np.dtype(np.float64)
      or accepted.values.dtype.metadata is not None
      or accepted.values.shape != layout.row_shape
      or not bool(np.isfinite(accepted.values).all())
    ):
      msg = "FE2 accepted state must match the compiled macro-strain layout"
      raise TypeError(msg)
    if inputs.signals or self.header.signal_ports:
      msg = "FE2 model operator does not accept program signal inputs"
      raise ValueError(msg)
    nested = inputs.nested_execution
    if type(nested) is not NestedEvaluationInput:
      msg = "FE2 evaluation requires its declared nested store and no macro owner"
      raise ValueError(msg)
    if nested.store is not self.nested_store:
      msg = "FE2 evaluation rejects a foreign nested store"
      raise ValueError(msg)
    if not nested.store.trial_open:
      msg = "FE2 evaluation requires a driver-opened nested trial scope"
      raise RuntimeError(msg)
    residual_ids = tuple(item.channel_id for item in self.header.residual_channels)
    jacobian_ids = tuple(item.channel_id for item in self.header.jacobian_channels)
    request = inputs.request
    if (
      type(request) is not ChannelRequest
      or request.derivative_channel_ids
      or len(set(request.residual_channel_ids)) != len(request.residual_channel_ids)
      or len(set(request.jacobian_channel_ids)) != len(request.jacobian_channel_ids)
      or not set(request.residual_channel_ids).issubset(residual_ids)
      or not set(request.jacobian_channel_ids).issubset(jacobian_ids)
    ):
      msg = "FE2 evaluation request contains an unavailable or duplicate channel"
      raise ValueError(msg)
    if not request.residual_channel_ids and not request.jacobian_channel_ids:
      return _new(
        OperatorEvaluation,
        residual_values=(),
        jacobian_values=(),
        trial_state=accepted,
        status=EvaluationStatus.OK,
        nested_references=(),
      )

    scales = self.payload.geometry_scales.values
    b_matrix = (
      self.payload.normalized_strain_displacement.values / scales[:, None, None, None]
    )
    weights = (
      self.payload.normalized_integration_weights.values
      * scales[:, None]
      * scales[:, None]
    )
    strain_voigt = np.einsum("epai,ei->epa", b_matrix, values, optimize=True)
    ip_strains = strain_voigt.reshape(layout.entity_count, 3)
    stresses = np.empty((layout.entity_count, 3), dtype=np.float64)
    tangents = np.empty((layout.entity_count, 3, 3), dtype=np.float64)
    store = nested.store
    for ip_index, (driver, strain) in enumerate(
      zip(self.drivers, ip_strains, strict=True)
    ):
      try:
        result = driver.run(
          base_point=_point(accepted.values[ip_index]),
          target_points=(_point(strain),),
          homogenization=self.homogenization,
        )
      except DriverEvaluationError:
        return self._rejected(accepted)
      if result.status is not DriverStatus.COMPLETED or not result.records:
        return self._rejected(accepted)
      record = result.records[-1]
      if record.status is not SubstepStatus.COMMITTED or record.observation is None:
        return self._rejected(accepted)
      observation = record.observation.homogenization
      if observation is None:
        msg = "FE2 nested driver completed without its homogenization observation"
        raise RuntimeError(msg)
      stresses[ip_index] = observation.stress.values
      tangents[ip_index] = observation.tangent.values
      store.trial_complete(ip_index)

    element_count = values.shape[0]
    stress3 = stresses.reshape(element_count, _POINT_COUNT, 3)
    tangent3 = tangents.reshape(element_count, _POINT_COUNT, 3, 3)
    residual = tangent = None
    if request.residual_channel_ids:
      residual = continuum_internal_force_batched(b_matrix, weights, stress3)
    if request.jacobian_channel_ids:
      tangent = np.einsum(
        "ep,epai,epab,epbj->eij",
        weights,
        b_matrix,
        tangent3,
        b_matrix,
        optimize=True,
      )
    if not bool(np.isfinite(stresses).all() and np.isfinite(tangents).all()):
      msg = "FE2 nested response is not representable as finite float64"
      raise ValueError(msg)
    if residual is not None and not bool(np.isfinite(residual).all()):
      msg = "FE2 internal force is not representable as finite float64"
      raise ValueError(msg)
    if tangent is not None and not bool(np.isfinite(tangent).all()):
      msg = "FE2 tangent is not representable as finite float64"
      raise ValueError(msg)
    references = tuple(
      store.trial_reference(ip_index) for ip_index in range(store.ip_count)
    )
    return _new(
      OperatorEvaluation,
      residual_values=(
        (FinalizedArray(residual, dtype=np.float64),) if residual is not None else ()
      ),
      jacobian_values=(
        (FinalizedArray(tangent, dtype=np.float64),) if tangent is not None else ()
      ),
      trial_state=FinalizedArray(ip_strains, dtype=np.float64),
      status=EvaluationStatus.OK,
      nested_references=references,
    )


def _homogenization_manifest(request: HomogenizationRequest) -> dict[str, object]:
  return {
    "strain_coordinates": request.strain_coordinates,
    "reductions": [
      {
        "component_id": reduction.component_id,
        "dofs": reduction.dofs,
        "scale": reduction.scale,
      }
      for reduction in request.reductions
    ],
  }


def _validate_micro_program(
  system: CompiledSystem,
  coordinate_map: CompiledConstraintMap,
  request: HomogenizationRequest,
  source: SourceContext,
) -> None:
  if type(system) is not CompiledSystem:
    msg = "FE2 micro problem requires an exact CompiledSystem"
    raise TypeError(msg)
  if type(coordinate_map) is not CompiledConstraintMap:
    msg = "FE2 micro problem requires an exact CompiledConstraintMap"
    raise TypeError(msg)
  if type(request) is not HomogenizationRequest:
    msg = "FE2 micro problem requires an exact HomogenizationRequest"
    raise TypeError(msg)
  if coordinate_map.coordinate_names != _STRAIN_COORDINATES:
    _fail(
      "invalid-fe2-strain-coordinates",
      "FE2 micro maps must declare eps11, eps22, gamma12 in that order",
      source,
    )
  if (
    request.strain_coordinates != _STRAIN_COORDINATES
    or tuple(reduction.component_id for reduction in request.reductions)
    != _STRESS_COMPONENTS
  ):
    _fail(
      "invalid-fe2-homogenization",
      "FE2 homogenization must request sigma11, sigma22, sigma12 over the "
      "three macro strain coordinates",
      source,
    )


def compile_fe2_continuum(
  base: CompiledSystem,
  *,
  micro_system: CompiledSystem,
  micro_coordinate_map: CompiledConstraintMap,
  homogenization: HomogenizationRequest,
  micro_loads: tuple[NodalLoadSpec, ...] = (),
  settings: NonlinearStaticSettings = NonlinearStaticSettings(),
  implementation_id: str = "pyfem-v3-fe2-continuum-v1",
  source: SourceContext = SourceContext(source="compile.fe2"),
) -> CompiledSystem:
  """Replace one direct Q8 operator with its FE² nested-response counterpart."""
  if type(base) is not CompiledSystem:
    msg = "FE2 compilation requires an exact base CompiledSystem"
    raise TypeError(msg)
  if len(base.operators) != 1 or type(base.operators[0]) is not Q8ContinuumOperator:
    _fail(
      "invalid-fe2-macro-operator",
      "FE2 currently replaces exactly one direct small-strain Q8 continuum operator",
      source,
    )
  if type(settings) is not NonlinearStaticSettings:
    msg = "FE2 micro settings must be exact NonlinearStaticSettings"
    raise TypeError(msg)
  if type(micro_loads) is not tuple or any(
    type(load) is not NodalLoadSpec for load in micro_loads
  ):
    msg = "FE2 micro loads must be an exact NodalLoadSpec tuple"
    raise TypeError(msg)
  if type(implementation_id) is not str or not implementation_id:
    msg = "FE2 implementation id must be a non-empty exact string"
    raise TypeError(msg)
  if type(source) is not SourceContext:
    msg = "FE2 compilation requires an exact SourceContext"
    raise TypeError(msg)
  _validate_micro_program(micro_system, micro_coordinate_map, homogenization, source)
  base_operator = base.operators[0]
  payload = base_operator.payload
  element_count, point_count = payload.normalized_strain_displacement.values.shape[:2]
  if point_count != _POINT_COUNT:
    _fail(
      "invalid-fe2-macro-operator",
      "FE2 requires nine macro Gauss points per Q8",
      source,
    )
  ip_count = element_count * point_count
  store = NestedStateStore(_nested_layout(micro_system, ip_count))
  _seed_store(micro_system, store)
  drivers = tuple(
    NonlinearStaticDriver(
      micro_system,
      micro_coordinate_map,
      micro_loads,
      settings,
      owner=StateTransactionOwner.for_nested_point(
        micro_system, store=store, ip_index=ip_index
      ),
    )
    for ip_index in range(ip_count)
  )
  index_dtype = base_operator.header.ports[0].coefficient_map.values.dtype
  state_layout = _new(
    OperatorStateLayout,
    schema=FE2_STATE_SCHEMA,
    block_id=base_operator.header.block_id,
    entity_count=ip_count,
    slots=(
      _new(
        OperatorStateSlot,
        name="macro-strain",
        width=3,
        dtype=np.dtype(np.float64).str,
        lifetime=StateLifetime.ACCEPTED_TRIAL,
        annotation="engineering-voigt",
      ),
    ),
    entity_offsets=FinalizedArray(
      np.arange(0, 3 * (ip_count + 1), 3, dtype=index_dtype), dtype=index_dtype
    ),
    row_width=3,
    dtype=np.dtype(np.float64).str,
    lifetime=StateLifetime.ACCEPTED_TRIAL,
  )
  residual = _new(
    ResidualChannel,
    channel_id="internal-force",
    target_port_id="displacement",
    balance_role=BalanceRole.INTERNAL,
    linear=False,
  )
  jacobian = _new(
    JacobianChannel,
    channel_id="material-tangent",
    residual_channel_id="internal-force",
    target_port_id="displacement",
    source_port_id="displacement",
    balance_role=BalanceRole.INTERNAL,
    linear=False,
    symmetric=False,
  )
  nested_execution = NestedExecutionDeclaration(
    modality=NestedExecutionModality.DRIVER_PER_INTEGRATION_POINT,
    store=store,
  )
  header = _new(
    OperatorHeader,
    block_id=base_operator.header.block_id,
    entity_block_id=base_operator.header.entity_block_id,
    implementations=(
      *base_operator.header.implementations,
      _new(
        ImplementationIdentity,
        kind="material",
        name="fe2-nested-continuum",
        version="1",
        implementation_id=implementation_id,
      ),
    ),
    ports=base_operator.header.ports,
    signal_ports=(),
    residual_channels=(residual,),
    jacobian_channels=(jacobian,),
    state_layout=state_layout,
    coupling_policy=CouplingPolicy.FIXED,
    parameters=(),
    derivative_channels=(),
    nested_execution=nested_execution,
    tangent_class=FE2_TANGENT_CLASS,
  )
  load_program = drivers[0].plan.loads
  manifest = CanonicalManifest(
    {
      "schema": FE2_SYSTEM_SCHEMA,
      "block_id": header.block_id,
      "implementation_id": implementation_id,
      "nested_execution": nested_execution.modality.value,
      "tangent_class": FE2_TANGENT_CLASS,
      "macro_geometry": {
        "entity_ids": base_operator.entity_block.entity_ids,
        "coefficient_map": header.ports[0].coefficient_map.values,
        "normalized_strain_displacement": payload.normalized_strain_displacement.values,
        "normalized_integration_weights": payload.normalized_integration_weights.values,
        "geometry_scales": payload.geometry_scales.values,
      },
      "macro_state": {
        "schema": FE2_STATE_SCHEMA,
        "entity_count": ip_count,
        "row_width": 3,
      },
      "micro_system": micro_system.provenance.manifest,
      "micro_coordinate_map": micro_coordinate_map.provenance.manifest,
      "micro_loads": {
        "coordinate_names": load_program.coordinate_names,
        "constant": load_program.constant.values,
        "coordinate_coefficients": load_program.coordinate_coefficients.values,
        "load_count": load_program.load_count,
      },
      "homogenization": _homogenization_manifest(homogenization),
      "settings": {
        "tolerance": settings.tolerance,
        "max_iterations": settings.max_iterations,
        "max_cutbacks": settings.max_cutbacks,
        "cutback_factor": settings.cutback_factor,
        "growth_factor": settings.growth_factor,
        "divergence_ratio": settings.divergence_ratio,
        "min_substep_size": settings.min_substep_size,
      },
    }
  )
  operator = _new(
    FE2ContinuumOperator,
    header=header,
    entity_block=base_operator.entity_block,
    payload=payload,
    content_manifest=manifest,
    drivers=drivers,
    nested_store=store,
    homogenization=homogenization,
    base_system_instance=base.instance_id,
  )

  before = store.encode()
  store.begin_trial()
  try:
    probe = operator.evaluate(
      OperatorEvaluationInput(
        port_values=(
          FinalizedArray(
            np.zeros(header.ports[0].coefficient_map.values.shape), dtype=np.float64
          ),
        ),
        accepted_state=FinalizedArray(
          np.zeros(state_layout.row_shape), dtype=np.float64
        ),
        signals=(),
        request=ChannelRequest(("internal-force",), ("material-tangent",)),
        nested_execution=NestedEvaluationInput(store),
      )
    )
    store.validate_trial()
  except (TypeError, ValueError, RuntimeError) as exc:
    _fail(
      "nested-probe-failed",
      f"FE2 virgin nested evaluation failed: {exc}",
      source,
    )
  finally:
    if store.trial_open:
      store.reject_trial()
  if evaluation_status(probe) is not EvaluationStatus.OK:
    _fail(
      "invalid-nested-probe",
      "FE2 virgin nested evaluation must converge",
      source,
    )
  if (
    probe.trial_state.values.tobytes() != np.zeros(state_layout.row_shape).tobytes()
    or len(evaluation_nested_references(probe)) != ip_count
    or any(not bool(np.isfinite(block.values).all()) for block in probe.jacobian_values)
    or store.encode() != before
  ):
    _fail(
      "invalid-nested-probe",
      "FE2 virgin probe must return zero macro strain, complete typed references, "
      "finite tangents, and byte-identical accepted nested state",
      source,
    )
  return compose_fe2_system(base, operator)


def compose_fe2_system(
  base: CompiledSystem,
  operator: FE2ContinuumOperator,
) -> CompiledSystem:
  """Compose the probed FE² operator into its geometry-providing base system."""
  if type(base) is not CompiledSystem or type(operator) is not FE2ContinuumOperator:
    msg = "FE2 composition requires an exact base system and FE2 operator"
    raise TypeError(msg)
  require_same_instance(
    operator.base_system_instance,
    base.instance_id,
    context="FE2 system composition",
  )
  operators = (operator, *base.operators[1:])
  manifest = CanonicalManifest(
    {
      "schema": FE2_SYSTEM_SCHEMA,
      "base_system": base.provenance.manifest,
      "fe2_operator": operator.content_manifest,
    }
  )
  provenance = _new(
    SystemProvenance,
    schema=FE2_SYSTEM_SCHEMA,
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
    point_blocks=base.point_blocks,
    entity_blocks=base.entity_blocks,
    spaces=base.spaces,
    operators=operators,
    source_attribution=base.source_attribution,
  )
