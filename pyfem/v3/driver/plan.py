"""Compiled driver assembly plan: cached sparsity, COO->CSR refill, affine loads.

The plan is the D2 compiled artifact for the nonlinear driver: everything
topological is built once per (compiled system, coordinate map) pair and never
re-derived per Newton iteration.

- Global tangent topology is compiled to a canonical CSR pattern plus a stable
  COO->CSR permutation (the single canonical compiler,
  ``fem.assembly.compile_csr_pattern``): ``csr_data = dedup_coo_segments(
  coo_values, perm, segment_offsets)`` refills values only, summing duplicate
  contributions strictly left-to-right in stable stream order so every refill
  is byte-deterministic and platform/SIMD-independent (``np.add.reduceat``'s
  per-segment SIMD inner loop is not). The pattern is validated once against
  scipy's own COO->CSR conversion at compile time.
- The internal-force scatter is one concatenated flat index vector consumed by
  a single ``np.bincount`` per evaluation.
- Parameter-sensitivity right-hand sides assemble through the SAME residual
  scatter: a requested parameter set resolves onto the operators' declared
  ``d(residual)/d(parameter)`` channels (``compile_sensitivity_program``,
  failing closed at request time on undeclared or ambiguous parameters), and
  ``assemble_parameter_rhs`` reduces the negated scatter through the
  coordinate map — zero new topology.
- Nodal loads compile to the same affine structure as the coordinate map's
  prescribed offsets: ``f_ext(p) = constant + coefficients @ p``.
- Declared operator signal ports compile to coordinate-index slices: every
  port's signal and derivative coordinates are validated against the
  coordinate map at plan time, and each evaluation forwards the bound point as
  ``ProgramSignalInput`` values (schedule-owned signals). Two binding rules
  are defined — identity binding for a signal id that exactly names a program
  coordinate, and committed-increment derivation for a signal id of the form
  ``d<coordinate>`` (``dtime`` binds the time increment: the bound time minus
  the time at the previous committed point). See ``OperatorSignalPortSlice``
  and ``evaluate_signals`` for the exact rule semantics.

The plan owns no mutable buffers and no factorization; those live in the
driver workspace. Constraint algebra is never re-implemented here: reduced
channels come from the M15 coordinate map API.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NoReturn

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from pyfem.v3.constraints.compile import (
  CompiledConstraintMap,
  reduce_residual,
  require_compatible_system,
)
from pyfem.v3.driver.diagnostics import (
  DriverDiagnostic,
  DriverEvaluationError,
  DriverPreparationError,
)
from pyfem.v3.fem.assembly import compile_csr_pattern, dedup_coo_segments
from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.identity import InstanceId
from pyfem.v3.model.operator import (
  ProgramSignalInput,
  SemanticId,
  SignalDerivativeInput,
)
from pyfem.v3.model.provenance import CanonicalManifest, ContentFingerprint
from pyfem.v3.model.system import CompiledSystem
from pyfem.v3.spec.diagnostics import SourceContext, render_diagnostic_value
from pyfem.v3.spec.program import (
  AffineValueSpec,
  DofRef,
  NodalLoadSpec,
  ProgramCoordinateValue,
  ProgramPoint,
)

DRIVER_ASSEMBLY_PLAN_MANIFEST_SCHEMA = "pyfem-v3-driver-assembly-plan-v1"

_FLOATING_DTYPE = np.dtype(np.float64)
_INDEX_DTYPE = np.dtype(np.int64)

# The two signal binding rules compiled into ``OperatorSignalPortSlice`` values.
_IDENTITY_BINDING = "identity"
_INCREMENT_BINDING = "increment"


def _preparation_fail(code: str, message: str, source: SourceContext) -> NoReturn:
  raise DriverPreparationError(
    (DriverDiagnostic(code=code, message=message, source=source),)
  )


def _evaluation_fail(code: str, message: str) -> NoReturn:
  raise DriverEvaluationError(
    (DriverDiagnostic(code=code, message=message, source=SourceContext()),)
  )


@dataclass(frozen=True, slots=True, eq=False)
class OperatorAssemblySlice:
  """One operator's gather maps and its segment of the plan COO buffers."""

  block_id: SemanticId
  entity_count: int
  element_dof_count: int
  gather: FinalizedArray
  residual_offset: int
  coo_offset: int
  coo_length: int


@dataclass(frozen=True, slots=True, eq=False)
class OperatorSignalPortSlice:
  """One operator signal port resolved onto program coordinate indices.

  Two binding rules exist, selected at plan time by the declared
  ``signal_id``:

  - ``identity`` — the signal id exactly names a program coordinate: the
    forwarded signal value is the bound value of the coordinate at
    ``signal_coordinate_index``.
  - ``increment`` — the signal id does not itself name a coordinate but is
    ``d`` prefixed onto one that does (``dtime`` derives from ``time``): the
    forwarded signal value is the bound value of the BASE coordinate at
    ``signal_coordinate_index`` minus its value at the previous committed
    program point. The rule is generic over base coordinates; the motivating
    case is the time-increment (``dtime``) channel of rate-form laws.

  Both rules forward ``d(signal)/d(p)`` per declared derivative coordinate as
  the Kronecker delta on ``signal_coordinate_index``: under the increment
  rule the previous committed point is constant with respect to the trial
  point, so the derivative is one for the base coordinate and zero for every
  other. Exact coordinate names always win over the ``d``-prefixed derivation
  namespace: a program coordinate literally named ``dtime`` binds identically,
  never derived.
  """

  port_id: str
  signal_id: str
  binding: str
  signal_coordinate_index: int
  derivative_coordinate_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True, eq=False)
class DriverPlanProvenance:
  """Canonical plan meaning and the exact numeric policy behind it."""

  schema: str
  manifest: CanonicalManifest
  floating_dtype: str
  index_dtype: str
  coordinate_names: tuple[str, ...]
  operator_block_ids: tuple[SemanticId, ...]
  constant_tangent: bool


@dataclass(frozen=True, slots=True, eq=False)
class CompiledLoadProgram:
  """Affine full-space external force ``f_ext(p) = constant + C_p @ p``."""

  coordinate_names: tuple[str, ...]
  constant: FinalizedArray
  coordinate_coefficients: FinalizedArray
  load_count: int


@dataclass(frozen=True, slots=True, eq=False)
class OperatorDerivativeSlice:
  """One operator's share of a validated sensitivity request.

  ``parameter_ids`` and ``derivative_channel_ids`` align pairwise in the
  header's declaration order filtered by the request, so the evaluation's
  derivative values index columns directly. Operators declaring no requested
  parameter carry empty tuples: they are never re-evaluated and contribute
  an exact zero batch to every right-hand side.
  """

  parameter_ids: tuple[str, ...]
  derivative_channel_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True, eq=False)
class CompiledSensitivityProgram:
  """One validated sensitivity request resolved onto operator channels.

  ``parameter_ids`` keeps the request order (the observation order);
  ``operator_slices`` aligns with the compiled system's operator order. The
  program is pure request resolution — it owns no topology, so assembly
  plans and their manifests are untouched by sensitivity requests.
  """

  parameter_ids: tuple[str, ...]
  operator_slices: tuple[OperatorDerivativeSlice, ...]


@dataclass(frozen=True, slots=True, eq=False)
class DriverAssemblyPlan:
  """Immutable assembly topology bound to one exact system and map pair."""

  instance_id: InstanceId
  content_fingerprint: ContentFingerprint
  provenance: DriverPlanProvenance
  compatible_system_instance_id: InstanceId
  compatible_system_content_fingerprint: ContentFingerprint
  compatible_map_instance_id: InstanceId
  compatible_map_content_fingerprint: ContentFingerprint
  full_dof_count: int
  reduced_dof_count: int
  coordinate_names: tuple[str, ...]
  constant_tangent: bool
  operator_slices: tuple[OperatorAssemblySlice, ...]
  signal_slices: tuple[tuple[OperatorSignalPortSlice, ...], ...]
  residual_scatter: FinalizedArray
  coo_row_indices: FinalizedArray
  coo_column_indices: FinalizedArray
  csr_sort_permutation: FinalizedArray
  csr_segment_offsets: FinalizedArray
  csr_indptr: FinalizedArray
  csr_indices: FinalizedArray
  loads: CompiledLoadProgram

  @property
  def coo_entry_count(self) -> int:
    """Return the raw COO entry count (with duplicate positions)."""
    return int(self.coo_row_indices.values.shape[0])

  @property
  def csr_shape(self) -> tuple[int, int]:
    """Return the full-space tangent shape."""
    return (self.full_dof_count, self.full_dof_count)

  @property
  def requires_committed_point(self) -> bool:
    """Whether any compiled signal port derives a committed-point increment."""
    return any(
      port.binding == _INCREMENT_BINDING
      for ports in self.signal_slices
      for port in ports
    )


class _LoadDofLookup:
  """Semantic DOF resolver for load declarations (mirrors the map compiler)."""

  def __init__(self, system: CompiledSystem) -> None:
    self.spaces = {space.space_id: space for space in system.spaces}
    supports = {block.block_id: block for block in system.point_blocks}
    self.dense_nodes = {
      space.space_id: {
        node_id: index
        for index, node_id in enumerate(supports[space.support_block_id].entity_ids)
      }
      for space in system.spaces
      if space.support_block_id in supports
    }

  def resolve(self, reference: DofRef, *, source: SourceContext) -> int:
    if type(reference) is not DofRef:
      msg = "load DOF references must be exact DofRef values"
      raise TypeError(msg)
    space = self.spaces.get(reference.field_id)
    if space is None:
      _preparation_fail(
        "unknown-load-space",
        f"load references field {render_diagnostic_value(reference.field_id)} "
        "the compiled system does not have",
        source,
      )
    dense = self.dense_nodes.get(space.space_id)
    if dense is None:
      _preparation_fail(
        "unknown-load-support-block",
        "load space support block is absent from the compiled system",
        source,
      )
    node_index = dense.get(reference.node_id)
    if node_index is None:
      _preparation_fail(
        "unknown-load-node",
        f"load references node {render_diagnostic_value(reference.node_id)} "
        "the compiled system does not have",
        source,
      )
    try:
      component_index = space.components.index(reference.component)
    except ValueError:
      _preparation_fail(
        "unknown-load-component",
        f"load references component {render_diagnostic_value(reference.component)} "
        f"of field {render_diagnostic_value(reference.field_id)} "
        "the compiled system does not have",
        source,
      )
    return int(space.coefficient_map.values[node_index, component_index])


def _finite_exact(value: object, *, label: str, source: SourceContext) -> float:
  if type(value) is not int and type(value) is not float:
    msg = f"{label} must be an exact int or float"
    raise TypeError(msg)
  result = float(value)
  if not math.isfinite(result):
    _preparation_fail(
      "non-finite-load-value",
      f"{label} must be finite",
      source,
    )
  return result


def _compile_loads(
  system: CompiledSystem,
  loads: tuple[NodalLoadSpec, ...],
  coordinate_names: tuple[str, ...],
) -> CompiledLoadProgram:
  if type(loads) is not tuple or any(type(load) is not NodalLoadSpec for load in loads):
    msg = "driver loads must be an exact tuple of NodalLoadSpec values"
    raise TypeError(msg)
  lookup = _LoadDofLookup(system)
  full_count = system.coefficient_count
  coordinate_indices = {name: index for index, name in enumerate(coordinate_names)}
  constant = np.zeros(full_count, dtype=_FLOATING_DTYPE)
  coefficients = np.zeros((full_count, len(coordinate_names)), dtype=_FLOATING_DTYPE)
  for load in loads:
    dof = lookup.resolve(load.target, source=load.source)
    value = load.value
    if type(value) is not AffineValueSpec:
      msg = "load values must be exact AffineValueSpec values"
      raise TypeError(msg)
    if type(value.coefficients) is not tuple:
      msg = "load value coefficients must be an exact tuple"
      raise TypeError(msg)
    constant[dof] += _finite_exact(
      value.constant,
      label="load value constant",
      source=value.source,
    )
    seen: set[str] = set()
    for item in value.coefficients:
      index = coordinate_indices.get(item.coordinate)
      if index is None:
        _preparation_fail(
          "unknown-load-coordinate",
          "load references undeclared program coordinate "
          f"{render_diagnostic_value(item.coordinate)}",
          item.source,
        )
      if item.coordinate in seen:
        _preparation_fail(
          "duplicate-load-coordinate-coefficient",
          "one load carries two coefficients for program coordinate "
          f"{render_diagnostic_value(item.coordinate)}",
          item.source,
        )
      seen.add(item.coordinate)
      coefficients[dof, index] += _finite_exact(
        item.coefficient,
        label="load coordinate coefficient",
        source=item.source,
      )
  return CompiledLoadProgram(
    coordinate_names=coordinate_names,
    constant=FinalizedArray(constant, dtype=_FLOATING_DTYPE),
    coordinate_coefficients=FinalizedArray(coefficients, dtype=_FLOATING_DTYPE),
    load_count=len(loads),
  )


def _compile_operator_slices(
  system: CompiledSystem,
) -> tuple[tuple[OperatorAssemblySlice, ...], np.ndarray, np.ndarray, np.ndarray]:
  if not system.operators:
    _preparation_fail(
      "empty-operator-list",
      "the driver assembly plan requires at least one compiled operator",
      SourceContext(),
    )
  slices: list[OperatorAssemblySlice] = []
  scatter_parts: list[np.ndarray] = []
  row_parts: list[np.ndarray] = []
  column_parts: list[np.ndarray] = []
  residual_offset = 0
  coo_offset = 0
  for operator in system.operators:
    header = operator.header
    if len(header.ports) != 1:
      _preparation_fail(
        "unsupported-operator-port-count",
        "the nonlinear driver supports exactly one coefficient port per operator",
        SourceContext(),
      )
    if len(header.residual_channels) != 1 or len(header.jacobian_channels) != 1:
      _preparation_fail(
        "unsupported-channel-signature",
        "the nonlinear driver supports exactly one residual and one Jacobian "
        "channel per operator",
        SourceContext(),
      )
    port = header.ports[0]
    gather = np.asarray(port.coefficient_map.values, dtype=_INDEX_DTYPE)
    if gather.ndim != 2:
      _preparation_fail(
        "unsupported-gather-shape",
        "operator port coefficient maps must be two-dimensional",
        SourceContext(),
      )
    entity_count, element_dof_count = gather.shape
    scatter_parts.append(gather.reshape(-1))
    rows = np.broadcast_to(gather[:, :, None], (*gather.shape, element_dof_count))
    columns = np.broadcast_to(gather[:, None, :], (*gather.shape, element_dof_count))
    row_parts.append(
      np.array(rows.reshape(-1), dtype=_INDEX_DTYPE, order="C", copy=True)
    )
    column_parts.append(
      np.array(columns.reshape(-1), dtype=_INDEX_DTYPE, order="C", copy=True)
    )
    coo_length = entity_count * element_dof_count * element_dof_count
    slices.append(
      OperatorAssemblySlice(
        block_id=header.block_id,
        entity_count=entity_count,
        element_dof_count=element_dof_count,
        gather=FinalizedArray(gather, dtype=_INDEX_DTYPE),
        residual_offset=residual_offset,
        coo_offset=coo_offset,
        coo_length=coo_length,
      )
    )
    residual_offset += entity_count * element_dof_count
    coo_offset += coo_length
  return (
    tuple(slices),
    np.concatenate(scatter_parts),
    np.concatenate(row_parts),
    np.concatenate(column_parts),
  )


def _compile_signal_slices(
  system: CompiledSystem,
  coordinate_names: tuple[str, ...],
) -> tuple[tuple[OperatorSignalPortSlice, ...], ...]:
  """Resolve every operator's declared signal ports onto coordinate indices.

  Ports reference program coordinates by name; an operator whose port binds an
  undeclared coordinate can never be driven honestly, so plan compilation
  fails closed with coded diagnostics. A port whose signal id does not name a
  coordinate but is ``d`` prefixed onto one declares an increment derivation
  (see ``OperatorSignalPortSlice``); its base coordinate must resolve, again
  fail-closed.
  """
  coordinate_indices = {name: index for index, name in enumerate(coordinate_names)}
  slices: list[tuple[OperatorSignalPortSlice, ...]] = []
  for operator in system.operators:
    ports = operator.header.signal_ports
    if len({port.port_id for port in ports}) != len(ports):
      _preparation_fail(
        "duplicate-signal-port",
        "an operator header declares the same signal port twice",
        SourceContext(),
      )
    resolved: list[OperatorSignalPortSlice] = []
    for port in ports:
      if type(port.port_id) is not str or not port.port_id:
        _preparation_fail(
          "invalid-signal-port",
          "operator signal port ids must be non-empty exact strings",
          SourceContext(),
        )
      signal_id = port.signal_id
      signal_index = (
        coordinate_indices.get(signal_id) if type(signal_id) is str else None
      )
      binding = _IDENTITY_BINDING
      if signal_index is None:
        if type(signal_id) is str and len(signal_id) > 1 and signal_id.startswith("d"):
          base_index = coordinate_indices.get(signal_id[1:])
          if base_index is None:
            _preparation_fail(
              "unknown-derived-signal-base-coordinate",
              f"operator signal port {render_diagnostic_value(port.port_id)} binds "
              f"the derived increment signal {render_diagnostic_value(signal_id)} "
              "whose base coordinate "
              f"{render_diagnostic_value(signal_id[1:])} is not a declared "
              "program coordinate",
              SourceContext(),
            )
          signal_index = base_index
          binding = _INCREMENT_BINDING
        else:
          _preparation_fail(
            "unknown-signal-coordinate",
            f"operator signal port {render_diagnostic_value(port.port_id)} binds "
            "undeclared program coordinate "
            f"{render_diagnostic_value(port.signal_id)}",
            SourceContext(),
          )
      derivative_indices: list[int] = []
      for coordinate in port.derivative_coordinate_ids:
        derivative_index = (
          coordinate_indices.get(coordinate) if type(coordinate) is str else None
        )
        if derivative_index is None:
          _preparation_fail(
            "unknown-signal-derivative-coordinate",
            f"operator signal port {render_diagnostic_value(port.port_id)} "
            "declares a derivative on undeclared program coordinate "
            f"{render_diagnostic_value(coordinate)}",
            SourceContext(),
          )
        if derivative_index in derivative_indices:
          _preparation_fail(
            "duplicate-signal-derivative-coordinate",
            f"operator signal port {render_diagnostic_value(port.port_id)} "
            "declares two derivative channels on program coordinate "
            f"{render_diagnostic_value(coordinate)}",
            SourceContext(),
          )
        derivative_indices.append(derivative_index)
      resolved.append(
        OperatorSignalPortSlice(
          port_id=port.port_id,
          signal_id=str(signal_id),
          binding=binding,
          signal_coordinate_index=signal_index,
          derivative_coordinate_indices=tuple(derivative_indices),
        )
      )
    slices.append(tuple(resolved))
  return tuple(slices)


def _compile_csr_pattern(
  rows: np.ndarray,
  columns: np.ndarray,
  full_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Canonical CSR pattern plus the stable COO->CSR refill permutation.

  The pattern itself comes from the single canonical compiler
  (``fem.assembly.compile_csr_pattern``); plan compilation additionally
  re-validates it against scipy's own COO->CSR conversion at runtime and
  fails closed on any disagreement, so a plan can never carry a topology
  scipy would not reproduce.
  """
  pattern = compile_csr_pattern(rows, columns, (full_count, full_count))
  permutation = pattern.permutation.astype(_INDEX_DTYPE)
  segment_offsets = pattern.segment_offsets.astype(_INDEX_DTYPE)
  csr_indptr = pattern.indptr.astype(_INDEX_DTYPE)
  csr_indices = pattern.indices.astype(_INDEX_DTYPE)
  entry_count = rows.shape[0]
  if entry_count == 0:
    return permutation, segment_offsets, csr_indptr, csr_indices
  reference = coo_matrix(
    (np.ones(entry_count, dtype=_FLOATING_DTYPE), (rows, columns)),
    shape=(full_count, full_count),
  ).tocsr()
  reference.sort_indices()
  if not np.array_equal(reference.indptr.astype(_INDEX_DTYPE), csr_indptr):
    _preparation_fail(
      "csr-pattern-mismatch",
      "compiled CSR row offsets contradict scipy's canonical COO conversion",
      SourceContext(),
    )
  if not np.array_equal(reference.indices.astype(_INDEX_DTYPE), csr_indices):
    _preparation_fail(
      "csr-pattern-mismatch",
      "compiled CSR column indices contradict scipy's canonical COO conversion",
      SourceContext(),
    )
  return permutation, segment_offsets, csr_indptr, csr_indices


def _signal_manifest_entry(
  port: OperatorSignalPortSlice,
  coordinate_names: tuple[str, ...],
) -> dict[str, object]:
  """Serialize one resolved signal port for the plan manifest.

  Identity entries keep the exact M29 shape (``port_id``,
  ``signal_coordinate``, ``derivative_coordinates``) so identity-only plans
  stay byte-identical; increment entries name the declared signal id, the
  binding rule, and the base coordinate explicitly.
  """
  if port.binding == _INCREMENT_BINDING:
    return {
      "port_id": port.port_id,
      "signal_id": port.signal_id,
      "binding": port.binding,
      "base_coordinate": coordinate_names[port.signal_coordinate_index],
      "derivative_coordinates": [
        coordinate_names[index] for index in port.derivative_coordinate_indices
      ],
    }
  return {
    "port_id": port.port_id,
    "signal_coordinate": (coordinate_names[port.signal_coordinate_index]),
    "derivative_coordinates": [
      coordinate_names[index] for index in port.derivative_coordinate_indices
    ],
  }


def compile_driver_plan(
  system: CompiledSystem,
  coordinate_map: CompiledConstraintMap,
  loads: tuple[NodalLoadSpec, ...] = (),
) -> DriverAssemblyPlan:
  """Compile the immutable assembly plan for one exact system and map pair."""
  if type(system) is not CompiledSystem:
    msg = "driver plan compilation requires an exact CompiledSystem"
    raise TypeError(msg)
  if type(coordinate_map) is not CompiledConstraintMap:
    msg = "driver plan compilation requires an exact CompiledConstraintMap"
    raise TypeError(msg)
  require_compatible_system(coordinate_map, system)
  if system.coefficient_count != coordinate_map.full_dof_count:
    _preparation_fail(
      "dof-count-mismatch",
      "system coefficient count and map full DOF count disagree",
      SourceContext(),
    )
  slices, scatter, coo_rows, coo_columns = _compile_operator_slices(system)
  signal_slices = _compile_signal_slices(system, coordinate_map.coordinate_names)
  permutation, segment_offsets, csr_indptr, csr_indices = _compile_csr_pattern(
    coo_rows,
    coo_columns,
    system.coefficient_count,
  )
  constant_tangent = all(
    operator.header.jacobian_channels[0].linear for operator in system.operators
  )
  load_program = _compile_loads(system, loads, coordinate_map.coordinate_names)
  manifest_content: dict[str, object] = {
    "schema": DRIVER_ASSEMBLY_PLAN_MANIFEST_SCHEMA,
    "numeric_policy": {
      "floating_dtype": _FLOATING_DTYPE.str,
      "index_dtype": _INDEX_DTYPE.str,
    },
    "compatible_system": str(system.content_fingerprint),
    "compatible_map": str(coordinate_map.content_fingerprint),
    "coordinates": list(coordinate_map.coordinate_names),
    "constant_tangent": constant_tangent,
    "operators": [
      {
        "block_id": slice_.block_id,
        "entity_count": slice_.entity_count,
        "element_dof_count": slice_.element_dof_count,
        "gather": slice_.gather.values,
      }
      for slice_ in slices
    ],
    "residual_scatter": scatter,
    "coo_row_indices": coo_rows,
    "coo_column_indices": coo_columns,
    "csr_sort_permutation": permutation,
    "csr_segment_offsets": segment_offsets,
    "csr_indptr": csr_indptr,
    "csr_indices": csr_indices,
    "loads": {
      "constant": load_program.constant.values,
      "coordinate_coefficients": load_program.coordinate_coefficients.values,
      "load_count": load_program.load_count,
    },
  }
  if any(signal_slices):
    manifest_content["signals"] = [
      {
        "block_id": slice_.block_id,
        "ports": [
          _signal_manifest_entry(port, coordinate_map.coordinate_names)
          for port in ports
        ],
      }
      for slice_, ports in zip(slices, signal_slices, strict=True)
      if ports
    ]
  manifest = CanonicalManifest(manifest_content)
  provenance = DriverPlanProvenance(
    schema=DRIVER_ASSEMBLY_PLAN_MANIFEST_SCHEMA,
    manifest=manifest,
    floating_dtype=_FLOATING_DTYPE.str,
    index_dtype=_INDEX_DTYPE.str,
    coordinate_names=coordinate_map.coordinate_names,
    operator_block_ids=tuple(slice_.block_id for slice_ in slices),
    constant_tangent=constant_tangent,
  )
  return DriverAssemblyPlan(
    instance_id=InstanceId(),
    content_fingerprint=ContentFingerprint.from_manifest(manifest),
    provenance=provenance,
    compatible_system_instance_id=system.instance_id,
    compatible_system_content_fingerprint=system.content_fingerprint,
    compatible_map_instance_id=coordinate_map.instance_id,
    compatible_map_content_fingerprint=coordinate_map.content_fingerprint,
    full_dof_count=coordinate_map.full_dof_count,
    reduced_dof_count=coordinate_map.reduced_dof_count,
    coordinate_names=coordinate_map.coordinate_names,
    constant_tangent=constant_tangent,
    operator_slices=slices,
    signal_slices=signal_slices,
    residual_scatter=FinalizedArray(scatter, dtype=_INDEX_DTYPE),
    coo_row_indices=FinalizedArray(coo_rows, dtype=_INDEX_DTYPE),
    coo_column_indices=FinalizedArray(coo_columns, dtype=_INDEX_DTYPE),
    csr_sort_permutation=FinalizedArray(permutation, dtype=_INDEX_DTYPE),
    csr_segment_offsets=FinalizedArray(segment_offsets, dtype=_INDEX_DTYPE),
    csr_indptr=FinalizedArray(csr_indptr, dtype=_INDEX_DTYPE),
    csr_indices=FinalizedArray(csr_indices, dtype=_INDEX_DTYPE),
    loads=load_program,
  )


def compile_sensitivity_program(
  system: CompiledSystem,
  parameter_ids: tuple[str, ...],
) -> CompiledSensitivityProgram:
  """Resolve one requested parameter set onto operator derivative channels.

  The resolution keys on the operators' declared ``ResidualDerivativeChannel``
  values (headers predating the channel read as empty — the
  ``OperatorStateLayout.initial_rows`` getattr precedent), keeping each
  operator's header declaration order. Request-time validation fails closed
  with coded diagnostics: a parameter no operator declares a derivative
  channel for, a channel not differentiating the operator's residual channel,
  or two channels of one operator answering the same parameter (which would
  double-count in the scatter) can never assemble an honest right-hand side,
  so the request is rejected before any substep runs.
  """
  if type(system) is not CompiledSystem:
    msg = "sensitivity program compilation requires an exact CompiledSystem"
    raise TypeError(msg)
  if (
    type(parameter_ids) is not tuple
    or not parameter_ids
    or any(type(parameter_id) is not str for parameter_id in parameter_ids)
  ):
    msg = "sensitivity parameters must be a non-empty exact tuple of exact strings"
    raise TypeError(msg)
  requested: set[str] = set()
  for parameter_id in parameter_ids:
    if parameter_id in requested:
      _preparation_fail(
        "duplicate-sensitivity-parameter",
        f"sensitivity parameter {render_diagnostic_value(parameter_id)} is "
        "requested twice",
        SourceContext(),
      )
    requested.add(parameter_id)
  slices: list[OperatorDerivativeSlice] = []
  declared: list[str] = []
  for operator in system.operators:
    header = operator.header
    channels = getattr(header, "derivative_channels", ())
    matched_ids: list[str] = []
    matched_channel_ids: list[str] = []
    for channel in channels:
      parameter_id = channel.parameter_id
      if type(parameter_id) is str and parameter_id not in declared:
        declared.append(parameter_id)
      if parameter_id not in requested:
        continue
      if (
        len(header.residual_channels) != 1
        or channel.residual_channel_id != header.residual_channels[0].channel_id
      ):
        _preparation_fail(
          "unsupported-derivative-channel-target",
          f"derivative channel {render_diagnostic_value(channel.channel_id)} "
          "does not differentiate the operator's single residual channel",
          SourceContext(),
        )
      if parameter_id in matched_ids:
        _preparation_fail(
          "duplicate-derivative-channel",
          f"an operator declares two derivative channels for parameter "
          f"{render_diagnostic_value(parameter_id)}",
          SourceContext(),
        )
      matched_ids.append(parameter_id)
      matched_channel_ids.append(channel.channel_id)
    slices.append(
      OperatorDerivativeSlice(
        parameter_ids=tuple(matched_ids),
        derivative_channel_ids=tuple(matched_channel_ids),
      )
    )
  for parameter_id in parameter_ids:
    if not any(parameter_id in slice_.parameter_ids for slice_ in slices):
      _preparation_fail(
        "unknown-sensitivity-parameter",
        f"sensitivity parameter {render_diagnostic_value(parameter_id)} is "
        "declared by no operator's derivative channels (declared: "
        f"{render_diagnostic_value(tuple(declared))})",
        SourceContext(),
      )
  return CompiledSensitivityProgram(
    parameter_ids=parameter_ids,
    operator_slices=tuple(slices),
  )


def _bound_coordinate_values(
  loads: CompiledLoadProgram,
  point: ProgramPoint,
) -> tuple[float, ...]:
  if type(point) is not ProgramPoint:
    msg = "load evaluation requires an exact ProgramPoint binding"
    raise TypeError(msg)
  if type(point.values) is not tuple:
    msg = "program point values must be an exact tuple"
    raise TypeError(msg)
  names = loads.coordinate_names
  values: dict[str, float] = {}
  for item in point.values:
    if type(item) is not ProgramCoordinateValue:
      msg = "program point values must be exact ProgramCoordinateValue values"
      raise TypeError(msg)
    if item.name not in names:
      _evaluation_fail(
        "unknown-program-coordinate",
        f"program point binds undeclared load coordinate {item.name!r}",
      )
    if item.name in values:
      _evaluation_fail(
        "duplicate-program-coordinate",
        f"program point binds load coordinate {item.name!r} twice",
      )
    if type(item.value) is not int and type(item.value) is not float:
      msg = "program point coordinate values must be exact int or float numbers"
      raise TypeError(msg)
    value = float(item.value)
    if not math.isfinite(value):
      _evaluation_fail(
        "non-finite-coordinate-value",
        "program point coordinate values must be finite",
      )
    values[item.name] = value
  missing = [name for name in names if name not in values]
  if missing:
    _evaluation_fail(
      "missing-program-coordinate",
      f"program point does not bind declared load coordinate {missing[0]!r}",
    )
  return tuple(values[name] for name in names)


def evaluate_loads(
  loads: CompiledLoadProgram,
  point: ProgramPoint,
) -> FinalizedArray:
  """Return the full external force ``f_ext(p)`` at one bound program point."""
  if type(loads) is not CompiledLoadProgram:
    msg = "load evaluation requires an exact CompiledLoadProgram"
    raise TypeError(msg)
  values = _bound_coordinate_values(loads, point)
  result = loads.constant.values + loads.coordinate_coefficients.values @ np.asarray(
    values,
    dtype=_FLOATING_DTYPE,
  )
  if not bool(np.isfinite(result).all()):
    _evaluation_fail(
      "non-finite-coordinate-value",
      "external force evaluation overflowed the finite float64 range",
    )
  return FinalizedArray(result, dtype=_FLOATING_DTYPE)


def evaluate_signals(
  plan: DriverAssemblyPlan,
  point: ProgramPoint,
  committed_point: ProgramPoint | None = None,
) -> tuple[tuple[ProgramSignalInput, ...], ...]:
  """Forward one bound program point as per-operator program signal inputs.

  Identity-bound ports (the signal id names a program coordinate) forward the
  bound value of the port's signal coordinate, and every declared derivative
  channel carries ``d(signal)/d(p)`` as the Kronecker delta (one for the
  signal's own coordinate, zero otherwise). Increment-bound ports (the signal
  id is ``d`` prefixed onto a declared base coordinate — ``dtime`` on
  ``time``) forward the bound value of the base coordinate minus its value at
  ``committed_point``, the previous committed program point; their derivative
  channels carry the same delta on the BASE coordinate, because the committed
  point is constant with respect to the trial point.

  The driver supplies the run's base point as the committed reference for the
  first substep, so the first increment derives from the base; a derivation
  requested without any committed reference fails closed with
  ``missing-committed-point``. Operators without declared signal ports receive
  an empty tuple, a plan with no signal ports at all never touches the point,
  and a plan without increment bindings never touches ``committed_point``.
  """
  if type(plan) is not DriverAssemblyPlan:
    msg = "signal evaluation requires an exact DriverAssemblyPlan"
    raise TypeError(msg)
  slices = plan.signal_slices
  if not any(slices):
    return tuple(() for _ in slices)
  values = _bound_coordinate_values(plan.loads, point)
  committed: tuple[float, ...] | None = None
  if plan.requires_committed_point:
    if committed_point is None:
      _evaluation_fail(
        "missing-committed-point",
        "increment-bound signal ports require the previous committed program "
        "point; at the first substep the run's base point is that reference",
      )
    committed = _bound_coordinate_values(plan.loads, committed_point)
  forwarded: list[tuple[ProgramSignalInput, ...]] = []
  for ports in slices:
    operator_signals: list[ProgramSignalInput] = []
    for port in ports:
      coordinate_index = port.signal_coordinate_index
      signal_value = values[coordinate_index]
      if port.binding == _INCREMENT_BINDING:
        if committed is None:
          _evaluation_fail(
            "missing-committed-point",
            "increment-bound signal ports require the previous committed program point",
          )
        signal_value -= committed[coordinate_index]
        if not math.isfinite(signal_value):
          _evaluation_fail(
            "non-finite-coordinate-value",
            "derived signal evaluation overflowed the finite float64 range",
          )
      operator_signals.append(
        ProgramSignalInput(
          port_id=port.port_id,
          values=FinalizedArray(
            np.array([signal_value], dtype=_FLOATING_DTYPE),
            dtype=_FLOATING_DTYPE,
          ),
          derivatives=tuple(
            SignalDerivativeInput(
              plan.coordinate_names[index],
              FinalizedArray(
                np.array(
                  [1.0 if index == port.signal_coordinate_index else 0.0],
                  dtype=_FLOATING_DTYPE,
                ),
                dtype=_FLOATING_DTYPE,
              ),
            )
            for index in port.derivative_coordinate_indices
          ),
        )
      )
    forwarded.append(tuple(operator_signals))
  return tuple(forwarded)


def assemble_internal_force(
  plan: DriverAssemblyPlan,
  residual_batches: tuple[np.ndarray, ...],
) -> np.ndarray:
  """Scatter operator residual batches into the full internal force vector.

  One flat ``np.bincount`` over the compiled scatter indices; summation order
  is fixed by the plan, so repeated assemblies are byte-deterministic.
  """
  if type(plan) is not DriverAssemblyPlan:
    msg = "internal force assembly requires an exact DriverAssemblyPlan"
    raise TypeError(msg)
  if type(residual_batches) is not tuple or len(residual_batches) != len(
    plan.operator_slices
  ):
    msg = "residual batches must be an exact tuple with one batch per operator"
    raise TypeError(msg)
  flat = np.empty(plan.residual_scatter.values.shape[0], dtype=_FLOATING_DTYPE)
  for slice_, batch in zip(plan.operator_slices, residual_batches, strict=True):
    if (
      type(batch) is not np.ndarray
      or batch.dtype != _FLOATING_DTYPE
      or batch.shape != (slice_.entity_count, slice_.element_dof_count)
    ):
      msg = "residual batches must be float64 arrays matching the compiled gather"
      raise TypeError(msg)
    end = slice_.residual_offset + slice_.entity_count * slice_.element_dof_count
    flat[slice_.residual_offset : end] = batch.reshape(-1)
  return np.bincount(
    plan.residual_scatter.values,
    weights=flat,
    minlength=plan.full_dof_count,
  )


def assemble_parameter_rhs(
  plan: DriverAssemblyPlan,
  coordinate_map: CompiledConstraintMap,
  derivative_batches: tuple[np.ndarray, ...],
) -> FinalizedArray:
  """Assemble one parameter's reduced IFT right-hand side ``-P.T @ dR/dp``.

  Derivative channel values follow the referenced residual channel's
  element-batch layout exactly (the M56 channel contract), so they scatter
  through the exact internal-force machinery — one flat ``np.bincount`` over
  the compiled scatter indices, no parallel assembly path — and then reduce
  through the coordinate map. The sign is the implicit function theorem's:
  at the converged state ``K_q @ (dq/dp) = -P.T @ dR/dp``.
  """
  if type(coordinate_map) is not CompiledConstraintMap:
    msg = "parameter right-hand side assembly requires an exact CompiledConstraintMap"
    raise TypeError(msg)
  full_derivative = assemble_internal_force(plan, derivative_batches)
  return reduce_residual(coordinate_map, np.negative(full_derivative))


def refill_tangent(
  plan: DriverAssemblyPlan,
  jacobian_batches: tuple[np.ndarray, ...],
) -> csr_matrix:
  """Refill the cached CSR pattern with fresh element tangent values only.

  The topology (pattern, permutation, segment offsets) was compiled once;
  this refill is the D2 values-only Newton reassembly. Duplicate positions
  are summed by ``fem.assembly.dedup_coo_segments`` strictly left-to-right
  in stable stream order — the canonical v3 accumulation order — so
  identical inputs refill byte-identical CSR data on any platform and at
  any thread count.
  """
  if type(plan) is not DriverAssemblyPlan:
    msg = "tangent refill requires an exact DriverAssemblyPlan"
    raise TypeError(msg)
  if type(jacobian_batches) is not tuple or len(jacobian_batches) != len(
    plan.operator_slices
  ):
    msg = "jacobian batches must be an exact tuple with one batch per operator"
    raise TypeError(msg)
  coo_values = np.empty(plan.coo_entry_count, dtype=_FLOATING_DTYPE)
  for slice_, batch in zip(plan.operator_slices, jacobian_batches, strict=True):
    if (
      type(batch) is not np.ndarray
      or batch.dtype != _FLOATING_DTYPE
      or batch.shape
      != (slice_.entity_count, slice_.element_dof_count, slice_.element_dof_count)
    ):
      msg = "jacobian batches must be float64 arrays matching the compiled gather"
      raise TypeError(msg)
    end = slice_.coo_offset + slice_.coo_length
    coo_values[slice_.coo_offset : end] = batch.reshape(-1)
  data = dedup_coo_segments(
    coo_values,
    plan.csr_sort_permutation.values,
    plan.csr_segment_offsets.values,
  )
  return csr_matrix(
    (data, plan.csr_indices.values, plan.csr_indptr.values),
    shape=plan.csr_shape,
  )
