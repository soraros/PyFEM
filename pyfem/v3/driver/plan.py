"""Compiled driver assembly plan: cached sparsity, COO->CSR refill, affine loads.

The plan is the D2 compiled artifact for the nonlinear driver: everything
topological is built once per (compiled system, coordinate map) pair and never
re-derived per Newton iteration.

- Global tangent topology is compiled to a canonical CSR pattern plus a stable
  COO->CSR permutation: ``csr_data = np.add.reduceat(coo_values[perm],
  segment_offsets)`` refills values only, summing duplicate contributions in
  stable element order so every refill is byte-deterministic. The pattern is
  validated once against scipy's own COO->CSR conversion at compile time.
- The internal-force scatter is one concatenated flat index vector consumed by
  a single ``np.bincount`` per evaluation.
- Nodal loads compile to the same affine structure as the coordinate map's
  prescribed offsets: ``f_ext(p) = constant + coefficients @ p``.

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
  require_compatible_system,
)
from pyfem.v3.driver.diagnostics import (
  DriverDiagnostic,
  DriverEvaluationError,
  DriverPreparationError,
)
from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.identity import InstanceId
from pyfem.v3.model.operator import SemanticId
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


def _compile_csr_pattern(
  rows: np.ndarray,
  columns: np.ndarray,
  full_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Canonical CSR pattern plus the stable COO->CSR refill permutation."""
  entry_count = rows.shape[0]
  if entry_count == 0:
    return (
      np.empty(0, dtype=_INDEX_DTYPE),
      np.empty(0, dtype=_INDEX_DTYPE),
      np.zeros(full_count + 1, dtype=_INDEX_DTYPE),
      np.empty(0, dtype=_INDEX_DTYPE),
    )
  permutation = np.lexsort((columns, rows)).astype(_INDEX_DTYPE)
  sorted_rows = rows[permutation]
  sorted_columns = columns[permutation]
  boundary = np.empty(entry_count, dtype=np.bool_)
  boundary[0] = True
  np.not_equal(sorted_rows[1:], sorted_rows[:-1], out=boundary[1:])
  boundary[1:] |= sorted_columns[1:] != sorted_columns[:-1]
  segment_offsets = np.flatnonzero(boundary).astype(_INDEX_DTYPE)
  unique_rows = sorted_rows[segment_offsets]
  csr_indices = sorted_columns[segment_offsets]
  row_counts = np.bincount(unique_rows, minlength=full_count)
  csr_indptr = np.zeros(full_count + 1, dtype=_INDEX_DTYPE)
  np.cumsum(row_counts, out=csr_indptr[1:])
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
  permutation, segment_offsets, csr_indptr, csr_indices = _compile_csr_pattern(
    coo_rows,
    coo_columns,
    system.coefficient_count,
  )
  constant_tangent = all(
    operator.header.jacobian_channels[0].linear for operator in system.operators
  )
  load_program = _compile_loads(system, loads, coordinate_map.coordinate_names)
  manifest = CanonicalManifest(
    {
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
  )
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
    residual_scatter=FinalizedArray(scatter, dtype=_INDEX_DTYPE),
    coo_row_indices=FinalizedArray(coo_rows, dtype=_INDEX_DTYPE),
    coo_column_indices=FinalizedArray(coo_columns, dtype=_INDEX_DTYPE),
    csr_sort_permutation=FinalizedArray(permutation, dtype=_INDEX_DTYPE),
    csr_segment_offsets=FinalizedArray(segment_offsets, dtype=_INDEX_DTYPE),
    csr_indptr=FinalizedArray(csr_indptr, dtype=_INDEX_DTYPE),
    csr_indices=FinalizedArray(csr_indices, dtype=_INDEX_DTYPE),
    loads=load_program,
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


def refill_tangent(
  plan: DriverAssemblyPlan,
  jacobian_batches: tuple[np.ndarray, ...],
) -> csr_matrix:
  """Refill the cached CSR pattern with fresh element tangent values only.

  The topology (pattern, permutation, segment offsets) was compiled once;
  this refill is the D2 values-only Newton reassembly. Duplicate positions are
  summed in stable element order by ``np.add.reduceat``, so identical inputs
  refill byte-identical CSR data.
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
  if plan.coo_entry_count == 0:
    data = np.empty(0, dtype=_FLOATING_DTYPE)
  else:
    data = np.add.reduceat(
      coo_values[plan.csr_sort_permutation.values],
      plan.csr_segment_offsets.values,
    )
  return csr_matrix(
    (data, plan.csr_indices.values, plan.csr_indptr.values),
    shape=plan.csr_shape,
  )
