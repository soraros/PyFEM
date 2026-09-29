"""Compiled affine coordinate maps ``u = P q + u_bar(p)`` for generic systems.

Fixed Dirichlet prescriptions and one-master affine MPC triplets are exact
coordinate transformations, never constraint-row surgery. This package compiles
authored declarations (reused from :mod:`pyfem.v3.spec.program`) against a
generic :class:`~pyfem.v3.model.system.CompiledSystem` into a versioned
:class:`CompiledConstraintMap` artifact:

- ``P`` is a sparse prolongation built once per mesh-plus-constraints and kept
  as immutable CSR plan arrays. Free DOFs carry identity rows, slaves a single
  ``(root master column, root factor)`` entry, and prescribed DOFs empty rows.
- ``u_bar(p) = offset_constant + C_p @ p`` is the prescribed offset at
  structured program coordinates ``p`` (typed time/load/continuation signals).
  Its derivatives ``du_bar/dp_k`` are exactly the compiled columns of ``C_p``,
  so load and continuation factors scale prescribed values without ever
  reconstructing the map.
- The compiler resolves chained ties to their root master and detects
  duplicate or conflicting prescriptions, self ties, dependency cycles, and
  incompatible field references at compile time. Cycles raise the coded
  ``cyclic-affine-constraints`` diagnostic from an iterative depth-first walk;
  no flatten loop can hang.

Reduced channels come from the map, not from row edits: ``r_q = P.T @ r`` and
``K_q = P.T @ K @ P``. Reactions are observed from the FULL residual
``r = f_int - f_ext``: on constrained DOFs the residual is exactly the force
the constraints exert on the structure (its negative is the force on the
supports), well defined at any state. At reduced equilibrium ``P.T @ r = 0``
the full residual itself is the complete constraint-force field, including the
slave contributions redistributed onto free master DOFs; constraint work under
this basis is ``W = f_c . u``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NoReturn

import numpy as np
from scipy.sparse import csr_matrix

from pyfem.v3.constraints.diagnostics import (
  ConstraintEvaluationError,
  ConstraintMapDiagnostic,
  ConstraintMapError,
)
from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.identity import InstanceId, require_same_instance
from pyfem.v3.model.operator import CompilerConstructed, SemanticId
from pyfem.v3.model.provenance import CanonicalManifest, ContentFingerprint
from pyfem.v3.model.system import CompiledSystem, DiscreteSpace
from pyfem.v3.spec.diagnostics import SourceContext, render_diagnostic_value
from pyfem.v3.spec.program import (
  AffineCoefficientSpec,
  AffineTieSpec,
  AffineValueSpec,
  DofRef,
  PrescribedDofSpec,
  ProgramConstraintSpec,
  ProgramCoordinateSpec,
  ProgramCoordinateValue,
  ProgramPoint,
)

CONSTRAINT_MAP_MANIFEST_SCHEMA = "pyfem-v3-constraint-map-v1"

_FLOATING_DTYPE = np.dtype(np.float64)
_INDEX_DTYPE = np.dtype(np.int64)
_COORDINATE_KINDS = ("time", "load", "continuation")


def _new[ValueT](cls: type[ValueT], /, **fields: object) -> ValueT:
  value = object.__new__(cls)
  for name, field in fields.items():
    object.__setattr__(value, name, field)
  return value


def _fail(code: str, message: str, source: SourceContext) -> NoReturn:
  raise ConstraintMapError(
    (ConstraintMapDiagnostic(code=code, message=message, source=source),)
  )


def _evaluation_fail(code: str, message: str) -> NoReturn:
  raise ConstraintEvaluationError(
    (
      ConstraintMapDiagnostic(
        code=code,
        message=message,
        source=SourceContext(),
      ),
    )
  )


@dataclass(frozen=True, slots=True, eq=False, init=False)
class ConstraintMapProvenance(CompilerConstructed):
  """Canonical map meaning and the exact numeric policy behind it."""

  schema: str
  manifest: CanonicalManifest
  floating_dtype: str
  index_dtype: str
  coordinate_names: tuple[str, ...]
  coordinate_kinds: tuple[str, ...]


@dataclass(frozen=True, slots=True, eq=False, init=False)
class CompiledConstraintMap(CompilerConstructed):
  """Immutable compiled affine coordinate map bound to one compiled system.

  The CSR plan arrays are the single source of truth for ``P``; they are
  built once at compile time and never mutated. ``prolongation`` hands out a
  fresh lightweight sparse wrapper per call so caller-side sparse operations
  cannot corrupt the artifact.
  """

  instance_id: InstanceId
  content_fingerprint: ContentFingerprint
  provenance: ConstraintMapProvenance
  compatible_system_instance_id: InstanceId
  compatible_system_content_fingerprint: ContentFingerprint
  full_dof_count: int
  reduced_dof_count: int
  coordinate_names: tuple[str, ...]
  coordinate_kinds: tuple[str, ...]
  free_dofs: FinalizedArray
  constrained_dofs: FinalizedArray
  row_offsets: FinalizedArray
  column_indices: FinalizedArray
  coefficients: FinalizedArray
  offset_constant: FinalizedArray
  offset_coordinate_coefficients: FinalizedArray

  @property
  def coordinate_count(self) -> int:
    """Return the number of declared structured program coordinates."""
    return len(self.coordinate_names)

  def prolongation(self) -> csr_matrix:
    """Return a fresh sparse view of the compiled prolongation ``P``.

    A new ``scipy.sparse.csr_matrix`` wrapper is constructed per call over
    the immutable plan buffers; mutating the wrapper never writes back into
    the artifact.
    """
    return csr_matrix(
      (
        np.array(self.coefficients.values, copy=True),
        np.array(self.column_indices.values, copy=True),
        np.array(self.row_offsets.values, copy=True),
      ),
      shape=(self.full_dof_count, self.reduced_dof_count),
    )


@dataclass(frozen=True, slots=True, eq=False)
class AffineOffsetEvaluation:
  """Prescribed offsets and signal derivatives at one bound program point.

  ``offsets`` is the full-space ``u_bar(p)``. ``derivatives`` has one column
  per declared coordinate in ``coordinate_names`` order; column ``k`` is
  ``du_bar/dp_k``, the first-class continuation/load-factor input.
  """

  coordinate_names: tuple[str, ...]
  coordinate_values: FinalizedArray
  offsets: FinalizedArray
  derivatives: FinalizedArray


@dataclass(frozen=True, slots=True)
class _ResolvedConstraint:
  root_dof: int | None
  root_factor: float
  offset_constant: float
  offset_coefficients: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class _Tie:
  slave: int
  master: int
  factor: float
  offset_constant: float
  offset_coefficients: tuple[float, ...]
  source: SourceContext


def _source_manifest(source: SourceContext) -> dict[str, object]:
  return {
    "source": source.source,
    "line": source.line,
    "column": source.column,
  }


def _dof_manifest(value: DofRef) -> dict[str, object]:
  return {
    "node_id": value.node_id,
    "field_id": value.field_id,
    "component": value.component,
  }


def _affine_manifest(value: AffineValueSpec) -> dict[str, object]:
  return {
    "constant": value.constant,
    "coefficients": [
      {
        "coordinate": item.coordinate,
        "coefficient": item.coefficient,
        "source": _source_manifest(item.source),
      }
      for item in value.coefficients
    ],
    "source": _source_manifest(value.source),
  }


def _finite_exact(value: object, *, label: str, source: SourceContext) -> float:
  if type(value) is not int and type(value) is not float:
    msg = f"{label} must be an exact int or float"
    raise TypeError(msg)
  result = float(value)
  if not math.isfinite(result):
    _fail(
      "non-finite-constraint-value",
      f"{label} must be finite",
      source,
    )
  return result


def _checked_product(value: float, factor: float, *, source: SourceContext) -> float:
  result = value * factor
  if not math.isfinite(result):
    _fail(
      "non-finite-constraint-value",
      "affine tie chain composition overflowed the finite float64 range",
      source,
    )
  return result


def _checked_sum(values: tuple[float, ...], *, source: SourceContext) -> float:
  result = math.fsum(values)
  if not math.isfinite(result):
    _fail(
      "non-finite-constraint-value",
      "affine tie chain composition overflowed the finite float64 range",
      source,
    )
  return result


def _coordinate_plan(
  coordinates: tuple[ProgramCoordinateSpec, ...],
) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, int]]:
  if type(coordinates) is not tuple:
    msg = "constraint map coordinates must be an exact tuple"
    raise TypeError(msg)
  names: list[str] = []
  kinds: list[str] = []
  indices: dict[str, int] = {}
  for coordinate in coordinates:
    if type(coordinate) is not ProgramCoordinateSpec:
      msg = "constraint map coordinates must be exact ProgramCoordinateSpec values"
      raise TypeError(msg)
    name = coordinate.name
    kind = coordinate.kind
    if type(name) is not str or not name:
      _fail(
        "invalid-program-coordinate",
        "program coordinate names must be non-empty exact strings",
        coordinate.source,
      )
    if type(kind) is not str or kind not in _COORDINATE_KINDS:
      _fail(
        "invalid-program-coordinate",
        "program coordinate kinds must be 'time', 'load', or 'continuation'",
        coordinate.source,
      )
    if name in indices:
      _fail(
        "duplicate-program-coordinate",
        f"program coordinate {render_diagnostic_value(name)} is declared twice",
        coordinate.source,
      )
    indices[name] = len(names)
    names.append(name)
    kinds.append(kind)
  return tuple(names), tuple(kinds), indices


def _affine_parts(
  value: AffineValueSpec,
  coordinate_indices: dict[str, int],
) -> tuple[float, tuple[float, ...]]:
  if type(value) is not AffineValueSpec:
    msg = "constraint affine values must be exact AffineValueSpec values"
    raise TypeError(msg)
  if type(value.coefficients) is not tuple:
    msg = "constraint affine value coefficients must be an exact tuple"
    raise TypeError(msg)
  constant = _finite_exact(
    value.constant,
    label="affine value constant",
    source=value.source,
  )
  coefficients = [0.0] * len(coordinate_indices)
  seen: set[str] = set()
  for item in value.coefficients:
    if type(item) is not AffineCoefficientSpec:
      msg = "affine value coefficients must be exact AffineCoefficientSpec values"
      raise TypeError(msg)
    if type(item.coordinate) is not str or not item.coordinate:
      _fail(
        "invalid-program-coordinate",
        "affine coefficient coordinates must be non-empty exact strings",
        item.source,
      )
    index = coordinate_indices.get(item.coordinate)
    if index is None:
      _fail(
        "unknown-program-coordinate",
        "affine coefficient references an undeclared program coordinate "
        f"{render_diagnostic_value(item.coordinate)}",
        item.source,
      )
    if item.coordinate in seen:
      _fail(
        "duplicate-coordinate-coefficient",
        "one affine value carries two coefficients for program coordinate "
        f"{render_diagnostic_value(item.coordinate)}",
        item.source,
      )
    seen.add(item.coordinate)
    coefficients[index] = _finite_exact(
      item.coefficient,
      label="affine coordinate coefficient",
      source=item.source,
    )
  return constant, tuple(coefficients)


class _DofLookup:
  def __init__(self, system: CompiledSystem) -> None:
    self.spaces: dict[SemanticId, DiscreteSpace] = {
      space.space_id: space for space in system.spaces
    }
    self.supports = {block.block_id: block for block in system.point_blocks}
    self.dense_nodes: dict[SemanticId, dict[SemanticId, int]] = {
      space.space_id: {
        node_id: index
        for index, node_id in enumerate(
          self.supports[space.support_block_id].entity_ids
        )
      }
      for space in system.spaces
      if space.support_block_id in self.supports
    }
    self.full_dof_count = system.coefficient_count

  def resolve(self, reference: DofRef, *, source: SourceContext) -> int:
    if type(reference) is not DofRef:
      msg = "constraint DOF references must be exact DofRef values"
      raise TypeError(msg)
    if type(reference.node_id) not in (str, int):
      msg = "constraint DOF reference node ids must be exact semantic ids"
      raise TypeError(msg)
    if type(reference.field_id) not in (str, int):
      msg = "constraint DOF reference field ids must be exact semantic ids"
      raise TypeError(msg)
    if type(reference.component) is not str or not reference.component:
      msg = "constraint DOF reference components must be non-empty exact strings"
      raise TypeError(msg)
    space = self.spaces.get(reference.field_id)
    if space is None:
      _fail(
        "unknown-constraint-space",
        "constraint references field "
        f"{render_diagnostic_value(reference.field_id)} "
        "the compiled system does not have",
        source,
      )
    dense = self.dense_nodes.get(space.space_id)
    if dense is None:
      _fail(
        "unknown-constraint-support-block",
        "constraint space support block is absent from the compiled system",
        source,
      )
    node_index = dense.get(reference.node_id)
    if node_index is None:
      _fail(
        "unknown-constraint-node",
        f"constraint references node {render_diagnostic_value(reference.node_id)} "
        "the compiled system does not have",
        source,
      )
    try:
      component_index = space.components.index(reference.component)
    except ValueError:
      _fail(
        "unknown-constraint-component",
        "constraint references component "
        f"{render_diagnostic_value(reference.component)} "
        f"of field {render_diagnostic_value(reference.field_id)} "
        "the compiled system does not have",
        source,
      )
    return int(space.coefficient_map.values[node_index, component_index])


def _compile_constraints(
  constraints: tuple[ProgramConstraintSpec, ...],
  lookup: _DofLookup,
  coordinate_indices: dict[str, int],
) -> tuple[dict[int, _ResolvedConstraint], dict[int, _Tie]]:
  if type(constraints) is not tuple:
    msg = "constraint map declarations must be an exact tuple"
    raise TypeError(msg)
  prescribed: dict[int, _ResolvedConstraint] = {}
  ties: dict[int, _Tie] = {}
  for constraint in constraints:
    if type(constraint) is PrescribedDofSpec:
      dof = lookup.resolve(constraint.target, source=constraint.source)
      if dof in prescribed:
        _fail(
          "duplicate-prescribed-dof",
          "more than one constraint prescribes the same semantic DOF",
          constraint.source,
        )
      if dof in ties:
        _fail(
          "prescribed-slave-conflict",
          "one semantic DOF cannot be both prescribed and an affine slave",
          constraint.source,
        )
      constant, coefficients = _affine_parts(constraint.value, coordinate_indices)
      prescribed[dof] = _ResolvedConstraint(
        root_dof=None,
        root_factor=0.0,
        offset_constant=constant,
        offset_coefficients=coefficients,
      )
      continue
    if type(constraint) is AffineTieSpec:
      slave = lookup.resolve(constraint.slave, source=constraint.source)
      master = lookup.resolve(constraint.master, source=constraint.source)
      if slave == master:
        _fail(
          "self-affine-tie",
          "an affine tie slave and master must be different semantic DOFs",
          constraint.source,
        )
      if slave in ties:
        _fail(
          "duplicate-affine-slave",
          "more than one affine tie owns the same dependent DOF",
          constraint.source,
        )
      if slave in prescribed:
        _fail(
          "prescribed-slave-conflict",
          "one semantic DOF cannot be both prescribed and an affine slave",
          constraint.source,
        )
      factor = _finite_exact(
        constraint.factor,
        label="affine tie factor",
        source=constraint.source,
      )
      constant, coefficients = _affine_parts(constraint.offset, coordinate_indices)
      ties[slave] = _Tie(
        slave=slave,
        master=master,
        factor=factor,
        offset_constant=constant,
        offset_coefficients=coefficients,
        source=constraint.source,
      )
      continue
    msg = (
      "constraint declarations must be exact PrescribedDofSpec or AffineTieSpec values"
    )
    raise TypeError(msg)
  return prescribed, ties


def _resolve_chains(
  prescribed: dict[int, _ResolvedConstraint],
  ties: dict[int, _Tie],
  coordinate_count: int,
) -> dict[int, _ResolvedConstraint]:
  """Flatten tie chains to root masters with an iterative depth-first walk.

  Every dependent DOF is visited at most twice, so a dependency cycle is
  reported from the tie that closes it instead of hanging in a rewrite loop.
  """
  resolved: dict[int, _ResolvedConstraint] = dict(prescribed)
  visit_state: dict[int, int] = {}
  zeros = tuple(0.0 for _ in range(coordinate_count))
  for start in sorted(ties):
    if visit_state.get(start) == 2:
      continue
    stack: list[tuple[int, bool]] = [(start, False)]
    while stack:
      dof, exiting = stack.pop()
      if exiting:
        tie = ties[dof]
        master = tie.master
        if master in resolved:
          master_value = resolved[master]
        else:
          master_value = _ResolvedConstraint(
            root_dof=master,
            root_factor=1.0,
            offset_constant=0.0,
            offset_coefficients=zeros,
          )
        root_factor = 0.0
        if master_value.root_dof is not None:
          root_factor = _checked_product(
            tie.factor,
            master_value.root_factor,
            source=tie.source,
          )
        offset_constant = _checked_sum(
          (
            tie.offset_constant,
            _checked_product(
              tie.factor,
              master_value.offset_constant,
              source=tie.source,
            ),
          ),
          source=tie.source,
        )
        offset_coefficients = tuple(
          _checked_sum(
            (
              own_coefficient,
              _checked_product(
                tie.factor,
                master_coefficient,
                source=tie.source,
              ),
            ),
            source=tie.source,
          )
          for master_coefficient, own_coefficient in zip(
            master_value.offset_coefficients,
            tie.offset_coefficients,
            strict=True,
          )
        )
        resolved[dof] = _ResolvedConstraint(
          root_dof=master_value.root_dof,
          root_factor=root_factor,
          offset_constant=offset_constant,
          offset_coefficients=offset_coefficients,
        )
        visit_state[dof] = 2
        continue
      state = visit_state.get(dof, 0)
      if state == 2:
        continue
      if state == 1:
        _fail(
          "cyclic-affine-constraints",
          "affine tie declarations contain a dependency cycle",
          ties[dof].source,
        )
      visit_state[dof] = 1
      stack.append((dof, True))
      master = ties[dof].master
      if master in ties:
        master_state = visit_state.get(master, 0)
        if master_state == 1:
          _fail(
            "cyclic-affine-constraints",
            "affine tie declarations contain a dependency cycle",
            ties[dof].source,
          )
        if master_state != 2:
          stack.append((master, False))
  return resolved


def _constraint_manifest(
  constraints: tuple[ProgramConstraintSpec, ...],
) -> list[dict[str, object]]:
  entries: list[dict[str, object]] = []
  for constraint in constraints:
    if type(constraint) is PrescribedDofSpec:
      entries.append(
        {
          "kind": "prescribed-dof",
          "id": constraint.id,
          "target": _dof_manifest(constraint.target),
          "value": _affine_manifest(constraint.value),
          "source": _source_manifest(constraint.source),
        }
      )
    else:
      entries.append(
        {
          "kind": "one-master-affine-tie",
          "id": constraint.id,
          "slave": _dof_manifest(constraint.slave),
          "master": _dof_manifest(constraint.master),
          "factor": float(constraint.factor),
          "offset": _affine_manifest(constraint.offset),
          "source": _source_manifest(constraint.source),
        }
      )
  return entries


def compile_constraint_map(
  system: CompiledSystem,
  constraints: tuple[ProgramConstraintSpec, ...] = (),
  coordinates: tuple[ProgramCoordinateSpec, ...] = (),
) -> CompiledConstraintMap:
  """Compile authored affine constraints into a versioned coordinate map.

  ``constraints`` holds exact ``PrescribedDofSpec`` and ``AffineTieSpec``
  declarations (for example the tuples returned by
  :func:`pyfem.v3.constraints.periodic_ties`); ``coordinates`` declares the
  structured program signals the affine offsets reference. Type-contract
  violations raise ``TypeError``; semantic failures raise
  :class:`ConstraintMapError` with coded source-context diagnostics.
  """
  if type(system) is not CompiledSystem:
    msg = "constraint map compilation requires an exact CompiledSystem"
    raise TypeError(msg)
  coordinate_names, coordinate_kinds, coordinate_indices = _coordinate_plan(coordinates)
  lookup = _DofLookup(system)
  prescribed, ties = _compile_constraints(constraints, lookup, coordinate_indices)
  resolved = _resolve_chains(prescribed, ties, len(coordinate_names))

  full_count = lookup.full_dof_count
  free_dofs = tuple(
    dof for dof in range(full_count) if dof not in prescribed and dof not in ties
  )
  reduced_count = len(free_dofs)
  free_columns = {dof: column for column, dof in enumerate(free_dofs)}
  zeros = tuple(0.0 for _ in coordinate_names)
  row_offsets = [0]
  column_indices: list[int] = []
  coefficients: list[float] = []
  offset_constant: list[float] = []
  offset_coordinate_coefficients: list[tuple[float, ...]] = []
  for dof in range(full_count):
    if dof in free_columns:
      column_indices.append(free_columns[dof])
      coefficients.append(1.0)
      offset_constant.append(0.0)
      offset_coordinate_coefficients.append(zeros)
    else:
      value = resolved[dof]
      if value.root_dof is not None:
        column_indices.append(free_columns[value.root_dof])
        coefficients.append(value.root_factor)
      offset_constant.append(value.offset_constant)
      offset_coordinate_coefficients.append(value.offset_coefficients)
    row_offsets.append(len(column_indices))

  constrained_dofs = tuple(sorted((*prescribed, *ties)))
  coefficient_count = len(coordinate_names)
  manifest = CanonicalManifest(
    {
      "schema": CONSTRAINT_MAP_MANIFEST_SCHEMA,
      "numeric_policy": {
        "floating_dtype": _FLOATING_DTYPE.str,
        "index_dtype": _INDEX_DTYPE.str,
      },
      "compatible_system": str(system.content_fingerprint),
      "coordinates": [
        {"name": name, "kind": kind}
        for name, kind in zip(coordinate_names, coordinate_kinds, strict=True)
      ],
      "spaces": [
        {
          "space_id": space.space_id,
          "support_block_id": space.support_block_id,
          "coefficient_range": space.coefficient_range,
        }
        for space in system.spaces
      ],
      "constraints": _constraint_manifest(constraints),
      "plan": {
        "full_dof_count": full_count,
        "reduced_dof_count": reduced_count,
        "free_dofs": np.asarray(free_dofs, dtype=_INDEX_DTYPE),
        "constrained_dofs": np.asarray(constrained_dofs, dtype=_INDEX_DTYPE),
        "row_offsets": np.asarray(row_offsets, dtype=_INDEX_DTYPE),
        "column_indices": np.asarray(column_indices, dtype=_INDEX_DTYPE),
        "coefficients": np.asarray(coefficients, dtype=_FLOATING_DTYPE),
        "offset_constant": np.asarray(offset_constant, dtype=_FLOATING_DTYPE),
        "offset_coordinate_coefficients": np.asarray(
          offset_coordinate_coefficients, dtype=_FLOATING_DTYPE
        ).reshape(full_count, coefficient_count),
      },
    }
  )
  provenance = _new(
    ConstraintMapProvenance,
    schema=CONSTRAINT_MAP_MANIFEST_SCHEMA,
    manifest=manifest,
    floating_dtype=_FLOATING_DTYPE.str,
    index_dtype=_INDEX_DTYPE.str,
    coordinate_names=coordinate_names,
    coordinate_kinds=coordinate_kinds,
  )
  return _new(
    CompiledConstraintMap,
    instance_id=InstanceId(),
    content_fingerprint=ContentFingerprint.from_manifest(manifest),
    provenance=provenance,
    compatible_system_instance_id=system.instance_id,
    compatible_system_content_fingerprint=system.content_fingerprint,
    full_dof_count=full_count,
    reduced_dof_count=reduced_count,
    coordinate_names=coordinate_names,
    coordinate_kinds=coordinate_kinds,
    free_dofs=FinalizedArray(free_dofs, dtype=_INDEX_DTYPE),
    constrained_dofs=FinalizedArray(constrained_dofs, dtype=_INDEX_DTYPE),
    row_offsets=FinalizedArray(row_offsets, dtype=_INDEX_DTYPE),
    column_indices=FinalizedArray(column_indices, dtype=_INDEX_DTYPE),
    coefficients=FinalizedArray(coefficients, dtype=_FLOATING_DTYPE),
    offset_constant=FinalizedArray(offset_constant, dtype=_FLOATING_DTYPE),
    offset_coordinate_coefficients=FinalizedArray(
      np.asarray(offset_coordinate_coefficients, dtype=_FLOATING_DTYPE).reshape(
        full_count,
        coefficient_count,
      ),
      dtype=_FLOATING_DTYPE,
    ),
  )


def require_compatible_system(
  coordinate_map: CompiledConstraintMap,
  system: CompiledSystem,
) -> None:
  """Require the exact live compiled system the map was compiled against."""
  if type(coordinate_map) is not CompiledConstraintMap:
    msg = "system compatibility requires an exact CompiledConstraintMap"
    raise TypeError(msg)
  if type(system) is not CompiledSystem:
    msg = "system compatibility requires an exact CompiledSystem"
    raise TypeError(msg)
  require_same_instance(
    coordinate_map.compatible_system_instance_id,
    system.instance_id,
    context="constraint map composition",
  )


def _validated_map(value: object) -> CompiledConstraintMap:
  if type(value) is not CompiledConstraintMap:
    msg = "constraint map evaluation requires an exact CompiledConstraintMap"
    raise TypeError(msg)
  return value


def _bound_coordinate_values(
  coordinate_map: CompiledConstraintMap,
  point: ProgramPoint,
) -> tuple[float, ...]:
  if type(point) is not ProgramPoint:
    msg = "offset evaluation requires an exact ProgramPoint binding"
    raise TypeError(msg)
  if type(point.values) is not tuple:
    msg = "program point values must be an exact tuple"
    raise TypeError(msg)
  names = coordinate_map.coordinate_names
  values: dict[str, float] = {}
  for item in point.values:
    if type(item) is not ProgramCoordinateValue:
      msg = "program point values must be exact ProgramCoordinateValue values"
      raise TypeError(msg)
    if type(item.name) is not str or not item.name:
      _evaluation_fail(
        "invalid-program-coordinate",
        "program point coordinate names must be non-empty exact strings",
      )
    if item.name not in names:
      _evaluation_fail(
        "unknown-program-coordinate",
        "program point binds undeclared program coordinate "
        f"{render_diagnostic_value(item.name)}",
      )
    if item.name in values:
      _evaluation_fail(
        "duplicate-program-coordinate",
        "program point binds program coordinate "
        f"{render_diagnostic_value(item.name)} twice",
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
      "program point does not bind declared program coordinate "
      f"{render_diagnostic_value(missing[0])}",
    )
  return tuple(values[name] for name in names)


def evaluate_offsets(
  coordinate_map: CompiledConstraintMap,
  point: ProgramPoint,
) -> AffineOffsetEvaluation:
  """Bind one exact program point into offsets ``u_bar(p)`` and derivatives.

  The derivatives need no recompilation: column ``k`` of the compiled
  coefficient matrix is ``du_bar/dp_k`` for every bound point, so sweeping a
  load or continuation factor only re-binds the point.
  """
  coordinate_map = _validated_map(coordinate_map)
  values = _bound_coordinate_values(coordinate_map, point)
  constants = coordinate_map.offset_constant.values
  coefficient_matrix = coordinate_map.offset_coordinate_coefficients.values
  offsets = np.empty(coordinate_map.full_dof_count, dtype=_FLOATING_DTYPE)
  for dof in range(coordinate_map.full_dof_count):
    offsets[dof] = math.fsum(
      (
        float(constants[dof]),
        *(
          float(coefficient_matrix[dof, index]) * value
          for index, value in enumerate(values)
        ),
      )
    )
  if not bool(np.isfinite(offsets).all()):
    _evaluation_fail(
      "non-finite-coordinate-value",
      "prescribed offset evaluation overflowed the finite float64 range",
    )
  return AffineOffsetEvaluation(
    coordinate_names=coordinate_map.coordinate_names,
    coordinate_values=FinalizedArray(values, dtype=_FLOATING_DTYPE),
    offsets=FinalizedArray(offsets, dtype=_FLOATING_DTYPE),
    derivatives=FinalizedArray(coefficient_matrix, dtype=_FLOATING_DTYPE),
  )


def _vector_values(
  value: object,
  *,
  size: int,
  label: str,
) -> np.ndarray:
  if type(value) is FinalizedArray:
    source = value.values
  elif type(value) is np.ndarray:
    source = value
  else:
    msg = f"{label} must be an exact FinalizedArray or plain ndarray"
    raise TypeError(msg)
  if (
    source.dtype != _FLOATING_DTYPE
    or source.dtype.metadata is not None
    or source.shape != (size,)
  ):
    msg = f"{label} must be a float64 vector of length {size}"
    raise ValueError(msg)
  if not bool(np.isfinite(source).all()):
    msg = f"{label} must be finite"
    raise ValueError(msg)
  return np.asarray(source, dtype=_FLOATING_DTYPE)


def full_coefficients(
  coordinate_map: CompiledConstraintMap,
  reduced: FinalizedArray | np.ndarray,
  point: ProgramPoint,
) -> FinalizedArray:
  """Return the full coefficient vector ``u = P q + u_bar(p)``."""
  coordinate_map = _validated_map(coordinate_map)
  reduced_values = _vector_values(
    reduced,
    size=coordinate_map.reduced_dof_count,
    label="reduced coefficients",
  )
  evaluation = evaluate_offsets(coordinate_map, point)
  full = coordinate_map.prolongation() @ reduced_values
  full = full + evaluation.offsets.values
  return FinalizedArray(full, dtype=_FLOATING_DTYPE)


def admissible_increment(
  coordinate_map: CompiledConstraintMap,
  reduced_increment: FinalizedArray | np.ndarray,
) -> FinalizedArray:
  """Return the full admissible increment ``delta_u = P delta_q``."""
  coordinate_map = _validated_map(coordinate_map)
  increment = _vector_values(
    reduced_increment,
    size=coordinate_map.reduced_dof_count,
    label="reduced increment",
  )
  return FinalizedArray(
    coordinate_map.prolongation() @ increment,
    dtype=_FLOATING_DTYPE,
  )


def reduce_residual(
  coordinate_map: CompiledConstraintMap,
  full_residual: FinalizedArray | np.ndarray,
) -> FinalizedArray:
  """Extract the reduced residual channel ``r_q = P.T @ r``."""
  coordinate_map = _validated_map(coordinate_map)
  residual = _vector_values(
    full_residual,
    size=coordinate_map.full_dof_count,
    label="full residual",
  )
  return FinalizedArray(
    coordinate_map.prolongation().T @ residual,
    dtype=_FLOATING_DTYPE,
  )


def reduce_tangent(
  coordinate_map: CompiledConstraintMap,
  full_tangent: csr_matrix,
) -> csr_matrix:
  """Extract the reduced tangent channel ``K_q = P.T @ K @ P`` (sparse)."""
  coordinate_map = _validated_map(coordinate_map)
  if type(full_tangent) is not csr_matrix:
    msg = "full tangent must be an exact scipy.sparse.csr_matrix"
    raise TypeError(msg)
  shape = (coordinate_map.full_dof_count, coordinate_map.full_dof_count)
  if full_tangent.shape != shape:
    msg = f"full tangent must have shape {shape}"
    raise ValueError(msg)
  prolongation = coordinate_map.prolongation()
  return (prolongation.T @ (full_tangent @ prolongation)).tocsr()


def reaction_forces(
  coordinate_map: CompiledConstraintMap,
  full_residual: FinalizedArray | np.ndarray,
) -> FinalizedArray:
  """Observe direct constraint reactions from the FULL residual.

  With the signed full residual ``r = f_int - f_ext`` the returned vector is
  zero on free DOFs and ``r`` on constrained DOFs: the force the constraints
  exert on the structure (its negative is the force on the supports). At
  reduced equilibrium ``P.T @ r = 0`` the full residual itself is the complete
  constraint-force field, slave contributions already redistributed onto
  their masters, so reduced equilibrium and full-residual reactions agree by
  construction. Constraint work under this basis is ``W = f_c . u``.
  """
  coordinate_map = _validated_map(coordinate_map)
  residual = _vector_values(
    full_residual,
    size=coordinate_map.full_dof_count,
    label="full residual",
  )
  reactions = np.zeros(coordinate_map.full_dof_count, dtype=_FLOATING_DTYPE)
  constrained = coordinate_map.constrained_dofs.values
  reactions[constrained] = residual[constrained]
  return FinalizedArray(reactions, dtype=_FLOATING_DTYPE)
