"""Periodic-cell declarations for RVE homogenization and FE² micro problems."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from pyfem.v3.compile.diagnostics import (
  ModelCompilationDiagnostic,
  ModelCompilationError,
)
from pyfem.v3.constraints import periodic_ties
from pyfem.v3.driver.contracts import HomogenizationReduction, HomogenizationRequest
from pyfem.v3.model.operator import SemanticId
from pyfem.v3.model.system import CompiledSystem
from pyfem.v3.spec.diagnostics import SourceContext
from pyfem.v3.spec.model import MeshSpec
from pyfem.v3.spec.program import (
  AffineCoefficientSpec,
  AffineValueSpec,
  DofRef,
  PrescribedDofSpec,
  ProgramConstraintSpec,
  ProgramCoordinateSpec,
)

STRAIN_COORDINATES = ("eps11", "eps22", "gamma12")


@dataclass(frozen=True, slots=True)
class PeriodicCell:
  """Validated boundary groups and their landed periodic-cell declarations."""

  groups: tuple[
    tuple[SemanticId, ...],
    tuple[SemanticId, ...],
    tuple[SemanticId, ...],
    tuple[SemanticId, ...],
  ]
  width: float
  height: float
  coordinates: tuple[ProgramCoordinateSpec, ...]
  constraints: tuple[ProgramConstraintSpec, ...]

  def homogenization_request(self, system: CompiledSystem) -> HomogenizationRequest:
    """Resolve the legacy Right/Top reaction reductions on a compiled system."""
    if type(system) is not CompiledSystem:
      msg = "periodic-cell homogenization requires an exact CompiledSystem"
      raise TypeError(msg)
    _, right, _, top = self.groups
    return HomogenizationRequest(
      strain_coordinates=STRAIN_COORDINATES,
      reductions=(
        HomogenizationReduction(
          "sigma11", _group_dofs(system, right, "x"), 1.0 / self.height
        ),
        HomogenizationReduction(
          "sigma22", _group_dofs(system, top, "y"), 1.0 / self.width
        ),
        HomogenizationReduction(
          "sigma12", _group_dofs(system, top, "x"), 1.0 / self.height
        ),
      ),
    )


def _fail(code: str, message: str, source: SourceContext) -> None:
  raise ModelCompilationError(
    (ModelCompilationDiagnostic(code=code, message=message, source=source),)
  )


def _size(value: object, *, label: str, source: SourceContext) -> float:
  if type(value) not in (int, float):
    msg = f"periodic-cell {label} must be an exact number"
    raise TypeError(msg)
  try:
    converted = float(value)
  except OverflowError:
    converted = math.inf
  if not math.isfinite(converted) or converted <= 0.0:
    _fail(
      "invalid-periodic-cell-size",
      f"periodic-cell {label} must be a positive finite float64",
      source,
    )
  return converted


def _affine(*pairs: tuple[str, float]) -> AffineValueSpec:
  return AffineValueSpec(
    coefficients=tuple(
      AffineCoefficientSpec(name, coefficient) for name, coefficient in pairs
    )
  )


def _prescribed(
  node: SemanticId,
  component: str,
  value: AffineValueSpec,
  *,
  field_id: SemanticId,
  source: SourceContext,
) -> PrescribedDofSpec:
  return PrescribedDofSpec(
    id=f"corner-{node}-{component}",
    target=DofRef(node_id=node, field_id=field_id, component=component),
    value=value,
    source=source,
  )


def _boundary_groups(
  mesh: MeshSpec,
  width: float,
  height: float,
  source: SourceContext,
) -> tuple[
  tuple[SemanticId, ...],
  tuple[SemanticId, ...],
  tuple[SemanticId, ...],
  tuple[SemanticId, ...],
]:
  tolerance = 1.0e-9 * max(width, height)
  coordinates = {node.id: node.coordinates for node in mesh.nodes}

  def group(
    predicate: Callable[[tuple[float, float]], bool],
    key: int,
  ) -> tuple[SemanticId, ...]:
    ids = tuple(node.id for node in mesh.nodes if predicate(node.coordinates))
    return tuple(sorted(ids, key=lambda node_id: coordinates[node_id][key]))

  left = group(lambda c: abs(c[0]) < tolerance, 1)
  right = group(lambda c: abs(c[0] - width) < tolerance, 1)
  bottom = group(lambda c: abs(c[1]) < tolerance, 0)
  top = group(lambda c: abs(c[1] - height) < tolerance, 0)
  if not (len(left) == len(right) >= 2 and len(bottom) == len(top) >= 2):
    _fail(
      "periodic-boundary-count",
      "Left/Right and Bottom/Top groups must pair one-to-one with at least "
      "two nodes on every side",
      source,
    )
  for node_left, node_right in zip(left, right, strict=True):
    if abs(coordinates[node_right][1] - coordinates[node_left][1]) >= tolerance:
      _fail(
        "periodic-boundary-alignment",
        f"Left node {node_left!r} and Right node {node_right!r} do not align in y",
        source,
      )
    x_offset = coordinates[node_right][0] - coordinates[node_left][0]
    if abs(x_offset - width) >= tolerance:
      _fail(
        "periodic-boundary-offset",
        f"Left/Right pair ({node_left!r}, {node_right!r}) does not have the "
        "declared constant cell-width offset",
        source,
      )
  for node_bottom, node_top in zip(bottom, top, strict=True):
    if abs(coordinates[node_top][0] - coordinates[node_bottom][0]) >= tolerance:
      _fail(
        "periodic-boundary-alignment",
        f"Bottom node {node_bottom!r} and Top node {node_top!r} do not align in x",
        source,
      )
    y_offset = coordinates[node_top][1] - coordinates[node_bottom][1]
    if abs(y_offset - height) >= tolerance:
      _fail(
        "periodic-boundary-offset",
        f"Bottom/Top pair ({node_bottom!r}, {node_top!r}) does not have the "
        "declared constant cell-height offset",
        source,
      )
  if not (
    left[0] == bottom[0]
    and right[0] == bottom[-1]
    and left[-1] == top[0]
    and right[-1] == top[-1]
  ):
    _fail(
      "periodic-corner-mismatch",
      "Left/Right/Top/Bottom groups must share the four rectangle corners",
      source,
    )
  return left, right, bottom, top


def _group_dofs(
  system: CompiledSystem,
  node_ids: tuple[SemanticId, ...],
  component: str,
) -> tuple[int, ...]:
  (space,) = system.spaces
  supports = {block.block_id: block for block in system.point_blocks}
  node_index = {
    node_id: index
    for index, node_id in enumerate(supports[space.support_block_id].entity_ids)
  }
  component_index = space.components.index(component)
  coefficient_map = space.coefficient_map.values
  return tuple(
    int(coefficient_map[node_index[node], component_index]) for node in node_ids
  )


def periodic_cell(
  mesh: MeshSpec,
  *,
  width: float,
  height: float,
  field_id: SemanticId = "displacement",
  source: str = "authoring.periodic_cell",
) -> PeriodicCell:
  """Emit the landed periodic-cell coordinates, constraints, and boundaries.

  The mesh occupies ``[0, width] x [0, height]``. Opposite boundary groups are
  extracted and validated with legacy ``RVE.getBoundaries``' pairing, alignment,
  constant-offset, and shared-corner rules; violations are coded compile-time
  diagnostics carrying ``source``. The bottom-left corner is fixed, the other
  corners carry affine ``eps11``/``eps22``/``gamma12`` columns, and interior
  boundary nodes use the landed ``periodic_ties`` declarations.
  """
  if type(mesh) is not MeshSpec:
    msg = "periodic_cell requires an exact MeshSpec from the authoring mesh helpers"
    raise TypeError(msg)
  if type(field_id) not in (str, int) or not field_id:
    msg = "periodic-cell field id must be a non-empty exact semantic id"
    raise TypeError(msg)
  if type(source) is not str or not source:
    msg = "periodic-cell source must be a non-empty exact string"
    raise TypeError(msg)
  context = SourceContext(source=source)
  dx = _size(width, label="width", source=context)
  dy = _size(height, label="height", source=context)
  left, right, bottom, top = groups = _boundary_groups(mesh, dx, dy, context)
  bottom_left, bottom_right, top_left, top_right = left[0], right[0], top[0], top[-1]
  constraints: list[ProgramConstraintSpec] = [
    _prescribed(bottom_left, "x", AffineValueSpec(), field_id=field_id, source=context),
    _prescribed(bottom_left, "y", AffineValueSpec(), field_id=field_id, source=context),
    _prescribed(
      bottom_right,
      "x",
      _affine(("eps11", dx)),
      field_id=field_id,
      source=context,
    ),
    _prescribed(
      bottom_right,
      "y",
      _affine(("gamma12", 0.5 * dx)),
      field_id=field_id,
      source=context,
    ),
    _prescribed(
      top_left,
      "x",
      _affine(("gamma12", 0.5 * dy)),
      field_id=field_id,
      source=context,
    ),
    _prescribed(
      top_left,
      "y",
      _affine(("eps22", dy)),
      field_id=field_id,
      source=context,
    ),
    _prescribed(
      top_right,
      "x",
      _affine(("eps11", dx), ("gamma12", 0.5 * dy)),
      field_id=field_id,
      source=context,
    ),
    _prescribed(
      top_right,
      "y",
      _affine(("gamma12", 0.5 * dx), ("eps22", dy)),
      field_id=field_id,
      source=context,
    ),
  ]
  if left[1:-1]:
    constraints += periodic_ties(
      primary_node_ids=left[1:-1],
      image_node_ids=right[1:-1],
      field_id=field_id,
      components=("x", "y"),
      offsets=(_affine(("eps11", dx)), _affine(("gamma12", 0.5 * dx))),
      id_prefix="tie-lr",
    )
  if bottom[1:-1]:
    constraints += periodic_ties(
      primary_node_ids=bottom[1:-1],
      image_node_ids=top[1:-1],
      field_id=field_id,
      components=("x", "y"),
      offsets=(_affine(("gamma12", 0.5 * dy)), _affine(("eps22", dy))),
      id_prefix="tie-bt",
    )
  return PeriodicCell(
    groups=groups,
    width=dx,
    height=dy,
    coordinates=tuple(
      ProgramCoordinateSpec(name=name, kind="load") for name in STRAIN_COORDINATES
    ),
    constraints=tuple(constraints),
  )
