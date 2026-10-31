# SPDX-License-Identifier: MIT

"""FE2 periodic-cell authoring and M75 declaration-equivalence witnesses."""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import asdict, replace

import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from pyfem.v3 import authoring
from pyfem.v3.compile.diagnostics import ModelCompilationError
from pyfem.v3.compile.system import compile_system
from pyfem.v3.constraints import periodic_ties
from pyfem.v3.driver.contracts import HomogenizationReduction, HomogenizationRequest
from pyfem.v3.model.provenance import CanonicalManifest
from pyfem.v3.model.system import CompiledSystem
from pyfem.v3.spec import MeshSpec, SourceContext
from pyfem.v3.spec.program import (
  AffineCoefficientSpec,
  AffineValueSpec,
  DofRef,
  PrescribedDofSpec,
  ProgramCoordinateSpec,
)

_STRAIN = ("eps11", "eps22", "gamma12")
_DX = _DY = 0.6


def _affine(*pairs: tuple[str, float]) -> AffineValueSpec:
  return AffineValueSpec(
    coefficients=tuple(AffineCoefficientSpec(name, coef) for name, coef in pairs)
  )


def _prescribed(node: int, component: str, value: AffineValueSpec) -> PrescribedDofSpec:
  return PrescribedDofSpec(
    id=f"corner-{node}-{component}",
    target=DofRef(node_id=node, field_id="displacement", component=component),
    value=value,
    source=SourceContext(source="test.rve.corner"),
  )


def _reference_groups(
  mesh: MeshSpec,
  dx: float,
  dy: float,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
  """Verbatim M75 programmatic boundary extraction (its assertions included)."""
  tolerance = 1.0e-9 * max(dx, dy)
  coordinates = {node.id: node.coordinates for node in mesh.nodes}

  def group(
    predicate: Callable[[tuple[float, float]], bool],
    key: int,
  ) -> tuple[int, ...]:
    ids = tuple(node.id for node in mesh.nodes if predicate(node.coordinates))
    return tuple(sorted(ids, key=lambda node_id: coordinates[node_id][key]))

  left = group(lambda c: abs(c[0]) < tolerance, 1)
  right = group(lambda c: abs(c[0] - dx) < tolerance, 1)
  bottom = group(lambda c: abs(c[1]) < tolerance, 0)
  top = group(lambda c: abs(c[1] - dy) < tolerance, 0)
  assert len(left) == len(right) >= 2
  assert len(bottom) == len(top) >= 2
  for node_left, node_right in zip(left, right, strict=True):
    assert abs(coordinates[node_right][1] - coordinates[node_left][1]) < tolerance
    x_offset = coordinates[node_right][0] - coordinates[node_left][0]
    assert abs(x_offset - dx) < tolerance
  for node_bottom, node_top in zip(bottom, top, strict=True):
    assert abs(coordinates[node_top][0] - coordinates[node_bottom][0]) < tolerance
    y_offset = coordinates[node_top][1] - coordinates[node_bottom][1]
    assert abs(y_offset - dy) < tolerance
  assert left[0] == bottom[0] and right[0] == bottom[-1]
  assert left[-1] == top[0] and right[-1] == top[-1]
  return left, right, bottom, top


def _reference_constraints(
  left: tuple[int, ...],
  right: tuple[int, ...],
  bottom: tuple[int, ...],
  top: tuple[int, ...],
  dx: float,
  dy: float,
) -> tuple:
  """Verbatim M75 corner columns and periodic tie construction."""
  bottom_left, bottom_right, top_left, top_right = left[0], right[0], top[0], top[-1]
  constraints = [
    _prescribed(bottom_left, "x", AffineValueSpec()),
    _prescribed(bottom_left, "y", AffineValueSpec()),
    _prescribed(bottom_right, "x", _affine(("eps11", dx))),
    _prescribed(bottom_right, "y", _affine(("gamma12", 0.5 * dx))),
    _prescribed(top_left, "x", _affine(("gamma12", 0.5 * dy))),
    _prescribed(top_left, "y", _affine(("eps22", dy))),
    _prescribed(top_right, "x", _affine(("eps11", dx), ("gamma12", 0.5 * dy))),
    _prescribed(top_right, "y", _affine(("gamma12", 0.5 * dx), ("eps22", dy))),
  ]
  constraints += periodic_ties(
    primary_node_ids=left[1:-1],
    image_node_ids=right[1:-1],
    field_id="displacement",
    components=("x", "y"),
    offsets=(_affine(("eps11", dx)), _affine(("gamma12", 0.5 * dx))),
    id_prefix="tie-lr",
  )
  constraints += periodic_ties(
    primary_node_ids=bottom[1:-1],
    image_node_ids=top[1:-1],
    field_id="displacement",
    components=("x", "y"),
    offsets=(_affine(("gamma12", 0.5 * dy)), _affine(("eps22", dy))),
    id_prefix="tie-bt",
  )
  return tuple(constraints)


def _group_dofs(
  system: CompiledSystem,
  node_ids: tuple[int, ...],
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


def _reference_request(
  system: CompiledSystem,
  groups: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]],
  dx: float,
  dy: float,
) -> HomogenizationRequest:
  _, right, _, top = groups
  return HomogenizationRequest(
    strain_coordinates=_STRAIN,
    reductions=(
      HomogenizationReduction("sigma11", _group_dofs(system, right, "x"), 1.0 / dy),
      HomogenizationReduction("sigma22", _group_dofs(system, top, "y"), 1.0 / dx),
      HomogenizationReduction("sigma12", _group_dofs(system, top, "x"), 1.0 / dy),
    ),
  )


def _declaration_bytes(coordinates: tuple, constraints: tuple) -> bytes:
  return CanonicalManifest(
    {
      "coordinates": [asdict(item) for item in coordinates],
      "constraints": [asdict(item) for item in constraints],
    }
  ).to_bytes()


def test_periodic_cell_reproduces_m75_programmatic_specs_byte_identically() -> None:
  mesh = authoring.quad8_patch(3, 3, width=_DX, height=_DY)
  groups = _reference_groups(mesh, _DX, _DY)
  constraints = _reference_constraints(*groups, _DX, _DY)
  coordinates = tuple(ProgramCoordinateSpec(name=name, kind="load") for name in _STRAIN)
  cell = authoring.periodic_cell(
    mesh,
    width=_DX,
    height=_DY,
    source="test.rve.corner",
  )
  assert cell.groups == groups
  assert cell.constraints == constraints
  assert cell.coordinates == coordinates
  assert _declaration_bytes(cell.coordinates, cell.constraints) == _declaration_bytes(
    coordinates, constraints
  )
  model = authoring.small_strain_continuum(
    mesh, material=authoring.linear_elastic(1.0e6, 0.25)
  )
  system = compile_system(model, authoring.q8_registry())
  request = cell.homogenization_request(system)
  reference = _reference_request(system, groups, _DX, _DY)
  assert asdict(request) == asdict(reference)


def test_periodic_cell_boundary_failures_are_coded_compile_diagnostics() -> None:
  mesh = authoring.quad8_patch(2, 2, width=_DX, height=_DY)
  nodes = list(mesh.nodes)
  right_midside = next(
    node
    for node in nodes
    if abs(node.coordinates[0] - _DX) < 1.0e-12
    and abs(node.coordinates[1] - 0.5 * _DY) < 1.0e-12
  )
  index = nodes.index(right_midside)
  nodes[index] = replace(
    right_midside,
    coordinates=(right_midside.coordinates[0], 0.51 * _DY),
  )
  misaligned = replace(mesh, nodes=tuple(nodes))
  with pytest.raises(ModelCompilationError, match="periodic-boundary-alignment"):
    authoring.periodic_cell(misaligned, width=_DX, height=_DY)

  without_right = replace(
    mesh,
    nodes=tuple(node for node in mesh.nodes if node.id != right_midside.id),
  )
  with pytest.raises(ModelCompilationError, match="periodic-boundary-count"):
    authoring.periodic_cell(without_right, width=_DX, height=_DY)
  with pytest.raises(ModelCompilationError, match="invalid-periodic-cell-size"):
    authoring.periodic_cell(mesh, width=0.0, height=_DY)


def test_periodic_cell_rejects_foreign_authoring_values() -> None:
  mesh = authoring.quad8_patch(1, 1, width=_DX, height=_DY)
  with pytest.raises(TypeError, match="exact MeshSpec"):
    authoring.periodic_cell("mesh", width=_DX, height=_DY)  # type: ignore[arg-type]
  with pytest.raises(TypeError, match="field id"):
    authoring.periodic_cell(mesh, width=_DX, height=_DY, field_id="")
  with pytest.raises(TypeError, match="source"):
    authoring.periodic_cell(mesh, width=_DX, height=_DY, source="")
  with pytest.raises(TypeError, match="exact number"):
    authoring.periodic_cell(mesh, width="wide", height=_DY)  # type: ignore[arg-type]
