# SPDX-License-Identifier: MIT

"""Numerical oracles for the compiled truss operator family.

Oracle hierarchy and documented tolerances:

- Legacy ``pyfem/elements/Truss.py`` (driven through the shallow-truss skim
  properties) is the tangent and residual oracle of record. The battery below
  compares with ``rtol=1e-8`` and ``atol=1e-8`` times the axial stiffness scale
  ``E*A*max(l0, 1/l0)``; the observed worst relative deviation is 4.3e-10 over
  455 drives across five geometries and 91 states each. Those deviations stem
  from last-ulp differences between ``scipy.linalg.norm`` (legacy) and
  ``numpy.hypot`` (compiled) element lengths, amplified by ``(du/l0)**2`` in
  the geometric stiffness at extreme strain. Tolerance headroom is 25x.
- The v3 prototype ``link2_tangent_batched`` kernel is a residual-only oracle
  (``rtol=1e-12``; observed deviations are at the 1e-17 level). Its tangent is
  excluded: the kernel rotates element matrices to global coordinates with the
  forward rotation instead of its transpose, a mirrored-direction stiffness
  for angled elements filed as a tower finding on 2026-09-29. The legacy
  element applies the chain-rule-consistent transpose rotation, and so does
  the compiled operator.
- Closed-form Fraction-derived values anchor the response at ulp level
  (``rtol=0``, ``atol=1e-8``, roughly 20 ulps at the 3e6 response scale).
"""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from pyfem.io.InputReader import InputRead
from pyfem.util.dataStructures import elementData
from pyfem.v3.compile.system import compile_system
from pyfem.v3.compile.truss import TrussOperator, truss_reference_registry
from pyfem.v3.fem.link2 import link2_tangent_batched
from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.operator import ChannelRequest, OperatorEvaluationInput
from pyfem.v3.spec import (
  CellBlockSpec,
  CellRef,
  CellSpec,
  FieldSpec,
  MaterialParameterSpec,
  MaterialSpec,
  MeshSpec,
  ModelSpec,
  NodeSpec,
  RegionSpec,
)
from pyfem.v3.types import GROUP_TRUSS

ROOT = Path(__file__).resolve().parents[2]
SKIM_PRO = ROOT / "skims" / "shallow_truss_riks" / "skim.pro"
YOUNGS_MODULUS = 5.0e6
AREA = 1.0
BATTERY_RTOL = 1.0e-8
LINK2_RTOL = 1.0e-12

_LEGACY_CONTEXT: tuple[object, object] | None = None


def _legacy_element() -> object:
  global _LEGACY_CONTEXT
  if _LEGACY_CONTEXT is None:
    _LEGACY_CONTEXT = InputRead(str(SKIM_PRO))
  _, globdat = _LEGACY_CONTEXT
  return next(iter(globdat.elements.iterElementGroup("TrussElem")))


def _legacy_drive(
  element: object,
  coordinates: np.ndarray,
  state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  """Legacy tangent and internal force at a total state from zero history."""
  props, _ = _LEGACY_CONTEXT
  template = elementData(state.copy(), state.copy())
  template.coords = coordinates
  template.props = getattr(props, "TrussElem")
  element.setHistoryParameter("sigma", 0.0)
  element.commitHistory()
  element.getTangentStiffness(template)
  return template.stiff.copy(), template.fint.copy()


def _model(
  coordinates: tuple[tuple[float, float], ...],
  cells: tuple[tuple[str, tuple[int, ...]], ...],
) -> ModelSpec:
  nodes = tuple(
    NodeSpec(id=index, coordinates=point)
    for index, point in enumerate(coordinates, start=1)
  )
  cell_specs = tuple(
    CellSpec(id=cell_id, node_ids=node_ids) for cell_id, node_ids in cells
  )
  block = CellBlockSpec(
    id="bars",
    reference_topology="line",
    topological_dimension=1,
    embedding_dimension=2,
    geometry_interpolation="line2",
    cells=cell_specs,
  )
  field = FieldSpec(id="displacement", components=("x", "y"), location="node")
  material = MaterialSpec(
    id="steel",
    model="uniaxial-linear-elastic",
    parameters=(
      MaterialParameterSpec("youngs_modulus", YOUNGS_MODULUS),
      MaterialParameterSpec("area", AREA),
    ),
  )
  region = RegionSpec(
    id="domain",
    cell_refs=tuple(CellRef("bars", cell_id) for cell_id, _ in cells),
    field_ids=(field.id,),
    material_id=material.id,
    formulation="total-lagrangian-truss",
    quadrature="none",
  )
  return ModelSpec(
    mesh=MeshSpec(nodes=nodes, cell_blocks=(block,)),
    fields=(field,),
    materials=(material,),
    regions=(region,),
  )


def _operator(model: ModelSpec) -> TrussOperator:
  operator = compile_system(model, truss_reference_registry()).operators[0]
  assert isinstance(operator, TrussOperator)
  return operator


def _evaluate(
  operator: TrussOperator,
  states: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  inputs = OperatorEvaluationInput(
    port_values=(FinalizedArray(states, dtype=np.float64),),
    accepted_state=FinalizedArray(
      np.empty(operator.header.state_layout.row_shape),
      dtype=np.float64,
    ),
    signals=(),
    request=ChannelRequest(("internal-force",), ("tangent",)),
  )
  result = operator.evaluate(inputs)
  return result.jacobian_values[0].values, result.residual_values[0].values


def test_zero_state_and_axial_response_match_closed_form() -> None:
  operator = _operator(_model(((0.0, 0.0), (2.0, 0.0)), (("bar", (1, 2)),)))
  stiffness = Fraction.from_float(YOUNGS_MODULUS) * Fraction.from_float(AREA)
  length = Fraction(2)

  tangent, residual = _evaluate(operator, np.zeros((1, 4)))
  axial = float(stiffness / length)
  np.testing.assert_array_equal(residual[0], np.zeros(4))
  np.testing.assert_allclose(
    tangent[0],
    axial
    * np.array(
      [
        [1.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [-1.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
      ],
    ),
    rtol=0.0,
    atol=1.0e-8,
  )

  stretch = np.array([[0.0, 0.0, 0.1, 0.0]])
  tangent, residual = _evaluate(operator, stretch)
  du = Fraction.from_float(0.1) / length
  strain = du + Fraction(1, 2) * du * du
  bl0 = -(1 + du) / length
  expected_force = stiffness * length * strain * bl0
  expected_tangent = stiffness * length * bl0 * bl0 + stiffness * strain / length
  np.testing.assert_allclose(
    [residual[0, 0], residual[0, 2], tangent[0, 0, 0]],
    [float(expected_force), -float(expected_force), float(expected_tangent)],
    rtol=0.0,
    atol=1.0e-8,
  )
  np.testing.assert_array_equal(residual[0, 1], 0.0)
  np.testing.assert_array_equal(residual[0, 3], 0.0)


@pytest.mark.parametrize(
  "coordinates",
  [
    ((0.0, 0.0), (2.0, 0.0)),
    ((0.0, 0.0), (0.0, 3.0)),
    ((-10.0, 0.0), (0.0, 0.5)),
    ((1.0, -2.0), (-3.0, 4.0)),
    ((0.0, 0.0), (1.0e-3, 1.0e-3)),
  ],
)
def test_compiled_truss_matches_legacy_element_battery(
  coordinates: tuple[tuple[float, float], ...],
) -> None:
  coords = np.array(coordinates, dtype=np.float64)
  operator = _operator(_model(coordinates, (("bar", (1, 2)),)))
  element = _legacy_element()
  length = float(np.hypot(*(coords[1] - coords[0])))
  scale = YOUNGS_MODULUS * AREA * max(length, 1.0 / length)
  rng = np.random.default_rng(20260929)
  states = [np.zeros(4)]
  states.extend(rng.normal(0.0, extent, (30, 4)) for extent in (0.01, 0.1, 0.5))
  for batch in states:
    batch2 = batch.reshape(-1, 4)
    for state in batch2:
      tangent, residual = _evaluate(operator, state.reshape(1, 4))
      legacy_tangent, legacy_residual = _legacy_drive(element, coords, state)
      np.testing.assert_allclose(
        residual[0],
        legacy_residual,
        rtol=BATTERY_RTOL,
        atol=BATTERY_RTOL * scale,
      )
      np.testing.assert_allclose(
        tangent[0],
        legacy_tangent,
        rtol=BATTERY_RTOL,
        atol=BATTERY_RTOL * scale,
      )


def test_compiled_truss_residual_matches_link2_kernel() -> None:
  coordinates = ((-10.0, 0.0), (10.0, 0.0), (0.0, 0.5))
  cells = (("cell-1", (1, 3)), ("cell-2", (2, 3)))
  operator = _operator(_model(coordinates, cells))
  coords = np.array(
    [[coordinates[0], coordinates[2]], [coordinates[1], coordinates[2]]],
    dtype=np.float64,
  )
  scale = YOUNGS_MODULUS * AREA * float(np.hypot(10.0, 0.5))
  rng = np.random.default_rng(20260704)
  for extent in (0.0, 0.05, 0.6):
    states = rng.normal(0.0, extent, (2, 4)) if extent else np.zeros((2, 4))
    _, forces = link2_tangent_batched(
      coords,
      states,
      GROUP_TRUSS,
      YOUNGS_MODULUS,
      AREA,
    )
    _, residual = _evaluate(operator, states)
    np.testing.assert_allclose(
      residual,
      forces,
      rtol=LINK2_RTOL,
      atol=LINK2_RTOL * scale,
    )


def test_shallow_truss_pair_matches_legacy_manual_assembly() -> None:
  coordinates = ((-10.0, 0.0), (10.0, 0.0), (0.0, 0.5))
  cells = (("cell-1", (1, 3)), ("cell-2", (2, 3)))
  operator = _operator(_model(coordinates, cells))
  element = _legacy_element()
  gather = np.array([[0, 1, 4, 5], [2, 3, 4, 5]])
  pair_coords = (
    np.array([coordinates[0], coordinates[2]], dtype=np.float64),
    np.array([coordinates[1], coordinates[2]], dtype=np.float64),
  )
  scale = YOUNGS_MODULUS * AREA * float(np.hypot(10.0, 0.5))
  global_states = (
    np.zeros(6),
    np.array([0.02, -0.01, -0.03, 0.02, 0.3, -0.4]),
    np.array([-0.1, 0.05, 0.08, -0.06, 0.9, -1.2]),
  )
  for global_state in global_states:
    tangent, residual = _evaluate(operator, global_state[gather])
    compiled_tangent = np.zeros((6, 6))
    compiled_residual = np.zeros(6)
    for element_index in range(2):
      dofs = gather[element_index]
      compiled_tangent[np.ix_(dofs, dofs)] += tangent[element_index]
      compiled_residual[dofs] += residual[element_index]
    legacy_tangent = np.zeros((6, 6))
    legacy_residual = np.zeros(6)
    for element_index in range(2):
      dofs = gather[element_index]
      stiff, fint = _legacy_drive(
        element,
        pair_coords[element_index],
        global_state[dofs],
      )
      legacy_tangent[np.ix_(dofs, dofs)] += stiff
      legacy_residual[dofs] += fint
    np.testing.assert_allclose(
      compiled_residual,
      legacy_residual,
      rtol=BATTERY_RTOL,
      atol=BATTERY_RTOL * scale,
    )
    np.testing.assert_allclose(
      compiled_tangent,
      legacy_tangent,
      rtol=BATTERY_RTOL,
      atol=BATTERY_RTOL * scale,
    )
