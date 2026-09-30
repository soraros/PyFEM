# SPDX-License-Identifier: MIT

"""Geometry and material breadth: structural patch tests and compiler wiring.

Covers the B4 parity-breadth wave of the small-strain continuum family: the
``linear-tria3``, ``bilinear-quad4``, and ``trilinear-hex8`` geometries and
the ``plane-strain-linear-elastic`` / ``isotropic-linear-elastic`` laws. The
skim decks drive to the exact constant-strain patch solution, the compiled
element tangents match the landed fem batched kernels and the legacy element,
rigid-body modes stay unresisted without spurious mechanisms, and
geometry/material/rank mismatches fail with coded diagnostics.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from _legacy_parity import legacy_element_stiffness

from pyfem.v3.compile.continuum import plasticity_reference_registry
from pyfem.v3.compile.contracts import continuum_reference_registry
from pyfem.v3.compile.diagnostics import ModelCompilationError
from pyfem.v3.compile.system import compile_system
from pyfem.v3.driver import DriverStatus
from pyfem.v3.fem.element import continuum_stiffness_batched
from pyfem.v3.io.legacy_deck import (
  ConvertedDeck,
  compile_deck,
  read_legacy_deck,
  run_deck,
)
from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.operator import ChannelRequest, OperatorEvaluationInput
from pyfem.v3.model.system import CompiledSystem
from pyfem.v3.spec.diagnostics import SourceContext
from pyfem.v3.spec.model import (
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

ROOT = Path(__file__).resolve().parents[2]
SKIMS = ROOT / "skims"

BREADTH_SKIMS = (
  "patch_test3",
  "patch_test4",
  "patch_test8_3d",
  "patch_test8_plane_strain",
)

_KE_ATOL = 1e-8
_SOURCE = SourceContext()


def _node_state_map(
  system: CompiledSystem,
  state: np.ndarray,
) -> dict[object, dict[str, float]]:
  """Gather one compiled coefficient vector into ``{node: {component: value}}``."""
  space = system.spaces[0]
  values: dict[object, dict[str, float]] = {}
  for row, (_field_id, node_id, component) in enumerate(space.coefficient_ids):
    values.setdefault(node_id, {})[component] = float(state[row])
  return values


# --- structural patch tests ------------------------------------------------------


@pytest.mark.parametrize(
  "skim_name",
  ("patch_test3", "patch_test4", "patch_test8_plane_strain"),
)
def test_constant_strain_field_is_recovered(skim_name: str) -> None:
  """The patch solution is the exact linear field at every node, free or not."""
  deck = read_legacy_deck(SKIMS / skim_name / "skim.pro")
  compiled = compile_deck(deck)
  run = run_deck(deck)
  assert run.result.status is DriverStatus.COMPLETED
  values = _node_state_map(compiled.system, run.state)
  coordinates = {node.id: node.coordinates for node in deck.model.mesh.nodes}
  assert len(values) == len(coordinates)
  for node_id, components in values.items():
    x, y = coordinates[node_id][:2]
    assert components["x"] == pytest.approx(1.0e-3 * x + 5.0e-4 * y, abs=1.0e-12)
    assert components["y"] == pytest.approx(5.0e-4 * x + 1.0e-3 * y, abs=1.0e-12)


def test_constant_strain_field_is_recovered_in_3d() -> None:
  """The 3D patch recovers the in-plane field and the free sigma_zz=0 strain."""
  deck = read_legacy_deck(SKIMS / "patch_test8_3d" / "skim.pro")
  compiled = compile_deck(deck)
  run = run_deck(deck)
  assert run.result.status is DriverStatus.COMPLETED
  values = _node_state_map(compiled.system, run.state)
  coordinates = {node.id: node.coordinates for node in deck.model.mesh.nodes}
  assert len(values) == len(coordinates) == 16
  thickness_strains = []
  for node_id, components in values.items():
    x, y, z = coordinates[node_id]
    assert components["x"] == pytest.approx(1.0e-3 * x + 5.0e-4 * y, abs=1.0e-12)
    assert components["y"] == pytest.approx(5.0e-4 * x + 1.0e-3 * y, abs=1.0e-12)
    if z == 0.0:
      assert components["z"] == pytest.approx(0.0, abs=1.0e-15)
    else:
      thickness_strains.append(components["z"] / z)
  assert len(thickness_strains) == 8
  # sigma_zz = 0 in the exact patch solution: eps_zz = -nu/(1-nu) (eps_xx+eps_yy).
  expected = -(0.25 / 0.75) * 2.0e-3
  for strain in thickness_strains:
    assert strain == pytest.approx(expected, rel=1.0e-9)


# --- element tangent oracles -------------------------------------------------------


def _element_coordinates(
  deck: ConvertedDeck,
  system: CompiledSystem,
) -> np.ndarray:
  coordinates = {node.id: node.coordinates for node in deck.model.mesh.nodes}
  operator = system.operators[0]
  connectivity = operator.entity_block.incidence.values
  ordered = sorted(coordinates)
  ordered_coordinates = np.array(
    [coordinates[node_id] for node_id in ordered], dtype=np.float64
  )
  return ordered_coordinates[connectivity]


@pytest.mark.parametrize("skim_name", BREADTH_SKIMS)
def test_element_tangents_match_fem_kernels(skim_name: str) -> None:
  """The compiled element tangent consumes the landed fem batched kernels."""
  deck = read_legacy_deck(SKIMS / skim_name / "skim.pro")
  compiled = compile_deck(deck)
  operator = compiled.system.operators[0]
  evaluation = operator.evaluate(
    OperatorEvaluationInput(
      port_values=(
        FinalizedArray(
          np.zeros(operator.header.ports[0].coefficient_map.values.shape),
          dtype=np.float64,
        ),
      ),
      accepted_state=FinalizedArray(
        np.zeros(operator.header.state_layout.row_shape), dtype=np.float64
      ),
      signals=(),
      request=ChannelRequest((), ("material-tangent",)),
    )
  )
  tangents = evaluation.jacobian_values[0].values
  constitutive = operator.payload.constitutive.values
  coordinates = _element_coordinates(deck, compiled.system)
  kernels = continuum_stiffness_batched(coordinates, constitutive)
  np.testing.assert_allclose(tangents, kernels, rtol=1.0e-12, atol=_KE_ATOL)


@pytest.mark.parametrize("skim_name", BREADTH_SKIMS)
def test_element_tangents_match_legacy_elements(skim_name: str) -> None:
  """Each compiled element tangent matches the legacy element at zero state."""
  skim_pro = SKIMS / skim_name / "skim.pro"
  deck = read_legacy_deck(skim_pro)
  compiled = compile_deck(deck)
  operator = compiled.system.operators[0]
  evaluation = operator.evaluate(
    OperatorEvaluationInput(
      port_values=(
        FinalizedArray(
          np.zeros(operator.header.ports[0].coefficient_map.values.shape),
          dtype=np.float64,
        ),
      ),
      accepted_state=FinalizedArray(
        np.zeros(operator.header.state_layout.row_shape), dtype=np.float64
      ),
      signals=(),
      request=ChannelRequest((), ("material-tangent",)),
    )
  )
  tangents = evaluation.jacobian_values[0].values
  coordinates = _element_coordinates(deck, compiled.system)
  for index in range(coordinates.shape[0]):
    legacy = legacy_element_stiffness(skim_pro, coordinates[index])
    np.testing.assert_allclose(tangents[index], legacy, rtol=0.0, atol=_KE_ATOL)


# --- rigid-body modes ----------------------------------------------------------------


def _single_element_model(
  block: CellBlockSpec,
  nodes: tuple[NodeSpec, ...],
  components: tuple[str, ...],
  quadrature: str,
  material_model: str,
  parameters: tuple[tuple[str, float], ...],
  formulation: str = "small-strain-continuum",
) -> ModelSpec:
  field = FieldSpec(
    id="displacement", components=components, location="node", source=_SOURCE
  )
  material = MaterialSpec(
    id="material",
    model=material_model,
    parameters=tuple(
      MaterialParameterSpec(name, value, _SOURCE) for name, value in parameters
    ),
    source=_SOURCE,
  )
  region = RegionSpec(
    id="domain",
    cell_refs=tuple(CellRef(block.id, cell.id) for cell in block.cells),
    field_ids=(field.id,),
    material_id=material.id,
    formulation=formulation,
    quadrature=quadrature,
    source=_SOURCE,
  )
  return ModelSpec(
    mesh=MeshSpec(nodes=nodes, cell_blocks=(block,), source=_SOURCE),
    fields=(field,),
    materials=(material,),
    regions=(region,),
    source=_SOURCE,
  )


def _block(
  cells: tuple[CellSpec, ...],
  *,
  topology: str,
  topological_dimension: int,
  embedding_dimension: int,
  interpolation: str,
) -> CellBlockSpec:
  return CellBlockSpec(
    id="cells",
    reference_topology=topology,
    topological_dimension=topological_dimension,
    embedding_dimension=embedding_dimension,
    geometry_interpolation=interpolation,
    cells=cells,
    source=_SOURCE,
  )


def _tria3_model(**overrides: object) -> ModelSpec:
  nodes = (
    NodeSpec(id=0, coordinates=(0.0, 0.0), source=_SOURCE),
    NodeSpec(id=1, coordinates=(1.0, 0.0), source=_SOURCE),
    NodeSpec(id=2, coordinates=(0.25, 1.0), source=_SOURCE),
  )
  options: dict[str, object] = {
    "block": _block(
      (CellSpec(id=1, node_ids=(0, 1, 2), source=_SOURCE),),
      topology="triangle",
      topological_dimension=2,
      embedding_dimension=2,
      interpolation="linear-tria3",
    ),
    "nodes": nodes,
    "components": ("x", "y"),
    "quadrature": "gauss-tria3-1",
    "material_model": "plane-stress-linear-elastic",
    "parameters": (("youngs_modulus", 1.0e6), ("poisson_ratio", 0.25)),
  }
  options.update(overrides)
  return _single_element_model(**options)  # type: ignore[arg-type]


def _quad4_model(**overrides: object) -> ModelSpec:
  nodes = (
    NodeSpec(id=0, coordinates=(0.0, 0.0), source=_SOURCE),
    NodeSpec(id=1, coordinates=(1.0, 0.0), source=_SOURCE),
    NodeSpec(id=2, coordinates=(1.0, 1.0), source=_SOURCE),
    NodeSpec(id=3, coordinates=(0.0, 1.0), source=_SOURCE),
  )
  options: dict[str, object] = {
    "block": _block(
      (CellSpec(id=1, node_ids=(0, 1, 2, 3), source=_SOURCE),),
      topology="quadrilateral",
      topological_dimension=2,
      embedding_dimension=2,
      interpolation="bilinear-quad4",
    ),
    "nodes": nodes,
    "components": ("x", "y"),
    "quadrature": "gauss-2x2",
    "material_model": "plane-stress-linear-elastic",
    "parameters": (("youngs_modulus", 1.0e6), ("poisson_ratio", 0.25)),
  }
  options.update(overrides)
  return _single_element_model(**options)  # type: ignore[arg-type]


def _hex8_model(**overrides: object) -> ModelSpec:
  nodes = (
    NodeSpec(id=0, coordinates=(0.0, 0.0, 0.0), source=_SOURCE),
    NodeSpec(id=1, coordinates=(1.0, 0.0, 0.0), source=_SOURCE),
    NodeSpec(id=2, coordinates=(1.0, 1.0, 0.0), source=_SOURCE),
    NodeSpec(id=3, coordinates=(0.0, 1.0, 0.0), source=_SOURCE),
    NodeSpec(id=4, coordinates=(0.0, 0.0, 1.0), source=_SOURCE),
    NodeSpec(id=5, coordinates=(1.0, 0.0, 1.0), source=_SOURCE),
    NodeSpec(id=6, coordinates=(1.0, 1.0, 1.0), source=_SOURCE),
    NodeSpec(id=7, coordinates=(0.0, 1.0, 1.0), source=_SOURCE),
  )
  options: dict[str, object] = {
    "block": _block(
      (CellSpec(id=1, node_ids=(0, 1, 2, 3, 4, 5, 6, 7), source=_SOURCE),),
      topology="hexahedron",
      topological_dimension=3,
      embedding_dimension=3,
      interpolation="trilinear-hex8",
    ),
    "nodes": nodes,
    "components": ("x", "y", "z"),
    "quadrature": "gauss-2x2x2",
    "material_model": "isotropic-linear-elastic",
    "parameters": (("youngs_modulus", 1.0e6), ("poisson_ratio", 0.25)),
  }
  options.update(overrides)
  return _single_element_model(**options)  # type: ignore[arg-type]


def _compiled_tangent(model: ModelSpec) -> np.ndarray:
  system = compile_system(model, continuum_reference_registry())
  operator = system.operators[0]
  evaluation = operator.evaluate(
    OperatorEvaluationInput(
      port_values=(
        FinalizedArray(
          np.zeros(operator.header.ports[0].coefficient_map.values.shape),
          dtype=np.float64,
        ),
      ),
      accepted_state=FinalizedArray(
        np.zeros(operator.header.state_layout.row_shape), dtype=np.float64
      ),
      signals=(),
      request=ChannelRequest((), ("material-tangent",)),
    )
  )
  return evaluation.jacobian_values[0].values[0]


@pytest.mark.parametrize(
  ("model_factory", "coordinates", "zero_modes"),
  [
    (
      _tria3_model,
      np.array([[0.0, 0.0], [1.0, 0.0], [0.25, 1.0]]),
      3,
    ),
    (
      _quad4_model,
      np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
      3,
    ),
    (
      _hex8_model,
      np.array(
        [
          [0.0, 0.0, 0.0],
          [1.0, 0.0, 0.0],
          [1.0, 1.0, 0.0],
          [0.0, 1.0, 0.0],
          [0.0, 0.0, 1.0],
          [1.0, 0.0, 1.0],
          [1.0, 1.0, 1.0],
          [0.0, 1.0, 1.0],
        ]
      ),
      6,
    ),
  ],
)
def test_rigid_body_modes_are_unresisted(
  model_factory: Callable[[], ModelSpec],
  coordinates: np.ndarray,
  zero_modes: int,
) -> None:
  """Rigid translations and rotations carry no stiffness; nothing else is free."""
  tangent = _compiled_tangent(model_factory())
  rank = coordinates.shape[1]
  node_count = coordinates.shape[0]
  center = coordinates.mean(axis=0)
  modes = []
  for axis in range(rank):
    translation = np.zeros((node_count, rank))
    translation[:, axis] = 1.0
    modes.append(translation.reshape(-1))
  relative = coordinates - center
  if rank == 2:
    rotation = np.stack((-relative[:, 1], relative[:, 0]), axis=1)
    modes.append(rotation.reshape(-1))
  else:
    modes.append(
      np.stack(
        (
          np.zeros(node_count),
          -relative[:, 2],
          relative[:, 1],
        ),
        axis=1,
      ).reshape(-1)
    )
    modes.append(
      np.stack(
        (
          relative[:, 2],
          np.zeros(node_count),
          -relative[:, 0],
        ),
        axis=1,
      ).reshape(-1)
    )
    modes.append(
      np.stack(
        (
          -relative[:, 1],
          relative[:, 0],
          np.zeros(node_count),
        ),
        axis=1,
      ).reshape(-1)
    )
  scale = float(np.max(np.abs(tangent)))
  for mode in modes:
    reaction = tangent @ mode
    np.testing.assert_allclose(reaction, 0.0, rtol=0.0, atol=1.0e-10 * scale)
  eigenvalues = np.linalg.eigvalsh(tangent)
  near_zero = np.abs(eigenvalues) <= 1.0e-10 * float(eigenvalues[-1])
  assert int(np.count_nonzero(near_zero)) == zero_modes
  assert bool(np.all(eigenvalues[~near_zero] > 0.0))


# --- compiled-manifest determinism ----------------------------------------------------


@pytest.mark.parametrize("skim_name", BREADTH_SKIMS)
def test_compiled_manifest_is_deterministic(skim_name: str) -> None:
  deck = read_legacy_deck(SKIMS / skim_name / "skim.pro")
  first = compile_deck(deck).system.content_fingerprint
  second = compile_deck(deck).system.content_fingerprint
  assert first == second


# --- coded compiler diagnostics --------------------------------------------------


_COMPILER_REJECTIONS = {
  "tria3-wrong-quadrature": (
    _tria3_model(quadrature="gauss-3x3"),
    continuum_reference_registry(),
    "incompatible-quadrature",
  ),
  "tria3-with-3d-law": (
    _tria3_model(material_model="isotropic-linear-elastic"),
    continuum_reference_registry(),
    "incompatible-material-model",
  ),
  "quad4-with-3d-law": (
    _quad4_model(material_model="isotropic-linear-elastic"),
    continuum_reference_registry(),
    "incompatible-material-model",
  ),
  "hex8-with-plane-stress": (
    _hex8_model(material_model="plane-stress-linear-elastic"),
    continuum_reference_registry(),
    "incompatible-material-model",
  ),
  "hex8-with-plane-strain": (
    _hex8_model(material_model="plane-strain-linear-elastic"),
    continuum_reference_registry(),
    "incompatible-material-model",
  ),
  "hex8-planar-field-signature": (
    _hex8_model(components=("x", "y")),
    continuum_reference_registry(),
    "incompatible-field-signature",
  ),
  "tria3-stateful-material": (
    _tria3_model(
      material_model="isotropic-hardening-plasticity",
      parameters=(
        ("youngs_modulus", 210000.0),
        ("poisson_ratio", 0.3),
        ("initial_yield_stress", 250.0),
        ("hardening_slope", 1000.0),
      ),
    ),
    plasticity_reference_registry(),
    "incompatible-material-model",
  ),
  "tria3-thermal-formulation": (
    _tria3_model(formulation="small-strain-thermal-continuum"),
    continuum_reference_registry(),
    "incompatible-cell-block",
  ),
  "tria3-bad-cell-arity": (
    _tria3_model(
      block=_block(
        (CellSpec(id=1, node_ids=(0, 1, 2, 3), source=_SOURCE),),
        topology="triangle",
        topological_dimension=2,
        embedding_dimension=2,
        interpolation="linear-tria3",
      ),
      nodes=(
        NodeSpec(id=0, coordinates=(0.0, 0.0), source=_SOURCE),
        NodeSpec(id=1, coordinates=(1.0, 0.0), source=_SOURCE),
        NodeSpec(id=2, coordinates=(0.25, 1.0), source=_SOURCE),
        NodeSpec(id=3, coordinates=(1.0, 1.0), source=_SOURCE),
      ),
    ),
    continuum_reference_registry(),
    "invalid-cell-arity",
  ),
  "unknown-interpolation": (
    _tria3_model(
      block=_block(
        (CellSpec(id=1, node_ids=(0, 1, 2), source=_SOURCE),),
        topology="triangle",
        topological_dimension=2,
        embedding_dimension=2,
        interpolation="tria3-by-name",
      ),
    ),
    continuum_reference_registry(),
    "incompatible-cell-block",
  ),
  "inconsistent-block-dimensions": (
    _tria3_model(
      block=_block(
        (CellSpec(id=1, node_ids=(0, 1, 2), source=_SOURCE),),
        topology="quadrilateral",
        topological_dimension=2,
        embedding_dimension=2,
        interpolation="linear-tria3",
      ),
    ),
    continuum_reference_registry(),
    "incompatible-cell-block",
  ),
}


@pytest.mark.parametrize(
  ("model", "registry", "code"),
  [
    pytest.param(model, registry, code, id=name)
    for name, (model, registry, code) in _COMPILER_REJECTIONS.items()
  ],
)
def test_compiler_geometry_material_mismatches_are_coded(
  model: ModelSpec,
  registry: dict,
  code: str,
) -> None:
  with pytest.raises(ModelCompilationError, match=code):
    compile_system(model, registry)
