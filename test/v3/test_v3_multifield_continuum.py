# SPDX-License-Identifier: MIT

"""Direct compiler proof for multi-field continuum compilation behind the seam."""

from __future__ import annotations

import copy
import hashlib
import pickle
import sys
from collections.abc import Callable
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from pyfem.v3.compile.continuum import (
  Q8_TOPOLOGY_KEY,
  THERMAL_FORMULATION_KEY,
  THERMAL_MATERIAL_KEY,
  THERMO_FORMULATION_KEY,
  THERMO_MATERIAL_KEY,
  Q8ContinuumOperator,
  Q8ThermalOperator,
  Q8ThermoElasticOperator,
  linear_thermo_elastic,
  q8_reference_registry,
  thermal_descriptor_metadata,
  thermal_gradient_map,
  thermal_reference_registry,
  thermo_elastic_descriptor_metadata,
  thermo_elastic_reference_registry,
)
from pyfem.v3.compile.diagnostics import ModelCompilationError
from pyfem.v3.compile.system import compile_discrete_spaces, compile_system
from pyfem.v3.compile.truss import truss_reference_registry
from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.operator import (
  BalanceRole,
  ChannelRequest,
  OperatorEvaluationInput,
  ProgramSignalInput,
  StateLifetime,
)
from pyfem.v3.model.registry import RegistryDescriptor, RegistryKey
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
  SourceContext,
)

_HUGE_INTEGER_ID = 10**5000
_MAX_HUGE_DIAGNOSTIC_LENGTH = 8192

# Content digests recorded from the pristine landed compiler at merge-base
# 6e32b56 (before the multi-field extension). Single-field compilation must
# reproduce these bytes exactly.
_Q8_ONE_CELL_DIGEST = "abf6d020f5e74d7b84c800739fd31241e486e3586116f14def507381f78ba733"
_Q8_TWO_CELL_DIGEST = "6a0fff777e43d72008508531d3f2af5d1abad82a5a2c5cec573a1a5e77e43552"
_TRUSS_TWO_CELL_DIGEST = (
  "e1953b337b0b74c4e6020b2783281d12f07f9790443d07cafdc3535ae49fe31e"
)

_UNIT_COORDINATES = (
  (0.0, 0.0),
  (0.5, 0.0),
  (1.0, 0.0),
  (1.0, 0.5),
  (1.0, 1.0),
  (0.5, 1.0),
  (0.0, 1.0),
  (0.0, 0.5),
)

_TWO_CELL_COORDINATES = (
  (0.0, 0.0),
  (0.5, 0.0),
  (1.0, 0.0),
  (1.0, 0.5),
  (1.0, 1.0),
  (0.5, 1.0),
  (0.0, 1.0),
  (0.0, 0.5),
  (1.5, 0.0),
  (2.0, 0.0),
  (2.0, 0.5),
  (2.0, 1.0),
  (1.5, 1.0),
)


def _source(label: str) -> SourceContext:
  return SourceContext(source=label)


def _displacement_field(field_id: str = "displacement") -> FieldSpec:
  return FieldSpec(
    id=field_id,
    components=("x", "y"),
    location="node",
    source=_source("field:displacement"),
  )


def _temperature_field(field_id: str = "temperature") -> FieldSpec:
  return FieldSpec(
    id=field_id,
    components=("theta",),
    location="node",
    source=_source("field:temperature"),
  )


def _elastic_material(
  youngs_modulus: object = 1.0,
  poisson_ratio: object = 0.0,
) -> MaterialSpec:
  return MaterialSpec(
    id="elastic",
    model="plane-stress-linear-elastic",
    parameters=(
      MaterialParameterSpec("youngs_modulus", youngs_modulus, _source("material:E")),
      MaterialParameterSpec("poisson_ratio", poisson_ratio, _source("material:nu")),
    ),
    source=_source("material:elastic"),
  )


def _conductor_material(
  parameters: tuple[tuple[str, object], ...] = (("conductivity", 3.0),),
) -> MaterialSpec:
  return MaterialSpec(
    id="conductor",
    model="linear-thermal-conductor",
    parameters=tuple(
      MaterialParameterSpec(name, value, _source(f"material:{name}"))
      for name, value in parameters
    ),
    source=_source("material:conductor"),
  )


def _thermo_elastic_material(
  parameters: tuple[tuple[str, object], ...] = (
    ("youngs_modulus", 1.0),
    ("poisson_ratio", 0.0),
    ("thermal_expansion", 0.5),
    ("conductivity", 2.0),
  ),
) -> MaterialSpec:
  return MaterialSpec(
    id="thermo-elastic",
    model="linear-thermo-elastic",
    parameters=tuple(
      MaterialParameterSpec(name, value, _source(f"material:{name}"))
      for name, value in parameters
    ),
    source=_source("material:thermo-elastic"),
  )


def _q8_block(block_id: str, cells: tuple[CellSpec, ...]) -> CellBlockSpec:
  return CellBlockSpec(
    id=block_id,
    reference_topology="quadrilateral",
    topological_dimension=2,
    embedding_dimension=2,
    geometry_interpolation="serendipity-quad8",
    cells=cells,
    source=_source(f"block:{block_id}"),
  )


def _unit_nodes() -> tuple[NodeSpec, ...]:
  return tuple(
    NodeSpec(
      id=index + 1,
      coordinates=point,
      source=_source(f"node:{index + 1}"),
    )
    for index, point in enumerate(_UNIT_COORDINATES)
  )


def _two_cell_nodes() -> tuple[NodeSpec, ...]:
  return tuple(
    NodeSpec(
      id=index,
      coordinates=point,
      source=_source(f"node:{index}"),
    )
    for index, point in enumerate(_TWO_CELL_COORDINATES, start=1)
  )


def _coupled_model(
  *,
  cell_id: str | int = "cell-1",
  coordinates: tuple[tuple[float, float], ...] | None = None,
  field_ids: tuple[str, ...] = ("displacement", "temperature"),
  fields: tuple[FieldSpec, ...] | None = None,
  materials: tuple[MaterialSpec, ...] | None = None,
  formulation: str = "small-strain-thermo-elastic-continuum",
  quadrature: str = "gauss-3x3",
) -> ModelSpec:
  raw_coordinates = _UNIT_COORDINATES if coordinates is None else coordinates
  nodes = tuple(
    NodeSpec(
      id=index + 1,
      coordinates=point,
      source=_source(f"node:{index + 1}"),
    )
    for index, point in enumerate(raw_coordinates)
  )
  cell = CellSpec(
    id=cell_id,
    node_ids=tuple(node.id for node in nodes),
    source=_source("cell"),
  )
  block = _q8_block("cells", (cell,))
  region = RegionSpec(
    id="domain",
    cell_refs=(CellRef(block.id, cell.id),),
    field_ids=field_ids,
    material_id="thermo-elastic",
    formulation=formulation,
    quadrature=quadrature,
    source=_source("region:domain"),
  )
  return ModelSpec(
    mesh=MeshSpec(nodes=nodes, cell_blocks=(block,), source=_source("mesh")),
    fields=(_displacement_field(), _temperature_field()) if fields is None else fields,
    materials=(_thermo_elastic_material(),) if materials is None else materials,
    regions=(region,),
    source=_source("model"),
  )


def _thermal_model(
  *,
  parameters: tuple[tuple[str, object], ...] = (("conductivity", 3.0),),
  field_components: tuple[str, ...] = ("theta",),
  materials: tuple[MaterialSpec, ...] | None = None,
  formulation: str = "small-strain-thermal-continuum",
  quadrature: str = "gauss-3x3",
) -> ModelSpec:
  nodes = _unit_nodes()
  cell = CellSpec(
    id="cell-1",
    node_ids=tuple(node.id for node in nodes),
    source=_source("cell"),
  )
  block = _q8_block("cells", (cell,))
  field = FieldSpec(
    id="temperature",
    components=field_components,
    location="node",
    source=_source("field:temperature"),
  )
  region = RegionSpec(
    id="domain",
    cell_refs=(CellRef(block.id, cell.id),),
    field_ids=(field.id,),
    material_id="conductor",
    formulation=formulation,
    quadrature=quadrature,
    source=_source("region:domain"),
  )
  return ModelSpec(
    mesh=MeshSpec(nodes=nodes, cell_blocks=(block,), source=_source("mesh")),
    fields=(field,),
    materials=((_conductor_material(parameters),) if materials is None else materials),
    regions=(region,),
    source=_source("model"),
  )


def _split_model() -> ModelSpec:
  """One coupled region and one thermal-only region sharing three nodes."""
  nodes = _two_cell_nodes()
  coupled_block = _q8_block(
    "coupled-cells",
    (
      CellSpec(
        id=101,
        node_ids=(1, 2, 3, 4, 5, 6, 7, 8),
        source=_source("cell:101"),
      ),
    ),
  )
  thermal_block = _q8_block(
    "thermal-cells",
    (
      CellSpec(
        id=102,
        node_ids=(3, 9, 10, 11, 12, 13, 5, 4),
        source=_source("cell:102"),
      ),
    ),
  )
  coupled_region = RegionSpec(
    id="coupled-region",
    cell_refs=(CellRef("coupled-cells", 101),),
    field_ids=("displacement", "temperature"),
    material_id="thermo-elastic",
    formulation="small-strain-thermo-elastic-continuum",
    quadrature="gauss-3x3",
    source=_source("region:coupled"),
  )
  thermal_region = RegionSpec(
    id="thermal-region",
    cell_refs=(CellRef("thermal-cells", 102),),
    field_ids=("temperature",),
    material_id="conductor",
    formulation="small-strain-thermal-continuum",
    quadrature="gauss-3x3",
    source=_source("region:thermal"),
  )
  return ModelSpec(
    mesh=MeshSpec(
      nodes=nodes,
      cell_blocks=(coupled_block, thermal_block),
      source=_source("mesh"),
    ),
    fields=(_displacement_field(), _temperature_field()),
    materials=(_thermo_elastic_material(), _conductor_material()),
    regions=(coupled_region, thermal_region),
    source=_source("model"),
  )


def _mechanical_unit_model() -> ModelSpec:
  nodes = tuple(
    NodeSpec(
      id=index + 1,
      coordinates=point,
      source=_source(f"node-source-{index + 1}"),
    )
    for index, point in enumerate(_UNIT_COORDINATES)
  )
  cell = CellSpec(
    id="cell-1",
    node_ids=tuple(range(1, 9)),
    source=_source("cell-source"),
  )
  block = CellBlockSpec(
    id="cells",
    reference_topology="quadrilateral",
    topological_dimension=2,
    embedding_dimension=2,
    geometry_interpolation="serendipity-quad8",
    cells=(cell,),
    source=_source("block-source"),
  )
  field = FieldSpec(
    id="displacement",
    components=("x", "y"),
    location="node",
    source=_source("field-source"),
  )
  material = MaterialSpec(
    id="elastic",
    model="plane-stress-linear-elastic",
    parameters=(
      MaterialParameterSpec("youngs_modulus", 1.0, _source("material:E")),
      MaterialParameterSpec("poisson_ratio", 0.0, _source("material:nu")),
    ),
    source=_source("material-source"),
  )
  region = RegionSpec(
    id="domain",
    cell_refs=(CellRef(block.id, cell.id),),
    field_ids=(field.id,),
    material_id=material.id,
    formulation="small-strain-continuum",
    quadrature="gauss-3x3",
    source=_source("region-source"),
  )
  return ModelSpec(
    mesh=MeshSpec(nodes=nodes, cell_blocks=(block,), source=_source("mesh-source")),
    fields=(field,),
    materials=(material,),
    regions=(region,),
    source=_source("model-source"),
  )


def _mechanical_two_cell_model() -> ModelSpec:
  nodes = _two_cell_nodes()
  cells = (
    CellSpec(id=101, node_ids=(1, 2, 3, 4, 5, 6, 7, 8), source=_source("cell:101")),
    CellSpec(
      id=102,
      node_ids=(3, 9, 10, 11, 12, 13, 5, 4),
      source=_source("cell:102"),
    ),
  )
  block = CellBlockSpec(
    id="q8-cells",
    reference_topology="quadrilateral",
    topological_dimension=2,
    embedding_dimension=2,
    geometry_interpolation="serendipity-quad8",
    cells=cells,
    source=_source("cell-block"),
  )
  material = MaterialSpec(
    id="steel",
    model="plane-stress-linear-elastic",
    parameters=(
      MaterialParameterSpec("youngs_modulus", 210.0e9, _source("material:E")),
      MaterialParameterSpec("poisson_ratio", 0.3, _source("material:nu")),
    ),
    source=_source("material"),
  )
  field = FieldSpec(
    id="displacement",
    components=("x", "y"),
    location="node",
    source=_source("field"),
  )
  region = RegionSpec(
    id="domain",
    cell_refs=tuple(CellRef(block.id, cell.id) for cell in cells),
    field_ids=(field.id,),
    material_id=material.id,
    formulation="small-strain-continuum",
    quadrature="gauss-3x3",
    source=_source("region"),
  )
  return ModelSpec(
    mesh=MeshSpec(nodes=nodes, cell_blocks=(block,), source=_source("mesh")),
    fields=(field,),
    materials=(material,),
    regions=(region,),
    source=_source("model"),
  )


def _truss_model() -> ModelSpec:
  coordinates = ((-10.0, 0.0), (10.0, 0.0), (0.0, 0.5))
  nodes = tuple(
    NodeSpec(id=index, coordinates=point, source=_source(f"node:{index}"))
    for index, point in enumerate(coordinates, start=1)
  )
  cells = (
    CellSpec(id="cell-1", node_ids=(1, 3), source=_source("cell:1")),
    CellSpec(id="cell-2", node_ids=(2, 3), source=_source("cell:2")),
  )
  block = CellBlockSpec(
    id="bars",
    reference_topology="line",
    topological_dimension=1,
    embedding_dimension=2,
    geometry_interpolation="line2",
    cells=cells,
    source=_source("cell-block"),
  )
  field = FieldSpec(
    id="displacement",
    components=("x", "y"),
    location="node",
    source=_source("field"),
  )
  material = MaterialSpec(
    id="steel",
    model="uniaxial-linear-elastic",
    parameters=(
      MaterialParameterSpec(
        "youngs_modulus", 5.0e6, _source("material:youngs_modulus")
      ),
      MaterialParameterSpec("area", 1.0, _source("material:area")),
    ),
    source=_source("material"),
  )
  region = RegionSpec(
    id="domain",
    cell_refs=(CellRef("bars", "cell-1"), CellRef("bars", "cell-2")),
    field_ids=(field.id,),
    material_id=material.id,
    formulation="total-lagrangian-truss",
    quadrature="none",
    source=_source("region"),
  )
  return ModelSpec(
    mesh=MeshSpec(nodes=nodes, cell_blocks=(block,), source=_source("mesh")),
    fields=(field,),
    materials=(material,),
    regions=(region,),
    source=_source("model"),
  )


def _combined_registry() -> dict[RegistryKey, RegistryDescriptor]:
  return {**thermo_elastic_reference_registry(), **thermal_reference_registry()}


def _descriptor_with(
  registry: dict[RegistryKey, RegistryDescriptor],
  metadata_for: Callable[[str, str], dict[str, object]],
  key: RegistryKey,
  *,
  implementation_id: str = "changed-implementation",
  metadata: dict[str, object] | None = None,
  binding: object | None = None,
) -> RegistryDescriptor:
  reference = registry[key]
  return RegistryDescriptor(
    kind=key[0],
    name=key[1],
    version=reference.version,
    implementation_id=implementation_id,
    metadata=metadata_for(*key) if metadata is None else metadata,
    binding=reference.binding if binding is None else binding,
  )


def _pickle_round_trip(value: object) -> object:
  return pickle.loads(pickle.dumps(value))


def _independent_shapes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  xi = points[:, 0]
  eta = points[:, 1]
  values = np.stack(
    (
      -Fraction(1, 4) * (1 - xi) * (1 - eta) * (1 + xi + eta),
      Fraction(1, 2) * (1 - xi) * (1 + xi) * (1 - eta),
      -Fraction(1, 4) * (1 + xi) * (1 - eta) * (1 - xi + eta),
      Fraction(1, 2) * (1 + xi) * (1 + eta) * (1 - eta),
      -Fraction(1, 4) * (1 + xi) * (1 + eta) * (1 - xi - eta),
      Fraction(1, 2) * (1 - xi) * (1 + xi) * (1 + eta),
      -Fraction(1, 4) * (1 - xi) * (1 + eta) * (1 + xi - eta),
      Fraction(1, 2) * (1 - xi) * (1 + eta) * (1 - eta),
    ),
    axis=1,
  ).astype(np.float64)
  dxi = np.stack(
    (
      -Fraction(1, 4) * (-1 + eta) * (2 * xi + eta),
      xi * (-1 + eta),
      Fraction(1, 4) * (-1 + eta) * (-2 * xi + eta),
      -Fraction(1, 2) * (1 + eta) * (-1 + eta),
      Fraction(1, 4) * (1 + eta) * (2 * xi + eta),
      -xi * (1 + eta),
      -Fraction(1, 4) * (1 + eta) * (-2 * xi + eta),
      Fraction(1, 2) * (1 + eta) * (-1 + eta),
    ),
    axis=1,
  ).astype(np.float64)
  deta = np.stack(
    (
      -Fraction(1, 4) * (-1 + xi) * (xi + 2 * eta),
      Fraction(1, 2) * (1 + xi) * (-1 + xi),
      Fraction(1, 4) * (1 + xi) * (-xi + 2 * eta),
      -eta * (1 + xi),
      Fraction(1, 4) * (1 + xi) * (xi + 2 * eta),
      -Fraction(1, 2) * (1 + xi) * (-1 + xi),
      -Fraction(1, 4) * (-1 + xi) * (-xi + 2 * eta),
      eta * (-1 + xi),
    ),
    axis=1,
  ).astype(np.float64)
  return values, np.stack((dxi, deta), axis=2)


def _independent_blocks(
  coordinates: np.ndarray,
  youngs_modulus: float,
  poisson_ratio: float,
  thermal_expansion: float,
  conductivity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Hand-derived coupled element blocks straight from the physical definitions."""
  abscissa = 0.7745966692414834
  points = np.array(
    [(x, y) for x in (-abscissa, 0.0, abscissa) for y in (-abscissa, 0.0, abscissa)],
    dtype=np.float64,
  )
  fifth = float(Fraction(5, 9))
  eighth = float(Fraction(8, 9))
  weights = np.array(
    [x * y for x in (fifth, eighth, fifth) for y in (fifth, eighth, fifth)],
    dtype=np.float64,
  )
  shape_values, parent_gradients = _independent_shapes(points)
  modulus = Fraction.from_float(youngs_modulus)
  ratio = Fraction.from_float(poisson_ratio)
  normal = modulus / ((1 - ratio) * (1 + ratio))
  coupling = normal * ratio
  shear = modulus / (2 * (1 + ratio))
  constitutive = np.array(
    [
      [float(normal), float(coupling), 0.0],
      [float(coupling), float(normal), 0.0],
      [0.0, 0.0, float(shear)],
    ],
    dtype=np.float64,
  )
  dilatation = constitutive @ np.array(
    [thermal_expansion, thermal_expansion, 0.0],
    dtype=np.float64,
  )
  k_uu = np.zeros((16, 16), dtype=np.float64)
  k_ut = np.zeros((16, 8), dtype=np.float64)
  k_tt = np.zeros((8, 8), dtype=np.float64)
  for point_index in range(9):
    jacobian = coordinates.T @ parent_gradients[point_index]
    determinant = float(
      jacobian[0, 0] * jacobian[1, 1] - jacobian[0, 1] * jacobian[1, 0]
    )
    inverse = np.array(
      ((jacobian[1, 1], -jacobian[0, 1]), (-jacobian[1, 0], jacobian[0, 0])),
      dtype=np.float64,
    )
    gradients = parent_gradients[point_index] @ (inverse / determinant)
    b_matrix = np.zeros((3, 16), dtype=np.float64)
    b_matrix[0, 0::2] = gradients[:, 0]
    b_matrix[1, 1::2] = gradients[:, 1]
    b_matrix[2, 0::2] = gradients[:, 1]
    b_matrix[2, 1::2] = gradients[:, 0]
    measure = weights[point_index] * determinant
    k_uu += measure * b_matrix.T @ constitutive @ b_matrix
    k_ut += measure * np.outer(b_matrix.T @ dilatation, shape_values[point_index])
    k_tt += measure * conductivity * gradients @ gradients.T
  return k_uu, k_ut, k_tt


def _coupled_operator(model: ModelSpec | None = None) -> Q8ThermoElasticOperator:
  system = compile_system(
    _coupled_model() if model is None else model,
    thermo_elastic_reference_registry(),
  )
  operator = system.operators[0]
  assert isinstance(operator, Q8ThermoElasticOperator)
  return operator


def _coupled_inputs(
  operator: Q8ThermoElasticOperator,
  displacements: np.ndarray | None = None,
  temperatures: np.ndarray | None = None,
  request: ChannelRequest | None = None,
) -> OperatorEvaluationInput:
  if displacements is None:
    displacements = np.zeros((1, 16), dtype=np.float64)
  if temperatures is None:
    temperatures = np.zeros((1, 8), dtype=np.float64)
  if request is None:
    request = ChannelRequest(
      tuple(item.channel_id for item in operator.header.residual_channels),
      tuple(item.channel_id for item in operator.header.jacobian_channels),
    )
  return OperatorEvaluationInput(
    port_values=(
      FinalizedArray(displacements, dtype=np.float64),
      FinalizedArray(temperatures, dtype=np.float64),
    ),
    accepted_state=FinalizedArray(np.empty((1, 0)), dtype=np.float64),
    signals=(),
    request=request,
  )


def test_single_field_systems_compile_byte_identical_to_landed_manifests() -> None:
  cases = (
    (
      compile_system(_mechanical_unit_model(), q8_reference_registry()),
      _Q8_ONE_CELL_DIGEST,
    ),
    (
      compile_system(_mechanical_two_cell_model(), q8_reference_registry()),
      _Q8_TWO_CELL_DIGEST,
    ),
    (
      compile_system(_truss_model(), truss_reference_registry()),
      _TRUSS_TWO_CELL_DIGEST,
    ),
  )
  for system, digest in cases:
    manifest_bytes = system.provenance.manifest.to_bytes()
    assert hashlib.sha256(manifest_bytes).hexdigest() == digest
    assert system.content_fingerprint.digest == digest


def test_coupled_system_compiles_two_spaces_ports_and_cross_field_channels() -> None:
  system = compile_system(_coupled_model(), thermo_elastic_reference_registry())

  assert not hasattr(system, "space")
  assert tuple(space.space_id for space in system.spaces) == (
    "displacement",
    "temperature",
  )
  assert tuple(space.coefficient_range for space in system.spaces) == (
    (0, 16),
    (16, 24),
  )
  assert system.coefficient_count == 24
  displacement, temperature = system.spaces
  np.testing.assert_array_equal(
    displacement.coefficient_map.values,
    np.arange(16, dtype=np.int64).reshape(8, 2),
  )
  np.testing.assert_array_equal(
    temperature.coefficient_map.values,
    np.arange(16, 24, dtype=np.int64).reshape(8, 1),
  )
  assert displacement.coefficient_ids[:2] == (
    ("displacement", 1, "x"),
    ("displacement", 1, "y"),
  )
  assert temperature.coefficient_ids[0] == ("temperature", 1, "theta")
  assert len(system.operators) == 1
  assert len(system.entity_blocks) == 1
  np.testing.assert_array_equal(
    system.entity_blocks[0].incidence.values,
    [tuple(range(8))],
  )

  operator = system.operators[0]
  assert isinstance(operator, Q8ThermoElasticOperator)
  header = operator.header
  assert header.block_id == ("cells", "domain")
  assert header.entity_block_id == "cells"
  assert header.signal_ports == ()
  assert tuple(port.port_id for port in header.ports) == (
    "displacement",
    "temperature",
  )
  assert tuple(port.space_id for port in header.ports) == (
    "displacement",
    "temperature",
  )
  np.testing.assert_array_equal(
    header.ports[0].coefficient_map.values,
    np.arange(16, dtype=np.int64).reshape(1, 16),
  )
  np.testing.assert_array_equal(
    header.ports[1].coefficient_map.values,
    np.arange(16, 24, dtype=np.int64).reshape(1, 8),
  )
  residuals = header.residual_channels
  assert tuple(item.channel_id for item in residuals) == (
    "internal-force",
    "heat-flux",
  )
  assert tuple(item.target_port_id for item in residuals) == (
    "displacement",
    "temperature",
  )
  assert all(item.balance_role is BalanceRole.INTERNAL for item in residuals)
  assert all(item.linear for item in residuals)
  jacobians = header.jacobian_channels
  assert tuple(item.channel_id for item in jacobians) == (
    "material-tangent",
    "thermal-expansion-tangent",
    "conduction-tangent",
  )
  assert tuple((item.target_port_id, item.source_port_id) for item in jacobians) == (
    ("displacement", "displacement"),
    ("displacement", "temperature"),
    ("temperature", "temperature"),
  )
  assert all(item.balance_role is BalanceRole.INTERNAL for item in jacobians)
  assert all(item.linear for item in jacobians)
  assert tuple(item.symmetric for item in jacobians) == (True, False, True)
  assert header.state_layout.slots == ()
  assert header.state_layout.row_shape == (1, 0)
  assert header.state_layout.lifetime is StateLifetime.ACCEPTED_TRIAL
  np.testing.assert_array_equal(header.state_layout.entity_offsets.values, [0, 0])
  identities = {(item.kind, item.name) for item in header.implementations}
  assert identities == {
    Q8_TOPOLOGY_KEY,
    ("quadrature", "gauss-3x3"),
    THERMO_FORMULATION_KEY,
    THERMO_MATERIAL_KEY,
  }
  assert system.source_for("operator", ("cells", "domain")).source == "region:domain"
  assert system.source_for("space", "temperature").source == "field:temperature"
  assert (
    system.source_for(
      "material_parameter", ("thermo-elastic", "thermal_expansion")
    ).source
    == "material:thermal_expansion"
  )


def test_coupled_operator_matches_hand_derived_blocks_and_mechanical_tangent() -> None:
  operator = _coupled_operator()
  result = operator.evaluate(_coupled_inputs(operator))
  coordinates = np.array(_UNIT_COORDINATES, dtype=np.float64)
  k_uu, k_ut, k_tt = _independent_blocks(coordinates, 1.0, 0.0, 0.5, 2.0)
  material, expansion, conduction = result.jacobian_values
  np.testing.assert_allclose(material.values[0], k_uu, rtol=0.0, atol=1.0e-14)
  np.testing.assert_allclose(expansion.values[0], -k_ut, rtol=0.0, atol=1.0e-14)
  np.testing.assert_allclose(conduction.values[0], k_tt, rtol=0.0, atol=1.0e-14)

  mechanical = compile_system(_mechanical_unit_model(), q8_reference_registry())
  mechanical_operator = mechanical.operators[0]
  assert isinstance(mechanical_operator, Q8ContinuumOperator)
  mechanical_tangent = mechanical_operator.evaluate(
    OperatorEvaluationInput(
      port_values=(FinalizedArray(np.zeros((1, 16)), dtype=np.float64),),
      accepted_state=FinalizedArray(np.empty((1, 0)), dtype=np.float64),
      signals=(),
      request=ChannelRequest((), ("material-tangent",)),
    )
  )
  np.testing.assert_array_equal(
    material.values,
    mechanical_tangent.jacobian_values[0].values,
  )

  displacements = np.linspace(-0.02, 0.05, 16, dtype=np.float64).reshape(1, 16)
  temperatures = np.linspace(0.1, 0.8, 8, dtype=np.float64).reshape(1, 8)
  evaluated = operator.evaluate(_coupled_inputs(operator, displacements, temperatures))
  np.testing.assert_allclose(
    evaluated.residual_values[0].values[0],
    k_uu @ displacements[0] - k_ut @ temperatures[0],
    rtol=0.0,
    atol=2.0e-14,
  )
  np.testing.assert_allclose(
    evaluated.residual_values[1].values[0],
    k_tt @ temperatures[0],
    rtol=0.0,
    atol=2.0e-14,
  )


def test_free_thermal_expansion_produces_no_internal_force() -> None:
  operator = _coupled_operator()
  alpha = 0.5
  displacements = np.array(
    [[alpha * x, alpha * y] for x, y in _UNIT_COORDINATES],
    dtype=np.float64,
  ).reshape(1, 16)
  temperatures = np.full((1, 8), 1.0, dtype=np.float64)
  result = operator.evaluate(_coupled_inputs(operator, displacements, temperatures))
  np.testing.assert_allclose(
    result.residual_values[0].values,
    np.zeros((1, 16)),
    rtol=0.0,
    atol=1.0e-14,
  )
  np.testing.assert_allclose(
    result.residual_values[1].values,
    np.zeros((1, 8)),
    rtol=0.0,
    atol=1.0e-14,
  )


def test_coupled_evaluation_is_linear_deterministic_and_channel_gated() -> None:
  operator = _coupled_operator()
  displacements = np.linspace(-0.03, 0.04, 16, dtype=np.float64).reshape(1, 16)
  temperatures = np.linspace(0.2, 0.9, 8, dtype=np.float64).reshape(1, 8)
  request = ChannelRequest(
    ("internal-force", "heat-flux"),
    ("material-tangent", "thermal-expansion-tangent", "conduction-tangent"),
  )
  first = operator.evaluate(
    _coupled_inputs(operator, displacements, temperatures, request)
  )
  second = operator.evaluate(
    _coupled_inputs(operator, displacements, temperatures, request)
  )
  for left, right in zip(first.residual_values, second.residual_values, strict=True):
    np.testing.assert_array_equal(left.values, right.values)
  doubled = operator.evaluate(
    _coupled_inputs(
      operator,
      2.0 * displacements,
      2.0 * temperatures,
      ChannelRequest(("internal-force", "heat-flux"), ()),
    )
  )
  assert doubled.jacobian_values == ()
  for base, scaled in zip(first.residual_values, doubled.residual_values, strict=True):
    np.testing.assert_array_equal(2.0 * base.values, scaled.values)

  force_only = operator.evaluate(
    _coupled_inputs(
      operator,
      displacements,
      temperatures,
      ChannelRequest(("internal-force",), ()),
    )
  )
  assert len(force_only.residual_values) == 1
  np.testing.assert_array_equal(
    force_only.residual_values[0].values,
    first.residual_values[0].values,
  )
  heat_only = operator.evaluate(
    _coupled_inputs(
      operator,
      displacements,
      temperatures,
      ChannelRequest(("heat-flux",), ()),
    )
  )
  assert len(heat_only.residual_values) == 1
  np.testing.assert_array_equal(
    heat_only.residual_values[0].values,
    first.residual_values[1].values,
  )
  jacobian_only = operator.evaluate(
    _coupled_inputs(
      operator,
      displacements,
      temperatures,
      ChannelRequest((), ("conduction-tangent", "material-tangent")),
    )
  )
  assert jacobian_only.residual_values == ()
  assert len(jacobian_only.jacobian_values) == 2
  np.testing.assert_array_equal(
    jacobian_only.jacobian_values[0].values,
    first.jacobian_values[0].values,
  )
  np.testing.assert_array_equal(
    jacobian_only.jacobian_values[1].values,
    first.jacobian_values[2].values,
  )
  zero = operator.evaluate(_coupled_inputs(operator))
  np.testing.assert_array_equal(
    zero.residual_values[0].values,
    np.zeros((1, 16)),
  )
  np.testing.assert_array_equal(
    zero.residual_values[1].values,
    np.zeros((1, 8)),
  )


def test_coupled_evaluation_validates_its_exact_inputs() -> None:
  operator = _coupled_operator()
  good = _coupled_inputs(operator)
  result = operator.evaluate(good)
  assert result.trial_state.values.shape == (1, 0)

  with pytest.raises(TypeError, match="exact immutable evaluation input"):
    operator.evaluate(object())  # type: ignore[arg-type]
  with pytest.raises(TypeError, match="displacement and temperature batches"):
    operator.evaluate(replace(good, port_values=good.port_values[:1]))
  wrong_shape = OperatorEvaluationInput(
    port_values=(
      FinalizedArray(np.zeros((1, 8)), dtype=np.float64),
      good.port_values[1],
    ),
    accepted_state=good.accepted_state,
    signals=(),
    request=good.request,
  )
  with pytest.raises(TypeError, match="finite metadata-free float64 batches"):
    operator.evaluate(wrong_shape)
  non_finite = OperatorEvaluationInput(
    port_values=(
      FinalizedArray(np.full((1, 16), np.nan), dtype=np.float64),
      good.port_values[1],
    ),
    accepted_state=good.accepted_state,
    signals=(),
    request=good.request,
  )
  with pytest.raises(TypeError, match="finite metadata-free float64 batches"):
    operator.evaluate(non_finite)
  wrong_state = OperatorEvaluationInput(
    port_values=good.port_values,
    accepted_state=FinalizedArray(np.empty((1, 1)), dtype=np.float64),
    signals=(),
    request=good.request,
  )
  with pytest.raises(TypeError, match="zero-width state layout"):
    operator.evaluate(wrong_state)
  signaled = OperatorEvaluationInput(
    port_values=good.port_values,
    accepted_state=good.accepted_state,
    signals=(
      ProgramSignalInput(
        port_id="amplitude",
        values=FinalizedArray([1.0], dtype=np.float64),
        derivatives=(),
      ),
    ),
    request=good.request,
  )
  with pytest.raises(ValueError, match="does not accept program signal inputs"):
    operator.evaluate(signaled)
  duplicated = OperatorEvaluationInput(
    port_values=good.port_values,
    accepted_state=good.accepted_state,
    signals=(),
    request=ChannelRequest(("internal-force", "internal-force"), ()),
  )
  with pytest.raises(ValueError, match="unavailable or duplicate channel"):
    operator.evaluate(duplicated)
  unavailable = OperatorEvaluationInput(
    port_values=good.port_values,
    accepted_state=good.accepted_state,
    signals=(),
    request=ChannelRequest((), ("material-tangent", "shear-tangent")),
  )
  with pytest.raises(ValueError, match="unavailable or duplicate channel"):
    operator.evaluate(unavailable)


def test_split_system_allocates_exact_supports_without_ghost_dofs() -> None:
  system = compile_system(_split_model(), _combined_registry())

  assert tuple(space.space_id for space in system.spaces) == (
    "displacement",
    "temperature",
  )
  displacement, temperature = system.spaces
  assert displacement.coefficient_range == (0, 16)
  assert temperature.coefficient_range == (16, 29)
  assert system.coefficient_count == 29
  displacement_nodes = sorted(
    {coefficient_id[1] for coefficient_id in displacement.coefficient_ids}
  )
  temperature_nodes = sorted(
    {coefficient_id[1] for coefficient_id in temperature.coefficient_ids}
  )
  assert displacement_nodes == [1, 2, 3, 4, 5, 6, 7, 8]
  assert temperature_nodes == list(range(1, 14))
  np.testing.assert_array_equal(
    displacement.coefficient_map.values,
    np.arange(16, dtype=np.int64).reshape(8, 2),
  )
  np.testing.assert_array_equal(
    temperature.coefficient_map.values,
    np.arange(16, 29, dtype=np.int64).reshape(13, 1),
  )

  assert len(system.operators) == 2
  assert len(system.entity_blocks) == 2
  assert tuple(block.block_id for block in system.entity_blocks) == (
    "coupled-cells",
    "thermal-cells",
  )
  coupled_operator, thermal_operator = system.operators
  assert isinstance(coupled_operator, Q8ThermoElasticOperator)
  assert isinstance(thermal_operator, Q8ThermalOperator)
  assert coupled_operator.header.block_id == ("coupled-cells", "coupled-region")
  assert thermal_operator.header.block_id == ("thermal-cells", "thermal-region")
  np.testing.assert_array_equal(
    coupled_operator.header.ports[0].coefficient_map.values,
    np.arange(16, dtype=np.int64).reshape(1, 16),
  )
  np.testing.assert_array_equal(
    coupled_operator.header.ports[1].coefficient_map.values,
    np.arange(16, 24, dtype=np.int64).reshape(1, 8),
  )
  np.testing.assert_array_equal(
    thermal_operator.header.ports[0].coefficient_map.values,
    [[18, 24, 25, 26, 27, 28, 20, 19]],
  )
  assert system.source_for("operator", ("thermal-cells", "thermal-region")).source == (
    "region:thermal"
  )

  k_uu, k_ut, k_tt = _independent_blocks(
    np.array(_UNIT_COORDINATES, dtype=np.float64),
    1.0,
    0.0,
    0.5,
    2.0,
  )
  coupled_result = coupled_operator.evaluate(
    OperatorEvaluationInput(
      port_values=(
        FinalizedArray(np.zeros((1, 16)), dtype=np.float64),
        FinalizedArray(np.zeros((1, 8)), dtype=np.float64),
      ),
      accepted_state=FinalizedArray(np.empty((1, 0)), dtype=np.float64),
      signals=(),
      request=ChannelRequest(
        (),
        ("material-tangent", "thermal-expansion-tangent", "conduction-tangent"),
      ),
    )
  )
  np.testing.assert_allclose(
    coupled_result.jacobian_values[0].values[0], k_uu, rtol=0.0, atol=1.0e-14
  )
  np.testing.assert_allclose(
    coupled_result.jacobian_values[1].values[0], -k_ut, rtol=0.0, atol=1.0e-14
  )
  np.testing.assert_allclose(
    coupled_result.jacobian_values[2].values[0], k_tt, rtol=0.0, atol=1.0e-14
  )


def test_thermal_operator_matches_hand_derived_conduction_block() -> None:
  system = compile_system(_split_model(), _combined_registry())
  thermal_operator = system.operators[1]
  assert isinstance(thermal_operator, Q8ThermalOperator)
  header = thermal_operator.header
  assert tuple(item.channel_id for item in header.residual_channels) == ("heat-flux",)
  assert tuple(item.channel_id for item in header.jacobian_channels) == (
    "conduction-tangent",
  )
  assert header.residual_channels[0].linear
  assert header.jacobian_channels[0].symmetric
  assert header.state_layout.row_shape == (1, 0)

  coordinates = np.array(
    [_TWO_CELL_COORDINATES[index - 1] for index in (3, 9, 10, 11, 12, 13, 5, 4)],
    dtype=np.float64,
  )
  _, _, k_tt_unit = _independent_blocks(coordinates, 1.0, 0.0, 0.5, 1.0)
  expected = 3.0 * k_tt_unit
  temperatures = np.linspace(0.3, 1.1, 8, dtype=np.float64).reshape(1, 8)
  result = thermal_operator.evaluate(
    OperatorEvaluationInput(
      port_values=(FinalizedArray(temperatures, dtype=np.float64),),
      accepted_state=FinalizedArray(np.empty((1, 0)), dtype=np.float64),
      signals=(),
      request=ChannelRequest(("heat-flux",), ("conduction-tangent",)),
    )
  )
  np.testing.assert_allclose(
    result.jacobian_values[0].values[0], expected, rtol=0.0, atol=1.0e-13
  )
  np.testing.assert_allclose(
    result.residual_values[0].values[0],
    expected @ temperatures[0],
    rtol=0.0,
    atol=1.0e-13,
  )
  np.testing.assert_allclose(
    result.jacobian_values[0].values[0].sum(1),
    np.zeros(8),
    rtol=0.0,
    atol=1.0e-13,
  )


def test_thermal_evaluation_is_channel_gated_and_validated() -> None:
  system = compile_system(_thermal_model(), thermal_reference_registry())
  operator = system.operators[0]
  assert isinstance(operator, Q8ThermalOperator)
  assert system.spaces[0].coefficient_range == (0, 8)
  good = OperatorEvaluationInput(
    port_values=(FinalizedArray(np.zeros((1, 8)), dtype=np.float64),),
    accepted_state=FinalizedArray(np.empty((1, 0)), dtype=np.float64),
    signals=(),
    request=ChannelRequest(("heat-flux",), ("conduction-tangent",)),
  )
  result = operator.evaluate(good)
  assert len(result.residual_values) == 1
  assert len(result.jacobian_values) == 1
  jacobian_only = operator.evaluate(
    replace(good, request=ChannelRequest((), ("conduction-tangent",)))
  )
  assert jacobian_only.residual_values == ()
  assert len(jacobian_only.jacobian_values) == 1

  with pytest.raises(TypeError, match="exact immutable evaluation input"):
    operator.evaluate(object())  # type: ignore[arg-type]
  with pytest.raises(TypeError, match="exactly one temperature port batch"):
    operator.evaluate(replace(good, port_values=()))
  wrong_shape = replace(
    good,
    port_values=(FinalizedArray(np.zeros((1, 16)), dtype=np.float64),),
  )
  with pytest.raises(TypeError, match="finite metadata-free float64 batch"):
    operator.evaluate(wrong_shape)
  wrong_state = replace(
    good,
    accepted_state=FinalizedArray(np.empty((1, 2)), dtype=np.float64),
  )
  with pytest.raises(TypeError, match="zero-width state layout"):
    operator.evaluate(wrong_state)
  duplicated = replace(good, request=ChannelRequest(("heat-flux", "heat-flux"), ()))
  with pytest.raises(ValueError, match="unavailable or duplicate channel"):
    operator.evaluate(duplicated)


def test_multifield_compilation_is_canonical_permutation_invariant() -> None:
  authored = _split_model()
  mesh = authored.mesh
  materials = authored.materials
  reordered = replace(
    authored,
    mesh=replace(
      mesh,
      nodes=tuple(reversed(mesh.nodes)),
      cell_blocks=tuple(
        replace(block, cells=tuple(reversed(block.cells)))
        for block in reversed(mesh.cell_blocks)
      ),
    ),
    fields=tuple(reversed(authored.fields)),
    materials=tuple(
      replace(material, parameters=tuple(reversed(material.parameters)))
      for material in reversed(materials)
    ),
    regions=tuple(
      replace(
        region,
        cell_refs=tuple(reversed(region.cell_refs)),
      )
      for region in reversed(authored.regions)
    ),
  )
  first = compile_system(authored, _combined_registry())
  second = compile_system(reordered, _combined_registry())
  assert first.content_fingerprint == second.content_fingerprint
  assert first.instance_id != second.instance_id
  assert (
    compile_system(
      _coupled_model(), thermo_elastic_reference_registry()
    ).content_fingerprint
    != first.content_fingerprint
  )


def test_multifield_compiled_system_owns_arrays_and_trusted_carriers() -> None:
  system = compile_system(_split_model(), _combined_registry())
  coupled_operator, thermal_operator = system.operators
  assert isinstance(coupled_operator, Q8ThermoElasticOperator)
  assert isinstance(thermal_operator, Q8ThermalOperator)
  arrays = (
    system.spaces[0].coefficient_map.values,
    system.spaces[1].coefficient_map.values,
    coupled_operator.header.ports[0].coefficient_map.values,
    coupled_operator.header.ports[1].coefficient_map.values,
    coupled_operator.payload.constitutive.values,
    coupled_operator.payload.thermal_expansion.values,
    coupled_operator.payload.conductivity.values,
    thermal_operator.payload.conductivity.values,
  )
  for array in arrays:
    assert array.flags.owndata and not array.flags.writeable
    assert array.dtype.metadata is None
  for carrier in (
    Q8ThermoElasticOperator,
    Q8ThermalOperator,
  ):
    with pytest.raises(TypeError, match="constructed only by their compiler"):
      carrier()
  trusted = (
    coupled_operator,
    coupled_operator.payload,
    thermal_operator,
    thermal_operator.payload,
    thermal_operator.header.ports[0],
    thermal_operator.header.jacobian_channels[0],
  )
  for value in trusted:
    for reconstruct in (copy.copy, copy.deepcopy, _pickle_round_trip):
      with pytest.raises(TypeError, match="cannot be reconstructed"):
        reconstruct(value)


def test_support_aware_space_allocation_validates_its_inputs() -> None:
  points = compile_system(
    _mechanical_unit_model(), q8_reference_registry()
  ).point_blocks[0]
  displacement = FieldSpec("displacement", ("x", "y"), "node")
  temperature = FieldSpec("temperature", ("theta",), "node")
  supports = {
    "displacement": (1, 2, 3, 4),
    "temperature": (3, 4, 5, 6, 7, 8),
  }
  spaces = compile_discrete_spaces(
    points,
    (displacement, temperature),
    supports=supports,
  )
  assert tuple(space.coefficient_range for space in spaces) == ((0, 8), (8, 14))
  assert spaces[0].coefficient_ids[-1] == ("displacement", 4, "y")
  assert spaces[1].coefficient_ids[0] == ("temperature", 3, "theta")
  np.testing.assert_array_equal(
    spaces[1].coefficient_map.values,
    np.arange(8, 14, dtype=np.int64).reshape(6, 1),
  )

  with pytest.raises(ModelCompilationError, match="invalid-space-support"):
    compile_discrete_spaces(
      points,
      (displacement, temperature),
      supports={"displacement": (1, 2)},
    )
  with pytest.raises(ModelCompilationError, match="invalid-space-support"):
    compile_discrete_spaces(
      points,
      (displacement, temperature),
      supports={"displacement": (1, 2, 3, 4), "temperature": (3, 4, 99)},
    )


def _material_with_model(material: MaterialSpec, model: str) -> MaterialSpec:
  return replace(material, model=model)


def _shared_block_model() -> ModelSpec:
  nodes = _two_cell_nodes()
  cells = (
    CellSpec(id=101, node_ids=(1, 2, 3, 4, 5, 6, 7, 8), source=_source("cell:101")),
    CellSpec(
      id=102,
      node_ids=(3, 9, 10, 11, 12, 13, 5, 4),
      source=_source("cell:102"),
    ),
  )
  block = _q8_block("cells", cells)
  mechanical = RegionSpec(
    id="mechanical",
    cell_refs=(CellRef("cells", 101),),
    field_ids=("displacement",),
    material_id="elastic",
    formulation="small-strain-continuum",
    quadrature="gauss-3x3",
    source=_source("region:mechanical"),
  )
  thermal = RegionSpec(
    id="thermal",
    cell_refs=(CellRef("cells", 102),),
    field_ids=("temperature",),
    material_id="conductor",
    formulation="small-strain-thermal-continuum",
    quadrature="gauss-3x3",
    source=_source("region:thermal"),
  )
  return ModelSpec(
    mesh=MeshSpec(nodes=nodes, cell_blocks=(block,), source=_source("mesh")),
    fields=(_displacement_field(), _temperature_field()),
    materials=(_elastic_material(), _conductor_material()),
    regions=(mechanical, thermal),
    source=_source("model"),
  )


@pytest.mark.parametrize(
  ("model", "registry", "code"),
  [
    (
      _coupled_model(field_ids=("temperature", "displacement")),
      thermo_elastic_reference_registry(),
      "incompatible-field-signature",
    ),
    (
      _coupled_model(field_ids=("displacement",)),
      thermo_elastic_reference_registry(),
      "incompatible-region-field-signature",
    ),
    (
      _coupled_model(
        fields=(
          _displacement_field(),
          FieldSpec("temperature", ("x", "y"), "node", _source("field:temperature")),
        )
      ),
      thermo_elastic_reference_registry(),
      "incompatible-field-signature",
    ),
    (
      _thermal_model(field_components=("theta", "phi")),
      thermal_reference_registry(),
      "incompatible-field-signature",
    ),
    (
      _thermal_model(formulation="mystery"),
      thermal_reference_registry(),
      "incompatible-formulation",
    ),
    (
      _thermal_model(quadrature="gauss-2x2"),
      thermal_reference_registry(),
      "incompatible-quadrature",
    ),
    (
      _coupled_model(
        materials=(
          _material_with_model(_thermo_elastic_material(), "linear-thermal-conductor"),
        )
      ),
      thermo_elastic_reference_registry(),
      "incompatible-material-model",
    ),
    (
      _thermal_model(
        materials=(
          _material_with_model(_conductor_material(), "plane-stress-linear-elastic"),
        )
      ),
      thermal_reference_registry(),
      "incompatible-material-model",
    ),
    (
      _coupled_model(
        fields=(
          _displacement_field(),
          _temperature_field(),
          FieldSpec("phase", ("phi",), "node", _source("field:phase")),
        )
      ),
      thermo_elastic_reference_registry(),
      "unreferenced-field-declaration",
    ),
    (
      _coupled_model(materials=(_thermo_elastic_material(), _conductor_material())),
      thermo_elastic_reference_registry(),
      "unreferenced-material-declaration",
    ),
    (
      replace(
        _split_model(),
        regions=(
          replace(
            _split_model().regions[0],
            cell_refs=(
              CellRef("coupled-cells", 101),
              CellRef("thermal-cells", 102),
            ),
          ),
        ),
      ),
      _combined_registry(),
      "unsupported-region-cell-block-span",
    ),
    (
      _shared_block_model(),
      _combined_registry(),
      "shared-cell-block",
    ),
    (
      replace(_split_model(), regions=_split_model().regions[:1]),
      _combined_registry(),
      "incomplete-cell-membership",
    ),
    (
      replace(
        _split_model(),
        regions=(
          _split_model().regions[0],
          replace(
            _split_model().regions[1],
            cell_refs=(
              CellRef("thermal-cells", 102),
              CellRef("coupled-cells", 101),
            ),
          ),
        ),
      ),
      _combined_registry(),
      "multiple-cell-membership",
    ),
    (
      _coupled_model(
        fields=(
          _displacement_field(),
          _temperature_field(),
          FieldSpec("cell-field", ("phi",), "cell", _source("field:cell")),
        )
      ),
      thermo_elastic_reference_registry(),
      "unsupported-space-support",
    ),
  ],
)
def test_incompatible_multifield_slices_fail_with_coded_diagnostics(
  model: ModelSpec,
  registry: dict[RegistryKey, RegistryDescriptor],
  code: str,
) -> None:
  with pytest.raises(ModelCompilationError, match=code):
    compile_system(model, registry)


@pytest.mark.parametrize(
  ("parameters", "code"),
  [
    ((), "invalid-material-parameter-schema"),
    (
      (("conductivity", 3.0), ("density", 1.0)),
      "invalid-material-parameter-schema",
    ),
    ((("conductivity", True),), "invalid-material-parameter-type"),
    ((("conductivity", _HUGE_INTEGER_ID),), "invalid-material-parameter-value"),
    ((("conductivity", 0.0),), "invalid-conductivity"),
    ((("conductivity", -2.0),), "invalid-conductivity"),
  ],
)
def test_conductor_parameter_domain_failures_are_coded(
  parameters: tuple[tuple[str, object], ...],
  code: str,
) -> None:
  with pytest.raises(ModelCompilationError, match=code):
    compile_system(
      _thermal_model(parameters=parameters),
      thermal_reference_registry(),
    )


@pytest.mark.parametrize(
  ("parameters", "code"),
  [
    (
      (
        ("youngs_modulus", 1.0),
        ("poisson_ratio", 0.0),
        ("conductivity", 2.0),
      ),
      "invalid-material-parameter-schema",
    ),
    (
      (
        ("youngs_modulus", 1.0),
        ("poisson_ratio", 0.0),
        ("thermal_expansion", 0.5),
        ("conductivity", 2.0),
        ("density", 1.0),
      ),
      "invalid-material-parameter-schema",
    ),
    (
      (
        ("youngs_modulus", 0.0),
        ("poisson_ratio", 0.0),
        ("thermal_expansion", 0.5),
        ("conductivity", 2.0),
      ),
      "invalid-youngs-modulus",
    ),
    (
      (
        ("youngs_modulus", 1.0),
        ("poisson_ratio", 0.5),
        ("thermal_expansion", 0.5),
        ("conductivity", 2.0),
      ),
      "invalid-poisson-ratio",
    ),
    (
      (
        ("youngs_modulus", 1.0),
        ("poisson_ratio", 0.0),
        ("thermal_expansion", True),
        ("conductivity", 2.0),
      ),
      "invalid-material-parameter-type",
    ),
    (
      (
        ("youngs_modulus", 1.0),
        ("poisson_ratio", 0.0),
        ("thermal_expansion", 0.5),
        ("conductivity", 0.0),
      ),
      "invalid-conductivity",
    ),
  ],
)
def test_thermo_elastic_parameter_domain_failures_are_coded(
  parameters: tuple[tuple[str, object], ...],
  code: str,
) -> None:
  with pytest.raises(ModelCompilationError, match=code):
    compile_system(
      _coupled_model(materials=(_thermo_elastic_material(parameters),)),
      thermo_elastic_reference_registry(),
    )


def test_integer_thermo_elastic_parameters_compile_to_float64() -> None:
  material = MaterialSpec(
    id="thermo-elastic",
    model="linear-thermo-elastic",
    parameters=(
      MaterialParameterSpec("youngs_modulus", 210, _source("material:E")),
      MaterialParameterSpec("poisson_ratio", 0, _source("material:nu")),
      MaterialParameterSpec("thermal_expansion", 1, _source("material:alpha")),
      MaterialParameterSpec("conductivity", 2, _source("material:k")),
    ),
    source=_source("material:thermo-elastic"),
  )
  system = compile_system(
    _coupled_model(materials=(material,)),
    thermo_elastic_reference_registry(),
  )
  operator = system.operators[0]
  assert isinstance(operator, Q8ThermoElasticOperator)
  parameters = operator.payload.material_parameters.values
  assert parameters.dtype == np.dtype(np.float64)
  np.testing.assert_array_equal(parameters, [[210.0, 0.0, 1.0, 2.0]])


def _zero_thermal_kinematics(gradients: np.ndarray) -> np.ndarray:
  return np.zeros((*gradients.shape[:2], 2, 8), dtype=np.float64)


def _short_thermal_kinematics(gradients: np.ndarray) -> np.ndarray:
  del gradients
  return np.zeros((9, 8), dtype=np.float64)


def _raising_thermal_kinematics(gradients: np.ndarray) -> np.ndarray:
  del gradients
  msg = "hostile thermal kinematics binding"
  raise RuntimeError(msg)


def _zero_conductor(conductivity: float) -> np.ndarray:
  del conductivity
  return np.zeros((2, 2), dtype=np.float64)


def _wide_conductor(conductivity: float) -> np.ndarray:
  del conductivity
  return np.zeros((3, 3), dtype=np.float64)


def _asymmetric_conductor(conductivity: float) -> np.ndarray:
  return np.array([[conductivity, conductivity], [0.0, conductivity]])


def _doubled_conductor(conductivity: float) -> np.ndarray:
  return np.array([[2.0 * conductivity, 0.0], [0.0, 2.0 * conductivity]])


def _raising_conductor(conductivity: float) -> np.ndarray:
  del conductivity
  msg = "hostile conductor binding"
  raise RuntimeError(msg)


@pytest.mark.parametrize(
  "missing_key",
  [Q8_TOPOLOGY_KEY, THERMAL_FORMULATION_KEY, THERMAL_MATERIAL_KEY],
)
def test_each_missing_thermal_registry_key_fails_closed(
  missing_key: RegistryKey,
) -> None:
  registry = thermal_reference_registry()
  registry.pop(missing_key)
  with pytest.raises(ModelCompilationError, match="registry-capture-failed"):
    compile_system(_thermal_model(), registry)


@pytest.mark.parametrize(
  "key",
  [THERMAL_FORMULATION_KEY, THERMAL_MATERIAL_KEY],
)
def test_thermal_descriptor_metadata_mismatch_is_rejected(key: RegistryKey) -> None:
  registry = thermal_reference_registry()
  metadata = thermal_descriptor_metadata(*key)
  metadata["incompatible_change"] = True
  registry[key] = _descriptor_with(
    registry, thermal_descriptor_metadata, key, metadata=metadata
  )
  with pytest.raises(ModelCompilationError, match="incompatible-registry-descriptor"):
    compile_system(_thermal_model(), registry)


@pytest.mark.parametrize(
  ("key", "binding", "code"),
  [
    (
      THERMAL_FORMULATION_KEY,
      _zero_thermal_kinematics,
      "incompatible-formulation-binding-output",
    ),
    (
      THERMAL_FORMULATION_KEY,
      _short_thermal_kinematics,
      "invalid-formulation-binding-output",
    ),
    (
      THERMAL_FORMULATION_KEY,
      _raising_thermal_kinematics,
      "formulation-binding-failed",
    ),
    (
      THERMAL_MATERIAL_KEY,
      _zero_conductor,
      "incompatible-material-binding-output",
    ),
    (
      THERMAL_MATERIAL_KEY,
      _wide_conductor,
      "invalid-material-binding-output",
    ),
    (
      THERMAL_MATERIAL_KEY,
      _asymmetric_conductor,
      "nonsymmetric-material-binding",
    ),
    (
      THERMAL_MATERIAL_KEY,
      _doubled_conductor,
      "incompatible-material-binding-output",
    ),
    (
      THERMAL_MATERIAL_KEY,
      _raising_conductor,
      "material-binding-failed",
    ),
  ],
)
def test_same_identity_spoofed_thermal_bindings_fail_at_compile_boundary(
  key: RegistryKey,
  binding: object,
  code: str,
) -> None:
  registry = thermal_reference_registry()
  registry[key] = _descriptor_with(
    registry, thermal_descriptor_metadata, key, binding=binding
  )
  with pytest.raises(ModelCompilationError, match=code):
    compile_system(_thermal_model(), registry)


def _single_coupled_kinematics(gradients: np.ndarray) -> np.ndarray:
  del gradients
  return np.zeros((9, 8), dtype=np.float64)


def _zero_b_coupled_kinematics(
  gradients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  return (
    np.zeros((*gradients.shape[:2], 3, 16), dtype=np.float64),
    thermal_gradient_map(gradients),
  )


def _zero_bt_coupled_kinematics(
  gradients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  del gradients
  return (
    np.zeros((1, 9, 3, 16), dtype=np.float64),
    np.zeros((1, 9, 2, 8), dtype=np.float64),
  )


def _raising_coupled_kinematics(gradients: np.ndarray) -> np.ndarray:
  del gradients
  msg = "hostile coupled kinematics binding"
  raise RuntimeError(msg)


def _two_tuple_coupled_material(
  youngs_modulus: float,
  poisson_ratio: float,
  thermal_expansion: float,
  conductivity: float,
) -> tuple[np.ndarray, np.ndarray]:
  matrix, expansion, _ = linear_thermo_elastic(
    youngs_modulus, poisson_ratio, thermal_expansion, conductivity
  )
  return matrix, expansion


def _doubled_coupled_matrix(
  youngs_modulus: float,
  poisson_ratio: float,
  thermal_expansion: float,
  conductivity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  matrix, expansion, conduction = linear_thermo_elastic(
    youngs_modulus, poisson_ratio, thermal_expansion, conductivity
  )
  return 2.0 * matrix, expansion, conduction


def _doubled_coupled_expansion(
  youngs_modulus: float,
  poisson_ratio: float,
  thermal_expansion: float,
  conductivity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  matrix, expansion, conduction = linear_thermo_elastic(
    youngs_modulus, poisson_ratio, thermal_expansion, conductivity
  )
  return matrix, 2.0 * expansion, conduction


def _doubled_coupled_conduction(
  youngs_modulus: float,
  poisson_ratio: float,
  thermal_expansion: float,
  conductivity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  matrix, expansion, conduction = linear_thermo_elastic(
    youngs_modulus, poisson_ratio, thermal_expansion, conductivity
  )
  return matrix, expansion, 2.0 * conduction


def _asymmetric_coupled_conduction(
  youngs_modulus: float,
  poisson_ratio: float,
  thermal_expansion: float,
  conductivity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  matrix, expansion, _ = linear_thermo_elastic(
    youngs_modulus, poisson_ratio, thermal_expansion, conductivity
  )
  return (
    matrix,
    expansion,
    np.array([[conductivity, conductivity], [0.0, conductivity]]),
  )


def _raising_coupled_material(
  youngs_modulus: float,
  poisson_ratio: float,
  thermal_expansion: float,
  conductivity: float,
) -> np.ndarray:
  del youngs_modulus, poisson_ratio, thermal_expansion, conductivity
  msg = "hostile coupled material binding"
  raise RuntimeError(msg)


@pytest.mark.parametrize(
  "missing_key",
  [THERMO_FORMULATION_KEY, THERMO_MATERIAL_KEY],
)
def test_each_missing_coupled_registry_key_fails_closed(
  missing_key: RegistryKey,
) -> None:
  registry = thermo_elastic_reference_registry()
  registry.pop(missing_key)
  with pytest.raises(ModelCompilationError, match="registry-capture-failed"):
    compile_system(_coupled_model(), registry)


@pytest.mark.parametrize(
  "key",
  [THERMO_FORMULATION_KEY, THERMO_MATERIAL_KEY],
)
def test_coupled_descriptor_metadata_mismatch_is_rejected(key: RegistryKey) -> None:
  registry = thermo_elastic_reference_registry()
  metadata = thermo_elastic_descriptor_metadata(*key)
  metadata["incompatible_change"] = True
  registry[key] = _descriptor_with(
    registry, thermo_elastic_descriptor_metadata, key, metadata=metadata
  )
  with pytest.raises(ModelCompilationError, match="incompatible-registry-descriptor"):
    compile_system(_coupled_model(), registry)


@pytest.mark.parametrize(
  ("key", "binding", "code"),
  [
    (
      THERMO_FORMULATION_KEY,
      _single_coupled_kinematics,
      "invalid-formulation-binding-output",
    ),
    (
      THERMO_FORMULATION_KEY,
      _zero_b_coupled_kinematics,
      "incompatible-formulation-binding-output",
    ),
    (
      THERMO_FORMULATION_KEY,
      _zero_bt_coupled_kinematics,
      "incompatible-formulation-binding-output",
    ),
    (
      THERMO_FORMULATION_KEY,
      _raising_coupled_kinematics,
      "formulation-binding-failed",
    ),
    (
      THERMO_MATERIAL_KEY,
      _two_tuple_coupled_material,
      "invalid-material-binding-output",
    ),
    (
      THERMO_MATERIAL_KEY,
      _doubled_coupled_matrix,
      "incompatible-material-binding-output",
    ),
    (
      THERMO_MATERIAL_KEY,
      _doubled_coupled_expansion,
      "incompatible-material-binding-output",
    ),
    (
      THERMO_MATERIAL_KEY,
      _doubled_coupled_conduction,
      "incompatible-material-binding-output",
    ),
    (
      THERMO_MATERIAL_KEY,
      _asymmetric_coupled_conduction,
      "nonsymmetric-material-binding",
    ),
    (
      THERMO_MATERIAL_KEY,
      _raising_coupled_material,
      "material-binding-failed",
    ),
  ],
)
def test_same_identity_spoofed_coupled_bindings_fail_at_compile_boundary(
  key: RegistryKey,
  binding: object,
  code: str,
) -> None:
  registry = thermo_elastic_reference_registry()
  registry[key] = _descriptor_with(
    registry, thermo_elastic_descriptor_metadata, key, binding=binding
  )
  with pytest.raises(ModelCompilationError, match=code):
    compile_system(_coupled_model(), registry)


def test_unselected_continuum_descriptors_do_not_enter_compiled_identity() -> None:
  combined = {**q8_reference_registry(), **thermo_elastic_reference_registry()}
  with_combined = compile_system(_mechanical_unit_model(), combined)
  with_q8_only = compile_system(_mechanical_unit_model(), q8_reference_registry())
  assert with_combined.content_fingerprint == with_q8_only.content_fingerprint
  assert len(with_combined.registry_snapshot.descriptors) == 4
  coupled = compile_system(_coupled_model(), combined)
  assert (
    coupled.content_fingerprint
    == compile_system(
      _coupled_model(),
      thermo_elastic_reference_registry(),
    ).content_fingerprint
  )


def test_huge_integer_geometry_diagnostic_is_bounded_under_digit_limit() -> None:
  model = _coupled_model(
    cell_id=_HUGE_INTEGER_ID,
    coordinates=tuple((x, y * 1.0e-14) for x, y in _UNIT_COORDINATES),
  )
  with pytest.raises(ModelCompilationError) as captured:
    compile_system(model, thermo_elastic_reference_registry())

  rendered = str(captured.value)
  assert len(rendered) < _MAX_HUGE_DIAGNOSTIC_LENGTH
  assert "<int sign=+ bits=" in rendered
  assert "near-singular-reference-geometry" in rendered
