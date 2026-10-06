# SPDX-License-Identifier: MIT

"""Finite-strain (total-Lagrangian) continuum family: structural oracles.

The total-Lagrangian Q8 slice compiles ``total-lagrangian-continuum`` regions
with the plane-stress Saint-Venant-Kirchhoff law onto the landed batched TL
kernel (``pyfem.v3.fem.tl_element.quad8_tl_tangent_batched``). The oracles
here are hand-derived:

- patch-level uniform deformation gradient recovery: an affine displacement
  field prescribed on a 2x2 Q8 patch yields exactly ``F = I + H`` at every
  quadrature point (recomputed from the compiled payload), and the per-element
  internal force equals an independent quadrature-loop assembly that
  re-derives the TL chain (F, Green-Lagrange E, PK2 S, B, B_NL) by hand from
  the pinned shape/quadrature recipes;
- objectivity: a pure rigid rotation carries zero Green-Lagrange strain, so
  the internal force vanishes (the small-strain operator on the same mesh
  develops a large spurious force — the contrast pins the strain measure);
- the consistent tangent is finite-difference checked against the residual at
  a genuinely deformed state;
- the cantilever8 skim converts, compiles, and drives to the legacy oracle
  state within its ``parity.toml``, and the assembled tangent and internal
  force at the converged state match the legacy tangent assembly.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from _legacy_parity import (
  legacy_nonlinear_state,
  legacy_tangent_at_state,
  load_parity_tolerances,
)

from pyfem.v3.compile.continuum import Q8FiniteStrainOperator, q8_reference_registry
from pyfem.v3.compile.contracts import (
  TL_FORMULATION_KEY,
  TL_MATERIAL_KEY,
  finite_strain_reference_registry,
  q8_descriptor_metadata,
)
from pyfem.v3.compile.diagnostics import ModelCompilationError
from pyfem.v3.compile.system import compile_system
from pyfem.v3.constraints import compile_constraint_map
from pyfem.v3.driver import DriverStatus
from pyfem.v3.driver.plan import (
  assemble_internal_force,
  compile_driver_plan,
  refill_tangent,
)
from pyfem.v3.fem.quadrature import gauss_tensor_product_2d
from pyfem.v3.fem.shapes import serendipity_quad8
from pyfem.v3.io.legacy_deck import compile_deck, read_legacy_deck, run_deck
from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.operator import (
  ChannelRequest,
  CompiledOperator,
  OperatorEvaluationInput,
)
from pyfem.v3.model.provenance import CanonicalManifest
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
from pyfem.v3.spec.program import ProgramCoordinateSpec

ROOT = Path(__file__).resolve().parents[2]
SKIMS = ROOT / "skims"
CANTILEVER8_PRO = SKIMS / "cantilever8" / "skim.pro"

_SOURCE = SourceContext()
_YOUNGS_MODULUS = 100.0
_POISSON_RATIO = 0.3
# The fixed moderate displacement gradient of the patch oracles: genuine
# finite strain (about 5-10%) with a healthy det(F) everywhere.
_PROBE_GRADIENT = np.array([[0.04, 0.02], [0.03, 0.07]], dtype=np.float64)

_TL_FORMULATION_METADATA = {
  "schema": "pyfem-v3-formulation-descriptor-v1",
  "field_quantity": "displacement",
  "field_location": "node",
  "field_components": ["x", "y"],
  "dofs_per_node": 2,
  "kinematic_regime": "finite-strain",
  "strain_measure": "green-lagrange",
  "stress_measure": "second-piola-kirchhoff",
  "reference_frame": "total-lagrangian",
  "strain_voigt_order": ["xx", "yy", "xy"],
  "shear_convention": "engineering",
  "formulation_history_width": 0,
  "tangent_contribution": "material-geometric",
  "tangent_symmetry": "symmetric",
}
_TL_MATERIAL_METADATA = {
  "schema": "pyfem-v3-material-descriptor-v1",
  "law": "saint-venant-kirchhoff",
  "stress_state": "plane-stress",
  "parameter_names": ["youngs_modulus", "poisson_ratio"],
  "parameter_dtype": "float64",
  "stress_voigt_order": ["xx", "yy", "xy"],
  "strain_shear_convention": "engineering",
  "material_history_width": 0,
  "tangent_class": "constant-symmetric",
}

# The one-cell TL system manifest digest: the byte pin of the compiled
# finite-strain path (the M16 digest idiom).
_TL_ONE_CELL_DIGEST = "5450d85d8039f2a70393eefb9cd38876151ea174ce7e9389671655518e2857e0"


def _tl_model(
  cell_nodes: tuple[tuple[int, ...], ...],
  coordinates: tuple[tuple[float, ...], ...],
) -> ModelSpec:
  nodes = tuple(
    NodeSpec(id=index, coordinates=point, source=_SOURCE)
    for index, point in enumerate(coordinates)
  )
  cells = tuple(
    CellSpec(id=index, node_ids=node_ids, source=_SOURCE)
    for index, node_ids in enumerate(cell_nodes)
  )
  block = CellBlockSpec(
    id="cells",
    reference_topology="quadrilateral",
    topological_dimension=2,
    embedding_dimension=2,
    geometry_interpolation="serendipity-quad8",
    cells=cells,
    source=_SOURCE,
  )
  field = FieldSpec(
    id="displacement",
    components=("x", "y"),
    location="node",
    source=_SOURCE,
  )
  material = MaterialSpec(
    id="stvk",
    model="plane-stress-saint-venant-kirchhoff",
    parameters=(
      MaterialParameterSpec("youngs_modulus", _YOUNGS_MODULUS),
      MaterialParameterSpec("poisson_ratio", _POISSON_RATIO),
    ),
    source=_SOURCE,
  )
  region = RegionSpec(
    id="domain",
    cell_refs=tuple(CellRef(block.id, cell.id) for cell in cells),
    field_ids=(field.id,),
    material_id=material.id,
    formulation="total-lagrangian-continuum",
    quadrature="gauss-3x3",
    source=_SOURCE,
  )
  return ModelSpec(
    mesh=MeshSpec(nodes=nodes, cell_blocks=(block,), source=_SOURCE),
    fields=(field,),
    materials=(material,),
    regions=(region,),
    source=_SOURCE,
  )


def _model_with(
  model: ModelSpec,
  *,
  formulation: str | None = None,
  material_model: str | None = None,
  quadrature: str | None = None,
) -> ModelSpec:
  region = model.regions[0]
  material = model.materials[0]
  if material_model is not None:
    material = MaterialSpec(
      id=material.id,
      model=material_model,
      parameters=material.parameters,
      source=material.source,
    )
  return ModelSpec(
    mesh=model.mesh,
    fields=model.fields,
    materials=(material,),
    regions=(
      RegionSpec(
        id=region.id,
        cell_refs=region.cell_refs,
        field_ids=region.field_ids,
        material_id=material.id,
        formulation=formulation or region.formulation,
        quadrature=quadrature or region.quadrature,
        source=region.source,
      ),
    ),
    source=model.source,
  )


_UNIT_CELL_NODES = ((0, 1, 2, 3, 4, 5, 6, 7),)
_UNIT_CELL_COORDINATES = (
  (0.0, 0.0),
  (0.5, 0.0),
  (1.0, 0.0),
  (1.0, 0.5),
  (1.0, 1.0),
  (0.5, 1.0),
  (0.0, 1.0),
  (0.0, 0.5),
)


def _unit_cell_model() -> ModelSpec:
  """One Q8 element on the unit square, nodes in the v3 serendipity order."""
  return _tl_model(_UNIT_CELL_NODES, _UNIT_CELL_COORDINATES)


def _patch_model() -> tuple[ModelSpec, np.ndarray, tuple[int, ...]]:
  """A 2x2 Q8 patch on the unit square plus its interior node ids."""
  coordinates = tuple((0.25 * a, 0.25 * b) for b in range(5) for a in range(5))

  def node(a: int, b: int) -> int:
    return b * 5 + a

  cell_nodes = []
  for p in range(2):
    for q in range(2):
      cell_nodes.append(
        (
          node(2 * p, 2 * q),
          node(2 * p + 1, 2 * q),
          node(2 * p + 2, 2 * q),
          node(2 * p + 2, 2 * q + 1),
          node(2 * p + 2, 2 * q + 2),
          node(2 * p + 1, 2 * q + 2),
          node(2 * p, 2 * q + 2),
          node(2 * p, 2 * q + 1),
        )
      )
  # Interior nodes: off the patch boundary. Under a uniform stress state the
  # assembled internal force vanishes at every one of them (their global shape
  # functions vanish on the patch boundary), whatever their element count.
  interior = tuple(node(a, b) for b in range(1, 4) for a in range(1, 4))
  model = _tl_model(tuple(cell_nodes), coordinates)
  return model, np.array(coordinates, dtype=np.float64), interior


def _compile(model: ModelSpec) -> CompiledSystem:
  return compile_system(model, finite_strain_reference_registry())


def _operator(system: CompiledSystem) -> Q8FiniteStrainOperator:
  (operator,) = system.operators
  assert isinstance(operator, Q8FiniteStrainOperator)
  return operator


def _evaluate(
  operator: CompiledOperator,
  states: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  """Evaluate ``(tangent, internal force)`` on element-state batches."""
  evaluation = operator.evaluate(
    OperatorEvaluationInput(
      port_values=(FinalizedArray(states, dtype=np.float64),),
      accepted_state=FinalizedArray(
        np.zeros(operator.header.state_layout.row_shape), dtype=np.float64
      ),
      signals=(),
      request=ChannelRequest(("internal-force",), ("material-tangent",)),
    )
  )
  return (
    evaluation.jacobian_values[0].values,
    evaluation.residual_values[0].values,
  )


def _hand_tl_response(
  cell_coordinates: np.ndarray,
  states: np.ndarray,
  constitutive: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  """Independent hand-derived TL element tangent and internal force.

  Re-derives the total-Lagrangian chain — jacobian, physical gradients,
  deformation gradient, Green-Lagrange strain, PK2 stress, B, B_NL, and the
  geometric stress block — point by point from the pinned shape/quadrature
  recipes. No symbol from ``pyfem.v3.fem.tl_kinematics``/``tl_element`` (or
  the compiler's own reference assembly) is consumed here.
  """
  points, weights = gauss_tensor_product_2d(3)
  _, parent_gradients = serendipity_quad8(points)
  tangent = np.zeros((len(states), 16, 16), dtype=np.float64)
  force = np.zeros((len(states), 16), dtype=np.float64)
  for element in range(len(states)):
    coords = cell_coordinates[element]
    displacement = states[element].reshape(8, 2)
    for point in range(9):
      jacobian = coords.T @ parent_gradients[point]
      determinant = jacobian[0, 0] * jacobian[1, 1] - jacobian[0, 1] * jacobian[1, 0]
      inverse = (
        np.array(((jacobian[1, 1], -jacobian[0, 1]), (-jacobian[1, 0], jacobian[0, 0])))
        / determinant
      )
      gradient = parent_gradients[point] @ inverse
      weight = weights[point] * determinant
      deformation = np.eye(2) + displacement.T @ gradient
      right_cauchy_green = deformation.T @ deformation
      strain = np.array(
        [
          0.5 * (right_cauchy_green[0, 0] - 1.0),
          0.5 * (right_cauchy_green[1, 1] - 1.0),
          right_cauchy_green[0, 1],
        ],
        dtype=np.float64,
      )
      stress = constitutive @ strain
      b_matrix = np.zeros((3, 16), dtype=np.float64)
      b_matrix[0, 0::2] = gradient[:, 0] * deformation[0, 0]
      b_matrix[0, 1::2] = gradient[:, 0] * deformation[1, 0]
      b_matrix[1, 0::2] = gradient[:, 1] * deformation[0, 1]
      b_matrix[1, 1::2] = gradient[:, 1] * deformation[1, 1]
      b_matrix[2, 0::2] = (
        gradient[:, 1] * deformation[0, 0] + gradient[:, 0] * deformation[0, 1]
      )
      b_matrix[2, 1::2] = (
        gradient[:, 0] * deformation[1, 1] + gradient[:, 1] * deformation[1, 0]
      )
      b_nl = np.zeros((4, 16), dtype=np.float64)
      b_nl[0, 0::2] = gradient[:, 0]
      b_nl[1, 0::2] = gradient[:, 1]
      b_nl[2, 1::2] = gradient[:, 0]
      b_nl[3, 1::2] = gradient[:, 1]
      stress_matrix = np.array(
        [
          [stress[0], stress[2], 0.0, 0.0],
          [stress[2], stress[1], 0.0, 0.0],
          [0.0, 0.0, stress[0], stress[2]],
          [0.0, 0.0, stress[2], stress[1]],
        ],
        dtype=np.float64,
      )
      tangent[element] += weight * (
        b_matrix.T @ (constitutive @ b_matrix) + b_nl.T @ (stress_matrix @ b_nl)
      )
      force[element] += weight * (b_matrix.T @ stress)
  return tangent, force


def _affine_states(
  operator: Q8FiniteStrainOperator,
  coordinates: np.ndarray,
  gradient: np.ndarray,
) -> np.ndarray:
  """Element-state batches of the affine field ``u(X) = H X`` per cell."""
  connectivity = operator.entity_block.incidence.values
  displaced = coordinates[connectivity] @ gradient.T
  return displaced.reshape(len(connectivity), 16)


def _plane_stress_constitutive() -> np.ndarray:
  modulus = _YOUNGS_MODULUS
  ratio = _POISSON_RATIO
  normal = modulus / (1.0 - ratio * ratio)
  return np.array(
    [
      [normal, normal * ratio, 0.0],
      [normal * ratio, normal, 0.0],
      [0.0, 0.0, modulus / (2.0 * (1.0 + ratio))],
    ],
    dtype=np.float64,
  )


# --- descriptor and registry contract -----------------------------------------


def test_finite_strain_descriptors_carry_the_pinned_metadata() -> None:
  registry = finite_strain_reference_registry()
  assert set(registry) == {
    ("topology", "serendipity-quad8"),
    ("quadrature", "gauss-3x3"),
    TL_FORMULATION_KEY,
    TL_MATERIAL_KEY,
  }
  formulation = registry[TL_FORMULATION_KEY]
  assert formulation.implementation_id == "pyfem-v3-total-lagrangian-quad8-v1"
  assert formulation.metadata.to_bytes() == (
    CanonicalManifest(_TL_FORMULATION_METADATA).to_bytes()
  )
  material = registry[TL_MATERIAL_KEY]
  assert material.implementation_id == "pyfem-v3-plane-stress-saint-venant-kirchhoff-v1"
  assert material.metadata.to_bytes() == (
    CanonicalManifest(_TL_MATERIAL_METADATA).to_bytes()
  )
  # The shared Q8 topology/quadrature descriptors are byte-identical to the
  # landed small-strain convention.
  for key in (("topology", "serendipity-quad8"), ("quadrature", "gauss-3x3")):
    assert registry[key].metadata.to_bytes() == (
      CanonicalManifest(q8_descriptor_metadata(*key)).to_bytes()
    )


def test_single_cell_system_compiles_deterministically() -> None:
  first = _compile(_unit_cell_model())
  second = _compile(_unit_cell_model())
  assert first.content_fingerprint == second.content_fingerprint
  assert first.content_fingerprint.digest == _TL_ONE_CELL_DIGEST
  assert hashlib.sha256(first.provenance.manifest.to_bytes()).hexdigest() == (
    first.content_fingerprint.digest
  )


# --- structural oracles -------------------------------------------------------


def test_zero_state_recovers_the_small_strain_stiffness() -> None:
  operator = _operator(_compile(_unit_cell_model()))
  zeros = np.zeros((1, 16), dtype=np.float64)
  tangent, force = _evaluate(operator, zeros)
  assert np.array_equal(force, np.zeros((1, 16), dtype=np.float64))
  np.testing.assert_allclose(
    tangent,
    tangent.transpose(0, 2, 1),
    rtol=0.0,
    atol=1e-13 * float(np.max(np.abs(tangent))),
  )
  # At the zero state the TL tangent is the small-strain stiffness: cross-check
  # against the landed small-strain operator compiled on the same mesh.
  small_strain_model = _model_with(
    _unit_cell_model(),
    formulation="small-strain-continuum",
    material_model="plane-stress-linear-elastic",
  )
  small_strain_system = compile_system(small_strain_model, q8_reference_registry())
  (small_strain_operator,) = small_strain_system.operators
  small_strain_evaluation = small_strain_operator.evaluate(
    OperatorEvaluationInput(
      port_values=(FinalizedArray(zeros, dtype=np.float64),),
      accepted_state=FinalizedArray(
        np.zeros(small_strain_operator.header.state_layout.row_shape),
        dtype=np.float64,
      ),
      signals=(),
      request=ChannelRequest(("internal-force",), ("material-tangent",)),
    )
  )
  small_strain_tangent = small_strain_evaluation.jacobian_values[0].values
  np.testing.assert_allclose(tangent, small_strain_tangent, rtol=1e-11, atol=1e-11)


def test_patch_recovers_uniform_deformation_gradient() -> None:
  model, coordinates, interior = _patch_model()
  system = _compile(model)
  operator = _operator(system)
  states = _affine_states(operator, coordinates, _PROBE_GRADIENT)

  # F recovery at every quadrature point, from the compiled payload alone.
  physical = operator.payload.physical_node_coordinates().values
  points, _weights = gauss_tensor_product_2d(3)
  _, parent_gradients = serendipity_quad8(points)
  expected_deformation = np.eye(2) + _PROBE_GRADIENT
  for element in range(physical.shape[0]):
    displacement = states[element].reshape(8, 2)
    for point in range(9):
      jacobian = physical[element].T @ parent_gradients[point]
      gradient = parent_gradients[point] @ np.linalg.inv(jacobian)
      deformation = np.eye(2) + displacement.T @ gradient
      np.testing.assert_allclose(
        deformation, expected_deformation, rtol=0.0, atol=1e-13
      )

  # Per-element internal force and tangent against the hand-derived assembly.
  tangent, force = _evaluate(operator, states)
  reference_tangent, reference_force = _hand_tl_response(
    physical, states, _plane_stress_constitutive()
  )
  np.testing.assert_allclose(force, reference_force, rtol=1e-11, atol=1e-11)
  np.testing.assert_allclose(tangent, reference_tangent, rtol=1e-11, atol=1e-11)

  # Patch equilibrium: with a uniform stress state the assembled internal
  # force vanishes at every interior node.
  coordinate_map = compile_constraint_map(
    system,
    constraints=(),
    coordinates=(ProgramCoordinateSpec(name="load", kind="load"),),
  )
  plan = compile_driver_plan(system, coordinate_map)
  internal = assemble_internal_force(plan, (force,))
  interior_rows = np.array(
    [2 * node_id + component for node_id in interior for component in (0, 1)]
  )
  np.testing.assert_allclose(
    internal[interior_rows],
    np.zeros(interior_rows.shape),
    rtol=0.0,
    atol=1e-10,
  )
  assert float(np.max(np.abs(internal))) > 1.0  # boundary tractions are real


def test_rigid_rotation_carries_no_internal_force() -> None:
  model, coordinates, _interior = _patch_model()
  operator = _operator(_compile(model))
  angle = 0.4
  rotation = np.array(
    [
      [np.cos(angle) - 1.0, -np.sin(angle)],
      [np.sin(angle), np.cos(angle) - 1.0],
    ],
    dtype=np.float64,
  )
  states = _affine_states(operator, coordinates, rotation)
  _tangent, force = _evaluate(operator, states)
  assert float(np.max(np.abs(force))) < 1e-10

  # The contrast pins the strain measure: the same rotation carries a large
  # infinitesimal strain, so the small-strain operator answers with an O(1)
  # spurious internal force.
  (small_strain_operator,) = compile_system(
    _model_with(
      model,
      formulation="small-strain-continuum",
      material_model="plane-stress-linear-elastic",
    ),
    q8_reference_registry(),
  ).operators
  _small_strain_tangent, small_strain_force = _evaluate(small_strain_operator, states)
  assert float(np.max(np.abs(small_strain_force))) > 0.5


def test_tangent_matches_finite_differences_of_the_residual() -> None:
  operator = _operator(_compile(_unit_cell_model()))
  rng = np.random.default_rng(20261006)
  base = 0.05 * rng.normal(size=(1, 16))
  tangent, _force = _evaluate(operator, base)
  step = 1.0e-6
  fd_tangent = np.empty((16, 16), dtype=np.float64)
  for dof in range(16):
    delta = np.zeros((1, 16), dtype=np.float64)
    delta[0, dof] = step
    _plus_tangent, plus = _evaluate(operator, base + delta)
    _minus_tangent, minus = _evaluate(operator, base - delta)
    fd_tangent[dof] = (plus[0] - minus[0]) / (2.0 * step)
  np.testing.assert_allclose(
    tangent[0],
    fd_tangent.T,
    rtol=1e-6,
    atol=1e-8 * float(np.max(np.abs(tangent))),
  )


# --- coded compiler diagnostics ------------------------------------------------


@pytest.mark.parametrize(
  ("options", "code"),
  (
    pytest.param(
      {"material_model": "plane-stress-linear-elastic"},
      "incompatible-material-model",
      id="small-strain-law-on-finite-strain-region",
    ),
    pytest.param(
      {"material_model": "saint-venant-kirchhoff-3d"},
      "incompatible-material-model",
      id="unknown-law",
    ),
    pytest.param(
      {"quadrature": "gauss-2x2"},
      "incompatible-quadrature",
      id="wrong-quadrature",
    ),
  ),
)
def test_compiler_rejections(options: dict[str, str], code: str) -> None:
  with pytest.raises(ModelCompilationError) as excinfo:
    compile_system(
      _model_with(_unit_cell_model(), **options),
      finite_strain_reference_registry(),
    )
  assert {diagnostic.code for diagnostic in excinfo.value.diagnostics} == {code}


def test_finite_strain_region_rejects_a_quad4_block() -> None:
  model = _tl_model(((0, 1, 2, 3),), _UNIT_CELL_COORDINATES[:4])
  quad4_model = ModelSpec(
    mesh=MeshSpec(
      nodes=model.mesh.nodes,
      cell_blocks=(
        CellBlockSpec(
          id="cells",
          reference_topology="quadrilateral",
          topological_dimension=2,
          embedding_dimension=2,
          geometry_interpolation="bilinear-quad4",
          cells=model.mesh.cell_blocks[0].cells,
          source=_SOURCE,
        ),
      ),
      source=_SOURCE,
    ),
    fields=model.fields,
    materials=model.materials,
    regions=model.regions,
    source=_SOURCE,
  )
  with pytest.raises(ModelCompilationError) as excinfo:
    compile_system(quad4_model, finite_strain_reference_registry())
  assert {diagnostic.code for diagnostic in excinfo.value.diagnostics} == {
    "incompatible-cell-block"
  }


def test_finite_strain_region_requires_its_registry_keys() -> None:
  with pytest.raises(ModelCompilationError) as excinfo:
    compile_system(_unit_cell_model(), q8_reference_registry())
  assert {diagnostic.code for diagnostic in excinfo.value.diagnostics} == {
    "registry-capture-failed"
  }


# --- evaluation contract -------------------------------------------------------


def test_evaluation_rejects_malformed_inputs() -> None:
  operator = _operator(_compile(_unit_cell_model()))
  zeros = np.zeros((1, 16), dtype=np.float64)
  state = FinalizedArray(
    np.zeros(operator.header.state_layout.row_shape), dtype=np.float64
  )
  request = ChannelRequest(("internal-force",), ("material-tangent",))

  def _input(**overrides: object) -> OperatorEvaluationInput:
    values: dict[str, object] = {
      "port_values": (FinalizedArray(zeros, dtype=np.float64),),
      "accepted_state": state,
      "signals": (),
      "request": request,
    }
    values.update(overrides)
    return OperatorEvaluationInput(**values)  # type: ignore[arg-type]

  with pytest.raises(ValueError, match="unavailable"):
    operator.evaluate(
      _input(request=ChannelRequest(("internal-force",), ("foreign-channel",)))
    )
  with pytest.raises(TypeError):
    operator.evaluate(_input(port_values=()))
  with pytest.raises(TypeError):
    operator.evaluate(_input(port_values=(FinalizedArray(np.full((1, 16), np.inf)),)))
  with pytest.raises(TypeError):
    operator.evaluate(
      _input(accepted_state=FinalizedArray(np.zeros((1, 1)), dtype=np.float64))
    )


# --- cantilever8 end-to-end parity ---------------------------------------------


def _legacy_to_compiled_permutation(
  deck_model: ModelSpec,
  system: CompiledSystem,
) -> np.ndarray:
  """Row index mapping: compiled coefficient row -> legacy state row."""
  position = {node.id: index for index, node in enumerate(deck_model.mesh.nodes)}
  components = deck_model.fields[0].components
  component_index = {component: index for index, component in enumerate(components)}
  ndof = len(components)
  space = system.spaces[0]
  return np.array(
    [
      position[node_id] * ndof + component_index[component]
      for _field_id, node_id, component in space.coefficient_ids
    ],
    dtype=np.int64,
  )


def test_cantilever8_deck_drives_to_the_legacy_state() -> None:
  deck = read_legacy_deck(CANTILEVER8_PRO)
  run = run_deck(deck)
  assert run.result.status is DriverStatus.COMPLETED
  statistics = run.result.statistics
  assert statistics.committed_substep_count == 20
  assert statistics.rejected_substep_count == 0
  assert statistics.cutback_count == 0
  # The finite-strain tangent is state-dependent: no factorization reuse.
  assert statistics.factorization_reuse_count == 0
  assert statistics.factorization_count > statistics.committed_substep_count
  legacy = legacy_nonlinear_state(CANTILEVER8_PRO)
  permutation = _legacy_to_compiled_permutation(deck.model, compile_deck(deck).system)
  rtol, atol = load_parity_tolerances(SKIMS / "cantilever8")
  np.testing.assert_allclose(run.state, legacy[permutation], rtol=rtol, atol=atol)
  # The converged reactions equilibrate the applied tip load (0.01 ramped to
  # the load factor 20): the supports carry -0.2 in y and nothing in x, up to
  # the residual level the deck's 1e-3 Newton tolerance leaves behind (the
  # observed sums are off by ~8e-8; the band carries ~10x headroom).
  observation = run.result.records[-1].observation
  assert observation is not None
  reactions = observation.reactions.values
  assert float(reactions[0::2].sum()) == pytest.approx(0.0, abs=1.0e-6)
  assert float(reactions[1::2].sum()) == pytest.approx(-0.2, abs=1.0e-6)


def test_cantilever8_tangent_and_force_match_legacy_at_the_converged_state() -> None:
  deck = read_legacy_deck(CANTILEVER8_PRO)
  compiled = compile_deck(deck)
  run = run_deck(deck)
  state = run.state
  operator = _operator(compiled.system)
  gather = operator.header.ports[0].coefficient_map.values
  tangent, force = _evaluate(operator, state[gather])
  plan = compile_driver_plan(compiled.system, compiled.constraint_map, compiled.loads)
  internal = assemble_internal_force(plan, (force,))
  stiffness = refill_tangent(plan, (tangent,)).toarray()

  permutation = _legacy_to_compiled_permutation(deck.model, compiled.system)
  legacy_k, legacy_fint = legacy_tangent_at_state(CANTILEVER8_PRO, state[permutation])
  legacy_k_dense = legacy_k.toarray()
  rtol, atol = load_parity_tolerances(SKIMS / "cantilever8")
  np.testing.assert_allclose(
    internal,
    legacy_fint[permutation],
    rtol=rtol,
    atol=atol,
  )
  np.testing.assert_allclose(
    stiffness,
    legacy_k_dense[np.ix_(permutation, permutation)],
    rtol=rtol,
    atol=atol,
  )
