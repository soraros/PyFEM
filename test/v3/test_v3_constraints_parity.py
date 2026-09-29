# SPDX-License-Identifier: MIT

"""Parity oracles: compiled constraint maps vs the legacy ``Constrainer``.

The patch_test8_mpc skim exercises ties that chain into prescribed masters
(legacy flattens them to prescribed values). The RVE-style periodic case
exercises genuine slave rows in the prolongation (free masters). Legacy
cannot represent tie chains ending at free masters at all (its flatten walk
crashes), so chain-to-free composition is covered by unit tests instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from _legacy_parity import (
  legacy_external_load,
  legacy_state,
  legacy_stiffness_coo,
  legacy_tangent_at_state,
  load_parity_tolerances,
)
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from pyfem.fem.Constrainer import Constrainer
from pyfem.v3.compile.continuum import q8_reference_registry
from pyfem.v3.compile.system import compile_system
from pyfem.v3.constraints import (
  CompiledConstraintMap,
  compile_constraint_map,
  evaluate_offsets,
  full_coefficients,
  periodic_ties,
  reaction_forces,
  reduce_residual,
  reduce_tangent,
)
from pyfem.v3.io.dat import read_dat_mesh
from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.operator import (
  ChannelRequest,
  OperatorEvaluationInput,
)
from pyfem.v3.model.system import CompiledSystem
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
from pyfem.v3.spec.program import (
  AffineCoefficientSpec,
  AffineTieSpec,
  AffineValueSpec,
  DofRef,
  PrescribedDofSpec,
  ProgramCoordinateSpec,
  ProgramCoordinateValue,
  ProgramPoint,
)
from pyfem.v3.types import MpcTie, PrescribedDof

ROOT = Path(__file__).resolve().parents[2]
SKIM_DIR = ROOT / "skims" / "patch_test8_mpc"
MPC_DAT = SKIM_DIR / "PatchTest8_mpc.dat"
SKIM_PRO = SKIM_DIR / "skim.pro"
_COMPONENT = {"u": "x", "v": "y"}


def _source(label: str) -> SourceContext:
  return SourceContext(source=label)


def _constraint_specs(
  prescribed: tuple[PrescribedDof, ...],
  ties: tuple[MpcTie, ...],
) -> tuple[PrescribedDofSpec | AffineTieSpec, ...]:
  specs: list[PrescribedDofSpec | AffineTieSpec] = [
    PrescribedDofSpec(
      id=f"prescribed:{item.node_id}:{item.dof_type}",
      target=DofRef(
        node_id=item.node_id,
        field_id="displacement",
        component=_COMPONENT[item.dof_type],
      ),
      value=AffineValueSpec(constant=item.value),
      source=_source(f"dat:prescribed:{item.node_id}:{item.dof_type}"),
    )
    for item in prescribed
  ]
  specs.extend(
    AffineTieSpec(
      id=f"tie:{item.slave_node_id}:{item.slave_dof_type}",
      slave=DofRef(
        node_id=item.slave_node_id,
        field_id="displacement",
        component=_COMPONENT[item.slave_dof_type],
      ),
      master=DofRef(
        node_id=item.master_node_id,
        field_id="displacement",
        component=_COMPONENT[item.master_dof_type],
      ),
      factor=item.factor,
      offset=AffineValueSpec(constant=item.offset),
      source=_source(f"dat:tie:{item.slave_node_id}:{item.slave_dof_type}"),
    )
    for item in ties
  )
  return tuple(specs)


def _skim_model() -> ModelSpec:
  mesh, _constraints, _ties, _loads = read_dat_mesh(MPC_DAT)
  nodes = tuple(
    NodeSpec(
      id=int(node_id),
      coordinates=tuple(float(value) for value in mesh.coords[index]),
      source=_source(f"dat:node:{int(node_id)}"),
    )
    for index, node_id in enumerate(mesh.node_ids)
  )
  cells = tuple(
    CellSpec(
      id=f"cell-{index}",
      node_ids=tuple(int(node_id) for node_id in mesh.conn[index]),
      source=_source(f"dat:cell:{index}"),
    )
    for index in range(mesh.conn.shape[0])
  )
  block = CellBlockSpec(
    id="cells",
    reference_topology="quadrilateral",
    topological_dimension=2,
    embedding_dimension=2,
    geometry_interpolation="serendipity-quad8",
    cells=cells,
    source=_source("dat:cells"),
  )
  field = FieldSpec(
    id="displacement",
    components=("x", "y"),
    location="node",
    source=_source("dat:field"),
  )
  material = MaterialSpec(
    id="elastic",
    model="plane-stress-linear-elastic",
    parameters=(
      MaterialParameterSpec("youngs_modulus", 1.0e6, _source("dat:E")),
      MaterialParameterSpec("poisson_ratio", 0.25, _source("dat:nu")),
    ),
    source=_source("dat:material"),
  )
  region = RegionSpec(
    id="domain",
    cell_refs=tuple(CellRef(block.id, cell.id) for cell in cells),
    field_ids=(field.id,),
    material_id=material.id,
    formulation="small-strain-continuum",
    quadrature="gauss-3x3",
    source=_source("dat:region"),
  )
  return ModelSpec(
    mesh=MeshSpec(nodes=nodes, cell_blocks=(block,), source=_source("dat:mesh")),
    fields=(field,),
    materials=(material,),
    regions=(region,),
    source=_source("dat:model"),
  )


def _skim_constraint_map() -> tuple[CompiledSystem, CompiledConstraintMap, Constrainer]:
  _mesh, prescribed, ties, _loads = read_dat_mesh(MPC_DAT)
  system = compile_system(_skim_model(), q8_reference_registry())
  constraint_map = compile_constraint_map(
    system,
    constraints=_constraint_specs(prescribed, ties),
  )
  legacy = _legacy_constrainer(
    40,
    tuple(
      (_legacy_dof(item.node_id, item.dof_type), item.value) for item in prescribed
    ),
    tuple(
      (
        _legacy_dof(item.slave_node_id, item.slave_dof_type),
        item.offset,
        _legacy_dof(item.master_node_id, item.master_dof_type),
        item.factor,
      )
      for item in ties
    ),
  )
  return system, constraint_map, legacy


def _legacy_dof(node_id: int, dof_type: str) -> int:
  return 2 * node_id + (0 if dof_type == "u" else 1)


def _legacy_constrainer(
  n_dofs: int,
  prescribed: tuple[tuple[int, float], ...],
  ties: tuple[tuple[int, float, int, float], ...],
  factor: float = 1.0,
  flatten: bool = True,
) -> Constrainer:
  """Build a legacy ``Constrainer`` the way the legacy paths do.

  With ``flatten`` the tie masters are wrapped as one-element arrays (the
  parser's convention) and ``checkConstraints`` flattens ties whose masters
  are themselves constrained — the only tie shape legacy fully supports.
  Without ``flatten`` the masters are plain scalars (the RVE model's
  convention) and ``flush`` runs directly, which legacy can only do when no
  tie master is constrained; the compiled map supersedes both limitations.
  """
  constrainer = Constrainer(n_dofs)
  label = "main"
  constrainer.constrainedDofs[label] = []
  constrainer.constrainedVals[label] = []
  constrainer.constrainedFac[label] = factor
  for dof, value in prescribed:
    constrainer.addConstraint(dof, value, label)
  for slave, offset, master, tie_factor in ties:
    master_entry = np.array([master]) if flatten else np.int64(master)
    constrainer.addConstraint(
      slave,
      [offset, master_entry, tie_factor],
      label,
    )
  if flatten:
    constrainer.checkConstraints(None, None)
  constrainer.flush()
  return constrainer


def _legacy_offset(constrainer: Constrainer, n_dofs: int) -> np.ndarray:
  offset = np.zeros(n_dofs, dtype=np.float64)
  constrainer.addConstrainedValues(offset)
  return offset


def _legacy_reduced_solve(
  constrainer: Constrainer,
  tangent: csr_matrix,
  load: np.ndarray,
) -> np.ndarray:
  offset = _legacy_offset(constrainer, tangent.shape[0])
  reduced_tangent = constrainer.C.T @ (tangent @ constrainer.C)
  reduced_rhs = constrainer.C.T @ (load - tangent @ offset)
  return (
    np.asarray(constrainer.C @ spsolve(reduced_tangent.tocsr(), reduced_rhs)) + offset
  )


def test_mpc_skim_prolongation_matches_legacy_c_matrix() -> None:
  _system, constraint_map, legacy = _skim_constraint_map()
  assert constraint_map.full_dof_count == 40
  assert constraint_map.reduced_dof_count == 25
  assert legacy.C.shape == (40, 25)
  np.testing.assert_allclose(
    constraint_map.prolongation().toarray(),
    legacy.C.toarray(),
    rtol=0.0,
    atol=1e-15,
  )


def test_mpc_skim_offsets_match_legacy_prescribed_values() -> None:
  _system, constraint_map, legacy = _skim_constraint_map()
  evaluation = evaluate_offsets(constraint_map, ProgramPoint())
  np.testing.assert_allclose(
    evaluation.offsets.values,
    _legacy_offset(legacy, 40),
    rtol=0.0,
    atol=1e-15,
  )
  assert evaluation.derivatives.values.shape == (40, 0)


def test_mpc_skim_reduced_system_matches_legacy() -> None:
  _system, constraint_map, legacy = _skim_constraint_map()
  tangent = legacy_stiffness_coo(SKIM_PRO).tocsr()
  load = legacy_external_load(SKIM_PRO)
  offset = _legacy_offset(legacy, 40)
  reduced_tangent = reduce_tangent(constraint_map, tangent)
  np.testing.assert_allclose(
    reduced_tangent.toarray(),
    (legacy.C.T @ (tangent @ legacy.C)).toarray(),
    rtol=0.0,
    atol=1e-9,
  )
  reduced_rhs = reduce_residual(constraint_map, load - tangent @ offset)
  np.testing.assert_allclose(
    reduced_rhs.values,
    np.asarray(legacy.C.T @ (load - tangent @ offset)),
    rtol=0.0,
    atol=1e-12,
  )


def test_mpc_skim_solve_matches_legacy_state() -> None:
  rtol, atol = load_parity_tolerances(SKIM_DIR)
  _system, constraint_map, legacy = _skim_constraint_map()
  tangent = legacy_stiffness_coo(SKIM_PRO).tocsr()
  load = legacy_external_load(SKIM_PRO)
  evaluation = evaluate_offsets(constraint_map, ProgramPoint())
  rhs = reduce_residual(constraint_map, load - tangent @ evaluation.offsets.values)
  q = spsolve(reduce_tangent(constraint_map, tangent), rhs.values)
  state = full_coefficients(constraint_map, q, ProgramPoint())
  legacy_reference = legacy_state(SKIM_PRO)
  np.testing.assert_allclose(state.values, legacy_reference, rtol=rtol, atol=atol)
  legacy_direct = _legacy_reduced_solve(legacy, tangent, load)
  np.testing.assert_allclose(state.values, legacy_direct, rtol=rtol, atol=atol)


def test_mpc_skim_reactions_match_legacy_full_residual() -> None:
  rtol, atol = load_parity_tolerances(SKIM_DIR)
  _system, constraint_map, _legacy = _skim_constraint_map()
  reference_state = legacy_state(SKIM_PRO)
  _tangent, internal = legacy_tangent_at_state(SKIM_PRO, reference_state)
  residual = internal - legacy_external_load(SKIM_PRO)
  reactions = reaction_forces(constraint_map, residual)
  constrained = constraint_map.constrained_dofs.values
  np.testing.assert_allclose(
    reactions.values[constrained],
    residual[constrained],
    rtol=rtol,
    atol=atol,
  )
  free = constraint_map.free_dofs.values
  assert np.all(reactions.values[free] == 0.0)
  # The legacy equilibrium is a reduced equilibrium of the compiled map.
  np.testing.assert_allclose(
    reduce_residual(constraint_map, residual).values,
    np.zeros(constraint_map.reduced_dof_count),
    rtol=0.0,
    atol=1e-6,
  )


# --- RVE-style periodic triplets ----------------------------------------------

_RVE_COORDINATES = (
  (0.0, 0.0),
  (0.5, 0.0),
  (1.0, 0.0),
  (1.0, 0.5),
  (1.0, 1.0),
  (0.5, 1.0),
  (0.0, 1.0),
  (0.0, 0.5),
)
_RVE_LEFT = (1, 8, 7)
_RVE_RIGHT = (3, 4, 5)
_RVE_EX = 1.0e-3
_RVE_NU = 0.25


def _rve_system() -> CompiledSystem:
  nodes = tuple(
    NodeSpec(
      id=index + 1,
      coordinates=point,
      source=_source(f"rve:node:{index + 1}"),
    )
    for index, point in enumerate(_RVE_COORDINATES)
  )
  cell = CellSpec(
    id="cell-1",
    node_ids=tuple(node.id for node in nodes),
    source=_source("rve:cell"),
  )
  block = CellBlockSpec(
    id="cells",
    reference_topology="quadrilateral",
    topological_dimension=2,
    embedding_dimension=2,
    geometry_interpolation="serendipity-quad8",
    cells=(cell,),
    source=_source("rve:cells"),
  )
  field = FieldSpec(
    id="displacement",
    components=("x", "y"),
    location="node",
    source=_source("rve:field"),
  )
  material = MaterialSpec(
    id="elastic",
    model="plane-stress-linear-elastic",
    parameters=(
      MaterialParameterSpec("youngs_modulus", 1.0e6, _source("rve:E")),
      MaterialParameterSpec("poisson_ratio", _RVE_NU, _source("rve:nu")),
    ),
    source=_source("rve:material"),
  )
  region = RegionSpec(
    id="domain",
    cell_refs=(CellRef(block.id, cell.id),),
    field_ids=(field.id,),
    material_id=material.id,
    formulation="small-strain-continuum",
    quadrature="gauss-3x3",
    source=_source("rve:region"),
  )
  model = ModelSpec(
    mesh=MeshSpec(nodes=nodes, cell_blocks=(block,), source=_source("rve:mesh")),
    fields=(field,),
    materials=(material,),
    regions=(region,),
    source=_source("rve:model"),
  )
  return compile_system(model, q8_reference_registry())


def _rve_constraints() -> tuple[PrescribedDofSpec | AffineTieSpec, ...]:
  periodic = periodic_ties(
    primary_node_ids=_RVE_LEFT,
    image_node_ids=_RVE_RIGHT,
    field_id="displacement",
    components=("x", "y"),
    offsets=(
      AffineValueSpec(
        coefficients=(AffineCoefficientSpec(coordinate="ex", coefficient=1.0),)
      ),
      AffineValueSpec(),
    ),
    id_prefix="rve-x",
    source=_source("rve:periodic"),
  )
  fixed = (
    PrescribedDofSpec(
      id="rve:fix-x",
      target=DofRef(node_id=1, field_id="displacement", component="x"),
      value=AffineValueSpec(),
      source=_source("rve:fix-x"),
    ),
    PrescribedDofSpec(
      id="rve:fix-y",
      target=DofRef(node_id=1, field_id="displacement", component="y"),
      value=AffineValueSpec(),
      source=_source("rve:fix-y"),
    ),
  )
  return (*periodic, *fixed)


def _rve_map() -> tuple[CompiledSystem, CompiledConstraintMap]:
  system = _rve_system()
  constraint_map = compile_constraint_map(
    system,
    constraints=_rve_constraints(),
    coordinates=(
      ProgramCoordinateSpec(name="ex", kind="continuation", source=_source("rve:ex")),
    ),
  )
  return system, constraint_map


def _rve_point(strain: float) -> ProgramPoint:
  return ProgramPoint(values=(ProgramCoordinateValue(name="ex", value=strain),))


def _rve_legacy_constrainer(x_offset: float) -> Constrainer:
  """Legacy oracle for the RVE tie set.

  Legacy cannot flatten node 3's ties into the prescribed corner and keep the
  surviving free-master ties in one constrainer (the R1 F2 rebuild friction),
  so the oracle expresses that promotion explicitly: node 3 is prescribed at
  the resolved values — exactly what the compiled map computes by chain
  resolution.
  """
  prescribed = ((0, 0.0), (1, 0.0), (4, x_offset), (5, 0.0))
  ties = (
    (6, x_offset, 14, 1.0),
    (7, 0.0, 15, 1.0),
    (8, x_offset, 12, 1.0),
    (9, 0.0, 13, 1.0),
  )
  return _legacy_constrainer(16, prescribed, ties, flatten=False)


def _rve_tangent(system: CompiledSystem) -> csr_matrix:
  operator = system.operators[0]
  result = operator.evaluate(
    OperatorEvaluationInput(
      port_values=(FinalizedArray(np.zeros((1, 16)), dtype=np.float64),),
      accepted_state=FinalizedArray(np.empty((1, 0)), dtype=np.float64),
      signals=(),
      request=ChannelRequest(
        (),
        tuple(item.channel_id for item in operator.header.jacobian_channels),
      ),
    )
  )
  return csr_matrix(result.jacobian_values[0].values[0])


def test_rve_periodic_prolongation_matches_legacy_c_matrix() -> None:
  _system, constraint_map = _rve_map()
  legacy = _rve_legacy_constrainer(0.0)
  assert constraint_map.reduced_dof_count == 8
  assert legacy.C.shape == (16, 8)
  np.testing.assert_allclose(
    constraint_map.prolongation().toarray(),
    legacy.C.toarray(),
    rtol=0.0,
    atol=1e-15,
  )


def test_rve_periodic_signal_offsets_match_legacy_scaled_values() -> None:
  _system, constraint_map = _rve_map()
  evaluation = evaluate_offsets(constraint_map, _rve_point(_RVE_EX))
  legacy = _rve_legacy_constrainer(_RVE_EX)
  np.testing.assert_allclose(
    evaluation.offsets.values,
    _legacy_offset(legacy, 16),
    rtol=0.0,
    atol=1e-15,
  )
  derivatives = evaluation.derivatives.values
  image_dofs = [2 * (image - 1) for image in _RVE_RIGHT]
  np.testing.assert_allclose(
    derivatives[image_dofs, 0],
    np.ones(3),
    rtol=0.0,
    atol=1e-15,
  )
  masked = derivatives.copy()
  masked[image_dofs, 0] = 0.0
  assert np.all(masked == 0.0)


def test_rve_periodic_reduced_solve_matches_legacy_and_analytic_field() -> None:
  rtol, atol = load_parity_tolerances(SKIM_DIR)
  system, constraint_map = _rve_map()
  legacy = _rve_legacy_constrainer(_RVE_EX)
  tangent = _rve_tangent(system)
  evaluation = evaluate_offsets(constraint_map, _rve_point(_RVE_EX))
  offset = evaluation.offsets.values
  rhs = reduce_residual(constraint_map, -(tangent @ offset))
  q = spsolve(reduce_tangent(constraint_map, tangent), rhs.values)
  state = full_coefficients(constraint_map, q, _rve_point(_RVE_EX))
  legacy_state_rve = _legacy_reduced_solve(legacy, tangent, np.zeros(16))
  np.testing.assert_allclose(state.values, legacy_state_rve, rtol=rtol, atol=atol)
  coordinates = np.asarray(_RVE_COORDINATES, dtype=np.float64)
  analytic = np.empty(16, dtype=np.float64)
  analytic[0::2] = _RVE_EX * coordinates[:, 0]
  analytic[1::2] = -_RVE_NU * _RVE_EX * coordinates[:, 1]
  np.testing.assert_allclose(state.values, analytic, rtol=rtol, atol=atol)
  residual = tangent @ state.values
  np.testing.assert_allclose(
    reduce_residual(constraint_map, residual).values,
    np.zeros(constraint_map.reduced_dof_count),
    rtol=0.0,
    atol=1e-6,
  )
  reactions = reaction_forces(constraint_map, residual)
  assert np.all(reactions.values[constraint_map.free_dofs.values] == 0.0)
  # Tie reactions redistribute onto the free masters through P.T r = 0.
  master_dofs = [2 * (primary - 1) for primary in (8, 7)]
  slave_dofs = [2 * (image - 1) for image in (4, 5)]
  np.testing.assert_allclose(
    residual[master_dofs],
    -residual[slave_dofs],
    rtol=1e-6,
    atol=1e-6,
  )
