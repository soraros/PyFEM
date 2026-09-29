# SPDX-License-Identifier: MIT

"""``pack_problem`` constraint/MPC packing semantics (incl. inverse DOF map)."""

from __future__ import annotations

import numpy as np
import pytest

from pyfem.v3.mesh import build_dof_map
from pyfem.v3.mesh.refined_patch import build_uniform_q8_patch
from pyfem.v3.pack import _dof_node_type, _inverse_dof_map, pack_problem
from pyfem.v3.types import (
  GROUP_CONTINUUM,
  GROUP_TRUSS,
  DofMap,
  ElementGroupSpec,
  Mesh,
  MpcTie,
  NodalLoad,
  PrescribedDof,
)


def _patch(nx: int = 1, ny: int = 1) -> tuple[Mesh, DofMap]:
  mesh, _constraints = build_uniform_q8_patch(nx, ny)
  return mesh, build_dof_map(mesh)


def test_prescribed_dofs_packed_in_input_order() -> None:
  mesh, dof_map = _patch()
  constraints = (
    PrescribedDof(node_id=5, dof_type="v", value=-1.5),
    PrescribedDof(node_id=0, dof_type="u", value=2.0),
    PrescribedDof(node_id=3, dof_type="u", value=0.25),
  )
  loads = (NodalLoad(node_id=7, dof_type="v", value=-250.0),)
  problem = pack_problem(mesh, dof_map, 1.0e6, 0.25, constraints, (), loads)
  expected_dofs = [
    dof_map.dof_index(5, "v"),
    dof_map.dof_index(0, "u"),
    dof_map.dof_index(3, "u"),
  ]
  np.testing.assert_array_equal(problem.constraint_dof, expected_dofs)
  np.testing.assert_array_equal(problem.constraint_val, [-1.5, 2.0, 0.25])
  external = np.zeros(dof_map.n_dofs)
  external[dof_map.dof_index(7, "v")] = -250.0
  np.testing.assert_array_equal(problem.external_load, external)


def test_duplicate_prescription_keeps_first_slot_last_value() -> None:
  mesh, dof_map = _patch()
  constraints = (
    PrescribedDof(node_id=0, dof_type="u", value=1.0),
    PrescribedDof(node_id=3, dof_type="v", value=-2.0),
    PrescribedDof(node_id=0, dof_type="u", value=9.0),
  )
  problem = pack_problem(mesh, dof_map, 1.0e6, 0.25, constraints)
  np.testing.assert_array_equal(
    problem.constraint_dof,
    [dof_map.dof_index(0, "u"), dof_map.dof_index(3, "v")],
  )
  np.testing.assert_array_equal(problem.constraint_val, [9.0, -2.0])


def test_mpc_tie_promoted_when_master_prescribed() -> None:
  mesh, dof_map = _patch()
  constraints = (PrescribedDof(node_id=0, dof_type="u", value=3.0),)
  ties = (
    MpcTie(
      slave_node_id=4,
      slave_dof_type="u",
      offset=0.5,
      master_node_id=0,
      master_dof_type="u",
      factor=2.0,
    ),
  )
  problem = pack_problem(mesh, dof_map, 1.0e6, 0.25, constraints, ties)
  assert problem.mpc_slave_dof.size == 0
  np.testing.assert_array_equal(
    problem.constraint_dof,
    [dof_map.dof_index(0, "u"), dof_map.dof_index(4, "u")],
  )
  np.testing.assert_array_equal(problem.constraint_val, [3.0, 0.5 + 2.0 * 3.0])


def test_mpc_chain_flattened_against_prescribed_root() -> None:
  mesh, dof_map = _patch()
  root = PrescribedDof(node_id=0, dof_type="u", value=2.0)
  ties = (
    MpcTie(
      slave_node_id=4,
      slave_dof_type="u",
      offset=1.0,
      master_node_id=5,
      master_dof_type="u",
      factor=2.0,
    ),
    MpcTie(
      slave_node_id=5,
      slave_dof_type="u",
      offset=0.5,
      master_node_id=0,
      master_dof_type="u",
      factor=3.0,
    ),
  )
  problem = pack_problem(mesh, dof_map, 1.0e6, 0.25, (root,), ties)
  assert problem.mpc_slave_dof.size == 0
  values = dict(zip(problem.constraint_dof.tolist(), problem.constraint_val.tolist()))
  assert values == {
    dof_map.dof_index(0, "u"): 2.0,
    dof_map.dof_index(5, "u"): 0.5 + 3.0 * 2.0,
    dof_map.dof_index(4, "u"): (1.0 + 2.0 * 0.5) + 2.0 * 3.0 * 2.0,
  }
  assert problem.constraint_dof[0] == dof_map.dof_index(0, "u")


def test_unresolved_tie_stays_in_mpc_arrays() -> None:
  mesh, dof_map = _patch()
  constraints = (PrescribedDof(node_id=0, dof_type="u", value=1.0),)
  ties = (
    MpcTie(
      slave_node_id=4,
      slave_dof_type="u",
      offset=0.25,
      master_node_id=7,
      master_dof_type="u",
      factor=1.5,
    ),
  )
  problem = pack_problem(mesh, dof_map, 1.0e6, 0.25, constraints, ties)
  np.testing.assert_array_equal(problem.constraint_dof, [dof_map.dof_index(0, "u")])
  np.testing.assert_array_equal(problem.mpc_slave_dof, [dof_map.dof_index(4, "u")])
  np.testing.assert_array_equal(problem.mpc_master_dof, [dof_map.dof_index(7, "u")])
  np.testing.assert_array_equal(problem.mpc_factor, [1.5])
  np.testing.assert_array_equal(problem.mpc_offset, [0.25])


def test_empty_constraints_and_ties_give_empty_arrays() -> None:
  mesh, dof_map = _patch()
  problem = pack_problem(mesh, dof_map, 1.0e6, 0.25)
  assert problem.constraint_dof.dtype == np.int32
  assert problem.constraint_val.dtype == np.float64
  assert problem.constraint_dof.size == 0
  assert problem.constraint_val.size == 0
  assert problem.mpc_slave_dof.size == 0
  assert problem.mpc_master_dof.size == 0
  assert problem.mpc_factor.size == 0
  assert problem.mpc_offset.size == 0
  np.testing.assert_array_equal(problem.external_load, np.zeros(dof_map.n_dofs))


def test_inverse_dof_map_first_occurrence_wins() -> None:
  _mesh, dof_map = _patch()
  dup_table = dof_map.global_dofs.copy()
  dup_table[3, 0] = dup_table[0, 0]
  dup_map = type(dof_map)(
    global_dofs=dup_table,
    dof_types=dof_map.dof_types,
    node_ids=dof_map.node_ids,
    node_id_to_index=dof_map.node_id_to_index,
  )
  inverse = _inverse_dof_map(dup_map)
  first_dof = int(dup_table[0, 0])
  assert inverse[first_dof] == (int(dof_map.node_ids[0]), dof_map.dof_types[0])
  normal_dof = int(dup_table[5, 1])
  assert inverse[normal_dof] == (int(dof_map.node_ids[5]), dof_map.dof_types[1])


def test_dof_node_type_unknown_raises() -> None:
  _mesh, dof_map = _patch()
  inverse = _inverse_dof_map(dof_map)
  with pytest.raises(ValueError, match="Unknown DOF index 999"):
    _dof_node_type(inverse, 999)


def test_non_identity_node_ids_resolved_to_correct_dofs() -> None:
  base, _ = _patch()
  node_ids = np.array([100, 105, 101, 107, 102, 106, 103, 104], dtype=np.int32)
  mesh = Mesh(
    coords=base.coords,
    conn=base.conn,
    node_ids=node_ids,
    elem_group_id=base.elem_group_id,
    node_id_to_index={int(nid): i for i, nid in enumerate(node_ids)},
  )
  dof_map = build_dof_map(mesh)
  constraints = (PrescribedDof(node_id=105, dof_type="u", value=4.0),)
  ties = (
    MpcTie(
      slave_node_id=107,
      slave_dof_type="v",
      offset=1.0,
      master_node_id=105,
      master_dof_type="u",
      factor=0.5,
    ),
    MpcTie(
      slave_node_id=100,
      slave_dof_type="u",
      offset=0.0,
      master_node_id=102,
      master_dof_type="u",
      factor=1.0,
    ),
  )
  problem = pack_problem(mesh, dof_map, 1.0e6, 0.25, constraints, ties)
  np.testing.assert_array_equal(
    problem.constraint_dof,
    [dof_map.dof_index(105, "u"), dof_map.dof_index(107, "v")],
  )
  np.testing.assert_array_equal(problem.constraint_val, [4.0, 1.0 + 0.5 * 4.0])
  np.testing.assert_array_equal(problem.mpc_slave_dof, [dof_map.dof_index(100, "u")])
  np.testing.assert_array_equal(problem.mpc_master_dof, [dof_map.dof_index(102, "u")])


def test_element_groups_populate_group_arrays() -> None:
  mesh, dof_map = _patch()
  groups = (
    ElementGroupSpec(
      name="ContElem",
      element_type="SmallStrainContinuum",
      props=(1.0e6, 0.25),
    ),
    ElementGroupSpec(name="TrussElem", element_type="Truss", props=(2.1e11, 1.0e-4)),
  )
  problem = pack_problem(mesh, dof_map, 0.0, 0.0, groups=groups)
  np.testing.assert_array_equal(problem.group_kind, [GROUP_CONTINUUM, GROUP_TRUSS])
  np.testing.assert_array_equal(problem.group_props, [[0.0, 0.0], [2.1e11, 1.0e-4]])
  np.testing.assert_array_equal(problem.constitutive, np.zeros((3, 3)))


def test_boundary_node_group_64x64_packs_all_prescribed() -> None:
  """Scale gate: full boundary prescription set packed exactly (was O(n^2) scan)."""
  mesh, constraints = build_uniform_q8_patch(64, 64)
  dof_map = build_dof_map(mesh)
  assert dof_map.n_dofs == 25090
  problem = pack_problem(mesh, dof_map, 1.0e6, 0.25, constraints)
  expected_dofs = [dof_map.dof_index(c.node_id, c.dof_type) for c in constraints]
  np.testing.assert_array_equal(problem.constraint_dof, expected_dofs)
  np.testing.assert_array_equal(
    problem.constraint_val,
    [c.value for c in constraints],
  )
