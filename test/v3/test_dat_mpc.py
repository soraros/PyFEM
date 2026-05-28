# SPDX-License-Identifier: MIT

"""`.dat` reader: MPC ties from ``<NodeConstraints>``."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from pyfem.v3 import load_problem
from pyfem.v3.io.dat import _parse_mpc_rhs, read_dat_mesh
from pyfem.v3.mesh import build_dof_map
from pyfem.v3.pack import pack_problem

ROOT = Path(__file__).resolve().parents[2]
MPC_DAT = ROOT / "skims" / "patch_test8_mpc" / "PatchTest8_mpc.dat"
PATCH8_DAT = ROOT / "examples" / "ch02" / "PatchTest8.dat"


def test_parse_mpc_rhs_factor_form() -> None:
  offset, factor, dof_type, node_id = _parse_mpc_rhs("2.0 * u[1]")
  assert offset == pytest.approx(0.0)
  assert factor == pytest.approx(2.0)
  assert dof_type == "u"
  assert node_id == 1


def test_parse_mpc_rhs_identity_form() -> None:
  offset, factor, dof_type, node_id = _parse_mpc_rhs("u[1]")
  assert offset == pytest.approx(0.0)
  assert factor == pytest.approx(1.0)
  assert dof_type == "u"
  assert node_id == 1


def test_patch_test8_dat_has_no_mpc_ties() -> None:
  _mesh, _constraints, ties, _loads = read_dat_mesh(PATCH8_DAT)
  assert ties == ()


def test_mpc_dat_parses_ties() -> None:
  _mesh, constraints, ties, _loads = read_dat_mesh(MPC_DAT)
  assert len(constraints) == 12
  assert len(ties) == 3
  assert ties[0].slave_node_id == 2
  assert ties[0].master_node_id == 1
  assert ties[0].factor == pytest.approx(2.0)


def test_pack_resolves_ties_to_prescribed() -> None:
  mesh, constraints, ties, _loads = read_dat_mesh(MPC_DAT)
  dof_map = build_dof_map(mesh)
  problem = pack_problem(mesh, dof_map, 1.0e6, 0.25, constraints, ties)
  assert len(problem.mpc_slave_dof) == 0
  assert len(problem.constraint_dof) == 15


def test_load_problem_mpc_skim() -> None:
  loaded = load_problem(ROOT / "skims" / "patch_test8_mpc" / "problem.toml")
  assert len(loaded.problem.mpc_slave_dof) == 0
  assert len(loaded.problem.constraint_dof) == 15
