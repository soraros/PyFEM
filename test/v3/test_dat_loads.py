# SPDX-License-Identifier: MIT

"""`.dat` reader: nodal loads from ``<ExternalForces>``."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from pyfem.v3 import load_problem
from pyfem.v3.io.dat import read_dat_mesh
from pyfem.v3.mesh import build_dof_map
from pyfem.v3.pack import pack_problem

ROOT = Path(__file__).resolve().parents[2]
LOADED_DAT = ROOT / "skims" / "patch_test8_loaded" / "PatchTest8_loaded.dat"
PATCH8_DAT = ROOT / "examples" / "ch02" / "PatchTest8.dat"


def test_patch_test8_dat_has_no_nodal_loads() -> None:
  _mesh, _constraints, _ties, loads = read_dat_mesh(PATCH8_DAT)
  assert loads == ()


def test_loaded_dat_parses_nodal_load() -> None:
  mesh, _constraints, _ties, loads = read_dat_mesh(LOADED_DAT)
  assert len(loads) == 1
  assert loads[0].node_id == 13
  assert loads[0].dof_type == "v"
  assert loads[0].value == pytest.approx(1000.0)


def test_pack_maps_load_to_global_dof() -> None:
  mesh, constraints, _ties, loads = read_dat_mesh(LOADED_DAT)
  dof_map = build_dof_map(mesh)
  problem = pack_problem(mesh, dof_map, 1.0e6, 0.25, constraints, loads=loads)
  dof_idx = dof_map.dof_index(13, "v")
  assert problem.external_load[dof_idx] == pytest.approx(1000.0)
  assert problem.external_load.sum() == pytest.approx(1000.0)


def test_load_problem_includes_external_load() -> None:
  loaded = load_problem(ROOT / "skims" / "patch_test8_loaded" / "problem.toml")
  assert loaded.problem.external_load.sum() == pytest.approx(1000.0)
