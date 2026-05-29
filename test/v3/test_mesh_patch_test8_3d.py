# SPDX-License-Identifier: MIT

"""Mesh I/O parity for PatchTest8_3D."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from pyfem.fem.NodeSet import NodeSet
from pyfem.v3.io.dat import read_dat_mesh

ROOT = Path(__file__).resolve().parents[2]
DAT = ROOT / "examples" / "ch02" / "PatchTest8_3D.dat"


def test_mesh_matches_legacy_node_coords() -> None:
  mesh, _constraints, _ties, _loads = read_dat_mesh(DAT)

  legacy = NodeSet()
  legacy.readFromFile(str(DAT))

  assert mesh.n_nodes == len(legacy)
  assert mesh.rank == legacy.rank == 3
  assert mesh.n_elems == 5
  assert mesh.conn.shape == (5, 8)

  for i, node_id in enumerate(mesh.node_ids):
    legacy_coords = np.asarray(legacy.get(int(node_id)))
    np.testing.assert_allclose(mesh.coords[i], legacy_coords, rtol=0, atol=0)


def test_mesh_w_constraints_count() -> None:
  _mesh, constraints, _ties, loads = read_dat_mesh(DAT)
  assert len(constraints) == 24
  assert len(loads) == 0
  w_constraints = [c for c in constraints if c.dof_type == "w"]
  assert len(w_constraints) == 8
