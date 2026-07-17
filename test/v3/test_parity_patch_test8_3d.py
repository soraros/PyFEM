# SPDX-License-Identifier: MIT

"""Extended parity checks for PatchTest8_3D vs legacy."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from _legacy_parity import (
  legacy_element_stiffness,
  legacy_external_load,
  legacy_state,
  legacy_stiffness_coo,
)

from pyfem.v3 import load_problem, solve_linear
from pyfem.v3._prototype_assembly import assemble_loaded
from pyfem.v3.fem.element import hex8_stiffness, tet4_stiffness
from pyfem.v3.materials.isotropic import isotropic_matrix
from pyfem.v3.types import LoadedProblem

ROOT = Path(__file__).resolve().parents[2]
SKIM_DIR = ROOT / "skims" / "patch_test8_3d"
SKIM_PRO = SKIM_DIR / "skim.pro"
PARITY = SKIM_DIR / "parity.toml"
_K_ATOL = 1e-8
_KE_ATOL = 1e-8


@pytest.fixture(scope="module")
def tolerances() -> tuple[float, float]:
  with PARITY.open("rb") as fh:
    data = tomllib.load(fh)
  return float(data["rtol"]), float(data["atol"])


@pytest.fixture(scope="module")
def loaded() -> LoadedProblem:
  return load_problem(SKIM_DIR / "problem.toml")


def test_parity_from_problem_toml(
  loaded: LoadedProblem,
  tolerances: tuple[float, float],
) -> None:
  rtol, atol = tolerances
  v3_state = solve_linear(loaded)
  legacy = legacy_state(SKIM_PRO)
  np.testing.assert_allclose(v3_state, legacy, rtol=rtol, atol=atol)


def test_parity_from_skim_pro(tolerances: tuple[float, float]) -> None:
  rtol, atol = tolerances
  loaded = load_problem(SKIM_PRO)
  v3_state = solve_linear(loaded)
  legacy = legacy_state(SKIM_PRO)
  np.testing.assert_allclose(v3_state, legacy, rtol=rtol, atol=atol)


def test_toml_and_skim_pro_states_identical(loaded: LoadedProblem) -> None:
  skim_loaded = load_problem(SKIM_PRO)
  np.testing.assert_allclose(
    solve_linear(loaded),
    solve_linear(skim_loaded),
    rtol=0.0,
    atol=0.0,
  )


def test_global_stiffness_coo_matches_legacy(loaded: LoadedProblem) -> None:
  v3 = assemble_loaded(loaded).stiffness.tocoo()
  legacy = legacy_stiffness_coo(SKIM_PRO)
  assert v3.shape == legacy.shape == (48, 48)
  np.testing.assert_array_equal(v3.row, legacy.row)
  np.testing.assert_array_equal(v3.col, legacy.col)
  np.testing.assert_allclose(v3.data, legacy.data, rtol=0.0, atol=_K_ATOL)


def test_external_load_matches_legacy(loaded: LoadedProblem) -> None:
  v3 = assemble_loaded(loaded).load
  legacy = legacy_external_load(SKIM_PRO)
  np.testing.assert_allclose(v3, legacy, rtol=0.0, atol=0.0)


def test_constitutive_matrix_matches_isotropic(loaded: LoadedProblem) -> None:
  np.testing.assert_allclose(
    loaded.problem.constitutive,
    isotropic_matrix(1.0e6, 0.25),
    rtol=0.0,
    atol=0.0,
  )


def test_prescribed_dofs_exact_in_solution(loaded: LoadedProblem) -> None:
  state = solve_linear(loaded)
  problem = loaded.problem
  for dof_id, value in zip(problem.constraint_dof, problem.constraint_val, strict=True):
    assert state[dof_id] == pytest.approx(value)


def test_equilibrium_residual_on_free_dofs(loaded: LoadedProblem) -> None:
  system = assemble_loaded(loaded)
  state = solve_linear(loaded)
  problem = loaded.problem
  constrained = set(problem.constraint_dof.tolist())
  free = np.array([i for i in range(problem.n_dofs) if i not in constrained])
  k = system.stiffness.tocsr()
  residual = k @ state - system.load
  np.testing.assert_allclose(
    residual[free],
    0.0,
    rtol=0.0,
    atol=1e-6,
  )


@pytest.mark.parametrize("elem_index", range(5))
def test_element_stiffness_matches_legacy(
  loaded: LoadedProblem,
  elem_index: int,
) -> None:
  coords = loaded.problem.coords[loaded.problem.conn[elem_index]]
  c = loaded.problem.constitutive
  k_v3 = hex8_stiffness(coords, c)
  k_legacy = legacy_element_stiffness(SKIM_PRO, coords)
  np.testing.assert_allclose(k_v3, k_legacy, rtol=0.0, atol=_KE_ATOL)


def test_tet4_stiffness_matches_legacy_on_unit_tet(tmp_path: Path) -> None:
  dat = tmp_path / "unit_tet.dat"
  pro = tmp_path / "unit_tet.pro"
  dat.write_text(
    "\n".join(
      [
        "<Nodes>",
        "  0 0.0 0.0 0.0;",
        "  1 1.0 0.0 0.0;",
        "  2 0.0 1.0 0.0;",
        "  3 0.0 0.0 1.0;",
        "</Nodes>",
        "<Elements>",
        '  1 "ContElem" 0 1 2 3;',
        "</Elements>",
        "<NodeConstraints>",
        "</NodeConstraints>",
        "<ExternalForces>",
        "</ExternalForces>",
      ]
    ),
    encoding="utf-8",
  )
  pro.write_text(
    "\n".join(
      [
        'input = "unit_tet.dat";',
        "ContElem = {",
        '  type = "SmallStrainContinuum";',
        "  material = {",
        '    type = "Isotropic";',
        "    E    = 1.e6;",
        "    nu   = 0.25;",
        "  };",
        "};",
        "solver = { type = \"LinearSolver\"; };",
      ]
    ),
    encoding="utf-8",
  )
  coords = np.array(
    [
      [0.0, 0.0, 0.0],
      [1.0, 0.0, 0.0],
      [0.0, 1.0, 0.0],
      [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
  )
  c = isotropic_matrix(1.0e6, 0.25)
  k_v3 = tet4_stiffness(coords, c)
  k_legacy = legacy_element_stiffness(pro, coords)
  np.testing.assert_allclose(k_v3, k_legacy, rtol=0.0, atol=_KE_ATOL)
