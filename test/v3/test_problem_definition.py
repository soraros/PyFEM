# SPDX-License-Identifier: MIT

"""ProblemDefinition NamedTuple (array-only, jitable)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from pyfem.v3 import ProblemDefinition, load_problem

ROOT = Path(__file__).resolve().parents[2]
SKIM = ROOT / "skims" / "patch_test8" / "problem.toml"


def test_problem_definition_is_namedtuple_with_contiguous_arrays() -> None:
  loaded = load_problem(SKIM)
  p = loaded.problem
  assert isinstance(p, ProblemDefinition)
  assert p.coords.dtype == np.float64
  assert p.conn.dtype == np.int32
  assert p.coords.flags.c_contiguous
  assert p.constitutive.shape == (3, 3)
  assert p.n_nodes == p.coords.shape[0]
  assert p.n_elems == p.conn.shape[0]
  assert p.n_dofs == p.global_dofs.size


def test_problem_definition_field_access() -> None:
  loaded = load_problem(SKIM)
  p = loaded.problem
  assert p.coords.ndim == 2
  assert p.conn.shape[1] == 8
  assert len(p.constraint_dof) == len(p.constraint_val)
