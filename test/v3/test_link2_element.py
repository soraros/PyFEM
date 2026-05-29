# SPDX-License-Identifier: MIT

"""2-node link element (truss + spring) stiffness checks."""

from __future__ import annotations

import sys

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from pyfem.v3.fem.link2 import link2_tangent_single
from pyfem.v3.types import GROUP_SPRING, GROUP_TRUSS


def test_truss_stiffness_at_zero_state() -> None:
  coords = np.array([[-10.0, 0.0], [0.0, 0.5]], dtype=np.float64)
  state = np.zeros(4, dtype=np.float64)
  ke, fe = link2_tangent_single(coords, state, GROUP_TRUSS, 5.0e6, 1.0)
  assert fe.shape == (4,)
  assert ke.shape == (4, 4)
  np.testing.assert_allclose(fe, np.zeros(4), atol=1.0e-12)
  assert ke[0, 0] > 0.0


def test_spring_stiffness_and_force() -> None:
  coords = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
  state = np.array([0.0, 0.0, 0.1, 0.0], dtype=np.float64)
  ke, fe = link2_tangent_single(coords, state, GROUP_SPRING, 100.0, 0.0)
  np.testing.assert_allclose(fe[2], 10.0, rtol=1.0e-12)
  np.testing.assert_allclose(fe[0], -10.0, rtol=1.0e-12)
  assert ke[0, 0] == pytest.approx(100.0)
