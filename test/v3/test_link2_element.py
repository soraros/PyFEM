# SPDX-License-Identifier: MIT

"""2-node link element (truss + spring) stiffness checks."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from pyfem.elements.Spring import Spring as LegacySpring
from pyfem.elements.Truss import Truss as LegacyTruss
from pyfem.io.InputReader import InputRead
from pyfem.util.dataStructures import elementData
from pyfem.v3.fem.link2 import link2_tangent_batched, link2_tangent_single
from pyfem.v3.types import GROUP_SPRING, GROUP_TRUSS

ROOT = Path(__file__).resolve().parents[2]
TRUSS_SKIM_PRO = ROOT / "skims" / "shallow_truss_riks" / "skim.pro"

TRUSS_E = 5.0e6
TRUSS_AREA = 1.0
SPRING_K = 100.0

# Angled orientations spanning all four quadrants. Forward (R K R^T) and
# transpose (R^T K R) rotations differ only where cos*sin != 0, so axis-aligned
# elements cannot distinguish the two conventions.
ANGLED_COORDS = (
  np.array([[0.0, 0.0], [3.0, 1.0]], dtype=np.float64),
  np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64),
  np.array([[1.0, -2.0], [-1.5, 0.5]], dtype=np.float64),
  np.array([[0.0, 0.0], [0.3, 2.7]], dtype=np.float64),
  np.array([[2.0, 1.0], [-1.0, 0.0]], dtype=np.float64),
)
ANGLED_IDS = ("alpha18", "alpha45", "alpha135", "alpha84", "alpha162")

PROBE_STATE = np.array([0.05, -0.02, 0.4, 0.15], dtype=np.float64)


def _unit_axial_vector(coords: np.ndarray) -> np.ndarray:
  """Unit axial vector ``b = (c, s, -c, -s)`` of a 2-node element."""
  direction = coords[1] - coords[0]
  c, s = direction / np.linalg.norm(direction)
  return np.array([c, s, -c, -s], dtype=np.float64)


def _unit_transverse_vector(coords: np.ndarray) -> np.ndarray:
  """Unit transverse vector ``t = (-s, c, s, -c)`` of a 2-node element."""
  direction = coords[1] - coords[0]
  c, s = direction / np.linalg.norm(direction)
  return np.array([-s, c, s, -c], dtype=np.float64)


@pytest.fixture(scope="module")
def legacy_truss_element() -> LegacyTruss:
  props, globdat = InputRead(str(TRUSS_SKIM_PRO))
  element = next(iter(globdat.elements.iterElementGroup("TrussElem")))
  element.globdat = globdat
  return element


def _legacy_truss_response(
  element: LegacyTruss,
  coords: np.ndarray,
  state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  """Legacy ``Truss`` tangent and internal force; state=Dstate with sigma reset."""
  element.setHistoryParameter("sigma", 0.0)
  element.commitHistory()
  elemdat = elementData(state, state)
  elemdat.coords = coords
  elemdat.props = SimpleNamespace(E=TRUSS_E, Area=TRUSS_AREA)
  elemdat.stiff.fill(0.0)
  elemdat.fint.fill(0.0)
  element.getTangentStiffness(elemdat)
  return elemdat.stiff.copy(), elemdat.fint.copy()


@pytest.fixture(scope="module")
def legacy_spring_element() -> LegacySpring:
  props, globdat = InputRead(str(TRUSS_SKIM_PRO))
  element = next(iter(globdat.elements.iterElementGroup("SpringElem")))
  element.globdat = globdat
  return element


def _legacy_spring_response(
  element: LegacySpring,
  coords: np.ndarray,
  state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  """Legacy ``Spring`` tangent and internal force (no history to reset)."""
  elemdat = elementData(state, state)
  elemdat.coords = coords
  elemdat.props = SimpleNamespace(k=SPRING_K)
  elemdat.stiff.fill(0.0)
  elemdat.fint.fill(0.0)
  element.getTangentStiffness(elemdat)
  return elemdat.stiff.copy(), elemdat.fint.copy()


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


@pytest.mark.parametrize("coords", ANGLED_COORDS, ids=ANGLED_IDS)
def test_truss_zero_state_tangent_matches_closed_form(coords: np.ndarray) -> None:
  state = np.zeros(4, dtype=np.float64)
  ke, fe = link2_tangent_single(coords, state, GROUP_TRUSS, TRUSS_E, TRUSS_AREA)
  np.testing.assert_allclose(fe, np.zeros(4), atol=1.0e-12)
  l0 = float(np.linalg.norm(coords[1] - coords[0]))
  b = _unit_axial_vector(coords)
  np.testing.assert_allclose(
    ke, (TRUSS_E * TRUSS_AREA / l0) * np.outer(b, b), rtol=1.0e-12, atol=1.0e-6
  )


@pytest.mark.parametrize("coords", ANGLED_COORDS, ids=ANGLED_IDS)
def test_spring_tangent_matches_closed_form(coords: np.ndarray) -> None:
  """Pin the axial spring semantics: ``K = k * b b^T``, ``b = (c, s, -c, -s)``.

  This tangent is the exact derivative of the axial residual
  ``f = -k * (b.a) * b``. The legacy Spring element now shares these
  semantics — see ``test_spring_legacy_tangent_repaired_matches_v3``.
  """
  ke, _ = link2_tangent_single(coords, PROBE_STATE, GROUP_SPRING, SPRING_K, 0.0)
  b = _unit_axial_vector(coords)
  np.testing.assert_allclose(ke, SPRING_K * np.outer(b, b), rtol=1.0e-12, atol=1.0e-10)


@pytest.mark.parametrize("coords", ANGLED_COORDS, ids=ANGLED_IDS)
def test_spring_transverse_motion_is_unresisted(coords: np.ndarray) -> None:
  """An axial spring resists only relative motion along its axis."""
  ke, _ = link2_tangent_single(coords, PROBE_STATE, GROUP_SPRING, SPRING_K, 0.0)
  t_perp = _unit_transverse_vector(coords)
  np.testing.assert_allclose(ke @ t_perp, np.zeros(4), atol=1.0e-10)


@pytest.mark.parametrize("coords", ANGLED_COORDS, ids=ANGLED_IDS)
def test_spring_legacy_tangent_repaired_matches_v3(
  legacy_spring_element: LegacySpring,
  coords: np.ndarray,
) -> None:
  """Repair confirmed: the legacy Spring tangent now matches v3 exactly.

  Legacy ``pyfem/elements/Spring.py`` used to assemble an isotropic tangent
  ``k * [[I, -I], [-I, I]]`` (stiffness ``k`` in the axial AND transverse
  directions, rotation-invariant) while its internal force is purely axial —
  a tangent that was not the derivative of its own residual (finding
  20260929-agent-b3), so the legacy spring TANGENT was not a valid oracle
  and differed from the consistent axial tangent by exactly the transverse
  projector ``k * t t^T`` (M24 pinned that divergence here as
  ``test_spring_legacy_tangent_is_not_the_oracle``). M55 repaired the legacy
  element to the axial-only tangent ``k * b b^T`` — the exact derivative of
  its unchanged axial residual (commit ed798ae, "correctness bug") — so the
  relationship is now parity: legacy tangent == v3 tangent, and the legacy
  RESIDUAL stays authoritative (internal-force parity holds, as before).
  """
  k_legacy, f_legacy = _legacy_spring_response(
    legacy_spring_element, coords, PROBE_STATE
  )
  ke, fe = link2_tangent_single(coords, PROBE_STATE, GROUP_SPRING, SPRING_K, 0.0)
  b = _unit_axial_vector(coords)
  np.testing.assert_allclose(
    k_legacy, SPRING_K * np.outer(b, b), rtol=1.0e-12, atol=1.0e-10
  )
  np.testing.assert_allclose(k_legacy, ke, rtol=1.0e-12, atol=1.0e-10)
  np.testing.assert_allclose(f_legacy, fe, rtol=1.0e-10, atol=1.0e-8)
  t_perp = _unit_transverse_vector(coords)
  np.testing.assert_allclose(k_legacy @ t_perp, np.zeros(4), atol=1.0e-10)


@pytest.mark.parametrize("coords", ANGLED_COORDS, ids=ANGLED_IDS)
@pytest.mark.parametrize(
  "state",
  (
    np.zeros(4, dtype=np.float64),
    PROBE_STATE,
    np.array([-0.03, 0.06, 0.12, -0.2], dtype=np.float64),
  ),
  ids=("zero", "tensile", "mixed"),
)
def test_truss_tangent_matches_legacy(
  legacy_truss_element: LegacyTruss,
  coords: np.ndarray,
  state: np.ndarray,
) -> None:
  k_legacy, f_legacy = _legacy_truss_response(legacy_truss_element, coords, state)
  ke, fe = link2_tangent_single(coords, state, GROUP_TRUSS, TRUSS_E, TRUSS_AREA)
  np.testing.assert_allclose(fe, f_legacy, rtol=1.0e-10, atol=1.0e-8)
  np.testing.assert_allclose(ke, k_legacy, rtol=1.0e-10, atol=1.0e-6)


@pytest.mark.parametrize("coords", ANGLED_COORDS, ids=ANGLED_IDS)
@pytest.mark.parametrize(
  "group_kind,prop0,prop1",
  ((GROUP_TRUSS, TRUSS_E, TRUSS_AREA), (GROUP_SPRING, SPRING_K, 0.0)),
  ids=("truss", "spring"),
)
def test_tangent_is_directional_derivative_of_force(
  coords: np.ndarray,
  group_kind: int,
  prop0: float,
  prop1: float,
) -> None:
  direction = np.array([0.3, -0.7, 0.6, 0.2], dtype=np.float64)
  direction /= np.linalg.norm(direction)
  eps = 1.0e-7
  ke, _ = link2_tangent_single(coords, PROBE_STATE, group_kind, prop0, prop1)
  _, f_plus = link2_tangent_single(
    coords, PROBE_STATE + eps * direction, group_kind, prop0, prop1
  )
  _, f_minus = link2_tangent_single(
    coords, PROBE_STATE - eps * direction, group_kind, prop0, prop1
  )
  finite_difference = (f_plus - f_minus) / (2.0 * eps)
  np.testing.assert_allclose(
    ke @ direction, finite_difference, rtol=1.0e-6, atol=1.0e-5
  )


@pytest.mark.parametrize(
  "group_kind,prop0,prop1",
  ((GROUP_TRUSS, TRUSS_E, TRUSS_AREA), (GROUP_SPRING, SPRING_K, 0.0)),
  ids=("truss", "spring"),
)
def test_batched_matches_single_on_angled_batch(
  group_kind: int,
  prop0: float,
  prop1: float,
) -> None:
  coords_batch = np.stack(ANGLED_COORDS)
  states_batch = np.tile(PROBE_STATE, (len(ANGLED_COORDS), 1))
  k_batch, f_batch = link2_tangent_batched(
    coords_batch, states_batch, group_kind, prop0, prop1
  )
  for elem, coords in enumerate(ANGLED_COORDS):
    ke, fe = link2_tangent_single(coords, states_batch[elem], group_kind, prop0, prop1)
    np.testing.assert_allclose(k_batch[elem], ke, rtol=1.0e-13, atol=1.0e-9)
    np.testing.assert_allclose(f_batch[elem], fe, rtol=1.0e-13, atol=1.0e-12)
