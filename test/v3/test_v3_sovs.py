# SPDX-License-Identifier: MIT

"""Skorohod-Olevsky viscous sintering: legacy parity oracles and the time-port path.

The legacy law ``pyfem/materials/SOVS.py`` is the numerical reference
(AGENTS.md). The parity harness drives it with the
``dstrain = eps_total - strain_committed`` identity the v3 kernel uses — the
same float64 subtraction — and sets ``solverStat.time`` per committed step,
the hidden channel the v3 law replaces with the declared identity signal
port. Because the kernel replicates the legacy ``getStress`` arithmetic
statement for statement and both sides run identical NumPy operations in
identical order, every law-level stress/state/tangent comparison is bitwise
on every platform — including at the density kinks (the
``rho >= 0.999 -> 1e20`` viscosity guard at SOVS.py:236-242 and the ``rho``
clamp into ``[rho0, 1]`` at :292-293), which are state-trajectory kinks, not
strain-response kinks: at fixed committed state the implemented map is AFFINE
in the trial strain (the clamp binds only ``rho_new`` while the volumetric
viscous strain reads the unclamped ``drho`` at :297), so the closed-form
tangent is exact at every state and the finite-difference legs need no kink
avoidance (the M26 skip-and-document precedent is noted for the record and
deliberately not exercised).

Dormancy (finding 20261009-agent-vp1-improve-sovs-vp-migration-relevant-
quirks-hard-coded-moduli-dormant): the shipped decks
(examples/materials/sintering/) carry eta0 = 1e12, Q = 5e5, T = 1600, so the
Arrhenius reference viscosity eta_ref = 2.1e28 Pa.s keeps the viscous
machinery inert (drho rounds to zero; measured response ~1e-12) — a
schedule-based oracle would be trivially elastic, so this battery owns
activated constants (eta0 = 1e10, Q = 1.0, T = 1600, rho0 = 0.6,
sigma_sint = 1e6, n_vol = 2, n_shear = 1: the survey's finding leg, rho =
0.6000062 after 20 free-sintering steps at dtime = 0.01). The dormant deck
configuration is pinned as its own bitwise leg.

The tangent relationship is parity-where-repaired (the M55/l2 flip
precedent). The legacy coded tangent used to be the closed form of an
IMPLICIT step while the stress update is explicit forward Euler (finding
20261009-agent-vp1-bug-sovs-tangent-is-the-implicit-step-form-of-an-explicit-update):
central FD of the legacy law's own response convicted it at 9.41e-3 relative
of max|tang| at dtime = 0.01 and 9.94e-4 at dtime = 0.001 (O(dt)
inconsistency; elastic branch exact), and this battery pinned the divergence
by mechanism against legacy at the fork base (the legacy oracle reproducing
the coded recomputation bit for bit, the v3 kernel the true recomputation).
M67 repaired the legacy side (commit 30a5f4e) to the same true explicit-map
derivative the v3 kernel writes — the three-leg closed form of the kernel
module docstring: ``K_alg = K(1 - 3K dt/2ηv)`` volumetric,
``G_dev = G(1 - G dt/ηs)`` on the normal-deviatoric block,
``G_alg = G(1 - G dt/2ηs)`` shear — FD-exact to 3.3e-15 at the activated
state (rounding level, the map is affine). The v3 kernel is unchanged, so
the pins below assert bitwise tangent equality on every step, kink states
included; the coded implicit-step form is retained as the divergence-record
witness in the FD leg, still failing the shared FD by the documented
9.41e-3/9.94e-4. The M63 survey sketch's two-modulus "isotropic
(K_alg, G_alg)" assembly is NOT the derivative of the implemented map
(1.4e-2 against the same FD — it misses the deviatoric viscous flow on the
normal components); the shared tangent remains the kernel's three-modulus
form.

The driver battery proves schedule-owned time flows through the M29 identity
signal port: committed state rows equal the kernel oracle stepped on the
committed integration-point strains (bitwise), the ``time`` and ``rho`` slots
record the bound time and the densification at every integration point, a
slower time schedule densifies less (positive control — SOVS is genuinely
rate-dependent), and a port-free law driven under two different time
schedules yields byte-identical results (isolation). The ABI proof is the
frozen validator accepting the law's v2 metadata unchanged plus
``pyfem/v3/compile/contracts.py`` carrying no diff for this mission; the
``rho = rho0`` initial rows are the first parameter-dependent initial state
of the stateful family (M25 G2 binding), pinned at compile level below.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from pyfem.materials.SOVS import SkorohodOlevsky
from pyfem.v3.compile.continuum import (
  q8_reference_registry,
  sovs_reference_registry,
)
from pyfem.v3.compile.contracts import (
  STATEFUL_MATERIAL_DESCRIPTOR_SCHEMA,
  StatefulContinuumKernelResult,
  StatefulContinuumSignalDerivative,
  StatefulContinuumSignalInput,
  resolve_material_state_slots,
  validate_stateful_material_metadata,
)
from pyfem.v3.compile.system import compile_system
from pyfem.v3.constraints import compile_constraint_map
from pyfem.v3.driver import (
  DriverStatus,
  NonlinearStaticDriver,
  NonlinearStaticSettings,
  evaluate_signals,
)
from pyfem.v3.materials.skorohod_olevsky import (
  SKOROHOD_OLEVSKY_BINDING,
  skorohod_olevsky_calibration,
  skorohod_olevsky_initial_state,
  skorohod_olevsky_kernel,
  skorohod_olevsky_metadata,
)
from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.operator import (
  ChannelRequest,
  EvaluationStatus,
  OperatorEvaluationInput,
  ProgramSignalInput,
  SignalDerivativeInput,
)
from pyfem.v3.model.provenance import CanonicalManifest
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
  AffineValueSpec,
  DofRef,
  PrescribedDofSpec,
  ProgramCoordinateSpec,
  ProgramCoordinateValue,
  ProgramPoint,
)
from pyfem.v3.state import StateCodecError

# Documented configurations: the dormant shipped deck
# (examples/materials/sintering/free_sintering.pro: eta0 = 1e12, Q = 5e5,
# T = 1600, rho0 = 0.6, sigma_sint = 1e6, n_vol = 2, n_shear = 1, dtime = 2)
# and the battery-owned activated family (eta0 = 1e10, Q = 1.0).
_DECK = {
  "eta0": 1.0e12,
  "Q": 5.0e5,
  "T": 1600.0,
  "rho0": 0.6,
  "sigma_sint": 1.0e6,
  "n_vol": 2.0,
  "n_shear": 1.0,
  "R": 8.314,
}
_ACTIVATED = {
  "eta0": 1.0e10,
  "Q": 1.0,
  "T": 1600.0,
  "rho0": 0.6,
  "sigma_sint": 1.0e6,
  "n_vol": 2.0,
  "n_shear": 1.0,
  "R": 8.314,
}
_FD_STEP = 1.0e-7

_UNIT_COORDINATES = (
  (0.0, 0.0),
  (0.5, 0.0),
  (1.0, 0.0),
  (1.0, 0.5),
  (1.0, 1.0),
  (0.5, 1.0),
  (0.0, 1.0),
  (0.0, 0.5),
)


def _source(label: str) -> SourceContext:
  return SourceContext(source=label)


def _parameters(config: dict) -> tuple[float, ...]:
  return (
    config["eta0"],
    config["Q"],
    config["T"],
    config["rho0"],
    config["sigma_sint"],
    config["R"],
    config["n_vol"],
    config["n_shear"],
  )


class _LegacyProps:
  """Minimal iterable props stand-in for ``BaseMaterial``."""

  def __init__(self, **values: object) -> None:
    self._values = values
    self.solverStat = None
    for name, value in values.items():
      setattr(self, name, value)

  def __iter__(self) -> object:
    return iter(self._values.items())


def _legacy_law(config: dict) -> SkorohodOlevsky:
  with contextlib.redirect_stdout(io.StringIO()):
    with warnings.catch_warnings(action="ignore", category=DeprecationWarning):
      return SkorohodOlevsky(_LegacyProps(**config))


def _legacy_row(mat: SkorohodOlevsky) -> np.ndarray:
  return np.concatenate(
    (
      np.atleast_1d(mat.getHistoryParameter("strain")),
      np.atleast_1d(mat.getHistoryParameter("strain_visc")),
      np.atleast_1d(mat.getHistoryParameter("rho")),
      np.atleast_1d(mat.getHistoryParameter("time_old")),
    )
  )


def _legacy_step(
  mat: SkorohodOlevsky,
  strain_total: np.ndarray,
  time_new: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Advance the legacy oracle one committed step on the total-strain path."""
  dstrain = np.array(strain_total - mat.getHistoryParameter("strain"), copy=True)
  mat.solverStat = SimpleNamespace(time=time_new)
  with contextlib.redirect_stdout(io.StringIO()):
    with warnings.catch_warnings(action="ignore", category=DeprecationWarning):
      sigma, tangent = mat.getStress(SimpleNamespace(dstrain=dstrain))
  mat.commitHistory()
  return np.array(sigma, copy=True), np.array(tangent, copy=True), _legacy_row(mat)


def _legacy_probe(
  mat: SkorohodOlevsky,
  strain_total: np.ndarray,
  time_new: float,
) -> tuple[np.ndarray, np.ndarray]:
  """Evaluate the legacy oracle without committing (finite-difference probe)."""
  dstrain = np.array(strain_total - mat.getHistoryParameter("strain"), copy=True)
  mat.solverStat = SimpleNamespace(time=time_new)
  with contextlib.redirect_stdout(io.StringIO()):
    with warnings.catch_warnings(action="ignore", category=DeprecationWarning):
      sigma, tangent = mat.getStress(SimpleNamespace(dstrain=dstrain))
  return np.array(sigma, copy=True), np.array(tangent, copy=True)


def _time_signal(time: float) -> StatefulContinuumSignalInput:
  return StatefulContinuumSignalInput(
    port_id="time",
    values=np.array([time], dtype=np.float64),
    derivatives=(
      StatefulContinuumSignalDerivative("time", np.array([1.0], dtype=np.float64)),
    ),
  )


def _program_time_signal(time: float) -> ProgramSignalInput:
  return ProgramSignalInput(
    port_id="time",
    values=FinalizedArray(np.array([time], dtype=np.float64), dtype=np.float64),
    derivatives=(
      SignalDerivativeInput(
        "time",
        FinalizedArray(np.array([1.0], dtype=np.float64), dtype=np.float64),
      ),
    ),
  )


def _v3_step(
  calibration: np.ndarray,
  rows: np.ndarray,
  strain_total: np.ndarray,
  time_new: float,
) -> StatefulContinuumKernelResult:
  result = skorohod_olevsky_kernel(
    strain_total[None, :], rows, calibration, (_time_signal(time_new),)
  )
  assert result.status is EvaluationStatus.OK
  return result


def _sovs_tangents(
  calibration: np.ndarray,
  row: np.ndarray,
  time_new: float,
) -> tuple[np.ndarray, np.ndarray, bool]:
  """Replicate the tangent construction, repaired-shared and pre-repair.

  Both forms depend only on the committed rho and the time increment (the
  map is affine in the trial strain, so the tangent carries no strain
  argument): the v3 kernel's true three-leg closed form (bitwise-equal to
  the kernel's tangent AND to the M67-repaired legacy oracle's tangent —
  commit 30a5f4e replaced the implicit-step form) and the pre-repair legacy
  coded implicit-step form, retained as the divergence-record witness
  (bitwise-equal to the PRE-repair legacy oracle's tangent). Returns
  ``active=False`` with two elastic-stiffness copies on dtime <= 0.
  """
  eta_ref = calibration[0]
  rho0 = calibration[1]
  n_vol = calibration[3]
  n_shear = calibration[4]
  e_base = calibration[5]
  rho_power = calibration[6]
  nu_eff = calibration[7]
  dtime = time_new - row[13]
  rho = float(row[12])
  if rho < 0.999:
    eta_vol = eta_ref * (rho ** (-n_vol) - 1.0)
    eta_shear = eta_ref * (rho ** (-n_shear) - 1.0)
  else:
    eta_vol = 1.0e20
    eta_shear = 1.0e20
  rho_factor = rho / rho0
  e_eff = e_base * (rho_factor**rho_power)
  ebulk3 = e_eff / (1.0 - 2.0 * nu_eff)
  eg2 = e_eff / (1.0 + nu_eff)
  eg = 0.5 * eg2
  elam = (ebulk3 - eg2) / 3.0
  ctang = np.zeros(shape=(6, 6))
  ctang[:3, :3] = elam
  ctang[0, 0] += eg2
  ctang[1, 1] = ctang[0, 0]
  ctang[2, 2] = ctang[0, 0]
  ctang[3, 3] = eg
  ctang[4, 4] = ctang[3, 3]
  ctang[5, 5] = ctang[3, 3]
  if dtime <= 0.0:
    return np.array(ctang, copy=True), np.array(ctang, copy=True), False
  k_mod = ebulk3 / 3.0
  g_mod = eg
  # Pre-repair legacy coded tangent: the implicit-step closed form (the
  # divergence record; M67 commit 30a5f4e replaced it with the true form).
  k_tang = k_mod / (1.0 + k_mod * dtime * 3.0 / (2.0 * eta_vol))
  g_tang = g_mod / (1.0 + g_mod * dtime / eta_shear)
  lam_tang = k_tang - 2.0 * g_tang / 3.0
  coded = np.zeros(shape=(6, 6))
  coded[:3, :3] = lam_tang
  coded[0, 0] += 2.0 * g_tang
  coded[1, 1] = coded[0, 0]
  coded[2, 2] = coded[0, 0]
  coded[3, 3] = g_tang
  coded[4, 4] = g_tang
  coded[5, 5] = g_tang
  # True explicit-map tangent: the v3 kernel's three-leg closed form, now
  # also the M67-repaired legacy oracle's (SOVS.py:320-341, commit 30a5f4e).
  k_alg = k_mod * (1.0 - 3.0 * k_mod * dtime / (2.0 * eta_vol))
  g_dev = g_mod * (1.0 - g_mod * dtime / eta_shear)
  g_alg = g_mod * (1.0 - g_mod * dtime / (2.0 * eta_shear))
  lam_alg = k_alg - 2.0 * g_dev / 3.0
  true = np.zeros(shape=(6, 6))
  true[:3, :3] = lam_alg
  true[0, 0] += 2.0 * g_dev
  true[1, 1] = true[0, 0]
  true[2, 2] = true[0, 0]
  true[3, 3] = g_alg
  true[4, 4] = true[3, 3]
  true[5, 5] = true[3, 3]
  return coded, true, True


def _fd_tangent_v3(
  calibration: np.ndarray,
  rows: np.ndarray,
  strain_total: np.ndarray,
  time_new: float,
) -> np.ndarray:
  """Central finite difference of the v3 kernel's own stress response."""
  fd = np.zeros((6, 6))
  for component in range(6):
    delta = _FD_STEP * np.eye(6)[component]
    plus = _v3_step(calibration, rows, strain_total + delta, time_new)
    minus = _v3_step(calibration, rows, strain_total - delta, time_new)
    fd[:, component] = (plus.stresses[0] - minus.stresses[0]) / (2.0 * _FD_STEP)
  return fd


def _fd_tangent_legacy(
  mat: SkorohodOlevsky,
  strain_total: np.ndarray,
  time_new: float,
) -> np.ndarray:
  """Central finite difference of the legacy oracle's own stress response."""
  fd = np.zeros((6, 6))
  for component in range(6):
    delta = _FD_STEP * np.eye(6)[component]
    plus, _ = _legacy_probe(mat, strain_total + delta, time_new)
    minus, _ = _legacy_probe(mat, strain_total - delta, time_new)
    fd[:, component] = (plus - minus) / (2.0 * _FD_STEP)
  return fd


def _run_path_parity(
  config: dict,
  path: list[tuple[np.ndarray, float]],
) -> tuple[np.ndarray, list[np.ndarray], list[float]]:
  """Step both laws along one (total-strain, time) path; bitwise everywhere.

  Stress, state, AND tangent are bitwise-identical on every step, kink
  steps included: repair confirmed (module docstring), the M67-repaired
  legacy oracle and the v3 kernel both reproduce the true explicit-map
  recomputation bit for bit. On dtime <= 0 steps both tangents are the
  elastic stiffness, bitwise. The pre-repair coded form is not exercised
  here; it survives as the divergence-record witness in the FD leg.
  Returns the final v3 rows, the legacy stresses, and the per-step
  committed rho trajectory.
  """
  calibration = skorohod_olevsky_calibration(*_parameters(config))
  legacy = _legacy_law(config)
  rows_v = np.zeros((1, 14))
  rows_v[:, 12] = config["rho0"]
  stresses = []
  rhos = []
  for strain_total, time_new in path:
    strain_total = np.array(strain_total, dtype=np.float64)
    sigma_l, tangent_l, row_l = _legacy_step(legacy, strain_total, time_new)
    result = _v3_step(calibration, rows_v, strain_total, time_new)
    assert np.array_equal(result.stresses[0], sigma_l)
    assert np.array_equal(result.trial_rows[0], row_l)
    _, true, _ = _sovs_tangents(calibration, rows_v[0], time_new)
    assert np.array_equal(tangent_l, true)
    assert np.array_equal(result.tangents[0], true)
    assert np.array_equal(tangent_l, result.tangents[0])
    rows_v = result.trial_rows
    stresses.append(sigma_l)
    rhos.append(float(rows_v[0, 12]))
  return rows_v, stresses, rhos


def test_dormant_deck_constants_elastic_limit_bitwise() -> None:
  # The shipped deck constants (eta_ref = 2.1e28): the viscous machinery is
  # inert — drho rounds to zero and the tangent corrections underflow, so
  # stress, state, AND tangent are all bitwise parity, elastic-limit. The
  # virgin zero-response step (zero strain, dtime = 0) returns exactly zero.
  calibration = skorohod_olevsky_calibration(*_parameters(_DECK))
  assert calibration[0] > 1.0e28  # eta_ref, the dormancy marker
  rows = np.zeros((1, 14))
  rows[:, 12] = _DECK["rho0"]
  result = _v3_step(calibration, rows, np.zeros(6), 0.0)
  assert np.array_equal(result.stresses[0], np.zeros(6))
  assert np.array_equal(result.trial_rows, rows)
  # Deck schedule: dtime = 2 steps at zero strain, then a small strain step.
  path = [(np.zeros(6), 2.0 * (k + 1)) for k in range(3)]
  path += [(np.array([1.0e-4, 0.0, 0.0, 0.0, 0.0, 0.0]), 8.0)]
  rows_v, stresses, _ = _run_path_parity_overrides(_DECK, path)
  # rho never left rho0, bit for bit; the stress response stays at the
  # elastic limit magnitude (the dormancy pin: ~1e-12 after free steps).
  assert np.all(rows_v[:, 12] == _DECK["rho0"])
  assert float(np.max(np.abs(stresses[0]))) < 1.0e-9
  assert float(np.max(np.abs(stresses[2]))) < 1.0e-9
  assert float(np.max(np.abs(stresses[3]))) > 1.0e6  # the strained step responds


def _run_path_parity_overrides(
  config: dict,
  path: list[tuple[np.ndarray, float]],
) -> tuple[np.ndarray, list[np.ndarray], list[float]]:
  """The parity loop at the dormant deck constants (the elastic limit).

  At the dormant deck constants the tangent correction underflows below the
  elastic-stiffness ulp, so the repaired shared tangent and the retained
  pre-repair coded form coincide bit for bit with the elastic stiffness —
  full bitwise parity, with the underflow coincidence as the dormancy
  witness.
  """
  calibration = skorohod_olevsky_calibration(*_parameters(config))
  legacy = _legacy_law(config)
  rows_v = np.zeros((1, 14))
  rows_v[:, 12] = config["rho0"]
  stresses = []
  rhos = []
  for strain_total, time_new in path:
    strain_total = np.array(strain_total, dtype=np.float64)
    sigma_l, tangent_l, row_l = _legacy_step(legacy, strain_total, time_new)
    result = _v3_step(calibration, rows_v, strain_total, time_new)
    assert np.array_equal(result.stresses[0], sigma_l)
    assert np.array_equal(result.trial_rows[0], row_l)
    assert np.array_equal(tangent_l, result.tangents[0])  # dormant: bitwise
    coded, true, _ = _sovs_tangents(calibration, rows_v[0], time_new)
    assert np.array_equal(tangent_l, true)
    assert np.array_equal(result.tangents[0], true)
    assert np.array_equal(coded, true)  # dormancy: both forms underflow to C
    rows_v = result.trial_rows
    stresses.append(sigma_l)
    rhos.append(float(rows_v[0, 12]))
  return rows_v, stresses, rhos


def test_activated_free_sintering_matches_legacy_bitwise() -> None:
  # The survey's finding configuration: 20 free-sintering steps at
  # dtime = 0.01 densify rho0 = 0.6 to rho = 0.6000062 (pinned exact value).
  path = [(np.zeros(6), 0.01 * (k + 1)) for k in range(20)]
  rows_v, stresses, rhos = _run_path_parity(_ACTIVATED, path)
  assert rows_v[0, 12] == 0.6000061724180394
  # rho densifies strictly monotonically over the free-sintering path.
  assert all(later > earlier for earlier, later in zip(rhos, rhos[1:]))
  # Densification physics: viscous strain is negative isotropic (shrinkage —
  # shear slots exactly zero on a zero-stress free-sintering path), and a
  # small tensile reaction develops under the constraint.
  assert np.all(rows_v[0, 6:9] < 0.0)
  assert np.all(rows_v[0, 6:9] == rows_v[0, 6])
  assert np.all(rows_v[0, 9:12] == 0.0)
  assert float(stresses[-1][0]) > 0.0


def test_activated_pressure_leg_matches_legacy_bitwise() -> None:
  # Pressure-assisted sintering: after ten free steps, a compressive normal
  # strain ramp (sigma_m < 0) increases the driving stress and the creep.
  path = [(np.zeros(6), 0.01 * (k + 1)) for k in range(10)]
  path += [
    (np.array([-2.0e-4 * k, -1.0e-4 * k, 0.0, 0.0, 0.0, 0.0]), 0.1 + 0.01 * k)
    for k in range(1, 6)
  ]
  rows_v, stresses, _ = _run_path_parity(_ACTIVATED, path)
  assert rows_v[0, 12] > 0.600003  # densified beyond the free trajectory
  assert np.all(stresses[-1][:3] < 0.0)  # compressive reaction


def test_rho_guard_kink_matches_legacy_bitwise() -> None:
  # The eta guard kink (rho >= 0.999 -> 1e20): rho0 = 0.998 with a large
  # sintering stress crosses the guard mid-path — the first step runs the
  # Skorohod branch, later steps the plateau branch. Bitwise state parity AT
  # the kink; the tangent parity holds bitwise on both branches (repair
  # confirmed — the harness asserts it step by step).
  config = {**_ACTIVATED, "rho0": 0.998, "sigma_sint": 1.0e8}
  path = [(np.zeros(6), 0.01 * (k + 1)) for k in range(4)]
  rows_v, _, rhos = _run_path_parity(config, path)
  # The first step jumps past the guard and pins rho = 1.0 exactly (the
  # upper clamp); the plateau branch then holds rho in the guard regime —
  # within a few ulps of 1.0, not bitwise, since the plateau's tiny but
  # nonzero drho keeps rounding (the bitwise pin is the legacy parity above).
  assert rhos[0] == 1.0
  assert np.all(rows_v[:, 12] >= 0.999)
  assert np.all(rows_v[:, 12] <= 1.0)


def test_rho_guard_from_the_boundary_matches_legacy_bitwise() -> None:
  # rho0 = 0.999 exactly: the guard fires from the very first step; rho
  # drifts by ~1 ulp per step under the plateau's residual drho, so the
  # exact pin is the bitwise legacy parity and the guard-regime bound.
  config = {**_ACTIVATED, "rho0": 0.999}
  path = [(np.zeros(6), 0.01 * (k + 1)) for k in range(3)]
  rows_v, _, rhos = _run_path_parity(config, path)
  assert all(rho >= 0.999 for rho in rhos)
  assert np.all(np.abs(rows_v[:, 12] - 0.999) < 1.0e-12)


def test_rho_clamps_match_legacy_bitwise() -> None:
  # Upper clamp: a huge sintering stress drives rho + drho past 1.0; the
  # clamped rho_new = 1.0 is committed while the volumetric viscous strain
  # reads the UNCLAMPED drho (SOVS.py:297) — pinned bitwise, and the
  # committed viscous strain equals the unclamped-drho reconstruction.
  config = {**_ACTIVATED, "sigma_sint": 1.0e12}
  rows_v, _, _ = _run_path_parity(config, [(np.zeros(6), 0.01)])
  assert rows_v[0, 12] == 1.0
  eta_vol = skorohod_olevsky_calibration(*_parameters(config))[0] * (0.6**-2.0 - 1.0)
  drho_unclamped = (3.0 * 0.6 / (2.0 * eta_vol)) * (1.0e12 - 0.0) * 0.01
  assert 0.6 + drho_unclamped > 1.0  # the clamp actually fired
  np.testing.assert_array_equal(rows_v[0, 6:9], -drho_unclamped / 0.6 / 3.0)
  # Lower clamp: a tensile hydrostatic state (sigma_m > sigma_sint) drives
  # drho < 0 and the clamp pins rho back to rho0 exactly.
  config_t = {**_ACTIVATED, "sigma_sint": 1.0e3}
  rows_t, _, _ = _run_path_parity(
    config_t, [(np.array([0.01, 0.01, 0.01, 0.0, 0.0, 0.0]), 0.01)]
  )
  assert rows_t[0, 12] == _ACTIVATED["rho0"]


def test_activated_tangent_repair_confirmed_by_fd() -> None:
  # At the 20-step free-sintered state (rho = 0.6000062), probing at
  # dtime = 0.01 and dtime = 0.001: the divergence was pinned here — the
  # pre-repair coded legacy tangent (the implicit-step form) contradicted a
  # central FD of the legacy law's own response by 9.41e-3 relative (9.94e-4
  # at the smaller dtime — the O(dt) inconsistency at two scales), while the
  # v3 closed form matched an FD of the kernel's own response to ~3e-15 (the
  # map is affine in the trial strain at fixed state). Repair confirmed (M67
  # commit 30a5f4e, the structured three-leg tangent): the legacy tangent is
  # now bitwise the v3 tangent and matches the FD of its own response to the
  # same rounding class; the coded recomputation, retained as the witness,
  # still fails by the documented bounds — the probe states and the
  # divergence record are unchanged.
  config = _ACTIVATED
  calibration = skorohod_olevsky_calibration(*_parameters(config))
  legacy = _legacy_law(config)
  rows_v = np.zeros((1, 14))
  rows_v[:, 12] = config["rho0"]
  for k in range(20):
    time = 0.01 * (k + 1)
    _legacy_step(legacy, np.zeros(6), time)
    rows_v = _v3_step(calibration, rows_v, np.zeros(6), time).trial_rows
  assert rows_v[0, 12] == 0.6000061724180394

  for probe_time, coded_bound in ((0.21, 1.0e-3), (0.201, 1.0e-4)):
    _, tangent_l = _legacy_probe(legacy, np.zeros(6), probe_time)
    base = _v3_step(calibration, rows_v, np.zeros(6), probe_time)
    coded, true, active = _sovs_tangents(calibration, rows_v[0], probe_time)
    assert active
    assert np.array_equal(tangent_l, true)
    assert np.array_equal(base.tangents[0], true)
    assert np.array_equal(tangent_l, base.tangents[0])
    fd_v3 = _fd_tangent_v3(calibration, rows_v, np.zeros(6), probe_time)
    fd_l = _fd_tangent_legacy(legacy, np.zeros(6), probe_time)
    scale = float(np.max(np.abs(fd_v3)))
    np.testing.assert_allclose(fd_v3, fd_l, rtol=1.0e-9, atol=scale * 1.0e-9)
    # Repair confirmed: both tangents are the exact derivative of the shared
    # affine map (the pre-repair coded form failed this check materially;
    # M67 commit 30a5f4e).
    legacy_error = float(np.max(np.abs(fd_l - tangent_l))) / scale
    v3_error = float(np.max(np.abs(fd_v3 - base.tangents[0]))) / scale
    assert legacy_error < 1.0e-9  # measured 3.3e-15 class, as v3
    assert v3_error < 1.0e-9  # measured 3.3e-15
    # The divergence record, reproduced by the retained coded form at these
    # same states: the witness convicts the pre-repair form at both scales.
    coded_error = float(np.max(np.abs(fd_l - coded))) / scale
    assert coded_error > coded_bound  # measured 9.41e-3 / 9.94e-4
  # The shared tangent is exactly symmetric by construction.
  assert np.array_equal(base.tangents[0], base.tangents[0].T)


def test_kernel_tangent_is_the_exact_stress_map_derivative_by_fd() -> None:
  """Central FD of the kernel's own stress update vs the returned tangent.

  Seeded nonzero states at activated constants — virgin (dtime > 0: the
  viscous update is active from the first step), free-sintered, pressured,
  and a seeded random state — at dtime = 0.01 and dtime = 0.001. The
  implemented map is affine in the trial strain at every fixed committed
  state (the rho guard reads the committed rho; the clamps bind only the
  committed rho trajectory), so the central difference is exact up to
  rounding at every state, kink-adjacent or not — worst observed ~1e-13.
  """
  config = _ACTIVATED
  calibration = skorohod_olevsky_calibration(*_parameters(config))
  worst = 0.0

  def check(rows: np.ndarray, strain_total: np.ndarray, time_new: float) -> None:
    nonlocal worst
    base = _v3_step(calibration, rows, strain_total, time_new)
    fd = _fd_tangent_v3(calibration, rows, strain_total, time_new)
    tangent = base.tangents[0]
    scale = float(np.max(np.abs(tangent)))
    worst = max(worst, float(np.max(np.abs(fd - tangent))) / scale)
    assert float(np.max(np.abs(fd - fd.T))) / scale < 1.0e-9

  # Virgin state, both dtime scales.
  virgin = np.zeros((1, 14))
  virgin[:, 12] = config["rho0"]
  check(virgin, np.zeros(6), 0.01)
  check(virgin, np.zeros(6), 0.001)
  check(virgin, np.array([1.0e-4, 0.0, 0.0, 0.0, 0.0, 5.0e-5]), 0.01)
  # Free-sintered state.
  rows = virgin
  for k in range(20):
    rows = _v3_step(calibration, rows, np.zeros(6), 0.01 * (k + 1)).trial_rows
  check(rows, np.zeros(6), 0.21)
  check(rows, np.zeros(6), 0.201)
  # Pressured state.
  strained = _v3_step(
    calibration, rows, np.array([-2.0e-4, -1.0e-4, 0.0, 0.0, 0.0, 0.0]), 0.22
  ).trial_rows
  check(strained, np.array([-4.0e-4, -2.0e-4, 0.0, 0.0, 0.0, 0.0]), 0.23)
  # A seeded random state (fixed seed, deterministic path).
  rng = np.random.default_rng(20261009)
  rows = np.zeros((1, 14))
  rows[:, 12] = config["rho0"]
  strain = np.zeros(6)
  for step in range(3):
    strain = strain + rng.normal(size=6) * 2.0e-4
    rows = _v3_step(calibration, rows, strain, 0.05 * (step + 1)).trial_rows
  probe = strain + rng.normal(size=6) * 1.0e-4
  check(rows, probe, 0.20)
  assert worst < 1.0e-9, worst  # observed: ~1e-13 class over all states


def test_fd_conviction_holds_at_the_kink_states() -> None:
  """FD conviction AT the density kinks — the tower's spec-correction leg.

  The guard (rho >= 0.999 -> eta = 1e20) switches on the COMMITTED rho and
  the clamp binds only the committed rho trajectory (the stress reads the
  unclamped drho, SOVS.py:297), so the strain->stress map at every fixed
  committed state is affine — across the kinks included — and central FD is
  exact to rounding everywhere. No one-sided stencil is needed; the reason
  is documented here per the M26 skip-and-document precedent (which is
  deliberately not exercised).
  """
  calibration = skorohod_olevsky_calibration(*_parameters(_ACTIVATED))
  worst = 0.0

  def check(rows: np.ndarray, strain_total: np.ndarray, time_new: float) -> None:
    nonlocal worst
    base = _v3_step(calibration, rows, strain_total, time_new)
    fd = _fd_tangent_v3(calibration, rows, strain_total, time_new)
    scale = float(np.max(np.abs(base.tangents[0])))
    worst = max(worst, float(np.max(np.abs(fd - base.tangents[0]))) / scale)

  # A guard-regime committed state (rho = 1.0 exactly, plateau branch).
  guard = np.zeros((1, 14))
  guard[:, 12] = 1.0
  check(guard, np.zeros(6), 0.01)
  check(guard, np.array([1.0e-4, 0.0, 0.0, 0.0, 0.0, -5.0e-5]), 0.02)
  # A boundary-adjacent Skorohod-branch state (rho just below the guard).
  edge = np.zeros((1, 14))
  edge[:, 12] = 0.9985
  check(edge, np.zeros(6), 0.01)
  assert worst < 1.0e-9, worst

  # A clamp-firing step: committed rho = 0.6 and the huge sintering stress
  # drives rho + drho past 1.0 mid-evaluation. The tangent must still be the
  # exact derivative of the stress map (the map reads the unclamped drho).
  config = {**_ACTIVATED, "sigma_sint": 1.0e12}
  calibration_c = skorohod_olevsky_calibration(*_parameters(config))
  rows = np.zeros((1, 14))
  rows[:, 12] = 0.6
  base = _v3_step(calibration_c, rows, np.zeros(6), 0.01)
  assert base.trial_rows[0, 12] == 1.0  # the clamp fired on this step
  fd = _fd_tangent_v3(calibration_c, rows, np.zeros(6), 0.01)
  scale = float(np.max(np.abs(base.tangents[0])))
  assert float(np.max(np.abs(fd - base.tangents[0]))) / scale < 1.0e-9


def _symbolic_sovs_jacobian(
  calibration: np.ndarray,
  row: np.ndarray,
  dtime: float,
) -> object:
  """The SOVS explicit map differentiated symbolically in the 6-Voigt strain.

  The committed state enters as numeric constants (committed rho, viscous
  strain, calibration); the trial strain is symbolic. Returns the sympy
  Jacobian of the step's stress map (dtime > 0 branch of SOVS.py:284-306).
  """
  import sympy as sp

  eta_ref = calibration[0]
  rho0 = calibration[1]
  sigma_sint = calibration[2]
  n_vol = calibration[3]
  n_shear = calibration[4]
  e_base = calibration[5]
  rho_power = calibration[6]
  nu_eff = calibration[7]
  rho = float(row[12])
  if rho < 0.999:
    eta_vol = eta_ref * (rho ** (-n_vol) - 1.0)
    eta_shear = eta_ref * (rho ** (-n_shear) - 1.0)
  else:
    eta_vol = 1.0e20
    eta_shear = 1.0e20
  e_eff = e_base * ((rho / rho0) ** rho_power)
  ebulk3 = e_eff / (1.0 - 2.0 * nu_eff)
  eg2 = e_eff / (1.0 + nu_eff)
  eg = 0.5 * eg2
  elam = (ebulk3 - eg2) / 3.0
  ctang = np.zeros((6, 6))
  ctang[:3, :3] = elam
  ctang[0, 0] += eg2
  ctang[1, 1] = ctang[0, 0]
  ctang[2, 2] = ctang[0, 0]
  ctang[3, 3] = eg
  ctang[4, 4] = ctang[3, 3]
  ctang[5, 5] = ctang[3, 3]

  eps = sp.Matrix(sp.symbols("e0:6"))
  elastic = eps - sp.Matrix(row[6:12])
  trial = sp.Matrix(ctang) * elastic
  sigma_m = (trial[0] + trial[1] + trial[2]) / 3.0
  deviatoric = trial - sigma_m * sp.Matrix([1, 1, 1, 0, 0, 0])
  drho = (3.0 * rho / (2.0 * eta_vol)) * (sigma_sint - sigma_m) * dtime
  dstrain_visc = sp.Matrix(
    [-(drho / rho) / 3.0, -(drho / rho) / 3.0, -(drho / rho) / 3.0, 0, 0, 0]
  )
  dstrain_visc += deviatoric / (2.0 * eta_shear) * dtime
  sigma = sp.Matrix(ctang) * (elastic - dstrain_visc)
  return sigma.jacobian(eps)


def test_symbolic_jacobian_confirms_the_closed_form() -> None:
  """The sympy leg of the tangent conviction (the tower's SOVS instrument).

  FD convicts at sampled states; the symbolic Jacobian of the explicit map
  convicts the closed form STRUCTURALLY. Measured agreement of the kernel's
  three-leg tangent with the symbolic derivative: ~1e-16 relative (exact
  arithmetic structure; float evaluation of the same expressions). The
  survey's two-modulus isotropic reassembly disagrees with the same symbolic
  derivative by ~1.4e-2 — the tower's 2026-10-09 spec correction pinned at
  the full-matrix level.
  """
  calibration = skorohod_olevsky_calibration(*_parameters(_ACTIVATED))
  probe = np.array([1.0e-4, -2.0e-4, 3.0e-5, 1.0e-4, 0.0, 5.0e-5])
  states = []
  virgin = np.zeros((1, 14))
  virgin[:, 12] = _ACTIVATED["rho0"]
  states.append((virgin, 0.01))
  states.append((virgin, 0.001))
  # Free-sintered state with nonzero viscous strain.
  rows = virgin
  for k in range(20):
    rows = _v3_step(calibration, rows, np.zeros(6), 0.01 * (k + 1)).trial_rows
  states.append((rows, 0.01))
  # Pressured state (nonzero committed total and viscous strain).
  strained = _v3_step(
    calibration, rows, np.array([-2.0e-4, -1.0e-4, 0.0, 0.0, 0.0, 3.0e-5]), 0.22
  ).trial_rows
  states.append((strained, 0.01))
  # Guard-regime state (plateau branch).
  guard = np.zeros((1, 14))
  guard[:, 12] = 1.0
  states.append((guard, 0.01))

  worst = 0.0
  for state_rows, dtime in states:
    time_new = float(state_rows[0, 13]) + dtime
    symbolic = _symbolic_sovs_jacobian(calibration, state_rows[0], dtime)
    numeric = np.array(
      symbolic.subs({f"e{i}": float(probe[i]) for i in range(6)})
    ).astype(float)
    base = _v3_step(calibration, state_rows, probe, time_new)
    tangent = base.tangents[0]
    scale = float(np.max(np.abs(tangent)))
    worst = max(worst, float(np.max(np.abs(numeric - tangent))) / scale)
    assert float(np.max(np.abs(numeric - numeric.T))) / scale < 1.0e-12
  assert worst < 1.0e-9, worst  # observed: ~1e-16 class over all states

  # Conviction of the corrected sketch: the two-modulus isotropic reassembly
  # (K_alg, G_alg) is structurally NOT the map's derivative.
  rows = states[2][0]
  symbolic = _symbolic_sovs_jacobian(calibration, rows[0], 0.01)
  numeric = np.array(
    symbolic.subs({f"e{i}": float(probe[i]) for i in range(6)})
  ).astype(float)
  eta_ref, rho0, _, _, _, e_base, rho_power, nu_eff = calibration
  rho = float(rows[0, 12])
  eta_vol = eta_ref * (rho ** (-calibration[3]) - 1.0)
  eta_shear = eta_ref * (rho ** (-calibration[4]) - 1.0)
  e_eff = e_base * ((rho / rho0) ** rho_power)
  k_mod = e_eff / (1.0 - 2.0 * nu_eff) / 3.0
  g_mod = 0.5 * (e_eff / (1.0 + nu_eff))
  k_alg = k_mod * (1.0 - 3.0 * k_mod * 0.01 / (2.0 * eta_vol))
  g_alg = g_mod * (1.0 - g_mod * 0.01 / (2.0 * eta_shear))
  lam_iso = k_alg - 2.0 * g_alg / 3.0
  iso = np.zeros((6, 6))
  iso[:3, :3] = lam_iso
  iso[0, 0] += 2.0 * g_alg
  iso[1, 1] = iso[0, 0]
  iso[2, 2] = iso[0, 0]
  iso[3, 3] = g_alg
  iso[4, 4] = g_alg
  iso[5, 5] = g_alg
  scale = float(np.max(np.abs(numeric)))
  gap = float(np.max(np.abs(numeric - iso))) / scale
  assert gap > 1.0e-2  # measured 1.4e-2 — the corrected survey sketch


def test_kernel_reports_typed_failures() -> None:
  calibration = skorohod_olevsky_calibration(*_parameters(_ACTIVATED))
  rows = np.zeros((1, 14))
  rows[:, 12] = _ACTIVATED["rho0"]
  signals = (_time_signal(0.5),)
  # Non-finite strain batch.
  result = skorohod_olevsky_kernel(
    np.array([[np.inf, 0.0, 0.0, 0.0, 0.0, 0.0]]), rows, calibration, signals
  )
  assert result.status is EvaluationStatus.REJECT_STEP
  assert np.array_equal(result.trial_rows, rows)
  # Non-finite response from a finite but extreme strain (overflow in the
  # elastic stiffness product).
  result = skorohod_olevsky_kernel(
    np.array([[1.0e308, 0.0, 0.0, 0.0, 0.0, 0.0]]), rows, calibration, signals
  )
  assert result.status is EvaluationStatus.REJECT_STEP
  assert np.array_equal(result.trial_rows, rows)
  # Non-finite bound time.
  result = skorohod_olevsky_kernel(
    np.zeros((1, 6)), rows, calibration, (_time_signal(np.nan),)
  )
  assert result.status is EvaluationStatus.REJECT_STEP
  assert np.array_equal(result.trial_rows, rows)
  # Contract violations stay typed exceptions.
  with pytest.raises(TypeError, match="calibration"):
    skorohod_olevsky_kernel(np.zeros((1, 6)), rows, np.zeros(3), signals)
  with pytest.raises(ValueError, match="strains and"):
    skorohod_olevsky_kernel(np.zeros((1, 3)), rows, calibration, signals)
  with pytest.raises(TypeError, match="time port"):
    skorohod_olevsky_kernel(np.zeros((1, 6)), rows, calibration, ())
  wrong_port = StatefulContinuumSignalInput("load", np.array([0.5]), ())
  with pytest.raises(TypeError, match="time port"):
    skorohod_olevsky_kernel(np.zeros((1, 6)), rows, calibration, (wrong_port,))


def test_virgin_probe_returns_rows_byte_equal() -> None:
  calibration = skorohod_olevsky_calibration(*_parameters(_ACTIVATED))
  rows = np.zeros((1, 14))
  rows[:, 12] = _ACTIVATED["rho0"]
  result = skorohod_olevsky_kernel(
    np.zeros((1, 6)), rows, calibration, (_time_signal(0.0),)
  )
  assert result.status is EvaluationStatus.OK
  assert np.array_equal(result.trial_rows, rows)
  # The virgin tangent is the elastic stiffness at rho0 (rho/rho0 == 1):
  # E_eff = 100e9 exactly, nu = 0.25 — isotropic, finite, symmetric.
  eg = 0.5 * (100.0e9 / 1.25)
  ebulk3 = 100.0e9 / 0.5
  elam = (ebulk3 - 2.0 * eg) / 3.0
  expected = np.zeros((6, 6))
  expected[:3, :3] = elam
  expected[0, 0] += 2.0 * eg
  expected[1, 1] = expected[0, 0]
  expected[2, 2] = expected[0, 0]
  expected[3, 3] = eg
  expected[4, 4] = eg
  expected[5, 5] = eg
  assert np.array_equal(result.tangents[0], expected)


def test_calibration_validation_is_strict() -> None:
  p = _parameters(_ACTIVATED)
  with pytest.raises(TypeError, match="exact float"):
    skorohod_olevsky_calibration(1, *p[1:])
  with pytest.raises(ValueError, match="must be finite"):
    skorohod_olevsky_calibration(np.inf, *p[1:])
  with pytest.raises(ValueError, match="reference_viscosity must be positive"):
    skorohod_olevsky_calibration(0.0, *p[1:])
  with pytest.raises(ValueError, match="activation_energy must be positive"):
    skorohod_olevsky_calibration(p[0], -1.0, *p[2:])
  with pytest.raises(ValueError, match="temperature must be positive"):
    skorohod_olevsky_calibration(p[0], p[1], 0.0, *p[3:])
  with pytest.raises(ValueError, match="initial_relative_density"):
    skorohod_olevsky_calibration(p[0], p[1], p[2], 0.0, *p[4:])
  with pytest.raises(ValueError, match="initial_relative_density"):
    skorohod_olevsky_calibration(p[0], p[1], p[2], 1.5, *p[4:])
  with pytest.raises(ValueError, match="sintering_stress must be positive"):
    skorohod_olevsky_calibration(p[0], p[1], p[2], p[3], -1.0, *p[5:])
  with pytest.raises(ValueError, match="gas_constant must be positive"):
    skorohod_olevsky_calibration(p[0], p[1], p[2], p[3], p[4], 0.0, *p[6:])
  with pytest.raises(ValueError, match="viscosity_exponent_volumetric"):
    skorohod_olevsky_calibration(p[0], p[1], p[2], p[3], p[4], p[5], 0.0, p[7])
  with pytest.raises(ValueError, match="viscosity_exponent_shear"):
    skorohod_olevsky_calibration(p[0], p[1], p[2], p[3], p[4], p[5], p[6], -1.0)
  with pytest.raises(TypeError, match="exactly eight"):
    SKOROHOD_OLEVSKY_BINDING(*p[:7])
  # The Arrhenius reference viscosity is bound at calibration time with the
  # legacy expression order (SOVS.py:163).
  calibration = skorohod_olevsky_calibration(*_parameters(_DECK))
  assert calibration[0] == 1.0e12 * np.exp(5.0e5 / (8.314 * 1600.0))


def test_frozen_v2_schema_covers_the_law_without_new_fields() -> None:
  metadata = skorohod_olevsky_metadata()
  assert metadata["schema"] == STATEFUL_MATERIAL_DESCRIPTOR_SCHEMA
  assert STATEFUL_MATERIAL_DESCRIPTOR_SCHEMA == "pyfem-v3-material-descriptor-v2"
  # The M25/M29 freeze accepts the law as-is: fixed-width slots and the
  # optional signal_ports field are already in the frozen v2 field set
  # (contracts.py carries no diff for this mission).
  assert validate_stateful_material_metadata(metadata) is metadata
  slots = resolve_material_state_slots(
    metadata["state_slots"],
    dict(
      zip(
        metadata["parameter_names"],
        _parameters(_ACTIVATED),
      )
    ),
  )
  assert [slot.name for slot in slots] == ["strain", "strain_visc", "rho", "time"]
  assert [slot.width for slot in slots] == [6, 6, 1, 1]


def _sovs_model(config: dict = _ACTIVATED) -> ModelSpec:
  nodes = tuple(
    NodeSpec(
      id=index + 1,
      coordinates=point,
      source=_source(f"node-{index + 1}"),
    )
    for index, point in enumerate(_UNIT_COORDINATES)
  )
  cell = CellSpec(
    id="cell-1",
    node_ids=tuple(node.id for node in nodes),
    source=_source("cell"),
  )
  names = (
    "reference_viscosity",
    "activation_energy",
    "temperature",
    "initial_relative_density",
    "sintering_stress",
    "gas_constant",
    "viscosity_exponent_volumetric",
    "viscosity_exponent_shear",
  )
  return ModelSpec(
    mesh=MeshSpec(
      nodes=nodes,
      cell_blocks=(
        CellBlockSpec(
          id="cells",
          reference_topology="quadrilateral",
          topological_dimension=2,
          embedding_dimension=2,
          geometry_interpolation="serendipity-quad8",
          cells=(cell,),
          source=_source("block"),
        ),
      ),
      source=_source("mesh"),
    ),
    fields=(
      FieldSpec(
        id="displacement",
        components=("x", "y"),
        location="node",
        source=_source("field"),
      ),
    ),
    materials=(
      MaterialSpec(
        id="ceramic",
        model="skorohod-olevsky",
        parameters=tuple(
          MaterialParameterSpec(name, value)
          for name, value in zip(names, _parameters(config))
        ),
        source=_source("material"),
      ),
    ),
    regions=(
      RegionSpec(
        id="domain",
        cell_refs=(CellRef("cells", "cell-1"),),
        field_ids=("displacement",),
        material_id="ceramic",
        formulation="small-strain-continuum",
        quadrature="gauss-3x3",
        source=_source("region"),
      ),
    ),
    source=_source("model"),
  )


def _elastic_model() -> ModelSpec:
  """The same one-element mesh with the port-free plane-stress linear law."""
  base = _sovs_model()
  return ModelSpec(
    mesh=base.mesh,
    fields=base.fields,
    materials=(
      MaterialSpec(
        id="steel",
        model="plane-stress-linear-elastic",
        parameters=(
          MaterialParameterSpec("youngs_modulus", 2.0e5),
          MaterialParameterSpec("poisson_ratio", 0.3),
        ),
        source=_source("material"),
      ),
    ),
    regions=base.regions[:-1]
    + (
      RegionSpec(
        id="domain",
        cell_refs=(CellRef("cells", "cell-1"),),
        field_ids=("displacement",),
        material_id="steel",
        formulation="small-strain-continuum",
        quadrature="gauss-3x3",
        source=_source("region"),
      ),
    ),
    source=_source("model"),
  )


def _compiled(model: ModelSpec, registry: dict) -> CompiledSystem:
  return compile_system(model, registry)


def _sovs_compiled(config: dict = _ACTIVATED) -> CompiledSystem:
  return _compiled(_sovs_model(config), sovs_reference_registry())


def test_compiled_state_layout_and_fingerprint() -> None:
  layout = _sovs_compiled().operators[0].header.state_layout
  assert layout.entity_count == 9
  assert layout.row_width == 14
  assert layout.schema == (
    "pyfem-v3-skorohod-olevsky-state-v1|strain:6,strain_visc:6,rho:1,time:1"
  )
  # The first parameter-dependent initial state of the stateful family:
  # every initial row carries rho = rho0 in the rho slot (M25 G2 binding).
  assert layout.initial_rows is not None
  initial = layout.initial_rows.values
  assert initial.shape == (9, 14)
  assert np.all(initial[:, :12] == 0.0)
  assert np.all(initial[:, 12] == 0.6)
  assert np.all(initial[:, 13] == 0.0)
  # Deterministic content identity across compilations.
  assert _sovs_compiled().content_fingerprint == _sovs_compiled().content_fingerprint


def test_initial_state_binding_carries_rho0_and_is_validated() -> None:
  layout = _sovs_compiled().operators[0].header.state_layout
  parameters = _parameters(_ACTIVATED)
  rows = skorohod_olevsky_initial_state(parameters, layout)
  assert rows.shape == layout.row_shape
  assert np.all(rows[:, 12] == 0.6)
  assert np.all(rows[:, :12] == 0.0)
  # A different rho0 binds a different initial row — the binding reads the
  # parameters, not a constant.
  denser = parameters[:3] + (0.95,) + parameters[4:]
  rows = skorohod_olevsky_initial_state(denser, layout)
  assert np.all(rows[:, 12] == 0.95)
  with pytest.raises(TypeError, match="eight parameters"):
    skorohod_olevsky_initial_state(parameters[:7], layout)
  with pytest.raises(ValueError, match="row layout"):
    skorohod_olevsky_initial_state(
      parameters,
      type("Layout", (), {"row_width": 13, "row_shape": layout.row_shape})(),
    )
  with pytest.raises(ValueError, match="initial_relative_density"):
    skorohod_olevsky_initial_state(parameters[:3] + (1.5,) + parameters[4:], layout)


def _decode_node(node: object) -> object:
  if type(node) is not list or not node:
    return node
  tag = node[0]
  if tag == "mapping":
    return {pair[0]: _decode_node(pair[1]) for pair in node[1]}
  if tag == "sequence":
    return [_decode_node(item) for item in node[1]]
  if tag == "str":
    return node[1]
  return node


def _manifest_content(manifest: CanonicalManifest) -> dict[str, object]:
  payload = manifest.to_bytes().split(b"\n", 1)[1]
  node = json.loads(payload)
  if type(node) is not list or not node or node[0] != "mapping":
    msg = "compiled manifests must decode to a canonical mapping"
    raise TypeError(msg)
  return {pair[0]: pair[1] for pair in node[1]}


def test_compiled_operator_versions_ports_state_and_manifest() -> None:
  operator = _sovs_compiled().operators[0]
  header = operator.header
  assert len(header.signal_ports) == 1
  port = header.signal_ports[0]
  assert port.port_id == "time"
  assert port.signal_id == "time"
  assert port.derivative_coordinate_ids == ("time",)
  (channel,) = header.jacobian_channels
  assert channel.linear is False
  assert channel.symmetric is True
  manifest = _manifest_content(operator.content_manifest)
  assert _decode_node(manifest["signal_ports"]) == [
    {
      "port_id": "time",
      "signal_id": "time",
      "derivative_coordinate_ids": ["time"],
    }
  ]
  state = _decode_node(manifest["state"])
  assert state["schema"] == (
    "pyfem-v3-skorohod-olevsky-state-v1|strain:6,strain_visc:6,rho:1,time:1"
  )
  assert [slot["name"] for slot in state["slots"]] == [
    "strain",
    "strain_visc",
    "rho",
    "time",
  ]


def test_compiled_operator_binds_the_time_port_exactly() -> None:
  operator = _sovs_compiled().operators[0]
  values = FinalizedArray(np.zeros((1, 16)), dtype=np.float64)
  accepted = FinalizedArray(
    np.array(operator.header.state_layout.initial_rows.values, copy=True),
    dtype=np.float64,
  )
  request = ChannelRequest(("internal-force",), ("material-tangent",))

  def evaluate(signals: tuple) -> object:
    return operator.evaluate(
      OperatorEvaluationInput(
        port_values=(values,),
        accepted_state=accepted,
        signals=signals,
        request=request,
      )
    )

  result = evaluate((_program_time_signal(0.25),))
  assert result.status is EvaluationStatus.OK
  assert result.trial_state.values.shape == (9, 14)
  np.testing.assert_array_equal(result.trial_state.values[:, 13], 0.25)
  # The dtime > 0 step densified from rho0 at every integration point.
  assert np.all(result.trial_state.values[:, 12] > 0.6)
  with pytest.raises(ValueError, match="missing a declared program signal port"):
    evaluate(())
  with pytest.raises(ValueError, match="undeclared program signal port"):
    evaluate(
      (
        _program_time_signal(0.25),
        ProgramSignalInput(
          port_id="temperature",
          values=FinalizedArray(np.array([1.0], dtype=np.float64), dtype=np.float64),
          derivatives=(),
        ),
      )
    )


def _ramp_driver(model: ModelSpec, registry: dict) -> NonlinearStaticDriver:
  """One Q8 element under a prescribed affine eps_xx ramp, one free DOF.

  All nodal displacements follow the homogeneous strain field exactly
  (``u_x = eps * x``, ``u_y = 0``) except node 6's x component, which stays
  free so every Newton iteration assembles and factorizes the tangent. The
  program declares the schedule-owned ``time`` coordinate alongside ``eps``.
  The tolerance is 1e-6 absolute: the law's hard-coded E = 100 GPa modulus
  puts assembled internal forces at ~1e7 on the pressure steps, so the
  reduced residual's cancellation floor (~1e-9) sits above the 1e-10 default
  (there is no external load vector to scale the tolerance against). The
  bitwise oracle comparison is unaffected — it reads the committed strains.
  """
  system = _compiled(model, registry)
  constraints = []
  for index, node in enumerate(_UNIT_COORDINATES):
    node_id = index + 1
    if node_id != 6:
      constraints.append(
        PrescribedDofSpec(
          id=f"ux-{node_id}",
          target=DofRef(
            node_id=node_id,
            field_id="displacement",
            component="x",
          ),
          value=AffineValueSpec(
            coefficients=(AffineCoefficientSpec("eps", node[0]),),
          ),
          source=_source(f"ux-{node_id}"),
        )
      )
    constraints.append(
      PrescribedDofSpec(
        id=f"uy-{node_id}",
        target=DofRef(node_id=node_id, field_id="displacement", component="y"),
        value=AffineValueSpec(constant=0.0),
        source=_source(f"uy-{node_id}"),
      )
    )
  coordinate_map = compile_constraint_map(
    system,
    constraints=tuple(constraints),
    coordinates=(
      ProgramCoordinateSpec(name="time", kind="time"),
      ProgramCoordinateSpec(name="eps", kind="load"),
    ),
  )
  return NonlinearStaticDriver(
    system, coordinate_map, (), NonlinearStaticSettings(tolerance=1.0e-6)
  )


def _sovs_driver() -> NonlinearStaticDriver:
  return _ramp_driver(_sovs_model(), sovs_reference_registry())


def _point(time: float, eps: float) -> ProgramPoint:
  return ProgramPoint(
    (
      ProgramCoordinateValue("time", time),
      ProgramCoordinateValue("eps", eps),
    )
  )


def _committed_ip_strains(driver: NonlinearStaticDriver) -> np.ndarray:
  """Recompute the committed per-integration-point 6-Voigt strains.

  Uses the operator's own physical strain-displacement expression and the
  committed coefficient vector, mirroring the evaluation's arithmetic exactly.
  """
  operator = driver.owner.system.operators[0]
  payload = operator.payload
  b_matrix = (
    payload.normalized_strain_displacement.values
    / payload.geometry_scales.values[:, None, None, None]
  )
  gather = operator.header.ports[0].coefficient_map.values
  values = driver.owner.accepted_physical().values
  strain3 = np.einsum("epai,ei->epa", b_matrix, values[gather], optimize=True)
  strains = np.zeros((np.prod(strain3.shape[:2]), 6), dtype=np.float64)
  flat = strain3.reshape(-1, 3)
  strains[:, 0] = flat[:, 0]
  strains[:, 1] = flat[:, 1]
  strains[:, 5] = flat[:, 2]
  return strains


def test_plan_forwards_the_identity_time_signal() -> None:
  driver = _sovs_driver()
  plan = driver.plan
  assert plan.constant_tangent is False
  assert not plan.requires_committed_point
  assert len(plan.signal_slices) == 1 and len(plan.signal_slices[0]) == 1
  port = plan.signal_slices[0][0]
  assert port.port_id == "time"
  assert port.signal_id == "time"
  assert port.binding == "identity"
  names = plan.coordinate_names
  assert port.signal_coordinate_index == names.index("time")
  assert port.derivative_coordinate_indices == (names.index("time"),)
  forwarded = evaluate_signals(plan, _point(0.3, 2.0e-4))
  signal = forwarded[0][0]
  np.testing.assert_array_equal(signal.values.values, [0.3])
  assert tuple(item.coordinate_id for item in signal.derivatives) == ("time",)
  np.testing.assert_array_equal(
    [item.values.values[0] for item in signal.derivatives], [1.0]
  )
  signals = _decode_node(_manifest_content(plan.provenance.manifest)["signals"])
  assert signals[0]["ports"] == [
    {"port_id": "time", "signal_coordinate": "time", "derivative_coordinates": ["time"]}
  ]


def test_driver_time_schedule_commits_oracle_state_and_records_time() -> None:
  driver = _sovs_driver()
  layout = driver.owner.system.operators[0].header.state_layout
  block_id = layout.block_id
  calibration = skorohod_olevsky_calibration(*_parameters(_ACTIVATED))
  # Free sintering under full constraint (eps = 0, time advancing), then a
  # pressure-assisted step. The committed rows equal the kernel oracle
  # stepped on the committed integration-point strain path — bitwise.
  oracle_rows = np.zeros((9, 14))
  oracle_rows[:, 12] = _ACTIVATED["rho0"]
  schedule = (
    (0.01, 0.0),
    (0.02, 0.0),
    (0.03, 0.0),
    (0.04, -2.0e-4),
    (0.05, -4.0e-4),
  )
  base = _point(0.0, 0.0)
  previous_rho = None
  for step, (time, eps) in enumerate(schedule, 1):
    result = driver.run(base_point=base, target_points=(_point(time, eps),))
    assert result.status is DriverStatus.COMPLETED
    base = _point(time, eps)
    oracle = skorohod_olevsky_kernel(
      _committed_ip_strains(driver), oracle_rows, calibration, (_time_signal(time),)
    )
    assert oracle.status is EvaluationStatus.OK
    oracle_rows = oracle.trial_rows
    rows = driver.owner.accepted_state(block_id).values
    assert rows.shape == (9, 14)
    np.testing.assert_array_equal(rows, oracle_rows)
    # State rows record the substep's bound time at every integration point,
    # and rho densifies monotonically from rho0.
    np.testing.assert_array_equal(rows[:, 13], time)
    assert np.all(rows[:, 12] >= _ACTIVATED["rho0"])
    if previous_rho is not None:
      assert np.all(rows[:, 12] > previous_rho)
    previous_rho = rows[:, 12]
    assert result.final_generation.ordinal == step
  statistics = result.statistics
  assert statistics.committed_substep_count == 5
  assert statistics.rejected_substep_count == 0
  # Factorization honesty: the algorithmic-symmetric channel is not linear,
  # so the driver re-assembles and re-factorizes every Newton iteration.
  assert statistics.factorization_reuse_count == 0
  assert statistics.factorization_count == statistics.tangent_refill_count
  # Homogeneous state: all nine integration points agree to solver tolerance.
  np.testing.assert_allclose(
    rows, np.broadcast_to(rows[0], rows.shape), rtol=1.0e-9, atol=1.0e-10
  )


def test_driver_response_depends_on_the_time_schedule() -> None:
  # Positive control: SOVS is genuinely rate-dependent — a slower schedule
  # densifies more at the same strain path.
  first = _sovs_driver()
  second = _sovs_driver()
  block_id = first.owner.system.operators[0].header.state_layout.block_id
  first_base = _point(0.0, 0.0)
  second_base = _point(0.0, 0.0)
  for first_time, second_time, eps in (
    (0.01, 0.10, 0.0),
    (0.02, 0.30, 0.0),
    (0.03, 1.00, 0.0),
  ):
    first_result = first.run(
      base_point=first_base, target_points=(_point(first_time, eps),)
    )
    second_result = second.run(
      base_point=second_base, target_points=(_point(second_time, eps),)
    )
    assert first_result.status is DriverStatus.COMPLETED
    assert second_result.status is DriverStatus.COMPLETED
    first_base = _point(first_time, eps)
    second_base = _point(second_time, eps)
  first_rows = first.owner.accepted_state(block_id).values
  second_rows = second.owner.accepted_state(block_id).values
  assert np.all(second_rows[:, 12] > first_rows[:, 12])  # slower: denser
  assert first_rows.tobytes() != second_rows.tobytes()
  assert (
    first_result.records[-1].observation.reactions.values.tobytes()
    != second_result.records[-1].observation.reactions.values.tobytes()
  )


def test_port_free_law_results_are_signal_independent() -> None:
  first = _ramp_driver(_elastic_model(), q8_reference_registry())
  second = _ramp_driver(_elastic_model(), q8_reference_registry())
  eps_path = (2.0e-4, 5.0e-4, 1.1e-3)
  first_base = _point(0.0, 0.0)
  second_base = _point(0.0, 0.0)
  for first_time, second_time, eps in (
    (0.0, 3.0, eps_path[0]),
    (0.0, 7.5, eps_path[1]),
    (0.0, 11.25, eps_path[2]),
  ):
    first_result = first.run(
      base_point=first_base, target_points=(_point(first_time, eps),)
    )
    second_result = second.run(
      base_point=second_base, target_points=(_point(second_time, eps),)
    )
    assert first_result.status is DriverStatus.COMPLETED
    assert second_result.status is DriverStatus.COMPLETED
    assert (
      first.owner.accepted_physical().values.tobytes()
      == second.owner.accepted_physical().values.tobytes()
    )
    assert (
      first_result.records[-1].observation.reactions.values.tobytes()
      == second_result.records[-1].observation.reactions.values.tobytes()
    )
    first_base = _point(first_time, eps)
    second_base = _point(second_time, eps)


def test_state_codec_round_trips_and_rejects_foreign_schemas() -> None:
  driver = _sovs_driver()
  result = driver.run(
    base_point=_point(0.0, 0.0), target_points=(_point(0.05, -4.0e-4),)
  )
  assert result.status is DriverStatus.COMPLETED
  owner = driver.owner
  layout = owner.system.operators[0].header.state_layout
  payload = owner.encode_state(layout.block_id)
  decoded = owner.decode_state(layout.block_id, payload)
  np.testing.assert_array_equal(
    decoded.values, owner.accepted_state(layout.block_id).values
  )
  with pytest.raises(StateCodecError):
    owner.decode_state(layout.block_id, payload.replace(b"rho:1", b"rho:2"))
