# SPDX-License-Identifier: MIT

"""Perzyna-branded viscoplasticity: legacy parity oracles and the time-port path.

The legacy law ``pyfem/materials/ViscoPlasticity.py`` is the numerical
reference (AGENTS.md). The parity harness drives it with the
``dstrain = eps_total - (eelas + eplas)`` identity the v3 kernel uses — the
same float64 subtraction — and sets ``solverStat.time`` per committed step,
the hidden channel the v3 law replaces with the declared identity signal
port. Because the kernel replicates the legacy ``getStress`` arithmetic
statement for statement and both sides run identical NumPy operations in
identical order, every law-level stress/state comparison is bitwise on every
platform.

Semantics pinned loudly (finding
20261009-agent-vp1-bug-viscoplasticity-stress-update-is-rate-independent-j2-perzyna):
the legacy law is Perzyna-branded but integrates the RATE-INDEPENDENT J2
consistency equation — gamma, n, and dtime seed only the discarded Newton
initial guess, so the converged stress is invariant to them (measured
1.4e-16 relative between gamma = 1e-4, n = 1 and gamma = 1e2, n = 2 over the
documented plastic ramp; the gamma-invariance leg asserts <= 1e-8). The ONLY
rate effect is the ``dtime > 0`` gate (ViscoPlasticity.py:205): a supra-yield
step at constant or backward time returns the elastic trial response above
the yield surface — pinned as oracle behavior here and end-to-end through
the driver. The legacy docstring's backward-Euler/Perzyna claims and the
example README's rate-dependence promises
(examples/materials/viscoplasticity/README.md:39-55) are false for the
implemented map; a true-Perzyna rate law is a deferred v2 feature, not this
migration.

The tangent is the pinned divergence of this mission (finding
20261009-agent-vp1-bug-viscoplasticity-tangent-inconsistent-with-its-own-stress-upd):
the legacy coded tangent adds a spurious ``rate_factor`` (:270) — the
derivative of the discarded initial guess, not of the converged map. The v3
kernel writes the TRUE rate-factor-free J2 consistent tangent. This battery
pins the divergence by mechanism against legacy AT THE FORK BASE: the v3
tangent reproduces the rate-factor-free recomputation bit for bit, the legacy
oracle reproduces the coded recomputation bit for bit, and central finite
differences of each law's own stress response convict the sides (v3 exact to
~1.6e-10; legacy off by 7.0e-2 relative at the amplified state gamma = 1e4,
dtime = 1e3). At deck constants the divergence is dormant (~2e-11 relative of
max|tang| at dtime ~ 1) — pinned by the dormancy band in the ramp legs. The
L3 mission (M67) repairs the legacy side in parallel; when it lands, the
divergence pins flip to parity-where-repaired by whoever integrates second
(M54/M55 precedent — this module is the flip target).

Documented oracle configuration: the shipped deck
(examples/materials/viscoplasticity/bar_tension.pro): E = 2e5, nu = 0.3,
syield = 250, hard = 1000, gamma = 1e-3, n = 1. Documented paths (total
6-Voigt, engineering shear, per committed step): an elastic ramp eps_xx =
0.0002, 0.0006, 0.0010; a uniaxial plastic ramp of ten equal steps eps_xx =
0.0002 .. 0.004; a mixed segment holding eps_xx = 0.004 while gamma_xy ramps
0.001 .. 0.006 in six steps; an unload-reload pair eps_xx = 0.0025 then
0.0055; and constant-time/backward-time steps exercising the dtime gate.

The driver battery proves schedule-owned time flows through the M29 identity
signal port: committed state rows equal the kernel oracle stepped on the
committed integration-point strains (bitwise), the ``time`` slot records the
substep's bound time at every integration point, a constant-time supra-yield
substep changes the committed response through the gate (positive control),
two all-advancing schedules give identical stress to solver tolerance
(rate-independence end-to-end), and a port-free law driven under two
different time schedules yields byte-identical results (isolation). The ABI
proof is the frozen validator accepting the law's v2 metadata unchanged plus
``pyfem/v3/compile/contracts.py`` carrying no diff for this mission.
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

from pyfem.materials.ViscoPlasticity import ViscoPlasticity
from pyfem.v3.compile.continuum import (
  q8_reference_registry,
  viscoplasticity_reference_registry,
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
  evaluate_signals,
)
from pyfem.v3.materials.perzyna_viscoplasticity import (
  PERZYNA_VISCOPLASTICITY_BINDING,
  perzyna_viscoplasticity_calibration,
  perzyna_viscoplasticity_initial_state,
  perzyna_viscoplasticity_kernel,
  perzyna_viscoplasticity_metadata,
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

# Documented oracle configuration: the shipped deck
# (examples/materials/viscoplasticity/bar_tension.pro).
_E = 2.0e5
_NU = 0.3
_SYIELD = 250.0
_HARD = 1000.0
_GAMMA = 1.0e-3
_N = 1.0
_YIELD_STRAIN = _SYIELD * (1.0 + _NU) / _E  # plane-strain deviatoric bound
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


class _LegacyProps:
  """Minimal iterable props stand-in for ``BaseMaterial``."""

  def __init__(self, **values: object) -> None:
    self._values = values
    self.solverStat = None
    for name, value in values.items():
      setattr(self, name, value)

  def __iter__(self) -> object:
    return iter(self._values.items())


def _legacy_law(gamma: float = _GAMMA, n: float = _N) -> ViscoPlasticity:
  """A fresh legacy oracle (the ctor's ``print(self)`` is redirected away)."""
  with contextlib.redirect_stdout(io.StringIO()):
    with warnings.catch_warnings(action="ignore", category=DeprecationWarning):
      return ViscoPlasticity(
        _LegacyProps(E=_E, nu=_NU, syield=_SYIELD, hard=_HARD, gamma=gamma, n=n)
      )


def _legacy_row(mat: ViscoPlasticity) -> np.ndarray:
  return np.concatenate(
    (
      np.atleast_1d(mat.getHistoryParameter("eelas")),
      np.atleast_1d(mat.getHistoryParameter("eplas")),
      np.atleast_1d(mat.getHistoryParameter("eqplas")),
      np.atleast_1d(mat.getHistoryParameter("time_old")),
    )
  )


def _legacy_step(
  mat: ViscoPlasticity,
  strain_total: np.ndarray,
  time_new: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Advance the legacy oracle one committed step on the total-strain path."""
  eelas = mat.getHistoryParameter("eelas")
  eplas = mat.getHistoryParameter("eplas")
  dstrain = np.array(strain_total - (eelas + eplas), copy=True)
  mat.solverStat = SimpleNamespace(time=time_new)
  with contextlib.redirect_stdout(io.StringIO()):
    with warnings.catch_warnings(action="ignore", category=DeprecationWarning):
      sigma, tangent = mat.getStress(SimpleNamespace(dstrain=dstrain))
  mat.commitHistory()
  return np.array(sigma, copy=True), np.array(tangent, copy=True), _legacy_row(mat)


def _legacy_probe(
  mat: ViscoPlasticity,
  strain_total: np.ndarray,
  time_new: float,
) -> tuple[np.ndarray, np.ndarray]:
  """Evaluate the legacy oracle without committing (finite-difference probe)."""
  eelas = mat.getHistoryParameter("eelas")
  eplas = mat.getHistoryParameter("eplas")
  dstrain = np.array(strain_total - (eelas + eplas), copy=True)
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
  result = perzyna_viscoplasticity_kernel(
    strain_total[None, :], rows, calibration, (_time_signal(time_new),)
  )
  assert result.status is EvaluationStatus.OK
  return result


def _von_mises(sigma: np.ndarray) -> float:
  smises = (
    (sigma[0] - sigma[1]) * (sigma[0] - sigma[1])
    + (sigma[1] - sigma[2]) * (sigma[1] - sigma[2])
    + (sigma[2] - sigma[0]) * (sigma[2] - sigma[0])
  )
  smises += 6.0 * np.dot(sigma[3:], sigma[3:])
  return float(np.sqrt(0.5 * smises))


def _map_tangents(
  calibration: np.ndarray,
  row: np.ndarray,
  strain_total: np.ndarray,
  time_new: float,
) -> tuple[np.ndarray, np.ndarray, bool]:
  """Replicate the tangent construction, coded and rate-factor-free.

  Recomputes the return map from the committed row with the kernel's own
  statement order, then assembles both tangent forms on the shared block
  base: the legacy coded form carrying the spurious ``rate_factor``
  (bitwise-equal to the legacy oracle's returned tangent) and the true
  rate-factor-free J2 form (bitwise-equal to the v3 kernel's tangent).
  Returns ``plastic=False`` with two elastic-tangent copies on gated or
  elastic steps.
  """
  eg = calibration[0]
  eg3 = calibration[2]
  ebulk3 = calibration[3]
  syield = calibration[5]
  hard = calibration[6]
  gamma = calibration[7]
  n = calibration[8]
  tolerance = calibration[9]
  ctang = calibration[10:46].reshape(6, 6)
  dtime = time_new - row[13]
  dstrain = strain_total - (row[0:6] + row[6:12])
  eelas_trial = row[0:6] + dstrain
  sigma_trial = np.dot(ctang, eelas_trial)
  smises = _von_mises(sigma_trial)
  syield_current = syield + hard * row[12]
  if not (smises > syield_current and dtime > 0.0):
    return np.array(ctang, copy=True), np.array(ctang, copy=True), False
  overstress = (smises - syield_current) / syield_current
  gamma_eff = gamma * (overstress**n)
  deqpl = gamma_eff * dtime
  shydro = 0.333333333333333 * (sigma_trial[0] + sigma_trial[1] + sigma_trial[2])
  flow = np.array(sigma_trial, copy=True)
  flow[:3] = flow[:3] - shydro
  flow *= 1.0 / smises
  eqplas = float(row[12])
  converged = False
  for _ in range(20):
    syield_iter = syield + hard * (eqplas + deqpl)
    residual = smises - eg3 * deqpl - syield_iter
    if abs(residual) < tolerance * syield:
      converged = True
      break
    deqpl += -residual / (-eg3 - hard)
  assert converged
  syield_final = syield + hard * (eqplas + deqpl)
  effg = eg * syield_final / smises
  effg2 = 2.0 * effg
  effg3 = 3.0 * effg
  efflam = (ebulk3 - effg2) / 3.0
  base = np.zeros(shape=(6, 6))
  base[:3, :3] = efflam
  for i in range(3):
    base[i, i] += effg2
    base[i + 3, i + 3] += effg
  rate_factor = gamma * n * (overstress ** (n - 1.0)) * dtime / syield_current
  effhdr_coded = eg3 * (hard + rate_factor) / (eg3 + hard + rate_factor) - effg3
  effhdr_true = eg3 * hard / (eg3 + hard) - effg3
  coded = base + effhdr_coded * np.outer(flow, flow)
  true = base + effhdr_true * np.outer(flow, flow)
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
  mat: ViscoPlasticity,
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


def _elastic_path() -> list[np.ndarray]:
  return [np.array([eps, 0.0, 0.0, 0.0, 0.0, 0.0]) for eps in (0.0002, 0.0006, 0.0010)]


def _uniaxial_path() -> list[np.ndarray]:
  return [
    np.array([eps, 0.0, 0.0, 0.0, 0.0, 0.0]) for eps in np.linspace(0.0002, 0.004, 10)
  ]


def _shear_path() -> list[np.ndarray]:
  return [
    np.array([0.004, 0.0, 0.0, 0.0, 0.0, gamma])
    for gamma in np.linspace(0.001, 0.006, 6)
  ]


def _ramped(path: list[np.ndarray], t0: float, dt: float) -> list:
  return [(strain, t0 + dt * (index + 1)) for index, strain in enumerate(path)]


def _run_path_parity(
  path: list[tuple[np.ndarray, float]],
  *,
  gamma: float = _GAMMA,
  n: float = _N,
  max_tangent_gap: float = 1.0e-9,
) -> tuple[np.ndarray, list[np.ndarray], bool]:
  """Step both laws along one (total-strain, time) path; bitwise everywhere.

  Stress and state are bitwise-identical on every step. The tangent is
  bitwise on elastic and gated steps; on plastic steps the divergence is
  pinned by mechanism — the legacy oracle reproduces the coded recomputation
  bit for bit, the v3 kernel reproduces the rate-factor-free recomputation
  bit for bit, and the relative gap sits inside ``max_tangent_gap`` of
  max|tang|. The default is the documented dormancy band of the deck
  constants (measured ~2e-11 at dtime ~ 1); amplified fluidities carry an
  active rate_factor and pass a looser bound.
  """
  calibration = perzyna_viscoplasticity_calibration(_E, _NU, _SYIELD, _HARD, gamma, n)
  legacy = _legacy_law(gamma, n)
  rows_v = np.zeros((1, 14))
  stresses = []
  went_plastic = False
  for strain_total, time_new in path:
    strain_total = np.array(strain_total, dtype=np.float64)
    sigma_l, tangent_l, row_l = _legacy_step(legacy, strain_total, time_new)
    result = _v3_step(calibration, rows_v, strain_total, time_new)
    assert np.array_equal(result.stresses[0], sigma_l)
    assert np.array_equal(result.trial_rows[0], row_l)
    coded, true, plastic = _map_tangents(calibration, rows_v[0], strain_total, time_new)
    assert plastic == bool(result.trial_rows[0, 12] > rows_v[0, 12])
    assert np.array_equal(tangent_l, coded)
    assert np.array_equal(result.tangents[0], true)
    if plastic:
      went_plastic = True
      gap = float(np.max(np.abs(tangent_l - result.tangents[0])))
      scale = float(np.max(np.abs(result.tangents[0])))
      assert 0.0 < gap / scale < max_tangent_gap
    else:
      assert np.array_equal(tangent_l, result.tangents[0])
    rows_v = result.trial_rows
    stresses.append(sigma_l)
  return rows_v, stresses, went_plastic


def test_elastic_ramp_matches_legacy_bitwise() -> None:
  calibration = perzyna_viscoplasticity_calibration(_E, _NU, _SYIELD, _HARD, _GAMMA, _N)
  elastic_tangent = calibration[10:46].reshape(6, 6)
  for strain in _elastic_path():
    assert strain[0] < _YIELD_STRAIN
  rows, _, went_plastic = _run_path_parity(_ramped(_elastic_path(), 0.0, 0.3))
  assert not went_plastic
  assert rows[0, 12] == 0.0
  result = _v3_step(calibration, np.zeros((1, 14)), _elastic_path()[0], 0.3)
  assert np.array_equal(result.tangents[0], elastic_tangent)


def test_uniaxial_plastic_ramp_matches_legacy_bitwise() -> None:
  rows, _, went_plastic = _run_path_parity(_ramped(_uniaxial_path(), 0.0, 0.1))
  assert went_plastic
  assert rows[0, 12] > 0.0


def test_mixed_shear_after_plastic_matches_legacy_bitwise() -> None:
  path = [*_ramped(_uniaxial_path(), 0.0, 0.1), *_ramped(_shear_path(), 1.0, 0.1)]
  rows, _, went_plastic = _run_path_parity(path)
  assert went_plastic
  # True shear plastic flow is real state on both sides.
  assert np.any(rows[0, 9:12] != 0.0)


def test_unload_reload_keeps_kappa_and_returns_to_the_surface() -> None:
  calibration = perzyna_viscoplasticity_calibration(_E, _NU, _SYIELD, _HARD, _GAMMA, _N)
  legacy = _legacy_law()
  rows_v = np.zeros((1, 14))
  for step, (strain, time_new) in enumerate(_ramped(_uniaxial_path(), 0.0, 0.1), 1):
    _legacy_step(legacy, strain, time_new)
    rows_v = _v3_step(calibration, rows_v, strain, time_new).trial_rows
  kappa_loaded = rows_v[0, 12]
  assert kappa_loaded > 0.0
  elastic_tangent = calibration[10:46].reshape(6, 6)

  unload = np.array([0.0025, 0.0, 0.0, 0.0, 0.0, 0.0])
  sigma_l, tangent_l, row_l = _legacy_step(legacy, unload, 1.1)
  result = _v3_step(calibration, rows_v, unload, 1.1)
  assert np.array_equal(result.stresses[0], sigma_l)
  assert np.array_equal(result.trial_rows[0], row_l)
  assert np.array_equal(result.tangents[0], elastic_tangent)
  assert np.array_equal(tangent_l, elastic_tangent)
  assert result.trial_rows[0, 12] == kappa_loaded
  assert result.trial_rows[0, 13] == 1.1

  reload_strain = np.array([0.0055, 0.0, 0.0, 0.0, 0.0, 0.0])
  pre_reload = result.trial_rows
  sigma_l, tangent_l, row_l = _legacy_step(legacy, reload_strain, 1.2)
  result = _v3_step(calibration, pre_reload, reload_strain, 1.2)
  assert np.array_equal(result.stresses[0], sigma_l)
  assert np.array_equal(result.trial_rows[0], row_l)
  assert result.trial_rows[0, 12] > kappa_loaded
  coded, true, plastic = _map_tangents(calibration, pre_reload[0], reload_strain, 1.2)
  assert plastic
  assert np.array_equal(tangent_l, coded)
  assert np.array_equal(result.tangents[0], true)


def test_dtime_gate_returns_elastic_above_yield_bitwise() -> None:
  # The semantic trap (ViscoPlasticity.py:205): a supra-yield step at
  # constant or backward time commits the ELASTIC trial response above the
  # yield surface — epsilon_e advances, epsilon_p and kappa freeze, the time
  # slot records the bound time. Pinned bitwise on both sides.
  calibration = perzyna_viscoplasticity_calibration(_E, _NU, _SYIELD, _HARD, _GAMMA, _N)
  elastic_tangent = calibration[10:46].reshape(6, 6)
  for gamma, n in ((_GAMMA, _N), (1.0e2, 2.0)):
    calibration_g = perzyna_viscoplasticity_calibration(
      _E, _NU, _SYIELD, _HARD, gamma, n
    )
    legacy = _legacy_law(gamma, n)
    rows_v = np.zeros((1, 14))
    # Commit one plastic step at t = 0.5.
    plastic_strain = np.array([0.004, 0.0, 0.0, 0.0, 0.0, 0.0])
    sigma_l, _, row_l = _legacy_step(legacy, plastic_strain, 0.5)
    result = _v3_step(calibration_g, rows_v, plastic_strain, 0.5)
    assert np.array_equal(result.stresses[0], sigma_l)
    assert np.array_equal(result.trial_rows[0], row_l)
    kappa_loaded = result.trial_rows[0, 12]
    assert kappa_loaded > 0.0
    rows_v = result.trial_rows

    # Constant-time supra-yield step: the gate fires.
    jump = np.array([0.008, 0.0, 0.0, 0.0, 0.0, 0.0])
    sigma_l, tangent_l, row_l = _legacy_step(legacy, jump, 0.5)
    result = _v3_step(calibration_g, rows_v, jump, 0.5)
    assert np.array_equal(result.stresses[0], sigma_l)
    assert np.array_equal(result.trial_rows[0], row_l)
    # Elastic response above yield: stress is the elastic trial of the
    # advanced epsilon_e, tangent is the elastic stiffness, plastic state
    # frozen.
    eelas_new = rows_v[0, 0:6] + (jump - (rows_v[0, 0:6] + rows_v[0, 6:12]))
    np.testing.assert_array_equal(
      result.stresses[0], np.dot(elastic_tangent, eelas_new)
    )
    assert _von_mises(result.stresses[0]) > _SYIELD + _HARD * kappa_loaded
    assert np.array_equal(result.tangents[0], elastic_tangent)
    assert np.array_equal(tangent_l, elastic_tangent)
    assert result.trial_rows[0, 12] == kappa_loaded
    assert np.array_equal(result.trial_rows[0, 6:12], rows_v[0, 6:12])
    assert result.trial_rows[0, 13] == 0.5
    rows_v = result.trial_rows

    # Backward-time step: the same gate.
    backward = np.array([0.010, 0.0, 0.0, 0.0, 0.0, 0.0])
    sigma_l, tangent_l, row_l = _legacy_step(legacy, backward, 0.2)
    result = _v3_step(calibration_g, rows_v, backward, 0.2)
    assert np.array_equal(result.stresses[0], sigma_l)
    assert np.array_equal(result.trial_rows[0], row_l)
    assert np.array_equal(result.tangents[0], elastic_tangent)
    assert result.trial_rows[0, 12] == kappa_loaded
    assert result.trial_rows[0, 13] == 0.2


def test_converged_stress_is_invariant_to_gamma_and_n() -> None:
  # The rate-independence semantics pin (finding
  # 20261009-agent-vp1-...-rate-independent-j2-perzyna): gamma and n seed the
  # discarded initial guess only. Legacy-vs-legacy and v3-vs-v3 between
  # (gamma=1e-4, n=1) and (gamma=1e2, n=2) agree to Newton-tolerance slack —
  # measured 1.4e-16 relative on this path; asserted at the mission's 1e-8
  # bound. Within each configuration the v3-vs-legacy comparison is bitwise
  # (the ramp harness asserts it step by step).
  path = _ramped(_uniaxial_path(), 0.0, 0.1)
  rows_a, stresses_a, plastic_a = _run_path_parity(path, gamma=1.0e-4, n=1.0)
  # At gamma = 1e2 the coded-legacy rate_factor is ACTIVE (the gap is far
  # outside the deck-constants dormancy band); the leg pins stress/state
  # parity and the divergence mechanism, not the dormancy bound.
  rows_b, stresses_b, plastic_b = _run_path_parity(
    path, gamma=1.0e2, n=2.0, max_tangent_gap=1.0
  )
  assert plastic_a and plastic_b
  for sigma_a, sigma_b in zip(stresses_a, stresses_b):
    np.testing.assert_allclose(sigma_a, sigma_b, rtol=1.0e-8, atol=1.0e-8)
  np.testing.assert_allclose(rows_a[0, :13], rows_b[0, :13], rtol=1.0e-8, atol=1.0e-12)
  # The time slot is schedule-recorded, identical by construction.
  assert rows_a[0, 13] == rows_b[0, 13]


def test_tangent_divergence_pin_at_the_amplified_state() -> None:
  # gamma = 1e4, dtime = 1e3 (rate_factor A ~ 4e4): the legacy coded tangent
  # contradicts a central FD of the legacy law's own stress response by
  # 7.0e-2 relative of max|tang|, while the v3 rate-factor-free tangent
  # matches an FD of the kernel's own response to ~1.6e-10. Both recomputed
  # forms reproduce their returned counterparts bit for bit.
  gamma, n = 1.0e4, 1.0
  calibration = perzyna_viscoplasticity_calibration(_E, _NU, _SYIELD, _HARD, gamma, n)
  legacy = _legacy_law(gamma, n)
  rows_v = np.zeros((1, 14))
  setup = np.array([0.002, 0.0, 0.0, 0.0, 0.0, 0.0])
  _legacy_step(legacy, setup, 1000.0)
  rows_v = _v3_step(calibration, rows_v, setup, 1000.0).trial_rows
  assert rows_v[0, 12] > 0.0

  probe_strain, probe_time = np.array([0.003, 0.0, 0.0, 0.0, 0.0, 0.0]), 2000.0
  _, tangent_l = _legacy_probe(legacy, probe_strain, probe_time)
  base = _v3_step(calibration, rows_v, probe_strain, probe_time)
  assert base.trial_rows[0, 12] > rows_v[0, 12]  # the probe step is plastic
  coded, true, plastic = _map_tangents(calibration, rows_v[0], probe_strain, probe_time)
  assert plastic
  assert np.array_equal(tangent_l, coded)
  assert np.array_equal(base.tangents[0], true)

  fd_v3 = _fd_tangent_v3(calibration, rows_v, probe_strain, probe_time)
  fd_l = _fd_tangent_legacy(legacy, probe_strain, probe_time)
  # Both laws implement the same stress map: the FDs agree to FD noise.
  scale = float(np.max(np.abs(fd_v3)))
  np.testing.assert_allclose(fd_v3, fd_l, rtol=1.0e-6, atol=scale * 1.0e-6)
  legacy_error = float(np.max(np.abs(fd_l - tangent_l))) / scale
  v3_error = float(np.max(np.abs(fd_v3 - base.tangents[0]))) / scale
  assert legacy_error > 1.0e-2  # measured 7.0e-2
  assert v3_error < 1.0e-8  # measured 1.6e-10
  gap = float(np.max(np.abs(tangent_l - base.tangents[0]))) / scale
  assert gap > 1.0e-2  # the pinned divergence, dominant at this state
  # Symmetry: the true tangent is exactly symmetric; the FDs are symmetric to
  # their noise floor (measured ~1.6e-10 at this state).
  assert float(np.max(np.abs(fd_v3 - fd_v3.T))) / scale < 1.0e-8
  assert float(np.max(np.abs(base.tangents[0] - base.tangents[0].T))) == 0.0


def test_kernel_tangent_is_the_exact_stress_map_derivative_by_fd() -> None:
  """Central FD of the kernel's own stress update vs the returned tangent.

  Seeded nonzero states from the documented paths (elastic, mid-plastic,
  mixed shear, unload, reload) plus the dtime=0 gate and a seeded random
  plastic state: the map is smooth on the 1e-7 stencil at every probed state
  (the nearest branch boundary is O(1e-4) of strain away; the gate is a time
  branch, not a strain branch), so the finite difference is exact up to
  rounding — worst observed 1.6e-10 relative of max|tang|.
  """
  calibration = perzyna_viscoplasticity_calibration(_E, _NU, _SYIELD, _HARD, _GAMMA, _N)
  rows = np.zeros((1, 14))
  worst = 0.0
  went_plastic = False
  path = [*_ramped(_uniaxial_path(), 0.0, 0.1), *_ramped(_shear_path(), 1.0, 0.1)]
  path += [
    (np.array([0.0025, 0.0, 0.0, 0.0, 0.0, 0.006]), 1.8),
    (np.array([0.0055, 0.0, 0.0, 0.0, 0.0, 0.006]), 2.0),
  ]
  for strain_total, time_new in path:
    strain_total = np.array(strain_total, dtype=np.float64)
    base = _v3_step(calibration, rows, strain_total, time_new)
    fd = _fd_tangent_v3(calibration, rows, strain_total, time_new)
    tangent = base.tangents[0]
    scale = float(np.max(np.abs(tangent)))
    worst = max(worst, float(np.max(np.abs(fd - tangent))) / scale)
    assert float(np.max(np.abs(fd - fd.T))) / scale < 1.0e-8
    went_plastic |= bool(base.trial_rows[0, 12] > rows[0, 12])
    rows = base.trial_rows
  assert went_plastic

  # The dtime=0 gate state: elastic branch above yield, affine in strain.
  gate_probe = np.array([0.008, 0.0, 0.0, 0.0, 0.0, 0.006])
  base = _v3_step(calibration, rows, gate_probe, 2.0)
  fd = _fd_tangent_v3(calibration, rows, gate_probe, 2.0)
  scale = float(np.max(np.abs(base.tangents[0])))
  worst = max(worst, float(np.max(np.abs(fd - base.tangents[0]))) / scale)

  # A seeded random plastic state (fixed seed, deterministic path).
  rng = np.random.default_rng(20261009)
  rows = np.zeros((1, 14))
  for step in range(3):
    strain = rng.normal(size=6) * 1.0e-3 * (step + 1)
    strain[0] += 0.002 * (step + 1)  # keep the path clearly plastic
    rows = _v3_step(calibration, rows, strain, 0.4 * (step + 1)).trial_rows
  assert rows[0, 12] > 0.0
  probe = strain + rng.normal(size=6) * 5.0e-4
  base = _v3_step(calibration, rows, probe, 2.0)
  fd = _fd_tangent_v3(calibration, rows, probe, 2.0)
  scale = float(np.max(np.abs(base.tangents[0])))
  worst = max(worst, float(np.max(np.abs(fd - base.tangents[0]))) / scale)
  assert worst < 1.0e-8, worst  # observed: ~1e-10 class over all states


def test_kernel_reports_typed_failures() -> None:
  calibration = perzyna_viscoplasticity_calibration(_E, _NU, _SYIELD, _HARD, _GAMMA, _N)
  rows = np.zeros((1, 14))
  signals = (_time_signal(0.5),)
  # Non-finite strain batch.
  result = perzyna_viscoplasticity_kernel(
    np.array([[np.inf, 0.0, 0.0, 0.0, 0.0, 0.0]]), rows, calibration, signals
  )
  assert result.status is EvaluationStatus.REJECT_STEP
  assert np.array_equal(result.trial_rows, rows)
  # Non-finite bound time.
  result = perzyna_viscoplasticity_kernel(
    np.zeros((1, 6)), rows, calibration, (_time_signal(np.nan),)
  )
  assert result.status is EvaluationStatus.REJECT_STEP
  assert np.array_equal(result.trial_rows, rows)
  # Return-map non-convergence (the legacy warn-and-continue, :242-244): a
  # tampered calibration with hard = -eg3 degenerates the Newton Jacobian to
  # zero, so the iteration exhausts the 20-iteration cap and rejects typed.
  tampered = np.array(calibration, copy=True)
  tampered[6] = -tampered[2]
  result = perzyna_viscoplasticity_kernel(
    np.array([[0.004, 0.0, 0.0, 0.0, 0.0, 0.0]]), rows, tampered, signals
  )
  assert result.status is EvaluationStatus.REJECT_STEP
  assert np.array_equal(result.trial_rows, rows)
  # Contract violations stay typed exceptions.
  with pytest.raises(TypeError, match="calibration"):
    perzyna_viscoplasticity_kernel(np.zeros((1, 6)), rows, np.zeros(3), signals)
  with pytest.raises(ValueError, match="strains and"):
    perzyna_viscoplasticity_kernel(np.zeros((1, 3)), rows, calibration, signals)
  with pytest.raises(TypeError, match="time port"):
    perzyna_viscoplasticity_kernel(np.zeros((1, 6)), rows, calibration, ())
  wrong_port = StatefulContinuumSignalInput("load", np.array([0.5]), ())
  with pytest.raises(TypeError, match="time port"):
    perzyna_viscoplasticity_kernel(np.zeros((1, 6)), rows, calibration, (wrong_port,))


def test_virgin_probe_returns_rows_byte_equal() -> None:
  calibration = perzyna_viscoplasticity_calibration(_E, _NU, _SYIELD, _HARD, _GAMMA, _N)
  rows = np.zeros((1, 14))
  result = perzyna_viscoplasticity_kernel(
    np.zeros((1, 6)), rows, calibration, (_time_signal(0.0),)
  )
  assert result.status is EvaluationStatus.OK
  assert np.array_equal(result.trial_rows, rows)
  assert np.array_equal(result.tangents[0], calibration[10:46].reshape(6, 6))


def test_calibration_validation_is_strict() -> None:
  with pytest.raises(TypeError, match="exact float"):
    perzyna_viscoplasticity_calibration(200000, _NU, _SYIELD, _HARD, _GAMMA, _N)
  with pytest.raises(ValueError, match="must be finite"):
    perzyna_viscoplasticity_calibration(np.nan, _NU, _SYIELD, _HARD, _GAMMA, _N)
  with pytest.raises(ValueError, match="youngs_modulus must be positive"):
    perzyna_viscoplasticity_calibration(-1.0, _NU, _SYIELD, _HARD, _GAMMA, _N)
  with pytest.raises(ValueError, match="between -1 and 0.5"):
    perzyna_viscoplasticity_calibration(_E, 0.5, _SYIELD, _HARD, _GAMMA, _N)
  with pytest.raises(ValueError, match="initial_yield_stress must be positive"):
    perzyna_viscoplasticity_calibration(_E, _NU, 0.0, _HARD, _GAMMA, _N)
  with pytest.raises(ValueError, match="hardening_slope must be non-negative"):
    perzyna_viscoplasticity_calibration(_E, _NU, _SYIELD, -1.0, _GAMMA, _N)
  with pytest.raises(ValueError, match="fluidity must be positive"):
    perzyna_viscoplasticity_calibration(_E, _NU, _SYIELD, _HARD, 0.0, _N)
  with pytest.raises(ValueError, match="rate_exponent must be positive"):
    perzyna_viscoplasticity_calibration(_E, _NU, _SYIELD, _HARD, _GAMMA, 0.0)
  with pytest.raises(TypeError, match="exactly six"):
    PERZYNA_VISCOPLASTICITY_BINDING(_E, _NU, _SYIELD, _HARD, _GAMMA)


def test_frozen_v2_schema_covers_the_law_without_new_fields() -> None:
  metadata = perzyna_viscoplasticity_metadata()
  assert metadata["schema"] == STATEFUL_MATERIAL_DESCRIPTOR_SCHEMA
  assert STATEFUL_MATERIAL_DESCRIPTOR_SCHEMA == "pyfem-v3-material-descriptor-v2"
  # The M25/M29 freeze accepts the law as-is: fixed-width slots, the
  # monotone annotation, and the optional signal_ports field are already in
  # the frozen v2 field set (contracts.py carries no diff for this mission).
  assert validate_stateful_material_metadata(metadata) is metadata
  slots = resolve_material_state_slots(
    metadata["state_slots"],
    {
      "youngs_modulus": _E,
      "poisson_ratio": _NU,
      "initial_yield_stress": _SYIELD,
      "hardening_slope": _HARD,
      "fluidity": _GAMMA,
      "rate_exponent": _N,
    },
  )
  assert [slot.name for slot in slots] == ["epsilon_e", "epsilon_p", "kappa", "time"]
  assert [slot.width for slot in slots] == [6, 6, 1, 1]
  assert slots[2].annotation == "monotone-nondecreasing"


def _vp_model() -> ModelSpec:
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
        id="steel",
        model="perzyna-viscoplasticity",
        parameters=(
          MaterialParameterSpec("youngs_modulus", _E),
          MaterialParameterSpec("poisson_ratio", _NU),
          MaterialParameterSpec("initial_yield_stress", _SYIELD),
          MaterialParameterSpec("hardening_slope", _HARD),
          MaterialParameterSpec("fluidity", _GAMMA),
          MaterialParameterSpec("rate_exponent", _N),
        ),
        source=_source("material"),
      ),
    ),
    regions=(
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


def _elastic_model() -> ModelSpec:
  """The same one-element mesh with the port-free plane-stress linear law."""
  base = _vp_model()
  return ModelSpec(
    mesh=base.mesh,
    fields=base.fields,
    materials=(
      MaterialSpec(
        id="steel",
        model="plane-stress-linear-elastic",
        parameters=(
          MaterialParameterSpec("youngs_modulus", _E),
          MaterialParameterSpec("poisson_ratio", _NU),
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


def _vp_compiled() -> CompiledSystem:
  return _compiled(_vp_model(), viscoplasticity_reference_registry())


def test_compiled_state_layout_and_fingerprint() -> None:
  layout = _vp_compiled().operators[0].header.state_layout
  assert layout.entity_count == 9
  assert layout.row_width == 14
  assert layout.schema == (
    "pyfem-v3-perzyna-viscoplastic-state-v1|epsilon_e:6,epsilon_p:6,kappa:1,time:1"
  )
  assert layout.initial_rows is not None
  assert np.all(layout.initial_rows.values == 0.0)
  # Deterministic content identity across compilations.
  assert _vp_compiled().content_fingerprint == _vp_compiled().content_fingerprint


def test_initial_state_binding_is_explicit_and_validated() -> None:
  layout = _vp_compiled().operators[0].header.state_layout
  parameters = (_E, _NU, _SYIELD, _HARD, _GAMMA, _N)
  rows = perzyna_viscoplasticity_initial_state(parameters, layout)
  assert rows.shape == layout.row_shape
  assert np.all(rows == 0.0)
  with pytest.raises(TypeError, match="six parameters"):
    perzyna_viscoplasticity_initial_state(parameters[:5], layout)
  with pytest.raises(ValueError, match="row layout"):
    perzyna_viscoplasticity_initial_state(
      parameters,
      type(
        "Layout",
        (),
        {"row_width": 13, "row_shape": layout.row_shape},
      )(),
    )


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
  operator = _vp_compiled().operators[0]
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
    "pyfem-v3-perzyna-viscoplastic-state-v1|epsilon_e:6,epsilon_p:6,kappa:1,time:1"
  )
  assert [slot["name"] for slot in state["slots"]] == [
    "epsilon_e",
    "epsilon_p",
    "kappa",
    "time",
  ]
  assert [slot["width"] for slot in state["slots"]] == [
    ["int", "6"],
    ["int", "6"],
    ["int", "1"],
    ["int", "1"],
  ]


def test_compiled_operator_binds_the_time_port_exactly() -> None:
  operator = _vp_compiled().operators[0]
  values = FinalizedArray(np.zeros((1, 16)), dtype=np.float64)
  accepted = FinalizedArray(
    np.zeros(operator.header.state_layout.row_shape), dtype=np.float64
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
  return NonlinearStaticDriver(system, coordinate_map, ())


def _vp_driver() -> NonlinearStaticDriver:
  return _ramp_driver(_vp_model(), viscoplasticity_reference_registry())


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
  driver = _vp_driver()
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
  driver = _vp_driver()
  layout = driver.owner.system.operators[0].header.state_layout
  block_id = layout.block_id
  calibration = perzyna_viscoplasticity_calibration(_E, _NU, _SYIELD, _HARD, _GAMMA, _N)
  oracle_rows = np.zeros((9, 14))
  # The third substep holds time constant while strain advances ABOVE YIELD:
  # the dtime gate branch, committed end-to-end through the driver (kappa
  # freezes while epsilon_e advances; a semantic trap pinned at driver level).
  schedule = (
    (0.05, 2.0e-4),
    (0.15, 5.0e-4),
    (0.40, 2.5e-3),
    (0.40, 4.0e-3),
    (1.00, 5.5e-3),
  )
  base = _point(0.0, 0.0)
  kappa_after_plastic = None
  for step, (time, eps) in enumerate(schedule, 1):
    result = driver.run(base_point=base, target_points=(_point(time, eps),))
    assert result.status is DriverStatus.COMPLETED
    base = _point(time, eps)
    # The committed rows equal the kernel stepped on the committed
    # integration-point strain path — bitwise, since the driver stages the
    # trial rows of the converged iterate.
    oracle = perzyna_viscoplasticity_kernel(
      _committed_ip_strains(driver), oracle_rows, calibration, (_time_signal(time),)
    )
    assert oracle.status is EvaluationStatus.OK
    oracle_rows = oracle.trial_rows
    rows = driver.owner.accepted_state(block_id).values
    assert rows.shape == (9, 14)
    np.testing.assert_array_equal(rows, oracle_rows)
    # State rows record the substep's bound time at every integration point.
    np.testing.assert_array_equal(rows[:, 13], time)
    if step == 3:
      assert np.all(rows[:, 12] > 0.0)
      kappa_after_plastic = rows[:, 12].copy()
    if step == 4:
      # The constant-time supra-yield substep: the gate froze plastic flow.
      np.testing.assert_array_equal(rows[:, 12], kappa_after_plastic)
    if step == 5:
      assert np.all(rows[:, 12] > kappa_after_plastic)
    assert result.final_generation.ordinal == step
  statistics = result.statistics
  assert statistics.committed_substep_count == 5
  assert statistics.rejected_substep_count == 0
  # Factorization honesty: the algorithmic-symmetric channel is not linear,
  # so the driver re-assembles and re-factorizes every Newton iteration.
  assert statistics.factorization_reuse_count == 0
  assert statistics.factorization_count == statistics.tangent_refill_count
  # Homogeneous strain: all nine integration points agree to solver tolerance.
  np.testing.assert_allclose(
    rows, np.broadcast_to(rows[0], rows.shape), rtol=1.0e-9, atol=1.0e-10
  )


def test_driver_gate_changes_the_committed_response() -> None:
  # Positive control: the same strain step with advancing versus constant
  # time commits materially different states — schedule-owned time reaches
  # the law through the port (here: through the dtime gate).
  advanced = _vp_driver()
  held = _vp_driver()
  block_id = advanced.owner.system.operators[0].header.state_layout.block_id
  base = _point(0.0, 0.0)
  for time, eps in ((0.4, 2.5e-3),):
    for driver in (advanced, held):
      result = driver.run(base_point=base, target_points=(_point(time, eps),))
      assert result.status is DriverStatus.COMPLETED
  advanced_result = advanced.run(
    base_point=_point(0.4, 2.5e-3), target_points=(_point(1.0, 4.0e-3),)
  )
  held_result = held.run(
    base_point=_point(0.4, 2.5e-3), target_points=(_point(0.4, 4.0e-3),)
  )
  assert advanced_result.status is DriverStatus.COMPLETED
  assert held_result.status is DriverStatus.COMPLETED
  advanced_rows = advanced.owner.accepted_state(block_id).values
  held_rows = held.owner.accepted_state(block_id).values
  # The held-time run froze kappa; the advancing run flowed further.
  assert np.all(advanced_rows[:, 12] > held_rows[:, 12])
  # The reactions (internal force at the prescribed DOFs) carry the
  # materially different stress response: elastic above yield vs plastic.
  # (The free DOF converges byte-identical through the homogeneous
  # prescribed field, so displacement bytes cannot discriminate here.)
  assert (
    advanced_result.records[-1].observation.reactions.values.tobytes()
    != held_result.records[-1].observation.reactions.values.tobytes()
  )


def test_driver_response_is_rate_invariant_under_advancing_schedules() -> None:
  # The rate-independence semantics end-to-end: two schedules that both
  # advance time on every substep (different rates) commit the same stress
  # response to Newton-tolerance slack (law-level measured 1.4e-16), while
  # the recorded time slots track their own schedules.
  first = _vp_driver()
  second = _vp_driver()
  block_id = first.owner.system.operators[0].header.state_layout.block_id
  first_base = _point(0.0, 0.0)
  second_base = _point(0.0, 0.0)
  eps_path = (2.0e-4, 2.5e-3, 5.5e-3)
  for first_time, second_time, eps in (
    (0.05, 0.50, eps_path[0]),
    (0.40, 3.00, eps_path[1]),
    (1.00, 12.0, eps_path[2]),
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
  # Stress/strain/kappa slots agree to solver tolerance; the time slot
  # records each schedule's own bound time.
  np.testing.assert_allclose(
    first_rows[:, :13], second_rows[:, :13], rtol=1.0e-8, atol=1.0e-10
  )
  np.testing.assert_array_equal(first_rows[:, 13], 1.00)
  np.testing.assert_array_equal(second_rows[:, 13], 12.0)
  np.testing.assert_allclose(
    first.owner.accepted_physical().values,
    second.owner.accepted_physical().values,
    rtol=1.0e-9,
    atol=1.0e-12,
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
  driver = _vp_driver()
  result = driver.run(base_point=_point(0.0, 0.0), target_points=(_point(0.5, 2.5e-3),))
  assert result.status is DriverStatus.COMPLETED
  owner = driver.owner
  layout = owner.system.operators[0].header.state_layout
  payload = owner.encode_state(layout.block_id)
  decoded = owner.decode_state(layout.block_id, payload)
  np.testing.assert_array_equal(
    decoded.values, owner.accepted_state(layout.block_id).values
  )
  with pytest.raises(StateCodecError):
    owner.decode_state(layout.block_id, payload.replace(b"kappa:1", b"kappa:2"))
