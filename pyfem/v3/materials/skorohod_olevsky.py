"""Skorohod-Olevsky viscous sintering: the explicit-update v3 stateful law.

A pure, batched port of the legacy reference ``pyfem/materials/SOVS.py`` (the
numerical oracle per AGENTS.md) onto the v2 stateful descriptor ABI with a
declared ``signal_ports`` time port. The law is the Skorohod-Olevsky viscous
sintering model: a porous ceramic compact densifies under the sintering
stress and creeps under deviatoric stress through the Skorohod viscosity
functions ``eta = eta_ref * (rho**-n - 1)`` with the Arrhenius reference
viscosity ``eta_ref = eta0 * exp(Q / (R * T))`` bound at calibration time
(legacy :163). The law is 3D internally (6-Voigt ``[xx, yy, zz, yz, zx, xy]``
with engineering shears); the continuum operator embeds the 3-Voigt port
strain as ``[xx, yy, 0, 0, 0, xy]`` (a plane-strain embedding) and truncates
stress/tangent back to ``[xx, yy, xy]`` (the M25 seam).

Parameters (``parameter_names`` order): ``reference_viscosity`` eta0 > 0,
``activation_energy`` Q > 0, ``temperature`` T > 0,
``initial_relative_density`` 0 < rho0 <= 1 (the legacy ctor validation
:143-159), ``sintering_stress`` sigma_sint > 0, ``gas_constant`` R > 0
(legacy default 8.314), ``viscosity_exponent_volumetric`` n_vol > 0 (legacy
default 2.0), and ``viscosity_exponent_shear`` n_shear > 0 (legacy default
1.0). The descriptor ABI fixes the parameter list per descriptor, so all
eight are required.

Time reaches the law exclusively through the declared identity signal port
(``port_id`` ``"time"``, ``signal_id`` ``"time"``), replacing the legacy
``self.solverStat.time`` read (:222) — there is no schedule back channel, and
state rows record the bound time so restarts are sound (M49 doctrine).

State rows (14 floats per entity, flattened element x ip x slot): ``strain``
6 (the committed total strain — the legacy law consumes the strain increment
supplied by its element while the frozen v3 kernel ABI supplies total strain,
so the committed total is state; the increment is
``dstrain = strains - strain`` on the accepted row, the identity the parity
harness uses to drive the legacy oracle), ``strain_visc`` 6, ``rho`` 1, and
``time`` 1. The legacy write-only ``sigma`` history slot (:169) is dropped.
**The initial state carries ``rho = rho0``** — the first parameter-dependent
initial row of the stateful family (the M25 prony law was all-zero); the
binding's ``initial_state`` writes the validated ``initial_relative_density``
parameter into the ``rho`` slot through the existing M25 G2 binding path,
no machinery change.

The kernel replicates the legacy ``getStress`` arithmetic statement for
statement (:215-344): the explicit forward-Euler viscous update
(:284-306 — ``drho`` and the deviatoric viscous strain from committed-rho
viscosities and the TRIAL stress, the mid-step ordering), the ``eta`` guard
``rho >= 0.999 -> 1e20`` (:236-242), the density clamp ``rho_new`` into
``[rho0, 1]`` (:292-293 — note the volumetric viscous strain reads the
UNCLAMPED ``drho`` at :297, so the clamp kinks the state trajectory but the
strain->stress map at fixed state stays affine), and the hard-coded modulus
law ``E = 100e9 * (rho/rho0)**2.5, nu = 0.25`` (:254-255), replicated as
calibration constants per the tower ruling (the constants ride the
calibration vector so the kernel stays pure; no new props). Three legacy
behaviors are documented, not silently changed:

- ``outData`` reports the PRE-update rho (:348 — ``rho`` is written to the
  history at :301 before ``outData[6]`` reads the stale local at :348): a
  reporting-lag quirk of the legacy output channel. The v2 kernel ABI has no
  output channel — the committed state row carries the post-update rho,
  exactly the value the legacy history holds — so the lag has no v3
  numerical consequence and is pinned here for the record (tower ruling M68).
- The ``maximum_principal_stress`` helper (:357-386) is unused by
  ``getStress`` and is not replicated.
- Legacy raises ``ValueError`` from the constructor on out-of-domain
  parameters; the v3 calibration binding does the same at bind time, and
  in-evaluation numerical failures (non-finite strain batch, non-finite bound
  time, non-finite predictor or response) report ``REJECT_STEP`` with
  byte-equal trial rows — typed, never exceptions.

Tangent class: ``algorithmic-symmetric`` — the Jacobian channel is nonlinear
(never ``linear``) and symmetric; the driver re-factorizes every Newton
iteration (counter-asserted in the battery).

The TRUE algorithmic tangent (one legacy statement NOT replicated). The
legacy coded tangent (:322-339) is the closed form of an IMPLICIT step —
``K/(1 + K dt 3/(2 eta_vol))``, ``G/(1 + G dt/eta_shear)`` — while the stress
update is explicit forward Euler: against a central finite difference of the
legacy law's own stress response at an activated state (eta0 = 1e10, Q = 1,
T = 1600, rho0 = 0.6, sigma_sint = 1e6; 20 free-sintering steps at
dtime = 0.01, rho = 0.6000062) the coded tangent errs at 9.41e-3 relative of
max|tang| at dtime = 0.01 and 9.94e-4 at dtime = 0.001 — O(dt) inconsistency
(finding
20261009-agent-vp1-bug-sovs-tangent-is-the-implicit-step-form-of-an-explicit-update;
the elastic branch dtime = 0 is exact, and the FD tangent is exactly
symmetric). At fixed committed state the implemented map is AFFINE in the
trial strain, so its exact derivative has closed form; differentiating the
explicit update gives three legs at the committed rho:

- volumetric: ``K_alg = K * (1 - 3*K*dt / (2*eta_vol))`` from the
  densification feedback (all nine normal-normal entries);
- normal-deviatoric: ``G_dev = G * (1 - G*dt / eta_shear)`` scaling
  ``2*G_dev * (I - J/3)`` — the deviatoric viscous strain increments the
  NORMAL components too (:305-306 reads the full deviatoric stress), and the
  normal deviatoric derivative ``d(s_dev)/d(e) = 2G(δ - 1/3)`` carries the
  law's own ``1/(2 eta_shear)`` factor to ``G*dt/eta_shear`` here;
- shear: ``G_alg = G * (1 - G*dt / (2*eta_shear))`` on the three shear
  diagonals.

assembled as ``tang[:3,:3] = K_alg - 2*G_dev/3`` plus ``2*G_dev`` on the
normal diagonal and ``G_alg`` on the shear diagonal, cross terms zero. This
closed form matches the central finite difference to 3.3e-15 relative at the
activated state above (rounding level — the map is affine), at both probed
dtime scales. Note the M63 survey sketch's two-modulus "isotropic
(K_alg, G_alg)" assembly is NOT the derivative of the implemented map: it
misses the deviatoric flow on the normal components and errs at 1.4e-2
against the same finite difference (worse than the legacy coded tangent on
the normal block) — the three-modulus form above is the FD-identified truth
the mission pins (the volumetric and shear legs agree with the survey; only
the normal-block deviatoric leg was under-specified). The v3 kernel writes
this true tangent; stress and state bookkeeping stay bitwise-identical to
legacy, and the parity battery pins the tangent divergence by mechanism. The
L3 mission (M67) repairs the legacy side in parallel; when it lands, the
divergence pins flip to parity-where-repaired by whoever integrates second
(M54/M55 precedent).

Dormancy note: both shipped sintering decks (``examples/materials/sintering/``)
carry eta0 = 1e12, Q = 5e5, T = 1600, so ``eta_ref = 2.1e28`` Pa.s and the
viscous machinery never activates measurably — the decks are effectively
elastic (the survey measured drho = 0.0 over a full dtime = 2 step). The
parity battery therefore owns activated constants; the dormant deck
configuration is pinned as its own bitwise leg.

The virgin probe (zero strain, zero bound time, initial rows with
``rho = rho0``) takes the ``dtime == 0`` branch with a zero increment and
returns the accepted rows byte-equal; the virgin tangent is the elastic
stiffness at ``rho0`` (``rho/rho0 == 1``), finite and symmetric.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyfem.v3.compile.contracts import (
  STATEFUL_MATERIAL_DESCRIPTOR_SCHEMA,
  StatefulContinuumKernelResult,
  StatefulContinuumSignalInput,
)
from pyfem.v3.model.operator import EvaluationStatus, OperatorStateLayout

SKOROHOD_OLEVSKY_STATE_SCHEMA = "pyfem-v3-skorohod-olevsky-state-v1"

_TIME_PORT_ID = "time"

# Legacy hard-coded constants (SOVS.py:236-242): the full-density guard level
# and viscosity plateau.
_RHO_GUARD = 0.999
_ETA_PLATEAU = 1.0e20

# Flat calibration vector layout: eta_ref, rho0, sigma_sint, n_vol, n_shear,
# then the hard-coded modulus law constants (E_base, rho_power, nu) of
# SOVS.py:254-255 packed so the kernel stays pure.
_CALIBRATION_SIZE = 8
_ROW_WIDTH = 14


def skorohod_olevsky_metadata() -> dict[str, object]:
  """Return the canonical v2 descriptor metadata for the sintering law."""
  return {
    "schema": STATEFUL_MATERIAL_DESCRIPTOR_SCHEMA,
    "law": "skorohod-olevsky",
    "stress_state": "plane-strain",
    "parameter_names": [
      "reference_viscosity",
      "activation_energy",
      "temperature",
      "initial_relative_density",
      "sintering_stress",
      "gas_constant",
      "viscosity_exponent_volumetric",
      "viscosity_exponent_shear",
    ],
    "parameter_dtype": "float64",
    "stress_voigt_order": ["xx", "yy", "xy"],
    "strain_shear_convention": "engineering",
    "internal_voigt_order": ["xx", "yy", "zz", "yz", "zx", "xy"],
    "tangent_class": "algorithmic-symmetric",
    "state_schema": SKOROHOD_OLEVSKY_STATE_SCHEMA,
    "state_slots": [
      {
        "name": "strain",
        "width": 6,
        "dtype": "float64",
        "lifetime": "accepted-trial",
      },
      {
        "name": "strain_visc",
        "width": 6,
        "dtype": "float64",
        "lifetime": "accepted-trial",
      },
      {
        "name": "rho",
        "width": 1,
        "dtype": "float64",
        "lifetime": "accepted-trial",
      },
      {
        "name": "time",
        "width": 1,
        "dtype": "float64",
        "lifetime": "accepted-trial",
      },
    ],
    "signal_ports": [
      {
        "port_id": _TIME_PORT_ID,
        "signal_id": _TIME_PORT_ID,
        "derivative_coordinate_ids": [_TIME_PORT_ID],
      },
    ],
  }


def skorohod_olevsky_calibration(
  reference_viscosity: float,
  activation_energy: float,
  temperature: float,
  initial_relative_density: float,
  sintering_stress: float,
  gas_constant: float,
  viscosity_exponent_volumetric: float,
  viscosity_exponent_shear: float,
) -> np.ndarray:
  """Pack the law's flat calibration vector with legacy-identical arithmetic."""
  for label, value in (
    ("reference_viscosity", reference_viscosity),
    ("activation_energy", activation_energy),
    ("temperature", temperature),
    ("initial_relative_density", initial_relative_density),
    ("sintering_stress", sintering_stress),
    ("gas_constant", gas_constant),
    ("viscosity_exponent_volumetric", viscosity_exponent_volumetric),
    ("viscosity_exponent_shear", viscosity_exponent_shear),
  ):
    if type(value) is not float:
      msg = f"skorohod-olevsky {label} must be an exact float"
      raise TypeError(msg)
    if not np.isfinite(value):
      msg = f"skorohod-olevsky {label} must be finite"
      raise ValueError(msg)
  # Legacy constructor domain validation (SOVS.py:149-159), plus the positive
  # domains the legacy defaults assume for R and the Skorohod exponents.
  if reference_viscosity <= 0.0:
    msg = "skorohod-olevsky reference_viscosity must be positive"
    raise ValueError(msg)
  if activation_energy <= 0.0:
    msg = "skorohod-olevsky activation_energy must be positive"
    raise ValueError(msg)
  if temperature <= 0.0:
    msg = "skorohod-olevsky temperature must be positive"
    raise ValueError(msg)
  if not 0.0 < initial_relative_density <= 1.0:
    msg = "skorohod-olevsky initial_relative_density must lie in (0, 1]"
    raise ValueError(msg)
  if sintering_stress <= 0.0:
    msg = "skorohod-olevsky sintering_stress must be positive"
    raise ValueError(msg)
  if gas_constant <= 0.0:
    msg = "skorohod-olevsky gas_constant must be positive"
    raise ValueError(msg)
  if viscosity_exponent_volumetric <= 0.0:
    msg = "skorohod-olevsky viscosity_exponent_volumetric must be positive"
    raise ValueError(msg)
  if viscosity_exponent_shear <= 0.0:
    msg = "skorohod-olevsky viscosity_exponent_shear must be positive"
    raise ValueError(msg)

  # Legacy constructor expression order (SOVS.py:163).
  eta_ref = reference_viscosity * np.exp(
    activation_energy / (gas_constant * temperature)
  )

  return np.array(
    [
      eta_ref,
      initial_relative_density,
      sintering_stress,
      viscosity_exponent_volumetric,
      viscosity_exponent_shear,
      100.0e9,
      2.5,
      0.25,
    ],
    dtype=np.float64,
  )


def _reject(accepted_rows: np.ndarray) -> StatefulContinuumKernelResult:
  entity_count = accepted_rows.shape[0]
  return StatefulContinuumKernelResult(
    stresses=np.zeros((entity_count, 6), dtype=np.float64),
    tangents=np.zeros((entity_count, 6, 6), dtype=np.float64),
    trial_rows=np.array(accepted_rows, dtype=np.float64, copy=True),
    status=EvaluationStatus.REJECT_STEP,
  )


def skorohod_olevsky_kernel(
  strains: np.ndarray,
  accepted_rows: np.ndarray,
  calibration: np.ndarray,
  signals: tuple[StatefulContinuumSignalInput, ...],
) -> StatefulContinuumKernelResult:
  """Evaluate the Skorohod-Olevsky law over flattened state entities.

  ``strains`` holds total engineering 6-Voigt strain per entity;
  ``accepted_rows`` holds the committed 14-float rows; ``signals`` carries
  exactly the declared identity time port. The strain increment is
  ``strains - strain`` from the accepted row, the identity the parity harness
  also uses to drive the legacy oracle; the time increment is the legacy
  ``time_new - time_old`` over the bound signal value and the row's committed
  time. Expected numerical failures report ``REJECT_STEP`` with byte-equal
  trial rows, never exceptions.
  """
  if (
    type(calibration) is not np.ndarray
    or calibration.dtype != np.dtype(np.float64)
    or calibration.dtype.metadata is not None
    or calibration.shape != (_CALIBRATION_SIZE,)
  ):
    msg = "skorohod-olevsky kernel requires the packed calibration vector"
    raise TypeError(msg)
  if (
    type(signals) is not tuple
    or len(signals) != 1
    or type(signals[0]) is not StatefulContinuumSignalInput
    or signals[0].port_id != _TIME_PORT_ID
  ):
    msg = "skorohod-olevsky kernel requires exactly the declared time port"
    raise TypeError(msg)
  entity_count = strains.shape[0]
  if strains.shape != (entity_count, 6) or accepted_rows.shape != (
    entity_count,
    _ROW_WIDTH,
  ):
    msg = "skorohod-olevsky kernel requires (n, 6) strains and (n, 14) rows"
    raise ValueError(msg)
  if not bool(np.isfinite(strains).all()):
    return _reject(accepted_rows)
  time_new = float(signals[0].values[0])
  if not np.isfinite(time_new):
    return _reject(accepted_rows)

  eta_ref = calibration[0]
  rho0 = calibration[1]
  sigma_sint = calibration[2]
  n_vol = calibration[3]
  n_shear = calibration[4]
  e_base = calibration[5]
  rho_power = calibration[6]
  nu_eff = calibration[7]

  stresses = np.empty((entity_count, 6), dtype=np.float64)
  tangents = np.empty((entity_count, 6, 6), dtype=np.float64)
  trial_rows = np.empty((entity_count, _ROW_WIDTH), dtype=np.float64)

  with np.errstate(over="ignore", invalid="ignore"):
    for index in range(entity_count):
      row = accepted_rows[index]
      # Legacy getStress statement order (SOVS.py:215-344).
      dtime = time_new - row[13]
      strain = row[0:6] + (strains[index] - row[0:6])
      rho = float(row[12])
      strain_visc_old = row[6:12]
      if rho < _RHO_GUARD:
        eta_vol = eta_ref * (rho ** (-n_vol) - 1.0)
        eta_shear = eta_ref * (rho ** (-n_shear) - 1.0)
      else:
        eta_vol = _ETA_PLATEAU
        eta_shear = _ETA_PLATEAU
      strain_elastic = strain - strain_visc_old
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
      sigma_trial = np.dot(ctang, strain_elastic)
      if not bool(np.isfinite(sigma_trial).all()):
        return _reject(accepted_rows)
      sigma_m = 0.333333333333333 * (sigma_trial[0] + sigma_trial[1] + sigma_trial[2])
      sigma_dev = np.array(sigma_trial, copy=True)
      sigma_dev[:3] = sigma_dev[:3] - sigma_m

      dstrain_visc = np.zeros(6)
      rho_new = rho
      if dtime > 0.0:
        driving_stress = sigma_sint - sigma_m
        drho_dt = (3.0 * rho / (2.0 * eta_vol)) * driving_stress
        drho = drho_dt * dtime
        rho_new = min(rho + drho, 1.0)
        rho_new = max(rho_new, rho0)
        # The volumetric viscous strain reads the UNCLAMPED drho (:297).
        dstrain_vol_visc = -drho / rho
        dstrain_visc[:3] = dstrain_vol_visc / 3.0
        dstrain_visc += sigma_dev / (2.0 * eta_shear) * dtime

      strain_visc = strain_visc_old + dstrain_visc
      strain_elastic = strain - strain_visc
      sigma = np.dot(ctang, strain_elastic)

      if dtime > 0.0:
        # The TRUE algorithmic tangent of the implemented explicit update:
        # the three-leg closed form of the module docstring (FD-identified,
        # exact at rounding level because the map is affine in the trial
        # strain at fixed committed state). The legacy coded tangent — the
        # implicit-step form — is not replicated (the pinned divergence).
        k_mod = ebulk3 / 3.0
        k_alg = k_mod * (1.0 - 3.0 * k_mod * dtime / (2.0 * eta_vol))
        g_dev = eg * (1.0 - eg * dtime / eta_shear)
        g_alg = eg * (1.0 - eg * dtime / (2.0 * eta_shear))
        lam_alg = k_alg - 2.0 * g_dev / 3.0
        tang = np.zeros(shape=(6, 6))
        tang[:3, :3] = lam_alg
        tang[0, 0] += 2.0 * g_dev
        tang[1, 1] = tang[0, 0]
        tang[2, 2] = tang[0, 0]
        tang[3, 3] = g_alg
        tang[4, 4] = tang[3, 3]
        tang[5, 5] = tang[3, 3]
      else:
        tang = np.array(ctang, copy=True)
      if not bool(np.isfinite(sigma).all()) or not bool(np.isfinite(tang).all()):
        return _reject(accepted_rows)

      trial_rows[index, 0:6] = strain
      trial_rows[index, 6:12] = strain_visc
      trial_rows[index, 12] = rho_new
      trial_rows[index, 13] = time_new
      stresses[index] = sigma
      tangents[index] = tang

  return StatefulContinuumKernelResult(
    stresses=stresses,
    tangents=tangents,
    trial_rows=trial_rows,
    status=EvaluationStatus.OK,
  )


def skorohod_olevsky_initial_state(
  parameters: tuple[float, ...],
  layout: OperatorStateLayout,
) -> np.ndarray:
  """Bind the initial rows: all zero except the parameter-dependent rho0.

  The first parameter-dependent initial row of the stateful family (M25 G2):
  the ``rho`` slot starts at the validated ``initial_relative_density``
  parameter, matching the legacy constructor's history commit (:166).
  """
  if len(parameters) != 8:
    msg = "skorohod-olevsky initial state requires the eight parameters"
    raise TypeError(msg)
  if layout.row_width != _ROW_WIDTH:
    msg = "skorohod-olevsky initial state requires the 14-float row layout"
    raise ValueError(msg)
  rho0 = parameters[3]
  if not 0.0 < rho0 <= 1.0:
    msg = "skorohod-olevsky initial_relative_density must lie in (0, 1]"
    raise ValueError(msg)
  rows = np.zeros(layout.row_shape, dtype=np.float64)
  rows[:, 12] = rho0
  return rows


@dataclass(frozen=True, slots=True)
class SkorohodOlevskyBinding:
  """The registry binding object wiring the sintering law to the ABI."""

  def __call__(self, *parameters: float) -> np.ndarray:
    if len(parameters) != 8:
      msg = "skorohod-olevsky requires exactly eight calibration parameters"
      raise TypeError(msg)
    return skorohod_olevsky_calibration(*parameters)

  def descriptor_metadata(self) -> dict[str, object]:
    return skorohod_olevsky_metadata()

  def kernel(
    self,
    strains: np.ndarray,
    accepted_rows: np.ndarray,
    calibration: np.ndarray,
    signals: tuple[StatefulContinuumSignalInput, ...],
  ) -> StatefulContinuumKernelResult:
    return skorohod_olevsky_kernel(strains, accepted_rows, calibration, signals)

  def initial_state(
    self,
    parameters: tuple[float, ...],
    layout: OperatorStateLayout,
  ) -> np.ndarray:
    return skorohod_olevsky_initial_state(parameters, layout)


SKOROHOD_OLEVSKY_BINDING = SkorohodOlevskyBinding()
