"""Perzyna-branded viscoplasticity: a rate-INDEPENDENT J2 map with a time gate.

A pure, batched port of the legacy reference
``pyfem/materials/ViscoPlasticity.py`` (the numerical oracle per AGENTS.md)
onto the v2 stateful descriptor ABI with a declared ``signal_ports`` time
port. Read this first: **the legacy law is Perzyna-branded but integrates no
rate law.** Its local Newton solves the rate-independent J2 consistency
equation ``smises - eg3*deqpl - syield_iter = 0``
(``ViscoPlasticity.py:228``) — no rate term enters the residual; the fluidity
``gamma``, the rate exponent ``n``, and ``dtime`` only seed the initial guess
(:210-213), which the linear-residual Newton discards. Measured on the deck
constants (E = 2e5, nu = 0.3, syield = 250, hard = 1000, 12-step plastic
ramp): running the full path with gamma = 1e-4, n = 1 versus gamma = 1e2,
n = 2 changes the converged stress by 1.95e-16 relative — pure Newton-tolerance
slack (finding 20261009-agent-vp1-bug-viscoplasticity-stress-update-is-
rate-independent-j2-perzyna). The ONLY rate effect in the law is the
``dtime > 0`` gate (:205): a supra-yield step at constant time returns the
elastic trial response above the yield surface — a semantic trap the parity
battery pins explicitly. The backward-Euler/Perzyna claims of the legacy
docstring (:25-34, :76) and of ``examples/materials/viscoplasticity/README.md``
(:39-55: "fast loading -> higher peak stress") are false for the implemented
map; a true-Perzyna rate law is a deferred v2 feature, NOT this mission
(tower ruling, M68). This kernel replicates the implemented map statement for
statement; the module and registry names keep the legacy branding so decks
map one-to-one.

The law is 3D internally (6-Voigt ``[xx, yy, zz, yz, zx, xy]`` with
engineering shears); the continuum operator embeds the 3-Voigt port strain as
``[xx, yy, 0, 0, 0, xy]`` (a plane-strain embedding) and truncates
stress/tangent back to ``[xx, yy, xy]`` (the M25 seam).

Parameters (``parameter_names`` order): ``youngs_modulus`` E > 0,
``poisson_ratio`` -1 < nu < 0.5, ``initial_yield_stress`` syield > 0,
``hardening_slope`` hard >= 0 (the legacy default family is linear hardening,
``syield_current = syield + hard * eqplas`` :199), ``fluidity`` gamma > 0, and
``rate_exponent`` n > 0 (legacy defaults n = 1.0; gamma/n seed the discarded
initial guess only). The descriptor ABI fixes the parameter list per
descriptor, so all six are required; the legacy ctor debug ``print(self)``
(:91) is not replicated (no numeric consequence).

Time reaches the law exclusively through the declared identity signal port
(``port_id`` ``"time"``, ``signal_id`` ``"time"``), replacing the legacy
``self.solverStat.time`` read (:182) — there is no schedule back channel, and
the M39 derived-dtime binding is deliberately NOT used: state rows record the
bound time so restarts are sound (M49 doctrine).

State rows (14 floats per entity, flattened element x ip x slot):
``epsilon_e`` 6, ``epsilon_p`` 6, ``kappa`` 1 (the accumulated equivalent
plastic strain, annotated ``monotone-nondecreasing``), ``time`` 1. The legacy
``sigma`` history slot (:126, fetched at :178 and overwritten unread in both
branches) is dropped; the strain increment is the M25 identity
``dstrain = strains - (epsilon_e + epsilon_p)`` evaluated on the accepted row,
the same float64 subtraction the parity harness uses to drive the legacy
oracle. The initial state is all zero, matching the legacy constructor's
zeroed history commit (:126-132); the binding declares ``initial_state``
explicitly so the owner never relies on implicit zero-fill.

The kernel replicates the legacy ``getStress`` arithmetic statement for
statement (:191-294), including the predictor order
``sigma_trial = dot(ctang, eelas + dstrain)`` (:192-193 — VP's own order, not
the J2 kernel's accumulator form), the ``0.333333333333333`` hydrostatic
scaling and ``1.0 / smises`` reciprocal multiply of the flow direction
(:216-219), the local Newton with tolerance ``1e-8 * syield`` and at most 20
iterations (:223-240 — linear residual, so it converges in one update past
the seed check; the seeded guess survives only on marginal crossings within
the tolerance, a measure-zero band the FD legs avoid), and the elastic branch
(:284-287). Stress and every state slot match the legacy oracle bit for bit
on every committed path.

Two legacy behaviors are documented, not silently inherited — the first is
deliberately NOT replicated; the second was NOT replicated at the fork and
has since been repaired legacy-side:

- The no-convergence path warns and continues (:242-244 — dead code in
  practice, the residual is linear): the v3 kernel reports ``REJECT_STEP``
  with byte-equal trial rows instead, as it does for a non-finite strain
  batch, a non-finite bound time, or a non-finite predictor. Divergences from
  legacy are typed, not silent.
- The plastic-branch tangent used to add a spurious ``rate_factor``
  (:270: ``gamma * n * overstress**(n-1) * dtime / syield_current``) to the
  hardening term — the derivative of the discarded initial guess, not of the
  converged map (finding
  20261009-agent-vp1-bug-viscoplasticity-tangent-inconsistent-with-its-own-stress-upd).
  Against a central finite difference of the legacy law's own stress response
  at an amplified state (gamma = 1e4, dtime = 1e3, rate_factor A = 4.0e4), the
  coded tangent errs at 7.0e-2 relative of max|tang| while the
  rate-factor-free J2 consistent form matches to 1.6e-10 (M68 battery
  measurements; the M63 survey measured 6.59e-2 / 2.83e-8 on its own probe
  state). The v3 kernel writes the TRUE algorithmic tangent of the
  implemented map: ``effhdr = eg3 * hard / (eg3 + hard) - effg3`` with the
  legacy block construction order (:264-282) otherwise untouched — class
  ``algorithmic-symmetric`` (the coded tangent is also symmetric, just wrong;
  no taxonomy extension). Stress and state bookkeeping stay bitwise-identical
  to legacy. M67 repaired the legacy side to the same rate-factor-free form
  (commit 494f30c, merge 8a3eae8) and M76 flipped the divergence pins to
  repair-confirmed parity (b0ad9ed, merge 2ffcf20); the pre-repair pin
  record: v3 == rate-factor-free form bitwise, legacy == coded form bitwise,
  the relative gap pinned per leg (dormant ~2.3e-11 of effhdr at deck
  constants with dtime ~ 1, dominant 7.0e-2 at the amplified state).

Tangent class: ``algorithmic-symmetric`` — the Jacobian channel is nonlinear
(never ``linear``) and symmetric; the driver re-factorizes every Newton
iteration (counter-asserted in the battery).

The virgin probe (zero strain, zero bound time, zero rows) takes the elastic
branch with a zero increment and returns the accepted rows byte-equal.
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

PERZYNA_VISCOPLASTICITY_STATE_SCHEMA = "pyfem-v3-perzyna-viscoplastic-state-v1"

_TIME_PORT_ID = "time"

# Legacy constants (ViscoPlasticity.py:139-140): the return-map convergence
# tolerance rides the calibration vector; the iteration cap is fixed here.
_RETURN_MAP_TOLERANCE = 1.0e-8
_LOCAL_NEWTON_LIMIT = 20
_ROW_WIDTH = 14

# Flat calibration vector layout: ten scalars, then the 6x6 elastic tangent in
# C order — 46 floats.
_CALIBRATION_SIZE = 10 + 36


def perzyna_viscoplasticity_metadata() -> dict[str, object]:
  """Return the canonical v2 descriptor metadata for the rate-gated J2 law."""
  return {
    "schema": STATEFUL_MATERIAL_DESCRIPTOR_SCHEMA,
    "law": "perzyna-viscoplasticity",
    "stress_state": "plane-strain",
    "parameter_names": [
      "youngs_modulus",
      "poisson_ratio",
      "initial_yield_stress",
      "hardening_slope",
      "fluidity",
      "rate_exponent",
    ],
    "parameter_dtype": "float64",
    "stress_voigt_order": ["xx", "yy", "xy"],
    "strain_shear_convention": "engineering",
    "internal_voigt_order": ["xx", "yy", "zz", "yz", "zx", "xy"],
    "tangent_class": "algorithmic-symmetric",
    "state_schema": PERZYNA_VISCOPLASTICITY_STATE_SCHEMA,
    "state_slots": [
      {
        "name": "epsilon_e",
        "width": 6,
        "dtype": "float64",
        "lifetime": "accepted-trial",
      },
      {
        "name": "epsilon_p",
        "width": 6,
        "dtype": "float64",
        "lifetime": "accepted-trial",
      },
      {
        "name": "kappa",
        "width": 1,
        "dtype": "float64",
        "lifetime": "accepted-trial",
        "annotation": "monotone-nondecreasing",
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


def perzyna_viscoplasticity_calibration(
  youngs_modulus: float,
  poisson_ratio: float,
  initial_yield_stress: float,
  hardening_slope: float,
  fluidity: float,
  rate_exponent: float,
) -> np.ndarray:
  """Pack the law's flat calibration vector with legacy-identical arithmetic."""
  for label, value in (
    ("youngs_modulus", youngs_modulus),
    ("poisson_ratio", poisson_ratio),
    ("initial_yield_stress", initial_yield_stress),
    ("hardening_slope", hardening_slope),
    ("fluidity", fluidity),
    ("rate_exponent", rate_exponent),
  ):
    if type(value) is not float:
      msg = f"perzyna viscoplasticity {label} must be an exact float"
      raise TypeError(msg)
    if not np.isfinite(value):
      msg = f"perzyna viscoplasticity {label} must be finite"
      raise ValueError(msg)
  if youngs_modulus <= 0.0:
    msg = "perzyna viscoplasticity youngs_modulus must be positive"
    raise ValueError(msg)
  if not -1.0 < poisson_ratio < 0.5:
    msg = "perzyna viscoplasticity poisson_ratio must lie between -1 and 0.5"
    raise ValueError(msg)
  if initial_yield_stress <= 0.0:
    msg = "perzyna viscoplasticity initial_yield_stress must be positive"
    raise ValueError(msg)
  if hardening_slope < 0.0:
    msg = "perzyna viscoplasticity hardening_slope must be non-negative"
    raise ValueError(msg)
  if fluidity <= 0.0:
    msg = "perzyna viscoplasticity fluidity must be positive"
    raise ValueError(msg)
  if rate_exponent <= 0.0:
    msg = "perzyna viscoplasticity rate_exponent must be positive"
    raise ValueError(msg)

  # Legacy constructor expression order (ViscoPlasticity.py:107-123).
  ebulk3 = youngs_modulus / (1.0 - 2.0 * poisson_ratio)
  eg2 = youngs_modulus / (1.0 + poisson_ratio)
  eg = 0.5 * eg2
  eg3 = 3.0 * eg
  elam = (ebulk3 - eg2) / 3.0

  ctang = np.zeros(shape=(6, 6))
  ctang[:3, :3] = elam
  ctang[0, 0] += eg2
  ctang[1, 1] = ctang[0, 0]
  ctang[2, 2] = ctang[0, 0]
  ctang[3, 3] = eg
  ctang[4, 4] = ctang[3, 3]
  ctang[5, 5] = ctang[3, 3]

  return np.concatenate(
    (
      np.array(
        [
          eg,
          eg2,
          eg3,
          ebulk3,
          elam,
          initial_yield_stress,
          hardening_slope,
          fluidity,
          rate_exponent,
          _RETURN_MAP_TOLERANCE,
        ],
        dtype=np.float64,
      ),
      ctang.reshape(-1),
    )
  )


def _von_mises(sigma: np.ndarray) -> float:
  """The legacy MatUtils.vonMisesStress arithmetic, statement for statement."""
  smises = (
    (sigma[0] - sigma[1]) * (sigma[0] - sigma[1])
    + (sigma[1] - sigma[2]) * (sigma[1] - sigma[2])
    + (sigma[2] - sigma[0]) * (sigma[2] - sigma[0])
  )
  smises += 6.0 * np.dot(sigma[3:], sigma[3:])
  return float(np.sqrt(0.5 * smises))


def _reject(accepted_rows: np.ndarray) -> StatefulContinuumKernelResult:
  entity_count = accepted_rows.shape[0]
  return StatefulContinuumKernelResult(
    stresses=np.zeros((entity_count, 6), dtype=np.float64),
    tangents=np.zeros((entity_count, 6, 6), dtype=np.float64),
    trial_rows=np.array(accepted_rows, dtype=np.float64, copy=True),
    status=EvaluationStatus.REJECT_STEP,
  )


def perzyna_viscoplasticity_kernel(
  strains: np.ndarray,
  accepted_rows: np.ndarray,
  calibration: np.ndarray,
  signals: tuple[StatefulContinuumSignalInput, ...],
) -> StatefulContinuumKernelResult:
  """Evaluate the rate-gated J2 law over flattened state entities.

  ``strains`` holds total engineering 6-Voigt strain per entity;
  ``accepted_rows`` holds the committed 14-float rows; ``signals`` carries
  exactly the declared identity time port. The strain increment is
  ``strains - (epsilon_e + epsilon_p)`` from the accepted row, the identity
  the parity harness also uses to drive the legacy oracle; the time increment
  is the legacy ``time_new - time_old`` over the bound signal value and the
  row's committed time. Expected numerical failures report ``REJECT_STEP``
  with byte-equal trial rows, never exceptions.
  """
  if (
    type(calibration) is not np.ndarray
    or calibration.dtype != np.dtype(np.float64)
    or calibration.dtype.metadata is not None
    or calibration.shape != (_CALIBRATION_SIZE,)
  ):
    msg = "perzyna viscoplasticity kernel requires the packed calibration vector"
    raise TypeError(msg)
  if (
    type(signals) is not tuple
    or len(signals) != 1
    or type(signals[0]) is not StatefulContinuumSignalInput
    or signals[0].port_id != _TIME_PORT_ID
  ):
    msg = "perzyna viscoplasticity kernel requires exactly the declared time port"
    raise TypeError(msg)
  entity_count = strains.shape[0]
  if strains.shape != (entity_count, 6) or accepted_rows.shape != (
    entity_count,
    _ROW_WIDTH,
  ):
    msg = "perzyna viscoplasticity kernel requires (n, 6) strains and (n, 14) rows"
    raise ValueError(msg)
  if not bool(np.isfinite(strains).all()):
    return _reject(accepted_rows)
  time_new = float(signals[0].values[0])
  if not np.isfinite(time_new):
    return _reject(accepted_rows)

  eg = calibration[0]
  eg3 = calibration[2]
  ebulk3 = calibration[3]
  syield = calibration[5]
  hard = calibration[6]
  gamma = calibration[7]
  n = calibration[8]
  tolerance = calibration[9]
  ctang = calibration[10:46].reshape(6, 6)

  stresses = np.empty((entity_count, 6), dtype=np.float64)
  tangents = np.empty((entity_count, 6, 6), dtype=np.float64)
  trial_rows = np.empty((entity_count, _ROW_WIDTH), dtype=np.float64)

  with np.errstate(over="ignore", invalid="ignore"):
    for index in range(entity_count):
      row = accepted_rows[index]
      # Legacy getStress statement order (ViscoPlasticity.py:191-294).
      dtime = time_new - row[13]
      dstrain = strains[index] - (row[0:6] + row[6:12])
      eelas_trial = row[0:6] + dstrain
      sigma_trial = np.dot(ctang, eelas_trial)
      smises = _von_mises(sigma_trial)
      syield_current = syield + hard * row[12]
      if not bool(np.isfinite(sigma_trial).all()) or not np.isfinite(smises):
        return _reject(accepted_rows)
      eplas = np.array(row[6:12], copy=True)
      eqplas = float(row[12])

      if smises > syield_current and dtime > 0.0:
        overstress = (smises - syield_current) / syield_current
        gamma_eff = gamma * (overstress**n)
        deqpl = gamma_eff * dtime
        shydro = 0.333333333333333 * (sigma_trial[0] + sigma_trial[1] + sigma_trial[2])
        flow = np.array(sigma_trial, copy=True)
        flow[:3] = flow[:3] - shydro
        flow *= 1.0 / smises

        converged = False
        for _ in range(_LOCAL_NEWTON_LIMIT):
          syield_iter = syield + hard * (eqplas + deqpl)
          residual = smises - eg3 * deqpl - syield_iter
          if abs(residual) < tolerance * syield:
            converged = True
            break
          jacobian = -eg3 - hard
          if jacobian == 0.0:
            # Unreachable from validated calibrations (jacobian = -eg3 - hard
            # <= -eg3 < 0); a degenerate hand-packed vector rejects typed
            # instead of dividing by zero.
            break
          deqpl_inc = -residual / jacobian
          deqpl += deqpl_inc
        if not converged or not np.isfinite(deqpl):
          # Legacy warns and continues (:242-244); the v3 law rejects typed.
          return _reject(accepted_rows)

        eplas[:3] += 1.5 * flow[:3] * deqpl
        eplas[3:] += 3.0 * flow[3:] * deqpl
        eelas = np.empty(6, dtype=np.float64)
        eelas[:3] = eelas_trial[:3] - 1.5 * flow[:3] * deqpl
        eelas[3:] = eelas_trial[3:] - 3.0 * flow[3:] * deqpl
        eqplas += deqpl

        syield_final = syield + hard * eqplas
        sigma = flow * syield_final
        sigma[:3] += shydro

        # The TRUE algorithmic tangent of the implemented rate-independent
        # map: the legacy block construction order (:264-282) with the
        # spurious rate_factor term deleted (module docstring). M67 repaired
        # the legacy side identically (494f30c), so the fork-base divergence
        # this kernel pinned is now repair-confirmed parity (M76, b0ad9ed).
        effg = eg * syield_final / smises
        effg2 = 2.0 * effg
        effg3 = 3.0 * effg
        efflam = (ebulk3 - effg2) / 3.0
        effhdr = eg3 * hard / (eg3 + hard) - effg3
        tang = np.zeros(shape=(6, 6))
        tang[:3, :3] = efflam
        for i in range(3):
          tang[i, i] += effg2
          tang[i + 3, i + 3] += effg
        tang += effhdr * np.outer(flow, flow)
        if not bool(np.isfinite(sigma).all()) or not bool(np.isfinite(tang).all()):
          return _reject(accepted_rows)
      else:
        # Elastic branch, including the dtime == 0 gate above yield (:284-287).
        eelas = eelas_trial
        sigma = sigma_trial
        tang = np.array(ctang, copy=True)

      trial_rows[index, 0:6] = eelas
      trial_rows[index, 6:12] = eplas
      trial_rows[index, 12] = eqplas
      trial_rows[index, 13] = time_new
      stresses[index] = sigma
      tangents[index] = tang

  return StatefulContinuumKernelResult(
    stresses=stresses,
    tangents=tangents,
    trial_rows=trial_rows,
    status=EvaluationStatus.OK,
  )


def perzyna_viscoplasticity_initial_state(
  parameters: tuple[float, ...],
  layout: OperatorStateLayout,
) -> np.ndarray:
  """Bind the all-zero initial rows of the legacy constructor explicitly."""
  if len(parameters) != 6:
    msg = "perzyna viscoplasticity initial state requires the six parameters"
    raise TypeError(msg)
  if layout.row_width != _ROW_WIDTH:
    msg = "perzyna viscoplasticity initial state requires the 14-float row layout"
    raise ValueError(msg)
  return np.zeros(layout.row_shape, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class PerzynaViscoPlasticityBinding:
  """The registry binding object wiring the rate-gated J2 law to the ABI."""

  def __call__(self, *parameters: float) -> np.ndarray:
    if len(parameters) != 6:
      msg = "perzyna viscoplasticity requires exactly six calibration parameters"
      raise TypeError(msg)
    return perzyna_viscoplasticity_calibration(*parameters)

  def descriptor_metadata(self) -> dict[str, object]:
    return perzyna_viscoplasticity_metadata()

  def kernel(
    self,
    strains: np.ndarray,
    accepted_rows: np.ndarray,
    calibration: np.ndarray,
    signals: tuple[StatefulContinuumSignalInput, ...],
  ) -> StatefulContinuumKernelResult:
    return perzyna_viscoplasticity_kernel(strains, accepted_rows, calibration, signals)

  def initial_state(
    self,
    parameters: tuple[float, ...],
    layout: OperatorStateLayout,
  ) -> np.ndarray:
    return perzyna_viscoplasticity_initial_state(parameters, layout)


PERZYNA_VISCOPLASTICITY_BINDING = PerzynaViscoPlasticityBinding()
