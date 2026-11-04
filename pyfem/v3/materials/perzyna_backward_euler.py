"""True backward-Euler Perzyna J2 viscoplasticity with isotropic hardening.

This is a distinct constitutive law, not a repair or reinterpretation of
``perzyna-viscoplasticity``. That legacy-mapped key remains a rate-independent
J2 consistency solve with a time gate. Here the Perzyna rate equation is part
of the converged residual. The overstress form follows P. Perzyna,
"Fundamental Problems in Viscoplasticity", Advances in Applied Mechanics 9
(1966), 243--377, doi:10.1016/S0065-2156(08)70009-7; the implicit radial
return and algorithmic-linearization construction follow J.C. Simo and
T.J.R. Hughes, *Computational Inelasticity*, Springer (1998), chapters 2 and
5, doi:10.1007/b98904.

With ``q`` the von Mises stress, ``y = syield + hard*kappa``, and Macauley
brackets ``<x> = max(x, 0)``, equivalent viscoplastic strain evolves as

  kappa_dot = fluidity * <(q - y) / y>**rate_exponent.

Associative J2 flow makes one backward-Euler step radial in the elastic trial
deviator. Writing ``l = Delta kappa``, ``a = fluidity*dtime``, ``qtr`` for the
trial von Mises stress, ``yn = syield + hard*kappa_n``, and
``A = 3*G + hard``, the scalar residual is

  r(l) = l - a * <(qtr - A*l - yn) / (yn + hard*l)>**n = 0
  q(l) = qtr - 3*G*l
  y(l) = yn + hard*l.

For a positive trial overstress and positive ``a``, ``r(0) < 0`` and
``r(f_trial/A) = f_trial/A > 0``. On the active branch define
``phi = (q-y)/y``, ``f = q-y``, and ``B = A*y + hard*f = 3*G*y + hard*q``.
Then

  dr/dl = 1 + a*n*phi**(n-1) * B/y**2 > 0,

so the bracket contains exactly one root. A safeguarded Newton iteration uses
this analytical derivative and falls back to bisection whenever a proposed
iterate leaves the bracket. A zero time increment is exactly elastic; a
negative or non-finite increment is a typed step rejection. As ``a -> 0`` the
root and response approach the elastic predictor, while as ``a -> infinity``
the normalized overstress approaches zero and the root approaches the
rate-independent J2 consistency solution ``f_trial/A``.

The consistent tangent differentiates the same implicit residual, not an
explicit predictor. With ``w = a*n*phi**(n-1)``,

  dl/dqtr = w*y / (y**2 + w*B)
  dq/dqtr = 1 - 3*G*dl/dqtr.

If ``N = dev(sigma_trial)/qtr``, the returned engineering-Voigt stress is
``sigma = p_trial*I + q*N`` and its algorithmic tangent is the radial-elastic
block plus ``(3*G*dq/dqtr - 3*G*q/qtr) * outer(N, N)``. It is exactly
symmetric. Elastic steps return the elastic stiffness.

State rows contain 13 floats: ``epsilon_e`` (6), ``epsilon_p`` (6), and the
monotone equivalent viscoplastic strain ``kappa`` (1). The derived ``dtime``
signal is computed from the previous committed program point, so no absolute
time slot is needed for run-boundary restart equivalence. Inputs are never
mutated; every rejection returns trial rows byte-equal to the accepted rows.
The internal convention is 6-Voigt ``[xx, yy, zz, yz, zx, xy]`` with
engineering shears; the Q8 operator supplies its usual plane-strain embedding.
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

BACKWARD_EULER_PERZYNA_STATE_SCHEMA = "pyfem-v3-backward-euler-perzyna-state-v1"

_DTIME_PORT_ID = "dtime"
_ROW_WIDTH = 13
_RETURN_MAP_TOLERANCE = 1.0e-12
_LOCAL_NEWTON_LIMIT = 100
_CALIBRATION_SIZE = 10 + 36


def backward_euler_perzyna_metadata() -> dict[str, object]:
  """Return the frozen-v2 descriptor of the true implicit Perzyna law."""
  return {
    "schema": STATEFUL_MATERIAL_DESCRIPTOR_SCHEMA,
    "law": "backward-euler-perzyna",
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
    "state_schema": BACKWARD_EULER_PERZYNA_STATE_SCHEMA,
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
    ],
    "signal_ports": [
      {
        "port_id": _DTIME_PORT_ID,
        "signal_id": _DTIME_PORT_ID,
        "derivative_coordinate_ids": ["time"],
      }
    ],
  }


def backward_euler_perzyna_calibration(
  youngs_modulus: float,
  poisson_ratio: float,
  initial_yield_stress: float,
  hardening_slope: float,
  fluidity: float,
  rate_exponent: float,
) -> np.ndarray:
  """Validate and pack the six physical parameters and elastic constants."""
  values = (
    ("youngs_modulus", youngs_modulus),
    ("poisson_ratio", poisson_ratio),
    ("initial_yield_stress", initial_yield_stress),
    ("hardening_slope", hardening_slope),
    ("fluidity", fluidity),
    ("rate_exponent", rate_exponent),
  )
  for label, value in values:
    if type(value) is not float:
      msg = f"backward-Euler Perzyna {label} must be an exact float"
      raise TypeError(msg)
    if not np.isfinite(value):
      msg = f"backward-Euler Perzyna {label} must be finite"
      raise ValueError(msg)
  if youngs_modulus <= 0.0:
    msg = "backward-Euler Perzyna youngs_modulus must be positive"
    raise ValueError(msg)
  if not -1.0 < poisson_ratio < 0.5:
    msg = "backward-Euler Perzyna poisson_ratio must lie between -1 and 0.5"
    raise ValueError(msg)
  if initial_yield_stress <= 0.0:
    msg = "backward-Euler Perzyna initial_yield_stress must be positive"
    raise ValueError(msg)
  if hardening_slope < 0.0:
    msg = "backward-Euler Perzyna hardening_slope must be non-negative"
    raise ValueError(msg)
  if fluidity <= 0.0:
    msg = "backward-Euler Perzyna fluidity must be positive"
    raise ValueError(msg)
  if rate_exponent <= 0.0:
    msg = "backward-Euler Perzyna rate_exponent must be positive"
    raise ValueError(msg)

  ebulk3 = youngs_modulus / (1.0 - 2.0 * poisson_ratio)
  eg2 = youngs_modulus / (1.0 + poisson_ratio)
  eg = 0.5 * eg2
  eg3 = 3.0 * eg
  elam = (ebulk3 - eg2) / 3.0
  ctang = np.zeros((6, 6), dtype=np.float64)
  ctang[:3, :3] = elam
  ctang[0, 0] += eg2
  ctang[1, 1] = ctang[0, 0]
  ctang[2, 2] = ctang[0, 0]
  ctang[3, 3] = eg
  ctang[4, 4] = ctang[3, 3]
  ctang[5, 5] = ctang[3, 3]
  scalars = np.array(
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
  )
  return np.concatenate((scalars, ctang.reshape(-1)))


def _von_mises(stress: np.ndarray) -> float:
  normal = (
    (stress[0] - stress[1]) ** 2
    + (stress[1] - stress[2]) ** 2
    + (stress[2] - stress[0]) ** 2
  )
  return float(np.sqrt(0.5 * (normal + 6.0 * np.dot(stress[3:], stress[3:]))))


def _perzyna_increment(
  trial_mises: float,
  accepted_yield: float,
  eg3: float,
  hard: float,
  fluidity: float,
  exponent: float,
  dtime: float,
  tolerance: float,
) -> tuple[float, float] | None:
  """Solve the scalar backward-Euler residual and return ``l, dq/dqtr``."""
  trial_overstress = trial_mises - accepted_yield
  stiffness = eg3 + hard
  upper = trial_overstress / stiffness
  rate_scale = fluidity * dtime
  trial_ratio = trial_overstress / accepted_yield
  explicit = rate_scale * trial_ratio**exponent
  if np.isfinite(explicit) and 0.0 < explicit < upper:
    increment = float(explicit)
  else:
    increment = 0.5 * upper
  lower = 0.0
  converged = False
  with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
    for _ in range(_LOCAL_NEWTON_LIMIT):
      overstress = trial_overstress - stiffness * increment
      current_yield = accepted_yield + hard * increment
      ratio = max(overstress / current_yield, 0.0)
      rate = rate_scale * ratio**exponent
      residual = increment - rate
      scale = tolerance * max(upper, abs(increment), 1.0e-30)
      if residual > 0.0:
        upper = increment
      else:
        lower = increment
      if abs(residual) <= scale:
        converged = True
        break
      if not np.isfinite(residual):
        candidate = 0.5 * (lower + upper)
      else:
        b_term = stiffness * current_yield + hard * overstress
        rate_derivative = rate_scale * exponent * ratio ** (exponent - 1.0)
        jacobian = 1.0 + rate_derivative * b_term / current_yield**2
        candidate = increment - residual / jacobian
        if not np.isfinite(candidate) or not lower < candidate < upper:
          candidate = 0.5 * (lower + upper)
      if candidate == increment:
        converged = (upper - lower) <= scale
        break
      increment = float(candidate)
    if not converged:
      bracket_tolerance = tolerance * max(trial_overstress / stiffness, 1.0e-30)
      converged = (upper - lower) <= bracket_tolerance
    if not converged:
      return None
    increment = float(np.clip(increment, 0.0, trial_overstress / stiffness))
    overstress = trial_overstress - stiffness * increment
    current_yield = accepted_yield + hard * increment
    ratio = max(overstress / current_yield, 0.0)
    b_term = stiffness * current_yield + hard * overstress
    rate_derivative = rate_scale * exponent * ratio ** (exponent - 1.0)
    if np.isposinf(rate_derivative):
      increment_derivative = current_yield / b_term
    else:
      denominator = current_yield**2 + rate_derivative * b_term
      increment_derivative = rate_derivative * current_yield / denominator
    q_derivative = 1.0 - eg3 * increment_derivative
  if not np.isfinite(increment) or not np.isfinite(q_derivative):
    return None
  return increment, float(q_derivative)


def _reject(accepted_rows: np.ndarray) -> StatefulContinuumKernelResult:
  entity_count = accepted_rows.shape[0]
  return StatefulContinuumKernelResult(
    stresses=np.zeros((entity_count, 6), dtype=np.float64),
    tangents=np.zeros((entity_count, 6, 6), dtype=np.float64),
    trial_rows=np.array(accepted_rows, dtype=np.float64, copy=True),
    status=EvaluationStatus.REJECT_STEP,
  )


def backward_euler_perzyna_kernel(
  strains: np.ndarray,
  accepted_rows: np.ndarray,
  calibration: np.ndarray,
  signals: tuple[StatefulContinuumSignalInput, ...],
) -> StatefulContinuumKernelResult:
  """Evaluate the true backward-Euler Perzyna map without mutating inputs."""
  if (
    type(calibration) is not np.ndarray
    or calibration.dtype != np.dtype(np.float64)
    or calibration.dtype.metadata is not None
    or calibration.shape != (_CALIBRATION_SIZE,)
  ):
    msg = "backward-Euler Perzyna kernel requires the packed calibration vector"
    raise TypeError(msg)
  if (
    type(signals) is not tuple
    or len(signals) != 1
    or type(signals[0]) is not StatefulContinuumSignalInput
    or signals[0].port_id != _DTIME_PORT_ID
  ):
    msg = "backward-Euler Perzyna kernel requires exactly the declared dtime port"
    raise TypeError(msg)
  entity_count = strains.shape[0]
  if strains.shape != (entity_count, 6) or accepted_rows.shape != (
    entity_count,
    _ROW_WIDTH,
  ):
    msg = "backward-Euler Perzyna kernel requires (n, 6) strains and (n, 13) rows"
    raise ValueError(msg)
  dtime = float(signals[0].values[0])
  if (
    not bool(np.isfinite(strains).all())
    or not bool(np.isfinite(accepted_rows).all())
    or not bool(np.isfinite(calibration).all())
    or not np.isfinite(dtime)
    or dtime < 0.0
  ):
    return _reject(accepted_rows)

  eg = calibration[0]
  eg3 = calibration[2]
  ebulk3 = calibration[3]
  syield = calibration[5]
  hard = calibration[6]
  fluidity = calibration[7]
  exponent = calibration[8]
  tolerance = calibration[9]
  ctang = calibration[10:46].reshape(6, 6)
  stresses = np.empty((entity_count, 6), dtype=np.float64)
  tangents = np.empty((entity_count, 6, 6), dtype=np.float64)
  trial_rows = np.empty((entity_count, _ROW_WIDTH), dtype=np.float64)

  with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
    for index in range(entity_count):
      row = accepted_rows[index]
      dstrain = strains[index] - (row[0:6] + row[6:12])
      eelas_trial = row[0:6] + dstrain
      sigma_trial = np.dot(ctang, eelas_trial)
      trial_mises = _von_mises(sigma_trial)
      accepted_yield = syield + hard * row[12]
      if not bool(np.isfinite(sigma_trial).all()) or not np.isfinite(trial_mises):
        return _reject(accepted_rows)
      eplas = np.array(row[6:12], copy=True)
      kappa = float(row[12])
      if trial_mises > accepted_yield and dtime > 0.0:
        returned = _perzyna_increment(
          trial_mises,
          accepted_yield,
          eg3,
          hard,
          fluidity,
          exponent,
          dtime,
          tolerance,
        )
        if returned is None:
          return _reject(accepted_rows)
        increment, q_derivative = returned
        hydrostatic = (sigma_trial[0] + sigma_trial[1] + sigma_trial[2]) / 3.0
        flow = np.array(sigma_trial, copy=True)
        flow[:3] -= hydrostatic
        flow *= 1.0 / trial_mises
        eplas[:3] += 1.5 * flow[:3] * increment
        eplas[3:] += 3.0 * flow[3:] * increment
        eelas = np.empty(6, dtype=np.float64)
        eelas[:3] = eelas_trial[:3] - 1.5 * flow[:3] * increment
        eelas[3:] = eelas_trial[3:] - 3.0 * flow[3:] * increment
        kappa += increment
        mises = trial_mises - eg3 * increment
        sigma = flow * mises
        sigma[:3] += hydrostatic

        effective_shear = eg * mises / trial_mises
        effective_shear2 = 2.0 * effective_shear
        effective_shear3 = 3.0 * effective_shear
        effective_lambda = (ebulk3 - effective_shear2) / 3.0
        normal_coefficient = eg3 * q_derivative - effective_shear3
        tangent = np.zeros((6, 6), dtype=np.float64)
        tangent[:3, :3] = effective_lambda
        for component in range(3):
          tangent[component, component] += effective_shear2
          tangent[component + 3, component + 3] += effective_shear
        tangent += normal_coefficient * np.outer(flow, flow)
      else:
        eelas = eelas_trial
        sigma = sigma_trial
        tangent = np.array(ctang, copy=True)

      if not bool(np.isfinite(sigma).all()) or not bool(np.isfinite(tangent).all()):
        return _reject(accepted_rows)
      trial_rows[index, 0:6] = eelas
      trial_rows[index, 6:12] = eplas
      trial_rows[index, 12] = kappa
      stresses[index] = sigma
      tangents[index] = tangent

  return StatefulContinuumKernelResult(
    stresses=stresses,
    tangents=tangents,
    trial_rows=trial_rows,
    status=EvaluationStatus.OK,
  )


def backward_euler_perzyna_initial_state(
  parameters: tuple[float, ...],
  layout: OperatorStateLayout,
) -> np.ndarray:
  """Initialize all strain and hardening state explicitly to zero."""
  if len(parameters) != 6:
    msg = "backward-Euler Perzyna initial state requires six parameters"
    raise TypeError(msg)
  if layout.row_width != _ROW_WIDTH:
    msg = "backward-Euler Perzyna initial state requires the 13-float row layout"
    raise ValueError(msg)
  return np.zeros(layout.row_shape, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class BackwardEulerPerzynaBinding:
  """Registry binding for the distinct implicit Perzyna constitutive law."""

  def __call__(self, *parameters: float) -> np.ndarray:
    if len(parameters) != 6:
      msg = "backward-Euler Perzyna requires exactly six calibration parameters"
      raise TypeError(msg)
    return backward_euler_perzyna_calibration(*parameters)

  def descriptor_metadata(self) -> dict[str, object]:
    return backward_euler_perzyna_metadata()

  def kernel(
    self,
    strains: np.ndarray,
    accepted_rows: np.ndarray,
    calibration: np.ndarray,
    signals: tuple[StatefulContinuumSignalInput, ...],
  ) -> StatefulContinuumKernelResult:
    return backward_euler_perzyna_kernel(strains, accepted_rows, calibration, signals)

  def initial_state(
    self,
    parameters: tuple[float, ...],
    layout: OperatorStateLayout,
  ) -> np.ndarray:
    return backward_euler_perzyna_initial_state(parameters, layout)


BACKWARD_EULER_PERZYNA_BINDING = BackwardEulerPerzynaBinding()
