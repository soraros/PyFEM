"""Small-strain rate-dependent crystal plasticity on complete cubic slip families.

This is a fresh literature-based implementation, not a port of the removed
legacy ``Crystal.py`` law. Resolved shear stress and viscoplastic slip follow
the Schmid power law of Peirce, Asaro and Needleman, *Material rate dependence
and localized deformation in crystalline solids*, Acta Metallurgica 31 (1983),
1951--1976, doi:10.1016/0001-6160(83)90014-7. Slip resistance follows the
integrated Kocks--Mecking saturation law reviewed by Kocks and Mecking,
*Physics and phenomenology of strain hardening: the FCC case*, Progress in
Materials Science 48 (2003), 171--273,
doi:10.1016/S0079-6425(02)00003-8::

  d(tau_c)/d(Gamma) = h0 * (1 - tau_c / taus)**a
  Gamma_alpha = sum_beta Q_alpha_beta * gamma_cum_beta
  Q_alpha_beta = delta_alpha_beta + qab * (1 - delta_alpha_beta).

The closed-form resistance is evaluated at the CURRENT Newton iterate. Thus
latent and self hardening both evolve inside the return map; no constant
hardening matrix is used. For ``a == 1`` the integral is exponential; otherwise
it is algebraic with ``y = 1 - tau_c / taus`` and
``y**(1-a) = y0**(1-a) + (a-1)*h0*Gamma/taus``. The exponent domain is
``a >= 1`` so saturation is approached smoothly.

The default FCC family is the complete {111}<110> set (4 planes times 3
directions, 12 systems). ``crystal_structure = 1`` selects the complete BCC
{110}<111> set (6 planes times 2 directions, also 12 systems). Systems are
generated from cubic symmetry, not authored as subsets. Plane normals and slip
directions must be orthonormal; invalid frames fail closed with the coded
``nonorthogonal-slip-frame`` diagnostic and are never silently projected.

For each system, ``g = gamma0 * sign(tau) * abs(tau/tau_c)**m``. The theta
method solves, for total step slip increments ``x = Delta gamma``,

  r = x - dtime * ((1-theta)*g_accepted + theta*g(x)) = 0
  sigma = C : (strain - sum_alpha S_alpha * (gamma_alpha + x_alpha)),

with ``S`` the Schmid vectors in engineering 6-Voigt order
``[xx, yy, zz, yz, zx, xy]``. Every residual evaluation uses the stress,
resistance, and rate at the current ``x``. The old rate comes only from the
accepted row. The consistent tangent is

  D = C - C*S * J^-1 * theta*dtime*diag(dg/dtau) * S^T*C
  J = dr/dx,

so the implicit rate factor is present in full. Its general form is
nonsymmetric when system rates or latent-hardening derivatives differ;
descriptor class ``algorithmic-nonsymmetric`` records that property rather
than imposing symmetry.

State rows contain ``4*nslip + 7`` floats: critical stress ``tau_c[nslip]``,
signed accumulated slip ``gamma[nslip]``, resolved stress ``tau[nslip]``,
absolute accumulated slip ``gamma_cum[nslip]``, stress ``sigma[6]``, and total
absolute slip ``gamma_total[1]``. Initial rows set every ``tau_c`` to ``tau0``.
The time step arrives only through the derived ``dtime`` signal port. The law
is 3D internally; the continuum operator supplies the usual plane-strain
embedding for two-dimensional models.

Non-finite data, negative time increments, singular or non-convergent local
Newton solves, and non-finite results report ``REJECT_STEP`` with trial rows
byte-equal to the accepted rows. Accepted inputs are never mutated.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from pyfem.v3.compile.contracts import (
  STATEFUL_MATERIAL_DESCRIPTOR_SCHEMA,
  StatefulContinuumKernelResult,
  StatefulContinuumSignalInput,
)
from pyfem.v3.model.operator import EvaluationStatus, OperatorStateLayout

CRYSTAL_PLASTICITY_STATE_SCHEMA = "pyfem-v3-crystal-plasticity-state-v1"

_DTIME_PORT_ID = "dtime"
_LOCAL_NEWTON_LIMIT = 50
_CALIBRATION_FIXED = 10 + 36
_PARAMETER_COUNT = 13
_SLIP_SYSTEM_COUNT = 12


class SlipFrameError(ValueError):
  """A coded invalid Schmid-frame diagnostic raised while binding a law."""

  code = "nonorthogonal-slip-frame"


def crystal_plasticity_metadata() -> dict[str, object]:
  """Return the frozen-v2 descriptor metadata for complete cubic crystals."""
  slip_width: dict[str, object] = {"parameter": "slip_system_count"}
  return {
    "schema": STATEFUL_MATERIAL_DESCRIPTOR_SCHEMA,
    "law": "crystal-plasticity",
    "stress_state": "plane-strain",
    "parameter_names": [
      "youngs_modulus",
      "poisson_ratio",
      "crystal_structure",
      "slip_system_count",
      "reference_slip_rate",
      "rate_exponent",
      "initial_critical_stress",
      "hardening_modulus",
      "saturation_stress",
      "hardening_exponent",
      "latent_hardening_ratio",
      "theta",
      "newton_tolerance",
    ],
    "parameter_dtype": "float64",
    "stress_voigt_order": ["xx", "yy", "xy"],
    "strain_shear_convention": "engineering",
    "internal_voigt_order": ["xx", "yy", "zz", "yz", "zx", "xy"],
    "tangent_class": "algorithmic-nonsymmetric",
    "state_schema": CRYSTAL_PLASTICITY_STATE_SCHEMA,
    "state_slots": [
      {
        "name": "tau_c",
        "width": dict(slip_width),
        "dtype": "float64",
        "lifetime": "accepted-trial",
      },
      {
        "name": "gamma",
        "width": dict(slip_width),
        "dtype": "float64",
        "lifetime": "accepted-trial",
      },
      {
        "name": "tau",
        "width": dict(slip_width),
        "dtype": "float64",
        "lifetime": "accepted-trial",
      },
      {
        "name": "gamma_cum",
        "width": dict(slip_width),
        "dtype": "float64",
        "lifetime": "accepted-trial",
        "annotation": "monotone-nondecreasing",
      },
      {
        "name": "sigma",
        "width": 6,
        "dtype": "float64",
        "lifetime": "accepted-trial",
      },
      {
        "name": "gamma_total",
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


def _canonical_integer_vector(vector: tuple[int, ...]) -> tuple[int, ...]:
  for value in vector:
    if value:
      return vector if value > 0 else tuple(-item for item in vector)
  msg = "cubic slip-family vectors must be nonzero"
  raise ValueError(msg)


def _cubic_integer_family(
  base: tuple[int, int, int],
) -> tuple[tuple[int, int, int], ...]:
  vectors = {
    _canonical_integer_vector(vector)
    for vector in itertools.permutations(base)
    for vector in itertools.product(
      *((-value, value) if value else (0,) for value in vector)
    )
  }
  return tuple(sorted(vectors))


def _validated_crystal_structure(crystal_structure: float) -> int:
  if crystal_structure not in (0.0, 1.0):
    msg = "crystal plasticity crystal_structure must be 0 (FCC) or 1 (BCC)"
    raise ValueError(msg)
  return int(crystal_structure)


def _validated_slip_count(slip_system_count: float) -> int:
  if not float(slip_system_count).is_integer() or slip_system_count < 1.0:
    msg = "crystal plasticity slip_system_count must be a positive integer value"
    raise ValueError(msg)
  return int(slip_system_count)


def schmid_matrix(
  plane_normals: np.ndarray,
  slip_directions: np.ndarray,
) -> np.ndarray:
  """Build engineering-Voigt Schmid vectors after strict frame validation."""
  if (
    type(plane_normals) is not np.ndarray
    or type(slip_directions) is not np.ndarray
    or plane_normals.ndim != 2
    or slip_directions.shape != plane_normals.shape
    or plane_normals.shape[1:] != (3,)
  ):
    msg = "Schmid frames require matching (nslip, 3) normal and direction arrays"
    raise TypeError(msg)
  normals = np.asarray(plane_normals, dtype=np.float64)
  directions = np.asarray(slip_directions, dtype=np.float64)
  schmid = np.empty((6, normals.shape[0]), dtype=np.float64)
  for index, (normal, direction) in enumerate(zip(normals, directions, strict=True)):
    if not bool(np.isfinite(normal).all()) or not bool(np.isfinite(direction).all()):
      msg = f"nonorthogonal-slip-frame: pair {index} has non-finite components"
      raise SlipFrameError(msg)
    normal_norm = float(np.linalg.norm(normal))
    direction_norm = float(np.linalg.norm(direction))
    dot = float(np.dot(normal, direction))
    if (
      abs(normal_norm - 1.0) > 1.0e-12
      or abs(direction_norm - 1.0) > 1.0e-12
      or abs(dot) > 1.0e-12
    ):
      msg = (
        f"nonorthogonal-slip-frame: pair {index}, normal={normal.tolist()}, "
        f"direction={direction.tolist()}, dot={dot:.17g}, "
        f"norms=({normal_norm:.17g}, {direction_norm:.17g})"
      )
      raise SlipFrameError(msg)
    schmid[0, index] = direction[0] * normal[0]
    schmid[1, index] = direction[1] * normal[1]
    schmid[2, index] = direction[2] * normal[2]
    schmid[3, index] = direction[1] * normal[2] + direction[2] * normal[1]
    schmid[4, index] = direction[0] * normal[2] + direction[2] * normal[0]
    schmid[5, index] = direction[0] * normal[1] + direction[1] * normal[0]
  return schmid


def crystal_slip_frames(
  crystal_structure: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
  """Generate every system in the selected complete FCC or BCC cubic family."""
  structure = _validated_crystal_structure(crystal_structure)
  if structure == 0:
    planes = _cubic_integer_family((1, 1, 1))
    directions = _cubic_integer_family((1, 1, 0))
  else:
    planes = _cubic_integer_family((1, 1, 0))
    directions = _cubic_integer_family((1, 1, 1))
  selected: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []
  for plane in planes:
    for direction in directions:
      if sum(plane[index] * direction[index] for index in range(3)) == 0:
        selected.append((plane, direction))
  if len(selected) != _SLIP_SYSTEM_COUNT:
    msg = "internal cubic slip-family generation did not produce 12 systems"
    raise RuntimeError(msg)
  normals = np.array(
    [
      np.asarray(plane, dtype=np.float64) / np.linalg.norm(plane)
      for plane, _ in selected
    ],
    dtype=np.float64,
  )
  directions = np.array(
    [
      np.asarray(direction, dtype=np.float64) / np.linalg.norm(direction)
      for _, direction in selected
    ],
    dtype=np.float64,
  )
  schmid_matrix(normals, directions)
  return normals, directions


def _elastic_matrix(youngs_modulus: float, poisson_ratio: float) -> np.ndarray:
  shear = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
  lame = (
    youngs_modulus
    * poisson_ratio
    / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
  )
  matrix = np.zeros((6, 6), dtype=np.float64)
  matrix[:3, :3] = lame
  for index in range(3):
    matrix[index, index] += 2.0 * shear
    matrix[index + 3, index + 3] = shear
  return matrix


def crystal_plasticity_calibration(
  youngs_modulus: float,
  poisson_ratio: float,
  crystal_structure: float,
  slip_system_count: float,
  reference_slip_rate: float,
  rate_exponent: float,
  initial_critical_stress: float,
  hardening_modulus: float,
  saturation_stress: float,
  hardening_exponent: float,
  latent_hardening_ratio: float,
  theta: float,
  newton_tolerance: float,
) -> np.ndarray:
  """Validate and pack one complete-crystal calibration vector."""
  values = (
    ("youngs_modulus", youngs_modulus),
    ("poisson_ratio", poisson_ratio),
    ("crystal_structure", crystal_structure),
    ("slip_system_count", slip_system_count),
    ("reference_slip_rate", reference_slip_rate),
    ("rate_exponent", rate_exponent),
    ("initial_critical_stress", initial_critical_stress),
    ("hardening_modulus", hardening_modulus),
    ("saturation_stress", saturation_stress),
    ("hardening_exponent", hardening_exponent),
    ("latent_hardening_ratio", latent_hardening_ratio),
    ("theta", theta),
    ("newton_tolerance", newton_tolerance),
  )
  for label, value in values:
    if type(value) is not float:
      msg = f"crystal plasticity {label} must be an exact float"
      raise TypeError(msg)
    if not np.isfinite(value):
      msg = f"crystal plasticity {label} must be finite"
      raise ValueError(msg)
  structure = _validated_crystal_structure(crystal_structure)
  del structure
  count = _validated_slip_count(slip_system_count)
  if count != _SLIP_SYSTEM_COUNT:
    msg = "crystal plasticity requires the complete 12-system FCC or BCC family"
    raise ValueError(msg)
  if youngs_modulus <= 0.0:
    msg = "crystal plasticity youngs_modulus must be positive"
    raise ValueError(msg)
  if not -1.0 < poisson_ratio < 0.5:
    msg = "crystal plasticity poisson_ratio must lie between -1 and 0.5"
    raise ValueError(msg)
  if reference_slip_rate <= 0.0:
    msg = "crystal plasticity reference_slip_rate must be positive"
    raise ValueError(msg)
  if rate_exponent <= 1.0:
    msg = "crystal plasticity rate_exponent must exceed one"
    raise ValueError(msg)
  if initial_critical_stress <= 0.0:
    msg = "crystal plasticity initial_critical_stress must be positive"
    raise ValueError(msg)
  if hardening_modulus < 0.0:
    msg = "crystal plasticity hardening_modulus must be non-negative"
    raise ValueError(msg)
  if saturation_stress <= initial_critical_stress:
    msg = "crystal plasticity saturation_stress must exceed initial_critical_stress"
    raise ValueError(msg)
  if hardening_exponent < 1.0:
    msg = "crystal plasticity hardening_exponent must be at least one"
    raise ValueError(msg)
  if latent_hardening_ratio < 1.0:
    msg = "crystal plasticity latent_hardening_ratio must be at least one"
    raise ValueError(msg)
  if not 0.0 <= theta <= 1.0:
    msg = "crystal plasticity theta must lie in [0, 1]"
    raise ValueError(msg)
  if newton_tolerance <= 0.0:
    msg = "crystal plasticity newton_tolerance must be positive"
    raise ValueError(msg)

  normals, directions = crystal_slip_frames(crystal_structure)
  schmid = schmid_matrix(normals, directions)
  elastic = _elastic_matrix(youngs_modulus, poisson_ratio)
  scalars = np.array(
    [
      float(count),
      reference_slip_rate,
      rate_exponent,
      initial_critical_stress,
      hardening_modulus,
      saturation_stress,
      hardening_exponent,
      latent_hardening_ratio,
      theta,
      newton_tolerance,
    ],
    dtype=np.float64,
  )
  return np.concatenate((scalars, elastic.reshape(-1), schmid.reshape(-1)))


def _resistance(
  effective_slip: np.ndarray,
  initial_critical_stress: float,
  hardening_modulus: float,
  saturation_stress: float,
  hardening_exponent: float,
) -> tuple[np.ndarray, np.ndarray]:
  """Integrated Kocks--Mecking resistance and its effective-slip derivative."""
  y0 = 1.0 - initial_critical_stress / saturation_stress
  if hardening_exponent == 1.0:
    y = y0 * np.exp(-hardening_modulus * effective_slip / saturation_stress)
  else:
    exponent = 1.0 / (1.0 - hardening_exponent)
    base = y0 ** (1.0 - hardening_exponent) + (
      (hardening_exponent - 1.0)
      * hardening_modulus
      * effective_slip
      / saturation_stress
    )
    y = base**exponent
  resistance = initial_critical_stress + (
    saturation_stress - initial_critical_stress
  ) * (1.0 - y / y0)
  resistance = np.where(effective_slip == 0.0, initial_critical_stress, resistance)
  derivative = hardening_modulus * y**hardening_exponent
  return resistance, derivative


def _slip_rates(
  resolved: np.ndarray,
  resistance: np.ndarray,
  reference_slip_rate: float,
  rate_exponent: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  ratio = resolved / resistance
  magnitude = np.abs(ratio)
  rate = reference_slip_rate * np.sign(resolved) * magnitude**rate_exponent
  rate_resolved = (
    reference_slip_rate
    * rate_exponent
    * magnitude ** (rate_exponent - 1.0)
    / resistance
  )
  rate_resistance = -rate_exponent * rate / resistance
  return rate, rate_resolved, rate_resistance


def _latent_matrix(count: int, latent_hardening_ratio: float) -> np.ndarray:
  matrix = np.full((count, count), latent_hardening_ratio, dtype=np.float64)
  np.fill_diagonal(matrix, 1.0)
  return matrix


@dataclass(frozen=True, slots=True)
class _IncrementEvaluation:
  stress: np.ndarray
  resolved: np.ndarray
  resistance: np.ndarray
  rate: np.ndarray
  rate_resolved: np.ndarray
  rate_resistance: np.ndarray
  resistance_increment: np.ndarray


def _evaluate_increment(
  increment: np.ndarray,
  trial_stress: np.ndarray,
  flow_stiffness: np.ndarray,
  schmid: np.ndarray,
  accepted_cumulative: np.ndarray,
  latent: np.ndarray,
  reference_slip_rate: float,
  rate_exponent: float,
  initial_critical_stress: float,
  hardening_modulus: float,
  saturation_stress: float,
  hardening_exponent: float,
) -> _IncrementEvaluation:
  stress = trial_stress - flow_stiffness @ increment
  resolved = schmid.T @ stress
  cumulative = accepted_cumulative + np.abs(increment)
  effective = latent @ cumulative
  resistance, resistance_effective = _resistance(
    effective,
    initial_critical_stress,
    hardening_modulus,
    saturation_stress,
    hardening_exponent,
  )
  rate, rate_resolved, rate_resistance = _slip_rates(
    resolved, resistance, reference_slip_rate, rate_exponent
  )
  resistance_increment = (
    resistance_effective[:, None] * latent * np.sign(increment)[None, :]
  )
  return _IncrementEvaluation(
    stress,
    resolved,
    resistance,
    rate,
    rate_resolved,
    rate_resistance,
    resistance_increment,
  )


def _crystal_step(
  strain: np.ndarray,
  row: np.ndarray,
  dtime: float,
  elastic: np.ndarray,
  schmid: np.ndarray,
  reference_slip_rate: float,
  rate_exponent: float,
  initial_critical_stress: float,
  hardening_modulus: float,
  saturation_stress: float,
  hardening_exponent: float,
  latent_hardening_ratio: float,
  theta: float,
  newton_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
  """Evaluate one entity; ``None`` denotes a typed step rejection."""
  count = schmid.shape[1]
  old_resistance = row[:count]
  old_gamma = row[count : 2 * count]
  old_resolved = row[2 * count : 3 * count]
  old_cumulative = row[3 * count : 4 * count]
  flow_stiffness = elastic @ schmid
  trial_stress = elastic @ (strain - schmid @ old_gamma)
  latent = _latent_matrix(count, latent_hardening_ratio)
  old_rate, _, _ = _slip_rates(
    old_resolved, old_resistance, reference_slip_rate, rate_exponent
  )

  if dtime == 0.0 or theta == 0.0:
    increment = dtime * old_rate
    evaluation = _evaluate_increment(
      increment,
      trial_stress,
      flow_stiffness,
      schmid,
      old_cumulative,
      latent,
      reference_slip_rate,
      rate_exponent,
      initial_critical_stress,
      hardening_modulus,
      saturation_stress,
      hardening_exponent,
    )
    tangent = np.array(elastic, copy=True)
  else:
    increment = dtime * np.array(old_rate, copy=True)
    converged = False
    evaluation = None
    jacobian = np.empty((count, count), dtype=np.float64)
    schmid_elastic = schmid.T @ elastic
    slip_elastic = schmid_elastic @ schmid
    for _ in range(_LOCAL_NEWTON_LIMIT):
      evaluation = _evaluate_increment(
        increment,
        trial_stress,
        flow_stiffness,
        schmid,
        old_cumulative,
        latent,
        reference_slip_rate,
        rate_exponent,
        initial_critical_stress,
        hardening_modulus,
        saturation_stress,
        hardening_exponent,
      )
      residual = increment - dtime * (
        (1.0 - theta) * old_rate + theta * evaluation.rate
      )
      scale = newton_tolerance * max(
        reference_slip_rate * dtime, float(np.max(np.abs(increment))), 1.0e-30
      )
      jacobian = np.eye(count) + theta * dtime * (
        evaluation.rate_resolved[:, None] * slip_elastic
        - evaluation.rate_resistance[:, None] * evaluation.resistance_increment
      )
      if float(np.max(np.abs(residual))) <= scale:
        converged = True
        break
      if not bool(np.isfinite(jacobian).all()) or not bool(np.isfinite(residual).all()):
        return None
      try:
        correction = np.linalg.solve(jacobian, -residual)
      except np.linalg.LinAlgError:
        return None
      if not bool(np.isfinite(correction).all()):
        return None
      increment += correction
      if not bool(np.isfinite(increment).all()):
        return None
    if not converged or evaluation is None:
      return None
    if np.any(increment):
      implicit_factor = theta * dtime * evaluation.rate_resolved
      try:
        tangent = elastic - flow_stiffness @ np.linalg.solve(
          jacobian, implicit_factor[:, None] * schmid_elastic
        )
      except np.linalg.LinAlgError:
        return None
    else:
      tangent = np.array(elastic, copy=True)

  trial = np.empty(4 * count + 7, dtype=np.float64)
  trial[:count] = evaluation.resistance
  trial[count : 2 * count] = old_gamma + increment
  trial[2 * count : 3 * count] = evaluation.resolved
  trial[3 * count : 4 * count] = old_cumulative + np.abs(increment)
  trial[4 * count : 4 * count + 6] = evaluation.stress
  trial[-1] = row[-1] + float(np.sum(np.abs(increment)))
  if (
    not bool(np.isfinite(evaluation.stress).all())
    or not bool(np.isfinite(tangent).all())
    or not bool(np.isfinite(trial).all())
  ):
    return None
  return evaluation.stress, tangent, trial


def _reject(accepted_rows: np.ndarray) -> StatefulContinuumKernelResult:
  entity_count = accepted_rows.shape[0]
  return StatefulContinuumKernelResult(
    stresses=np.zeros((entity_count, 6), dtype=np.float64),
    tangents=np.zeros((entity_count, 6, 6), dtype=np.float64),
    trial_rows=np.array(accepted_rows, dtype=np.float64, copy=True),
    status=EvaluationStatus.REJECT_STEP,
  )


def crystal_plasticity_kernel(
  strains: np.ndarray,
  accepted_rows: np.ndarray,
  calibration: np.ndarray,
  signals: tuple[StatefulContinuumSignalInput, ...],
) -> StatefulContinuumKernelResult:
  """Evaluate the complete-crystal return map over flattened state entities."""
  if (
    type(calibration) is not np.ndarray
    or calibration.dtype != np.dtype(np.float64)
    or calibration.dtype.metadata is not None
    or calibration.ndim != 1
    or calibration.size < _CALIBRATION_FIXED + 6
  ):
    msg = "crystal plasticity kernel requires the packed calibration vector"
    raise TypeError(msg)
  count = _validated_slip_count(float(calibration[0]))
  if count != _SLIP_SYSTEM_COUNT or calibration.shape != (
    _CALIBRATION_FIXED + 6 * count,
  ):
    msg = "crystal plasticity kernel requires a complete 12-system calibration"
    raise TypeError(msg)
  if (
    type(signals) is not tuple
    or len(signals) != 1
    or type(signals[0]) is not StatefulContinuumSignalInput
    or signals[0].port_id != _DTIME_PORT_ID
  ):
    msg = "crystal plasticity kernel requires exactly the declared dtime port"
    raise TypeError(msg)
  entity_count = strains.shape[0]
  row_width = 4 * count + 7
  if strains.shape != (entity_count, 6) or accepted_rows.shape != (
    entity_count,
    row_width,
  ):
    msg = (
      "crystal plasticity kernel requires (n, 6) strains and "
      "(n, 4 * slip_system_count + 7) rows"
    )
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

  reference_slip_rate = float(calibration[1])
  rate_exponent = float(calibration[2])
  initial_critical_stress = float(calibration[3])
  hardening_modulus = float(calibration[4])
  saturation_stress = float(calibration[5])
  hardening_exponent = float(calibration[6])
  latent_hardening_ratio = float(calibration[7])
  theta = float(calibration[8])
  newton_tolerance = float(calibration[9])
  elastic = calibration[10:46].reshape(6, 6)
  schmid = calibration[46:].reshape(6, count)

  stresses = np.empty((entity_count, 6), dtype=np.float64)
  tangents = np.empty((entity_count, 6, 6), dtype=np.float64)
  trial_rows = np.empty((entity_count, row_width), dtype=np.float64)
  with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
    for index in range(entity_count):
      result = _crystal_step(
        strains[index],
        accepted_rows[index],
        dtime,
        elastic,
        schmid,
        reference_slip_rate,
        rate_exponent,
        initial_critical_stress,
        hardening_modulus,
        saturation_stress,
        hardening_exponent,
        latent_hardening_ratio,
        theta,
        newton_tolerance,
      )
      if result is None:
        return _reject(accepted_rows)
      stresses[index], tangents[index], trial_rows[index] = result
  return StatefulContinuumKernelResult(
    stresses=stresses,
    tangents=tangents,
    trial_rows=trial_rows,
    status=EvaluationStatus.OK,
  )


def crystal_plasticity_initial_state(
  parameters: tuple[float, ...],
  layout: OperatorStateLayout,
) -> np.ndarray:
  """Initialize every slip resistance to the authored critical stress."""
  if len(parameters) != _PARAMETER_COUNT:
    msg = "crystal plasticity initial state requires thirteen parameters"
    raise TypeError(msg)
  structure = _validated_crystal_structure(parameters[2])
  count = _validated_slip_count(parameters[3])
  if count != _SLIP_SYSTEM_COUNT:
    msg = "crystal plasticity requires the complete 12-system family"
    raise ValueError(msg)
  if layout.row_width != 4 * count + 7:
    msg = "crystal plasticity initial state requires the 4*nslip + 7 row layout"
    raise ValueError(msg)
  expected_count = _SLIP_SYSTEM_COUNT if structure in (0, 1) else count
  if count != expected_count:
    msg = "crystal plasticity slip count does not match the crystal structure"
    raise ValueError(msg)
  rows = np.zeros(layout.row_shape, dtype=np.float64)
  rows[..., :count] = parameters[6]
  return rows


@dataclass(frozen=True, slots=True)
class CrystalPlasticityBinding:
  """Registry binding for the fresh complete-crystal stateful law."""

  def __call__(self, *parameters: float) -> np.ndarray:
    if len(parameters) != _PARAMETER_COUNT:
      msg = "crystal plasticity requires exactly thirteen calibration parameters"
      raise TypeError(msg)
    return crystal_plasticity_calibration(*parameters)

  def descriptor_metadata(self) -> dict[str, object]:
    return crystal_plasticity_metadata()

  def kernel(
    self,
    strains: np.ndarray,
    accepted_rows: np.ndarray,
    calibration: np.ndarray,
    signals: tuple[StatefulContinuumSignalInput, ...],
  ) -> StatefulContinuumKernelResult:
    return crystal_plasticity_kernel(strains, accepted_rows, calibration, signals)

  def initial_state(
    self,
    parameters: tuple[float, ...],
    layout: OperatorStateLayout,
  ) -> np.ndarray:
    return crystal_plasticity_initial_state(parameters, layout)


CRYSTAL_PLASTICITY_BINDING = CrystalPlasticityBinding()
