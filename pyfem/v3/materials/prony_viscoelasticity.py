"""Prony-series viscoelasticity: the first signal-consuming v3 material law.

A pure, batched port of the legacy reference ``pyfem/materials/ViscoElasticity.py``
(the numerical oracle per AGENTS.md) onto the v2 stateful descriptor ABI with a
declared ``signal_ports`` time port. The law is a generalized Maxwell model: a
long-term spring in parallel with ``n`` Maxwell elements, with relaxation
modulus ``E(t) = Einf + sum_i E_i * exp(-t / tau_i)``. It is 3D internally
(6-Voigt ``[xx, yy, zz, yz, zx, xy]`` with engineering shears); the continuum
operator embeds the 3-Voigt port strain as ``[xx, yy, 0, 0, 0, xy]`` (a
plane-strain embedding) and truncates stress/tangent back to ``[xx, yy, xy]``.

Parameters (``parameter_names`` order): ``youngs_modulus`` E > 0 (the legacy
instantaneous modulus), ``poisson_ratio`` -1 < nu < 0.5,
``equilibrium_modulus`` 0 < Einf < E, ``prony_term_count`` integer n >= 1 (the
legacy ``nMaxwell``), ``relaxation_time_first`` tau_first > 0, and
``relaxation_time_last`` tau_last >= tau_first (strictly greater for n > 1).
The descriptor parameterizes the legacy default family: equal relaxation
moduli ``E_i = (E - Einf) / n`` (``ViscoElasticity.py:87-90``) and
logarithmically spaced relaxation times ``tau_i`` between the two endpoints
(the legacy ``np.logspace`` default at :93-96), which covers the shipped
example deck (``examples/materials/viscoelasticity/creep_test.pro``: E = 1000,
nu = 0.3, Einf = 100, n = 3, times log-spaced 0.1 .. 10, moduli 300 each).
Decks with arbitrary per-term constants are out of scope for this descriptor:
the ABI's scalar parameter interface fixes the parameter list per descriptor.
For ``prony_term_count`` = 1 the series is the single term at
``relaxation_time_first`` (the ``np.logspace(start, stop, 1)`` convention).

Time reaches the law exclusively through the declared identity signal port
(``port_id`` ``"time"``, ``signal_id`` ``"time"``), replacing the legacy
``self.solverStat.time`` read (``ViscoElasticity.py:175``) — there is no
schedule back channel. The time increment is the legacy ``dtime = time_new -
time_old`` (:176), with ``time_old`` carried in the state row.

State rows (6*n + 13 floats per entity, flattened element x ip x slot):
``eps_i`` 6n (the per-term internal strains, one parameterized-width slot
resolved through the M25 ``{"parameter": "prony_term_count", "scale": 6}``
declaration), ``sigma`` 6, ``epsilon`` 6 (committed total strain), ``time`` 1.

The F3 analysis behind the mission pinned the legacy state inventory
(6*n + 7: ``eps_i``, ``sigma``, ``time_old``). That inventory is not a
sufficient v3 state: the legacy law consumes the strain *increment* supplied
externally by its element (``kinematics.dstrain``, :179-186), while the frozen
v3 kernel ABI supplies *total* strain. The committed total strain is not
recoverable from the legacy state — unrolling the recursion gives
``sigma^n = (1 + sum f) Cinf : eps^n - sum_i f_i Cinf : (sum_{m<=n} eps_i^m)``,
which needs the full running sum of internal strains, and constructively two
different strain/time histories can reach the identical ``{eps_i, sigma,
time}`` state with different committed totals (scalar f = 1: increments (1, 1)
at constant relaxation factor a = 0.5 versus increments (0.25, 1.59375) at
a_1 = 0.25, a_2 = 0.6 both land on ``{eps_i = 0.75, sigma = 2.75}`` with
committed totals 2.0 versus 1.84375). The ``epsilon`` slot internalizes the
committed-strain channel the legacy element supplied — the width is
6*n + 7 + 6 = 6n + 13, one committed-strain slot beyond the F3
inventory (escalated to the tower on the M49 clarify-request; the
parameterized-width machinery covers it with no ABI or schema change).

The kernel replicates the legacy ``getStress`` arithmetic statement for
statement (:185-221), so stress and every state slot match the legacy oracle
bit for bit on every committed path, including the legacy
``dtime > 0`` guard (:192 — a constant-time or backward-time substep adds only
the long-term elastic increment and freezes the internal strains, exactly the
legacy behavior, now exact because the committed strain is stored):

- ``sigma += Cinf : dstrain`` with ``dstrain = strains - epsilon`` (the same
  float64 subtraction the parity harness uses to drive the legacy oracle);
- per term, ``exp_factor = exp(-dtime / tau_i)``,
  ``eps_i <- exp_factor * eps_i + (1 - exp_factor) * dstrain`` (:203),
  ``sigma += (E_i / Einf) * Cinf : (dstrain - eps_i)`` (:207-209);
- the exact algorithmic tangent ``Cinf * (1 + sum_i (E_i/Einf) *
  exp(-dtime/tau_i))`` accumulated as ``tang += (factor * exp_factor) *
  Cinf`` — class ``algorithmic-symmetric``, so the Jacobian channel is
  nonlinear and the driver re-factorizes every Newton iteration.

One legacy statement was deliberately NOT replicated at the fork: the legacy
tangent accumulation ``tang += (factor * (1 - exp_factor)) * Cinf``
(``ViscoElasticity.py:213-214``) yielded ``Cinf * (1 + sum f_i (1 - a_i))``,
which contradicted the legacy stress update it accompanied — the exact
derivative of ``sigma += f_i Cinf : (dstrain - eps_i)`` with
``eps_i <- a_i eps_i + (1 - a_i) dstrain`` is ``Cinf * (1 + sum f_i a_i)``
(the pre-repair legacy tangent belonged to the standard branch-stress
recurrence, a different discretization). The divergence was not
rounding-level: against a finite difference of the legacy law's own stress
response at dtime = 0.05 on the example-deck constants, the pre-repair
legacy tangent errs at 0.73 relative while ``Cinf * (1 + sum f_i a_i)``
matches to 2e-12, and a Newton iteration on the pre-repair tangent contracts
as ``|1 - K_true/K_legacy| = 2.7 > 1`` there — divergence that driver
cutback worsens (halving dtime moves the ratio toward 10x; the shipped
creep_test.pro survived only because its dtime = 0.5 sits mid-spectrum,
where the swap nearly cancels). The v3 kernel therefore writes the true
algorithmic tangent (the M25 ``flow[3:]`` precedent: fix, and pin the
divergence). M55 repaired the legacy side to the same true tangent (commit
bc4a434, merge 945a95d); the v3 kernel is unchanged, stress and state
bookkeeping remain bitwise-identical to legacy, and the relationship is now
parity-where-repaired — the parity battery pins the shared tangent bitwise,
with the pre-repair evidence above retained as the record of why the
divergence existed.

Divergences from legacy are typed, not silent: a non-finite strain batch or a
non-finite bound time reports ``REJECT_STEP`` with byte-equal trial rows
(legacy would silently poison its history). The initial state is all zero,
matching the legacy constructor's zeroed history commit (:123-131); the
binding declares ``initial_state`` explicitly so the owner never relies on
implicit zero-fill. The virgin probe (zero strain, zero bound time, zero
rows) takes the ``dtime == 0`` branch with a zero increment and returns the
accepted rows byte-equal.
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

PRONY_VISCOELASTICITY_STATE_SCHEMA = "pyfem-v3-prony-viscoelastic-state-v1"

_TIME_PORT_ID = "time"

# Flat calibration vector layout: the term count, then the per-term relaxation
# factors E_i / Einf, the per-term relaxation times, and the 6x6 long-term
# elastic tangent in C order.
_CALIBRATION_FIXED = 1 + 36

# State row layout: eps_i (6n) | sigma (6) | epsilon (6) | time (1).
_ROW_FIXED = 13


def prony_viscoelasticity_metadata() -> dict[str, object]:
  """Return the canonical v2 descriptor metadata for the first ported law."""
  return {
    "schema": STATEFUL_MATERIAL_DESCRIPTOR_SCHEMA,
    "law": "prony-viscoelasticity",
    "stress_state": "plane-strain",
    "parameter_names": [
      "youngs_modulus",
      "poisson_ratio",
      "equilibrium_modulus",
      "prony_term_count",
      "relaxation_time_first",
      "relaxation_time_last",
    ],
    "parameter_dtype": "float64",
    "stress_voigt_order": ["xx", "yy", "xy"],
    "strain_shear_convention": "engineering",
    "internal_voigt_order": ["xx", "yy", "zz", "yz", "zx", "xy"],
    "tangent_class": "algorithmic-symmetric",
    "state_schema": PRONY_VISCOELASTICITY_STATE_SCHEMA,
    "state_slots": [
      {
        "name": "eps_i",
        "width": {"parameter": "prony_term_count", "scale": 6},
        "dtype": "float64",
        "lifetime": "accepted-trial",
      },
      {
        "name": "sigma",
        "width": 6,
        "dtype": "float64",
        "lifetime": "accepted-trial",
      },
      {
        "name": "epsilon",
        "width": 6,
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


def _validated_term_count(value: float, label: str) -> int:
  if not float(value).is_integer() or value < 1.0:
    msg = f"prony viscoelasticity {label} must be a positive integer value"
    raise ValueError(msg)
  return int(value)


def prony_viscoelasticity_calibration(
  youngs_modulus: float,
  poisson_ratio: float,
  equilibrium_modulus: float,
  prony_term_count: float,
  relaxation_time_first: float,
  relaxation_time_last: float,
) -> np.ndarray:
  """Pack the law's flat calibration vector with legacy-identical arithmetic."""
  for label, value in (
    ("youngs_modulus", youngs_modulus),
    ("poisson_ratio", poisson_ratio),
    ("equilibrium_modulus", equilibrium_modulus),
    ("prony_term_count", prony_term_count),
    ("relaxation_time_first", relaxation_time_first),
    ("relaxation_time_last", relaxation_time_last),
  ):
    if type(value) is not float:
      msg = f"prony viscoelasticity {label} must be an exact float"
      raise TypeError(msg)
    if not np.isfinite(value):
      msg = f"prony viscoelasticity {label} must be finite"
      raise ValueError(msg)
  if youngs_modulus <= 0.0:
    msg = "prony viscoelasticity youngs_modulus must be positive"
    raise ValueError(msg)
  if not -1.0 < poisson_ratio < 0.5:
    msg = "prony viscoelasticity poisson_ratio must lie between -1 and 0.5"
    raise ValueError(msg)
  if not 0.0 < equilibrium_modulus < youngs_modulus:
    msg = (
      "prony viscoelasticity equilibrium_modulus must lie between zero and "
      "youngs_modulus"
    )
    raise ValueError(msg)
  term_count = _validated_term_count(prony_term_count, "prony_term_count")
  if relaxation_time_first <= 0.0:
    msg = "prony viscoelasticity relaxation_time_first must be positive"
    raise ValueError(msg)
  if relaxation_time_last < relaxation_time_first:
    msg = (
      "prony viscoelasticity relaxation_time_last must not precede "
      "relaxation_time_first"
    )
    raise ValueError(msg)
  if term_count > 1 and relaxation_time_last == relaxation_time_first:
    msg = (
      "prony viscoelasticity relaxation times must span a non-empty interval "
      "for multiple terms"
    )
    raise ValueError(msg)

  # Legacy constructor expression order (ViscoElasticity.py:106-121).
  ebulk3 = equilibrium_modulus / (1.0 - 2.0 * poisson_ratio)
  eg2 = equilibrium_modulus / (1.0 + poisson_ratio)
  eg = 0.5 * eg2
  elam = (ebulk3 - eg2) / 3.0

  cinf = np.zeros(shape=(6, 6))
  cinf[:3, :3] = elam
  cinf[0, 0] += eg2
  cinf[1, 1] = cinf[0, 0]
  cinf[2, 2] = cinf[0, 0]
  cinf[3, 3] = eg
  cinf[4, 4] = cinf[3, 3]
  cinf[5, 5] = cinf[3, 3]

  # Legacy default family: equal relaxation moduli (:87-90) and
  # logarithmically spaced relaxation times (:93-96). The stress factor is the
  # legacy per-term ratio E_i / Einf (:207).
  term_modulus = (youngs_modulus - equilibrium_modulus) / term_count
  factors = np.full(term_count, term_modulus / equilibrium_modulus)
  times = np.logspace(
    np.log10(relaxation_time_first),
    np.log10(relaxation_time_last),
    term_count,
  )

  return np.concatenate(
    (
      np.array([float(term_count)], dtype=np.float64),
      factors,
      times,
      cinf.reshape(-1),
    )
  )


def _reject(accepted_rows: np.ndarray) -> StatefulContinuumKernelResult:
  entity_count = accepted_rows.shape[0]
  return StatefulContinuumKernelResult(
    stresses=np.zeros((entity_count, 6), dtype=np.float64),
    tangents=np.zeros((entity_count, 6, 6), dtype=np.float64),
    trial_rows=np.array(accepted_rows, dtype=np.float64, copy=True),
    status=EvaluationStatus.REJECT_STEP,
  )


def prony_viscoelasticity_kernel(
  strains: np.ndarray,
  accepted_rows: np.ndarray,
  calibration: np.ndarray,
  signals: tuple[StatefulContinuumSignalInput, ...],
) -> StatefulContinuumKernelResult:
  """Evaluate the Prony-series law over flattened state entities.

  ``strains`` holds total engineering 6-Voigt strain per entity;
  ``accepted_rows`` holds the committed 6*n + 13 rows; ``signals`` carries
  exactly the declared identity time port. The strain increment is
  ``strains - epsilon`` from the accepted row, the identity the parity harness
  also uses to drive the legacy oracle; the time increment is the legacy
  ``time_new - time_old`` over the bound signal value and the row's committed
  time. Expected numerical failures report ``REJECT_STEP`` with byte-equal
  trial rows, never exceptions.
  """
  if (
    type(calibration) is not np.ndarray
    or calibration.dtype != np.dtype(np.float64)
    or calibration.dtype.metadata is not None
    or calibration.ndim != 1
    or calibration.size < _CALIBRATION_FIXED + 2
  ):
    msg = "prony viscoelasticity kernel requires the packed calibration vector"
    raise TypeError(msg)
  term_count = _validated_term_count(calibration[0], "calibration term count")
  if calibration.shape != (2 * term_count + _CALIBRATION_FIXED,):
    msg = "prony viscoelasticity kernel requires the packed calibration vector"
    raise TypeError(msg)
  if (
    type(signals) is not tuple
    or len(signals) != 1
    or type(signals[0]) is not StatefulContinuumSignalInput
    or signals[0].port_id != _TIME_PORT_ID
  ):
    msg = "prony viscoelasticity kernel requires exactly the declared time port"
    raise TypeError(msg)
  row_width = 6 * term_count + _ROW_FIXED
  entity_count = strains.shape[0]
  if strains.shape != (entity_count, 6) or accepted_rows.shape != (
    entity_count,
    row_width,
  ):
    msg = (
      "prony viscoelasticity kernel requires (n, 6) strains and "
      "(n, 6 * term_count + 13) rows"
    )
    raise ValueError(msg)
  if not bool(np.isfinite(strains).all()):
    return _reject(accepted_rows)
  time_new = float(signals[0].values[0])
  if not np.isfinite(time_new):
    return _reject(accepted_rows)

  factors = calibration[1 : 1 + term_count]
  times = calibration[1 + term_count : 1 + 2 * term_count]
  cinf = calibration[1 + 2 * term_count :].reshape(6, 6)

  sigma_offset = 6 * term_count
  epsilon_offset = sigma_offset + 6
  time_offset = epsilon_offset + 6

  stresses = np.empty((entity_count, 6), dtype=np.float64)
  tangents = np.empty((entity_count, 6, 6), dtype=np.float64)
  trial_rows = np.empty((entity_count, row_width), dtype=np.float64)

  with np.errstate(over="ignore", invalid="ignore"):
    for index in range(entity_count):
      row = accepted_rows[index]
      # Legacy getStress statement order (ViscoElasticity.py:171-221).
      dtime = time_new - row[time_offset]
      dstrain = strains[index] - row[epsilon_offset : epsilon_offset + 6]
      sigma = row[sigma_offset : sigma_offset + 6] + np.dot(cinf, dstrain)
      tang = cinf.copy()
      trial = trial_rows[index]
      if dtime > 0.0:
        for term in range(term_count):
          eps_i = row[6 * term : 6 * term + 6]
          exp_factor = np.exp(-dtime / times[term])
          deps_i = exp_factor * eps_i + (1.0 - exp_factor) * dstrain
          factor = factors[term]
          sigma += factor * np.dot(cinf, dstrain - deps_i)
          tang += (factor * exp_factor) * cinf
          trial[6 * term : 6 * term + 6] = deps_i
      else:
        trial[:sigma_offset] = row[:sigma_offset]
      if not bool(np.isfinite(sigma).all()) or not bool(np.isfinite(tang).all()):
        return _reject(accepted_rows)
      trial[sigma_offset : sigma_offset + 6] = sigma
      trial[epsilon_offset : epsilon_offset + 6] = strains[index]
      trial[time_offset] = time_new
      stresses[index] = sigma
      tangents[index] = tang

  return StatefulContinuumKernelResult(
    stresses=stresses,
    tangents=tangents,
    trial_rows=trial_rows,
    status=EvaluationStatus.OK,
  )


def prony_viscoelastic_initial_state(
  parameters: tuple[float, ...],
  layout: OperatorStateLayout,
) -> np.ndarray:
  """Bind the all-zero initial rows of the legacy constructor explicitly."""
  if len(parameters) != 6:
    msg = "prony viscoelasticity initial state requires the six parameters"
    raise TypeError(msg)
  term_count = _validated_term_count(parameters[3], "prony_term_count")
  if layout.row_width != 6 * term_count + _ROW_FIXED:
    msg = "prony viscoelasticity initial state requires the 6n + 13 row layout"
    raise ValueError(msg)
  return np.zeros(layout.row_shape, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class PronyViscoelasticityBinding:
  """The registry binding object wiring the first ported law to the ABI."""

  def __call__(self, *parameters: float) -> np.ndarray:
    if len(parameters) != 6:
      msg = "prony viscoelasticity requires exactly six calibration parameters"
      raise TypeError(msg)
    return prony_viscoelasticity_calibration(*parameters)

  def descriptor_metadata(self) -> dict[str, object]:
    return prony_viscoelasticity_metadata()

  def kernel(
    self,
    strains: np.ndarray,
    accepted_rows: np.ndarray,
    calibration: np.ndarray,
    signals: tuple[StatefulContinuumSignalInput, ...],
  ) -> StatefulContinuumKernelResult:
    return prony_viscoelasticity_kernel(strains, accepted_rows, calibration, signals)

  def initial_state(
    self,
    parameters: tuple[float, ...],
    layout: OperatorStateLayout,
  ) -> np.ndarray:
    return prony_viscoelastic_initial_state(parameters, layout)


PRONY_VISCOELASTICITY_BINDING = PronyViscoelasticityBinding()
