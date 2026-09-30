"""J2 isotropic-hardening plasticity: the first stateful v3 material law.

A pure, batched port of the legacy reference
``pyfem/materials/IsotropicHardeningPlasticity.py`` (the numerical oracle per
AGENTS.md) onto the v2 stateful descriptor ABI. The law is 3D internally
(6-Voigt ``[xx, yy, zz, yz, zx, xy]`` with engineering shears); the continuum
operator embeds the 3-Voigt port strain as ``[xx, yy, 0, 0, 0, xy]``, which is
a plane-strain embedding, and truncates stress/tangent back to
``[xx, yy, xy]``.

Parameters (``parameter_names`` order): ``youngs_modulus`` E > 0,
``poisson_ratio`` -1 < nu < 0.5, ``initial_yield_stress`` syield0 > 0, and
``hardening_slope`` hard >= 0. The legacy law's plastic branch reads
``self.hard`` for its tangent (``IsotropicHardeningPlasticity.py:119``), so
only decks carrying a ``hard`` property ever defined its plastic behavior; the
canonical consistent configuration is linear hardening, where the tabulated
slope equals ``hard``. This law ships exactly that configuration: the stress
update interpolates the legacy two-point table ``EqPlasStrains = [0, 1]``,
``Stresses = [syield0, syield0 + hard]`` (the legacy ``np.insert`` prepend is a
discarded-return no-op, so the table is used as given), and the algorithmic
tangent uses the declared slope — identical numbers, and exactly consistent.

State rows (19 floats per entity, flattened element x ip x slot):
``sigma`` 6 + ``epsilon_e`` 6 + ``epsilon_p`` 6 + ``kappa`` 1 (the accumulated
equivalent plastic strain, annotated ``monotone-nondecreasing``). The initial
state is all zero, matching the legacy constructor's zeroed history commit;
the binding still declares ``initial_state`` explicitly so the owner never
relies on implicit zero-fill for this law.

Tangent class: ``algorithmic-symmetric`` — the consistent tangent of the
return map. The Jacobian channel is therefore nonlinear (never ``linear``):
the driver must re-factorize every Newton iteration.

Numerical contract with the legacy oracle (exercised by
``test/v3/test_v3_stateful_plasticity.py``):

- Documented parity configuration: E = 210000, nu = 0.3, syield0 = 250,
  hard = 1000 (linear hardening). Documented strain paths (total 6-Voigt,
  engineering shear, per committed step): an elastic ramp eps_xx = 0.0002,
  0.0006, 0.0010 (below the plane-strain yield strain syield0*(1+nu)/E); a
  uniaxial plastic ramp of ten equal steps eps_xx = 0.0002 .. 0.004; a mixed
  segment holding eps_xx = 0.004 while gamma_xy ramps 0.001 .. 0.006 in six
  steps; and an unload-reload pair eps_xx = 0.0025 then 0.0055.
- The kernel replicates the legacy ``getStress`` arithmetic statement by
  statement (same NumPy operation order, including ``0.333333333333333``
  hydrostatic scaling, the ``1.0 / smises`` reciprocal multiply, the
  ``(1.0 + 1.0e-6) * syield`` yield shift, and the ``1.0e-6 * syield0``
  Newton convergence test). Stress, tangent, and kappa match the legacy law
  bit for bit on the documented proportional normal-strain paths; paths
  mixing plastic shear agree to a 1e-12 relative tolerance, the drift being
  rounding-level accumulation through the strain-split identity.
- One legacy statement is deliberately NOT replicated:
  ``IsotropicHardeningPlasticity.py:105-106`` transfers ``flow[:3]`` (the
  normal deviatoric direction) into the shear slots of ``epsilon_p`` and
  ``epsilon_e`` where ``flow[3:]`` belongs. The bug is physically inert — the
  ``epsilon_e + epsilon_p`` sum driving the next increment is preserved — but
  it pollutes the shear split the v3 slot schema promises. The v3 kernel
  writes ``flow[3:]``; the parity battery pins the divergence (legacy shear
  bookkeeping is nonzero on plastic normal paths, v3 is exactly zero) instead
  of inheriting it.
- The legacy law aliases ``tang = self.ctang`` and mutates the elastic
  tangent in the plastic branch, so its later elastic predictors depend on
  evaluation history. That is an impurity the pure v3 contract (accepted rows
  in, trial rows out) cannot reproduce; the parity harness therefore repairs
  the oracle with a pristine elastic predictor before each step. Single-step
  paths match bitwise without the repair; multi-step paths match with it.
- The parity harness drives the legacy law with ``dstrain = eps_total -
  (eelas + eplas)`` from its own committed history, the same identity the
  kernel uses, so no accumulation drift enters the comparison.
- Divergences from legacy are typed, not silent: a return map exhausting the
  hardening table (legacy falls into an ``IndexError``/``NameError``), a
  non-finite predictor, or more than 100 local Newton iterations (legacy only
  prints past 10) report ``REJECT_STEP`` with byte-equal trial rows.

Throughput implementation (M30). Two interchangeable kernels implement the
contract above:

- ``isotropic_hardening_plasticity_kernel_reference`` — the M25 kernel, pure
  NumPy evaluating one entity at a time, kept verbatim as the numerical
  oracle and as the "before" side of the bench material workload.
- ``isotropic_hardening_plasticity_kernel`` — the production kernel: the same
  per-entity arithmetic compiled into one batched SoA pass over the flattened
  entity axis (``@njit(cache=True, parallel=True)``, the landed batch-SoA
  idiom of ``pyfem/v3/fem/element.py``). The numba kernels reference only
  NumPy and same-file symbols, so cache=True is cache-friendly per
  NUMBA_CACHING.md §5.

The optimized kernel reproduces the reference bit for bit on every path —
elastic, plastic, mixed-shear, and REJECT_STEP. It evaluates in two batched
stages: a NumPy predictor stage computes ``dstrain``, ``sigma_trial``, and
``smises`` over the whole SoA block, where the ``einsum(..., optimize=False)``
contractions are per-entity-bitwise equal to the reference's per-entity
``ctang @ dstrain`` and ``np.dot(shear, shear)`` on the reference platform
(``optimize=True`` selects a different BLAS path and is not); a compiled
``@njit(cache=True, parallel=True)`` return-map stage then holds only scalar
and per-component arithmetic, so parallel threading cannot perturb numerics
(a reduction inside the parallel region would lower to a different,
non-bitwise code path — the split exists for exactly that reason). Every
statement preserves the reference's expression order (the
``shydro * np.ones(3)`` no-op multiplies by an exact 1.0 and is elided).
Parity is enforced, not assumed: the landed parity battery runs the optimized
kernel against the legacy oracle, the test suite pins optimized-vs-reference
on deterministic and seeded batches, and the bench material gate
(``material/j2-isotropic-hardening/36864``) admits timings only after bitwise
equality holds. If a future platform's BLAS or einsum path selection breaks
the bitwise identity, those checks fail loudly rather than silently changing
numerics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from pyfem.v3.compile.contracts import (
  STATEFUL_MATERIAL_DESCRIPTOR_SCHEMA,
  StatefulContinuumKernelResult,
)
from pyfem.v3.model.operator import EvaluationStatus, OperatorStateLayout
from pyfem.v3.types import F64

ISOTROPIC_HARDENING_PLASTICITY_STATE_SCHEMA = "pyfem-v3-j2-isotropic-hardening-state-v1"

_RETURN_MAP_TOLERANCE = 1.0e-6
_LOCAL_NEWTON_LIMIT = 100
_ROW_WIDTH = 19

# Flat calibration vector layout: eight scalars, the 6x6 elastic tangent in C
# order, then the two-point hardening table abscissae and ordinates.
_CALIBRATION_SIZE = 8 + 36 + 2 + 2


def isotropic_hardening_plasticity_metadata() -> dict[str, object]:
  """Return the canonical v2 descriptor metadata for the first stateful law."""
  return {
    "schema": STATEFUL_MATERIAL_DESCRIPTOR_SCHEMA,
    "law": "isotropic-hardening-plasticity",
    "stress_state": "plane-strain",
    "parameter_names": [
      "youngs_modulus",
      "poisson_ratio",
      "initial_yield_stress",
      "hardening_slope",
    ],
    "parameter_dtype": "float64",
    "stress_voigt_order": ["xx", "yy", "xy"],
    "strain_shear_convention": "engineering",
    "internal_voigt_order": ["xx", "yy", "zz", "yz", "zx", "xy"],
    "tangent_class": "algorithmic-symmetric",
    "state_schema": ISOTROPIC_HARDENING_PLASTICITY_STATE_SCHEMA,
    "state_slots": [
      {
        "name": "sigma",
        "width": 6,
        "dtype": "float64",
        "lifetime": "accepted-trial",
      },
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
  }


def isotropic_hardening_calibration(
  youngs_modulus: float,
  poisson_ratio: float,
  initial_yield_stress: float,
  hardening_slope: float,
) -> np.ndarray:
  """Pack the law's flat calibration vector with legacy-identical arithmetic."""
  for label, value in (
    ("youngs_modulus", youngs_modulus),
    ("poisson_ratio", poisson_ratio),
    ("initial_yield_stress", initial_yield_stress),
    ("hardening_slope", hardening_slope),
  ):
    if type(value) is not float:
      msg = f"isotropic hardening {label} must be an exact float"
      raise TypeError(msg)
    if not np.isfinite(value):
      msg = f"isotropic hardening {label} must be finite"
      raise ValueError(msg)
  if youngs_modulus <= 0.0:
    msg = "isotropic hardening youngs_modulus must be positive"
    raise ValueError(msg)
  if not -1.0 < poisson_ratio < 0.5:
    msg = "isotropic hardening poisson_ratio must lie between -1 and 0.5"
    raise ValueError(msg)
  if initial_yield_stress <= 0.0:
    msg = "isotropic hardening initial_yield_stress must be positive"
    raise ValueError(msg)
  if hardening_slope < 0.0:
    msg = "isotropic hardening hardening_slope must be non-negative"
    raise ValueError(msg)

  # Legacy constructor expression order (IsotropicHardeningPlasticity.py:18-23).
  e = youngs_modulus
  nu = poisson_ratio
  syield0 = initial_yield_stress
  ebulk3 = e / (1.0 - 2.0 * nu)
  eg2 = e / (1.0 + nu)
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

  # Legacy linear-hardening table as used by the working decks: the prepend
  # in MatUtils.Hardening is a discarded-return no-op, so the two given points
  # are the whole table.
  table_strains = np.array([0.0, 1.0])
  table_stresses = np.array([syield0, syield0 + hardening_slope])

  return np.concatenate(
    (
      np.array(
        [eg, eg2, eg3, ebulk3, elam, syield0, _RETURN_MAP_TOLERANCE],
        dtype=np.float64,
      ),
      np.array([hardening_slope], dtype=np.float64),
      ctang.reshape(-1),
      table_strains,
      table_stresses,
    )
  )


def _hardening(
  table_strains: np.ndarray,
  table_stresses: np.ndarray,
  eqplas: float,
) -> tuple[float, float] | None:
  """Piecewise-linear hardening lookup; ``None`` beyond the tabulated range."""
  for i in range(len(table_strains) - 1):
    eqpl1 = table_strains[i + 1]
    if eqplas < eqpl1:
      eqpl0 = table_strains[i]
      deqpl = eqpl1 - eqpl0
      syiel0 = table_stresses[i]
      dsyiel = table_stresses[i + 1] - syiel0
      hard = dsyiel / deqpl
      return syiel0 + (eqplas - eqpl0) * hard, hard
  return None


def _von_mises(sigma: np.ndarray) -> float:
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


def isotropic_hardening_plasticity_kernel_reference(
  strains: np.ndarray,
  accepted_rows: np.ndarray,
  calibration: np.ndarray,
) -> StatefulContinuumKernelResult:
  """The M25 reference kernel: pure NumPy, one entity at a time.

  Kept verbatim as the numerical oracle for the optimized
  ``isotropic_hardening_plasticity_kernel`` (same arithmetic, one batched SoA
  pass) and as the "before" side of the bench material workload.

  ``strains`` holds total engineering 6-Voigt strain per entity;
  ``accepted_rows`` holds the committed 19-float rows. The strain increment is
  ``strains - (epsilon_e + epsilon_p)`` from the accepted row, the identity the
  parity harness also uses to drive the legacy oracle.
  """
  if calibration.shape != (_CALIBRATION_SIZE,):
    msg = "isotropic hardening kernel requires the packed calibration vector"
    raise TypeError(msg)
  if not bool(np.isfinite(strains).all()):
    return _reject(accepted_rows)
  eg = calibration[0]
  eg3 = calibration[2]
  ebulk3 = calibration[3]
  syield0 = calibration[5]
  tolerance = calibration[6]
  hard_slope = calibration[7]
  ctang = calibration[8:44].reshape(6, 6)
  table_strains = calibration[44:46]
  table_stresses = calibration[46:48]

  entity_count = strains.shape[0]
  stresses = np.empty((entity_count, 6), dtype=np.float64)
  tangents = np.empty((entity_count, 6, 6), dtype=np.float64)
  trial_rows = np.empty((entity_count, _ROW_WIDTH), dtype=np.float64)

  for index in range(entity_count):
    row = accepted_rows[index]
    with np.errstate(over="ignore", invalid="ignore"):
      dstrain = strains[index] - (row[6:12] + row[12:18])
      eelas = row[6:12] + dstrain
      sigma = row[0:6] + ctang @ dstrain
      smises = _von_mises(sigma)
    eplas = np.array(row[12:18], copy=True)
    eqplas = float(row[18])
    if not bool(np.isfinite(sigma).all()) or not np.isfinite(smises):
      return _reject(accepted_rows)
    hardening = _hardening(table_strains, table_stresses, eqplas)
    if hardening is None:
      return _reject(accepted_rows)
    syield, hard = hardening

    if smises > (1.0 + tolerance) * syield:
      shydro = 0.333333333333333 * (sigma[0] + sigma[1] + sigma[2])
      flow = np.array(sigma, copy=True)
      flow[:3] = flow[:3] - shydro * np.ones(3)
      flow *= 1.0 / smises

      syield = syield0
      deqpl = 0.0
      rhs = syield
      iterations = 0
      while abs(rhs) > tolerance * syield0:
        iterations += 1
        if iterations > _LOCAL_NEWTON_LIMIT:
          return _reject(accepted_rows)
        rhs = smises - eg3 * deqpl - syield
        deqpl = deqpl + rhs / (eg3 + hard)
        if not np.isfinite(deqpl):
          return _reject(accepted_rows)
        hardening = _hardening(table_strains, table_stresses, eqplas + deqpl)
        if hardening is None:
          return _reject(accepted_rows)
        syield, hard = hardening

      eplas[:3] += 1.5 * flow[:3] * deqpl
      eelas[:3] += -1.5 * flow[:3] * deqpl
      eplas[3:] += 3.0 * flow[3:] * deqpl
      eelas[3:] += -3.0 * flow[3:] * deqpl

      sigma = flow * syield
      sigma[:3] += shydro
      eqplas += deqpl

      effg = eg * syield / smises
      effg2 = 2.0 * effg
      effg3 = 3.0 * effg
      efflam = 1.0 / 3.0 * (ebulk3 - effg2)
      # Legacy tangent semantics: the declared slope (self.hard), which the
      # two-point table matches exactly in this linear-hardening law.
      effhdr = eg3 * hard_slope / (eg3 + hard_slope) - effg3

      # The legacy law aliases and mutates its elastic tangent here; the pure
      # v3 kernel writes the algorithmic tangent into a fresh copy instead.
      tang = np.array(ctang, copy=True)
      tang[:3, :3] = efflam
      for i in range(3):
        tang[i, i] += effg2
        tang[i + 3, i + 3] += effg
      tang += effhdr * np.outer(flow, flow)
    else:
      tang = ctang

    stresses[index] = sigma
    tangents[index] = tang
    trial_rows[index] = np.concatenate((sigma, eelas, eplas, [eqplas]))

  return StatefulContinuumKernelResult(
    stresses=stresses,
    tangents=tangents,
    trial_rows=trial_rows,
    status=EvaluationStatus.OK,
  )


@njit(cache=True)
def _hardening_lookup(
  table_strains: F64,
  table_stresses: F64,
  eqplas: float,
) -> tuple[float, float, bool]:
  """Piecewise-linear hardening lookup; ``found=False`` beyond the table."""
  for i in range(len(table_strains) - 1):
    eqpl1 = table_strains[i + 1]
    if eqplas < eqpl1:
      eqpl0 = table_strains[i]
      deqpl = eqpl1 - eqpl0
      syiel0 = table_stresses[i]
      dsyiel = table_stresses[i + 1] - syiel0
      hard = dsyiel / deqpl
      return syiel0 + (eqplas - eqpl0) * hard, hard, True
  return 0.0, 0.0, False


@njit(cache=True, parallel=True)
def _return_map_batched(
  dstrain: F64,
  sigma_trial: F64,
  smises: F64,
  accepted_rows: F64,
  calibration: F64,
  stresses: F64,
  tangents: F64,
  trial_rows: F64,
) -> int:
  """Batched radial return: one compiled pass over the flattened state block.

  The elastic predictor (``dstrain``, ``sigma_trial``, ``smises``) arrives
  precomputed by the wrapper's batched NumPy stage, so this pass holds only
  scalar and per-component arithmetic — parallel threading cannot perturb
  numerics, and outputs stay bitwise identical to
  ``isotropic_hardening_plasticity_kernel_reference`` (pinned by the parity
  battery and the bench material gate). Returns the number of rejected
  entities; any nonzero count rejects the whole batch.
  """
  eg = calibration[0]
  eg3 = calibration[2]
  ebulk3 = calibration[3]
  syield0 = calibration[5]
  tolerance = calibration[6]
  hard_slope = calibration[7]
  ctang = calibration[8:44].reshape(6, 6)
  table_strains = calibration[44:46]
  table_stresses = calibration[46:48]

  rejects = 0
  for index in prange(dstrain.shape[0]):
    row = accepted_rows[index]
    eelas = row[6:12] + dstrain[index]
    eplas = row[12:18].copy()
    eqplas = row[18]
    sigma = sigma_trial[index]
    sm = smises[index]

    finite = np.isfinite(sm)
    for component in range(6):
      if not np.isfinite(sigma[component]):
        finite = False
    if not finite:
      rejects += 1
      continue

    syield, hard, found = _hardening_lookup(table_strains, table_stresses, eqplas)
    if not found:
      rejects += 1
      continue

    tang = np.empty((6, 6))
    if sm > (1.0 + tolerance) * syield:
      shydro = 0.333333333333333 * (sigma[0] + sigma[1] + sigma[2])
      flow = sigma.copy()
      for component in range(3):
        flow[component] -= shydro
      reciprocal = 1.0 / sm
      for component in range(6):
        flow[component] *= reciprocal

      syield = syield0
      deqpl = 0.0
      rhs = syield
      iterations = 0
      converged = True
      while abs(rhs) > tolerance * syield0:
        iterations += 1
        if iterations > _LOCAL_NEWTON_LIMIT:
          converged = False
          break
        rhs = sm - eg3 * deqpl - syield
        deqpl = deqpl + rhs / (eg3 + hard)
        if not np.isfinite(deqpl):
          converged = False
          break
        syield, hard, found = _hardening_lookup(
          table_strains, table_stresses, eqplas + deqpl
        )
        if not found:
          converged = False
          break
      if not converged:
        rejects += 1
        continue

      for component in range(3):
        eplas[component] += 1.5 * flow[component] * deqpl
        eelas[component] += -1.5 * flow[component] * deqpl
      for component in range(3, 6):
        eplas[component] += 3.0 * flow[component] * deqpl
        eelas[component] += -3.0 * flow[component] * deqpl

      sigma = flow * syield
      for component in range(3):
        sigma[component] += shydro
      eqplas += deqpl

      effg = eg * syield / sm
      effg2 = 2.0 * effg
      effg3 = 3.0 * effg
      efflam = 1.0 / 3.0 * (ebulk3 - effg2)
      # Legacy tangent semantics: the declared slope (self.hard), which the
      # two-point table matches exactly in this linear-hardening law.
      effhdr = eg3 * hard_slope / (eg3 + hard_slope) - effg3

      for i in range(6):
        for j in range(6):
          tang[i, j] = ctang[i, j]
      for i in range(3):
        for j in range(3):
          tang[i, j] = efflam
      for i in range(3):
        tang[i, i] += effg2
        tang[i + 3, i + 3] += effg
      for i in range(6):
        for j in range(6):
          tang[i, j] += effhdr * (flow[i] * flow[j])
    else:
      for i in range(6):
        for j in range(6):
          tang[i, j] = ctang[i, j]

    stresses[index] = sigma
    tangents[index] = tang
    for component in range(6):
      trial_rows[index, component] = sigma[component]
      trial_rows[index, component + 6] = eelas[component]
      trial_rows[index, component + 12] = eplas[component]
    trial_rows[index, 18] = eqplas
  return rejects


def isotropic_hardening_plasticity_kernel(
  strains: np.ndarray,
  accepted_rows: np.ndarray,
  calibration: np.ndarray,
) -> StatefulContinuumKernelResult:
  """Evaluate the batched radial-return law over flattened state entities.

  ``strains`` holds total engineering 6-Voigt strain per entity;
  ``accepted_rows`` holds the committed 19-float rows. The strain increment is
  ``strains - (epsilon_e + epsilon_p)`` from the accepted row, the identity the
  parity harness also uses to drive the legacy oracle. Outputs are bitwise
  identical to ``isotropic_hardening_plasticity_kernel_reference`` (module
  docstring, "Throughput implementation").
  """
  if calibration.shape != (_CALIBRATION_SIZE,):
    msg = "isotropic hardening kernel requires the packed calibration vector"
    raise TypeError(msg)
  if not bool(np.isfinite(strains).all()):
    return _reject(accepted_rows)
  entity_count = strains.shape[0]
  if strains.shape != (entity_count, 6) or accepted_rows.shape != (
    entity_count,
    _ROW_WIDTH,
  ):
    msg = "isotropic hardening kernel requires (n, 6) strains and (n, 19) rows"
    raise ValueError(msg)

  strains64 = np.ascontiguousarray(strains, dtype=np.float64)
  rows64 = np.ascontiguousarray(accepted_rows, dtype=np.float64)
  ctang = calibration[8:44].reshape(6, 6)
  with np.errstate(over="ignore", invalid="ignore"):
    # Batched elastic predictor over the SoA block. The einsum contractions
    # with optimize=False are per-entity-bitwise equal to the reference
    # kernel's `ctang @ dstrain` and `np.dot(shear, shear)` on the reference
    # platform (optimize=True selects a different BLAS path and is not).
    dstrain = strains64 - (rows64[:, 6:12] + rows64[:, 12:18])
    sigma_trial = rows64[:, 0:6] + np.einsum(
      "ij,nj->ni", ctang, dstrain, optimize=False
    )
    shear = sigma_trial[:, 3:]
    smises = (
      (sigma_trial[:, 0] - sigma_trial[:, 1]) * (sigma_trial[:, 0] - sigma_trial[:, 1])
      + (sigma_trial[:, 1] - sigma_trial[:, 2])
      * (sigma_trial[:, 1] - sigma_trial[:, 2])
      + (sigma_trial[:, 2] - sigma_trial[:, 0])
      * (sigma_trial[:, 2] - sigma_trial[:, 0])
    )
    smises += 6.0 * np.einsum("ni,ni->n", shear, shear, optimize=False)
    smises = np.sqrt(0.5 * smises)

  stresses = np.empty((entity_count, 6), dtype=np.float64)
  tangents = np.empty((entity_count, 6, 6), dtype=np.float64)
  trial_rows = np.empty((entity_count, _ROW_WIDTH), dtype=np.float64)
  rejects = _return_map_batched(
    dstrain,
    sigma_trial,
    smises,
    rows64,
    np.ascontiguousarray(calibration, dtype=np.float64),
    stresses,
    tangents,
    trial_rows,
  )
  if rejects:
    return _reject(accepted_rows)
  return StatefulContinuumKernelResult(
    stresses=stresses,
    tangents=tangents,
    trial_rows=trial_rows,
    status=EvaluationStatus.OK,
  )


def isotropic_hardening_initial_state(
  parameters: tuple[float, ...],
  layout: OperatorStateLayout,
) -> np.ndarray:
  """Bind the all-zero initial rows of the legacy constructor explicitly."""
  if layout.row_width != _ROW_WIDTH:
    msg = "isotropic hardening initial state requires the 19-float row layout"
    raise ValueError(msg)
  del parameters
  return np.zeros(layout.row_shape, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class IsotropicHardeningPlasticityBinding:
  """The registry binding object wiring the first stateful law to the ABI."""

  def __call__(self, *parameters: float) -> np.ndarray:
    if len(parameters) != 4:
      msg = "isotropic hardening requires exactly four calibration parameters"
      raise TypeError(msg)
    return isotropic_hardening_calibration(*parameters)

  def descriptor_metadata(self) -> dict[str, object]:
    return isotropic_hardening_plasticity_metadata()

  def kernel(
    self,
    strains: np.ndarray,
    accepted_rows: np.ndarray,
    calibration: np.ndarray,
  ) -> StatefulContinuumKernelResult:
    return isotropic_hardening_plasticity_kernel(strains, accepted_rows, calibration)

  def initial_state(
    self,
    parameters: tuple[float, ...],
    layout: OperatorStateLayout,
  ) -> np.ndarray:
    return isotropic_hardening_initial_state(parameters, layout)


ISOTROPIC_HARDENING_PLASTICITY_BINDING = IsotropicHardeningPlasticityBinding()
