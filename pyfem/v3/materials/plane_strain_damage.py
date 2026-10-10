"""Plane-strain isotropic damage: the second stateful v3 material law.

A pure, batched port of the legacy reference
``pyfem/materials/PlaneStrainDamage.py`` (the numerical oracle per AGENTS.md)
onto the v2 stateful descriptor ABI. The law is the de Vree modified
von Mises isotropic damage model: a strain-space equivalent measure
``eps(eq)`` drives a max-envelope internal variable ``kappa`` and a damage
factor ``omega(kappa)`` scaling the plane-strain elastic stiffness. The law is
3-Voigt internally (``[xx, yy, xy]``, engineering shear) with the
out-of-plane strain reconstructed as ``ezz = nu/(nu-1) * (exx + eyy)`` inside
the equivalent measure exactly as the legacy law does; the continuum operator
embeds the port strain as ``[xx, yy, 0, 0, 0, xy]`` in the 6-Voigt internal
order and truncates stress/tangent back to ``[xx, yy, xy]``.

Parameters (``parameter_names`` order): ``youngs_modulus`` E > 0,
``poisson_ratio`` -1 < nu < 0.5, ``kappa_0`` > 0 (damage onset threshold),
``kappa_c`` > kappa_0 (failure equivalent strain), and ``strength_ratio``
k > 0 (the de Vree compressive-to-tensile strength ratio; k = 1 degenerates
the measure to the von Mises strain norm). The documented configuration is
the landed chapter-6 deck ``examples/ch06/ContDamExample.pro``: E = 100,
nu = 0.3, k = 1.0, kappa0 = 1.0e-6, kappac = 1.0e-5.

State rows (1 float per entity, flattened element x ip x slot): ``kappa``,
the maximum equivalent strain ever accepted, annotated ``envelope-max``. The
initial state is all zero, matching the legacy constructor's zeroed history
commit; the binding still declares ``initial_state`` explicitly so the owner
never relies on implicit zero-fill for this law.

Tangent class: ``algorithmic-nonsymmetric`` — the first nonsymmetric
algorithmic tangent in v3. On the progressive branch (``eps(eq)`` exceeds the
accepted kappa) the consistent tangent adds the rank-1 correction
``-domegadkappa * outer(effStress, detadstrain)`` (legacy
PlaneStrainDamage.py:69-70) to the symmetric ``(1 - omega) * De``; on elastic
and unloading branches it is exactly ``(1 - omega) * De``. The Jacobian
channel therefore reports ``linear=False, symmetric=False`` honestly.

The legacy zero-return divergence (repair-confirmed history, not inherited).
Pre-repair legacy ``getEquivStrain`` computed ``detadstrain`` and then
returned the never-assigned ``depsdstrain = zeros(3)``
(``PlaneStrainDamage.py:99,133``) — present since the first commit
(``d041176``) and in the shipped v1.0 tarballs — so the legacy rank-1
correction vanished identically and the legacy tangent was always the
symmetric ``(1 - omega) * De``. The discarded legacy expression was itself
additionally inconsistent in its shear component: it weighed
``d(exy)/d(strain)`` as ``0.5 * O3`` (``PlaneStrainDamage.py:96``) against
its own ``J2 = ... + exy**2`` (``PlaneStrainDamage.py:90``), halving the true
shear derivative. The v3 kernel differentiates the equivalent strain exactly
(``d(exy)/d(strain) = O3``), so its tangent is the algorithmically
consistent tangent of the shared stress response — finite-difference
verified in the parity battery. M55 repaired both legacy defects (commits
5ff40d5 / cb4f23a, merge 945a95d): the legacy law now returns the computed
``detadstrain`` with the exact ``O3`` shear weight, arithmetic identical to
this kernel. The v3 kernel is unchanged and the relationship is now
parity-where-repaired: legacy and v3 tangents match bit for bit on every
branch. This is what makes the declared ``algorithmic-nonsymmetric``
class honest rather than aspirational. The parity contract with the legacy
oracle (exercised by ``test/v3/test_v3_damage.py``):

- Documented strain paths (total 3-Voigt, engineering shear, per committed
  step): an elastic ramp eps_xx = 0.5e-6, 0.8e-6 (below the onset equivalent
  strain ~1.024e-6); a progressive ramp eps_xx = 2e-6 .. 8e-6; a mixed
  segment holding eps_xx = 4e-6 while gamma_xy ramps 1e-6 .. 4e-6; an
  unload-reload pair eps_xx = 3e-6 then 9e-6; and a full-softening step
  eps_xx = 2e-5 (beyond kappac, omega = 1).
- Stresses and the kappa envelope match the legacy law bit for bit on every
  documented path — the stress response never reads the tangent, and the
  kernel replicates the legacy ``getStress``/``getEquivStrain``/``getDamage``
  arithmetic statement by statement (same NumPy operation order, including
  the branch-local recomputation of ``eps`` at ``PlaneStrainDamage.py:116``
  and :126).
- Tangents match the legacy law bit for bit on every branch. On elastic and
  unloading branches both sides return ``(1 - omega) * De``; on progressive
  branches both sides carry the rank-1 correction (the repaired state of the
  zero-return divergence above). The parity battery recomputes the legacy
  side independently (the ``getDamage`` factor on the committed kappa), and
  a finite-difference check pins both tangents as the algorithmically
  consistent derivative of the shared stress response (the pre-repair legacy
  tangent — the symmetric secant — failed that check materially; the pinned
  pre-repair difference was exactly rank-1 with the effective-stress left
  vector).
- Divergences from legacy failures are typed, not silent: a non-finite
  strain batch or a non-finite equivalent strain reports ``REJECT_STEP``
  with byte-equal trial rows, never an exception. Beyond kappac the law
  returns omega = 1 with a zero stress and zero tangent (legacy semantics);
  the driver owns the resulting singular-tangent cutback.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyfem.v3.compile.contracts import (
  STATEFUL_MATERIAL_DESCRIPTOR_SCHEMA,
  StatefulContinuumKernelResult,
)
from pyfem.v3.model.operator import EvaluationStatus, OperatorStateLayout

PLANE_STRAIN_DAMAGE_STATE_SCHEMA = "pyfem-v3-plane-strain-damage-state-v1"

_ROW_WIDTH = 1

# Flat calibration vector layout: the two damage envelope bounds, the five
# legacy derived constants (a1, a2, a3, a4, c), then the 3x3 plane-strain
# elastic tangent in C order.
_CALIBRATION_SIZE = 7 + 9


def plane_strain_damage_metadata() -> dict[str, object]:
  """Return the canonical v2 descriptor metadata for the damage law."""
  return {
    "schema": STATEFUL_MATERIAL_DESCRIPTOR_SCHEMA,
    "law": "plane-strain-damage",
    "stress_state": "plane-strain",
    "parameter_names": [
      "youngs_modulus",
      "poisson_ratio",
      "kappa_0",
      "kappa_c",
      "strength_ratio",
    ],
    "parameter_dtype": "float64",
    "stress_voigt_order": ["xx", "yy", "xy"],
    "strain_shear_convention": "engineering",
    "internal_voigt_order": ["xx", "yy", "zz", "yz", "zx", "xy"],
    "tangent_class": "algorithmic-nonsymmetric",
    "state_schema": PLANE_STRAIN_DAMAGE_STATE_SCHEMA,
    "state_slots": [
      {
        "name": "kappa",
        "width": 1,
        "dtype": "float64",
        "lifetime": "accepted-trial",
        "annotation": "envelope-max",
      },
    ],
  }


def plane_strain_damage_calibration(
  youngs_modulus: float,
  poisson_ratio: float,
  kappa_0: float,
  kappa_c: float,
  strength_ratio: float,
) -> np.ndarray:
  """Pack the law's flat calibration vector with legacy-identical arithmetic."""
  for label, value in (
    ("youngs_modulus", youngs_modulus),
    ("poisson_ratio", poisson_ratio),
    ("kappa_0", kappa_0),
    ("kappa_c", kappa_c),
    ("strength_ratio", strength_ratio),
  ):
    if type(value) is not float:
      msg = f"plane-strain damage {label} must be an exact float"
      raise TypeError(msg)
    if not np.isfinite(value):
      msg = f"plane-strain damage {label} must be finite"
      raise ValueError(msg)
  if youngs_modulus <= 0.0:
    msg = "plane-strain damage youngs_modulus must be positive"
    raise ValueError(msg)
  if not -1.0 < poisson_ratio < 0.5:
    msg = "plane-strain damage poisson_ratio must lie between -1 and 0.5"
    raise ValueError(msg)
  if kappa_0 <= 0.0:
    msg = "plane-strain damage kappa_0 must be positive"
    raise ValueError(msg)
  if kappa_c <= kappa_0:
    msg = "plane-strain damage kappa_c must exceed kappa_0"
    raise ValueError(msg)
  if strength_ratio <= 0.0:
    msg = "plane-strain damage strength_ratio must be positive"
    raise ValueError(msg)

  # Legacy constructor expression order (PlaneStrainDamage.py:20-35).
  e = youngs_modulus
  nu = poisson_ratio
  k = strength_ratio
  de = np.zeros(shape=(3, 3))
  de[0, 0] = e * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
  de[0, 1] = de[0, 0] * nu / (1.0 - nu)
  de[1, 0] = de[0, 1]
  de[1, 1] = de[0, 0]
  de[2, 2] = de[0, 0] * 0.5 * (1.0 - 2.0 * nu) / (1.0 - nu)

  a1 = 1.0 / (2.0 * k)
  a2 = (k - 1.0) / (1.0 - 2.0 * nu)
  a3 = 12.0 * k / ((1.0 + nu) ** 2)
  a4 = np.sqrt(a2**2 + a3 * (1.0 / 3.0))
  c = nu / (nu - 1.0)

  return np.concatenate(
    (
      np.array([kappa_0, kappa_c, a1, a2, a3, a4, c], dtype=np.float64),
      de.reshape(-1),
    )
  )


def _damage(kappa: float, kappa_0: float, kappa_c: float) -> tuple[float, float]:
  """Legacy ``getDamage`` (PlaneStrainDamage.py:140-152), statement order kept."""
  if kappa <= kappa_0:
    return 0.0, 0.0
  if kappa_0 < kappa < kappa_c:
    fac = kappa_c / kappa
    omega = fac * (kappa - kappa_0) / (kappa_c - kappa_0)
    domegadkappa = fac / (kappa_c - kappa_0) - (omega / kappa)
    return omega, domegadkappa
  return 1.0, 0.0


def _reject(accepted_rows: np.ndarray) -> StatefulContinuumKernelResult:
  entity_count = accepted_rows.shape[0]
  return StatefulContinuumKernelResult(
    stresses=np.zeros((entity_count, 6), dtype=np.float64),
    tangents=np.zeros((entity_count, 6, 6), dtype=np.float64),
    trial_rows=np.array(accepted_rows, dtype=np.float64, copy=True),
    status=EvaluationStatus.REJECT_STEP,
  )


def plane_strain_damage_kernel(
  strains: np.ndarray,
  accepted_rows: np.ndarray,
  calibration: np.ndarray,
) -> StatefulContinuumKernelResult:
  """Evaluate the batched isotropic damage law over flattened state entities.

  ``strains`` holds total engineering 6-Voigt strain per entity (the
  plane-strain embedding ``[xx, yy, 0, 0, 0, xy]``); ``accepted_rows`` holds
  the committed 1-float kappa rows. The stress/tangent arithmetic replicates
  the legacy ``getStress`` statement by statement — post-M55 without
  residue: the two pre-repair divergences of the rank-1 correction (module
  docstring, "The legacy zero-return divergence") were repaired legacy-side
  (5ff40d5 / cb4f23a), so legacy now also returns the computed derivative
  and differentiates the shear component exactly.
  """
  if calibration.shape != (_CALIBRATION_SIZE,):
    msg = "plane-strain damage kernel requires the packed calibration vector"
    raise TypeError(msg)
  if not bool(np.isfinite(strains).all()):
    return _reject(accepted_rows)
  entity_count = strains.shape[0]
  if strains.shape != (entity_count, 6) or accepted_rows.shape != (
    entity_count,
    _ROW_WIDTH,
  ):
    msg = "plane-strain damage kernel requires (n, 6) strains and (n, 1) rows"
    raise ValueError(msg)

  kappa_0 = calibration[0]
  kappa_c = calibration[1]
  a1 = calibration[2]
  a2 = calibration[3]
  a3 = calibration[4]
  a4 = calibration[5]
  c = calibration[6]
  de = calibration[7:16].reshape(3, 3)
  sc = 1.0 / 3.0  # legacy class attribute ``sc = 1./3.`` (PlaneStrainDamage.py:14)
  o4 = np.array([a4, a4, 2.0 * a3])

  stresses = np.zeros((entity_count, 6), dtype=np.float64)
  tangents = np.zeros((entity_count, 6, 6), dtype=np.float64)
  trial_rows = np.empty((entity_count, _ROW_WIDTH), dtype=np.float64)

  for index in range(entity_count):
    strain = strains[index]
    exx = strain[0]
    eyy = strain[1]
    exy = strain[5]
    with np.errstate(over="ignore", invalid="ignore"):
      # Legacy getEquivStrain (PlaneStrainDamage.py:82-131), statement order
      # kept; post-M55 the legacy return is the same computed
      # ``detadstrain`` (pre-repair it read the never-assigned
      # ``depsdstrain`` — repaired in 5ff40d5).
      ezz = c * (exx + eyy)
      i1 = exx + eyy + ezz
      j2 = (exx**2 + eyy**2 + ezz**2 - exx * eyy - eyy * ezz - exx * ezz) / 3.0 + exy**2

      dexxdstrain = np.array([1.0, 0.0, 0.0])
      deyydstrain = np.array([0.0, 1.0, 0.0])
      # Legacy wrote ``0.5 * self.O3`` here pre-M55 (PlaneStrainDamage.py:96),
      # halving the shear derivative of its own ``J2 = ... + exy**2``; the
      # cb4f23a repair made legacy differentiate its equivalent strain
      # exactly, matching this kernel.
      dexydstrain = np.array([0.0, 0.0, 1.0])
      dezzdstrain = c * (dexxdstrain + deyydstrain)

      d_i1_dstrain = dexxdstrain + deyydstrain + dezzdstrain

      d_j2_dstrain = sc * (2.0 * exx - eyy - ezz) * dexxdstrain
      d_j2_dstrain += sc * (2.0 * eyy - exx - ezz) * deyydstrain
      d_j2_dstrain += sc * (2.0 * ezz - exx - eyy) * dezzdstrain
      d_j2_dstrain += 2.0 * exy * dexydstrain

      disc = (a2 * i1) ** 2 + a3 * j2
      ddiscd_i1 = 2.0 * a2**2 * i1
      ddiscd_j2 = a3

      if disc < 1e-16:
        tmp = 0.0
        dtmpdstrain = o4
        eps = a1 * (a2 * i1 + tmp)
        detad_i1 = a1 * a2
        detadstrain = detad_i1 * d_i1_dstrain + a1 * dtmpdstrain
      else:
        tmp = np.sqrt(disc)
        dtmpd_i1 = (0.5 / tmp) * ddiscd_i1
        dtmpd_j2 = (0.5 / tmp) * ddiscd_j2
        eps = a1 * (a2 * i1 + tmp)
        detad_i1 = a1 * (a2 + dtmpd_i1)
        detad_j2 = a1 * dtmpd_j2
        detadstrain = detad_i1 * d_i1_dstrain + detad_j2 * d_j2_dstrain

    if not np.isfinite(eps):
      return _reject(accepted_rows)

    # Legacy getStress (PlaneStrainDamage.py:48-75), statement order kept.
    kappa = accepted_rows[index, 0]
    if eps > kappa:
      progressive = True
      kappa = eps
    else:
      progressive = False

    omega, domegadkappa = _damage(kappa, kappa_0, kappa_c)

    strain3 = np.array([exx, eyy, exy])
    eff_stress = np.dot(de, strain3)
    stress3 = (1.0 - omega) * eff_stress
    tang3 = (1.0 - omega) * de
    if progressive:
      tang3 += -domegadkappa * np.outer(eff_stress, detadstrain)
    if not bool(np.isfinite(stress3).all()) or not bool(np.isfinite(tang3).all()):
      return _reject(accepted_rows)

    stresses[index, 0] = stress3[0]
    stresses[index, 1] = stress3[1]
    stresses[index, 5] = stress3[2]
    for row, row6 in enumerate((0, 1, 5)):
      for column, column6 in enumerate((0, 1, 5)):
        tangents[index, row6, column6] = tang3[row, column]
    trial_rows[index, 0] = kappa

  return StatefulContinuumKernelResult(
    stresses=stresses,
    tangents=tangents,
    trial_rows=trial_rows,
    status=EvaluationStatus.OK,
  )


def plane_strain_damage_initial_state(
  parameters: tuple[float, ...],
  layout: OperatorStateLayout,
) -> np.ndarray:
  """Bind the all-zero initial rows of the legacy constructor explicitly."""
  if layout.row_width != _ROW_WIDTH:
    msg = "plane-strain damage initial state requires the 1-float kappa row layout"
    raise ValueError(msg)
  del parameters
  return np.zeros(layout.row_shape, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class PlaneStrainDamageBinding:
  """The registry binding object wiring the damage law to the v2 ABI."""

  def __call__(self, *parameters: float) -> np.ndarray:
    if len(parameters) != 5:
      msg = "plane-strain damage requires exactly five calibration parameters"
      raise TypeError(msg)
    return plane_strain_damage_calibration(*parameters)

  def descriptor_metadata(self) -> dict[str, object]:
    return plane_strain_damage_metadata()

  def kernel(
    self,
    strains: np.ndarray,
    accepted_rows: np.ndarray,
    calibration: np.ndarray,
  ) -> StatefulContinuumKernelResult:
    return plane_strain_damage_kernel(strains, accepted_rows, calibration)

  def initial_state(
    self,
    parameters: tuple[float, ...],
    layout: OperatorStateLayout,
  ) -> np.ndarray:
    return plane_strain_damage_initial_state(parameters, layout)


PLANE_STRAIN_DAMAGE_BINDING = PlaneStrainDamageBinding()
