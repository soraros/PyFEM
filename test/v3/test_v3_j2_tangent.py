# SPDX-License-Identifier: MIT

"""J2 plastic tangent exactness: the FD pin and the Newton-rate protocol.

Finding 20261007-agent-a3 (HIGH): the legacy-inherited J2 consistent
tangent carried the elastic shear modulus G one time too many on the
shear-shear diagonals at plastic integration points (~31% of max|tang|),
invisible at converged states but degrading global Newton from quadratic to
linear and corrupting every derivative-as-tangent consumer (IFT
sensitivities, bifurcation, arc-length quality). The v3 kernels assign the
shear block like the normal block — the documented divergence of
``pyfem/v3/materials/isotropic_hardening_plasticity.py``, pinned against
the legacy oracle in ``test_v3_stateful_plasticity.py``. This battery pins
the repair itself:

- the returned tangent is the exact derivative of the kernel's own stress
  map: central finite differences at accepted states agree to 3.9e-10
  relative of max|tang| over every plastic step of the documented paths
  (the pre-fix construction fails the same finite difference by ~0.33);
- a load-controlled J2 bending case pins quadratic Newton convergence as an
  iteration-count bound: the ``quad8_patch(4, 1)`` cantilever (40 free
  DOFs, 4/36 plastic integration points) commits one load step to 1e-12 in
  5 iterations, where the pre-fix tangent needed 41 with residual ratio
  ~0.52 — a future tangent regression surfaces here as a convergence-rate
  failure.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from pyfem.v3 import authoring
from pyfem.v3.compile.continuum import plasticity_reference_registry
from pyfem.v3.compile.system import compile_system
from pyfem.v3.constraints import compile_constraint_map
from pyfem.v3.driver import (
  DriverStatus,
  NonlinearStaticDriver,
  NonlinearStaticSettings,
)
from pyfem.v3.materials.isotropic_hardening_plasticity import (
  isotropic_hardening_calibration,
  isotropic_hardening_plasticity_kernel,
)
from pyfem.v3.model.operator import EvaluationStatus
from pyfem.v3.spec.program import (
  AffineCoefficientSpec,
  AffineValueSpec,
  DofRef,
  NodalLoadSpec,
  ProgramCoordinateSpec,
  ProgramCoordinateValue,
  ProgramPoint,
)

# The documented M25 parity configuration (test_v3_stateful_plasticity.py).
_E = 210000.0
_NU = 0.3
_SYIELD = 250.0
_HARD = 1000.0
# The elastic shear modulus (calibration slot 0, constructor expression
# order): the pre-fix tangent's excess on the shear-shear diagonals.
_EG = 0.5 * (_E / (1.0 + _NU))
_FD_STEP = 1.0e-7


def _plastic_paths() -> list[np.ndarray]:
  uniaxial = [
    np.array([eps, 0.0, 0.0, 0.0, 0.0, 0.0]) for eps in np.linspace(0.0002, 0.004, 10)
  ]
  shear = [
    np.array([0.004, 0.0, 0.0, 0.0, 0.0, gamma])
    for gamma in np.linspace(0.001, 0.006, 6)
  ]
  return [*uniaxial, *shear]


def test_plastic_tangent_is_the_exact_stress_map_derivative_by_fd() -> None:
  """Central FD of the kernel's own stress update vs the returned tangent.

  At every accepted state of the documented paths the stress map is smooth
  on the 1e-7 stencil (the nearest branch boundary is O(1e-4) of strain
  away), so the finite difference is exact up to rounding. The conviction
  half of the pin: the pre-fix construction — the same tangent with the
  elastic G added back onto the shear-shear diagonals — fails the same
  comparison by ~0.33 of max|tang| at every plastic step.
  """
  calibration = isotropic_hardening_calibration(_E, _NU, _SYIELD, _HARD)
  rows = np.zeros((1, 19))
  worst = 0.0
  worst_legacy_like = 1.0
  went_plastic = False
  for strain in _plastic_paths():
    strain = np.array(strain, dtype=np.float64)
    base = isotropic_hardening_plasticity_kernel(strain[None, :], rows, calibration)
    assert base.status is EvaluationStatus.OK
    tangent = base.tangents[0]
    scale = float(np.max(np.abs(tangent)))
    fd = np.zeros((6, 6))
    for component in range(6):
      delta = _FD_STEP * np.eye(6)[component]
      plus = isotropic_hardening_plasticity_kernel(
        (strain + delta)[None, :], rows, calibration
      )
      minus = isotropic_hardening_plasticity_kernel(
        (strain - delta)[None, :], rows, calibration
      )
      assert plus.status is EvaluationStatus.OK
      assert minus.status is EvaluationStatus.OK
      fd[:, component] = (plus.stresses[0] - minus.stresses[0]) / (2.0 * _FD_STEP)
    deviation = float(np.max(np.abs(fd - tangent))) / scale
    worst = max(worst, deviation)
    if base.trial_rows[0, 18] > 0.0:
      went_plastic = True
      legacy_like = tangent.copy()
      for index in (3, 4, 5):
        legacy_like[index, index] += _EG
      legacy_deviation = float(np.max(np.abs(fd - legacy_like))) / scale
      worst_legacy_like = min(worst_legacy_like, legacy_deviation)
    rows = base.trial_rows
  assert went_plastic
  assert worst < 1.0e-8, worst  # observed: 3.9e-10 over all steps
  assert worst_legacy_like > 0.1, worst_legacy_like  # observed: >= 0.32


def _cantilever_driver(settings: NonlinearStaticSettings) -> NonlinearStaticDriver:
  """The protocol case: a quad8 cantilever under load-controlled tip bending.

  ``quad8_patch(4, 1)`` (23 nodes, 46 DOFs) with the left edge clamped
  (40 free DOFs) and the downward tip load shared across the three
  right-edge nodes, so the patch bends symmetrically into plasticity.
  """
  mesh = authoring.quad8_patch(4, 1)
  model = authoring.small_strain_continuum(
    mesh, material=authoring.plasticity(_E, _NU, _SYIELD, _HARD, id="steel")
  )
  clamped = tuple(node.id for node in mesh.nodes if node.coordinates[0] == 0.0)
  loaded = [node.id for node in mesh.nodes if node.coordinates[0] == 1.0]
  system = compile_system(model, plasticity_reference_registry())
  coordinate_map = compile_constraint_map(
    system,
    constraints=tuple(authoring.fixed(nodes=clamped, components=("x", "y"))),
    coordinates=(ProgramCoordinateSpec(name="load", kind="load"),),
  )
  loads = tuple(
    NodalLoadSpec(
      id=f"tip-{node}",
      target=DofRef(node_id=node, field_id="displacement", component="y"),
      value=AffineValueSpec(
        coefficients=(AffineCoefficientSpec("load", -1.0 / len(loaded)),),
      ),
    )
    for node in loaded
  )
  return NonlinearStaticDriver(system, coordinate_map, loads, settings)


def test_load_controlled_bending_converges_quadratically() -> None:
  """One load step into plastic bending commits within the quadratic budget.

  The pre-fix tangent converged linearly on this identical case: 41
  iterations to 1e-12 with residual ratio ~0.52 (finding 20261007-agent-a3
  measured 33 on its 6/36-plastic configuration, and 28 to 1e-10, which
  tripped the default max_iterations=25 into a cutback cascade). With the
  exact tangent the case commits in 5 iterations. The bound below fails
  loudly on any future tangent regression instead of silently accepting
  linear convergence.
  """
  settings = NonlinearStaticSettings(
    tolerance=1.0e-12, max_iterations=60, max_cutbacks=8
  )
  driver = _cantilever_driver(settings)
  result = driver.run(
    base_point=ProgramPoint((ProgramCoordinateValue("load", 0.0),)),
    target_points=(ProgramPoint((ProgramCoordinateValue("load", 70.0),)),),
  )
  assert result.status is DriverStatus.COMPLETED
  assert result.statistics.committed_substep_count == 1
  assert result.statistics.rejected_substep_count == 0
  # The load level lands exactly four integration points in plasticity (the
  # transition to eight is ~3% of load away), so the plastic-branch tangent
  # drives the Newton iteration.
  layout = driver.owner.system.operators[0].header.state_layout
  rows = driver.owner.accepted_state(layout.block_id).values
  assert int(np.count_nonzero(rows[:, 18] > 0.0)) == 4
  # Quadratic convergence commits the single step in 5 iterations (pre-fix:
  # 41, linear at ratio ~0.52). 10 is 2x the measured count and 4x below
  # the linear regime; the lower bound proves genuine plastic nonlinearity
  # (an elastic step takes 2).
  (record,) = result.records
  assert 3 <= len(record.iterations) <= 10
  # The tangent channel is genuinely exercised: re-factorized, never reused.
  assert result.statistics.factorization_reuse_count == 0
  assert result.statistics.factorization_count == result.statistics.linear_solve_count
