"""Correctness gates: no timing is recorded unless parity passes first.

A faster wrong result is a failed benchmark. The reference is the legacy
implementation (mission tolerance rtol 1e-10 for generated patches and linear
skims; nonlinear finite-strain and Riks skims use their own ``parity.toml``,
the repo's established numerical oracle). Generated uniform patches are also
checked against the analytic patch displacement field, so a bug shared by
both implementations cannot slip through.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from bench.legacy_cases import (
  legacy_linear_state,
  legacy_linear_state_fast,
  legacy_load,
  legacy_nonlinear_state,
  legacy_riks_state,
  write_legacy_q8_patch,
)
from bench.workloads import (
  GATE_ATOL,
  GATE_RTOL,
  MaterialKernelCase,
  Q8Workload,
  SkimCase,
)
from pyfem.v3 import load_problem, solve_linear, solve_nonlinear, solve_riks
from pyfem.v3.compile.contracts import StatefulContinuumKernelResult
from pyfem.v3.materials.isotropic_hardening_plasticity import (
  isotropic_hardening_calibration,
  isotropic_hardening_plasticity_kernel,
  isotropic_hardening_plasticity_kernel_reference,
)
from pyfem.v3.mesh.refined_patch import (
  build_uniform_q8_loaded,
  patch_displacement,
)
from pyfem.v3.types import LoadedProblem

# At or below this size the gate uses the canonical legacy ``LinearSolver``
# (three element loops); above it the numerically identical single-assembly
# path keeps the gate affordable in the legacy superlinear regime.
CANONICAL_LEGACY_MAX_SIZE = 16


@dataclass
class CheckResult:
  name: str
  passed: bool
  max_rel_diff: float
  rtol: float
  atol: float
  detail: str = ""

  def to_json(self) -> dict[str, Any]:
    return {
      "name": self.name,
      "passed": self.passed,
      "max_rel_diff": self.max_rel_diff,
      "rtol": self.rtol,
      "atol": self.atol,
      "detail": self.detail,
    }


@dataclass
class GateResult:
  workload: str
  passed: bool
  checks: list[CheckResult] = field(default_factory=list)

  def to_json(self) -> dict[str, Any]:
    return {
      "workload": self.workload,
      "passed": self.passed,
      "checks": [c.to_json() for c in self.checks],
    }


def _compare(
  name: str, candidate: np.ndarray, reference: np.ndarray, rtol: float, atol: float
) -> CheckResult:
  diff = np.abs(candidate - reference)
  denom = np.maximum(np.abs(reference), 1.0e-30)
  max_rel = float((diff / denom).max()) if diff.size else 0.0
  max_abs = float(diff.max()) if diff.size else 0.0
  passed = bool(np.allclose(candidate, reference, rtol=rtol, atol=atol))
  return CheckResult(
    name=name,
    passed=passed,
    max_rel_diff=max_rel,
    rtol=rtol,
    atol=atol,
    detail=f"max_abs={max_abs:.3e}",
  )


def _analytic_patch_state(loaded: LoadedProblem) -> np.ndarray:
  coords = loaded.problem.coords
  state = np.zeros(loaded.problem.n_dofs, dtype=np.float64)
  for node in range(coords.shape[0]):
    u, v = patch_displacement(float(coords[node, 0]), float(coords[node, 1]))
    state[2 * node] = u
    state[2 * node + 1] = v
  return state


def gate_q8(
  workload: Q8Workload,
  *,
  reference: np.ndarray | None = None,
  reference_kind: str | None = None,
  loaded: LoadedProblem | None = None,
) -> GateResult:
  """
  Gate one uniform Q8 workload: v3 vs legacy, plus the analytic field.

  The caller may pass a precomputed legacy ``reference`` state (the warm phase
  reuses its timed legacy run at large sizes instead of assembling twice);
  otherwise the gate computes it: canonical ``LinearSolver`` up to
  ``CANONICAL_LEGACY_MAX_SIZE``, the self-tested single-assembly path above.
  """
  if reference is None:
    pro_path = write_legacy_q8_patch(
      workload.nx,
      workload.ny,
      material_type=workload.material_type,
    )
    if workload.nx <= CANONICAL_LEGACY_MAX_SIZE:
      reference = legacy_linear_state(pro_path)
      reference_kind = "legacy-LinearSolver"
    else:
      props, globdat = legacy_load(pro_path)
      reference = legacy_linear_state_fast(props, globdat)
      reference_kind = "legacy-single-assembly"
  assert reference_kind is not None

  if loaded is None:
    loaded = build_uniform_q8_loaded(
      workload.nx,
      workload.ny,
      material_type=workload.material_type,
    )
  v3 = solve_linear(loaded)

  parity = _compare(reference_kind, v3, reference, GATE_RTOL, GATE_ATOL)
  analytic = _compare(
    "analytic-patch-field",
    v3,
    _analytic_patch_state(loaded),
    1.0e-8,
    1.0e-12,
  )
  checks = [parity, analytic]
  return GateResult(
    workload=workload.name,
    passed=all(c.passed for c in checks),
    checks=checks,
  )


def gate_skim(case: SkimCase) -> GateResult:
  """Gate one skim case: v3 solver vs the legacy driver of the same kind."""
  pro_path = case.pro_path
  loaded = load_problem(pro_path)
  if case.kind == "linear":
    v3 = np.asarray(solve_linear(loaded))
    legacy = legacy_linear_state(pro_path)
  elif case.kind == "nonlinear":
    v3 = np.asarray(solve_nonlinear(loaded).state)
    legacy = legacy_nonlinear_state(pro_path)
  elif case.kind == "riks":
    v3 = np.asarray(solve_riks(loaded).state)
    legacy = legacy_riks_state(pro_path)
  else:
    msg = f"Unknown skim kind {case.kind!r}"
    raise ValueError(msg)

  check = _compare("legacy-parity", v3, legacy, case.rtol, case.atol)
  return GateResult(workload=case.workload, passed=check.passed, checks=[check])


def check_fast_reference_selftest(materials: tuple[str, ...]) -> list[CheckResult]:
  """
  Verify the single-assembly fast reference equals canonical ``LinearSolver``.

  Runs once at the smallest size per material; protects the large-size gate
  from a reference-path substitution error.
  """
  results: list[CheckResult] = []
  for material_type in materials:
    pro_path = write_legacy_q8_patch(2, 2, material_type=material_type)
    canonical = legacy_linear_state(pro_path)
    props, globdat = legacy_load(pro_path)
    fast = legacy_linear_state_fast(props, globdat)
    results.append(
      _compare(
        f"fast-reference-selftest/{material_type}", fast, canonical, 1.0e-12, 1.0e-15
      )
    )
  return results


# Documented parity configuration of the J2 law (the battery in
# test/v3/test_v3_stateful_plasticity.py uses the same values).
MATERIAL_CALIBRATION = (210000.0, 0.3, 250.0, 1000.0)
MATERIAL_SWEEP_ENTITIES = 4096
MATERIAL_GATE_SEED = 42


def material_kernel_batch(
  case: MaterialKernelCase,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """The documented benchmark batch: strains, virgin rows, calibration.

  Deterministic construction: eps_xx sweeps the documented parity ramp
  (0.0002 .. 0.004, straddling the plane-strain yield strain) and every third
  entity carries a gamma_xy = 0.003 shear, so elastic, plastic-normal, and
  plastic-mixed branches all appear in one batch.
  """
  calibration = isotropic_hardening_calibration(*MATERIAL_CALIBRATION)
  strains = np.zeros((case.n_entities, 6), dtype=np.float64)
  strains[:, 0] = np.linspace(0.0002, 0.004, case.n_entities)
  strains[::3, 5] = 0.003
  rows = np.zeros((case.n_entities, 19), dtype=np.float64)
  return strains, rows, calibration


def _bitwise_check(
  name: str,
  optimized: StatefulContinuumKernelResult,
  reference: StatefulContinuumKernelResult,
) -> CheckResult:
  """Bitwise (uint64-view) equality of one optimized-vs-reference evaluation."""
  if optimized.status is not reference.status:
    return CheckResult(
      name=name,
      passed=False,
      max_rel_diff=float("inf"),
      rtol=0.0,
      atol=0.0,
      detail=f"status {optimized.status} != {reference.status}",
    )
  worst_rel = 0.0
  detail = "bitwise equal"
  passed = True
  for label in ("stresses", "tangents", "trial_rows"):
    candidate = getattr(optimized, label)
    referent = getattr(reference, label)
    if np.array_equal(candidate.view(np.uint64), referent.view(np.uint64)):
      continue
    passed = False
    diff = np.abs(candidate - referent)
    denom = np.maximum(np.abs(referent), 1.0e-30)
    worst_rel = max(worst_rel, float((diff / denom).max()))
    detail = f"{label} not bitwise equal"
  return CheckResult(
    name=name, passed=passed, max_rel_diff=worst_rel, rtol=0.0, atol=0.0, detail=detail
  )


def gate_material(case: MaterialKernelCase) -> GateResult:
  """Gate the optimized J2 kernel against its M25 reference: bitwise parity.

  No timing is recorded unless the optimized kernel reproduces the reference
  bit for bit — statuses included — on the documented batch (virgin and
  stepped), a seeded random sweep (virgin and stepped), and two rejecting
  batches (beyond the hardening table, non-finite predictor).
  """
  optimized = isotropic_hardening_plasticity_kernel
  reference = isotropic_hardening_plasticity_kernel_reference
  strains, virgin, calibration = material_kernel_batch(case)

  checks: list[CheckResult] = []
  stepped = reference(strains, virgin, calibration)
  checks.append(
    _bitwise_check(
      "documented-virgin", optimized(strains, virgin, calibration), stepped
    )
  )
  checks.append(
    _bitwise_check(
      "documented-stepped",
      optimized(strains, stepped.trial_rows, calibration),
      reference(strains, stepped.trial_rows, calibration),
    )
  )

  rng = np.random.default_rng(MATERIAL_GATE_SEED)
  random_strains = rng.normal(size=(MATERIAL_SWEEP_ENTITIES, 6)) * 1.5e-3
  random_virgin = np.zeros((MATERIAL_SWEEP_ENTITIES, 19))
  random_stepped = reference(random_strains, random_virgin, calibration)
  checks.append(
    _bitwise_check(
      "seeded-virgin",
      optimized(random_strains, random_virgin, calibration),
      random_stepped,
    )
  )
  next_strains = rng.normal(size=(MATERIAL_SWEEP_ENTITIES, 6)) * 1.0e-3
  checks.append(
    _bitwise_check(
      "seeded-stepped",
      optimized(next_strains, random_stepped.trial_rows, calibration),
      reference(next_strains, random_stepped.trial_rows, calibration),
    )
  )

  extreme = np.zeros((8, 6))
  extreme[3, 0] = 2.0
  checks.append(
    _bitwise_check(
      "reject-beyond-table",
      optimized(extreme, np.zeros((8, 19)), calibration),
      reference(extreme, np.zeros((8, 19)), calibration),
    )
  )
  nonfinite = np.zeros((8, 6))
  nonfinite[5, 4] = np.inf
  checks.append(
    _bitwise_check(
      "reject-nonfinite",
      optimized(nonfinite, np.zeros((8, 19)), calibration),
      reference(nonfinite, np.zeros((8, 19)), calibration),
    )
  )
  return GateResult(
    workload=case.workload,
    passed=all(check.passed for check in checks),
    checks=checks,
  )
