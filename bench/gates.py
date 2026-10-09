"""Correctness gates: no timing is recorded unless parity passes first.

A faster wrong result is a failed benchmark. The reference is the legacy
implementation (mission tolerance rtol 1e-10 for generated patches and linear
skims; nonlinear finite-strain and Riks skims use their own ``parity.toml``,
the repo's established numerical oracle). Generated uniform patches are also
checked against the analytic patch displacement field, so a bug shared by
both implementations cannot slip through.

Ratio-gate adjudication (the regression gate itself lives in ``bench/db.py``;
the policy is codified in bench/README.md, "Adjudicating a failed ratio
gate"). One environmental failure class is documented and accepted:
interpreter-bound and fixed-overhead cells — the legacy per-element Python
loops, the pure-Python LinearSolver skim e2e cells, cold import/solve/wall,
family load/e2e — can trip the 1.25 default ratio en masse on a fleet-loaded
or thermally throttled machine with no code cause. The class is pinned by two
committed M51-era runs sharing one ``git_revision``: the candidate
``bench/results/run_20261007T163300Z.json`` (137 regressions) and the
base-revision control ``bench/results/run_control_oldcode.json`` (85
regressions, 39 shared cells as re-measured in the M51 r1 review). THE
accepted adjudication path is the M51 evidence protocol — a candidate run
plus a base-revision control run under matched conditions, both with
``working_tree.clean`` true in their manifests — never a threshold change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from bench.family_v3 import (
  cantilever_family_load,
  cantilever_family_solve,
  cantilever_family_tangent,
  compiled_to_legacy_permutation,
  riks_fan_load,
  riks_fan_solve,
)
from bench.legacy_cases import (
  legacy_linear_state,
  legacy_linear_state_fast,
  legacy_load,
  legacy_nonlinear_state,
  legacy_riks_cycles,
  legacy_riks_state,
  legacy_tangent_at_state,
  write_legacy_cantilever,
  write_legacy_q8_patch,
  write_legacy_truss_fan,
)
from bench.workloads import (
  GATE_ATOL,
  GATE_RTOL,
  FiniteStrainWorkload,
  MaterialKernelCase,
  Q8Workload,
  RiksFanWorkload,
  SkimCase,
)
from pyfem.v3 import load_problem, solve_linear, solve_nonlinear, solve_riks
from pyfem.v3.compile.contracts import StatefulContinuumKernelResult
from pyfem.v3.driver import DriverStatus, SubstepStatus
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


# --- family gates (M47): legacy parity per the landed family oracles --------


def _status_check(name: str, passed: bool, detail: str) -> CheckResult:
  """Exact (non-numeric) gate check: driver status, cycle-count equality."""
  return CheckResult(
    name=name,
    passed=passed,
    max_rel_diff=0.0 if passed else float("inf"),
    rtol=0.0,
    atol=0.0,
    detail=detail,
  )


def gate_finite_strain(case: FiniteStrainWorkload) -> GateResult:
  """Gate one refined cantilever through the F4 deck stack, per the landed oracle.

  The v3 side drives the landed finite-strain family (deck conversion +
  compilation + ``NonlinearStaticDriver``) on the generated refined
  cantilever8 deck; the reference is the legacy ``NonlinearSolver`` on the
  identical files. Checks are the landed F4 oracle set, all within
  ``skims/cantilever8/parity.toml``: the run completes, the final state
  matches legacy, and the assembled tangent and internal force at the
  converged state match the legacy tangent assembly.
  """
  pro_path = write_legacy_cantilever(case.nx, case.ny)
  legacy = legacy_nonlinear_state(pro_path)
  prepared = cantilever_family_load(pro_path)
  run = cantilever_family_solve(prepared)
  deck, compiled = prepared

  checks: list[CheckResult] = []
  completed = run.result.status is DriverStatus.COMPLETED
  checks.append(
    _status_check(
      "driver-status",
      completed,
      f"status={run.result.status.name}; "
      f"committed={run.result.statistics.committed_substep_count}",
    )
  )
  if not completed:
    return GateResult(workload=case.name, passed=False, checks=checks)

  state = run.state
  permutation = compiled_to_legacy_permutation(deck.model, compiled.system)
  checks.append(
    _compare("legacy-parity", state, legacy[permutation], case.rtol, case.atol)
  )

  stiffness, internal = cantilever_family_tangent(prepared, state)
  legacy_k, legacy_fint = legacy_tangent_at_state(pro_path, state[permutation])
  legacy_k_dense = legacy_k.toarray()
  checks.append(
    _compare(
      "legacy-internal-force",
      internal,
      legacy_fint[permutation],
      case.rtol,
      case.atol,
    )
  )
  checks.append(
    _compare(
      "legacy-tangent",
      stiffness,
      legacy_k_dense[np.ix_(permutation, permutation)],
      case.rtol,
      case.atol,
    )
  )
  return GateResult(
    workload=case.name,
    passed=all(check.passed for check in checks),
    checks=checks,
  )


def gate_riks_fan(case: RiksFanWorkload) -> GateResult:
  """Gate one truss-only fan through the M33 Riks driver, per the landed oracle.

  The v3 side runs the landed ``RiksDriver`` over the programmatic fan
  model; the reference is the legacy ``RiksSolver`` on the identical
  generated deck, captured cycle by cycle. Checks are the landed M33 oracle
  set: the run completes, the committed cycle count matches legacy exactly
  (fixedStep makes the trajectory deterministic), and per-cycle committed
  load parameters and states match within
  ``skims/shallow_truss_riks/parity.toml`` with exact per-cycle
  correction-count equality.
  """
  pro_path = write_legacy_truss_fan(case.n_rays)
  props, globdat = legacy_load(pro_path)
  legacy_cycles = legacy_riks_cycles(props, globdat, pro_path)
  result = riks_fan_solve(riks_fan_load(case.n_rays))
  committed = tuple(
    record for record in result.records if record.status is SubstepStatus.COMMITTED
  )

  checks: list[CheckResult] = []
  completed = result.status is DriverStatus.COMPLETED
  checks.append(
    _status_check(
      "driver-status",
      completed,
      f"status={result.status.name}; termination={result.termination_reason}",
    )
  )
  count_ok = len(committed) == len(legacy_cycles)
  checks.append(
    _status_check(
      "cycle-count",
      count_ok,
      f"v3={len(committed)} legacy={len(legacy_cycles)}",
    )
  )
  if not (completed and count_ok):
    return GateResult(workload=case.name, passed=False, checks=checks)

  v3_lam = np.array([record.lam for record in committed])
  legacy_lam = np.array([lam for lam, _iiter, _state in legacy_cycles])
  checks.append(_compare("per-cycle-lam", v3_lam, legacy_lam, case.rtol, case.atol))
  v3_states = np.array([record.committed_coefficients.values for record in committed])
  legacy_states = np.array([state for _lam, _iiter, state in legacy_cycles])
  checks.append(
    _compare("per-cycle-state", v3_states, legacy_states, case.rtol, case.atol)
  )
  v3_iters = np.array(
    [len(record.iterations) - 1 for record in committed], dtype=np.float64
  )
  legacy_iters = np.array(
    [iiter for _lam, iiter, _state in legacy_cycles], dtype=np.float64
  )
  checks.append(
    _compare("per-cycle-correction-count", v3_iters, legacy_iters, 0.0, 0.0)
  )
  return GateResult(
    workload=case.name,
    passed=all(check.passed for check in checks),
    checks=checks,
  )
