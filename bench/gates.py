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
from bench.workloads import GATE_ATOL, GATE_RTOL, Q8Workload, SkimCase
from pyfem.v3 import load_problem, solve_linear, solve_nonlinear, solve_riks
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
