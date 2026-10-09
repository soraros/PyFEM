"""Regression DB: flatten run records into cells and gate on ratios vs a baseline.

Gates compare ratios, not absolutes (D2 §7.10): a candidate cell regresses when
``candidate_median / baseline_median > threshold`` AND the absolute slowdown
exceeds ``NOISE_FLOOR_MS``. Baseline cells in a declared high-variance
class (e.g. legacy superlinear sizes, +/-50% per M3) may carry a per-cell
``gate_threshold`` that overrides the default for that cell only. The floor
(1 ms) is D2's measured warm fixed
overhead (~1.03 ms at the 2x2 toy size): below it, ratios are jitter, and a
gate that trips on 0.1 ms of timer noise trains people to ignore it. Cells
present only in the candidate are new (informational); cells present only in
the baseline are missing (a workload or metric silently disappeared — a
harness regression and a gate failure). Any candidate record whose
correctness gate failed is itself a hard failure, independent of timings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

DEFAULT_THRESHOLD = 1.25
IMPROVEMENT_RATIO = 0.8
NOISE_FLOOR_MS = 1.0

CellKey = tuple[str, str, str, str, int, str]


@dataclass
class CellRow:
  key: CellKey
  baseline_ms: float
  candidate_ms: float
  ratio: float
  status: str  # "ok" | "regression" | "improvement" | "missing" | "new"


@dataclass
class ComparisonReport:
  threshold: float
  rows: list[CellRow] = field(default_factory=list)
  correctness_failures: list[str] = field(default_factory=list)

  @property
  def regressions(self) -> list[CellRow]:
    return [r for r in self.rows if r.status == "regression"]

  @property
  def missing(self) -> list[CellRow]:
    return [r for r in self.rows if r.status == "missing"]

  @property
  def skipped(self) -> list[CellRow]:
    return [r for r in self.rows if r.status == "skipped"]

  @property
  def passed(self) -> bool:
    return not self.regressions and not self.missing and not self.correctness_failures

  def render(self) -> str:
    lines: list[str] = []
    ordered = sorted(
      self.rows,
      key=lambda r: (r.status != "regression", -(r.ratio if r.ratio else 0.0)),
    )
    lines.append(f"{'status':>11} {'ratio':>8} {'base_ms':>12} {'cand_ms':>12}  cell")
    for row in ordered:
      base = f"{row.baseline_ms:12.4g}" if row.baseline_ms else f"{'-':>12}"
      cand = f"{row.candidate_ms:12.4g}" if row.candidate_ms else f"{'-':>12}"
      ratio = f"{row.ratio:8.3f}" if row.ratio else f"{'-':>8}"
      cell = "/".join(str(part) for part in row.key)
      lines.append(f"{row.status:>11} {ratio} {base} {cand}  {cell}")
    lines.append(
      f"cells: {len(self.rows)} | regressions: {len(self.regressions)} "
      f"| missing: {len(self.missing)} | skipped (partial coverage): "
      f"{len(self.skipped)} "
      f"| correctness failures: {len(self.correctness_failures)} "
      f"| threshold: {self.threshold}"
    )
    lines.append("GATE: " + ("PASS" if self.passed else "FAIL"))
    return "\n".join(lines)


def flatten(run: dict[str, Any]) -> dict[CellKey, dict[str, Any]]:
  """Flatten a run/baseline JSON into metric cells keyed for comparison."""
  cells: dict[CellKey, dict[str, Any]] = {}
  for record in run.get("records", []):
    correctness = record.get("correctness", {})
    passed = bool(correctness.get("passed", True))
    for metric, stats in record.get("metrics", {}).items():
      key = (
        record["category"],
        record["workload"],
        record["side"],
        record["mode"],
        int(record["threads"]),
        metric,
      )
      cells[key] = {
        "min_ms": float(stats["min_ms"]),
        "median_ms": float(stats["median_ms"]),
        "reps": stats.get("reps", 0),
        "correctness_passed": passed,
        "gate_threshold": stats.get("gate_threshold"),
      }
  return cells


def correctness_failures(run: dict[str, Any]) -> list[str]:
  """Workloads in a candidate run whose correctness gate did not pass."""
  failures: list[str] = []
  seen: set[str] = set()
  for record in run.get("records", []):
    correctness = record.get("correctness", {})
    if not correctness.get("passed", True):
      workload = str(record.get("workload", "?"))
      if workload not in seen:
        seen.add(workload)
        failures.append(workload)
  return failures


def working_tree_state(run: dict[str, Any]) -> dict[str, Any]:
  """Working-tree provenance recorded in a run's manifest (M60).

  Returns ``{"clean": ..., "dirty_files": ...}`` as captured by
  ``bench.manifest``: ``clean`` True/False with ``dirty_files`` listing the
  porcelain entries when False. Backward-compatible read: run JSONs predating
  the field (and captures where git itself failed) report
  ``{"clean": None, "dirty_files": None}`` — provenance unknown, never an
  error. Adjudication should prefer ``clean`` True runs and treat None as
  "not recorded", not as clean.
  """
  manifest = run.get("manifest")
  if not isinstance(manifest, dict):
    return {"clean": None, "dirty_files": None}
  state = manifest.get("working_tree")
  if not isinstance(state, dict):
    return {"clean": None, "dirty_files": None}
  return {"clean": state.get("clean"), "dirty_files": state.get("dirty_files")}


def compare(
  baseline: dict[str, Any],
  candidate: dict[str, Any],
  *,
  threshold: float = DEFAULT_THRESHOLD,
) -> ComparisonReport:
  """Ratio-gate ``candidate`` against ``baseline`` (typically the M3 seed)."""
  base_cells = flatten(baseline)
  cand_cells = flatten(candidate)
  # Runs that did not produce the complete workload matrix (quick/reduced,
  # single-phase) cannot distinguish "workload disappeared" from "not run":
  # their unproduced baseline cells are informational skips. Full-coverage
  # runs keep missing-is-failure. Absent coverage metadata means full
  # (conservative for pre-coverage artifacts).
  coverage = candidate.get("coverage", "full")
  missing_status = "missing" if coverage == "full" else "skipped"
  report = ComparisonReport(
    threshold=threshold,
    correctness_failures=correctness_failures(candidate),
  )

  for key, base in sorted(base_cells.items()):
    cand = cand_cells.get(key)
    if cand is None:
      report.rows.append(CellRow(key, base["median_ms"], 0.0, 0.0, missing_status))
      continue
    ratio = cand["median_ms"] / base["median_ms"] if base["median_ms"] else 0.0
    delta_ms = cand["median_ms"] - base["median_ms"]
    cell_threshold = float(base.get("gate_threshold") or threshold)
    if ratio > cell_threshold and delta_ms > NOISE_FLOOR_MS:
      status = "regression"
    elif 0.0 < ratio < IMPROVEMENT_RATIO:
      status = "improvement"
    else:
      status = "ok"
    report.rows.append(
      CellRow(key, base["median_ms"], cand["median_ms"], ratio, status)
    )

  for key, cand in sorted(cand_cells.items()):
    if key not in base_cells:
      report.rows.append(CellRow(key, 0.0, cand["median_ms"], 0.0, "new"))

  return report
