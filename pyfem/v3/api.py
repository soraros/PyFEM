"""Approved Phase 1 reusable and one-shot PyFEM v3 public flows."""

from __future__ import annotations

from pyfem.v3.analysis import (
  AnalysisDiagnostic,
  AnalysisError,
  AnalysisPreparationError,
  AnalysisSolveError,
  LinearStatic,
  PreparedAnalysis,
  StateTransactionError,
  prepare_analysis,
)
from pyfem.v3.compile import compile_model, compile_program
from pyfem.v3.model import RegistryDescriptor, RegistryKey
from pyfem.v3.results.contracts import VerificationReport
from pyfem.v3.results.diagnostics import SolutionVerificationError
from pyfem.v3.results.solution import Solution
from pyfem.v3.spec import ModelSpec, ProgramPoint, ProgramSpec


def solve(
  model_spec: ModelSpec,
  program_spec: ProgramSpec,
  request: LinearStatic,
  *,
  registry: dict[RegistryKey, RegistryDescriptor],
  initial_point: ProgramPoint,
  point: ProgramPoint,
) -> Solution:
  """Compile, prepare, accept once, and freshly verify one linear solution."""
  if type(request) is not LinearStatic:
    raise AnalysisPreparationError(
      (
        AnalysisDiagnostic(
          "invalid-analysis-request",
          "analysis request must be exactly LinearStatic",
        ),
      )
    )
  model = compile_model(model_spec, registry)
  program = compile_program(model, program_spec)
  analysis = prepare_analysis(model, program, request)
  initial = analysis.initialize(point=initial_point)
  solution = analysis.solve(initial=initial, point=point)
  report = solution.verify()
  if not report.passed:
    failed = ", ".join(item.name for item in report.checks if not item.passed)
    raise AnalysisSolveError(
      (
        AnalysisDiagnostic(
          "one-shot-verification-failed",
          f"fresh one-shot verification failed: {failed}",
        ),
      )
    )
  return solution


__all__ = [
  "AnalysisError",
  "AnalysisPreparationError",
  "AnalysisSolveError",
  "LinearStatic",
  "PreparedAnalysis",
  "Solution",
  "SolutionVerificationError",
  "StateTransactionError",
  "VerificationReport",
  "prepare_analysis",
  "solve",
]
