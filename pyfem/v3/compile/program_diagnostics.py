"""Source-aware failures from program compilation and evaluation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from pyfem.v3.spec.diagnostics import SourceContext
from pyfem.v3.spec.program_diagnostics import (
  render_program_diagnostic,
  render_program_diagnostics,
)


@dataclass(frozen=True, slots=True)
class ProgramCompilationDiagnostic:
  """One deterministic program/model compatibility or affine-plan finding."""

  code: str
  message: str
  source: SourceContext

  def render(self) -> str:
    """Render the code with its normalized source location."""
    return render_program_diagnostic(
      code=self.code,
      message=self.message,
      source=self.source,
    )


@dataclass(frozen=True, slots=True)
class ProgramEvaluationDiagnostic:
  """One deterministic structured-point or arithmetic finding."""

  code: str
  message: str
  source: SourceContext

  def render(self) -> str:
    """Render the code with its normalized source location."""
    return render_program_diagnostic(
      code=self.code,
      message=self.message,
      source=self.source,
    )


class ProgramCompilationError(ValueError):
  """Raised for every covered program compilation failure."""

  diagnostics: tuple[ProgramCompilationDiagnostic, ...]

  def __init__(self, diagnostics: Iterable[ProgramCompilationDiagnostic]) -> None:
    self.diagnostics = tuple(diagnostics)
    if not self.diagnostics:
      msg = "ProgramCompilationError requires at least one diagnostic"
      raise ValueError(msg)
    super().__init__(render_program_diagnostics(self.diagnostics))


class ProgramEvaluationError(ValueError):
  """Raised before a malformed or nonfinite program evaluation can escape."""

  diagnostics: tuple[ProgramEvaluationDiagnostic, ...]

  def __init__(self, diagnostics: Iterable[ProgramEvaluationDiagnostic]) -> None:
    self.diagnostics = tuple(diagnostics)
    if not self.diagnostics:
      msg = "ProgramEvaluationError requires at least one diagnostic"
      raise ValueError(msg)
    super().__init__(render_program_diagnostics(self.diagnostics))
