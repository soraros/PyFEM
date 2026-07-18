"""Stable source-aware diagnostics for assembly preparation and evaluation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from pyfem.v3.spec.diagnostics import SourceContext
from pyfem.v3.spec.program_diagnostics import (
  render_program_diagnostic,
  render_program_diagnostics,
)


@dataclass(frozen=True, slots=True)
class AssemblyPreparationDiagnostic:
  """One deterministic preparation or compatibility finding."""

  code: str
  message: str
  source: SourceContext

  def render(self) -> str:
    """Render one bounded, escaped diagnostic."""
    return render_program_diagnostic(
      code=self.code,
      message=self.message,
      source=self.source,
    )


@dataclass(frozen=True, slots=True)
class AssemblyEvaluationDiagnostic:
  """One deterministic reference-evaluation finding."""

  code: str
  message: str
  source: SourceContext

  def render(self) -> str:
    """Render one bounded, escaped diagnostic."""
    return render_program_diagnostic(
      code=self.code,
      message=self.message,
      source=self.source,
    )


class AssemblyPreparationError(ValueError):
  """Raised before an invalid prepared assembly plan can escape."""

  diagnostics: tuple[AssemblyPreparationDiagnostic, ...]

  def __init__(
    self,
    diagnostics: Iterable[AssemblyPreparationDiagnostic],
  ) -> None:
    self.diagnostics = tuple(diagnostics)
    if not self.diagnostics:
      msg = "AssemblyPreparationError requires at least one diagnostic"
      raise ValueError(msg)
    super().__init__(render_program_diagnostics(self.diagnostics))


class AssemblyEvaluationError(ValueError):
  """Raised before malformed or nonfinite assembly values can escape."""

  diagnostics: tuple[AssemblyEvaluationDiagnostic, ...]

  def __init__(
    self,
    diagnostics: Iterable[AssemblyEvaluationDiagnostic],
  ) -> None:
    self.diagnostics = tuple(diagnostics)
    if not self.diagnostics:
      msg = "AssemblyEvaluationError requires at least one diagnostic"
      raise ValueError(msg)
    super().__init__(render_program_diagnostics(self.diagnostics))
