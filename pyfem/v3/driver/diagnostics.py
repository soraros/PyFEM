"""Stable coded diagnostics for the purified nonlinear driver."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from pyfem.v3.spec.diagnostics import SourceContext
from pyfem.v3.spec.program_diagnostics import (
  render_program_diagnostic,
  render_program_diagnostics,
)


@dataclass(frozen=True, slots=True)
class DriverDiagnostic:
  """One deterministic driver preparation finding."""

  code: str
  message: str
  source: SourceContext = field(default_factory=SourceContext)

  def render(self) -> str:
    """Render one bounded, escaped diagnostic."""
    return render_program_diagnostic(
      code=self.code,
      message=self.message,
      source=self.source,
    )


class DriverPreparationError(ValueError):
  """Raised before an invalid driver or assembly plan can escape."""

  diagnostics: tuple[DriverDiagnostic, ...]

  def __init__(self, diagnostics: Iterable[DriverDiagnostic]) -> None:
    self.diagnostics = tuple(diagnostics)
    if not self.diagnostics:
      msg = "DriverPreparationError requires at least one diagnostic"
      raise ValueError(msg)
    super().__init__(render_program_diagnostics(self.diagnostics))


class DriverEvaluationError(ValueError):
  """Raised before malformed or nonfinite driver evaluation values escape."""

  diagnostics: tuple[DriverDiagnostic, ...]

  def __init__(self, diagnostics: Iterable[DriverDiagnostic]) -> None:
    self.diagnostics = tuple(diagnostics)
    if not self.diagnostics:
      msg = "DriverEvaluationError requires at least one diagnostic"
      raise ValueError(msg)
    super().__init__(render_program_diagnostics(self.diagnostics))
