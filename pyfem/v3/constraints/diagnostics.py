"""Source-aware failures from constraint-map compilation and evaluation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from pyfem.v3.spec.diagnostics import SourceContext


@dataclass(frozen=True, slots=True)
class ConstraintMapDiagnostic:
  """One deterministic constraint-map compilation or evaluation finding."""

  code: str
  message: str
  source: SourceContext

  def render(self) -> str:
    """Render the machine-readable code with its normalized source context."""
    return f"{self.source.render()}: [{self.code}] {self.message}"


class ConstraintMapError(ValueError):
  """Raised for every covered constraint-map compilation failure."""

  diagnostics: tuple[ConstraintMapDiagnostic, ...]

  def __init__(self, diagnostics: Iterable[ConstraintMapDiagnostic]) -> None:
    self.diagnostics = tuple(diagnostics)
    if not self.diagnostics:
      msg = "ConstraintMapError requires at least one diagnostic"
      raise ValueError(msg)
    super().__init__("\n".join(item.render() for item in self.diagnostics))


class ConstraintEvaluationError(ValueError):
  """Raised before a malformed signal binding or map evaluation can escape."""

  diagnostics: tuple[ConstraintMapDiagnostic, ...]

  def __init__(self, diagnostics: Iterable[ConstraintMapDiagnostic]) -> None:
    self.diagnostics = tuple(diagnostics)
    if not self.diagnostics:
      msg = "ConstraintEvaluationError requires at least one diagnostic"
      raise ValueError(msg)
    super().__init__("\n".join(item.render() for item in self.diagnostics))
