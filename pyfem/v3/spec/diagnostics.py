"""Source-aware diagnostics for model specification normalization."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SourceContext:
  """Location supplied by an authored adapter or Python caller."""

  source: str = "<python>"
  line: int | None = None
  column: int | None = None

  def render(self) -> str:
    """Render a stable human-readable source location."""
    rendered = self.source
    if self.line is not None:
      rendered = f"{rendered}:{self.line}"
      if self.column is not None:
        rendered = f"{rendered}:{self.column}"
    return rendered


@dataclass(frozen=True, slots=True)
class SpecDiagnostic:
  """One deterministic model-specification validation finding."""

  code: str
  message: str
  source: SourceContext

  def render(self) -> str:
    """Render the diagnostic without losing its machine-readable code."""
    return f"{self.source.render()}: [{self.code}] {self.message}"


class ModelSpecValidationError(ValueError):
  """Raised with all model-specification diagnostics in validation order."""

  diagnostics: tuple[SpecDiagnostic, ...]

  def __init__(self, diagnostics: Iterable[SpecDiagnostic]) -> None:
    self.diagnostics = tuple(diagnostics)
    if not self.diagnostics:
      msg = "ModelSpecValidationError requires at least one diagnostic"
      raise ValueError(msg)
    super().__init__("\n".join(item.render() for item in self.diagnostics))
