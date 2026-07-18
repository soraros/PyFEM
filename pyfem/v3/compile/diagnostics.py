"""Source-aware failures from normalized-model compilation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from pyfem.v3.spec.diagnostics import SourceContext


@dataclass(frozen=True, slots=True)
class ModelCompilationDiagnostic:
  """One deterministic compiler compatibility or geometry finding."""

  code: str
  message: str
  source: SourceContext

  def render(self) -> str:
    """Render the machine-readable code with its normalized source context."""
    return f"{self.source.render()}: [{self.code}] {self.message}"


class ModelCompilationError(ValueError):
  """Raised for every covered compiler compatibility or geometry failure."""

  diagnostics: tuple[ModelCompilationDiagnostic, ...]

  def __init__(self, diagnostics: Iterable[ModelCompilationDiagnostic]) -> None:
    self.diagnostics = tuple(diagnostics)
    if not self.diagnostics:
      msg = "ModelCompilationError requires at least one diagnostic"
      raise ValueError(msg)
    super().__init__("\n".join(item.render() for item in self.diagnostics))
