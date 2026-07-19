"""Structured diagnostics for preparation, solve, and state transactions."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AnalysisDiagnostic:
  """One bounded analysis diagnostic."""

  code: str
  message: str

  def render(self) -> str:
    return f"[{self.code}] {self.message}"


class AnalysisError(ValueError):
  """Base class for structured analysis failures."""

  diagnostics: tuple[AnalysisDiagnostic, ...]

  def __init__(self, diagnostics: Iterable[AnalysisDiagnostic]) -> None:
    self.diagnostics = tuple(diagnostics)
    if not self.diagnostics:
      msg = "AnalysisError requires at least one diagnostic"
      raise ValueError(msg)
    super().__init__("\n".join(item.render() for item in self.diagnostics))


class AnalysisPreparationError(AnalysisError):
  """Raised before an incompatible prepared analysis can escape."""


class AnalysisSolveError(AnalysisError):
  """Raised when a candidate cannot satisfy the frozen linear policy."""


class StateTransactionError(AnalysisError):
  """Raised when a state, transaction, trial, or acceptance is invalid."""
