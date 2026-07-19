"""Structured verification-boundary diagnostics."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SolutionVerificationDiagnostic:
  """One malformed-record or verification-boundary finding."""

  code: str
  message: str

  def render(self) -> str:
    return f"[{self.code}] {self.message}"


class SolutionVerificationError(ValueError):
  """Raised when a solution or retained carrier is structurally malformed."""

  diagnostics: tuple[SolutionVerificationDiagnostic, ...]

  def __init__(
    self,
    diagnostics: Iterable[SolutionVerificationDiagnostic],
  ) -> None:
    self.diagnostics = tuple(diagnostics)
    if not self.diagnostics:
      msg = "SolutionVerificationError requires at least one diagnostic"
      raise ValueError(msg)
    super().__init__("\n".join(item.render() for item in self.diagnostics))
