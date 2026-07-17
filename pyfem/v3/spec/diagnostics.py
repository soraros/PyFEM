"""Source-aware diagnostics for model specification normalization."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass

_PLAIN_INTEGER_MAX_BITS = 256
_INTEGER_EDGE_BITS = 32
_INTEGER_EDGE_MASK = (1 << _INTEGER_EDGE_BITS) - 1


def _plain_integer_decimal(value: int) -> str:
  """Render a small exact integer without using Python's decimal conversion."""
  if value == 0:
    return "0"
  sign = "-" if value < 0 else ""
  remaining = -value if value < 0 else value
  digits: list[str] = []
  while remaining:
    remaining, digit = divmod(remaining, 10)
    digits.append(chr(ord("0") + digit))
  return sign + "".join(reversed(digits))


def _render_exact_integer(value: int) -> str:
  """Render an exact integer with output independent of decimal digit limits."""
  magnitude = -value if value < 0 else value
  bit_length = magnitude.bit_length()
  if bit_length <= _PLAIN_INTEGER_MAX_BITS:
    return _plain_integer_decimal(value)

  byte_length = (bit_length + 7) // 8
  digest = hashlib.sha256()
  digest.update(b"-" if value < 0 else b"+")
  digest.update(magnitude.to_bytes(byte_length, byteorder="big"))
  high = magnitude >> (bit_length - _INTEGER_EDGE_BITS)
  low = magnitude & _INTEGER_EDGE_MASK
  return (
    f"<int sign={'-' if value < 0 else '+'} "
    f"bits={_plain_integer_decimal(bit_length)} "
    f"high=0x{high:08x} low=0x{low:08x} sha256={digest.hexdigest()}>"
  )


def _render_diagnostic_value(value: object) -> str:
  """Render trusted exact diagnostic values without polymorphic conversion."""
  if type(value) is int:
    return _render_exact_integer(value)
  if type(value) is str:
    return repr(value)
  if type(value) is tuple:
    rendered = ", ".join(_render_diagnostic_value(item) for item in value)
    if len(value) == 1:
      rendered += ","
    return f"({rendered})"
  if value is None:
    return "None"
  return "<unrenderable value>"


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
      rendered = f"{rendered}:{_render_diagnostic_value(self.line)}"
      if self.column is not None:
        rendered = f"{rendered}:{_render_diagnostic_value(self.column)}"
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
