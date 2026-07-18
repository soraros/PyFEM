"""Source-aware diagnostics for authored program normalization."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass

from pyfem.v3.spec.diagnostics import SourceContext, _render_diagnostic_value

PROGRAM_DIAGNOSTIC_MAX_LENGTH = 8191
_PROGRAM_COMPONENT_MAX_LENGTH = 1800
_TRUNCATED_TEXT = "...<truncated>"
_OMITTED_DIAGNOSTICS = "...<additional diagnostics omitted>"


def _escaped_exact_string(value: str) -> str:
  content_limit = _PROGRAM_COMPONENT_MAX_LENGTH - len(_TRUNCATED_TEXT) - len('""')
  pieces: list[str] = []
  length = 0
  truncated = False
  for character in value:
    escaped = json.dumps(character, ensure_ascii=True)[1:-1]
    if length + len(escaped) > content_limit:
      truncated = True
      break
    pieces.append(escaped)
    length += len(escaped)
  suffix = _TRUNCATED_TEXT if truncated else ""
  return f'"{"".join(pieces)}{suffix}"'


def render_program_value(value: object) -> str:
  """Render one trusted exact diagnostic value without unbounded text."""
  if type(value) is str:
    return _escaped_exact_string(value)
  if type(value) is int or value is None:
    return _render_diagnostic_value(value)
  return "<unrenderable value>"


def render_program_source(source: object) -> str:
  """Render an exact source location with escaped, bounded source text."""
  if type(source) is not SourceContext:
    return "<invalid-program-source>"
  try:
    source_name = object.__getattribute__(source, "source")
    line = object.__getattribute__(source, "line")
    column = object.__getattribute__(source, "column")
  except AttributeError:
    return "<invalid-program-source>"
  if (
    type(source_name) is not str
    or (line is not None and type(line) is not int)
    or (column is not None and type(column) is not int)
  ):
    return "<invalid-program-source>"
  rendered = _escaped_exact_string(source_name)
  if line is not None:
    rendered = f"{rendered}:{_render_diagnostic_value(line)}"
    if column is not None:
      rendered = f"{rendered}:{_render_diagnostic_value(column)}"
  return rendered


def render_program_diagnostic(
  *,
  code: object,
  message: object,
  source: object,
) -> str:
  """Render one escaped program diagnostic below the frozen ceiling."""
  safe_code = code if type(code) is str else "invalid-program-diagnostic-code"
  safe_message = (
    message if type(message) is str else "invalid program diagnostic message"
  )
  rendered = f"{render_program_source(source)}: [{safe_code}] {safe_message}"
  if len(rendered) <= PROGRAM_DIAGNOSTIC_MAX_LENGTH:
    return rendered
  keep = PROGRAM_DIAGNOSTIC_MAX_LENGTH - len(_TRUNCATED_TEXT)
  return rendered[:keep] + _TRUNCATED_TEXT


def render_program_diagnostics(diagnostics: Iterable[object]) -> str:
  """Join structured diagnostics without allowing aggregate text growth."""
  rendered: list[str] = []
  length = 0
  for diagnostic in diagnostics:
    try:
      item = diagnostic.render()
    except (AttributeError, TypeError, ValueError):
      item = "<invalid program diagnostic>"
    separator_length = 1 if rendered else 0
    if length + separator_length + len(item) > PROGRAM_DIAGNOSTIC_MAX_LENGTH:
      marker_length = len(_OMITTED_DIAGNOSTICS) + separator_length
      if length + marker_length <= PROGRAM_DIAGNOSTIC_MAX_LENGTH:
        rendered.append(_OMITTED_DIAGNOSTICS)
      break
    rendered.append(item)
    length += separator_length + len(item)
  return "\n".join(rendered)


@dataclass(frozen=True, slots=True)
class ProgramSpecDiagnostic:
  """One deterministic program-specification validation finding."""

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


class ProgramSpecValidationError(ValueError):
  """Raised with all authored-program diagnostics in validation order."""

  diagnostics: tuple[ProgramSpecDiagnostic, ...]

  def __init__(self, diagnostics: Iterable[ProgramSpecDiagnostic]) -> None:
    self.diagnostics = tuple(diagnostics)
    if not self.diagnostics:
      msg = "ProgramSpecValidationError requires at least one diagnostic"
      raise ValueError(msg)
    super().__init__(render_program_diagnostics(self.diagnostics))
