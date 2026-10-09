"""Field-level diffs for registry-descriptor and sensitivity mismatches.

The landed builders reject an incompatible descriptor with a bare
``incompatible-registry-descriptor`` code and no detail. The authoring layer
decodes both canonical manifests back into plain values and names every
mismatched field, so an author sees exactly which metadata entry disagrees
with the qualified convention. The diagnostic keeps the landed machine-readable
code and source context; only the message gains the field-level diff. The same
idiom serves run-time sensitivity requests: a name no operator differentiates
keeps the landed ``unknown-sensitivity-parameter`` code, and the diff names
each offending entry against what the compiled system declares for it —
a constant of the calibration or no parameter at all.
"""

from __future__ import annotations

import json
from typing import NoReturn

from pyfem.v3.compile.diagnostics import (
  ModelCompilationDiagnostic,
  ModelCompilationError,
)
from pyfem.v3.driver.diagnostics import DriverDiagnostic, DriverPreparationError
from pyfem.v3.model.provenance import CANONICAL_MANIFEST_FORMAT, CanonicalManifest
from pyfem.v3.model.registry import RegistryDescriptor, RegistryKey
from pyfem.v3.spec.diagnostics import SourceContext, render_diagnostic_value

_MANIFEST_HEADER_LENGTH = len(CANONICAL_MANIFEST_FORMAT) + 1
_MAX_RENDERED_VALUE_LENGTH = 80


class _Opaque(str):
  """Rendered placeholder for manifest values that are not plain scalars."""

  def __repr__(self) -> str:
    return str(self)


def _decode_node(node: object) -> object:
  if type(node) is not list or not node or type(node[0]) is not str:
    msg = "canonical manifest payload is not a typed node"
    raise ValueError(msg)
  tag = node[0]
  if tag == "none":
    return None
  if tag == "bool":
    return bool(node[1])
  if tag == "int":
    try:
      return int(node[1])
    except ValueError:
      return _Opaque(f"<integer with {len(node[1])} digits>")
  if tag == "float":
    return float.fromhex(node[1])
  if tag == "str":
    return str(node[1])
  if tag == "sequence":
    return [_decode_node(item) for item in node[1]]
  if tag == "mapping":
    return {str(key): _decode_node(item) for key, item in node[1]}
  if tag == "ndarray":
    return _Opaque(f"<{node[1]} array of shape {tuple(node[2])}>")
  if tag == "captured-manifest":
    return _Opaque("<nested canonical manifest>")
  if tag == "unordered-declarations":
    return _Opaque("<unordered declarations>")
  msg = f"unsupported canonical manifest node tag {tag!r}"
  raise ValueError(msg)


def _decode_payload(manifest: CanonicalManifest) -> object:
  payload = manifest.to_bytes()[_MANIFEST_HEADER_LENGTH:]
  return _decode_node(json.loads(payload))


def _render(value: object) -> str:
  rendered = repr(value)
  if len(rendered) > _MAX_RENDERED_VALUE_LENGTH:
    rendered = f"{rendered[: _MAX_RENDERED_VALUE_LENGTH - 3]}..."
  return rendered


def _walk(expected: object, authored: object, path: str, lines: list[str]) -> None:
  if type(expected) is not type(authored):
    lines.append(
      f"field {path!r}: expected {_render(expected)}, authored {_render(authored)}"
    )
    return
  if type(expected) is dict:
    for key, item in expected.items():
      child = f"{path}.{key}" if path else str(key)
      if key not in authored:
        lines.append(f"missing field {child!r}")
      else:
        _walk(item, authored[key], child, lines)
    for key in authored:
      if key not in expected:
        child = f"{path}.{key}" if path else str(key)
        lines.append(f"unexpected field {child!r}")
    return
  if type(expected) is list:
    if len(expected) != len(authored):
      lines.append(
        f"field {path!r}: expected {len(expected)} entries, authored {len(authored)}"
      )
    for index, (wanted, given) in enumerate(zip(expected, authored, strict=False)):
      _walk(wanted, given, f"{path}[{index}]", lines)
    return
  if expected != authored:
    lines.append(
      f"field {path!r}: expected {_render(expected)}, authored {_render(authored)}"
    )


def diff_manifest_fields(
  expected: CanonicalManifest,
  authored: CanonicalManifest,
) -> tuple[str, ...]:
  """Return one line per metadata field where ``authored`` disagrees."""
  try:
    expected_value = _decode_payload(expected)
    authored_value = _decode_payload(authored)
  except (TypeError, ValueError, json.JSONDecodeError):
    return ("metadata payloads differ but cannot be decoded for a field diff",)
  lines: list[str] = []
  _walk(expected_value, authored_value, "", lines)
  if not lines:
    lines.append("metadata payloads differ")
  return tuple(lines)


def raise_descriptor_mismatch(
  *,
  family: str,
  key: RegistryKey,
  expected_metadata: dict[str, object],
  descriptor: RegistryDescriptor,
  source: SourceContext,
) -> NoReturn:
  """Raise the landed mismatch code with a field-level diff in the message."""
  diff = diff_manifest_fields(
    CanonicalManifest(expected_metadata),
    descriptor.metadata,
  )
  detail = "\n".join(f"  {line}" for line in diff)
  raise ModelCompilationError(
    (
      ModelCompilationDiagnostic(
        code="incompatible-registry-descriptor",
        message=(
          f"{family} descriptor metadata for registry key {key!r} does not "
          f"match the qualified convention:\n{detail}"
        ),
        source=source,
      ),
    )
  )


def raise_sensitivity_mismatch(
  *,
  offenders: tuple[str, ...],
  differentiable: tuple[str, ...],
  constants: tuple[str, ...],
  source: SourceContext,
) -> NoReturn:
  """Raise the landed unknown-parameter code with a field-level diff.

  The landed driver rejects an unservable sensitivity request with the coded
  ``unknown-sensitivity-parameter`` diagnostic and the declared channel list.
  The authoring layer names every offending request entry field by field:
  what the system would have to declare for the name to resolve (a
  parameterized derivative declaration) against what it actually declares —
  ``'constant'`` for a qualified-convention parameter baked into the compiled
  calibration without a derivative channel, ``'unknown'`` for a name no
  operator declares at all. The declared surface trails the diff.
  """
  expected = CanonicalManifest(
    {"sensitivities": {name: {"parameter": name} for name in offenders}}
  )
  authored = CanonicalManifest(
    {
      "sensitivities": {
        name: ("constant" if name in constants else "unknown") for name in offenders
      }
    }
  )
  diff = diff_manifest_fields(expected, authored)
  detail = "\n".join(f"  {line}" for line in diff)
  raise DriverPreparationError(
    (
      DriverDiagnostic(
        code="unknown-sensitivity-parameter",
        message=(
          "sensitivity parameters must name the compiled system's declared "
          f"derivative channels; the request does not match the declared "
          f"parameter surface:\n{detail}\n"
          "declared differentiable parameters: "
          f"{render_diagnostic_value(differentiable)}; declared constant "
          f"parameters: {render_diagnostic_value(constants)}"
        ),
        source=source,
      ),
    )
  )
