"""Program authoring: plain fixities and nodal loads as landed spec values.

The driver seam consumes the landed program declarations
(``PrescribedDofSpec``/``NodalLoadSpec`` bound to named program coordinates);
these helpers build them from plain node ids, component names, and numbers with
deterministic source labels. Hand-written landed declarations pass through
everywhere the helpers' output is accepted, so research code can mix authored
fixities with arbitrary affine ties or prescribed offsets.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from pyfem.v3.model.operator import SemanticId
from pyfem.v3.spec.diagnostics import SourceContext
from pyfem.v3.spec.program import (
  AffineCoefficientSpec,
  AffineValueSpec,
  DofRef,
  NodalLoadSpec,
  PrescribedDofSpec,
)


def _source(label: str) -> SourceContext:
  return SourceContext(source=label)


def _semantic_id(value: object, *, label: str) -> SemanticId:
  if type(value) is str and value:
    return value
  if type(value) is int:
    return value
  msg = f"{label} must be a non-empty string or integer id"
  raise TypeError(msg)


def _component(value: object) -> str:
  if type(value) is str and value:
    return value
  msg = "program components must be non-empty exact strings"
  raise TypeError(msg)


def _finite_number(value: object, *, label: str) -> float:
  if type(value) not in (int, float):
    msg = f"{label} must be an exact number"
    raise TypeError(msg)
  try:
    converted = float(value)
  except OverflowError:
    msg = f"{label} must fit finite float64"
    raise ValueError(msg) from None
  if not math.isfinite(converted):
    msg = f"{label} must be finite"
    raise ValueError(msg)
  return converted


def fixed(
  nodes: Sequence[SemanticId],
  components: Sequence[str] = ("x", "y"),
  *,
  field_id: SemanticId = "displacement",
) -> tuple[PrescribedDofSpec, ...]:
  """Declare zero-prescribed DOFs for every node/component pair.

  ``components`` defaults to the in-plane displacement pair, matching the
  model builders' field convention. Each emitted constraint carries the
  deterministic id ``fixed-<node>-<component>`` and the matching
  ``authoring.fixed`` source label.
  """
  field = _semantic_id(field_id, label="fixed field id")
  if not isinstance(nodes, Sequence) or isinstance(nodes, str):
    msg = "fixed nodes must be a sequence of node ids"
    raise TypeError(msg)
  if not isinstance(components, Sequence) or isinstance(components, str):
    msg = "fixed components must be a sequence of component names"
    raise TypeError(msg)
  node_ids = tuple(_semantic_id(node, label="fixed node id") for node in nodes)
  names = tuple(_component(component) for component in components)
  if not node_ids:
    msg = "fixed requires at least one node"
    raise ValueError(msg)
  if not names:
    msg = "fixed requires at least one component"
    raise ValueError(msg)
  seen: set[tuple[SemanticId, str]] = set()
  declarations: list[PrescribedDofSpec] = []
  for node_id in node_ids:
    for component in names:
      pair = (node_id, component)
      if pair in seen:
        msg = f"fixed declares node {node_id!r} component {component!r} twice"
        raise ValueError(msg)
      seen.add(pair)
      label = f"authoring.fixed:{node_id}:{component}"
      declarations.append(
        PrescribedDofSpec(
          id=f"fixed-{node_id}-{component}",
          target=DofRef(
            node_id=node_id,
            field_id=field,
            component=component,
          ),
          value=AffineValueSpec(constant=0.0, source=_source(label)),
          source=_source(label),
        )
      )
  return tuple(declarations)


def nodal_load(
  node: SemanticId,
  component: str,
  scale: float,
  *,
  field_id: SemanticId = "displacement",
  coordinate: str = "load",
  id: SemanticId | None = None,
) -> NodalLoadSpec:
  """Declare one nodal load proportional to a named program coordinate.

  The applied value is ``scale * coordinate`` at any program point, so a
  ``{"load": 0.5}`` point applies half the scale; negative scales reverse the
  direction. The declaration id defaults to ``load-<node>-<component>``.
  """
  node_id = _semantic_id(node, label="nodal load node id")
  name = _component(component)
  field = _semantic_id(field_id, label="nodal load field id")
  factor = _finite_number(scale, label="nodal load scale")
  if type(coordinate) is not str or not coordinate:
    msg = "nodal load coordinate must be a non-empty exact string"
    raise TypeError(msg)
  load_id = (
    f"load-{node_id}-{name}" if id is None else _semantic_id(id, label="nodal load id")
  )
  label = f"authoring.nodal_load:{load_id}"
  return NodalLoadSpec(
    id=load_id,
    target=DofRef(node_id=node_id, field_id=field, component=name),
    value=AffineValueSpec(
      coefficients=(
        AffineCoefficientSpec(coordinate, factor, _source(f"{label}:{coordinate}")),
      ),
      source=_source(label),
    ),
    source=_source(label),
  )
