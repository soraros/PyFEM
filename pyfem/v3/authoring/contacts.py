"""Penalty contact authoring: a disc obstacle over a declared surface set.

The contact seam composes the landed frictionless penalty law onto an
existing compiled system: the student names the surface node set and the
analytic disc (``centre``, ``direction``, ``radius``, ``penalty``), and the
obstacle centre rides the declared program coordinate — at coordinate value
``lam`` the centre sits at ``centre + lam * direction``. One call instead of
the declaration/compile/compose triple, mirroring
:mod:`pyfem.v3.authoring.springs`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from pyfem.v3.compile.contact import (
  ContactSignalPort,
  compile_contact_operator,
  compose_contact_system,
  penalty_disc_declaration,
)
from pyfem.v3.model.operator import SemanticId
from pyfem.v3.model.system import CompiledSystem
from pyfem.v3.spec.diagnostics import SourceContext


def _source(label: str) -> SourceContext:
  return SourceContext(source=label)


def _semantic_id(value: object, *, label: str) -> SemanticId:
  if type(value) is str and value:
    return value
  if type(value) is int:
    return value
  msg = f"{label} must be a non-empty string or integer id"
  raise TypeError(msg)


def _surface_nodes(
  nodes: object,
) -> tuple[tuple[SemanticId, ...], tuple[SemanticId, ...]]:
  if isinstance(nodes, Mapping):
    pairs = list(nodes.items())
  elif isinstance(nodes, Sequence) and not isinstance(nodes, str):
    pairs = [(f"contact-{index}", node_id) for index, node_id in enumerate(nodes, 1)]
  else:
    msg = (
      "contact nodes must be a sequence of surface node ids or a mapping of "
      "contact id to surface node id"
    )
    raise TypeError(msg)
  if not pairs:
    msg = "a contact network requires at least one surface node"
    raise ValueError(msg)
  contact_ids = tuple(
    _semantic_id(contact_id, label="contact id") for contact_id, _ in pairs
  )
  node_ids = tuple(
    _semantic_id(node_id, label="contact surface node id") for _, node_id in pairs
  )
  return contact_ids, node_ids


def _space_id(system: CompiledSystem, space_id: object) -> SemanticId:
  if space_id is not None:
    return _semantic_id(space_id, label="contact space id")
  if len(system.spaces) != 1:
    msg = (
      "contact networks on a multi-space system require an explicit space_id, "
      f"the compiled system has {len(system.spaces)} spaces"
    )
    raise ValueError(msg)
  return system.spaces[0].space_id


def _finite_pair(value: object, *, label: str) -> tuple[float, float]:
  if not isinstance(value, Sequence) or isinstance(value, str) or len(value) != 2:
    msg = f"{label} must be an (x, y) pair"
    raise TypeError(msg)
  components: list[float] = []
  for component in value:
    if type(component) not in (int, float):
      msg = f"{label} components must be exact numbers"
      raise TypeError(msg)
    try:
      converted = float(component)
    except OverflowError:
      msg = f"{label} components must fit finite float64"
      raise ValueError(msg) from None
    if not math.isfinite(converted):
      msg = f"{label} components must be finite"
      raise ValueError(msg)
    components.append(converted)
  return components[0], components[1]


def _positive_float(value: object, *, label: str) -> float:
  if type(value) not in (int, float):
    msg = f"{label} must be an exact number"
    raise TypeError(msg)
  try:
    converted = float(value)
  except OverflowError:
    msg = f"{label} must fit finite float64"
    raise ValueError(msg) from None
  if not math.isfinite(converted) or converted <= 0.0:
    msg = f"{label} must be a positive finite number"
    raise ValueError(msg)
  return converted


def penalty_contact(
  system: CompiledSystem,
  *,
  nodes: Sequence[SemanticId] | Mapping[SemanticId, SemanticId],
  centre: tuple[float, float],
  direction: tuple[float, float],
  radius: float,
  penalty: float,
  coordinate: str = "load",
  block_id: SemanticId = "contact",
  space_id: SemanticId | None = None,
  source: str = "authoring.penalty_contact",
) -> CompiledSystem:
  """Compile and compose the penalty disc contact law onto a system.

  ``nodes`` is the surface set: a plain sequence of node ids (contact
  entities named ``"contact-1"`` onward) or a mapping of contact id to node
  id. The disc obstacle sits at ``centre + lam * direction`` with ``lam``
  the value of the program coordinate named by ``coordinate`` — declare that
  coordinate on the stepping session and schedule it per step; there is no
  other obstacle channel. ``radius`` and ``penalty`` must be positive. The
  law is frictionless with the exact symmetric tangent; the compiler pins
  two-component spaces.
  """
  if type(system) is not CompiledSystem:
    msg = "penalty_contact requires an exact CompiledSystem to compose onto"
    raise TypeError(msg)
  if type(coordinate) is not str or not coordinate:
    msg = "penalty_contact coordinate must be a non-empty exact string"
    raise TypeError(msg)
  contact_ids, node_ids = _surface_nodes(nodes)
  declaration = penalty_disc_declaration(
    block_id=_semantic_id(block_id, label="contact block id"),
    space_id=_space_id(system, space_id),
    contact_ids=contact_ids,
    node_ids=node_ids,
    centre=_finite_pair(centre, label="penalty_contact centre"),
    direction=_finite_pair(direction, label="penalty_contact direction"),
    radius=_positive_float(radius, label="penalty_contact radius"),
    penalty=_positive_float(penalty, label="penalty_contact penalty"),
    signal_port=ContactSignalPort(port_id="load-factor", signal_id=coordinate),
    source=_source(source),
  )
  block, operator = compile_contact_operator(system, declaration)
  return compose_contact_system(system, block, operator)
