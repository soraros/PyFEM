"""Point-spring authoring: plain kernels composed into compiled systems.

The spring seam is the researcher-facing stateful extension path: a kernel is
a plain batched function of displacements, accepted state rows, and
parameters, returning a :class:`SpringKernelResult`. These helpers turn such
a kernel plus a small declaration into a validated :class:`SpringDeclaration`,
compile it against an existing compiled system, and compose the extended
system — one call instead of three.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from pyfem.v3.compile.spring import (
  SpringDeclaration,
  SpringKernel,
  SpringStateSlot,
  compile_spring_operator,
  compose_system,
  damage_envelope_declaration,
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


def _spring_pairs(
  nodes: object,
) -> tuple[tuple[SemanticId, ...], tuple[SemanticId, ...]]:
  if isinstance(nodes, Mapping):
    pairs = list(nodes.items())
  elif isinstance(nodes, Sequence) and not isinstance(nodes, str):
    pairs = [(f"spring-{index}", node_id) for index, node_id in enumerate(nodes, 1)]
  else:
    msg = (
      "spring nodes must be a sequence of support node ids or a mapping of "
      "spring id to support node id"
    )
    raise TypeError(msg)
  if not pairs:
    msg = "a spring network requires at least one support node"
    raise ValueError(msg)
  spring_ids = tuple(
    _semantic_id(spring_id, label="spring id") for spring_id, _ in pairs
  )
  node_ids = tuple(
    _semantic_id(node_id, label="spring support node id") for _, node_id in pairs
  )
  return spring_ids, node_ids


def _space_id(system: CompiledSystem, space_id: object) -> SemanticId:
  if space_id is not None:
    return _semantic_id(space_id, label="spring space id")
  if len(system.spaces) != 1:
    msg = (
      "spring networks on a multi-space system require an explicit space_id, "
      f"the compiled system has {len(system.spaces)} spaces"
    )
    raise ValueError(msg)
  return system.spaces[0].space_id


def _state_slots(state: object) -> tuple[SpringStateSlot, ...]:
  if not isinstance(state, Sequence) or isinstance(state, str):
    msg = "spring state must be a sequence of (name, width) slot pairs"
    raise TypeError(msg)
  slots: list[SpringStateSlot] = []
  for entry in state:
    if not isinstance(entry, Sequence) or isinstance(entry, str) or len(entry) != 2:
      msg = "spring state slots must be (name, width) pairs"
      raise TypeError(msg)
    name, width = entry
    slots.append(SpringStateSlot(name, width))
  return tuple(slots)


def _parameters(parameters: object) -> tuple[float, ...]:
  if not isinstance(parameters, Sequence) or isinstance(parameters, str):
    msg = "spring parameters must be a sequence of numbers"
    raise TypeError(msg)
  values: list[float] = []
  for parameter in parameters:
    if type(parameter) not in (int, float):
      msg = "spring parameters must be exact numbers"
      raise TypeError(msg)
    try:
      converted = float(parameter)
    except OverflowError:
      msg = "spring parameters must fit finite float64"
      raise ValueError(msg) from None
    if not math.isfinite(converted):
      msg = "spring parameters must be finite"
      raise ValueError(msg)
    values.append(converted)
  return tuple(values)


def _label(value: object, *, label: str) -> str:
  if type(value) is not str or not value:
    msg = f"spring {label} must be a non-empty exact string"
    raise TypeError(msg)
  return value


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


def spring(
  system: CompiledSystem,
  *,
  nodes: Sequence[SemanticId] | Mapping[SemanticId, SemanticId],
  kernel: SpringKernel,
  state: Sequence[tuple[str, int]] = (),
  parameters: Sequence[float] = (),
  name: str,
  implementation_id: str,
  version: str = "1",
  state_schema: str | None = None,
  block_id: SemanticId = "springs",
  space_id: SemanticId | None = None,
  source: str = "authoring.spring",
) -> CompiledSystem:
  """Compile and compose a stateful point-spring network onto a system.

  ``nodes`` maps each spring onto its support node (a plain sequence of node
  ids names the springs ``"spring-1"`` onward). ``kernel`` is a plain batched
  function; ``state`` declares per-spring state slots as ``(name, width)``
  pairs; ``parameters`` are the float values handed to every kernel call. The
  kernel is probed once at this compile boundary, exactly as the landed
  spring seam requires.
  """
  if type(system) is not CompiledSystem:
    msg = "spring requires an exact CompiledSystem to compose onto"
    raise TypeError(msg)
  if not callable(kernel):
    msg = "spring kernel must be callable"
    raise TypeError(msg)
  spring_ids, node_ids = _spring_pairs(nodes)
  declaration = SpringDeclaration(
    block_id=_semantic_id(block_id, label="spring block id"),
    space_id=_space_id(system, space_id),
    spring_ids=spring_ids,
    node_ids=node_ids,
    state_schema=implementation_id if state_schema is None else state_schema,
    state_slots=_state_slots(state),
    kernel_name=_label(name, label="kernel name"),
    kernel_version=_label(version, label="kernel version"),
    implementation_id=_label(implementation_id, label="implementation id"),
    parameters=_parameters(parameters),
    kernel=kernel,
    source=_source(source),
  )
  block, operator = compile_spring_operator(system, declaration)
  return compose_system(system, block, operator)


def damage_envelope_spring(
  system: CompiledSystem,
  *,
  nodes: Sequence[SemanticId] | Mapping[SemanticId, SemanticId],
  stiffness: float,
  critical_extension: float,
  max_increment: float,
  block_id: SemanticId = "damage-springs",
  space_id: SemanticId | None = None,
  source: str = "authoring.damage_envelope_spring",
) -> CompiledSystem:
  """Compose the in-tree isotropic damage-envelope spring law onto a system.

  Wraps :func:`pyfem.v3.compile.spring.damage_envelope_declaration` and the
  compile/compose pair in one call.
  """
  if type(system) is not CompiledSystem:
    msg = "damage_envelope_spring requires an exact CompiledSystem"
    raise TypeError(msg)
  spring_ids, node_ids = _spring_pairs(nodes)
  declaration = damage_envelope_declaration(
    block_id=_semantic_id(block_id, label="spring block id"),
    space_id=_space_id(system, space_id),
    spring_ids=spring_ids,
    node_ids=node_ids,
    stiffness=_positive_float(stiffness, label="damage envelope stiffness"),
    critical_extension=_positive_float(
      critical_extension,
      label="damage envelope critical_extension",
    ),
    max_increment=_positive_float(
      max_increment,
      label="damage envelope max_increment",
    ),
    source=_source(source),
  )
  block, operator = compile_spring_operator(system, declaration)
  return compose_system(system, block, operator)
