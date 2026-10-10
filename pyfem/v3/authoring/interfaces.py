"""Cohesive interface authoring: plain node quads composed into systems.

The interface seam is the researcher-facing traction-separation extension
path: an interface network is a plain set of four-node element declarations
(the bottom pair defines the integration line, the top pair the displaced
side) plus one of the four landed stateless laws. These helpers validate the
authored slice, build the landed law declaration, compile it against an
existing compiled system, and compose the extended system — one call instead
of three, mirroring :mod:`pyfem.v3.authoring.springs`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from pyfem.v3.compile.interface import (
  InterfaceDeclaration,
  compile_interface_operator,
  compose_interface_system,
  dummy_interface_declaration,
  power_law_mode_i_declaration,
  thouless_mode_i_declaration,
  xu_needleman_declaration,
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


def _interface_quads(
  elements: object,
) -> tuple[
  tuple[SemanticId, ...],
  tuple[tuple[SemanticId, SemanticId, SemanticId, SemanticId], ...],
]:
  if isinstance(elements, Mapping):
    pairs = list(elements.items())
  elif isinstance(elements, Sequence) and not isinstance(elements, str):
    pairs = [(f"interface-{index}", quad) for index, quad in enumerate(elements, 1)]
  else:
    msg = (
      "interface elements must be a sequence of four-node quads or a mapping "
      "of interface id to node quad"
    )
    raise TypeError(msg)
  if not pairs:
    msg = "an interface network requires at least one element"
    raise ValueError(msg)
  interface_ids = tuple(
    _semantic_id(interface_id, label="interface id") for interface_id, _ in pairs
  )
  quads: list[tuple[SemanticId, SemanticId, SemanticId, SemanticId]] = []
  for interface_id, quad in pairs:
    if not isinstance(quad, Sequence) or isinstance(quad, str) or len(quad) != 4:
      msg = (
        f"interface element {interface_id!r} must reference exactly four node "
        "ids in (bottom_1, bottom_2, top_1, top_2) order"
      )
      raise ValueError(msg)
    quads.append(
      (
        _semantic_id(quad[0], label=f"interface {interface_id!r} node id"),
        _semantic_id(quad[1], label=f"interface {interface_id!r} node id"),
        _semantic_id(quad[2], label=f"interface {interface_id!r} node id"),
        _semantic_id(quad[3], label=f"interface {interface_id!r} node id"),
      )
    )
  return interface_ids, tuple(quads)


def _space_id(system: CompiledSystem, space_id: object) -> SemanticId:
  if type(system) is not CompiledSystem:
    msg = "interface helpers require an exact CompiledSystem to compose onto"
    raise TypeError(msg)
  if space_id is not None:
    return _semantic_id(space_id, label="interface space id")
  if len(system.spaces) != 1:
    msg = (
      "interface networks on a multi-space system require an explicit "
      f"space_id, the compiled system has {len(system.spaces)} spaces"
    )
    raise ValueError(msg)
  return system.spaces[0].space_id


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


def _interface_system(
  system: CompiledSystem,
  declaration: InterfaceDeclaration,
) -> CompiledSystem:
  if type(system) is not CompiledSystem:
    msg = "interface helpers require an exact CompiledSystem to compose onto"
    raise TypeError(msg)
  block, operator = compile_interface_operator(system, declaration)
  return compose_interface_system(system, block, operator)


def xu_needleman(
  system: CompiledSystem,
  *,
  elements: Sequence[Sequence[SemanticId]] | Mapping[SemanticId, Sequence[SemanticId]],
  fracture_energy: float,
  ultimate_traction: float,
  block_id: SemanticId = "interfaces",
  space_id: SemanticId | None = None,
  source: str = "authoring.xu_needleman",
) -> CompiledSystem:
  """Compile and compose a Xu-Needleman (rank-2) cohesive network.

  ``elements`` maps each interface element onto its four support node ids in
  the legacy ``(bottom_1, bottom_2, top_1, top_2)`` order (a plain sequence
  names the elements ``"interface-1"`` onward). ``fracture_energy`` is the
  work of separation ``Gc`` and ``ultimate_traction`` the peak traction
  ``Tult``; both must be positive, checked here and by the landed declaration
  builder.
  """
  interface_ids, node_quads = _interface_quads(elements)
  declaration = xu_needleman_declaration(
    block_id=_semantic_id(block_id, label="interface block id"),
    space_id=_space_id(system, space_id),
    interface_ids=interface_ids,
    node_quads=node_quads,
    fracture_energy=_positive_float(
      fracture_energy, label="xu_needleman fracture_energy"
    ),
    ultimate_traction=_positive_float(
      ultimate_traction, label="xu_needleman ultimate_traction"
    ),
    source=_source(source),
  )
  return _interface_system(system, declaration)


def power_law_mode_i(
  system: CompiledSystem,
  *,
  elements: Sequence[Sequence[SemanticId]] | Mapping[SemanticId, Sequence[SemanticId]],
  fracture_energy: float,
  ultimate_traction: float,
  block_id: SemanticId = "interfaces",
  space_id: SemanticId | None = None,
  source: str = "authoring.power_law_mode_i",
) -> CompiledSystem:
  """Compile and compose a power-law mode-I cohesive network.

  Mirrors :func:`xu_needleman` for the landed power-law mode-I law:
  ``fracture_energy`` is ``Gc`` and ``ultimate_traction`` the peak traction
  ``Tult``; both must be positive.
  """
  interface_ids, node_quads = _interface_quads(elements)
  declaration = power_law_mode_i_declaration(
    block_id=_semantic_id(block_id, label="interface block id"),
    space_id=_space_id(system, space_id),
    interface_ids=interface_ids,
    node_quads=node_quads,
    fracture_energy=_positive_float(
      fracture_energy, label="power_law_mode_i fracture_energy"
    ),
    ultimate_traction=_positive_float(
      ultimate_traction, label="power_law_mode_i ultimate_traction"
    ),
    source=_source(source),
  )
  return _interface_system(system, declaration)


def thouless_mode_i(
  system: CompiledSystem,
  *,
  elements: Sequence[Sequence[SemanticId]] | Mapping[SemanticId, Sequence[SemanticId]],
  fracture_energy: float,
  ultimate_traction: float,
  d1d3: float,
  d2d3: float,
  block_id: SemanticId = "interfaces",
  space_id: SemanticId | None = None,
  source: str = "authoring.thouless_mode_i",
) -> CompiledSystem:
  """Compile and compose a Thouless mode-I cohesive network.

  ``d1d3``/``d2d3`` are the rise-to-plateau and plateau-to-softening jump
  ratios; the landed declaration builder requires ``0 < d1d3 < d2d3 < 1``.
  """
  interface_ids, node_quads = _interface_quads(elements)
  declaration = thouless_mode_i_declaration(
    block_id=_semantic_id(block_id, label="interface block id"),
    space_id=_space_id(system, space_id),
    interface_ids=interface_ids,
    node_quads=node_quads,
    fracture_energy=_positive_float(
      fracture_energy, label="thouless_mode_i fracture_energy"
    ),
    ultimate_traction=_positive_float(
      ultimate_traction, label="thouless_mode_i ultimate_traction"
    ),
    d1d3=_positive_float(d1d3, label="thouless_mode_i d1d3"),
    d2d3=_positive_float(d2d3, label="thouless_mode_i d2d3"),
    source=_source(source),
  )
  return _interface_system(system, declaration)


def dummy_interface(
  system: CompiledSystem,
  *,
  elements: Sequence[Sequence[SemanticId]] | Mapping[SemanticId, Sequence[SemanticId]],
  stiffness: float,
  block_id: SemanticId = "interfaces",
  space_id: SemanticId | None = None,
  source: str = "authoring.dummy_interface",
) -> CompiledSystem:
  """Compile and compose a linear dummy-interface network.

  ``stiffness`` is the isotropic penalty stiffness ``D`` of the landed linear
  law; it must be positive.
  """
  interface_ids, node_quads = _interface_quads(elements)
  declaration = dummy_interface_declaration(
    block_id=_semantic_id(block_id, label="interface block id"),
    space_id=_space_id(system, space_id),
    interface_ids=interface_ids,
    node_quads=node_quads,
    stiffness=_positive_float(stiffness, label="dummy_interface stiffness"),
    source=_source(source),
  )
  return _interface_system(system, declaration)
