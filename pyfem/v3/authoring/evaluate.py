"""Evaluation helpers: plain displacements in, compiled operator results out.

A trial state is authored as a mapping of node id to displacement components;
the helpers map it onto the compiled system's coefficient vector, gather the
operator's port batch, supply a zero accepted state of the compiled layout,
and request every channel. This covers the teaching path (stateless Q8 and
truss operators); stateful flows step through the transaction helpers in
:mod:`pyfem.v3.authoring.transactions` (``state_owner``/``nonlinear_static``),
passing the trial's accepted snapshot as ``accepted_state`` here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.operator import (
  ChannelRequest,
  CompiledOperator,
  OperatorEvaluation,
  OperatorEvaluationInput,
  SemanticId,
)
from pyfem.v3.model.system import CompiledSystem, DiscreteSpace


def _finite_scalar(value: object, *, label: str) -> float:
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


def _space(system: CompiledSystem, space_id: object) -> DiscreteSpace:
  if space_id is None:
    if len(system.spaces) != 1:
      msg = (
        "a multi-space system requires an explicit space_id, the compiled "
        f"system has {len(system.spaces)} spaces"
      )
      raise ValueError(msg)
    return system.spaces[0]
  for space in system.spaces:
    if space.space_id == space_id:
      return space
  msg = f"the compiled system has no space {space_id!r}"
  raise ValueError(msg)


def trial_vector(
  system: CompiledSystem,
  displacements: Mapping[SemanticId, Sequence[float]] | None = None,
  *,
  space_id: SemanticId | None = None,
) -> np.ndarray:
  """Build the global trial coefficient vector from plain nodal displacements.

  ``displacements`` maps a node id to one value per field component of the
  selected space (``(ux, uy)`` for the displacement field); every
  unmentioned coefficient stays zero.
  """
  if type(system) is not CompiledSystem:
    msg = "trial_vector requires an exact CompiledSystem"
    raise TypeError(msg)
  values = np.zeros(system.coefficient_count, dtype=np.float64)
  if displacements is None:
    return values
  if type(displacements) is not dict:
    msg = "displacements must map node ids to per-component sequences"
    raise TypeError(msg)
  space = _space(system, space_id)
  support = next(
    (
      block for block in system.point_blocks if block.block_id == space.support_block_id
    ),
    None,
  )
  if support is None:
    msg = "the selected space support block is absent from the compiled system"
    raise ValueError(msg)
  node_index = {node_id: index for index, node_id in enumerate(support.entity_ids)}
  for node_id, components in displacements.items():
    index = node_index.get(node_id)
    if index is None:
      msg = f"displacements reference unknown node {node_id!r}"
      raise ValueError(msg)
    if (
      not isinstance(components, Sequence)
      or isinstance(components, str)
      or len(components) != len(space.components)
    ):
      msg = (
        f"displacements for node {node_id!r} must supply exactly "
        f"{len(space.components)} components {space.components!r}"
      )
      raise ValueError(msg)
    values[space.coefficient_map.values[index]] = [
      _finite_scalar(component, label=f"displacement at node {node_id!r}")
      for component in components
    ]
  return values


def evaluate(
  system: CompiledSystem,
  displacements: Mapping[SemanticId, Sequence[float]] | np.ndarray | None = None,
  *,
  operator: int = 0,
  residual: bool = True,
  jacobian: bool = True,
  accepted_state: np.ndarray | None = None,
) -> OperatorEvaluation:
  """Evaluate one compiled operator on authored nodal displacements.

  ``displacements`` accepts the plain mapping form of :func:`trial_vector`
  or a ready global coefficient vector; ``operator`` selects the system
  operator by index. The request covers all of the operator's residual and
  jacobian channels unless disabled; the accepted state defaults to the zero
  state of the compiled layout.
  """
  if type(system) is not CompiledSystem:
    msg = "evaluate requires an exact CompiledSystem"
    raise TypeError(msg)
  if type(operator) is not int or not 0 <= operator < len(system.operators):
    msg = (
      f"operator must be an index into the {len(system.operators)} compiled operators"
    )
    raise ValueError(msg)
  selected: CompiledOperator = system.operators[operator]
  port = selected.header.ports[0]
  if type(displacements) is np.ndarray:
    trial = displacements
  else:
    trial = trial_vector(system, displacements, space_id=port.space_id)
  if type(trial) is not np.ndarray or trial.shape != (system.coefficient_count,):
    msg = (
      f"a trial vector must be a float64 array of shape ({system.coefficient_count},)"
    )
    raise ValueError(msg)
  layout = selected.header.state_layout
  if accepted_state is None:
    state = FinalizedArray(np.zeros(layout.row_shape), dtype=np.float64)
  else:
    state = FinalizedArray(accepted_state, dtype=np.float64)
  request = ChannelRequest(
    tuple(channel.channel_id for channel in selected.header.residual_channels)
    if residual
    else (),
    tuple(channel.channel_id for channel in selected.header.jacobian_channels)
    if jacobian
    else (),
  )
  return selected.evaluate(
    OperatorEvaluationInput(
      port_values=(
        FinalizedArray(trial[port.coefficient_map.values], dtype=np.float64),
      ),
      accepted_state=state,
      signals=(),
      request=request,
    )
  )
