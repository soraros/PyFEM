"""Private stateless evaluation helpers shared by the static drivers."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.sparse import csr_matrix

from pyfem.v3.driver.nested import (
  PreparedNestedExecution,
  begin_nested_trial,
  commit_nested_trial,
  reject_nested_trial,
  validate_nested_trial,
)
from pyfem.v3.driver.plan import (
  DriverAssemblyPlan,
  OperatorAssemblySlice,
  assemble_residual_role,
  refill_tangent_channels,
)
from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.operator import (
  BalanceRole,
  ChannelRequest,
  EvaluationStatus,
  NestedEvaluationInput,
  OperatorEvaluation,
  OperatorEvaluationInput,
  ProgramSignalInput,
)
from pyfem.v3.model.system import CompiledSystem
from pyfem.v3.state import StateTransaction

type ResidualBatches = tuple[tuple[np.ndarray, ...], ...]
type JacobianBatches = tuple[tuple[np.ndarray, ...], ...]


def primal_channel_requests(system: CompiledSystem) -> tuple[ChannelRequest, ...]:
  """Request every primal channel in each header's declaration order."""
  return tuple(
    ChannelRequest(
      tuple(channel.channel_id for channel in operator.header.residual_channels),
      tuple(channel.channel_id for channel in operator.header.jacobian_channels),
    )
    for operator in system.operators
  )


def gather_port_values(
  plan: DriverAssemblyPlan,
  operator_index: int,
  full: FinalizedArray,
) -> tuple[FinalizedArray, ...]:
  """Gather every coefficient port in header order from one full trial vector."""
  slice_ = plan.operator_slices[operator_index]
  return tuple(
    FinalizedArray(
      np.array(full.values[gather.values], dtype=np.float64, order="C", copy=True),
      dtype=np.float64,
    )
    for gather in slice_.port_gathers
  )


def _validate_channel_values(
  evaluation: OperatorEvaluation,
  residual_request: tuple[str, ...],
  jacobian_request: tuple[str, ...],
  slice_: OperatorAssemblySlice,
) -> None:
  if type(evaluation) is not OperatorEvaluation:
    msg = "operator evaluation must return an exact OperatorEvaluation"
    raise TypeError(msg)
  if evaluation.status is not EvaluationStatus.OK:
    return
  if (
    type(evaluation.residual_values) is not tuple
    or len(evaluation.residual_values) != len(residual_request)
    or type(evaluation.jacobian_values) is not tuple
    or len(evaluation.jacobian_values) != len(jacobian_request)
  ):
    msg = "operator evaluation returned the wrong primal channel count"
    raise TypeError(msg)
  residual_channels = tuple(
    channel
    for channel in slice_.residual_channels
    if channel.channel_id in residual_request
  )
  jacobian_channels = tuple(
    channel
    for channel in slice_.jacobian_channels
    if channel.channel_id in jacobian_request
  )
  for value, channel in zip(
    evaluation.residual_values,
    residual_channels,
    strict=True,
  ):
    if (
      type(value) is not FinalizedArray
      or value.values.dtype != np.dtype(np.float64)
      or value.values.shape != (slice_.entity_count, channel.target_count)
    ):
      msg = "operator residual values do not match their compiled channel shape"
      raise TypeError(msg)
  for value, channel in zip(
    evaluation.jacobian_values,
    jacobian_channels,
    strict=True,
  ):
    if (
      type(value) is not FinalizedArray
      or value.values.dtype != np.dtype(np.float64)
      or value.values.shape
      != (slice_.entity_count, channel.target_count, channel.source_count)
    ):
      msg = "operator Jacobian values do not match their compiled channel shape"
      raise TypeError(msg)


def evaluate_operator_round(
  system: CompiledSystem,
  plan: DriverAssemblyPlan,
  requests: tuple[ChannelRequest, ...],
  full: FinalizedArray,
  transaction: StateTransaction,
  signal_inputs: tuple[tuple[ProgramSignalInput, ...], ...],
  nested_inputs: tuple[NestedEvaluationInput | None, ...],
) -> tuple[tuple[OperatorEvaluation, ...], tuple[EvaluationStatus, ...]]:
  """Evaluate each operator once, then validate its requested channel batches."""
  evaluations: list[OperatorEvaluation] = []
  for operator_index, (operator, request, signals, nested_input) in enumerate(
    zip(
      system.operators,
      requests,
      signal_inputs,
      nested_inputs,
      strict=True,
    )
  ):
    evaluation = operator.evaluate(
      OperatorEvaluationInput(
        port_values=gather_port_values(plan, operator_index, full),
        accepted_state=transaction.accepted_state(
          operator.header.state_layout.block_id
        ),
        signals=signals,
        request=request,
        nested_execution=nested_input,
      )
    )
    _validate_channel_values(
      evaluation,
      request.residual_channel_ids,
      request.jacobian_channel_ids,
      plan.operator_slices[operator_index],
    )
    evaluations.append(evaluation)
  return tuple(evaluations), tuple(evaluation.status for evaluation in evaluations)


def evaluation_channel_batches(
  evaluations: tuple[OperatorEvaluation, ...],
) -> tuple[ResidualBatches, JacobianBatches]:
  """Expose all primal channels in operator/header declaration order."""
  residual = tuple(
    tuple(value.values for value in evaluation.residual_values)
    for evaluation in evaluations
  )
  jacobian = tuple(
    tuple(value.values for value in evaluation.jacobian_values)
    for evaluation in evaluations
  )
  return residual, jacobian


def assemble_evaluation_balance(
  plan: DriverAssemblyPlan,
  evaluations: tuple[OperatorEvaluation, ...],
  external: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, JacobianBatches]:
  """Return signed full residual, Newton right-hand side, and tangent batches.

  The legacy branch deliberately retains the original subtraction order on both
  vectors. Other signatures evaluate ``I-X-L`` and ``L+X-I`` from per-role
  vectors assembled once each.
  """
  residual, jacobian = evaluation_channel_batches(evaluations)
  internal = assemble_residual_role(plan, residual, BalanceRole.INTERNAL)
  if plan.legacy_single_port:
    return internal - external, external - internal, jacobian
  operator_external = assemble_residual_role(plan, residual, BalanceRole.EXTERNAL)
  return (
    internal - operator_external - external,
    external + operator_external - internal,
    jacobian,
  )


def refill_evaluation_tangent(
  plan: DriverAssemblyPlan,
  jacobian: JacobianBatches,
) -> csr_matrix:
  """Refill the single prepared tangent from all Jacobian channels."""
  return refill_tangent_channels(plan, jacobian)


def restart_nested_trial(
  nested_executions: tuple[PreparedNestedExecution | None, ...],
) -> None:
  """Discard the current nested trial and reopen it from accepted state."""
  reject_nested_trial(nested_executions)
  begin_nested_trial(nested_executions)


def stage_and_commit_trial(
  system: CompiledSystem,
  transaction: StateTransaction,
  full: FinalizedArray,
  evaluations: tuple[OperatorEvaluation, ...],
  nested_executions: tuple[PreparedNestedExecution | None, ...],
  after_owner_commit: Callable[[], None],
) -> None:
  """Stage one complete primal trial, then commit owner before nested resources."""
  transaction.stage_physical(full)
  for operator, evaluation in zip(system.operators, evaluations, strict=True):
    transaction.stage_state(
      operator.header.state_layout.block_id,
      evaluation.trial_state,
    )
    for reference in evaluation.nested_references:
      transaction.stage_nested(reference)
  validate_nested_trial(nested_executions)
  transaction.commit()
  after_owner_commit()
  commit_nested_trial(nested_executions)
