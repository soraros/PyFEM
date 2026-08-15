"""Generic compiled finite-element operator contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np

from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.provenance import CanonicalManifest

type SemanticId = str | int | tuple[str | int, ...]


class BalanceRole(Enum):
  """Ledger role of one attributed residual or derivative channel."""

  INTERNAL = "internal"
  EXTERNAL = "external"
  CONSTRAINT = "constraint"


class PortMode(Enum):
  """How an operator consumes one bound discrete space."""

  COEFFICIENTS = "coefficients"


class StateLifetime(Enum):
  """Lifetime shared by accepted and trial operator-owned state."""

  ACCEPTED_TRIAL = "accepted-trial"


class CouplingPolicy(Enum):
  """Structural lifetime of an operator's coefficient coupling."""

  FIXED = "fixed"


@dataclass(frozen=True, slots=True, eq=False)
class ImplementationIdentity:
  """Selected descriptor identity captured at compilation."""

  kind: str
  name: str
  version: str
  implementation_id: str


@dataclass(frozen=True, slots=True, eq=False)
class PortBinding:
  """One typed operator port and its native coefficient gather map."""

  port_id: str
  space_id: SemanticId
  mode: PortMode
  coefficient_map: FinalizedArray


@dataclass(frozen=True, slots=True, eq=False)
class ResidualChannel:
  """Attributed vector-valued contribution."""

  channel_id: str
  target_port_id: str
  balance_role: BalanceRole
  linear: bool


@dataclass(frozen=True, slots=True, eq=False)
class JacobianChannel:
  """Attributed derivative between two operator ports."""

  channel_id: str
  residual_channel_id: str
  target_port_id: str
  source_port_id: str
  balance_role: BalanceRole
  linear: bool
  symmetric: bool


@dataclass(frozen=True, slots=True, eq=False)
class OperatorStateSlot:
  """One named contiguous part of an operator-owned state row."""

  name: str
  width: int
  dtype: str
  lifetime: StateLifetime


@dataclass(frozen=True, slots=True, eq=False)
class OperatorStateLayout:
  """Allocation schema for accepted/trial block state, including width zero."""

  schema: str
  block_id: SemanticId
  entity_count: int
  slots: tuple[OperatorStateSlot, ...]
  entity_offsets: FinalizedArray
  row_width: int
  dtype: str
  lifetime: StateLifetime

  @property
  def row_shape(self) -> tuple[int, int]:
    """Return the exact accepted/trial row-array shape."""
    return self.entity_count, self.row_width


@dataclass(frozen=True, slots=True, eq=False)
class OperatorHeader:
  """Small non-enumerating header shared by every compiled operator."""

  block_id: SemanticId
  entity_block_id: SemanticId
  implementations: tuple[ImplementationIdentity, ...]
  ports: tuple[PortBinding, ...]
  residual_channels: tuple[ResidualChannel, ...]
  jacobian_channels: tuple[JacobianChannel, ...]
  state_layout: OperatorStateLayout
  coupling_policy: CouplingPolicy


@dataclass(frozen=True, slots=True, eq=False)
class OperatorEvaluation:
  """Detached local channel values and the matching trial-state rows."""

  residual_values: tuple[FinalizedArray, ...]
  jacobian_values: tuple[FinalizedArray, ...]
  trial_state: FinalizedArray


class CompiledOperator(Protocol):
  """Open structural protocol; concrete payloads remain builder-owned."""

  @property
  def header(self) -> OperatorHeader: ...

  @property
  def content_manifest(self) -> CanonicalManifest: ...

  def evaluate(
    self,
    port_values: tuple[np.ndarray, ...],
    accepted_state: np.ndarray,
  ) -> OperatorEvaluation: ...
