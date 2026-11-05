"""v3-side family case drivers for the finite-strain TL and Riks deck stacks.

Both families drive the landed deck stack (``read_legacy_deck`` ->
``compile_deck`` -> driver) on generated decks, distinct from the prototype
solver the skims time. The Riks fan remains the historical truss-only,
no-point-spring discretization; its geometry and solver settings have one
source, ``bench.legacy_cases.write_legacy_truss_fan``.

Handles are prepared once per case (the ``load`` metric); each solve builds
a fresh driver on the prepared system, so warm repetitions rerun the full
solve from the virgin state honestly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bench.legacy_cases import fan_pro_path_for
from pyfem.v3.driver import (
  ArcLengthResult,
  ArcLengthSettings,
  NonlinearStaticDriver,
  NonlinearStaticSettings,
  RiksDriver,
)
from pyfem.v3.driver.plan import (
  assemble_internal_force,
  compile_driver_plan,
  refill_tangent,
)
from pyfem.v3.io.legacy_deck import (
  _LOAD_COORDINATE,
  CompiledDeck,
  ConvertedDeck,
  DeckRun,
  compile_deck,
  read_legacy_deck,
)
from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.operator import ChannelRequest, OperatorEvaluationInput
from pyfem.v3.model.system import CompiledSystem
from pyfem.v3.spec.model import ModelSpec
from pyfem.v3.spec.program import ProgramCoordinateValue, ProgramPoint

CantileverPrepared = tuple[ConvertedDeck, CompiledDeck]
RiksFanPrepared = tuple[ConvertedDeck, CompiledDeck]


# --- finite-strain TL family (cantilever8-derived) -------------------------


def cantilever_family_load(pro_path: Path) -> CantileverPrepared:
  """v3 load path: convert the generated deck and compile it (the load metric)."""
  deck = read_legacy_deck(pro_path)
  return deck, compile_deck(deck)


def cantilever_family_solve(prepared: CantileverPrepared) -> DeckRun:
  """Drive a prepared cantilever deck through the landed nonlinear driver.

  A fresh driver is built per call (``run_deck``'s driver + schedule half),
  so warm repetitions rerun the full ramp from the virgin state.
  """
  deck, compiled = prepared
  settings = NonlinearStaticSettings(
    tolerance=deck.solver.tolerance,
    max_iterations=deck.solver.max_iterations,
  )
  driver = NonlinearStaticDriver(
    compiled.system,
    compiled.constraint_map,
    compiled.loads,
    settings,
  )
  base_point = ProgramPoint(
    (ProgramCoordinateValue(_LOAD_COORDINATE, 0.0),),
  )
  target_points = tuple(
    ProgramPoint((ProgramCoordinateValue(_LOAD_COORDINATE, factor),))
    for factor in deck.solver.load_factors
  )
  result = driver.run(base_point=base_point, target_points=target_points)
  return DeckRun(driver=driver, result=result)


def compiled_to_legacy_permutation(
  model: ModelSpec, system: CompiledSystem
) -> np.ndarray:
  """Row index mapping: compiled coefficient row -> legacy state row.

  The landed F4 oracle's helper: deck node order is the generated ``.dat``
  row order, which is the legacy DOF order on the identical files.
  """
  position = {node.id: index for index, node in enumerate(model.mesh.nodes)}
  components = model.fields[0].components
  component_index = {component: index for index, component in enumerate(components)}
  ndof = len(components)
  space = system.spaces[0]
  return np.array(
    [
      position[node_id] * ndof + component_index[component]
      for _field_id, node_id, component in space.coefficient_ids
    ],
    dtype=np.int64,
  )


def cantilever_family_tangent(
  prepared: CantileverPrepared, state: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
  """Assembled tangent and internal force at ``state`` (compiled ordering).

  The operator-evaluation half of the landed F4 oracle: one evaluation of
  the compiled finite-strain operator gathered at the converged state,
  scattered through the driver assembly plan.
  """
  _deck, compiled = prepared
  (operator,) = compiled.system.operators
  gather = operator.header.ports[0].coefficient_map.values
  evaluation = operator.evaluate(
    OperatorEvaluationInput(
      port_values=(FinalizedArray(state[gather], dtype=np.float64),),
      accepted_state=FinalizedArray(
        np.zeros(operator.header.state_layout.row_shape), dtype=np.float64
      ),
      signals=(),
      request=ChannelRequest(("internal-force",), ("material-tangent",)),
    )
  )
  plan = compile_driver_plan(compiled.system, compiled.constraint_map, compiled.loads)
  internal = assemble_internal_force(plan, (evaluation.residual_values[0].values,))
  stiffness = refill_tangent(plan, (evaluation.jacobian_values[0].values,)).toarray()
  return np.asarray(stiffness), np.asarray(internal)


# --- Riks arc-length family (shallow-truss-derived) ------------------------


def riks_fan_load(n_rays: int) -> RiksFanPrepared:
  """Read and compile the already-generated no-point-spring fan deck."""
  deck = read_legacy_deck(fan_pro_path_for(n_rays))
  return deck, compile_deck(deck)


def riks_fan_solve(prepared: RiksFanPrepared) -> ArcLengthResult:
  """Drive a prepared fan through the landed Riks driver (fresh driver per call)."""
  deck, compiled = prepared
  settings = deck.solver
  driver = RiksDriver(
    compiled.system,
    compiled.constraint_map,
    compiled.loads,
    ArcLengthSettings(
      tolerance=settings.tolerance,
      max_iterations=settings.max_iterations,
      optimal_iterations=settings.optimal_iterations,
      fixed_step=settings.fixed_step,
      max_lam=settings.max_lam,
      max_factor=settings.max_factor,
      cycle_cap=settings.cycle_cap,
    ),
  )
  return driver.run(base_point=ProgramPoint())
