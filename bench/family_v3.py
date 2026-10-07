"""v3-side family case drivers: finite-strain TL deck stack and Riks fan.

The finite-strain side drives the landed deck stack (``read_legacy_deck`` ->
``compile_deck`` -> ``NonlinearStaticDriver``) on refined cantilever8 decks —
the M38 family path, distinct from the prototype solver the skim times. The
Riks side builds the landed M33 ``RiksDriver`` over a truss-only
shallow-truss fan ModelSpec: the deck converter rejects RiksSolver decks, so
the fan follows the landed driver parity oracle's programmatic construction
(``test/v3/test_v3_driver_parity.py``).

Handles are prepared once per case (the ``load`` metric); each solve builds
a fresh driver on the prepared system, so warm repetitions rerun the full
solve from the virgin state honestly. The fan geometry and Riks settings
mirror the generated legacy deck (``bench.legacy_cases.write_legacy_truss_fan``);
the parity gate protects the correspondence, like ``bench.legacy_settings``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pyfem.v3.compile.system import compile_system
from pyfem.v3.compile.truss import truss_reference_registry
from pyfem.v3.constraints import CompiledConstraintMap, compile_constraint_map
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
from pyfem.v3.spec.diagnostics import SourceContext
from pyfem.v3.spec.model import (
  CellBlockSpec,
  CellRef,
  CellSpec,
  FieldSpec,
  MaterialParameterSpec,
  MaterialSpec,
  MeshSpec,
  ModelSpec,
  NodeSpec,
  RegionSpec,
)
from pyfem.v3.spec.program import (
  AffineCoefficientSpec,
  AffineValueSpec,
  DofRef,
  NodalLoadSpec,
  PrescribedDofSpec,
  ProgramCoordinateSpec,
  ProgramCoordinateValue,
  ProgramPoint,
)

CantileverPrepared = tuple[ConvertedDeck, CompiledDeck]
RiksFanPrepared = tuple[
  CompiledSystem, CompiledConstraintMap, tuple[NodalLoadSpec, ...]
]

# Truss-only shallow-truss fan constants, mirrored by the generated legacy
# deck (write_legacy_truss_fan): the landed M33 oracle's values.
FAN_SPAN = 20.0
FAN_HEIGHT = 0.5
FAN_YOUNGS_MODULUS = 5.0e6
FAN_AREA = 1.0
FAN_APEX_LOAD = -100.0
FAN_TOLERANCE = 1.0e-10
FAN_MAX_ITERATIONS = 25
FAN_MAX_LAM = 10.0


def _source(label: str) -> SourceContext:
  return SourceContext(source=label)


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


def riks_fan_model(n_rays: int) -> ModelSpec:
  """Truss-only fan ModelSpec; ``n_rays=2`` is the ch.4 shallow-truss geometry."""
  apex = n_rays
  nodes = tuple(
    NodeSpec(
      id=i,
      coordinates=(-FAN_SPAN / 2.0 + i * FAN_SPAN / max(n_rays - 1, 1), 0.0),
      source=_source(f"n{i}"),
    )
    for i in range(n_rays)
  ) + (
    NodeSpec(
      id=apex,
      coordinates=(0.0, FAN_HEIGHT),
      source=_source(f"n{apex}"),
    ),
  )
  cells = tuple(
    CellSpec(id=f"ray-{i}", node_ids=(i, apex), source=_source(f"c{i}"))
    for i in range(n_rays)
  )
  block = CellBlockSpec(
    id="rays",
    reference_topology="line",
    topological_dimension=1,
    embedding_dimension=2,
    geometry_interpolation="line2",
    cells=cells,
    source=_source("block"),
  )
  field = FieldSpec(
    id="displacement",
    components=("x", "y"),
    location="node",
    source=_source("field"),
  )
  material = MaterialSpec(
    id="steel",
    model="uniaxial-linear-elastic",
    parameters=(
      MaterialParameterSpec("youngs_modulus", FAN_YOUNGS_MODULUS),
      MaterialParameterSpec("area", FAN_AREA),
    ),
    source=_source("material"),
  )
  region = RegionSpec(
    id="domain",
    cell_refs=tuple(CellRef("rays", f"ray-{i}") for i in range(n_rays)),
    field_ids=("displacement",),
    material_id="steel",
    formulation="total-lagrangian-truss",
    quadrature="none",
    source=_source("region"),
  )
  return ModelSpec(
    mesh=MeshSpec(nodes=nodes, cell_blocks=(block,), source=_source("mesh")),
    fields=(field,),
    materials=(material,),
    regions=(region,),
    source=_source("model"),
  )


def riks_fan_load(n_rays: int) -> RiksFanPrepared:
  """v3 load path: compile the fan system, constraint map, and load program."""
  system = compile_system(riks_fan_model(n_rays), truss_reference_registry())
  apex = n_rays
  coordinate_map = compile_constraint_map(
    system,
    constraints=tuple(
      PrescribedDofSpec(
        id=f"fix-{node}-{component}",
        target=DofRef(
          node_id=node,
          field_id="displacement",
          component=component,
        ),
        value=AffineValueSpec(constant=0.0),
        source=_source(f"fix-{node}-{component}"),
      )
      for node in range(n_rays)
      for component in ("x", "y")
    ),
    coordinates=(ProgramCoordinateSpec(name="load", kind="load"),),
  )
  loads = (
    NodalLoadSpec(
      id="apex",
      target=DofRef(node_id=apex, field_id="displacement", component="y"),
      value=AffineValueSpec(
        coefficients=(AffineCoefficientSpec("load", FAN_APEX_LOAD, _source("coef")),),
        source=_source("load"),
      ),
      source=_source("apex"),
    ),
  )
  return system, coordinate_map, loads


def riks_fan_solve(prepared: RiksFanPrepared) -> ArcLengthResult:
  """Drive a prepared fan through the landed Riks driver (fresh driver per call)."""
  system, coordinate_map, loads = prepared
  driver = RiksDriver(
    system,
    coordinate_map,
    loads,
    ArcLengthSettings(
      tolerance=FAN_TOLERANCE,
      max_iterations=FAN_MAX_ITERATIONS,
      fixed_step=True,
      max_lam=FAN_MAX_LAM,
    ),
  )
  return driver.run(base_point=ProgramPoint())
