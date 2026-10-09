# SPDX-License-Identifier: MIT

"""Contact skim parity vs legacy (M66 task 3, end-to-end leg).

``skims/contact_test02`` trims ``examples/contact/contact_test02.pro`` (the
only wired legacy contact deck — the M62 survey's finding 4) onto the landed
v3 subset: SmallStrainContinuum in place of FiniteStrainContinuum (the v3
finite-strain slice is serendipity-quad8-only and rejects this quad4 mesh;
both sides run the trimmed deck, so parity is exact by construction), a
pinned load table replicating the original ``dtime=0.2`` ramp's first five
cycles, and the ``c1`` Contact block intact as the legacy oracle input. The
v3 side reads only the ``.dat`` and the solver settings and wires the same
obstacle through :mod:`pyfem.v3.compile.contact` (converter support is the
mission's declared NOT-yet), so the test pins converged-state parity per
committed substep, active-set equality at the final point, and the exact
tangent's correction counts against legacy's inexact ones.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from _legacy_parity import load_parity_tolerances

from pyfem.io.InputReader import InputRead
from pyfem.solvers.NonlinearSolver import NonlinearSolver as LegacyNonlinearSolver
from pyfem.v3.compile.contact import (
  ContactSignalPort,
  compile_contact_operator,
  compose_contact_system,
  penalty_disc_declaration,
)
from pyfem.v3.compile.contracts import continuum_reference_registry
from pyfem.v3.compile.system import compile_system
from pyfem.v3.constraints import compile_constraint_map
from pyfem.v3.constraints.compile import CompiledConstraintMap
from pyfem.v3.driver import (
  DriverStatus,
  NonlinearStaticDriver,
  NonlinearStaticSettings,
  SubstepStatus,
)
from pyfem.v3.driver.contracts import NonlinearStaticResult
from pyfem.v3.io.dat import read_dat_mesh
from pyfem.v3.io.solver_pro import parse_nonlinear_solver_settings
from pyfem.v3.model.system import CompiledSystem
from pyfem.v3.spec import (
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
  SourceContext,
)
from pyfem.v3.spec.program import (
  AffineValueSpec,
  DofRef,
  PrescribedDofSpec,
  ProgramCoordinateSpec,
  ProgramCoordinateValue,
  ProgramPoint,
)

ROOT = Path(__file__).resolve().parents[2]
SKIM_DIR = ROOT / "skims" / "contact_test02"
_COMPONENT = {"u": "x", "v": "y"}

# The trimmed deck's obstacle (the c1 block of the legacy skim).
_CENTRE = (8.0, 1.4)
_DIRECTION = (0.0, -0.1)
_RADIUS = 1.0
_PENALTY = 1.0e6


def _source(label: str) -> SourceContext:
  return SourceContext(source=label)


def _legacy_contact_cycles(
  pro_path: Path,
) -> list[tuple[float, int, np.ndarray]]:
  """Legacy ``NonlinearSolver`` run cycle by cycle, capturing committed points."""
  props, globdat = InputRead(str(pro_path))
  solver = LegacyNonlinearSolver(props, globdat)
  settings = parse_nonlinear_solver_settings(pro_path.read_text(encoding="utf-8"))
  assert settings is not None
  solver.tol = settings.tol
  solver.iterMax = settings.iter_max
  solver.maxCycle = settings.max_cycle
  solver.dtime = settings.dtime
  if settings.load_table is not None:
    solver.loadTable = settings.load_table
  cycles: list[tuple[float, int, np.ndarray]] = []
  while globdat.active:
    solver.run(props, globdat)
    cycles.append(
      (
        float(globdat.lam),
        globdat.solverStatus.iiter,
        np.asarray(globdat.state).copy(),
      )
    )
  return cycles


def _contact_system() -> tuple[
  CompiledSystem,
  CompiledConstraintMap,
  NonlinearStaticSettings,
  tuple[float, ...],
  np.ndarray,
]:
  """Compile the skim's quad4 strip and wire the penalty contact operator."""
  pro_text = (SKIM_DIR / "skim.pro").read_text(encoding="utf-8")
  settings = parse_nonlinear_solver_settings(pro_text)
  assert settings is not None
  assert settings.load_table is not None
  mesh, prescribed, ties, loads = read_dat_mesh(SKIM_DIR / "contact_test02.dat")
  assert not ties
  assert not loads
  nodes = tuple(
    NodeSpec(
      id=int(node_id),
      coordinates=tuple(float(value) for value in mesh.coords[index]),
      source=_source(f"dat:node:{int(node_id)}"),
    )
    for index, node_id in enumerate(mesh.node_ids)
  )
  cells = tuple(
    CellSpec(
      id=f"cell-{index}",
      node_ids=tuple(int(mesh.node_ids[position]) for position in mesh.conn[index]),
      source=_source(f"dat:cell:{index}"),
    )
    for index in range(mesh.conn.shape[0])
  )
  block = CellBlockSpec(
    id="cells",
    reference_topology="quadrilateral",
    topological_dimension=2,
    embedding_dimension=2,
    geometry_interpolation="bilinear-quad4",
    cells=cells,
    source=_source("dat:cells"),
  )
  field = FieldSpec(
    id="displacement",
    components=("x", "y"),
    location="node",
    source=_source("dat:field"),
  )
  material = MaterialSpec(
    id="elastic",
    model="plane-stress-linear-elastic",
    parameters=(
      MaterialParameterSpec("youngs_modulus", 1.0e6, _source("dat:E")),
      MaterialParameterSpec("poisson_ratio", 0.25, _source("dat:nu")),
    ),
    source=_source("dat:material"),
  )
  region = RegionSpec(
    id="domain",
    cell_refs=tuple(CellRef(block.id, cell.id) for cell in cells),
    field_ids=(field.id,),
    material_id=material.id,
    formulation="small-strain-continuum",
    quadrature="gauss-2x2",
    source=_source("dat:region"),
  )
  model = ModelSpec(
    mesh=MeshSpec(nodes=nodes, cell_blocks=(block,), source=_source("dat:mesh")),
    fields=(field,),
    materials=(material,),
    regions=(region,),
    source=_source("dat:model"),
  )
  base = compile_system(model, continuum_reference_registry())
  # Legacy Contact loops over all nodes (no search): the declared surface set
  # is the whole mesh, in .dat order.
  node_ids = tuple(int(node_id) for node_id in mesh.node_ids)
  declaration = penalty_disc_declaration(
    block_id="c1",
    space_id="displacement",
    contact_ids=tuple(f"contact-{node_id}" for node_id in node_ids),
    node_ids=node_ids,
    centre=_CENTRE,
    direction=_DIRECTION,
    radius=_RADIUS,
    penalty=_PENALTY,
    signal_port=ContactSignalPort(port_id="load-factor", signal_id="load"),
    source=_source("skim:c1"),
  )
  contact_block, contact_operator = compile_contact_operator(base, declaration)
  system = compose_contact_system(base, contact_block, contact_operator)
  seen: set[tuple[int, str]] = set()
  constraints = []
  for item in prescribed:
    key = (item.node_id, item.dof_type)
    if key in seen:
      continue  # the legacy deck lists u[81] twice
    seen.add(key)
    assert item.value == 0.0
    constraints.append(
      PrescribedDofSpec(
        id=f"p:{item.node_id}:{item.dof_type}",
        target=DofRef(
          node_id=item.node_id,
          field_id="displacement",
          component=_COMPONENT[item.dof_type],
        ),
        value=AffineValueSpec(constant=0.0),
        source=_source(f"dat:p:{item.node_id}:{item.dof_type}"),
      )
    )
  coordinate_map = compile_constraint_map(
    system,
    constraints=tuple(constraints),
    coordinates=(ProgramCoordinateSpec(name="load", kind="load"),),
  )
  driver_settings = NonlinearStaticSettings(
    tolerance=settings.tol,
    max_iterations=settings.iter_max,
  )
  table = tuple(float(value) for value in settings.load_table[1:])
  node_xy = np.array(mesh.coords, dtype=np.float64)
  return system, coordinate_map, driver_settings, table, node_xy


def _ramp(factors: tuple[float, ...]) -> tuple[ProgramPoint, ...]:
  return tuple(
    ProgramPoint((ProgramCoordinateValue("load", factor),)) for factor in factors
  )


def _drive(
  system: CompiledSystem,
  coordinate_map: CompiledConstraintMap,
  settings: NonlinearStaticSettings,
  factors: tuple[float, ...],
) -> tuple[NonlinearStaticDriver, NonlinearStaticResult]:
  driver = NonlinearStaticDriver(system, coordinate_map, (), settings)
  result = driver.run(
    base_point=_ramp((0.0,))[0],
    target_points=_ramp(factors),
  )
  return driver, result


def _engaged_nodes(state: np.ndarray, lam: float, node_xy: np.ndarray) -> set[int]:
  """Active contact set, computed test-side from a committed state."""
  centre = np.array(_CENTRE) + lam * np.array(_DIRECTION)
  positions = node_xy + state.reshape(-1, 2)
  distance = np.sqrt(np.sum((positions - centre) ** 2, axis=1))
  return {index for index in range(len(node_xy)) if _RADIUS - distance[index] > 0.0}


def test_contact_test02_skim_matches_legacy() -> None:
  rtol, atol = load_parity_tolerances(SKIM_DIR)
  legacy = _legacy_contact_cycles(SKIM_DIR / "skim.pro")
  assert [round(lam, 12) for lam, _, _ in legacy] == [0.2, 0.4, 0.6, 0.8, 1.0]
  system, coordinate_map, settings, table, node_xy = _contact_system()
  driver, result = _drive(system, coordinate_map, settings, table)
  assert result.status is DriverStatus.COMPLETED
  assert result.statistics.committed_substep_count == len(legacy) == 5
  committed = [
    record for record in result.records if record.status is SubstepStatus.COMMITTED
  ]
  # Correction counts: the v3 Newton loop records one evaluation at the
  # entering state, so len(iterations) - 1 corrections per committed substep
  # (the Riks precedent); the exact tangent never needs more corrections than
  # legacy's inexact one — measured on this host: v3 [3, 3, 3, 3, 3]
  # corrections vs legacy [3, 3, 3, 3, 3] iterations (the deck's converged
  # overlaps reach only ~2% of the radius, where legacy's overlap/d
  # inexactness is mildest; the FD conviction battery measures the gap
  # growing with penetration in test_v3_contact.py).
  for record, (_, legacy_iiter, _) in zip(committed, legacy, strict=True):
    assert len(record.iterations) - 1 <= legacy_iiter
  # Converged-state parity at the final committed point.
  final_state = driver.owner.accepted_physical().values
  np.testing.assert_allclose(final_state, legacy[-1][2], rtol=rtol, atol=atol)
  assert float(final_state.min()) == pytest.approx(-1.032008e-01, rel=1e-4)
  # Per-substep parity: prefix runs reproduce each legacy cycle's state.
  for count in range(1, len(legacy) + 1):
    prefix_driver, prefix_result = _drive(
      system,
      coordinate_map,
      settings,
      table[:count],
    )
    assert prefix_result.status is DriverStatus.COMPLETED
    np.testing.assert_allclose(
      prefix_driver.owner.accepted_physical().values,
      legacy[count - 1][2],
      rtol=rtol,
      atol=atol,
    )
  # The active set at the final point matches legacy's, computed test-side.
  assert _engaged_nodes(final_state, 1.0, node_xy) == _engaged_nodes(
    legacy[-1][2],
    1.0,
    node_xy,
  )
  assert len(_engaged_nodes(final_state, 1.0, node_xy)) > 0
