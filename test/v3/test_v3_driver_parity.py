# SPDX-License-Identifier: MIT

"""Driver oracles vs legacy: shallow-truss nonlinear/Riks and patch_test8 skims.

Documented tolerances:

- Shallow truss (load-controlled Newton on the ch.4 two-bar geometry, spring
  dropped on both sides): ``rtol=1e-8``, ``atol=1e-10`` — the values of the
  landed ``skims/shallow_truss_riks/parity.toml``. Both sides run Newton to
  ``tol=1e-10`` relative residual; the observed final-state deviation on this
  host is below 1e-12 (the residual norms differ slightly: legacy norms the
  constrained residual via ``C.T`` on free DOFs, the driver norms ``P.T r`` in
  reduced coordinates), so the documented band carries ~100x headroom.
- Shallow truss Riks arc-length (same truss-only geometry, ``fixedStep``,
  ``maxLam=10.0``, ``tol=1e-10`` both sides): the same documented band. The
  observed per-cycle deviation on this host is below 7e-14 on the committed
  load parameters and below 6e-17 on the committed states — the reduced-space
  two-solve and the residual/reference norms differ from legacy's full-space
  ``DofSpace`` arithmetic at the last-ulp level — with exact integer
  cycle-count and per-cycle correction-count equality (fixedStep makes the
  trajectory deterministic), so the documented band carries ~1000x headroom.
- patch_test8_nonlinear / _ramp / _prescribed: the skim's own
  ``parity.toml`` (``rtol=1e-10``, ``atol=1e-12``); the observed final-state
  deviation is at the 1e-19 level (direct solves of an exactly linear
  problem). The ramp case additionally pins the cached-factorization
  contract: exactly one reduced factorization across the whole 4-step ramp.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from _legacy_parity import legacy_nonlinear_state, load_parity_tolerances

from pyfem.v3.compile.continuum import q8_reference_registry
from pyfem.v3.compile.system import compile_system
from pyfem.v3.compile.truss import truss_reference_registry
from pyfem.v3.constraints import compile_constraint_map
from pyfem.v3.driver import (
  ArcLengthSettings,
  ArcLengthTermination,
  DriverStatus,
  NonlinearStaticDriver,
  RiksDriver,
  SubstepStatus,
)
from pyfem.v3.io.dat import read_dat_mesh
from pyfem.v3.io.solver_pro import (
  parse_nonlinear_solver_settings,
  parse_riks_solver_settings,
)
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
  AffineCoefficientSpec,
  AffineValueSpec,
  DofRef,
  NodalLoadSpec,
  PrescribedDofSpec,
  ProgramCoordinateSpec,
  ProgramCoordinateValue,
  ProgramPoint,
)

ROOT = Path(__file__).resolve().parents[2]
_COMPONENT = {"u": "x", "v": "y"}

SHALLOW_TRUSS_RTOL = 1.0e-8
SHALLOW_TRUSS_ATOL = 1.0e-10

_SHALLOW_TRUSS_DAT = """<Nodes>
 0 -10.0 0.0 ;
 1  10.0 0.0 ;
 2   0.0 0.5 ;
</Nodes>

<Elements>
 1 'TrussElem' 0 2 ;
 2 'TrussElem' 1 2 ;
</Elements>

<NodeConstraints>
 u[0] = 0.0;
 v[0] = 0.0;
 u[1] = 0.0;
 v[1] = 0.0;
</NodeConstraints>

<ExternalForces>
 v[2] = -100.0 ;
</ExternalForces>
"""

_SHALLOW_TRUSS_PRO = """
#  Shallow truss (ch.4 geometry) + NonlinearSolver under load control.
#  Spring element of the Riks skim intentionally absent: pure truss pair.

input = "{dat}";

TrussElem =
{{
  type = "Truss";
  E    = 5e6;
  Area = 1.0;
}};

solver =
{{
  type = "NonlinearSolver";

  tol = 1.0e-10;
  iterMax = 25;
  loadTable = [0.25, 0.5, 0.75, 1.0];
}};
"""

_SHALLOW_TRUSS_RIKS_PRO = """
#  Shallow truss (ch.4 geometry) + RiksSolver arc-length continuation.
#  Spring element of the Riks skim intentionally absent: pure truss pair.

input = "{dat}";

TrussElem =
{{
  type = "Truss";
  E    = 5e6;
  Area = 1.0;
}};

solver =
{{
  type = "RiksSolver";

  tol = 1.0e-10;
  iterMax = 25;
  fixedStep = true;
  maxLam = 10.0;
}};
"""


def _source(label: str) -> SourceContext:
  return SourceContext(source=label)


def _shallow_truss_model() -> ModelSpec:
  nodes = (
    NodeSpec(id=0, coordinates=(-10.0, 0.0), source=_source("n0")),
    NodeSpec(id=1, coordinates=(10.0, 0.0), source=_source("n1")),
    NodeSpec(id=2, coordinates=(0.0, 0.5), source=_source("n2")),
  )
  cells = (
    CellSpec(id="left", node_ids=(0, 2), source=_source("c0")),
    CellSpec(id="right", node_ids=(1, 2), source=_source("c1")),
  )
  block = CellBlockSpec(
    id="bars",
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
      MaterialParameterSpec("youngs_modulus", 5.0e6),
      MaterialParameterSpec("area", 1.0),
    ),
    source=_source("material"),
  )
  region = RegionSpec(
    id="domain",
    cell_refs=(CellRef("bars", "left"), CellRef("bars", "right")),
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


def _shallow_truss_driver() -> NonlinearStaticDriver:
  system = compile_system(_shallow_truss_model(), truss_reference_registry())
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
      for node in (0, 1)
      for component in ("x", "y")
    ),
    coordinates=(ProgramCoordinateSpec(name="load", kind="load"),),
  )
  loads = (
    NodalLoadSpec(
      id="apex",
      target=DofRef(node_id=2, field_id="displacement", component="y"),
      value=AffineValueSpec(
        coefficients=(AffineCoefficientSpec("load", -100.0, _source("coef")),),
        source=_source("load"),
      ),
      source=_source("apex"),
    ),
  )
  return NonlinearStaticDriver(system, coordinate_map, loads)


def _ramp_points(factors: tuple[float, ...]) -> tuple[ProgramPoint, ...]:
  return tuple(
    ProgramPoint((ProgramCoordinateValue("load", factor),)) for factor in factors
  )


def test_shallow_truss_nonlinear_matches_legacy(tmp_path: Path) -> None:
  dat = tmp_path / "ShallowTrussNonlinear.dat"
  pro = tmp_path / "skim.pro"
  dat.write_text(_SHALLOW_TRUSS_DAT, encoding="utf-8")
  pro.write_text(_SHALLOW_TRUSS_PRO.format(dat=dat), encoding="utf-8")
  legacy = legacy_nonlinear_state(pro)

  driver = _shallow_truss_driver()
  result = driver.run(
    base_point=_ramp_points((0.0,))[0],
    target_points=_ramp_points((0.25, 0.5, 0.75, 1.0)),
  )
  assert result.status is DriverStatus.COMPLETED
  assert result.statistics.committed_substep_count == 4
  # Genuine nonlinearity: every committed substep needed several corrections.
  assert all(len(record.iterations) >= 3 for record in result.records)
  state = driver.owner.accepted_physical().values
  np.testing.assert_allclose(
    state,
    legacy,
    rtol=SHALLOW_TRUSS_RTOL,
    atol=SHALLOW_TRUSS_ATOL,
  )
  # The full-residual reactions equilibrate the applied apex load.
  observation = result.records[-1].observation
  assert observation is not None
  reactions = observation.reactions.values
  np.testing.assert_allclose(
    reactions[[1, 3]],
    [50.0, 50.0],
    rtol=0.0,
    atol=1.0e-6,
  )


def _shallow_truss_riks_driver() -> RiksDriver:
  system = compile_system(_shallow_truss_model(), truss_reference_registry())
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
      for node in (0, 1)
      for component in ("x", "y")
    ),
    coordinates=(ProgramCoordinateSpec(name="load", kind="load"),),
  )
  loads = (
    NodalLoadSpec(
      id="apex",
      target=DofRef(node_id=2, field_id="displacement", component="y"),
      value=AffineValueSpec(
        coefficients=(AffineCoefficientSpec("load", -100.0, _source("coef")),),
        source=_source("load"),
      ),
      source=_source("apex"),
    ),
  )
  return RiksDriver(
    system,
    coordinate_map,
    loads,
    ArcLengthSettings(
      tolerance=1.0e-10,
      max_iterations=25,
      fixed_step=True,
      max_lam=10.0,
    ),
  )


def _legacy_riks_cycles(pro_path: Path) -> list[tuple[float, int, np.ndarray]]:
  """Legacy ``RiksSolver`` run cycle by cycle, capturing committed points."""
  from pyfem.io.InputReader import InputRead
  from pyfem.solvers.RiksSolver import RiksSolver

  props, globdat = InputRead(str(pro_path))
  solver = RiksSolver(props, globdat)
  settings = parse_riks_solver_settings(pro_path.read_text(encoding="utf-8"))
  assert settings is not None
  solver.tol = settings.tol
  solver.iterMax = settings.iter_max
  solver.optiter = settings.opt_iter
  solver.fixedStep = settings.fixed_step
  solver.maxLam = settings.max_lam
  solver.maxFactor = settings.max_factor
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


def test_shallow_truss_riks_matches_legacy_per_cycle(tmp_path: Path) -> None:
  dat = tmp_path / "ShallowtrussRiksTrussOnly.dat"
  pro = tmp_path / "skim.pro"
  dat.write_text(_SHALLOW_TRUSS_DAT, encoding="utf-8")
  pro.write_text(_SHALLOW_TRUSS_RIKS_PRO.format(dat=dat), encoding="utf-8")
  legacy = _legacy_riks_cycles(pro)

  driver = _shallow_truss_riks_driver()
  result = driver.run(base_point=ProgramPoint())
  assert result.status is DriverStatus.COMPLETED
  assert result.termination_reason is ArcLengthTermination.LOAD_PARAMETER_LIMIT
  committed = [
    record for record in result.records if record.status is SubstepStatus.COMMITTED
  ]
  # Exact integer cycle-count equality: fixedStep makes the path deterministic.
  assert len(committed) == len(legacy)
  for record, (legacy_lam, legacy_iiter, legacy_state) in zip(
    committed,
    legacy,
    strict=True,
  ):
    # Per-cycle committed points (lam_k, u_k) and correction-count equality.
    np.testing.assert_allclose(
      record.lam,
      legacy_lam,
      rtol=SHALLOW_TRUSS_RTOL,
      atol=SHALLOW_TRUSS_ATOL,
    )
    assert record.committed_coefficients is not None
    np.testing.assert_allclose(
      record.committed_coefficients.values,
      legacy_state,
      rtol=SHALLOW_TRUSS_RTOL,
      atol=SHALLOW_TRUSS_ATOL,
    )
    assert len(record.iterations) - 1 == legacy_iiter
  # Final state equality (implied per-cycle, pinned explicitly).
  np.testing.assert_allclose(
    driver.owner.accepted_physical().values,
    legacy[-1][2],
    rtol=SHALLOW_TRUSS_RTOL,
    atol=SHALLOW_TRUSS_ATOL,
  )
  # The final full-residual reactions equilibrate lam * fhat: symmetric
  # supports each carry half the apex load, and the x reactions cancel.
  observation = result.records[-1].observation
  assert observation is not None
  lam_final = result.final_continuation.lam
  reactions = observation.reactions.values
  np.testing.assert_allclose(
    reactions[[1, 3]],
    [50.0 * lam_final, 50.0 * lam_final],
    rtol=1.0e-9,
    atol=1.0e-8,
  )
  assert float(reactions[0] + reactions[2]) == pytest.approx(0.0, abs=1.0e-8)


def _patch_model(dat_path: Path) -> tuple[ModelSpec, tuple, tuple, tuple]:
  mesh, prescribed, ties, loads = read_dat_mesh(dat_path)
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
      node_ids=tuple(int(node_id) for node_id in mesh.conn[index]),
      source=_source(f"dat:cell:{index}"),
    )
    for index in range(mesh.conn.shape[0])
  )
  block = CellBlockSpec(
    id="cells",
    reference_topology="quadrilateral",
    topological_dimension=2,
    embedding_dimension=2,
    geometry_interpolation="serendipity-quad8",
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
    quadrature="gauss-3x3",
    source=_source("dat:region"),
  )
  model = ModelSpec(
    mesh=MeshSpec(nodes=nodes, cell_blocks=(block,), source=_source("dat:mesh")),
    fields=(field,),
    materials=(material,),
    regions=(region,),
    source=_source("dat:model"),
  )
  return model, prescribed, ties, loads


def _run_patch_skim(
  skim_name: str,
  dat_path: Path,
) -> tuple[NonlinearStaticDriver, object, np.ndarray, tuple[float, float]]:
  """Drive one skims-format nonlinear case end-to-end through the driver."""
  skim_dir = ROOT / "skims" / skim_name
  pro_text = (skim_dir / "skim.pro").read_text(encoding="utf-8")
  settings = parse_nonlinear_solver_settings(pro_text)
  assert settings is not None
  assert settings.load_table is not None
  model, prescribed, ties, loads = _patch_model(dat_path)
  assert not ties
  system = compile_system(model, q8_reference_registry())
  constraints = tuple(
    PrescribedDofSpec(
      id=f"p:{item.node_id}:{item.dof_type}",
      target=DofRef(
        node_id=item.node_id,
        field_id="displacement",
        component=_COMPONENT[item.dof_type],
      ),
      value=AffineValueSpec(
        coefficients=(AffineCoefficientSpec("load", item.value, _source("dat:coef")),),
        source=_source(f"dat:p:{item.node_id}:{item.dof_type}"),
      ),
      source=_source(f"dat:p:{item.node_id}:{item.dof_type}"),
    )
    for item in prescribed
  )
  coordinate_map = compile_constraint_map(
    system,
    constraints=constraints,
    coordinates=(ProgramCoordinateSpec(name="load", kind="load"),),
  )
  load_specs = tuple(
    NodalLoadSpec(
      id=f"l:{item.node_id}:{item.dof_type}",
      target=DofRef(
        node_id=item.node_id,
        field_id="displacement",
        component=_COMPONENT[item.dof_type],
      ),
      value=AffineValueSpec(
        coefficients=(AffineCoefficientSpec("load", item.value, _source("dat:coef")),),
        source=_source(f"dat:l:{item.node_id}:{item.dof_type}"),
      ),
      source=_source(f"dat:l:{item.node_id}:{item.dof_type}"),
    )
    for item in loads
  )
  driver = NonlinearStaticDriver(system, coordinate_map, load_specs)
  table = settings.load_table
  result = driver.run(
    base_point=_ramp_points((float(table[0]),))[0],
    target_points=_ramp_points(tuple(float(value) for value in table[1:])),
  )
  legacy = legacy_nonlinear_state(skim_dir / "skim.pro")
  tolerances = load_parity_tolerances(skim_dir)
  return driver, result, legacy, tolerances


def test_patch_test8_nonlinear_end_to_end_matches_legacy() -> None:
  dat = ROOT / "skims" / "patch_test8_loaded" / "PatchTest8_loaded.dat"
  driver, result, legacy, (rtol, atol) = _run_patch_skim(
    "patch_test8_nonlinear",
    dat,
  )
  assert result.status is DriverStatus.COMPLETED
  assert result.statistics.committed_substep_count == 1
  np.testing.assert_allclose(
    driver.owner.accepted_physical().values,
    legacy,
    rtol=rtol,
    atol=atol,
  )
  observation = result.records[-1].observation
  assert observation is not None
  # The applied v-load on node 13 is +1000.0 at full factor; the reactions
  # observed from the full residual equilibrate it: zero net in x, -1000 in y
  # (the force the constraints exert on the supports, per the map's sign
  # convention). Legacy's full residual agrees to 5.5e-13 on this host.
  reactions = observation.reactions.values
  assert float(reactions[0::2].sum()) == pytest.approx(0.0, abs=1.0e-6)
  assert float(reactions[1::2].sum()) == pytest.approx(-1000.0, abs=1.0e-6)


def test_patch_test8_nonlinear_ramp_reuses_one_factorization() -> None:
  dat = ROOT / "skims" / "patch_test8_loaded" / "PatchTest8_loaded.dat"
  driver, result, legacy, (rtol, atol) = _run_patch_skim(
    "patch_test8_nonlinear_ramp",
    dat,
  )
  assert result.status is DriverStatus.COMPLETED
  statistics = result.statistics
  assert statistics.committed_substep_count == 4
  assert statistics.factorization_count == 1
  assert statistics.tangent_refill_count == 1
  assert statistics.factorization_reuse_count == 3
  assert [record.committed_ordinal for record in result.records] == [1, 2, 3, 4]
  np.testing.assert_allclose(
    driver.owner.accepted_physical().values,
    legacy,
    rtol=rtol,
    atol=atol,
  )


def test_patch_test8_nonlinear_prescribed_end_to_end_matches_legacy() -> None:
  dat = ROOT / "examples" / "ch02" / "PatchTest8.dat"
  driver, result, legacy, (rtol, atol) = _run_patch_skim(
    "patch_test8_nonlinear_prescribed",
    dat,
  )
  assert result.status is DriverStatus.COMPLETED
  assert result.statistics.committed_substep_count == 4
  np.testing.assert_allclose(
    driver.owner.accepted_physical().values,
    legacy,
    rtol=rtol,
    atol=atol,
  )
  # Prescribed-driven: offsets move constrained DOFs, so the full-residual
  # reactions do genuine constraint work.
  observation = result.records[-1].observation
  assert observation is not None
  assert observation.constraint_work != 0.0
