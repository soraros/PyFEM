# SPDX-License-Identifier: MIT

"""Legacy-deck converter: subset reading, coded rejections, round-trip oracles.

The round-trip oracles convert every in-subset ``skims/`` deck (the
``patch_test*`` small-strain family, the finite-strain ``cantilever8``, and
the multi-group ``shallow_truss_riks``), compile the emitted specs with the
landed compiler unchanged, drive them with the landed drivers, and compare
against the landed legacy oracle states within each skim's ``parity.toml``
tolerances. ``shallow_truss_riks`` is the M32 stretch-path oracle: the full
deck — truss group plus its ``SpringElem`` group — runs end-to-end through
the landed ``RiksDriver``, matching the legacy-instrumented per-cycle
``(lam, u)`` trajectory with exact integer cycle-count equality.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from _legacy_parity import (
  legacy_nonlinear_state,
  legacy_state,
  load_parity_tolerances,
)

from pyfem.v3.driver import (
  ArcLengthTermination,
  DriverStatus,
  NonlinearStaticDriver,
  NonlinearStaticSettings,
  SubstepStatus,
)
from pyfem.v3.io.legacy_deck import (
  ConvertedDeck,
  DeckConversionError,
  compile_deck,
  read_legacy_deck,
  run_deck,
)
from pyfem.v3.io.solver_pro import parse_riks_solver_settings
from pyfem.v3.materials.isotropic_hardening_plasticity import (
  isotropic_hardening_calibration,
  isotropic_hardening_plasticity_kernel,
)
from pyfem.v3.model.operator import EvaluationStatus
from pyfem.v3.model.system import CompiledSystem
from pyfem.v3.spec.program import (
  AffineTieSpec,
  PrescribedDofSpec,
  ProgramCoordinateValue,
  ProgramPoint,
)

ROOT = Path(__file__).resolve().parents[2]
SKIMS = ROOT / "skims"

LINEAR_SKIMS = (
  "patch_test8",
  "patch_test8_loaded",
  "patch_test8_mpc",
  "patch_test3",
  "patch_test4",
  "patch_test8_3d",
  "patch_test8_plane_strain",
)
NONLINEAR_SKIMS = (
  "patch_test8_nonlinear",
  "patch_test8_nonlinear_prescribed",
  "patch_test8_nonlinear_ramp",
  "cantilever8",
)
ROUND_TRIP_SKIMS = LINEAR_SKIMS + NONLINEAR_SKIMS

_MINI_NODES = (
  " 0 0.0 0.0; 1 1.0 0.0; 2 1.0 1.0; 3 0.0 1.0;\n"
  " 4 0.5 0.0; 5 1.0 0.5; 6 0.5 1.0; 7 0.0 0.5;"
)
_MINI_ELEMENTS = ' 1 "ContElem" 0 4 1 5 2 6 3 7;'
_MINI_CONSTRAINTS = " u[0] = 0.0;\n v[0] = 0.0;\n v[3] = 0.0;"
_MINI_FORCES = " v[2] = 1.0;"

_MINI_ELEMENT_BLOCK = """ContElem =
{
  type = "SmallStrainContinuum";
  material =
  {
    type = "PlaneStress";
    E = 1.0e6;
    nu = 0.25;
  };
};
"""

_MINI_SOLVER_BLOCK = 'solver = { type = "LinearSolver"; };'


def _mini_dat(
  *,
  nodes: str = _MINI_NODES,
  elements: str = _MINI_ELEMENTS,
  constraints: str = _MINI_CONSTRAINTS,
  forces: str = _MINI_FORCES,
  constraints_tag: str = "<NodeConstraints>",
  extra: str = "",
) -> str:
  return (
    "<Nodes>\n"
    f"{nodes}\n"
    "</Nodes>\n"
    "<Elements>\n"
    f"{elements}\n"
    "</Elements>\n"
    f"{constraints_tag}\n"
    f"{constraints}\n"
    "</NodeConstraints>\n"
    "<ExternalForces>\n"
    f"{forces}\n"
    "</ExternalForces>\n"
    f"{extra}"
  )


def _mini_pro(
  *,
  element_block: str = _MINI_ELEMENT_BLOCK,
  solver_block: str = _MINI_SOLVER_BLOCK,
  extra: str = "",
  input_line: str = 'input = "mini.dat";',
) -> str:
  return f"{input_line}\n{element_block}{solver_block}\n{extra}"


def _write_deck(tmp_path: Path, pro_text: str, dat_text: str) -> Path:
  (tmp_path / "mini.dat").write_text(dat_text, encoding="utf-8")
  pro_path = tmp_path / "mini.pro"
  pro_path.write_text(pro_text, encoding="utf-8")
  return pro_path


def _convert_mini(tmp_path: Path, pro_text: str, dat_text: str) -> ConvertedDeck:
  return read_legacy_deck(_write_deck(tmp_path, pro_text, dat_text))


def _rejection_codes(excinfo: pytest.ExceptionInfo[DeckConversionError]) -> set[str]:
  assert type(excinfo.value) is DeckConversionError
  return {diagnostic.code for diagnostic in excinfo.value.diagnostics}


# --- reading the supported subset -----------------------------------------------


def test_patch_test8_deck_reads_model_and_program() -> None:
  deck = read_legacy_deck(SKIMS / "patch_test8" / "skim.pro")
  model = deck.model
  assert len(model.mesh.nodes) == 20
  assert model.mesh.nodes[0].id == 0
  assert model.mesh.nodes[0].coordinates == (0.0, 0.0)
  assert model.mesh.nodes[0].source.line is not None
  assert model.mesh.nodes[0].source.source.endswith("PatchTest8.dat")
  (block,) = model.mesh.cell_blocks
  assert block.id == "ContElem"
  assert block.geometry_interpolation == "serendipity-quad8"
  assert [cell.id for cell in block.cells] == [1, 2, 3, 4, 5]
  (material,) = model.materials
  assert material.model == "plane-stress-linear-elastic"
  parameters = {parameter.name: parameter.value for parameter in material.parameters}
  assert parameters == {"youngs_modulus": 1.0e6, "poisson_ratio": 0.25}
  (region,) = model.regions
  assert region.formulation == "small-strain-continuum"
  assert region.quadrature == "gauss-3x3"
  assert len(region.cell_refs) == 5
  assert len(deck.program.constraints) == 16
  assert deck.program.loads == ()
  assert deck.solver.solver_type == "LinearSolver"
  assert deck.solver.load_factors == (1.0,)
  assert [note.code for note in deck.not_converted] == [
    "not-converted-output-module",
    "not-converted-output-module",
  ]


def test_linear_deck_prescribed_values_are_constant() -> None:
  deck = read_legacy_deck(SKIMS / "patch_test8" / "skim.pro")
  for constraint in deck.program.constraints:
    assert type(constraint) is PrescribedDofSpec
    assert constraint.value.coefficients == ()
    assert type(constraint.value.constant) is float
  (coordinate,) = deck.program.coordinates
  assert coordinate.name == "load"
  assert coordinate.kind == "load"


def test_nonlinear_deck_prescribed_values_scale_with_load() -> None:
  deck = read_legacy_deck(SKIMS / "patch_test8_nonlinear_prescribed" / "skim.pro")
  for constraint in deck.program.constraints:
    assert type(constraint) is PrescribedDofSpec
    assert constraint.value.constant == 0.0
    (coefficient,) = constraint.value.coefficients
    assert coefficient.coordinate == "load"
  assert deck.solver.solver_type == "NonlinearSolver"
  assert deck.solver.load_factors == (0.25, 0.5, 0.75, 1.0)
  assert deck.solver.tolerance == pytest.approx(1.0e-10)
  assert deck.solver.max_iterations == 10


def test_patch_test8_mpc_deck_reads_one_master_ties() -> None:
  deck = read_legacy_deck(SKIMS / "patch_test8_mpc" / "skim.pro")
  ties = [
    constraint
    for constraint in deck.program.constraints
    if type(constraint) is AffineTieSpec
  ]
  prescribed = [
    constraint
    for constraint in deck.program.constraints
    if type(constraint) is PrescribedDofSpec
  ]
  assert len(prescribed) == 12
  assert len(ties) == 3
  first = ties[0]
  assert first.slave.node_id == 2
  assert first.slave.component == "x"
  assert first.master.node_id == 1
  assert first.master.component == "x"
  assert first.factor == pytest.approx(2.0)
  assert first.offset.constant == pytest.approx(0.0)


def test_patch_test8_loaded_deck_reads_nodal_loads() -> None:
  deck = read_legacy_deck(SKIMS / "patch_test8_loaded" / "skim.pro")
  (load,) = deck.program.loads
  assert load.target.node_id == 13
  assert load.target.component == "y"
  (coefficient,) = load.value.coefficients
  assert coefficient.coordinate == "load"
  assert coefficient.coefficient == pytest.approx(1000.0)


def test_patch_test3_deck_reads_the_tri3_slice() -> None:
  deck = read_legacy_deck(SKIMS / "patch_test3" / "skim.pro")
  (block,) = deck.model.mesh.cell_blocks
  assert block.reference_topology == "triangle"
  assert block.topological_dimension == 2
  assert block.embedding_dimension == 2
  assert block.geometry_interpolation == "linear-tria3"
  assert len(block.cells) == 10
  assert all(len(cell.node_ids) == 3 for cell in block.cells)
  (field,) = deck.model.fields
  assert field.components == ("x", "y")
  (material,) = deck.model.materials
  assert material.model == "plane-stress-linear-elastic"
  (region,) = deck.model.regions
  assert region.formulation == "small-strain-continuum"
  assert region.quadrature == "gauss-tria3-1"
  assert len(deck.program.constraints) == 8
  assert ("topology", "linear-tria3") in deck.registry
  assert ("quadrature", "gauss-tria3-1") in deck.registry


def test_patch_test4_deck_reads_the_quad4_slice() -> None:
  deck = read_legacy_deck(SKIMS / "patch_test4" / "skim.pro")
  (block,) = deck.model.mesh.cell_blocks
  assert block.reference_topology == "quadrilateral"
  assert block.geometry_interpolation == "bilinear-quad4"
  assert len(block.cells) == 5
  assert all(len(cell.node_ids) == 4 for cell in block.cells)
  (material,) = deck.model.materials
  assert material.model == "plane-stress-linear-elastic"
  (region,) = deck.model.regions
  assert region.quadrature == "gauss-2x2"
  assert ("topology", "bilinear-quad4") in deck.registry
  assert ("quadrature", "gauss-2x2") in deck.registry


def test_patch_test8_3d_deck_reads_the_hex8_3d_slice() -> None:
  deck = read_legacy_deck(SKIMS / "patch_test8_3d" / "skim.pro")
  assert len(deck.model.mesh.nodes) == 16
  assert deck.model.mesh.nodes[2].coordinates == (0.24, 0.12, 0.0)
  (block,) = deck.model.mesh.cell_blocks
  assert block.reference_topology == "hexahedron"
  assert block.topological_dimension == 3
  assert block.embedding_dimension == 3
  assert block.geometry_interpolation == "trilinear-hex8"
  assert len(block.cells) == 5
  assert all(len(cell.node_ids) == 8 for cell in block.cells)
  (field,) = deck.model.fields
  assert field.components == ("x", "y", "z")
  (material,) = deck.model.materials
  assert material.model == "isotropic-linear-elastic"
  parameters = {parameter.name: parameter.value for parameter in material.parameters}
  assert parameters == {"youngs_modulus": 1.0e6, "poisson_ratio": 0.25}
  (region,) = deck.model.regions
  assert region.formulation == "small-strain-continuum"
  assert region.quadrature == "gauss-2x2x2"
  assert len(deck.program.constraints) == 24
  w_constraints = [
    constraint
    for constraint in deck.program.constraints
    if constraint.target.component == "z"
  ]
  assert len(w_constraints) == 8
  assert {constraint.target.node_id for constraint in w_constraints} == {
    0,
    1,
    22,
    3,
    14,
    5,
    6,
    99,
  }
  assert all(constraint.value.constant == 0.0 for constraint in w_constraints)
  assert ("topology", "trilinear-hex8") in deck.registry
  assert ("quadrature", "gauss-2x2x2") in deck.registry
  assert ("formulation", "small-strain-continuum-3d") in deck.registry
  assert ("material", "isotropic-linear-elastic") in deck.registry


def test_patch_test8_plane_strain_deck_reads_the_plane_strain_law() -> None:
  deck = read_legacy_deck(SKIMS / "patch_test8_plane_strain" / "skim.pro")
  (block,) = deck.model.mesh.cell_blocks
  assert block.geometry_interpolation == "serendipity-quad8"
  (material,) = deck.model.materials
  assert material.model == "plane-strain-linear-elastic"
  parameters = {parameter.name: parameter.value for parameter in material.parameters}
  assert parameters == {"youngs_modulus": 1.0e6, "poisson_ratio": 0.25}
  (region,) = deck.model.regions
  assert region.formulation == "small-strain-continuum"
  assert region.quadrature == "gauss-3x3"
  assert ("material", "plane-strain-linear-elastic") in deck.registry


def test_cantilever8_deck_reads_the_finite_strain_slice() -> None:
  deck = read_legacy_deck(SKIMS / "cantilever8" / "skim.pro")
  assert len(deck.model.mesh.nodes) == 43
  (block,) = deck.model.mesh.cell_blocks
  assert block.geometry_interpolation == "serendipity-quad8"
  assert len(block.cells) == 8
  (material,) = deck.model.materials
  assert material.model == "plane-stress-saint-venant-kirchhoff"
  parameters = {parameter.name: parameter.value for parameter in material.parameters}
  assert parameters == {"youngs_modulus": 100.0, "poisson_ratio": 0.3}
  (region,) = deck.model.regions
  assert region.formulation == "total-lagrangian-continuum"
  assert region.quadrature == "gauss-3x3"
  assert ("formulation", "total-lagrangian-continuum") in deck.registry
  assert ("material", "plane-stress-saint-venant-kirchhoff") in deck.registry
  assert deck.solver.solver_type == "NonlinearSolver"
  assert deck.solver.load_factors == tuple(float(k) for k in range(1, 21))
  assert deck.solver.tolerance == pytest.approx(1.0e-3)
  assert deck.solver.max_iterations == 10
  assert [note.code for note in deck.not_converted] == [
    "not-converted-output-module",
    "not-converted-solver-option",
  ]


def test_compile_deck_uses_landed_compiler_unchanged() -> None:
  deck = read_legacy_deck(SKIMS / "patch_test8" / "skim.pro")
  compiled = compile_deck(deck)
  assert type(compiled.system).__name__ == "CompiledSystem"
  assert sum(len(space.coefficient_ids) for space in compiled.system.spaces) == 40
  assert compiled.constraint_map.full_dof_count == 40
  assert compiled.constraint_map.reduced_dof_count == 24


def test_compile_deck_rejects_foreign_input() -> None:
  with pytest.raises(TypeError):
    compile_deck(object())


def test_mini_deck_converts(tmp_path: Path) -> None:
  deck = _convert_mini(tmp_path, _mini_pro(), _mini_dat())
  assert len(deck.model.mesh.nodes) == 8
  assert len(deck.model.mesh.cell_blocks[0].cells) == 1
  assert len(deck.program.constraints) == 3
  assert len(deck.program.loads) == 1
  assert deck.not_converted == ()


# --- round-trip oracles ----------------------------------------------------------


def _legacy_state_in_compiled_order(
  deck: ConvertedDeck,
  system: CompiledSystem,
  legacy: np.ndarray,
) -> np.ndarray:
  """Permute a legacy state vector into the compiled coefficient order.

  Legacy assigns DOFs in ``.dat`` declaration order; the compiled system
  orders coefficients by sorted node id. The orders agree on decks with
  sequentially declared ids (every previously landed parity deck, where this
  mapping is the identity) and differ on the breadth decks (patch_test3,
  patch_test4, patch_test8_3d), so the faithful per-coefficient comparison
  maps the legacy state through the compiled ``coefficient_ids``.
  """
  position = {node.id: index for index, node in enumerate(deck.model.mesh.nodes)}
  components = deck.model.fields[0].components
  component_index = {component: index for index, component in enumerate(components)}
  ndof = len(components)
  state = np.empty(legacy.shape, dtype=np.float64)
  space = system.spaces[0]
  for row, (_field_id, node_id, component) in enumerate(space.coefficient_ids):
    state[row] = legacy[position[node_id] * ndof + component_index[component]]
  return state


@pytest.mark.parametrize("skim_name", ROUND_TRIP_SKIMS)
def test_skim_deck_round_trip_matches_legacy(skim_name: str) -> None:
  skim_dir = SKIMS / skim_name
  deck = read_legacy_deck(skim_dir / "skim.pro")
  compiled = compile_deck(deck)
  run = run_deck(deck)
  assert run.result.status is DriverStatus.COMPLETED
  if deck.solver.solver_type == "LinearSolver":
    legacy = legacy_state(skim_dir / "skim.pro")
  else:
    legacy = legacy_nonlinear_state(skim_dir / "skim.pro")
  legacy = _legacy_state_in_compiled_order(deck, compiled.system, legacy)
  rtol, atol = load_parity_tolerances(skim_dir)
  np.testing.assert_allclose(run.state, legacy, rtol=rtol, atol=atol)


def test_ramp_deck_commits_four_substeps() -> None:
  deck = read_legacy_deck(SKIMS / "patch_test8_nonlinear_ramp" / "skim.pro")
  run = run_deck(deck)
  assert run.result.statistics.committed_substep_count == 4


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

_SHALLOW_TRUSS_PRO = """input = "@dat@";

TrussElem =
{
  type = "Truss";
  E    = 5e6;
  Area = 1.0;
};

solver =
{
  type = "NonlinearSolver";

  tol = 1.0e-10;
  iterMax = 25;
  loadTable = [0.25, 0.5, 0.75, 1.0];
};
"""


def test_truss_deck_round_trip_matches_legacy(tmp_path: Path) -> None:
  dat = tmp_path / "ShallowTrussNonlinear.dat"
  pro = tmp_path / "shallow_truss.pro"
  dat.write_text(_SHALLOW_TRUSS_DAT, encoding="utf-8")
  pro.write_text(
    _SHALLOW_TRUSS_PRO.replace("@dat@", str(dat)),
    encoding="utf-8",
  )
  deck = read_legacy_deck(pro)
  block = deck.model.mesh.cell_blocks[0]
  assert block.geometry_interpolation == "line2"
  assert deck.model.regions[0].formulation == "total-lagrangian-truss"
  material = deck.model.materials[0]
  assert material.model == "uniaxial-linear-elastic"
  parameters = {parameter.name: parameter.value for parameter in material.parameters}
  assert parameters == {"youngs_modulus": 5.0e6, "area": 1.0}

  run = run_deck(deck)
  assert run.result.status is DriverStatus.COMPLETED
  assert run.result.statistics.committed_substep_count == 4
  legacy = legacy_nonlinear_state(pro)
  np.testing.assert_allclose(run.state, legacy, rtol=1.0e-8, atol=1.0e-10)


# --- multi-group spring decks and the Riks dispatch -------------------------------

# The M32 stretch path: the full shallow_truss_riks skim — the truss group
# (region-routed base ModelSpec) plus its SpringElem group (one
# SpringDeclaration through compile_system's springs channel) — converts and
# runs end-to-end through the landed RiksDriver.


def test_shallow_truss_riks_deck_reads_multi_group_model_and_riks_settings() -> None:
  deck = read_legacy_deck(SKIMS / "shallow_truss_riks" / "skim.pro")
  model = deck.model
  assert len(model.mesh.nodes) == 4
  (block,) = model.mesh.cell_blocks
  assert block.id == "TrussElem"
  assert block.geometry_interpolation == "line2"
  assert [cell.id for cell in block.cells] == [1, 2]
  (material,) = model.materials
  assert material.model == "uniaxial-linear-elastic"
  (region,) = model.regions
  assert region.formulation == "total-lagrangian-truss"
  # The SpringElem group is declaration-routed: no cell block or material in
  # the base spec, exactly one SpringDeclaration with the grounded apex
  # spring's stiffness and unit chord direction (0, 1).
  (declaration,) = deck.springs
  assert declaration.block_id == "SpringElem"
  assert declaration.space_id == "displacement"
  assert declaration.spring_ids == (3,)
  assert declaration.node_ids == (4,)
  assert declaration.parameters == (100.0, 0.0, 1.0)
  assert declaration.state_slots == ()
  assert declaration.kernel_name == "legacy-axial-point-spring"
  # RiksSolver settings: the legacy defaults where the block is silent.
  solver = deck.solver
  assert solver.solver_type == "RiksSolver"
  assert solver.load_factors == ()
  assert solver.tolerance == pytest.approx(1.0e-5)
  assert solver.max_iterations == 10
  assert solver.optimal_iterations == 5
  assert solver.fixed_step is True
  assert solver.max_lam == pytest.approx(10.0)
  assert solver.max_factor == pytest.approx(1.0e20)
  assert solver.cycle_cap == 1000
  # Under RiksSolver prescribed values stay constant (the legacy solver never
  # scales constraints); only the loads ride the arc-length parameter.
  for constraint in deck.program.constraints:
    assert type(constraint) is PrescribedDofSpec
    assert constraint.value.coefficients == ()
    assert type(constraint.value.constant) is float
  (load,) = deck.program.loads
  assert load.target.node_id == 4
  assert load.target.component == "y"
  (coefficient,) = load.value.coefficients
  assert coefficient.coordinate == "load"
  assert coefficient.coefficient == pytest.approx(-100.0)
  assert [note.code for note in deck.not_converted] == ["not-converted-output-module"]


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


def test_shallow_truss_riks_deck_matches_legacy_per_cycle() -> None:
  """The full skim deck, SpringElem included, through the RiksDriver.

  The converted multi-group deck — the truss region plus the grounded apex
  spring declaration — matches the legacy-instrumented per-cycle trajectory:
  exact integer cycle-count equality (``fixedStep`` makes the path
  deterministic), the committed ``(lam_k, u_k)`` points, and the per-cycle
  correction counts. The deck's constraint map is homogeneous, so the
  reduced-space Riks dots equal the legacy full-space dots (the documented
  M33 MPC note does not apply here).
  """
  skim_dir = SKIMS / "shallow_truss_riks"
  pro_path = skim_dir / "skim.pro"
  legacy = _legacy_riks_cycles(pro_path)
  rtol, atol = load_parity_tolerances(skim_dir)

  deck = read_legacy_deck(pro_path)
  compiled = compile_deck(deck)
  assert [type(operator).__name__ for operator in compiled.system.operators] == [
    "TrussOperator",
    "SpringOperator",
  ]
  run = run_deck(deck)
  result = run.result
  assert result.status is DriverStatus.COMPLETED
  assert result.termination_reason is ArcLengthTermination.LOAD_PARAMETER_LIMIT
  committed = [
    record for record in result.records if record.status is SubstepStatus.COMMITTED
  ]
  # Exact integer cycle-count equality: fixedStep makes the path deterministic.
  assert len(committed) == len(legacy)
  for record, (legacy_lam, legacy_iiter, legacy_cycle_state) in zip(
    committed,
    legacy,
    strict=True,
  ):
    np.testing.assert_allclose(record.lam, legacy_lam, rtol=rtol, atol=atol)
    assert record.committed_coefficients is not None
    np.testing.assert_allclose(
      record.committed_coefficients.values,
      _legacy_state_in_compiled_order(deck, compiled.system, legacy_cycle_state),
      rtol=rtol,
      atol=atol,
    )
    assert len(record.iterations) - 1 == legacy_iiter
  # Final state equality (implied per-cycle, pinned explicitly).
  np.testing.assert_allclose(
    run.state,
    _legacy_state_in_compiled_order(deck, compiled.system, legacy[-1][2]),
    rtol=rtol,
    atol=atol,
  )
  # The grounded spring relieves the supports: the v3 declaration grounds the
  # spring absolutely, so the anchor node carries no reaction and each truss
  # support carries half of (lam * 100 - k * |v_apex|) by symmetry.
  observation = result.records[-1].observation
  assert observation is not None
  reactions = observation.reactions.values
  lam_final = result.final_continuation.lam
  apex_vertical = float(run.state[7])
  assert apex_vertical < 0.0
  assert reactions[1] == pytest.approx(0.0, abs=1.0e-10)
  np.testing.assert_allclose(
    reactions[[3, 5]],
    [0.5 * (100.0 * lam_final + 100.0 * apex_vertical)] * 2,
    rtol=1.0e-9,
    atol=1.0e-8,
  )
  assert float(reactions[2] + reactions[4]) == pytest.approx(0.0, abs=1.0e-8)


# A shallow-truss-with-spring mini deck (the skim's shape, renumbered) for the
# spring/riks rejection battery and the multi-spring-group reading test.
_SPRING_TRUSS_DAT = """<Nodes>
 0 -10.0 0.0 ;
 1  10.0 0.0 ;
 2   0.0 0.5 ;
 3   0.0 0.0 ;
</Nodes>

<Elements>
 1 'TrussElem' 0 2 ;
 2 'TrussElem' 1 2 ;
 3 'SpringElem' 3 2 ;
</Elements>

<NodeConstraints>
 u[0] = 0.0;
 v[0] = 0.0;
 u[1] = 0.0;
 v[1] = 0.0;
 u[3] = 0.0;
 v[3] = 0.0;
</NodeConstraints>

<ExternalForces>
 v[2] = -100.0 ;
</ExternalForces>
"""

_SPRING_TRUSS_BLOCKS = """TrussElem =
{
  type = "Truss";
  E    = 5e6;
  Area = 1.0;
};

SpringElem =
{
  type = "Spring";
  k    = 100.0;
};
"""

_SPRING_RIKS_SOLVER_BLOCK = """solver =
{
  type = "RiksSolver";

  fixedStep = true;
  maxLam    = 10.0;
};
"""


def _spring_truss_pro(
  *,
  element_blocks: str = _SPRING_TRUSS_BLOCKS,
  solver_block: str = _SPRING_RIKS_SOLVER_BLOCK,
) -> str:
  return _mini_pro(element_block=element_blocks, solver_block=solver_block)


def test_riks_solver_block_reads_authored_keys(tmp_path: Path) -> None:
  pro = _spring_truss_pro(
    solver_block="""solver =
{
  type = "RiksSolver";
  tol = 1.0e-8;
  iterMax = 25;
  optiter = 3;
  maxLam = 5.0;
  maxFactor = 2.0;
};
""",
  )
  deck = _convert_mini(tmp_path, pro, _SPRING_TRUSS_DAT)
  solver = deck.solver
  assert solver.solver_type == "RiksSolver"
  assert solver.load_factors == ()
  assert solver.tolerance == pytest.approx(1.0e-8)
  assert solver.max_iterations == 25
  assert solver.optimal_iterations == 3
  assert solver.fixed_step is False
  assert solver.max_lam == pytest.approx(5.0)
  assert solver.max_factor == pytest.approx(2.0)
  assert solver.cycle_cap == 1000
  assert deck.not_converted == ()


def test_multiple_spring_groups_emit_one_declaration_each(tmp_path: Path) -> None:
  # A second spring group anchored at the grounded node 0, plus a second
  # spring inside the first group anchored at node 1: per-group declarations
  # pack each spring's own chord direction.
  dat = _SPRING_TRUSS_DAT.replace(
    " 3 'SpringElem' 3 2 ;",
    " 3 'SpringElem' 3 2 ;\n 4 'SpringElem2' 0 2 ;\n 5 'SpringElem' 1 2 ;",
  )
  blocks = (
    _SPRING_TRUSS_BLOCKS
    + """SpringElem2 =
{
  type = "Spring";
  k    = 50.0;
};
"""
  )
  deck = _convert_mini(tmp_path, _spring_truss_pro(element_blocks=blocks), dat)
  first, second = deck.springs
  assert first.block_id == "SpringElem"
  assert first.spring_ids == (3, 5)
  assert first.node_ids == (2, 2)
  # Element 3: chord (0,0)->(0,0.5) gives d = (0,1); element 5: chord
  # (10,0)->(0,0.5) gives d = (-10,0.5)/sqrt(100.25).
  length = math.sqrt(100.25)
  assert first.parameters == pytest.approx(
    (100.0, 0.0, 1.0, -10.0 / length, 0.5 / length)
  )
  assert second.block_id == "SpringElem2"
  assert second.spring_ids == (4,)
  assert second.node_ids == (2,)
  # Element 4: chord (-10,0)->(0,0.5).
  assert second.parameters == pytest.approx((50.0, 10.0 / length, 0.5 / length))
  compiled = compile_deck(deck)
  assert [type(operator).__name__ for operator in compiled.system.operators] == [
    "TrussOperator",
    "SpringOperator",
    "SpringOperator",
  ]


_SPRING_DAT_REJECTIONS = {
  "both-ends-free": (
    _SPRING_TRUSS_DAT.replace(" u[3] = 0.0;\n v[3] = 0.0;\n", ""),
    {"unsupported-spring-support"},
  ),
  "both-ends-grounded": (
    _SPRING_TRUSS_DAT.replace(
      "</NodeConstraints>",
      " u[2] = 0.0;\n v[2] = 0.0;\n</NodeConstraints>",
    ),
    {"unsupported-spring-support"},
  ),
  "anchor-nonzero-prescription": (
    _SPRING_TRUSS_DAT.replace(" u[3] = 0.0;", " u[3] = 0.5;"),
    {"unsupported-spring-support"},
  ),
  "anchor-partial-prescription": (
    _SPRING_TRUSS_DAT.replace(" v[3] = 0.0;\n", ""),
    {"unsupported-spring-support"},
  ),
  "spring-wrong-arity": (
    _SPRING_TRUSS_DAT.replace(" 3 'SpringElem' 3 2 ;", " 3 'SpringElem' 3 2 1 ;"),
    {"unsupported-cell-arity"},
  ),
  "spring-on-3d-mesh": (
    _SPRING_TRUSS_DAT.replace(
      " 0 -10.0 0.0 ;\n 1  10.0 0.0 ;\n 2   0.0 0.5 ;\n 3   0.0 0.0 ;",
      " 0 -10.0 0.0 0.0 ;\n 1  10.0 0.0 0.0 ;\n 2   0.0 0.5 0.0 ;\n 3   0.0 0.0 0.0 ;",
    ),
    {"unsupported-cell-arity"},
  ),
  "springs-only-deck": (
    _SPRING_TRUSS_DAT.replace(
      " 1 'TrussElem' 0 2 ;\n 2 'TrussElem' 1 2 ;\n",
      "",
    ),
    {"unsupported-element-groups", "unsupported-pro-construct"},
  ),
}

_SPRING_PRO_REJECTIONS = {
  "spring-linear-solver": (
    _spring_truss_pro(solver_block=_MINI_SOLVER_BLOCK),
    {"incompatible-solver-type"},
  ),
  "spring-missing-k": (
    _spring_truss_pro(
      element_blocks=_SPRING_TRUSS_BLOCKS.replace("  k    = 100.0;\n", ""),
    ),
    {"missing-element-parameter"},
  ),
  "spring-extra-key": (
    _spring_truss_pro(
      element_blocks=_SPRING_TRUSS_BLOCKS.replace(
        "  k    = 100.0;\n",
        "  k    = 100.0;\n  rho  = 1.0;\n",
      ),
    ),
    {"unsupported-element-parameter"},
  ),
  "riks-unknown-key": (
    _spring_truss_pro(
      solver_block=_SPRING_RIKS_SOLVER_BLOCK.replace(
        "  maxLam    = 10.0;\n",
        "  maxLam    = 10.0;\n  maxCycle  = 5;\n",
      ),
    ),
    {"unsupported-solver-parameter"},
  ),
  "riks-fixedstep-not-bool": (
    _spring_truss_pro(
      solver_block=_SPRING_RIKS_SOLVER_BLOCK.replace(
        "  fixedStep = true;",
        "  fixedStep = 1;",
      ),
    ),
    {"deck-pro-syntax"},
  ),
  "riks-tol-not-a-number": (
    _spring_truss_pro(
      solver_block=_SPRING_RIKS_SOLVER_BLOCK.replace(
        "  maxLam    = 10.0;",
        '  maxLam    = 10.0;\n  tol       = "tight";',
      ),
    ),
    {"deck-pro-syntax"},
  ),
}


@pytest.mark.parametrize(
  ("dat_text", "expected"),
  [
    pytest.param(text, codes, id=name)
    for name, (text, codes) in _SPRING_DAT_REJECTIONS.items()
  ],
)
def test_spring_dat_construct_rejections(
  tmp_path: Path,
  dat_text: str,
  expected: set[str],
) -> None:
  with pytest.raises(DeckConversionError) as excinfo:
    _convert_mini(tmp_path, _spring_truss_pro(), dat_text)
  assert expected <= _rejection_codes(excinfo)


@pytest.mark.parametrize(
  ("pro_text", "expected"),
  [
    pytest.param(text, codes, id=name)
    for name, (text, codes) in _SPRING_PRO_REJECTIONS.items()
  ],
)
def test_spring_pro_construct_rejections(
  tmp_path: Path,
  pro_text: str,
  expected: set[str],
) -> None:
  with pytest.raises(DeckConversionError) as excinfo:
    _convert_mini(tmp_path, pro_text, _SPRING_TRUSS_DAT)
  assert expected <= _rejection_codes(excinfo)


# --- plasticity material decks --------------------------------------------------

# The documented M25 parity configuration (test_v3_stateful_plasticity.py):
# linear hardening, the only configuration the legacy law ever defined on the
# plastic path.
_M25_E = 210000.0
_M25_NU = 0.3
_M25_SYIELD = 250.0
_M25_HARD = 1000.0
_M25_LOAD_TABLE = (0.001, 0.002, 0.004)

_PLASTICITY_NODES = (
  " 1 0.0 0.0; 2 0.5 0.0; 3 1.0 0.0; 4 1.0 0.5;\n"
  " 5 1.0 1.0; 6 0.5 1.0; 7 0.0 1.0; 8 0.0 0.5;"
)
_PLASTICITY_ELEMENT = ' 1 "ContElem" 1 2 3 4 5 6 7 8;'
# A homogeneous eps_xx ramp: u_x is prescribed to the node x coordinate and
# scales with the load coordinate under NonlinearSolver (node 6's x stays
# free so every Newton iteration assembles and factorizes the tangent).
_PLASTICITY_CONSTRAINTS = (
  " u[1] = 0.0; u[2] = 0.5; u[3] = 1.0; u[4] = 1.0;\n"
  " u[5] = 1.0; u[7] = 0.0; u[8] = 0.0;\n"
  " v[1] = 0.0; v[2] = 0.0; v[3] = 0.0; v[4] = 0.0;\n"
  " v[5] = 0.0; v[6] = 0.0; v[7] = 0.0; v[8] = 0.0;"
)
_PLASTICITY_ELEMENT_BLOCK = """ContElem =
{
  type = "SmallStrainContinuum";
  material =
  {
    type = "IsotropicHardeningPlasticity";
    E = 210000.0;
    nu = 0.3;
    syield = 250.0;
    hard = 1000.0;
  };
};
"""
_PLASTICITY_SOLVER_BLOCK = """solver =
{
  type = "NonlinearSolver";
  tol = 1.0e-10;
  iterMax = 25;
  loadTable = [0.001, 0.002, 0.004];
};
"""


def _plasticity_pro(
  *,
  element_block: str = _PLASTICITY_ELEMENT_BLOCK,
  solver_block: str = _PLASTICITY_SOLVER_BLOCK,
) -> str:
  return _mini_pro(element_block=element_block, solver_block=solver_block)


def _plasticity_dat() -> str:
  return _mini_dat(
    nodes=_PLASTICITY_NODES,
    elements=_PLASTICITY_ELEMENT,
    constraints=_PLASTICITY_CONSTRAINTS,
    forces="",
  )


def _committed_ip_strains(system: CompiledSystem, values: np.ndarray) -> np.ndarray:
  """Recompute the committed per-integration-point 6-Voigt strains.

  The M25 parity harness expression (test_v3_stateful_plasticity.py): the
  operator's own physical strain-displacement map applied to the committed
  coefficient vector.
  """
  operator = system.operators[0]
  payload = operator.payload
  b_matrix = (
    payload.normalized_strain_displacement.values
    / payload.geometry_scales.values[:, None, None, None]
  )
  gather = operator.header.ports[0].coefficient_map.values
  strain3 = np.einsum("epai,ei->epa", b_matrix, values[gather], optimize=True)
  strains = np.zeros((np.prod(strain3.shape[:2]), 6), dtype=np.float64)
  flat = strain3.reshape(-1, 3)
  strains[:, 0] = flat[:, 0]
  strains[:, 1] = flat[:, 1]
  strains[:, 5] = flat[:, 2]
  return strains


def test_plasticity_deck_reads_the_stateful_material_form(tmp_path: Path) -> None:
  deck = _convert_mini(tmp_path, _plasticity_pro(), _plasticity_dat())
  (material,) = deck.model.materials
  assert material.model == "isotropic-hardening-plasticity"
  assert tuple(parameter.name for parameter in material.parameters) == (
    "youngs_modulus",
    "poisson_ratio",
    "initial_yield_stress",
    "hardening_slope",
  )
  values = {parameter.name: parameter.value for parameter in material.parameters}
  assert values == {
    "youngs_modulus": _M25_E,
    "poisson_ratio": _M25_NU,
    "initial_yield_stress": _M25_SYIELD,
    "hardening_slope": _M25_HARD,
  }
  (region,) = deck.model.regions
  assert region.formulation == "small-strain-continuum"
  assert region.quadrature == "gauss-3x3"
  assert ("material", "isotropic-hardening-plasticity") in deck.registry
  assert deck.solver.solver_type == "NonlinearSolver"
  assert deck.solver.load_factors == _M25_LOAD_TABLE


def test_plasticity_deck_drives_the_m25_parity_path(tmp_path: Path) -> None:
  """The converted plasticity deck commits the M25 kernel-oracle state.

  Mirrors the driver battery of test_v3_stateful_plasticity.py on the
  documented ramp eps_xx = 0.001, 0.002, 0.004: the committed rows equal the
  batched M25 kernel stepped on the committed per-integration-point strain
  path — bitwise, since the driver stages the trial rows of the converged
  iterate — and the algorithmic channel re-factorizes every iteration.
  """
  deck = _convert_mini(tmp_path, _plasticity_pro(), _plasticity_dat())
  compiled = compile_deck(deck)
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
  assert driver.plan.constant_tangent is False
  layout = compiled.system.operators[0].header.state_layout
  assert layout.schema == (
    "pyfem-v3-j2-isotropic-hardening-state-v1|sigma:6,epsilon_e:6,epsilon_p:6,kappa:1"
  )
  assert layout.initial_rows is not None
  assert np.all(layout.initial_rows.values == 0.0)
  block_id = layout.block_id
  calibration = isotropic_hardening_calibration(_M25_E, _M25_NU, _M25_SYIELD, _M25_HARD)
  oracle_rows = np.zeros((9, 19))
  base = ProgramPoint((ProgramCoordinateValue("load", 0.0),))
  for step, factor in enumerate(deck.solver.load_factors, 1):
    target = ProgramPoint((ProgramCoordinateValue("load", factor),))
    result = driver.run(base_point=base, target_points=(target,))
    assert result.status is DriverStatus.COMPLETED
    base = target
    oracle = isotropic_hardening_plasticity_kernel(
      _committed_ip_strains(compiled.system, driver.owner.accepted_physical().values),
      oracle_rows,
      calibration,
    )
    assert oracle.status is EvaluationStatus.OK
    oracle_rows = oracle.trial_rows
    rows = driver.owner.accepted_state(block_id).values
    assert rows.shape == (9, 19)
    np.testing.assert_array_equal(rows, oracle_rows)
    # Homogeneous strain: all nine integration points agree to ~1e-10.
    np.testing.assert_allclose(
      rows, np.broadcast_to(rows[0], rows.shape), rtol=1.0e-9, atol=1.0e-10
    )
    assert result.final_generation.ordinal == step
  assert rows[0, 18] > 0.0  # kappa: the ramp went plastic
  statistics = result.statistics
  assert statistics.committed_substep_count == 3
  assert statistics.rejected_substep_count == 0
  assert statistics.factorization_reuse_count == 0
  assert statistics.factorization_count == statistics.tangent_refill_count
  assert statistics.factorization_count == statistics.linear_solve_count
  assert statistics.factorization_count > statistics.committed_substep_count

  # The one-call run_deck schedule lands on the same committed bytes.
  run = run_deck(deck)
  assert run.result.status is DriverStatus.COMPLETED
  np.testing.assert_array_equal(
    run.driver.owner.accepted_state(block_id).values,
    driver.owner.accepted_state(block_id).values,
  )
  np.testing.assert_array_equal(run.state, driver.owner.accepted_physical().values)


_PLASTICITY_TABLE_BLOCK = _PLASTICITY_ELEMENT_BLOCK.replace(
  "    hard = 1000.0;\n",
  "    hard = 1000.0;\n"
  "    EqPlasStrains = [0.0, 1.0];\n"
  "    Stresses = [250.0, 1250.0];\n",
)

_PLASTICITY_REJECTIONS = {
  "table-hardening": (
    _plasticity_pro(element_block=_PLASTICITY_TABLE_BLOCK),
    {"unsupported-hardening-form"},
  ),
  "power-law-hardening": (
    _plasticity_pro(
      element_block=_PLASTICITY_ELEMENT_BLOCK.replace(
        "    hard = 1000.0;\n",
        "    q = 0.05;\n",
      ),
    ),
    {"unsupported-hardening-form", "missing-material-parameter"},
  ),
  "missing-hardening-slope": (
    _plasticity_pro(
      element_block=_PLASTICITY_ELEMENT_BLOCK.replace("    hard = 1000.0;\n", ""),
    ),
    {"missing-material-parameter"},
  ),
  "extra-material-key": (
    _plasticity_pro(
      element_block=_PLASTICITY_ELEMENT_BLOCK.replace(
        "    hard = 1000.0;\n",
        "    hard = 1000.0;\n    rho = 7850.0;\n",
      ),
    ),
    {"unsupported-material-parameter"},
  ),
  "kinematic-hardening-model": (
    _plasticity_pro(
      element_block=_PLASTICITY_ELEMENT_BLOCK.replace(
        'type = "IsotropicHardeningPlasticity";',
        'type = "IsotropicKinematicHardening";',
      ),
    ),
    {"unsupported-material-model", "unsupported-material-parameter"},
  ),
}


@pytest.mark.parametrize(
  ("pro_text", "expected"),
  [
    pytest.param(text, codes, id=name)
    for name, (text, codes) in _PLASTICITY_REJECTIONS.items()
  ],
)
def test_plasticity_construct_rejections(
  tmp_path: Path,
  pro_text: str,
  expected: set[str],
) -> None:
  with pytest.raises(DeckConversionError) as excinfo:
    _convert_mini(tmp_path, pro_text, _plasticity_dat())
  assert expected == _rejection_codes(excinfo)


def test_table_hardening_rejection_cites_the_m25_scoping(tmp_path: Path) -> None:
  with pytest.raises(DeckConversionError) as excinfo:
    _convert_mini(
      tmp_path,
      _plasticity_pro(element_block=_PLASTICITY_TABLE_BLOCK),
      _plasticity_dat(),
    )
  assert _rejection_codes(excinfo) == {"unsupported-hardening-form"}
  (diagnostic,) = excinfo.value.diagnostics
  assert "M25" in diagnostic.message
  assert "EqPlasStrains" in diagnostic.message


def test_plasticity_on_non_quad8_mesh_rejects_with_coded_arity(tmp_path: Path) -> None:
  dat = _plasticity_dat().replace(_PLASTICITY_ELEMENT, ' 1 "ContElem" 1 2 3;')
  with pytest.raises(DeckConversionError) as excinfo:
    _convert_mini(tmp_path, _plasticity_pro(), dat)
  assert _rejection_codes(excinfo) == {"unsupported-cell-arity"}


# --- finite-strain (total-Lagrangian) decks --------------------------------------

_FINITE_STRAIN_ELEMENT_BLOCK = """ContElem =
{
  type = "FiniteStrainContinuum";
  material =
  {
    type = "PlaneStress";
    E = 100.0;
    nu = 0.3;
  };
};
"""

_FINITE_STRAIN_SOLVER_BLOCK = """solver =
{
  type = "NonlinearSolver";
  tol = 1.0e-10;
  iterMax = 25;
  loadTable = [1.0];
};
"""


def _finite_strain_pro(
  *,
  element_block: str = _FINITE_STRAIN_ELEMENT_BLOCK,
  solver_block: str = _FINITE_STRAIN_SOLVER_BLOCK,
) -> str:
  return _mini_pro(element_block=element_block, solver_block=solver_block)


def test_finite_strain_mini_deck_converts_and_drives(tmp_path: Path) -> None:
  # The drive uses u[3] = 0.0 (not the stock mini deck's v[3] = 0.0): the
  # stock constraint set leaves the rotation about node 0 unconstrained.
  deck = _convert_mini(
    tmp_path,
    _finite_strain_pro(),
    _mini_dat(constraints=" u[0] = 0.0;\n v[0] = 0.0;\n u[3] = 0.0;"),
  )
  (material,) = deck.model.materials
  assert material.model == "plane-stress-saint-venant-kirchhoff"
  parameters = {parameter.name: parameter.value for parameter in material.parameters}
  assert parameters == {"youngs_modulus": 100.0, "poisson_ratio": 0.3}
  (region,) = deck.model.regions
  assert region.formulation == "total-lagrangian-continuum"
  assert ("formulation", "total-lagrangian-continuum") in deck.registry
  assert deck.solver.load_factors == (1.0,)

  run = run_deck(deck)
  assert run.result.status is DriverStatus.COMPLETED
  # A genuinely nonlinear path: more corrections than committed substeps.
  statistics = run.result.statistics
  assert statistics.committed_substep_count == 1
  assert statistics.factorization_reuse_count == 0
  assert statistics.factorization_count > 1


_FINITE_STRAIN_REJECTIONS = {
  "plane-strain-material": (
    _finite_strain_pro(
      element_block=_FINITE_STRAIN_ELEMENT_BLOCK.replace(
        'type = "PlaneStress";',
        'type = "PlaneStrain";',
      ),
    ),
    {"unsupported-material-model"},
  ),
  "plasticity-material": (
    _finite_strain_pro(
      element_block=_FINITE_STRAIN_ELEMENT_BLOCK.replace(
        'type = "PlaneStress";',
        'type = "IsotropicHardeningPlasticity";',
      ),
    ),
    {"unsupported-material-model"},
  ),
  "missing-youngs-modulus": (
    _finite_strain_pro(
      element_block=_FINITE_STRAIN_ELEMENT_BLOCK.replace("    E = 100.0;\n", ""),
    ),
    {"missing-material-parameter"},
  ),
  "extra-material-key": (
    _finite_strain_pro(
      element_block=_FINITE_STRAIN_ELEMENT_BLOCK.replace(
        "    nu = 0.3;\n",
        "    nu = 0.3;\n    rho = 7850.0;\n",
      ),
    ),
    {"unsupported-material-parameter"},
  ),
  "extra-element-key": (
    _finite_strain_pro(
      element_block=_FINITE_STRAIN_ELEMENT_BLOCK.replace(
        '  type = "FiniteStrainContinuum";\n',
        '  type = "FiniteStrainContinuum";\n  rho = 7850.0;\n',
      ),
    ),
    {"unsupported-element-parameter"},
  ),
  "missing-material-block": (
    _finite_strain_pro(
      element_block='ContElem = { type = "FiniteStrainContinuum"; };\n',
    ),
    {"missing-material-parameter"},
  ),
  "linear-solver": (
    _finite_strain_pro(solver_block=_MINI_SOLVER_BLOCK),
    {"incompatible-solver-type"},
  ),
}


@pytest.mark.parametrize(
  ("pro_text", "expected"),
  [
    pytest.param(text, codes, id=name)
    for name, (text, codes) in _FINITE_STRAIN_REJECTIONS.items()
  ],
)
def test_finite_strain_construct_rejections(
  tmp_path: Path,
  pro_text: str,
  expected: set[str],
) -> None:
  with pytest.raises(DeckConversionError) as excinfo:
    _convert_mini(tmp_path, pro_text, _mini_dat())
  assert expected == _rejection_codes(excinfo)


def test_finite_strain_on_non_quad8_mesh_rejects(tmp_path: Path) -> None:
  dat = _mini_dat(elements=' 1 "ContElem" 0 1 2 3;')
  with pytest.raises(DeckConversionError) as excinfo:
    _convert_mini(tmp_path, _finite_strain_pro(), dat)
  assert _rejection_codes(excinfo) == {"unsupported-cell-arity"}


# --- coded rejections ------------------------------------------------------------


def test_rejections_render_source_context(tmp_path: Path) -> None:
  dat = _mini_dat(elements=' 1 "ContElem" 0 4 1 5 2;')
  with pytest.raises(DeckConversionError) as excinfo:
    _convert_mini(tmp_path, _mini_pro(), dat)
  rendered = str(excinfo.value)
  assert "[unsupported-cell-arity]" in rendered
  assert "mini.dat:6" in rendered


def test_missing_pro_file_rejects(tmp_path: Path) -> None:
  with pytest.raises(DeckConversionError) as excinfo:
    read_legacy_deck(tmp_path / "absent.pro")
  assert _rejection_codes(excinfo) == {"deck-input-missing"}


_PRO_REJECTIONS = {
  "include": (
    _mini_pro(extra='include "other.pro";\n'),
    {"unsupported-pro-construct"},
  ),
  "dotted-key": (
    _mini_pro(extra="fem.model = 1;\n"),
    {"unsupported-pro-construct"},
  ),
  "unknown-top-level": (
    _mini_pro(extra="mystery = 42;\n"),
    {"unsupported-pro-construct"},
  ),
  "missing-input": (
    _mini_pro(input_line=""),
    {"deck-input-missing"},
  ),
  "missing-solver": (
    _mini_pro(solver_block=""),
    {"deck-solver-missing"},
  ),
  "pro-syntax": (
    _mini_pro(solver_block='solver { type = "LinearSolver"; };'),
    {"deck-pro-syntax"},
  ),
  "non-finite-value": (
    _mini_pro(
      element_block=_MINI_ELEMENT_BLOCK.replace("E = 1.0e6;", "E = 1e999;"),
    ),
    {"deck-pro-syntax"},
  ),
  "unknown-output-module": (
    _mini_pro(
      extra='outputModules = ["odd"];\nodd = { type = "OddWriter"; };\n',
    ),
    {"unsupported-output-module"},
  ),
  "output-module-without-block": (
    _mini_pro(extra='outputModules = ["odd"];\n'),
    {"unsupported-output-module"},
  ),
  "unsupported-element-type": (
    _mini_pro(
      element_block='ContElem = { type = "Beam"; E = 1.0; };\n',
    ),
    {"unsupported-element-type"},
  ),
  "missing-material-parameter": (
    _mini_pro(element_block=_MINI_ELEMENT_BLOCK.replace("    nu = 0.25;\n", "")),
    {"missing-material-parameter"},
  ),
  "unsupported-material-parameter": (
    _mini_pro(
      element_block=_MINI_ELEMENT_BLOCK.replace(
        "    nu = 0.25;\n",
        "    nu = 0.25;\n    rho = 1.0;\n",
      ),
    ),
    {"unsupported-material-parameter"},
  ),
  "unknown-material-model": (
    _mini_pro(
      element_block=_MINI_ELEMENT_BLOCK.replace(
        'type = "PlaneStress";',
        'type = "MooneyRivlin";',
      ),
    ),
    {"unsupported-material-model"},
  ),
  "isotropic-on-2d-mesh": (
    _mini_pro(
      element_block=_MINI_ELEMENT_BLOCK.replace(
        'type = "PlaneStress";',
        'type = "Isotropic";',
      ),
    ),
    {"incompatible-material-geometry"},
  ),
  "missing-element-block": (
    _mini_pro(
      element_block=_MINI_ELEMENT_BLOCK.replace("ContElem", "OtherElem"),
    ),
    {"missing-element-block", "unsupported-pro-construct"},
  ),
  "truss-missing-area": (
    _mini_pro(
      element_block='ContElem = { type = "Truss"; E = 5e6; };\n',
    ),
    {"missing-element-parameter"},
  ),
  "truss-extra-parameter": (
    _mini_pro(
      element_block='ContElem = { type = "Truss"; E = 5e6; Area = 1.0; rho = 1.0; };\n',
    ),
    {"unsupported-element-parameter"},
  ),
  "truss-wrong-arity": (
    _mini_pro(
      element_block='ContElem = { type = "Truss"; E = 5e6; Area = 1.0; };\n',
    ),
    {"unsupported-cell-arity"},
  ),
  "unknown-solver-type": (
    _mini_pro(solver_block='solver = { type = "WeirdSolver"; maxLam = 1.0; };'),
    {"unsupported-solver-type"},
  ),
  "linear-solver-extra-key": (
    _mini_pro(solver_block='solver = { type = "LinearSolver"; tol = 1.0e-8; };'),
    {"unsupported-solver-parameter"},
  ),
  "nonlinear-solver-unknown-key": (
    _mini_pro(solver_block='solver = { type = "NonlinearSolver"; optiter = 5; };'),
    {"unsupported-solver-parameter"},
  ),
  "unsupported-load-func": (
    _mini_pro(
      solver_block='solver = { type = "NonlinearSolver"; loadFunc = "f"; };',
    ),
    {"unsupported-load-func"},
  ),
  "empty-load-table": (
    _mini_pro(solver_block='solver = { type = "NonlinearSolver"; loadTable = []; };'),
    {"unsupported-load-table"},
  ),
}

_DAT_REJECTIONS = {
  "gmsh-reference": (
    'gmsh = "mesh.msh";\n',
    {"unsupported-gmsh-reference"},
  ),
  "node-group-section": (
    _mini_dat(extra='<NodeGroup name="left">\n 0;\n</NodeGroup>\n'),
    {"unsupported-section"},
  ),
  "named-constraint-table": (
    _mini_dat(constraints_tag='<NodeConstraints name="main">'),
    {"unsupported-section"},
  ),
  "unknown-section": (
    _mini_dat(extra="<Weird>\n 1 2;\n</Weird>\n"),
    {"unsupported-section"},
  ),
  "duplicate-section": (
    _mini_dat(extra="<Nodes>\n 8 2.0 2.0;\n</Nodes>\n"),
    {"unsupported-section"},
  ),
  "statement-outside-section": (
    _mini_dat(extra=" 0 0.0 0.0;\n"),
    {"deck-dat-syntax"},
  ),
  "single-coordinate-node": (
    _mini_dat(nodes=" 0 0.0;"),
    {"unsupported-mesh-rank", "unknown-node-reference"},
  ),
  "unterminated-statement": (
    _mini_dat(forces=" v[2] = 1.0"),
    {"deck-dat-syntax"},
  ),
  "duplicate-node-id": (
    _mini_dat(nodes=_MINI_NODES + "\n 0 9.0 9.0;"),
    {"duplicate-node-id"},
  ),
  "duplicate-cell-id": (
    _mini_dat(
      elements=_MINI_ELEMENTS + '\n 1 "ContElem" 0 4 1 5 2 6 3 7;',
    ),
    {"duplicate-cell-id"},
  ),
  "unknown-node-in-element": (
    _mini_dat(elements=' 1 "ContElem" 0 4 1 5 2 6 3 42;'),
    {"unknown-node-reference"},
  ),
  "unknown-node-in-constraint": (
    _mini_dat(constraints=" u[42] = 0.0;"),
    {"unknown-node-reference"},
  ),
  "inconsistent-node-coordinates": (
    _mini_dat(nodes=_MINI_NODES + "\n 8 2.0 2.0 2.0;"),
    {"inconsistent-node-coordinates"},
  ),
  "three-dimensional-mesh": (
    _mini_dat(
      nodes=(
        " 0 0.0 0.0 0.0; 1 1.0 0.0 0.0; 2 1.0 1.0 0.0; 3 0.0 1.0 0.0;\n"
        " 4 0.5 0.0 0.0; 5 1.0 0.5 0.0; 6 0.5 1.0 0.0; 7 0.0 0.5 0.0;"
      ),
    ),
    {"incompatible-material-geometry"},
  ),
  "five-node-cell": (
    _mini_dat(elements=' 1 "ContElem" 0 4 1 5 2;'),
    {"unsupported-cell-arity"},
  ),
  "mixed-cell-arities": (
    _mini_dat(
      elements=_MINI_ELEMENTS + '\n 2 "ContElem" 0 4 1;',
    ),
    {"unsupported-cell-arity"},
  ),
  "unsupported-dof-type": (
    _mini_dat(constraints=_MINI_CONSTRAINTS + "\n w[1] = 0.0;"),
    {"unsupported-dof-type"},
  ),
  "unknown-dof-type": (
    _mini_dat(constraints=_MINI_CONSTRAINTS + "\n q[1] = 0.0;"),
    {"unsupported-dof-type"},
  ),
  "node-group-reference": (
    _mini_dat(constraints=_MINI_CONSTRAINTS + "\n u[left] = 0.0;"),
    {"unsupported-node-group"},
  ),
  "multi-master-tie": (
    _mini_dat(constraints=_MINI_CONSTRAINTS + "\n u[1] = 0.5 * u[0] + 0.5 * v[3];"),
    {"unsupported-tie-form"},
  ),
  "tie-without-master": (
    _mini_dat(constraints=_MINI_CONSTRAINTS + "\n u[1] = 0.5 * [2];"),
    {"deck-dat-syntax"},
  ),
  "empty-mesh": (
    _mini_dat(nodes="", elements=""),
    {"empty-mesh"},
  ),
  "two-element-groups": (
    _mini_dat(
      elements=_MINI_ELEMENTS + '\n 2 "OtherElem" 0 4 1 5 2 6 3 7;',
    ),
    {"unsupported-element-groups", "missing-element-block"},
  ),
}


@pytest.mark.parametrize(
  ("pro_text", "expected"),
  [
    pytest.param(text, codes, id=name)
    for name, (text, codes) in _PRO_REJECTIONS.items()
  ],
)
def test_pro_construct_rejections(
  tmp_path: Path,
  pro_text: str,
  expected: set[str],
) -> None:
  with pytest.raises(DeckConversionError) as excinfo:
    _convert_mini(tmp_path, pro_text, _mini_dat())
  assert expected <= _rejection_codes(excinfo)


@pytest.mark.parametrize(
  ("dat_text", "expected"),
  [
    pytest.param(text, codes, id=name)
    for name, (text, codes) in _DAT_REJECTIONS.items()
  ],
)
def test_dat_construct_rejections(
  tmp_path: Path,
  dat_text: str,
  expected: set[str],
) -> None:
  with pytest.raises(DeckConversionError) as excinfo:
    _convert_mini(tmp_path, _mini_pro(), dat_text)
  assert expected <= _rejection_codes(excinfo)


# --- affine tie grammar ----------------------------------------------------------


def test_tie_grammar_variants_convert(tmp_path: Path) -> None:
  variants = (
    " u[1] = 2.0 * u[0];",
    " u[1] = u[0];",
    " u[1] = -u[0];",
    " u[1] = 0.001 + 2.0 * u[0];",
    " u[1] = u[0] * 2.0;",
    " u[1] = 1.0e-4 + u[0];",
  )
  for variant in variants:
    deck = _convert_mini(
      tmp_path,
      _mini_pro(),
      _mini_dat(constraints=variant),
    )
    (tie,) = (
      constraint
      for constraint in deck.program.constraints
      if type(constraint) is AffineTieSpec
    )
    assert tie.slave.node_id == 1
    assert tie.master.node_id == 0


def test_tie_offset_and_factor_values(tmp_path: Path) -> None:
  deck = _convert_mini(
    tmp_path,
    _mini_pro(),
    _mini_dat(constraints=" u[1] = 0.001 - 2.5 * u[0];"),
  )
  (tie,) = (
    constraint
    for constraint in deck.program.constraints
    if type(constraint) is AffineTieSpec
  )
  assert tie.factor == pytest.approx(-2.5)
  assert tie.offset.constant == pytest.approx(0.001)
