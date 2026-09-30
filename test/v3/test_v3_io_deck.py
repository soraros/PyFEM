# SPDX-License-Identifier: MIT

"""Legacy-deck converter: subset reading, coded rejections, round-trip oracles.

The round-trip oracles convert every in-subset ``skims/patch_test*`` deck,
compile the emitted specs with the landed compiler unchanged, drive them with
the landed M18 driver, and compare against the landed legacy oracle states
within each skim's ``parity.toml`` tolerances.
"""

from __future__ import annotations

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

from pyfem.v3.driver import DriverStatus, NonlinearStaticDriver, NonlinearStaticSettings
from pyfem.v3.io.legacy_deck import (
  ConvertedDeck,
  DeckConversionError,
  compile_deck,
  read_legacy_deck,
  run_deck,
)
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
)
ROUND_TRIP_SKIMS = LINEAR_SKIMS + NONLINEAR_SKIMS

# code sets expected when converting the out-of-subset skims decks
SKIM_REJECTIONS = {
  "cantilever8": {"unsupported-element-type"},
  "shallow_truss_riks": {
    "unsupported-element-type",
    "unsupported-element-groups",
    "unsupported-solver-type",
  },
}

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


# --- coded rejections ------------------------------------------------------------


@pytest.mark.parametrize("skim_name", sorted(SKIM_REJECTIONS))
def test_out_of_subset_skims_reject_with_coded_diagnostics(skim_name: str) -> None:
  with pytest.raises(DeckConversionError) as excinfo:
    read_legacy_deck(SKIMS / skim_name / "skim.pro")
  assert _rejection_codes(excinfo) == SKIM_REJECTIONS[skim_name]


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
      element_block='ContElem = { type = "Spring"; k = 100.0; };\n',
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
  "riks-solver": (
    _mini_pro(solver_block='solver = { type = "RiksSolver"; maxLam = 1.0; };'),
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
