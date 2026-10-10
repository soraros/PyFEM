# SPDX-License-Identifier: MIT

"""RVE homogenization as a v3 program kind (M75, the FE2-1 mission).

The homogenized response of a periodic cell is observed at the driver as a
named boundary-group reaction reduction, per the M74 survey's build-ready
sketch and generic_core.md:268-273 (observations are solve-free reductions;
a quantity requiring an auxiliary solve comes from an explicit channel —
here the map-level implicit-function-theorem solve, never an operator
channel). This battery pins, per committed substep:

- the homogenized stress: the reduction of the COMMITTED reactions
  (solve-free, zero extra evaluations);
- the homogenized consistent tangent: one IFT column per requested strain
  coordinate, ``dq/de = -(P.T K P)^-1 P.T (K du_bar/de - df_ext/de)``,
  ONE back-substitution on the factorization the converged Newton loop
  last used (the M57 stash/cache/plateau-corner discipline verbatim), with
  the right-hand side assembled from ONE values-only refill of the
  committed tangent and the map's compiled offset-derivative columns —
  the legacy ``globdat.K`` side channel and its 2-of-13-solvers silent
  zero-tangent degradation class are eliminated by construction: a
  substep whose converged tangent is singular fails closed with a coded
  diagnostic instead of reporting an uncertified column;
- the cost contract: per committed substep exactly 3 right-hand-side
  assemblies + 3 back-substitutions + 1 tangent refill + 0
  refactorizations + 0 extra evaluations (3 requested coordinates);
- the primal contract: an unrequested run is bitwise untouched, requested
  runs leave committed states, reactions, schedule decisions, and
  iteration norms bitwise identical (M57's finding: cross-run iteration
  norms carry an ambient 1-ulp floor on some decks, so cross-run float
  pins use rtol 1e-12 while states/counters/structure stay exact);
- the reject path: rejected/failed records never carry the observation
  and stay byte-identical to the plain run's.

AUTHORING SEAM (the M74 scope split): the ``periodic_cell`` authoring
helper is deferred to the next authoring wave, so the periodic-cell spec
declarations are built PROGRAMMATICALLY here — corner prescriptions as
``AffineValueSpec`` columns bound to the named strain coordinates
eps11/eps22/gamma12, periodic tie pairs via the landed
``periodic_ties`` helper, and the boundary-group extraction below ports
legacy ``RVE.getBoundaries``' validation (pairing, alignment, constant
offsets) into the test setup.

Oracles, with the measured deviations every pin cites:

1. closed-form anchor: the homogeneous plane-stress cell's homogenized
   tangent == the material moduli (legacy measured col 0
   [1.0666667e6, 2.6666667e5] exactly; here max abs dev 2.4e-9 on the
   analytically-zero entries, max rel dev 2.2e-15 on 1.07e6 scale);
2. legacy parity: rve_test01's sigma = [7.627e4, 1.691e5, 4.0e3] at
   eps = [0.034, 0.15, 0.01] — the homogeneous cell's response is the
   analytic moduli x strain on any mesh, so the cross-implementation
   deviation is the summation-order class (measured 1.2e-10 abs);
3. FD conviction at elastic/damaged/shear states of the damage cell
   (legacy micro_rve.pro parameters, kappa_c made 10x more ductile so
   the damaged/shear states converge with nonsingular tangents — with
   the legacy kappa_c the only converged damaged branch on this mesh is
   the degenerate fully-damaged equilibrium, which serves the
   fail-closed witness below): elastic 2.8e-11 (h=1e-5), shear 1.3e-8
   (h=1e-8, rel 2.8e-11), damaged loading-axis column 1.6e-10 (h=1e-6).
   The damaged state's off-axis columns sit on the de Vree envelope
   kinks (transverse/shear perturbations unload Gauss points at the
   kappa envelope — one-sided FDs of opposite signs disagree, measured
   7.4e-2/5.4e-1), so the full-matrix FD agreement is pinned at its
   measured h-independent class (2.8e-1) with the loading-axis column
   carrying the 1e-9-class conviction;
4. sympy symbolic leg: a 1-cell quad4 homogeneous cell's boundary
   reduction simplifies EXACTLY to the plane-stress moduli (structural
   conviction of the reduction: corner sharing, engineering-shear
   halves, dx/dy normalization), the map-level IFT identity simplifies
   exactly to zero on a rational toy system, and the symbolic moduli
   densely evaluated at the deck parameters agree with the driver's
   observed anchor tangent;
5. the plateau corner: a zero-advance substep owns no Newton solve, so
   the homogenization factorizes the converged tangent ONCE from the
   shared refill, counted truthfully (first solve not a reuse) — and
   observes the UNLOADING/secant branch (the converged point sits
   exactly at the kappa envelope), convicted by Richardson-extrapolated
   one-sided unloading FD (measured 3.1e-10/3.9e-10/5.6e-6 per column);
   the loading-vs-plateau tangent jump (measured 516.5) is the de Vree
   branch switch witnessed, not an error.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import fields

import numpy as np
import pytest
import sympy as sp

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from pyfem.v3 import authoring
from pyfem.v3.compile.continuum import (
  damage_reference_registry,
  q8_reference_registry,
)
from pyfem.v3.compile.system import compile_system
from pyfem.v3.constraints import compile_constraint_map, periodic_ties
from pyfem.v3.driver import (
  DriverEvaluationError,
  DriverPreparationError,
  DriverStatus,
  NonlinearStaticDriver,
  NonlinearStaticResult,
  NonlinearStaticSettings,
  SubstepStatus,
)
from pyfem.v3.driver.contracts import (
  HomogenizationReduction,
  HomogenizationRequest,
)
from pyfem.v3.model.system import CompiledSystem
from pyfem.v3.spec import MaterialSpec, MeshSpec, SourceContext
from pyfem.v3.spec.program import (
  AffineCoefficientSpec,
  AffineValueSpec,
  DofRef,
  PrescribedDofSpec,
  ProgramCoordinateSpec,
  ProgramCoordinateValue,
  ProgramPoint,
)

# The three macroscopic strain coordinates every deck declares.
_STRAIN = ("eps11", "eps22", "gamma12")

# Elastic deck (legacy rve_test01.pro: plane-stress, E = 1e6, nu = 0.25,
# 0.6 x 0.6 cell; the legacy 3x3 quad4 mesh is a 3x3 quad8 patch here — the
# response of the homogeneous cell is mesh-independent).
_E, _NU = 1.0e6, 0.25
_DX = _DY = 0.6

# Damage deck (legacy micro_rve.pro: PlaneStrainDamage E = 1000, nu = 0.25,
# k = 1, kappa0 = 7.5e-4 on the 1 x 1 cell with a 2x2 mesh). kappa_c is a
# deck parameter: legacy's 2.5e-3 serves the fail-closed and reject-path
# witnesses; the FD legs use the 10x ductile 2.5e-2 (see the module
# docstring).
_DAMAGE_E, _DAMAGE_NU, _DAMAGE_KAPPA0, _DAMAGE_K = 1000.0, 0.25, 7.5e-4, 1.0
_DAMAGE_KAPPAC_DUCTILE = 2.5e-2
_DAMAGE_KAPPAC_LEGACY = 2.5e-3
_DUCTILE = (1.0, 1.0)  # cell size

# Cross-run float pin for iteration-history norms (M57's documented ambient
# 1-ulp floor on some decks; measured 0.0 on this file's decks in-process).
_NORM_RTOL = 1.0e-12


def _affine(*pairs: tuple[str, float]) -> AffineValueSpec:
  return AffineValueSpec(
    coefficients=tuple(AffineCoefficientSpec(name, coef) for name, coef in pairs)
  )


def _prescribed(node: int, component: str, value: AffineValueSpec) -> PrescribedDofSpec:
  return PrescribedDofSpec(
    id=f"corner-{node}-{component}",
    target=DofRef(node_id=node, field_id="displacement", component=component),
    value=value,
    source=SourceContext(source="test.rve.corner"),
  )


def _boundary_groups(
  mesh: MeshSpec,
  dx: float,
  dy: float,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
  """Sorted Left/Right/Top/Bottom node ids with the getBoundaries validation.

  Port of legacy ``RVE.getBoundaries`` (pyfem/models/RVE.py:240-343):
  opposite groups must pair one-to-one in the pairing coordinate with a
  constant offset (the cell's width/height), and corners are the shared
  group endpoints. This is the authoring seam: the deferred
  ``periodic_cell`` helper will own this extraction once the authoring
  wave lands it.
  """
  tolerance = 1.0e-9 * max(dx, dy)
  coordinates = {node.id: node.coordinates for node in mesh.nodes}

  def group(
    predicate: Callable[[tuple[float, float]], bool],
    key: int,
  ) -> tuple[int, ...]:
    ids = tuple(node.id for node in mesh.nodes if predicate(node.coordinates))
    return tuple(sorted(ids, key=lambda node_id: coordinates[node_id][key]))

  left = group(lambda c: abs(c[0]) < tolerance, 1)
  right = group(lambda c: abs(c[0] - dx) < tolerance, 1)
  bottom = group(lambda c: abs(c[1]) < tolerance, 0)
  top = group(lambda c: abs(c[1] - dy) < tolerance, 0)
  assert len(left) == len(right) >= 2
  assert len(bottom) == len(top) >= 2
  for node_left, node_right in zip(left, right, strict=True):
    assert abs(coordinates[node_right][1] - coordinates[node_left][1]) < tolerance
    x_offset = coordinates[node_right][0] - coordinates[node_left][0]
    assert abs(x_offset - dx) < tolerance
  for node_bottom, node_top in zip(bottom, top, strict=True):
    assert abs(coordinates[node_top][0] - coordinates[node_bottom][0]) < tolerance
    y_offset = coordinates[node_top][1] - coordinates[node_bottom][1]
    assert abs(y_offset - dy) < tolerance
  assert left[0] == bottom[0] and right[0] == bottom[-1]
  assert left[-1] == top[0] and right[-1] == top[-1]
  return left, right, bottom, top


def _periodic_constraints(
  left: tuple[int, ...],
  right: tuple[int, ...],
  bottom: tuple[int, ...],
  top: tuple[int, ...],
  dx: float,
  dy: float,
) -> tuple:
  """Corner prescriptions as affine strain columns + periodic tie pairs.

  The legacy ``applyPeriodicBC`` structure (pyfem/models/RVE.py:348-433):
  the bottom-left corner is fixed, the other three corners carry the
  affine macroscopic-strain columns, and interior boundary nodes tie
  across with the same strain-column offsets (u_right = u_left + u12,
  u_top = u_bottom + u14).
  """
  bottom_left, bottom_right, top_left, top_right = left[0], right[0], top[0], top[-1]
  constraints = [
    _prescribed(bottom_left, "x", AffineValueSpec()),
    _prescribed(bottom_left, "y", AffineValueSpec()),
    _prescribed(bottom_right, "x", _affine(("eps11", dx))),
    _prescribed(bottom_right, "y", _affine(("gamma12", 0.5 * dx))),
    _prescribed(top_left, "x", _affine(("gamma12", 0.5 * dy))),
    _prescribed(top_left, "y", _affine(("eps22", dy))),
    _prescribed(top_right, "x", _affine(("eps11", dx), ("gamma12", 0.5 * dy))),
    _prescribed(top_right, "y", _affine(("gamma12", 0.5 * dx), ("eps22", dy))),
  ]
  constraints += periodic_ties(
    primary_node_ids=left[1:-1],
    image_node_ids=right[1:-1],
    field_id="displacement",
    components=("x", "y"),
    offsets=(_affine(("eps11", dx)), _affine(("gamma12", 0.5 * dx))),
    id_prefix="tie-lr",
  )
  constraints += periodic_ties(
    primary_node_ids=bottom[1:-1],
    image_node_ids=top[1:-1],
    field_id="displacement",
    components=("x", "y"),
    offsets=(_affine(("gamma12", 0.5 * dy)), _affine(("eps22", dy))),
    id_prefix="tie-bt",
  )
  return tuple(constraints)


def _group_dofs(
  system: CompiledSystem,
  node_ids: tuple[int, ...],
  component: str,
) -> tuple[int, ...]:
  """Resolve boundary-group node ids to full-space DOF indices."""
  (space,) = system.spaces
  supports = {block.block_id: block for block in system.point_blocks}
  node_index = {
    node_id: index
    for index, node_id in enumerate(supports[space.support_block_id].entity_ids)
  }
  component_index = space.components.index(component)
  coefficient_map = space.coefficient_map.values
  return tuple(
    int(coefficient_map[node_index[node], component_index]) for node in node_ids
  )


def _rve_request(
  system: CompiledSystem,
  groups: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]],
  dx: float,
  dy: float,
) -> HomogenizationRequest:
  """The legacy homogenization reduction (RVE.py:151-168)."""
  _, right, _, top = groups
  return HomogenizationRequest(
    strain_coordinates=_STRAIN,
    reductions=(
      HomogenizationReduction("sigma11", _group_dofs(system, right, "x"), 1.0 / dy),
      HomogenizationReduction("sigma22", _group_dofs(system, top, "y"), 1.0 / dx),
      HomogenizationReduction("sigma12", _group_dofs(system, top, "x"), 1.0 / dy),
    ),
  )


def _rve_deck(
  nx: int,
  ny: int,
  dx: float,
  dy: float,
  material: MaterialSpec,
  registry: dict,
  settings: NonlinearStaticSettings | None = None,
  *,
  default_tolerance: float = 1.0e-12,
) -> tuple[NonlinearStaticDriver, HomogenizationRequest]:
  mesh = authoring.quad8_patch(nx, ny, width=dx, height=dy)
  model = authoring.small_strain_continuum(mesh, material=material)
  system = compile_system(model, registry)
  groups = _boundary_groups(mesh, dx, dy)
  coordinate_map = compile_constraint_map(
    system,
    constraints=_periodic_constraints(*groups, dx, dy),
    coordinates=tuple(
      ProgramCoordinateSpec(name=name, kind="load") for name in _STRAIN
    ),
  )
  driver = NonlinearStaticDriver(
    system,
    coordinate_map,
    (),
    settings or NonlinearStaticSettings(tolerance=default_tolerance),
  )
  return driver, _rve_request(system, groups, dx, dy)


def _elastic_deck(
  settings: NonlinearStaticSettings | None = None,
) -> tuple[NonlinearStaticDriver, HomogenizationRequest]:
  # The elastic deck runs at the default tolerance: with no external loads
  # the convergence measure is absolute, and the 1e6-scale stiffness floors
  # the reduced residual near 1e-10, so 1e-12 never trips (it would force
  # budget exhaustion on a converged trajectory).
  return _rve_deck(
    3,
    3,
    _DX,
    _DY,
    authoring.linear_elastic(_E, _NU),
    q8_reference_registry(),
    settings,
    default_tolerance=1.0e-10,
  )


def _damage_deck(
  kappa_c: float = _DAMAGE_KAPPAC_DUCTILE,
  settings: NonlinearStaticSettings | None = None,
) -> tuple[NonlinearStaticDriver, HomogenizationRequest]:
  dx, dy = _DUCTILE
  return _rve_deck(
    2,
    2,
    dx,
    dy,
    authoring.damage(_DAMAGE_E, _DAMAGE_NU, _DAMAGE_KAPPA0, kappa_c, _DAMAGE_K),
    damage_reference_registry(),
    settings,
  )


def _point(eps11: float, eps22: float, gamma12: float) -> ProgramPoint:
  return ProgramPoint(
    (
      ProgramCoordinateValue("eps11", float(eps11)),
      ProgramCoordinateValue("eps22", float(eps22)),
      ProgramCoordinateValue("gamma12", float(gamma12)),
    )
  )


def _run(
  driver: NonlinearStaticDriver,
  strains: tuple[tuple[float, ...], ...],
  request: HomogenizationRequest | None = None,
  base: tuple[float, ...] = (0.0, 0.0, 0.0),
) -> NonlinearStaticResult:
  points = [_point(*base)] + [_point(*strain) for strain in strains]
  return driver.run(
    base_point=points[0],
    target_points=tuple(points[1:]),
    homogenization=request,
  )


def _ramp(strain: tuple[float, ...], step: float) -> list[tuple[float, ...]]:
  count = max(1, round(max(strain) / step))
  return [tuple(float(c) * (k + 1) / count for c in strain) for k in range(count)]


def _committed(result: NonlinearStaticResult) -> list:
  return [
    record for record in result.records if record.status is SubstepStatus.COMMITTED
  ]


def _fd_stress_columns(
  prefix: list[tuple[float, ...]],
  strain: tuple[float, ...],
  h: float,
) -> np.ndarray:
  """Central FD of the last-increment stress map: shared entering state."""
  columns = []
  for axis in range(3):
    values = []
    for sign in (1.0, -1.0):
      perturbed = list(strain)
      perturbed[axis] += sign * h
      values.append(_stress_at(prefix, tuple(perturbed)))
    columns.append((values[0] - values[1]) / (2.0 * h))
  return np.column_stack(columns)


def _stress_at(
  prefix: list[tuple[float, ...]],
  final: tuple[float, ...],
) -> np.ndarray:
  driver, request = _damage_deck()
  result = _run(driver, tuple(prefix) + (tuple(float(v) for v in final),), request)
  assert result.status is DriverStatus.COMPLETED
  return result.records[-1].observation.homogenization.stress.values


def _plane_stress_moduli(youngs_modulus: float, poisson_ratio: float) -> np.ndarray:
  factor = youngs_modulus / (1.0 - poisson_ratio**2)
  return np.array(
    [
      [factor, poisson_ratio * factor, 0.0],
      [poisson_ratio * factor, factor, 0.0],
      [0.0, 0.0, youngs_modulus / (2.0 * (1.0 + poisson_ratio))],
    ]
  )


def _plane_strain_moduli(youngs_modulus: float, poisson_ratio: float) -> np.ndarray:
  factor = youngs_modulus / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
  return factor * np.array(
    [
      [1.0 - poisson_ratio, poisson_ratio, 0.0],
      [poisson_ratio, 1.0 - poisson_ratio, 0.0],
      [0.0, 0.0, (1.0 - 2.0 * poisson_ratio) / 2.0],
    ]
  )


# --- request-time validation ---------------------------------------------------


def test_homogenization_request_is_validated_before_any_substep() -> None:
  driver, request = _elastic_deck()
  strains = ((0.034, 0.15, 0.01),)
  reductions = request.reductions

  def run_with(homogenization: HomogenizationRequest) -> None:
    _run(driver, strains, homogenization)

  # Contract violations on the request shape itself remain TypeErrors.
  with pytest.raises(TypeError, match="exact HomogenizationRequest or None"):
    _run(driver, strains, request="eps11")  # type: ignore[arg-type]
  with pytest.raises(TypeError, match="non-empty exact tuple"):
    run_with(HomogenizationRequest("eps11", reductions))  # type: ignore[arg-type]
  with pytest.raises(TypeError, match="non-empty exact tuple"):
    run_with(HomogenizationRequest((), reductions))
  with pytest.raises(TypeError, match="non-empty exact strings"):
    run_with(HomogenizationRequest((5,), reductions))  # type: ignore[arg-type]
  with pytest.raises(TypeError, match="non-empty exact tuple"):
    run_with(HomogenizationRequest(_STRAIN, ()))
  with pytest.raises(TypeError, match="exact HomogenizationReduction values"):
    run_with(HomogenizationRequest(_STRAIN, ("sigma11",)))  # type: ignore[arg-type]
  bad_component = HomogenizationRequest(
    _STRAIN,
    (HomogenizationReduction(5, (0,), 1.0),),  # type: ignore[arg-type]
  )
  with pytest.raises(TypeError, match="non-empty exact strings"):
    run_with(bad_component)
  with pytest.raises(TypeError, match="non-empty exact tuple"):
    run_with(HomogenizationRequest(_STRAIN, (HomogenizationReduction("s", (), 1.0),)))
  bad_dof = HomogenizationRequest(_STRAIN, (HomogenizationReduction("s", (0.5,), 1.0),))
  with pytest.raises(TypeError, match="exact integers"):
    run_with(bad_dof)
  bad_scale = HomogenizationRequest(
    _STRAIN,
    (HomogenizationReduction("s", (0,), "x"),),  # type: ignore[arg-type]
  )
  with pytest.raises(TypeError, match="exact int or float numbers"):
    run_with(bad_scale)
  # A coordinate the map does not declare, a duplicated coordinate or
  # component, an out-of-range or duplicated DOF, and a non-finite scale
  # can never assemble an honest observation: coded diagnostics.
  with pytest.raises(DriverPreparationError, match="unknown-homogenization-coordinate"):
    run_with(HomogenizationRequest(("eps33",), reductions))
  duplicate_coordinate = "duplicate-homogenization-coordinate"
  duplicate_component = "duplicate-homogenization-component"
  with pytest.raises(DriverPreparationError, match=duplicate_coordinate):
    run_with(HomogenizationRequest(("eps11", "eps11"), reductions))
  with pytest.raises(DriverPreparationError, match=duplicate_component):
    run_with(HomogenizationRequest(_STRAIN, reductions + reductions[:1]))
  full_dof_count = driver.plan.full_dof_count
  out_of_range = HomogenizationRequest(
    _STRAIN, (HomogenizationReduction("s", (full_dof_count,), 1.0),)
  )
  with pytest.raises(DriverPreparationError, match="homogenization-dof-out-of-range"):
    run_with(out_of_range)
  negative = HomogenizationRequest(_STRAIN, (HomogenizationReduction("s", (-1,), 1.0),))
  with pytest.raises(DriverPreparationError, match="homogenization-dof-out-of-range"):
    run_with(negative)
  duplicated = HomogenizationRequest(
    _STRAIN, (HomogenizationReduction("s", (0, 0), 1.0),)
  )
  with pytest.raises(DriverPreparationError, match="duplicate-homogenization-dof"):
    run_with(duplicated)
  infinite = HomogenizationRequest(
    _STRAIN, (HomogenizationReduction("s", (0,), float("inf")),)
  )
  with pytest.raises(DriverPreparationError, match="non-finite-homogenization-scale"):
    run_with(infinite)
  # Every rejection happened before the first substep: no evaluation, no
  # state transition.
  statistics = driver.statistics
  for field in fields(statistics):
    assert getattr(statistics, field.name) == 0
  assert driver.owner.generation.ordinal == 0


# --- closed-form anchor and legacy parity (oracles 1 and 2) ---------------------


def test_homogeneous_cell_tangent_is_the_material_matrix() -> None:
  driver, request = _elastic_deck()
  result = _run(driver, ((0.034, 0.15, 0.01),), request)
  assert result.status is DriverStatus.COMPLETED
  (record,) = result.records
  observation = record.observation.homogenization
  assert observation.component_ids == ("sigma11", "sigma22", "sigma12")
  assert observation.strain_coordinates == _STRAIN
  moduli = _plane_stress_moduli(_E, _NU)
  # Measured on this deck: max abs dev 2.4e-9 (analytically-zero entries,
  # solve/reduction summation order), max rel dev 2.2e-15; pins carry
  # ~400x headroom over the observation.
  np.testing.assert_allclose(
    observation.tangent.values, moduli, rtol=1.0e-12, atol=1.0e-6
  )
  sigma = moduli @ np.array([0.034, 0.15, 0.01])
  # Measured max abs dev 1.2e-10 on 1.7e5 scale (rel 7e-16).
  np.testing.assert_allclose(
    observation.stress.values, sigma, rtol=1.0e-12, atol=1.0e-6
  )


def test_homogeneous_cell_matches_legacy_rve_test01() -> None:
  """Legacy parity: rve_test01 sigma = [7.627e4, 1.691e5, 4.0e3]."""
  driver, request = _elastic_deck()
  result = _run(driver, ((0.034, 0.15, 0.01),), request)
  assert result.status is DriverStatus.COMPLETED
  observation = result.records[-1].observation.homogenization
  # Legacy's report prints four significant digits (RVE.py:224-232); the
  # homogeneous cell's response is the analytic moduli x strain on any
  # mesh, so the parity pin is the analytic value (legacy measured it
  # exactly) plus consistency with legacy's printed values within their
  # print precision. Measured cross-implementation deviation 1.2e-10 abs.
  sigma = _plane_stress_moduli(_E, _NU) @ np.array([0.034, 0.15, 0.01])
  np.testing.assert_allclose(
    observation.stress.values, sigma, rtol=1.0e-12, atol=1.0e-6
  )
  printed = np.array([7.627e4, 1.691e5, 4.0e3])
  # Half-units of legacy's last printed digit: 7.627e4 -> 5, 1.691e5 -> 50,
  # 4.0e3 -> 50; the observed values sit 3.3 / 33.3 / 0.0 from the prints.
  deviation = np.abs(observation.stress.values - printed)
  assert np.all(deviation <= np.array([5.0, 50.0, 50.0]))
  tangent = observation.tangent.values
  # Legacy measured the tangent column 0 as the analytic moduli exactly:
  # [1.0666667e6, 2.6666667e5, ~0] (M74 note 2/4).
  np.testing.assert_allclose(
    tangent[:, 0],
    np.array([1.0666667e6, 2.6666667e5, 0.0]),
    rtol=1.0e-7,
    atol=1.0e-2,
  )


def test_reduction_over_free_dofs_contributes_exact_zeros() -> None:
  """The reduction reads the reaction OBSERVATION: free DOFs are exact zeros."""
  driver, request = _elastic_deck()
  constrained = set(driver._map.constrained_dofs.values.tolist())
  full_dof_count = driver.plan.full_dof_count
  free_dof = next(dof for dof in range(full_dof_count) if dof not in constrained)
  reductions = request.reductions + (HomogenizationReduction("free", (free_dof,), 1.0),)
  result = _run(
    driver,
    ((0.034, 0.15, 0.01),),
    HomogenizationRequest(_STRAIN, reductions),
  )
  assert result.status is DriverStatus.COMPLETED
  observation = result.records[-1].observation.homogenization
  index = observation.component_ids.index("free")
  assert observation.stress.values[index] == 0.0
  assert np.array_equal(observation.tangent.values[index], np.zeros(3))


# --- FD conviction (oracle 3) ---------------------------------------------------


def test_elastic_state_tangent_is_fd_exact() -> None:
  strain = (5.0e-4, 0.0, 0.0)
  path = _ramp(strain, 1.0e-4)
  prefix, final = path[:-1], path[-1]
  driver, request = _damage_deck()
  result = _run(driver, tuple(prefix) + (final,), request)
  assert result.status is DriverStatus.COMPLETED
  observation = result.records[-1].observation.homogenization
  moduli = _plane_strain_moduli(_DAMAGE_E, _DAMAGE_NU)
  # Below the onset the damage tangent is exactly De (kappa = 0); measured
  # max abs dev 2.3e-12 (zero entries), stress [0.6, 0.2, 0] to 1.2e-16.
  np.testing.assert_allclose(observation.tangent.values, moduli, rtol=0.0, atol=1.0e-10)
  np.testing.assert_allclose(
    observation.stress.values,
    moduli @ np.array(strain),
    rtol=0.0,
    atol=1.0e-12,
  )
  # Central FD of the one-step map from the shared entering state, h-table
  # 1e-5/1e-6/1e-7: measured 2.8e-11 / 1.0e-10 / 1.6e-9 (roundoff class).
  fd = _fd_stress_columns(prefix, strain, 1.0e-5)
  np.testing.assert_allclose(observation.tangent.values, fd, rtol=0.0, atol=1.0e-9)


def test_damaged_state_tangent_fd_conviction_and_witnesses() -> None:
  strain = (2.0e-3, 0.0, 0.0)
  path = _ramp(strain, 1.0e-4)
  prefix, final = path[:-1], path[-1]
  driver, request = _damage_deck()
  result = _run(driver, tuple(prefix) + (final,), request)
  assert result.status is DriverStatus.COMPLETED
  observation = result.records[-1].observation.homogenization
  tangent = observation.tangent.values
  scale = float(np.max(np.abs(tangent)))
  # The damaged tangent is the nonsymmetric softening rank-1 class (the
  # de Vree algorithmic tangent; legacy measured |C - C.T| = 248.6 on its
  # own deck/kappa_c): witnessed, not pinned bitwise. Measured nonsym 85.7.
  assert float(np.max(np.abs(tangent - tangent.T))) > 1.0e-2 * scale
  assert tangent[0, 0] < 0.0 < tangent[1, 1]
  # The state ratcheted past the onset: kappa > kappa0 somewhere.
  (block_id,) = driver.owner.block_ids
  kappa = driver.owner.accepted_state(block_id).values.reshape(-1)
  assert float(kappa.max()) > _DAMAGE_KAPPA0
  # The loading-axis column (eps11, the monotone loading direction) is
  # FD-exact: central FD at h=1e-6 measures max abs dev 1.2e-10-class
  # (one-sided h=1e-6: 1.6e-10); pinned with 60x headroom.
  fd = _fd_stress_columns(prefix, strain, 1.0e-6)
  np.testing.assert_allclose(tangent[:, 0], fd[:, 0], rtol=0.0, atol=1.0e-8)
  # The full matrix agrees with central FD at the measured h-independent
  # 2.8e-1 class: the off-axis columns sit on the de Vree envelope kinks
  # (transverse/shear perturbations unload Gauss points at the kappa
  # envelope — one-sided FDs of opposite signs disagree, measured 7.4e-2
  # and 5.4e-1), so central FD straddles there while the observed column
  # is the algorithmic loading-branch derivative a Newton consumer needs.
  assert float(np.max(np.abs(fd - tangent))) <= 0.5


def test_shear_state_tangent_fd_conviction() -> None:
  strain = (0.0, 0.0, 1.5e-3)
  path = _ramp(strain, 1.0e-4)
  prefix, final = path[:-1], path[-1]
  driver, request = _damage_deck()
  result = _run(driver, tuple(prefix) + (final,), request)
  assert result.status is DriverStatus.COMPLETED
  observation = result.records[-1].observation.homogenization
  tangent = observation.tangent.values
  (block_id,) = driver.owner.block_ids
  kappa = driver.owner.accepted_state(block_id).values.reshape(-1)
  assert float(kappa.max()) > _DAMAGE_KAPPA0
  # The shear map is smooth in every perturbation direction (the de Vree
  # equivalent strain grows at second order in eps11/eps22 and linearly
  # in |gamma12|), so central FD convicts the whole matrix: h-table
  # 1e-5..1e-8 measured 1.2e-2 / 1.2e-4 / 1.2e-6 / 1.3e-8 (O(h^2)
  # truncation down to the 1.3e-8 roundoff floor, rel 2.8e-11).
  fd = _fd_stress_columns(prefix, strain, 1.0e-8)
  np.testing.assert_allclose(tangent, fd, rtol=0.0, atol=1.0e-6)


# --- sympy symbolic leg (oracle 4) ----------------------------------------------


def _symbolic_cell_tangent() -> tuple[sp.Matrix, sp.Matrix, sp.Matrix, tuple]:
  """The 1-cell quad4 homogeneous periodic cell, fully symbolic."""
  youngs, nu, dx, dy = sp.symbols("E nu dx dy", positive=True)
  e11, e22, g12 = sp.symbols("e11 e22 g12")
  c = youngs / (1 - nu**2) * sp.Matrix([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
  # Bilinear quad4 on [0, dx] x [0, dy], nodes BL, BR, TR, TL; exact 2x2
  # Gauss integration of B.T C B (the integrand is biquadratic).
  signs = ((-1, -1), (1, -1), (1, 1), (-1, 1))
  xi, eta = sp.symbols("xi eta")
  shape = tuple((1 + s[0] * xi) * (1 + s[1] * eta) / 4 for s in signs)
  jacobian = sp.diag(dx / 2, dy / 2)
  det_j = sp.simplify(sp.det(jacobian))
  inv_j = jacobian.inv()
  stiffness = sp.zeros(8)
  gauss_points = ((g1, g2) for g1 in (-1, 1) for g2 in (-1, 1))
  for point in gauss_points:
    gauss = (point[0] / sp.sqrt(3), point[1] / sp.sqrt(3))
    binding = dict(zip((xi, eta), gauss, strict=True))
    grads = []
    for n in shape:
      d_ref = sp.Matrix([sp.diff(n, xi), sp.diff(n, eta)]).subs(binding)
      grads.append(inv_j * d_ref)
    b = sp.zeros(3, 8)
    for i, grad in enumerate(grads):
      b[0, 2 * i] = grad[0]
      b[1, 2 * i + 1] = grad[1]
      b[2, 2 * i] = grad[1]
      b[2, 2 * i + 1] = grad[0]
    stiffness += b.T * c * b * det_j
  # Corner prescriptions under the macroscopic strain (legacy's u12/u14).
  u_bar = sp.Matrix(
    [
      0,
      0,
      dx * e11,
      dx * g12 / 2,
      dx * e11 + dy * g12 / 2,
      dx * g12 / 2 + dy * e22,
      dy * g12 / 2,
      dy * e22,
    ]
  )
  reactions = stiffness * u_bar
  # Legacy's boundary reduction: Right = {BR, TR}, Top = {TL, TR}, with
  # sigma12 normalized by dy (pyfem/models/RVE.py:166) — exact for the
  # square cells of every legacy deck; the Hill-Mandel-consistent
  # normalization of the top-boundary x-traction is dx instead.
  sigma_legacy = sp.Matrix(
    [
      (reactions[2] + reactions[4]) / dy,
      (reactions[5] + reactions[7]) / dx,
      (reactions[4] + reactions[6]) / dy,
    ]
  )
  sigma_consistent = sp.Matrix(
    [
      (reactions[2] + reactions[4]) / dy,
      (reactions[5] + reactions[7]) / dx,
      (reactions[4] + reactions[6]) / dx,
    ]
  )
  strains = (e11, e22, g12)
  return (
    sigma_legacy.jacobian(strains),
    sigma_consistent.jacobian(strains),
    c,
    (youngs, nu, dx, dy),
  )


def test_sympy_symbolic_leg() -> None:
  tangent_legacy, tangent_consistent, moduli, parameters = _symbolic_cell_tangent()
  # S1a: the Hill-Mandel-consistent boundary reduction of the homogeneous
  # 1-cell cell simplifies EXACTLY to the material moduli for a general
  # rectangle — the reduction (corner sharing, engineering-shear halves,
  # normalization) is structurally exact.
  assert sp.simplify(tangent_consistent - moduli) == sp.zeros(3)
  # S1b: legacy's sigma12 = (sum of top x-reactions)/dy reproduces the
  # moduli only on square cells: the symbolic difference is exactly the
  # rectangular-cell residual E (dx - dy) / (2 dy (nu + 1)) in the shear
  # entry — the square-cell-masked legacy normalization, pinned
  # structurally (finding filed; every legacy deck is square, so the
  # parity oracles are unaffected).
  youngs, nu, dx, dy = parameters
  residual = sp.zeros(3)
  residual[2, 2] = youngs * (dx - dy) / (2 * dy * (nu + 1))
  assert sp.simplify(tangent_legacy - moduli - residual) == sp.zeros(3)

  # S2: the map-level IFT identity on a rational toy system: with the
  # constrained equilibrium solved symbolically, d(sigma)/d(eps) equals
  # the coded column S.T K (P dq + v) with dq = -(P.T K P)^-1 P.T K v.
  eps = sp.symbols("eps")
  k_full = sp.Matrix([[4, 1, 0, 1], [1, 3, 1, 0], [0, 1, 5, 1], [1, 0, 1, 6]])
  prolongation = sp.Matrix([[1, 0], [0, 1], [1, 1], [0, 0]])
  offset = sp.Matrix([0, 0, 2, 1])
  reduction = sp.Matrix([[0, 0, 1, 0], [0, 0, 0, 2]])
  q1, q2 = sp.symbols("q1 q2")
  field = prolongation * sp.Matrix([q1, q2]) + offset * eps
  equilibrium = prolongation.T * k_full * field
  (solution,) = sp.solve(list(equilibrium), (q1, q2), dict=True)
  sigma_map = (
    reduction
    * k_full
    * (prolongation * sp.Matrix([solution[q1], solution[q2]]) + offset * eps)
  )
  direct = sigma_map.jacobian((eps,))
  reduced_tangent = prolongation.T * k_full * prolongation
  dq = -(reduced_tangent.inv()) * (prolongation.T * k_full * offset)
  coded = reduction * k_full * (prolongation * dq + offset)
  assert sp.simplify(direct - coded) == sp.zeros(2, 1)

  # S3: the symbolic moduli densely evaluated at the elastic deck's
  # parameters agree with the driver's observed anchor tangent (the
  # cross-code agreement; pin class of the anchor leg). The deck is
  # square, so the legacy and consistent formulas coincide there.
  evaluated = np.array(
    tangent_consistent.subs(
      {youngs: _E, nu: _NU, dx: _DX, dy: _DY},
      simultaneous=True,
    ),
    dtype=np.float64,
  )
  driver, request = _elastic_deck()
  result = _run(driver, ((0.034, 0.15, 0.01),), request)
  assert result.status is DriverStatus.COMPLETED
  observed = result.records[-1].observation.homogenization.tangent.values
  np.testing.assert_allclose(observed, evaluated, rtol=1.0e-12, atol=1.0e-6)


# --- cost contract (oracle 5) ---------------------------------------------------


def _statistics_delta(
  plain: NonlinearStaticResult,
  requested: NonlinearStaticResult,
) -> dict:
  return {
    field.name: getattr(requested.statistics, field.name)
    - getattr(plain.statistics, field.name)
    for field in fields(plain.statistics)
  }


def test_cost_pins_constant_tangent_deck() -> None:
  strains = ((0.02, 0.0, 0.0), (0.034, 0.15, 0.01))
  plain_driver, _ = _elastic_deck()
  plain = _run(plain_driver, strains)
  requested_driver, request = _elastic_deck()
  requested = _run(requested_driver, strains, request)
  assert plain.status is requested.status is DriverStatus.COMPLETED
  committed = len(_committed(requested))
  assert committed == 2
  delta = _statistics_delta(plain, requested)
  # Per committed substep: 3 rhs assemblies + 3 back-substitutions + 1
  # tangent refill, ZERO refactorizations, ZERO extra evaluations (the
  # derivative source is the map, never an operator channel).
  assert delta["evaluation_count"] == 0
  assert delta["residual_assembly_count"] == 3 * committed
  assert delta["tangent_refill_count"] == committed
  assert delta["factorization_count"] == 0
  assert delta["factorization_reuse_count"] == 3 * committed
  assert delta["linear_solve_count"] == 3 * committed
  assert delta["cutback_count"] == 0
  # The primal trajectory is untouched: schedule structure, committed
  # states, and reactions bitwise; the unrequested records carry no
  # homogenization observation.
  assert [r.status for r in plain.records] == [r.status for r in requested.records]
  assert [r.progress for r in plain.records] == [r.progress for r in requested.records]
  assert np.array_equal(
    plain_driver.owner.accepted_physical().values,
    requested_driver.owner.accepted_physical().values,
  )
  for first, second in zip(plain.records, requested.records, strict=True):
    assert np.array_equal(
      first.observation.reactions.values,
      second.observation.reactions.values,
    )
    assert first.observation.homogenization is None
    assert second.observation.homogenization is not None


def test_cost_pins_nonconstant_tangent_deck() -> None:
  # The proven 20-substep ramp (the damaged FD leg's path): the one-step
  # schedule's iteration-1 overshoot drives the softening deck onto the
  # degenerate branch, so the cost leg ramps.
  strains = tuple(_ramp((2.0e-3, 0.0, 0.0), 1.0e-4))
  plain_driver, _ = _damage_deck()
  plain = _run(plain_driver, strains)
  requested_driver, request = _damage_deck()
  requested = _run(requested_driver, strains, request)
  assert plain.status is requested.status is DriverStatus.COMPLETED
  committed = len(_committed(requested))
  delta = _statistics_delta(plain, requested)
  # The stash discipline factorizes nothing extra on the nonconstant
  # deck either, and the solve side needs no operator re-evaluation.
  assert delta["evaluation_count"] == 0
  assert delta["factorization_count"] == 0
  assert delta["residual_assembly_count"] == 3 * committed
  assert delta["tangent_refill_count"] == committed
  assert delta["linear_solve_count"] == 3 * committed
  assert delta["factorization_reuse_count"] == 3 * committed
  assert delta["cutback_count"] == 0
  assert np.array_equal(
    plain_driver.owner.accepted_physical().values,
    requested_driver.owner.accepted_physical().values,
  )
  (block_id,) = plain_driver.owner.block_ids
  plain_states = plain_driver.owner.accepted_state(block_id).values
  requested_states = requested_driver.owner.accepted_state(block_id).values
  assert np.array_equal(plain_states, requested_states)


# --- plateau corner and reject path (oracle 6, M57 discipline) -------------------


def test_plateau_corner_factorizes_once_and_observes_the_unloading_branch() -> None:
  strain = (2.0e-3, 0.0, 0.0)
  path = _ramp(strain, 1.0e-4)
  driver, request = _damage_deck()
  first = _run(driver, tuple(path), request)
  assert first.status is DriverStatus.COMPLETED
  # A second run targeting the ALREADY committed point: the substep enters
  # converged (the entering state satisfies the point), owns no Newton
  # solve, and so owns no converged factorization — the M57 plateau corner.
  second = _run(driver, (path[-1],), request, base=path[-1])
  assert second.status is DriverStatus.COMPLETED
  (plateau,) = second.records
  assert plateau.status is SubstepStatus.COMMITTED
  assert len(plateau.iterations) == 1
  assert plateau.iterations[0].increment_norm is None
  stats_first, stats_second = first.statistics, second.statistics
  delta = {
    field.name: getattr(stats_second, field.name) - getattr(stats_first, field.name)
    for field in fields(stats_first)
  }
  # One primal evaluation for the plateau substep; the homogenization
  # factorizes the converged tangent ONCE from the shared refill (counted
  # truthfully), and the first of the 3 solves is not a reuse: fact +1,
  # reuse +2 = solves +3 — the M57 corner rule verbatim.
  assert delta["evaluation_count"] == 1
  assert delta["residual_assembly_count"] == 4
  assert delta["tangent_refill_count"] == 1
  assert delta["factorization_count"] == 1
  assert delta["factorization_reuse_count"] == 2
  assert delta["linear_solve_count"] == 3
  assert delta["committed_substep_count"] == 1
  previous = first.records[-1].observation.homogenization
  observed = plateau.observation.homogenization
  # Same committed point: the stress observation is bitwise identical.
  assert np.array_equal(previous.stress.values, observed.stress.values)
  # The tangent is the UNLOADING/secant branch: the plateau's converged
  # evaluation sits exactly at the kappa envelope, so the material takes
  # the unloading branch — the de Vree branch semantics, witnessed by the
  # jump against the loading substep's tangent (measured 516.5).
  jump = float(np.max(np.abs(previous.tangent.values - observed.tangent.values)))
  assert jump > 1.0e2

  # Convicted by Richardson-extrapolated one-sided unloading FD (the
  # perturbed substep steps DOWN from the plateau's committed state):
  # measured 3.1e-10 / 3.9e-10 / 5.6e-6 per column; pinned at 1e-4.
  def unloading_column(axis: int, h: float) -> np.ndarray:
    perturbed = list(strain)
    perturbed[axis] -= h
    value = _stress_at(list(path) + [path[-1]], tuple(perturbed))
    return (observed.stress.values - value) / h

  richardson = np.column_stack(
    [
      2.0 * unloading_column(axis, 5.0e-7) - unloading_column(axis, 1.0e-6)
      for axis in range(3)
    ]
  )
  np.testing.assert_allclose(observed.tangent.values, richardson, rtol=0.0, atol=1.0e-4)


def test_reject_path_records_carry_no_observation_and_match_the_plain_run() -> None:
  """The legacy-parameter shear ramp fails past the onset; rejects stay clean."""
  strains = tuple((0.0, 0.0, 1.5e-3 * (k + 1) / 8) for k in range(8))
  plain_driver, _ = _damage_deck(_DAMAGE_KAPPAC_LEGACY)
  plain = _run(plain_driver, strains)
  requested_driver, request = _damage_deck(_DAMAGE_KAPPAC_LEGACY)
  requested = _run(requested_driver, strains, request)
  assert plain.status is requested.status is DriverStatus.STEP_FAILED
  assert plain.failed_target_index == requested.failed_target_index
  assert len(plain.records) == len(requested.records)
  for first, second in zip(plain.records, requested.records, strict=True):
    assert first.status is second.status
    assert first.progress == second.progress
    assert first.cutback_level == second.cutback_level
    assert len(first.iterations) == len(second.iterations)
    for one, two in zip(first.iterations, second.iterations, strict=True):
      assert one.status is two.status
      if one.residual_norm is not None:
        # M57's ambient cross-run 1-ulp floor on iteration norms;
        # measured 0.0 on this deck in-process.
        np.testing.assert_allclose(
          one.residual_norm, two.residual_norm, rtol=_NORM_RTOL, atol=0.0
        )
      if one.increment_norm is not None:
        np.testing.assert_allclose(
          one.increment_norm, two.increment_norm, rtol=_NORM_RTOL, atol=0.0
        )
    if first.status is SubstepStatus.COMMITTED:
      assert first.observation.homogenization is None
      assert second.observation.homogenization is not None
      assert np.array_equal(
        first.observation.reactions.values,
        second.observation.reactions.values,
      )
    else:
      # Rejected and failed records never carry the observation.
      assert first.observation is None
      assert second.observation is None
  assert np.array_equal(
    plain_driver.owner.accepted_physical().values,
    requested_driver.owner.accepted_physical().values,
  )
  # StateGeneration equality includes a per-driver lineage UUID; the
  # schedule-level identity is the ordinal.
  assert plain.final_generation.ordinal == requested.final_generation.ordinal


def test_singular_converged_tangent_fails_closed() -> None:
  """A degenerate converged tangent reports a coded diagnostic, never silence."""
  strains = tuple((1.0e-4 * (k + 1), 0.0, 0.0) for k in range(8))
  plain_driver, _ = _damage_deck(_DAMAGE_KAPPAC_LEGACY)
  plain = _run(plain_driver, strains)
  # The primal run itself completes: the fully-damaged branch is a valid
  # equilibrium; the driver never needs its homogenized tangent.
  assert plain.status is DriverStatus.COMPLETED
  requested_driver, request = _damage_deck(_DAMAGE_KAPPAC_LEGACY)
  with pytest.raises(DriverEvaluationError, match="singular-homogenization-tangent"):
    _run(requested_driver, strains, request)
