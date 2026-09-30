# SPDX-License-Identifier: MIT

"""IsotropicHardeningPlasticity: legacy parity oracles and the driver path.

The legacy law ``pyfem/materials/IsotropicHardeningPlasticity.py`` is the
numerical reference (AGENTS.md). The parity harness drives it with the
``dstrain = eps_total - (eelas + eplas)`` identity the v3 kernel uses and
repairs the ``tang = self.ctang`` aliasing impurity with a pristine elastic
predictor before each step; single-step paths need no repair. The deliberate
``flow[3:]`` shear-transfer fix is pinned by asserting the legacy shear
bookkeeping slots drift from zero while the v3 slots stay exactly zero on
normal plastic paths. Documented tolerance for mixed shear paths: 1e-12
relative (observed drift is rounding-level, <= a few ulps).
"""

from __future__ import annotations

import contextlib
import io
import sys
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from pyfem.materials.IsotropicHardeningPlasticity import IsotropicHardeningPlasticity
from pyfem.v3.compile.continuum import plasticity_reference_registry
from pyfem.v3.compile.system import compile_system
from pyfem.v3.constraints import compile_constraint_map
from pyfem.v3.driver import (
  DriverStatus,
  NonlinearStaticDriver,
  NonlinearStaticSettings,
)
from pyfem.v3.materials.isotropic_hardening_plasticity import (
  ISOTROPIC_HARDENING_PLASTICITY_BINDING,
  isotropic_hardening_calibration,
  isotropic_hardening_plasticity_kernel,
  isotropic_hardening_plasticity_kernel_reference,
)
from pyfem.v3.model.operator import EvaluationStatus
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
  PrescribedDofSpec,
  ProgramCoordinateSpec,
  ProgramCoordinateValue,
  ProgramPoint,
)
from pyfem.v3.state import StateCodecError

# Documented oracle configuration: linear hardening, the only configuration
# the legacy law ever defined on the plastic path (its tangent reads the
# `hard` property; IsotropicHardeningPlasticity.py:119).
_E = 210000.0
_NU = 0.3
_SYIELD = 250.0
_HARD = 1000.0
_YIELD_STRAIN = _SYIELD * (1.0 + _NU) / _E  # plane-strain deviatoric bound
_PARITY_RTOL = 1.0e-12
_PARITY_ATOL = 1.0e-12

_UNIT_COORDINATES = (
  (0.0, 0.0),
  (0.5, 0.0),
  (1.0, 0.0),
  (1.0, 0.5),
  (1.0, 1.0),
  (0.5, 1.0),
  (0.0, 1.0),
  (0.0, 0.5),
)


class _LegacyProps:
  """Minimal iterable props stand-in for ``BaseMaterial``/``Hardening``."""

  def __init__(self, **values: object) -> None:
    self._values = values
    self.solverStat = None
    for name, value in values.items():
      setattr(self, name, value)

  def __iter__(self) -> object:
    return iter(self._values.items())


def _legacy_law() -> IsotropicHardeningPlasticity:
  with contextlib.redirect_stdout(io.StringIO()):
    with warnings.catch_warnings(action="ignore", category=DeprecationWarning):
      return IsotropicHardeningPlasticity(
        _LegacyProps(
          E=_E,
          nu=_NU,
          syield=_SYIELD,
          hard=_HARD,
          EqPlasStrains=np.array([0.0, 1.0]),
          Stresses=np.array([_SYIELD, _SYIELD + _HARD]),
        )
      )


def _pristine_ctang(mat: IsotropicHardeningPlasticity) -> np.ndarray:
  """Rebuild the elastic tangent the legacy constructor computed."""
  ctang = np.zeros(shape=(6, 6))
  ctang[:3, :3] = mat.elam
  ctang[0, 0] += mat.eg2
  ctang[1, 1] = ctang[0, 0]
  ctang[2, 2] = ctang[0, 0]
  ctang[3, 3] = mat.eg
  ctang[4, 4] = ctang[3, 3]
  ctang[5, 5] = ctang[3, 3]
  return ctang


def _legacy_row(mat: IsotropicHardeningPlasticity) -> np.ndarray:
  return np.concatenate(
    (
      np.atleast_1d(mat.getHistoryParameter("sigma")),
      np.atleast_1d(mat.getHistoryParameter("eelas")),
      np.atleast_1d(mat.getHistoryParameter("eplas")),
      np.atleast_1d(mat.getHistoryParameter("eqplas")),
    )
  )


def _legacy_step(
  mat: IsotropicHardeningPlasticity,
  strain_total: np.ndarray,
  *,
  repair: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Advance the legacy oracle one committed step on the total-strain path."""
  eelas = mat.getHistoryParameter("eelas")
  eplas = mat.getHistoryParameter("eplas")
  dstrain = np.array(strain_total - (eelas + eplas), copy=True)
  if repair:
    mat.ctang = _pristine_ctang(mat)
  with contextlib.redirect_stdout(io.StringIO()):
    with warnings.catch_warnings(action="ignore", category=DeprecationWarning):
      sigma, tangent = mat.getStress(SimpleNamespace(dstrain=dstrain))
  mat.commitHistory()
  return np.array(sigma, copy=True), np.array(tangent, copy=True), _legacy_row(mat)


def _v3_step(
  calibration: np.ndarray,
  rows: np.ndarray,
  strain_total: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  result = isotropic_hardening_plasticity_kernel(
    strain_total[None, :], rows, calibration
  )
  assert result.status is EvaluationStatus.OK
  return (
    np.array(result.stresses[0], copy=True),
    np.array(result.tangents[0], copy=True),
    result.trial_rows,
  )


def _elastic_path() -> list[np.ndarray]:
  return [np.array([eps, 0.0, 0.0, 0.0, 0.0, 0.0]) for eps in (0.0002, 0.0006, 0.0010)]


def _uniaxial_path() -> list[np.ndarray]:
  return [
    np.array([eps, 0.0, 0.0, 0.0, 0.0, 0.0]) for eps in np.linspace(0.0002, 0.004, 10)
  ]


def _shear_path() -> list[np.ndarray]:
  return [
    np.array([0.004, 0.0, 0.0, 0.0, 0.0, gamma])
    for gamma in np.linspace(0.001, 0.006, 6)
  ]


def test_single_step_paths_match_legacy_bitwise_without_repair() -> None:
  calibration = isotropic_hardening_calibration(_E, _NU, _SYIELD, _HARD)
  for strain in (
    np.array([0.001, 0.0, 0.0, 0.0, 0.0, 0.0]),
    np.array([0.004, 0.0, 0.0, 0.0, 0.0, 0.0]),
    np.array([0.004, 0.0, 0.0, 0.0, 0.0, 0.006]),
  ):
    legacy = _legacy_law()
    sigma_l, tangent_l, row_l = _legacy_step(legacy, strain, repair=False)
    sigma_v, tangent_v, rows_v = _v3_step(calibration, np.zeros((1, 19)), strain)
    assert np.array_equal(sigma_l, sigma_v)
    assert np.array_equal(tangent_l, tangent_v)
    assert row_l[18] == rows_v[0, 18]
    # The virgin-predictor equivalence holds with and without the repair.
    repaired = _legacy_law()
    sigma_r, tangent_r, _ = _legacy_step(repaired, strain, repair=True)
    assert np.array_equal(sigma_r, sigma_v)
    assert np.array_equal(tangent_r, tangent_v)


def test_elastic_ramp_matches_legacy_bitwise() -> None:
  calibration = isotropic_hardening_calibration(_E, _NU, _SYIELD, _HARD)
  legacy = _legacy_law()
  rows_v = np.zeros((1, 19))
  elastic_tangent = calibration[8:44].reshape(6, 6)
  for strain in _elastic_path():
    assert strain[0] < _YIELD_STRAIN
    sigma_l, tangent_l, row_l = _legacy_step(legacy, strain)
    sigma_v, tangent_v, rows_v = _v3_step(calibration, rows_v, strain)
    assert np.array_equal(sigma_l, sigma_v)
    assert np.array_equal(tangent_l, tangent_v)
    assert np.array_equal(tangent_v, elastic_tangent)
    assert np.array_equal(row_l, rows_v[0])
  assert rows_v[0, 18] == 0.0


def test_uniaxial_plastic_ramp_matches_legacy_bitwise() -> None:
  calibration = isotropic_hardening_calibration(_E, _NU, _SYIELD, _HARD)
  legacy = _legacy_law()
  rows_v = np.zeros((1, 19))
  went_plastic = False
  for strain in _uniaxial_path():
    sigma_l, tangent_l, row_l = _legacy_step(legacy, strain)
    sigma_v, tangent_v, rows_v = _v3_step(calibration, rows_v, strain)
    assert np.array_equal(sigma_l, sigma_v)
    assert np.array_equal(tangent_l, tangent_v)
    assert row_l[18] == rows_v[0, 18]
    # The strain-split normal slots agree bitwise; the shear slots pin the
    # documented flow[3:] fix: legacy pollutes them (its flow[:3] transfer),
    # v3 keeps them exactly zero on a normal plastic path.
    assert np.array_equal(row_l[6:9], rows_v[0, 6:9])
    assert np.array_equal(row_l[12:15], rows_v[0, 12:15])
    assert np.all(rows_v[0, 9:12] == 0.0)
    assert np.all(rows_v[0, 15:18] == 0.0)
    went_plastic |= bool(rows_v[0, 18] > 0.0)
  assert went_plastic
  assert np.any(row_l[9:12] != 0.0)
  assert np.any(row_l[15:18] != 0.0)


def test_mixed_shear_ramp_matches_legacy_within_documented_tolerance() -> None:
  calibration = isotropic_hardening_calibration(_E, _NU, _SYIELD, _HARD)
  legacy = _legacy_law()
  rows_v = np.zeros((1, 19))
  for strain in [*_uniaxial_path(), *_shear_path()]:
    sigma_l, tangent_l, row_l = _legacy_step(legacy, strain)
    sigma_v, tangent_v, rows_v = _v3_step(calibration, rows_v, strain)
    np.testing.assert_allclose(sigma_l, sigma_v, rtol=_PARITY_RTOL, atol=_PARITY_ATOL)
    np.testing.assert_allclose(
      tangent_l, tangent_v, rtol=_PARITY_RTOL, atol=_PARITY_ATOL
    )
    # The shear split slots (9:12, 15:18) are the pinned flow[3:] divergence:
    # legacy pollutes them with the normal flow, v3 carries the true shear
    # plastic flow, so they differ materially rather than at rounding level.
    np.testing.assert_allclose(
      row_l[6:9], rows_v[0, 6:9], rtol=_PARITY_RTOL, atol=_PARITY_ATOL
    )
    np.testing.assert_allclose(
      row_l[12:15], rows_v[0, 12:15], rtol=_PARITY_RTOL, atol=_PARITY_ATOL
    )
    np.testing.assert_allclose(
      row_l[18], rows_v[0, 18], rtol=_PARITY_RTOL, atol=_PARITY_ATOL
    )
  # The shear plastic flow is real state, not the legacy bookkeeping bug.
  assert np.any(rows_v[0, 15:18] != 0.0)


def test_unload_reload_keeps_kappa_and_returns_to_the_surface() -> None:
  calibration = isotropic_hardening_calibration(_E, _NU, _SYIELD, _HARD)
  legacy = _legacy_law()
  rows_v = np.zeros((1, 19))
  for strain in _uniaxial_path():
    _legacy_step(legacy, strain)
    _, _, rows_v = _v3_step(calibration, rows_v, strain)
  kappa_loaded = rows_v[0, 18]
  elastic_tangent = calibration[8:44].reshape(6, 6)

  unload = np.array([0.0025, 0.0, 0.0, 0.0, 0.0, 0.0])
  sigma_l, tangent_l, row_l = _legacy_step(legacy, unload)
  sigma_v, tangent_v, rows_v = _v3_step(calibration, rows_v, unload)
  assert np.array_equal(tangent_v, elastic_tangent)
  assert np.array_equal(tangent_l, tangent_v)
  assert rows_v[0, 18] == kappa_loaded
  assert row_l[18] == kappa_loaded
  np.testing.assert_allclose(sigma_l, sigma_v, rtol=_PARITY_RTOL, atol=_PARITY_ATOL)

  reload_strain = np.array([0.0055, 0.0, 0.0, 0.0, 0.0, 0.0])
  sigma_l, tangent_l, row_l = _legacy_step(legacy, reload_strain)
  sigma_v, tangent_v, rows_v = _v3_step(calibration, rows_v, reload_strain)
  np.testing.assert_allclose(sigma_l, sigma_v, rtol=_PARITY_RTOL, atol=_PARITY_ATOL)
  np.testing.assert_allclose(tangent_l, tangent_v, rtol=_PARITY_RTOL, atol=_PARITY_ATOL)
  assert rows_v[0, 18] > kappa_loaded


def test_kernel_reports_typed_failures() -> None:
  calibration = isotropic_hardening_calibration(_E, _NU, _SYIELD, _HARD)
  rows = np.zeros((1, 19))
  # Beyond the hardening table (legacy falls into IndexError/NameError).
  result = isotropic_hardening_plasticity_kernel(
    np.array([[2.0, 0.0, 0.0, 0.0, 0.0, 0.0]]), rows, calibration
  )
  assert result.status is EvaluationStatus.REJECT_STEP
  assert np.array_equal(result.trial_rows, rows)
  # Non-finite predictor.
  result = isotropic_hardening_plasticity_kernel(
    np.array([[np.inf, 0.0, 0.0, 0.0, 0.0, 0.0]]), rows, calibration
  )
  assert result.status is EvaluationStatus.REJECT_STEP
  assert np.array_equal(result.trial_rows, rows)
  # Contract violations stay typed exceptions.
  with pytest.raises(TypeError, match="calibration"):
    isotropic_hardening_plasticity_kernel(np.zeros((1, 6)), rows, np.zeros(3))
  with pytest.raises(ValueError, match="must be positive"):
    isotropic_hardening_calibration(_E, _NU, -1.0, _HARD)
  with pytest.raises(ValueError, match="between -1 and 0.5"):
    isotropic_hardening_calibration(_E, 0.6, _SYIELD, _HARD)
  with pytest.raises(TypeError, match="exactly four"):
    ISOTROPIC_HARDENING_PLASTICITY_BINDING(_E, _NU, _SYIELD)


def _plasticity_model() -> ModelSpec:
  nodes = tuple(
    NodeSpec(
      id=index + 1,
      coordinates=point,
      source=SourceContext(source=f"node-{index + 1}"),
    )
    for index, point in enumerate(_UNIT_COORDINATES)
  )
  cell = CellSpec(
    id="cell-1",
    node_ids=tuple(node.id for node in nodes),
    source=SourceContext(source="cell"),
  )
  return ModelSpec(
    mesh=MeshSpec(
      nodes=nodes,
      cell_blocks=(
        CellBlockSpec(
          id="cells",
          reference_topology="quadrilateral",
          topological_dimension=2,
          embedding_dimension=2,
          geometry_interpolation="serendipity-quad8",
          cells=(cell,),
          source=SourceContext(source="block"),
        ),
      ),
      source=SourceContext(source="mesh"),
    ),
    fields=(
      FieldSpec(
        id="displacement",
        components=("x", "y"),
        location="node",
        source=SourceContext(source="field"),
      ),
    ),
    materials=(
      MaterialSpec(
        id="steel",
        model="isotropic-hardening-plasticity",
        parameters=(
          MaterialParameterSpec("youngs_modulus", _E),
          MaterialParameterSpec("poisson_ratio", _NU),
          MaterialParameterSpec("initial_yield_stress", _SYIELD),
          MaterialParameterSpec("hardening_slope", _HARD),
        ),
        source=SourceContext(source="material"),
      ),
    ),
    regions=(
      RegionSpec(
        id="domain",
        cell_refs=(CellRef("cells", "cell-1"),),
        field_ids=("displacement",),
        material_id="steel",
        formulation="small-strain-continuum",
        quadrature="gauss-3x3",
        source=SourceContext(source="region"),
      ),
    ),
    source=SourceContext(source="model"),
  )


def _strain_ramp_driver(
  settings: NonlinearStaticSettings | None = None,
) -> NonlinearStaticDriver:
  """One Q8 element under a prescribed affine eps_xx ramp, one free DOF.

  All nodal displacements follow the homogeneous strain field exactly
  (u_x = eps * x, u_y = 0) except node 6's x component, which stays free so
  every Newton iteration must assemble and factorize the tangent.
  """
  system = compile_system(_plasticity_model(), plasticity_reference_registry())
  constraints = []
  for node in _UNIT_COORDINATES:
    node_id = _UNIT_COORDINATES.index(node) + 1
    if node_id != 6:
      constraints.append(
        PrescribedDofSpec(
          id=f"ux-{node_id}",
          target=DofRef(
            node_id=node_id,
            field_id="displacement",
            component="x",
          ),
          value=AffineValueSpec(
            coefficients=(AffineCoefficientSpec("eps", node[0]),),
          ),
          source=SourceContext(source=f"ux-{node_id}"),
        )
      )
    constraints.append(
      PrescribedDofSpec(
        id=f"uy-{node_id}",
        target=DofRef(node_id=node_id, field_id="displacement", component="y"),
        value=AffineValueSpec(constant=0.0),
        source=SourceContext(source=f"uy-{node_id}"),
      )
    )
  coordinate_map = compile_constraint_map(
    system,
    constraints=tuple(constraints),
    coordinates=(ProgramCoordinateSpec(name="eps", kind="load"),),
  )
  if settings is None:
    return NonlinearStaticDriver(system, coordinate_map, ())
  return NonlinearStaticDriver(system, coordinate_map, (), settings)


def _eps_point(value: float) -> ProgramPoint:
  return ProgramPoint((ProgramCoordinateValue("eps", value),))


def _committed_ip_strains(driver: NonlinearStaticDriver) -> np.ndarray:
  """Recompute the committed per-integration-point 6-Voigt strains.

  Uses the operator's own physical strain-displacement expression and the
  committed coefficient vector, mirroring the evaluation's arithmetic exactly.
  """
  operator = driver.owner.system.operators[0]
  payload = operator.payload
  b_matrix = (
    payload.normalized_strain_displacement.values
    / payload.geometry_scales.values[:, None, None, None]
  )
  gather = operator.header.ports[0].coefficient_map.values
  values = driver.owner.accepted_physical().values
  strain3 = np.einsum("epai,ei->epa", b_matrix, values[gather], optimize=True)
  strains = np.zeros((np.prod(strain3.shape[:2]), 6), dtype=np.float64)
  flat = strain3.reshape(-1, 3)
  strains[:, 0] = flat[:, 0]
  strains[:, 1] = flat[:, 1]
  strains[:, 5] = flat[:, 2]
  return strains


def test_driver_ramp_commits_oracle_state_with_fresh_factorizations() -> None:
  driver = _strain_ramp_driver()
  assert driver.plan.constant_tangent is False
  strain_path = (0.001, 0.002, 0.004)
  base = _eps_point(0.0)
  layout = driver.owner.system.operators[0].header.state_layout
  block_id = layout.block_id
  calibration = isotropic_hardening_calibration(_E, _NU, _SYIELD, _HARD)
  oracle_rows = np.zeros((9, 19))
  for step, eps in enumerate(strain_path, 1):
    result = driver.run(base_point=base, target_points=(_eps_point(eps),))
    assert result.status is DriverStatus.COMPLETED
    base = _eps_point(eps)
    # The committed rows equal the batched kernel stepped on the committed
    # per-integration-point strain path — bitwise, since the driver stages the
    # trial rows of the converged iterate.
    oracle = isotropic_hardening_plasticity_kernel(
      _committed_ip_strains(driver), oracle_rows, calibration
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
  statistics = result.statistics
  assert statistics.committed_substep_count == 3
  assert statistics.rejected_substep_count == 0
  # Factorization honesty: the algorithmic-symmetric channel is not linear,
  # so the driver re-assembles and re-factorizes every Newton iteration.
  assert statistics.factorization_reuse_count == 0
  assert statistics.factorization_count == statistics.tangent_refill_count
  assert statistics.factorization_count == statistics.linear_solve_count
  assert statistics.factorization_count > statistics.committed_substep_count


def test_driver_overshoot_rejects_and_leaves_state_byte_identical() -> None:
  driver = _strain_ramp_driver(NonlinearStaticSettings(max_cutbacks=2))
  layout = driver.owner.system.operators[0].header.state_layout
  block_id = layout.block_id
  # A huge-but-tabulated base state commits; the table edge is kappa = 1.0.
  setup = driver.run(base_point=_eps_point(0.0), target_points=(_eps_point(0.9),))
  assert setup.status is DriverStatus.COMPLETED
  before = driver.statistics
  snapshot = (
    driver.owner.accepted_physical().values.tobytes(),
    driver.owner.generation.ordinal,
    driver.owner.history,
    driver.owner.encode_state(block_id),
  )
  result = driver.run(
    base_point=_eps_point(0.9),
    target_points=(_eps_point(3.0),),
  )
  assert result.status is DriverStatus.STEP_FAILED
  assert result.failed_target_index == 0
  # Every attempted increment crosses the hardening-table edge: one initial
  # rejection plus two cutbacks, and nothing new commits.
  assert result.statistics.rejected_substep_count - before.rejected_substep_count == 3
  assert result.statistics.committed_substep_count - before.committed_substep_count == 0
  assert driver.owner.accepted_physical().values.tobytes() == snapshot[0]
  assert driver.owner.generation.ordinal == snapshot[1]
  assert driver.owner.history == snapshot[2]
  assert driver.owner.encode_state(block_id) == snapshot[3]


def test_state_codec_round_trips_and_rejects_foreign_schemas() -> None:
  driver = _strain_ramp_driver()
  result = driver.run(base_point=_eps_point(0.0), target_points=(_eps_point(0.004),))
  assert result.status is DriverStatus.COMPLETED
  owner = driver.owner
  layout = owner.system.operators[0].header.state_layout
  payload = owner.encode_state(layout.block_id)
  decoded = owner.decode_state(layout.block_id, payload)
  np.testing.assert_array_equal(
    decoded.values, owner.accepted_state(layout.block_id).values
  )
  with pytest.raises(StateCodecError):
    owner.decode_state(layout.block_id, payload.replace(b"sigma:6", b"sigma:7"))


def test_compiled_plasticity_system_fingerprint_is_deterministic() -> None:
  first = compile_system(_plasticity_model(), plasticity_reference_registry())
  second = compile_system(_plasticity_model(), plasticity_reference_registry())
  assert first.content_fingerprint == second.content_fingerprint
  harder = compile_system(_plasticity_model(), plasticity_reference_registry())
  assert harder.content_fingerprint == first.content_fingerprint
  layout = first.operators[0].header.state_layout
  assert layout.schema == (
    "pyfem-v3-j2-isotropic-hardening-state-v1|sigma:6,epsilon_e:6,epsilon_p:6,kappa:1"
  )
  assert layout.entity_count == 9
  assert layout.row_width == 19
  assert layout.initial_rows is not None
  assert np.all(layout.initial_rows.values == 0.0)


def test_optimized_kernel_matches_reference_bitwise() -> None:
  """The M30 optimized kernel reproduces the M25 reference bit for bit.

  Deterministic and seeded batches over virgin, stepped, and rejecting
  states: statuses agree and stresses, tangents, and trial rows compare
  equal as raw uint64 (signed-zero distinctions included).
  """
  calibration = isotropic_hardening_calibration(_E, _NU, _SYIELD, _HARD)

  def assert_bitwise(strains: np.ndarray, rows: np.ndarray) -> None:
    reference = isotropic_hardening_plasticity_kernel_reference(
      strains, rows, calibration
    )
    optimized = isotropic_hardening_plasticity_kernel(strains, rows, calibration)
    assert optimized.status is reference.status
    np.testing.assert_array_equal(
      optimized.stresses.view(np.uint64), reference.stresses.view(np.uint64)
    )
    np.testing.assert_array_equal(
      optimized.tangents.view(np.uint64), reference.tangents.view(np.uint64)
    )
    np.testing.assert_array_equal(
      optimized.trial_rows.view(np.uint64), reference.trial_rows.view(np.uint64)
    )

  # Deterministic documented ramp magnitudes, every third entity in mixed
  # shear, so elastic, plastic-normal, and plastic-mixed branches all appear.
  n = 1024
  strains = np.zeros((n, 6))
  strains[:, 0] = np.linspace(0.0002, 0.004, n)
  strains[::3, 5] = 0.003
  virgin = np.zeros((n, 19))
  assert_bitwise(strains, virgin)
  stepped = isotropic_hardening_plasticity_kernel_reference(
    strains, virgin, calibration
  ).trial_rows
  assert_bitwise(strains * 1.5, stepped)

  # Seeded random batches, virgin then stepped (nonzero state, mixed paths).
  rng = np.random.default_rng(42)
  random_strains = rng.normal(size=(512, 6)) * 1.5e-3
  assert_bitwise(random_strains, np.zeros((512, 19)))
  stepped_random = isotropic_hardening_plasticity_kernel_reference(
    random_strains, np.zeros((512, 19)), calibration
  ).trial_rows
  assert_bitwise(rng.normal(size=(512, 6)) * 1.0e-3, stepped_random)

  # Whole-batch rejects: beyond the hardening table, non-finite predictor.
  extreme = np.zeros((8, 6))
  extreme[3, 0] = 2.0
  assert_bitwise(extreme, np.zeros((8, 19)))
  nonfinite = np.zeros((8, 6))
  nonfinite[5, 4] = np.inf
  assert_bitwise(nonfinite, np.zeros((8, 19)))
