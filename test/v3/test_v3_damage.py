# SPDX-License-Identifier: MIT

"""PlaneStrainDamage: legacy parity oracles and the G4 tangent-class proofs.

The legacy law ``pyfem/materials/PlaneStrainDamage.py`` is the numerical
reference (AGENTS.md) for the v3 damage kernel. Stresses, the kappa
envelope, and the tangent match it bit for bit on every documented
branch — elastic, progressive, and unloading. That full parity is the
repaired state: legacy ``getEquivStrain`` used to return the
never-assigned ``depsdstrain = zeros(3)`` (killing the rank-1 tangent
correction, so the legacy tangent stayed the symmetric
``(1 - omega) * De`` on progressive branches) and its discarded
derivative expression halved the shear component
(``dexydstrain = 0.5 * O3`` against ``J2 = ... + exy**2``) — two
correctness bugs (finding 20261007-agent-g4) that this battery pinned as
a structural divergence (v3-minus-legacy exactly rank-1 with the
effective-stress left vector, the legacy tangent failing the
finite-difference check materially). M55 repaired both defects in the
legacy law (commits bad3512 / 5b1bee2, "correctness bug"); the v3
kernel, which had deliberately inherited neither, is unchanged, and the
relationship is now parity-where-repaired: the pins below assert
bitwise tangent equality on progressive branches, with the shared
tangent still witnessed as the nonsymmetric rank-1-corrected one, and a
finite-difference check pins both tangents as the algorithmically
consistent derivative of the shared stress response.

Documented oracle configuration (the landed chapter-6 deck
``examples/ch06/ContDamExample.pro``): E = 100, nu = 0.3, k = 1.0,
kappa0 = 1.0e-6, kappac = 1.0e-5.

The file also hosts the G4 secant-branch witness: a mock law whose kernel
reports the secant stiffness below a strain threshold and the true tangent
above it (the mission's branch direction; the legacy XuNeedleman 3D shear
branch at XuNeedleman.py:110-113 switches the same two stiffnesses the other
way — the class semantics are branch dependence, not direction). The mock
compiles through the generic v2 stateful path with honest
``linear=False, symmetric=False`` channel flags.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from pyfem.materials.PlaneStrainDamage import PlaneStrainDamage
from pyfem.v3.compile.continuum import (
  damage_reference_registry,
  q8_reference_registry,
)
from pyfem.v3.compile.contracts import (
  STATEFUL_MATERIAL_DESCRIPTOR_SCHEMA,
  StatefulContinuumKernelResult,
  validate_stateful_material_metadata,
)
from pyfem.v3.compile.system import compile_system
from pyfem.v3.constraints import compile_constraint_map
from pyfem.v3.driver import (
  DriverStatus,
  NonlinearStaticDriver,
)
from pyfem.v3.materials.plane_strain_damage import (
  PLANE_STRAIN_DAMAGE_BINDING,
  plane_strain_damage_calibration,
  plane_strain_damage_kernel,
  plane_strain_damage_metadata,
)
from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.operator import (
  ChannelRequest,
  EvaluationStatus,
  OperatorEvaluationInput,
  evaluation_status,
)
from pyfem.v3.model.provenance import CanonicalManifest
from pyfem.v3.model.registry import RegistryDescriptor
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

# Documented oracle configuration (examples/ch06/ContDamExample.pro).
_E = 100.0
_NU = 0.3
_K = 1.0
_KAPPA0 = 1.0e-6
_KAPPAC = 1.0e-5
_FD_RTOL = 1.0e-6
_FD_ATOL = 1.0e-8

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
  """Minimal iterable props stand-in for ``BaseMaterial``."""

  def __init__(self, **values: object) -> None:
    self._values = values
    self.solverStat = None
    for name, value in values.items():
      setattr(self, name, value)

  def __iter__(self) -> object:
    return iter(self._values.items())


def _legacy_law() -> PlaneStrainDamage:
  return PlaneStrainDamage(
    _LegacyProps(E=_E, nu=_NU, k=_K, kappa0=_KAPPA0, kappac=_KAPPAC)
  )


def _legacy_step(
  mat: PlaneStrainDamage,
  strain3: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
  """Advance the legacy oracle one committed step on the total-strain path."""
  sigma, tangent = mat.getStress(SimpleNamespace(strain=np.array(strain3, copy=True)))
  mat.commitHistory()
  return (
    np.array(sigma, copy=True),
    np.array(tangent, copy=True),
    float(mat.getHistoryParameter("kappa")),
  )


def _v3_step(
  calibration: np.ndarray,
  rows: np.ndarray,
  strain3: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  strain6 = np.zeros(6)
  strain6[[0, 1, 5]] = strain3
  result = plane_strain_damage_kernel(strain6[None, :], rows, calibration)
  assert result.status is EvaluationStatus.OK
  tangent3 = result.tangents[0][[0, 1, 5]][:, [0, 1, 5]]
  return (
    np.array(result.stresses[0][[0, 1, 5]], copy=True),
    np.array(tangent3, copy=True),
    result.trial_rows,
  )


def _omega(kappa: float) -> float:
  """The legacy ``getDamage`` factor (PlaneStrainDamage.py:140-152)."""
  if kappa <= _KAPPA0:
    return 0.0
  if kappa < _KAPPAC:
    fac = _KAPPAC / kappa
    return fac * (kappa - _KAPPA0) / (_KAPPAC - _KAPPA0)
  return 1.0


def _elastic_path() -> list[np.ndarray]:
  return [np.array([eps, 0.0, 0.0]) for eps in (0.5e-6, 0.8e-6)]


def _progressive_path() -> list[np.ndarray]:
  return [np.array([eps, 0.0, 0.0]) for eps in (2.0e-6, 4.0e-6, 6.0e-6, 8.0e-6)]


def _mixed_shear_path() -> list[np.ndarray]:
  return [np.array([4.0e-6, 0.0, gamma]) for gamma in (1.0e-6, 2.0e-6, 3.0e-6, 4.0e-6)]


def _assert_progressive_parity(
  tang_l: np.ndarray,
  tang_v: np.ndarray,
  de: np.ndarray,
  strain3: np.ndarray,
  kappa: float,
) -> None:
  """Pin the repaired progressive-branch tangent parity.

  The repaired legacy tangent equals the v3 tangent bit for bit (M55: the
  legacy zero-return and shear-halving defects are repaired, and the v3
  kernel had inherited neither). The shared tangent is witnessed as the
  rank-1-corrected one: its difference from the symmetric secant
  ``(1 - omega) * De`` is exactly rank-1 (up to rounding) with the
  effective stress as its left vector, and it is materially nonsymmetric.
  """
  assert kappa > _KAPPA0
  assert np.array_equal(tang_l, tang_v)
  difference = tang_v - (1.0 - _omega(kappa)) * de
  u, s, vt = np.linalg.svd(difference)
  assert s[0] > 0.0
  assert s[1] <= 1.0e-9 * s[0]
  eff_stress = np.dot(de, strain3)
  cosine = abs(float(u[:, 0] @ eff_stress)) / float(np.linalg.norm(eff_stress))
  assert cosine >= 1.0 - 1.0e-9
  del vt
  scale = float(np.max(np.abs(tang_v)))
  assert float(np.max(np.abs(tang_v - tang_v.T))) > 1.0e-3 * scale


def test_elastic_ramp_matches_legacy_bitwise() -> None:
  calibration = plane_strain_damage_calibration(_E, _NU, _KAPPA0, _KAPPAC, _K)
  de = calibration[7:16].reshape(3, 3)
  legacy = _legacy_law()
  rows_v = np.zeros((1, 1))
  for strain in _elastic_path():
    sigma_l, tang_l, kappa_l = _legacy_step(legacy, strain)
    sigma_v, tang_v, rows_v = _v3_step(calibration, rows_v, strain)
    assert np.array_equal(sigma_l, sigma_v)
    assert np.array_equal(tang_l, tang_v)
    # Below the onset the damage factor is exactly zero: undamaged De.
    assert np.array_equal(tang_v, de)
    assert float(kappa_l) == float(rows_v[0, 0])
  assert rows_v[0, 0] < _KAPPA0


def test_progressive_ramp_stress_kappa_bitwise_tangent_parity_rank1() -> None:
  calibration = plane_strain_damage_calibration(_E, _NU, _KAPPA0, _KAPPAC, _K)
  de = calibration[7:16].reshape(3, 3)
  legacy = _legacy_law()
  rows_v = np.zeros((1, 1))
  for strain in _progressive_path():
    sigma_l, tang_l, kappa_l = _legacy_step(legacy, strain)
    sigma_v, tang_v, rows_v = _v3_step(calibration, rows_v, strain)
    assert np.array_equal(sigma_l, sigma_v)
    assert float(kappa_l) == float(rows_v[0, 0])
    _assert_progressive_parity(tang_l, tang_v, de, strain, kappa_l)


def test_mixed_shear_path_stress_kappa_bitwise_tangent_parity_rank1() -> None:
  calibration = plane_strain_damage_calibration(_E, _NU, _KAPPA0, _KAPPAC, _K)
  de = calibration[7:16].reshape(3, 3)
  legacy = _legacy_law()
  rows_v = np.zeros((1, 1))
  for strain in _mixed_shear_path():
    sigma_l, tang_l, kappa_l = _legacy_step(legacy, strain)
    sigma_v, tang_v, rows_v = _v3_step(calibration, rows_v, strain)
    assert np.array_equal(sigma_l, sigma_v)
    assert float(kappa_l) == float(rows_v[0, 0])
    _assert_progressive_parity(tang_l, tang_v, de, strain, kappa_l)


def test_unload_reload_keeps_kappa_envelope_and_tangent_parity() -> None:
  calibration = plane_strain_damage_calibration(_E, _NU, _KAPPA0, _KAPPAC, _K)
  de = calibration[7:16].reshape(3, 3)
  legacy = _legacy_law()
  rows_v = np.zeros((1, 1))
  for strain in _progressive_path()[:3]:
    _legacy_step(legacy, strain)
    _, _, rows_v = _v3_step(calibration, rows_v, strain)
  kappa_loaded = float(rows_v[0, 0])
  assert kappa_loaded > _KAPPA0

  unload = np.array([3.0e-6, 0.0, 0.0])
  sigma_l, tang_l, kappa_l = _legacy_step(legacy, unload)
  sigma_v, tang_v, rows_v = _v3_step(calibration, rows_v, unload)
  # Elastic unload: both tangents are exactly the damaged-elastic matrix.
  assert np.array_equal(tang_l, tang_v)
  assert np.array_equal(tang_v, (1.0 - _omega(kappa_loaded)) * de)
  assert not np.array_equal(tang_v, de)
  assert np.array_equal(sigma_l, sigma_v)
  assert float(kappa_l) == kappa_loaded
  assert float(rows_v[0, 0]) == kappa_loaded

  reload_strain = np.array([9.0e-6, 0.0, 0.0])
  sigma_l, tang_l, kappa_l = _legacy_step(legacy, reload_strain)
  sigma_v, tang_v, rows_v = _v3_step(calibration, rows_v, reload_strain)
  assert np.array_equal(sigma_l, sigma_v)
  assert float(kappa_l) == float(rows_v[0, 0])
  assert float(kappa_l) > kappa_loaded
  _assert_progressive_parity(tang_l, tang_v, de, reload_strain, kappa_l)


def test_full_softening_zeros_the_response_bitwise() -> None:
  calibration = plane_strain_damage_calibration(_E, _NU, _KAPPA0, _KAPPAC, _K)
  legacy = _legacy_law()
  rows_v = np.zeros((1, 1))
  for strain in (np.array([2.0e-5, 0.0, 0.0]), np.array([3.0e-5, 0.0, 0.0])):
    sigma_l, tang_l, kappa_l = _legacy_step(legacy, strain)
    sigma_v, tang_v, rows_v = _v3_step(calibration, rows_v, strain)
    assert float(kappa_l) > _KAPPAC
    assert np.array_equal(sigma_l, sigma_v)
    assert np.array_equal(tang_l, tang_v)
    assert np.all(sigma_v == 0.0)
    assert np.all(tang_v == 0.0)
  # The envelope keeps tracking the equivalent strain past kappac.
  assert float(rows_v[0, 0]) > _KAPPAC


def test_progressive_tangent_matches_stress_finite_difference() -> None:
  calibration = plane_strain_damage_calibration(_E, _NU, _KAPPA0, _KAPPAC, _K)
  for strain in (np.array([3.0e-6, 0.0, 0.0]), np.array([4.0e-6, 0.0, 2.0e-6])):
    rows = np.zeros((1, 1))
    _, tang_v, _ = _v3_step(calibration, rows, strain)
    fd = np.zeros((3, 3))
    step = 1.0e-9
    for column in range(3):
      delta = np.zeros(3)
      delta[column] = step
      up = _v3_step(calibration, rows, strain + delta)[0]
      down = _v3_step(calibration, rows, strain - delta)[0]
      fd[:, column] = (up - down) / (2.0 * step)
    np.testing.assert_allclose(fd, tang_v, rtol=_FD_RTOL, atol=_FD_ATOL)
    # Repair confirmed: the legacy tangent passes the same check — it is the
    # algorithmic tangent of the shared stress response (the pre-repair
    # symmetric secant failed this materially; M55 commits bad3512/5b1bee2).
    legacy = _legacy_law()
    _, tang_l, _ = _legacy_step(legacy, strain)
    np.testing.assert_allclose(fd, tang_l, rtol=_FD_RTOL, atol=_FD_ATOL)
    assert np.array_equal(tang_l, tang_v)


def test_kernel_reports_typed_failures() -> None:
  calibration = plane_strain_damage_calibration(_E, _NU, _KAPPA0, _KAPPAC, _K)
  rows = np.zeros((1, 1))
  # Non-finite strain batch.
  result = plane_strain_damage_kernel(
    np.array([[np.inf, 0.0, 0.0, 0.0, 0.0, 0.0]]), rows, calibration
  )
  assert result.status is EvaluationStatus.REJECT_STEP
  assert np.array_equal(result.trial_rows, rows)
  # Overflowing equivalent strain (a2 * i1 is 0 * inf = nan for k = 1).
  result = plane_strain_damage_kernel(
    np.array([[1.0e200, 0.0, 0.0, 0.0, 0.0, 0.0]]), rows, calibration
  )
  assert result.status is EvaluationStatus.REJECT_STEP
  assert np.array_equal(result.trial_rows, rows)
  # Contract violations stay typed exceptions.
  with pytest.raises(TypeError, match="calibration"):
    plane_strain_damage_kernel(np.zeros((1, 6)), rows, np.zeros(3))
  with pytest.raises(ValueError, match=r"\(n, 6\) strains"):
    plane_strain_damage_kernel(np.zeros((1, 3)), rows, calibration)
  with pytest.raises(ValueError, match="kappa_0 must be positive"):
    plane_strain_damage_calibration(_E, _NU, -1.0, _KAPPAC, _K)
  with pytest.raises(ValueError, match="kappa_c must exceed kappa_0"):
    plane_strain_damage_calibration(_E, _NU, 1.0e-4, _KAPPAC, _K)
  with pytest.raises(ValueError, match="strength_ratio must be positive"):
    plane_strain_damage_calibration(_E, _NU, _KAPPA0, _KAPPAC, 0.0)
  with pytest.raises(ValueError, match="between -1 and 0.5"):
    plane_strain_damage_calibration(_E, 0.6, _KAPPA0, _KAPPAC, _K)
  with pytest.raises(TypeError, match="exactly five"):
    PLANE_STRAIN_DAMAGE_BINDING(_E, _NU, _KAPPA0, _KAPPAC)


def test_damage_descriptor_is_pinned_and_self_consistent() -> None:
  metadata = plane_strain_damage_metadata()
  validate_stateful_material_metadata(metadata)
  assert metadata["tangent_class"] == "algorithmic-nonsymmetric"
  assert PLANE_STRAIN_DAMAGE_BINDING.descriptor_metadata() == metadata
  assert (
    CanonicalManifest(metadata).to_bytes()
    == CanonicalManifest(plane_strain_damage_metadata()).to_bytes()
  )


def _damage_model() -> ModelSpec:
  return _continuum_model(
    "plane-strain-damage",
    (
      ("youngs_modulus", _E),
      ("poisson_ratio", _NU),
      ("kappa_0", _KAPPA0),
      ("kappa_c", _KAPPAC),
      ("strength_ratio", _K),
    ),
  )


def _continuum_model(
  model: str,
  parameters: tuple[tuple[str, float], ...],
) -> ModelSpec:
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
        id="material",
        model=model,
        parameters=tuple(
          MaterialParameterSpec(name, value) for name, value in parameters
        ),
        source=SourceContext(source="material"),
      ),
    ),
    regions=(
      RegionSpec(
        id="domain",
        cell_refs=(CellRef("cells", "cell-1"),),
        field_ids=("displacement",),
        material_id="material",
        formulation="small-strain-continuum",
        quadrature="gauss-3x3",
        source=SourceContext(source="region"),
      ),
    ),
    source=SourceContext(source="model"),
  )


def test_compiled_damage_operator_declares_honest_nonsymmetric_channels() -> None:
  system = compile_system(_damage_model(), damage_reference_registry())
  header = system.operators[0].header
  (residual,) = header.residual_channels
  assert residual.channel_id == "internal-force"
  assert residual.linear is False
  (jacobian,) = header.jacobian_channels
  assert jacobian.channel_id == "material-tangent"
  assert jacobian.linear is False
  assert jacobian.symmetric is False
  layout = header.state_layout
  assert layout.schema == "pyfem-v3-plane-strain-damage-state-v1|kappa:1"
  assert layout.row_width == 1
  assert layout.entity_count == 9
  (slot,) = layout.slots
  assert slot.name == "kappa"
  assert slot.annotation == "envelope-max"
  assert layout.initial_rows is not None
  assert np.all(layout.initial_rows.values == 0.0)
  again = compile_system(_damage_model(), damage_reference_registry())
  assert again.content_fingerprint == system.content_fingerprint


def _ramp_driver(model: ModelSpec, registry: dict) -> NonlinearStaticDriver:
  """One Q8 element under a prescribed affine eps_xx ramp, one free DOF.

  All nodal displacements follow the homogeneous strain field exactly
  (u_x = eps * x, u_y = 0) except node 6's x component, which stays free so
  every Newton iteration must assemble and factorize the tangent.
  """
  system = compile_system(model, registry)
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
  return NonlinearStaticDriver(system, coordinate_map, ())


def _eps_point(value: float) -> ProgramPoint:
  return ProgramPoint((ProgramCoordinateValue("eps", value),))


def _committed_ip_strains(driver: NonlinearStaticDriver) -> np.ndarray:
  """Recompute the committed per-integration-point 6-Voigt strains."""
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


def _operator_tangent(
  driver: NonlinearStaticDriver,
  scale: float,
) -> np.ndarray:
  """Evaluate the compiled operator tangent at ``scale`` x committed dofs."""
  operator = driver.owner.system.operators[0]
  layout = operator.header.state_layout
  gather = operator.header.ports[0].coefficient_map.values
  values = driver.owner.accepted_physical().values
  evaluation = operator.evaluate(
    OperatorEvaluationInput(
      port_values=(FinalizedArray(scale * values[gather], dtype=np.float64),),
      accepted_state=driver.owner.accepted_state(layout.block_id),
      signals=(),
      request=ChannelRequest((), ("material-tangent",)),
    )
  )
  assert evaluation_status(evaluation) is EvaluationStatus.OK
  return evaluation.jacobian_values[0].values


def test_driver_ramp_commits_oracle_state_with_fresh_factorizations() -> None:
  driver = _ramp_driver(_damage_model(), damage_reference_registry())
  assert driver.plan.constant_tangent is False
  base = _eps_point(0.0)
  layout = driver.owner.system.operators[0].header.state_layout
  block_id = layout.block_id
  calibration = plane_strain_damage_calibration(_E, _NU, _KAPPA0, _KAPPAC, _K)
  oracle_rows = np.zeros((9, 1))
  result = None
  for step, eps in enumerate((0.5e-6, 2.0e-6, 5.0e-6), 1):
    result = driver.run(base_point=base, target_points=(_eps_point(eps),))
    assert result.status is DriverStatus.COMPLETED
    base = _eps_point(eps)
    # The committed rows equal the batched kernel stepped on the committed
    # per-integration-point strain path — bitwise. (No homogeneity pin: the
    # softening law localizes, so the converged field is legitimately
    # non-affine and the kappa rows differ across integration points.)
    oracle = plane_strain_damage_kernel(
      _committed_ip_strains(driver), oracle_rows, calibration
    )
    assert oracle.status is EvaluationStatus.OK
    oracle_rows = oracle.trial_rows
    rows = driver.owner.accepted_state(block_id).values
    assert rows.shape == (9, 1)
    np.testing.assert_array_equal(rows, oracle_rows)
    assert result.final_generation.ordinal == step
  assert result is not None
  statistics = result.statistics
  assert statistics.committed_substep_count == 3
  assert statistics.rejected_substep_count == 0
  # Factorization honesty: the algorithmic-nonsymmetric channel is not
  # linear, so the driver re-assembles and re-factorizes every Newton
  # iteration through the general splu path.
  assert statistics.factorization_reuse_count == 0
  assert statistics.factorization_count == statistics.tangent_refill_count
  assert statistics.factorization_count == statistics.linear_solve_count
  assert statistics.factorization_count > statistics.committed_substep_count
  # Above the committed kappa envelope (a mid-Newton-iteration state) the
  # assembled element tangent is genuinely nonsymmetric — exactly the
  # tangent the splu path factorized during the run.
  tangent = _operator_tangent(driver, 1.2)
  scale = float(np.max(np.abs(tangent)))
  assert float(np.max(np.abs(tangent - tangent.transpose(0, 2, 1)))) > 1.0e-3 * scale
  # At the committed envelope the response sits exactly on the kappa kink:
  # the tangent falls back to the symmetric damaged-elastic matrix.
  committed = _operator_tangent(driver, 1.0)
  np.testing.assert_allclose(
    committed, committed.transpose(0, 2, 1), rtol=1.0e-12, atol=1.0e-12
  )


def test_state_codec_round_trips_and_rejects_foreign_schemas() -> None:
  driver = _ramp_driver(_damage_model(), damage_reference_registry())
  result = driver.run(base_point=_eps_point(0.0), target_points=(_eps_point(5.0e-6),))
  assert result.status is DriverStatus.COMPLETED
  owner = driver.owner
  layout = owner.system.operators[0].header.state_layout
  payload = owner.encode_state(layout.block_id)
  decoded = owner.decode_state(layout.block_id, payload)
  np.testing.assert_array_equal(
    decoded.values, owner.accepted_state(layout.block_id).values
  )
  with pytest.raises(StateCodecError):
    owner.decode_state(layout.block_id, payload.replace(b"kappa:1", b"kappa:2"))


# ---------------------------------------------------------------------------
# G4 secant-branch witness: a branch-dependent mock law (secant stiffness
# below a strain threshold, true tangent above it).

_SECANT_MODEL = "mock-secant-branch-cohesive"
_SECANT_STATE_SCHEMA = "pyfem-v3-mock-secant-branch-state-v1"
_SECANT_STIFFNESS = 1000.0
_SECANT_E0 = 1.0e-5
_SECANT_THRESHOLD = 2.0e-6


def _secant_metadata() -> dict[str, object]:
  return {
    "schema": STATEFUL_MATERIAL_DESCRIPTOR_SCHEMA,
    "law": _SECANT_MODEL,
    "stress_state": "plane-strain",
    "parameter_names": [
      "stiffness",
      "characteristic_strain",
      "branch_threshold",
    ],
    "parameter_dtype": "float64",
    "stress_voigt_order": ["xx", "yy", "xy"],
    "strain_shear_convention": "engineering",
    "internal_voigt_order": ["xx", "yy", "zz", "yz", "zx", "xy"],
    "tangent_class": "secant-branch",
    "state_schema": _SECANT_STATE_SCHEMA,
    "state_slots": [
      {
        "name": "max_strain",
        "width": 1,
        "dtype": "float64",
        "lifetime": "accepted-trial",
        "annotation": "envelope-max",
      },
    ],
  }


def _secant_traction(strain: np.ndarray) -> np.ndarray:
  """The mock's Xu-Needleman-flavored saturating exponential traction."""
  return _SECANT_STIFFNESS * _SECANT_E0 * (1.0 - np.exp(-strain / _SECANT_E0))


def _secant_kernel(
  strains: np.ndarray,
  accepted_rows: np.ndarray,
  calibration: np.ndarray,
) -> StatefulContinuumKernelResult:
  """Branch-dependent mock kernel: secant below threshold, tangent above."""
  stiffness = float(calibration[0])
  characteristic = float(calibration[1])
  threshold = float(calibration[2])
  strain = strains[:, 0]
  traction = stiffness * characteristic * (1.0 - np.exp(-strain / characteristic))
  with np.errstate(divide="ignore", invalid="ignore"):
    ratio = traction / strain
  secant = np.where(strain != 0.0, ratio, stiffness)
  tangent = stiffness * np.exp(-strain / characteristic)
  stiffnesses = np.where(np.abs(strain) <= threshold, secant, tangent)
  entity_count = strains.shape[0]
  stresses = np.zeros((entity_count, 6))
  tangents = np.zeros((entity_count, 6, 6))
  stresses[:, 0] = traction
  tangents[:, 0, 0] = stiffnesses
  trial_rows = np.array(accepted_rows, copy=True)
  trial_rows[:, 0] = np.maximum(accepted_rows[:, 0], np.abs(strain))
  return StatefulContinuumKernelResult(
    stresses=stresses,
    tangents=tangents,
    trial_rows=trial_rows,
    status=EvaluationStatus.OK,
  )


@dataclass(frozen=True, slots=True)
class _SecantBinding:
  """The mock law's v2 stateful binding (three calibration parameters)."""

  def __call__(self, *parameters: float) -> np.ndarray:
    if len(parameters) != 3:
      msg = "the secant-branch mock requires exactly three parameters"
      raise TypeError(msg)
    return np.array(parameters, dtype=np.float64)

  def descriptor_metadata(self) -> dict[str, object]:
    return _secant_metadata()

  def kernel(
    self,
    strains: np.ndarray,
    accepted_rows: np.ndarray,
    calibration: np.ndarray,
  ) -> StatefulContinuumKernelResult:
    return _secant_kernel(strains, accepted_rows, calibration)

  def initial_state(
    self,
    parameters: tuple[float, ...],
    layout: object,
  ) -> np.ndarray:
    del parameters
    return np.zeros(layout.row_shape, dtype=np.float64)


def _secant_registry() -> dict:
  registry = dict(q8_reference_registry())
  descriptor = RegistryDescriptor(
    kind="material",
    name=_SECANT_MODEL,
    version="1",
    implementation_id="pyfem-v3-test-mock-secant-branch-v1",
    metadata=_secant_metadata(),
    binding=_SecantBinding(),
  )
  registry[descriptor.key] = descriptor
  return registry


def _secant_model() -> ModelSpec:
  return _continuum_model(
    _SECANT_MODEL,
    (
      ("stiffness", _SECANT_STIFFNESS),
      ("characteristic_strain", _SECANT_E0),
      ("branch_threshold", _SECANT_THRESHOLD),
    ),
  )


def test_secant_branch_kernel_switches_stiffness_branches() -> None:
  calibration = np.array([_SECANT_STIFFNESS, _SECANT_E0, _SECANT_THRESHOLD])
  rows = np.zeros((2, 1))
  strains = np.zeros((2, 6))
  strains[0, 0] = 1.0e-6  # below the branch threshold
  strains[1, 0] = 4.0e-6  # above it
  result = _secant_kernel(strains, rows, calibration)
  assert result.status is EvaluationStatus.OK
  # The traction is branch-independent and identical in both branches.
  strain = strains[:, 0]
  traction = _secant_traction(strain)
  assert np.array_equal(result.stresses[:, 0], traction)
  # Below the threshold the reported stiffness is the secant T(e)/e,
  # materially different from the true tangent E * exp(-e/e0).
  tangent_below = _SECANT_STIFFNESS * np.exp(-strain[0] / _SECANT_E0)
  assert result.tangents[0, 0, 0] == traction[0] / strain[0]
  assert abs(result.tangents[0, 0, 0] - tangent_below) > 1.0e-3 * tangent_below
  # Above the threshold it is the true tangent, materially different from
  # the secant.
  secant_above = traction[1] / strain[1]
  assert result.tangents[1, 0, 0] == tangent_above(strain[1])
  assert abs(result.tangents[1, 0, 0] - secant_above) > 1.0e-3 * secant_above
  # The zero-strain secant limit is the elastic stiffness (virgin probe).
  virgin = _secant_kernel(np.zeros((1, 6)), np.zeros((1, 1)), calibration)
  assert virgin.tangents[0, 0, 0] == _SECANT_STIFFNESS
  assert np.array_equal(virgin.trial_rows, np.zeros((1, 1)))
  # The envelope slot tracks the strain magnitude.
  assert result.trial_rows[1, 0] == strain[1]


def tangent_above(strain: float) -> float:
  return _SECANT_STIFFNESS * float(np.exp(-strain / _SECANT_E0))


def test_secant_branch_descriptor_compiles_with_honest_flags() -> None:
  system = compile_system(_secant_model(), _secant_registry())
  header = system.operators[0].header
  (residual,) = header.residual_channels
  assert residual.linear is False
  (jacobian,) = header.jacobian_channels
  assert jacobian.channel_id == "material-tangent"
  assert jacobian.linear is False
  assert jacobian.symmetric is False
  layout = header.state_layout
  assert layout.schema == f"{_SECANT_STATE_SCHEMA}|max_strain:1"
  (slot,) = layout.slots
  assert slot.name == "max_strain"
  assert slot.annotation == "envelope-max"
  again = compile_system(_secant_model(), _secant_registry())
  assert again.content_fingerprint == system.content_fingerprint


def test_secant_branch_driver_runs_with_fresh_factorizations() -> None:
  driver = _ramp_driver(_secant_model(), _secant_registry())
  assert driver.plan.constant_tangent is False
  base = _eps_point(0.0)
  result = None
  for eps in (1.0e-6, 4.0e-6):  # below, then above the branch threshold
    result = driver.run(base_point=base, target_points=(_eps_point(eps),))
    assert result.status is DriverStatus.COMPLETED
    base = _eps_point(eps)
  assert result is not None
  statistics = result.statistics
  assert statistics.committed_substep_count == 2
  assert statistics.rejected_substep_count == 0
  assert statistics.factorization_reuse_count == 0
  assert statistics.factorization_count == statistics.tangent_refill_count
  assert statistics.factorization_count == statistics.linear_solve_count
  # The committed envelope rows track the homogeneous ramp strain.
  layout = driver.owner.system.operators[0].header.state_layout
  rows = driver.owner.accepted_state(layout.block_id).values
  assert rows.shape == (9, 1)
  np.testing.assert_allclose(rows, np.full((9, 1), 4.0e-6), rtol=1.0e-9, atol=1.0e-12)
