# SPDX-License-Identifier: MIT

"""Instrument-first verification of the fresh crystal-plasticity law.

No legacy implementation is used as an oracle. The constitutive references are
Peirce--Asaro--Needleman (1983), doi:10.1016/0001-6160(83)90014-7 for the
Schmid power law and Kocks--Mecking (2003),
doi:10.1016/S0079-6425(02)00003-8 for saturation hardening. A scalar
single-system return map is solved independently by bisection on its analytical
residual, sympy differentiates the implicit return map, and central finite
differences convict the full multislip tangent at a nonzero mixed state.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

import pyfem.v3.materials.crystal_plasticity as crystal
from pyfem.v3.compile.continuum import crystal_reference_registry
from pyfem.v3.compile.contracts import (
  STATEFUL_MATERIAL_DESCRIPTOR_SCHEMA,
  StatefulContinuumSignalDerivative,
  StatefulContinuumSignalInput,
  resolve_material_state_slots,
  validate_stateful_material_metadata,
)
from pyfem.v3.compile.system import compile_system
from pyfem.v3.constraints import compile_constraint_map
from pyfem.v3.driver import DriverStatus, NonlinearStaticDriver, evaluate_signals
from pyfem.v3.materials.crystal_plasticity import (
  crystal_plasticity_calibration,
  crystal_plasticity_kernel,
  crystal_slip_frames,
  schmid_matrix,
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

# Clean-parsed constants of the CR1 minimal_single_slip deck. Its [110]/[111]
# frame is nonorthogonal, so the analytical case uses the first corrected FCC
# {111}<110> frame instead. No numerical response is inherited from that deck.
_PARAMETERS = (
  70000.0,
  0.33,
  0.0,
  12.0,
  1.0e-3,
  20.0,
  25.0,
  150.0,
  120.0,
  2.0,
  1.0,
  0.5,
  1.0e-6,
)
_ROW_WIDTH = 55
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


def _calibration(**changes: float) -> np.ndarray:
  names = crystal.crystal_plasticity_metadata()["parameter_names"]
  values = dict(zip(names, _PARAMETERS, strict=True))
  values.update(changes)
  return crystal_plasticity_calibration(*(values[name] for name in names))


def _rows() -> np.ndarray:
  rows = np.zeros((1, _ROW_WIDTH), dtype=np.float64)
  rows[0, :12] = _PARAMETERS[6]
  return rows


def _signal(dtime: float) -> StatefulContinuumSignalInput:
  return StatefulContinuumSignalInput(
    port_id="dtime",
    values=np.array([dtime], dtype=np.float64),
    derivatives=(
      StatefulContinuumSignalDerivative("time", np.array([1.0], dtype=np.float64)),
    ),
  )


def _step(
  calibration: np.ndarray,
  rows: np.ndarray,
  strain: np.ndarray,
  dtime: float = 0.1,
) -> object:
  result = crystal_plasticity_kernel(
    np.asarray(strain, dtype=np.float64)[None, :], rows, calibration, (_signal(dtime),)
  )
  assert result.status is EvaluationStatus.OK
  return result


def _single_system_case(tolerance: float = 1.0e-6) -> tuple[np.ndarray, ...]:
  calibration = _calibration(newton_tolerance=tolerance)
  elastic = calibration[10:46].reshape(6, 6)
  normals, directions = crystal_slip_frames(0.0)
  schmid = schmid_matrix(normals, directions)[:, 0].reshape(6, 1)
  row = np.zeros(11, dtype=np.float64)
  row[0] = 25.0
  trial_stress = 35.0 * schmid[:, 0] / float(schmid[:, 0] @ schmid[:, 0])
  strain = np.linalg.solve(elastic, trial_stress)
  result = crystal._crystal_step(
    strain,
    row,
    0.1,
    elastic,
    schmid,
    1.0e-3,
    20.0,
    25.0,
    150.0,
    120.0,
    2.0,
    1.0,
    0.5,
    tolerance,
  )
  assert result is not None
  return elastic, schmid, row, strain, trial_stress, result


def test_complete_cubic_families_and_fail_closed_frames() -> None:
  for structure, plane_count, unique_direction_count in ((0.0, 3, 6), (1.0, 2, 4)):
    normals, directions = crystal_slip_frames(structure)
    assert normals.shape == directions.shape == (12, 3)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1.0e-15)
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0, atol=1.0e-15)
    np.testing.assert_array_equal(np.sum(normals * directions, axis=1), 0.0)
    unique_planes = {tuple(np.round(normal, 14)) for normal in normals}
    assert len(unique_planes) == 12 // plane_count
    counts = {
      plane: np.count_nonzero(np.all(np.isclose(normals, plane), axis=1))
      for plane in unique_planes
    }
    assert set(counts.values()) == {plane_count}
    unique_directions = {tuple(np.round(direction, 14)) for direction in directions}
    assert len(unique_directions) == unique_direction_count
    assert np.linalg.matrix_rank(schmid_matrix(normals, directions)) == 5

  # The broken deck pair must never be silently projected into a Schmid frame.
  normal = np.array([[1.0, 1.0, 0.0]]) / np.sqrt(2.0)
  direction = np.array([[1.0, 1.0, 1.0]]) / np.sqrt(3.0)
  with pytest.raises(crystal.SlipFrameError) as error:
    schmid_matrix(normal, direction)
  assert error.value.code == "nonorthogonal-slip-frame"
  assert "pair 0" in str(error.value) and "dot=" in str(error.value)
  with pytest.raises(ValueError, match="complete 12-system"):
    _calibration(slip_system_count=6.0)


def test_single_slip_analytical_hardening_response() -> None:
  elastic, schmid, _, _, trial_stress, result = _single_system_case()
  stiffness = float((schmid.T @ elastic @ schmid)[0, 0])
  trial_tau = float(schmid[:, 0] @ trial_stress)

  def resistance(slip: float) -> float:
    y0 = 1.0 - 25.0 / 120.0
    y = (y0**-1.0 + 150.0 * slip / 120.0) ** -1.0
    return 25.0 + 95.0 * (1.0 - y / y0)

  def residual(slip: float) -> float:
    tau = trial_tau - stiffness * slip
    tau_c = resistance(slip)
    rate = 1.0e-3 * np.sign(tau) * abs(tau / tau_c) ** 20.0
    return slip - 0.5 * 0.1 * rate

  # The anti-goal is an identically zero initial residual: at the trial state
  # this corrected return map starts far from zero and iterates to the root.
  initial_residual = residual(0.0)
  assert abs(initial_residual) > 1.0e-2
  lower, upper = 0.0, trial_tau / stiffness
  assert residual(lower) < 0.0 < residual(upper)
  for _ in range(200):
    midpoint = 0.5 * (lower + upper)
    if residual(midpoint) > 0.0:
      upper = midpoint
    else:
      lower = midpoint
  slip = 0.5 * (lower + upper)
  tau_c = resistance(slip)
  tau = trial_tau - stiffness * slip
  stress = trial_stress - (elastic @ schmid[:, 0]) * slip
  np.testing.assert_allclose(result[2][1], slip, rtol=1.0e-7, atol=1.0e-12)
  np.testing.assert_allclose(result[2][0], tau_c, rtol=1.0e-9, atol=1.0e-10)
  np.testing.assert_allclose(result[2][2], tau, rtol=1.0e-7, atol=1.0e-8)
  np.testing.assert_allclose(result[0], stress, rtol=1.0e-9, atol=1.0e-6)
  assert result[2][0] > 25.0


def test_sympy_implicit_return_map_derivative() -> None:
  import sympy as sp

  elastic, schmid, _, strain, _, result = _single_system_case(tolerance=1.0e-8)
  c = sp.Matrix(elastic)
  s = sp.Matrix(schmid)
  epsilon = sp.Matrix(sp.symbols("e0:6"))
  slip = sp.symbols("slip")
  trial = c * epsilon
  stiffness = (s.T * c * s)[0]
  tau = (s.T * trial)[0] - stiffness * slip
  y0 = 1.0 - 25.0 / 120.0
  y = (y0**-1.0 + 150.0 * slip / 120.0) ** -1.0
  tau_c = 25.0 + 95.0 * (1.0 - y / y0)
  residual = slip - 0.5 * 0.1 * 1.0e-3 * (tau / tau_c) ** 20
  columns = []
  for component in range(6):
    dslip = -sp.diff(residual, epsilon[component]) / sp.diff(residual, slip)
    columns.append(c[:, component] - c * s * dslip)
  symbolic_tangent = sp.Matrix.hstack(*columns)
  substitutions = {f"e{index}": float(strain[index]) for index in range(6)}
  substitutions[slip] = float(result[2][1])
  numeric = np.asarray(symbolic_tangent.subs(substitutions), dtype=np.float64)
  np.testing.assert_allclose(numeric, result[1], rtol=1.0e-9, atol=1.0e-6)


def _mixed_plastic_probe() -> tuple:
  calibration = _calibration(latent_hardening_ratio=1.4, newton_tolerance=1.0e-8)
  rows = _rows()
  path = (
    np.array([0.0004, -0.0002, 0.0001, 0.0002, -0.0001, 0.0015]),
    np.array([0.0008, -0.0003, 0.0002, 0.0005, -0.0002, 0.0030]),
    np.array([0.0010, 0.0002, -0.0001, 0.0008, 0.0004, 0.0040]),
    np.array([0.0015, 0.0003, -0.0002, -0.0004, 0.0007, 0.0050]),
  )
  for strain in path:
    rows = _step(calibration, rows, strain).trial_rows
  probe = np.array([0.0018, 0.0005, -0.0001, -0.0002, 0.0010, 0.0060])
  return calibration, rows, probe, _step(calibration, rows, probe)


def test_full_tangent_fd_conviction_and_measured_class() -> None:
  calibration, rows, probe, base = _mixed_plastic_probe()
  assert base.trial_rows[0, -1] > rows[0, -1] > 0.0
  step = 1.0e-7
  fd = np.empty((6, 6), dtype=np.float64)
  for component in range(6):
    delta = np.zeros(6)
    delta[component] = step
    plus = _step(calibration, rows, probe + delta).stresses[0]
    minus = _step(calibration, rows, probe - delta).stresses[0]
    fd[:, component] = (plus - minus) / (2.0 * step)
  scale = float(np.max(np.abs(fd)))
  error = float(np.max(np.abs(fd - base.tangents[0]))) / scale
  assert error < 1.0e-7  # measured 2.51e-9 at h=1e-7; 2.25e-10 at h=3e-8
  asymmetry = float(np.max(np.abs(fd - fd.T))) / scale
  assert asymmetry > 1.0e-6  # measured 6.823e-5: algorithmic-nonsymmetric

  elastic = calibration[10:46].reshape(6, 6)
  schmid = calibration[46:].reshape(6, 12)
  old_gamma = rows[0, 12:24]
  increment = base.trial_rows[0, 12:24] - old_gamma
  flow_stiffness = elastic @ schmid
  evaluation = crystal._evaluate_increment(
    increment,
    elastic @ (probe - schmid @ old_gamma),
    flow_stiffness,
    schmid,
    rows[0, 36:48],
    crystal._latent_matrix(12, 1.4),
    1.0e-3,
    20.0,
    25.0,
    150.0,
    120.0,
    2.0,
  )
  jacobian = np.eye(12) + 0.5 * 0.1 * (
    evaluation.rate_resolved[:, None] * (schmid.T @ elastic @ schmid)
    - evaluation.rate_resistance[:, None] * evaluation.resistance_increment
  )
  omitted_factor = elastic - flow_stiffness @ np.linalg.solve(
    jacobian, schmid.T @ elastic
  )
  convicted = float(np.max(np.abs(fd - omitted_factor))) / scale
  assert convicted > 1.0e-3  # measured 2.25e4 at this low-rate, stiff state


def test_elastic_limits_are_exact() -> None:
  calibration = _calibration()
  elastic = calibration[10:46].reshape(6, 6)
  rows = _rows()
  virgin = _step(calibration, rows, np.zeros(6))
  assert np.array_equal(virgin.trial_rows, rows)
  assert np.array_equal(virgin.tangents[0], elastic)

  strain = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0e-4])
  subyield = _step(calibration, rows, strain)
  assert np.array_equal(subyield.stresses[0], elastic @ strain)
  assert np.array_equal(subyield.tangents[0], elastic)
  assert np.array_equal(subyield.trial_rows[0, 12:24], rows[0, 12:24])
  assert np.array_equal(subyield.trial_rows[0, 36:48], rows[0, 36:48])

  explicit_calibration = _calibration(theta=0.0)
  large_strain = 10.0 * strain
  explicit = _step(explicit_calibration, rows, large_strain)
  assert np.array_equal(explicit.stresses[0], elastic @ large_strain)
  assert np.array_equal(explicit.tangents[0], elastic)
  assert np.array_equal(explicit.trial_rows[0, 12:24], rows[0, 12:24])
  accepted = explicit.trial_rows
  following = _step(explicit_calibration, accepted, 1.01 * large_strain)
  old_rate = (
    1.0e-3
    * np.sign(accepted[0, 24:36])
    * np.abs(accepted[0, 24:36] / accepted[0, :12]) ** 20.0
  )
  np.testing.assert_allclose(
    following.trial_rows[0, 12:24] - accepted[0, 12:24], 0.1 * old_rate
  )
  assert np.array_equal(following.tangents[0], elastic)


def test_state_transactions_and_typed_rejections() -> None:
  calibration = _calibration()
  rows = _rows()
  strain = np.array([0.0004, -0.0002, 0.0001, 0.0002, -0.0001, 0.002])
  rows_before = rows.copy()
  strain_before = strain.copy()
  calibration_before = calibration.copy()
  first = _step(calibration, rows, strain)
  assert np.array_equal(rows, rows_before)
  assert np.array_equal(strain, strain_before)
  assert np.array_equal(calibration, calibration_before)

  expected_stress = first.stresses.copy()
  expected_trial = first.trial_rows.copy()
  first.stresses[:] = np.nan
  first.trial_rows[:] = -1.0
  repeated = _step(calibration, rows, strain)
  assert np.array_equal(repeated.stresses, expected_stress)
  assert np.array_equal(repeated.trial_rows, expected_trial)
  assert np.array_equal(rows, rows_before)

  committed = repeated.trial_rows
  advanced = _step(calibration, committed, strain)
  assert advanced.trial_rows[0, -1] > committed[0, -1]
  assert np.array_equal(committed, expected_trial)

  for bad_strain, bad_time in (
    (np.full(6, np.inf), 0.1),
    (strain, np.nan),
    (strain, -0.1),
  ):
    rejected = crystal_plasticity_kernel(
      bad_strain[None, :], committed, calibration, (_signal(bad_time),)
    )
    assert rejected.status is EvaluationStatus.REJECT_STEP
    assert np.array_equal(rejected.trial_rows, committed)
  nonconvergent = calibration.copy()
  nonconvergent[9] = 1.0e-300
  rejected = crystal_plasticity_kernel(
    (20.0 * strain)[None, :], rows, nonconvergent, (_signal(0.1),)
  )
  assert rejected.status is EvaluationStatus.REJECT_STEP
  assert np.array_equal(rejected.trial_rows, rows)


def test_two_system_competition_uses_the_complete_fcc_family() -> None:
  axis = np.array([3.0, 1.0, 0.0])
  axis /= np.linalg.norm(axis)
  stress = 70.0 * np.array(
    [
      axis[0] ** 2,
      axis[1] ** 2,
      axis[2] ** 2,
      axis[1] * axis[2],
      axis[0] * axis[2],
      axis[0] * axis[1],
    ]
  )
  results = {}
  for latent in (1.0, 1.4):
    calibration = _calibration(latent_hardening_ratio=latent, newton_tolerance=1.0e-8)
    elastic = calibration[10:46].reshape(6, 6)
    strain = np.linalg.solve(elastic, stress)
    results[latent] = _step(calibration, _rows(), strain, dtime=0.001)
  increment = results[1.0].trial_rows[0, 12:24] - _rows()[0, 12:24]
  dominant = np.argsort(-np.abs(increment))
  assert set(dominant[:2]) == {8, 11}
  assert abs(increment[dominant[2]]) < 1.0e-2 * abs(increment[dominant[0]])
  np.testing.assert_allclose(increment[8], increment[11], rtol=1.0e-12)
  latent_increment = results[1.4].trial_rows[0, 12:24]
  assert np.all(latent_increment[[8, 11]] < increment[[8, 11]])
  assert np.all(
    results[1.4].trial_rows[0, [8, 11]] > results[1.0].trial_rows[0, [8, 11]]
  )
  assert results[1.0].trial_rows.shape == (1, 4 * 12 + 7)


def test_frozen_descriptor_and_registry_cover_crystal_without_abi_changes() -> None:
  metadata = crystal.crystal_plasticity_metadata()
  assert metadata["schema"] == STATEFUL_MATERIAL_DESCRIPTOR_SCHEMA
  assert validate_stateful_material_metadata(metadata) is metadata
  names = metadata["parameter_names"]
  slots = resolve_material_state_slots(
    metadata["state_slots"], dict(zip(names, _PARAMETERS))
  )
  assert [slot.width for slot in slots] == [12, 12, 12, 12, 6, 1]
  assert sum(slot.width for slot in slots) == 55
  descriptor = crystal_reference_registry()[("material", "crystal-plasticity")]
  assert descriptor.implementation_id == "pyfem-v3-crystal-plasticity-v1"
  assert descriptor.binding.descriptor_metadata()["tangent_class"] == (
    "algorithmic-nonsymmetric"
  )


def _source(label: str) -> SourceContext:
  return SourceContext(source=label)


def _crystal_model() -> ModelSpec:
  nodes = tuple(
    NodeSpec(id=index + 1, coordinates=point, source=_source(f"node-{index + 1}"))
    for index, point in enumerate(_UNIT_COORDINATES)
  )
  cell = CellSpec(
    id="cell-1", node_ids=tuple(node.id for node in nodes), source=_source("cell")
  )
  names = crystal.crystal_plasticity_metadata()["parameter_names"]
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
          source=_source("block"),
        ),
      ),
      source=_source("mesh"),
    ),
    fields=(
      FieldSpec(
        id="displacement",
        components=("x", "y"),
        location="node",
        source=_source("field"),
      ),
    ),
    materials=(
      MaterialSpec(
        id="crystal",
        model="crystal-plasticity",
        parameters=tuple(
          MaterialParameterSpec(name, value) for name, value in zip(names, _PARAMETERS)
        ),
        source=_source("material"),
      ),
    ),
    regions=(
      RegionSpec(
        id="domain",
        cell_refs=(CellRef("cells", "cell-1"),),
        field_ids=("displacement",),
        material_id="crystal",
        formulation="small-strain-continuum",
        quadrature="gauss-3x3",
        source=_source("region"),
      ),
    ),
    source=_source("model"),
  )


def _point(time: float, eps: float) -> ProgramPoint:
  return ProgramPoint(
    (ProgramCoordinateValue("time", time), ProgramCoordinateValue("eps", eps))
  )


def _crystal_driver() -> NonlinearStaticDriver:
  system = compile_system(_crystal_model(), crystal_reference_registry())
  constraints = []
  for index, node in enumerate(_UNIT_COORDINATES):
    node_id = index + 1
    if node_id != 6:
      constraints.append(
        PrescribedDofSpec(
          id=f"ux-{node_id}",
          target=DofRef(node_id=node_id, field_id="displacement", component="x"),
          value=AffineValueSpec(coefficients=(AffineCoefficientSpec("eps", node[0]),)),
          source=_source(f"ux-{node_id}"),
        )
      )
    constraints.append(
      PrescribedDofSpec(
        id=f"uy-{node_id}",
        target=DofRef(node_id=node_id, field_id="displacement", component="y"),
        value=AffineValueSpec(constant=0.0),
        source=_source(f"uy-{node_id}"),
      )
    )
  coordinate_map = compile_constraint_map(
    system,
    constraints=tuple(constraints),
    coordinates=(
      ProgramCoordinateSpec(name="time", kind="time"),
      ProgramCoordinateSpec(name="eps", kind="load"),
    ),
  )
  return NonlinearStaticDriver(system, coordinate_map, ())


def _committed_ip_strains(driver: NonlinearStaticDriver) -> np.ndarray:
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


def test_compiled_layout_derived_dtime_and_driver_commit() -> None:
  driver = _crystal_driver()
  operator = driver.owner.system.operators[0]
  layout = operator.header.state_layout
  assert layout.row_shape == (9, 55)
  assert layout.initial_rows is not None
  np.testing.assert_array_equal(layout.initial_rows.values[:, :12], 25.0)
  np.testing.assert_array_equal(layout.initial_rows.values[:, 12:], 0.0)
  (channel,) = operator.header.jacobian_channels
  assert channel.linear is False and channel.symmetric is False
  (port,) = driver.plan.signal_slices[0]
  assert port.port_id == port.signal_id == "dtime"
  assert port.binding == "increment"
  target = _point(0.35, 0.004)
  signal = evaluate_signals(driver.plan, target, _point(0.1, 0.003))[0][0]
  np.testing.assert_allclose(signal.values.values, [0.25], rtol=0.0, atol=1.0e-16)

  calibration = _calibration()
  oracle_rows = np.zeros((9, 55), dtype=np.float64)
  oracle_rows[:, :12] = 25.0
  base = _point(0.0, 0.0)
  previous_time = 0.0
  for time, eps in ((0.1, 0.003), (0.35, 0.004)):
    result = driver.run(base_point=base, target_points=(_point(time, eps),))
    assert result.status is DriverStatus.COMPLETED
    oracle = crystal_plasticity_kernel(
      _committed_ip_strains(driver),
      oracle_rows,
      calibration,
      (_signal(time - previous_time),),
    )
    assert oracle.status is EvaluationStatus.OK
    oracle_rows = oracle.trial_rows
    accepted = driver.owner.accepted_state(layout.block_id).values
    np.testing.assert_array_equal(accepted, oracle_rows)
    assert np.all(accepted[:, -1] > 0.0)
    base = _point(time, eps)
    previous_time = time
