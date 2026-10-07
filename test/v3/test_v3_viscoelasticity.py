# SPDX-License-Identifier: MIT

"""Prony-series viscoelasticity: legacy parity oracles and the time-port path.

The legacy law ``pyfem/materials/ViscoElasticity.py`` is the numerical
reference (AGENTS.md). The parity harness drives it with the
``dstrain = eps_total - eps_committed`` identity the v3 kernel uses — the same
float64 subtraction: the kernel reads its committed total from the ``epsilon``
state slot — and sets ``solverStat.time`` per committed step, the hidden
channel the v3 law replaces with the declared identity signal port. Because
the kernel replicates the legacy ``getStress`` arithmetic statement for
statement and both sides run identical NumPy operations in identical order,
every law-level comparison is bitwise on every platform. The one deliberate
divergence is the tangent: the legacy accumulation (ViscoElasticity.py:213-214)
contradicts the legacy stress update, so the v3 kernel writes the true
algorithmic tangent ``Cinf * (1 + sum f_i a_i)`` and this battery pins the
divergence explicitly (the module docstring carries the finite-difference and
Newton-convergence evidence).

Documented time-dependent paths: a relaxation jump-then-hold schedule, a
creep-like strain-time ramp over six strain components, a stepped ramp with
holds and a partial unload, a constant-time pair plus a backward-time step
(the legacy ``dtime > 0`` guard at ViscoElasticity.py:192, pinned as oracle
behavior), and the shipped example deck configuration
(``examples/materials/viscoelasticity/creep_test.pro``: E = 1000, nu = 0.3,
Einf = 100, three terms, log-spaced times 0.1 .. 10, equal moduli 300).

The driver battery proves schedule-owned time flows through the M29 identity
signal port: committed state rows equal the kernel oracle stepped on the
committed integration-point strains (bitwise), the ``time`` slot records the
substep's bound time at every integration point, the response changes when the
same strain path runs under a different time schedule (positive control), and
a port-free law driven under two different time schedules yields byte
-identical results (isolation: without a declared port no schedule value
reaches the law). The ABI proof is the frozen validator accepting the law's
v2 metadata unchanged — parameterized width and signal ports included — plus
``pyfem/v3/compile/contracts.py`` carrying no diff for this mission.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from pyfem.materials.ViscoElasticity import ViscoElasticity
from pyfem.v3.compile.continuum import (
  q8_reference_registry,
  viscoelasticity_reference_registry,
)
from pyfem.v3.compile.contracts import (
  STATEFUL_MATERIAL_DESCRIPTOR_SCHEMA,
  StatefulContinuumKernelResult,
  StatefulContinuumSignalDerivative,
  StatefulContinuumSignalInput,
  resolve_material_state_slots,
  validate_stateful_material_metadata,
)
from pyfem.v3.compile.system import compile_system
from pyfem.v3.constraints import compile_constraint_map
from pyfem.v3.driver import (
  DriverStatus,
  NonlinearStaticDriver,
  evaluate_signals,
)
from pyfem.v3.materials.prony_viscoelasticity import (
  PRONY_VISCOELASTICITY_BINDING,
  prony_viscoelastic_initial_state,
  prony_viscoelasticity_calibration,
  prony_viscoelasticity_kernel,
  prony_viscoelasticity_metadata,
)
from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.operator import (
  ChannelRequest,
  EvaluationStatus,
  OperatorEvaluationInput,
  ProgramSignalInput,
  SignalDerivativeInput,
)
from pyfem.v3.model.provenance import CanonicalManifest
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
  AffineCoefficientSpec,
  AffineValueSpec,
  DofRef,
  PrescribedDofSpec,
  ProgramCoordinateSpec,
  ProgramCoordinateValue,
  ProgramPoint,
)
from pyfem.v3.state import StateCodecError

# Documented oracle configuration: the shipped example deck
# (examples/materials/viscoelasticity/creep_test.pro).
_E = 1000.0
_NU = 0.3
_EINF = 100.0
_T_FIRST = 0.1
_T_LAST = 10.0

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


def _source(label: str) -> SourceContext:
  return SourceContext(source=label)


class _LegacyProps:
  """Minimal iterable props stand-in for ``BaseMaterial``."""

  def __init__(self, **values: object) -> None:
    self._values = values
    self.solverStat = None
    for name, value in values.items():
      setattr(self, name, value)

  def __iter__(self) -> object:
    return iter(self._values.items())


def _legacy_law(term_count: int) -> ViscoElasticity:
  """The legacy oracle on the descriptor's equal-moduli, log-spaced family."""
  term_modulus = (_E - _EINF) / term_count
  times = list(np.logspace(np.log10(_T_FIRST), np.log10(_T_LAST), term_count))
  with contextlib.redirect_stdout(io.StringIO()):
    with warnings.catch_warnings(action="ignore", category=DeprecationWarning):
      return ViscoElasticity(
        _LegacyProps(
          E=_E,
          nu=_NU,
          Einf=_EINF,
          nMaxwell=term_count,
          relaxTimes=times,
          relaxModuli=[term_modulus] * term_count,
        )
      )


def _legacy_step(
  mat: ViscoElasticity,
  dstrain: np.ndarray,
  time_new: float,
) -> tuple[np.ndarray, np.ndarray]:
  """Advance the legacy oracle one committed step; time rides solverStat."""
  mat.solverStat = SimpleNamespace(time=time_new)
  with contextlib.redirect_stdout(io.StringIO()):
    with warnings.catch_warnings(action="ignore", category=DeprecationWarning):
      sigma, tangent = mat.getStress(SimpleNamespace(dstrain=dstrain))
  mat.commitHistory()
  return np.array(sigma, copy=True), np.array(tangent, copy=True)


def _legacy_eps_i(mat: ViscoElasticity, term_count: int) -> np.ndarray:
  return np.concatenate(
    [
      np.atleast_1d(mat.getHistoryParameter(f"eps_i_{index}"))
      for index in range(term_count)
    ]
  )


def _time_signal(time: float) -> StatefulContinuumSignalInput:
  return StatefulContinuumSignalInput(
    port_id="time",
    values=np.array([time], dtype=np.float64),
    derivatives=(
      StatefulContinuumSignalDerivative("time", np.array([1.0], dtype=np.float64)),
    ),
  )


def _program_time_signal(time: float) -> ProgramSignalInput:
  return ProgramSignalInput(
    port_id="time",
    values=FinalizedArray(np.array([time], dtype=np.float64), dtype=np.float64),
    derivatives=(
      SignalDerivativeInput(
        "time",
        FinalizedArray(np.array([1.0], dtype=np.float64), dtype=np.float64),
      ),
    ),
  )


def _v3_step(
  calibration: np.ndarray,
  rows: np.ndarray,
  strain_total: np.ndarray,
  time_new: float,
) -> StatefulContinuumKernelResult:
  result = prony_viscoelasticity_kernel(
    strain_total[None, :], rows, calibration, (_time_signal(time_new),)
  )
  assert result.status is EvaluationStatus.OK
  return result


def _expected_tangents(
  calibration: np.ndarray, term_count: int, dtime: float
) -> tuple[np.ndarray, np.ndarray]:
  """Hand-derived tangents in each law's own arithmetic, per-term order.

  The v3 kernel accumulates the true algorithmic tangent ``Cinf * (1 + sum
  f_i a_i)``; the legacy oracle accumulates ``Cinf * (1 + sum f_i (1 - a_i))``
  (ViscoElasticity.py:213-214) — the pinned, deliberate divergence (module
  docstring): both are exact Cinf when dtime <= 0.
  """
  cinf = calibration[1 + 2 * term_count :].reshape(6, 6)
  factors = calibration[1 : 1 + term_count]
  times = calibration[1 + term_count : 1 + 2 * term_count]
  true_tangent = cinf.copy()
  legacy_tangent = cinf.copy()
  if dtime > 0.0:
    for term in range(term_count):
      exp_factor = np.exp(-dtime / times[term])
      true_tangent += (factors[term] * exp_factor) * cinf
      legacy_tangent += (factors[term] * (1.0 - exp_factor)) * cinf
  return true_tangent, legacy_tangent


def _run_path_parity(term_count: int, path: list) -> tuple[np.ndarray, list]:
  """Step both laws along one (time, total-strain) path, bitwise everywhere."""
  calibration = prony_viscoelasticity_calibration(
    _E, _NU, _EINF, float(term_count), _T_FIRST, _T_LAST
  )
  legacy = _legacy_law(term_count)
  rows_v = np.zeros((1, 6 * term_count + 13))
  previous = np.zeros(6)
  previous_time = 0.0
  stresses = []
  for time_new, strain in path:
    strain_total = np.array(strain, dtype=np.float64)
    dstrain = strain_total - previous
    sigma_l, tangent_l = _legacy_step(legacy, dstrain, time_new)
    result = _v3_step(calibration, rows_v, strain_total, time_new)
    assert np.array_equal(result.stresses[0], sigma_l)
    true_tangent, legacy_tangent = _expected_tangents(
      calibration, term_count, time_new - previous_time
    )
    # Stress and state bookkeeping are bitwise-identical to legacy; the
    # tangent pins the documented divergence: v3 is the true algorithmic
    # tangent, legacy carries its own (1 - a) arithmetic, and the two differ
    # materially on every time-advancing substep.
    assert np.array_equal(result.tangents[0], true_tangent)
    assert np.array_equal(tangent_l, legacy_tangent)
    if time_new - previous_time > 0.0:
      assert not np.array_equal(result.tangents[0], tangent_l)
    row = result.trial_rows[0]
    assert np.array_equal(row[: 6 * term_count], _legacy_eps_i(legacy, term_count))
    assert np.array_equal(
      row[6 * term_count : 6 * term_count + 6],
      np.atleast_1d(legacy.getHistoryParameter("sigma")),
    )
    assert np.array_equal(row[6 * term_count + 6 : 6 * term_count + 12], strain_total)
    assert row[6 * term_count + 12] == time_new
    assert legacy.getHistoryParameter("time_old") == time_new
    rows_v = result.trial_rows
    previous = strain_total
    previous_time = time_new
    stresses.append(sigma_l)
  return rows_v, stresses


def _relaxation_path() -> list:
  jump = np.array([5.0e-4, 0.0, 0.0, 0.0, 0.0, 0.0])
  hold = jump
  return [
    (0.05, jump),
    (0.10, hold),
    (0.35, hold),
    (1.2, hold),
    (5.0, hold),
    (25.0, hold),
    (200.0, hold),
  ]


def _creep_ramp_path() -> list:
  return [
    (0.10, [1.0e-4, 0.0, 0.0, 0.0, 0.0, 0.0]),
    (0.35, [2.0e-4, 0.0, 0.0, 0.0, 0.0, 1.0e-4]),
    (0.90, [4.0e-4, 5.0e-5, 0.0, 0.0, 0.0, 2.5e-4]),
    (2.50, [7.0e-4, 1.0e-4, 2.0e-5, 0.0, 0.0, 4.0e-4]),
    (8.00, [1.1e-3, 1.5e-4, 3.0e-5, 1.0e-5, 0.0, 6.0e-4]),
  ]


def _stepped_ramp_path() -> list:
  return [
    (0.05, [2.0e-4, 0.0, 0.0, 0.0, 0.0, 0.0]),
    (0.08, [6.0e-4, 0.0, 0.0, 0.0, 0.0, 0.0]),
    (0.50, [6.0e-4, 0.0, 0.0, 0.0, 0.0, 0.0]),
    (0.55, [1.2e-3, 0.0, 0.0, 0.0, 0.0, 0.0]),
    (1.50, [1.2e-3, 0.0, 0.0, 0.0, 0.0, 0.0]),
    (2.00, [9.0e-4, 0.0, 0.0, 0.0, 0.0, 0.0]),
  ]


def test_single_step_paths_match_legacy_bitwise() -> None:
  for term_count in (1, 2, 3, 5):
    for time_new, strain in (
      (0.05, [1.0e-4, 0.0, 0.0, 0.0, 0.0, 0.0]),
      (0.7, [4.0e-4, 1.0e-4, 2.0e-5, 0.0, 0.0, 3.0e-4]),
      (3.0, [1.0e-3, -2.0e-4, 0.0, 0.0, 0.0, -1.0e-4]),
    ):
      _run_path_parity(term_count, [(time_new, strain)])


def test_relaxation_paths_match_legacy_bitwise() -> None:
  for term_count in (1, 2, 3, 5):
    rows, stresses = _run_path_parity(term_count, _relaxation_path())
    # Relaxation physics of the legacy update: the stress decreases
    # monotonically over the holds and converges to the legacy equilibrium,
    # while the internal strains decay to zero at constant total strain.
    sigma_xx = [stress[0] for stress in stresses]
    for earlier, later in zip(sigma_xx, sigma_xx[1:]):
      assert later <= earlier
    assert sigma_xx[1] < sigma_xx[0]
    np.testing.assert_allclose(sigma_xx[-1], sigma_xx[-2], rtol=0.0, atol=1.0e-11)
    assert np.max(np.abs(rows[0, : 6 * term_count])) < 1.0e-6


def test_creep_ramp_matches_legacy_bitwise() -> None:
  _run_path_parity(3, _creep_ramp_path())


def test_stepped_ramp_with_holds_and_unload_matches_legacy_bitwise() -> None:
  _run_path_parity(3, _stepped_ramp_path())
  _run_path_parity(2, _stepped_ramp_path())


def test_constant_time_and_backward_time_branches_match_legacy_bitwise() -> None:
  # The legacy ``dtime > 0`` guard (ViscoElasticity.py:192): a constant-time
  # or backward-time substep adds only the long-term elastic increment and
  # freezes the internal strains. Pinned as oracle behavior; the driver never
  # produces such steps from monotone schedules.
  _run_path_parity(
    3,
    [
      (0.2, [1.0e-4, 0.0, 0.0, 0.0, 0.0, 0.0]),
      (0.2, [3.0e-4, 0.0, 0.0, 0.0, 0.0, 0.0]),
      (0.6, [3.0e-4, 0.0, 0.0, 0.0, 0.0, 0.0]),
      (0.4, [5.0e-4, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ],
  )


def test_tangent_is_the_exact_algorithmic_derivative_by_fd() -> None:
  # Given the committed state, the response is affine in the trial strain, so
  # the finite difference is exact up to rounding. This is the property the
  # legacy tangent fails (module docstring: 0.73 relative error against the
  # same finite difference on the legacy law at dtime = 0.05).
  term_count = 3
  calibration = prony_viscoelasticity_calibration(
    _E, _NU, _EINF, float(term_count), _T_FIRST, _T_LAST
  )
  eps0 = np.array([4.0e-4, 1.0e-4, 2.0e-5, 0.0, 0.0, 3.0e-4])
  rows = np.zeros((1, 6 * term_count + 13))
  rows = _v3_step(calibration, rows, 0.5 * eps0, 0.2).trial_rows
  base = _v3_step(calibration, rows, eps0, 0.7)
  step = 1.0e-7
  for component in range(6):
    moved = _v3_step(
      calibration, rows, eps0 + step * np.eye(6)[component], 0.7
    )
    np.testing.assert_allclose(
      (moved.stresses[0] - base.stresses[0]) / step,
      base.tangents[0][:, component],
      rtol=1.0e-6,
      atol=1.0e-6,
    )


def test_example_deck_configuration_relaxes_toward_equilibrium() -> None:
  # The creep_test.pro material: three terms, moduli 300 each, times 0.1..10.
  calibration = prony_viscoelasticity_calibration(
    _E, _NU, _EINF, 3.0, _T_FIRST, _T_LAST
  )
  assert calibration[1] == 3.0  # factor 300 / 100 exactly
  np.testing.assert_array_equal(
    calibration[4:7],
    np.logspace(np.log10(_T_FIRST), np.log10(_T_LAST), 3),
  )
  rows, stresses = _run_path_parity(3, _relaxation_path())
  # The jump step is hand-derived: sigma = (1 + sum_i f_i a_i) Cinf : eps
  # with a_i = exp(-0.05 / tau_i) on the first (t = 0.05) substep.
  cinf = calibration[7:].reshape(6, 6)
  factors = calibration[1:4]
  times = calibration[4:7]
  first_step_factor = 1.0 + float(np.sum(factors * np.exp(-0.05 / times)))
  np.testing.assert_allclose(
    stresses[0],
    first_step_factor * (cinf @ np.array([5.0e-4, 0, 0, 0, 0, 0])),
    rtol=1.0e-12,
    atol=1.0e-12,
  )
  # The legacy update relaxes toward, and reaches, its own equilibrium.
  sigma_xx = [stress[0] for stress in stresses]
  for earlier, later in zip(sigma_xx, sigma_xx[1:]):
    assert later <= earlier
  assert sigma_xx[1] < sigma_xx[0]
  np.testing.assert_allclose(sigma_xx[-1], sigma_xx[-2], rtol=0.0, atol=1.0e-11)
  assert np.max(np.abs(rows[0, :18])) < 1.0e-6


def test_calibration_validation_is_strict() -> None:
  with pytest.raises(TypeError, match="exact float"):
    prony_viscoelasticity_calibration(1000, _NU, _EINF, 3.0, _T_FIRST, _T_LAST)
  with pytest.raises(ValueError, match="must be finite"):
    prony_viscoelasticity_calibration(np.nan, _NU, _EINF, 3.0, _T_FIRST, _T_LAST)
  with pytest.raises(ValueError, match="youngs_modulus must be positive"):
    prony_viscoelasticity_calibration(-1.0, _NU, _EINF, 3.0, _T_FIRST, _T_LAST)
  with pytest.raises(ValueError, match="between -1 and 0.5"):
    prony_viscoelasticity_calibration(_E, 0.5, _EINF, 3.0, _T_FIRST, _T_LAST)
  with pytest.raises(ValueError, match="equilibrium_modulus"):
    prony_viscoelasticity_calibration(_E, _NU, 0.0, 3.0, _T_FIRST, _T_LAST)
  with pytest.raises(ValueError, match="equilibrium_modulus"):
    prony_viscoelasticity_calibration(_E, _NU, _E, 3.0, _T_FIRST, _T_LAST)
  with pytest.raises(ValueError, match="positive integer"):
    prony_viscoelasticity_calibration(_E, _NU, _EINF, 2.5, _T_FIRST, _T_LAST)
  with pytest.raises(ValueError, match="positive integer"):
    prony_viscoelasticity_calibration(_E, _NU, _EINF, 0.0, _T_FIRST, _T_LAST)
  with pytest.raises(ValueError, match="relaxation_time_first must be positive"):
    prony_viscoelasticity_calibration(_E, _NU, _EINF, 3.0, 0.0, _T_LAST)
  with pytest.raises(ValueError, match="must not precede"):
    prony_viscoelasticity_calibration(_E, _NU, _EINF, 3.0, _T_LAST, _T_FIRST)
  with pytest.raises(ValueError, match="non-empty interval"):
    prony_viscoelasticity_calibration(_E, _NU, _EINF, 3.0, 1.0, 1.0)
  # A single term is the legacy logspace(start, stop, 1) convention: the term
  # sits at the first endpoint and equal endpoints are legal.
  single = prony_viscoelasticity_calibration(_E, _NU, _EINF, 1.0, 0.5, 0.5)
  assert single[0] == 1.0
  assert single[2] == np.logspace(np.log10(0.5), np.log10(0.5), 1)[0]
  with pytest.raises(TypeError, match="exactly six"):
    PRONY_VISCOELASTICITY_BINDING(_E, _NU, _EINF)


def test_kernel_reports_typed_failures() -> None:
  calibration = prony_viscoelasticity_calibration(
    _E, _NU, _EINF, 3.0, _T_FIRST, _T_LAST
  )
  rows = np.zeros((1, 31))
  signals = (_time_signal(0.5),)
  result = prony_viscoelasticity_kernel(
    np.array([[np.inf, 0.0, 0.0, 0.0, 0.0, 0.0]]), rows, calibration, signals
  )
  assert result.status is EvaluationStatus.REJECT_STEP
  assert np.array_equal(result.trial_rows, rows)
  result = prony_viscoelasticity_kernel(
    np.zeros((1, 6)), rows, calibration, (_time_signal(np.nan),)
  )
  assert result.status is EvaluationStatus.REJECT_STEP
  assert np.array_equal(result.trial_rows, rows)
  with pytest.raises(TypeError, match="calibration"):
    prony_viscoelasticity_kernel(np.zeros((1, 6)), rows, np.zeros(3), signals)
  with pytest.raises(ValueError, match="strains and"):
    prony_viscoelasticity_kernel(np.zeros((1, 3)), rows, calibration, signals)
  with pytest.raises(TypeError, match="time port"):
    prony_viscoelasticity_kernel(np.zeros((1, 6)), rows, calibration, ())
  wrong_port = StatefulContinuumSignalInput("load", np.array([0.5]), ())
  with pytest.raises(TypeError, match="time port"):
    prony_viscoelasticity_kernel(np.zeros((1, 6)), rows, calibration, (wrong_port,))


def test_frozen_v2_schema_covers_the_law_without_new_fields() -> None:
  metadata = prony_viscoelasticity_metadata()
  assert metadata["schema"] == STATEFUL_MATERIAL_DESCRIPTOR_SCHEMA
  assert STATEFUL_MATERIAL_DESCRIPTOR_SCHEMA == "pyfem-v3-material-descriptor-v2"
  # The M25/M29 freeze accepts the law as-is: parameterized widths and the
  # optional signal_ports field are already in the frozen v2 field set.
  assert validate_stateful_material_metadata(metadata) is metadata
  parameters = {
    "youngs_modulus": _E,
    "poisson_ratio": _NU,
    "equilibrium_modulus": _EINF,
    "prony_term_count": 3.0,
    "relaxation_time_first": _T_FIRST,
    "relaxation_time_last": _T_LAST,
  }
  for term_count in (1.0, 2.0, 3.0, 5.0):
    slots = resolve_material_state_slots(
      metadata["state_slots"], {**parameters, "prony_term_count": term_count}
    )
    assert [slot.name for slot in slots] == ["eps_i", "sigma", "epsilon", "time"]
    assert sum(slot.width for slot in slots) == 6 * int(term_count) + 13


def _visco_model(term_count: float = 3.0) -> ModelSpec:
  nodes = tuple(
    NodeSpec(
      id=index + 1,
      coordinates=point,
      source=_source(f"node-{index + 1}"),
    )
    for index, point in enumerate(_UNIT_COORDINATES)
  )
  cell = CellSpec(
    id="cell-1",
    node_ids=tuple(node.id for node in nodes),
    source=_source("cell"),
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
        id="polymer",
        model="prony-viscoelasticity",
        parameters=(
          MaterialParameterSpec("youngs_modulus", _E),
          MaterialParameterSpec("poisson_ratio", _NU),
          MaterialParameterSpec("equilibrium_modulus", _EINF),
          MaterialParameterSpec("prony_term_count", term_count),
          MaterialParameterSpec("relaxation_time_first", _T_FIRST),
          MaterialParameterSpec("relaxation_time_last", _T_LAST),
        ),
        source=_source("material"),
      ),
    ),
    regions=(
      RegionSpec(
        id="domain",
        cell_refs=(CellRef("cells", "cell-1"),),
        field_ids=("displacement",),
        material_id="polymer",
        formulation="small-strain-continuum",
        quadrature="gauss-3x3",
        source=_source("region"),
      ),
    ),
    source=_source("model"),
  )


def _elastic_model() -> ModelSpec:
  """The same one-element mesh with the port-free plane-stress linear law."""
  base = _visco_model()
  return ModelSpec(
    mesh=base.mesh,
    fields=base.fields,
    materials=(
      MaterialSpec(
        id="steel",
        model="plane-stress-linear-elastic",
        parameters=(
          MaterialParameterSpec("youngs_modulus", _E),
          MaterialParameterSpec("poisson_ratio", _NU),
        ),
        source=_source("material"),
      ),
    ),
    regions=base.regions[:-1]
    + (
      RegionSpec(
        id="domain",
        cell_refs=(CellRef("cells", "cell-1"),),
        field_ids=("displacement",),
        material_id="steel",
        formulation="small-strain-continuum",
        quadrature="gauss-3x3",
        source=_source("region"),
      ),
    ),
    source=_source("model"),
  )


def _compiled(model: ModelSpec, registry: dict) -> CompiledSystem:
  return compile_system(model, registry)


def _visco_compiled(term_count: float = 3.0) -> CompiledSystem:
  return _compiled(_visco_model(term_count), viscoelasticity_reference_registry())


def test_parameterized_state_width_resolves_per_term_count() -> None:
  for term_count, width in ((1, 19), (2, 25), (3, 31), (5, 43)):
    layout = _visco_compiled(float(term_count)).operators[0].header.state_layout
    assert layout.entity_count == 9
    assert layout.row_width == width == 6 * term_count + 13
    assert layout.schema == (
      "pyfem-v3-prony-viscoelastic-state-v1"
      f"|eps_i:{6 * term_count},sigma:6,epsilon:6,time:1"
    )
    assert layout.initial_rows is not None
    assert np.all(layout.initial_rows.values == 0.0)
  # Deterministic content identity across compilations.
  assert (
    _visco_compiled().content_fingerprint == _visco_compiled().content_fingerprint
  )


def test_initial_state_binding_is_explicit_and_validated() -> None:
  layout = _visco_compiled().operators[0].header.state_layout
  parameters = (_E, _NU, _EINF, 3.0, _T_FIRST, _T_LAST)
  rows = prony_viscoelastic_initial_state(parameters, layout)
  assert rows.shape == layout.row_shape
  assert np.all(rows == 0.0)
  with pytest.raises(TypeError, match="six parameters"):
    prony_viscoelastic_initial_state(parameters[:5], layout)
  with pytest.raises(ValueError, match="positive integer"):
    prony_viscoelastic_initial_state((_E, _NU, _EINF, 2.5, _T_FIRST, _T_LAST), layout)
  with pytest.raises(ValueError, match="row layout"):
    prony_viscoelastic_initial_state((_E, _NU, _EINF, 2.0, _T_FIRST, _T_LAST), layout)


def _decode_node(node: object) -> object:
  if type(node) is not list or not node:
    return node
  tag = node[0]
  if tag == "mapping":
    return {pair[0]: _decode_node(pair[1]) for pair in node[1]}
  if tag == "sequence":
    return [_decode_node(item) for item in node[1]]
  if tag == "str":
    return node[1]
  return node


def _manifest_content(manifest: CanonicalManifest) -> dict[str, object]:
  payload = manifest.to_bytes().split(b"\n", 1)[1]
  node = json.loads(payload)
  if type(node) is not list or not node or node[0] != "mapping":
    msg = "compiled manifests must decode to a canonical mapping"
    raise TypeError(msg)
  return {pair[0]: pair[1] for pair in node[1]}


def test_compiled_operator_versions_ports_state_and_manifest() -> None:
  operator = _visco_compiled().operators[0]
  header = operator.header
  assert len(header.signal_ports) == 1
  port = header.signal_ports[0]
  assert port.port_id == "time"
  assert port.signal_id == "time"
  assert port.derivative_coordinate_ids == ("time",)
  (channel,) = header.jacobian_channels
  assert channel.linear is False
  assert channel.symmetric is True
  manifest = _manifest_content(operator.content_manifest)
  assert _decode_node(manifest["signal_ports"]) == [
    {
      "port_id": "time",
      "signal_id": "time",
      "derivative_coordinate_ids": ["time"],
    }
  ]
  state = _decode_node(manifest["state"])
  assert state["schema"] == (
    "pyfem-v3-prony-viscoelastic-state-v1|eps_i:18,sigma:6,epsilon:6,time:1"
  )
  assert [slot["name"] for slot in state["slots"]] == [
    "eps_i",
    "sigma",
    "epsilon",
    "time",
  ]
  assert [slot["width"] for slot in state["slots"]] == [
    ["int", "18"],
    ["int", "6"],
    ["int", "6"],
    ["int", "1"],
  ]


def test_compiled_operator_binds_the_time_port_exactly() -> None:
  operator = _visco_compiled().operators[0]
  values = FinalizedArray(np.zeros((1, 16)), dtype=np.float64)
  accepted = FinalizedArray(
    np.zeros(operator.header.state_layout.row_shape), dtype=np.float64
  )
  request = ChannelRequest(("internal-force",), ("material-tangent",))

  def evaluate(signals: tuple) -> object:
    return operator.evaluate(
      OperatorEvaluationInput(
        port_values=(values,),
        accepted_state=accepted,
        signals=signals,
        request=request,
      )
    )

  result = evaluate((_program_time_signal(0.25),))
  assert result.status is EvaluationStatus.OK
  assert result.trial_state.values.shape == (9, 31)
  np.testing.assert_array_equal(result.trial_state.values[:, 30], 0.25)
  with pytest.raises(ValueError, match="missing a declared program signal port"):
    evaluate(())
  with pytest.raises(ValueError, match="undeclared program signal port"):
    evaluate(
      (
        _program_time_signal(0.25),
        ProgramSignalInput(
          port_id="temperature",
          values=FinalizedArray(np.array([1.0], dtype=np.float64), dtype=np.float64),
          derivatives=(),
        ),
      )
    )


def _ramp_driver(model: ModelSpec, registry: dict) -> NonlinearStaticDriver:
  """One Q8 element under a prescribed affine eps_xx ramp, one free DOF.

  All nodal displacements follow the homogeneous strain field exactly
  (``u_x = eps * x``, ``u_y = 0``) except node 6's x component, which stays
  free so every Newton iteration assembles and factorizes the tangent. The
  program declares the schedule-owned ``time`` coordinate alongside ``eps``.
  """
  system = _compiled(model, registry)
  constraints = []
  for index, node in enumerate(_UNIT_COORDINATES):
    node_id = index + 1
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


def _visco_driver() -> NonlinearStaticDriver:
  return _ramp_driver(_visco_model(), viscoelasticity_reference_registry())


def _point(time: float, eps: float) -> ProgramPoint:
  return ProgramPoint(
    (
      ProgramCoordinateValue("time", time),
      ProgramCoordinateValue("eps", eps),
    )
  )


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


def test_plan_forwards_the_identity_time_signal() -> None:
  driver = _visco_driver()
  plan = driver.plan
  assert plan.constant_tangent is False
  assert not plan.requires_committed_point
  assert len(plan.signal_slices) == 1 and len(plan.signal_slices[0]) == 1
  port = plan.signal_slices[0][0]
  assert port.port_id == "time"
  assert port.signal_id == "time"
  assert port.binding == "identity"
  names = plan.coordinate_names
  assert port.signal_coordinate_index == names.index("time")
  assert port.derivative_coordinate_indices == (names.index("time"),)
  forwarded = evaluate_signals(plan, _point(0.3, 2.0e-4))
  signal = forwarded[0][0]
  np.testing.assert_array_equal(signal.values.values, [0.3])
  assert tuple(item.coordinate_id for item in signal.derivatives) == ("time",)
  np.testing.assert_array_equal(
    [item.values.values[0] for item in signal.derivatives], [1.0]
  )
  signals = _decode_node(_manifest_content(plan.provenance.manifest)["signals"])
  assert signals[0]["ports"] == [
    {"port_id": "time", "signal_coordinate": "time", "derivative_coordinates": ["time"]}
  ]


def test_driver_time_schedule_commits_oracle_state_and_records_time() -> None:
  driver = _visco_driver()
  layout = driver.owner.system.operators[0].header.state_layout
  block_id = layout.block_id
  calibration = prony_viscoelasticity_calibration(
    _E, _NU, _EINF, 3.0, _T_FIRST, _T_LAST
  )
  oracle_rows = np.zeros((9, 31))
  # The third substep holds time constant while strain advances: the legacy
  # guard branch, committed end-to-end through the driver.
  schedule = (
    (0.05, 2.0e-4),
    (0.15, 5.0e-4),
    (0.15, 8.0e-4),
    (0.40, 1.1e-3),
    (1.00, 1.6e-3),
  )
  base = _point(0.0, 0.0)
  for step, (time, eps) in enumerate(schedule, 1):
    result = driver.run(base_point=base, target_points=(_point(time, eps),))
    assert result.status is DriverStatus.COMPLETED
    base = _point(time, eps)
    # The committed rows equal the kernel stepped on the committed
    # integration-point strain path — bitwise, since the driver stages the
    # trial rows of the converged iterate.
    oracle = prony_viscoelasticity_kernel(
      _committed_ip_strains(driver), oracle_rows, calibration, (_time_signal(time),)
    )
    assert oracle.status is EvaluationStatus.OK
    oracle_rows = oracle.trial_rows
    rows = driver.owner.accepted_state(block_id).values
    assert rows.shape == (9, 31)
    np.testing.assert_array_equal(rows, oracle_rows)
    # State rows record the substep's bound time at every integration point.
    np.testing.assert_array_equal(rows[:, 30], time)
    assert result.final_generation.ordinal == step
  statistics = result.statistics
  assert statistics.committed_substep_count == 5
  assert statistics.rejected_substep_count == 0
  # Factorization honesty: the algorithmic-symmetric channel is not linear,
  # so the driver re-assembles and re-factorizes every Newton iteration.
  assert statistics.factorization_reuse_count == 0
  assert statistics.factorization_count == statistics.tangent_refill_count
  # Homogeneous strain: all nine integration points agree to solver tolerance.
  np.testing.assert_allclose(
    rows, np.broadcast_to(rows[0], rows.shape), rtol=1.0e-9, atol=1.0e-10
  )


def test_driver_response_depends_on_the_time_schedule() -> None:
  first = _visco_driver()
  second = _visco_driver()
  block_id = first.owner.system.operators[0].header.state_layout.block_id
  first_base = _point(0.0, 0.0)
  second_base = _point(0.0, 0.0)
  for first_time, second_time, eps in (
    (0.05, 0.50, 2.0e-4),
    (0.15, 3.00, 5.0e-4),
    (0.40, 12.0, 1.1e-3),
  ):
    first_result = first.run(
      base_point=first_base, target_points=(_point(first_time, eps),)
    )
    second_result = second.run(
      base_point=second_base, target_points=(_point(second_time, eps),)
    )
    assert first_result.status is DriverStatus.COMPLETED
    assert second_result.status is DriverStatus.COMPLETED
    first_base = _point(first_time, eps)
    second_base = _point(second_time, eps)
  # The same strain path under different time schedules relaxes differently:
  # schedule-owned time reaches the law through the port.
  assert (
    first.owner.accepted_state(block_id).values.tobytes()
    != second.owner.accepted_state(block_id).values.tobytes()
  )
  assert (
    first.owner.accepted_physical().values.tobytes()
    != second.owner.accepted_physical().values.tobytes()
  )


def test_port_free_law_results_are_signal_independent() -> None:
  first = _ramp_driver(_elastic_model(), q8_reference_registry())
  second = _ramp_driver(_elastic_model(), q8_reference_registry())
  eps_path = (2.0e-4, 5.0e-4, 1.1e-3)
  first_base = _point(0.0, 0.0)
  second_base = _point(0.0, 0.0)
  for first_time, second_time, eps in (
    (0.0, 3.0, eps_path[0]),
    (0.0, 7.5, eps_path[1]),
    (0.0, 11.25, eps_path[2]),
  ):
    first_result = first.run(
      base_point=first_base, target_points=(_point(first_time, eps),)
    )
    second_result = second.run(
      base_point=second_base, target_points=(_point(second_time, eps),)
    )
    assert first_result.status is DriverStatus.COMPLETED
    assert second_result.status is DriverStatus.COMPLETED
    assert (
      first.owner.accepted_physical().values.tobytes()
      == second.owner.accepted_physical().values.tobytes()
    )
    assert (
      first_result.records[-1].observation.reactions.values.tobytes()
      == second_result.records[-1].observation.reactions.values.tobytes()
    )
    first_base = _point(first_time, eps)
    second_base = _point(second_time, eps)


def test_state_codec_round_trips_and_rejects_foreign_schemas() -> None:
  driver = _visco_driver()
  result = driver.run(base_point=_point(0.0, 0.0), target_points=(_point(0.5, 8.0e-4),))
  assert result.status is DriverStatus.COMPLETED
  owner = driver.owner
  layout = owner.system.operators[0].header.state_layout
  payload = owner.encode_state(layout.block_id)
  decoded = owner.decode_state(layout.block_id, payload)
  np.testing.assert_array_equal(
    decoded.values, owner.accepted_state(layout.block_id).values
  )
  with pytest.raises(StateCodecError):
    owner.decode_state(layout.block_id, payload.replace(b"eps_i:18", b"eps_i:19"))
