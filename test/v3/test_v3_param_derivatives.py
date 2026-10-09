# SPDX-License-Identifier: MIT

"""Parameter derivative channel (M56): exact d(sigma)/dp columns end to end.

The S1A channel (``ParameterBinding`` / ``ResidualDerivativeChannel`` in
pyfem/v3/model/operator.py) lets a stateful continuum operator answer
``d(residual)/d(parameter)`` requests with exact analytic derivatives — no
autodiff framework and no finite differences in the shipped path (FD appears
here only as the verification oracle). The J2 law supplies the columns from
the converged return-map state
(pyfem/v3/materials/isotropic_hardening_plasticity.py, "Parameter derivative
channel"), and both kernel twins (reference and optimized) produce them
bitwise-identically (that pin lives in test_v3_stateful_plasticity.py). This
battery pins:

- header declarations and fail-closed request handling: the J2 operator
  declares one ``ParameterBinding`` and one ``dinternal-force/d<name>``
  channel per ``parameter_names`` entry; channel-free operators (damage,
  linear-elastic, spring) carry no derivative declarations and reject
  derivative requests; the frozen v2 metadata validator is byte-untouched —
  a foreign ``differentiable_parameters`` metadata key is still rejected,
  because the capability is declared behaviorally by the binding's
  ``param_derivative_kernel`` member (that inference is what keeps the
  teaching-law pattern of test_v3_authoring_plasticity.py — a primal-only
  binding reusing the qualified J2 metadata verbatim — compiling unchanged);
- the kernel columns against central finite differences of the kernel's own
  stress map under parameter perturbation (calibration repacking), at
  elastic and plastified accepted states along the documented paths, to
  ~1e-9 relative per column (observed <= 2.0e-9; pins at 1e-8), with the
  IFT-less truncated columns convicted materially wrong at plastic states
  (the M48 truncated-unrolling hazard class at the kernel level);
- the assembled operator derivative values against finite differences of
  the residual map itself — perturbed recompiles evaluated at the same
  point and accepted state — to the same class (observed <= 2.6e-9);
- state-transaction discipline: derivative evaluations never mutate the
  accepted rows, return trial rows bitwise identical to the primal
  evaluation at the same point, and report rejections with byte-equal
  accepted rows and empty derivative values;
- X1 increment 1: the assembled consistent tangent is the exact derivative
  of the residual map at NONZERO states — plastified (J2,
  ``algorithmic-symmetric``) and damaged (plane-strain damage,
  ``algorithmic-nonsymmetric``) trial states along committed paths, not only
  the zero state (observed <= 3.6e-9; pins at 1e-8) — and documents the
  tangent-class boundary behavior: at converged committed states the
  residual map carries the yield-surface/damage-envelope kink, so the same
  probe measures the elastic/algorithmic branch mixture (observed 1.0e-1
  and 2.4e-1 — material, a map property rather than a tangent defect),
  while the damage class's skew part is materially nonzero on progressive
  branches (observed 0.16-0.28 of max|K|) and ~1e-16 for the symmetric J2
  class.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from pyfem.v3 import authoring
from pyfem.v3.compile.continuum import (
  damage_reference_registry,
  plasticity_reference_registry,
  q8_reference_registry,
)
from pyfem.v3.compile.contracts import validate_stateful_material_metadata
from pyfem.v3.compile.spring import compile_spring_operator, damage_envelope_declaration
from pyfem.v3.compile.system import compile_system
from pyfem.v3.materials.isotropic_hardening_plasticity import (
  isotropic_hardening_calibration,
  isotropic_hardening_plasticity_kernel,
  isotropic_hardening_plasticity_metadata,
  isotropic_hardening_plasticity_param_kernel,
  isotropic_hardening_plasticity_param_kernel_reference,
)
from pyfem.v3.materials.plane_strain_damage import plane_strain_damage_metadata
from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.operator import (
  ChannelRequest,
  CompiledOperator,
  EvaluationStatus,
  OperatorEvaluation,
  OperatorEvaluationInput,
  evaluation_derivative_values,
  evaluation_status,
)
from pyfem.v3.model.system import CompiledSystem
from pyfem.v3.spec.diagnostics import SourceContext

# The documented M25 parity configuration (test_v3_stateful_plasticity.py).
_E = 210000.0
_NU = 0.3
_SYIELD = 250.0
_HARD = 1000.0
_PARAMETERS = (_E, _NU, _SYIELD, _HARD)
_PARAM_NAMES = (
  "youngs_modulus",
  "poisson_ratio",
  "initial_yield_stress",
  "hardening_slope",
)
_DERIVATIVE_CHANNELS = tuple(f"dinternal-force/d{name}" for name in _PARAM_NAMES)

# The documented damage configuration (the chapter-6 deck; test_v3_damage.py).
_DMG_PARAMETERS = (100.0, 0.3, 1.0e-6, 1.0e-5, 1.0)

_FD_REL_STEP = 1.0e-7
# Per-parameter FD steps. nu's absolute step is larger to keep the
# difference quotient above the stress-rounding floor: the map's nu
# sensitivity (|d(sigma)/d(nu)| ~ 5.6 at the documented states) is much
# smaller than the stress magnitudes (~720), so a 1e-7-relative stencil
# (2h = 2e-7) puts the FD numerator within a few ulps of the stress values
# and the quotient rounding dominates at ~9e-8 relative; the 1e-6 step keeps
# the observed deviation at 1.8e-9 with the branch boundary (O(1e-4) of
# strain) still far from the stencil.
_FD_STEPS = (_FD_REL_STEP * _E, 1.0e-6, _FD_REL_STEP * _SYIELD, _FD_REL_STEP * _HARD)
# Per-column deviation bound for the exact-derivative pins (kernel columns,
# assembled dR/dp columns, assembled tangent at plastified and damaged trial
# states). Observed maxima: 2.0e-9 (kernel columns along the documented
# plastic paths), 2.6e-9 (assembled dR/dp), 3.6e-9 (damage tangent) — the
# bound carries ~3x headroom over the worst observation.
_FD_RTOL = 1.0e-8


def _finalized(values: np.ndarray) -> FinalizedArray:
  return FinalizedArray(np.array(values, dtype=np.float64), dtype=np.float64)


def _j2_system(parameters: tuple[float, ...] = _PARAMETERS) -> CompiledSystem:
  mesh = authoring.quad8_patch(1, 1)
  model = authoring.small_strain_continuum(
    mesh, material=authoring.plasticity(*parameters, id="steel")
  )
  return compile_system(model, plasticity_reference_registry())


def _damage_system() -> CompiledSystem:
  mesh = authoring.quad8_patch(1, 1)
  model = authoring.small_strain_continuum(
    mesh, material=authoring.damage(*_DMG_PARAMETERS, id="dmg")
  )
  return compile_system(model, damage_reference_registry())


def _evaluate(
  operator: CompiledOperator,
  values: np.ndarray,
  accepted: np.ndarray,
  request: ChannelRequest,
) -> OperatorEvaluation:
  return operator.evaluate(
    OperatorEvaluationInput(
      port_values=(_finalized(np.asarray(values, dtype=np.float64)[None, :]),),
      accepted_state=_finalized(accepted),
      signals=(),
      request=request,
    )
  )


def _patch_field(ax: float, axy: float) -> np.ndarray:
  """Quadratic displacement field on the unit patch: varying strain per IP."""
  mesh = authoring.quad8_patch(1, 1)
  field = np.zeros(16)
  for index, node in enumerate(mesh.nodes):
    x, y = node.coordinates
    field[2 * index] = ax * x * x
    field[2 * index + 1] = axy * x * y
  return field


def _kernel_paths() -> list[np.ndarray]:
  """The documented paths: elastic ramp, uniaxial plastic ramp, mixed shear."""
  elastic = [
    np.array([eps, 0.0, 0.0, 0.0, 0.0, 0.0]) for eps in (0.0002, 0.0006, 0.0010)
  ]
  uniaxial = [
    np.array([eps, 0.0, 0.0, 0.0, 0.0, 0.0]) for eps in np.linspace(0.0002, 0.004, 10)
  ]
  shear = [
    np.array([0.004, 0.0, 0.0, 0.0, 0.0, gamma])
    for gamma in np.linspace(0.001, 0.006, 6)
  ]
  return [*elastic, *uniaxial, *shear]


# --- declaration and fail-closed handling ------------------------------------


def test_j2_operator_declares_parameter_derivative_channels() -> None:
  operator = _j2_system().operators[0]
  header = operator.header
  assert tuple(item.parameter_id for item in header.parameters) == _PARAM_NAMES
  assert tuple(item.channel_id for item in header.derivative_channels) == (
    _DERIVATIVE_CHANNELS
  )
  for channel in header.derivative_channels:
    assert channel.residual_channel_id == "internal-force"
  # The primal channel declarations are untouched.
  (residual,) = header.residual_channels
  assert residual.channel_id == "internal-force"
  (jacobian,) = header.jacobian_channels
  assert jacobian.channel_id == "material-tangent"
  # Requests predating the channel construct unchanged and serve no columns.
  request = ChannelRequest(("internal-force",), ("material-tangent",))
  assert request.derivative_channel_ids == ()
  layout = header.state_layout
  evaluation = _evaluate(operator, np.zeros(16), np.zeros(layout.row_shape), request)
  assert evaluation_derivative_values(evaluation) == ()
  with pytest.raises(TypeError, match="exact immutable tuple"):
    ChannelRequest(("internal-force",), (), ["dinternal-force/dyoungs_modulus"])  # type: ignore[arg-type]
  with pytest.raises(TypeError, match="exact OperatorEvaluation"):
    evaluation_derivative_values(object())  # type: ignore[arg-type]
  # Foreign or duplicate derivative channels fail closed.
  for bad in (("foreign-channel",), (_DERIVATIVE_CHANNELS[0],) * 2):
    with pytest.raises(ValueError, match="unavailable or duplicate channel"):
      _evaluate(
        operator,
        np.zeros(16),
        np.zeros(layout.row_shape),
        ChannelRequest(("internal-force",), (), bad),
      )


def test_channel_free_operators_and_validator_are_unchanged() -> None:
  """Channel-free operators carry no declarations and reject requests."""
  mesh = authoring.quad8_patch(1, 1)
  systems = (
    compile_system(
      authoring.small_strain_continuum(
        mesh, material=authoring.damage(*_DMG_PARAMETERS, id="dmg")
      ),
      damage_reference_registry(),
    ),
    compile_system(
      authoring.small_strain_continuum(
        mesh, material=authoring.linear_elastic(1000.0, 0.3, id="lin")
      ),
      q8_reference_registry(),
    ),
  )
  for system in systems:
    operator = system.operators[0]
    assert not hasattr(operator.header, "parameters")
    assert not hasattr(operator.header, "derivative_channels")
    layout = operator.header.state_layout
    evaluation = _evaluate(
      operator,
      np.zeros(16),
      np.zeros(layout.row_shape),
      ChannelRequest(("internal-force",), ()),
    )
    assert evaluation_derivative_values(evaluation) == ()
    with pytest.raises(ValueError, match="unavailable or duplicate channel"):
      _evaluate(
        operator,
        np.zeros(16),
        np.zeros(layout.row_shape),
        ChannelRequest(("internal-force",), (), (_DERIVATIVE_CHANNELS[0],)),
      )
  # The spring family's request gate rejects derivative channels identically.
  spring_system = compile_system(
    authoring.small_strain_continuum(
      mesh, material=authoring.linear_elastic(1000.0, 0.3, id="lin")
    ),
    q8_reference_registry(),
  )
  _block, spring = compile_spring_operator(
    spring_system,
    damage_envelope_declaration(
      block_id="springs",
      space_id="displacement",
      spring_ids=("spring-1",),
      node_ids=(1,),
      stiffness=2.0,
      critical_extension=2.0,
      max_increment=5.0,
      source=SourceContext(source="spring"),
    ),
  )
  assert not hasattr(spring.header, "derivative_channels")
  with pytest.raises(ValueError, match="unavailable or duplicate channel"):
    spring.evaluate(
      OperatorEvaluationInput(
        port_values=(_finalized(np.zeros((1, 2))),),
        accepted_state=_finalized(np.zeros(spring.header.state_layout.row_shape)),
        signals=(),
        request=ChannelRequest(("internal-force",), (), (_DERIVATIVE_CHANNELS[0],)),
      )
    )
  # The frozen v2 metadata validator is byte-untouched: the J2 metadata (no
  # derivative field — capability is the binding's behavioral member) and the
  # damage metadata validate exactly as before, and a foreign
  # ``differentiable_parameters`` key is rejected as outside the frozen set.
  metadata = isotropic_hardening_plasticity_metadata()
  assert "differentiable_parameters" not in metadata
  assert validate_stateful_material_metadata(metadata) is metadata
  assert validate_stateful_material_metadata(plane_strain_damage_metadata())
  spoofed = isotropic_hardening_plasticity_metadata()
  spoofed["differentiable_parameters"] = list(_PARAM_NAMES)
  with pytest.raises(ValueError, match="frozen v2 field set"):
    validate_stateful_material_metadata(spoofed)


# --- kernel column FD verification --------------------------------------------


def _perturbed_calibration(index: int, sign: float) -> tuple[np.ndarray, float]:
  step = _FD_STEPS[index]
  values = list(_PARAMETERS)
  values[index] += sign * step
  return isotropic_hardening_calibration(*values), step


def test_param_derivative_columns_match_central_fd() -> None:
  """Central FD of the stress map vs the exact columns, per parameter.

  Both twins are verified against the production kernel's primal map as the
  oracle (the twins' primal halves are pinned bitwise-identical to it in
  test_v3_stateful_plasticity.py). At every accepted state of the documented
  paths the map is smooth on the 1e-7-relative stencil (the nearest branch
  boundary is O(1e-4) of strain away — the same spacing argument as the
  tangent FD pin of test_v3_j2_tangent.py). Elastic states pin the exact
  zeros of the two hardening columns bitwise: the elastic map does not
  reference them.
  """
  oracle_calibration = isotropic_hardening_calibration(*_PARAMETERS)
  for kernel in (
    isotropic_hardening_plasticity_param_kernel_reference,
    isotropic_hardening_plasticity_param_kernel,
  ):
    rows = np.zeros((1, 19))
    worst = 0.0
    went_plastic = False
    for strain in _kernel_paths():
      strain = np.array(strain, dtype=np.float64)
      base = kernel(strain[None, :], rows, oracle_calibration)
      assert base.status is EvaluationStatus.OK
      assert base.param_derivatives is not None
      columns = base.param_derivatives[:, 0, :]
      plastic = base.trial_rows[0, 18] > rows[0, 18]
      went_plastic = went_plastic or plastic
      for index in range(4):
        plus_calibration, step = _perturbed_calibration(index, 1.0)
        minus_calibration, _ = _perturbed_calibration(index, -1.0)
        plus = isotropic_hardening_plasticity_kernel(
          strain[None, :], rows, plus_calibration
        )
        minus = isotropic_hardening_plasticity_kernel(
          strain[None, :], rows, minus_calibration
        )
        assert plus.status is EvaluationStatus.OK
        assert minus.status is EvaluationStatus.OK
        fd = (plus.stresses[0] - minus.stresses[0]) / (2.0 * step)
        if not plastic and index >= 2:
          # The elastic map is parameter-free in the hardening entries: the
          # exact column and its FD oracle both vanish to rounding.
          assert np.all(columns[index] == 0.0)
        scale = max(1.0, float(np.max(np.abs(fd))))
        worst = max(worst, float(np.max(np.abs(fd - columns[index]))) / scale)
      rows = base.trial_rows
    assert went_plastic
    # Observed worst: 2.0e-9 (initial_yield_stress column, plastic steps).
    assert worst < _FD_RTOL, worst


def test_truncated_columns_fail_materially_at_plastic_states() -> None:
  """Conviction: the IFT-less truncated derivative is not what FD measures.

  The truncated construction — the yield-stress correction without the
  return-map motion (``d(deqpl)/dp`` dropped) — is the kernel-level instance
  of the M48 truncated-unrolling hazard. At plastic states it deviates from
  the exact column by the factor ``hard / (3G + hard)`` (~4.1e-3 here),
  three orders of magnitude above the FD pin, so the battery convicts a
  missing IFT correction by name. The flow direction is reconstructed from
  the returned stress: ``sigma = flow * syield + shydro`` with a traceless
  ``flow`` of unit von Mises norm.
  """
  calibration = isotropic_hardening_calibration(*_PARAMETERS)
  rows = np.zeros((1, 19))
  for strain in (0.002, 0.004):
    rows = isotropic_hardening_plasticity_kernel(
      np.array([strain, 0.0, 0.0, 0.0, 0.0, 0.0])[None, :], rows, calibration
    ).trial_rows
  strain = np.array([0.006, 0.0, 0.0, 0.0, 0.0, 0.0])
  base = isotropic_hardening_plasticity_param_kernel(strain[None, :], rows, calibration)
  assert base.status is EvaluationStatus.OK
  assert base.param_derivatives is not None
  sigma = base.stresses[0]
  shydro = (sigma[0] + sigma[1] + sigma[2]) / 3.0
  flow_hat = np.array(sigma, copy=True)
  flow_hat[:3] -= shydro
  flow_hat /= np.sqrt(
    0.5
    * (
      (sigma[0] - sigma[1]) ** 2
      + (sigma[1] - sigma[2]) ** 2
      + (sigma[2] - sigma[0]) ** 2
      + 6.0 * np.dot(sigma[3:], sigma[3:])
    )
  )
  truncated_sig0 = flow_hat  # d(sigma)/d(sig0) without the deqpl motion
  truncated_hard = flow_hat * base.trial_rows[0, 18]
  for index, truncated in ((2, truncated_sig0), (3, truncated_hard)):
    column = base.param_derivatives[index, 0, :]
    deviation = float(np.max(np.abs(column - truncated))) / float(
      np.max(np.abs(column))
    )
    assert deviation > 1.0e-3, deviation  # observed: ~4.1e-3


# --- assembled operator derivative channels ------------------------------------


def test_operator_derivative_channels_match_fd_of_residual_map() -> None:
  """Assembled dR/dp columns vs FD of the residual map (perturbed recompiles).

  The FD oracle perturbs one material parameter at the spec level, recompiles
  the operator, and differences the internal-force residual at the same
  evaluation point over the same accepted rows — exactly the channel's
  semantics (committed state held fixed). The probe states are the mid-step
  trial points Newton actually iterates at, threaded along a heterogeneous
  ramp that lands six of nine integration points in plasticity.
  """
  base_operator = _j2_system().operators[0]
  perturbed = {}
  for index in range(4):
    for sign in (1, -1):
      values = list(_PARAMETERS)
      values[index] += sign * _FD_REL_STEP * max(abs(_PARAMETERS[index]), 1.0)
      perturbed[index, sign] = _j2_system(tuple(values)).operators[0]
  field = _patch_field(0.003, 0.0005)
  accepted = np.zeros(base_operator.header.state_layout.row_shape)
  worst = 0.0
  for scale_factor in (0.25, 0.5, 0.75, 1.0):
    values = scale_factor * field
    evaluation = _evaluate(
      base_operator,
      values,
      accepted,
      ChannelRequest(("internal-force",), (), _DERIVATIVE_CHANNELS),
    )
    assert evaluation_status(evaluation) is EvaluationStatus.OK
    columns = evaluation_derivative_values(evaluation)
    assert len(columns) == 4
    for index in range(4):
      step = _FD_REL_STEP * max(abs(_PARAMETERS[index]), 1.0)
      plus = _evaluate(
        perturbed[index, 1], values, accepted, ChannelRequest(("internal-force",), ())
      )
      minus = _evaluate(
        perturbed[index, -1], values, accepted, ChannelRequest(("internal-force",), ())
      )
      fd = (plus.residual_values[0].values[0] - minus.residual_values[0].values[0]) / (
        2.0 * step
      )
      scale = max(1.0, float(np.max(np.abs(fd))))
      worst = max(worst, float(np.max(np.abs(fd - columns[index].values[0]))) / scale)
    accepted = np.array(evaluation.trial_state.values, copy=True)
  # Observed worst: 2.6e-9 (initial_yield_stress channel).
  assert worst < _FD_RTOL, worst


def test_derivative_channel_order_and_reject_semantics() -> None:
  operator = _j2_system().operators[0]
  accepted = np.zeros(operator.header.state_layout.row_shape)
  values = 0.5 * _patch_field(0.003, 0.0005)
  # A scrambled subset request follows the header declaration order.
  evaluation = _evaluate(
    operator,
    values,
    accepted,
    ChannelRequest((), (), (_DERIVATIVE_CHANNELS[3], _DERIVATIVE_CHANNELS[0])),
  )
  subset = evaluation_derivative_values(evaluation)
  full = evaluation_derivative_values(
    _evaluate(operator, values, accepted, ChannelRequest((), (), _DERIVATIVE_CHANNELS))
  )
  assert len(subset) == 2
  np.testing.assert_array_equal(subset[0].values, full[0].values)  # youngs_modulus
  np.testing.assert_array_equal(subset[1].values, full[3].values)  # hardening_slope
  # A rejecting evaluation (beyond the hardening table) reports the typed
  # status with byte-equal accepted rows and empty derivative values.
  evaluation = _evaluate(
    operator,
    1.0e3 * _patch_field(0.003, 0.0005),
    accepted,
    ChannelRequest(("internal-force",), (), _DERIVATIVE_CHANNELS),
  )
  assert evaluation_status(evaluation) is EvaluationStatus.REJECT_STEP
  assert evaluation_derivative_values(evaluation) == ()
  assert evaluation.trial_state.values.tobytes() == accepted.tobytes()


def test_derivative_evaluation_state_transaction_discipline() -> None:
  operator = _j2_system().operators[0]
  accepted = np.zeros(operator.header.state_layout.row_shape)
  field = _patch_field(0.003, 0.0005)
  # Commit one step so the accepted state is genuinely plastified.
  accepted = np.array(
    _evaluate(
      operator,
      0.5 * field,
      accepted,
      ChannelRequest(("internal-force",), ("material-tangent",)),
    ).trial_state.values,
    copy=True,
  )
  assert np.any(accepted[:, 18] > 0.0)
  snapshot = accepted.tobytes()
  primal = _evaluate(
    operator,
    0.75 * field,
    accepted,
    ChannelRequest(("internal-force",), ("material-tangent",)),
  )
  derivative = _evaluate(
    operator,
    0.75 * field,
    accepted,
    ChannelRequest(("internal-force",), ("material-tangent",), _DERIVATIVE_CHANNELS),
  )
  # The accepted rows are input-pure and the derivative evaluation's trial
  # rows are bitwise identical to the primal evaluation's at the same point.
  assert accepted.tobytes() == snapshot
  assert derivative.trial_state.values.tobytes() == primal.trial_state.values.tobytes()
  np.testing.assert_array_equal(
    derivative.residual_values[0].values, primal.residual_values[0].values
  )
  np.testing.assert_array_equal(
    derivative.jacobian_values[0].values, primal.jacobian_values[0].values
  )


# --- X1 increment 1: assembled tangent FD probes at nonzero states ------------


def _fd_probe_tangent(
  operator: CompiledOperator,
  values: np.ndarray,
  accepted: np.ndarray,
  step: float,
) -> tuple[float, float]:
  """Central FD of the internal-force map vs the assembled tangent channel."""
  base = _evaluate(
    operator,
    values,
    accepted,
    ChannelRequest(("internal-force",), ("material-tangent",)),
  )
  assert evaluation_status(base) is EvaluationStatus.OK
  tangent = base.jacobian_values[0].values[0]
  fd = np.zeros_like(tangent)
  for column in range(16):
    delta = np.zeros(16)
    delta[column] = step
    plus = _evaluate(
      operator, values + delta, accepted, ChannelRequest(("internal-force",), ())
    )
    minus = _evaluate(
      operator, values - delta, accepted, ChannelRequest(("internal-force",), ())
    )
    fd[:, column] = (
      plus.residual_values[0].values[0] - minus.residual_values[0].values[0]
    ) / (2.0 * step)
  scale = float(np.max(np.abs(tangent)))
  deviation = float(np.max(np.abs(fd - tangent))) / scale
  skew = float(np.max(np.abs(tangent - tangent.T))) / scale
  return deviation, skew


def test_assembled_tangent_fd_at_plastified_states() -> None:
  """X1 inc. 1: the J2 assembled tangent is the residual map's exact derivative.

  Probes the virgin zero state and every mid-step trial point of a committed
  heterogeneous ramp (six of nine IPs plastic at the final step) — the points
  Newton actually iterates at, where the branch margin is the step increment.
  The committed-state probe documents the yield-surface kink: at the
  converged state the map is C0 with an elastic/algorithmic branch switch,
  so FD across it measures the mixture (material deviation, pinned below as
  a lower bound — the conviction that the nonzero-state pins are not
  vacuous). The symmetric class is confirmed by the skew measurements.
  """
  operator = _j2_system().operators[0]
  field = _patch_field(0.003, 0.0005)
  accepted = np.zeros(operator.header.state_layout.row_shape)
  deviation, skew = _fd_probe_tangent(operator, np.zeros(16), accepted, 1.0e-7)
  assert deviation < _FD_RTOL  # zero state, observed 2.1e-16 (linear elastic)
  assert skew < 1.0e-12
  worst = 0.0
  values = np.zeros(16)
  for scale_factor in (0.25, 0.5, 0.75, 1.0):
    values = scale_factor * field
    deviation, skew = _fd_probe_tangent(operator, values, accepted, 1.0e-7)
    worst = max(worst, deviation)
    assert skew < 1.0e-12  # algorithmic-symmetric, observed <= 2.2e-16
    evaluation = _evaluate(
      operator, values, accepted, ChannelRequest(("internal-force",), ())
    )
    accepted = np.array(evaluation.trial_state.values, copy=True)
  assert np.count_nonzero(accepted[:, 18] > 0.0) == 6
  # Observed worst: 1.0e-9 at the first plastic step (1e-7 stencil).
  assert worst < _FD_RTOL, worst
  # The committed-state kink: branch straddling makes the same probe see the
  # elastic/algorithmic mixture — material, not a tangent defect.
  deviation, _ = _fd_probe_tangent(operator, values, accepted, 1.0e-7)
  assert deviation > 1.0e-3, deviation  # observed: 1.0e-1


def test_assembled_tangent_fd_at_damaged_states() -> None:
  """X1 inc. 1: the damage law's nonsymmetric tangent is exact on its branch.

  The progressive-branch tangent of the ``algorithmic-nonsymmetric`` class
  (the rank-1-corrected ``(1 - omega) De``) is the exact derivative of the
  residual map at damaged trial states (a 1e-10 stencil: the softening
  curvature puts FD truncation at O(h^2), observed 3.6e-9); its skew part is
  materially nonzero there, so the class declaration is honest, not
  aspirational. At the committed state the kappa envelope sits exactly on
  the equivalent strain, the evaluation takes the elastic unloading branch,
  and the FD legs straddle the progressive/elastic kink (observed 2.4e-1).
  """
  operator = _damage_system().operators[0]
  field = _patch_field(4.5e-6, 0.5e-6)
  accepted = np.zeros(operator.header.state_layout.row_shape)
  deviation, skew = _fd_probe_tangent(operator, np.zeros(16), accepted, 1.0e-10)
  assert deviation < _FD_RTOL  # zero state, observed 4.4e-16
  assert skew < 1.0e-12
  worst = 0.0
  skews = []
  values = np.zeros(16)
  for scale_factor in (0.5, 1.0):
    values = scale_factor * field
    deviation, skew = _fd_probe_tangent(operator, values, accepted, 1.0e-10)
    worst = max(worst, deviation)
    skews.append(skew)
    evaluation = _evaluate(
      operator, values, accepted, ChannelRequest(("internal-force",), ())
    )
    accepted = np.array(evaluation.trial_state.values, copy=True)
  assert np.all(accepted[:, 0] > 0.0)  # all nine IPs damaged
  # Observed worst: 3.6e-9; the skew part is 16-28% of max|K| on progressive
  # branches (the honest asymmetry of the declared nonsymmetric class).
  assert worst < _FD_RTOL, worst
  assert min(skews) > 1.0e-2, skews
  deviation, skew = _fd_probe_tangent(operator, values, accepted, 1.0e-10)
  assert deviation > 1.0e-3, deviation  # committed-state kink, observed 2.4e-1
  assert skew < 1.0e-12  # the committed-state unloading tangent is symmetric
