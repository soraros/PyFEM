# SPDX-License-Identifier: MIT

"""Cohesive interface authoring: plain element quads composed into systems.

Covers the four landed traction-separation law helpers
(``xu_needleman``/``power_law_mode_i``/``thouless_mode_i``/
``dummy_interface``): declaration shape and compiled operator identity,
authoring-time validation with the landed domain diagnostics, and the
stepped persona — a student authors each law on the M65 minimal seam
configuration (the family's own documented compile harness) with the
documented peel60pres constants and prescribed-opening ramp, and steps a
``nonlinear_static`` session into softening. Committed coefficients and
interface state rows equal the landed declaration/compile/compose path
BITWISE per substep, and the operator's committed residual equals the law
kernel's tractions assembled test-side on the committed jump path, bitwise.
The laws ship no derivative kernels, so their qualified parameters read
'constant' in the M58 sensitivity diagnostics — pinned literally.
"""

from __future__ import annotations

import sys
from collections.abc import Callable

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from pyfem.v3 import authoring
from pyfem.v3.compile.diagnostics import ModelCompilationError
from pyfem.v3.compile.interface import (
  InterfaceDeclaration,
  compile_interface_operator,
  compose_interface_system,
  dummy_interface_declaration,
  power_law_mode_i_declaration,
  thouless_mode_i_declaration,
  xu_needleman_declaration,
  xu_needleman_rank2_kernel,
)
from pyfem.v3.compile.system import compile_system
from pyfem.v3.compile.truss import truss_reference_registry
from pyfem.v3.constraints import compile_constraint_map
from pyfem.v3.driver import DriverStatus, NonlinearStaticDriver
from pyfem.v3.driver.diagnostics import DriverPreparationError
from pyfem.v3.model.operator import EvaluationStatus
from pyfem.v3.model.system import CompiledSystem
from pyfem.v3.spec import ModelSpec, SourceContext
from pyfem.v3.spec.program import (
  AffineCoefficientSpec,
  AffineValueSpec,
  DofRef,
  PrescribedDofSpec,
  ProgramConstraintSpec,
  ProgramCoordinateSpec,
  ProgramCoordinateValue,
  ProgramPoint,
)

# The documented M65 peel60pres law constants (skims/peel60pres/skim.pro) and
# the converter battery's canonical values for the remaining laws.
_FRACTURE_ENERGY = 0.1
_ULTIMATE_TRACTION = 0.5
_D1D3 = 0.2
_D2D3 = 0.6
_STIFFNESS = 1.0e5
# The Xu-Needleman peak-opening scale vnmax = Gc / (e * Tult): the peel ramp
# crosses it between the second and third substep, exactly as the documented
# skim remarks (lam = 0.75 sits just below the peak opening 0.0736).
_VNMAX = _FRACTURE_ENERGY / (2.71828183 * _ULTIMATE_TRACTION)
# The documented peel60pres load table; the anchor prescription v = 0.1*lam
# reaches 2.7x vnmax at lam = 2.0, deep into softening.
_LOAD_TABLE = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
_LAW_CONSTANTS = {
  "xu-needleman-rank2": (_FRACTURE_ENERGY, _ULTIMATE_TRACTION),
  "power-law-mode-i": (_FRACTURE_ENERGY, _ULTIMATE_TRACTION),
  "thouless-mode-i": (_FRACTURE_ENERGY, _ULTIMATE_TRACTION, _D1D3, _D2D3),
  "dummy-linear-interface": (_STIFFNESS,),
}


def _source(label: str) -> SourceContext:
  return SourceContext(source=label)


def _seam_model() -> ModelSpec:
  """The M65 minimal seam configuration, authored: two bar-connected banks.

  Nodes 0/1 are the bottom seam pair, 2/3 the coincident top pair, and 4/5
  the anchor row; the two vertical connector bars (2-4, 3-5) pull the seam
  open against the cohesive traction, so the interface law genuinely
  equilibrates the Newton solve (the family's own documented minimal
  compile harness, test_v3_interface.py's unit quad, authored).
  """
  mesh = authoring.line2_mesh(
    {
      0: (0.0, 0.0),
      1: (1.0, 0.0),
      2: (0.0, 0.0),
      3: (1.0, 0.0),
      4: (0.0, 1.0),
      5: (1.0, 1.0),
    },
    {"bar-left": (2, 4), "bar-right": (3, 5)},
  )
  return authoring.truss(mesh, material=authoring.uniaxial_elastic(1.0e6, 1.0))


def _peel_constraints() -> tuple[ProgramConstraintSpec, ...]:
  """Bottom seam fixed, anchor-x fixed; anchor row prescribed v = 0.1 * load."""
  constraints: list[ProgramConstraintSpec] = list(authoring.fixed(nodes=(0, 1)))
  constraints.extend(authoring.fixed(nodes=(4, 5), components=("x",)))
  constraints.extend(
    PrescribedDofSpec(
      id=f"v-{node_id}",
      target=DofRef(node_id=node_id, field_id="displacement", component="y"),
      value=AffineValueSpec(
        coefficients=(AffineCoefficientSpec("load", 0.1),),
      ),
    )
    for node_id in (4, 5)
  )
  return tuple(constraints)


def _law_helper(law: str) -> Callable[..., CompiledSystem]:
  return {
    "xu-needleman-rank2": authoring.xu_needleman,
    "power-law-mode-i": authoring.power_law_mode_i,
    "thouless-mode-i": authoring.thouless_mode_i,
    "dummy-linear-interface": authoring.dummy_interface,
  }[law]


def _law_kwargs(law: str) -> dict[str, float]:
  if law == "thouless-mode-i":
    return {
      "fracture_energy": _FRACTURE_ENERGY,
      "ultimate_traction": _ULTIMATE_TRACTION,
      "d1d3": _D1D3,
      "d2d3": _D2D3,
    }
  if law == "dummy-linear-interface":
    return {"stiffness": _STIFFNESS}
  return {"fracture_energy": _FRACTURE_ENERGY, "ultimate_traction": _ULTIMATE_TRACTION}


def _author_system(law: str) -> CompiledSystem:
  base = authoring.compile(_seam_model())
  return _law_helper(law)(base, elements={"seam-1": (0, 1, 2, 3)}, **_law_kwargs(law))


def _twin_system(law: str) -> CompiledSystem:
  """The landed path directly: declaration builder plus compile/compose."""
  base = compile_system(_seam_model(), truss_reference_registry())
  builder = {
    "xu-needleman-rank2": xu_needleman_declaration,
    "power-law-mode-i": power_law_mode_i_declaration,
    "thouless-mode-i": thouless_mode_i_declaration,
    "dummy-linear-interface": dummy_interface_declaration,
  }[law]
  declaration = builder(
    block_id="interfaces",
    space_id="displacement",
    interface_ids=("seam-1",),
    node_quads=((0, 1, 2, 3),),
    source=_source("test:twin"),
    **_law_kwargs(law),
  )
  block, operator = compile_interface_operator(base, declaration)
  return compose_interface_system(base, block, operator)


def _session(system: CompiledSystem) -> authoring.NonlinearStaticSession:
  return authoring.nonlinear_static(system, constraints=_peel_constraints())


def _twin_driver(system: CompiledSystem) -> NonlinearStaticDriver:
  coordinate_map = compile_constraint_map(
    system,
    constraints=_peel_constraints(),
    coordinates=(ProgramCoordinateSpec(name="load", kind="load"),),
  )
  return NonlinearStaticDriver(system, coordinate_map, ())


def _point(load: float) -> ProgramPoint:
  return ProgramPoint((ProgramCoordinateValue("load", load),))


def _committed_residual_oracle(
  system: CompiledSystem,
  coefficients: np.ndarray,
  normal_rows: np.ndarray,
) -> np.ndarray:
  """Reassemble the interface residual from the law kernel on committed jumps.

  The parity-harness expression: the operator's own published geometry (the
  committed frame normals, the Gauss-2 Line2 shapes, and the integration
  weights) applied to the committed coefficient vector, with the landed law
  kernel supplying the per-integration-point tractions — mirroring the
  evaluation's arithmetic operation for operation, so the comparison is
  bitwise.
  """
  operator = system.operators[1]
  payload = operator.payload
  gather = operator.header.ports[0].coefficient_map.values
  displacements = coefficients[gather]
  shapes = payload.ip_shape_values.values
  weights = payload.integration_weights.values
  normals = normal_rows
  rotation = np.empty((len(normals), 2, 2), dtype=np.float64)
  rotation[:, 0, 0] = normals[:, 0]
  rotation[:, 0, 1] = normals[:, 1]
  rotation[:, 1, 0] = -normals[:, 1]
  rotation[:, 1, 1] = normals[:, 0]
  rot_ip = rotation[:, None, :, :]
  phi0 = shapes[None, :, 0, None, None]
  phi1 = shapes[None, :, 1, None, None]
  entity_count = len(normals)
  b_matrix = np.empty((entity_count, 2, 2, 8), dtype=np.float64)
  b_matrix[:, :, :, 0:2] = -rot_ip * phi0
  b_matrix[:, :, :, 2:4] = -rot_ip * phi1
  b_matrix[:, :, :, 4:6] = rot_ip * phi0
  b_matrix[:, :, :, 6:8] = rot_ip * phi1
  jumps = np.matmul(b_matrix, displacements[:, None, :, None])[:, :, :, 0]
  flat_jumps = np.array(jumps.reshape(entity_count * 2, 2), dtype=np.float64, order="C")
  law_rows = np.zeros((entity_count * 2, 0), dtype=np.float64)
  result = operator.kernel(flat_jumps, law_rows, payload.parameters.values)
  assert result.status is EvaluationStatus.OK
  traction = result.traction.reshape(entity_count, 2, 2)
  force0 = np.matmul(
    b_matrix[:, 0].transpose(0, 2, 1),
    traction[:, 0][:, :, None],
  )[:, :, 0]
  force1 = np.matmul(
    b_matrix[:, 1].transpose(0, 2, 1),
    traction[:, 1][:, :, None],
  )[:, :, 0]
  return force0 * weights[:, None] + force1 * weights[:, None]


# --- helper surface: declaration shape and validation ----------------------------


@pytest.mark.parametrize(
  ("law", "parameters"),
  tuple(_LAW_CONSTANTS.items()),
  ids=tuple(_LAW_CONSTANTS),
)
def test_interface_helpers_compose_the_landed_operator(
  law: str,
  parameters: tuple[float, ...],
) -> None:
  system = _author_system(law)
  assert len(system.operators) == 2
  operator = system.operators[1]
  (implementation,) = operator.header.implementations
  assert implementation.kind == "constitutive-kernel"
  assert implementation.name == law
  assert tuple(operator.payload.parameters.values) == parameters
  layout = operator.header.state_layout
  assert layout.block_id == ("interfaces", "pyfem-v3-interface-frame-state-v1")
  assert layout.schema == "pyfem-v3-interface-frame-state-v1"
  assert layout.row_shape == (1, 2)
  (slot,) = layout.slots
  assert slot.name == "normal"
  assert tuple(operator.interface_block.entity_ids) == ("seam-1",)
  # The honest channel flags of the landed exact tangent.
  (jacobian,) = operator.header.jacobian_channels
  assert jacobian.linear is False
  assert jacobian.symmetric is False


def test_interface_elements_accept_sequence_and_mapping_forms() -> None:
  sequenced = authoring.xu_needleman(
    authoring.compile(_seam_model()),
    elements=((0, 1, 2, 3),),
    fracture_energy=_FRACTURE_ENERGY,
    ultimate_traction=_ULTIMATE_TRACTION,
  )
  operator = sequenced.operators[1]
  assert tuple(operator.interface_block.entity_ids) == ("interface-1",)
  mapped = authoring.dummy_interface(
    authoring.compile(_seam_model()),
    elements={10: (0, 1, 2, 3)},
    stiffness=_STIFFNESS,
    block_id="cohesive",
  )
  assert tuple(mapped.operators[1].interface_block.entity_ids) == (10,)
  assert mapped.operators[1].interface_block.block_id == "cohesive"
  assert mapped.operators[1].header.block_id == (
    "cohesive",
    "pyfem-v3-interface-frame-state-v1",
  )


def test_interface_authoring_rejects_bad_inputs_early() -> None:
  base = authoring.compile(_seam_model())
  with pytest.raises(ValueError, match="exactly four node ids"):
    authoring.xu_needleman(
      base,
      elements=((0, 1, 2),),
      fracture_energy=_FRACTURE_ENERGY,
      ultimate_traction=_ULTIMATE_TRACTION,
    )
  with pytest.raises(ValueError, match="at least one element"):
    authoring.xu_needleman(
      base,
      elements=(),
      fracture_energy=_FRACTURE_ENERGY,
      ultimate_traction=_ULTIMATE_TRACTION,
    )
  with pytest.raises(TypeError, match="exact number"):
    authoring.xu_needleman(
      base,
      elements=((0, 1, 2, 3),),
      fracture_energy="0.1",  # type: ignore[arg-type]
      ultimate_traction=_ULTIMATE_TRACTION,
    )
  with pytest.raises(ValueError, match="positive finite"):
    authoring.power_law_mode_i(
      base,
      elements=((0, 1, 2, 3),),
      fracture_energy=-_FRACTURE_ENERGY,
      ultimate_traction=_ULTIMATE_TRACTION,
    )
  # The landed domain diagnostic for the Thouless ratio ordering.
  with pytest.raises(ValueError, match=r"0 < d1d3 < d2d3 < 1"):
    authoring.thouless_mode_i(
      base,
      elements=((0, 1, 2, 3),),
      fracture_energy=_FRACTURE_ENERGY,
      ultimate_traction=_ULTIMATE_TRACTION,
      d1d3=0.6,
      d2d3=0.2,
    )
  with pytest.raises(TypeError, match="exact CompiledSystem"):
    authoring.dummy_interface(
      base.operators[0],  # type: ignore[arg-type]
      elements=((0, 1, 2, 3),),
      stiffness=_STIFFNESS,
    )
  with pytest.raises(ModelCompilationError, match="unknown-interface-node"):
    authoring.dummy_interface(
      base,
      elements=((0, 1, 2, 99),),
      stiffness=_STIFFNESS,
    )


def test_interface_declaration_metadata_stays_landed() -> None:
  # The helpers build exactly the landed declaration values: constructing the
  # declaration through the landed builder with the helper's arguments must
  # reproduce the compiled operator's parameter payload and kernel identity.
  declaration = xu_needleman_declaration(
    block_id="interfaces",
    space_id="displacement",
    interface_ids=("seam-1",),
    node_quads=((0, 1, 2, 3),),
    fracture_energy=_FRACTURE_ENERGY,
    ultimate_traction=_ULTIMATE_TRACTION,
    source=_source("authoring.xu_needleman"),
  )
  assert type(declaration) is InterfaceDeclaration
  system = _author_system("xu-needleman-rank2")
  operator = system.operators[1]
  assert operator.kernel is xu_needleman_rank2_kernel
  np.testing.assert_array_equal(
    operator.payload.parameters.values,
    declaration.parameters and np.array(declaration.parameters),
  )


# --- sensitivity surface: no derivative kernel reads 'constant' -------------------


def test_interface_parameters_read_constant_in_the_sensitivity_surface() -> None:
  """The declaration-routed law ships no derivative kernel either.

  The qualified interface parameter names resolve through the M58 convention
  resolver (the landed kernel's fixed parameter tuple), so a sensitivity
  request fails pre-substep with the parameterized-vs-constant diff — the
  trailer lists the truss bank's constants alongside the interface law's.
  """
  session = _session(_author_system("xu-needleman-rank2"))
  with pytest.raises(DriverPreparationError) as captured:
    session.run({"load": 0.0}, {"load": 0.25}, sensitivities=("fracture_energy",))
  (diagnostic,) = captured.value.diagnostics
  assert diagnostic.code == "unknown-sensitivity-parameter"
  assert (
    "field 'sensitivities.fracture_energy': expected {'parameter': "
    "'fracture_energy'}, authored 'constant'" in diagnostic.message
  )
  assert (
    "declared differentiable parameters: (); declared constant parameters: "
    "('youngs_modulus', 'area', 'fracture_energy', 'ultimate_traction')"
    in diagnostic.message
  )
  assert diagnostic.source == _source("authoring.run:sensitivities")
  assert session.driver.statistics.evaluation_count == 0
  assert session.ordinal == 0


# --- stepped personas: authored law versus the landed path, bitwise ---------------


def _step_persona(law: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
  """Step the authored and the landed-twin sessions over the peel ramp.

  Returns the per-substep committed jump-traction profiles of the authored
  session for the law-specific witness legs; asserts the bitwise claims.
  """
  system = _author_system(law)
  session = _session(system)
  twin = _twin_driver(_twin_system(law))
  base = {"load": 0.0}
  twin_base = _point(0.0)
  openings: list[np.ndarray] = []
  tractions: list[np.ndarray] = []
  for step, lam in enumerate(_LOAD_TABLE, 1):
    result = session.run(base, {"load": lam})
    assert result.status is DriverStatus.COMPLETED
    base = {"load": lam}
    twin_result = twin.run(base_point=twin_base, target_points=(_point(lam),))
    assert twin_result.status is DriverStatus.COMPLETED
    twin_base = _point(lam)
    coefficients = session.accepted_coefficients()
    block_key = system.operators[1].header.state_layout.block_id
    normal_rows = session.accepted_state(block_key)
    # The committed interface state equals the landed path's, bitwise, and
    # the coefficients agree bitwise per substep.
    np.testing.assert_array_equal(coefficients, twin.owner.accepted_physical().values)
    np.testing.assert_array_equal(
      normal_rows, twin.owner.accepted_state(block_key).values
    )
    # The frame replay: the seam stays horizontal under the mirror-symmetric
    # opening, so the corrected normal bootstraps to (0, 1) and holds.
    np.testing.assert_array_equal(normal_rows, np.array([[0.0, 1.0]]))
    # The committed residual equals the law kernel's tractions assembled on
    # the committed jump path — bitwise against the operator's evaluation.
    evaluation = authoring.evaluate(
      system,
      coefficients,
      operator=1,
      jacobian=False,
      accepted_state=normal_rows,
    )
    np.testing.assert_array_equal(
      evaluation.residual_values[0].values,
      _committed_residual_oracle(system, coefficients, normal_rows),
    )
    assert session.ordinal == step
    # Witness data: the committed normal jump at both integration points and
    # the kernel's normal traction there (uniform seam: both points agree).
    operator = system.operators[1]
    gather = operator.header.ports[0].coefficient_map.values
    displacements = coefficients[gather]
    jump_left = displacements[:, 4:6] - displacements[:, 0:2]
    jump_right = displacements[:, 6:8] - displacements[:, 2:4]
    shapes = operator.payload.ip_shape_values.values
    global_jumps = shapes[None, :, 0, None] * jump_left[:, None, :] + (
      shapes[None, :, 1, None] * jump_right[:, None, :]
    )
    rotation = np.array([[[0.0, 1.0], [-1.0, 0.0]]])
    local_jumps = np.matmul(rotation[:, None, :, :], global_jumps[:, :, :, None])[
      :, :, :, 0
    ]
    kernel_result = operator.kernel(
      np.array(local_jumps.reshape(2, 2), dtype=np.float64, order="C"),
      np.zeros((2, 0), dtype=np.float64),
      operator.payload.parameters.values,
    )
    assert kernel_result.status is EvaluationStatus.OK
    openings.append(local_jumps[0, :, 0])
    tractions.append(kernel_result.traction[:, 0])
  return openings, tractions


def test_persona_xu_needleman_peel_into_softening() -> None:
  """The documented peel60pres harness, authored: ramp past the XN strength.

  The committed path carries the classic traction-separation profile: the
  normal traction rises toward the ultimate traction Tult as the opening
  approaches vnmax, then softens exponentially as the ramp reaches 2.7x
  vnmax — the profile the skim documents, witnessed on the kernel oracle.
  """
  openings, tractions = _step_persona("xu-needleman-rank2")
  peak_openings = np.array([pair[0] for pair in openings])
  peak_tractions = np.array([pair[0] for pair in tractions])
  # The stiff connector bars keep the seam opening at the prescribed ramp.
  np.testing.assert_allclose(peak_openings, 0.1 * np.array(_LOAD_TABLE), rtol=1.0e-3)
  # Rise past the strength: the profile climbs through the second substep,
  # peaks within 1% of Tult as the opening crosses vnmax, then softens to
  # under 60% of the peak by the end of the ramp.
  assert peak_tractions[1] > peak_tractions[0]
  assert peak_tractions[:3].max() == pytest.approx(_ULTIMATE_TRACTION, rel=0.01)
  assert peak_tractions[-1] < 0.6 * peak_tractions.max()
  assert peak_openings[-1] > 2.5 * _VNMAX


def test_persona_power_law_mode_i_steps_bitwise_like_the_landed_path() -> None:
  openings, tractions = _step_persona("power-law-mode-i")
  # The mode-I law carries a finite peak and softening tail on the same ramp.
  peak_tractions = np.array([pair[0] for pair in tractions])
  assert np.all(np.isfinite(peak_tractions))
  assert peak_tractions.max() > 0.0


def test_persona_thouless_mode_i_steps_bitwise_like_the_landed_path() -> None:
  openings, tractions = _step_persona("thouless-mode-i")
  peak_tractions = np.array([pair[0] for pair in tractions])
  assert np.all(np.isfinite(peak_tractions))
  assert peak_tractions.max() > 0.0


def test_persona_dummy_interface_steps_bitwise_like_the_landed_path() -> None:
  openings, tractions = _step_persona("dummy-linear-interface")
  # The dummy law is exactly linear: traction = D * jump on the whole ramp.
  normal_openings = np.array([pair[0] for pair in openings])
  normal_tractions = np.array([pair[0] for pair in tractions])
  np.testing.assert_allclose(
    normal_tractions, _STIFFNESS * normal_openings, rtol=1.0e-12
  )
