# SPDX-License-Identifier: MIT

"""Cohesive interface operator family: compile boundary, frame machine, channels.

The parity batteries against the legacy oracle live in test_v3_cohesive.py;
this file pins the compile-time contract: declaration validation, the
fail-closed virgin probes, the seeded finite-difference tangent probes (law
level and assembled-operator level, each with a conviction leg), the frame
state machine, the honest channel flags, and the legacy-deck converter's
Interface admission.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from pyfem.v3.compile.contracts import continuum_reference_registry
from pyfem.v3.compile.diagnostics import ModelCompilationError
from pyfem.v3.compile.interface import (
  InterfaceDeclaration,
  InterfaceKernelResult,
  compile_interface_operator,
  compose_interface_system,
  dummy_interface_declaration,
  power_law_mode_i_declaration,
  thouless_mode_i_declaration,
  xu_needleman_declaration,
)
from pyfem.v3.compile.system import compile_system
from pyfem.v3.compile.truss import truss_reference_registry
from pyfem.v3.driver import DriverStatus
from pyfem.v3.fem.quadrature import gauss_legendre_1d
from pyfem.v3.fem.shapes import linear_line2
from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.operator import (
  BalanceRole,
  ChannelRequest,
  EvaluationStatus,
  OperatorEvaluationInput,
)
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


def _source(label: str) -> SourceContext:
  return SourceContext(source=label)


def _base_model(node_coordinates: tuple[tuple[float, float], ...]) -> ModelSpec:
  """A minimal truss-family system whose point block carries all nodes.

  The interface network under test hangs off nodes 0..3; the truss cell on
  the last two nodes exists only so the base system compiles.
  """
  nodes = tuple(
    NodeSpec(id=index, coordinates=coordinates, source=_source(f"n{index}"))
    for index, coordinates in enumerate(node_coordinates)
  )
  truss_nodes = (len(node_coordinates) - 2, len(node_coordinates) - 1)
  block = CellBlockSpec(
    id="bars",
    reference_topology="line",
    topological_dimension=1,
    embedding_dimension=2,
    geometry_interpolation="line2",
    cells=(CellSpec(id="c0", node_ids=truss_nodes, source=_source("c0")),),
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
      MaterialParameterSpec("youngs_modulus", 1.0e6),
      MaterialParameterSpec("area", 1.0),
    ),
    source=_source("material"),
  )
  region = RegionSpec(
    id="domain",
    cell_refs=(CellRef("bars", "c0"),),
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


# The interface quad: bottom pair nodes 0/1 on y=0, top pair nodes 2/3
# coincident with them; the truss cell closes the base system on nodes 4/5.
_UNIT_QUAD_COORDINATES = (
  (0.0, 0.0),
  (1.0, 0.0),
  (0.0, 0.0),
  (1.0, 0.0),
  (0.0, 1.0),
  (1.0, 1.0),
)


def _compile_base(
  coordinates: tuple[tuple[float, float], ...] = _UNIT_QUAD_COORDINATES,
) -> CompiledSystem:
  return compile_system(_base_model(coordinates), truss_reference_registry())


def _dummy_declaration(**overrides: object) -> InterfaceDeclaration:
  options: dict[str, object] = {
    "block_id": "coh",
    "space_id": "displacement",
    "interface_ids": (10,),
    "node_quads": ((0, 1, 2, 3),),
    "stiffness": 1.0e5,
    "source": _source("coh"),
  }
  options.update(overrides)
  return dummy_interface_declaration(**options)  # type: ignore[arg-type]


def _evaluate(
  operator: object,
  state: np.ndarray,
  accepted: np.ndarray,
  *,
  jacobian: bool = True,
) -> object:
  return operator.evaluate(
    OperatorEvaluationInput(
      port_values=(FinalizedArray(np.array(state), dtype=np.float64),),
      accepted_state=FinalizedArray(np.array(accepted), dtype=np.float64),
      signals=(),
      request=ChannelRequest(
        ("interface-traction",),
        ("interface-tangent",) if jacobian else (),
      ),
    )
  )


# ---------------------------------------------------------------------------
# Line2 shapes and the de-facto Gauss-2 quadrature


def test_line2_shape_values_match_legacy_bitwise(
  bitwise_pin: object,
) -> None:
  from pyfem.util.shapeFunctions import getShapeLine2

  gauss_points, _ = gauss_legendre_1d(2)
  xi = np.concatenate(
    (
      gauss_points,
      np.array([0.0, -1.0, 1.0, -0.377, 0.891, -0.999999, 0.5]),
    )
  )
  values, derivatives = linear_line2(xi.reshape(-1, 1))
  legacy_values = np.empty((len(xi), 2))
  legacy_derivatives = np.empty((len(xi), 2))
  for index, point in enumerate(xi):
    legacy = getShapeLine2(float(point))
    legacy_values[index] = legacy.h
    legacy_derivatives[index] = legacy.dhdxi[:, 0]
  # Byte-identity pins: both sides compute 0.5 * (1 +- xi) with no
  # transcendentals, so off-reference-platform deviation is exactly the
  # constant 0.5 arithmetic — the documented tolerance is pure formality.
  bitwise_pin(values, legacy_values, rtol=0.0, atol=0.0)
  bitwise_pin(derivatives[:, :, 0], legacy_derivatives, rtol=0.0, atol=0.0)
  np.testing.assert_array_equal(values.sum(axis=1), np.ones(len(xi)))


def test_legacy_newton_cotes_flag_is_silent_gauss(bitwise_pin: object) -> None:
  """Legacy 'NewtonCotes' on Line2 never existed: it returns Gauss-2 (M62 F2)."""
  from pyfem.util.shapeFunctions import getIntegrationPoints

  gauss_points, gauss_weights = getIntegrationPoints("Line2", 0, "Gauss")
  cotes_points, cotes_weights = getIntegrationPoints("Line2", 0, "NewtonCotes")
  assert gauss_points == cotes_points
  np.testing.assert_array_equal(np.asarray(gauss_weights), np.asarray(cotes_weights))
  v3_points, v3_weights = gauss_legendre_1d(2)
  # The v3 quadrature the interface operator reuses is bitwise the legacy
  # de-facto rule on the reference platform (scipy p_roots ==
  # roots_legendre at order 2, measured); 1e-15 covers any residual ulp
  # drift of the scipy root finder off it.
  bitwise_pin(
    v3_points,
    np.array(gauss_points, dtype=np.float64),
    rtol=0.0,
    atol=1.0e-15,
  )
  bitwise_pin(
    v3_weights,
    np.array(gauss_weights, dtype=np.float64),
    rtol=0.0,
    atol=1.0e-15,
  )


# ---------------------------------------------------------------------------
# Compile boundary: declaration validation and system resolution


def test_compile_rejects_nonexact_arguments() -> None:
  system = _compile_base()
  declaration = _dummy_declaration()
  with pytest.raises(TypeError, match="exact CompiledSystem"):
    compile_interface_operator(object(), declaration)  # type: ignore[arg-type]
  with pytest.raises(TypeError, match="exact InterfaceDeclaration"):
    compile_interface_operator(system, object())  # type: ignore[arg-type]


def test_declaration_validation_battery() -> None:
  with pytest.raises(TypeError, match="paired unique"):
    _dummy_declaration(node_quads=((0, 1, 2, 3), (0, 1, 2, 3)))
  with pytest.raises(TypeError, match="four semantic ids"):
    _dummy_declaration(node_quads=((0, 1, 2),))
  with pytest.raises(TypeError, match="non-empty exact string"):
    _dummy_declaration(state_schema="")
  with pytest.raises(ValueError, match="positive finite"):
    _dummy_declaration(stiffness=-1.0)
  with pytest.raises(ValueError, match="0 < d1d3 < d2d3 < 1"):
    thouless_mode_i_declaration(
      block_id="coh",
      space_id="displacement",
      interface_ids=(10,),
      node_quads=((0, 1, 2, 3),),
      fracture_energy=0.1,
      ultimate_traction=0.5,
      d1d3=0.7,
      d2d3=0.2,
      source=_source("coh"),
    )
  declaration = _dummy_declaration()
  with pytest.raises(TypeError, match="callable"):
    InterfaceDeclaration(
      block_id=declaration.block_id,
      space_id=declaration.space_id,
      interface_ids=declaration.interface_ids,
      node_quads=declaration.node_quads,
      state_schema=declaration.state_schema,
      kernel_name=declaration.kernel_name,
      kernel_version=declaration.kernel_version,
      implementation_id=declaration.implementation_id,
      parameters=declaration.parameters,
      kernel=None,  # type: ignore[arg-type]
      source=declaration.source,
    )


def test_compile_rejects_unknown_space() -> None:
  system = _compile_base()
  with pytest.raises(ModelCompilationError, match="unknown-interface-space"):
    compile_interface_operator(system, _dummy_declaration(space_id="strain"))


def test_compile_rejects_unknown_node() -> None:
  system = _compile_base()
  with pytest.raises(ModelCompilationError, match="unknown-interface-node"):
    compile_interface_operator(system, _dummy_declaration(node_quads=((0, 1, 2, 99),)))


def _hex8_system() -> CompiledSystem:
  """A minimal 3D system: the interface family pins a two-component space."""
  nodes = tuple(
    NodeSpec(id=index, coordinates=coordinates, source=_source(f"n{index}"))
    for index, coordinates in enumerate(
      (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
        (0.0, 1.0, 1.0),
      )
    )
  )
  block = CellBlockSpec(
    id="cells",
    reference_topology="hexahedron",
    topological_dimension=3,
    embedding_dimension=3,
    geometry_interpolation="trilinear-hex8",
    cells=(CellSpec(id=1, node_ids=tuple(range(8)), source=_source("c")),),
    source=_source("block"),
  )
  field = FieldSpec(
    id="displacement",
    components=("x", "y", "z"),
    location="node",
    source=_source("field"),
  )
  material = MaterialSpec(
    id="solid",
    model="isotropic-linear-elastic",
    parameters=(
      MaterialParameterSpec("youngs_modulus", 1.0e6),
      MaterialParameterSpec("poisson_ratio", 0.25),
    ),
    source=_source("material"),
  )
  region = RegionSpec(
    id="domain",
    cell_refs=(CellRef("cells", 1),),
    field_ids=("displacement",),
    material_id="solid",
    formulation="small-strain-continuum",
    quadrature="gauss-2x2x2",
    source=_source("region"),
  )
  model = ModelSpec(
    mesh=MeshSpec(nodes=nodes, cell_blocks=(block,), source=_source("mesh")),
    fields=(field,),
    materials=(material,),
    regions=(region,),
    source=_source("model"),
  )
  return compile_system(model, continuum_reference_registry())


def test_compile_rejects_non_two_component_space() -> None:
  system = _hex8_system()
  with pytest.raises(ModelCompilationError, match="unsupported-interface-space"):
    compile_interface_operator(system, _dummy_declaration())


def test_compile_rejects_degenerate_geometry() -> None:
  # Coincident bottom pair: the integration line collapses.
  degenerate_chord = (
    (0.0, 0.0),
    (0.0, 0.0),
    (0.0, 0.0),
    (1.0, 0.0),
    (0.0, 1.0),
    (1.0, 1.0),
  )
  system = _compile_base(degenerate_chord)
  with pytest.raises(ModelCompilationError, match="degenerate-interface-geometry"):
    compile_interface_operator(system, _dummy_declaration())
  # Coincident pair midpoints: the frame segment collapses.
  degenerate_frame = (
    (0.0, 0.0),
    (0.0, 0.0),
    (1.0, 0.0),
    (1.0, 0.0),
    (0.0, 1.0),
    (1.0, 1.0),
  )
  system = _compile_base(degenerate_frame)
  with pytest.raises(ModelCompilationError, match="degenerate-interface-geometry"):
    compile_interface_operator(system, _dummy_declaration())


# ---------------------------------------------------------------------------
# Fail-closed compile probes


def test_kernel_raising_at_the_virgin_state_fails_closed() -> None:
  def raising_kernel(
    jumps: np.ndarray,
    accepted_rows: np.ndarray,
    parameters: np.ndarray,
  ) -> InterfaceKernelResult:
    raise RuntimeError("boom")

  system = _compile_base()
  declaration = InterfaceDeclaration(
    block_id="coh",
    space_id="displacement",
    interface_ids=(10,),
    node_quads=((0, 1, 2, 3),),
    state_schema="test-raising",
    kernel_name="raising",
    kernel_version="1",
    implementation_id="test-raising-v1",
    parameters=(1.0,),
    kernel=raising_kernel,
    source=_source("coh"),
  )
  with pytest.raises(ModelCompilationError, match="kernel-probe-failed"):
    compile_interface_operator(system, declaration)


def test_kernel_rejecting_at_the_virgin_state_fails_closed() -> None:
  def rejecting_kernel(
    jumps: np.ndarray,
    accepted_rows: np.ndarray,
    parameters: np.ndarray,
  ) -> InterfaceKernelResult:
    return InterfaceKernelResult(
      traction=np.zeros_like(jumps),
      tangent=np.zeros((len(jumps), 2, 2)),
      trial_rows=np.array(accepted_rows, copy=True),
      status=EvaluationStatus.REJECT_STEP,
    )

  system = _compile_base()
  declaration = InterfaceDeclaration(
    block_id="coh",
    space_id="displacement",
    interface_ids=(10,),
    node_quads=((0, 1, 2, 3),),
    state_schema="test-rejecting",
    kernel_name="rejecting",
    kernel_version="1",
    implementation_id="test-rejecting-v1",
    parameters=(1.0,),
    kernel=rejecting_kernel,
    source=_source("coh"),
  )
  with pytest.raises(ModelCompilationError, match="invalid-kernel-probe"):
    compile_interface_operator(system, declaration)


def test_planted_kernel_tangent_wrong_only_at_nonzero_jumps_is_rejected() -> None:
  """The conviction leg of the law-level seeded FD probe."""

  def tampered_kernel(
    jumps: np.ndarray,
    accepted_rows: np.ndarray,
    parameters: np.ndarray,
  ) -> InterfaceKernelResult:
    (stiffness,) = parameters
    traction = stiffness * jumps
    tangent = np.zeros((len(jumps), 2, 2))
    # Exact at the virgin state, 1% off once both jump components are live.
    wrong = stiffness * (1.0 + 0.01 * (np.abs(jumps[:, 0]) > 0.0))
    tangent[:, 0, 0] = wrong
    tangent[:, 1, 1] = stiffness
    return InterfaceKernelResult(
      traction=traction,
      tangent=tangent,
      trial_rows=np.array(accepted_rows, copy=True),
      status=EvaluationStatus.OK,
    )

  system = _compile_base()
  declaration = InterfaceDeclaration(
    block_id="coh",
    space_id="displacement",
    interface_ids=(10,),
    node_quads=((0, 1, 2, 3),),
    state_schema="test-tampered",
    kernel_name="tampered",
    kernel_version="1",
    implementation_id="test-tampered-v1",
    parameters=(100.0,),
    kernel=tampered_kernel,
    source=_source("coh"),
  )
  with pytest.raises(ModelCompilationError) as excinfo:
    compile_interface_operator(system, declaration)
  diagnostic = excinfo.value.diagnostics[0]
  assert diagnostic.code == "inconsistent-kernel-tangent"
  assert "seed 20261009" in diagnostic.message
  # The failure replays bit-for-bit: recompiling raises the same diagnostic.
  with pytest.raises(ModelCompilationError) as replay:
    compile_interface_operator(system, declaration)
  assert replay.value.diagnostics[0].message == diagnostic.message


def test_operator_probe_convicts_a_tampered_assembly() -> None:
  """The conviction leg of the assembled-tangent FD probe."""
  import pyfem.v3.compile.interface as interface_module

  system = _compile_base()
  declaration = _dummy_declaration()
  _, operator = compile_interface_operator(system, declaration)
  payload = operator.payload

  def tampered_response(
    midpoints: np.ndarray,
    weights: np.ndarray,
    shapes: np.ndarray,
    kernel: object,
    parameters: np.ndarray,
    displacements: np.ndarray,
    accepted_rows: np.ndarray,
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, EvaluationStatus]:
    residual, tangent, trial_rows, status = interface_module._evaluate_response(
      midpoints,
      weights,
      shapes,
      kernel,
      parameters,
      displacements,
      accepted_rows,
    )
    return residual, tangent * 1.001, trial_rows, status

  with pytest.raises(ModelCompilationError) as excinfo:
    interface_module._probe_operator_tangent(
      payload.reference_midpoints.values,
      payload.integration_weights.values,
      payload.ip_shape_values.values,
      declaration.kernel,
      payload.parameters.values,
      source=_source("coh"),
      response=tampered_response,
    )
  diagnostic = excinfo.value.diagnostics[0]
  assert diagnostic.code == "inconsistent-operator-tangent"
  assert "seed 20261010" in diagnostic.message


def test_landed_law_kernels_pass_the_compile_probes() -> None:
  builders = (
    lambda **kwargs: dummy_interface_declaration(stiffness=1.0e5, **kwargs),
    lambda **kwargs: xu_needleman_declaration(
      fracture_energy=0.1,
      ultimate_traction=0.5,
      **kwargs,
    ),
    lambda **kwargs: power_law_mode_i_declaration(
      fracture_energy=0.1,
      ultimate_traction=0.5,
      **kwargs,
    ),
    lambda **kwargs: thouless_mode_i_declaration(
      fracture_energy=0.1,
      ultimate_traction=0.5,
      d1d3=0.2,
      d2d3=0.6,
      **kwargs,
    ),
  )
  for build in builders:
    system = _compile_base()
    declaration = build(
      block_id="coh",
      space_id="displacement",
      interface_ids=(10,),
      node_quads=((0, 1, 2, 3),),
      source=_source("coh"),
    )
    block, operator = compile_interface_operator(system, declaration)
    assert operator.header.entity_block_id == block.block_id == "coh"


# ---------------------------------------------------------------------------
# Frame state machine and channel honesty


def _compiled_dummy_operator() -> object:
  system = _compile_base()
  _, operator = compile_interface_operator(system, _dummy_declaration())
  return operator


def test_virgin_frame_bootstrap_uses_the_corrected_normal() -> None:
  operator = _compiled_dummy_operator()
  state = np.zeros((1, 8))
  accepted = np.zeros((1, 2))
  evaluation = _evaluate(operator, state, accepted)
  # The horizontal segment lifts the corrected normal (-ds_y, ds_x) / |ds|
  # to (0, 1) — legacy's reflecting frame agrees here by construction.
  np.testing.assert_allclose(
    evaluation.trial_state.values,
    np.array([[0.0, 1.0]]),
    rtol=0.0,
    atol=0.0,
  )


def test_frame_sign_continuity_flip_and_alignment() -> None:
  operator = _compiled_dummy_operator()
  state = np.zeros((1, 8))
  # An accepted normal anti-aligned with the candidate flips the trial.
  flipped = _evaluate(operator, state, np.array([[0.0, -1.0]]))
  np.testing.assert_allclose(
    flipped.trial_state.values,
    np.array([[0.0, -1.0]]),
    rtol=0.0,
    atol=0.0,
  )
  # An aligned accepted normal keeps the candidate.
  aligned = _evaluate(operator, state, np.array([[0.0, 1.0]]))
  np.testing.assert_allclose(
    aligned.trial_state.values,
    np.array([[0.0, 1.0]]),
    rtol=0.0,
    atol=0.0,
  )


def test_frame_follows_the_displaced_midpoint_segment() -> None:
  operator = _compiled_dummy_operator()
  # Rotate the segment vertical: the right pair moves up and left onto the
  # left pair's station (mid1 (1, 0) -> (0, 1)), so ds = (0, 1) and the
  # corrected normal is (-1, 0).
  state = np.array([[0.0, 0.0, -1.0, 1.0, 0.0, 0.0, -1.0, 1.0]])
  evaluation = _evaluate(operator, state, np.zeros((1, 2)))
  np.testing.assert_allclose(
    evaluation.trial_state.values,
    np.array([[-1.0, 0.0]]),
    rtol=0.0,
    atol=1.0e-15,
  )


def test_channel_flags_are_honest() -> None:
  operator = _compiled_dummy_operator()
  header = operator.header
  assert [channel.channel_id for channel in header.residual_channels] == [
    "interface-traction"
  ]
  assert [channel.channel_id for channel in header.jacobian_channels] == [
    "interface-tangent"
  ]
  residual = header.residual_channels[0]
  jacobian = header.jacobian_channels[0]
  assert residual.balance_role is BalanceRole.INTERNAL
  assert residual.linear is False
  assert jacobian.balance_role is BalanceRole.INTERNAL
  assert jacobian.linear is False
  # The exact tangent carries the frame-motion terms; its true Jacobian is
  # measurably nonsymmetric (survey: ~1% at finite openings), so the channel
  # declares symmetric=False.
  assert jacobian.symmetric is False
  assert header.ports[0].coefficient_map.values.shape == (1, 8)
  assert header.state_layout.row_width == 2
  assert header.state_layout.slots[0].name == "normal"


def test_virgin_tangent_is_symmetric_and_deformed_tangent_is_not() -> None:
  operator = _compiled_dummy_operator()
  virgin = _evaluate(operator, np.zeros((1, 8)), np.zeros((1, 2)))
  tangent = virgin.jacobian_values[0].values[0]
  # At the virgin state the traction and the jump vanish, so both
  # frame-motion terms vanish and the tangent is the symmetric material part.
  np.testing.assert_array_equal(tangent, tangent.T)

  # The nonsymmetry needs a law with distinct normal/shear stiffnesses: for
  # the isotropic Dummy law the two frame-motion terms cancel exactly (the
  # assembled response is frame-independent). Xu-Needleman at a finite
  # sheared opening shows the survey's ~1% true-Jacobian nonsymmetry.
  system = _compile_base()
  _, xn_operator = compile_interface_operator(
    system,
    xu_needleman_declaration(
      block_id="coh",
      space_id="displacement",
      interface_ids=(10,),
      node_quads=((0, 1, 2, 3),),
      fracture_energy=0.1,
      ultimate_traction=0.5,
      source=_source("coh"),
    ),
  )
  state = np.array([[0.02, 0.01, -0.03, 0.02, 0.01, -0.02, -0.02, 0.03]])
  deformed = _evaluate(xn_operator, state, np.zeros((1, 2)))
  tangent = deformed.jacobian_values[0].values[0]
  scale = float(np.abs(tangent).max())
  assert float(np.abs(tangent - tangent.T).max()) > 1.0e-3 * scale


def test_kernel_rejection_propagates_byte_equal_trial_state() -> None:
  """A typed law rejection mid-solve returns accepted rows byte-equal."""

  def selective_kernel(
    jumps: np.ndarray,
    accepted_rows: np.ndarray,
    parameters: np.ndarray,
  ) -> InterfaceKernelResult:
    (stiffness,) = parameters
    if bool((np.abs(jumps) > 5.0).any()):
      return InterfaceKernelResult(
        traction=np.zeros_like(jumps),
        tangent=np.zeros((len(jumps), 2, 2)),
        trial_rows=np.array(accepted_rows, copy=True),
        status=EvaluationStatus.REJECT_STEP,
      )
    tangent = np.zeros((len(jumps), 2, 2))
    tangent[:, 0, 0] = stiffness
    tangent[:, 1, 1] = stiffness
    return InterfaceKernelResult(
      traction=stiffness * jumps,
      tangent=tangent,
      trial_rows=np.array(accepted_rows, copy=True),
      status=EvaluationStatus.OK,
    )

  system = _compile_base()
  declaration = InterfaceDeclaration(
    block_id="coh",
    space_id="displacement",
    interface_ids=(10,),
    node_quads=((0, 1, 2, 3),),
    state_schema="test-selective",
    kernel_name="selective",
    kernel_version="1",
    implementation_id="test-selective-v1",
    parameters=(100.0,),
    kernel=selective_kernel,
    source=_source("coh"),
  )
  # The seeded probe states sit at |jump| <= 5, so the typed rejection is
  # skipped there and the declaration compiles clean (spring idiom).
  _, operator = compile_interface_operator(system, declaration)
  accepted = np.array([[0.25, -0.5]])
  # A genuine top-minus-bottom jump of (10, 10) trips the law's rejection.
  rejected = _evaluate(
    operator,
    np.array([[0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0]]),
    accepted,
  )
  assert rejected.status is EvaluationStatus.REJECT_STEP
  assert rejected.residual_values == ()
  assert rejected.jacobian_values == ()
  np.testing.assert_array_equal(
    rejected.trial_state.values.view(np.uint64),
    accepted.view(np.uint64),
  )


def test_evaluate_rejects_malformed_inputs() -> None:
  operator = _compiled_dummy_operator()
  state = np.zeros((1, 8))
  accepted = np.zeros((1, 2))
  with pytest.raises(TypeError, match="exact immutable evaluation input"):
    operator.evaluate(object())  # type: ignore[arg-type]
  with pytest.raises(TypeError, match="exactly one displacement port batch"):
    operator.evaluate(
      OperatorEvaluationInput(
        port_values=(),
        accepted_state=FinalizedArray(accepted, dtype=np.float64),
        signals=(),
        request=ChannelRequest(("interface-traction",), ()),
      )
    )
  with pytest.raises(TypeError, match="finite float64 batch"):
    operator.evaluate(
      OperatorEvaluationInput(
        port_values=(FinalizedArray(np.full((1, 8), np.nan), dtype=np.float64),),
        accepted_state=FinalizedArray(accepted, dtype=np.float64),
        signals=(),
        request=ChannelRequest(("interface-traction",), ()),
      )
    )
  with pytest.raises(TypeError, match="match the compiled state layout"):
    operator.evaluate(
      OperatorEvaluationInput(
        port_values=(FinalizedArray(state, dtype=np.float64),),
        accepted_state=FinalizedArray(np.zeros((1, 3)), dtype=np.float64),
        signals=(),
        request=ChannelRequest(("interface-traction",), ()),
      )
    )
  with pytest.raises(ValueError, match="unavailable or duplicate channel"):
    operator.evaluate(
      OperatorEvaluationInput(
        port_values=(FinalizedArray(state, dtype=np.float64),),
        accepted_state=FinalizedArray(accepted, dtype=np.float64),
        signals=(),
        request=ChannelRequest(("spring-force",), ()),
      )
    )
  with pytest.raises(ValueError, match="unavailable or duplicate channel"):
    operator.evaluate(
      OperatorEvaluationInput(
        port_values=(FinalizedArray(state, dtype=np.float64),),
        accepted_state=FinalizedArray(accepted, dtype=np.float64),
        signals=(),
        request=ChannelRequest(("interface-traction",), (), ("d-residual",)),
      )
    )


def test_compose_rejects_foreign_and_duplicate_blocks() -> None:
  system = _compile_base()
  block, operator = compile_interface_operator(system, _dummy_declaration())
  composed = compose_interface_system(system, block, operator)
  assert composed.operators[-1] is operator
  assert composed.entity_blocks[-1] is block
  with pytest.raises(TypeError, match="exact base CompiledSystem"):
    compose_interface_system(object(), block, operator)  # type: ignore[arg-type]
  # A second declaration with the same block id compiles against the composed
  # system but collides at composition.
  block2, operator2 = compile_interface_operator(composed, _dummy_declaration())
  with pytest.raises(ValueError, match="collides with an existing compiled"):
    compose_interface_system(composed, block2, operator2)
  # The operator's owning instance is pinned: composing onto another live
  # system of identical content still fails closed.
  other_system = _compile_base()
  with pytest.raises(ValueError, match="exact same live instance"):
    compose_interface_system(other_system, block, operator)


# ---------------------------------------------------------------------------
# Legacy-deck converter admission

_MINI_INTERFACE_DAT = """<Nodes>
 1 0.0 0.0 ;
 2 1.0 0.0 ;
 3 0.0 0.5 ;
 4 1.0 0.5 ;
 5 0.0 0.5 ;
 6 1.0 0.5 ;
 7 0.0 1.0 ;
 8 1.0 1.0 ;
</Nodes>

<Elements>
 1 'ContElem' 1 2 4 3 ;
 2 'ContElem' 5 6 8 7 ;
 3 'InterfaceElem' 3 4 5 6 ;
</Elements>

<NodeConstraints>
 u[1] = 0.0;
 v[1] = 0.0;
 u[3] = 0.0;
 v[3] = 0.0;
 u[5] = 0.0;
 v[5] = 0.0;
 u[7] = 0.0;
 v[7] = 0.0;
 v[2] = -0.05;
 v[8] = 0.05;
</NodeConstraints>

<ExternalForces>
</ExternalForces>
"""

_NONLINEAR_SOLVER_BODY = """  type = "NonlinearSolver";
  tol = 1.0e-10;
  iterMax = 25;
  loadTable = [0.5, 1.0];
"""

_LINEAR_SOLVER_BODY = '  type = "LinearSolver";'


def _mini_pro(
  material_block: str | None,
  solver_body: str = _NONLINEAR_SOLVER_BODY,
) -> str:
  material = (
    ""
    if material_block is None
    else "  material =\n  {\n" + material_block + "\n  };\n"
  )
  return (
    'input = "mini.dat";\n\n'
    "ContElem =\n{\n"
    '  type = "SmallStrainContinuum";\n\n'
    "  material =\n  {\n"
    '    type = "PlaneStrain";\n'
    "    E    = 100.0;\n"
    "    nu   = 0.3;\n"
    "  };\n};\n\n"
    "InterfaceElem =\n{\n"
    '  type = "Interface";\n\n'
    f"{material}"
    "};\n\n"
    "solver =\n{\n"
    f"{solver_body}"
    "};\n\n"
    'outputModules = ["output"];\n\n'
    "output =\n{\n"
    '  type = "OutputWriter";\n'
    "};\n"
  )


def _write_deck(
  tmp_path: Path, pro_text: str, dat_text: str = _MINI_INTERFACE_DAT
) -> Path:
  (tmp_path / "mini.dat").write_text(dat_text, encoding="utf-8")
  pro_path = tmp_path / "mini.pro"
  pro_path.write_text(pro_text, encoding="utf-8")
  return pro_path


def _convert(
  tmp_path: Path,
  pro_text: str,
  dat_text: str = _MINI_INTERFACE_DAT,
) -> object:
  from pyfem.v3.io.legacy_deck import read_legacy_deck

  return read_legacy_deck(_write_deck(tmp_path, pro_text, dat_text))


def _rejection_codes(excinfo: pytest.ExceptionInfo) -> set[str]:
  return {diagnostic.code for diagnostic in excinfo.value.diagnostics}


_LAW_BLOCKS = (
  (
    "xu-needleman",
    '    type = "XuNeedleman";\n    Tult = 0.5;\n    Gc   = 0.1;',
    "xu-needleman-rank2",
    (0.1, 0.5),
  ),
  (
    "power-law",
    '    type = "PowerLawModeI";\n    Tult = 0.5;\n    Gc   = 0.1;',
    "power-law-mode-i",
    (0.1, 0.5),
  ),
  (
    "thouless",
    '    type = "ThoulessModeI";\n    Tult = 0.5;\n    Gc   = 0.1;\n'
    "    d1d3 = 0.2;\n    d2d3 = 0.6;",
    "thouless-mode-i",
    (0.1, 0.5, 0.2, 0.6),
  ),
  (
    "dummy",
    '    type = "Dummy";\n    D    = 1.0e5;',
    "dummy-linear-interface",
    (1.0e5,),
  ),
)


@pytest.mark.parametrize(
  ("_case", "material_block", "kernel_name", "expected_parameters"),
  _LAW_BLOCKS,
  ids=[case[0] for case in _LAW_BLOCKS],
)
def test_interface_deck_converts_each_law(
  _case: str,
  material_block: str,
  kernel_name: str,
  expected_parameters: tuple[float, ...],
  tmp_path: Path,
) -> None:
  deck = _convert(tmp_path, _mini_pro(material_block))
  assert deck.springs == ()
  assert len(deck.interfaces) == 1
  declaration = deck.interfaces[0]
  assert declaration.block_id == "InterfaceElem"
  assert declaration.interface_ids == (3,)
  assert declaration.node_quads == ((3, 4, 5, 6),)
  assert declaration.kernel_name == kernel_name
  assert declaration.parameters == expected_parameters
  assert declaration.state_schema == "legacy-deck-interface-frame-state-v1"


def test_interface_deck_compiles_and_runs(tmp_path: Path) -> None:
  from pyfem.v3.io.legacy_deck import compile_deck, run_deck

  deck = _convert(tmp_path, _mini_pro(_LAW_BLOCKS[0][1]))
  compiled = compile_deck(deck)
  assert [type(operator).__name__ for operator in compiled.system.operators] == [
    "Q8ContinuumOperator",
    "InterfaceOperator",
  ]
  run = run_deck(deck)
  assert run.result.status is DriverStatus.COMPLETED
  assert run.result.statistics.committed_substep_count == 2


def test_committed_frame_state_matches_the_corrected_frame(tmp_path: Path) -> None:
  """The driver's committed 'normal' rows are the corrected frame of the
  committed geometry (virgin bootstrap, no flips on this deck)."""
  from pyfem.v3.io.legacy_deck import run_deck

  deck = _convert(tmp_path, _mini_pro(_LAW_BLOCKS[0][1]))
  run = run_deck(deck)
  assert run.result.status is DriverStatus.COMPLETED
  driver = run.driver
  operator = next(
    item for item in driver.owner.system.operators if hasattr(item, "interface_block")
  )
  layout = operator.header.state_layout
  accepted = driver.owner.accepted_state(layout.block_id).values
  assert accepted.shape == (1, 2)
  # Reconstruct the corrected frame from the committed state by hand.
  payload = operator.payload
  midpoints = payload.reference_midpoints.values[0].copy()
  state = driver.owner.accepted_physical().values
  element_dofs = operator.header.ports[0].coefficient_map.values[0]
  a = state[element_dofs]
  midpoints[0, 0] += 0.5 * (a[0] + a[4])
  midpoints[0, 1] += 0.5 * (a[1] + a[5])
  midpoints[1, 0] += 0.5 * (a[2] + a[6])
  midpoints[1, 1] += 0.5 * (a[3] + a[7])
  ds = midpoints[1] - midpoints[0]
  length = math.hypot(ds[0], ds[1])
  expected = np.array([-ds[1] / length, ds[0] / length])
  np.testing.assert_allclose(accepted[0], expected, rtol=0.0, atol=1.0e-15)
  # The peel stays near-horizontal: the normal is the upward unit vector.
  np.testing.assert_allclose(accepted[0], [0.0, 1.0], rtol=0.0, atol=1.0e-9)


def test_interface_deck_rejection_battery(tmp_path: Path) -> None:
  from pyfem.v3.io.legacy_deck import DeckConversionError

  cases = {
    "unknown-law": (
      _mini_pro('    type = "CohesiveZoneModel";\n    Tult = 0.5;\n    Gc = 0.1;'),
      {"unsupported-material-model"},
    ),
    "xu-needleman-q-key": (
      _mini_pro(
        '    type = "XuNeedleman";\n    Tult = 0.5;\n    Gc = 0.1;\n    q = 0.5;'
      ),
      {"unsupported-material-parameter"},
    ),
    "xu-needleman-r-key": (
      _mini_pro(
        '    type = "XuNeedleman";\n    Tult = 0.5;\n    Gc = 0.1;\n    r = 0.0;'
      ),
      {"unsupported-material-parameter"},
    ),
    "missing-gc": (
      _mini_pro('    type = "XuNeedleman";\n    Tult = 0.5;'),
      {"missing-material-parameter"},
    ),
    "non-positive-gc": (
      _mini_pro('    type = "XuNeedleman";\n    Tult = 0.5;\n    Gc = -0.1;'),
      {"invalid-material-parameter-value"},
    ),
    "thouless-bad-ordering": (
      _mini_pro(
        '    type = "ThoulessModeI";\n    Tult = 0.5;\n    Gc = 0.1;\n'
        "    d1d3 = 0.6;\n    d2d3 = 0.2;"
      ),
      {"invalid-material-parameter-value"},
    ),
    "contact-still-excluded": (
      _mini_pro('    type = "Dummy";\n    D = 1.0e5;').replace(
        'type = "Interface"',
        'type = "Contact"',
      ),
      {"unsupported-element-type"},
    ),
    "no-material-block": (
      _mini_pro(None),
      {"missing-material-parameter"},
    ),
    "nonlinear-law-under-linear-solver": (
      _mini_pro(
        '    type = "XuNeedleman";\n    Tult = 0.5;\n    Gc   = 0.1;',
        solver_body=_LINEAR_SOLVER_BODY,
      ),
      {"incompatible-solver-type"},
    ),
  }
  for name, (pro_text, expected) in cases.items():
    with pytest.raises(DeckConversionError) as excinfo:
      _convert(tmp_path, pro_text)
    missing = expected - _rejection_codes(excinfo)
    assert not missing, (
      f"{name}: missing codes {missing} in {_rejection_codes(excinfo)}"
    )


def test_interface_arity_rejection(tmp_path: Path) -> None:
  from pyfem.v3.io.legacy_deck import DeckConversionError

  dat = _MINI_INTERFACE_DAT.replace(
    " 3 'InterfaceElem' 3 4 5 6 ;",
    " 3 'InterfaceElem' 3 4 5 ;",
  )
  with pytest.raises(DeckConversionError) as excinfo:
    _convert(tmp_path, _mini_pro(_LAW_BLOCKS[3][1]), dat)
  assert "unsupported-cell-arity" in _rejection_codes(excinfo)


def test_dummy_interface_under_linear_solver_converts(tmp_path: Path) -> None:
  from pyfem.v3.io.legacy_deck import run_deck

  deck = _convert(
    tmp_path,
    _mini_pro(_LAW_BLOCKS[3][1], solver_body=_LINEAR_SOLVER_BODY),
  )
  run = run_deck(deck)
  assert run.result.status is DriverStatus.COMPLETED
  # The linear law converges in one Newton correction.
  assert run.result.statistics.linear_solve_count == 1


def test_multiple_interface_groups_convert(tmp_path: Path) -> None:
  dat = _MINI_INTERFACE_DAT.replace(
    " 3 'InterfaceElem' 3 4 5 6 ;",
    " 3 'InterfaceElem' 3 4 5 6 ;\n 4 'InterfaceElem2' 3 4 5 6 ;",
  )
  pro = (
    _mini_pro(_LAW_BLOCKS[3][1])
    + """InterfaceElem2 =
{
  type = "Interface";

  material =
  {
    type = "PowerLawModeI";
    Tult = 0.5;
    Gc   = 0.1;
  };
};
"""
  )
  deck = _convert(tmp_path, pro, dat)
  assert [declaration.block_id for declaration in deck.interfaces] == [
    "InterfaceElem",
    "InterfaceElem2",
  ]
  assert deck.interfaces[0].kernel_name == "dummy-linear-interface"
  assert deck.interfaces[1].kernel_name == "power-law-mode-i"


def test_unused_interface_block_rejects(tmp_path: Path) -> None:
  from pyfem.v3.io.legacy_deck import DeckConversionError

  pro = (
    _mini_pro(_LAW_BLOCKS[3][1])
    + """GhostElem =
{
  type = "Interface";

  material =
  {
    type = "Dummy";
    D    = 1.0e5;
  };
};
"""
  )
  with pytest.raises(DeckConversionError) as excinfo:
    _convert(tmp_path, pro)
  assert "unsupported-pro-construct" in _rejection_codes(excinfo)
