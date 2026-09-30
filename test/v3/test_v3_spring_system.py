# SPDX-License-Identifier: MIT

"""System-level oracles for the spring family dispatch wiring (M35).

The spring operator family routes into compiled multi-group systems through
``compile_system``'s ``springs`` channel: one exact ``SpringDeclaration`` per
authored spring group, each binding its own kernel. Built programmatically
here — the converter's multi-group emission is wave 8 — the oracles pin, for
the shallow-truss-with-spring configuration of the M18/M32 decks:

- composes: the wired system is content-identical to the landed standalone
  spring compile path (``compile_spring_operator`` + ``compose_system``), up
  to equal content fingerprints;
- evaluates: both systems' operators return bit-identical residual and
  tangent values at a probe state, and the spring residual matches the
  closed-form axial force;
- drives: the landed nonlinear driver walks both systems through the same
  ramp with bit-identical trajectories, and the grounded spring relieves the
  support reactions by exactly its closed-form force;
- multi-group: two spring groups with distinct kernel bindings compose in
  one system — the acceptance shape prepared for the wave-8 converter, which
  only adds parsing (one declaration per parsed spring group);
- diagnostics: the landed coded diagnostics and multi-group membership
  validation are preserved on the wired path, and a spec region naming the
  spring formulation fails with a truthful coded refusal.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from pyfem.v3.compile.diagnostics import ModelCompilationError
from pyfem.v3.compile.spring import (
  SpringDeclaration,
  SpringKernelResult,
  SpringOperator,
  SpringStateSlot,
  compile_spring_operator,
  compose_system,
  damage_envelope_declaration,
)
from pyfem.v3.compile.system import compile_system
from pyfem.v3.compile.truss import TrussOperator, truss_reference_registry
from pyfem.v3.constraints import compile_constraint_map
from pyfem.v3.driver import (
  DriverStatus,
  NonlinearStaticDriver,
  NonlinearStaticResult,
)
from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.operator import (
  ChannelRequest,
  EvaluationStatus,
  OperatorEvaluation,
  OperatorEvaluationInput,
  evaluation_status,
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
from pyfem.v3.spec.program import (
  AffineCoefficientSpec,
  AffineValueSpec,
  DofRef,
  NodalLoadSpec,
  PrescribedDofSpec,
  ProgramCoordinateSpec,
  ProgramCoordinateValue,
  ProgramPoint,
)
from pyfem.v3.state import StateTransactionOwner


def _source(label: str) -> SourceContext:
  return SourceContext(source=label)


def _shallow_truss_model() -> ModelSpec:
  """The ch.4 shallow-truss geometry of the M18/M32 decks (nodes renumbered)."""
  nodes = (
    NodeSpec(id=0, coordinates=(-10.0, 0.0), source=_source("n0")),
    NodeSpec(id=1, coordinates=(10.0, 0.0), source=_source("n1")),
    NodeSpec(id=2, coordinates=(0.0, 0.5), source=_source("n2")),
  )
  cells = (
    CellSpec(id="left", node_ids=(0, 2), source=_source("c0")),
    CellSpec(id="right", node_ids=(1, 2), source=_source("c1")),
  )
  block = CellBlockSpec(
    id="bars",
    reference_topology="line",
    topological_dimension=1,
    embedding_dimension=2,
    geometry_interpolation="line2",
    cells=cells,
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
      MaterialParameterSpec("youngs_modulus", 5.0e6),
      MaterialParameterSpec("area", 1.0),
    ),
    source=_source("material"),
  )
  region = RegionSpec(
    id="domain",
    cell_refs=(CellRef("bars", "left"), CellRef("bars", "right")),
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


def axial_spring_kernel(
  displacements: np.ndarray,
  accepted_rows: np.ndarray,
  parameters: np.ndarray,
) -> SpringKernelResult:
  """Linear point spring grounded along a unit global direction.

  Internal force ``k (u.d) d`` with its exact tangent ``k d d^T``: the
  H1-pinned consistent axial pair. The Riks skim's ``SpringElem`` is the
  vertical (``d = (0, 1)``) grounding of the apex with ``k = 100``.
  """
  stiffness, direction_x, direction_y = parameters
  direction = np.array([direction_x, direction_y], dtype=np.float64)
  force = stiffness * (displacements @ direction)[:, None] * direction[None, :]
  tangent = stiffness * np.outer(direction, direction)
  return SpringKernelResult(
    force=force,
    tangent=np.broadcast_to(tangent, (len(displacements), 2, 2)).copy(),
    trial_rows=np.array(accepted_rows, copy=True),
    status=EvaluationStatus.OK,
  )


def planted_nonzero_tangent_kernel(
  displacements: np.ndarray,
  accepted_rows: np.ndarray,
  parameters: np.ndarray,
) -> SpringKernelResult:
  """Planted violation: the force is a consistent linear spring everywhere,
  but the tangent halves off the virgin state, so the kernel passes a
  virgin-only probe and evaluates silently wrong at nonzero accepted state.
  """
  stiffness = float(parameters[0])
  tangent_scale = stiffness if not bool(np.any(accepted_rows)) else 0.5 * stiffness
  return SpringKernelResult(
    force=stiffness * displacements,
    tangent=tangent_scale * np.tile(np.eye(2), (len(displacements), 1, 1)),
    trial_rows=np.array(accepted_rows, copy=True),
    status=EvaluationStatus.OK,
  )


def _apex_spring_declaration() -> SpringDeclaration:
  """The Riks skim's grounded vertical apex spring, as one spring group."""
  return SpringDeclaration(
    block_id="apex-spring",
    space_id="displacement",
    spring_ids=("apex-support",),
    node_ids=(2,),
    state_schema="apex-spring-state-v1",
    state_slots=(),
    kernel_name="axial-linear-spring",
    kernel_version="1",
    implementation_id="apex-spring-v1",
    parameters=(100.0, 0.0, 1.0),
    kernel=axial_spring_kernel,
    source=_source("apex-spring-source"),
  )


def _damage_declaration() -> SpringDeclaration:
  """A second spring group on the apex binding the landed damage kernel."""
  return damage_envelope_declaration(
    block_id="apex-damage",
    space_id="displacement",
    spring_ids=("apex-damage-1",),
    node_ids=(2,),
    stiffness=2.0,
    critical_extension=2.0,
    max_increment=5.0,
    source=_source("apex-damage-source"),
  )


def _base_system() -> CompiledSystem:
  return compile_system(_shallow_truss_model(), truss_reference_registry())


def _landed_system(*declarations: SpringDeclaration) -> CompiledSystem:
  """Compile through the standalone spring path: stepwise compile + compose."""
  system = _base_system()
  for declaration in declarations:
    block, operator = compile_spring_operator(system, declaration)
    system = compose_system(system, block, operator)
  return system


def _wired_system(*declarations: SpringDeclaration) -> CompiledSystem:
  """Compile through the spring family dispatch: one compile_system call."""
  return compile_system(
    _shallow_truss_model(),
    truss_reference_registry(),
    springs=declarations,
  )


def _driver(system: CompiledSystem) -> NonlinearStaticDriver:
  coordinate_map = compile_constraint_map(
    system,
    constraints=tuple(
      PrescribedDofSpec(
        id=f"fix-{node}-{component}",
        target=DofRef(
          node_id=node,
          field_id="displacement",
          component=component,
        ),
        value=AffineValueSpec(constant=0.0),
        source=_source(f"fix-{node}-{component}"),
      )
      for node in (0, 1)
      for component in ("x", "y")
    ),
    coordinates=(ProgramCoordinateSpec(name="load", kind="load"),),
  )
  loads = (
    NodalLoadSpec(
      id="apex",
      target=DofRef(node_id=2, field_id="displacement", component="y"),
      value=AffineValueSpec(
        coefficients=(AffineCoefficientSpec("load", -100.0, _source("coef")),),
        source=_source("load"),
      ),
      source=_source("apex"),
    ),
  )
  return NonlinearStaticDriver(system, coordinate_map, loads)


def _ramp_points(factors: tuple[float, ...]) -> tuple[ProgramPoint, ...]:
  return tuple(
    ProgramPoint((ProgramCoordinateValue("load", factor),)) for factor in factors
  )


def _run_ramp(
  system: CompiledSystem,
  factors: tuple[float, ...],
) -> tuple[NonlinearStaticResult, NonlinearStaticDriver]:
  driver = _driver(system)
  result = driver.run(
    base_point=_ramp_points((0.0,))[0],
    target_points=_ramp_points(factors),
  )
  return result, driver


def _evaluate(
  system: CompiledSystem,
  trial_physical: np.ndarray,
) -> tuple[OperatorEvaluation, ...]:
  """Evaluate every operator of a system at one trial vector, zero state."""
  owner = StateTransactionOwner(system)
  transaction = owner.begin()
  evaluations = []
  for operator in system.operators:
    header = operator.header
    batch = trial_physical[header.ports[0].coefficient_map.values]
    evaluations.append(
      operator.evaluate(
        OperatorEvaluationInput(
          port_values=(FinalizedArray(batch, dtype=np.float64),),
          accepted_state=transaction.accepted_state(header.state_layout.block_id),
          signals=(),
          request=ChannelRequest(
            tuple(item.channel_id for item in header.residual_channels),
            tuple(item.channel_id for item in header.jacobian_channels),
          ),
        )
      )
    )
  transaction.reject()
  return tuple(evaluations)


def test_spring_dispatch_composes_shallow_truss_with_spring() -> None:
  declaration = _apex_spring_declaration()
  landed = _landed_system(declaration)
  wired = _wired_system(declaration)
  # Content identity: the wired path reproduces the landed spring compile
  # path byte-for-byte; only the instance identity is fresh.
  assert wired.content_fingerprint == landed.content_fingerprint
  assert wired.instance_id != landed.instance_id
  assert wired.provenance.schema == landed.provenance.schema
  assert wired.registry_snapshot.fingerprint == landed.registry_snapshot.fingerprint
  assert tuple(type(operator) for operator in wired.operators) == (
    TrussOperator,
    SpringOperator,
  )
  assert tuple(block.block_id for block in wired.point_blocks) == (
    "nodes",
    "apex-spring",
  )
  assert tuple(block.block_id for block in wired.entity_blocks) == ("bars",)
  np.testing.assert_array_equal(
    wired.entity_blocks[0].incidence.values,
    landed.entity_blocks[0].incidence.values,
  )
  assert tuple(space.space_id for space in wired.spaces) == ("displacement",)
  assert wired.coefficient_count == 6
  spring = wired.operators[1]
  assert isinstance(spring, SpringOperator)
  header = spring.header
  assert header.block_id == ("apex-spring", "apex-spring-state-v1")
  assert header.entity_block_id == "apex-spring"
  assert tuple(item.channel_id for item in header.residual_channels) == (
    "spring-force",
  )
  assert tuple(item.channel_id for item in header.jacobian_channels) == (
    "spring-tangent",
  )
  assert header.state_layout.row_shape == (1, 0)
  np.testing.assert_array_equal(
    header.ports[0].coefficient_map.values,
    [[4, 5]],
  )
  np.testing.assert_array_equal(
    spring.spring_block.reference_coordinates.values,
    [[0.0, 0.5]],
  )


def test_wired_system_evaluates_identically_to_landed_spring_path() -> None:
  declaration = _apex_spring_declaration()
  trial = np.array([0.0, 0.0, 0.0, 0.0, 0.3, -0.7], dtype=np.float64)
  landed_evaluations = _evaluate(_landed_system(declaration), trial)
  wired_evaluations = _evaluate(_wired_system(declaration), trial)
  for landed_eval, wired_eval in zip(
    landed_evaluations,
    wired_evaluations,
    strict=True,
  ):
    assert evaluation_status(landed_eval) is EvaluationStatus.OK
    assert evaluation_status(wired_eval) is EvaluationStatus.OK
    for left, right in zip(
      landed_eval.residual_values,
      wired_eval.residual_values,
      strict=True,
    ):
      np.testing.assert_array_equal(left.values, right.values)
    for left, right in zip(
      landed_eval.jacobian_values,
      wired_eval.jacobian_values,
      strict=True,
    ):
      np.testing.assert_array_equal(left.values, right.values)
  # Closed-form axial spring response at the probe: k (u.d) d with k = 100,
  # d = (0, 1), u_apex = (0.3, -0.7).
  spring_eval = wired_evaluations[1]
  np.testing.assert_array_equal(
    spring_eval.residual_values[0].values,
    [[0.0, -70.0]],
  )
  np.testing.assert_array_equal(
    spring_eval.jacobian_values[0].values,
    [[[0.0, 0.0], [0.0, 100.0]]],
  )


def test_wired_system_drives_identically_to_landed_spring_path() -> None:
  declaration = _apex_spring_declaration()
  landed_result, landed_driver = _run_ramp(
    _landed_system(declaration),
    (0.25, 0.5, 0.75, 1.0),
  )
  wired_result, wired_driver = _run_ramp(
    _wired_system(declaration),
    (0.25, 0.5, 0.75, 1.0),
  )
  assert landed_result.status is DriverStatus.COMPLETED
  assert wired_result.status is DriverStatus.COMPLETED
  assert wired_result.statistics.committed_substep_count == 4
  assert wired_result.statistics.factorization_count == (
    landed_result.statistics.factorization_count
  )
  assert [len(record.iterations) for record in wired_result.records] == [
    len(record.iterations) for record in landed_result.records
  ]
  landed_state = landed_driver.owner.accepted_physical().values
  wired_state = wired_driver.owner.accepted_physical().values
  np.testing.assert_array_equal(wired_state, landed_state)
  # The grounded spring carries k |v_apex| of the applied -100 apex load, so
  # the support reactions equilibrate only the remainder.
  observation = wired_result.records[-1].observation
  assert observation is not None
  reactions = observation.reactions.values
  apex_vertical = float(wired_state[5])
  assert apex_vertical < 0.0
  assert float(reactions[[1, 3]].sum()) == pytest.approx(
    100.0 + 100.0 * apex_vertical,
    abs=1.0e-6,
  )


def test_multiple_spring_groups_with_distinct_kernels_compose() -> None:
  """Two spring groups, two kernel bindings, one compiled system.

  This is the acceptance shape prepared for the wave-8 converter: it parses
  each legacy spring element group into one ``SpringDeclaration`` — distinct
  block, kernel, parameters, and state schema per group — and the system
  wiring composes them all; the io mission only adds parsing.
  """
  declarations = (_apex_spring_declaration(), _damage_declaration())
  landed = _landed_system(*declarations)
  wired = _wired_system(*declarations)
  assert wired.content_fingerprint == landed.content_fingerprint
  assert tuple(type(operator) for operator in wired.operators) == (
    TrussOperator,
    SpringOperator,
    SpringOperator,
  )
  assert tuple(block.block_id for block in wired.point_blocks) == (
    "nodes",
    "apex-spring",
    "apex-damage",
  )
  axial = wired.operators[1]
  damage = wired.operators[2]
  assert isinstance(axial, SpringOperator)
  assert isinstance(damage, SpringOperator)
  assert axial.header.state_layout.row_shape == (1, 0)
  assert damage.header.state_layout.row_shape == (1, 1)
  assert damage.header.state_layout.slots[0].name == "max_extension"
  np.testing.assert_array_equal(
    axial.payload.parameters.values,
    [100.0, 0.0, 1.0],
  )
  np.testing.assert_array_equal(
    damage.payload.parameters.values,
    [2.0, 2.0, 5.0],
  )
  landed_result, landed_driver = _run_ramp(landed, (0.5, 1.0))
  wired_result, wired_driver = _run_ramp(wired, (0.5, 1.0))
  assert landed_result.status is DriverStatus.COMPLETED
  assert wired_result.status is DriverStatus.COMPLETED
  np.testing.assert_array_equal(
    wired_driver.owner.accepted_physical().values,
    landed_driver.owner.accepted_physical().values,
  )
  # The stateful group committed its damage envelope row through the driver,
  # identically on both paths.
  damage_block_id = damage.header.state_layout.block_id
  wired_rows = wired_driver.owner.accepted_state(damage_block_id).values
  landed_rows = landed_driver.owner.accepted_state(damage_block_id).values
  np.testing.assert_array_equal(wired_rows, landed_rows)
  assert float(wired_rows[0, 0]) > 0.0


def test_spring_groups_channel_validates_its_container() -> None:
  declaration = _apex_spring_declaration()
  invalid_channels = (
    [declaration],
    (declaration, "not-a-declaration"),
    declaration,
    None,
  )
  for invalid in invalid_channels:
    with pytest.raises(ModelCompilationError) as excinfo:
      compile_system(
        _shallow_truss_model(),
        truss_reference_registry(),
        springs=invalid,
      )
    (diagnostic,) = excinfo.value.diagnostics
    assert diagnostic.code == "invalid-spring-declarations"
    assert diagnostic.source.source == "model"
  # The default channel is byte-identical to the region-routed path alone.
  assert _wired_system().content_fingerprint == _base_system().content_fingerprint


def _point_spring_region_model(*, mixed: bool) -> ModelSpec:
  base = _shallow_truss_model()
  (region,) = base.regions
  spring_region = RegionSpec(
    id="springs" if mixed else region.id,
    cell_refs=region.cell_refs,
    field_ids=region.field_ids,
    material_id=region.material_id,
    formulation="point-spring",
    quadrature=region.quadrature,
    source=_source("springs-region"),
  )
  regions = (region, spring_region) if mixed else (spring_region,)
  return ModelSpec(
    mesh=base.mesh,
    fields=base.fields,
    materials=base.materials,
    regions=regions,
    source=base.source,
  )


def test_point_spring_region_formulation_fails_coded() -> None:
  with pytest.raises(ModelCompilationError) as excinfo:
    compile_system(
      _point_spring_region_model(mixed=False),
      truss_reference_registry(),
    )
  (diagnostic,) = excinfo.value.diagnostics
  assert diagnostic.code == "unsupported-spring-region"
  assert "springs" in diagnostic.message
  assert diagnostic.source.source == "springs-region"


def test_mixed_family_region_spec_keeps_landed_coded_diagnostics() -> None:
  # A spec carrying both a truss and a point-spring region still routes to
  # the truss family, whose landed membership validation describes the
  # double-referenced cells — the wiring changes no mixed-family diagnostic.
  with pytest.raises(ModelCompilationError) as excinfo:
    compile_system(
      _point_spring_region_model(mixed=True),
      truss_reference_registry(),
    )
  (diagnostic,) = excinfo.value.diagnostics
  assert diagnostic.code == "multiple-cell-membership"


def test_landed_coded_diagnostics_are_preserved_on_the_wired_path() -> None:
  unknown_space = damage_envelope_declaration(
    block_id="damage-springs",
    space_id="thermal",
    spring_ids=("spring-1",),
    node_ids=(0,),
    stiffness=2.0,
    critical_extension=2.0,
    max_increment=5.0,
    source=_source("damage-spring-source"),
  )
  unknown_node = damage_envelope_declaration(
    block_id="damage-springs",
    space_id="displacement",
    spring_ids=("spring-1",),
    node_ids=(99,),
    stiffness=2.0,
    critical_extension=2.0,
    max_increment=5.0,
    source=_source("damage-spring-source"),
  )
  planted = SpringDeclaration(
    block_id="planted-springs",
    space_id="displacement",
    spring_ids=("planted-1",),
    node_ids=(0,),
    state_schema="planted-nonzero-tangent-v1",
    state_slots=(SpringStateSlot("marker", 1),),
    kernel_name="planted-nonzero-tangent",
    kernel_version="1",
    implementation_id="planted-nonzero-tangent-v1",
    parameters=(3.0,),
    kernel=planted_nonzero_tangent_kernel,
    source=_source("planted-kernel-source"),
  )
  cases = (
    (unknown_space, "unknown-spring-space"),
    (unknown_node, "unknown-spring-support-node"),
    (planted, "inconsistent-kernel-tangent"),
  )
  for declaration, code in cases:
    with pytest.raises(ModelCompilationError) as landed_exc:
      compile_spring_operator(_base_system(), declaration)
    with pytest.raises(ModelCompilationError) as wired_exc:
      _wired_system(declaration)
    (diagnostic,) = wired_exc.value.diagnostics
    assert diagnostic.code == code
    assert str(wired_exc.value) == str(landed_exc.value)


def test_multi_group_membership_collisions_are_preserved() -> None:
  first = _apex_spring_declaration()
  overlapping = SpringDeclaration(
    block_id="apex-spring",
    space_id="displacement",
    spring_ids=("other-spring",),
    node_ids=(2,),
    state_schema="other-state-v1",
    state_slots=(),
    kernel_name="axial-linear-spring",
    kernel_version="1",
    implementation_id="other-spring-v1",
    parameters=(50.0, 1.0, 0.0),
    kernel=axial_spring_kernel,
    source=_source("other-spring-source"),
  )
  # Two groups sharing one block id collide exactly as the landed compose
  # seam reports it, on both the standalone and the wired path.
  with pytest.raises(ValueError, match="collides"):
    _landed_system(first, overlapping)
  with pytest.raises(ValueError, match="collides"):
    _wired_system(first, overlapping)
