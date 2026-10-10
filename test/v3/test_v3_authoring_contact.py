# SPDX-License-Identifier: MIT

"""Penalty contact authoring: a disc obstacle over a declared surface set.

Covers the ``penalty_contact(...)`` helper: declaration shape and compiled
operator identity (the packed obstacle parameters and the obstacle-schedule
signal port), authoring-time validation, and the stepped persona — a student
authors the documented M66 contact_test02 skim configuration (the quad4
strip read from the skim's ``.dat``, the c1 disc obstacle, the pinned load
table), composes the contact law through the helper, and steps a
``nonlinear_static`` session. Committed coefficients equal the landed
declaration/compile/compose path BITWISE per substep, the committed active
set matches the penalty kernel's engagement row by row, and the law ships no
derivative kernel, so its qualified parameters read 'constant' in the M58
sensitivity diagnostics — pinned literally.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from pyfem.v3 import authoring
from pyfem.v3.compile.contact import (
  ContactSignalInput,
  ContactSignalPort,
  compile_contact_operator,
  compose_contact_system,
  penalty_disc_declaration,
  penalty_disc_kernel,
)
from pyfem.v3.compile.contracts import continuum_reference_registry
from pyfem.v3.compile.diagnostics import ModelCompilationError
from pyfem.v3.compile.system import compile_system
from pyfem.v3.constraints import compile_constraint_map
from pyfem.v3.driver import DriverStatus, NonlinearStaticDriver, NonlinearStaticSettings
from pyfem.v3.driver.diagnostics import DriverPreparationError
from pyfem.v3.io.dat import read_dat_mesh
from pyfem.v3.io.solver_pro import parse_nonlinear_solver_settings
from pyfem.v3.model.operator import EvaluationStatus
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
  AffineValueSpec,
  DofRef,
  PrescribedDofSpec,
  ProgramConstraintSpec,
  ProgramCoordinateSpec,
  ProgramCoordinateValue,
  ProgramPoint,
)

ROOT = Path(__file__).resolve().parents[2]
SKIM_DIR = ROOT / "skims" / "contact_test02"
_COMPONENT = {"u": "x", "v": "y"}

# The documented M66 skim obstacle (the c1 block of skims/contact_test02).
_CENTRE = (8.0, 1.4)
_DIRECTION = (0.0, -0.1)
_RADIUS = 1.0
_PENALTY = 1.0e6
_PARAMETERS = np.array((_PENALTY, _RADIUS, *_CENTRE, *_DIRECTION), dtype=np.float64)


def _source(label: str) -> SourceContext:
  return SourceContext(source=label)


def _truss_base() -> CompiledSystem:
  mesh = authoring.line2_mesh(
    {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (0.0, 1.0), 3: (1.0, 1.0)},
    {"bar-left": (0, 2), "bar-right": (1, 3)},
  )
  model = authoring.truss(mesh, material=authoring.uniaxial_elastic(1.0e6, 1.0))
  return authoring.compile(model)


# --- helper surface: declaration shape and validation ----------------------------


def test_penalty_contact_composes_the_landed_operator() -> None:
  system = authoring.penalty_contact(
    _truss_base(),
    nodes=(2, 3),
    centre=_CENTRE,
    direction=_DIRECTION,
    radius=_RADIUS,
    penalty=_PENALTY,
  )
  assert len(system.operators) == 2
  operator = system.operators[1]
  (implementation,) = operator.header.implementations
  assert implementation.kind == "constitutive-kernel"
  assert implementation.name == "penalty-disc-contact"
  assert implementation.implementation_id == "pyfem-v3-contact-penalty-disc-v1"
  np.testing.assert_array_equal(operator.payload.parameters.values, _PARAMETERS)
  layout = operator.header.state_layout
  assert layout.schema == "pyfem-v3-contact-penalty-disc-v1"
  assert layout.row_shape == (2, 0)  # penalty contact is stateless
  (port,) = operator.header.signal_ports
  assert port.port_id == "load-factor"
  assert port.signal_id == "load"
  # The honest channel flags of the landed exact symmetric tangent.
  (jacobian,) = operator.header.jacobian_channels
  assert jacobian.linear is False
  assert jacobian.symmetric is True


def test_penalty_contact_nodes_accept_sequence_and_mapping_forms() -> None:
  sequenced = authoring.penalty_contact(
    _truss_base(),
    nodes=(2, 3),
    centre=_CENTRE,
    direction=_DIRECTION,
    radius=_RADIUS,
    penalty=_PENALTY,
  )
  assert tuple(sequenced.operators[1].contact_block.entity_ids) == (
    "contact-1",
    "contact-2",
  )
  mapped = authoring.penalty_contact(
    _truss_base(),
    nodes={"tip": 3},
    centre=_CENTRE,
    direction=_DIRECTION,
    radius=_RADIUS,
    penalty=_PENALTY,
    coordinate="lam",
    block_id="c9",
  )
  assert tuple(mapped.operators[1].contact_block.entity_ids) == ("tip",)
  assert mapped.operators[1].contact_block.block_id == "c9"
  (port,) = mapped.operators[1].header.signal_ports
  assert port.signal_id == "lam"


def test_penalty_contact_rejects_bad_inputs_early() -> None:
  base = _truss_base()
  with pytest.raises(ValueError, match="at least one surface node"):
    authoring.penalty_contact(
      base,
      nodes=(),
      centre=_CENTRE,
      direction=_DIRECTION,
      radius=_RADIUS,
      penalty=_PENALTY,
    )
  with pytest.raises(TypeError, match=r"centre must be an \(x, y\) pair"):
    authoring.penalty_contact(
      base,
      nodes=(2,),
      centre=(1.0,),  # type: ignore[arg-type]
      direction=_DIRECTION,
      radius=_RADIUS,
      penalty=_PENALTY,
    )
  with pytest.raises(ValueError, match="components must be finite"):
    authoring.penalty_contact(
      base,
      nodes=(2,),
      centre=_CENTRE,
      direction=(0.0, float("nan")),
      radius=_RADIUS,
      penalty=_PENALTY,
    )
  with pytest.raises(ValueError, match="positive finite"):
    authoring.penalty_contact(
      base,
      nodes=(2,),
      centre=_CENTRE,
      direction=_DIRECTION,
      radius=-_RADIUS,
      penalty=_PENALTY,
    )
  with pytest.raises(ValueError, match="positive finite"):
    authoring.penalty_contact(
      base,
      nodes=(2,),
      centre=_CENTRE,
      direction=_DIRECTION,
      radius=_RADIUS,
      penalty=0.0,
    )
  with pytest.raises(TypeError, match="coordinate must be a non-empty exact string"):
    authoring.penalty_contact(
      base,
      nodes=(2,),
      centre=_CENTRE,
      direction=_DIRECTION,
      radius=_RADIUS,
      penalty=_PENALTY,
      coordinate=3.0,  # type: ignore[arg-type]
    )
  with pytest.raises(TypeError, match="exact CompiledSystem"):
    authoring.penalty_contact(
      base.operators[0],  # type: ignore[arg-type]
      nodes=(2,),
      centre=_CENTRE,
      direction=_DIRECTION,
      radius=_RADIUS,
      penalty=_PENALTY,
    )
  with pytest.raises(ModelCompilationError, match="unknown-contact-support-node"):
    authoring.penalty_contact(
      base,
      nodes=(99,),
      centre=_CENTRE,
      direction=_DIRECTION,
      radius=_RADIUS,
      penalty=_PENALTY,
    )


def test_penalty_contact_parameters_read_constant_in_the_sensitivity_surface() -> None:
  """The declaration-routed law ships no derivative kernel either.

  The qualified contact parameter names resolve through the M58 convention
  resolver (the landed kernel's fixed scalar parameter pair), so a
  sensitivity request fails pre-substep with the parameterized-vs-constant
  diff — the trailer lists the truss bank's constants alongside the contact
  law's.
  """
  system = authoring.penalty_contact(
    _truss_base(),
    nodes=(2, 3),
    centre=_CENTRE,
    direction=_DIRECTION,
    radius=_RADIUS,
    penalty=_PENALTY,
  )
  session = authoring.nonlinear_static(
    system,
    constraints=authoring.fixed(nodes=(0, 1)),
  )
  with pytest.raises(DriverPreparationError) as captured:
    session.run({"load": 0.0}, {"load": 0.2}, sensitivities=("penalty",))
  (diagnostic,) = captured.value.diagnostics
  assert diagnostic.code == "unknown-sensitivity-parameter"
  assert (
    "field 'sensitivities.penalty': expected {'parameter': 'penalty'}, "
    "authored 'constant'" in diagnostic.message
  )
  assert (
    "declared differentiable parameters: (); declared constant parameters: "
    "('youngs_modulus', 'area', 'penalty', 'radius')" in diagnostic.message
  )
  assert diagnostic.source == _source("authoring.run:sensitivities")
  assert session.driver.statistics.evaluation_count == 0
  assert session.ordinal == 0


# --- stepped persona: the documented contact_test02 skim, authored -----------------


def _skim_model() -> tuple[ModelSpec, tuple[ProgramConstraintSpec, ...], np.ndarray]:
  """The skim's quad4 strip as landed spec values plus its dat constraints.

  Hand-written landed declarations pass through every authoring seam: the
  mesh and material slices below are exactly the M66 parity harness's, read
  from the skim's ``.dat``.
  """
  mesh, prescribed, ties, loads = read_dat_mesh(SKIM_DIR / "contact_test02.dat")
  assert not ties
  assert not loads
  nodes = tuple(
    NodeSpec(
      id=int(node_id),
      coordinates=tuple(float(value) for value in mesh.coords[index]),
      source=_source(f"dat:node:{int(node_id)}"),
    )
    for index, node_id in enumerate(mesh.node_ids)
  )
  cells = tuple(
    CellSpec(
      id=f"cell-{index}",
      node_ids=tuple(int(mesh.node_ids[position]) for position in mesh.conn[index]),
      source=_source(f"dat:cell:{index}"),
    )
    for index in range(mesh.conn.shape[0])
  )
  block = CellBlockSpec(
    id="cells",
    reference_topology="quadrilateral",
    topological_dimension=2,
    embedding_dimension=2,
    geometry_interpolation="bilinear-quad4",
    cells=cells,
    source=_source("dat:cells"),
  )
  field = FieldSpec(
    id="displacement",
    components=("x", "y"),
    location="node",
    source=_source("dat:field"),
  )
  material = MaterialSpec(
    id="elastic",
    model="plane-stress-linear-elastic",
    parameters=(
      MaterialParameterSpec("youngs_modulus", 1.0e6, _source("dat:E")),
      MaterialParameterSpec("poisson_ratio", 0.25, _source("dat:nu")),
    ),
    source=_source("dat:material"),
  )
  region = RegionSpec(
    id="domain",
    cell_refs=tuple(CellRef(block.id, cell.id) for cell in cells),
    field_ids=(field.id,),
    material_id=material.id,
    formulation="small-strain-continuum",
    quadrature="gauss-2x2",
    source=_source("dat:region"),
  )
  model = ModelSpec(
    mesh=MeshSpec(nodes=nodes, cell_blocks=(block,), source=_source("dat:mesh")),
    fields=(field,),
    materials=(material,),
    regions=(region,),
    source=_source("dat:model"),
  )
  seen: set[tuple[int, str]] = set()
  constraints: list[ProgramConstraintSpec] = []
  for item in prescribed:
    key = (item.node_id, item.dof_type)
    if key in seen:
      continue  # the legacy deck lists u[81] twice
    seen.add(key)
    assert item.value == 0.0
    constraints.append(
      PrescribedDofSpec(
        id=f"p:{item.node_id}:{item.dof_type}",
        target=DofRef(
          node_id=item.node_id,
          field_id="displacement",
          component=_COMPONENT[item.dof_type],
        ),
        value=AffineValueSpec(constant=0.0),
        source=_source(f"dat:p:{item.node_id}:{item.dof_type}"),
      )
    )
  return model, tuple(constraints), np.array(mesh.coords, dtype=np.float64)


def _skim_node_ids() -> tuple[int, ...]:
  mesh, _, _, _ = read_dat_mesh(SKIM_DIR / "contact_test02.dat")
  return tuple(int(node_id) for node_id in mesh.node_ids)


def _author_system(
  model: ModelSpec,
) -> CompiledSystem:
  base = authoring.compile(model, continuum_reference_registry())
  # Legacy Contact loops over all nodes (no search): the declared surface set
  # is the whole mesh, in .dat order.
  return authoring.penalty_contact(
    base,
    nodes=_skim_node_ids(),
    centre=_CENTRE,
    direction=_DIRECTION,
    radius=_RADIUS,
    penalty=_PENALTY,
    block_id="c1",
  )


def _twin_system(model: ModelSpec) -> CompiledSystem:
  """The landed M66 parity path directly: declaration plus compile/compose."""
  base = compile_system(model, continuum_reference_registry())
  node_ids = _skim_node_ids()
  declaration = penalty_disc_declaration(
    block_id="c1",
    space_id="displacement",
    contact_ids=tuple(f"contact-{node_id}" for node_id in node_ids),
    node_ids=node_ids,
    centre=_CENTRE,
    direction=_DIRECTION,
    radius=_RADIUS,
    penalty=_PENALTY,
    signal_port=ContactSignalPort(port_id="load-factor", signal_id="load"),
    source=_source("skim:c1"),
  )
  block, operator = compile_contact_operator(base, declaration)
  return compose_contact_system(base, block, operator)


def _point(load: float) -> ProgramPoint:
  return ProgramPoint((ProgramCoordinateValue("load", load),))


def _engaged(positions: np.ndarray, lam: float) -> set[int]:
  centre = np.array(_CENTRE) + lam * np.array(_DIRECTION)
  distance = np.sqrt(np.sum((positions - centre) ** 2, axis=1))
  return {index for index in range(len(positions)) if _RADIUS - distance[index] > 0.0}


def test_persona_steps_the_contact_skim_bitwise_like_the_landed_path() -> None:
  """Persona: penalty_contact() on the M66 skim, stepped and oracle-checked.

  The student reads the skim's mesh, compiles it, composes the documented c1
  obstacle through the helper, and steps the pinned load table. Per
  committed substep the coefficients equal the landed M66 parity path
  BITWISE, and the penalty kernel evaluated at the committed positions (the
  obstacle driven by the bound ``load`` coordinate) engages exactly the
  test-side active set — the converged path carries the legacy force law.
  """
  model, constraints, node_xy = _skim_model()
  pro_text = (SKIM_DIR / "skim.pro").read_text(encoding="utf-8")
  settings = parse_nonlinear_solver_settings(pro_text)
  assert settings is not None and settings.load_table is not None
  table = tuple(float(value) for value in settings.load_table[1:])
  system = _author_system(model)
  session = authoring.nonlinear_static(
    system,
    constraints=constraints,
    settings=NonlinearStaticSettings(
      tolerance=settings.tol,
      max_iterations=settings.iter_max,
    ),
  )
  twin_system = _twin_system(model)
  twin_map = compile_constraint_map(
    twin_system,
    constraints=constraints,
    coordinates=(ProgramCoordinateSpec(name="load", kind="load"),),
  )
  twin = NonlinearStaticDriver(
    twin_system,
    twin_map,
    (),
    NonlinearStaticSettings(tolerance=settings.tol, max_iterations=settings.iter_max),
  )
  base = {"load": 0.0}
  twin_base = _point(0.0)
  for lam in table:
    result = session.run(base, {"load": lam})
    assert result.status is DriverStatus.COMPLETED
    base = {"load": lam}
    twin_result = twin.run(base_point=twin_base, target_points=(_point(lam),))
    assert twin_result.status is DriverStatus.COMPLETED
    twin_base = _point(lam)
    committed = session.accepted_coefficients()
    np.testing.assert_array_equal(committed, twin.owner.accepted_physical().values)
    # The kernel oracle at the committed positions: the exact legacy force
    # law, engaging exactly the test-side active set under the bound lam.
    positions = node_xy + committed.reshape(-1, 2)
    signal = ContactSignalInput(
      port_id="load-factor",
      values=np.array([lam], dtype=np.float64),
      derivatives=(),
    )
    oracle = penalty_disc_kernel(
      positions,
      np.zeros((len(node_xy), 0), dtype=np.float64),
      _PARAMETERS,
      (signal,),
    )
    assert oracle.status is EvaluationStatus.OK
    kernel_engaged = {
      index for index in range(len(node_xy)) if bool((oracle.force[index] != 0.0).any())
    }
    assert kernel_engaged == _engaged(positions, lam)
  assert _engaged(positions, 1.0)
  # The documented M66 final-state measurement (parity harness).
  assert float(committed.min()) == pytest.approx(-1.032008e-01, rel=1e-4)


def test_obstacle_schedule_changes_the_response_through_the_authoring_path() -> None:
  """The bound coordinate is the only obstacle channel: slower approach,
  later engagement — two schedules over the same mesh commit different
  states, the positive control for the signal-port wiring.
  """
  model, constraints, node_xy = _skim_model()
  pro_text = (SKIM_DIR / "skim.pro").read_text(encoding="utf-8")
  settings = parse_nonlinear_solver_settings(pro_text)
  assert settings is not None and settings.load_table is not None
  table = tuple(float(value) for value in settings.load_table[1:])
  skim_settings = NonlinearStaticSettings(
    tolerance=settings.tol,
    max_iterations=settings.iter_max,
  )
  first = authoring.nonlinear_static(
    _author_system(model), constraints=constraints, settings=skim_settings
  )
  second = authoring.nonlinear_static(
    _author_system(model), constraints=constraints, settings=skim_settings
  )
  base = {"load": 0.0}
  for lam in table:
    assert first.run(base, {"load": lam}).status is DriverStatus.COMPLETED
    base = {"load": lam}
  assert second.run({"load": 0.0}, {"load": table[0]}).status is DriverStatus.COMPLETED
  first_committed = first.accepted_coefficients()
  second_committed = second.accepted_coefficients()
  assert first_committed.tobytes() != second_committed.tobytes()
  # The documented engagement of this skim: the near-field disc already
  # touches the strip corner (node index 50) at the first table entry and
  # holds it through the ramp — the active set the M66 parity harness
  # implies (no substep solves linearly).
  assert _engaged(node_xy + first_committed.reshape(-1, 2), table[-1]) == {50}
  assert _engaged(node_xy + second_committed.reshape(-1, 2), table[0]) == {50}
