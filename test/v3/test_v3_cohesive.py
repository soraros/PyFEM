# SPDX-License-Identifier: MIT

"""Cohesive traction-separation laws and interface skims: the parity evidence.

Law-level bitwise pins compare the v3 kernels against the legacy ``getStress``
implementations over a jump battery (normal/shear/mixed/opening/closing,
including sign corners and the virgin state). Element-level bitwise pins
compare per-element residuals against legacy ``Interface.getInternalForce``
on axis-aligned mirror-symmetric states (where the legacy reflecting frame
and the v3 corrected frame provably coincide, the tower's parity scope).
Oblique states diverge by construction — the 45-degree reflection measurement
is documented in test_v3_interface.py's frame tests and cited here.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from _legacy_parity import (
  legacy_state,
  load_parity_tolerances,
)

from pyfem.elements.Interface import Interface
from pyfem.materials.Dummy import Dummy
from pyfem.materials.PowerLawModeI import PowerLawModeI
from pyfem.materials.ThoulessModeI import ThoulessModeI
from pyfem.materials.XuNeedleman import XuNeedleman
from pyfem.util.dataStructures import Properties, elementData
from pyfem.util.kinematics import Kinematics
from pyfem.v3.compile.interface import (
  InterfaceKernel,
  compile_interface_operator,
  dummy_interface_declaration,
  power_law_mode_i_declaration,
  thouless_mode_i_declaration,
  xu_needleman_declaration,
)
from pyfem.v3.driver import DriverStatus
from pyfem.v3.io.legacy_deck import read_legacy_deck, run_deck
from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.operator import ChannelRequest, OperatorEvaluationInput
from pyfem.v3.spec.diagnostics import SourceContext

_SOURCE = SourceContext(source="test_v3_cohesive")

ROOT = Path(__file__).resolve().parents[2]

# Cross-platform tolerance note for every bitwise pin below: the reference-
# platform branch asserts raw-uint64 identity (signed zeros included) — only
# the off-reference branch uses these bands. rtol=1e-12 carries >100x
# headroom over the documented drift class for transcendentals (32 ulps,
# ~7.1e-15 relative, the M30 plasticity kernel on ubuntu-latest CI runs
# 37440534665 and 37441926273, cited in test/v3/conftest.py).
#
# The atol floors guard the batteries' exact-zero entries, where rtol
# contributes nothing. CI run 38023817372 (ubuntu x86_64) measured max abs
# 2.22e-16 against an expected exact-zero XuNeedleman traction (2/136
# entries) — ~1 ulp of the O(1) exp intermediates at the ~1e-1 drift site,
# i.e. ~2e-15 relative to battery scale. The floors sit at the 4-ulp class
# of the transcendental-carrying laws' measured maxima (XuNeedleman /
# PowerLawModeI, this host): traction max 17.8 (ulp 3.6e-15; 2.0e-14 =
# 5.6 ulps), tangent max 367.1 (ulp 5.7e-14; 5.0e-13 = 8.8 ulps) —
# respectively ~90x and ~2300x above the measured drift, and ~3 orders
# below the 1e-9-class order bug the pins must catch. Dummy needs no such
# floor: its D*jump law has no libm content, so its exact zeros are
# structural and its nonzero entries are single correctly-rounded products
# (rtol=1e-12 covers a hypothetical FMA contraction at 1e5 by ~7000x).
_LAW_RTOL = 1.0e-12
_TRACTION_ATOL = 2.0e-14
_TANGENT_ATOL = 5.0e-13


def _legacy_material(law: str) -> object:
  """Instantiate the legacy material exactly as an Interface element would."""
  table = {
    "XuNeedleman": (XuNeedleman, {"rank": 2, "Gc": 0.1, "Tult": 0.5}),
    "PowerLawModeI": (PowerLawModeI, {"rank": 2, "Gc": 0.1, "Tult": 0.5}),
    "ThoulessModeI": (
      ThoulessModeI,
      {"rank": 2, "Gc": 0.1, "Tult": 0.5, "d1d3": 0.2, "d2d3": 0.6},
    ),
    "Dummy": (Dummy, {"rank": 2, "D": 1.0e5}),
  }
  cls, props_dict = table[law]
  return cls(Properties({**props_dict, "solverStat": None}))


def _legacy_law_response(
  material: object,
  jumps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  stress = np.zeros((len(jumps), 2))
  tangent = np.zeros((len(jumps), 2, 2))
  for index, jump in enumerate(jumps):
    kinematics = Kinematics(2, 2)
    kinematics.strain = np.array(jump, dtype=np.float64)
    sigma, tang = material.getStress(kinematics)
    stress[index] = sigma
    tangent[index] = tang
  return stress, tangent


def _law_kernel(law: str) -> tuple[InterfaceKernel, tuple[float, ...]]:
  declarations = {
    "XuNeedleman": (
      xu_needleman_declaration,
      {"fracture_energy": 0.1, "ultimate_traction": 0.5},
    ),
    "PowerLawModeI": (
      power_law_mode_i_declaration,
      {"fracture_energy": 0.1, "ultimate_traction": 0.5},
    ),
    "ThoulessModeI": (
      thouless_mode_i_declaration,
      {"fracture_energy": 0.1, "ultimate_traction": 0.5, "d1d3": 0.2, "d2d3": 0.6},
    ),
    "Dummy": (dummy_interface_declaration, {"stiffness": 1.0e5}),
  }
  build, parameters = declarations[law]
  declaration = build(
    block_id="coh",
    space_id="displacement",
    interface_ids=(1,),
    node_quads=((0, 1, 2, 3),),
    source=_SOURCE,
    **parameters,
  )
  return declaration.kernel, declaration.parameters


def _jump_battery() -> np.ndarray:
  """Normal/shear/mixed/opening/closing jumps, sign corners, virgin state.

  Magnitudes stay below the legacy ``math.exp`` overflow on the closing side
  (vnmax = 0.0736 for the battery parameters).
  """
  vnmax = 0.1 / (2.71828183 * 0.5)
  vtmax = 0.1 / (1.16580058 * 0.5)
  rows = [
    (0.0, 0.0),
    (-0.0, -0.0),
    (0.0, 0.01),
    (0.0, -0.01),
    (1.0e-4, 0.0),
    (-1.0e-4, 0.0),
    (0.5 * vnmax, 0.0),
    (vnmax, 0.0),
    (1.5 * vnmax, 0.0),
    (4.0 * vnmax, 0.0),
    (-0.5 * vnmax, 0.0),
    (-vnmax, 0.0),
    (0.0, 0.25 * vtmax),
    (0.0, -0.25 * vtmax),
    (0.5 * vnmax, 0.5 * vtmax),
    (0.5 * vnmax, -0.5 * vtmax),
    (-0.5 * vnmax, 0.5 * vtmax),
    (-0.5 * vnmax, -0.5 * vtmax),
    (vnmax, vtmax),
    (-vnmax, -vtmax),
  ]
  rng = np.random.default_rng(20261009)
  randoms = rng.standard_normal((24, 2)) * np.array([vnmax, 0.5 * vtmax])
  return np.vstack((np.array(rows), randoms, -randoms))


@pytest.mark.parametrize(
  "law", ("XuNeedleman", "PowerLawModeI", "ThoulessModeI", "Dummy")
)
def test_law_kernel_matches_legacy_bitwise(law: str, bitwise_pin: object) -> None:
  battery = _jump_battery()
  material = _legacy_material(law)
  legacy_stress, legacy_tangent = _legacy_law_response(material, battery)
  kernel, parameters = _law_kernel(law)
  result = kernel(
    battery,
    np.zeros((len(battery), 0)),
    np.array(parameters, dtype=np.float64),
  )
  bitwise_pin(result.traction, legacy_stress, rtol=_LAW_RTOL, atol=_TRACTION_ATOL)
  bitwise_pin(result.tangent, legacy_tangent, rtol=_LAW_RTOL, atol=_TANGENT_ATOL)


@pytest.mark.parametrize(
  "law", ("XuNeedleman", "PowerLawModeI", "ThoulessModeI", "Dummy")
)
def test_law_tangent_fd_verified_at_nonzero_jumps(law: str) -> None:
  """Each law tangent convicts itself against a central FD of its traction.

  The battery points sit at law scale (fractions and multiples of the
  characteristic jumps), where central differences of these smooth laws are
  accurate to ~1e-9 relative; the pin at 1e-6 carries headroom. The
  conviction leg perturbs the tangent by 1e-3 and requires the same check to
  fail.
  """
  kernel, parameters = _law_kernel(law)
  params = np.array(parameters, dtype=np.float64)
  vnmax = 0.1 / (2.71828183 * 0.5)
  vtmax = 0.1 / (1.16580058 * 0.5)
  points = np.array(
    [
      [0.5 * vnmax, 0.3 * vtmax],
      [vnmax, -0.1 * vtmax],
      [2.0 * vnmax, 0.05 * vtmax],
      [-0.5 * vnmax, -0.4 * vtmax],
      [3.0 * vnmax, 0.6 * vtmax],
    ]
  )

  def check(jump: np.ndarray, tangent: np.ndarray) -> float:
    step = 1.0e-7 * np.maximum(1.0, np.abs(jump))
    worst = 0.0
    for component in range(2):
      plus = jump.copy()
      minus = jump.copy()
      plus[component] += step[component]
      minus[component] -= step[component]
      forward = kernel(plus[None, :], np.zeros((1, 0)), params)
      backward = kernel(minus[None, :], np.zeros((1, 0)), params)
      difference = (forward.traction[0] - backward.traction[0]) / (
        2.0 * step[component]
      )
      column = tangent[:, component]
      scale = max(1.0, float(np.abs(column).max()))
      worst = max(worst, float(np.abs(column - difference).max()) / scale)
    return worst

  for jump in points:
    result = kernel(jump[None, :], np.zeros((1, 0)), params)
    tangent = result.tangent[0]
    assert check(jump, tangent) < 1.0e-6
    # Conviction leg: where the tangent is materially nonzero, a 1e-3-relative
    # tamper is detected. (At plateau/peak points the tangent is exactly zero
    # and the FD agreement above is the content.)
    if float(np.abs(tangent).max()) > 1.0e-3:
      assert check(jump, tangent * 1.001) > 1.0e-6


# --- element-level bitwise battery on axis-aligned states --------------------


class _StubNodes:
  """The four-node index map the legacy element's nodal output expects."""

  def __len__(self) -> int:
    return 4

  def getIndices(self, element: object) -> list[int]:
    return [0, 1, 2, 3]


def _legacy_interface_element(law: str) -> Interface:
  material_props = {
    "XuNeedleman": {"type": "XuNeedleman", "Tult": 0.5, "Gc": 0.1},
    "PowerLawModeI": {"type": "PowerLawModeI", "Tult": 0.5, "Gc": 0.1},
    "ThoulessModeI": {
      "type": "ThoulessModeI",
      "Tult": 0.5,
      "Gc": 0.1,
      "d1d3": 0.2,
      "d2d3": 0.6,
    },
    "Dummy": {"type": "Dummy", "D": 1.0e5},
  }[law]
  props = Properties(
    {
      "rank": 2,
      "material": Properties(material_props),
      "solverStat": None,
    }
  )
  element = Interface([0, 1, 2, 3], props)
  element.globdat = Properties({"nodes": _StubNodes(), "outputNames": []})
  element.iElm = 0
  return element


_AXIS_ALIGNED_COORDS = np.array(
  [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [1.0, 0.0]],
  dtype=np.float64,
)


def _legacy_element_response(
  law: str,
  state: np.ndarray,
  accepted_normal: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  element = _legacy_interface_element(law)
  if accepted_normal is not None:
    element.history["normal"] = np.array(accepted_normal, dtype=np.float64)
  data = elementData(np.array(state, dtype=np.float64), np.zeros(8))
  data.coords = _AXIS_ALIGNED_COORDS
  data.nodes = [0, 1, 2, 3]
  element.getTangentStiffness(data)
  return (
    np.array(data.fint, dtype=np.float64),
    np.array(data.stiff, dtype=np.float64),
    np.array(element.current["normal"], dtype=np.float64),
  )


def _mirror_symmetric_states() -> list[np.ndarray]:
  """States with u_top = u_bottom and v_top = -v_bottom per coincident pair:
  the frame segment stays exactly horizontal (ds_y == +0.0), so the legacy
  reflecting frame and the corrected frame coincide."""
  rng = np.random.default_rng(20261010)
  states = [np.zeros(8)]
  for scale in (1.0e-4, 1.0e-2, 5.0e-2):
    for _ in range(4):
      u1, v1, u2, v2 = rng.standard_normal(4) * scale
      states.append(np.array([u1, v1, u2, v2, u1, -v1, u2, -v2]))
  return states


@pytest.mark.parametrize(
  "law", ("XuNeedleman", "PowerLawModeI", "ThoulessModeI", "Dummy")
)
def test_element_residual_matches_legacy_bitwise_axis_aligned(
  law: str,
  bitwise_pin: object,
) -> None:
  kernel, parameters = _law_kernel(law)
  declaration_builders = {
    "XuNeedleman": xu_needleman_declaration,
    "PowerLawModeI": power_law_mode_i_declaration,
    "ThoulessModeI": thouless_mode_i_declaration,
    "Dummy": dummy_interface_declaration,
  }
  from pyfem.v3.compile.system import compile_system
  from pyfem.v3.compile.truss import truss_reference_registry
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
  )

  nodes = tuple(
    NodeSpec(
      id=index,
      coordinates=coordinates,
      source=_SOURCE,
    )
    for index, coordinates in enumerate(
      (
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (1.0, 1.0),
      )
    )
  )
  model = ModelSpec(
    mesh=MeshSpec(
      nodes=nodes,
      cell_blocks=(
        CellBlockSpec(
          id="bars",
          reference_topology="line",
          topological_dimension=1,
          embedding_dimension=2,
          geometry_interpolation="line2",
          cells=(CellSpec(id="c0", node_ids=(4, 5), source=_SOURCE),),
          source=_SOURCE,
        ),
      ),
      source=_SOURCE,
    ),
    fields=(
      FieldSpec(
        id="displacement",
        components=("x", "y"),
        location="node",
        source=_SOURCE,
      ),
    ),
    materials=(
      MaterialSpec(
        id="steel",
        model="uniaxial-linear-elastic",
        parameters=(
          MaterialParameterSpec("youngs_modulus", 1.0e6),
          MaterialParameterSpec("area", 1.0),
        ),
        source=_SOURCE,
      ),
    ),
    regions=(
      RegionSpec(
        id="domain",
        cell_refs=(CellRef("bars", "c0"),),
        field_ids=("displacement",),
        material_id="steel",
        formulation="total-lagrangian-truss",
        quadrature="none",
        source=_SOURCE,
      ),
    ),
    source=_SOURCE,
  )
  system = compile_system(model, truss_reference_registry())
  build = declaration_builders[law]
  kwargs = {
    "XuNeedleman": {"fracture_energy": 0.1, "ultimate_traction": 0.5},
    "PowerLawModeI": {"fracture_energy": 0.1, "ultimate_traction": 0.5},
    "ThoulessModeI": {
      "fracture_energy": 0.1,
      "ultimate_traction": 0.5,
      "d1d3": 0.2,
      "d2d3": 0.6,
    },
    "Dummy": {"stiffness": 1.0e5},
  }[law]
  _, operator = compile_interface_operator(
    system,
    build(
      block_id="coh",
      space_id="displacement",
      interface_ids=(1,),
      node_quads=((0, 1, 2, 3),),
      source=_SOURCE,
      **kwargs,
    ),
  )
  branches = (
    np.zeros(2),
    np.array([0.0, 1.0]),
    np.array([0.0, -1.0]),
  )
  for state in _mirror_symmetric_states():
    for accepted in branches:
      legacy_fint, _, legacy_normal = _legacy_element_response(
        law,
        state,
        None if accepted.sum() == 0.0 and accepted[1] == 0.0 else accepted,
      )
      evaluation = operator.evaluate(
        OperatorEvaluationInput(
          port_values=(FinalizedArray(state.reshape(1, 8).copy(), dtype=np.float64),),
          accepted_state=FinalizedArray(
            accepted.reshape(1, 2).copy(), dtype=np.float64
          ),
          signals=(),
          request=ChannelRequest(("interface-traction",), ()),
        )
      )
      # Residual floor audit (M77): max|fint| over this battery is 6.4e3
      # (Dummy D=1e5 states; ulp 9.1e-13), so atol=1.0e-9 sits ~280x above
      # the 4-ulp class and needs no widening — CI run 38023817372 passed
      # this battery on ubuntu with it.
      bitwise_pin(
        evaluation.residual_values[0].values[0],
        legacy_fint,
        rtol=_LAW_RTOL,
        atol=1.0e-9,
      )
      # The trial frame agrees with legacy's updated history on these
      # branches (the virgin/aligned/flipped cases all coincide here).
      np.testing.assert_allclose(
        evaluation.trial_state.values[0],
        legacy_normal,
        rtol=0.0,
        atol=1.0e-15,
      )


# --- the 45-degree reflection documentation -----------------------------------
#
# M62 survey finding 1 (agent-coh1, measured on the legacy element): the
# legacy frame is a REFLECTION — on a 45-degree element it reports a pure
# geometric-normal jump as pure shear and vice versa (normal/shear exactly
# swapped). The v3 family ships the corrected proper-rotation frame, so the
# same jump maps to pure normal. This test measures the v3 side; the legacy
# measurement is cited, not re-run, because M67 repairs the legacy frame
# concurrently (the repair provably leaves axis-aligned behavior unchanged —
# the runtime batteries above pin only that invariant scope).


def _build_operator(
  coordinates: tuple[tuple[float, float], ...],
  law: str = "Dummy",
  **law_kwargs: float,
) -> object:
  from pyfem.v3.compile.system import compile_system
  from pyfem.v3.compile.truss import truss_reference_registry
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
  )

  extra = ((0.0, 2.0), (1.0, 2.0))
  nodes = tuple(
    NodeSpec(id=index, coordinates=pair, source=_SOURCE)
    for index, pair in enumerate((*coordinates, *extra))
  )
  count = len(nodes)
  model = ModelSpec(
    mesh=MeshSpec(
      nodes=nodes,
      cell_blocks=(
        CellBlockSpec(
          id="bars",
          reference_topology="line",
          topological_dimension=1,
          embedding_dimension=2,
          geometry_interpolation="line2",
          cells=(CellSpec(id="c0", node_ids=(count - 2, count - 1), source=_SOURCE),),
          source=_SOURCE,
        ),
      ),
      source=_SOURCE,
    ),
    fields=(
      FieldSpec(
        id="displacement",
        components=("x", "y"),
        location="node",
        source=_SOURCE,
      ),
    ),
    materials=(
      MaterialSpec(
        id="steel",
        model="uniaxial-linear-elastic",
        parameters=(
          MaterialParameterSpec("youngs_modulus", 1.0e6),
          MaterialParameterSpec("area", 1.0),
        ),
        source=_SOURCE,
      ),
    ),
    regions=(
      RegionSpec(
        id="domain",
        cell_refs=(CellRef("bars", "c0"),),
        field_ids=("displacement",),
        material_id="steel",
        formulation="total-lagrangian-truss",
        quadrature="none",
        source=_SOURCE,
      ),
    ),
    source=_SOURCE,
  )
  system = compile_system(model, truss_reference_registry())
  builders = {
    "Dummy": dummy_interface_declaration,
    "XuNeedleman": xu_needleman_declaration,
  }
  kwargs = {"stiffness": 1.0} if law == "Dummy" else law_kwargs
  _, operator = compile_interface_operator(
    system,
    builders[law](
      block_id="coh",
      space_id="displacement",
      interface_ids=(1,),
      node_quads=((0, 1, 2, 3),),
      source=_SOURCE,
      **kwargs,
    ),
  )
  return operator


def test_corrected_frame_at_45_degrees_maps_normal_to_normal() -> None:
  # Element along the 45-degree line: coincident pairs at (0,0) and (1,1).
  operator = _build_operator(((0.0, 0.0), (1.0, 1.0), (0.0, 0.0), (1.0, 1.0)))
  evaluation = operator.evaluate(
    OperatorEvaluationInput(
      port_values=(FinalizedArray(np.zeros((1, 8)), dtype=np.float64),),
      accepted_state=FinalizedArray(np.zeros((1, 2)), dtype=np.float64),
      signals=(),
      request=ChannelRequest(("interface-traction",), ()),
    )
  )
  # The corrected normal of the (1, 1) segment is (-1, 1)/sqrt(2), and the
  # frame is a proper rotation (det +1) — legacy's is a reflection (det -1).
  normal = evaluation.trial_state.values[0]
  np.testing.assert_allclose(
    normal,
    np.array([-1.0, 1.0]) / math.sqrt(2.0),
    rtol=0.0,
    atol=1.0e-15,
  )
  rotation = np.array([[normal[0], normal[1]], [-normal[1], normal[0]]])
  assert np.linalg.det(rotation) == pytest.approx(1.0, rel=0.0, abs=1.0e-15)

  # A pure geometric-normal jump of the top pair: u_top = delta * n with
  # delta = 1e-3, bottom pair fixed. With the corrected frame the local jump
  # is pure normal: the linear Dummy law answers with traction (D*delta, 0),
  # and the top-left node feels that traction times its shape weight along
  # the frame rows. Legacy's reflecting frame reads this state as pure SHEAR
  # (finding 1's measured swap) — its nodal force would sit along the
  # geometric tangent instead.
  delta = 1.0e-3
  state = np.array(
    [
      [
        0.0,
        0.0,
        0.0,
        0.0,
        delta * normal[0],
        delta * normal[1],
        delta * normal[0],
        delta * normal[1],
      ]
    ]
  )
  response = operator.evaluate(
    OperatorEvaluationInput(
      port_values=(FinalizedArray(state, dtype=np.float64),),
      accepted_state=FinalizedArray(np.zeros((1, 2)), dtype=np.float64),
      signals=(),
      request=ChannelRequest(("interface-traction",), ()),
    )
  )
  residual = response.residual_values[0].values[0]
  # The jump is constant along the element, so each Gauss point carries the
  # same traction; the nodal forces follow the frame: top nodes pull along
  # +n * D * delta * (integration weight sums), bottom nodes mirror them.
  length = math.sqrt(2.0)
  jacobian_weight = 0.5 * length
  top_left = residual[4:6]
  expected_direction = np.array([rotation[0, 0], rotation[0, 1]])
  np.testing.assert_allclose(
    top_left,
    expected_direction * delta * jacobian_weight,
    rtol=1.0e-12,
    atol=1.0e-15,
  )
  # And the shear channel is exactly silent: the force is along the normal,
  # not the tangent (legacy would answer along (n1, n0)-style shear rows).
  tangent_direction = np.array([rotation[1, 0], rotation[1, 1]])
  assert float(np.dot(top_left, tangent_direction)) == pytest.approx(
    0.0,
    abs=1.0e-18,
  )


# --- tangent honesty: the documented divergence from legacy --------------------


def test_exact_tangent_diverges_from_legacy_by_the_frame_terms() -> None:
  """The M65 tangent-honesty evidence battery, measured on one element.

  Legacy omits the frame-motion terms (the survey measured 6e-3..1e-2
  relative against a finite difference of its own residual); at this
  battery's moderate sheared-opening state the legacy tangent sits 2.2e-2
  off its own residual's FD Jacobian, while the v3 exact tangent matches the
  FD of the v3 residual to 3.6e-10. The v3/legacy tangent divergence equals
  that legacy inconsistency (v3 is FD-true), and the true Jacobian is
  nonsymmetric at the same order — the honest channel flags carry it.
  """
  operator = _build_operator(
    ((0.0, 0.0), (1.0, 0.0), (0.0, 0.0), (1.0, 0.0)),
    law="XuNeedleman",
    fracture_energy=0.1,
    ultimate_traction=0.5,
  )
  rng = np.random.default_rng(3)
  u1, v1, u2, v2 = rng.standard_normal(4) * 0.02
  state = np.array([u1, v1, u2, v2, u1, -v1, u2, -v2])

  def v3_response(
    point: np.ndarray,
  ) -> tuple[np.ndarray, np.ndarray]:
    evaluation = operator.evaluate(
      OperatorEvaluationInput(
        port_values=(FinalizedArray(point.reshape(1, 8).copy(), dtype=np.float64),),
        accepted_state=FinalizedArray(np.zeros((1, 2)), dtype=np.float64),
        signals=(),
        request=ChannelRequest(("interface-traction",), ("interface-tangent",)),
      )
    )
    return (
      evaluation.residual_values[0].values[0],
      evaluation.jacobian_values[0].values[0],
    )

  legacy_fint, legacy_tangent, _ = _legacy_element_response("XuNeedleman", state, None)
  residual_v3, tangent_v3 = v3_response(state)
  # Residual parity on the axis-aligned state holds bitwise while the
  # tangents diverge by the frame terms.
  np.testing.assert_array_equal(
    residual_v3.view(np.uint64),
    legacy_fint.view(np.uint64),
  )
  step = 1.0e-7
  fd_legacy = np.zeros((8, 8))
  fd_v3 = np.zeros((8, 8))
  for dof in range(8):
    plus = state.copy()
    minus = state.copy()
    plus[dof] += step
    minus[dof] -= step
    fd_legacy[:, dof] = (
      _legacy_element_response("XuNeedleman", plus, None)[0]
      - _legacy_element_response("XuNeedleman", minus, None)[0]
    ) / (2.0 * step)
    fd_v3[:, dof] = (v3_response(plus)[0] - v3_response(minus)[0]) / (2.0 * step)
  scale = float(np.abs(legacy_tangent).max())
  legacy_inconsistency = float(np.abs(legacy_tangent - fd_legacy).max()) / scale
  v3_inconsistency = float(np.abs(tangent_v3 - fd_v3).max()) / scale
  divergence = float(np.abs(tangent_v3 - legacy_tangent).max()) / scale
  # Measured on the reference platform: legacy 2.164e-2, v3 3.6e-10,
  # divergence 2.164e-2, true nonsymmetry 2.164e-2 — consistent with the
  # survey's 6e-3..1e-2 band (different probe states).
  assert legacy_inconsistency > 1.0e-3
  assert v3_inconsistency < 1.0e-6
  assert divergence > 1.0e-3
  assert abs(divergence - legacy_inconsistency) < 0.1 * divergence
  assert float(np.abs(tangent_v3 - tangent_v3.T).max()) / scale > 1.0e-3, (
    "the true Jacobian's nonsymmetry justifies symmetric=False"
  )


# --- skim oracles -------------------------------------------------------------


def _legacy_peel_cycles() -> list[tuple[int, int, np.ndarray]]:
  """Legacy ``NonlinearSolver`` on the peel skim, cycle by cycle."""
  from pyfem.io.InputReader import InputRead
  from pyfem.solvers.NonlinearSolver import NonlinearSolver
  from pyfem.v3.io.solver_pro import parse_nonlinear_solver_settings

  pro_path = ROOT / "skims" / "peel60pres" / "skim.pro"
  props, globdat = InputRead(str(pro_path))
  solver = NonlinearSolver(props, globdat)
  settings = parse_nonlinear_solver_settings(pro_path.read_text(encoding="utf-8"))
  assert settings is not None
  solver.tol = settings.tol
  solver.iterMax = settings.iter_max
  solver.maxCycle = settings.max_cycle
  if settings.load_table is not None:
    solver.loadTable = settings.load_table
  cycles: list[tuple[int, int, np.ndarray]] = []
  while globdat.active:
    solver.run(props, globdat)
    cycles.append(
      (
        globdat.solverStatus.cycle,
        globdat.solverStatus.iiter,
        np.asarray(globdat.state).copy(),
      )
    )
  return cycles


def _peel_drivers() -> tuple[object, object, tuple]:
  """The compiled peel skim plus its driver settings, run_deck-style."""
  from pyfem.v3.driver import NonlinearStaticDriver, NonlinearStaticSettings
  from pyfem.v3.io.legacy_deck import compile_deck
  from pyfem.v3.spec.program import ProgramCoordinateValue, ProgramPoint

  skim_dir = ROOT / "skims" / "peel60pres"
  deck = read_legacy_deck(skim_dir / "skim.pro")
  compiled = compile_deck(deck)
  driver = NonlinearStaticDriver(
    compiled.system,
    compiled.constraint_map,
    compiled.loads,
    NonlinearStaticSettings(
      tolerance=deck.solver.tolerance,
      max_iterations=deck.solver.max_iterations,
    ),
  )
  points = tuple(
    ProgramPoint((ProgramCoordinateValue("load", factor),))
    for factor in deck.solver.load_factors
  )
  return deck, driver, points


def test_peel_skim_converts_and_matches_legacy() -> None:
  skim_dir = ROOT / "skims" / "peel60pres"
  deck = read_legacy_deck(skim_dir / "skim.pro")
  assert len(deck.interfaces) == 1
  declaration = deck.interfaces[0]
  assert declaration.block_id == "InterfaceElem"
  assert declaration.kernel_name == "xu-needleman-rank2"
  assert len(declaration.interface_ids) == 4
  assert declaration.parameters == (0.1, 0.5)

  run = run_deck(deck)
  result = run.result
  assert result.status is DriverStatus.COMPLETED
  rtol, atol = load_parity_tolerances(skim_dir)
  cycles = _legacy_peel_cycles()
  legacy = cycles[-1][2]
  # Observed deviation on the reference platform: 1.1e-16 — the mirror-
  # symmetric peel keeps the frame segment exactly horizontal, so both sides
  # converge to the same state; the parity.toml band (1e-9/1e-11) carries
  # orders of headroom for cross-platform solver-path rounding.
  np.testing.assert_allclose(run.state, legacy, rtol=rtol, atol=atol)

  # Committed-state parity per substep: fresh drivers on every target prefix
  # land on the legacy cycle states.
  from pyfem.v3.spec.program import ProgramCoordinateValue, ProgramPoint

  for step, (_, _, legacy_state_k) in enumerate(cycles, start=1):
    _, prefix_driver, points = _peel_drivers()
    prefix = prefix_driver.run(
      base_point=ProgramPoint((ProgramCoordinateValue("load", 0.0),)),
      target_points=points[:step],
    )
    assert prefix.status is DriverStatus.COMPLETED
    np.testing.assert_allclose(
      prefix_driver.owner.accepted_physical().values,
      legacy_state_k,
      rtol=rtol,
      atol=atol,
    )

  # The exact-tangent driver-level consequence: the v3 Newton path never
  # needs more evaluations than the legacy cycles needed solves (plus the
  # one entering-state evaluation per substep, which legacy skips by always
  # solving once). Measured on the reference platform: per-substep v3
  # evaluations == legacy solves + 1, and total solves equal (22).
  assert len(result.records) == len(cycles)
  for record, (_, legacy_iiter, _) in zip(result.records, cycles, strict=True):
    assert len(record.iterations) <= legacy_iiter + 1
  legacy_solves = sum(iiter for _, iiter, _ in cycles)
  assert result.statistics.linear_solve_count <= legacy_solves
  assert result.statistics.cutback_count == 0

  # The peel equilibrates: the mirror-symmetric prescribed ends carry equal
  # and opposite reactions, and the seam carries the peel force.
  observation = result.records[-1].observation
  assert observation is not None
  reactions = observation.reactions.values
  assert float(reactions[1::2].sum()) == pytest.approx(0.0, abs=1.0e-8)
  assert float(reactions[::2].sum()) == pytest.approx(0.0, abs=1.0e-8)


def test_traction_oscillations_linear_oracle() -> None:
  skim_dir = ROOT / "skims" / "traction_oscillations"
  deck = read_legacy_deck(skim_dir / "skim.pro")
  assert len(deck.interfaces) == 1
  assert deck.interfaces[0].kernel_name == "dummy-linear-interface"
  assert len(deck.interfaces[0].interface_ids) == 6

  run = run_deck(deck)
  result = run.result
  assert result.status is DriverStatus.COMPLETED
  rtol, atol = load_parity_tolerances(skim_dir)
  legacy = legacy_state(skim_dir / "skim.pro")
  # The Dummy law's assembled response is frame-independent linear, so both
  # sides solve the same linear system once: one committed substep, one
  # factorization, one linear solve. Observed deviation 5.2e-16 on the
  # reference platform (solver-path rounding only); the parity.toml band
  # (1e-10/1e-12) carries the usual headroom.
  np.testing.assert_allclose(run.state, legacy, rtol=rtol, atol=atol)
  statistics = result.statistics
  assert statistics.committed_substep_count == 1
  assert statistics.factorization_count == 1
  assert statistics.linear_solve_count == 1

  # The oracle's raison d'être: the seam tractions oscillate along x (the
  # book's section 13.2 demonstration). The per-element normal force profile
  # on the top bank is the classic bathtub: mirror-symmetric with the center
  # materially below the ends (measured here: -119.8 at the ends, -55.3 at
  # the center).
  operator = next(
    item
    for item in run.driver.owner.system.operators
    if hasattr(item, "interface_block")
  )
  gather = operator.header.ports[0].coefficient_map.values
  state = run.state
  evaluation = operator.evaluate(
    OperatorEvaluationInput(
      port_values=(FinalizedArray(state[gather].copy(), dtype=np.float64),),
      accepted_state=run.driver.owner.accepted_state(
        operator.header.state_layout.block_id
      ),
      signals=(),
      request=ChannelRequest(("interface-traction",), ()),
    )
  )
  residual = evaluation.residual_values[0].values
  left_profile = residual[:, 5]  # top-left node normal force per element
  right_profile = residual[:, 7]  # top-right node normal force per element
  # Mirror symmetry maps element i's left node onto element (n-1-i)'s right.
  np.testing.assert_allclose(
    left_profile,
    right_profile[::-1],
    rtol=1.0e-9,
    atol=1.0e-9,
  )
  assert abs(left_profile[0]) > 1.5 * abs(left_profile[len(left_profile) // 2])
