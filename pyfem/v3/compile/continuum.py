"""Concrete direct compiler and evaluators for the qualified Q8 continuum slices.

The continuum family compiles three formulations sharing two-dimensional
serendipity-quad8 geometry and 3x3 Gauss quadrature:

- ``small-strain-continuum``: one node displacement field and a linear-elastic
  or stateful material. This is the landed single-field slice; its compiled
  output is byte-identical to the original direct compiler. The small-strain
  mechanical formulation additionally covers the geometry-and-material breadth
  wave: the ``linear-tria3`` and ``bilinear-quad4`` two-dimensional geometries
  (pinned to the ``gauss-tria3-1`` and ``gauss-2x2`` rules), the
  ``trilinear-hex8`` three-dimensional geometry (pinned to ``gauss-2x2x2``,
  displacement components ``('x', 'y', 'z')``, six-component Voigt
  kinematics), the ``plane-strain-linear-elastic`` law beside plane stress on
  every two-dimensional geometry, and the ``isotropic-linear-elastic`` 3D law
  on the hexahedral geometry. The region's authored formulation string names
  the physics; the cell block's authored geometry selects the pinned
  topology/quadrature recipe and the rank-specific kinematics descriptor.
- ``small-strain-thermal-continuum``: one node temperature field carrying
  exactly one component and an isotropic linear conductor (quad8 only).
- ``small-strain-thermo-elastic-continuum``: one displacement field followed
  by one temperature field in the region field signature, with a coupled
  linear thermo-elastic material (quad8 only). The operator binds both spaces
  with per-port coefficient maps and carries exactly the nonzero static
  Jacobian blocks: the mechanical tangent (displacement from displacement,
  symmetric), the thermal-expansion tangent (displacement from temperature,
  nonsymmetric), and the conduction tangent (temperature from temperature,
  symmetric). A stateless static thermo-elastic operator honestly has no
  mechanical-to-thermal block: that coupling is a rate effect owned by
  stateful formulations with accepted state and signals.

The small-strain formulation is also the open stateful seam: a region whose
material is not one of the pinned linear-elastic laws compiles through the
generic descriptor-driven stateful path whenever its registry descriptor
carries valid v2 stateful metadata and a binding implementing the
``StatefulContinuumBinding`` protocol (calibration call, batched kernel,
optional ``initial_state``) — on the serendipity-quad8 geometry only. The path
emits nonzero-width ``OperatorStateLayout`` values from descriptor
``state_slots`` with the entity axis flattened (element x integration point x
material slot), wires the descriptor's initial-state rows into the layout, and
marks every channel nonlinear per the declared tangent class. The first such
law is ``isotropic-hardening-plasticity`` (J2, 19-float rows); the second is
``plane-strain-damage`` (the legacy isotropic damage law, 1-float kappa
envelope rows), the first ``algorithmic-nonsymmetric`` tangent-class witness:
its progressive-branch rank-1 correction tangent honestly drives the Jacobian
channel flags ``linear=False, symmetric=False``, which the driver consumes
through the general splu path with no symmetry assumption. The third is
``prony-viscoelasticity`` (the legacy generalized-Maxwell Prony law, 6*n + 13
rows with a parameterized per-term internal-strain slot), the first
signal-consuming production law: it declares the optional ``signal_ports``
field below and consumes the schedule-owned time coordinate through its
identity port. The fourth and fifth are the rate-dependent family's stateful
pair, signal-consuming through the same identity time port:
``perzyna-viscoplasticity`` (the legacy ViscoPlasticity law — Perzyna-branded
but integrating a rate-INDEPENDENT J2 map with a ``dtime > 0`` gate, 14-float
rows) and ``skorohod-olevsky`` (the legacy SOVS explicit viscous-sintering
law, 14-float rows, the first parameter-dependent initial state of the
family: ``rho = rho0``). Both ship the true algorithmic tangents of their
implemented maps with class ``algorithmic-symmetric`` — the legacy coded
tangents used to diverge from the maps they accompanied; M67 repaired the
legacy sides (494f30c, 30a5f4e; merge 8a3eae8) and M76 flipped the v3 pins
to repair-confirmed parity (b0ad9ed, merge 2ffcf20). The pre-repair
divergence records are retained in the kernel modules.

- ``total-lagrangian-continuum``: the finite-strain slice (serendipity-quad8
  only): one node displacement field and the plane-stress
  Saint-Venant-Kirchhoff law. The formulation descriptor binds the landed
  batched Q8 total-Lagrangian element kernel
  (``pyfem.v3.fem.tl_element.quad8_tl_tangent_batched``), which the compiler
  probes at compile time against a qualified reference assembly built from
  the validated quadrature/topology/material ingredients — at the zero state
  (where the response is the small-strain stiffness and an exactly-zero
  internal force) and at a fixed uniform displacement-gradient probe state.
  The operator carries a zero-width state layout and nonlinear, symmetric
  channels (material and geometric tangent parts).

A stateful descriptor may additionally declare the optional ``signal_ports``
field: typed program-signal ports emitted as ``SignalPortBinding`` values on
the operator header. The compiled operator then accepts one bound
``ProgramSignalInput`` per declared port and forwards the validated scalar
values to the kernel's fourth positional argument, so schedule-owned signals
(time-like coordinates) reach the law without any hidden global. Descriptors
without the field compile byte-identical operators that reject every signal.

A stateful descriptor may likewise open the parameter-derivative channel: a
binding implementing the optional ``param_derivative_kernel`` member (the
``StatefulContinuumBinding`` protocol) differentiates exactly the descriptor's
``parameter_names``, and the compiled header then carries
``ParameterBinding`` and ``ResidualDerivativeChannel`` declarations (one
``dinternal-force/d<name>`` channel per declared parameter, in
``parameter_names`` order). The operator answers derivative channel requests
with per-element residual derivatives assembled from the kernel's exact
per-IP ``d(stress)/d(parameter)`` columns through the same internal-force
expression as the residual itself, and the virgin-state probe exercises every
declared derivative channel before the operator can escape. Bindings without
the member compile byte-identical channel-free operators that reject every
derivative request fail-closed.

Capability boundary: every cell belongs to exactly one region; each region
draws its cells from exactly one cell block; each cell block feeds exactly
one region; every declared field and material is referenced by at least one
region. Geometry/material/rank mismatches (a two-dimensional law on a
three-dimensional block, a 3D law on a planar block, a stateful law on a
non-quad8 block, a quadrature rule foreign to the block geometry) fail with
coded source-context diagnostics, as do all other violations.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from typing import NoReturn

import numpy as np

from pyfem.v3.compile.contracts import (
  CONTINUUM_3D_FORMULATION_KEY,
  HEX8_QUADRATURE_KEY,
  HEX8_TOPOLOGY_KEY,
  ISOTROPIC_MATERIAL_KEY,
  PLANE_STRAIN_MATERIAL_KEY,
  QUAD4_QUADRATURE_KEY,
  QUAD4_TOPOLOGY_KEY,
  TL_FORMULATION_KEY,
  TL_MATERIAL_KEY,
  TRIA3_QUADRATURE_KEY,
  TRIA3_TOPOLOGY_KEY,
  StatefulContinuumKernel,
  StatefulContinuumKernelResult,
  StatefulContinuumSignalDerivative,
  StatefulContinuumSignalInput,
  StatefulContinuumSignalKernel,
  breadth_descriptor_metadata,
  build_material_state_layout,
  finite_strain_descriptor_metadata,
  resolve_material_signal_ports,
  resolve_material_state_slots,
  stateful_tangent_channel_flags,
  validate_stateful_material_metadata,
  validated_material_initial_rows,
)
from pyfem.v3.compile.diagnostics import (
  ModelCompilationDiagnostic,
  ModelCompilationError,
)
from pyfem.v3.fem.element import continuum_internal_force_batched
from pyfem.v3.fem.kinematics import strain_displacement
from pyfem.v3.fem.quadrature import gauss_tensor_product_2d
from pyfem.v3.fem.shapes import (
  bilinear_quad4,
  linear_tria3,
  serendipity_quad8,
  trilinear_hex8,
)
from pyfem.v3.materials.isotropic import isotropic_matrix
from pyfem.v3.materials.isotropic_hardening_plasticity import (
  ISOTROPIC_HARDENING_PLASTICITY_BINDING,
  isotropic_hardening_plasticity_metadata,
)
from pyfem.v3.materials.perzyna_viscoplasticity import (
  PERZYNA_VISCOPLASTICITY_BINDING,
  perzyna_viscoplasticity_metadata,
)
from pyfem.v3.materials.plane_strain import plane_strain_matrix
from pyfem.v3.materials.plane_strain_damage import (
  PLANE_STRAIN_DAMAGE_BINDING,
  plane_strain_damage_metadata,
)
from pyfem.v3.materials.plane_stress import plane_stress_matrix
from pyfem.v3.materials.prony_viscoelasticity import (
  PRONY_VISCOELASTICITY_BINDING,
  prony_viscoelasticity_metadata,
)
from pyfem.v3.materials.skorohod_olevsky import (
  SKOROHOD_OLEVSKY_BINDING,
  skorohod_olevsky_metadata,
)
from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.operator import (
  BalanceRole,
  ChannelRequest,
  CompiledOperator,
  CompilerConstructed,
  CouplingPolicy,
  EvaluationStatus,
  ImplementationIdentity,
  JacobianChannel,
  OperatorEvaluation,
  OperatorEvaluationInput,
  OperatorHeader,
  OperatorStateLayout,
  ParameterBinding,
  PortBinding,
  PortMode,
  ProgramSignalInput,
  ResidualChannel,
  ResidualDerivativeChannel,
  SignalDerivativeInput,
  SignalPortBinding,
  StateLifetime,
  evaluation_derivative_values,
  evaluation_status,
)
from pyfem.v3.model.provenance import CanonicalManifest
from pyfem.v3.model.registry import (
  RegistryDescriptor,
  RegistryKey,
  RegistrySnapshot,
)
from pyfem.v3.model.system import CompiledSource, DiscreteSpace, IncidenceEntityBlock
from pyfem.v3.spec.diagnostics import SourceContext, render_diagnostic_value
from pyfem.v3.spec.model import (
  CellBlockSpec,
  CellSpec,
  FieldSpec,
  MaterialParameterSpec,
  MaterialSpec,
  ModelSpec,
  RegionSpec,
  SpecId,
)

Q8_TOPOLOGY_KEY: RegistryKey = ("topology", "serendipity-quad8")
Q8_QUADRATURE_KEY: RegistryKey = ("quadrature", "gauss-3x3")
Q8_FORMULATION_KEY: RegistryKey = ("formulation", "small-strain-continuum")
Q8_MATERIAL_KEY: RegistryKey = ("material", "plane-stress-linear-elastic")
THERMAL_FORMULATION_KEY: RegistryKey = (
  "formulation",
  "small-strain-thermal-continuum",
)
THERMAL_MATERIAL_KEY: RegistryKey = ("material", "linear-thermal-conductor")
THERMO_FORMULATION_KEY: RegistryKey = (
  "formulation",
  "small-strain-thermo-elastic-continuum",
)
THERMO_MATERIAL_KEY: RegistryKey = ("material", "linear-thermo-elastic")
PLASTIC_MATERIAL_KEY: RegistryKey = ("material", "isotropic-hardening-plasticity")
DAMAGE_MATERIAL_KEY: RegistryKey = ("material", "plane-strain-damage")
VISCOELASTIC_MATERIAL_KEY: RegistryKey = ("material", "prony-viscoelasticity")
VISCOPLASTIC_MATERIAL_KEY: RegistryKey = ("material", "perzyna-viscoplasticity")
SOVS_MATERIAL_KEY: RegistryKey = ("material", "skorohod-olevsky")
_PARAMETER_NAMES = ("youngs_modulus", "poisson_ratio")
_THERMAL_PARAMETER_NAMES = ("conductivity",)
_THERMO_PARAMETER_NAMES = (
  "youngs_modulus",
  "poisson_ratio",
  "thermal_expansion",
  "conductivity",
)
_POINT_COUNT = 9
_NODE_COUNT = 8
_LOCAL_COEFFICIENT_COUNT = 16
_MATERIAL_RELATIVE_TOLERANCE = 16.0 * float(np.finfo(np.float64).eps)

_DISPLACEMENT_ROLE = ("displacement", ("x", "y"))
_DISPLACEMENT_3D_ROLE = ("displacement", ("x", "y", "z"))
_TEMPERATURE_ROLE = ("temperature", "scalar")
_FORMULATION_CONTRACTS = {
  Q8_FORMULATION_KEY[1]: (_DISPLACEMENT_ROLE,),
  THERMAL_FORMULATION_KEY[1]: (_TEMPERATURE_ROLE,),
  THERMO_FORMULATION_KEY[1]: (_DISPLACEMENT_ROLE, _TEMPERATURE_ROLE),
  TL_FORMULATION_KEY[1]: (_DISPLACEMENT_ROLE,),
}
# The small-strain formulation is the open stateful seam (any captured v2
# stateful material descriptor may bind); the thermal formulations keep their
# closed material sets, and so does the finite-strain slice (the plane-stress
# Saint-Venant-Kirchhoff law the legacy FiniteStrainContinuum deck ships).
_FORMULATION_MATERIAL_MODELS = {
  THERMAL_FORMULATION_KEY[1]: THERMAL_MATERIAL_KEY[1],
  THERMO_FORMULATION_KEY[1]: THERMO_MATERIAL_KEY[1],
  TL_FORMULATION_KEY[1]: TL_MATERIAL_KEY[1],
}


@dataclass(frozen=True, slots=True)
class ContinuumGeometryProfile:
  """The pinned geometry recipe of one supported continuum cell-block shape.

  ``quadrature`` is the authored region quadrature name the shape requires;
  ``formulation_key`` resolves the rank-specific small-strain kinematics
  descriptor; ``quadrature_measure`` is the parent-domain measure the pinned
  rule's weights must sum to.
  """

  geometry_interpolation: str
  reference_topology: str
  topological_dimension: int
  embedding_dimension: int
  node_count: int
  point_count: int
  quadrature: str
  quadrature_order: int
  quadrature_measure: float
  topology_key: RegistryKey
  quadrature_key: RegistryKey
  formulation_key: RegistryKey
  field_components: tuple[str, ...]
  local_coefficient_count: int
  voigt_size: int


_Q8_GEOMETRY = ContinuumGeometryProfile(
  geometry_interpolation="serendipity-quad8",
  reference_topology="quadrilateral",
  topological_dimension=2,
  embedding_dimension=2,
  node_count=_NODE_COUNT,
  point_count=_POINT_COUNT,
  quadrature=Q8_QUADRATURE_KEY[1],
  quadrature_order=3,
  quadrature_measure=4.0,
  topology_key=Q8_TOPOLOGY_KEY,
  quadrature_key=Q8_QUADRATURE_KEY,
  formulation_key=Q8_FORMULATION_KEY,
  field_components=("x", "y"),
  local_coefficient_count=_LOCAL_COEFFICIENT_COUNT,
  voigt_size=3,
)
_TRIA3_GEOMETRY = ContinuumGeometryProfile(
  geometry_interpolation="linear-tria3",
  reference_topology="triangle",
  topological_dimension=2,
  embedding_dimension=2,
  node_count=3,
  point_count=1,
  quadrature=TRIA3_QUADRATURE_KEY[1],
  quadrature_order=1,
  quadrature_measure=0.5,
  topology_key=TRIA3_TOPOLOGY_KEY,
  quadrature_key=TRIA3_QUADRATURE_KEY,
  formulation_key=Q8_FORMULATION_KEY,
  field_components=("x", "y"),
  local_coefficient_count=6,
  voigt_size=3,
)
_QUAD4_GEOMETRY = ContinuumGeometryProfile(
  geometry_interpolation="bilinear-quad4",
  reference_topology="quadrilateral",
  topological_dimension=2,
  embedding_dimension=2,
  node_count=4,
  point_count=4,
  quadrature=QUAD4_QUADRATURE_KEY[1],
  quadrature_order=2,
  quadrature_measure=4.0,
  topology_key=QUAD4_TOPOLOGY_KEY,
  quadrature_key=QUAD4_QUADRATURE_KEY,
  formulation_key=Q8_FORMULATION_KEY,
  field_components=("x", "y"),
  local_coefficient_count=8,
  voigt_size=3,
)
_HEX8_GEOMETRY = ContinuumGeometryProfile(
  geometry_interpolation="trilinear-hex8",
  reference_topology="hexahedron",
  topological_dimension=3,
  embedding_dimension=3,
  node_count=8,
  point_count=8,
  quadrature=HEX8_QUADRATURE_KEY[1],
  quadrature_order=2,
  quadrature_measure=8.0,
  topology_key=HEX8_TOPOLOGY_KEY,
  quadrature_key=HEX8_QUADRATURE_KEY,
  formulation_key=CONTINUUM_3D_FORMULATION_KEY,
  field_components=("x", "y", "z"),
  local_coefficient_count=24,
  voigt_size=6,
)
_CONTINUUM_GEOMETRIES = {
  profile.geometry_interpolation: profile
  for profile in (_Q8_GEOMETRY, _TRIA3_GEOMETRY, _QUAD4_GEOMETRY, _HEX8_GEOMETRY)
}
# The thermal formulations remain pinned to the landed quad8 slice; only the
# small-strain mechanical formulation admits the breadth geometries.
_FORMULATION_GEOMETRIES = {
  Q8_FORMULATION_KEY[1]: frozenset(
    (
      _Q8_GEOMETRY.geometry_interpolation,
      _TRIA3_GEOMETRY.geometry_interpolation,
      _QUAD4_GEOMETRY.geometry_interpolation,
      _HEX8_GEOMETRY.geometry_interpolation,
    )
  ),
  THERMAL_FORMULATION_KEY[1]: frozenset((_Q8_GEOMETRY.geometry_interpolation,)),
  THERMO_FORMULATION_KEY[1]: frozenset((_Q8_GEOMETRY.geometry_interpolation,)),
  # The landed TL kernel is the two-dimensional serendipity-quad8 slice only.
  TL_FORMULATION_KEY[1]: frozenset((_Q8_GEOMETRY.geometry_interpolation,)),
}
_BREADTH_DESCRIPTOR_KEYS = frozenset(
  (
    TRIA3_TOPOLOGY_KEY,
    QUAD4_TOPOLOGY_KEY,
    HEX8_TOPOLOGY_KEY,
    TRIA3_QUADRATURE_KEY,
    QUAD4_QUADRATURE_KEY,
    HEX8_QUADRATURE_KEY,
    CONTINUUM_3D_FORMULATION_KEY,
    PLANE_STRAIN_MATERIAL_KEY,
    ISOTROPIC_MATERIAL_KEY,
  )
)
_PLANAR_LINEAR_MATERIAL_MODELS = frozenset(
  (Q8_MATERIAL_KEY[1], PLANE_STRAIN_MATERIAL_KEY[1])
)


def _new[ValueT](cls: type[ValueT], /, **fields: object) -> ValueT:
  value = object.__new__(cls)
  for name, field in fields.items():
    object.__setattr__(value, name, field)
  return value


def _fail(code: str, message: str, source: SourceContext) -> NoReturn:
  raise ModelCompilationError(
    (ModelCompilationDiagnostic(code=code, message=message, source=source),)
  )


def _sort_key(value: SpecId | tuple[SpecId, ...]) -> tuple[int, object]:
  if type(value) is int:
    return 0, value
  if type(value) is str:
    return 1, value
  return 2, tuple(_sort_key(item) for item in value)


def _source(value: SourceContext) -> CompiledSource:
  return _new(
    CompiledSource,
    source=value.source,
    line=value.line,
    column=value.column,
  )


@dataclass(frozen=True, slots=True, eq=False)
class RegionSelection:
  block: CellBlockSpec
  fields: tuple[FieldSpec, ...]
  material: MaterialSpec
  region: RegionSpec
  geometry: ContinuumGeometryProfile
  cells: tuple[CellSpec, ...]


@dataclass(frozen=True, slots=True, eq=False)
class ContinuumSelection:
  regions: tuple[RegionSelection, ...]


@dataclass(frozen=True, slots=True, eq=False, init=False)
class Q8ContinuumPayload(CompilerConstructed):
  quadrature_points: FinalizedArray
  quadrature_weights: FinalizedArray
  shape_values: FinalizedArray
  parent_gradients: FinalizedArray
  geometry_scales: FinalizedArray
  normalized_gradients: FinalizedArray
  normalized_strain_displacement: FinalizedArray
  normalized_integration_weights: FinalizedArray
  constitutive: FinalizedArray
  material_parameters: FinalizedArray

  def physical_gradients(self) -> FinalizedArray:
    values = (
      self.normalized_gradients.values
      / self.geometry_scales.values[:, None, None, None]
    )
    return FinalizedArray(values, dtype=np.float64)

  def physical_strain_displacement(self) -> FinalizedArray:
    values = (
      self.normalized_strain_displacement.values
      / self.geometry_scales.values[:, None, None, None]
    )
    return FinalizedArray(values, dtype=np.float64)

  def physical_integration_weights(self) -> FinalizedArray:
    scales = self.geometry_scales.values[:, None]
    values = self.normalized_integration_weights.values * scales * scales
    return FinalizedArray(values, dtype=np.float64)


@dataclass(frozen=True, slots=True, eq=False, init=False)
class Q8ContinuumOperator(CompilerConstructed):
  header: OperatorHeader
  entity_block: IncidenceEntityBlock
  payload: Q8ContinuumPayload
  content_manifest: CanonicalManifest

  def evaluate(
    self,
    inputs: OperatorEvaluationInput,
  ) -> OperatorEvaluation:
    """Evaluate local internal force and material tangent from compiled meaning."""
    if type(inputs) is not OperatorEvaluationInput:
      msg = "Q8 evaluation requires an exact immutable evaluation input"
      raise TypeError(msg)
    if type(inputs.port_values) is not tuple or len(inputs.port_values) != 1:
      msg = "Q8 evaluation requires exactly one displacement port batch"
      raise TypeError(msg)
    values = inputs.port_values[0].values
    expected = self.header.ports[0].coefficient_map.values.shape
    if (
      values.dtype != np.dtype(np.float64)
      or values.dtype.metadata is not None
      or values.shape != expected
      or not bool(np.isfinite(values).all())
    ):
      msg = "Q8 displacement port values must be a finite metadata-free float64 batch"
      raise TypeError(msg)
    layout = self.header.state_layout
    accepted_state = inputs.accepted_state.values
    if (
      accepted_state.dtype != np.dtype(np.float64)
      or accepted_state.dtype.metadata is not None
      or accepted_state.shape != layout.row_shape
      or not bool(np.isfinite(accepted_state).all())
    ):
      msg = "Q8 accepted state must match the compiled zero-width state layout"
      raise TypeError(msg)
    if inputs.signals or self.header.signal_ports:
      msg = "Q8 model operator does not accept program signal inputs"
      raise ValueError(msg)
    residual_ids = tuple(item.channel_id for item in self.header.residual_channels)
    jacobian_ids = tuple(item.channel_id for item in self.header.jacobian_channels)
    derivative_ids = tuple(
      item.channel_id for item in getattr(self.header, "derivative_channels", ())
    )
    request = inputs.request
    derivative_request = request.derivative_channel_ids
    if (
      type(request) is not ChannelRequest
      or len(set(request.residual_channel_ids)) != len(request.residual_channel_ids)
      or len(set(request.jacobian_channel_ids)) != len(request.jacobian_channel_ids)
      or len(set(derivative_request)) != len(derivative_request)
      or not set(request.residual_channel_ids).issubset(residual_ids)
      or not set(request.jacobian_channel_ids).issubset(jacobian_ids)
      or not set(derivative_request).issubset(derivative_ids)
    ):
      msg = "Q8 evaluation request contains an unavailable or duplicate channel"
      raise ValueError(msg)

    b_matrix = self.payload.normalized_strain_displacement.values
    weights = self.payload.normalized_integration_weights.values
    constitutive = self.payload.constitutive.values
    tangent = np.einsum(
      "ep,epai,ab,epbj->eij",
      weights,
      b_matrix,
      constitutive,
      b_matrix,
      optimize=True,
    )
    residual_values = ()
    if request.residual_channel_ids:
      residual = np.einsum("eij,ej->ei", tangent, values, optimize=True)
      residual_values = (FinalizedArray(residual, dtype=np.float64),)
    jacobian_values = (
      (FinalizedArray(tangent, dtype=np.float64),)
      if request.jacobian_channel_ids
      else ()
    )
    return _new(
      OperatorEvaluation,
      residual_values=residual_values,
      jacobian_values=jacobian_values,
      trial_state=FinalizedArray(accepted_state, dtype=np.float64),
    )


@dataclass(frozen=True, slots=True, eq=False, init=False)
class Q8FiniteStrainPayload(CompilerConstructed):
  """Scale-normalized payload of the finite-strain (TL) Q8 continuum slice.

  ``normalized_node_coordinates`` carries the translation-free reference
  coordinates (per cell relative to the cell's first node, divided by the
  cell geometry scale); :meth:`physical_node_coordinates` rescales them for
  the TL kernel. The dropped translation leaves the kernel output invariant:
  the kernel differentiates coordinates through the shape gradients, which
  sum to zero.
  """

  quadrature_points: FinalizedArray
  quadrature_weights: FinalizedArray
  shape_values: FinalizedArray
  parent_gradients: FinalizedArray
  geometry_scales: FinalizedArray
  normalized_node_coordinates: FinalizedArray
  constitutive: FinalizedArray
  material_parameters: FinalizedArray

  def physical_node_coordinates(self) -> FinalizedArray:
    values = (
      self.normalized_node_coordinates.values
      * self.geometry_scales.values[:, None, None]
    )
    return FinalizedArray(values, dtype=np.float64)


def _tl_evaluation_array(
  value: object,
  *,
  shape: tuple[int, ...],
  label: str,
) -> np.ndarray:
  if (
    type(value) is not np.ndarray
    or value.shape != shape
    or value.dtype.metadata is not None
    or value.dtype.kind not in "iuf"
  ):
    msg = f"{label} must return a metadata-free numeric array {shape!r}"
    raise TypeError(msg)
  captured = np.array(value, dtype=np.float64, order="C", copy=True, subok=False)
  if not bool(np.isfinite(captured).all()):
    msg = "finite-strain evaluation response is not representable as finite float64"
    raise ValueError(msg)
  return captured


@dataclass(frozen=True, slots=True, eq=False, init=False)
class Q8FiniteStrainOperator(CompilerConstructed):
  """Finite-strain total-Lagrangian continuum operator (Q8, plane stress).

  Evaluation is pure and stateless: element displacement batches in, internal
  force and the consistent (material + geometric) tangent out, computed by
  the descriptor-bound landed TL kernel on the physical reference
  coordinates. Both channels are nonlinear; the tangent is symmetric.
  """

  header: OperatorHeader
  entity_block: IncidenceEntityBlock
  payload: Q8FiniteStrainPayload
  content_manifest: CanonicalManifest
  kernel: Callable[[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]

  def evaluate(
    self,
    inputs: OperatorEvaluationInput,
  ) -> OperatorEvaluation:
    """Evaluate internal force and consistent tangent from compiled meaning."""
    if type(inputs) is not OperatorEvaluationInput:
      msg = "finite-strain evaluation requires an exact immutable evaluation input"
      raise TypeError(msg)
    if type(inputs.port_values) is not tuple or len(inputs.port_values) != 1:
      msg = "finite-strain evaluation requires exactly one displacement port batch"
      raise TypeError(msg)
    values = inputs.port_values[0].values
    expected = self.header.ports[0].coefficient_map.values.shape
    if (
      values.dtype != np.dtype(np.float64)
      or values.dtype.metadata is not None
      or values.shape != expected
      or not bool(np.isfinite(values).all())
    ):
      msg = (
        "finite-strain displacement port values must be a finite metadata-free "
        "float64 batch"
      )
      raise TypeError(msg)
    layout = self.header.state_layout
    accepted_state = inputs.accepted_state.values
    if (
      accepted_state.dtype != np.dtype(np.float64)
      or accepted_state.dtype.metadata is not None
      or accepted_state.shape != layout.row_shape
      or not bool(np.isfinite(accepted_state).all())
    ):
      msg = "finite-strain accepted state must match the compiled zero-width layout"
      raise TypeError(msg)
    if inputs.signals or self.header.signal_ports:
      msg = "finite-strain model operator does not accept program signal inputs"
      raise ValueError(msg)
    residual_ids = tuple(item.channel_id for item in self.header.residual_channels)
    jacobian_ids = tuple(item.channel_id for item in self.header.jacobian_channels)
    derivative_ids = tuple(
      item.channel_id for item in getattr(self.header, "derivative_channels", ())
    )
    request = inputs.request
    derivative_request = request.derivative_channel_ids
    if (
      type(request) is not ChannelRequest
      or len(set(request.residual_channel_ids)) != len(request.residual_channel_ids)
      or len(set(request.jacobian_channel_ids)) != len(request.jacobian_channel_ids)
      or len(set(derivative_request)) != len(derivative_request)
      or not set(request.residual_channel_ids).issubset(residual_ids)
      or not set(request.jacobian_channel_ids).issubset(jacobian_ids)
      or not set(derivative_request).issubset(derivative_ids)
    ):
      msg = "finite-strain evaluation request contains an unavailable channel"
      raise ValueError(msg)

    coordinates = self.payload.physical_node_coordinates().values
    try:
      raw_response = self.kernel(
        coordinates,
        np.array(values, dtype=np.float64, order="C", copy=True),
        self.payload.constitutive.values,
      )
    except Exception:
      msg = "finite-strain formulation binding failed during evaluation"
      raise ValueError(msg) from None
    if type(raw_response) is not tuple or len(raw_response) != 2:
      msg = (
        "finite-strain formulation binding must return exactly the tangent "
        "and the internal force"
      )
      raise TypeError(msg)
    tangent = _tl_evaluation_array(
      raw_response[0],
      shape=(expected[0], expected[1], expected[1]),
      label="finite-strain tangent binding",
    )
    force = _tl_evaluation_array(
      raw_response[1],
      shape=expected,
      label="finite-strain internal force binding",
    )
    residual_values = (
      (FinalizedArray(force, dtype=np.float64),) if request.residual_channel_ids else ()
    )
    jacobian_values = (
      (FinalizedArray(tangent, dtype=np.float64),)
      if request.jacobian_channel_ids
      else ()
    )
    return _new(
      OperatorEvaluation,
      residual_values=residual_values,
      jacobian_values=jacobian_values,
      trial_state=FinalizedArray(accepted_state, dtype=np.float64),
    )


@dataclass(frozen=True, slots=True, eq=False, init=False)
class Hex8ContinuumPayload(CompilerConstructed):
  quadrature_points: FinalizedArray
  quadrature_weights: FinalizedArray
  shape_values: FinalizedArray
  parent_gradients: FinalizedArray
  geometry_scales: FinalizedArray
  normalized_gradients: FinalizedArray
  normalized_strain_displacement: FinalizedArray
  normalized_integration_weights: FinalizedArray
  constitutive: FinalizedArray
  material_parameters: FinalizedArray

  def physical_gradients(self) -> FinalizedArray:
    values = (
      self.normalized_gradients.values
      / self.geometry_scales.values[:, None, None, None]
    )
    return FinalizedArray(values, dtype=np.float64)

  def physical_strain_displacement(self) -> FinalizedArray:
    values = (
      self.normalized_strain_displacement.values
      / self.geometry_scales.values[:, None, None, None]
    )
    return FinalizedArray(values, dtype=np.float64)

  def physical_integration_weights(self) -> FinalizedArray:
    scales = self.geometry_scales.values[:, None]
    values = self.normalized_integration_weights.values * scales * scales * scales
    return FinalizedArray(values, dtype=np.float64)


@dataclass(frozen=True, slots=True, eq=False, init=False)
class Hex8ContinuumOperator(CompilerConstructed):
  """Three-dimensional small-strain linear continuum operator (hex8).

  The payload factors are scale-normalized like the two-dimensional slices,
  but the 3D stiffness does not share the planar scale cancellation:
  ``K = scale * ∫ B_normᵀ C B_norm dV_norm``, so evaluation multiplies the
  normalized einsum by the per-cell geometry scale.
  """

  header: OperatorHeader
  entity_block: IncidenceEntityBlock
  payload: Hex8ContinuumPayload
  content_manifest: CanonicalManifest

  def evaluate(
    self,
    inputs: OperatorEvaluationInput,
  ) -> OperatorEvaluation:
    """Evaluate local internal force and material tangent from compiled meaning."""
    if type(inputs) is not OperatorEvaluationInput:
      msg = "hex8 evaluation requires an exact immutable evaluation input"
      raise TypeError(msg)
    if type(inputs.port_values) is not tuple or len(inputs.port_values) != 1:
      msg = "hex8 evaluation requires exactly one displacement port batch"
      raise TypeError(msg)
    values = inputs.port_values[0].values
    expected = self.header.ports[0].coefficient_map.values.shape
    if (
      values.dtype != np.dtype(np.float64)
      or values.dtype.metadata is not None
      or values.shape != expected
      or not bool(np.isfinite(values).all())
    ):
      msg = "hex8 displacement port values must be a finite metadata-free float64 batch"
      raise TypeError(msg)
    layout = self.header.state_layout
    accepted_state = inputs.accepted_state.values
    if (
      accepted_state.dtype != np.dtype(np.float64)
      or accepted_state.dtype.metadata is not None
      or accepted_state.shape != layout.row_shape
      or not bool(np.isfinite(accepted_state).all())
    ):
      msg = "hex8 accepted state must match the compiled zero-width state layout"
      raise TypeError(msg)
    if inputs.signals or self.header.signal_ports:
      msg = "hex8 model operator does not accept program signal inputs"
      raise ValueError(msg)
    residual_ids = tuple(item.channel_id for item in self.header.residual_channels)
    jacobian_ids = tuple(item.channel_id for item in self.header.jacobian_channels)
    derivative_ids = tuple(
      item.channel_id for item in getattr(self.header, "derivative_channels", ())
    )
    request = inputs.request
    derivative_request = request.derivative_channel_ids
    if (
      type(request) is not ChannelRequest
      or len(set(request.residual_channel_ids)) != len(request.residual_channel_ids)
      or len(set(request.jacobian_channel_ids)) != len(request.jacobian_channel_ids)
      or len(set(derivative_request)) != len(derivative_request)
      or not set(request.residual_channel_ids).issubset(residual_ids)
      or not set(request.jacobian_channel_ids).issubset(jacobian_ids)
      or not set(derivative_request).issubset(derivative_ids)
    ):
      msg = "hex8 evaluation request contains an unavailable or duplicate channel"
      raise ValueError(msg)

    b_matrix = self.payload.normalized_strain_displacement.values
    weights = self.payload.normalized_integration_weights.values
    constitutive = self.payload.constitutive.values
    tangent = self.payload.geometry_scales.values[:, None, None] * np.einsum(
      "ep,epai,ab,epbj->eij",
      weights,
      b_matrix,
      constitutive,
      b_matrix,
      optimize=True,
    )
    residual_values = ()
    if request.residual_channel_ids:
      residual = np.einsum("eij,ej->ei", tangent, values, optimize=True)
      residual_values = (FinalizedArray(residual, dtype=np.float64),)
    jacobian_values = (
      (FinalizedArray(tangent, dtype=np.float64),)
      if request.jacobian_channel_ids
      else ()
    )
    return _new(
      OperatorEvaluation,
      residual_values=residual_values,
      jacobian_values=jacobian_values,
      trial_state=FinalizedArray(accepted_state, dtype=np.float64),
    )


@dataclass(frozen=True, slots=True, eq=False, init=False)
class Q8ThermalPayload(CompilerConstructed):
  quadrature_points: FinalizedArray
  quadrature_weights: FinalizedArray
  shape_values: FinalizedArray
  parent_gradients: FinalizedArray
  geometry_scales: FinalizedArray
  normalized_gradients: FinalizedArray
  normalized_temperature_gradients: FinalizedArray
  normalized_integration_weights: FinalizedArray
  conductivity: FinalizedArray
  material_parameters: FinalizedArray

  def physical_gradients(self) -> FinalizedArray:
    values = (
      self.normalized_gradients.values
      / self.geometry_scales.values[:, None, None, None]
    )
    return FinalizedArray(values, dtype=np.float64)

  def physical_temperature_gradients(self) -> FinalizedArray:
    values = (
      self.normalized_temperature_gradients.values
      / self.geometry_scales.values[:, None, None, None]
    )
    return FinalizedArray(values, dtype=np.float64)

  def physical_integration_weights(self) -> FinalizedArray:
    scales = self.geometry_scales.values[:, None]
    values = self.normalized_integration_weights.values * scales * scales
    return FinalizedArray(values, dtype=np.float64)


@dataclass(frozen=True, slots=True, eq=False, init=False)
class Q8ThermalOperator(CompilerConstructed):
  header: OperatorHeader
  entity_block: IncidenceEntityBlock
  payload: Q8ThermalPayload
  content_manifest: CanonicalManifest

  def evaluate(
    self,
    inputs: OperatorEvaluationInput,
  ) -> OperatorEvaluation:
    """Evaluate local heat flux and conduction tangent from compiled meaning."""
    if type(inputs) is not OperatorEvaluationInput:
      msg = "thermal evaluation requires an exact immutable evaluation input"
      raise TypeError(msg)
    if type(inputs.port_values) is not tuple or len(inputs.port_values) != 1:
      msg = "thermal evaluation requires exactly one temperature port batch"
      raise TypeError(msg)
    values = inputs.port_values[0].values
    expected = self.header.ports[0].coefficient_map.values.shape
    if (
      values.dtype != np.dtype(np.float64)
      or values.dtype.metadata is not None
      or values.shape != expected
      or not bool(np.isfinite(values).all())
    ):
      msg = (
        "thermal temperature port values must be a finite metadata-free float64 batch"
      )
      raise TypeError(msg)
    layout = self.header.state_layout
    accepted_state = inputs.accepted_state.values
    if (
      accepted_state.dtype != np.dtype(np.float64)
      or accepted_state.dtype.metadata is not None
      or accepted_state.shape != layout.row_shape
      or not bool(np.isfinite(accepted_state).all())
    ):
      msg = "thermal accepted state must match the compiled zero-width state layout"
      raise TypeError(msg)
    if inputs.signals or self.header.signal_ports:
      msg = "thermal model operator does not accept program signal inputs"
      raise ValueError(msg)
    residual_ids = tuple(item.channel_id for item in self.header.residual_channels)
    jacobian_ids = tuple(item.channel_id for item in self.header.jacobian_channels)
    derivative_ids = tuple(
      item.channel_id for item in getattr(self.header, "derivative_channels", ())
    )
    request = inputs.request
    derivative_request = request.derivative_channel_ids
    if (
      type(request) is not ChannelRequest
      or len(set(request.residual_channel_ids)) != len(request.residual_channel_ids)
      or len(set(request.jacobian_channel_ids)) != len(request.jacobian_channel_ids)
      or len(set(derivative_request)) != len(derivative_request)
      or not set(request.residual_channel_ids).issubset(residual_ids)
      or not set(request.jacobian_channel_ids).issubset(jacobian_ids)
      or not set(derivative_request).issubset(derivative_ids)
    ):
      msg = "thermal evaluation request contains an unavailable or duplicate channel"
      raise ValueError(msg)

    requested_residuals = set(request.residual_channel_ids)
    requested_jacobians = set(request.jacobian_channel_ids)
    tangent = None
    if (
      "heat-flux" in requested_residuals or "conduction-tangent" in requested_jacobians
    ):
      b_t = self.payload.normalized_temperature_gradients.values
      weights = self.payload.normalized_integration_weights.values
      conduction = self.payload.conductivity.values
      with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        tangent = np.einsum(
          "ep,epai,ab,epbj->eij",
          weights,
          b_t,
          conduction,
          b_t,
          optimize=True,
        )
      if not bool(np.isfinite(tangent).all()):
        msg = "thermal evaluation response is not representable as finite float64"
        raise ValueError(msg)
    residual_values: tuple[FinalizedArray, ...] = ()
    if "heat-flux" in requested_residuals:
      assert tangent is not None
      with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        residual = np.einsum("eij,ej->ei", tangent, values, optimize=True)
      if not bool(np.isfinite(residual).all()):
        msg = "thermal evaluation response is not representable as finite float64"
        raise ValueError(msg)
      residual_values = (FinalizedArray(residual, dtype=np.float64),)
    jacobian_values = (
      (FinalizedArray(tangent, dtype=np.float64),)
      if "conduction-tangent" in requested_jacobians
      else ()
    )
    return _new(
      OperatorEvaluation,
      residual_values=residual_values,
      jacobian_values=jacobian_values,
      trial_state=FinalizedArray(accepted_state, dtype=np.float64),
    )


@dataclass(frozen=True, slots=True, eq=False, init=False)
class Q8ThermoElasticPayload(CompilerConstructed):
  quadrature_points: FinalizedArray
  quadrature_weights: FinalizedArray
  shape_values: FinalizedArray
  parent_gradients: FinalizedArray
  geometry_scales: FinalizedArray
  normalized_gradients: FinalizedArray
  normalized_strain_displacement: FinalizedArray
  normalized_temperature_gradients: FinalizedArray
  normalized_integration_weights: FinalizedArray
  constitutive: FinalizedArray
  thermal_expansion: FinalizedArray
  conductivity: FinalizedArray
  material_parameters: FinalizedArray

  def physical_gradients(self) -> FinalizedArray:
    values = (
      self.normalized_gradients.values
      / self.geometry_scales.values[:, None, None, None]
    )
    return FinalizedArray(values, dtype=np.float64)

  def physical_strain_displacement(self) -> FinalizedArray:
    values = (
      self.normalized_strain_displacement.values
      / self.geometry_scales.values[:, None, None, None]
    )
    return FinalizedArray(values, dtype=np.float64)

  def physical_temperature_gradients(self) -> FinalizedArray:
    values = (
      self.normalized_temperature_gradients.values
      / self.geometry_scales.values[:, None, None, None]
    )
    return FinalizedArray(values, dtype=np.float64)

  def physical_integration_weights(self) -> FinalizedArray:
    scales = self.geometry_scales.values[:, None]
    values = self.normalized_integration_weights.values * scales * scales
    return FinalizedArray(values, dtype=np.float64)


@dataclass(frozen=True, slots=True, eq=False, init=False)
class Q8ThermoElasticOperator(CompilerConstructed):
  header: OperatorHeader
  entity_block: IncidenceEntityBlock
  payload: Q8ThermoElasticPayload
  content_manifest: CanonicalManifest

  def evaluate(
    self,
    inputs: OperatorEvaluationInput,
  ) -> OperatorEvaluation:
    """Evaluate coupled internal force and heat flux from compiled meaning."""
    if type(inputs) is not OperatorEvaluationInput:
      msg = "coupled evaluation requires an exact immutable evaluation input"
      raise TypeError(msg)
    if type(inputs.port_values) is not tuple or len(inputs.port_values) != 2:
      msg = "coupled evaluation requires exactly displacement and temperature batches"
      raise TypeError(msg)
    port_values = []
    for port, port_input in zip(self.header.ports, inputs.port_values, strict=True):
      values = port_input.values
      expected = port.coefficient_map.values.shape
      if (
        values.dtype != np.dtype(np.float64)
        or values.dtype.metadata is not None
        or values.shape != expected
        or not bool(np.isfinite(values).all())
      ):
        msg = "coupled port values must be finite metadata-free float64 batches"
        raise TypeError(msg)
      port_values.append(values)
    displacements, temperatures = port_values
    layout = self.header.state_layout
    accepted_state = inputs.accepted_state.values
    if (
      accepted_state.dtype != np.dtype(np.float64)
      or accepted_state.dtype.metadata is not None
      or accepted_state.shape != layout.row_shape
      or not bool(np.isfinite(accepted_state).all())
    ):
      msg = "coupled accepted state must match the compiled zero-width state layout"
      raise TypeError(msg)
    if inputs.signals or self.header.signal_ports:
      msg = "coupled model operator does not accept program signal inputs"
      raise ValueError(msg)
    residual_ids = tuple(item.channel_id for item in self.header.residual_channels)
    jacobian_ids = tuple(item.channel_id for item in self.header.jacobian_channels)
    derivative_ids = tuple(
      item.channel_id for item in getattr(self.header, "derivative_channels", ())
    )
    request = inputs.request
    derivative_request = request.derivative_channel_ids
    if (
      type(request) is not ChannelRequest
      or len(set(request.residual_channel_ids)) != len(request.residual_channel_ids)
      or len(set(request.jacobian_channel_ids)) != len(request.jacobian_channel_ids)
      or len(set(derivative_request)) != len(derivative_request)
      or not set(request.residual_channel_ids).issubset(residual_ids)
      or not set(request.jacobian_channel_ids).issubset(jacobian_ids)
      or not set(derivative_request).issubset(derivative_ids)
    ):
      msg = "coupled evaluation request contains an unavailable or duplicate channel"
      raise ValueError(msg)

    requested_residuals = set(request.residual_channel_ids)
    requested_jacobians = set(request.jacobian_channel_ids)
    mechanical = (
      "internal-force" in requested_residuals
      or "material-tangent" in requested_jacobians
    )
    coupling = (
      "internal-force" in requested_residuals
      or "thermal-expansion-tangent" in requested_jacobians
    )
    thermal = (
      "heat-flux" in requested_residuals or "conduction-tangent" in requested_jacobians
    )
    b_matrix = self.payload.normalized_strain_displacement.values
    b_t = self.payload.normalized_temperature_gradients.values
    weights = self.payload.normalized_integration_weights.values
    constitutive = self.payload.constitutive.values
    tangent_uu = tangent_ut = tangent_tt = None
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
      if mechanical:
        tangent_uu = np.einsum(
          "ep,epai,ab,epbj->eij",
          weights,
          b_matrix,
          constitutive,
          b_matrix,
          optimize=True,
        )
      if coupling:
        dilatation = constitutive @ self.payload.thermal_expansion.values
        tangent_ut = self.payload.geometry_scales.values[:, None, None] * np.einsum(
          "ep,epai,a,pb->eib",
          weights,
          b_matrix,
          dilatation,
          self.payload.shape_values.values,
          optimize=True,
        )
      if thermal:
        tangent_tt = np.einsum(
          "ep,epai,ab,epbj->eij",
          weights,
          b_t,
          self.payload.conductivity.values,
          b_t,
          optimize=True,
        )
    computed = tuple(
      block for block in (tangent_uu, tangent_ut, tangent_tt) if block is not None
    )
    if any(not bool(np.isfinite(block).all()) for block in computed):
      msg = "coupled evaluation response is not representable as finite float64"
      raise ValueError(msg)

    residual_u = residual_t = None
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
      if "internal-force" in requested_residuals:
        assert tangent_uu is not None and tangent_ut is not None
        residual_u = np.einsum(
          "eij,ej->ei", tangent_uu, displacements, optimize=True
        ) - np.einsum("eij,ej->ei", tangent_ut, temperatures, optimize=True)
      if "heat-flux" in requested_residuals:
        assert tangent_tt is not None
        residual_t = np.einsum("eij,ej->ei", tangent_tt, temperatures, optimize=True)
    residuals = tuple(block for block in (residual_u, residual_t) if block is not None)
    if any(not bool(np.isfinite(block).all()) for block in residuals):
      msg = "coupled evaluation response is not representable as finite float64"
      raise ValueError(msg)

    residual_arrays = {"internal-force": residual_u, "heat-flux": residual_t}
    residual_values = tuple(
      FinalizedArray(residual_arrays[item.channel_id], dtype=np.float64)
      for item in self.header.residual_channels
      if item.channel_id in requested_residuals
    )
    jacobian_arrays = {
      "material-tangent": tangent_uu,
      "thermal-expansion-tangent": (None if tangent_ut is None else -tangent_ut),
      "conduction-tangent": tangent_tt,
    }
    jacobian_values = tuple(
      FinalizedArray(jacobian_arrays[item.channel_id], dtype=np.float64)
      for item in self.header.jacobian_channels
      if item.channel_id in requested_jacobians
    )
    return _new(
      OperatorEvaluation,
      residual_values=residual_values,
      jacobian_values=jacobian_values,
      trial_state=FinalizedArray(accepted_state, dtype=np.float64),
    )


@dataclass(frozen=True, slots=True, eq=False, init=False)
class Q8StatefulContinuumPayload(CompilerConstructed):
  quadrature_points: FinalizedArray
  quadrature_weights: FinalizedArray
  shape_values: FinalizedArray
  parent_gradients: FinalizedArray
  geometry_scales: FinalizedArray
  normalized_gradients: FinalizedArray
  normalized_strain_displacement: FinalizedArray
  normalized_integration_weights: FinalizedArray
  calibration: FinalizedArray
  material_parameters: FinalizedArray

  def physical_gradients(self) -> FinalizedArray:
    values = (
      self.normalized_gradients.values
      / self.geometry_scales.values[:, None, None, None]
    )
    return FinalizedArray(values, dtype=np.float64)

  def physical_strain_displacement(self) -> FinalizedArray:
    values = (
      self.normalized_strain_displacement.values
      / self.geometry_scales.values[:, None, None, None]
    )
    return FinalizedArray(values, dtype=np.float64)

  def physical_integration_weights(self) -> FinalizedArray:
    scales = self.geometry_scales.values[:, None]
    values = self.normalized_integration_weights.values * scales * scales
    return FinalizedArray(values, dtype=np.float64)


def _validated_stateful_kernel_result(
  result: object,
  *,
  entity_count: int,
  row_width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, EvaluationStatus]:
  if type(result) is not StatefulContinuumKernelResult:
    msg = "stateful kernels must return an exact StatefulContinuumKernelResult"
    raise TypeError(msg)
  arrays = (result.stresses, result.tangents, result.trial_rows)
  shapes = ((entity_count, 6), (entity_count, 6, 6), (entity_count, row_width))
  for array, shape in zip(arrays, shapes, strict=True):
    if array.shape != shape:
      msg = "stateful kernel arrays must match the declared batched shapes"
      raise TypeError(msg)
  if result.status is EvaluationStatus.OK and not all(
    bool(np.isfinite(array).all()) for array in arrays
  ):
    msg = "stateful kernel arrays must be finite for a successful evaluation"
    raise TypeError(msg)
  return result.stresses, result.tangents, result.trial_rows, result.status


def _validated_stateful_param_kernel_result(
  result: object,
  *,
  entity_count: int,
  row_width: int,
  parameter_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, EvaluationStatus]:
  """Validate the derivative-kernel response, adding the derivative columns.

  ``param_derivatives`` must stack exactly ``parameter_count`` declared
  columns of shape ``(entity_count, 6)`` on a successful evaluation, and is
  discarded (may be ``None``) on a rejected one — mirroring the primal
  arrays, which the rejection path also discards.
  """
  stresses, tangents, trial_rows, status = _validated_stateful_kernel_result(
    result,
    entity_count=entity_count,
    row_width=row_width,
  )
  assert type(result) is StatefulContinuumKernelResult
  param_derivatives = result.param_derivatives
  if status is EvaluationStatus.OK:
    if param_derivatives is None or param_derivatives.shape != (
      parameter_count,
      entity_count,
      6,
    ):
      msg = (
        "stateful kernel param_derivatives must stack the declared parameter "
        "columns in (parameter_count, entity_count, 6) layout"
      )
      raise TypeError(msg)
    if not bool(np.isfinite(param_derivatives).all()):
      msg = (
        "stateful kernel param_derivatives must be finite for a successful evaluation"
      )
      raise TypeError(msg)
  return stresses, tangents, trial_rows, param_derivatives, status


def _signal_scalar(value: np.ndarray, label: str) -> None:
  if (
    value.dtype != np.dtype(np.float64)
    or value.dtype.metadata is not None
    or value.shape != (1,)
    or not bool(np.isfinite(value).all())
  ):
    msg = f"stateful {label} must be a finite metadata-free float64 scalar"
    raise TypeError(msg)


def _validated_stateful_signals(
  ports: tuple[SignalPortBinding, ...],
  signals: tuple[ProgramSignalInput, ...],
) -> tuple[StatefulContinuumSignalInput, ...]:
  """Bind evaluation signal inputs exactly onto the declared signal ports.

  Every declared port must be bound exactly once and every input must name a
  declared port; derivative channels must follow the port's declared
  ``derivative_coordinate_ids`` order. The forwarded carriers preserve
  declaration order so kernels read a deterministic layout.
  """
  by_port: dict[str, ProgramSignalInput] = {}
  for signal in signals:
    if signal.port_id in by_port:
      msg = "stateful evaluation received a duplicate program signal port"
      raise ValueError(msg)
    by_port[signal.port_id] = signal
  declared = {port.port_id: port for port in ports}
  for port_id in by_port:
    if port_id not in declared:
      msg = "stateful evaluation received an undeclared program signal port"
      raise ValueError(msg)
  kernel_signals: list[StatefulContinuumSignalInput] = []
  for port in ports:
    signal = by_port.get(port.port_id)
    if signal is None:
      msg = "stateful evaluation is missing a declared program signal port"
      raise ValueError(msg)
    values = signal.values.values
    _signal_scalar(values, "signal values")
    if (
      tuple(item.coordinate_id for item in signal.derivatives)
      != port.derivative_coordinate_ids
    ):
      msg = "stateful signal derivatives must match the declared coordinates"
      raise ValueError(msg)
    kernel_derivatives: list[StatefulContinuumSignalDerivative] = []
    for item in signal.derivatives:
      derivative_values = item.values.values
      _signal_scalar(derivative_values, "signal derivative values")
      kernel_derivatives.append(
        StatefulContinuumSignalDerivative(
          coordinate_id=item.coordinate_id,
          values=derivative_values,
        )
      )
    kernel_signals.append(
      StatefulContinuumSignalInput(
        port_id=port.port_id,
        values=values,
        derivatives=tuple(kernel_derivatives),
      )
    )
  return tuple(kernel_signals)


@dataclass(frozen=True, slots=True, eq=False, init=False)
class Q8StatefulContinuumOperator(CompilerConstructed):
  """Kernel-driven stateful small-strain continuum operator (v2 descriptor ABI).

  Evaluation is pure: total displacements and accepted state rows in, internal
  force, algorithmic tangent, and trial rows out. Port strains are embedded in
  the law's 6-Voigt internal order as ``[xx, yy, 0, 0, 0, xy]`` (plane strain);
  stresses and tangents truncate back to the ``[xx, yy, xy]`` port convention.
  An operator compiled with declared signal ports accepts exactly one bound
  ``ProgramSignalInput`` per port and forwards the validated scalar values and
  derivative channels to the kernel's fourth positional argument; an operator
  compiled without ports rejects every signal input.

  An operator whose binding provides the optional ``param_derivative_kernel``
  member carries ``ParameterBinding`` and ``ResidualDerivativeChannel``
  declarations on its header and answers derivative channel requests through
  that kernel: the binding's derivative twin returns the primal response
  bitwise identical to ``kernel`` plus the exact per-IP
  ``d(stress)/d(parameter)`` columns, which the operator assembles into
  per-element residual derivatives through the same internal-force expression
  as the residual itself. Derivative values are pure functions of the
  accepted state, the port values, and the parameters — the committed record
  never sees trial-state increments.
  """

  header: OperatorHeader
  entity_block: IncidenceEntityBlock
  payload: Q8StatefulContinuumPayload
  content_manifest: CanonicalManifest
  kernel: StatefulContinuumKernel | StatefulContinuumSignalKernel
  param_kernel: StatefulContinuumKernel | StatefulContinuumSignalKernel | None

  def evaluate(
    self,
    inputs: OperatorEvaluationInput,
  ) -> OperatorEvaluation:
    """Evaluate stateful internal force and tangent from compiled meaning."""
    if type(inputs) is not OperatorEvaluationInput:
      msg = "stateful evaluation requires an exact immutable evaluation input"
      raise TypeError(msg)
    if type(inputs.port_values) is not tuple or len(inputs.port_values) != 1:
      msg = "stateful evaluation requires exactly one displacement port batch"
      raise TypeError(msg)
    values = inputs.port_values[0].values
    expected = self.header.ports[0].coefficient_map.values.shape
    if (
      values.dtype != np.dtype(np.float64)
      or values.dtype.metadata is not None
      or values.shape != expected
      or not bool(np.isfinite(values).all())
    ):
      msg = "stateful displacement port values must be a finite float64 batch"
      raise TypeError(msg)
    layout = self.header.state_layout
    accepted_state = inputs.accepted_state.values
    if (
      accepted_state.dtype != np.dtype(np.float64)
      or accepted_state.dtype.metadata is not None
      or accepted_state.shape != layout.row_shape
      or not bool(np.isfinite(accepted_state).all())
    ):
      msg = "stateful accepted state must match the compiled state layout"
      raise TypeError(msg)
    signal_ports = self.header.signal_ports
    if signal_ports:
      kernel_signals = _validated_stateful_signals(signal_ports, inputs.signals)
    elif inputs.signals:
      msg = "stateful model operator does not accept program signal inputs"
      raise ValueError(msg)
    else:
      kernel_signals = ()
    residual_ids = tuple(item.channel_id for item in self.header.residual_channels)
    jacobian_ids = tuple(item.channel_id for item in self.header.jacobian_channels)
    parameters: tuple[ParameterBinding, ...] = getattr(self.header, "parameters", ())
    derivative_channels: tuple[ResidualDerivativeChannel, ...] = getattr(
      self.header, "derivative_channels", ()
    )
    derivative_ids = tuple(item.channel_id for item in derivative_channels)
    request = inputs.request
    derivative_request = request.derivative_channel_ids
    if (
      type(request) is not ChannelRequest
      or len(set(request.residual_channel_ids)) != len(request.residual_channel_ids)
      or len(set(request.jacobian_channel_ids)) != len(request.jacobian_channel_ids)
      or len(set(derivative_request)) != len(derivative_request)
      or not set(request.residual_channel_ids).issubset(residual_ids)
      or not set(request.jacobian_channel_ids).issubset(jacobian_ids)
      or not set(derivative_request).issubset(derivative_ids)
    ):
      msg = "stateful evaluation request contains an unavailable or duplicate channel"
      raise ValueError(msg)

    scales = self.payload.geometry_scales.values
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
      b_matrix = (
        self.payload.normalized_strain_displacement.values / scales[:, None, None, None]
      )
      weights = (
        self.payload.normalized_integration_weights.values
        * scales[:, None]
        * scales[:, None]
      )
      strain_voigt = np.einsum("epai,ei->epa", b_matrix, values, optimize=True)
    strains = np.zeros((layout.entity_count, 6), dtype=np.float64)
    flat = strain_voigt.reshape(layout.entity_count, 3)
    strains[:, 0] = flat[:, 0]
    strains[:, 1] = flat[:, 1]
    strains[:, 5] = flat[:, 2]

    requested_derivatives = set(derivative_request)
    if requested_derivatives:
      active_kernel = self.param_kernel
      if active_kernel is None:
        msg = (
          "stateful operator declares derivative channels without a derivative kernel"
        )
        raise ValueError(msg)
    else:
      active_kernel = self.kernel
    if signal_ports:
      kernel_result = active_kernel(
        strains,
        accepted_state,
        self.payload.calibration.values,
        kernel_signals,
      )
    else:
      kernel_result = active_kernel(
        strains,
        accepted_state,
        self.payload.calibration.values,
      )
    if requested_derivatives:
      (
        stresses,
        tangents,
        trial_rows,
        param_derivatives,
        status,
      ) = _validated_stateful_param_kernel_result(
        kernel_result,
        entity_count=layout.entity_count,
        row_width=layout.row_width,
        parameter_count=len(parameters),
      )
    else:
      stresses, tangents, trial_rows, status = _validated_stateful_kernel_result(
        kernel_result,
        entity_count=layout.entity_count,
        row_width=layout.row_width,
      )
      param_derivatives = None
    if status is not EvaluationStatus.OK:
      return _new(
        OperatorEvaluation,
        residual_values=(),
        jacobian_values=(),
        trial_state=FinalizedArray(accepted_state, dtype=np.float64),
        status=status,
      )

    element_count = values.shape[0]
    stress3 = stresses.reshape(element_count, _POINT_COUNT, 6)[:, :, [0, 1, 5]]
    tangent6 = tangents.reshape(element_count, _POINT_COUNT, 6, 6)
    tangent3 = tangent6[:, :, [0, 1, 5], :][:, :, :, [0, 1, 5]]
    requested_residuals = set(request.residual_channel_ids)
    requested_jacobians = set(request.jacobian_channel_ids)
    residual = tangent = None
    derivative_blocks: list[np.ndarray] = []
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
      if "internal-force" in requested_residuals:
        residual = continuum_internal_force_batched(b_matrix, weights, stress3)
      if "material-tangent" in requested_jacobians:
        tangent = np.einsum(
          "ep,epai,epab,epbj->eij",
          weights,
          b_matrix,
          tangent3,
          b_matrix,
          optimize=True,
        )
      if requested_derivatives:
        assert param_derivatives is not None
        parameter_ids = tuple(item.parameter_id for item in parameters)
        derivative_blocks.extend(
          continuum_internal_force_batched(
            b_matrix,
            weights,
            param_derivatives[parameter_ids.index(channel.parameter_id)].reshape(
              element_count, _POINT_COUNT, 6
            )[:, :, [0, 1, 5]],
          )
          for channel in derivative_channels
          if channel.channel_id in requested_derivatives
        )
    for block in (residual, tangent, *derivative_blocks):
      if block is not None and not bool(np.isfinite(block).all()):
        msg = "stateful evaluation response is not representable as finite float64"
        raise ValueError(msg)
    residual_values = (
      (FinalizedArray(residual, dtype=np.float64),) if residual is not None else ()
    )
    jacobian_values = (
      (FinalizedArray(tangent, dtype=np.float64),) if tangent is not None else ()
    )
    return _new(
      OperatorEvaluation,
      residual_values=residual_values,
      jacobian_values=jacobian_values,
      trial_state=FinalizedArray(trial_rows, dtype=np.float64),
      status=status,
      derivative_values=tuple(
        FinalizedArray(block, dtype=np.float64) for block in derivative_blocks
      ),
    )


def q8_descriptor_metadata(kind: str, name: str) -> dict[str, object]:
  """Return the exact semantic metadata for one selected Q8 implementation."""
  key = kind, name
  if key == Q8_TOPOLOGY_KEY:
    return {
      "schema": "pyfem-v3-topology-descriptor-v1",
      "reference_topology": "quadrilateral",
      "parent_dimension": 2,
      "embedding_dimension": 2,
      "node_count": 8,
      "parent_coordinates": ["xi", "eta"],
      "local_node_parent_coordinates": [
        [-1.0, -1.0],
        [0.0, -1.0],
        [1.0, -1.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [-1.0, 1.0],
        [-1.0, 0.0],
      ],
      "shape_value_layout": ["point", "node"],
      "parent_gradient_layout": ["point", "node", "parent_coordinate"],
    }
  if key == Q8_QUADRATURE_KEY:
    return {
      "schema": "pyfem-v3-quadrature-descriptor-v1",
      "family": "tensor-gauss-legendre",
      "parent_coordinates": ["xi", "eta"],
      "orders": [3, 3],
      "point_count": 9,
      "binding_arguments": [3],
    }
  if key == Q8_FORMULATION_KEY:
    return {
      "schema": "pyfem-v3-formulation-descriptor-v1",
      "field_quantity": "displacement",
      "field_location": "node",
      "field_components": ["x", "y"],
      "dofs_per_node": 2,
      "kinematic_regime": "small-strain",
      "strain_measure": "infinitesimal",
      "strain_voigt_order": ["xx", "yy", "xy"],
      "shear_convention": "engineering",
      "formulation_history_width": 0,
      "tangent_contribution": "material",
      "tangent_symmetry": "symmetric",
    }
  if key == Q8_MATERIAL_KEY:
    return {
      "schema": "pyfem-v3-material-descriptor-v1",
      "law": "linear-elastic",
      "stress_state": "plane-stress",
      "parameter_names": ["youngs_modulus", "poisson_ratio"],
      "parameter_dtype": "float64",
      "stress_voigt_order": ["xx", "yy", "xy"],
      "strain_shear_convention": "engineering",
      "material_history_width": 0,
      "tangent_class": "constant-symmetric",
    }
  msg = "no Q8 descriptor metadata exists for that exact registry key"
  raise KeyError(msg)


def thermal_descriptor_metadata(kind: str, name: str) -> dict[str, object]:
  """Return the exact semantic metadata for one selected thermal implementation."""
  key = kind, name
  if key == THERMAL_FORMULATION_KEY:
    return {
      "schema": "pyfem-v3-formulation-descriptor-v1",
      "field_quantity": "temperature",
      "field_location": "node",
      "field_components": "scalar",
      "kinematic_regime": "steady-diffusion",
      "thermal_measure": "temperature-gradient",
      "formulation_history_width": 0,
      "tangent_contribution": "material",
      "tangent_symmetry": "symmetric",
    }
  if key == THERMAL_MATERIAL_KEY:
    return {
      "schema": "pyfem-v3-material-descriptor-v1",
      "law": "linear-isotropic-conduction",
      "parameter_names": ["conductivity"],
      "parameter_dtype": "float64",
      "material_history_width": 0,
      "tangent_class": "constant-symmetric",
    }
  msg = "no thermal descriptor metadata exists for that exact registry key"
  raise KeyError(msg)


def thermo_elastic_descriptor_metadata(kind: str, name: str) -> dict[str, object]:
  """Return the exact semantic metadata for one selected coupled implementation."""
  key = kind, name
  if key == THERMO_FORMULATION_KEY:
    return {
      "schema": "pyfem-v3-formulation-descriptor-v1",
      "field_quantities": ["displacement", "temperature"],
      "field_location": "node",
      "field_signatures": {
        "displacement": ["x", "y"],
        "temperature": "scalar",
      },
      "kinematic_regime": "small-strain",
      "strain_measure": "infinitesimal",
      "strain_voigt_order": ["xx", "yy", "xy"],
      "shear_convention": "engineering",
      "thermal_measure": "temperature-gradient",
      "coupling": "thermo-elastic-dilatation",
      "formulation_history_width": 0,
      "tangent_contribution": "material",
      "tangent_symmetry": "nonsymmetric",
    }
  if key == THERMO_MATERIAL_KEY:
    return {
      "schema": "pyfem-v3-material-descriptor-v1",
      "law": "linear-thermo-elastic",
      "stress_state": "plane-stress",
      "parameter_names": [
        "youngs_modulus",
        "poisson_ratio",
        "thermal_expansion",
        "conductivity",
      ],
      "parameter_dtype": "float64",
      "stress_voigt_order": ["xx", "yy", "xy"],
      "strain_shear_convention": "engineering",
      "material_history_width": 0,
      "tangent_class": "constant-nonsymmetric-coupled",
    }
  msg = "no thermo-elastic descriptor metadata exists for that exact registry key"
  raise KeyError(msg)


def _qualified_descriptor_metadata(key: RegistryKey) -> dict[str, object]:
  if key in (THERMAL_FORMULATION_KEY, THERMAL_MATERIAL_KEY):
    return thermal_descriptor_metadata(*key)
  if key in (THERMO_FORMULATION_KEY, THERMO_MATERIAL_KEY):
    return thermo_elastic_descriptor_metadata(*key)
  if key in (TL_FORMULATION_KEY, TL_MATERIAL_KEY):
    return finite_strain_descriptor_metadata(*key)
  if key == PLASTIC_MATERIAL_KEY:
    return isotropic_hardening_plasticity_metadata()
  if key == DAMAGE_MATERIAL_KEY:
    return plane_strain_damage_metadata()
  if key == VISCOELASTIC_MATERIAL_KEY:
    return prony_viscoelasticity_metadata()
  if key == VISCOPLASTIC_MATERIAL_KEY:
    return perzyna_viscoplasticity_metadata()
  if key == SOVS_MATERIAL_KEY:
    return skorohod_olevsky_metadata()
  if key in _BREADTH_DESCRIPTOR_KEYS:
    return breadth_descriptor_metadata(*key)
  return q8_descriptor_metadata(*key)


def q8_reference_registry() -> dict[RegistryKey, RegistryDescriptor]:
  """Build a fresh injectable registry for the qualified Q8 convention."""
  bindings = {
    Q8_TOPOLOGY_KEY: ("pyfem-v3-serendipity-quad8-v1", serendipity_quad8),
    Q8_QUADRATURE_KEY: (
      "pyfem-v3-gauss-tensor-product-2d-order-3-v1",
      gauss_tensor_product_2d,
    ),
    Q8_FORMULATION_KEY: (
      "pyfem-v3-small-strain-engineering-shear-v1",
      strain_displacement,
    ),
    Q8_MATERIAL_KEY: (
      "pyfem-v3-plane-stress-linear-elastic-v1",
      plane_stress_matrix,
    ),
  }
  descriptors = tuple(
    RegistryDescriptor(
      kind=key[0],
      name=key[1],
      version="1",
      implementation_id=implementation_id,
      metadata=q8_descriptor_metadata(*key),
      binding=binding,
    )
    for key, (implementation_id, binding) in bindings.items()
  )
  return {descriptor.key: descriptor for descriptor in descriptors}


def thermal_gradient_map(gradients: np.ndarray) -> np.ndarray:
  """Reference steady thermal kinematics: temperature-gradient operator rows."""
  return np.ascontiguousarray(np.swapaxes(gradients, 2, 3))


def linear_thermal_conductor(conductivity: float) -> np.ndarray:
  """Reference isotropic conduction law: conductivity as a diagonal 2x2 matrix."""
  return np.array(
    [[conductivity, 0.0], [0.0, conductivity]],
    dtype=np.float64,
  )


def thermo_elastic_kinematics(
  gradients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  """Reference coupled kinematics: engineering-shear and temperature gradients."""
  return strain_displacement(gradients), thermal_gradient_map(gradients)


def linear_thermo_elastic(
  youngs_modulus: float,
  poisson_ratio: float,
  thermal_expansion: float,
  conductivity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Reference coupled law: plane-stress matrix, thermal vector, conductivity."""
  matrix = plane_stress_matrix(youngs_modulus, poisson_ratio)
  expansion = np.array(
    [thermal_expansion, thermal_expansion, 0.0],
    dtype=np.float64,
  )
  return matrix, expansion, linear_thermal_conductor(conductivity)


def thermal_reference_registry() -> dict[RegistryKey, RegistryDescriptor]:
  """Build a fresh injectable registry for the qualified thermal convention."""
  bindings = {
    Q8_TOPOLOGY_KEY: ("pyfem-v3-serendipity-quad8-v1", serendipity_quad8),
    Q8_QUADRATURE_KEY: (
      "pyfem-v3-gauss-tensor-product-2d-order-3-v1",
      gauss_tensor_product_2d,
    ),
    THERMAL_FORMULATION_KEY: (
      "pyfem-v3-steady-temperature-gradient-v1",
      thermal_gradient_map,
    ),
    THERMAL_MATERIAL_KEY: (
      "pyfem-v3-linear-isotropic-conductor-v1",
      linear_thermal_conductor,
    ),
  }
  descriptors = tuple(
    RegistryDescriptor(
      kind=key[0],
      name=key[1],
      version="1",
      implementation_id=implementation_id,
      metadata=_qualified_descriptor_metadata(key),
      binding=binding,
    )
    for key, (implementation_id, binding) in bindings.items()
  )
  return {descriptor.key: descriptor for descriptor in descriptors}


def thermo_elastic_reference_registry() -> dict[RegistryKey, RegistryDescriptor]:
  """Build a fresh injectable registry for the qualified coupled convention."""
  bindings = {
    Q8_TOPOLOGY_KEY: ("pyfem-v3-serendipity-quad8-v1", serendipity_quad8),
    Q8_QUADRATURE_KEY: (
      "pyfem-v3-gauss-tensor-product-2d-order-3-v1",
      gauss_tensor_product_2d,
    ),
    THERMO_FORMULATION_KEY: (
      "pyfem-v3-thermo-elastic-kinematics-v1",
      thermo_elastic_kinematics,
    ),
    THERMO_MATERIAL_KEY: (
      "pyfem-v3-linear-thermo-elastic-v1",
      linear_thermo_elastic,
    ),
  }
  descriptors = tuple(
    RegistryDescriptor(
      kind=key[0],
      name=key[1],
      version="1",
      implementation_id=implementation_id,
      metadata=_qualified_descriptor_metadata(key),
      binding=binding,
    )
    for key, (implementation_id, binding) in bindings.items()
  )
  return {descriptor.key: descriptor for descriptor in descriptors}


def plasticity_reference_registry() -> dict[RegistryKey, RegistryDescriptor]:
  """Build the Q8 reference registry plus the first stateful law descriptor."""
  registry = dict(q8_reference_registry())
  descriptor = RegistryDescriptor(
    kind=PLASTIC_MATERIAL_KEY[0],
    name=PLASTIC_MATERIAL_KEY[1],
    version="1",
    implementation_id="pyfem-v3-isotropic-hardening-plasticity-v1",
    metadata=isotropic_hardening_plasticity_metadata(),
    binding=ISOTROPIC_HARDENING_PLASTICITY_BINDING,
  )
  registry[descriptor.key] = descriptor
  return registry


def damage_reference_registry() -> dict[RegistryKey, RegistryDescriptor]:
  """Build the Q8 reference registry plus the second stateful law descriptor.

  The plane-strain damage law is the first ``algorithmic-nonsymmetric``
  tangent-class witness of the frozen v2 descriptor ABI: its descriptor pins
  the qualified damage convention, and the compiled operator's Jacobian
  channel carries ``linear=False, symmetric=False``.
  """
  registry = dict(q8_reference_registry())
  descriptor = RegistryDescriptor(
    kind=DAMAGE_MATERIAL_KEY[0],
    name=DAMAGE_MATERIAL_KEY[1],
    version="1",
    implementation_id="pyfem-v3-plane-strain-damage-v1",
    metadata=plane_strain_damage_metadata(),
    binding=PLANE_STRAIN_DAMAGE_BINDING,
  )
  registry[descriptor.key] = descriptor
  return registry


def viscoelasticity_reference_registry() -> dict[RegistryKey, RegistryDescriptor]:
  """Build the Q8 reference registry plus the first signal-consuming law.

  The Prony-series viscoelastic law is the first descriptor declaring the
  optional ``signal_ports`` field on a production law: its compiled operator
  carries one ``SignalPortBinding`` for the identity time port, and its
  ``eps_i`` state slot resolves through the parameterized-width declaration
  ``{"parameter": "prony_term_count", "scale": 6}`` — the frozen v2 schema
  covers both, so no contracts change accompanies this registration.
  """
  registry = dict(q8_reference_registry())
  descriptor = RegistryDescriptor(
    kind=VISCOELASTIC_MATERIAL_KEY[0],
    name=VISCOELASTIC_MATERIAL_KEY[1],
    version="1",
    implementation_id="pyfem-v3-prony-viscoelasticity-v1",
    metadata=prony_viscoelasticity_metadata(),
    binding=PRONY_VISCOELASTICITY_BINDING,
  )
  registry[descriptor.key] = descriptor
  return registry


def viscoplasticity_reference_registry() -> dict[RegistryKey, RegistryDescriptor]:
  """Build the Q8 reference registry plus the rate-gated J2 law descriptor.

  The Perzyna-branded viscoplastic law (a rate-INDEPENDENT J2 map with a
  ``dtime > 0`` gate — see the kernel module docstring) declares the identity
  time port: its compiled operator carries one ``SignalPortBinding``, and its
  Jacobian channel is ``linear=False, symmetric=True`` per the declared
  ``algorithmic-symmetric`` class. The frozen v2 schema covers the
  declaration, so no contracts change accompanies this registration.
  """
  registry = dict(q8_reference_registry())
  descriptor = RegistryDescriptor(
    kind=VISCOPLASTIC_MATERIAL_KEY[0],
    name=VISCOPLASTIC_MATERIAL_KEY[1],
    version="1",
    implementation_id="pyfem-v3-perzyna-viscoplasticity-v1",
    metadata=perzyna_viscoplasticity_metadata(),
    binding=PERZYNA_VISCOPLASTICITY_BINDING,
  )
  registry[descriptor.key] = descriptor
  return registry


def sovs_reference_registry() -> dict[RegistryKey, RegistryDescriptor]:
  """Build the Q8 reference registry plus the sintering law descriptor.

  The Skorohod-Olevsky explicit viscous-sintering law declares the identity
  time port and binds the first parameter-dependent initial state of the
  stateful family (``rho = rho0`` in every row) through the existing
  initial-state binding path — the frozen v2 schema covers both, so no
  contracts change accompanies this registration. Its Jacobian channel is
  ``linear=False, symmetric=True`` per the declared ``algorithmic-symmetric``
  class.
  """
  registry = dict(q8_reference_registry())
  descriptor = RegistryDescriptor(
    kind=SOVS_MATERIAL_KEY[0],
    name=SOVS_MATERIAL_KEY[1],
    version="1",
    implementation_id="pyfem-v3-skorohod-olevsky-v1",
    metadata=skorohod_olevsky_metadata(),
    binding=SKOROHOD_OLEVSKY_BINDING,
  )
  registry[descriptor.key] = descriptor
  return registry


def select_model(spec: ModelSpec) -> ContinuumSelection:
  """Validate the concrete authored slice after the sole normalization pass."""
  for field in spec.fields:
    if field.location != "node":
      _fail(
        "unsupported-space-support",
        "the direct Q8 slice cannot allocate a non-node field",
        field.source,
      )
  cells_by_key = {
    (block.id, cell.id): cell for block in spec.mesh.cell_blocks for cell in block.cells
  }
  counts = {key: 0 for key in cells_by_key}
  for region in spec.regions:
    for cell_ref in region.cell_refs:
      key = cell_ref.block_id, cell_ref.cell_id
      if key in counts:
        counts[key] += 1
  for key in sorted(counts, key=_sort_key):
    count = counts[key]
    cell = cells_by_key[key]
    rendered = render_diagnostic_value(key)
    if count == 0:
      _fail(
        "incomplete-cell-membership",
        f"source cell {rendered} does not belong to a compiled region",
        cell.source,
      )
    if count > 1:
      _fail(
        "multiple-cell-membership",
        f"source cell {rendered} belongs to more than one compiled region",
        cell.source,
      )
  fields_by_id = {field.id: field for field in spec.fields}
  materials_by_id = {material.id: material for material in spec.materials}
  blocks_by_id = {block.id: block for block in spec.mesh.cell_blocks}
  block_owners: dict[SpecId, RegionSpec] = {}
  selections = tuple(
    _select_region(
      region,
      fields_by_id,
      materials_by_id,
      blocks_by_id,
      cells_by_key,
      block_owners,
    )
    for region in sorted(spec.regions, key=lambda item: _sort_key(item.id))
  )
  referenced_fields = {
    field_id for region in spec.regions for field_id in region.field_ids
  }
  for field in sorted(spec.fields, key=lambda item: _sort_key(item.id)):
    if field.id not in referenced_fields:
      _fail(
        "unreferenced-field-declaration",
        f"declared field {render_diagnostic_value(field.id)} is not referenced "
        "by a compiled region",
        field.source,
      )
  referenced_materials = {region.material_id for region in spec.regions}
  for material in sorted(spec.materials, key=lambda item: _sort_key(item.id)):
    if material.id not in referenced_materials:
      _fail(
        "unreferenced-material-declaration",
        f"declared material {render_diagnostic_value(material.id)} is not "
        "referenced by a compiled region",
        material.source,
      )
  return ContinuumSelection(regions=selections)


def _select_region(
  region: RegionSpec,
  fields_by_id: dict[SpecId, FieldSpec],
  materials_by_id: dict[SpecId, MaterialSpec],
  blocks_by_id: dict[SpecId, CellBlockSpec],
  cells_by_key: dict[tuple[SpecId, SpecId], CellSpec],
  block_owners: dict[SpecId, RegionSpec],
) -> RegionSelection:
  block_ids = {cell_ref.block_id for cell_ref in region.cell_refs}
  if len(block_ids) != 1:
    _fail(
      "unsupported-region-cell-block-span",
      "each compiled region draws its cells from exactly one cell block",
      region.source,
    )
  block_id = next(iter(block_ids))
  owner = block_owners.get(block_id)
  if owner is not None:
    _fail(
      "shared-cell-block",
      f"cell block {render_diagnostic_value(block_id)} already feeds region "
      f"{render_diagnostic_value(owner.id)}",
      region.source,
    )
  block_owners[block_id] = region
  block = blocks_by_id[block_id]
  profile = _CONTINUUM_GEOMETRIES.get(block.geometry_interpolation)
  if (
    profile is None
    or block.reference_topology != profile.reference_topology
    or block.topological_dimension != profile.topological_dimension
    or block.embedding_dimension != profile.embedding_dimension
  ):
    _fail(
      "incompatible-cell-block",
      "unsupported or inconsistent continuum cell-block geometry",
      block.source,
    )
  for cell in block.cells:
    if len(cell.node_ids) != profile.node_count:
      if profile is _Q8_GEOMETRY:
        _fail(
          "invalid-q8-arity",
          f"Q8 cell {render_diagnostic_value(cell.id)} must reference eight nodes",
          cell.source,
        )
      _fail(
        "invalid-cell-arity",
        f"{profile.geometry_interpolation} cell "
        f"{render_diagnostic_value(cell.id)} must reference "
        f"{profile.node_count} nodes",
        cell.source,
      )
  contract = _FORMULATION_CONTRACTS.get(region.formulation)
  if contract is None:
    _fail(
      "incompatible-formulation",
      "unsupported continuum formulation",
      region.source,
    )
  if profile.geometry_interpolation not in _FORMULATION_GEOMETRIES[region.formulation]:
    _fail(
      "incompatible-cell-block",
      f"the {region.formulation} formulation does not compile on the "
      f"{profile.geometry_interpolation} geometry",
      block.source,
    )
  if len(region.field_ids) != len(contract) or any(
    field_id not in fields_by_id for field_id in region.field_ids
  ):
    _fail(
      "incompatible-region-field-signature",
      f"the {region.formulation} region must reference exactly "
      f"{len(contract)} declared field(s) in role order",
      region.source,
    )
  fields = tuple(fields_by_id[field_id] for field_id in region.field_ids)
  for field, (role, signature) in zip(fields, contract, strict=True):
    if signature == "scalar":
      if len(field.components) != 1:
        _fail(
          "incompatible-field-signature",
          f"the {role} field must carry exactly one component",
          field.source,
        )
      continue
    expected_components = (
      profile.field_components
      if region.formulation == Q8_FORMULATION_KEY[1]
      else signature
    )
    if field.components != expected_components:
      rendered = (
        "('x', 'y')" if expected_components == ("x", "y") else repr(expected_components)
      )
      _fail(
        "incompatible-field-signature",
        f"the {role} field requires physical components {rendered}",
        field.source,
      )
  material = materials_by_id.get(region.material_id)
  if material is None:
    _fail(
      "incompatible-region-material",
      "the region must reference a declared material",
      region.source,
    )
  if region.quadrature != profile.quadrature:
    _fail(
      "incompatible-quadrature",
      "unsupported continuum quadrature",
      region.source,
    )
  expected_model = _FORMULATION_MATERIAL_MODELS.get(region.formulation)
  if expected_model is not None and material.model != expected_model:
    _fail(
      "incompatible-material-model",
      f"the {region.formulation} region requires material model {expected_model!r}",
      material.source,
    )
  if region.formulation == Q8_FORMULATION_KEY[1]:
    if profile.voigt_size != 3:
      if material.model != ISOTROPIC_MATERIAL_KEY[1]:
        _fail(
          "incompatible-material-model",
          f"the three-dimensional {profile.geometry_interpolation} geometry "
          f"requires material model {ISOTROPIC_MATERIAL_KEY[1]!r}",
          material.source,
        )
    elif material.model == ISOTROPIC_MATERIAL_KEY[1]:
      _fail(
        "incompatible-material-model",
        f"material model {ISOTROPIC_MATERIAL_KEY[1]!r} requires a "
        "three-dimensional cell-block geometry",
        material.source,
      )
    elif material.model not in _PLANAR_LINEAR_MATERIAL_MODELS and (
      profile is not _Q8_GEOMETRY
    ):
      _fail(
        "incompatible-material-model",
        "stateful continuum materials compile on the serendipity-quad8 geometry only",
        material.source,
      )
  return RegionSelection(
    block=block,
    fields=fields,
    material=material,
    region=region,
    geometry=profile,
    cells=tuple(
      sorted(
        (
          cells_by_key[cell_ref.block_id, cell_ref.cell_id]
          for cell_ref in region.cell_refs
        ),
        key=lambda item: _sort_key(item.id),
      )
    ),
  )


def _required_keys(selection: ContinuumSelection) -> tuple[RegistryKey, ...]:
  keys: set[RegistryKey] = set()
  for region_selection in selection.regions:
    profile = region_selection.geometry
    keys.add(profile.topology_key)
    keys.add(profile.quadrature_key)
    formulation = region_selection.region.formulation
    if formulation == TL_FORMULATION_KEY[1]:
      keys.add(TL_FORMULATION_KEY)
      keys.add(("material", region_selection.material.model))
    elif formulation == Q8_FORMULATION_KEY[1]:
      keys.add(profile.formulation_key)
      keys.add(("material", region_selection.material.model))
    elif formulation == THERMAL_FORMULATION_KEY[1]:
      keys.update((THERMAL_FORMULATION_KEY, THERMAL_MATERIAL_KEY))
    else:
      keys.update((THERMO_FORMULATION_KEY, THERMO_MATERIAL_KEY))
  return tuple(sorted(keys))


def capture_registry(
  registry: dict[RegistryKey, RegistryDescriptor],
  selection: ContinuumSelection,
) -> RegistrySnapshot:
  """Capture and validate exactly the implementations selected by this builder."""
  required = _required_keys(selection)
  try:
    snapshot = RegistrySnapshot.capture(registry, required=required)
  except (KeyError, TypeError, ValueError):
    _fail(
      "registry-capture-failed",
      "the injected registry could not capture the required continuum implementations",
      selection.regions[0].region.source,
    )
  sources: dict[RegistryKey, SourceContext] = {}
  for region_selection in selection.regions:
    profile = region_selection.geometry
    sources.setdefault(profile.topology_key, region_selection.block.source)
    sources.setdefault(profile.quadrature_key, region_selection.region.source)
    formulation = region_selection.region.formulation
    if formulation == TL_FORMULATION_KEY[1]:
      selected = (
        TL_FORMULATION_KEY,
        ("material", region_selection.material.model),
      )
    elif formulation == Q8_FORMULATION_KEY[1]:
      selected = (
        profile.formulation_key,
        ("material", region_selection.material.model),
      )
    elif formulation == THERMAL_FORMULATION_KEY[1]:
      selected = (THERMAL_FORMULATION_KEY, THERMAL_MATERIAL_KEY)
    else:
      selected = (THERMO_FORMULATION_KEY, THERMO_MATERIAL_KEY)
    sources.setdefault(selected[0], region_selection.region.source)
    sources.setdefault(selected[1], region_selection.material.source)
  for key in required:
    try:
      descriptor = snapshot.resolve(*key)
    except (KeyError, TypeError, ValueError):
      _fail(
        "malformed-registry-descriptor",
        "malformed continuum descriptor",
        sources[key],
      )
    try:
      qualified = _qualified_descriptor_metadata(key)
    except KeyError:
      qualified = None
    if qualified is None:
      if key[0] != "material":
        _fail(
          "malformed-registry-descriptor",
          "malformed continuum descriptor",
          sources[key],
        )
      _validate_stateful_material_descriptor(descriptor, sources[key])
      continue
    try:
      expected = CanonicalManifest(qualified)
      compatible = descriptor.metadata.to_bytes() == expected.to_bytes()
    except (TypeError, ValueError):
      _fail(
        "malformed-registry-descriptor",
        "malformed continuum descriptor",
        sources[key],
      )
    if not compatible:
      _fail(
        "incompatible-registry-descriptor",
        "continuum descriptor metadata does not match the qualified convention",
        sources[key],
      )
  return snapshot


def _validate_stateful_material_descriptor(
  descriptor: RegistryDescriptor,
  source: SourceContext,
) -> None:
  """Validate one researcher-supplied v2 stateful material descriptor.

  The binding must re-declare the descriptor metadata it was authored with;
  the re-declaration is canonicalized and byte-compared against the captured
  descriptor manifest, then structurally validated against the frozen v2
  schema. This is the open extension seam: no in-tree convention is pinned
  for keys the reference registries do not carry.
  """
  binding = descriptor.binding
  declared = getattr(binding, "descriptor_metadata", None)
  if not callable(declared):
    _fail(
      "malformed-registry-descriptor",
      "stateful material bindings must re-declare their descriptor metadata",
      source,
    )
  try:
    metadata = declared()
  except Exception:
    _fail(
      "malformed-registry-descriptor",
      "stateful material binding metadata declaration failed",
      source,
    )
  try:
    canonical = CanonicalManifest(metadata)
  except (TypeError, ValueError):
    _fail(
      "malformed-registry-descriptor",
      "stateful material binding metadata is not canonicalizable",
      source,
    )
  if canonical.to_bytes() != descriptor.metadata.to_bytes():
    _fail(
      "incompatible-registry-descriptor",
      "stateful material binding metadata does not match its descriptor",
      source,
    )
  try:
    validate_stateful_material_metadata(metadata)
  except (TypeError, ValueError):
    _fail(
      "malformed-registry-descriptor",
      "stateful material metadata violates the v2 descriptor schema",
      source,
    )


def _binding_array(
  value: object,
  *,
  shape: tuple[int, ...],
  code: str,
  label: str,
  source: SourceContext,
) -> np.ndarray:
  if (
    type(value) is not np.ndarray
    or value.shape != shape
    or value.dtype.metadata is not None
    or value.dtype.kind not in "iuf"
  ):
    _fail(code, f"{label} must return a metadata-free numeric array {shape!r}", source)
  try:
    captured = np.array(value, dtype=np.float64, order="C", copy=True, subok=False)
  except (OverflowError, TypeError, ValueError):
    _fail(code, f"{label} cannot be represented as float64", source)
  if not bool(np.isfinite(captured).all()):
    _fail(code, f"{label} must contain finite values", source)
  return captured


def _corresponds(actual: np.ndarray, expected: np.ndarray) -> bool:
  scale = float(np.max(np.abs(expected)))
  zero_tolerance = 8.0 * float(np.finfo(np.float64).eps) * scale
  ulps = 16.0 * np.abs(np.spacing(expected))
  tolerance = np.where(expected == 0.0, zero_tolerance, ulps)
  return bool(np.all(np.abs(actual - expected) <= tolerance))


def _tl_probe_corresponds(actual: np.ndarray, expected: np.ndarray) -> bool:
  """Scale-relative correspondence for the TL formulation-binding probe.

  The probe compares a full element response (tangent and internal force)
  computed two ways — the descriptor-bound kernel and the qualified reference
  assembly — whose summation orders differ, so analytically-vanishing entries
  carry cancellation noise at the ``eps * scale`` level. A per-entry ulp
  envelope (``_corresponds``) would reject exactly that noise; the probe
  therefore uses the scale-relative tolerance idiom of the landed symmetry
  checks (64 eps on the response scale). Structural errors in a binding (a
  wrong B row, a dropped geometric term) are O(scale) and never pass.
  """
  scale = float(np.max(np.abs(expected)))
  if scale == 0.0:
    return bool(np.array_equal(actual, expected))
  tolerance = 64.0 * float(np.finfo(np.float64).eps) * scale
  return bool(
    np.all(
      np.abs(actual - expected)
      <= tolerance + 64.0 * float(np.finfo(np.float64).eps) * np.abs(expected)
    )
  )


def _qualified_constitutive(
  youngs_modulus: float,
  poisson_ratio: float,
  source: SourceContext,
) -> tuple[np.ndarray, np.ndarray]:
  modulus = Fraction.from_float(youngs_modulus)
  ratio = Fraction.from_float(poisson_ratio)
  normal = modulus / ((1 - ratio) * (1 + ratio))
  coupling = normal * ratio
  shear = modulus / (2 * (1 + ratio))
  try:
    exact = np.array(
      [
        [float(normal), float(coupling), 0.0],
        [float(coupling), float(normal), 0.0],
        [0.0, 0.0, float(shear)],
      ],
      dtype=np.float64,
    )
    if not bool(np.isfinite(exact).all()):
      raise OverflowError
  except (ArithmeticError, OverflowError, ValueError):
    _fail(
      "unrepresentable-material-law",
      "Q8 plane-stress matrix cannot be represented as finite float64",
      source,
    )
  with warnings.catch_warnings(action="ignore", category=RuntimeWarning):
    binary64_route = plane_stress_matrix(youngs_modulus, poisson_ratio)
  if not bool(np.isfinite(binary64_route).all()):
    binary64_route = exact
  return exact, binary64_route


def _qualified_plane_strain_constitutive(
  youngs_modulus: float,
  poisson_ratio: float,
  source: SourceContext,
) -> tuple[np.ndarray, np.ndarray]:
  modulus = Fraction.from_float(youngs_modulus)
  ratio = Fraction.from_float(poisson_ratio)
  normal = modulus * (1 - ratio) / ((1 + ratio) * (1 - 2 * ratio))
  coupling = normal * ratio / (1 - ratio)
  shear = modulus / (2 * (1 + ratio))
  try:
    exact = np.array(
      [
        [float(normal), float(coupling), 0.0],
        [float(coupling), float(normal), 0.0],
        [0.0, 0.0, float(shear)],
      ],
      dtype=np.float64,
    )
    if not bool(np.isfinite(exact).all()):
      raise OverflowError
  except (ArithmeticError, OverflowError, ValueError):
    _fail(
      "unrepresentable-material-law",
      "plane-strain matrix cannot be represented as finite float64",
      source,
    )
  with warnings.catch_warnings(action="ignore", category=RuntimeWarning):
    binary64_route = plane_strain_matrix(youngs_modulus, poisson_ratio)
  if not bool(np.isfinite(binary64_route).all()):
    binary64_route = exact
  return exact, binary64_route


def _qualified_isotropic_constitutive(
  youngs_modulus: float,
  poisson_ratio: float,
  source: SourceContext,
) -> tuple[np.ndarray, np.ndarray]:
  modulus = Fraction.from_float(youngs_modulus)
  ratio = Fraction.from_float(poisson_ratio)
  denominator = 2 * ratio * ratio + ratio - 1
  normal = modulus * (ratio - 1) / denominator
  coupling = -modulus * ratio / denominator
  shear = modulus / (2 + 2 * ratio)
  try:
    exact = np.array(
      [
        [float(normal), float(coupling), float(coupling), 0.0, 0.0, 0.0],
        [float(coupling), float(normal), float(coupling), 0.0, 0.0, 0.0],
        [float(coupling), float(coupling), float(normal), 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, float(shear), 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, float(shear), 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, float(shear)],
      ],
      dtype=np.float64,
    )
    if not bool(np.isfinite(exact).all()):
      raise OverflowError
  except (ArithmeticError, OverflowError, ValueError):
    _fail(
      "unrepresentable-material-law",
      "isotropic 3D matrix cannot be represented as finite float64",
      source,
    )
  with warnings.catch_warnings(action="ignore", category=RuntimeWarning):
    binary64_route = isotropic_matrix(youngs_modulus, poisson_ratio)
  if not bool(np.isfinite(binary64_route).all()):
    binary64_route = exact
  return exact, binary64_route


def _outside_relative_envelope(actual: float, first: float, second: float) -> bool:
  lower, upper = sorted((first, second))
  return not (
    lower <= actual <= upper
    or any(
      math.isclose(
        actual,
        boundary,
        rel_tol=_MATERIAL_RELATIVE_TOLERANCE,
        abs_tol=0.0,
      )
      for boundary in (lower, upper)
    )
  )


def _constitutive_corresponds(
  actual: np.ndarray,
  exact: np.ndarray,
  binary64_route: np.ndarray,
) -> bool:
  return not any(
    _outside_relative_envelope(float(value), float(qualified), float(route))
    for value, qualified, route in zip(
      actual.flat,
      exact.flat,
      binary64_route.flat,
      strict=True,
    )
  )


def _qualified_shapes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  xi, eta = points[:, 0], points[:, 1]
  values = np.stack(
    (
      -0.25 * (1 - xi) * (1 - eta) * (1 + xi + eta),
      0.5 * (1 - xi) * (1 + xi) * (1 - eta),
      -0.25 * (1 + xi) * (1 - eta) * (1 - xi + eta),
      0.5 * (1 + xi) * (1 + eta) * (1 - eta),
      -0.25 * (1 + xi) * (1 + eta) * (1 - xi - eta),
      0.5 * (1 - xi) * (1 + xi) * (1 + eta),
      -0.25 * (1 - xi) * (1 + eta) * (1 + xi - eta),
      0.5 * (1 - xi) * (1 + eta) * (1 - eta),
    ),
    axis=1,
  )
  dxi = np.stack(
    (
      -0.25 * (-1 + eta) * (2 * xi + eta),
      xi * (-1 + eta),
      0.25 * (-1 + eta) * (-2 * xi + eta),
      -0.5 * (1 + eta) * (-1 + eta),
      0.25 * (1 + eta) * (2 * xi + eta),
      -xi * (1 + eta),
      -0.25 * (1 + eta) * (-2 * xi + eta),
      0.5 * (1 + eta) * (-1 + eta),
    ),
    axis=1,
  )
  deta = np.stack(
    (
      -0.25 * (-1 + xi) * (xi + 2 * eta),
      0.5 * (1 + xi) * (-1 + xi),
      0.25 * (1 + xi) * (-xi + 2 * eta),
      -eta * (1 + xi),
      0.25 * (1 + xi) * (xi + 2 * eta),
      -0.5 * (1 + xi) * (-1 + xi),
      -0.25 * (-1 + xi) * (-xi + 2 * eta),
      eta * (-1 + xi),
    ),
    axis=1,
  )
  return values, np.stack((dxi, deta), axis=2)


def _qualified_quadrature(
  profile: ContinuumGeometryProfile,
) -> tuple[np.ndarray, np.ndarray]:
  """Return the pinned parent points and weights of one geometry's rule."""
  if profile is _Q8_GEOMETRY:
    abscissa = math.sqrt(3.0 / 5.0)
    expected_points = np.array(
      [(x, y) for x in (-abscissa, 0.0, abscissa) for y in (-abscissa, 0.0, abscissa)],
      dtype=np.float64,
    )
    expected_weights = np.array(
      [
        x * y
        for x in (5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0)
        for y in (5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0)
      ],
      dtype=np.float64,
    )
    return expected_points, expected_weights
  if profile is _TRIA3_GEOMETRY:
    return (
      np.array([[1.0 / 3.0, 1.0 / 3.0]], dtype=np.float64),
      np.array([0.5], dtype=np.float64),
    )
  abscissa = 1.0 / math.sqrt(3.0)
  if profile is _QUAD4_GEOMETRY:
    return (
      np.array(
        [(x, y) for x in (-abscissa, abscissa) for y in (-abscissa, abscissa)],
        dtype=np.float64,
      ),
      np.ones(profile.point_count, dtype=np.float64),
    )
  return (
    np.array(
      [
        (x, y, z)
        for x in (-abscissa, abscissa)
        for y in (-abscissa, abscissa)
        for z in (-abscissa, abscissa)
      ],
      dtype=np.float64,
    ),
    np.ones(profile.point_count, dtype=np.float64),
  )


def _qualified_profile_shapes(
  profile: ContinuumGeometryProfile,
  points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  """Return the trusted shape values and parent gradients at the rule points.

  The quad8 reference mirrors the qualified in-builder formulas; the breadth
  geometries consume the landed ``pyfem.v3.fem.shapes`` implementations as
  their reference instead of duplicating them.
  """
  if profile is _Q8_GEOMETRY:
    return _qualified_shapes(points)
  if profile is _TRIA3_GEOMETRY:
    values, gradients = linear_tria3(points)
  elif profile is _QUAD4_GEOMETRY:
    values, gradients = bilinear_quad4(points)
  else:
    values, gradients = trilinear_hex8(points)
  return (
    np.array(values, dtype=np.float64, order="C", copy=True, subok=False),
    np.array(gradients, dtype=np.float64, order="C", copy=True, subok=False),
  )


def _recipes(
  snapshot: RegistrySnapshot,
  selection: RegionSelection,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  profile = selection.geometry
  label = "Q8" if profile is _Q8_GEOMETRY else profile.geometry_interpolation
  quadrature = snapshot.resolve(*profile.quadrature_key).binding
  try:
    raw_quadrature = quadrature(profile.quadrature_order)
  except Exception:
    _fail(
      "quadrature-binding-failed",
      f"{label} quadrature binding failed",
      selection.region.source,
    )
  if type(raw_quadrature) is not tuple or len(raw_quadrature) != 2:
    _fail(
      "invalid-quadrature-binding-output",
      f"{label} quadrature must return exactly points and weights",
      selection.region.source,
    )
  points = _binding_array(
    raw_quadrature[0],
    shape=(profile.point_count, profile.topological_dimension),
    code="invalid-quadrature-binding-output",
    label=f"{label} quadrature points",
    source=selection.region.source,
  )
  weights = _binding_array(
    raw_quadrature[1],
    shape=(profile.point_count,),
    code="invalid-quadrature-binding-output",
    label=f"{label} quadrature weights",
    source=selection.region.source,
  )
  expected_points, expected_weights = _qualified_quadrature(profile)
  if (
    bool(np.any(weights <= 0.0))
    or bool(np.any(np.abs(points) > 1.0))
    or not math.isclose(
      float(weights.sum()),
      profile.quadrature_measure,
      rel_tol=1.0e-14,
      abs_tol=1.0e-14,
    )
    or not _corresponds(points, expected_points)
    or not _corresponds(weights, expected_weights)
  ):
    measure = "four" if profile is _Q8_GEOMETRY else "the parent measure"
    _fail(
      "invalid-quadrature-binding-output",
      f"{label} quadrature requires positive in-domain weights summing to {measure}",
      selection.region.source,
    )
  topology = snapshot.resolve(*profile.topology_key).binding
  try:
    raw_topology = topology(np.array(points, copy=True))
  except Exception:
    _fail(
      "topology-binding-failed",
      f"{label} topology binding failed",
      selection.block.source,
    )
  if type(raw_topology) is not tuple or len(raw_topology) != 2:
    _fail(
      "invalid-topology-binding-output",
      f"{label} topology must return exactly shape values and parent gradients",
      selection.block.source,
    )
  shape_values = _binding_array(
    raw_topology[0],
    shape=(profile.point_count, profile.node_count),
    code="invalid-topology-binding-output",
    label=f"{label} shape values",
    source=selection.block.source,
  )
  parent_gradients = _binding_array(
    raw_topology[1],
    shape=(profile.point_count, profile.node_count, profile.topological_dimension),
    code="invalid-topology-binding-output",
    label=f"{label} parent gradients",
    source=selection.block.source,
  )
  expected_values, expected_gradients = _qualified_profile_shapes(profile, points)
  if (
    not bool(np.allclose(shape_values.sum(1), 1.0, rtol=0.0, atol=1.0e-12))
    or not bool(np.allclose(parent_gradients.sum(1), 0.0, rtol=0.0, atol=1.0e-12))
    or not _corresponds(shape_values, expected_values)
    or not _corresponds(parent_gradients, expected_gradients)
  ):
    _fail(
      "invalid-topology-binding-output",
      f"{label} topology violates partition or gradient completeness",
      selection.block.source,
    )
  return points, weights, shape_values, parent_gradients


def _parameters(
  selection: RegionSelection,
  profile: ContinuumGeometryProfile,
) -> tuple[float, float]:
  by_name = {item.name: item for item in selection.material.parameters}
  if tuple(sorted(by_name)) != tuple(sorted(_PARAMETER_NAMES)):
    if profile is _Q8_GEOMETRY:
      message = "Q8 plane stress requires youngs_modulus and poisson_ratio"
    else:
      message = (
        f"the {selection.material.model} material on {profile.geometry_interpolation} "
        "requires youngs_modulus and poisson_ratio"
      )
    _fail(
      "invalid-material-parameter-schema",
      message,
      selection.material.source,
    )
  values: list[float] = []
  for name in _PARAMETER_NAMES:
    parameter = by_name[name]
    if type(parameter.value) is not int and type(parameter.value) is not float:
      _fail(
        "invalid-material-parameter-type",
        f"{name} must be an exact scalar",
        parameter.source,
      )
    try:
      value = float(parameter.value)
    except OverflowError:
      _fail(
        "invalid-material-parameter-value",
        f"{name} must fit finite float64",
        parameter.source,
      )
    if not math.isfinite(value):
      _fail(
        "invalid-material-parameter-value",
        f"{name} must fit finite float64",
        parameter.source,
      )
    values.append(value)
  youngs_modulus, poisson_ratio = values
  if youngs_modulus <= 0.0:
    _fail(
      "invalid-youngs-modulus",
      "youngs_modulus must be positive",
      by_name[_PARAMETER_NAMES[0]].source,
    )
  if not -1.0 < poisson_ratio < 0.5:
    _fail(
      "invalid-poisson-ratio",
      "poisson_ratio must lie between -1 and 0.5",
      by_name[_PARAMETER_NAMES[1]].source,
    )
  return youngs_modulus, poisson_ratio


def _scalar_parameters(
  selection: RegionSelection,
  names: tuple[str, ...],
) -> dict[str, tuple[MaterialParameterSpec, float]]:
  by_name = {item.name: item for item in selection.material.parameters}
  if tuple(sorted(by_name)) != tuple(sorted(names)):
    _fail(
      "invalid-material-parameter-schema",
      f"the {selection.material.model} material requires exactly {', '.join(names)}",
      selection.material.source,
    )
  values: dict[str, tuple[MaterialParameterSpec, float]] = {}
  for name in names:
    parameter = by_name[name]
    if type(parameter.value) is not int and type(parameter.value) is not float:
      _fail(
        "invalid-material-parameter-type",
        f"{name} must be an exact scalar",
        parameter.source,
      )
    try:
      value = float(parameter.value)
    except OverflowError:
      _fail(
        "invalid-material-parameter-value",
        f"{name} must fit finite float64",
        parameter.source,
      )
    if not math.isfinite(value):
      _fail(
        "invalid-material-parameter-value",
        f"{name} must fit finite float64",
        parameter.source,
      )
    values[name] = parameter, value
  return values


def _thermal_parameters(selection: RegionSelection) -> float:
  values = _scalar_parameters(selection, _THERMAL_PARAMETER_NAMES)
  parameter, conductivity = values[_THERMAL_PARAMETER_NAMES[0]]
  if conductivity <= 0.0:
    _fail(
      "invalid-conductivity",
      "conductivity must be positive",
      parameter.source,
    )
  return conductivity


def _thermo_elastic_parameters(selection: RegionSelection) -> tuple[float, ...]:
  values = _scalar_parameters(selection, _THERMO_PARAMETER_NAMES)
  youngs_parameter, youngs_modulus = values[_THERMO_PARAMETER_NAMES[0]]
  ratio_parameter, poisson_ratio = values[_THERMO_PARAMETER_NAMES[1]]
  conductivity_parameter, conductivity = values[_THERMO_PARAMETER_NAMES[3]]
  if youngs_modulus <= 0.0:
    _fail(
      "invalid-youngs-modulus",
      "youngs_modulus must be positive",
      youngs_parameter.source,
    )
  if not -1.0 < poisson_ratio < 0.5:
    _fail(
      "invalid-poisson-ratio",
      "poisson_ratio must lie between -1 and 0.5",
      ratio_parameter.source,
    )
  if conductivity <= 0.0:
    _fail(
      "invalid-conductivity",
      "conductivity must be positive",
      conductivity_parameter.source,
    )
  return (
    youngs_modulus,
    poisson_ratio,
    values[_THERMO_PARAMETER_NAMES[2]][1],
    conductivity,
  )


def _normalized_coordinates(
  coordinates: np.ndarray, cell: CellSpec
) -> tuple[np.ndarray, float]:
  coordinate_scale = 1.0
  with np.errstate(over="ignore", invalid="ignore", under="ignore"):
    relative = coordinates - coordinates[0]
  if not bool(np.isfinite(relative).all()):
    coordinate_scale = float(np.max(np.abs(coordinates)))
    if not math.isfinite(coordinate_scale) or coordinate_scale == 0.0:
      _fail(
        "non-finite-reference-geometry",
        "Q8 has non-finite or zero-scale geometry",
        cell.source,
      )
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
      relative = coordinates / coordinate_scale - coordinates[0] / coordinate_scale
  cell_scale = float(np.max(np.abs(relative)))
  if not math.isfinite(cell_scale) or cell_scale == 0.0:
    _fail(
      "non-finite-reference-geometry",
      "Q8 has non-finite or zero-scale geometry",
      cell.source,
    )
  normalized = relative / cell_scale
  if not bool(np.isfinite(normalized).all()):
    _fail(
      "non-finite-reference-geometry",
      "Q8 normalized geometry is non-finite",
      cell.source,
    )
  physical_scale = coordinate_scale * cell_scale
  if not math.isfinite(physical_scale) or physical_scale == 0.0:
    _fail(
      "non-finite-reference-geometry", "Q8 physical scale is not finite", cell.source
    )
  return normalized, physical_scale


def _geometry(
  coordinates: np.ndarray,
  connectivity: np.ndarray,
  parent_gradients: np.ndarray,
  cells: tuple[CellSpec, ...],
  tolerance: float,
  profile: ContinuumGeometryProfile,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  label = "Q8" if profile is _Q8_GEOMETRY else profile.geometry_interpolation
  point_count = profile.point_count
  gradients = np.empty(
    (len(cells), point_count, profile.node_count, 2), dtype=np.float64
  )
  determinants = np.empty((len(cells), point_count), dtype=np.float64)
  scales = np.empty(len(cells), dtype=np.float64)
  for cell_index, cell in enumerate(cells):
    normalized, scales[cell_index] = _normalized_coordinates(
      coordinates[connectivity[cell_index]], cell
    )
    norm_squares = np.empty(point_count, dtype=np.float64)
    for point_index in range(point_count):
      jacobian = normalized.T @ parent_gradients[point_index]
      determinant = float(
        jacobian[0, 0] * jacobian[1, 1] - jacobian[0, 1] * jacobian[1, 0]
      )
      norm_square = math.fsum(float(value * value) for value in jacobian.flat)
      if (
        not math.isfinite(determinant)
        or not math.isfinite(norm_square)
        or norm_square == 0.0
      ):
        _fail(
          "non-finite-reference-geometry",
          f"{label} has a non-finite reference Jacobian",
          cell.source,
        )
      determinants[cell_index, point_index] = determinant
      norm_squares[point_index] = norm_square
    signs = determinants[cell_index]
    if bool(np.any(signs > 0.0)) and bool(np.any(signs < 0.0)):
      _fail(
        "sign-changing-reference-geometry",
        f"{label} changes orientation across quadrature points",
        cell.source,
      )
    if bool(np.any(signs < 0.0)):
      _fail(
        "inverted-reference-geometry",
        f"{label} has negative reference orientation",
        cell.source,
      )
    for point_index in range(point_count):
      determinant = float(determinants[cell_index, point_index])
      if (
        determinant <= tolerance
        or determinant / float(norm_squares[point_index]) <= tolerance
      ):
        _fail(
          "near-singular-reference-geometry",
          f"{label} cell {render_diagnostic_value(cell.id)} has a scale-relative "
          f"near-singular Jacobian at point {point_index}",
          cell.source,
        )
      jacobian = normalized.T @ parent_gradients[point_index]
      inverse = np.array(
        ((jacobian[1, 1], -jacobian[0, 1]), (-jacobian[1, 0], jacobian[0, 0])),
        dtype=np.float64,
      )
      inverse /= determinant
      gradients[cell_index, point_index] = parent_gradients[point_index] @ inverse
      if not bool(np.isfinite(gradients[cell_index, point_index]).all()):
        _fail(
          "non-finite-reference-geometry",
          f"{label} physical gradients are non-finite",
          cell.source,
        )
  return gradients, determinants, scales


def _geometry_3d(
  coordinates: np.ndarray,
  connectivity: np.ndarray,
  parent_gradients: np.ndarray,
  cells: tuple[CellSpec, ...],
  tolerance: float,
  profile: ContinuumGeometryProfile,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Three-dimensional analogue of ``_geometry`` (explicit 3x3 adjugate)."""
  label = profile.geometry_interpolation
  point_count = profile.point_count
  gradients = np.empty(
    (len(cells), point_count, profile.node_count, 3), dtype=np.float64
  )
  determinants = np.empty((len(cells), point_count), dtype=np.float64)
  scales = np.empty(len(cells), dtype=np.float64)
  for cell_index, cell in enumerate(cells):
    normalized, scales[cell_index] = _normalized_coordinates(
      coordinates[connectivity[cell_index]], cell
    )
    norm_squares = np.empty(point_count, dtype=np.float64)
    for point_index in range(point_count):
      jacobian = normalized.T @ parent_gradients[point_index]
      determinant = float(
        jacobian[0, 0]
        * (jacobian[1, 1] * jacobian[2, 2] - jacobian[1, 2] * jacobian[2, 1])
        - jacobian[0, 1]
        * (jacobian[1, 0] * jacobian[2, 2] - jacobian[1, 2] * jacobian[2, 0])
        + jacobian[0, 2]
        * (jacobian[1, 0] * jacobian[2, 1] - jacobian[1, 1] * jacobian[2, 0])
      )
      norm_square = math.fsum(float(value * value) for value in jacobian.flat)
      if (
        not math.isfinite(determinant)
        or not math.isfinite(norm_square)
        or norm_square == 0.0
      ):
        _fail(
          "non-finite-reference-geometry",
          f"{label} has a non-finite reference Jacobian",
          cell.source,
        )
      determinants[cell_index, point_index] = determinant
      norm_squares[point_index] = norm_square
    signs = determinants[cell_index]
    if bool(np.any(signs > 0.0)) and bool(np.any(signs < 0.0)):
      _fail(
        "sign-changing-reference-geometry",
        f"{label} changes orientation across quadrature points",
        cell.source,
      )
    if bool(np.any(signs < 0.0)):
      _fail(
        "inverted-reference-geometry",
        f"{label} has negative reference orientation",
        cell.source,
      )
    for point_index in range(point_count):
      determinant = float(determinants[cell_index, point_index])
      condition = float(norm_squares[point_index]) ** 1.5
      if determinant <= tolerance or determinant / condition <= tolerance:
        _fail(
          "near-singular-reference-geometry",
          f"{label} cell {render_diagnostic_value(cell.id)} has a scale-relative "
          f"near-singular Jacobian at point {point_index}",
          cell.source,
        )
      jacobian = normalized.T @ parent_gradients[point_index]
      inverse = np.array(
        (
          (
            jacobian[1, 1] * jacobian[2, 2] - jacobian[1, 2] * jacobian[2, 1],
            jacobian[0, 2] * jacobian[2, 1] - jacobian[0, 1] * jacobian[2, 2],
            jacobian[0, 1] * jacobian[1, 2] - jacobian[0, 2] * jacobian[1, 1],
          ),
          (
            jacobian[1, 2] * jacobian[2, 0] - jacobian[1, 0] * jacobian[2, 2],
            jacobian[0, 0] * jacobian[2, 2] - jacobian[0, 2] * jacobian[2, 0],
            jacobian[0, 2] * jacobian[1, 0] - jacobian[0, 0] * jacobian[1, 2],
          ),
          (
            jacobian[1, 0] * jacobian[2, 1] - jacobian[1, 1] * jacobian[2, 0],
            jacobian[0, 1] * jacobian[2, 0] - jacobian[0, 0] * jacobian[2, 1],
            jacobian[0, 0] * jacobian[1, 1] - jacobian[0, 1] * jacobian[1, 0],
          ),
        ),
        dtype=np.float64,
      )
      inverse /= determinant
      gradients[cell_index, point_index] = parent_gradients[point_index] @ inverse
      if not bool(np.isfinite(gradients[cell_index, point_index]).all()):
        _fail(
          "non-finite-reference-geometry",
          f"{label} physical gradients are non-finite",
          cell.source,
        )
  return gradients, determinants, scales


def _validate_physical_recovery(
  gradients: np.ndarray,
  b_matrix: np.ndarray,
  integration_weights: np.ndarray,
  scales: np.ndarray,
  cells: tuple[CellSpec, ...],
  profile: ContinuumGeometryProfile,
) -> None:
  label = "Q8" if profile is _Q8_GEOMETRY else profile.geometry_interpolation
  with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
    physical_gradients = gradients / scales[:, None, None, None]
    physical_b = b_matrix / scales[:, None, None, None]
    physical_weights = integration_weights * scales[:, None] * scales[:, None]
    if profile.embedding_dimension == 3:
      physical_weights = physical_weights * scales[:, None]
  for index, cell in enumerate(cells):
    if (
      not bool(np.isfinite(physical_gradients[index]).all())
      or not bool(np.isfinite(physical_b[index]).all())
      or not bool(np.isfinite(physical_weights[index]).all())
      or bool(np.any(physical_weights[index] <= 0.0))
    ):
      _fail(
        "unrepresentable-physical-geometry",
        f"{label} cell {render_diagnostic_value(cell.id)} physical gradients or "
        "integration measure cannot be represented as float64",
        cell.source,
      )


def _gather_map(
  space: DiscreteSpace,
  connectivity: np.ndarray,
  node_dense: dict[SpecId, int],
) -> np.ndarray:
  coefficient_map = space.coefficient_map.values
  if coefficient_map.shape[0] == len(node_dense):
    return coefficient_map[connectivity]
  component_count = len(space.components)
  support_rows = {
    node_dense[point_id]: row
    for row, point_id in enumerate(
      space.coefficient_ids[index][1]
      for index in range(0, len(space.coefficient_ids), component_count)
    )
  }
  rows = np.fromiter(
    (support_rows[int(dense)] for dense in connectivity.flat),
    dtype=connectivity.dtype,
    count=connectivity.size,
  ).reshape(connectivity.shape)
  return coefficient_map[rows]


def _identities(snapshot: RegistrySnapshot) -> tuple[ImplementationIdentity, ...]:
  return tuple(
    _new(
      ImplementationIdentity,
      kind=descriptor.kind,
      name=descriptor.name,
      version=descriptor.version,
      implementation_id=descriptor.implementation_id,
    )
    for descriptor in snapshot.descriptors
  )


def compile_operator(
  selection: RegionSelection,
  *,
  coordinates: FinalizedArray,
  node_dense: dict[SpecId, int],
  spaces: dict[SpecId, DiscreteSpace],
  snapshot: RegistrySnapshot,
  index_dtype: np.dtype,
  geometry_relative_tolerance: float,
) -> tuple[IncidenceEntityBlock, CompiledOperator]:
  """Compile the concrete payload and return it behind the open operator header."""
  formulation = selection.region.formulation
  if formulation == THERMAL_FORMULATION_KEY[1]:
    return _compile_thermal(
      selection,
      coordinates=coordinates,
      node_dense=node_dense,
      spaces=spaces,
      snapshot=snapshot,
      index_dtype=index_dtype,
      geometry_relative_tolerance=geometry_relative_tolerance,
    )
  if formulation == THERMO_FORMULATION_KEY[1]:
    return _compile_coupled(
      selection,
      coordinates=coordinates,
      node_dense=node_dense,
      spaces=spaces,
      snapshot=snapshot,
      index_dtype=index_dtype,
      geometry_relative_tolerance=geometry_relative_tolerance,
    )
  if formulation == TL_FORMULATION_KEY[1]:
    return _compile_mechanical_tl(
      selection,
      coordinates=coordinates,
      node_dense=node_dense,
      spaces=spaces,
      snapshot=snapshot,
      index_dtype=index_dtype,
      geometry_relative_tolerance=geometry_relative_tolerance,
    )
  return _compile_mechanical(
    selection,
    coordinates=coordinates,
    node_dense=node_dense,
    spaces=spaces,
    snapshot=snapshot,
    index_dtype=index_dtype,
    geometry_relative_tolerance=geometry_relative_tolerance,
  )


def _compile_mechanical(
  selection: RegionSelection,
  *,
  coordinates: FinalizedArray,
  node_dense: dict[SpecId, int],
  spaces: dict[SpecId, DiscreteSpace],
  snapshot: RegistrySnapshot,
  index_dtype: np.dtype,
  geometry_relative_tolerance: float,
) -> tuple[IncidenceEntityBlock, CompiledOperator]:
  model = selection.material.model
  if model == ISOTROPIC_MATERIAL_KEY[1]:
    return _compile_mechanical_3d(
      selection,
      coordinates=coordinates,
      node_dense=node_dense,
      spaces=spaces,
      snapshot=snapshot,
      index_dtype=index_dtype,
      geometry_relative_tolerance=geometry_relative_tolerance,
    )
  if model not in _PLANAR_LINEAR_MATERIAL_MODELS:
    return _compile_mechanical_stateful(
      selection,
      coordinates=coordinates,
      node_dense=node_dense,
      spaces=spaces,
      snapshot=snapshot,
      index_dtype=index_dtype,
      geometry_relative_tolerance=geometry_relative_tolerance,
    )
  return _compile_mechanical_linear(
    selection,
    coordinates=coordinates,
    node_dense=node_dense,
    spaces=spaces,
    snapshot=snapshot,
    index_dtype=index_dtype,
    geometry_relative_tolerance=geometry_relative_tolerance,
  )


def _compile_mechanical_linear(
  selection: RegionSelection,
  *,
  coordinates: FinalizedArray,
  node_dense: dict[SpecId, int],
  spaces: dict[SpecId, DiscreteSpace],
  snapshot: RegistrySnapshot,
  index_dtype: np.dtype,
  geometry_relative_tolerance: float,
) -> tuple[IncidenceEntityBlock, CompiledOperator]:
  profile = selection.geometry
  label = "Q8" if profile is _Q8_GEOMETRY else profile.geometry_interpolation
  space = spaces[selection.fields[0].id]
  connectivity_values = [
    [node_dense[node_id] for node_id in cell.node_ids] for cell in selection.cells
  ]
  connectivity = FinalizedArray(connectivity_values, dtype=index_dtype)
  entity_block_id = selection.block.id
  entity_block = _new(
    IncidenceEntityBlock,
    block_id=entity_block_id,
    entity_ids=tuple(cell.id for cell in selection.cells),
    sources=tuple(_source(cell.source) for cell in selection.cells),
    incidence=connectivity,
  )
  points, weights, shape_values, parent_gradients = _recipes(snapshot, selection)
  gradients, determinants, geometry_scales = _geometry(
    coordinates.values,
    connectivity.values,
    parent_gradients,
    selection.cells,
    geometry_relative_tolerance,
    profile,
  )
  formulation = snapshot.resolve(*profile.formulation_key).binding
  try:
    raw_b_matrix = formulation(np.array(gradients, copy=True))
  except Exception:
    _fail(
      "formulation-binding-failed",
      f"{label} formulation binding failed",
      selection.region.source,
    )
  b_matrix = _binding_array(
    raw_b_matrix,
    shape=(
      len(selection.cells),
      profile.point_count,
      3,
      profile.local_coefficient_count,
    ),
    code="invalid-formulation-binding-output",
    label=f"{label} strain-displacement binding",
    source=selection.region.source,
  )
  expected_b = np.zeros_like(b_matrix)
  expected_b[..., 0, 0::2] = gradients[..., :, 0]
  expected_b[..., 1, 1::2] = gradients[..., :, 1]
  expected_b[..., 2, 0::2] = gradients[..., :, 1]
  expected_b[..., 2, 1::2] = gradients[..., :, 0]
  if not _corresponds(b_matrix, expected_b):
    _fail(
      "incompatible-formulation-binding-output",
      f"{label} formulation output contradicts the qualified engineering-shear map",
      selection.region.source,
    )
  youngs_modulus, poisson_ratio = _parameters(selection, profile)
  if selection.material.model == PLANE_STRAIN_MATERIAL_KEY[1]:
    exact_constitutive, binary64_route = _qualified_plane_strain_constitutive(
      youngs_modulus,
      poisson_ratio,
      selection.material.source,
    )
  else:
    exact_constitutive, binary64_route = _qualified_constitutive(
      youngs_modulus,
      poisson_ratio,
      selection.material.source,
    )
  material = snapshot.resolve("material", selection.material.model).binding
  try:
    with warnings.catch_warnings():
      warnings.simplefilter("error", RuntimeWarning)
      raw_constitutive = material(youngs_modulus, poisson_ratio)
  except Exception:
    _fail(
      "material-binding-failed",
      f"{label} material binding failed",
      selection.material.source,
    )
  constitutive = _binding_array(
    raw_constitutive,
    shape=(3, 3),
    code="invalid-material-binding-output",
    label=f"{label} material binding",
    source=selection.material.source,
  )
  if not bool(np.array_equal(constitutive, constitutive.T)):
    _fail(
      "nonsymmetric-material-binding",
      f"{label} material tangent must be symmetric",
      selection.material.source,
    )
  law = (
    "plane-strain"
    if selection.material.model == PLANE_STRAIN_MATERIAL_KEY[1]
    else "plane-stress"
  )
  if not _constitutive_corresponds(
    constitutive,
    exact_constitutive,
    binary64_route,
  ):
    _fail(
      "incompatible-material-binding-output",
      f"{label} material output contradicts the qualified {law} law",
      selection.material.source,
    )
  integration_weights = determinants * weights[None, :]
  _validate_physical_recovery(
    gradients,
    b_matrix,
    integration_weights,
    geometry_scales,
    selection.cells,
    profile,
  )
  with np.errstate(invalid="ignore", over="ignore", under="ignore"):
    tangent = np.einsum(
      "ep,epai,ab,epbj->eij",
      integration_weights,
      b_matrix,
      constitutive,
      b_matrix,
      optimize=True,
    )
  if not bool(np.isfinite(tangent).all()):
    _fail(
      "non-finite-element-operator",
      f"{label} element operator is non-finite",
      selection.region.source,
    )
  tangent_scale = float(np.max(np.abs(tangent)))
  symmetry_tolerance = 64.0 * max(
    float(np.finfo(np.float64).eps) * tangent_scale,
    abs(tangent_scale - math.nextafter(tangent_scale, 0.0)),
  )
  if not bool(
    np.allclose(
      tangent,
      tangent.transpose(0, 2, 1),
      rtol=0.0,
      atol=symmetry_tolerance,
    )
  ):
    _fail(
      "nonsymmetric-element-operator",
      f"{label} element operator is not symmetric",
      selection.region.source,
    )

  gather = FinalizedArray(
    _gather_map(space, connectivity.values, node_dense).reshape(
      len(selection.cells), -1
    ),
    dtype=index_dtype,
  )
  block_id = selection.block.id, selection.region.id
  state_layout = _new(
    OperatorStateLayout,
    schema="pyfem-v3-operator-state-layout-v1",
    block_id=block_id,
    entity_count=len(selection.cells),
    slots=(),
    entity_offsets=FinalizedArray(
      np.zeros(len(selection.cells) + 1), dtype=index_dtype
    ),
    row_width=0,
    dtype=np.dtype(np.float64).str,
    lifetime=StateLifetime.ACCEPTED_TRIAL,
  )
  port = _new(
    PortBinding,
    port_id="displacement",
    space_id=space.space_id,
    mode=PortMode.COEFFICIENTS,
    coefficient_map=gather,
  )
  residual_channel = _new(
    ResidualChannel,
    channel_id="internal-force",
    target_port_id=port.port_id,
    balance_role=BalanceRole.INTERNAL,
    linear=True,
  )
  jacobian_channel = _new(
    JacobianChannel,
    channel_id="material-tangent",
    residual_channel_id="internal-force",
    target_port_id=port.port_id,
    source_port_id=port.port_id,
    balance_role=BalanceRole.INTERNAL,
    linear=True,
    symmetric=True,
  )
  header = _new(
    OperatorHeader,
    block_id=block_id,
    entity_block_id=entity_block_id,
    implementations=_identities(snapshot),
    ports=(port,),
    signal_ports=(),
    residual_channels=(residual_channel,),
    jacobian_channels=(jacobian_channel,),
    state_layout=state_layout,
    coupling_policy=CouplingPolicy.FIXED,
  )
  payload = _new(
    Q8ContinuumPayload,
    quadrature_points=FinalizedArray(points, dtype=np.float64),
    quadrature_weights=FinalizedArray(weights, dtype=np.float64),
    shape_values=FinalizedArray(shape_values, dtype=np.float64),
    parent_gradients=FinalizedArray(parent_gradients, dtype=np.float64),
    geometry_scales=FinalizedArray(geometry_scales, dtype=np.float64),
    normalized_gradients=FinalizedArray(gradients, dtype=np.float64),
    normalized_strain_displacement=FinalizedArray(b_matrix, dtype=np.float64),
    normalized_integration_weights=FinalizedArray(
      integration_weights, dtype=np.float64
    ),
    constitutive=FinalizedArray(constitutive, dtype=np.float64),
    material_parameters=FinalizedArray(
      [[youngs_modulus, poisson_ratio]], dtype=np.float64
    ),
  )
  manifest = CanonicalManifest(
    {
      "block_id": block_id,
      "entity_block_id": entity_block_id,
      "entity_ids": entity_block.entity_ids,
      "implementations": [
        {
          "kind": item.kind,
          "name": item.name,
          "version": item.version,
          "implementation_id": item.implementation_id,
        }
        for item in header.implementations
      ],
      "port": {
        "port_id": port.port_id,
        "space_id": port.space_id,
        "coefficient_map": port.coefficient_map.values,
      },
      "channels": ["internal-force", "material-tangent"],
      "state": {
        "schema": state_layout.schema,
        "row_width": 0,
        "entity_offsets": state_layout.entity_offsets.values,
      },
      "payload": {
        "quadrature_points": payload.quadrature_points.values,
        "quadrature_weights": payload.quadrature_weights.values,
        "shape_values": payload.shape_values.values,
        "parent_gradients": payload.parent_gradients.values,
        "geometry_scales": payload.geometry_scales.values,
        "normalized_gradients": payload.normalized_gradients.values,
        "normalized_strain_displacement": (
          payload.normalized_strain_displacement.values
        ),
        "normalized_integration_weights": (
          payload.normalized_integration_weights.values
        ),
        "constitutive": payload.constitutive.values,
        "material_parameters": payload.material_parameters.values,
      },
    }
  )
  return entity_block, _new(
    Q8ContinuumOperator,
    header=header,
    entity_block=entity_block,
    payload=payload,
    content_manifest=manifest,
  )


def _compile_mechanical_3d(
  selection: RegionSelection,
  *,
  coordinates: FinalizedArray,
  node_dense: dict[SpecId, int],
  spaces: dict[SpecId, DiscreteSpace],
  snapshot: RegistrySnapshot,
  index_dtype: np.dtype,
  geometry_relative_tolerance: float,
) -> tuple[IncidenceEntityBlock, CompiledOperator]:
  """Compile the three-dimensional linear mechanical slice (hex8, isotropic)."""
  profile = selection.geometry
  label = profile.geometry_interpolation
  space = spaces[selection.fields[0].id]
  connectivity_values = [
    [node_dense[node_id] for node_id in cell.node_ids] for cell in selection.cells
  ]
  connectivity = FinalizedArray(connectivity_values, dtype=index_dtype)
  entity_block_id = selection.block.id
  entity_block = _new(
    IncidenceEntityBlock,
    block_id=entity_block_id,
    entity_ids=tuple(cell.id for cell in selection.cells),
    sources=tuple(_source(cell.source) for cell in selection.cells),
    incidence=connectivity,
  )
  points, weights, shape_values, parent_gradients = _recipes(snapshot, selection)
  gradients, determinants, geometry_scales = _geometry_3d(
    coordinates.values,
    connectivity.values,
    parent_gradients,
    selection.cells,
    geometry_relative_tolerance,
    profile,
  )
  formulation = snapshot.resolve(*profile.formulation_key).binding
  try:
    raw_b_matrix = formulation(np.array(gradients, copy=True))
  except Exception:
    _fail(
      "formulation-binding-failed",
      f"{label} formulation binding failed",
      selection.region.source,
    )
  b_matrix = _binding_array(
    raw_b_matrix,
    shape=(
      len(selection.cells),
      profile.point_count,
      6,
      profile.local_coefficient_count,
    ),
    code="invalid-formulation-binding-output",
    label=f"{label} strain-displacement binding",
    source=selection.region.source,
  )
  expected_b = np.zeros_like(b_matrix)
  expected_b[..., 0, 0::3] = gradients[..., :, 0]
  expected_b[..., 1, 1::3] = gradients[..., :, 1]
  expected_b[..., 2, 2::3] = gradients[..., :, 2]
  expected_b[..., 3, 1::3] = gradients[..., :, 2]
  expected_b[..., 3, 2::3] = gradients[..., :, 1]
  expected_b[..., 4, 0::3] = gradients[..., :, 2]
  expected_b[..., 4, 2::3] = gradients[..., :, 0]
  expected_b[..., 5, 0::3] = gradients[..., :, 1]
  expected_b[..., 5, 1::3] = gradients[..., :, 0]
  if not _corresponds(b_matrix, expected_b):
    _fail(
      "incompatible-formulation-binding-output",
      f"{label} formulation output contradicts the qualified engineering-shear map",
      selection.region.source,
    )
  youngs_modulus, poisson_ratio = _parameters(selection, profile)
  exact_constitutive, binary64_route = _qualified_isotropic_constitutive(
    youngs_modulus,
    poisson_ratio,
    selection.material.source,
  )
  material = snapshot.resolve("material", selection.material.model).binding
  try:
    with warnings.catch_warnings():
      warnings.simplefilter("error", RuntimeWarning)
      raw_constitutive = material(youngs_modulus, poisson_ratio)
  except Exception:
    _fail(
      "material-binding-failed",
      f"{label} material binding failed",
      selection.material.source,
    )
  constitutive = _binding_array(
    raw_constitutive,
    shape=(6, 6),
    code="invalid-material-binding-output",
    label=f"{label} material binding",
    source=selection.material.source,
  )
  if not bool(np.array_equal(constitutive, constitutive.T)):
    _fail(
      "nonsymmetric-material-binding",
      f"{label} material tangent must be symmetric",
      selection.material.source,
    )
  if not _constitutive_corresponds(
    constitutive,
    exact_constitutive,
    binary64_route,
  ):
    _fail(
      "incompatible-material-binding-output",
      f"{label} material output contradicts the qualified isotropic 3D law",
      selection.material.source,
    )
  integration_weights = determinants * weights[None, :]
  _validate_physical_recovery(
    gradients,
    b_matrix,
    integration_weights,
    geometry_scales,
    selection.cells,
    profile,
  )
  with np.errstate(invalid="ignore", over="ignore", under="ignore"):
    tangent = geometry_scales[:, None, None] * np.einsum(
      "ep,epai,ab,epbj->eij",
      integration_weights,
      b_matrix,
      constitutive,
      b_matrix,
      optimize=True,
    )
  if not bool(np.isfinite(tangent).all()):
    _fail(
      "non-finite-element-operator",
      f"{label} element operator is non-finite",
      selection.region.source,
    )
  tangent_scale = float(np.max(np.abs(tangent)))
  symmetry_tolerance = 64.0 * max(
    float(np.finfo(np.float64).eps) * tangent_scale,
    abs(tangent_scale - math.nextafter(tangent_scale, 0.0)),
  )
  if not bool(
    np.allclose(
      tangent,
      tangent.transpose(0, 2, 1),
      rtol=0.0,
      atol=symmetry_tolerance,
    )
  ):
    _fail(
      "nonsymmetric-element-operator",
      f"{label} element operator is not symmetric",
      selection.region.source,
    )

  gather = FinalizedArray(
    _gather_map(space, connectivity.values, node_dense).reshape(
      len(selection.cells), -1
    ),
    dtype=index_dtype,
  )
  block_id = selection.block.id, selection.region.id
  state_layout = _new(
    OperatorStateLayout,
    schema="pyfem-v3-operator-state-layout-v1",
    block_id=block_id,
    entity_count=len(selection.cells),
    slots=(),
    entity_offsets=FinalizedArray(
      np.zeros(len(selection.cells) + 1), dtype=index_dtype
    ),
    row_width=0,
    dtype=np.dtype(np.float64).str,
    lifetime=StateLifetime.ACCEPTED_TRIAL,
  )
  port = _new(
    PortBinding,
    port_id="displacement",
    space_id=space.space_id,
    mode=PortMode.COEFFICIENTS,
    coefficient_map=gather,
  )
  residual_channel = _new(
    ResidualChannel,
    channel_id="internal-force",
    target_port_id=port.port_id,
    balance_role=BalanceRole.INTERNAL,
    linear=True,
  )
  jacobian_channel = _new(
    JacobianChannel,
    channel_id="material-tangent",
    residual_channel_id="internal-force",
    target_port_id=port.port_id,
    source_port_id=port.port_id,
    balance_role=BalanceRole.INTERNAL,
    linear=True,
    symmetric=True,
  )
  header = _new(
    OperatorHeader,
    block_id=block_id,
    entity_block_id=entity_block_id,
    implementations=_identities(snapshot),
    ports=(port,),
    signal_ports=(),
    residual_channels=(residual_channel,),
    jacobian_channels=(jacobian_channel,),
    state_layout=state_layout,
    coupling_policy=CouplingPolicy.FIXED,
  )
  payload = _new(
    Hex8ContinuumPayload,
    quadrature_points=FinalizedArray(points, dtype=np.float64),
    quadrature_weights=FinalizedArray(weights, dtype=np.float64),
    shape_values=FinalizedArray(shape_values, dtype=np.float64),
    parent_gradients=FinalizedArray(parent_gradients, dtype=np.float64),
    geometry_scales=FinalizedArray(geometry_scales, dtype=np.float64),
    normalized_gradients=FinalizedArray(gradients, dtype=np.float64),
    normalized_strain_displacement=FinalizedArray(b_matrix, dtype=np.float64),
    normalized_integration_weights=FinalizedArray(
      integration_weights, dtype=np.float64
    ),
    constitutive=FinalizedArray(constitutive, dtype=np.float64),
    material_parameters=FinalizedArray(
      [[youngs_modulus, poisson_ratio]], dtype=np.float64
    ),
  )
  manifest = CanonicalManifest(
    {
      "block_id": block_id,
      "entity_block_id": entity_block_id,
      "entity_ids": entity_block.entity_ids,
      "implementations": [
        {
          "kind": item.kind,
          "name": item.name,
          "version": item.version,
          "implementation_id": item.implementation_id,
        }
        for item in header.implementations
      ],
      "port": {
        "port_id": port.port_id,
        "space_id": port.space_id,
        "coefficient_map": port.coefficient_map.values,
      },
      "channels": ["internal-force", "material-tangent"],
      "state": {
        "schema": state_layout.schema,
        "row_width": 0,
        "entity_offsets": state_layout.entity_offsets.values,
      },
      "payload": {
        "quadrature_points": payload.quadrature_points.values,
        "quadrature_weights": payload.quadrature_weights.values,
        "shape_values": payload.shape_values.values,
        "parent_gradients": payload.parent_gradients.values,
        "geometry_scales": payload.geometry_scales.values,
        "normalized_gradients": payload.normalized_gradients.values,
        "normalized_strain_displacement": (
          payload.normalized_strain_displacement.values
        ),
        "normalized_integration_weights": (
          payload.normalized_integration_weights.values
        ),
        "constitutive": payload.constitutive.values,
        "material_parameters": payload.material_parameters.values,
      },
    }
  )
  return entity_block, _new(
    Hex8ContinuumOperator,
    header=header,
    entity_block=entity_block,
    payload=payload,
    content_manifest=manifest,
  )


def _flat_binding_array(
  value: object,
  *,
  code: str,
  label: str,
  source: SourceContext,
) -> np.ndarray:
  if (
    type(value) is not np.ndarray
    or value.ndim != 1
    or not value.size
    or value.dtype.metadata is not None
    or value.dtype.kind not in "iuf"
  ):
    _fail(
      code,
      f"{label} must return a metadata-free non-empty 1D numeric array",
      source,
    )
  try:
    captured = np.array(value, dtype=np.float64, order="C", copy=True, subok=False)
  except (OverflowError, TypeError, ValueError):
    _fail(code, f"{label} cannot be represented as float64", source)
  if not bool(np.isfinite(captured).all()):
    _fail(code, f"{label} must contain finite values", source)
  return captured


def _compile_mechanical_stateful(
  selection: RegionSelection,
  *,
  coordinates: FinalizedArray,
  node_dense: dict[SpecId, int],
  spaces: dict[SpecId, DiscreteSpace],
  snapshot: RegistrySnapshot,
  index_dtype: np.dtype,
  geometry_relative_tolerance: float,
) -> tuple[IncidenceEntityBlock, Q8StatefulContinuumOperator]:
  """Compile the generic descriptor-driven stateful small-strain operator.

  Everything numerical comes from the descriptor: the resolved ``state_slots``
  (widths possibly computed from validated parameters) drive the emitted
  ``OperatorStateLayout``, the binding's optional ``initial_state`` wires the
  owner's construction-time rows, and the declared tangent class sets channel
  linearity and symmetry. The optional ``signal_ports`` declarations emit
  ``SignalPortBinding`` values that open the operator's signal slot for exactly
  the declared ports. A virgin-state probe over the initial rows enforces the
  byte-equal no-evolution invariant before the operator can escape.
  """
  space = spaces[selection.fields[0].id]
  connectivity, entity_block = _element_block(selection, node_dense, index_dtype)
  points, weights, shape_values, parent_gradients = _recipes(snapshot, selection)
  gradients, determinants, geometry_scales = _geometry(
    coordinates.values,
    connectivity.values,
    parent_gradients,
    selection.cells,
    geometry_relative_tolerance,
    selection.geometry,
  )
  formulation = snapshot.resolve(*Q8_FORMULATION_KEY).binding
  try:
    raw_b_matrix = formulation(np.array(gradients, copy=True))
  except Exception:
    _fail(
      "formulation-binding-failed",
      "Q8 formulation binding failed",
      selection.region.source,
    )
  b_matrix = _binding_array(
    raw_b_matrix,
    shape=(len(selection.cells), _POINT_COUNT, 3, _LOCAL_COEFFICIENT_COUNT),
    code="invalid-formulation-binding-output",
    label="Q8 strain-displacement binding",
    source=selection.region.source,
  )
  expected_b = np.zeros_like(b_matrix)
  expected_b[..., 0, 0::2] = gradients[..., :, 0]
  expected_b[..., 1, 1::2] = gradients[..., :, 1]
  expected_b[..., 2, 0::2] = gradients[..., :, 1]
  expected_b[..., 2, 1::2] = gradients[..., :, 0]
  if not _corresponds(b_matrix, expected_b):
    _fail(
      "incompatible-formulation-binding-output",
      "Q8 formulation output contradicts the qualified engineering-shear map",
      selection.region.source,
    )

  descriptor = snapshot.resolve("material", selection.material.model)
  binding = descriptor.binding
  kernel = getattr(binding, "kernel", None)
  metadata_binding = getattr(binding, "descriptor_metadata", None)
  if not callable(kernel) or not callable(metadata_binding):
    _fail(
      "malformed-registry-descriptor",
      "stateful material bindings must provide kernel and descriptor_metadata",
      selection.material.source,
    )
  try:
    metadata = metadata_binding()
  except Exception:
    _fail(
      "material-binding-failed",
      "stateful material metadata binding failed",
      selection.material.source,
    )
  try:
    validate_stateful_material_metadata(metadata)
    canonical = CanonicalManifest(metadata)
  except (TypeError, ValueError):
    _fail(
      "malformed-registry-descriptor",
      "stateful material metadata violates the v2 descriptor schema",
      selection.material.source,
    )
  if canonical.to_bytes() != descriptor.metadata.to_bytes():
    _fail(
      "incompatible-registry-descriptor",
      "stateful material binding metadata does not match its descriptor",
      selection.material.source,
    )
  if (
    metadata["stress_voigt_order"] != ["xx", "yy", "xy"]
    or metadata["internal_voigt_order"] != ["xx", "yy", "zz", "yz", "zx", "xy"]
    or metadata["strain_shear_convention"] != "engineering"
  ):
    _fail(
      "incompatible-registry-descriptor",
      "the Q8 stateful slice requires xx/yy/xy stress ports over the 6-Voigt "
      "internal order with engineering shear",
      selection.material.source,
    )
  parameter_names = tuple(metadata["parameter_names"])
  parameter_specs = _scalar_parameters(selection, parameter_names)
  parameter_values = tuple(parameter_specs[name][1] for name in parameter_names)
  parameters_by_name = {name: parameter_specs[name][1] for name in parameter_names}
  try:
    slots = resolve_material_state_slots(metadata["state_slots"], parameters_by_name)
  except (TypeError, ValueError):
    _fail(
      "invalid-material-state-schema",
      "material state slots violate the v2 descriptor schema",
      selection.material.source,
    )
  if "signal_ports" in metadata:
    try:
      signal_declarations = resolve_material_signal_ports(metadata["signal_ports"])
    except (TypeError, ValueError):
      _fail(
        "invalid-material-signal-ports",
        "material signal ports violate the v2 descriptor schema",
        selection.material.source,
      )
  else:
    signal_declarations = ()
  # The derivative channel is declared behaviorally: a binding implementing
  # the optional ``param_derivative_kernel`` member differentiates exactly the
  # descriptor's ``parameter_names``; bindings without it compile
  # byte-identical channel-free operators (contracts.py module docstring).
  param_kernel = getattr(binding, "param_derivative_kernel", None)
  if not callable(param_kernel):
    param_kernel = None
  differentiable_names = parameter_names if param_kernel is not None else ()
  try:
    with warnings.catch_warnings():
      warnings.simplefilter("error", RuntimeWarning)
      raw_calibration = binding(*parameter_values)
  except Exception:
    _fail(
      "material-binding-failed",
      "stateful material calibration binding failed",
      selection.material.source,
    )
  calibration = _flat_binding_array(
    raw_calibration,
    code="invalid-material-binding-output",
    label="stateful material calibration binding",
    source=selection.material.source,
  )
  block_id = selection.block.id, selection.region.id
  entity_count = len(selection.cells) * _POINT_COUNT
  try:
    layout = build_material_state_layout(
      schema_prefix=str(metadata["state_schema"]),
      block_id=block_id,
      entity_count=entity_count,
      slots=slots,
      initial_rows=None,
      index_dtype=index_dtype,
    )
  except (TypeError, ValueError):
    _fail(
      "invalid-material-state-schema",
      "material state schema cannot emit a valid layout",
      selection.material.source,
    )
  initial_binding = getattr(binding, "initial_state", None)
  initial_rows = None
  if initial_binding is not None:
    if not callable(initial_binding):
      _fail(
        "malformed-registry-descriptor",
        "stateful material initial_state binding must be callable",
        selection.material.source,
      )
    try:
      raw_initial = initial_binding(parameter_values, layout)
    except Exception:
      _fail(
        "material-binding-failed",
        "stateful material initial-state binding failed",
        selection.material.source,
      )
    try:
      initial_rows = validated_material_initial_rows(
        raw_initial,
        entity_count=entity_count,
        row_width=layout.row_width,
      )
      layout = build_material_state_layout(
        schema_prefix=str(metadata["state_schema"]),
        block_id=block_id,
        entity_count=entity_count,
        slots=slots,
        initial_rows=initial_rows,
        index_dtype=index_dtype,
      )
    except (TypeError, ValueError):
      _fail(
        "invalid-material-binding-output",
        "stateful material initial-state rows are invalid",
        selection.material.source,
      )
  linear, symmetric = stateful_tangent_channel_flags(metadata["tangent_class"])
  integration_weights = determinants * weights[None, :]
  _validate_physical_recovery(
    gradients,
    b_matrix,
    integration_weights,
    geometry_scales,
    selection.cells,
    selection.geometry,
  )
  gather = FinalizedArray(
    _gather_map(space, connectivity.values, node_dense).reshape(
      len(selection.cells), -1
    ),
    dtype=index_dtype,
  )
  port = _new(
    PortBinding,
    port_id="displacement",
    space_id=space.space_id,
    mode=PortMode.COEFFICIENTS,
    coefficient_map=gather,
  )
  residual_channel = _new(
    ResidualChannel,
    channel_id="internal-force",
    target_port_id=port.port_id,
    balance_role=BalanceRole.INTERNAL,
    linear=linear,
  )
  jacobian_channel = _new(
    JacobianChannel,
    channel_id="material-tangent",
    residual_channel_id=residual_channel.channel_id,
    target_port_id=port.port_id,
    source_port_id=port.port_id,
    balance_role=BalanceRole.INTERNAL,
    linear=linear,
    symmetric=symmetric,
  )
  signal_ports = tuple(
    _new(
      SignalPortBinding,
      port_id=declaration.port_id,
      signal_id=declaration.signal_id,
      derivative_coordinate_ids=declaration.derivative_coordinate_ids,
    )
    for declaration in signal_declarations
  )
  parameters: tuple[ParameterBinding, ...] = ()
  derivative_channels: tuple[ResidualDerivativeChannel, ...] = ()
  channel_fields: dict[str, object] = {}
  if differentiable_names:
    parameters = tuple(
      _new(ParameterBinding, parameter_id=name) for name in differentiable_names
    )
    derivative_channels = tuple(
      _new(
        ResidualDerivativeChannel,
        channel_id=f"dinternal-force/d{name}",
        residual_channel_id=residual_channel.channel_id,
        parameter_id=name,
      )
      for name in differentiable_names
    )
    channel_fields["parameters"] = parameters
    channel_fields["derivative_channels"] = derivative_channels
  header = _new(
    OperatorHeader,
    block_id=block_id,
    entity_block_id=entity_block.block_id,
    implementations=_identities(snapshot),
    ports=(port,),
    signal_ports=signal_ports,
    residual_channels=(residual_channel,),
    jacobian_channels=(jacobian_channel,),
    state_layout=layout,
    coupling_policy=CouplingPolicy.FIXED,
    **channel_fields,
  )
  payload = _new(
    Q8StatefulContinuumPayload,
    quadrature_points=FinalizedArray(points, dtype=np.float64),
    quadrature_weights=FinalizedArray(weights, dtype=np.float64),
    shape_values=FinalizedArray(shape_values, dtype=np.float64),
    parent_gradients=FinalizedArray(parent_gradients, dtype=np.float64),
    geometry_scales=FinalizedArray(geometry_scales, dtype=np.float64),
    normalized_gradients=FinalizedArray(gradients, dtype=np.float64),
    normalized_strain_displacement=FinalizedArray(b_matrix, dtype=np.float64),
    normalized_integration_weights=FinalizedArray(
      integration_weights, dtype=np.float64
    ),
    calibration=FinalizedArray(calibration, dtype=np.float64),
    material_parameters=FinalizedArray([list(parameter_values)], dtype=np.float64),
  )
  manifest_content: dict[str, object] = {
    "block_id": block_id,
    "entity_block_id": entity_block.block_id,
    "entity_ids": entity_block.entity_ids,
    "implementations": [
      {
        "kind": item.kind,
        "name": item.name,
        "version": item.version,
        "implementation_id": item.implementation_id,
      }
      for item in header.implementations
    ],
    "port": {
      "port_id": port.port_id,
      "space_id": port.space_id,
      "coefficient_map": port.coefficient_map.values,
    },
    "channels": ["internal-force", "material-tangent"],
    "state": {
      "schema": layout.schema,
      "row_width": layout.row_width,
      "entity_offsets": layout.entity_offsets.values,
      "slots": [
        {"name": slot.name, "width": slot.width, "annotation": slot.annotation}
        for slot in layout.slots
      ],
      "initial_rows": (
        None if layout.initial_rows is None else layout.initial_rows.values
      ),
    },
    "payload": {
      "quadrature_points": payload.quadrature_points.values,
      "quadrature_weights": payload.quadrature_weights.values,
      "shape_values": payload.shape_values.values,
      "parent_gradients": payload.parent_gradients.values,
      "geometry_scales": payload.geometry_scales.values,
      "normalized_gradients": payload.normalized_gradients.values,
      "normalized_strain_displacement": (payload.normalized_strain_displacement.values),
      "normalized_integration_weights": (payload.normalized_integration_weights.values),
      "calibration": payload.calibration.values,
      "material_parameters": payload.material_parameters.values,
    },
  }
  if signal_ports:
    manifest_content["signal_ports"] = [
      {
        "port_id": signal_port.port_id,
        "signal_id": signal_port.signal_id,
        "derivative_coordinate_ids": list(signal_port.derivative_coordinate_ids),
      }
      for signal_port in signal_ports
    ]
  if differentiable_names:
    manifest_content["parameters"] = [
      {"parameter_id": parameter.parameter_id} for parameter in parameters
    ]
    manifest_content["derivative_channels"] = [
      {
        "channel_id": channel.channel_id,
        "residual_channel_id": channel.residual_channel_id,
        "parameter_id": channel.parameter_id,
      }
      for channel in derivative_channels
    ]
  manifest = CanonicalManifest(manifest_content)
  operator = _new(
    Q8StatefulContinuumOperator,
    header=header,
    entity_block=entity_block,
    payload=payload,
    content_manifest=manifest,
    kernel=kernel,
    param_kernel=param_kernel,
  )

  # The virgin probe mirrors runtime input mutability exactly: read-only
  # accepted rows and fresh writable port values, over the initial state.
  # Ported descriptors probe with zero-valued scalar signals and synthesize
  # identity-style derivative deltas (one where a declared derivative
  # coordinate equals the signal id, zero otherwise) regardless of the port's
  # runtime binding rule: the driver forwards the Kronecker delta keyed on the
  # bound coordinate (identity rule) or on the base coordinate
  # (committed-increment ``d<coordinate>`` rule), and a zero signal with
  # identity-style deltas is a legal input either way, so a time-like law
  # sees its no-elapsed-time evaluation at the origin.
  probe_state = (
    layout.initial_rows
    if layout.initial_rows is not None
    else FinalizedArray(np.zeros(layout.row_shape), dtype=np.float64)
  )
  probe_signals = tuple(
    ProgramSignalInput(
      port_id=signal_port.port_id,
      values=FinalizedArray(np.zeros(1, dtype=np.float64), dtype=np.float64),
      derivatives=tuple(
        SignalDerivativeInput(
          coordinate,
          FinalizedArray(
            np.array(
              [1.0 if coordinate == signal_port.signal_id else 0.0],
              dtype=np.float64,
            ),
            dtype=np.float64,
          ),
        )
        for coordinate in signal_port.derivative_coordinate_ids
      ),
    )
    for signal_port in signal_ports
  )
  try:
    probe = operator.evaluate(
      OperatorEvaluationInput(
        port_values=(
          FinalizedArray(
            np.zeros((len(selection.cells), _LOCAL_COEFFICIENT_COUNT)),
            dtype=np.float64,
          ),
        ),
        accepted_state=probe_state,
        signals=probe_signals,
        # Derivative-capable bindings probe the full derivative channel set
        # at the virgin state: the derivative twin's primal half must
        # reproduce the byte-equal no-evolution invariant and its columns
        # must be finite and complete before the operator can escape.
        request=ChannelRequest(
          ("internal-force",),
          ("material-tangent",),
          tuple(channel.channel_id for channel in derivative_channels),
        ),
      )
    )
  except (TypeError, ValueError):
    _fail(
      "kernel-probe-failed",
      "stateful kernel failed its virgin-state compile probe",
      selection.material.source,
    )
  if evaluation_status(probe) is not EvaluationStatus.OK:
    _fail(
      "invalid-kernel-probe",
      "stateful kernel must evaluate its virgin initial state successfully",
      selection.material.source,
    )
  if probe.trial_state.values.tobytes() != probe_state.values.tobytes():
    _fail(
      "invalid-kernel-probe",
      "stateful kernel virgin probe must return the accepted rows byte-equal",
      selection.material.source,
    )
  probe_tangent = probe.jacobian_values[0].values
  if not bool(np.isfinite(probe_tangent).all()):
    _fail(
      "invalid-kernel-probe",
      "stateful kernel virgin tangent must be finite",
      selection.material.source,
    )
  if symmetric:
    tangent_scale = float(np.max(np.abs(probe_tangent)))
    symmetry_tolerance = 64.0 * max(
      float(np.finfo(np.float64).eps) * tangent_scale,
      abs(tangent_scale - math.nextafter(tangent_scale, 0.0)),
    )
    if not bool(
      np.allclose(
        probe_tangent,
        probe_tangent.transpose(0, 2, 1),
        rtol=0.0,
        atol=symmetry_tolerance,
      )
    ):
      _fail(
        "invalid-kernel-probe",
        "stateful kernel virgin tangent must be symmetric for its declared class",
        selection.material.source,
      )
  if derivative_channels:
    probe_derivatives = evaluation_derivative_values(probe)
    if len(probe_derivatives) != len(derivative_channels) or any(
      block.values.shape != (len(selection.cells), _LOCAL_COEFFICIENT_COUNT)
      or not bool(np.isfinite(block.values).all())
      for block in probe_derivatives
    ):
      _fail(
        "invalid-kernel-probe",
        "stateful kernel virgin probe derivative values must be finite and complete",
        selection.material.source,
      )
  return entity_block, operator


_TL_PROBE_GRADIENT = ((0.04, 0.02), (0.03, 0.07))


def _tl_probe_states(coordinates: np.ndarray) -> np.ndarray:
  """Fixed formulation-binding probes: zero, then one uniform displacement gradient.

  The nonzero probe applies the affine field ``u(X) = H X`` per cell, which
  the Q8 isoparametric map reproduces exactly, so the binding and the
  qualified reference assembly meet at a genuinely deformed finite-strain
  state (uniform ``F = I + H`` per cell).
  """
  count = coordinates.shape[0]
  gradient = np.array(_TL_PROBE_GRADIENT, dtype=np.float64)
  states = np.zeros((2 * count, _LOCAL_COEFFICIENT_COUNT), dtype=np.float64)
  states[count:] = (coordinates @ gradient.T).reshape(count, _LOCAL_COEFFICIENT_COUNT)
  return states


def _tl_reference_response(
  gradients: np.ndarray,
  integration_weights: np.ndarray,
  geometry_scales: np.ndarray,
  constitutive: np.ndarray,
  states: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  """Qualified TL element response assembled from the validated recipes.

  The compiler's independent reference for the formulation-binding probe, in
  the idiom of the truss family's ``_expected_kinematics``: physical shape
  gradients (normalized gradients rescaled), deformation gradient,
  Green-Lagrange strain, PK2 stress, and the material + geometric tangent
  accumulated point by point from the trusted ingredients.
  """
  probe_count = states.shape[0]
  cell_count = gradients.shape[0]
  tangent = np.zeros(
    (probe_count, _LOCAL_COEFFICIENT_COUNT, _LOCAL_COEFFICIENT_COUNT),
    dtype=np.float64,
  )
  force = np.zeros((probe_count, _LOCAL_COEFFICIENT_COUNT), dtype=np.float64)
  for probe in range(probe_count):
    cell = probe % cell_count
    scale = geometry_scales[cell]
    displacement = states[probe].reshape(_NODE_COUNT, 2)
    for point in range(_POINT_COUNT):
      gradient = gradients[cell, point] / scale
      weight = integration_weights[cell, point] * scale * scale
      deformation = np.eye(2) + displacement.T @ gradient
      right_cauchy_green = deformation.T @ deformation
      strain = np.array(
        [
          0.5 * (right_cauchy_green[0, 0] - 1.0),
          0.5 * (right_cauchy_green[1, 1] - 1.0),
          right_cauchy_green[0, 1],
        ],
        dtype=np.float64,
      )
      stress = constitutive @ strain
      b_matrix = np.zeros((3, _LOCAL_COEFFICIENT_COUNT), dtype=np.float64)
      b_matrix[0, 0::2] = gradient[:, 0] * deformation[0, 0]
      b_matrix[0, 1::2] = gradient[:, 0] * deformation[1, 0]
      b_matrix[1, 0::2] = gradient[:, 1] * deformation[0, 1]
      b_matrix[1, 1::2] = gradient[:, 1] * deformation[1, 1]
      b_matrix[2, 0::2] = (
        gradient[:, 1] * deformation[0, 0] + gradient[:, 0] * deformation[0, 1]
      )
      b_matrix[2, 1::2] = (
        gradient[:, 0] * deformation[1, 1] + gradient[:, 1] * deformation[1, 0]
      )
      b_nl = np.zeros((4, _LOCAL_COEFFICIENT_COUNT), dtype=np.float64)
      b_nl[0, 0::2] = gradient[:, 0]
      b_nl[1, 0::2] = gradient[:, 1]
      b_nl[2, 1::2] = gradient[:, 0]
      b_nl[3, 1::2] = gradient[:, 1]
      stress_matrix = np.array(
        [
          [stress[0], stress[2], 0.0, 0.0],
          [stress[2], stress[1], 0.0, 0.0],
          [0.0, 0.0, stress[0], stress[2]],
          [0.0, 0.0, stress[2], stress[1]],
        ],
        dtype=np.float64,
      )
      tangent[probe] += weight * (
        b_matrix.T @ (constitutive @ b_matrix) + b_nl.T @ (stress_matrix @ b_nl)
      )
      force[probe] += weight * (b_matrix.T @ stress)
  return tangent, force


def _compile_mechanical_tl(
  selection: RegionSelection,
  *,
  coordinates: FinalizedArray,
  node_dense: dict[SpecId, int],
  spaces: dict[SpecId, DiscreteSpace],
  snapshot: RegistrySnapshot,
  index_dtype: np.dtype,
  geometry_relative_tolerance: float,
) -> tuple[IncidenceEntityBlock, Q8FiniteStrainOperator]:
  """Compile the finite-strain (total-Lagrangian) Q8 plane-stress slice.

  The payload keeps the reference geometry scale-normalized (translation-free
  node coordinates plus the per-cell scale); evaluation rescales them and
  calls the descriptor-bound landed TL kernel. The kernel is probed at
  compile time against :func:`_tl_reference_response` — at the zero state and
  at one fixed uniform displacement-gradient state — so a binding whose
  response contradicts the qualified reference never escapes compilation.
  """
  space = spaces[selection.fields[0].id]
  connectivity, entity_block = _element_block(selection, node_dense, index_dtype)
  points, weights, shape_values, parent_gradients = _recipes(snapshot, selection)
  gradients, determinants, geometry_scales = _geometry(
    coordinates.values,
    connectivity.values,
    parent_gradients,
    selection.cells,
    geometry_relative_tolerance,
    selection.geometry,
  )
  expected_b = np.zeros(
    (len(selection.cells), _POINT_COUNT, 3, _LOCAL_COEFFICIENT_COUNT),
    dtype=np.float64,
  )
  expected_b[..., 0, 0::2] = gradients[..., :, 0]
  expected_b[..., 1, 1::2] = gradients[..., :, 1]
  expected_b[..., 2, 0::2] = gradients[..., :, 1]
  expected_b[..., 2, 1::2] = gradients[..., :, 0]
  youngs_modulus, poisson_ratio = _parameters(selection, selection.geometry)
  exact_constitutive, binary64_route = _qualified_constitutive(
    youngs_modulus,
    poisson_ratio,
    selection.material.source,
  )
  material = snapshot.resolve("material", selection.material.model).binding
  try:
    with warnings.catch_warnings():
      warnings.simplefilter("error", RuntimeWarning)
      raw_constitutive = material(youngs_modulus, poisson_ratio)
  except Exception:
    _fail(
      "material-binding-failed",
      "total-Lagrangian material binding failed",
      selection.material.source,
    )
  constitutive = _binding_array(
    raw_constitutive,
    shape=(3, 3),
    code="invalid-material-binding-output",
    label="total-Lagrangian material binding",
    source=selection.material.source,
  )
  if not bool(np.array_equal(constitutive, constitutive.T)):
    _fail(
      "nonsymmetric-material-binding",
      "total-Lagrangian material tangent must be symmetric",
      selection.material.source,
    )
  if not _constitutive_corresponds(
    constitutive,
    exact_constitutive,
    binary64_route,
  ):
    _fail(
      "incompatible-material-binding-output",
      "total-Lagrangian material output contradicts the qualified plane-stress "
      "Saint-Venant-Kirchhoff law",
      selection.material.source,
    )
  integration_weights = determinants * weights[None, :]
  _validate_physical_recovery(
    gradients,
    expected_b,
    integration_weights,
    geometry_scales,
    selection.cells,
    selection.geometry,
  )
  normalized_nodes = np.empty((len(selection.cells), _NODE_COUNT, 2), dtype=np.float64)
  for cell_index, cell in enumerate(selection.cells):
    normalized_nodes[cell_index] = _normalized_coordinates(
      coordinates.values[connectivity.values[cell_index]],
      cell,
    )[0]
  physical_nodes = normalized_nodes * geometry_scales[:, None, None]
  if not bool(np.isfinite(physical_nodes).all()):
    _fail(
      "non-finite-reference-geometry",
      "total-Lagrangian physical reference geometry is non-finite",
      selection.region.source,
    )

  formulation = snapshot.resolve(*TL_FORMULATION_KEY).binding
  if not callable(formulation):
    _fail(
      "malformed-registry-descriptor",
      "the total-Lagrangian formulation binding must be callable",
      selection.region.source,
    )
  probe_states = _tl_probe_states(physical_nodes)
  probe_coordinates = np.concatenate((physical_nodes, physical_nodes))
  try:
    raw_probe = formulation(probe_coordinates, probe_states, constitutive)
  except Exception:
    _fail(
      "formulation-binding-failed",
      "total-Lagrangian formulation binding failed",
      selection.region.source,
    )
  if type(raw_probe) is not tuple or len(raw_probe) != 2:
    _fail(
      "invalid-formulation-binding-output",
      "total-Lagrangian formulation must return exactly the tangent and the "
      "internal force",
      selection.region.source,
    )
  probe_tangent = _binding_array(
    raw_probe[0],
    shape=(
      2 * len(selection.cells),
      _LOCAL_COEFFICIENT_COUNT,
      _LOCAL_COEFFICIENT_COUNT,
    ),
    code="invalid-formulation-binding-output",
    label="total-Lagrangian tangent binding",
    source=selection.region.source,
  )
  probe_force = _binding_array(
    raw_probe[1],
    shape=(2 * len(selection.cells), _LOCAL_COEFFICIENT_COUNT),
    code="invalid-formulation-binding-output",
    label="total-Lagrangian internal force binding",
    source=selection.region.source,
  )
  reference_tangent, reference_force = _tl_reference_response(
    gradients,
    integration_weights,
    geometry_scales,
    constitutive,
    probe_states,
  )
  if not _tl_probe_corresponds(probe_tangent, reference_tangent):
    _fail(
      "incompatible-formulation-binding-output",
      "total-Lagrangian formulation tangent contradicts the qualified "
      "total-Lagrangian reference assembly",
      selection.region.source,
    )
  if not _tl_probe_corresponds(probe_force, reference_force):
    _fail(
      "incompatible-formulation-binding-output",
      "total-Lagrangian formulation internal force contradicts the qualified "
      "total-Lagrangian reference assembly",
      selection.region.source,
    )
  tangent_scale = float(np.max(np.abs(probe_tangent)))
  symmetry_tolerance = 64.0 * max(
    float(np.finfo(np.float64).eps) * tangent_scale,
    abs(tangent_scale - math.nextafter(tangent_scale, 0.0)),
  )
  if not bool(
    np.allclose(
      probe_tangent,
      probe_tangent.transpose(0, 2, 1),
      rtol=0.0,
      atol=symmetry_tolerance,
    )
  ):
    _fail(
      "nonsymmetric-element-operator",
      "total-Lagrangian element operator is not symmetric",
      selection.region.source,
    )

  gather = FinalizedArray(
    _gather_map(space, connectivity.values, node_dense).reshape(
      len(selection.cells), -1
    ),
    dtype=index_dtype,
  )
  block_id = selection.block.id, selection.region.id
  state_layout = _zero_width_state_layout(block_id, len(selection.cells), index_dtype)
  port = _new(
    PortBinding,
    port_id="displacement",
    space_id=space.space_id,
    mode=PortMode.COEFFICIENTS,
    coefficient_map=gather,
  )
  residual_channel = _new(
    ResidualChannel,
    channel_id="internal-force",
    target_port_id=port.port_id,
    balance_role=BalanceRole.INTERNAL,
    linear=False,
  )
  jacobian_channel = _new(
    JacobianChannel,
    channel_id="material-tangent",
    residual_channel_id=residual_channel.channel_id,
    target_port_id=port.port_id,
    source_port_id=port.port_id,
    balance_role=BalanceRole.INTERNAL,
    linear=False,
    symmetric=True,
  )
  header = _new(
    OperatorHeader,
    block_id=block_id,
    entity_block_id=entity_block.block_id,
    implementations=_identities(snapshot),
    ports=(port,),
    signal_ports=(),
    residual_channels=(residual_channel,),
    jacobian_channels=(jacobian_channel,),
    state_layout=state_layout,
    coupling_policy=CouplingPolicy.FIXED,
  )
  payload = _new(
    Q8FiniteStrainPayload,
    quadrature_points=FinalizedArray(points, dtype=np.float64),
    quadrature_weights=FinalizedArray(weights, dtype=np.float64),
    shape_values=FinalizedArray(shape_values, dtype=np.float64),
    parent_gradients=FinalizedArray(parent_gradients, dtype=np.float64),
    geometry_scales=FinalizedArray(geometry_scales, dtype=np.float64),
    normalized_node_coordinates=FinalizedArray(normalized_nodes, dtype=np.float64),
    constitutive=FinalizedArray(constitutive, dtype=np.float64),
    material_parameters=FinalizedArray(
      [[youngs_modulus, poisson_ratio]], dtype=np.float64
    ),
  )
  manifest = CanonicalManifest(
    {
      "block_id": block_id,
      "entity_block_id": entity_block.block_id,
      "entity_ids": entity_block.entity_ids,
      "implementations": [
        {
          "kind": item.kind,
          "name": item.name,
          "version": item.version,
          "implementation_id": item.implementation_id,
        }
        for item in header.implementations
      ],
      "port": {
        "port_id": port.port_id,
        "space_id": port.space_id,
        "coefficient_map": port.coefficient_map.values,
      },
      "channels": ["internal-force", "material-tangent"],
      "state": {
        "schema": state_layout.schema,
        "row_width": 0,
        "entity_offsets": state_layout.entity_offsets.values,
      },
      "payload": {
        "quadrature_points": payload.quadrature_points.values,
        "quadrature_weights": payload.quadrature_weights.values,
        "shape_values": payload.shape_values.values,
        "parent_gradients": payload.parent_gradients.values,
        "geometry_scales": payload.geometry_scales.values,
        "normalized_node_coordinates": payload.normalized_node_coordinates.values,
        "constitutive": payload.constitutive.values,
        "material_parameters": payload.material_parameters.values,
      },
    }
  )
  return entity_block, _new(
    Q8FiniteStrainOperator,
    header=header,
    entity_block=entity_block,
    payload=payload,
    content_manifest=manifest,
    kernel=formulation,
  )


def _element_block(
  selection: RegionSelection,
  node_dense: dict[SpecId, int],
  index_dtype: np.dtype,
) -> tuple[FinalizedArray, IncidenceEntityBlock]:
  connectivity_values = [
    [node_dense[node_id] for node_id in cell.node_ids] for cell in selection.cells
  ]
  connectivity = FinalizedArray(connectivity_values, dtype=index_dtype)
  entity_block = _new(
    IncidenceEntityBlock,
    block_id=selection.block.id,
    entity_ids=tuple(cell.id for cell in selection.cells),
    sources=tuple(_source(cell.source) for cell in selection.cells),
    incidence=connectivity,
  )
  return connectivity, entity_block


def _zero_width_state_layout(
  block_id: tuple[SpecId, SpecId],
  entity_count: int,
  index_dtype: np.dtype,
) -> OperatorStateLayout:
  return _new(
    OperatorStateLayout,
    schema="pyfem-v3-operator-state-layout-v1",
    block_id=block_id,
    entity_count=entity_count,
    slots=(),
    entity_offsets=FinalizedArray(np.zeros(entity_count + 1), dtype=index_dtype),
    row_width=0,
    dtype=np.dtype(np.float64).str,
    lifetime=StateLifetime.ACCEPTED_TRIAL,
  )


def _compile_thermal(
  selection: RegionSelection,
  *,
  coordinates: FinalizedArray,
  node_dense: dict[SpecId, int],
  spaces: dict[SpecId, DiscreteSpace],
  snapshot: RegistrySnapshot,
  index_dtype: np.dtype,
  geometry_relative_tolerance: float,
) -> tuple[IncidenceEntityBlock, Q8ThermalOperator]:
  space = spaces[selection.fields[0].id]
  connectivity, entity_block = _element_block(selection, node_dense, index_dtype)
  points, weights, shape_values, parent_gradients = _recipes(snapshot, selection)
  gradients, determinants, geometry_scales = _geometry(
    coordinates.values,
    connectivity.values,
    parent_gradients,
    selection.cells,
    geometry_relative_tolerance,
    selection.geometry,
  )
  formulation = snapshot.resolve(*THERMAL_FORMULATION_KEY).binding
  try:
    raw_gradients = formulation(np.array(gradients, copy=True))
  except Exception:
    _fail(
      "formulation-binding-failed",
      "thermal formulation binding failed",
      selection.region.source,
    )
  temperature_gradients = _binding_array(
    raw_gradients,
    shape=(len(selection.cells), _POINT_COUNT, 2, _NODE_COUNT),
    code="invalid-formulation-binding-output",
    label="thermal temperature-gradient binding",
    source=selection.region.source,
  )
  if not _corresponds(temperature_gradients, np.swapaxes(gradients, 2, 3)):
    _fail(
      "incompatible-formulation-binding-output",
      "thermal formulation output contradicts the qualified temperature-gradient map",
      selection.region.source,
    )
  conductivity = _thermal_parameters(selection)
  material = snapshot.resolve(*THERMAL_MATERIAL_KEY).binding
  try:
    with warnings.catch_warnings():
      warnings.simplefilter("error", RuntimeWarning)
      raw_conduction = material(conductivity)
  except Exception:
    _fail(
      "material-binding-failed",
      "thermal material binding failed",
      selection.material.source,
    )
  conduction = _binding_array(
    raw_conduction,
    shape=(2, 2),
    code="invalid-material-binding-output",
    label="thermal material binding",
    source=selection.material.source,
  )
  if not bool(np.array_equal(conduction, conduction.T)):
    _fail(
      "nonsymmetric-material-binding",
      "thermal conductivity must be symmetric",
      selection.material.source,
    )
  qualified = np.array(
    [[conductivity, 0.0], [0.0, conductivity]],
    dtype=np.float64,
  )
  if not _constitutive_corresponds(conduction, qualified, qualified):
    _fail(
      "incompatible-material-binding-output",
      "thermal material output contradicts the qualified isotropic conductor",
      selection.material.source,
    )
  integration_weights = determinants * weights[None, :]
  _validate_physical_recovery(
    gradients,
    np.swapaxes(gradients, 2, 3),
    integration_weights,
    geometry_scales,
    selection.cells,
    selection.geometry,
  )
  with np.errstate(invalid="ignore", over="ignore", under="ignore"):
    tangent = np.einsum(
      "ep,epai,ab,epbj->eij",
      integration_weights,
      temperature_gradients,
      conduction,
      temperature_gradients,
      optimize=True,
    )
  if not bool(np.isfinite(tangent).all()):
    _fail(
      "non-finite-element-operator",
      "thermal element operator is non-finite",
      selection.region.source,
    )
  tangent_scale = float(np.max(np.abs(tangent)))
  symmetry_tolerance = 64.0 * max(
    float(np.finfo(np.float64).eps) * tangent_scale,
    abs(tangent_scale - math.nextafter(tangent_scale, 0.0)),
  )
  if not bool(
    np.allclose(
      tangent,
      tangent.transpose(0, 2, 1),
      rtol=0.0,
      atol=symmetry_tolerance,
    )
  ):
    _fail(
      "nonsymmetric-element-operator",
      "thermal element operator is not symmetric",
      selection.region.source,
    )

  gather = FinalizedArray(
    _gather_map(space, connectivity.values, node_dense).reshape(
      len(selection.cells), -1
    ),
    dtype=index_dtype,
  )
  block_id = selection.block.id, selection.region.id
  state_layout = _zero_width_state_layout(block_id, len(selection.cells), index_dtype)
  port = _new(
    PortBinding,
    port_id="temperature",
    space_id=space.space_id,
    mode=PortMode.COEFFICIENTS,
    coefficient_map=gather,
  )
  residual_channel = _new(
    ResidualChannel,
    channel_id="heat-flux",
    target_port_id=port.port_id,
    balance_role=BalanceRole.INTERNAL,
    linear=True,
  )
  jacobian_channel = _new(
    JacobianChannel,
    channel_id="conduction-tangent",
    residual_channel_id="heat-flux",
    target_port_id=port.port_id,
    source_port_id=port.port_id,
    balance_role=BalanceRole.INTERNAL,
    linear=True,
    symmetric=True,
  )
  header = _new(
    OperatorHeader,
    block_id=block_id,
    entity_block_id=entity_block.block_id,
    implementations=_identities(snapshot),
    ports=(port,),
    signal_ports=(),
    residual_channels=(residual_channel,),
    jacobian_channels=(jacobian_channel,),
    state_layout=state_layout,
    coupling_policy=CouplingPolicy.FIXED,
  )
  payload = _new(
    Q8ThermalPayload,
    quadrature_points=FinalizedArray(points, dtype=np.float64),
    quadrature_weights=FinalizedArray(weights, dtype=np.float64),
    shape_values=FinalizedArray(shape_values, dtype=np.float64),
    parent_gradients=FinalizedArray(parent_gradients, dtype=np.float64),
    geometry_scales=FinalizedArray(geometry_scales, dtype=np.float64),
    normalized_gradients=FinalizedArray(gradients, dtype=np.float64),
    normalized_temperature_gradients=FinalizedArray(
      temperature_gradients, dtype=np.float64
    ),
    normalized_integration_weights=FinalizedArray(
      integration_weights, dtype=np.float64
    ),
    conductivity=FinalizedArray(conduction, dtype=np.float64),
    material_parameters=FinalizedArray([[conductivity]], dtype=np.float64),
  )
  manifest = CanonicalManifest(
    {
      "block_id": block_id,
      "entity_block_id": entity_block.block_id,
      "entity_ids": entity_block.entity_ids,
      "implementations": [
        {
          "kind": item.kind,
          "name": item.name,
          "version": item.version,
          "implementation_id": item.implementation_id,
        }
        for item in header.implementations
      ],
      "port": {
        "port_id": port.port_id,
        "space_id": port.space_id,
        "coefficient_map": port.coefficient_map.values,
      },
      "channels": ["heat-flux", "conduction-tangent"],
      "state": {
        "schema": state_layout.schema,
        "row_width": 0,
        "entity_offsets": state_layout.entity_offsets.values,
      },
      "payload": {
        "quadrature_points": payload.quadrature_points.values,
        "quadrature_weights": payload.quadrature_weights.values,
        "shape_values": payload.shape_values.values,
        "parent_gradients": payload.parent_gradients.values,
        "geometry_scales": payload.geometry_scales.values,
        "normalized_gradients": payload.normalized_gradients.values,
        "normalized_temperature_gradients": (
          payload.normalized_temperature_gradients.values
        ),
        "normalized_integration_weights": (
          payload.normalized_integration_weights.values
        ),
        "conductivity": payload.conductivity.values,
        "material_parameters": payload.material_parameters.values,
      },
    }
  )
  return entity_block, _new(
    Q8ThermalOperator,
    header=header,
    entity_block=entity_block,
    payload=payload,
    content_manifest=manifest,
  )


def _compile_coupled(
  selection: RegionSelection,
  *,
  coordinates: FinalizedArray,
  node_dense: dict[SpecId, int],
  spaces: dict[SpecId, DiscreteSpace],
  snapshot: RegistrySnapshot,
  index_dtype: np.dtype,
  geometry_relative_tolerance: float,
) -> tuple[IncidenceEntityBlock, Q8ThermoElasticOperator]:
  displacement_space = spaces[selection.fields[0].id]
  temperature_space = spaces[selection.fields[1].id]
  connectivity, entity_block = _element_block(selection, node_dense, index_dtype)
  points, weights, shape_values, parent_gradients = _recipes(snapshot, selection)
  gradients, determinants, geometry_scales = _geometry(
    coordinates.values,
    connectivity.values,
    parent_gradients,
    selection.cells,
    geometry_relative_tolerance,
    selection.geometry,
  )
  formulation = snapshot.resolve(*THERMO_FORMULATION_KEY).binding
  try:
    raw_kinematics = formulation(np.array(gradients, copy=True))
  except Exception:
    _fail(
      "formulation-binding-failed",
      "coupled formulation binding failed",
      selection.region.source,
    )
  if type(raw_kinematics) is not tuple or len(raw_kinematics) != 2:
    _fail(
      "invalid-formulation-binding-output",
      "coupled formulation must return exactly strain-displacement and "
      "temperature-gradient maps",
      selection.region.source,
    )
  b_matrix = _binding_array(
    raw_kinematics[0],
    shape=(len(selection.cells), _POINT_COUNT, 3, _LOCAL_COEFFICIENT_COUNT),
    code="invalid-formulation-binding-output",
    label="coupled strain-displacement binding",
    source=selection.region.source,
  )
  temperature_gradients = _binding_array(
    raw_kinematics[1],
    shape=(len(selection.cells), _POINT_COUNT, 2, _NODE_COUNT),
    code="invalid-formulation-binding-output",
    label="coupled temperature-gradient binding",
    source=selection.region.source,
  )
  expected_b = np.zeros_like(b_matrix)
  expected_b[..., 0, 0::2] = gradients[..., :, 0]
  expected_b[..., 1, 1::2] = gradients[..., :, 1]
  expected_b[..., 2, 0::2] = gradients[..., :, 1]
  expected_b[..., 2, 1::2] = gradients[..., :, 0]
  if not _corresponds(b_matrix, expected_b) or not _corresponds(
    temperature_gradients, np.swapaxes(gradients, 2, 3)
  ):
    _fail(
      "incompatible-formulation-binding-output",
      "coupled formulation output contradicts the qualified kinematic maps",
      selection.region.source,
    )
  youngs_modulus, poisson_ratio, thermal_expansion, conductivity = (
    _thermo_elastic_parameters(selection)
  )
  exact_constitutive, binary64_route = _qualified_constitutive(
    youngs_modulus,
    poisson_ratio,
    selection.material.source,
  )
  material = snapshot.resolve(*THERMO_MATERIAL_KEY).binding
  try:
    with warnings.catch_warnings():
      warnings.simplefilter("error", RuntimeWarning)
      raw_response = material(
        youngs_modulus,
        poisson_ratio,
        thermal_expansion,
        conductivity,
      )
  except Exception:
    _fail(
      "material-binding-failed",
      "coupled material binding failed",
      selection.material.source,
    )
  if type(raw_response) is not tuple or len(raw_response) != 3:
    _fail(
      "invalid-material-binding-output",
      "coupled material must return exactly the constitutive matrix, the "
      "thermal-expansion vector, and the conductivity matrix",
      selection.material.source,
    )
  constitutive = _binding_array(
    raw_response[0],
    shape=(3, 3),
    code="invalid-material-binding-output",
    label="coupled constitutive binding",
    source=selection.material.source,
  )
  expansion = _binding_array(
    raw_response[1],
    shape=(3,),
    code="invalid-material-binding-output",
    label="coupled thermal-expansion binding",
    source=selection.material.source,
  )
  conduction = _binding_array(
    raw_response[2],
    shape=(2, 2),
    code="invalid-material-binding-output",
    label="coupled conductivity binding",
    source=selection.material.source,
  )
  if not bool(np.array_equal(constitutive, constitutive.T)) or not bool(
    np.array_equal(conduction, conduction.T)
  ):
    _fail(
      "nonsymmetric-material-binding",
      "coupled constitutive and conductivity matrices must be symmetric",
      selection.material.source,
    )
  if not _constitutive_corresponds(
    constitutive,
    exact_constitutive,
    binary64_route,
  ):
    _fail(
      "incompatible-material-binding-output",
      "coupled constitutive output contradicts the qualified plane-stress law",
      selection.material.source,
    )
  qualified_expansion = np.array(
    [thermal_expansion, thermal_expansion, 0.0],
    dtype=np.float64,
  )
  if not _constitutive_corresponds(
    expansion,
    qualified_expansion,
    qualified_expansion,
  ):
    _fail(
      "incompatible-material-binding-output",
      "coupled thermal expansion contradicts the qualified isotropic dilatation",
      selection.material.source,
    )
  qualified_conduction = np.array(
    [[conductivity, 0.0], [0.0, conductivity]],
    dtype=np.float64,
  )
  if not _constitutive_corresponds(
    conduction,
    qualified_conduction,
    qualified_conduction,
  ):
    _fail(
      "incompatible-material-binding-output",
      "coupled conductivity contradicts the qualified isotropic conductor",
      selection.material.source,
    )
  integration_weights = determinants * weights[None, :]
  _validate_physical_recovery(
    gradients,
    b_matrix,
    integration_weights,
    geometry_scales,
    selection.cells,
    selection.geometry,
  )
  with np.errstate(invalid="ignore", over="ignore", under="ignore"):
    tangent_uu = np.einsum(
      "ep,epai,ab,epbj->eij",
      integration_weights,
      b_matrix,
      constitutive,
      b_matrix,
      optimize=True,
    )
    dilatation = constitutive @ expansion
    tangent_ut = geometry_scales[:, None, None] * np.einsum(
      "ep,epai,a,pb->eib",
      integration_weights,
      b_matrix,
      dilatation,
      shape_values,
      optimize=True,
    )
    tangent_tt = np.einsum(
      "ep,epai,ab,epbj->eij",
      integration_weights,
      temperature_gradients,
      conduction,
      temperature_gradients,
      optimize=True,
    )
  if not bool(np.isfinite(tangent_uu).all()) or not bool(np.isfinite(tangent_tt).all()):
    _fail(
      "non-finite-element-operator",
      "coupled element operator is non-finite",
      selection.region.source,
    )
  if not bool(np.isfinite(tangent_ut).all()):
    _fail(
      "non-finite-element-operator",
      "coupled thermal-expansion operator is non-finite",
      selection.region.source,
    )
  tangent_scale = float(np.max(np.abs(tangent_uu)))
  symmetry_tolerance = 64.0 * max(
    float(np.finfo(np.float64).eps) * tangent_scale,
    abs(tangent_scale - math.nextafter(tangent_scale, 0.0)),
  )
  if not bool(
    np.allclose(
      tangent_uu,
      tangent_uu.transpose(0, 2, 1),
      rtol=0.0,
      atol=symmetry_tolerance,
    )
  ):
    _fail(
      "nonsymmetric-element-operator",
      "coupled mechanical element operator is not symmetric",
      selection.region.source,
    )
  thermal_scale = float(np.max(np.abs(tangent_tt)))
  thermal_tolerance = 64.0 * max(
    float(np.finfo(np.float64).eps) * thermal_scale,
    abs(thermal_scale - math.nextafter(thermal_scale, 0.0)),
  )
  if not bool(
    np.allclose(
      tangent_tt,
      tangent_tt.transpose(0, 2, 1),
      rtol=0.0,
      atol=thermal_tolerance,
    )
  ):
    _fail(
      "nonsymmetric-element-operator",
      "coupled thermal element operator is not symmetric",
      selection.region.source,
    )

  displacement_gather = FinalizedArray(
    _gather_map(displacement_space, connectivity.values, node_dense).reshape(
      len(selection.cells), -1
    ),
    dtype=index_dtype,
  )
  temperature_gather = FinalizedArray(
    _gather_map(temperature_space, connectivity.values, node_dense).reshape(
      len(selection.cells), -1
    ),
    dtype=index_dtype,
  )
  block_id = selection.block.id, selection.region.id
  state_layout = _zero_width_state_layout(block_id, len(selection.cells), index_dtype)
  displacement_port = _new(
    PortBinding,
    port_id="displacement",
    space_id=displacement_space.space_id,
    mode=PortMode.COEFFICIENTS,
    coefficient_map=displacement_gather,
  )
  temperature_port = _new(
    PortBinding,
    port_id="temperature",
    space_id=temperature_space.space_id,
    mode=PortMode.COEFFICIENTS,
    coefficient_map=temperature_gather,
  )
  force_channel = _new(
    ResidualChannel,
    channel_id="internal-force",
    target_port_id=displacement_port.port_id,
    balance_role=BalanceRole.INTERNAL,
    linear=True,
  )
  heat_channel = _new(
    ResidualChannel,
    channel_id="heat-flux",
    target_port_id=temperature_port.port_id,
    balance_role=BalanceRole.INTERNAL,
    linear=True,
  )
  mechanical_jacobian = _new(
    JacobianChannel,
    channel_id="material-tangent",
    residual_channel_id="internal-force",
    target_port_id=displacement_port.port_id,
    source_port_id=displacement_port.port_id,
    balance_role=BalanceRole.INTERNAL,
    linear=True,
    symmetric=True,
  )
  coupling_jacobian = _new(
    JacobianChannel,
    channel_id="thermal-expansion-tangent",
    residual_channel_id="internal-force",
    target_port_id=displacement_port.port_id,
    source_port_id=temperature_port.port_id,
    balance_role=BalanceRole.INTERNAL,
    linear=True,
    symmetric=False,
  )
  thermal_jacobian = _new(
    JacobianChannel,
    channel_id="conduction-tangent",
    residual_channel_id="heat-flux",
    target_port_id=temperature_port.port_id,
    source_port_id=temperature_port.port_id,
    balance_role=BalanceRole.INTERNAL,
    linear=True,
    symmetric=True,
  )
  header = _new(
    OperatorHeader,
    block_id=block_id,
    entity_block_id=entity_block.block_id,
    implementations=_identities(snapshot),
    ports=(displacement_port, temperature_port),
    signal_ports=(),
    residual_channels=(force_channel, heat_channel),
    jacobian_channels=(mechanical_jacobian, coupling_jacobian, thermal_jacobian),
    state_layout=state_layout,
    coupling_policy=CouplingPolicy.FIXED,
  )
  payload = _new(
    Q8ThermoElasticPayload,
    quadrature_points=FinalizedArray(points, dtype=np.float64),
    quadrature_weights=FinalizedArray(weights, dtype=np.float64),
    shape_values=FinalizedArray(shape_values, dtype=np.float64),
    parent_gradients=FinalizedArray(parent_gradients, dtype=np.float64),
    geometry_scales=FinalizedArray(geometry_scales, dtype=np.float64),
    normalized_gradients=FinalizedArray(gradients, dtype=np.float64),
    normalized_strain_displacement=FinalizedArray(b_matrix, dtype=np.float64),
    normalized_temperature_gradients=FinalizedArray(
      temperature_gradients, dtype=np.float64
    ),
    normalized_integration_weights=FinalizedArray(
      integration_weights, dtype=np.float64
    ),
    constitutive=FinalizedArray(constitutive, dtype=np.float64),
    thermal_expansion=FinalizedArray(expansion, dtype=np.float64),
    conductivity=FinalizedArray(conduction, dtype=np.float64),
    material_parameters=FinalizedArray(
      [[youngs_modulus, poisson_ratio, thermal_expansion, conductivity]],
      dtype=np.float64,
    ),
  )
  manifest = CanonicalManifest(
    {
      "block_id": block_id,
      "entity_block_id": entity_block.block_id,
      "entity_ids": entity_block.entity_ids,
      "implementations": [
        {
          "kind": item.kind,
          "name": item.name,
          "version": item.version,
          "implementation_id": item.implementation_id,
        }
        for item in header.implementations
      ],
      "ports": [
        {
          "port_id": port.port_id,
          "space_id": port.space_id,
          "coefficient_map": port.coefficient_map.values,
        }
        for port in header.ports
      ],
      "channels": [
        "internal-force",
        "heat-flux",
        "material-tangent",
        "thermal-expansion-tangent",
        "conduction-tangent",
      ],
      "state": {
        "schema": state_layout.schema,
        "row_width": 0,
        "entity_offsets": state_layout.entity_offsets.values,
      },
      "payload": {
        "quadrature_points": payload.quadrature_points.values,
        "quadrature_weights": payload.quadrature_weights.values,
        "shape_values": payload.shape_values.values,
        "parent_gradients": payload.parent_gradients.values,
        "geometry_scales": payload.geometry_scales.values,
        "normalized_gradients": payload.normalized_gradients.values,
        "normalized_strain_displacement": (
          payload.normalized_strain_displacement.values
        ),
        "normalized_temperature_gradients": (
          payload.normalized_temperature_gradients.values
        ),
        "normalized_integration_weights": (
          payload.normalized_integration_weights.values
        ),
        "constitutive": payload.constitutive.values,
        "thermal_expansion": payload.thermal_expansion.values,
        "conductivity": payload.conductivity.values,
        "material_parameters": payload.material_parameters.values,
      },
    }
  )
  return entity_block, _new(
    Q8ThermoElasticOperator,
    header=header,
    entity_block=entity_block,
    payload=payload,
    content_manifest=manifest,
  )
