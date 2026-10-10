"""Cohesive interface operator family: batched two-sided jump kinematics.

This module is the researcher-facing extension seam for 2D cohesive
interfaces: four-node line pairs (two bottom nodes, two top nodes) gluing two
continuum banks, evaluated with a corotational normal/shear frame at two
internal Gauss points. A researcher authors a plain batched traction-separation
law kernel — local jumps and accepted state rows in, traction, tangent, trial
rows, and a typed evaluation status out — plus a small declaration of element
connectivity and parameters. The compiler owns the frame state machine, the
kinematics, the quadrature, and every trusted carrier, validates the kernel
once at the compile boundary, and checks both the kernel tangent and the
assembled element tangent against central finite differences at seeded nonzero
states, so a tangent wrong only off the virgin probe state fails compilation
with a coded diagnostic instead of evaluating silently wrong.

Two decisions are pinned by the M62 survey (legacy evidence in
pyfem/elements/Interface.py and pyfem/materials/):

- Frame: legacy built a REFLECTING frame (``rot = [[n0, n1], [n1, -n0]]``,
  determinant -1) with ``normal = (ds_y, ds_x)/|ds|`` — the normal/shear
  components were exactly swapped on a 45-degree element (survey finding 1).
  This family ships the proper-rotation frame
  ``normal = (-ds_y, ds_x)/|ds|`` with ``rot = [[n0, n1], [-n1, n0]]``
  (determinant +1); M67 repaired the legacy side to the same proper rotation
  (commit 9965442, merge 8a3eae8), so the frames now agree on any geometry.
  On axis-aligned horizontal decks the two frames always agreed on the
  normal, and the pre-repair flipped shear row cancels pairwise through ``B``
  and ``B.T`` for every shipped law (all odd in the shear jump), so element
  residuals there are bitwise reproducible against legacy — the axis-aligned
  parity scope is unchanged (pre-repair, oblique decks diverged by
  construction and were outside it).
- Tangent: legacy assembles ``sum w * B.T @ D @ B`` and omits the
  ``d(rot)/da`` geometric term (survey: ~6e-3..1e-2 relative deviation from a
  finite difference of its own residual; the true Jacobian is ~1%
  nonsymmetric). This family assembles the EXACT tangent
  ``sum w * (B.T @ D @ (B + P) + G)`` with ``P`` the frame-motion jump
  derivative and ``G`` the frame-motion force term, and declares
  the honest channel flags ``linear=False, symmetric=False``. The converged
  path is pinned by residual parity; the tangent divergence from legacy is
  documented and measured in the test suite.

Parity integration is two-point Gauss (``gauss_legendre_1d(2)``): legacy's
``intMethod="NewtonCotes"`` flag was a silent no-op from v1.0 (survey finding
2) until M67 made it a validated alias for ``"Gauss"`` (commit 4d2093a, merge
8a3eae8 — no Newton-Cotes rule exists in the code base, and unknown schemes
now raise), so Gauss-2 IS the legacy behavior. A Newton-Cotes family is
explicitly NOT-yet. All four shipped laws are stateless reversible potentials of the
local jump (zero material history); irreversible laws, the rank-3/3D branch
(dead+broken legacy-side, survey finding 3), dissipation channels, and
traction output channels are NOT-yet.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NoReturn, Protocol

import numpy as np

from pyfem.v3.compile.diagnostics import (
  ModelCompilationDiagnostic,
  ModelCompilationError,
)
from pyfem.v3.fem.quadrature import gauss_legendre_1d
from pyfem.v3.fem.shapes import linear_line2
from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.identity import InstanceId, require_same_instance
from pyfem.v3.model.operator import (
  BalanceRole,
  ChannelRequest,
  CompilerConstructed,
  CouplingPolicy,
  EvaluationStatus,
  ImplementationIdentity,
  JacobianChannel,
  OperatorEvaluation,
  OperatorEvaluationInput,
  OperatorHeader,
  OperatorStateLayout,
  OperatorStateSlot,
  PortBinding,
  PortMode,
  ResidualChannel,
  SemanticId,
  StateLifetime,
)
from pyfem.v3.model.provenance import CanonicalManifest, ContentFingerprint
from pyfem.v3.model.system import (
  CompiledSource,
  CompiledSystem,
  IncidenceEntityBlock,
  SourceAttribution,
  SystemProvenance,
)
from pyfem.v3.spec.diagnostics import SourceContext

INTERFACE_SYSTEM_EXTENSION_SCHEMA = "pyfem-v3-compiled-system-interface-extension-v1"
INTERFACE_FRAME_STATE_SCHEMA = "pyfem-v3-interface-frame-state-v1"
_FLOAT64_DTYPE = np.dtype(np.float64).str

# Legacy de-facto integration: getIntegrationPoints("Line2", 0, <scheme>)
# ignores the scheme and returns the 2-point Gauss rule (survey finding 2).
_GAUSS_ORDER = 2
_NODE_COUNT = 4
_COMPONENTS = 2
_ELEMENT_DOFS = _NODE_COUNT * _COMPONENTS

# The tangent probes draw a handful of nonzero states from one fixed-seed
# generator and central-difference the response at each accepted state. The
# seeds, state count, relative tolerance, and step are part of the conformance
# contract: the seed is recorded in every probe diagnostic so a failure
# replays bit-for-bit. The operator probe scales its displacement draws by the
# element chord (the truss probe's 0.05/0.3 factors) so both the elastic and
# the finite-rotation regimes convict the assembled tangent.
_TANGENT_PROBE_SEED = 20261009
_TANGENT_PROBE_STATE_COUNT = 3
_TANGENT_PROBE_RTOL = 1.0e-4
_TANGENT_PROBE_STEP = float(np.cbrt(np.finfo(np.float64).eps))
_OPERATOR_PROBE_SEED = 20261010
_OPERATOR_PROBE_FACTORS = (0.05, 0.3)


def _new[ValueT](cls: type[ValueT], /, **fields: object) -> ValueT:
  value = object.__new__(cls)
  for name, field in fields.items():
    object.__setattr__(value, name, field)
  return value


def _fail(code: str, message: str, source: SourceContext) -> NoReturn:
  raise ModelCompilationError(
    (ModelCompilationDiagnostic(code=code, message=message, source=source),)
  )


def _source(value: SourceContext) -> CompiledSource:
  return _new(
    CompiledSource,
    source=value.source,
    line=value.line,
    column=value.column,
  )


@dataclass(frozen=True, slots=True)
class InterfaceKernelResult:
  """One batched local response of a traction-separation law evaluation.

  ``traction`` has shape ``(point_count, 2)`` (normal/shear in the element's
  corotational frame), ``tangent`` has shape ``(point_count, 2, 2)``, and
  ``trial_rows`` has shape ``(point_count, row_width)``. When ``status`` is
  not ``OK`` the operator discards the arrays and returns the accepted rows
  byte-equal, so kernels report expected numerical outcomes instead of
  raising. The shipped laws are stateless: ``row_width`` is zero and the
  trial rows echo the accepted rows.
  """

  traction: np.ndarray
  tangent: np.ndarray
  trial_rows: np.ndarray
  status: EvaluationStatus


class InterfaceKernel(Protocol):
  """Researcher-authored batched traction-separation law kernel."""

  def __call__(
    self,
    jumps: np.ndarray,
    accepted_rows: np.ndarray,
    parameters: np.ndarray,
  ) -> InterfaceKernelResult: ...


def _stateless_trial_rows(accepted_rows: np.ndarray) -> np.ndarray:
  return np.array(accepted_rows, copy=True)


def xu_needleman_rank2_kernel(
  jumps: np.ndarray,
  accepted_rows: np.ndarray,
  parameters: np.ndarray,
) -> InterfaceKernelResult:
  """Xu-Needleman exponential law, rank-2 (normal + single shear component).

  A transcription of the legacy rank-2 ``getStress`` path
  (pyfem/materials/XuNeedleman.py:52-92) with the legacy-effective constants:
  the legacy constructor hardcodes ``r = 0.0`` and ``q = 1.0`` AFTER reading
  the deck properties (:14-15), so deck-supplied ``q``/``r`` values never take
  effect and are rejected at the deck converter instead. ``vnmax``/``vtmax``
  use the legacy truncated constants 2.71828183 and 1.16580058. The tangent
  block recomputes its own temporaries exactly as the legacy second block
  does, so both outputs are bitwise-identical to legacy on the reference
  platform. The rank-3 branch is dead and broken upstream (survey finding 3)
  and is explicitly NOT-yet.
  """
  gc = float(parameters[0])
  tult = float(parameters[1])
  q = 1.0
  r = 0.0
  vnmax = gc / (2.71828183 * tult)
  vtmax = q * gc / (1.16580058 * tult)
  jump_normal = jumps[:, 0]
  jump_shear = jumps[:, 1]

  t1 = 1.0 / vnmax
  t3 = jump_normal * t1
  t4 = np.exp(-t3)
  t6 = 1.0 - q
  t9 = 1.0 / (r - 1.0)
  t12 = (r - q) * t9
  t14 = q + t12 * t3
  t15 = jump_shear * jump_shear
  t16 = vtmax * vtmax
  t17 = 1.0 / t16
  t19 = np.exp(-t15 * t17)
  t24 = gc * t4

  traction = np.zeros((len(jumps), 2), dtype=np.float64)
  traction[:, 0] = -t4 * ((1.0 - r + t3) * t6 * t9 - t14 * t19) * gc * t1 + t24 * (
    t1 * t6 * t9 - t12 * t1 * t19
  )
  traction[:, 1] = 2.0 * t24 * t14 * jump_shear * t17 * t19

  s1 = vnmax * vnmax
  s4 = 1 / vnmax
  s5 = jump_normal * s4
  s6 = np.exp(-s5)
  s8 = 1.0 - q
  s11 = 1 / (r - 1.0)
  s14 = (r - q) * s11
  s16 = q + s14 * s5
  s17 = jump_shear * jump_shear
  s18 = vtmax * vtmax
  s19 = 1 / s18
  s21 = np.exp(-s17 * s19)
  s26 = gc * s4
  s38 = s19 * s21
  s41 = gc * s6
  s46 = -s26 * s6 * s16 * jump_shear * s38 + s41 * s14 * s4 * jump_shear * s38
  s52 = s18 * s18

  tangent = np.zeros((len(jumps), 2, 2), dtype=np.float64)
  tangent[:, 0, 0] = gc / s1 * s6 * (
    (1.0 - r + s5) * s8 * s11 - s16 * s21
  ) - 2.0 * s26 * s6 * (s4 * s8 * s11 - s14 * s4 * s21)
  tangent[:, 0, 1] = 2.0 * s46
  tangent[:, 1, 0] = 2.0 * s46
  tangent[:, 1, 1] = 2.0 * s41 * s16 * s19 * s21 - 4.0 * s41 * s16 * s17 / s52 * s21
  return InterfaceKernelResult(
    traction=traction,
    tangent=tangent,
    trial_rows=_stateless_trial_rows(accepted_rows),
    status=EvaluationStatus.OK,
  )


def power_law_mode_i_kernel(
  jumps: np.ndarray,
  accepted_rows: np.ndarray,
  parameters: np.ndarray,
) -> InterfaceKernelResult:
  """Single-exponential mode-I law (legacy PowerLawModeI.py transcription).

  Only the normal traction component is nonzero. ``deltan`` derives from the
  full-precision ``exp(1.0)`` exactly as the legacy constructor computes it.
  """
  gc = float(parameters[0])
  tult = float(parameters[1])
  deltan = gc / (np.exp(1.0) * tult)
  deltan2 = deltan * deltan
  jump_normal = jumps[:, 0]

  traction = np.zeros((len(jumps), 2), dtype=np.float64)
  traction[:, 0] = gc / deltan2 * np.exp(-jump_normal / deltan) * jump_normal
  tangent = np.zeros((len(jumps), 2, 2), dtype=np.float64)
  tangent[:, 0, 0] = (
    gc / deltan2 * np.exp(-jump_normal / deltan) * (1.0 - jump_normal / deltan)
  )
  return InterfaceKernelResult(
    traction=traction,
    tangent=tangent,
    trial_rows=_stateless_trial_rows(accepted_rows),
    status=EvaluationStatus.OK,
  )


def thouless_mode_i_kernel(
  jumps: np.ndarray,
  accepted_rows: np.ndarray,
  parameters: np.ndarray,
) -> InterfaceKernelResult:
  """Piecewise rise/plateau/softening mode-I law (ThoulessModeI.py).

  Parameters are ``(Gc, Tult, d1d3, d2d3)`` with the legacy derived constants
  ``d3 = 2 Gc / ((1 - d1d3 + d2d3) Tult)``, ``d1 = d1d3 d3``,
  ``d2 = d2d3 d3``. Only the normal traction component is nonzero. The
  batched ``np.where`` chain reproduces the legacy branch ladder exactly: the
  masks are disjoint and ordered, and every branch expression is the legacy
  one.
  """
  gc = float(parameters[0])
  tult = float(parameters[1])
  d1d3 = float(parameters[2])
  d2d3 = float(parameters[3])
  d3 = 2.0 * gc / ((-d1d3 + d2d3 + 1.0) * tult)
  d1 = d1d3 * d3
  d2 = d2d3 * d3
  dummy = tult / d1
  eps_n = jumps[:, 0]

  rise = eps_n < d1
  plateau = (d1 <= eps_n) & (eps_n < d2)
  softening = (d2 <= eps_n) & (eps_n < d3)
  traction_normal = np.where(
    rise,
    dummy * eps_n,
    np.where(
      plateau,
      tult,
      np.where(softening, tult * (1.0 - (eps_n - d2) / (d3 - d2)), 0.0),
    ),
  )
  tangent_normal = np.where(
    rise,
    dummy,
    np.where(
      plateau,
      0.0,
      np.where(softening, tult * (-1.0) / (d3 - d2), 0.0),
    ),
  )
  traction = np.zeros((len(jumps), 2), dtype=np.float64)
  traction[:, 0] = traction_normal
  tangent = np.zeros((len(jumps), 2, 2), dtype=np.float64)
  tangent[:, 0, 0] = tangent_normal
  return InterfaceKernelResult(
    traction=traction,
    tangent=tangent,
    trial_rows=_stateless_trial_rows(accepted_rows),
    status=EvaluationStatus.OK,
  )


def dummy_interface_kernel(
  jumps: np.ndarray,
  accepted_rows: np.ndarray,
  parameters: np.ndarray,
) -> InterfaceKernelResult:
  """Linear diagonal law ``t = D * jump`` (legacy Dummy.py, rank 2).

  The traction runs through an explicit ``D * I`` matmul, exactly the legacy
  ``self.H @ deformation.strain`` with ``H = D * eye(2)``.
  """
  stiffness = float(parameters[0])
  matrix = stiffness * np.eye(2)
  traction = np.matmul(matrix, jumps[:, :, None])[:, :, 0]
  tangent = np.zeros((len(jumps), 2, 2), dtype=np.float64)
  tangent[:, 0, 0] = stiffness
  tangent[:, 1, 1] = stiffness
  return InterfaceKernelResult(
    traction=traction,
    tangent=tangent,
    trial_rows=_stateless_trial_rows(accepted_rows),
    status=EvaluationStatus.OK,
  )


@dataclass(frozen=True, slots=True)
class InterfaceDeclaration:
  """Authored meaning for one network of cohesive interface elements.

  ``node_quads`` carries each element's four support node ids in the legacy
  order ``(bottom_1, bottom_2, top_1, top_2)``: the bottom pair defines the
  integration line, and the displaced pair midpoints define the corotational
  frame segment. The operator's state layout is structural — one width-2
  ``normal`` slot per element holding the accepted frame normal (zero-initial,
  virgin bootstrap) — because the wave-1 laws are stateless; law-owned state
  slots are explicitly NOT-yet.
  """

  block_id: SemanticId
  space_id: SemanticId
  interface_ids: tuple[SemanticId, ...]
  node_quads: tuple[tuple[SemanticId, SemanticId, SemanticId, SemanticId], ...]
  state_schema: str
  kernel_name: str
  kernel_version: str
  implementation_id: str
  parameters: tuple[float, ...]
  kernel: InterfaceKernel
  source: SourceContext

  def __post_init__(self) -> None:
    if type(self.block_id) not in (str, int, tuple):
      msg = "interface block id must be an exact semantic id"
      raise TypeError(msg)
    if (
      type(self.interface_ids) is not tuple
      or type(self.node_quads) is not tuple
      or not self.interface_ids
      or len(self.interface_ids) != len(self.node_quads)
      or len(set(self.interface_ids)) != len(self.interface_ids)
    ):
      msg = "interface declarations require paired unique id and node quad tuples"
      raise TypeError(msg)
    for quad in self.node_quads:
      if (
        type(quad) is not tuple
        or len(quad) != _NODE_COUNT
        or any(type(node_id) not in (str, int) for node_id in quad)
      ):
        msg = "interface node quads must be exact tuples of four semantic ids"
        raise TypeError(msg)
    if type(self.state_schema) is not str or not self.state_schema:
      msg = "interface state schema must be a non-empty exact string"
      raise TypeError(msg)
    for label in ("kernel_name", "kernel_version", "implementation_id"):
      value = getattr(self, label)
      if type(value) is not str or not value:
        msg = f"interface {label} must be a non-empty exact string"
        raise TypeError(msg)
    if type(self.parameters) is not tuple or any(
      type(value) is not float or not math.isfinite(value) for value in self.parameters
    ):
      msg = "interface parameters must be finite exact floats"
      raise TypeError(msg)
    if not callable(self.kernel):
      msg = "interface kernel must be callable"
      raise TypeError(msg)
    if type(self.source) is not SourceContext:
      msg = "interface declarations require an exact SourceContext"
      raise TypeError(msg)


def _positive_finite(value: object, *, label: str) -> float:
  if type(value) not in (int, float):
    msg = f"interface law {label} must be an exact number"
    raise TypeError(msg)
  converted = float(value)
  if not math.isfinite(converted) or converted <= 0.0:
    msg = f"interface law {label} must be a positive finite number"
    raise ValueError(msg)
  return converted


def xu_needleman_declaration(
  *,
  block_id: SemanticId,
  space_id: SemanticId,
  interface_ids: tuple[SemanticId, ...],
  node_quads: tuple[tuple[SemanticId, SemanticId, SemanticId, SemanticId], ...],
  fracture_energy: float,
  ultimate_traction: float,
  state_schema: str = INTERFACE_FRAME_STATE_SCHEMA,
  source: SourceContext,
) -> InterfaceDeclaration:
  """Build a validated declaration for the rank-2 Xu-Needleman law."""
  return InterfaceDeclaration(
    block_id=block_id,
    space_id=space_id,
    interface_ids=interface_ids,
    node_quads=node_quads,
    state_schema=state_schema,
    kernel_name="xu-needleman-rank2",
    kernel_version="1",
    implementation_id="pyfem-v3-xu-needleman-rank2-v1",
    parameters=(
      _positive_finite(fracture_energy, label="fracture_energy"),
      _positive_finite(ultimate_traction, label="ultimate_traction"),
    ),
    kernel=xu_needleman_rank2_kernel,
    source=source,
  )


def power_law_mode_i_declaration(
  *,
  block_id: SemanticId,
  space_id: SemanticId,
  interface_ids: tuple[SemanticId, ...],
  node_quads: tuple[tuple[SemanticId, SemanticId, SemanticId, SemanticId], ...],
  fracture_energy: float,
  ultimate_traction: float,
  state_schema: str = INTERFACE_FRAME_STATE_SCHEMA,
  source: SourceContext,
) -> InterfaceDeclaration:
  """Build a validated declaration for the power-law mode-I law."""
  return InterfaceDeclaration(
    block_id=block_id,
    space_id=space_id,
    interface_ids=interface_ids,
    node_quads=node_quads,
    state_schema=state_schema,
    kernel_name="power-law-mode-i",
    kernel_version="1",
    implementation_id="pyfem-v3-power-law-mode-i-v1",
    parameters=(
      _positive_finite(fracture_energy, label="fracture_energy"),
      _positive_finite(ultimate_traction, label="ultimate_traction"),
    ),
    kernel=power_law_mode_i_kernel,
    source=source,
  )


def thouless_mode_i_declaration(
  *,
  block_id: SemanticId,
  space_id: SemanticId,
  interface_ids: tuple[SemanticId, ...],
  node_quads: tuple[tuple[SemanticId, SemanticId, SemanticId, SemanticId], ...],
  fracture_energy: float,
  ultimate_traction: float,
  d1d3: float,
  d2d3: float,
  state_schema: str = INTERFACE_FRAME_STATE_SCHEMA,
  source: SourceContext,
) -> InterfaceDeclaration:
  """Build a validated declaration for the Thouless mode-I law.

  ``d1d3``/``d2d3`` are the rise-to-plateau and plateau-to-softening jump
  ratios; the piecewise branches require ``0 < d1d3 < d2d3 < 1``.
  """
  first = _positive_finite(d1d3, label="d1d3")
  second = _positive_finite(d2d3, label="d2d3")
  if not first < second < 1.0:
    msg = "thouless mode-I requires 0 < d1d3 < d2d3 < 1"
    raise ValueError(msg)
  return InterfaceDeclaration(
    block_id=block_id,
    space_id=space_id,
    interface_ids=interface_ids,
    node_quads=node_quads,
    state_schema=state_schema,
    kernel_name="thouless-mode-i",
    kernel_version="1",
    implementation_id="pyfem-v3-thouless-mode-i-v1",
    parameters=(
      _positive_finite(fracture_energy, label="fracture_energy"),
      _positive_finite(ultimate_traction, label="ultimate_traction"),
      first,
      second,
    ),
    kernel=thouless_mode_i_kernel,
    source=source,
  )


def dummy_interface_declaration(
  *,
  block_id: SemanticId,
  space_id: SemanticId,
  interface_ids: tuple[SemanticId, ...],
  node_quads: tuple[tuple[SemanticId, SemanticId, SemanticId, SemanticId], ...],
  stiffness: float,
  state_schema: str = INTERFACE_FRAME_STATE_SCHEMA,
  source: SourceContext,
) -> InterfaceDeclaration:
  """Build a validated declaration for the linear dummy interface law."""
  return InterfaceDeclaration(
    block_id=block_id,
    space_id=space_id,
    interface_ids=interface_ids,
    node_quads=node_quads,
    state_schema=state_schema,
    kernel_name="dummy-linear-interface",
    kernel_version="1",
    implementation_id="pyfem-v3-dummy-linear-interface-v1",
    parameters=(_positive_finite(stiffness, label="stiffness"),),
    kernel=dummy_interface_kernel,
    source=source,
  )


def _validated_kernel_arrays(
  result: object,
  *,
  point_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, EvaluationStatus]:
  if type(result) is not InterfaceKernelResult:
    msg = "interface kernels must return an exact InterfaceKernelResult"
    raise TypeError(msg)
  status = result.status
  if type(status) is not EvaluationStatus:
    msg = "interface kernel status must be an exact EvaluationStatus"
    raise TypeError(msg)
  arrays = (result.traction, result.tangent, result.trial_rows)
  shapes = ((point_count, 2), (point_count, 2, 2), (point_count, 0))
  for array, shape in zip(arrays, shapes, strict=True):
    if (
      type(array) is not np.ndarray
      or array.dtype != np.dtype(np.float64)
      or array.dtype.metadata is not None
      or array.shape != shape
    ):
      msg = "interface kernel arrays must match the declared batched float64 shapes"
      raise TypeError(msg)
  if status is EvaluationStatus.OK and not all(
    bool(np.isfinite(array).all()) for array in arrays
  ):
    msg = "interface kernel arrays must be finite for a successful evaluation"
    raise TypeError(msg)
  return result.traction, result.tangent, result.trial_rows, status


def _probe_kernel_tangent(
  kernel: InterfaceKernel,
  parameters: np.ndarray,
  *,
  point_count: int,
  source: SourceContext,
) -> None:
  """Central-difference the law kernel tangent at seeded nonzero jumps.

  The virgin-state probe proves array and status plumbing only, so a tangent
  wrong solely at nonzero jumps would compile clean and evaluate silently
  wrong. Every seeded jump the kernel accepts is checked by a central finite
  difference of the traction along each jump component. Jumps — or stencil
  legs — the kernel rejects with a typed status carry no channels to verify
  and are skipped. The draws come from one fixed-seed generator and the seed
  is recorded in every diagnostic, so a failure replays bit-for-bit. This is
  the spring.py tangent probe verbatim at the law boundary; the per-law
  deep-scale conviction batteries live in the test suite.
  """
  generator = np.random.default_rng(_TANGENT_PROBE_SEED)
  for state_index in range(_TANGENT_PROBE_STATE_COUNT):
    jumps = generator.standard_normal((point_count, 2))
    accepted_rows = generator.standard_normal((point_count, 0))
    # The probe mirrors runtime input mutability exactly, exactly as at the
    # virgin state: every array handed to the kernel is read-only.
    jumps.setflags(write=False)
    accepted_rows.setflags(write=False)
    try:
      probed = kernel(jumps, accepted_rows, parameters)
    except Exception:
      _fail(
        "kernel-probe-failed",
        "interface kernel failed its seeded nonzero-jump compile probe "
        f"(seed {_TANGENT_PROBE_SEED}, state {state_index})",
        source,
      )
    _, tangent, _, status = _validated_kernel_arrays(probed, point_count=point_count)
    if status is not EvaluationStatus.OK:
      continue
    for point_index in range(point_count):
      for component in range(2):
        step = _TANGENT_PROBE_STEP * max(
          1.0,
          abs(float(jumps[point_index, component])),
        )
        plus = np.array(jumps, copy=True)
        minus = np.array(jumps, copy=True)
        plus[point_index, component] += step
        minus[point_index, component] -= step
        plus.setflags(write=False)
        minus.setflags(write=False)
        try:
          plus_result = kernel(plus, accepted_rows, parameters)
          minus_result = kernel(minus, accepted_rows, parameters)
        except Exception:
          _fail(
            "kernel-probe-failed",
            "interface kernel failed its seeded nonzero-jump compile probe "
            f"(seed {_TANGENT_PROBE_SEED}, state {state_index}, point "
            f"{point_index}, component {component})",
            source,
          )
        plus_traction, _, _, plus_status = _validated_kernel_arrays(
          plus_result,
          point_count=point_count,
        )
        minus_traction, _, _, minus_status = _validated_kernel_arrays(
          minus_result,
          point_count=point_count,
        )
        if (
          plus_status is not EvaluationStatus.OK
          or minus_status is not EvaluationStatus.OK
        ):
          continue
        difference = (plus_traction[point_index] - minus_traction[point_index]) / (
          2.0 * step
        )
        column = tangent[point_index, :, component]
        scale = max(
          1.0,
          float(np.abs(column).max()),
          float(np.abs(difference).max()),
        )
        mismatch = float(np.abs(column - difference).max())
        if mismatch > _TANGENT_PROBE_RTOL * scale:
          _fail(
            "inconsistent-kernel-tangent",
            "interface kernel tangent contradicts a central finite difference "
            "of its traction at a seeded nonzero jump (seed "
            f"{_TANGENT_PROBE_SEED}, state {state_index}, point "
            f"{point_index}, component {component}): kernel tangent column "
            f"{column.tolist()} vs finite difference {difference.tolist()}",
            source,
          )


# The residual/assembly path uses np.matmul everywhere and never einsum:
# batched matmul reproduces the legacy np.dot summation order bit-for-bit on
# the parity-critical products (measured on the reference platform), which the
# element-level bitwise pins rely on.
_FLIP_JACOBI = np.array([[0.0, -1.0], [1.0, 0.0]])
_DOF_SEGMENTS = np.array([-0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5])
_DOF_COMPONENT_INDEX = np.array([0, 1, 0, 1, 0, 1, 0, 1])


def _evaluate_response(
  midpoints: np.ndarray,
  weights: np.ndarray,
  shapes: np.ndarray,
  kernel: InterfaceKernel,
  parameters: np.ndarray,
  displacements: np.ndarray,
  accepted_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, EvaluationStatus]:
  """Evaluate the batched element response from accepted state.

  ``midpoints`` carries the reference pair midpoints ``0.5 * (x1 + x3)`` and
  ``0.5 * (x2 + x4)`` per element, ``weights`` the constant per-element
  Jacobian weights, and ``shapes`` the Line2 values at the two internal Gauss
  points. Returns the element residual ``(e, 8)``, the exact element tangent
  ``(e, 8, 8)`` (material part plus the ``d(rot)/da`` geometric term), the
  trial frame normals ``(e, 2)``, and the law's typed status. On a non-``OK``
  law status the channels carry zeros and the trial rows echo the accepted
  rows byte-equal — the operator discards them either way.
  """
  entity_count = displacements.shape[0]
  # Corotational frame segment from the displaced pair midpoints, in the
  # legacy evaluation order (Interface.py:133-140).
  mids = np.array(midpoints, dtype=np.float64, order="C", copy=True)
  mids[:, 0, 0] += 0.5 * (displacements[:, 0] + displacements[:, 4])
  mids[:, 0, 1] += 0.5 * (displacements[:, 1] + displacements[:, 5])
  mids[:, 1, 0] += 0.5 * (displacements[:, 2] + displacements[:, 6])
  mids[:, 1, 1] += 0.5 * (displacements[:, 3] + displacements[:, 7])
  ds = mids[:, 1] - mids[:, 0]
  length = np.sqrt(ds[:, 0] * ds[:, 0] + ds[:, 1] * ds[:, 1])

  # The sign-continuity frame machine with the CORRECTED (non-reflecting)
  # normal candidate: legacy's virgin bootstrap (accepted norm below 0.5) and
  # anti-alignment flip, applied to the proper normal (-ds_y, ds_x) / |ds|.
  candidate = np.stack((-ds[:, 1] / length, ds[:, 0] / length), axis=1)
  accepted_length = np.sqrt(
    accepted_rows[:, 0] * accepted_rows[:, 0]
    + accepted_rows[:, 1] * accepted_rows[:, 1]
  )
  virgin = accepted_length < 0.5
  flip = (candidate * accepted_rows).sum(axis=1) < 0.0
  sign = np.where(virgin, 1.0, np.where(flip, -1.0, 1.0))
  trial_normal = candidate * sign[:, None]

  rotation = np.empty((entity_count, 2, 2), dtype=np.float64)
  rotation[:, 0, 0] = trial_normal[:, 0]
  rotation[:, 0, 1] = trial_normal[:, 1]
  rotation[:, 1, 0] = -trial_normal[:, 1]
  rotation[:, 1, 1] = trial_normal[:, 0]

  # B = [-rot phi0, -rot phi1, +rot phi0, +rot phi1] per integration point,
  # the legacy block order (Interface.py:113-122).
  rot_ip = rotation[:, None, :, :]
  phi0 = shapes[None, :, 0, None, None]
  phi1 = shapes[None, :, 1, None, None]
  b_matrix = np.empty((entity_count, _GAUSS_ORDER, 2, _ELEMENT_DOFS), dtype=np.float64)
  b_matrix[:, :, :, 0:2] = -rot_ip * phi0
  b_matrix[:, :, :, 2:4] = -rot_ip * phi1
  b_matrix[:, :, :, 4:6] = rot_ip * phi0
  b_matrix[:, :, :, 6:8] = rot_ip * phi1

  jumps = np.matmul(b_matrix, displacements[:, None, :, None])[:, :, :, 0]
  flat_jumps = np.array(
    jumps.reshape(entity_count * _GAUSS_ORDER, 2),
    dtype=np.float64,
    order="C",
    copy=True,
  )
  flat_jumps.setflags(write=False)
  law_rows = np.zeros((entity_count * _GAUSS_ORDER, 0), dtype=np.float64)
  law_rows.setflags(write=False)
  traction_flat, tangent_flat, _, status = _validated_kernel_arrays(
    kernel(flat_jumps, law_rows, parameters),
    point_count=entity_count * _GAUSS_ORDER,
  )
  if status is not EvaluationStatus.OK:
    return (
      np.zeros((entity_count, _ELEMENT_DOFS), dtype=np.float64),
      np.zeros((entity_count, _ELEMENT_DOFS, _ELEMENT_DOFS), dtype=np.float64),
      np.array(accepted_rows, dtype=np.float64, order="C", copy=True),
      status,
    )
  traction = traction_flat.reshape(entity_count, _GAUSS_ORDER, 2)
  law_tangent = tangent_flat.reshape(entity_count, _GAUSS_ORDER, 2, 2)

  force0 = np.matmul(
    b_matrix[:, 0].transpose(0, 2, 1),
    traction[:, 0][:, :, None],
  )[:, :, 0]
  force1 = np.matmul(
    b_matrix[:, 1].transpose(0, 2, 1),
    traction[:, 1][:, :, None],
  )[:, :, 0]
  residual = force0 * weights[:, None] + force1 * weights[:, None]

  # The exact tangent is sum_ip w [Bᵀ D (B + P) + G]: beyond the material
  # part BᵀDB the frame rotation enters twice — through the traction change
  # D · d(rot)/da · q on the nonzero global jump q (the P term), and through
  # d(Bᵀ)/da · t at fixed traction (the geometric term G). Legacy omits both.
  normal0 = trial_normal[:, 0]
  normal1 = trial_normal[:, 1]
  shat0 = ds[:, 0] / length
  shat1 = ds[:, 1] / length
  dn0_ds0 = (sign * _FLIP_JACOBI[0, 0] - normal0 * shat0) / length
  dn0_ds1 = (sign * _FLIP_JACOBI[0, 1] - normal0 * shat1) / length
  dn1_ds0 = (sign * _FLIP_JACOBI[1, 0] - normal1 * shat0) / length
  dn1_ds1 = (sign * _FLIP_JACOBI[1, 1] - normal1 * shat1) / length

  # The global jump q per integration point: phi0 * (u3 - u1) + phi1 *
  # (u4 - u2), so that the local jump is rot @ q.
  jump_left = displacements[:, 4:6] - displacements[:, 0:2]
  jump_right = displacements[:, 6:8] - displacements[:, 2:4]
  global_jump = (
    shapes[None, :, 0, None] * jump_left[:, None, :]
    + shapes[None, :, 1, None] * jump_right[:, None, :]
  )

  # p_k = (d rot / ds_k) q per integration point, k the segment component.
  p_term = np.empty((entity_count, _GAUSS_ORDER, 2, 2), dtype=np.float64)
  p_term[:, :, 0, 0] = dn0_ds0[:, None] * global_jump[:, :, 0] + (
    dn1_ds0[:, None] * global_jump[:, :, 1]
  )
  p_term[:, :, 0, 1] = dn0_ds0[:, None] * global_jump[:, :, 1] - (
    dn1_ds0[:, None] * global_jump[:, :, 0]
  )
  p_term[:, :, 1, 0] = dn0_ds1[:, None] * global_jump[:, :, 0] + (
    dn1_ds1[:, None] * global_jump[:, :, 1]
  )
  p_term[:, :, 1, 1] = dn0_ds1[:, None] * global_jump[:, :, 1] - (
    dn1_ds1[:, None] * global_jump[:, :, 0]
  )
  # P[r, l] = c_l * p[l % 2, r] — the frame-motion jump derivative of column l.
  p_columns = p_term[:, :, _DOF_COMPONENT_INDEX, :]
  jump_tangent_input = b_matrix + _DOF_SEGMENTS[None, None, None, :] * (
    p_columns.swapaxes(-1, -2)
  )
  traction_tangent = np.matmul(
    b_matrix.transpose(0, 1, 3, 2),
    np.matmul(law_tangent, jump_tangent_input),
  )

  # g_k = (d rot / ds_k).T t per integration point, k the segment component.
  g_term = np.empty((entity_count, _GAUSS_ORDER, 2, 2), dtype=np.float64)
  g_term[:, :, 0, 0] = dn0_ds0[:, None] * traction[:, :, 0] - (
    dn1_ds0[:, None] * traction[:, :, 1]
  )
  g_term[:, :, 0, 1] = dn1_ds0[:, None] * traction[:, :, 0] + (
    dn0_ds0[:, None] * traction[:, :, 1]
  )
  g_term[:, :, 1, 0] = dn0_ds1[:, None] * traction[:, :, 0] - (
    dn1_ds1[:, None] * traction[:, :, 1]
  )
  g_term[:, :, 1, 1] = dn1_ds1[:, None] * traction[:, :, 0] + (
    dn0_ds1[:, None] * traction[:, :, 1]
  )

  row_scale = np.empty((entity_count, _GAUSS_ORDER, _ELEMENT_DOFS), dtype=np.float64)
  row_scale[:, :, 0] = -shapes[:, 0]
  row_scale[:, :, 1] = -shapes[:, 0]
  row_scale[:, :, 2] = -shapes[:, 1]
  row_scale[:, :, 3] = -shapes[:, 1]
  row_scale[:, :, 4] = shapes[:, 0]
  row_scale[:, :, 5] = shapes[:, 0]
  row_scale[:, :, 6] = shapes[:, 1]
  row_scale[:, :, 7] = shapes[:, 1]
  # column_term[l, r] = c_l * g[l % 2, r]; G[i, l] = row_scale[i] *
  # column_term[l, i % 2].
  column_term = (
    _DOF_SEGMENTS[None, None, :, None]
    * g_term[
      :,
      :,
      _DOF_COMPONENT_INDEX,
      :,
    ]
  )
  geometric = row_scale[:, :, :, None] * np.take(
    column_term,
    _DOF_COMPONENT_INDEX,
    axis=-1,
  ).swapaxes(-1, -2)

  tangent = (traction_tangent[:, 0] + geometric[:, 0]) * weights[:, None, None] + (
    traction_tangent[:, 1] + geometric[:, 1]
  ) * weights[:, None, None]
  return residual, tangent, trial_normal, status


def _probe_operator_tangent(
  midpoints: np.ndarray,
  weights: np.ndarray,
  shapes: np.ndarray,
  kernel: InterfaceKernel,
  parameters: np.ndarray,
  *,
  source: SourceContext,
  response: object = None,
) -> None:
  """Central-difference the assembled element tangent at seeded states.

  The law-level probe cannot see the frame term: the assembled tangent's
  geometric part multiplies the law traction with the frame's displacement
  derivative, so it is convicted here on the whole network response. Draws
  follow the truss probe's element-chord scaling (0.05 and 0.3 of the chord)
  and three frame-branch variants — virgin (zero accepted normal), aligned
  (reference-frame normal), and flipped (its negation) — so every frame
  branch convicts the tangent at nonzero jump states. Element responses are
  block-diagonal across the network, so one displacement component is
  perturbed on all elements at once and the central difference convicts every
  element's tangent column at once. The seed is recorded in every diagnostic,
  so a failure replays bit-for-bit. ``response`` is the evaluation core under
  test — ``_evaluate_response`` in production; the test suite injects a
  tampered response to convict this probe's own failure leg.
  """
  evaluate_response = _evaluate_response if response is None else response
  entity_count = len(weights)
  generator = np.random.default_rng(_OPERATOR_PROBE_SEED)
  reference_ds = midpoints[:, 1] - midpoints[:, 0]
  reference_length = np.sqrt(
    reference_ds[:, 0] * reference_ds[:, 0] + reference_ds[:, 1] * reference_ds[:, 1]
  )
  reference_normal = np.stack(
    (-reference_ds[:, 1] / reference_length, reference_ds[:, 0] / reference_length),
    axis=1,
  )
  chord = 2.0 * weights
  variants = (
    np.zeros((entity_count, 2), dtype=np.float64),
    reference_normal,
    -reference_normal,
  )
  for factor in _OPERATOR_PROBE_FACTORS:
    for variant_index, accepted in enumerate(variants):
      displacements = (
        factor
        * chord[:, None]
        * generator.standard_normal((entity_count, _ELEMENT_DOFS))
      )
      accepted_read = np.array(accepted, dtype=np.float64, order="C", copy=True)
      displacements.setflags(write=False)
      accepted_read.setflags(write=False)
      try:
        _, tangent, _, status = evaluate_response(
          midpoints,
          weights,
          shapes,
          kernel,
          parameters,
          displacements,
          accepted_read,
        )
      except Exception:
        _fail(
          "operator-probe-failed",
          "interface operator failed its seeded nonzero-state compile probe "
          f"(seed {_OPERATOR_PROBE_SEED}, factor {factor}, variant "
          f"{variant_index})",
          source,
        )
      if status is not EvaluationStatus.OK:
        continue
      for dof in range(_ELEMENT_DOFS):
        steps = _TANGENT_PROBE_STEP * np.maximum(
          1.0,
          np.abs(displacements[:, dof]),
        )
        plus = np.array(displacements, copy=True)
        minus = np.array(displacements, copy=True)
        plus[:, dof] += steps
        minus[:, dof] -= steps
        plus.setflags(write=False)
        minus.setflags(write=False)
        try:
          plus_residual, _, _, plus_status = evaluate_response(
            midpoints,
            weights,
            shapes,
            kernel,
            parameters,
            plus,
            accepted_read,
          )
          minus_residual, _, _, minus_status = evaluate_response(
            midpoints,
            weights,
            shapes,
            kernel,
            parameters,
            minus,
            accepted_read,
          )
        except Exception:
          _fail(
            "operator-probe-failed",
            "interface operator failed its seeded nonzero-state compile probe "
            f"(seed {_OPERATOR_PROBE_SEED}, factor {factor}, variant "
            f"{variant_index}, component {dof})",
            source,
          )
        if (
          plus_status is not EvaluationStatus.OK
          or minus_status is not EvaluationStatus.OK
        ):
          continue
        difference = (plus_residual - minus_residual) / (2.0 * steps[:, None])
        column = tangent[:, :, dof]
        scale = np.maximum(
          1.0,
          np.maximum(np.abs(column).max(axis=1), np.abs(difference).max(axis=1)),
        )
        mismatch = np.abs(column - difference).max(axis=1)
        if bool((mismatch > _TANGENT_PROBE_RTOL * scale).any()):
          element_index = int(np.argmax(mismatch / scale))
          _fail(
            "inconsistent-operator-tangent",
            "interface operator tangent contradicts a central finite "
            "difference of its residual at a seeded nonzero state (seed "
            f"{_OPERATOR_PROBE_SEED}, factor {factor}, variant "
            f"{variant_index}, element {element_index}, component {dof}): "
            f"tangent column {column[element_index].tolist()} vs finite "
            f"difference {difference[element_index].tolist()}",
            source,
          )


@dataclass(frozen=True, slots=True, eq=False, init=False)
class InterfacePayload(CompilerConstructed):
  reference_midpoints: FinalizedArray
  integration_weights: FinalizedArray
  ip_shape_values: FinalizedArray
  parameters: FinalizedArray


@dataclass(frozen=True, slots=True, eq=False, init=False)
class InterfaceOperator(CompilerConstructed):
  header: OperatorHeader
  interface_block: IncidenceEntityBlock
  payload: InterfacePayload
  content_manifest: CanonicalManifest
  kernel: InterfaceKernel
  system_instance: InstanceId

  def evaluate(
    self,
    inputs: OperatorEvaluationInput,
  ) -> OperatorEvaluation:
    """Evaluate the interface network from accepted state with typed outcomes."""
    if type(inputs) is not OperatorEvaluationInput:
      msg = "interface evaluation requires an exact immutable evaluation input"
      raise TypeError(msg)
    if type(inputs.port_values) is not tuple or len(inputs.port_values) != 1:
      msg = "interface evaluation requires exactly one displacement port batch"
      raise TypeError(msg)
    values = inputs.port_values[0].values
    expected = self.header.ports[0].coefficient_map.values.shape
    if (
      values.dtype != np.dtype(np.float64)
      or values.dtype.metadata is not None
      or values.shape != expected
      or not bool(np.isfinite(values).all())
    ):
      msg = "interface displacement port values must be a finite float64 batch"
      raise TypeError(msg)
    layout = self.header.state_layout
    accepted_state = inputs.accepted_state.values
    if (
      accepted_state.dtype != np.dtype(np.float64)
      or accepted_state.dtype.metadata is not None
      or accepted_state.shape != layout.row_shape
      or not bool(np.isfinite(accepted_state).all())
    ):
      msg = "interface accepted state must match the compiled state layout"
      raise TypeError(msg)
    if inputs.signals or self.header.signal_ports:
      msg = "interface model operator does not accept program signal inputs"
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
      msg = "interface evaluation request contains an unavailable or duplicate channel"
      raise ValueError(msg)

    residual, tangent, trial_normal, status = _evaluate_response(
      self.payload.reference_midpoints.values,
      self.payload.integration_weights.values,
      self.payload.ip_shape_values.values,
      self.kernel,
      self.payload.parameters.values,
      values,
      accepted_state,
    )
    if status is not EvaluationStatus.OK:
      return _new(
        OperatorEvaluation,
        residual_values=(),
        jacobian_values=(),
        trial_state=FinalizedArray(accepted_state, dtype=np.float64),
        status=status,
      )
    residual_values = (
      (FinalizedArray(residual, dtype=np.float64),)
      if request.residual_channel_ids
      else ()
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
      trial_state=FinalizedArray(trial_normal, dtype=np.float64),
      status=status,
    )


def compile_interface_operator(
  system: CompiledSystem,
  declaration: InterfaceDeclaration,
) -> tuple[IncidenceEntityBlock, InterfaceOperator]:
  """Compile one authored interface network against an existing compiled system."""
  if type(system) is not CompiledSystem:
    msg = "interface compilation requires an exact CompiledSystem"
    raise TypeError(msg)
  if type(declaration) is not InterfaceDeclaration:
    msg = "interface compilation requires an exact InterfaceDeclaration"
    raise TypeError(msg)
  source = declaration.source
  space = next(
    (item for item in system.spaces if item.space_id == declaration.space_id),
    None,
  )
  if space is None:
    _fail(
      "unknown-interface-space",
      "interface network references a space the compiled system does not have",
      source,
    )
  if len(space.components) != _COMPONENTS:
    _fail(
      "unsupported-interface-space",
      "cohesive interfaces require a two-component displacement space",
      source,
    )
  support = next(
    (item for item in system.point_blocks if item.block_id == space.support_block_id),
    None,
  )
  if support is None:
    _fail(
      "unknown-interface-support-block",
      "interface space support block is absent from the compiled system",
      source,
    )
  node_dense = {node_id: index for index, node_id in enumerate(support.entity_ids)}
  quad_indices: list[list[int]] = []
  for interface_id, quad in zip(
    declaration.interface_ids,
    declaration.node_quads,
    strict=True,
  ):
    indices: list[int] = []
    for node_id in quad:
      index = node_dense.get(node_id)
      if index is None:
        _fail(
          "unknown-interface-node",
          f"interface element {interface_id!r} references a support node the "
          "compiled system does not have",
          source,
        )
      indices.append(index)
    quad_indices.append(indices)

  entity_count = len(declaration.interface_ids)
  connectivity = np.array(quad_indices, dtype=space.coefficient_map.values.dtype)
  coordinates = np.array(
    support.reference_coordinates.values[connectivity],
    dtype=np.float64,
    order="C",
    copy=True,
  )
  # Geometry, in the legacy evaluation order: the bottom pair carries the
  # integration line (weight = |d x_bottom / d xi|, Gauss weights are 1), and
  # the pair midpoints carry the frame segment.
  jac0 = coordinates[:, 0, 0] * -0.5 + coordinates[:, 1, 0] * 0.5
  jac1 = coordinates[:, 0, 1] * -0.5 + coordinates[:, 1, 1] * 0.5
  integration_weights = np.sqrt(jac0 * jac0 + jac1 * jac1)
  reference_midpoints = 0.5 * (coordinates[:, :2] + coordinates[:, 2:])
  reference_ds = reference_midpoints[:, 1] - reference_midpoints[:, 0]
  reference_length = np.sqrt(
    reference_ds[:, 0] * reference_ds[:, 0] + reference_ds[:, 1] * reference_ds[:, 1]
  )
  tolerance = system.provenance.geometry_relative_tolerance
  for index, interface_id in enumerate(declaration.interface_ids):
    scale = float(np.abs(coordinates[index]).max())
    if not np.isfinite(scale):
      _fail(
        "degenerate-interface-geometry",
        f"interface element {interface_id!r} has non-finite reference coordinates",
        source,
      )
    if not math.isfinite(float(integration_weights[index])) or not (
      float(integration_weights[index]) > tolerance * scale
    ):
      _fail(
        "degenerate-interface-geometry",
        f"interface element {interface_id!r} has a zero or scale-unresolvable "
        "bottom-pair chord; the integration line is undefined",
        source,
      )
    if not math.isfinite(float(reference_length[index])) or not (
      float(reference_length[index]) > tolerance * scale
    ):
      _fail(
        "degenerate-interface-geometry",
        f"interface element {interface_id!r} has coincident pair midpoints; "
        "the corotational frame is undefined",
        source,
      )

  gauss_points, _ = gauss_legendre_1d(_GAUSS_ORDER)
  ip_shapes, _ = linear_line2(gauss_points.reshape(-1, 1))
  ip_shapes = np.array(ip_shapes, dtype=np.float64, order="C", copy=True)
  parameters = FinalizedArray(declaration.parameters, dtype=np.float64)

  # The probes mirror runtime input mutability exactly: evaluate hands the
  # kernel read-only arrays, so the probes do too, or an input-mutating kernel
  # would compile and fail untyped at first evaluation.
  probe_jumps = np.zeros((entity_count * _GAUSS_ORDER, 2), dtype=np.float64)
  probe_rows = np.zeros((entity_count * _GAUSS_ORDER, 0), dtype=np.float64)
  probe_jumps.setflags(write=False)
  probe_rows.setflags(write=False)
  try:
    probe_result = declaration.kernel(probe_jumps, probe_rows, parameters.values)
  except Exception:
    _fail(
      "kernel-probe-failed",
      "interface kernel failed its virgin-state compile probe",
      source,
    )
  probe = _validated_kernel_arrays(
    probe_result,
    point_count=entity_count * _GAUSS_ORDER,
  )
  if probe[3] is not EvaluationStatus.OK:
    _fail(
      "invalid-kernel-probe",
      "interface kernel must evaluate its virgin zero state successfully",
      source,
    )
  _probe_kernel_tangent(
    declaration.kernel,
    parameters.values,
    point_count=entity_count * _GAUSS_ORDER,
    source=source,
  )
  probe_displacements = np.zeros((entity_count, _ELEMENT_DOFS), dtype=np.float64)
  probe_accepted = np.zeros((entity_count, 2), dtype=np.float64)
  probe_displacements.setflags(write=False)
  probe_accepted.setflags(write=False)
  virgin_residual, virgin_tangent, _, virgin_status = _evaluate_response(
    reference_midpoints,
    integration_weights,
    ip_shapes,
    declaration.kernel,
    parameters.values,
    probe_displacements,
    probe_accepted,
  )
  if (
    virgin_status is not EvaluationStatus.OK
    or not bool(np.isfinite(virgin_residual).all())
    or not bool(np.isfinite(virgin_tangent).all())
  ):
    _fail(
      "invalid-operator-probe",
      "interface operator must evaluate its virgin zero state successfully",
      source,
    )
  _probe_operator_tangent(
    reference_midpoints,
    integration_weights,
    ip_shapes,
    declaration.kernel,
    parameters.values,
    source=source,
  )

  interface_block = _new(
    IncidenceEntityBlock,
    block_id=declaration.block_id,
    entity_ids=declaration.interface_ids,
    sources=tuple(_source(source) for _ in declaration.interface_ids),
    incidence=FinalizedArray(connectivity, dtype=connectivity.dtype),
  )
  block_id = declaration.block_id, declaration.state_schema
  state_layout = _new(
    OperatorStateLayout,
    schema=declaration.state_schema,
    block_id=block_id,
    entity_count=entity_count,
    slots=(
      _new(
        OperatorStateSlot,
        name="normal",
        width=2,
        dtype=_FLOAT64_DTYPE,
        lifetime=StateLifetime.ACCEPTED_TRIAL,
      ),
    ),
    entity_offsets=FinalizedArray(
      np.arange(entity_count + 1, dtype=connectivity.dtype) * 2,
      dtype=connectivity.dtype,
    ),
    row_width=2,
    dtype=_FLOAT64_DTYPE,
    lifetime=StateLifetime.ACCEPTED_TRIAL,
  )
  port = _new(
    PortBinding,
    port_id="displacement",
    space_id=space.space_id,
    mode=PortMode.COEFFICIENTS,
    coefficient_map=FinalizedArray(
      space.coefficient_map.values[connectivity].reshape(entity_count, _ELEMENT_DOFS),
      dtype=space.coefficient_map.values.dtype,
    ),
  )
  residual_channel = _new(
    ResidualChannel,
    channel_id="interface-traction",
    target_port_id=port.port_id,
    balance_role=BalanceRole.INTERNAL,
    linear=False,
  )
  # The exact tangent carries the frame-motion term, whose true Jacobian is
  # nonsymmetric (survey: ~1% at finite openings) — the channel declares it.
  jacobian_channel = _new(
    JacobianChannel,
    channel_id="interface-tangent",
    residual_channel_id=residual_channel.channel_id,
    target_port_id=port.port_id,
    source_port_id=port.port_id,
    balance_role=BalanceRole.INTERNAL,
    linear=False,
    symmetric=False,
  )
  header = _new(
    OperatorHeader,
    block_id=block_id,
    entity_block_id=interface_block.block_id,
    implementations=(
      _new(
        ImplementationIdentity,
        kind="constitutive-kernel",
        name=declaration.kernel_name,
        version=declaration.kernel_version,
        implementation_id=declaration.implementation_id,
      ),
    ),
    ports=(port,),
    signal_ports=(),
    residual_channels=(residual_channel,),
    jacobian_channels=(jacobian_channel,),
    state_layout=state_layout,
    coupling_policy=CouplingPolicy.FIXED,
  )
  payload = _new(
    InterfacePayload,
    reference_midpoints=FinalizedArray(reference_midpoints, dtype=np.float64),
    integration_weights=FinalizedArray(integration_weights, dtype=np.float64),
    ip_shape_values=FinalizedArray(ip_shapes, dtype=np.float64),
    parameters=parameters,
  )
  manifest = CanonicalManifest(
    {
      "block_id": block_id,
      "entity_block_id": interface_block.block_id,
      "entity_ids": interface_block.entity_ids,
      "node_quads": [list(quad) for quad in declaration.node_quads],
      "implementations": [
        {
          "kind": "constitutive-kernel",
          "name": declaration.kernel_name,
          "version": declaration.kernel_version,
          "implementation_id": declaration.implementation_id,
        }
      ],
      "frame": {
        "kind": "corrected-corotational-sign-continuity",
        "quadrature": f"gauss-{_GAUSS_ORDER}",
        "quadrature_points": gauss_points,
      },
      "port": {
        "port_id": port.port_id,
        "space_id": port.space_id,
        "coefficient_map": port.coefficient_map.values,
      },
      "channels": ["interface-traction", "interface-tangent"],
      "state": {
        "schema": state_layout.schema,
        "row_width": state_layout.row_width,
        "slots": [{"name": "normal", "width": 2}],
        "entity_offsets": state_layout.entity_offsets.values,
      },
      "payload": {
        "reference_midpoints": payload.reference_midpoints.values,
        "integration_weights": payload.integration_weights.values,
        "ip_shape_values": payload.ip_shape_values.values,
        "parameters": payload.parameters.values,
      },
    }
  )
  operator = _new(
    InterfaceOperator,
    header=header,
    interface_block=interface_block,
    payload=payload,
    content_manifest=manifest,
    kernel=declaration.kernel,
    system_instance=system.instance_id,
  )
  return interface_block, operator


def compose_interface_system(
  base: CompiledSystem,
  interface_block: IncidenceEntityBlock,
  interface_operator: InterfaceOperator,
) -> CompiledSystem:
  """Compose one compiled interface network into a base compiled system."""
  if type(base) is not CompiledSystem:
    msg = "interface composition requires an exact base CompiledSystem"
    raise TypeError(msg)
  if type(interface_block) is not IncidenceEntityBlock:
    msg = "interface composition requires an exact interface IncidenceEntityBlock"
    raise TypeError(msg)
  if type(interface_operator) is not InterfaceOperator:
    msg = "interface composition requires an exact InterfaceOperator"
    raise TypeError(msg)
  require_same_instance(
    interface_operator.system_instance,
    base.instance_id,
    context="interface system composition",
  )
  if interface_operator.interface_block is not interface_block:
    msg = "interface operator must bind the exact composed interface block"
    raise ValueError(msg)
  if any(
    block.block_id == interface_block.block_id for block in base.point_blocks
  ) or any(block.block_id == interface_block.block_id for block in base.entity_blocks):
    msg = "interface block id collides with an existing compiled entity block"
    raise ValueError(msg)
  if any(
    operator.header.block_id == interface_operator.header.block_id
    for operator in base.operators
  ):
    msg = "interface operator block id collides with an existing compiled operator"
    raise ValueError(msg)
  space_ids = {space.space_id for space in base.spaces}
  if interface_operator.header.ports[0].space_id not in space_ids:
    msg = "interface operator port references a space outside the base system"
    raise ValueError(msg)

  interface_attribution = (
    _new(
      SourceAttribution,
      kind="entity_block",
      semantic_id=interface_block.block_id,
      source=interface_block.sources[0],
    ),
    *(
      _new(
        SourceAttribution,
        kind="interface",
        semantic_id=entity_id,
        source=source,
      )
      for entity_id, source in zip(
        interface_block.entity_ids,
        interface_block.sources,
        strict=True,
      )
    ),
  )
  attributions = (*base.source_attribution, *interface_attribution)
  manifest = CanonicalManifest(
    {
      "schema": INTERFACE_SYSTEM_EXTENSION_SCHEMA,
      "base_system": base.provenance.manifest,
      "interface_entity_block": {
        "block_id": interface_block.block_id,
        "entity_ids": interface_block.entity_ids,
        "incidence": interface_block.incidence.values,
      },
      "interface_operator": interface_operator.content_manifest,
      "source_attribution": [
        {
          "kind": record.kind,
          "semantic_id": record.semantic_id,
          "source": {
            "source": record.source.source,
            "line": record.source.line,
            "column": record.source.column,
          },
        }
        for record in interface_attribution
      ],
    }
  )
  provenance = _new(
    SystemProvenance,
    schema=INTERFACE_SYSTEM_EXTENSION_SCHEMA,
    manifest=manifest,
    registry_fingerprint=base.provenance.registry_fingerprint,
    floating_dtype=base.provenance.floating_dtype,
    dense_index_dtype=base.provenance.dense_index_dtype,
    geometry_relative_tolerance=base.provenance.geometry_relative_tolerance,
  )
  return _new(
    CompiledSystem,
    instance_id=InstanceId(),
    content_fingerprint=ContentFingerprint.from_manifest(manifest),
    provenance=provenance,
    registry_snapshot=base.registry_snapshot,
    point_blocks=base.point_blocks,
    entity_blocks=(*base.entity_blocks, interface_block),
    spaces=base.spaces,
    operators=(*base.operators, interface_operator),
    source_attribution=attributions,
  )
