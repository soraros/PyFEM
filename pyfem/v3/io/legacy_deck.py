"""Legacy input-deck converter: skim-format ``.pro``/``.dat`` decks to v3 specs.

This module reads the legacy input deck subset used by the ``skims/`` parity
cases and emits authored v3 values — a :class:`~pyfem.v3.spec.model.ModelSpec`
plus a :class:`~pyfem.v3.spec.program.ProgramSpec`, one
:class:`~pyfem.v3.compile.spring.SpringDeclaration` per spring group, and one
:class:`~pyfem.v3.compile.interface.InterfaceDeclaration` per interface
group —
through the landed spec contracts only. The emitted specs compile with the
landed compiler unchanged: :func:`compile_deck` composes ``compile_system``
(with the ``springs`` channel) and ``compile_constraint_map``, and
:func:`run_deck` drives the landed ``NonlinearStaticDriver`` or, for
``RiksSolver`` decks, the landed ``RiksDriver``; nothing here re-implements
evaluation or reaches into builder internals.

Supported subset
----------------

Deck structure (``.pro`` side):

- exactly one ``input = "<mesh>.dat";`` reference, resolved relative to the
  ``.pro`` file;
- one ``solver = { ... };`` block of type ``LinearSolver``,
  ``NonlinearSolver``, or ``RiksSolver``; the nonlinear block may carry
  ``tol``, ``iterMax``, ``maxCycle``, ``dtime``, ``loadTable = [...]``,
  ``loadFunc`` (only the identity ramp ``t``), and ``fixedStep`` (parsed,
  recorded as not converted — the v3 driver owns its own cutback schedule);
  the riks block may carry ``tol``, ``iterMax``, ``optiter``, ``fixedStep``,
  ``maxLam``, and ``maxFactor`` (all converted into the arc-length settings;
  the legacy cycle cap of 1000 is fixed);
- one element block per mesh group, named after the group, of type
  ``SmallStrainContinuum`` (with a nested ``material`` block of type
  ``PlaneStress``, ``PlaneStrain``, or ``Isotropic`` carrying ``E`` and
  ``nu``, or of type ``IsotropicHardeningPlasticity`` carrying ``E``,
  ``nu``, ``syield``, and ``hard`` — the linear-hardening configuration the
  v3 stateful law ships; the legacy hardening-table
  (``EqPlasStrains``/``Stresses``) and power-law (``q``/``K``) properties
  reject, citing the M25 scoping decision), of type ``FiniteStrainContinuum``
  (with a nested ``material`` block of type ``PlaneStress`` carrying ``E``
  and ``nu`` — the legacy stateless Saint-Venant-Kirchhoff law the v3
  total-Lagrangian slice ships), ``Truss`` (carrying ``E`` and ``Area``
  directly), ``Spring`` (carrying ``k`` directly), or ``Interface`` (with a
  nested ``material`` block of type ``XuNeedleman``, ``PowerLawModeI``,
  ``ThoulessModeI``, or ``Dummy`` — the four stateless traction-separation
  laws the v3 cohesive interface family ships, carrying exactly their legacy
  parameter sets: ``Tult``/``Gc``, ``Tult``/``Gc``, ``Tult``/``Gc``/
  ``d1d3``/``d2d3``, and ``D`` respectively; the legacy ``XuNeedleman``
  constructor silently overwrites deck-supplied ``q``/``r`` with ``1.0``/
  ``0.0``, so those keys reject rather than convert dishonestly, and the
  legacy rank-3 branch — dead and broken upstream — is not admitted);
- exactly one region-routed element group (``SmallStrainContinuum``,
  ``FiniteStrainContinuum``, or ``Truss``) plus any number of ``Spring``
  and ``Interface`` groups; the region-routed group forms the base
  ``ModelSpec`` and each spring group emits one
  :class:`~pyfem.v3.compile.spring.SpringDeclaration`, composed through
  ``compile_system``'s ``springs`` channel. Each two-node spring element
  must have exactly one end with every component prescribed to zero (the
  grounded end): it then reduces exactly to the v3 grounded point spring at
  the other node along the element chord, the consistent axial form the H1
  decision pins (the legacy isotropic spring tangent is not the oracle — its
  axial residual is). Each four-node interface element (bottom pair, then
  top pair) emits into one
  :class:`~pyfem.v3.compile.interface.InterfaceDeclaration` per group,
  composed after the springs; the v3 operator ships the corrected
  (non-reflecting) corotational frame and the exact tangent, so residual
  parity is the contract on axis-aligned decks and the tangent divergence
  from legacy is documented (see
  :mod:`pyfem.v3.compile.interface`);
- ``outputModules = [...]`` naming blocks whose ``type`` is a known legacy
  writer (``MeshWriter``, ``OutputWriter``, ``GraphWriter``, ``HDF5Writer``,
  ``DataDump``, ``ContourWriter``, ``ROMSnapshotWriter``). Writers do not
  affect the computed state; each is recorded in
  :attr:`ConvertedDeck.not_converted` rather than silently skipped.

Mesh structure (``.dat`` side):

- ``<Nodes>``: ``id x y;`` statements on 2D decks, ``id x y z;`` on 3D
  decks;
- ``<Elements>``: ``id "Group" n1 ... nk;`` statements — two nodes for
  ``Truss`` (line2) and ``Spring``; three (tria3), four (quad4), or eight
  (serendipity quad8) nodes for 2D ``SmallStrainContinuum``; eight nodes
  (hex8) for 3D ``SmallStrainContinuum``; eight nodes (serendipity quad8)
  for 2D ``FiniteStrainContinuum``;
- ``<NodeConstraints>``: prescribed DOFs ``u[i] = value;`` / ``v[i] = value;``
  (plus ``w[i] = value;`` on 3D decks) and one-master affine ties
  ``u[i] = offset + factor * v[j];`` (factor may be omitted or follow the
  master: ``u[i] = v[j];``, ``u[i] = 2.0 * u[j];``);
- ``<ExternalForces>``: nodal loads ``v[i] = value;``.

Semantics: every deck declares one ``load`` program coordinate. Under
``LinearSolver`` prescribed values and tie offsets are constant and loads scale
linearly to a single full step (matching the legacy full-value application);
under ``NonlinearSolver`` prescribed values, tie offsets, and loads all scale
with the load coordinate, whose targets are the authored ``loadTable`` values
or the ``dtime``/``maxCycle`` ramp. Under ``RiksSolver`` prescribed values
and tie offsets are constant (the legacy solver never scales constraints)
while loads scale with the arc-length parameter the continuation owns;
``run_deck`` routes the deck to the landed ``RiksDriver`` and terminates on
the legacy conditions (``lam > maxLam`` or more than 1000 committed cycles,
checked after each commit). DOF types map ``u -> x``, ``v -> y`` on
2D decks and additionally ``w -> z`` on 3D decks, on the node-supported
``displacement`` field. The ``PlaneStress`` and ``PlaneStrain`` laws require
a 2D mesh; the ``Isotropic`` law requires a 3D mesh;
``IsotropicHardeningPlasticity`` requires a 2D serendipity-quad8 mesh.
``FiniteStrainContinuum`` requires a 2D serendipity-quad8 mesh and a
``NonlinearSolver`` block (the legacy ``LinearSolver`` assembles the
zero-state tangent once, while the v3 driver integrates the finite-strain
residual by Newton iteration — the combination has no faithful mapping).
``Spring`` groups require a 2D mesh and a Newton-based solver block
(``NonlinearSolver`` or ``RiksSolver``): the legacy ``LinearSolver``
assembles the legacy isotropic spring tangent once, while the v3 spring
family ships the consistent axial tangent, so the single linear solve has no
faithful mapping.

Rejection codes
---------------

Every out-of-subset construct raises :class:`DeckConversionError` carrying
coded :class:`~pyfem.v3.spec.diagnostics.SpecDiagnostic` values with source
context — never a silent skip. Known legacy constructs and their codes:

- ``deck-input-missing`` — unreadable ``.pro`` file, absent ``input = ...``
  reference, or an unreadable referenced mesh file;
- ``deck-pro-syntax`` / ``deck-dat-syntax`` — malformed ``.pro`` grammar
  (bad token, missing ``;``/``=``/``}``, mistyped or non-finite value) or
  malformed ``.dat`` statements (bad node/element/constraint line, statements
  outside any section, unterminated statements or sections);
- ``deck-solver-missing`` — no ``solver`` block;
- ``unsupported-pro-construct`` — ``include`` directives, dotted keys
  (``a.b = ...``), duplicate or unknown top-level items;
- ``unsupported-output-module`` — an ``outputModules`` entry without a block,
  or a block ``type`` outside the known writer set;
- ``unsupported-element-type`` — element block type outside
  ``{SmallStrainContinuum, FiniteStrainContinuum, Truss, Spring,
  Interface}`` (e.g. ``Beam``, ``Contact``);
- ``unsupported-element-parameter`` / ``missing-element-parameter`` —
  extra keys in an element block, or a ``Truss`` block without ``E``/``Area``
  or a ``Spring`` block without ``k``;
- ``missing-element-block`` — a mesh group with no same-named ``.pro`` block;
- ``unsupported-element-groups`` — the mesh declares anything but exactly
  one region-routed element group: several region-routed groups, or none
  (a springs-only or interfaces-only deck); ``Spring`` and ``Interface``
  groups are auxiliary and unrestricted in number;
- ``unsupported-spring-support`` — a ``Spring`` element whose support
  pattern has no faithful grounded point-spring mapping: both ends free, both
  ends fully prescribed, an anchor end with a nonzero or partial
  prescription, or coincident endpoints;
- ``unsupported-material-model`` — material type outside
  ``{PlaneStress, PlaneStrain, Isotropic, IsotropicHardeningPlasticity}``
  on a ``SmallStrainContinuum`` block, outside ``{PlaneStress}`` on a
  ``FiniteStrainContinuum`` block (e.g. ``IsotropicKinematicHardening``), or
  outside ``{XuNeedleman, PowerLawModeI, ThoulessModeI, Dummy}`` on an
  ``Interface`` block;
- ``incompatible-solver-type`` — a ``FiniteStrainContinuum`` deck whose
  solver block is a ``LinearSolver`` (legacy assembles the zero-state
  tangent once; the v3 driver only integrates the finite-strain residual by
  Newton iteration, so the combination has no faithful mapping), a deck
  with ``Spring`` groups whose solver block is a ``LinearSolver`` (legacy
  assembles its isotropic spring tangent once — not the oracle per the H1
  decision — while the v3 spring family ships the consistent axial tangent,
  so the single linear solve has no faithful mapping), or a deck whose
  ``Interface`` groups carry a nonlinear law (``XuNeedleman``,
  ``PowerLawModeI``, ``ThoulessModeI``) under a ``LinearSolver`` (the same
  one-shot mismatch; the linear ``Dummy`` law converts, since its assembled
  residual is frame-independent and the single linear solve is exact);
- ``incompatible-material-geometry`` — a supported material on an
  incompatible mesh: ``PlaneStress``/``PlaneStrain`` on a 3D mesh,
  ``Isotropic`` on a 2D mesh;
- ``unsupported-hardening-form`` — an ``IsotropicHardeningPlasticity``
  material carrying the legacy hardening-table (``EqPlasStrains``,
  ``Stresses``) or power-law (``q``, ``K``) properties; the v3 stateful law
  ships exactly the linear-hardening ``hard`` configuration (the M25 scoping
  decision: the legacy law's plastic branch reads its ``hard`` property for
  the tangent, so no other hardening form ever defined plastic behavior);
- ``unsupported-material-parameter`` / ``missing-material-parameter`` —
  extra keys in a material block, or a block without its model's required
  properties (``E``/``nu`` for ``PlaneStress``/``PlaneStrain``/``Isotropic``;
  ``E``/``nu``/``syield``/``hard`` for ``IsotropicHardeningPlasticity``;
  ``Tult``/``Gc`` for ``XuNeedleman`` and ``PowerLawModeI`` — whose ``q``/``r``
  keys reject here, since legacy silently overwrites them —
  ``Tult``/``Gc``/``d1d3``/``d2d3`` for ``ThoulessModeI``; ``D`` for
  ``Dummy``);
- ``invalid-material-parameter-value`` — an interface law parameter that is
  non-positive where the law requires positive (``Tult``/``Gc``/``D``), or a
  ``ThoulessModeI`` ratio pair outside ``0 < d1d3 < d2d3 < 1``;
- ``unsupported-solver-type`` — solver type outside
  ``{LinearSolver, NonlinearSolver, RiksSolver}``;
- ``unsupported-solver-parameter`` — solver key outside the supported set;
- ``unsupported-load-func`` — ``loadFunc`` other than the identity ramp ``t``;
- ``unsupported-load-table`` — an empty or malformed ``loadTable``;
- ``unsupported-section`` — unknown or unsupported ``.dat`` sections
  (e.g. ``<NodeGroup>``, named ``<NodeConstraints name=...>`` tables,
  duplicate sections);
- ``unsupported-gmsh-reference`` — ``gmsh = "file.msh";`` mesh references;
- ``unsupported-mesh-rank`` — node coordinates that are not 2D or 3D;
- ``unsupported-cell-arity`` — cell node counts with no landed v3 geometry
  for the deck's element type and mesh rank (e.g. five-node cells, non-hex8
  cells on a 3D mesh, non-line2 cells for ``Truss``, non-quad8 cells for the
  stateful law or the finite-strain family, non-four-node cells for
  ``Interface``), or mixed cell arities inside one mesh group;
- ``unsupported-dof-type`` — DOF types outside ``{u, v, w}``, or ``w`` on a
  2D mesh;
- ``unsupported-node-group`` — named node-group references in constraint
  left- or right-hand sides (``u[name]``);
- ``unsupported-tie-form`` — constraint right-hand sides outside the
  one-master affine form (e.g. multi-master relations);
- ``duplicate-node-id`` / ``duplicate-cell-id`` / ``unknown-node-reference`` /
  ``inconsistent-node-coordinates`` / ``empty-mesh`` — structurally malformed
  meshes.

Two further codes are non-fatal acknowledgments, collected in
:attr:`ConvertedDeck.not_converted` instead of raising:
``not-converted-output-module`` (a known legacy writer block — output does not
affect the computed state) and ``not-converted-solver-option``
(``fixedStep`` on a ``NonlinearSolver`` block, and a ``maxCycle``/``dtime``
ramp overridden by ``loadTable`` — the v3 driver owns its own substep
schedule; on a ``RiksSolver`` block every parsed key converts).
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

import numpy as np

from pyfem.v3.compile.continuum import (
  plasticity_reference_registry,
  q8_reference_registry,
)
from pyfem.v3.compile.contracts import (
  continuum_reference_registry,
  finite_strain_reference_registry,
)
from pyfem.v3.compile.interface import (
  InterfaceDeclaration,
  compile_interface_operator,
  compose_interface_system,
  dummy_interface_declaration,
  power_law_mode_i_declaration,
  thouless_mode_i_declaration,
  xu_needleman_declaration,
)
from pyfem.v3.compile.spring import SpringDeclaration, SpringKernelResult
from pyfem.v3.compile.system import compile_system
from pyfem.v3.compile.truss import truss_reference_registry
from pyfem.v3.constraints import CompiledConstraintMap, compile_constraint_map
from pyfem.v3.driver import (
  ArcLengthResult,
  ArcLengthSettings,
  NonlinearStaticDriver,
  NonlinearStaticResult,
  NonlinearStaticSettings,
  RiksDriver,
)
from pyfem.v3.model.operator import EvaluationStatus
from pyfem.v3.model.registry import RegistryDescriptor, RegistryKey
from pyfem.v3.model.system import CompiledSystem
from pyfem.v3.spec.diagnostics import SourceContext, SpecDiagnostic
from pyfem.v3.spec.model import (
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
from pyfem.v3.spec.program import (
  AffineCoefficientSpec,
  AffineTieSpec,
  AffineValueSpec,
  DofRef,
  NodalLoadSpec,
  PrescribedDofSpec,
  ProgramConstraintSpec,
  ProgramCoordinateSpec,
  ProgramCoordinateValue,
  ProgramPoint,
  ProgramSpec,
)

__all__ = [
  "CompiledDeck",
  "ConvertedDeck",
  "DeckConversionError",
  "DeckRun",
  "DeckSolverSettings",
  "compile_deck",
  "read_legacy_deck",
  "run_deck",
]

_LOAD_COORDINATE = "load"
_FIELD_ID = "displacement"
_DOF_COMPONENTS = {"u": "x", "v": "y", "w": "z"}
_KNOWN_OUTPUT_TYPES = frozenset(
  {
    "ContourWriter",
    "DataDump",
    "GraphWriter",
    "HDF5Writer",
    "MeshWriter",
    "OutputWriter",
    "ROMSnapshotWriter",
  }
)
_NONLINEAR_SOLVER_KEYS = frozenset(
  {"type", "tol", "iterMax", "maxCycle", "dtime", "loadTable", "loadFunc", "fixedStep"}
)
_RIKS_SOLVER_KEYS = frozenset(
  {"type", "tol", "iterMax", "optiter", "fixedStep", "maxLam", "maxFactor"}
)
# Legacy ``RiksSolver`` defaults (pyfem/solvers/RiksSolver.py): the cycle cap
# is hard-coded on the legacy side (``cycle > 1000`` terminates the schedule).
_RIKS_DEFAULT_TOLERANCE = 1.0e-5
_RIKS_DEFAULT_MAX_ITERATIONS = 10
_RIKS_DEFAULT_OPTIMAL_ITERATIONS = 5
_RIKS_DEFAULT_MAX_LAM = 1.0e20
_RIKS_DEFAULT_MAX_FACTOR = 1.0e20
_RIKS_CYCLE_CAP = 1000
# A legacy ``Spring`` group converts to the grounded point-spring family:
# two-node cells on a 2D mesh, exactly one end prescribed to zero.
_SPRING_GEOMETRIES = frozenset({(2, 2)})
_SPRING_STATE_SCHEMA = "legacy-deck-spring-state-v1"
_SPRING_KERNEL_NAME = "legacy-axial-point-spring"
_SPRING_IMPLEMENTATION_ID = "legacy-axial-point-spring-v1"
# A legacy ``Interface`` group converts to the cohesive interface family:
# four-node (bottom pair, top pair) cells on a 2D mesh — de-facto Gauss-2
# integration, since the legacy ``NewtonCotes`` flag never existed.
_INTERFACE_GEOMETRIES = frozenset({(2, 4)})
_INTERFACE_STATE_SCHEMA = "legacy-deck-interface-frame-state-v1"
# Legacy interface material type -> its required parameter keys, in the
# v3 declaration's canonical order (Tult -> ultimate_traction, Gc ->
# fracture_energy, D -> stiffness).
_INTERFACE_LAW_PARAMETERS: dict[str, tuple[str, ...]] = {
  "XuNeedleman": ("Tult", "Gc"),
  "PowerLawModeI": ("Tult", "Gc"),
  "ThoulessModeI": ("Tult", "Gc", "d1d3", "d2d3"),
  "Dummy": ("D",),
}
_PLASTICITY_MODEL = "IsotropicHardeningPlasticity"
_PLASTICITY_VALUE_KEYS = frozenset({"E", "nu", "syield", "hard"})
_PLASTICITY_TABLE_KEYS = frozenset({"EqPlasStrains", "Stresses", "q", "K"})
# Legacy elastic material type -> (v1 material model, required mesh rank).
_ELASTIC_MATERIAL_MODELS = {
  "PlaneStress": ("plane-stress-linear-elastic", 2),
  "PlaneStrain": ("plane-strain-linear-elastic", 2),
  "Isotropic": ("isotropic-linear-elastic", 3),
}


class DeckConversionError(ValueError):
  """Raised with every coded diagnostic produced while converting a deck."""

  diagnostics: tuple[SpecDiagnostic, ...]

  def __init__(self, diagnostics: Iterable[SpecDiagnostic]) -> None:
    self.diagnostics = tuple(diagnostics)
    if not self.diagnostics:
      msg = "DeckConversionError requires at least one diagnostic"
      raise ValueError(msg)
    super().__init__("\n".join(item.render() for item in self.diagnostics))


@dataclass(frozen=True, slots=True)
class DeckSolverSettings:
  """Typed solver schedule harvested from the deck's solver block.

  ``load_factors`` holds the absolute target values of the ``load`` program
  coordinate in schedule order (the single full step ``(1.0,)`` for
  ``LinearSolver`` decks); it is empty for ``RiksSolver`` decks, whose
  continuation owns the load parameter autonomously. ``tolerance`` and
  ``max_iterations`` follow the legacy ``NonlinearSolver`` defaults when the
  deck leaves them unset. The remaining fields carry the ``RiksSolver``
  arc-length policy — the legacy defaults when the riks block leaves them
  unset, and the fixed legacy cycle cap of 1000 — and are inert on
  ``LinearSolver``/``NonlinearSolver`` decks.
  """

  solver_type: str
  load_factors: tuple[float, ...]
  tolerance: float
  max_iterations: int
  optimal_iterations: int = 5
  fixed_step: bool = False
  max_lam: float = 1.0e20
  max_factor: float = 1.0e20
  cycle_cap: int = 1000


@dataclass(frozen=True, slots=True, eq=False)
class ConvertedDeck:
  """One converted legacy deck: authored specs plus their execution context.

  ``model`` and ``program`` are plain authored spec values; ``springs``
  holds one :class:`~pyfem.v3.compile.spring.SpringDeclaration` per parsed
  spring group and ``interfaces`` one
  :class:`~pyfem.v3.compile.interface.InterfaceDeclaration` per parsed
  interface group (both empty on a bare continuum deck), composed through
  ``compile_system``'s ``springs`` channel and the interface family's
  compile/compose pair; ``not_converted`` lists the
  acknowledged non-physics constructs (output writers, subsumed solver
  policy flags) with their source contexts.
  """

  name: str
  pro_path: Path
  dat_path: Path
  model: ModelSpec
  program: ProgramSpec
  registry: dict[RegistryKey, RegistryDescriptor]
  solver: DeckSolverSettings
  not_converted: tuple[SpecDiagnostic, ...]
  springs: tuple[SpringDeclaration, ...] = ()
  interfaces: tuple[InterfaceDeclaration, ...] = ()


@dataclass(frozen=True, slots=True, eq=False)
class CompiledDeck:
  """A converted deck compiled through the landed compiler, unchanged."""

  deck: ConvertedDeck
  system: CompiledSystem
  constraint_map: CompiledConstraintMap
  loads: tuple[NodalLoadSpec, ...]


@dataclass(frozen=True, slots=True, eq=False)
class DeckRun:
  """One finished driver execution of a converted deck.

  ``LinearSolver`` and ``NonlinearSolver`` decks run through the landed
  ``NonlinearStaticDriver``; ``RiksSolver`` decks run through the landed
  ``RiksDriver`` and return its :class:`ArcLengthResult`.
  """

  driver: NonlinearStaticDriver | RiksDriver
  result: NonlinearStaticResult | ArcLengthResult

  @property
  def state(self) -> np.ndarray:
    """Return the committed physical coefficient vector after the run."""
    return self.driver.owner.accepted_physical().values


def _diagnostic(code: str, message: str, source: SourceContext) -> SpecDiagnostic:
  return SpecDiagnostic(code=code, message=message, source=source)


def _raise(code: str, message: str, source: SourceContext) -> NoReturn:
  raise DeckConversionError((_diagnostic(code, message, source),))


# --- .pro grammar --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Token:
  kind: str
  text: str
  line: int


_PRO_TOKEN_RE = re.compile(
  r"(?P<comment>\#|//)[^\n]*"
  r"|(?P<string>\"(?:[^\"\\]|\\.)*\")"
  r"|(?P<number>[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)"
  r"|(?P<word>[A-Za-z_][\w.]*)"
  r"|(?P<punct>[=\{\}\[\];,])"
  r"|(?P<space>\s+)"
  r"|(?P<bad>[^\s])"
)


def _tokenize_pro(text: str, path: Path) -> tuple[_Token, ...]:
  tokens: list[_Token] = []
  line = 1
  for match in _PRO_TOKEN_RE.finditer(text):
    kind = match.lastgroup
    token_text = match.group()
    if kind is None:
      continue
    if kind == "bad":
      _raise(
        "deck-pro-syntax",
        f"unexpected character {token_text!r}",
        SourceContext(source=str(path), line=line),
      )
    if kind in ("space", "comment"):
      line += token_text.count("\n")
      continue
    tokens.append(_Token(kind=kind, text=token_text, line=line))
    line += token_text.count("\n")
  return tuple(tokens)


@dataclass(frozen=True, slots=True)
class _ProValue:
  value: str | bool | int | float | tuple[_ProValue, ...]
  source: SourceContext


@dataclass(frozen=True, slots=True)
class _ProAssignment:
  name: str
  value: _ProValue
  source: SourceContext


@dataclass(frozen=True, slots=True)
class _ProBlock:
  name: str
  items: tuple[_ProAssignment | _ProBlock, ...]
  source: SourceContext


class _ProParser:
  """Recursive-descent parser for the legacy ``.pro`` assignment grammar."""

  def __init__(self, tokens: tuple[_Token, ...], path: Path) -> None:
    self._tokens = tokens
    self._path = path
    self._pos = 0
    self.diagnostics: list[SpecDiagnostic] = []

  def parse(self) -> tuple[_ProAssignment | _ProBlock, ...]:
    return self._parse_items(depth=0)

  def _source(self, token: _Token | None) -> SourceContext:
    line = token.line if token is not None else None
    return SourceContext(source=str(self._path), line=line)

  def _syntax(self, message: str, token: _Token | None) -> NoReturn:
    _raise("deck-pro-syntax", message, self._source(token))

  def _peek(self) -> _Token | None:
    if self._pos < len(self._tokens):
      return self._tokens[self._pos]
    return None

  def _advance(self) -> _Token:
    token = self._peek()
    if token is None:
      self._syntax("unexpected end of file", self._tokens[-1] if self._tokens else None)
    self._pos += 1
    return token

  def _expect_punct(self, text: str) -> _Token:
    token = self._advance()
    if token.kind != "punct" or token.text != text:
      self._syntax(f"expected {text!r}, found {token.text!r}", token)
    return token

  def _parse_items(self, *, depth: int) -> tuple[_ProAssignment | _ProBlock, ...]:
    items: list[_ProAssignment | _ProBlock] = []
    while True:
      token = self._peek()
      if token is None:
        if depth > 0:
          self._syntax("unclosed block — missing '}'", self._tokens[-1])
        return tuple(items)
      if token.kind == "punct" and token.text == "}":
        if depth == 0:
          self._syntax("unmatched '}'", token)
        return tuple(items)
      item = self._parse_item()
      if item is not None:
        items.append(item)

  def _parse_item(self) -> _ProAssignment | _ProBlock | None:
    name = self._advance()
    if name.kind != "word":
      self._syntax(f"expected a block or assignment name, found {name.text!r}", name)
    if name.text == "include" or "." in name.text:
      construct = "include directive" if name.text == "include" else "dotted key"
      self.diagnostics.append(
        _diagnostic(
          "unsupported-pro-construct",
          f"{construct} {name.text!r} is outside the supported deck subset",
          self._source(name),
        ),
      )
      self._skip_statement()
      return None
    self._expect_punct("=")
    token = self._peek()
    if token is not None and token.kind == "punct" and token.text == "{":
      self._advance()
      body = self._parse_items(depth=1)
      self._expect_punct("}")
      closing = self._peek()
      if closing is not None and closing.kind == "punct" and closing.text == ";":
        self._advance()
      return _ProBlock(name=name.text, items=body, source=self._source(name))
    value = self._parse_value()
    self._expect_punct(";")
    return _ProAssignment(name=name.text, value=value, source=self._source(name))

  def _skip_statement(self) -> None:
    depth = 0
    while True:
      token = self._peek()
      if token is None:
        return
      self._advance()
      if token.kind == "punct":
        if token.text in ("{", "["):
          depth += 1
        elif token.text in ("}", "]"):
          depth -= 1
        elif token.text == ";" and depth <= 0:
          return

  def _parse_value(self) -> _ProValue:
    token = self._advance()
    if token.kind == "string":
      return _ProValue(value=token.text[1:-1], source=self._source(token))
    if token.kind == "number":
      if any(mark in token.text for mark in (".", "e", "E")):
        number: int | float = float(token.text)
        if not math.isfinite(number):
          self._syntax(f"non-finite number {token.text!r}", token)
      else:
        number = int(token.text)
      return _ProValue(value=number, source=self._source(token))
    if token.kind == "word":
      if token.text == "true":
        return _ProValue(value=True, source=self._source(token))
      if token.text == "false":
        return _ProValue(value=False, source=self._source(token))
      return _ProValue(value=token.text, source=self._source(token))
    if token.kind == "punct" and token.text == "[":
      values: list[_ProValue] = []
      while True:
        token = self._peek()
        if token is None:
          self._syntax("unclosed list — missing ']'", self._tokens[-1])
        if token.kind == "punct" and token.text == "]":
          self._advance()
          break
        values.append(self._parse_value())
        separator = self._advance()
        if separator.kind == "punct" and separator.text == "]":
          break
        if separator.kind != "punct" or separator.text != ",":
          self._syntax("expected ',' or ']' in list", separator)
      return _ProValue(value=tuple(values), source=self._source(token))
    self._syntax(f"expected a value, found {token.text!r}", token)


@dataclass(frozen=True, slots=True)
class _ProDeck:
  input_ref: str | None
  solver_block: _ProBlock | None
  element_blocks: tuple[_ProBlock, ...]


def _string_assignment(
  item: _ProAssignment | _ProBlock,
  *,
  label: str,
  diagnostics: list[SpecDiagnostic],
) -> tuple[str, SourceContext] | None:
  if type(item) is not _ProAssignment or type(item.value.value) is not str:
    diagnostics.append(
      _diagnostic(
        "unsupported-pro-construct",
        f"{label} must be a quoted string assignment",
        item.source,
      ),
    )
    return None
  return item.value.value, item.source


def _walk_pro(
  items: tuple[_ProAssignment | _ProBlock, ...],
  diagnostics: list[SpecDiagnostic],
  not_converted: list[SpecDiagnostic],
) -> _ProDeck:
  input_ref: str | None = None
  solver_block: _ProBlock | None = None
  output_names: tuple[str, ...] = ()
  output_source = SourceContext()
  blocks: list[_ProBlock] = []
  seen: set[str] = set()

  for item in items:
    if item.name in seen:
      diagnostics.append(
        _diagnostic(
          "unsupported-pro-construct",
          f"duplicate top-level item {item.name!r}",
          item.source,
        ),
      )
      continue
    seen.add(item.name)
    if type(item) is _ProBlock:
      if item.name == "solver":
        solver_block = item
      else:
        blocks.append(item)
      continue
    if item.name == "input":
      parsed = _string_assignment(item, label="input", diagnostics=diagnostics)
      if parsed is not None:
        input_ref = parsed[0]
    elif item.name == "outputModules":
      output_names = _parse_output_modules(item, diagnostics)
      output_source = item.source
    else:
      diagnostics.append(
        _diagnostic(
          "unsupported-pro-construct",
          f"unsupported top-level assignment {item.name!r}",
          item.source,
        ),
      )

  output_name_set = set(output_names)
  element_blocks: list[_ProBlock] = []
  for block in blocks:
    if block.name in output_name_set:
      _check_output_module(block, diagnostics, not_converted)
    else:
      element_blocks.append(block)
  for name in output_names:
    if all(block.name != name for block in blocks):
      diagnostics.append(
        _diagnostic(
          "unsupported-output-module",
          f"declared output module {name!r} has no matching block",
          output_source,
        ),
      )

  return _ProDeck(
    input_ref=input_ref,
    solver_block=solver_block,
    element_blocks=tuple(element_blocks),
  )


def _parse_output_modules(
  item: _ProAssignment,
  diagnostics: list[SpecDiagnostic],
) -> tuple[str, ...]:
  value = item.value.value
  if type(value) is not tuple or any(type(entry.value) is not str for entry in value):
    diagnostics.append(
      _diagnostic(
        "unsupported-pro-construct",
        "outputModules must be a list of module names",
        item.source,
      ),
    )
    return ()
  return tuple(entry.value for entry in value if type(entry.value) is str)


def _check_output_module(
  block: _ProBlock,
  diagnostics: list[SpecDiagnostic],
  not_converted: list[SpecDiagnostic],
) -> None:
  module_type: str | None = None
  for item in block.items:
    if type(item) is _ProAssignment and item.name == "type":
      raw = item.value.value
      if type(raw) is str:
        module_type = raw
  if module_type not in _KNOWN_OUTPUT_TYPES:
    diagnostics.append(
      _diagnostic(
        "unsupported-output-module",
        f"output module {block.name!r} has unsupported type "
        f"{module_type!r}; known writers: {sorted(_KNOWN_OUTPUT_TYPES)}",
        block.source,
      ),
    )
    return
  not_converted.append(
    _diagnostic(
      "not-converted-output-module",
      f"output module {block.name!r} of type {module_type!r} does not affect "
      "the computed state and is intentionally not converted",
      block.source,
    ),
  )


# --- .dat grammar --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _DatNode:
  id: int
  coordinates: tuple[float, ...]
  source: SourceContext


@dataclass(frozen=True, slots=True)
class _DatElement:
  id: int
  group: str
  node_ids: tuple[int, ...]
  source: SourceContext


@dataclass(frozen=True, slots=True)
class _DatPrescribed:
  node_id: int
  dof_type: str
  value: float
  source: SourceContext


@dataclass(frozen=True, slots=True)
class _DatTie:
  slave_node_id: int
  slave_dof_type: str
  master_node_id: int
  master_dof_type: str
  factor: float
  offset: float
  source: SourceContext


@dataclass(frozen=True, slots=True)
class _DatLoad:
  node_id: int
  dof_type: str
  value: float
  source: SourceContext


@dataclass(frozen=True, slots=True)
class _DatDeck:
  nodes: tuple[_DatNode, ...]
  elements: tuple[_DatElement, ...]
  prescribed: tuple[_DatPrescribed, ...]
  ties: tuple[_DatTie, ...]
  loads: tuple[_DatLoad, ...]


_SECTION_TAGS = frozenset({"Nodes", "Elements", "NodeConstraints", "ExternalForces"})
_SECTION_TAG_RE = re.compile(
  r"^<(?P<closing>/?)(?P<name>[A-Za-z]\w*)(?P<attrs>[^>]*)>$"
)
_GMSH_RE = re.compile(r"^gmsh\s*=")
_CONSTRAINT_LHS_RE = re.compile(
  r"^([A-Za-z]\w*)\s*\[\s*([^\]]+?)\s*\]\s*=\s*(.*)$",
  re.S,
)
_GROUP_NAME_RE = re.compile(r"[A-Za-z_]\w*")
_TIE_DOF_RE = re.compile(r"([A-Za-z]\w*)\[([^\]]+)\]")
_TIE_TOKEN_RE = re.compile(
  r"(?P<number>[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)"
  r"|(?P<dof>[A-Za-z]\w*\[[^\]]*\])"
  r"|(?P<star>\*)"
  r"|(?P<sign>\+|-)"
  r"|(?P<space>\s+)"
  r"|(?P<bad>.)"
)


def _strip_dat_comment(line: str) -> str:
  quote: str | None = None
  for index, char in enumerate(line):
    if quote is not None:
      if char == quote:
        quote = None
    elif char in ('"', "'"):
      quote = char
    elif char == "#":
      return line[:index]
    elif char == "/" and line[index + 1 : index + 2] == "/":
      return line[:index]
  return line


def _dat_int(text: str) -> int | None:
  try:
    return int(text)
  except ValueError:
    return None


def _dat_float(text: str) -> float | None:
  try:
    value = float(text)
  except ValueError:
    return None
  if not math.isfinite(value):
    return None
  return value


def _read_dat_deck(
  text: str,
  path: Path,
  diagnostics: list[SpecDiagnostic],
) -> _DatDeck:
  nodes: list[_DatNode] = []
  elements: list[_DatElement] = []
  prescribed: list[_DatPrescribed] = []
  ties: list[_DatTie] = []
  loads: list[_DatLoad] = []

  section: str | None = None
  skip_tag: str | None = None
  seen_sections: set[str] = set()
  buffer = ""
  buffer_line = 1
  line_number = 0

  def source(line: int) -> SourceContext:
    return SourceContext(source=str(path), line=line)

  for line_number, raw_line in enumerate(text.splitlines(), start=1):
    line = _strip_dat_comment(raw_line).strip()
    if not line:
      continue

    tag = _SECTION_TAG_RE.match(line)
    if tag is not None:
      if buffer.strip():
        diagnostics.append(
          _diagnostic(
            "deck-dat-syntax",
            f"statement {buffer.strip()!r} is not terminated by ';'",
            source(buffer_line),
          ),
        )
      buffer = ""
      name = tag.group("name")
      attrs = tag.group("attrs").strip()
      if tag.group("closing"):
        if skip_tag is not None:
          if name == skip_tag:
            skip_tag = None
          continue
        if section is None or name != section:
          diagnostics.append(
            _diagnostic(
              "deck-dat-syntax",
              f"closing tag </{name}> does not match the open section",
              source(line_number),
            ),
          )
          continue
        section = None
        continue
      if section is not None or skip_tag is not None:
        diagnostics.append(
          _diagnostic(
            "deck-dat-syntax",
            f"nested section <{name}> inside an open section",
            source(line_number),
          ),
        )
        continue
      if name in seen_sections:
        diagnostics.append(
          _diagnostic(
            "unsupported-section",
            f"duplicate <{name}> section",
            source(line_number),
          ),
        )
        skip_tag = name
        continue
      seen_sections.add(name)
      if name not in _SECTION_TAGS or attrs:
        reason = (
          f"unknown section <{name}>"
          if name not in _SECTION_TAGS
          else f"section <{name}> with attributes (named tables or groups)"
        )
        diagnostics.append(
          _diagnostic(
            "unsupported-section",
            f"{reason} is outside the supported deck subset",
            source(line_number),
          ),
        )
        skip_tag = name
        continue
      section = name
      continue

    if skip_tag is not None:
      continue
    if section is None:
      code = "unsupported-gmsh-reference" if _GMSH_RE.match(line) else "deck-dat-syntax"
      message = (
        "gmsh mesh references are outside the supported deck subset"
        if code == "unsupported-gmsh-reference"
        else f"statement {line!r} appears outside any section"
      )
      diagnostics.append(_diagnostic(code, message, source(line_number)))
      continue

    if not buffer.strip():
      buffer = ""
      buffer_line = line_number
    buffer += f"{line} "
    while ";" in buffer:
      statement, buffer = buffer.split(";", 1)
      statement = statement.strip()
      if statement:
        _parse_dat_statement(
          section,
          statement,
          source(buffer_line),
          diagnostics,
          nodes,
          elements,
          prescribed,
          ties,
          loads,
        )
      buffer_line = line_number

  if section is not None:
    diagnostics.append(
      _diagnostic(
        "deck-dat-syntax",
        f"unclosed <{section}> section",
        source(max(line_number, 1)),
      ),
    )
  if skip_tag is not None:
    diagnostics.append(
      _diagnostic(
        "deck-dat-syntax",
        f"unclosed skipped section <{skip_tag}>",
        source(max(line_number, 1)),
      ),
    )
  if buffer.strip():
    diagnostics.append(
      _diagnostic(
        "deck-dat-syntax",
        f"statement {buffer.strip()!r} is not terminated by ';'",
        source(buffer_line),
      ),
    )

  return _DatDeck(
    nodes=tuple(nodes),
    elements=tuple(elements),
    prescribed=tuple(prescribed),
    ties=tuple(ties),
    loads=tuple(loads),
  )


def _parse_dat_statement(
  section: str,
  statement: str,
  source: SourceContext,
  diagnostics: list[SpecDiagnostic],
  nodes: list[_DatNode],
  elements: list[_DatElement],
  prescribed: list[_DatPrescribed],
  ties: list[_DatTie],
  loads: list[_DatLoad],
) -> None:
  if section == "Nodes":
    _parse_dat_node(statement, source, diagnostics, nodes)
  elif section == "Elements":
    _parse_dat_element(statement, source, diagnostics, elements)
  elif section == "NodeConstraints":
    _parse_dat_constraint(statement, source, diagnostics, prescribed, ties)
  else:
    _parse_dat_load(statement, source, diagnostics, loads)


def _parse_dat_node(
  statement: str,
  source: SourceContext,
  diagnostics: list[SpecDiagnostic],
  nodes: list[_DatNode],
) -> None:
  parts = statement.split()
  node_id = _dat_int(parts[0]) if parts else None
  coordinates = tuple(_dat_float(part) for part in parts[1:])
  if node_id is None or not coordinates or any(value is None for value in coordinates):
    diagnostics.append(
      _diagnostic(
        "deck-dat-syntax",
        f"malformed node statement {statement!r} (expected 'id x y [z]')",
        source,
      ),
    )
    return
  nodes.append(
    _DatNode(
      id=node_id,
      coordinates=tuple(value for value in coordinates if value is not None),
      source=source,
    ),
  )


def _parse_dat_element(
  statement: str,
  source: SourceContext,
  diagnostics: list[SpecDiagnostic],
  elements: list[_DatElement],
) -> None:
  parts = statement.split()
  element_id = _dat_int(parts[0]) if parts else None
  group = parts[1] if len(parts) > 1 else None
  node_ids = tuple(_dat_int(part) for part in parts[2:])
  quoted = (
    group is not None
    and len(group) >= 2
    and group[0] == group[-1]
    and group[0] in ('"', "'")
  )
  if (
    element_id is None
    or not quoted
    or not node_ids
    or any(node_id is None for node_id in node_ids)
  ):
    diagnostics.append(
      _diagnostic(
        "deck-dat-syntax",
        f"malformed element statement {statement!r} "
        "(expected 'id \"Group\" n1 ... nk')",
        source,
      ),
    )
    return
  elements.append(
    _DatElement(
      id=element_id,
      group=group[1:-1],
      node_ids=tuple(node_id for node_id in node_ids if node_id is not None),
      source=source,
    ),
  )


def _constraint_target(
  lhs: re.Match[str],
  diagnostics: list[SpecDiagnostic],
  source: SourceContext,
) -> tuple[str, int] | None:
  dof_type = lhs.group(1)
  node_id = _dat_int(lhs.group(2))
  if node_id is None:
    code = (
      "unsupported-node-group"
      if _GROUP_NAME_RE.fullmatch(lhs.group(2).strip())
      else "deck-dat-syntax"
    )
    message = (
      f"named node-group reference {lhs.group(0).split('=')[0].strip()!r} is "
      "outside the supported deck subset"
      if code == "unsupported-node-group"
      else f"malformed constraint target {lhs.group(0).split('=')[0].strip()!r}"
    )
    diagnostics.append(_diagnostic(code, message, source))
    return None
  return dof_type, node_id


def _parse_dat_constraint(
  statement: str,
  source: SourceContext,
  diagnostics: list[SpecDiagnostic],
  prescribed: list[_DatPrescribed],
  ties: list[_DatTie],
) -> None:
  lhs = _CONSTRAINT_LHS_RE.match(statement)
  if lhs is None:
    diagnostics.append(
      _diagnostic(
        "deck-dat-syntax",
        f"malformed constraint statement {statement!r} (expected 'u[i] = ...')",
        source,
      ),
    )
    return
  target = _constraint_target(lhs, diagnostics, source)
  if target is None:
    return
  dof_type, node_id = target
  rhs = lhs.group(3).strip()
  if "[" not in rhs:
    value = _dat_float(rhs)
    if value is None:
      diagnostics.append(
        _diagnostic(
          "deck-dat-syntax",
          f"malformed prescribed value {rhs!r}",
          source,
        ),
      )
      return
    prescribed.append(
      _DatPrescribed(node_id=node_id, dof_type=dof_type, value=value, source=source),
    )
    return
  parsed = _parse_tie_rhs(rhs, source, diagnostics)
  if parsed is None:
    return
  offset, factor, master_dof_type, master_node_id = parsed
  ties.append(
    _DatTie(
      slave_node_id=node_id,
      slave_dof_type=dof_type,
      master_node_id=master_node_id,
      master_dof_type=master_dof_type,
      factor=factor,
      offset=offset,
      source=source,
    ),
  )


def _parse_tie_rhs(
  rhs: str,
  source: SourceContext,
  diagnostics: list[SpecDiagnostic],
) -> tuple[float, float, str, int] | None:
  """Parse the one-master affine tie form ``offset + factor * master``."""
  tokens: list[tuple[str, str]] = []
  for match in _TIE_TOKEN_RE.finditer(rhs):
    kind = match.lastgroup
    if kind is None or kind == "space":
      continue
    if kind == "bad":
      diagnostics.append(
        _diagnostic(
          "deck-dat-syntax",
          f"unexpected character {match.group()!r} in constraint right-hand side",
          source,
        ),
      )
      return None
    tokens.append((kind, match.group()))

  terms: list[list[tuple[str, str]]] = [[]]
  for kind, text in tokens:
    if kind == "sign":
      terms.append([(kind, text)])
    else:
      terms[-1].append((kind, text))

  offset = 0.0
  factor = 0.0
  master: tuple[str, int] | None = None

  for term in terms:
    if not term:
      continue
    sign = 1.0
    body = term
    if body[0][0] == "sign":
      sign = -1.0 if body[0][1] == "-" else 1.0
      body = body[1:]
      if not body:
        diagnostics.append(
          _diagnostic(
            "deck-dat-syntax",
            f"dangling sign in constraint right-hand side {rhs!r}",
            source,
          ),
        )
        return None
    kinds = tuple(kind for kind, _text in body)
    master_text: str | None = None
    if kinds == ("number",):
      number = _dat_float(body[0][1])
      if number is None:
        diagnostics.append(
          _diagnostic("deck-dat-syntax", f"non-finite tie term {body[0][1]!r}", source),
        )
        return None
      offset += sign * number
      continue
    if kinds == ("dof",):
      factor = sign
      master_text = body[0][1]
    elif kinds == ("number", "star", "dof"):
      number = _dat_float(body[0][1])
      if number is None:
        diagnostics.append(
          _diagnostic(
            "deck-dat-syntax",
            f"non-finite tie factor {body[0][1]!r}",
            source,
          ),
        )
        return None
      factor = sign * number
      master_text = body[2][1]
    elif kinds == ("dof", "star", "number"):
      number = _dat_float(body[2][1])
      if number is None:
        diagnostics.append(
          _diagnostic(
            "deck-dat-syntax",
            f"non-finite tie factor {body[2][1]!r}",
            source,
          ),
        )
        return None
      factor = sign * number
      master_text = body[0][1]
    else:
      diagnostics.append(
        _diagnostic(
          "unsupported-tie-form",
          f"constraint right-hand side {rhs!r} is outside the one-master "
          "affine form 'offset + factor * master'",
          source,
        ),
      )
      return None

    if master is not None:
      diagnostics.append(
        _diagnostic(
          "unsupported-tie-form",
          f"multi-master constraint {rhs!r} is outside the supported deck subset",
          source,
        ),
      )
      return None
    dof_match = _TIE_DOF_RE.fullmatch(master_text.strip())
    master_node = _dat_int(dof_match.group(2)) if dof_match is not None else None
    if dof_match is None or master_node is None:
      diagnostics.append(
        _diagnostic(
          "unsupported-node-group",
          f"named node-group master {master_text!r} is outside the supported "
          "deck subset",
          source,
        ),
      )
      return None
    master = (dof_match.group(1), master_node)

  if master is None:
    diagnostics.append(
      _diagnostic(
        "unsupported-tie-form",
        f"constraint right-hand side {rhs!r} has no master DOF",
        source,
      ),
    )
    return None
  return offset, factor, master[0], master[1]


def _parse_dat_load(
  statement: str,
  source: SourceContext,
  diagnostics: list[SpecDiagnostic],
  loads: list[_DatLoad],
) -> None:
  lhs = _CONSTRAINT_LHS_RE.match(statement)
  if lhs is None:
    diagnostics.append(
      _diagnostic(
        "deck-dat-syntax",
        f"malformed force statement {statement!r} (expected 'v[i] = value')",
        source,
      ),
    )
    return
  target = _constraint_target(lhs, diagnostics, source)
  if target is None:
    return
  dof_type, node_id = target
  rhs = lhs.group(3).strip()
  value = _dat_float(rhs) if "[" not in rhs else None
  if value is None:
    diagnostics.append(
      _diagnostic(
        "deck-dat-syntax",
        f"external force value {rhs!r} must be a plain number",
        source,
      ),
    )
    return
  loads.append(_DatLoad(node_id=node_id, dof_type=dof_type, value=value, source=source))


# --- semantic conversion -------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _MaterialProfile:
  """One validated element-block family: physics plus law, geometry deferred.

  ``geometries`` lists the mesh ``(rank, cell_arity)`` pairs the family
  accepts (its own unsupported-cell-arity rejections); ``required_rank``
  pins the mesh rank the material law is defined on (``None`` when the
  geometry set already pins it).
  """

  formulation: str
  material_model: str
  parameters: tuple[tuple[str, float, SourceContext], ...]
  registry_kind: str
  geometries: frozenset[tuple[int, int]]
  required_rank: int | None
  source: SourceContext


@dataclass(frozen=True, slots=True)
class _GeometryProfile:
  """The resolved cell geometry of one deck's mesh group."""

  reference_topology: str
  topological_dimension: int
  embedding_dimension: int
  geometry_interpolation: str
  cell_arity: int
  quadrature: str
  field_components: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _SpringProfile:
  """One validated ``Spring`` element block: the legacy grounded axial spring.

  ``stiffness`` is the block's ``k``; the per-element chord directions and
  support pattern resolve later, against the mesh and constraints.
  """

  stiffness: float
  stiffness_source: SourceContext
  source: SourceContext


@dataclass(frozen=True, slots=True)
class _InterfaceProfile:
  """One validated ``Interface`` element block: law plus its parameters.

  ``law`` is one of the four shipped traction-separation laws;
  ``parameters`` carries the law's values in canonical declaration order,
  each with its source. The per-element node quads resolve later, against
  the mesh.
  """

  law: str
  parameters: tuple[tuple[str, float, SourceContext], ...]
  source: SourceContext


@dataclass(frozen=True, slots=True)
class _ElementProfile:
  """One parsed element block: its declared type plus the validated payload.

  Exactly one of ``family`` (a region-routed group), ``spring`` (a
  declaration-routed ``Spring`` group), or ``interface`` (a
  declaration-routed ``Interface`` group) is set on success; all are
  ``None`` when the block's diagnostics already record the failure.
  ``element_type`` keeps the declared type either way, so group
  classification follows the authored intent even for unsound blocks.
  """

  name: str
  element_type: str | None
  family: _MaterialProfile | None
  spring: _SpringProfile | None
  interface: _InterfaceProfile | None
  source: SourceContext


_LINE2_GEOMETRY = _GeometryProfile(
  reference_topology="line",
  topological_dimension=1,
  embedding_dimension=2,
  geometry_interpolation="line2",
  cell_arity=2,
  quadrature="none",
  field_components=("x", "y"),
)
_TRIA3_GEOMETRY = _GeometryProfile(
  reference_topology="triangle",
  topological_dimension=2,
  embedding_dimension=2,
  geometry_interpolation="linear-tria3",
  cell_arity=3,
  quadrature="gauss-tria3-1",
  field_components=("x", "y"),
)
_QUAD4_GEOMETRY = _GeometryProfile(
  reference_topology="quadrilateral",
  topological_dimension=2,
  embedding_dimension=2,
  geometry_interpolation="bilinear-quad4",
  cell_arity=4,
  quadrature="gauss-2x2",
  field_components=("x", "y"),
)
_QUAD8_GEOMETRY = _GeometryProfile(
  reference_topology="quadrilateral",
  topological_dimension=2,
  embedding_dimension=2,
  geometry_interpolation="serendipity-quad8",
  cell_arity=8,
  quadrature="gauss-3x3",
  field_components=("x", "y"),
)
_HEX8_GEOMETRY = _GeometryProfile(
  reference_topology="hexahedron",
  topological_dimension=3,
  embedding_dimension=3,
  geometry_interpolation="trilinear-hex8",
  cell_arity=8,
  quadrature="gauss-2x2x2",
  field_components=("x", "y", "z"),
)
_MESH_GEOMETRIES = {
  (2, 2): _LINE2_GEOMETRY,
  (2, 3): _TRIA3_GEOMETRY,
  (2, 4): _QUAD4_GEOMETRY,
  (2, 8): _QUAD8_GEOMETRY,
  (3, 8): _HEX8_GEOMETRY,
}
_CONTINUUM_GEOMETRY_KEYS = frozenset(key for key in _MESH_GEOMETRIES if key != (2, 2))
_SUPPORTED_MESH_RANKS = (2, 3)


def _block_number(
  block: _ProBlock,
  name: str,
  *,
  extra_code: str,
  allowed: frozenset[str],
  diagnostics: list[SpecDiagnostic],
  ignored: frozenset[str] = frozenset(),
) -> tuple[dict[str, tuple[float, SourceContext]], bool]:
  """Collect numeric assignments of ``allowed`` names from one flat block."""
  values: dict[str, tuple[float, SourceContext]] = {}
  sound = True
  for item in block.items:
    if type(item) is _ProBlock:
      diagnostics.append(
        _diagnostic(
          extra_code,
          f"unsupported nested block {item.name!r} in {block.name!r}",
          item.source,
        ),
      )
      sound = False
      continue
    if item.name in ignored:
      continue
    if item.name not in allowed:
      diagnostics.append(
        _diagnostic(
          extra_code,
          f"unsupported key {item.name!r} in {block.name!r}",
          item.source,
        ),
      )
      sound = False
      continue
    raw = item.value.value
    if item.name == "type":
      if type(raw) is not str:
        diagnostics.append(
          _diagnostic(
            "deck-pro-syntax",
            f"{name} 'type' must be a quoted string",
            item.source,
          ),
        )
        sound = False
      continue
    if type(raw) is bool or type(raw) not in (int, float):
      diagnostics.append(
        _diagnostic(
          "deck-pro-syntax",
          f"{name} {item.name!r} must be a number",
          item.source,
        ),
      )
      sound = False
      continue
    values[item.name] = (float(raw), item.source)
  return values, sound


def _continuum_family(
  block: _ProBlock,
  diagnostics: list[SpecDiagnostic],
) -> _MaterialProfile | None:
  material_block: _ProBlock | None = None
  sound = True
  for item in block.items:
    if type(item) is _ProBlock:
      if item.name == "material" and material_block is None:
        material_block = item
        continue
      diagnostics.append(
        _diagnostic(
          "unsupported-element-parameter",
          f"unsupported nested block {item.name!r} in element block {block.name!r}",
          item.source,
        ),
      )
      sound = False
    elif item.name != "type":
      diagnostics.append(
        _diagnostic(
          "unsupported-element-parameter",
          f"unsupported key {item.name!r} in element block {block.name!r}",
          item.source,
        ),
      )
      sound = False
  if material_block is None:
    diagnostics.append(
      _diagnostic(
        "missing-material-parameter",
        f"element block {block.name!r} has no material block",
        block.source,
      ),
    )
    return None

  material_type: str | None = None
  for item in material_block.items:
    if type(item) is _ProAssignment and item.name == "type":
      raw = item.value.value
      material_type = raw if type(raw) is str else None
  if material_type == _PLASTICITY_MODEL:
    return _plasticity_family(block, material_block, diagnostics)
  elastic = _ELASTIC_MATERIAL_MODELS.get(material_type or "")
  if elastic is None:
    diagnostics.append(
      _diagnostic(
        "unsupported-material-model",
        f"material model {material_type!r} is outside the supported deck "
        "subset ('PlaneStress', 'PlaneStrain', 'Isotropic', and "
        "'IsotropicHardeningPlasticity' only)",
        material_block.source,
      ),
    )
    sound = False
  parameters, params_sound = _block_number(
    material_block,
    "material",
    extra_code="unsupported-material-parameter",
    allowed=frozenset({"type", "E", "nu"}),
    diagnostics=diagnostics,
  )
  sound = sound and params_sound
  missing = {"E", "nu"} - set(parameters)
  if missing:
    diagnostics.append(
      _diagnostic(
        "missing-material-parameter",
        f"material block in {block.name!r} misses {sorted(missing)}",
        material_block.source,
      ),
    )
    sound = False
  if not sound or elastic is None:
    return None
  material_model, required_rank = elastic
  youngs = parameters["E"]
  poisson = parameters["nu"]
  return _MaterialProfile(
    formulation="small-strain-continuum",
    material_model=material_model,
    parameters=(
      ("youngs_modulus", youngs[0], youngs[1]),
      ("poisson_ratio", poisson[0], poisson[1]),
    ),
    registry_kind="continuum",
    geometries=_CONTINUUM_GEOMETRY_KEYS,
    required_rank=required_rank,
    source=material_block.source,
  )


def _plasticity_family(
  block: _ProBlock,
  material_block: _ProBlock,
  diagnostics: list[SpecDiagnostic],
) -> _MaterialProfile | None:
  """Read the linear-hardening plasticity form into the v2 stateful descriptor.

  Only the ``hard`` configuration converts: the legacy law's plastic branch
  reads its ``hard`` property for the tangent, so the table and power-law
  hardening forms never defined consistent plastic behavior (the M25 scoping
  decision); carrying their properties rejects with a coded diagnostic.
  """
  sound = True
  table_keys = sorted(
    {
      item.name
      for item in material_block.items
      if type(item) is _ProAssignment and item.name in _PLASTICITY_TABLE_KEYS
    }
  )
  if table_keys:
    diagnostics.append(
      _diagnostic(
        "unsupported-hardening-form",
        f"material model 'IsotropicHardeningPlasticity' with hardening "
        f"properties {table_keys} is outside the supported deck subset: the "
        "v3 stateful law ships exactly the linear-hardening configuration "
        "(E, nu, syield, hard) — the M25 scoping decision, since the legacy "
        "law's plastic branch reads its 'hard' property for the tangent and "
        "the table/power-law forms never defined consistent plastic behavior",
        material_block.source,
      ),
    )
    sound = False
  parameters, params_sound = _block_number(
    material_block,
    "material",
    extra_code="unsupported-material-parameter",
    allowed=frozenset({"type", *_PLASTICITY_VALUE_KEYS}),
    diagnostics=diagnostics,
    ignored=_PLASTICITY_TABLE_KEYS,
  )
  sound = sound and params_sound
  missing = _PLASTICITY_VALUE_KEYS - set(parameters)
  if missing:
    diagnostics.append(
      _diagnostic(
        "missing-material-parameter",
        f"material block in {block.name!r} misses {sorted(missing)}",
        material_block.source,
      ),
    )
    sound = False
  if not sound:
    return None
  youngs = parameters["E"]
  poisson = parameters["nu"]
  syield = parameters["syield"]
  hard = parameters["hard"]
  return _MaterialProfile(
    formulation="small-strain-continuum",
    material_model="isotropic-hardening-plasticity",
    parameters=(
      ("youngs_modulus", youngs[0], youngs[1]),
      ("poisson_ratio", poisson[0], poisson[1]),
      ("initial_yield_stress", syield[0], syield[1]),
      ("hardening_slope", hard[0], hard[1]),
    ),
    registry_kind="plasticity",
    geometries=frozenset({(2, 8)}),
    required_rank=2,
    source=material_block.source,
  )


def _finite_strain_family(
  block: _ProBlock,
  diagnostics: list[SpecDiagnostic],
) -> _MaterialProfile | None:
  """Read a FiniteStrainContinuum block: the total-Lagrangian Q8 slice.

  The legacy element pairs with any ``getStress`` law, but the only
  configuration the supported deck subset ships is the stateless
  Saint-Venant-Kirchhoff form: a ``PlaneStress`` material block (``E``,
  ``nu``), whose constant matrix the total-Lagrangian formulation contracts
  with the Green-Lagrange strain into the second Piola-Kirchhoff stress.
  Every other material model rejects with a coded diagnostic.
  """
  material_block: _ProBlock | None = None
  sound = True
  for item in block.items:
    if type(item) is _ProBlock:
      if item.name == "material" and material_block is None:
        material_block = item
        continue
      diagnostics.append(
        _diagnostic(
          "unsupported-element-parameter",
          f"unsupported nested block {item.name!r} in element block {block.name!r}",
          item.source,
        ),
      )
      sound = False
    elif item.name != "type":
      diagnostics.append(
        _diagnostic(
          "unsupported-element-parameter",
          f"unsupported key {item.name!r} in element block {block.name!r}",
          item.source,
        ),
      )
      sound = False
  if material_block is None:
    diagnostics.append(
      _diagnostic(
        "missing-material-parameter",
        f"element block {block.name!r} has no material block",
        block.source,
      ),
    )
    return None

  material_type: str | None = None
  for item in material_block.items:
    if type(item) is _ProAssignment and item.name == "type":
      raw = item.value.value
      material_type = raw if type(raw) is str else None
  if material_type != "PlaneStress":
    diagnostics.append(
      _diagnostic(
        "unsupported-material-model",
        f"material model {material_type!r} on a FiniteStrainContinuum element "
        "is outside the supported deck subset ('PlaneStress' only)",
        material_block.source,
      ),
    )
    return None
  parameters, params_sound = _block_number(
    material_block,
    "material",
    extra_code="unsupported-material-parameter",
    allowed=frozenset({"type", "E", "nu"}),
    diagnostics=diagnostics,
  )
  sound = sound and params_sound
  missing = {"E", "nu"} - set(parameters)
  if missing:
    diagnostics.append(
      _diagnostic(
        "missing-material-parameter",
        f"material block in {block.name!r} misses {sorted(missing)}",
        material_block.source,
      ),
    )
    sound = False
  if not sound:
    return None
  youngs = parameters["E"]
  poisson = parameters["nu"]
  return _MaterialProfile(
    formulation="total-lagrangian-continuum",
    material_model="plane-stress-saint-venant-kirchhoff",
    parameters=(
      ("youngs_modulus", youngs[0], youngs[1]),
      ("poisson_ratio", poisson[0], poisson[1]),
    ),
    registry_kind="finite-strain",
    geometries=frozenset({(2, 8)}),
    required_rank=2,
    source=material_block.source,
  )


def _truss_family(
  block: _ProBlock,
  diagnostics: list[SpecDiagnostic],
) -> _MaterialProfile | None:
  parameters, sound = _block_number(
    block,
    "element block",
    extra_code="unsupported-element-parameter",
    allowed=frozenset({"type", "E", "Area"}),
    diagnostics=diagnostics,
  )
  missing = {"E", "Area"} - set(parameters)
  if missing:
    diagnostics.append(
      _diagnostic(
        "missing-element-parameter",
        f"Truss block {block.name!r} misses {sorted(missing)}",
        block.source,
      ),
    )
    sound = False
  if not sound:
    return None
  youngs = parameters["E"]
  area = parameters["Area"]
  return _MaterialProfile(
    formulation="total-lagrangian-truss",
    material_model="uniaxial-linear-elastic",
    parameters=(
      ("youngs_modulus", youngs[0], youngs[1]),
      ("area", area[0], area[1]),
    ),
    registry_kind="truss",
    geometries=frozenset({(2, 2)}),
    required_rank=None,
    source=block.source,
  )


def _spring_profile(
  block: _ProBlock,
  diagnostics: list[SpecDiagnostic],
) -> _SpringProfile | None:
  """Read a ``Spring`` block: the legacy grounded axial spring's ``k``."""
  parameters, sound = _block_number(
    block,
    "element block",
    extra_code="unsupported-element-parameter",
    allowed=frozenset({"type", "k"}),
    diagnostics=diagnostics,
  )
  if "k" not in parameters:
    diagnostics.append(
      _diagnostic(
        "missing-element-parameter",
        f"Spring block {block.name!r} misses ['k']",
        block.source,
      ),
    )
    sound = False
  if not sound:
    return None
  stiffness, stiffness_source = parameters["k"]
  return _SpringProfile(
    stiffness=stiffness,
    stiffness_source=stiffness_source,
    source=block.source,
  )


def _interface_family(
  block: _ProBlock,
  diagnostics: list[SpecDiagnostic],
) -> _InterfaceProfile | None:
  """Read an ``Interface`` block: one of the four shipped cohesive laws.

  The legacy ``XuNeedleman`` constructor overwrites deck-supplied ``q``/``r``
  with hardcoded values AFTER reading the properties, so carrying those keys
  converts dishonestly — they reject here as unsupported parameters.
  """
  material_block: _ProBlock | None = None
  sound = True
  for item in block.items:
    if type(item) is _ProBlock:
      if item.name == "material" and material_block is None:
        material_block = item
        continue
      diagnostics.append(
        _diagnostic(
          "unsupported-element-parameter",
          f"unsupported nested block {item.name!r} in element block {block.name!r}",
          item.source,
        ),
      )
      sound = False
    elif item.name != "type":
      diagnostics.append(
        _diagnostic(
          "unsupported-element-parameter",
          f"unsupported key {item.name!r} in element block {block.name!r}",
          item.source,
        ),
      )
      sound = False
  if material_block is None:
    diagnostics.append(
      _diagnostic(
        "missing-material-parameter",
        f"element block {block.name!r} has no material block",
        block.source,
      ),
    )
    return None

  material_type: str | None = None
  for item in material_block.items:
    if type(item) is _ProAssignment and item.name == "type":
      raw = item.value.value
      material_type = raw if type(raw) is str else None
  law_parameters = _INTERFACE_LAW_PARAMETERS.get(material_type or "")
  if law_parameters is None:
    diagnostics.append(
      _diagnostic(
        "unsupported-material-model",
        f"material model {material_type!r} on an Interface element is outside "
        "the supported deck subset ('XuNeedleman', 'PowerLawModeI', "
        "'ThoulessModeI', and 'Dummy' only)",
        material_block.source,
      ),
    )
    sound = False
  parameters, params_sound = _block_number(
    material_block,
    "material",
    extra_code="unsupported-material-parameter",
    allowed=frozenset(("type", *(law_parameters or ()))),
    diagnostics=diagnostics,
  )
  sound = sound and params_sound
  if law_parameters is None:
    return None
  missing = set(law_parameters) - set(parameters)
  if missing:
    diagnostics.append(
      _diagnostic(
        "missing-material-parameter",
        f"material block in {block.name!r} misses {sorted(missing)}",
        material_block.source,
      ),
    )
    sound = False
  if not sound:
    return None
  ordered: list[tuple[str, float, SourceContext]] = []
  for name in law_parameters:
    value, value_source = parameters[name]
    if value <= 0.0:
      diagnostics.append(
        _diagnostic(
          "invalid-material-parameter-value",
          f"interface law {material_type!r} parameter {name!r} must be positive",
          value_source,
        ),
      )
      sound = False
    ordered.append((name, value, value_source))
  if material_type == "ThoulessModeI" and sound:
    d1d3 = parameters["d1d3"][0]
    d2d3 = parameters["d2d3"][0]
    if not d1d3 < d2d3 < 1.0:
      diagnostics.append(
        _diagnostic(
          "invalid-material-parameter-value",
          "interface law 'ThoulessModeI' requires 0 < d1d3 < d2d3 < 1 for "
          "the rise/plateau/softening branches",
          parameters["d1d3"][1],
        ),
      )
      sound = False
  if not sound:
    return None
  return _InterfaceProfile(
    law=material_type,
    parameters=tuple(ordered),
    source=material_block.source,
  )


def _interface_declaration(
  group: str,
  profile: _InterfaceProfile,
  deck: _DatDeck,
) -> InterfaceDeclaration:
  """Build one interface group's declaration from its validated profile."""
  interface_ids: list[int] = []
  node_quads: list[tuple[int, int, int, int]] = []
  for element in deck.elements:
    if element.group != group:
      continue
    interface_ids.append(element.id)
    node_quads.append(tuple(element.node_ids))
  values = {name: value for name, value, _ in profile.parameters}
  if profile.law == "Dummy":
    return dummy_interface_declaration(
      block_id=group,
      space_id=_FIELD_ID,
      interface_ids=tuple(interface_ids),
      node_quads=tuple(node_quads),
      stiffness=values["D"],
      state_schema=_INTERFACE_STATE_SCHEMA,
      source=profile.source,
    )
  if profile.law == "ThoulessModeI":
    return thouless_mode_i_declaration(
      block_id=group,
      space_id=_FIELD_ID,
      interface_ids=tuple(interface_ids),
      node_quads=tuple(node_quads),
      fracture_energy=values["Gc"],
      ultimate_traction=values["Tult"],
      d1d3=values["d1d3"],
      d2d3=values["d2d3"],
      state_schema=_INTERFACE_STATE_SCHEMA,
      source=profile.source,
    )
  if profile.law == "PowerLawModeI":
    return power_law_mode_i_declaration(
      block_id=group,
      space_id=_FIELD_ID,
      interface_ids=tuple(interface_ids),
      node_quads=tuple(node_quads),
      fracture_energy=values["Gc"],
      ultimate_traction=values["Tult"],
      state_schema=_INTERFACE_STATE_SCHEMA,
      source=profile.source,
    )
  return xu_needleman_declaration(
    block_id=group,
    space_id=_FIELD_ID,
    interface_ids=tuple(interface_ids),
    node_quads=tuple(node_quads),
    fracture_energy=values["Gc"],
    ultimate_traction=values["Tult"],
    state_schema=_INTERFACE_STATE_SCHEMA,
    source=profile.source,
  )


def _check_interface_arities(
  group: str,
  rank: int,
  elements: tuple[_DatElement, ...],
  diagnostics: list[SpecDiagnostic],
) -> None:
  """Require four-node (bottom pair, top pair) interface cells on a 2D mesh."""
  for element in elements:
    if element.group != group:
      continue
    if (rank, len(element.node_ids)) not in _INTERFACE_GEOMETRIES:
      diagnostics.append(
        _diagnostic(
          "unsupported-cell-arity",
          f"element {element.id} has {len(element.node_ids)} nodes on a "
          f"{rank}D mesh; interface elements require four nodes (a bottom "
          "pair and a top pair) on a 2D mesh",
          element.source,
        ),
      )


def _element_profile(
  block: _ProBlock,
  diagnostics: list[SpecDiagnostic],
) -> _ElementProfile:
  element_type: str | None = None
  for item in block.items:
    if type(item) is _ProAssignment and item.name == "type":
      raw = item.value.value
      element_type = raw if type(raw) is str else None
  family: _MaterialProfile | None = None
  spring: _SpringProfile | None = None
  interface: _InterfaceProfile | None = None
  if element_type == "SmallStrainContinuum":
    family = _continuum_family(block, diagnostics)
  elif element_type == "FiniteStrainContinuum":
    family = _finite_strain_family(block, diagnostics)
  elif element_type == "Truss":
    family = _truss_family(block, diagnostics)
  elif element_type == "Spring":
    spring = _spring_profile(block, diagnostics)
  elif element_type == "Interface":
    interface = _interface_family(block, diagnostics)
  else:
    diagnostics.append(
      _diagnostic(
        "unsupported-element-type",
        f"element type {element_type!r} is outside the supported deck subset "
        "('SmallStrainContinuum', 'FiniteStrainContinuum', 'Truss', "
        "'Spring', and 'Interface' only)",
        block.source,
      ),
    )
  return _ElementProfile(
    name=block.name,
    element_type=element_type,
    family=family,
    spring=spring,
    interface=interface,
    source=block.source,
  )


def _solver_settings(
  block: _ProBlock,
  diagnostics: list[SpecDiagnostic],
  not_converted: list[SpecDiagnostic],
) -> DeckSolverSettings | None:
  solver_type: str | None = None
  assignments: dict[str, _ProValue] = {}
  for item in block.items:
    if type(item) is _ProBlock:
      diagnostics.append(
        _diagnostic(
          "unsupported-solver-parameter",
          f"unsupported nested block {item.name!r} in the solver block",
          item.source,
        ),
      )
      continue
    assignments[item.name] = item.value
    if item.name == "type":
      raw = item.value.value
      solver_type = raw if type(raw) is str else None

  if solver_type not in ("LinearSolver", "NonlinearSolver", "RiksSolver"):
    diagnostics.append(
      _diagnostic(
        "unsupported-solver-type",
        f"solver type {solver_type!r} is outside the supported deck subset "
        "('LinearSolver', 'NonlinearSolver', and 'RiksSolver' only)",
        block.source,
      ),
    )
    return None

  allowed = frozenset({"type"})
  if solver_type == "NonlinearSolver":
    allowed = _NONLINEAR_SOLVER_KEYS
  elif solver_type == "RiksSolver":
    allowed = _RIKS_SOLVER_KEYS
  sound = True
  for name, value in assignments.items():
    if name not in allowed:
      diagnostics.append(
        _diagnostic(
          "unsupported-solver-parameter",
          f"unsupported {solver_type} key {name!r}",
          value.source,
        ),
      )
      sound = False
  if solver_type == "LinearSolver":
    if not sound:
      return None
    return DeckSolverSettings(
      solver_type="LinearSolver",
      load_factors=(1.0,),
      tolerance=1.0e-10,
      max_iterations=25,
    )

  if solver_type == "RiksSolver":
    return _riks_solver_settings(assignments, sound, diagnostics)

  tol = _solver_number(assignments, "tol", 1.0e-3, diagnostics)
  iter_max = _solver_number(assignments, "iterMax", 10, diagnostics, integer=True)
  max_cycle = _solver_number(assignments, "maxCycle", 5, diagnostics, integer=True)
  dtime = _solver_number(assignments, "dtime", 1.0, diagnostics)
  if None in (tol, iter_max, max_cycle, dtime):
    sound = False

  load_func = assignments.get("loadFunc")
  if load_func is not None:
    raw = load_func.value
    if raw != "t":
      diagnostics.append(
        _diagnostic(
          "unsupported-load-func",
          f"loadFunc {raw!r} is outside the supported deck subset "
          "(identity ramp 't' only)",
          load_func.source,
        ),
      )
      sound = False

  fixed_step = assignments.get("fixedStep")
  if fixed_step is not None:
    if type(fixed_step.value) is not bool:
      diagnostics.append(
        _diagnostic(
          "deck-pro-syntax",
          "fixedStep must be a boolean",
          fixed_step.source,
        ),
      )
      sound = False
    else:
      not_converted.append(
        _diagnostic(
          "not-converted-solver-option",
          "fixedStep is a legacy step-size policy; the v3 driver owns its own "
          "cutback schedule, so the flag is intentionally not converted",
          fixed_step.source,
        ),
      )

  load_table = assignments.get("loadTable")
  factors: tuple[float, ...] | None = None
  if load_table is not None:
    factors = _solver_load_table(load_table, diagnostics)
    if factors is None:
      sound = False
    elif "maxCycle" in assignments or "dtime" in assignments:
      not_converted.append(
        _diagnostic(
          "not-converted-solver-option",
          "loadTable overrides the maxCycle/dtime ramp",
          load_table.source,
        ),
      )
  if not sound:
    return None
  if factors is None:
    factors = tuple(float(dtime) * float(k) for k in range(1, int(max_cycle) + 1))
  return DeckSolverSettings(
    solver_type="NonlinearSolver",
    load_factors=factors,
    tolerance=float(tol),
    max_iterations=int(iter_max),
  )


def _riks_solver_settings(
  assignments: dict[str, _ProValue],
  sound: bool,
  diagnostics: list[SpecDiagnostic],
) -> DeckSolverSettings | None:
  """Translate a ``RiksSolver`` block into the arc-length policy.

  Every parsed key converts (the legacy defaults stand in when unset);
  ``load_factors`` stays empty because the continuation owns the load
  parameter autonomously, and the cycle cap is the legacy hard-coded 1000.
  """
  tol = _solver_number(assignments, "tol", _RIKS_DEFAULT_TOLERANCE, diagnostics)
  iter_max = _solver_number(
    assignments, "iterMax", _RIKS_DEFAULT_MAX_ITERATIONS, diagnostics, integer=True
  )
  opt_iter = _solver_number(
    assignments,
    "optiter",
    _RIKS_DEFAULT_OPTIMAL_ITERATIONS,
    diagnostics,
    integer=True,
  )
  max_lam = _solver_number(assignments, "maxLam", _RIKS_DEFAULT_MAX_LAM, diagnostics)
  max_factor = _solver_number(
    assignments, "maxFactor", _RIKS_DEFAULT_MAX_FACTOR, diagnostics
  )
  if None in (tol, iter_max, opt_iter, max_lam, max_factor):
    sound = False

  fixed_step_value = assignments.get("fixedStep")
  fixed_step = False
  if fixed_step_value is not None:
    if type(fixed_step_value.value) is not bool:
      diagnostics.append(
        _diagnostic(
          "deck-pro-syntax",
          "fixedStep must be a boolean",
          fixed_step_value.source,
        ),
      )
      sound = False
    else:
      fixed_step = fixed_step_value.value

  if not sound:
    return None
  return DeckSolverSettings(
    solver_type="RiksSolver",
    load_factors=(),
    tolerance=float(tol),
    max_iterations=int(iter_max),
    optimal_iterations=int(opt_iter),
    fixed_step=fixed_step,
    max_lam=float(max_lam),
    max_factor=float(max_factor),
    cycle_cap=_RIKS_CYCLE_CAP,
  )


def _solver_number(
  assignments: dict[str, _ProValue],
  name: str,
  default: float,
  diagnostics: list[SpecDiagnostic],
  *,
  integer: bool = False,
) -> float | None:
  value = assignments.get(name)
  if value is None:
    return float(default)
  raw = value.value
  if type(raw) is bool or type(raw) not in (int, float):
    diagnostics.append(
      _diagnostic(
        "deck-pro-syntax",
        f"solver key {name!r} must be {'an integer' if integer else 'a number'}",
        value.source,
      ),
    )
    return None
  if integer and type(raw) is not int:
    diagnostics.append(
      _diagnostic(
        "deck-pro-syntax",
        f"solver key {name!r} must be an integer",
        value.source,
      ),
    )
    return None
  return float(raw)


def _solver_load_table(
  value: _ProValue,
  diagnostics: list[SpecDiagnostic],
) -> tuple[float, ...] | None:
  raw = value.value
  if type(raw) is not tuple or not raw:
    diagnostics.append(
      _diagnostic(
        "unsupported-load-table",
        "loadTable must be a non-empty list of numbers",
        value.source,
      ),
    )
    return None
  factors: list[float] = []
  for entry in raw:
    item = entry.value
    if type(item) is bool or type(item) not in (int, float):
      diagnostics.append(
        _diagnostic(
          "unsupported-load-table",
          "loadTable entries must be numbers",
          entry.source,
        ),
      )
      return None
    factors.append(float(item))
  return tuple(factors)


def _check_mesh(
  deck: _DatDeck,
  file_source: SourceContext,
  diagnostics: list[SpecDiagnostic],
) -> int | None:
  """Validate mesh structure and return the mesh rank (coordinate count)."""
  if not deck.nodes:
    diagnostics.append(
      _diagnostic("empty-mesh", "the mesh declares no nodes", file_source)
    )
  if not deck.elements:
    diagnostics.append(
      _diagnostic("empty-mesh", "the mesh declares no elements", file_source)
    )
  seen_nodes: set[int] = set()
  arity: int | None = None
  for node in deck.nodes:
    if node.id in seen_nodes:
      diagnostics.append(
        _diagnostic(
          "duplicate-node-id",
          f"node id {node.id} is declared twice",
          node.source,
        ),
      )
    seen_nodes.add(node.id)
    node_arity = len(node.coordinates)
    if arity is None:
      arity = node_arity
    elif node_arity != arity:
      diagnostics.append(
        _diagnostic(
          "inconsistent-node-coordinates",
          f"node {node.id} has {node_arity} coordinates; expected {arity}",
          node.source,
        ),
      )
  if arity is not None and arity not in _SUPPORTED_MESH_RANKS:
    diagnostics.append(
      _diagnostic(
        "unsupported-mesh-rank",
        f"node coordinates are {arity}D; the supported deck subset is 2D or 3D",
        deck.nodes[0].source,
      ),
    )
  seen_cells: set[int] = set()
  for element in deck.elements:
    if element.id in seen_cells:
      diagnostics.append(
        _diagnostic(
          "duplicate-cell-id",
          f"element id {element.id} is declared twice",
          element.source,
        ),
      )
    seen_cells.add(element.id)
    for node_id in element.node_ids:
      if node_id not in seen_nodes:
        diagnostics.append(
          _diagnostic(
            "unknown-node-reference",
            f"element {element.id} references undeclared node {node_id}",
            element.source,
          ),
        )
  return arity


def _check_dofs(
  deck: _DatDeck,
  rank: int | None,
  diagnostics: list[SpecDiagnostic],
) -> None:
  node_ids = {node.id for node in deck.nodes}
  for item in deck.prescribed:
    _check_dof_target(
      item.node_id, item.dof_type, item.source, node_ids, rank, diagnostics
    )
  for item in deck.loads:
    _check_dof_target(
      item.node_id, item.dof_type, item.source, node_ids, rank, diagnostics
    )
  for tie in deck.ties:
    _check_dof_target(
      tie.slave_node_id,
      tie.slave_dof_type,
      tie.source,
      node_ids,
      rank,
      diagnostics,
    )
    _check_dof_target(
      tie.master_node_id,
      tie.master_dof_type,
      tie.source,
      node_ids,
      rank,
      diagnostics,
    )


def _check_dof_target(
  node_id: int,
  dof_type: str,
  source: SourceContext,
  node_ids: set[int],
  rank: int | None,
  diagnostics: list[SpecDiagnostic],
) -> None:
  if dof_type not in _DOF_COMPONENTS:
    diagnostics.append(
      _diagnostic(
        "unsupported-dof-type",
        f"DOF type {dof_type!r} is outside the supported deck subset "
        f"({sorted(_DOF_COMPONENTS)} only)",
        source,
      ),
    )
  elif dof_type == "w" and rank != 3:
    diagnostics.append(
      _diagnostic(
        "unsupported-dof-type",
        "DOF type 'w' requires a 3D mesh",
        source,
      ),
    )
  if node_id not in node_ids:
    diagnostics.append(
      _diagnostic(
        "unknown-node-reference",
        f"constraint or load references undeclared node {node_id}",
        source,
      ),
    )


def _affine_value(
  value: float,
  *,
  ramped: bool,
  source: SourceContext,
) -> AffineValueSpec:
  if not ramped:
    return AffineValueSpec(constant=value, source=source)
  return AffineValueSpec(
    coefficients=(AffineCoefficientSpec(_LOAD_COORDINATE, value, source),),
    source=source,
  )


def _emit_program(
  deck: _DatDeck,
  *,
  ramped: bool,
  solver_source: SourceContext,
) -> ProgramSpec:
  constraints: list[ProgramConstraintSpec] = []
  for item in deck.prescribed:
    constraints.append(
      PrescribedDofSpec(
        id=f"prescribed:{item.node_id}:{item.dof_type}",
        target=DofRef(
          node_id=item.node_id,
          field_id=_FIELD_ID,
          component=_DOF_COMPONENTS[item.dof_type],
        ),
        value=_affine_value(item.value, ramped=ramped, source=item.source),
        source=item.source,
      ),
    )
  for tie in deck.ties:
    constraints.append(
      AffineTieSpec(
        id=f"tie:{tie.slave_node_id}:{tie.slave_dof_type}",
        slave=DofRef(
          node_id=tie.slave_node_id,
          field_id=_FIELD_ID,
          component=_DOF_COMPONENTS[tie.slave_dof_type],
        ),
        master=DofRef(
          node_id=tie.master_node_id,
          field_id=_FIELD_ID,
          component=_DOF_COMPONENTS[tie.master_dof_type],
        ),
        factor=tie.factor,
        offset=_affine_value(tie.offset, ramped=ramped, source=tie.source),
        source=tie.source,
      ),
    )
  loads = tuple(
    NodalLoadSpec(
      id=f"load:{item.node_id}:{item.dof_type}",
      target=DofRef(
        node_id=item.node_id,
        field_id=_FIELD_ID,
        component=_DOF_COMPONENTS[item.dof_type],
      ),
      value=_affine_value(item.value, ramped=True, source=item.source),
      source=item.source,
    )
    for item in deck.loads
  )
  return ProgramSpec(
    coordinates=(
      ProgramCoordinateSpec(
        name=_LOAD_COORDINATE,
        kind="load",
        source=solver_source,
      ),
    ),
    constraints=tuple(constraints),
    loads=loads,
    source=solver_source,
  )


def _resolve_geometry(
  group: str,
  rank: int,
  geometries: frozenset[tuple[int, int]],
  elements: tuple[_DatElement, ...],
  diagnostics: list[SpecDiagnostic],
) -> _GeometryProfile | None:
  """Resolve the single cell geometry of one mesh group.

  Every element's ``(rank, arity)`` must name one of the geometry keys the
  deck's group accepts, and all elements of the group must resolve to the
  same geometry. Violations record coded ``unsupported-cell-arity``
  diagnostics.
  """
  geometry: _GeometryProfile | None = None
  geometry_key: tuple[int, int] | None = None
  supported = sorted(
    {_MESH_GEOMETRIES[key].geometry_interpolation for key in geometries}
  )
  for element in elements:
    if element.group != group:
      continue
    arity = len(element.node_ids)
    key = (rank, arity)
    if key not in geometries:
      diagnostics.append(
        _diagnostic(
          "unsupported-cell-arity",
          f"element {element.id} has {arity} nodes on a {rank}D mesh; the "
          f"supported cells for this deck are {supported}",
          element.source,
        ),
      )
      continue
    if geometry_key is None:
      geometry_key = key
      geometry = _MESH_GEOMETRIES[key]
    elif key != geometry_key:
      diagnostics.append(
        _diagnostic(
          "unsupported-cell-arity",
          f"element {element.id} has {arity} nodes; the deck's geometry for "
          f"{group!r} is {geometry.geometry_interpolation} "
          f"({geometry.cell_arity} nodes)",
          element.source,
        ),
      )
  return geometry


def _legacy_spring_kernel(
  displacements: np.ndarray,
  accepted_rows: np.ndarray,
  parameters: np.ndarray,
) -> SpringKernelResult:
  """The legacy ``Spring`` residual as a network of grounded point springs.

  ``parameters`` is ``(k, d0_x, d0_y, d1_x, d1_y, ...)``: the group's shared
  stiffness plus each spring's unit chord direction. The response is the
  consistent axial pair ``k (u.d) d`` / ``k d d^T`` the H1 decision pins —
  the legacy element's residual restricted to a grounded end (the legacy
  isotropic tangent is not the oracle). The spring is stateless, so the
  trial rows echo the accepted rows.
  """
  stiffness = parameters[0]
  directions = parameters[1:].reshape(-1, 2)
  axial = (displacements * directions).sum(axis=1)
  force = stiffness * axial[:, None] * directions
  tangent = stiffness * directions[:, :, None] * directions[:, None, :]
  return SpringKernelResult(
    force=force,
    tangent=tangent,
    trial_rows=np.array(accepted_rows, copy=True),
    status=EvaluationStatus.OK,
  )


def _spring_declaration(
  group: str,
  profile: _SpringProfile,
  deck: _DatDeck,
  diagnostics: list[SpecDiagnostic],
) -> SpringDeclaration | None:
  """Build one spring group's declaration, validating the support pattern.

  Each element must have exactly one grounded end — every displacement
  component prescribed to zero — so the two-node legacy spring reduces
  exactly to a grounded point spring at the other node along the element
  chord (the H1 consistent axial form). Every other support pattern records
  a coded ``unsupported-spring-support`` diagnostic.
  """
  prescribed: dict[int, dict[str, float]] = {}
  for item in deck.prescribed:
    prescribed.setdefault(item.node_id, {})[item.dof_type] = item.value

  def grounded(node_id: int) -> bool:
    values = prescribed.get(node_id)
    if values is None:
      return False
    return all(values.get(dof_type) == 0.0 for dof_type in ("u", "v"))

  coordinates = {node.id: node.coordinates for node in deck.nodes}
  spring_ids: list[int] = []
  node_ids: list[int] = []
  directions: list[float] = []
  sound = True
  for element in deck.elements:
    if element.group != group:
      continue
    first, second = element.node_ids
    first_grounded = grounded(first)
    second_grounded = grounded(second)
    if first_grounded == second_grounded:
      diagnostics.append(
        _diagnostic(
          "unsupported-spring-support",
          f"Spring element {element.id} must have exactly one end with every "
          "component prescribed to zero (the grounded point-spring form); "
          f"node {first} is {'grounded' if first_grounded else 'not grounded'} "
          f"and node {second} is "
          f"{'grounded' if second_grounded else 'not grounded'}",
          element.source,
        ),
      )
      sound = False
      continue
    chord_x = coordinates[second][0] - coordinates[first][0]
    chord_y = coordinates[second][1] - coordinates[first][1]
    length = math.hypot(chord_x, chord_y)
    if length == 0.0:
      diagnostics.append(
        _diagnostic(
          "unsupported-spring-support",
          f"Spring element {element.id} has coincident endpoints; the chord "
          "direction is undefined",
          element.source,
        ),
      )
      sound = False
      continue
    spring_ids.append(element.id)
    node_ids.append(second if first_grounded else first)
    directions.extend((chord_x / length, chord_y / length))
  if not sound:
    return None
  return SpringDeclaration(
    block_id=group,
    space_id=_FIELD_ID,
    spring_ids=tuple(spring_ids),
    node_ids=tuple(node_ids),
    state_schema=_SPRING_STATE_SCHEMA,
    state_slots=(),
    kernel_name=_SPRING_KERNEL_NAME,
    kernel_version="1",
    implementation_id=_SPRING_IMPLEMENTATION_ID,
    parameters=(profile.stiffness, *directions),
    kernel=_legacy_spring_kernel,
    source=profile.source,
  )


def _emit_model(
  deck: _DatDeck,
  group: str,
  profile: _ElementProfile,
  geometry: _GeometryProfile,
  pro_source: SourceContext,
) -> ModelSpec:
  family = profile.family
  if family is None:
    msg = "emission requires a validated element family"
    raise ValueError(msg)
  nodes = tuple(
    NodeSpec(id=node.id, coordinates=node.coordinates, source=node.source)
    for node in deck.nodes
  )
  cells = tuple(
    CellSpec(
      id=element.id,
      node_ids=element.node_ids,
      source=element.source,
    )
    for element in deck.elements
    if element.group == group
  )
  block = CellBlockSpec(
    id=group,
    reference_topology=geometry.reference_topology,
    topological_dimension=geometry.topological_dimension,
    embedding_dimension=geometry.embedding_dimension,
    geometry_interpolation=geometry.geometry_interpolation,
    cells=cells,
    source=profile.source,
  )
  field = FieldSpec(
    id=_FIELD_ID,
    components=geometry.field_components,
    location="node",
    source=profile.source,
  )
  material = MaterialSpec(
    id=f"{group}:material",
    model=family.material_model,
    parameters=tuple(
      MaterialParameterSpec(name, value, source)
      for name, value, source in family.parameters
    ),
    source=profile.source,
  )
  region = RegionSpec(
    id="domain",
    cell_refs=tuple(CellRef(block.id, cell.id) for cell in cells),
    field_ids=(field.id,),
    material_id=material.id,
    formulation=family.formulation,
    quadrature=geometry.quadrature,
    source=profile.source,
  )
  return ModelSpec(
    mesh=MeshSpec(nodes=nodes, cell_blocks=(block,), source=pro_source),
    fields=(field,),
    materials=(material,),
    regions=(region,),
    source=pro_source,
  )


# --- public API ----------------------------------------------------------------


def read_legacy_deck(path: Path | str) -> ConvertedDeck:
  """Convert one legacy ``.pro`` deck (and its referenced ``.dat``) to specs.

  The emitted :class:`ConvertedDeck` carries an authored ``ModelSpec`` and
  ``ProgramSpec`` plus the landed reference registry for the deck's operator
  family. Every out-of-subset construct raises :class:`DeckConversionError`
  with all coded diagnostics in detection order.
  """
  pro_path = Path(path)
  try:
    text = pro_path.read_text(encoding="utf-8")
  except (OSError, UnicodeError):
    _raise(
      "deck-input-missing",
      f"legacy deck file is not readable: {pro_path}",
      SourceContext(source=str(pro_path)),
    )

  tokens = _tokenize_pro(text, pro_path)
  parser = _ProParser(tokens, pro_path)
  items = parser.parse()

  diagnostics: list[SpecDiagnostic] = list(parser.diagnostics)
  not_converted: list[SpecDiagnostic] = []
  pro_deck = _walk_pro(items, diagnostics, not_converted)

  settings: DeckSolverSettings | None = None
  if pro_deck.solver_block is not None:
    settings = _solver_settings(pro_deck.solver_block, diagnostics, not_converted)
  else:
    diagnostics.append(
      _diagnostic(
        "deck-solver-missing",
        "the deck declares no solver block",
        SourceContext(source=str(pro_path)),
      ),
    )

  profiles: dict[str, _ElementProfile] = {}
  for block in pro_deck.element_blocks:
    profile = _element_profile(block, diagnostics)
    profiles[block.name] = profile

  dat_deck: _DatDeck | None = None
  dat_path: Path | None = None
  if pro_deck.input_ref is None:
    diagnostics.append(
      _diagnostic(
        "deck-input-missing",
        "the deck declares no readable 'input = \"<mesh>.dat\";' reference",
        SourceContext(source=str(pro_path)),
      ),
    )
  else:
    dat_path = (pro_path.parent / pro_deck.input_ref).resolve()
    try:
      dat_text = dat_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
      diagnostics.append(
        _diagnostic(
          "deck-input-missing",
          f"referenced mesh file is not readable: {pro_deck.input_ref!r}",
          SourceContext(source=str(pro_path)),
        ),
      )
    else:
      dat_deck = _read_dat_deck(dat_text, dat_path, diagnostics)

  if dat_deck is not None:
    rank = _check_mesh(dat_deck, SourceContext(source=str(dat_path)), diagnostics)
    _check_dofs(dat_deck, rank, diagnostics)
    groups = tuple(dict.fromkeys(element.group for element in dat_deck.elements))
    spring_groups = tuple(
      group
      for group in groups
      if (group_profile := profiles.get(group)) is not None
      and group_profile.element_type == "Spring"
    )
    interface_groups = tuple(
      group
      for group in groups
      if (group_profile := profiles.get(group)) is not None
      and group_profile.element_type == "Interface"
    )
    declaration_groups = spring_groups + interface_groups
    region_groups = tuple(group for group in groups if group not in declaration_groups)
    if dat_deck.elements and len(region_groups) != 1:
      diagnostics.append(
        _diagnostic(
          "unsupported-element-groups",
          f"the mesh declares {len(region_groups)} region-routed element groups "
          f"{region_groups}; the supported deck subset covers exactly one "
          "region-routed group plus any number of 'Spring' and 'Interface' "
          "groups",
          dat_deck.elements[0].source,
        ),
      )
    geometries: dict[str, _GeometryProfile | None] = {}
    spring_declarations: dict[str, SpringDeclaration] = {}
    interface_declarations: dict[str, InterfaceDeclaration] = {}
    for group in groups:
      profile = profiles.get(group)
      if profile is None:
        diagnostics.append(
          _diagnostic(
            "missing-element-block",
            f"mesh group {group!r} has no same-named .pro element block",
            SourceContext(source=str(pro_path)),
          ),
        )
        continue
      if profile.spring is not None:
        if rank in _SUPPORTED_MESH_RANKS:
          recorded = len(diagnostics)
          _resolve_geometry(
            group,
            rank,
            _SPRING_GEOMETRIES,
            dat_deck.elements,
            diagnostics,
          )
          if len(diagnostics) == recorded:
            declaration = _spring_declaration(
              group,
              profile.spring,
              dat_deck,
              diagnostics,
            )
            if declaration is not None:
              spring_declarations[group] = declaration
        continue
      if profile.interface is not None:
        if rank in _SUPPORTED_MESH_RANKS:
          recorded = len(diagnostics)
          _check_interface_arities(group, rank, dat_deck.elements, diagnostics)
          if len(diagnostics) == recorded:
            interface_declarations[group] = _interface_declaration(
              group,
              profile.interface,
              dat_deck,
            )
        continue
      if profile.family is None:
        geometries[group] = None
        continue
      geometry: _GeometryProfile | None = None
      if rank in _SUPPORTED_MESH_RANKS:
        geometry = _resolve_geometry(
          group,
          rank,
          profile.family.geometries,
          dat_deck.elements,
          diagnostics,
        )
      geometries[group] = geometry
      if (
        geometry is not None
        and profile.family.required_rank is not None
        and profile.family.required_rank != rank
      ):
        diagnostics.append(
          _diagnostic(
            "incompatible-material-geometry",
            f"material model {profile.family.material_model!r} requires a "
            f"{profile.family.required_rank}D mesh; the deck mesh is {rank}D",
            profile.family.source,
          ),
        )
    unused = set(profiles) - set(groups)
    for name in sorted(unused):
      diagnostics.append(
        _diagnostic(
          "unsupported-pro-construct",
          f"element block {name!r} matches no mesh group",
          profiles[name].source,
        ),
      )

  if settings is not None and settings.solver_type == "LinearSolver":
    for profile in profiles.values():
      if (
        profile.family is not None
        and profile.family.formulation == "total-lagrangian-continuum"
      ):
        diagnostics.append(
          _diagnostic(
            "incompatible-solver-type",
            "a FiniteStrainContinuum deck requires a NonlinearSolver block: "
            "the legacy LinearSolver assembles the zero-state tangent once, "
            "while the v3 driver integrates the finite-strain residual by "
            "Newton iteration — the combination has no faithful mapping",
            (
              pro_deck.solver_block.source
              if pro_deck.solver_block is not None
              else profile.source
            ),
          ),
        )
    if dat_deck is not None and spring_groups:
      diagnostics.append(
        _diagnostic(
          "incompatible-solver-type",
          "a deck with 'Spring' groups requires a NonlinearSolver or "
          "RiksSolver block: the legacy LinearSolver assembles the legacy "
          "isotropic spring tangent once — not the oracle per the H1 "
          "decision — while the v3 spring family ships the consistent axial "
          "tangent, so the single linear solve has no faithful mapping",
          (
            pro_deck.solver_block.source
            if pro_deck.solver_block is not None
            else SourceContext(source=str(pro_path))
          ),
        ),
      )
    if dat_deck is not None:
      for name in interface_groups:
        profile = profiles[name]
        if profile.interface is not None and profile.interface.law != "Dummy":
          diagnostics.append(
            _diagnostic(
              "incompatible-solver-type",
              f"a deck with 'Interface' group {name!r} on the nonlinear law "
              f"{profile.interface.law!r} requires a NonlinearSolver or "
              "RiksSolver block: the legacy LinearSolver assembles the "
              "zero-state tangent once, while the v3 driver integrates the "
              "cohesive residual by Newton iteration — the combination has "
              "no faithful mapping (the linear 'Dummy' law converts, since "
              "its assembled response is frame-independent and the single "
              "linear solve is exact)",
              (
                pro_deck.solver_block.source
                if pro_deck.solver_block is not None
                else profile.source
              ),
            ),
          )

  if diagnostics:
    raise DeckConversionError(tuple(diagnostics))

  if settings is None or dat_deck is None or dat_path is None:
    msg = "conversion invariants failed without a recorded diagnostic"
    raise ValueError(msg)

  group = region_groups[0]
  profile = profiles[group]
  geometry = geometries[group]
  if profile.family is None or geometry is None:
    msg = "conversion invariants failed without a recorded diagnostic"
    raise ValueError(msg)
  pro_source = SourceContext(source=str(pro_path))
  model = _emit_model(dat_deck, group, profile, geometry, pro_source)
  program = _emit_program(
    dat_deck,
    ramped=settings.solver_type == "NonlinearSolver",
    solver_source=pro_deck.solver_block.source
    if pro_deck.solver_block is not None
    else pro_source,
  )
  springs = tuple(spring_declarations[name] for name in spring_groups)
  interfaces = tuple(interface_declarations[name] for name in interface_groups)
  family = profile.family
  if family.registry_kind == "plasticity":
    registry = plasticity_reference_registry()
  elif family.registry_kind == "truss":
    registry = truss_reference_registry()
  elif family.registry_kind == "finite-strain":
    registry = finite_strain_reference_registry()
  elif (
    family.material_model == "plane-stress-linear-elastic"
    and geometry.geometry_interpolation == "serendipity-quad8"
  ):
    registry = q8_reference_registry()
  else:
    registry = continuum_reference_registry()
  return ConvertedDeck(
    name=pro_path.stem,
    pro_path=pro_path,
    dat_path=dat_path,
    model=model,
    program=program,
    registry=registry,
    solver=settings,
    not_converted=tuple(not_converted),
    springs=springs,
    interfaces=interfaces,
  )


def compile_deck(deck: ConvertedDeck) -> CompiledDeck:
  """Compile a converted deck through the landed compiler, unchanged."""
  if type(deck) is not ConvertedDeck:
    msg = "compile_deck requires an exact ConvertedDeck"
    raise TypeError(msg)
  system = compile_system(deck.model, deck.registry, springs=deck.springs)
  for declaration in deck.interfaces:
    interface_block, interface_operator = compile_interface_operator(
      system,
      declaration,
    )
    system = compose_interface_system(system, interface_block, interface_operator)
  constraint_map = compile_constraint_map(
    system,
    constraints=deck.program.constraints,
    coordinates=deck.program.coordinates,
  )
  return CompiledDeck(
    deck=deck,
    system=system,
    constraint_map=constraint_map,
    loads=deck.program.loads,
  )


def run_deck(deck: ConvertedDeck) -> DeckRun:
  """Compile and drive a converted deck through the landed drivers.

  ``RiksSolver`` decks route to the landed ``RiksDriver``: the deck's riks
  settings become the exact :class:`ArcLengthSettings` and the continuation
  owns the ``load`` coordinate autonomously (the base point binds nothing).
  Every other deck keeps the landed ``NonlinearStaticDriver`` schedule (a
  ``LinearSolver`` deck's single full step).
  """
  compiled = compile_deck(deck)
  if deck.solver.solver_type == "RiksSolver":
    riks_driver = RiksDriver(
      compiled.system,
      compiled.constraint_map,
      compiled.loads,
      ArcLengthSettings(
        tolerance=deck.solver.tolerance,
        max_iterations=deck.solver.max_iterations,
        optimal_iterations=deck.solver.optimal_iterations,
        fixed_step=deck.solver.fixed_step,
        max_lam=deck.solver.max_lam,
        max_factor=deck.solver.max_factor,
        cycle_cap=deck.solver.cycle_cap,
      ),
    )
    riks_result = riks_driver.run(base_point=ProgramPoint())
    return DeckRun(driver=riks_driver, result=riks_result)
  settings = NonlinearStaticSettings(
    tolerance=deck.solver.tolerance,
    max_iterations=deck.solver.max_iterations,
  )
  driver = NonlinearStaticDriver(
    compiled.system,
    compiled.constraint_map,
    compiled.loads,
    settings,
  )
  base_point = ProgramPoint(
    (ProgramCoordinateValue(_LOAD_COORDINATE, 0.0),),
  )
  target_points = tuple(
    ProgramPoint((ProgramCoordinateValue(_LOAD_COORDINATE, factor),))
    for factor in deck.solver.load_factors
  )
  result = driver.run(base_point=base_point, target_points=target_points)
  return DeckRun(driver=driver, result=result)
