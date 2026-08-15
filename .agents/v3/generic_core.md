# PyFEM v3 generic semantic core

- Status: **authoritative architecture amendment**
- Decision date: 2026-08-16
- Scope: the semantic intermediate representation, extension model, trust boundary,
  and the next proof portfolio
- Supersedes: the fixed `topology + quadrature + formulation + material + section`
  decomposition as a core contract; the frozen P2-A through P2-G execution plan
- Retains: the ownership, identity, state-transaction, contribution-ledger, and
  result-verification invariants in [design.md](design.md)

This document is the answer to a specific course correction: legacy PyFEM is a
coverage set, not a taxonomy to reproduce. Q8, Q4, T3, beams, shells, contact,
thermal response, phase field, RVE/FE2, ROM, loads, and output writers must become
compositions of a smaller generic finite-element core. Their old class boundaries
do not receive architectural status merely because they exist in v1.

The word *generic* here does not mean an entity-component framework, a universal
ragged table, or an untyped dictionary protocol. It means that the core names the
few mathematical roles shared by the legacy portfolio, while concrete builders and
numeric kernels retain the specialized data required by their physics.

## 1. Why the frontier is being reset

The Phase 0/1 work proved several valuable contracts, but it also produced a
decisive counterexample to the planned migration shape.

From the closed foundation checkpoint `c50ca70` to the integrated linear-slice
code head `7ee65c3`, the branch added 14,572 production lines and 6,323 focused
slice-test lines for one explicit stateless Q8 plane-stress flow. The resulting
implementation is correct for its frozen slice, but its growth is structural:

- `DomainBlock` persists separate topology, quadrature, formulation, and material
  identities even though the core runtime ultimately needs one bound local
  operator;
- `DofPlan` represents one nodal field instead of a collection of discrete spaces;
- the compiler, program compiler, assembly preparation, reference evaluator, and
  result verifier repeatedly inspect or reconstruct the same compiler-owned
  carrier meaning;
- assembly and result code import the program compiler's private
  `_validated_model` helper;
- Q8 descriptor bindings are replayed after compilation to prove that compiled
  shape and quadrature arrays still correspond to their source callables; and
- many tests target manually altered in-memory carrier slots rather than malformed
  authored input, restore data, numerical behavior, or a supported public flow.

Those choices made sense as an experiment in explicit correctness. They do not
scale to the 154-row legacy capability inventory. Adding Q4 and T3 to the same
shape would prove variable widths but retain the deeper taxonomy, duplicated trust
checks, and one-field assumptions.

The implementation through `7ee65c3` remains valuable executable evidence and a
reference oracle. It is not the representation that future breadth must extend.

## 2. Preserved decisions

The following ideas survived the re-evaluation and remain mandatory:

1. Authored convenience is normalized before numerical execution.
2. Compiled arrays are detached, owning, read-only, and organized into homogeneous
   hot blocks.
3. Semantic identity is independent of block order and dense execution indices.
4. Physical discretization, program inputs, accepted physical state, accepted
   evolution, trial evaluation, workspace, and published results have distinct
   owners and lifetimes.
5. Fixed affine constraints use one canonical coordinate map
   `x = P q + x_bar(p)`.
6. Repeated compatible contributions add; balance terms remain attributable.
7. Stateful evaluation is repeatable from one accepted state; rejection changes
   nothing and acceptance commits exactly one matching trial.
8. Sparse structure and backend workspaces are prepared outside numeric kernels
   and reused when their declared topology permits it.
9. Analyses request mathematical contribution channels; they do not import element
   or material classes.
10. Results are immutable accepted snapshots with provenance, ledgers, and a
    genuinely fresh verification path where the contract requires it.

These are semantic invariants. The current class names and field decomposition are
not.

## 3. The generic model

The core representation is a compiled discrete system:

```text
authored cases and adapters
        |
        v
normalized declarations
        |
        v
compiler --------------------------------------------------+
        |                                                   |
        v                                                   v
CompiledSystem                                      CompiledProgram
  entity blocks                                       signals
  discrete spaces                                     coordinate maps
  operator blocks                                     program operators
  state layout                                        initial/evolution data
  observations
        |                                                   |
        +---------------------- join -----------------------+
                               |
                               v
                     PreparedExecution
                       execution schedule
                       gather/scatter and sparse plans
                       backend workspace policy
                       effective capabilities
                               |
                               v
                    Analysis + state transaction
                               |
                               v
                  accepted result + verification
```

Six concepts form the stable **numerical** semantic IR. Input adapters, application
frontends, artifact encoders/decoders, logging, documentation, and output sinks are
typed producers or consumers around it; they are not forced to masquerade as
finite-element operators or spaces.

### 3.1 Entity blocks

An entity block is a homogeneous table of the things on which an operator acts.
Its incidence width and entity role are fixed inside the block. Examples include:

- points or nodes;
- volume/surface cells;
- boundary facets;
- cohesive/interface pairs;
- contact candidate pairs;
- whole-domain or global entities; and
- nested sample points used by a response backend.

An entity block owns stable semantic IDs, a dense incidence table, source
provenance, and any geometry/reference data common to every consumer. Different
incidence widths are different blocks. The core never pads them into a universal
rectangular mesh and never infers physics from their width.

### 3.2 Discrete spaces

A discrete space owns the global coefficients for one explicit field/basis
combination and the maps required to gather local coefficients. It records:

- semantic field and space identity;
- support entity kind and basis/interpolation identity;
- component names and physical/value shape;
- global coefficient count and semantic coefficient IDs;
- block-local gather maps; and
- evolution classification when a request binds the space as algebraic,
  first-order, or second-order.

Geometry is an explicit immutable input space or geometry map, not an implicit
consequence of the unknown field. Nodal displacement, cell temperature, phase
field, beam rotations, shell directors, internal condensed variables, and reduced
coordinates therefore differ by declared space semantics rather than optional
whole-model arrays.

The current single-field `DofPlan` becomes one special case of a tuple of spaces.

### 3.3 Operator blocks

An operator block is one homogeneous batch bound to one evaluator. It is the
runtime concept that replaces core knowledge of legacy element, material, section,
model-action, and load classes.

Every block has a small common header:

```text
OperatorBlock
  semantic block/entity IDs and sources
  entity block binding
  operator descriptor identity and implementation fingerprint
  port bindings to discrete spaces and program signals
  requested/available contribution channels
  local state schema and accepted/trial row layout
  coupling/topology policy
  typed operator-owned payload
```

The payload is deliberately specialized. A continuum payload may own shape
tables, quadrature, material parameters, and reference gradients. A beam payload
may own local frames and section data. A contact payload may own pair geometry and
active-set storage. A nested FE2 payload may own microproblem recipes. The generic
core does not flatten these into one mega-record.

The descriptor owns three boundaries:

1. **compile** authored operator intent into the typed payload and state layout;
2. **evaluate** one homogeneous block from plain arrays, accepted state, program
   signals, and a typed channel request; and
3. **observe** accepted state without accidentally advancing history.

Python orchestration dispatches once per block. Hot evaluators receive arrays and
scalars, never the registry, model, program, or dictionaries.

An evaluator also declares its execution modality:

- a direct array kernel;
- a local recomputation such as condensation or recovery; or
- an explicit auxiliary/nested execution rooted at a declared accepted state.

The modality changes scheduling, state, and verification obligations without
creating another operator kind. In particular, an RVE tangent obtained by auxiliary
unit-strain micro solves is a derivative channel from explicit auxiliary execution;
it is not hidden inside a passive observation or commit hook.

Topology, quadrature, kinematics, response law, and section may remain reusable
implementation components and public authoring helpers. They are not mandatory
core registry kinds. A standard continuum builder may compose them into one
operator descriptor; a beam or nested solver may use a different internal
composition while satisfying the same block contract.

### 3.4 Contribution channels

Operators publish typed mathematical channels. The initial standard vocabulary
includes:

- attributed residual-vector terms;
- Jacobian/tangent terms and their derivative target/source spaces;
- mass, capacity, and damping bilinear operators;
- scalar functionals such as energy or dissipation and their derivatives;
- constraint/contact terms when they cannot be compiled into a fixed coordinate
  map; and
- named observations with explicit entity/support and reduction semantics.

Channels carry sign, balance role, linearity/state-dependence, symmetry, and
topology guarantees where those are real. An evaluator may produce a fused total
for performance while a diagnostic request asks for attributable parts.

An analysis consumes only the channels it declares. Adding a new operator that
already provides those channels requires no analysis edit.

### 3.5 Coordinate maps and programs

Fixed Dirichlet conditions, affine MPCs, prescribed values, and reduced bases are
coordinate transformations, not element variants. The canonical full-to-free map
remains separate from residual-producing operators. Coordinate maps that cross
process or run boundaries, including ROM bases, are versioned artifacts with schema,
content identity, source/target-space identity, implementation compatibility, and
restore validation; they are not anonymous arrays.

A compiled program owns:

- named typed signals such as time, load, continuation, and environmental values;
- fixed coordinate maps and parameter-dependent offsets;
- program-owned operator blocks such as nodal, boundary, body, follower, and
  interaction contributions;
- initial conditions and signal schedules; and
- program/interaction state layouts when behavior itself has history.

Model-owned and program-owned operator blocks share the same evaluation contract;
their ownership and lifetime remain distinct.

### 3.6 State, execution, and observation

Physical state stores global discrete-space coefficients plus operator-local
accepted state by block. The core does not divide local state into mandatory
“material” and “formulation” arrays. A descriptor may expose named sublayouts for
diagnostics, but atomic trial/commit applies to the whole operator-owned state row.

Prepared execution joins model, program, request, coordinate maps, operator
couplings, and backend policy. It owns the execution schedule, sparse/action plans,
gather/scatter slots, caches, and mutable scratch. It owns no authoritative
accepted physical history.

Observations are solve-free typed projections or reductions from accepted spaces,
operator state, and attributed ledgers. Stress recovery, failure criteria,
homogenized RVE stress, reactions, contour fields, histories, and export payloads
use this boundary. A quantity requiring an auxiliary solve is produced by an
explicit execution/channel request and may then be observed. File writers are sinks
over observations; they are not solver actions.

## 4. The variation axes behind the legacy portfolio

Legacy classes are combinations of a small set of independent axes:

| Axis | Representative values |
|---|---|
| entity/incidence | point, cell, boundary facet, interface pair, contact pair, global |
| discrete spaces | displacement, temperature, phase, rotations, mixed/coupled, reduced |
| operator channels | residual, tangent, mass/capacity, damping, scalar functional, observation |
| local state | stateless, material-like, kinematic, condensed, irreversible, active-set, nested |
| structure lifetime | fixed, safely over-allocated, dynamic, matrix-free |
| evaluator backend | vectorized array kernel, compiled loop, orchestrated nested solve |
| analysis evolution | algebraic, continuation, first-order, second-order, spectral, composite |

The 154-row inventory is retained unchanged as evidence, but future planning maps
each row onto these axes and the six IR concepts. It no longer assigns architecture
from the row's legacy directory or class family.

## 5. Legacy cases as instances

The following mappings are architectural acceptance examples.

| Legacy family | Generic instance |
|---|---|
| small/finite-strain continuum | cell entity block + displacement space + stateful/stateless residual/Jacobian operator |
| truss, spring, beam, plate, shell | structural entity block + explicit component/frame spaces + specialized operator payload |
| thermal and thermomechanical | temperature and optional displacement spaces + capacity/residual/Jacobian channels |
| phase field | coupled displacement/damage ports + irreversible block state + residual/Jacobian channels |
| cohesive interface | fixed interface-pair entity block + jump operator + local damage state |
| distributed/nodal loads | program-owned facet/point operator blocks producing attributed residual terms |
| contact | program/model interaction pair block + active-set state + dynamic/over-allocated topology policy |
| fixed constraints and periodic RVE BCs | compiled coordinate map and signal-dependent offset |
| RVE homogenization | observation/reduction over boundary reactions and accepted fields |
| FE2 | ordinary operator descriptor whose evaluator is an orchestrated nested response and whose state owns micro checkpoints |
| linear/Newton/Riks/dissipated/explicit | analysis algorithms over requested channels and explicit evolution state |
| modal/buckling | spectral analyses over named bilinear operators and a preloaded accepted state |
| staggered/multi-stage | composite analysis schedule over field partitions with one atomic accepted transition policy |
| ROM | alternate/reduced discrete space plus coordinate map, full-residual verification, and basis compatibility |
| failure criteria/output projection | observation descriptors over accepted response/state, not material mutations |
| VTK/HDF5/text/graph | sinks consuming typed observations and provenance |

This table is not a parity claim. It is a falsifiable representation claim: a
legacy capability that cannot be expressed without changing the six core concepts
is a counterexample to this design and must be recorded before implementation
continues.

## 6. Trust and validation boundary

Validation occurs where data changes authority or representation:

1. adapters validate external syntax and retain source locations;
2. normalization validates authored semantic declarations and detaches inputs;
3. compilation resolves descriptors, compatibility, geometry, spaces, operator
   payloads, state layouts, and source/entity maps;
4. restore/rebind validates persisted schema, content identity, implementation
   compatibility, and array ownership once; and
5. public result verification recomputes the mathematical claims it advertises.

Compiler-produced objects are then trusted internal values. Downstream modules
check live instance compatibility, state generation, request capability, and
numerical preconditions, but do not recursively reconstruct every dataclass or
replay every compiler binding on each call.

Unsupported manual alteration of private frozen carriers is outside the normal
runtime contract. Python assertions may detect internal inconsistencies; the
library does not promise bounded public diagnostics after deliberate alteration
through low-level object primitives. Content fingerprints are provenance and
restore compatibility keys, not a substitute for the compiler boundary.

This changes the emphasis of existing tests:

- keep malformed authored input, adapter, restore, identity mismatch, aliasing,
  state transaction, geometry, balance, and numerical edge cases;
- keep independent mathematical verification and source-aware diagnostics;
- move carrier decoder cases to the actual decoder/restore boundary; and
- retire repeated downstream tests whose only purpose is to alter an already
  compiler-owned exact object and demand full public revalidation at every layer.

The expected consequence is causal deletion of duplicated validators and meaning
witnesses as the generic path replaces the frozen Q8 path.

## 7. Extension laws

The architecture passes only if these laws remain true:

1. Adding Q4 or T3 changes a continuum descriptor/builder and operator payload,
   not `CompiledSystem`, an analysis, or a result type.
2. Adding a beam or contact operator changes no global carrier field and no solver
   dispatch on legacy type names.
3. Adding a material or section with no new channel/state semantics changes only
   operator construction/evaluation components and tests.
4. Adding a new field creates a discrete space and port bindings; unrelated
   entities receive no padded DOFs.
5. Adding a new analysis names required channels/evolution semantics and imports
   no concrete operator implementation.
6. Adding a file format produces normalized declarations; it does not create a
   second compiler or solve path.
7. Adding an output uses observations/ledgers; it does not mutate accepted state or
   call trial-update kernels.
8. A backend may change schedules, sparse formats, or fusion, but not the semantic
   system, state ownership, channel meaning, or verification result.

## 8. Designs not selected

### 8.1 Continue the frozen P2-A topology expansion

Not selected. It would remove some Q8 literals and prove native-width blocks, but
it would preserve the core's legacy descriptor taxonomy, one-field carrier, and
repeated validation structure. The lost temporary worktree had no commit or test
evidence, so there is no implementation asset to recover.

### 8.2 Make every legacy class a plugin kind

Rejected. Renaming `Element`, `Material`, `Section`, `Model`, and `Solver` as five
plugin registries reproduces their coupling and forces the core to know their old
boundaries.

### 8.3 One universal array schema

Rejected. Specialized payloads and native-width entity blocks are required for
clarity and hot-loop regularity. Genericity lives in ports, channels, state, and
execution contracts—not in flattening all numerical data.

### 8.4 A free-form operator dictionary

Rejected. Operator headers, ports, channel descriptors, state layouts, identities,
and payload schema are typed and versioned. Extension does not mean silent keys or
runtime guessing.

### 8.5 Treat every constraint as a residual operator

Rejected. Fixed affine constraints and reduced bases are exact coordinate maps.
Nonlinear inequalities/contact may be operators, but forcing simple kinematics
into penalty or multiplier assembly would weaken semantics and conditioning.

### 8.6 Delete the proven Phase 1 path immediately

Rejected. It remains the reference until the generic path reproduces its numerical,
balance, state, and verification evidence. Replacement must migrate meaningful
tests before causal deletion.

## 9. Decisive proof portfolio

The next work is not “port the next element.” It must test whether the generic IR
survives unlike cases.

### Proof A — three unlike operators

One compiled system combines:

- a continuum cell operator;
- a structural link/spring operator with a different local layout; and
- a program-owned boundary or point load operator.

They share one displacement space and assemble through one channel/plan boundary.
The analysis contains no topology, formulation, material, or ownership branch.
The exact matrix, force, balance, and public result are independently known.

### Proof B — spaces and coupling

Compile mechanical-only, thermal-only, and coupled field/operator signatures with
nonstandard component ordering. Only active entity-space pairs receive
coefficients. The proof need not claim broad thermal physics, but the IR and
coupling graph must represent the real layouts without padding.

### Proof C — real local state

A stateful response block is evaluated twice from one accepted state, rejected,
then evaluated and accepted. A stateless block coexists. No analysis code knows
which state subfields are conventionally called material or formulation history.

### Proof D — difficult structure

Before broad porting, read-only counterexamples must cover dynamic contact pairs,
nested FE2 response, spectral operators, staggered coupled fields, and ROM
coordinate transforms. An unrepresentable case stops the design and amends this
document.

## 10. Structural success bar

The generic path is not accepted because its diagrams look cleaner. It must show:

- executable representation of Proof A and B before broad feature work;
- no concrete operator import or type-name branch in analysis/result code;
- one validation pass per authority boundary, with no compiler-private recursive
  model validator imported by assembly or results;
- a causal deletion map for Q8-only carrier, replay, and repeated-validation code;
- retained Phase 1 numerical, balance, ownership, transaction, and fresh-result
  evidence;
- representative allocation/runtime measurements before hot-path specialization;
  and
- at least two unlike legacy families added without changing the six IR concepts.

Temporary coexistence is allowed only for a bounded proof wave. The new path must
either demonstrate a credible one-wave migration/deletion cut or remain an
unintegrated experiment. We do not permanently maintain two semantic cores.

## 11. Current bounded simplification

No existing P2-A writer is resumable. Its temporary worktree is gone, its branch
contains no child commit, and the frozen card is superseded by this amendment.

R2-E, P2-H, and R2-F are complete and adjudicated. They found no seventh numerical
concept, proved one generic continuum/link/load boundary, and selected a bounded
direct vertical replacement with causal deletion. The production cut is not part
of the current pass.

The only live packet is **S2-A — proof-core simplification**. It removes accidental
owner/role duplication, mirrored identifiers, repeated factory ceremony, redundant
test scaffolding, and naming drift from the unexported P2-H proof. It adds no
capability and touches no Phase 1 file or public export. It must preserve all seven
focused proof cases, exact Q4/T3/spring/load/global-balance oracles, typed
ports/channels, native-width entity blocks, provenance attribution, immutable
ownership, normal and restricted-digit gates, and the normalize-once trust
boundary. The target is at most 650 nonblank production and 360 nonblank test lines.

After one independent correctness review and any bounded repair, I0 records the
future vertical-cut card and pauses. No subsequent implementation packet is
dispatched in this pass.

## 12. Decision record

### 2026-08-16 — Replace the legacy-shaped block taxonomy with operator IR

- **Changed boundary:** topology/formulation/material/section/model/load are no
  longer mandatory core descriptor kinds; bound operator blocks over discrete
  spaces and typed channels are the core runtime unit.
- **Forcing evidence:** one correct Q8 linear slice grew by 14,572 production lines
  after the closed foundation, while compiler-owned meaning is recursively checked
  across program, assembly, and results. The planned Q4/T3 expansion would not
  remove that structure.
- **Strongest alternative:** complete frozen P2-A, then generalize materials,
  fields, boundaries, and sparse plans incrementally. It is smaller immediately
  but entrenches the wrong extension seam and delays the first unlike-family test.
- **Chosen consequence:** freeze the existing path as an oracle, map legacy cases
  onto a six-concept semantic IR, prototype three unlike operators, and require a
  bounded causal deletion cut before production integration.
- **Acceptance surface:** the extension laws, Proof A-D, structural success bar,
  and R2-E/P2-H/R2-F batch above.

### 2026-08-16 — Qualify the core boundary after the 154-row instance audit

- **Evidence:** R2-E classified 95 rows as direct, 56 as composed, two as
  questionable, and the rejected prototype carrier itself as not representable.
- **Counterexample:** the legacy RVE tangent performs auxiliary constrained micro
  solves and therefore cannot truthfully inhabit a passive observation contract.
- **Decision:** the six concepts describe numerical semantics, not ecosystem or
  adapter surfaces; observations are solve-free; evaluator modality makes
  auxiliary/nested execution explicit; persisted ROM coordinate maps are versioned
  artifacts.
- **Consequence:** no seventh finite-element concept or restoration of v1
  Element/Material/Section/Model boundaries is required. Proof-A code proceeds only
  to a bounded simplification pass; the production vertical cut remains a later,
  separately frozen decision.
