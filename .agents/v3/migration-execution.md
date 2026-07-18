# PyFEM v3 migration execution ledger

- Status: foundation gate GO; R0-E ingested; P0-D dispatched from frozen Horizon
- Owner: delegating/integration thread
- Target branch: `v3`
- Design authority: [design.md](design.md)
- Migration method: [migration_workflow.md](migration_workflow.md)
- Supporting structural method: [refactor_playbook.md](refactor_playbook.md)
- Original dispatch base: `c75cbf3523349deb40bd2b07de7959e7606c3b1f`
- Integrated foundation implementation: `108552ddd3382163a0e15c2fef7ca34e75f974fd`
- Integrated exact-integer diagnostic repair: `461a8a85de622820eb627a24b8e53749489d020a`
- Combined repaired foundation code head: `fb358fc0609b81a12cee4a1a66c2dae98edf5cae`
- Integrated registry-snapshot repair: source `0947dd7ca3b70bef2ebdf986f2336f62f5b3c08b`, integrated `fb358fc0609b81a12cee4a1a66c2dae98edf5cae`
- Workflow proposal integrated: `9b26574f52c39be756e2cdeb275dcfe7691e5bc4`
- Active milestone: `P0-D · model compiler — Q8 block frozen`

## Exact next safe action

Let the single Sol/max P0-D task `019f75dd-f597-7c21-adfc-d78b1e2580c0` execute
the frozen Horizon from exact clean base
`2842ef84b86f04f82587f5fc378584f30f6b62bd`. It is performing its required
base/document preflight in an isolated worktree. Do not dispatch a second writer,
R0-F, ProgramSpec, assembly, state, or solver work. On P0-D's own final plus direct
callback, I0 reviews ancestry, owned paths, design/physics meaning, diagnostics,
correctness matrix, and all reported gates before any integration or repair action.

Blocked condition: the P0-D Horizon cannot assign one target owner and one exact
descriptor/compiler contract without a new design, physics, compatibility,
dependency, or authority decision, or its exact clean base and required evidence
cannot be established.

## Semantic decisions and open questions

Decided:

- `design.md` remains authoritative; the legacy runtime and current v3 prototype are
  requirements and evidence, not target architecture.
- Development continues on local branch `v3`; the separate unmerged modernization
  line is not silently imported.
- Full migration uses the bounded packet conveyor in `migration_workflow.md`, direct
  terminal callbacks, serial integration, and independent finite-element
  correctness proof.
- A task is complete only when its own final response and callback satisfy the
  terminal-result contract. Idle state, commentary, recovered evidence, or a
  completed replacement never completes the original task.
- Task prompts use only domain-specific finite-element correctness language and
  repository-local evidence: finite-element correctness review, local edge-case
  matrix, correctness matrix, failure case, and independent reviewer. Historical
  packet wording is not reusable prompt text, and unrelated labels, metaphors,
  skills, or review frames are not routed into this migration.
- The 2026-07-18 audit found 12 properly completed migration tasks and five
  incomplete tasks. All five are visibly prefixed `INCOMPLETE` in the task list and
  recorded separately below.
- R0-E completed with both terminal signals. Its exact E0 inventory is preserved in
  [the dated evidence note](evidence/2026-07-18-r0e-capability-inventory.md) at
  SHA-256 `f9e326da3adc8e6bb3a777fb4134965ade6f9b364df352c53e68c2cbb7e377cf`;
  all 154 rows remain E0, and classification does not upgrade behavioral evidence.
- The R0-E preserve/change/retire values are accepted as working migration
  dispositions. A working `retire` disposition records the intended successor or
  loss decision; it does not authorize deletion, compatibility loss, or a public
  deprecation. The six public retirement candidates below remain decision-blocked.
- RVE/FE2 and implemented ROM intent are preserved through explicit redesign; they
  are not silently excluded. The current v3 whole-problem carrier, string registry,
  duplicate assembly bridge, and direct solve surface remain causal-retirement
  candidates only after their approved replacements are proved.
- Thread IDs are never reused. `R0-E` and `R0-F` remain reserved for the planned
  legacy-breadth and compiler reviews; replacement/follow-up foundation reviews use
  the next otherwise-unreserved IDs, hence `R0-G`, `R0-H`, and `R0-I`.
- dtype metadata is outside the current v1 manifest/finalization boundary and must
  fail closed; do not invent a recursive dtype-metadata schema or manifest v2 before
  an executable compiler slice demonstrates that semantic requirement;
- structured arrays remain unsupported by manifest v1, while metadata-free
  structured finalization may remain only if it preserves full dtype/value ownership;
  and
- total `FinalizedArray` identity semantics cover the carrier and standard NumPy
  comparison protocols, not arbitrary unrelated left operands whose own equality
  method controls dispatch.

Open and not implicitly decided:

- root Python API, `pyfem` CLI, and `.pro`/`.dat` compatibility;
- optional GUI preservation, replacement, or retirement;
- approval of public retirement candidates `ECO-ROOT-MESH-API`,
  `ECO-PYTHON-API`, `ECO-GUI`, `ECO-ARCHIVES`, `ADP-PICKLE-IN`, and
  `RES-PICKLE`;
- exact RVE/FE2 nesting, state, restart, cost, and result semantics, plus exact ROM
  snapshot/basis/reduced-solve contracts; and
- the later-phase registry-callable trust and accepted-generation transaction
  obligations recorded below.

## Capability coverage ledger

The complete E0 discovery payload is frozen in
[2026-07-18-r0e-capability-inventory.md](evidence/2026-07-18-r0e-capability-inventory.md).
That immutable evidence note contains all 154 exact 15-field JSONL rows, the source
audit, the 606-path zero-remnant coverage proof, all 122 property-file mappings,
all 33 skim paths, all 41 test/support paths, the dependency view, limits, and
machine audit. This section is the sole mutable status overlay; corrections to
disposition, lifecycle, grade, or proof are recorded here and never rewrite the
dated E0 evidence.

| Live scope | Rows | Lifecycle and evidence | Disposition state | Next transition |
|---|---:|---|---|---|
| P0-D immediate dependency cut | 18 | `contracted`, `E0` at `2842ef8` | 11 preserve, 7 change; frozen Q8 compiler Horizon below | P0-D implementation, then component proof toward `provisional`/E2 |
| Later preserve/change portfolio | 123 | `inventoried`, `E0` | Working dispositions accepted; slice-specific E1 extraction and semantic adjudication still required | Select only when dependencies pass |
| Internal/duplicate retirement candidates | 7 | `inventoried`, `E0` | Working `retire`; no deletion before replacement or unique-behavior proof | Dedicated causal-retirement proof |
| Public retirement candidates | 6 | `blocked`, `E0` | Working `retire`; explicit approval and compatibility/loss statement absent | Delegator/public decision packet |
| **Total** | **154** | all rows accounted for | 52 preserve, 89 change, 13 retire | no hidden or undecided row |

The 18 classified P0-D rows are:

```text
V3-AUTHORED-SPEC      V3-SPEC-NORMALIZE     V3-ARRAY-OWNERSHIP
V3-LIVE-ID            V3-CONTENT-ID         V3-REGISTRY-SNAPSHOT
MESH-NODES            MESH-CELLS            MESH-GROUPS
COMP-MESH             COMP-DOF              COMP-REGISTRY
COMP-MODEL            KERN-SHAPES           KERN-QUADRATURE
KERN-KINEMATICS       MAT-PLANE-STRESS      FORM-SMALL-CONT
```

The seven internal/duplicate retirement candidates are `ANAL-MODAL-DUP`,
`ROM-LINEAR-MANIFOLD`, `ROM-QUADRATIC-MANIFOLD`, `V3-PROTOTYPE-CARRIER`,
`V3-PROTOTYPE-REGISTRY`, `V3-PROTOTYPE-ASSEMBLY`, and
`V3-PROTOTYPE-ANALYSIS`. The six public candidates are listed under open decisions
above. [feature-parity.md](feature-parity.md) is now historical input: its old
`done`, `deferred`, and `out of scope` labels have no live status authority.

## P0-D Horizon card — `HORIZON_FROZEN`

**ID/title:** `P0-D · model compiler — Q8 block frozen`

**Outcome and ownership invariant:** `compile_model(...)` consumes only the exact,
detached result of `normalize_model_spec(...)` plus an injected registry. For the
single supported slice it returns one structurally frozen `CompiledModel` whose
arrays are detached, owning, contiguous, and read-only; whose live identity is
fresh; whose content fingerprint includes normalized semantics, source/entity
mapping, numeric policy, and the exact selected registry snapshot; and whose
physical-state layout describes sizes without allocating any evolving state value.
The compiler owns semantic resolution and homogeneous recipe construction. Numeric
kernels own only explicit array/scalar operations, and the prototype carrier owns
nothing on this path.

**Coverage rows advanced:** `V3-AUTHORED-SPEC`, `V3-SPEC-NORMALIZE`,
`V3-ARRAY-OWNERSHIP`, `V3-LIVE-ID`, `V3-CONTENT-ID`, `V3-REGISTRY-SNAPSHOT`,
`MESH-NODES`, `MESH-CELLS`, `MESH-GROUPS`, `COMP-MESH`, `COMP-DOF`,
`COMP-REGISTRY`, `COMP-MODEL`, `KERN-SHAPES`, `KERN-QUADRATURE`,
`KERN-KINEMATICS`, `MAT-PLANE-STRESS`, and `FORM-SMALL-CONT`. Freezing this card
moves the rows to `contracted` but leaves them at E0. Implementation plus the
independent R0-F review may advance component evidence to E2; no row reaches E3
before the authored-to-verified-result Phase 1 flow exists.

**Exact base and required parent packets:** required parent is the clean inventory
ingestion commit `80edd4af7456321e42376a006ee428d6c690080b`; the writer starts from the
single documentation commit that freezes this card, whose exact hash is supplied
in the dispatch prompt and packet ledger. P0-A through P0-H, R0-M, R0-E, and I0
foundation proof are required and complete. No parallel writer is permitted.

**Owned paths:** new `pyfem/v3/compile/**`; new compiled-carrier modules under
`pyfem/v3/model/**`; `pyfem/v3/model/__init__.py` only for internal model exports;
and new `test/v3/test_v3_model_compile.py`. Existing spec, identity, provenance,
registry, shape, quadrature, kinematics, and plane-stress modules are read-only
dependencies.

**Forbidden paths and non-goals:** `.agents/v3/**`, `pyproject.toml`, `uv.lock`,
root configuration, `pyfem/v3/__init__.py`, `pyfem/v3/spec/**`, existing foundation
tests, prototype `types.py`, `pack.py`, `_prototype_assembly.py`, `solver/**`,
`io/**`, and existing numeric kernels. Do not add `ProgramSpec`, sections,
constraints, loads, initial conditions, a final sparse pattern, SciPy objects,
prepared assembly, physical-state values, state transactions, analysis/solver/
result APIs, file parsing, root exports, legacy adapters, compatibility wrappers,
GUI, RVE/FE2, ROM, output behavior, Numba policy, or performance thresholds.

**Supported authored slice and descriptor contract:** exactly one cell block with
one or more cells and exactly one region are executable in P0-D. The block declares
`reference_topology="quadrilateral"`, topological and embedding dimension 2, and
`geometry_interpolation="serendipity-quad8"`; every cell has exactly eight node
IDs in the existing serendipity Q8 local order. The region covers every source cell
exactly once, references exactly one node field with components `("x", "y")` in
that physical order, and declares `formulation="small-strain-continuum"`,
`quadrature="gauss-3x3"`, and one material whose
`model="plane-stress-linear-elastic"`. That material has exactly finite
`youngs_modulus` and `poisson_ratio` scalars, with `E > 0` and
`-1 < nu < 0.5`. The first slice is per-unit out-of-plane thickness; no implicit
section value is stored, and any non-unit thickness remains a later explicit
section contract.

The compiler selects only these injected registry keys:

| Kind | Name | Required meaning |
|---|---|---|
| `topology` | `serendipity-quad8` | quadrilateral, 2D parent/embedding, eight-node local convention, shape values and parent gradients |
| `quadrature` | `gauss-3x3` | deterministic nine-point tensor Gauss rule in the same parent coordinates |
| `formulation` | `small-strain-continuum` | two-component nodal displacement, engineering-shear Voigt order, no formulation history, symmetric material tangent contribution |
| `material` | `plane-stress-linear-elastic` | ordered `[E, nu]` schema, plane-stress engineering-shear law, no material history |

Descriptor metadata must carry and the compiler must validate these compatibility
facts; matching names alone are insufficient. The selected callables are exact
references captured by `RegistrySnapshot`. The topology, quadrature, kinematics,
and plane-stress behavior may reuse the existing mathematically relevant functions
without modifying them or binding the prototype shape/rank-dispatch stiffness
wrapper. Callable implementation truth and restore/rebind remain recorded later
obligations; P0-D does not claim to solve them.

**Required compiled meaning:** the returned carrier has the semantic content of
`CompiledMesh`, `DofPlan`, one `DomainBlock`, `ModelAssemblyTopology`,
`PhysicalStateLayout`, `ModelCapabilities`, `EntityIndex`, `SourceMap`, registry
snapshot, provenance manifest/fingerprint, and live `InstanceId`. Exact internal
field splitting may remain compact, but it must prove all of the following:

- canonical dense node and cell indices derive from stable semantic IDs, not source
  declaration order; Q8 connectivity retains physical local-node order;
- global DOFs are deterministic node/declared-component pairs, and the block owns
  an explicit `(n_cell, 16)` local-to-global map rather than relying on a kernel's
  node-major assumption;
- source/entity records preserve node, cell, field, region, block, and stable local
  integration-point identities independently of execution ordering;
- the domain recipe contains explicit descriptor identities, connectivity, DOF
  map, quadrature/shape recipe, ordered material parameters, integration layout,
  and backend-neutral element-coupling information; no final sparse indices or
  matrices are built;
- dense index arrays use a declared integer dtype whose capacity is checked before
  conversion; numeric compiled values use `float64`;
- `PhysicalStateLayout` describes the global primary-field size and zero-width
  material/formulation history for every element/integration point, but contains no
  displacement, stress, history, or other evolving value array; and
- capabilities are derived, never accepted as authored claims: linear elastic,
  fixed model coupling, symmetric constant material tangent, conservative internal
  contribution, no mass/damping/storage channels, and no restart history.

All compiled carrier dataclasses use structural freezing and identity equality.
No mutable list/dictionary/parser object is retained. Sharing a single finalized
read-only array between two compiled views is allowed; retaining a caller-owned
array or two independently mutable semantic owners is not.

**Geometry policy and failure behavior:** compilation evaluates the explicit Q8
shape gradients at the explicit nine quadrature points and audits each source cell
in reference configuration. Jacobian classification is translation- and
scale-invariant, based on `float64` Jacobian magnitude rather than a fixed absolute
length. A non-finite mapping, scale-relative near-singular Jacobian, uniformly
negative orientation, or determinant sign change fails during compilation with a
bounded deterministic diagnostic carrying the source cell identity/context. The
compiler never applies `abs(det J)`, reorders connectivity, or repairs geometry.
Wrong topology/dimension/arity, incomplete or multiply assigned cells, incompatible
field/material/descriptor metadata, missing registry keys, invalid material
domains, index overflow, and malformed exact inputs also fail before a compiled
model is returned. Normalization failures remain `ModelSpecValidationError`;
compiler compatibility/geometry failures use one explicit compilation-error type
and must not leak incidental Python/NumPy exceptions for covered inputs.

**Public flows and consumers:** P0-D is internal and has no root public flow. Its
only immediate consumers are P1-A compiled-program compatibility, P1-B assembly
composition, P1-C state/linear analysis, and R0-F read-only verification. Later
block compilers must be able to add Q4/T3 without changing `CompiledModel` into a
union-shaped carrier.

**Strongest competing design:** extend `ProblemDefinition`/`pack.py`, or make a
thin immutable wrapper over their rectangular arrays and string registry. Rejected:
that preserves shape/rank inference, mutable caller meaning, one global
constitutive carrier, missing source/entity ownership, and solver-coupled topology.
Also rejected for this packet are a universal ragged element table and premature
`SectionSpec`/program/assembly/state APIs. The Q8 compiler recipe is the smallest
owner-plus-consumer boundary that can later replace the prototype rather than hide
it.

**Baseline and independent reference:** foundation baseline is 42 normalized-spec
tests plus 68 identity/storage tests, each also under
`PYTHONINTMAXSTRDIGITS=640`; the exact pre-dispatch v3 baseline is 240 tests. Q8
local ordering and shape/quadrature values are checked against the existing direct
mathematical tests and the `PatchTest8` mesh only as independent component/oracle
evidence. Prototype `ProblemDefinition`, `pack`, or solve results are not API
fixtures for this packet.

**Required correctness matrix:** successful compilation proves detached/read-only
ownership, fresh live identity, stable content identity, exact registry capture,
canonical declaration-order independence, complete source/entity lookup, explicit
DOF and coupling maps, empty state values, and derived capabilities. Failure cases
cover caller mutation after compile; duplicate/missing cross-region membership;
wrong Q8 arity/topology/dimensions; incompatible or absent descriptor metadata;
missing/extra/invalid material parameters; non-finite and out-of-domain values;
valid geometry at very small and very large scales; inverted, sign-changing, and
scale-relative near-singular cells; mixed string/integer semantic IDs; and bounded
diagnostics for very large exact integer IDs. A second compilation of semantically
equivalent reordered declarations has the same fingerprint and canonical arrays
but a different live instance ID. A semantic, source-map, numeric-policy, or
selected-descriptor change changes the fingerprint.

**Expected migrations and causal deletions:** no prototype path is deleted in
P0-D. The new compiler path becomes the sole owner for later Phase 1 work. P1-C may
retire the Q8 use of `ProblemDefinition`/`LoadedProblem`, `pack.py`, string
registry, prototype assembly, and direct linear solve only after its public flow is
proved; other prototype consumers remain until their own slices migrate.

**Compatibility/API consequence:** all new names are internal and are not exported
from `pyfem.v3`. Legacy class/type/property strings remain adapter questions. The
P0-D exact descriptor vocabulary is a versioned internal contract, not approval of
root API, CLI, `.pro`/`.dat`, GUI, archive, or output compatibility.

**Performance relevance and envelope:** no performance claim or gate. Use clear
NumPy/reference compilation code, do not add a backend or Numba policy, and do not
retain both a canonical dense connectivity/DOF map and a second ragged execution
copy. Compiler temporaries may exceed retained payload during validation; report a
measured concern only if the focused correctness fixture reveals an unbounded or
obviously quadratic owner error.

**Acceptance commands and evidence:** one focused commit and clean worker tree;
focused compiler plus foundation tests normally and with
`PYTHONINTMAXSTRDIGITS=640`; direct Q8 shape/quadrature/element component tests;
both required Ruff configurations and focused format check; `pytest -q test/v3`;
full `pytest -q`; `git diff --check`; a search proving the new compiler does not
import prototype `types`, `pack`, `_prototype_assembly`, solver, or I/O modules;
and the complete edge-case evidence above. Report all warnings without upgrading
them into findings unless they change this contract.

**Merge order:** writer returns one source commit; I0 checks ancestry, owned paths,
diff, contract, focused/static/v3/full evidence, and then integrates serially. Only
after combined proof does a fresh Sol/max R0-F task independently verify the
integrated block invariants. Accepted findings return to the same writer for at
most two bounded repair turns. P1-A stays closed until R0-F is adjudicated.

**Blocked condition and minimum unlock:** stop without broadening scope if the
exact descriptor compatibility cannot represent the Q8 recipe, a correct geometry
audit requires modifying existing numeric kernels, the current spec cannot express
the supported slice without a new authored owner, a forbidden shared path is
required, or a physical convention other than the explicit per-unit-thickness
slice must be chosen. Return the smallest reproducible conflict; I0 decides whether
to amend design/card or create a new packet.

**Integration task ID:** `019f6f49-0b72-7d73-86da-c6b85519eeaf`.

## Outcome and invariant

Establish the smallest foundations needed for the Phase 1 Q8 vertical slice without
letting the current prototype carrier define the new API.

The first chunk must prove three facts independently:

1. the colliding prototype assembly module is visibly quarantined without changing
   behavior;
2. authored/normalized model intent can represent explicit topology, fields,
   regions, materials, and quadrature without shape inference; and
3. compiler-owned data primitives provide strict live identity, deterministic
   content provenance, owned read-only arrays, and frozen registry meaning.

No packet may introduce a second whole-problem carrier, public solver API, final
assembly plan, or compatibility shim.

## Authority and non-goals

- Start from the committed `v3` HEAD that contains this execution artifact and the
  adopted playbook; each dispatched prompt records the resolved commit.
- Work only in the named worktree branch and owned paths.
- Commits are authorized; merge and push are not.
- Shared files owned by the delegator and forbidden to workers:
  `.agents/v3/**`, `pyproject.toml`, `uv.lock`, `pyfem/v3/__init__.py`, and root
  project configuration.
- Do not fix the known prototype correctness failures in these packets; they are
  edge cases for the replacement boundary.
- Do not optimize, add Numba policy, expose public exports, or continue the old
  roadmap.

## Strongest competing hypothesis

The prototype could be extended in place by adding fields to `ProblemDefinition`
and more registry/group switches. This chunk fails if the new foundations
need a universal optional-array carrier, rank/node-count formulation inference,
mutable compiled arrays, runtime registry rebinding, or edits to current solver
behavior merely to express identity/specification.

## Work packets

### P0-A — Quarantine the prototype assembly module

Owned paths:

- `pyfem/v3/assembly.py` (rename only)
- `pyfem/v3/_prototype_assembly.py`
- existing `pyfem/v3/**/*.py` and `test/v3/**/*.py` import references only where
  required by the rename

Deliverable:

- rename the colliding module exactly as required by design section 8;
- preserve all current imports/behavior through explicit prototype imports; and
- make no architectural cleanup or correctness fixes.

Acceptance:

- v3 Ruff checks;
- `pytest -q test/v3` remains 130 passing tests; and
- repository search finds no live import of `pyfem.v3.assembly`.

Merge dependency: first. This opens the canonical `pyfem/v3/assembly/` namespace.

### P0-B — Authored and normalized model specification

Owned paths:

- `pyfem/v3/spec/**`
- `test/v3/test_v3_model_spec.py`

Deliverable:

- minimal immutable `MeshSpec`, explicit cell-block/topology metadata,
  `FieldSpec`, `MaterialSpec`, `RegionSpec`, and `ModelSpec` contracts;
- normalization/validation that diagnoses duplicate IDs, bad connectivity, invalid
  field/material/region references, and topology/embedding collisions with source
  context; and
- no global DOF allocation, compiled arrays, kernel dispatch, or file parsing.

Acceptance:

- a 3D Tet4 volume and four-node Quad4 surface remain distinct by explicit
  topology/topological dimension;
- mixed Q4/T3 declarations require no padded connectivity;
- caller mutation cannot silently change an already normalized immutable spec; and
- focused tests plus v3 Ruff pass.

Merge dependency: independent of P0-C; integrate after P0-A to minimize path churn.

### P0-C — Compiler-owned identity and array primitives

Owned paths:

- `pyfem/v3/model/**`
- `test/v3/test_v3_model_identity.py`

Deliverable:

- live instance identity, deterministic content-fingerprint primitives, and state
  generation identity with clearly distinct semantics;
- an owned/contiguous/read-only NumPy finalization helper that never aliases caller
  storage;
- an immutable registry-manifest/snapshot primitive whose recorded meaning cannot
  change by later source mapping mutation; and
- identity-equality array carriers (no generated elementwise equality).

Acceptance:

- mutate every caller input after finalization and prove compiled storage unchanged;
- write attempts fail;
- equivalent canonical manifests fingerprint identically, semantic changes do not;
- live instance mismatch fails even when fingerprints match; and
- focused tests plus v3 Ruff pass.

Merge dependency: independent of P0-B; both feed a later compiler-integration
packet. Do not invent `CompiledModel` before P0-B is available.

### P0-E — Caller-detached canonical model specification

State: source `246114ba32b8367675f2b3954e67f9e3847ca235` reviewed and
integrated as `108552ddd3382163a0e15c2fef7ca34e75f974fd`; integration gates
are green and a fresh independent edge-case review is required. The task was dispatched from
exact base `47e94752e93b7424c73e4d2979bebbe0f7567291`. This is a
post-integration repair, not a rewrite of P0-B history.

Outcome and ownership invariant:

- successful normalization constructs a new, exact, recursively caller-detached
  canonical `ModelSpec` tree;
- every dataclass slot, container, scalar, and `SourceContext` is validated before
  unsafe iteration, lookup, comparison, formatting, or child access; and
- every malformed authored value produces deterministic `ModelSpecValidationError`
  diagnostics anchored at the nearest trusted source, never raw Python exceptions.

Accepted R0-C blockers to repair:

1. `normalize_model_spec()` returns the caller tree, so manually altered mutable slots and
   source values remain aliased.
2. `str`/`int` subclasses can retain polymorphic hash, comparison, stripping, and
   rendering behavior or escape as raw exceptions.
3. exact-but-manually-constructed dataclasses and wrong internal containers are trusted before
   their slots and collection shapes are validated.

Owned paths:

- `pyfem/v3/spec/**`
- `test/v3/test_v3_model_spec.py`

Forbidden paths and non-goals:

- all other production/tests and all `.agents/v3/**` documents;
- compiler, registry compatibility, DOF allocation, section/program binding,
  public exports, file parsing, solvers, compatibility shims, and dependency changes;
- weakening exact canonical type policy or treating deliberate raw exceptions as
  acceptable failure behavior.

Required edge cases and evidence:

- normalized result and every nested spec/source/value/container are newly owned
  exact canonical values; mutating manually altered caller slots afterward cannot affect it;
- mutable-hash, unhashable, raising-repr `str` subclasses and raising-comparison `int`
  subclasses reject deterministically without invoking caller-defined behavior;
- exact uninitialized or wrong-slot `ModelSpec`, `MeshSpec`, `NodeSpec`, and
  `SourceContext`, plus non-iterable/list-valued collections, never escape raw
  `AttributeError`, `TypeError`, or rendering failures;
- ordinary multi-defect ordering, 0D topology, block-local identity, differing
  uniform block arities, caller detachment, and all earlier P0-B repairs remain;
- focused spec tests, both v3 Ruff configurations, focused format, `pytest -q
  test/v3`, full `pytest -q`, `git diff --check`, and a clean worker worktree pass.

Merge order: completed independently of the identity reviewer. If R0-I or R0-H
requires another owner repair, integrate disjoint repairs serially and re-run the
combined foundation gates before any compiler packet.

### R0-H — Independently check the repaired canonical model boundary

State: `COMPLETE` in task `019f711d-c79c-7652-96dc-f07e55fdb71b`, reviewed
read-only from exact base `13c68e302cf8f4e0f7e211f8af46eff09c863368`.
One blocker is accepted: exact integers beyond Python's decimal conversion limit
normalize successfully but duplicate/reference/source diagnostics interpolate them
with ordinary decimal rendering and leak raw `ValueError`.

Review lens:

- prove the returned tree is recursively exact, newly owned, and detached from
  caller-side mutation at every nested spec/source/value/container boundary;
- check every spec family with exact uninitialized objects, manually set slots, wrong
  collection shapes, container subclasses, cycles, deep values, and malformed
  `SourceContext` values;
- prove custom scalar subclasses cannot execute polymorphic hash, representation,
  comparison, or string-normalization behavior and every failure remains a
  deterministic `ModelSpecValidationError` at the nearest trusted source;
- preserve ordinary multi-defect ordering, 0D topology, block-local identity,
  differing uniform cross-block arities, and the absence of compiler/program/
  section policy in the authored boundary; and
- scrutinize the second defensive reconstruction, but report it as a blocker only
  if evidence shows a correctness failure or a required foundation-scale violation.

Owned output: terminal review report and reproducible local edge-case checks only. No file
edits, commits, merge, push, API expansion, or compiler work.

Evidence and adjudication:

- 35 focused spec tests passed in 0.76 seconds;
- all 11 exact spec/source families, 27 custom scalar subclasses, 13 custom
  collection slots, depth 5,000, cycles, deterministic fallback ordering, and a
  complete post-normalization mutation sweep survived;
- a valid 5,001-digit exact-integer ID normalized, while duplicate-ID,
  invalid-node, and huge `SourceContext.line` diagnostics leaked raw `ValueError`
  at the default 4,300-digit conversion limit; and
- double reconstruction measured 1.251 seconds and 11.36 MiB peak, 2.37 times
  retained memory, for 20,000 nodes, 5,000 cells, and 5,000 references. This remains
  a nonblocking later measurement obligation.

### P0-F — Total bounded exact-integer diagnostics

State: `COMPLETE` in task `019f7136-c2f1-77b2-81bb-f9aa41627a92`,
source `c0b3c578e6641cad428244065221170136534b51` from exact base
`f9e1867a2c458531b61fd3d8e5107445c0229733`, reviewed and integrated as
`461a8a85de622820eb627a24b8e53749489d020a`.

Outcome and invariant:

- every exact integer already accepted as a spec ID, dimension, parameter, or
  source line/column renders deterministically and with bounded output in every
  normalization diagnostic;
- large integers remain valid authored IDs where the existing contract permits
  them; the repair must not impose an arbitrary ID limit or mutate Python's global
  integer-string conversion setting; and
- malformed or conflicting models still fail only as source-anchored
  `ModelSpecValidationError`, never a raw decimal-conversion exception.

Owned paths:

- `pyfem/v3/spec/diagnostics.py`
- `pyfem/v3/spec/normalize.py`
- `test/v3/test_v3_model_spec.py`

Required edge cases and evidence:

- duplicate 5,001-digit node/block/cell/field/material/region IDs use one bounded,
  stable representation and preserve diagnostic order;
- unknown/duplicate huge node, cell, block, field, and material references never
  invoke unbounded `repr`/`str` conversion;
- huge exact `SourceContext.line` and `.column` values render safely when another
  defect is reported, including nearest-trusted-source fallback;
- ordinary small integer/string diagnostic text, 0D topology, block-local identity,
  cross-block arity policy, and all P0-E custom-behavior/detachment cases remain unchanged;
- focused spec tests, both v3 Ruff configurations, focused format, `pytest -q
  test/v3`, full `pytest -q`, `git diff --check`, and a clean worker worktree pass.

Forbidden: new ID-range policy, global interpreter-setting changes, compiler/
program/section work, double-reconstruction optimization, public exports,
dependency/config changes, shared-document edits, merge, or push.

### R0-I — Independently check identity, provenance, registry, and storage repairs

State: `COMPLETE` in task `019f711f-339b-7300-aa1c-17e6e7ea9974`, reviewed
read-only from exact base `13c68e302cf8f4e0f7e211f8af46eff09c863368`.

Accepted foundation blockers:

1. semantic NumPy scalar subclasses execute conversion methods during manifest
   capture, and nested semantic ndarray subclasses such as `MaskedArray` lose their
   meaning when hidden inside a list passed to `finalize_array`;
2. dtype metadata is omitted from fingerprints and its nested mutable objects remain
   caller-aliased across finalizations;
3. `FinalizedArray` comparison becomes elementwise under standard NumPy reverse/
   ufunc dispatch, and a public carrier subclass can imitate base-carrier equality;
4. canonical carriers plus mapping/list subclasses retain polymorphic execution
   paths, `ContentFingerprint` subclasses imitate equality, and invalid `id_key`
   values can escape as raw exceptions; and
5. `RegistrySnapshot` accepts and retains descriptor/key/source subclasses, allowing
   exposed key/binding meaning to drift under a stable snapshot fingerprint.

Evidence and exclusions:

- 18 focused identity tests passed in 2.06 seconds; independent local edge-case checks
  reproduced all five blockers under NumPy 2.3.5; ordinary exact scalar/array,
  endian/stride, cycle, detachment, and registry-rebinding cases held;
- exact manually constructed identity fields required `object.__new__` plus deliberate
  `object.__setattr__` and are not a current primitive blocker; restore/rebind must
  validate exact `UUID`/integer fields before comparing externally reconstructed
  identity values;
- arbitrary unrelated left-operand equality and deliberate re-enabling of an owning
  NumPy array's write flag remain outside the carrier's accidental-mutation contract;
  and
- no depth-probe output was obtained, so R0-I makes no deep-acyclic-manifest claim.

### P0-G — Close polymorphic identity/storage semantic drift

State: `COMPLETE` in task `019f7146-e871-76c2-a3e6-75d87fac83fd` from exact
base `7d1ededb6d11fc183ff506b989b6d1f72fd12600`; source
`95044678c54a5d9f5141bed504d160fd1bfbbd33` and its direct child
`8612d0140dc0cb7e68bc1bc61d3947acf3f81de0` were reviewed and integrated as
`e04f86a67e5fa225f6191b7ef16ca5572598cbb9` followed by
`afaac4d189979c861fb0463e6aa9c07bc1bc4ed5`.

Outcome and invariant:

- only exact supported NumPy scalar/array and builtin container forms cross generic
  canonicalization/finalization boundaries; semantic subclasses fail before any
  user conversion, iteration, mapping, array, or ufunc behavior executes;
- every accepted dtype is fully represented by existing manifest v1 semantics and
  detached by finalization. Any dtype metadata at any nested dtype level rejects
  deterministically rather than colliding or aliasing;
- `FinalizedArray` is runtime-final and has scalar identity-only equality/hash
  semantics for both orders of standard ndarray and NumPy equal/not-equal dispatch;
- canonical manifest/fingerprint/unordered-declaration carriers are runtime-final,
  consumed only as exact trusted values, and exact builtin mapping/sequence policy
  cannot execute subclass methods; and
- registry snapshots accept exact keys, sources, and runtime-final exact descriptors
  only, so later source mutation cannot alter exposed key/binding meaning under a
  stable fingerprint.

Owned paths:

- `pyfem/v3/model/arrays.py`
- `pyfem/v3/model/provenance.py`
- `pyfem/v3/model/registry.py`
- `test/v3/test_v3_model_identity.py`

Required edge cases and evidence:

- NumPy integer/float subclasses with raising conversion methods reject at every nested
  manifest position, while exact `float16`/`float32`/`float64` and integers retain
  the documented lossless behavior;
- top-level and list/tuple-nested `MaskedArray`/custom ndarray subclasses reject
  before `np.array(..., subok=False)` can erase semantics;
- arrays/dtypes with metadata, including metadata nested in structured/subdtypes,
  reject in both provenance and finalization without changing the manifest-v1 tag;
- plain ndarray endian/stride/order/value ownership and metadata-free structured
  finalization, if retained, remain detached and exact;
- `FinalizedArray` subclass creation fails; standard ndarray equality/inequality and
  NumPy ufunc comparison return scalar identity booleans in either order; hashes
  remain object identity;
- canonical-carrier, mapping, list, fingerprint, descriptor, source-mapping, and
  tuple-key subclasses reject without executing overrides; `id_key` is an exact
  nonempty string before lookup; ordinary cycles still reject deterministically;
- ordinary registry source clearing/rebinding/nested metadata mutation remains
  detached, while a descriptor cannot change exposed key/binding meaning; and
- focused identity tests, both v3 Ruff configurations, focused format, `pytest -q
  test/v3`, full `pytest -q`, `git diff --check`, and a clean worker worktree pass.

Forbidden: manifest-v2/schema invention, dtype-metadata lowering, identity.py/
restore/transaction changes, callable introspection, arbitrary foreign equality
control, deliberate write-flag hardening, compiler/program/solver/public API work,
shared-document/config/dependency edits, merge, or push.

Integration audit follow-up:

- a normal 5,001-digit exact Python integer leaks the interpreter's raw decimal
  conversion-limit `ValueError`, although exact integers are documented supported;
- manually constructed exact float carrier text can leak `OverflowError` from `float.fromhex`;
- a manually constructed exact v1 float-array payload containing `NaN` passes validation even
  though constructor capture rejects non-finite values; and
- a manually constructed exact unordered carrier can contain a non-mapping node without its
  declared ID key, unique identity, or canonical identity ordering.

The provenance/test-only follow-up repaired all four findings without changing
manifest v1. It additionally closes noncanonical base64/bool/endian payloads and
NumPy-unreachable rank, dimension, and zero-sized nonzero-product shapes. It is the
exact child `8612d0140dc0cb7e68bc1bc61d3947acf3f81de0` of the source commit and is
integrated directly after it.

### R0-L — Independently verify the identity/storage boundary

State: `COMPLETE` with a `NO-GO` verdict as task
`019f7551-0075-7383-a895-fd2d77f02419` from exact clean base
`dc9e588a50c21596dfc2b7f2a879c3a4dd31ee92`. Its own final response began with
the required terminal sentinel and its direct callback reached I0. This replaces
incomplete R0-K; it does not change R0-K's status.

Role and outcome:

- read-only Sol/max correctness reviewer for the local finite-element Python
  library;
- return one evidence-backed GO or NO-GO verdict on the current identity,
  provenance, array ownership, and registry snapshot primitives; and
- make no repository edits, commits, configuration changes, dependency changes,
  external-system access, compiler/program/solver work, or additional tasks.

Relevant paths:

- `pyfem/v3/model/arrays.py`
- `pyfem/v3/model/identity.py`
- `pyfem/v3/model/provenance.py`
- `pyfem/v3/model/registry.py`
- `test/v3/test_v3_model_identity.py`

Required local correctness checks:

1. Exact supported NumPy scalar, ndarray, dtype, list, tuple, and mapping forms are
   validated before custom subclass behavior can run. Unsupported semantic array
   subclasses, object arrays, and dtype metadata reject deterministically at every
   nested position.
2. Array finalization preserves value, dtype, endian meaning, requested C/F order,
   ownership, read-only item assignment, caller isolation, and isolation between
   separate finalizations for native, non-native, sliced, and structured inputs
   allowed by the current contract.
3. `FinalizedArray` remains runtime-final with scalar identity-only equality and
   hashing in both operand orders and through standard NumPy equal/not-equal calls;
   unsupported keyword/output forms reject without mutation.
4. Manifest v1 bytes and fingerprints are stable for ordinary values. Positive and
   negative 5,001-digit integers work at interpreter digit limits 4,300 and 640
   without changing global settings. Float text, base64, finite array payloads,
   endian tags, rank, dimensions, payload length, nonzero product/itemsize, and
   zero-sized boundaries are validated deterministically.
5. Reused exact manifest, unordered-declaration, fingerprint, descriptor, and
   snapshot values validate all required fields and canonical ordering. Missing or
   altered fields, duplicate/missing declaration IDs, cycles, and invalid nested
   nodes produce stable library exceptions rather than incidental Python errors.
6. Registry capture preserves exact key/descriptor meaning, detached metadata,
   selected binding references, required-key behavior, and source mutation
   isolation. Existing documented callable, transaction, and restore/rebind
   obligations remain later work unless a current primitive contract is directly
   violated.
7. Record the bounded depth reached by a deeply nested but acyclic manifest input.
   Treat externally reconstructed-data decoding as later work unless ordinary current use fails.

Required evidence and environment:

- first verify the exact dispatch base and an empty `git status --short`;
- use `/Users/sora/Projects/python/PyFEM/.venv/bin/python` and
  `/Users/sora/Projects/python/PyFEM/.venv/bin/ruff` with the review worktree as the
  current directory;
- task-specific scripts under `/private/tmp` are authorized when useful; do not
  modify the repository or install anything;
- run `test/v3/test_v3_model_identity.py` normally and with
  `PYTHONINTMAXSTRDIGITS=640`, both repository-defined Ruff configurations for the
  relevant paths, and bounded local edge-case checks; run `test/v3` if time permits;
- finish with a final response beginning `RESULT: COMPLETE` or `RESULT: BLOCKED`,
  then exact base, blocker count, command evidence, warnings/deferred items,
  worktree state, and smallest next action; and
- send the same compact result to I0. Commentary, idle state, or partial results do
  not complete this task.

Result and adjudication:

- 47 focused identity/storage tests passed normally and under
  `PYTHONINTMAXSTRDIGITS=640`; both Ruff configurations passed; the bounded local
  matrix passed 162 checks in each digit mode; and `test/v3` passed 219 tests with
  40 existing SciPy `SparseEfficiencyWarning` notices;
- acyclic manifest capture and `to_bytes()` passed through depth 247; depth 248
  returned the stable library malformed-carrier `TypeError`, which is recorded as a
  bounded later-phase decoding obligation rather than a current blocker;
- one P1 defect is accepted: `RegistrySnapshot.resolve()` trusts retained or reused
  snapshot/descriptor fields. Changing the caller-owned descriptor's selected
  binding changes the resolved binding while manifest and fingerprint stay stable,
  and missing snapshot or nested descriptor fields leak `AttributeError`;
- the integration owner reproduced all three manifestations at the current branch
  head. They are one root failure of the frozen-snapshot contract in `design.md`,
  not three separate repair packets; and
- callable implementation truth, transaction semantics, restore/rebind, and
  externally reconstructed-data decoding remain later-phase obligations.

### P0-H — Preserve registry snapshot meaning

State: `COMPLETE` as task `019f7568-a1e0-7c33-bf83-0a490e54c520` from exact clean
base `55fcf990da8f47f612f5193529c2a6b77a823a10`. Source commit
`0947dd7ca3b70bef2ebdf986f2336f62f5b3c08b` is exactly one two-path child and was
integrated without conflict as `fb358fc0609b81a12cee4a1a66c2dae98edf5cae`.
The task supplied both required terminal signals and left its worktree clean.

Outcome and ownership:

- repair the one accepted R0-L root defect without widening the registry's public
  persistence, restore, compiler, program, solver, dependency, or compatibility
  surface;
- compare at least two minimal internal representations and choose the smallest one
  that preserves independently captured descriptor meaning and exact selected
  callable references;
- validate reused exact descriptor/snapshot fields, canonical ordering/uniqueness,
  manifest/fingerprint consistency, and selected callable-reference consistency
  before resolution; and
- own only `pyfem/v3/model/registry.py` and
  `test/v3/test_v3_model_identity.py`, returning one focused commit.

Required proof:

- caller-side changes to the source descriptor's binding, key fields, or metadata
  carrier cannot change an existing snapshot;
- missing or altered snapshot/nested-descriptor fields, reordered/duplicate
  descriptors, and mismatched manifest/fingerprint fail with stable library
  exceptions;
- ordinary multi-descriptor capture, required keys, source clearing/rebinding,
  canonical ordering, and exact callable identity remain green; and
- focused identity tests pass normally and at the 640-digit interpreter limit,
  both Ruff configurations and focused format pass, `test/v3` and the full suite
  pass, `git diff --check` passes, and the committed worktree is clean.

Integrated result:

- one detached descriptor and detached canonical metadata/manifest carrier are
  captured per snapshot while the exact selected callable object is retained in a
  parallel private tuple;
- every resolution revalidates descriptor fields, canonical ordering/uniqueness,
  snapshot manifest/fingerprint, and selected callable identity before returning;
- both writer and I0 integration runs passed 68 focused tests in each digit mode,
  both Ruff configurations, focused format, 240 v3 tests, 429 full-repository
  tests, and `git diff --check`; and
- I0 independently replayed source descriptor key/binding/metadata changes and all
  four missing snapshot fields successfully. The cold integration v3 run emitted
  four existing Numba performance notices plus 40 existing SciPy warnings; the
  warm full run emitted only the 40 SciPy warnings.

### R0-M — Independently recheck captured registry meaning

State: `DISPATCHED` as read-only task `019f757a-e84f-70c3-a6d6-8c4ff7874204` from
exact clean integrated base `fb358fc0609b81a12cee4a1a66c2dae98edf5cae`.

Outcome and boundary:

- return one GO/NO-GO decision only on the repaired frozen-registry invariant;
- replay source descriptor/mapping changes, missing snapshot/nested descriptor
  fields, descriptor ordering/duplicates, snapshot manifest/fingerprint changes,
  selected callable-reference changes, required-key selection, and ordinary stable
  resolution;
- run focused identity tests in both digit modes, both Ruff configurations, and
  `test/v3` if time permits; and
- remain read-only with no repository changes, commits, dependencies, additional
  tasks, or compiler/program/solver work. Completion requires both terminal signals.

Result:

- state is `COMPLETE` with a `GO` verdict, zero blockers, a valid sentinel-bearing
  final response, and a direct callback to I0;
- 42 of 42 independent local correctness checks passed, covering caller/source
  detachment, every required missing field, altered fields/counts, canonical
  ordering/uniqueness, snapshot manifest/fingerprint, selected callable identity,
  required-key selection, source rebinding/clearing, and stable descriptor identity;
- 68 focused tests passed in both interpreter digit modes, both Ruff configurations
  passed, and the v3 suite passed 240 tests with 40 existing SciPy warnings and four
  existing Numba performance notices; and
- the first task turn ended with an app-level output-capture `systemError` while the
  v3 process was still running. The single allowed resume recovered that same
  process at exit code zero, reconfirmed exact HEAD, empty porcelain status, and
  `git diff --check`, then supplied both terminal signals. No product gate is
  missing and no replacement task was created.

### R0-E — Seed the complete legacy capability inventory

State: `COMPLETE` as read-only task
`019f7587-31f2-7642-b162-5fd5b7a2d07b` from exact clean foundation-closure commit
`c50ca70bff884157c5645dde276c25acc6672d4a`. It returned both the required
sentinel-bearing final and direct I0 callback. The exact report is now preserved as
[dated evidence](evidence/2026-07-18-r0e-capability-inventory.md) with SHA-256
`f9e326da3adc8e6bb3a777fb4134965ade6f9b364df352c53e68c2cbb7e377cf`.

Outcome and boundary:

- create one complete E0 report at
  `/private/tmp/pyfem-r0e-capability-inventory.md` without modifying the repository;
- scan and map public entry points, packaging, loaders, mesh/DOF/constraint/model/
  assembly/solver/element/material/section/writer surfaces, every legacy test and
  checked-in `.pro` example, every parity skim and v3 test, public docs/notebooks,
  GUI, RVE/FE2, ROM, contact, multiphysics, structural, dynamics/eigen,
  nonlinear/path-following, ecosystem I/O, and prototype duplicate owners;
- provide every required capability-ledger field, stable semantic IDs, proposed
  preserve/change/retire dispositions, exact evidence pointers, P0-D dependencies,
  coverage mappings, and an explicit unclassified-remnants count; and
- keep public retirement and API/GUI/RVE-FE2/ROM choices visibly proposed until I0
  and the user adjudicate them. Code/example existence is E0, not proof of behavior.

Completion requires a required-field/ID/disposition/owner/status/grade/coverage
audit, exact HEAD and clean-status recheck, a sentinel-bearing final response, and a
direct I0 callback. R0-E has no repository file ownership and cannot create tasks or
design/implement compiler code.

Completion evidence: 154 valid E0 rows; 606/606 tracked paths, 122/122 `.pro`
files, 33/33 skim paths, and 41/41 test/support paths classified; zero duplicate
IDs, missing required fields, undefined dependency IDs, unclassified remnants, or
coordinator-terminology matches; 22 historical-manual pages inspected; exact HEAD,
empty porcelain, and `git diff --check` green. I0 independently reproduced the row,
schema, dependency, vocabulary, manifest-hash, `.pro`, skim, and test-set checks.

## Merge and continuation order

1. Completed: review and integrate P0-A.
2. Completed: review P0-B and P0-C independently against the Horizon Gate.
3. Completed: integrate P0-B/P0-C, repair the six accepted original reviewer
   review findings, and rerun combined v3 plus full repository tests.
4. Completed: P0-F repaired the R0-H diagnostic blocker; P0-G plus its direct
   provenance follow-up repaired the five R0-I seams and integration-audit findings.
5. Completed: R0-L returned a valid terminal result and callback. Its one accepted
   frozen-registry defect is now the only active foundation repair. R0-J and R0-K
   remain incomplete; their statuses do not change.
6. Completed: P0-H repaired the R0-L defect, I0 reviewed/integrated its single
   commit, and all combined focused/static/v3/full gates passed.
7. Completed: R0-M independently returned GO with zero blockers at the integrated
   commit after one bounded resume recovered the already-running v3 result.
8. Completed: R0-E produced the complete E0 legacy capability inventory; I0
   preserved its exact report, independently checked it, and ingested the live state
   overlay into this ledger.
9. Active: the dependent compiler-integration packet is dispatched from the frozen
   Horizon: one explicit Q8 region ->
   immutable `CompiledModel` recipe with entity/source maps and empty physical-state
   layout.
10. Only after P0-D and its `R0-F` reviewer dispatch `ProgramSpec`/affine constraints and
   `PreparedAssemblyPlan` work.

## Packet ledger

Titles follow the compact coordinate/owner/outcome convention in
[refactor_playbook.md](refactor_playbook.md#thread-titles). Status and execution
metadata stay here rather than being encoded in the title.

Completion signalling: each task must produce both its own terminal-result final and
a direct callback to `I0` (or explicitly report callback unavailability). Native
idle/completed status and commentary are insufficient. No polling automation is
active; a watchdog is unnecessary while callbacks and native status are available.

| Exact title | Thread | Exact base | Owner/output | State and evidence |
|---|---|---|---|---|
| `I0 · foundations — combined proof` | `019f6f49-0b72-7d73-86da-c6b85519eeaf` | `c75cbf3523349deb40bd2b07de7959e7606c3b1f` | Integration, shared docs, combined proof | Complete GO through integrated repair `fb358fc`; coordinator continues to next milestone |
| `P0-A · assembly — prototype quarantined` | `019f7060-0bb3-7a72-bb6b-47697f1c5747` | `c75cbf3523349deb40bd2b07de7959e7606c3b1f` | Assembly quarantine | Integrated as `92bc87d` from `db486f5`; repair 0 |
| `P0-B · model spec — explicit immutable intent` | `019f7060-0bb9-7b40-8cfb-f056155afe37` | `c75cbf3523349deb40bd2b07de7959e7606c3b1f` | `pyfem/v3/spec/**` | Integrated through `f2a0cd2`; two integration repairs plus one reviewer repair |
| `P0-C · identity/storage — owned and frozen` | `019f7060-0bb1-7a72-b438-5c2274f3d5e8` | `c75cbf3523349deb40bd2b07de7959e7606c3b1f` | `pyfem/v3/model/**` | Integrated through `e1d7fe7`; one integration repair plus one reviewer repair |
| `R0-A · model spec — semantic gaps identified` | `019f7087-61f2-78a2-9df7-5174dbc5b8a3` | `2bda241719e2c236abe4711528abd82f47b4633d` | Original spec reviewer | Complete; 3 accepted blockers repaired |
| `R0-B · identity/storage — invariant gaps identified` | `019f7087-61f0-71e0-9082-122e7ea75894` | `2bda241719e2c236abe4711528abd82f47b4633d` | Original identity/storage reviewer | Complete; 3 accepted blockers repaired |
| `D0-A · migration workflow — autonomy bounded` | `019f708b-2980-7703-8fca-7ea26d5826ba` | `ad95149e2e34b8eff55c0896c1dea53ac1cbc71d` | `migration_workflow.md` | Complete; source `50cc663`, integrated `9b26574`, adopted |
| `INCOMPLETE R0-C · model spec — superseded` | `019f70a3-2650-7181-8a05-fc2b72b111a5` | `faab0c938705f59fc5a22e702413f295af1dcadb` | Initial read-only spec reviewer | Incomplete; unrelated policy reframing interrupted the work; no final result or callback |
| `R0-C · model spec — repairs independently checked` | `019f70af-2c98-7563-b303-0a5a66fd6ef5` | `faab0c938705f59fc5a22e702413f295af1dcadb` | Replacement read-only spec reviewer | Complete; 3 accepted blockers; 23 focused tests passed |
| `INCOMPLETE R0-D · identity/storage — no verdict` | `019f70a3-264c-7831-8509-a3ffbf9235f4` | `faab0c938705f59fc5a22e702413f295af1dcadb` | Initial identity/storage reviewer | Incomplete; partial focused evidence only; no final verdict or callback |
| `INCOMPLETE R0-G · identity/storage — superseded` | `019f70b1-793c-7c90-a051-07911fff3134` | `faab0c938705f59fc5a22e702413f295af1dcadb` | First R0-D replacement | Incomplete after repeated turns; no final verdict or callback; later replaced by completed R0-I |
| `P0-E · model spec — canonical tree owned` | `019f70b8-3e74-7d82-b27b-67991cf50e3c` | `47e94752e93b7424c73e4d2979bebbe0f7567291` | `pyfem/v3/spec/**`, focused spec tests | Complete; source `246114b`, integrated `108552d`; 35 focused, 183 v3, 372 full tests passed |
| `R0-H · model spec — boundary independently checked` | `019f711d-c79c-7652-96dc-f07e55fdb71b` | `13c68e302cf8f4e0f7e211f8af46eff09c863368` | Fresh read-only P0-E reviewer | Complete; 1 accepted exact-integer rendering blocker; 35 focused passed |
| `R0-I · identity/storage — repairs independently checked` | `019f711f-339b-7300-aa1c-17e6e7ea9974` | `13c68e302cf8f4e0f7e211f8af46eff09c863368` | Exact-scope R0-G replacement with independent reproduction | Complete; 5 accepted blockers; 18 focused passed |
| `P0-F · model spec — integer diagnostics total` | `019f7136-c2f1-77b2-81bb-f9aa41627a92` | `f9e1867a2c458531b61fd3d8e5107445c0229733` | Bounded exact-integer diagnostic repair | Complete; source `c0b3c57`, integrated `461a8a8`; 42 focused, 190 v3, 379 full tests passed |
| `P0-G · identity/storage — exact boundaries enforced` | `019f7146-e871-76c2-a3e6-75d87fac83fd` | `7d1ededb6d11fc183ff506b989b6d1f72fd12600` | Close five accepted R0-I seams plus integration-audit carrier gaps | Complete; source `9504467` + child `8612d01`, integrated `e04f86a` + `afaac4d`; 47 focused, 212 v3, 401 full passed |
| `INCOMPLETE R0-J · model spec — evidence only` | `019f7184-2fe6-7403-a5fd-683e58dcee79` | `8a952e7668f4f6c52d39352dee7e0685593906a3` | Read-only spec/diagnostic repeat reviewer | Incomplete; substantial zero-blocker evidence recovered, but no final result or callback; evidence is supporting only |
| `INCOMPLETE R0-K · identity/storage — checks not run` | `019f7184-2fe6-7403-a5fd-685f9ee1995d` | `8a952e7668f4f6c52d39352dee7e0685593906a3` | Read-only identity/provenance/registry/storage repeat reviewer | Incomplete; temporary script created but never executed; no required tests, static gates, verdict, or callback |
| `R0-L · identity/storage — boundary independently verified` | `019f7551-0075-7383-a895-fd2d77f02419` | `dc9e588a50c21596dfc2b7f2a879c3a4dd31ee92` | Read-only identity/provenance/registry/storage replacement reviewer | Complete, valid final and callback; NO-GO with one accepted frozen-registry defect; 47 focused in both digit modes, 162 local checks per mode, 219 v3 tests, both Ruff gates |
| `P0-H · registry snapshot — meaning detached` | `019f7568-a1e0-7c33-bf83-0a490e54c520` | `55fcf990da8f47f612f5193529c2a6b77a823a10` | `registry.py` plus focused identity tests | Complete; source `0947dd7`, integrated `fb358fc`; 68 focused twice, 240 v3, 429 full, static gates green |
| `R0-M · registry snapshot — capture invariant verified` | `019f757a-e84f-70c3-a6d6-8c4ff7874204` | `fb358fc0609b81a12cee4a1a66c2dae98edf5cae` | Read-only P0-H integrated recheck | Complete GO, zero blockers; 42 local checks, 68 focused twice, both Ruff gates, 240 v3; one app-level resume, both terminal signals valid |
| `R0-E · legacy breadth — semantic ledger seeded` | `019f7587-31f2-7642-b162-5fd5b7a2d07b` | `c50ca70bff884157c5645dde276c25acc6672d4a` | Complete read-only E0 capability inventory report | Complete with both terminal signals; 154 rows, 606/606 paths, zero remnants; durable report SHA `f9e326d`; independently checked and ingested |
| `P0-D · model compiler — Q8 block frozen` | `019f75dd-f597-7c21-adfc-d78b1e2580c0` | `2842ef84b86f04f82587f5fc378584f30f6b62bd` | `pyfem/v3/compile/**`, new compiled-model carriers, one focused compiler test | Dispatched to Sol/max from frozen Horizon; isolated worktree preflight active; no callback, commit, or integration claim yet |

## Task completion audit

The 2026-07-18 audit inspected the actual final turns of the first 17 user-visible
migration tasks rather than relying on titles, idle state, or ledger summaries.
R0-L, P0-H, R0-M, and R0-E subsequently completed under the corrected terminal
contract. With P0-D dispatched, the record is 22 tasks: 16 properly complete, five
explicitly incomplete, and one active bounded writer.

- Properly completed: P0-A, P0-B, P0-C, R0-A, R0-B, D0-A, replacement R0-C,
  P0-E, R0-H, R0-I, P0-F, P0-G, R0-L, P0-H, R0-M, and R0-E.
- Incomplete: initial R0-C, R0-D, R0-G, R0-J, and R0-K.
- Active: P0-D on the frozen Q8 compiler Horizon.
- Completed replacements provide valid evidence for their own task IDs; they do not
  change the recorded status of the tasks they replaced.
- No incomplete writer commit was integrated. The current risk is review/process
  integrity, not contamination of the integrated foundation code.

## Active watchdogs

None. Direct terminal callbacks and bounded native waits are active; unchanged
thread state does not trigger a model-consuming polling loop.

## Required combined evidence

- focused tests owned by every packet;
- both v3 Ruff configurations;
- `pytest -q test/v3` after each merge;
- full `pytest -q` after the first combined foundation; and
- disconfirming review of identity, caller aliasing, explicit topology, and
  prototype-import quarantine before the integration packet begins.

Current combined evidence at
`108552ddd3382163a0e15c2fef7ca34e75f974fd`:

- 35 focused P0-E model-spec tests passed;
- 183 v3 tests passed with 40 pre-existing SciPy warnings and four existing
  cold-cache Numba performance warnings;
- 372 full-repository tests passed with the same 44 warnings;
- both v3 Ruff gates, focused format, and `git diff --check` passed; and
- all accepted R0-A/R0-B/R0-C blockers are repaired under implementation gates;
  R0-H accepted one spec-diagnostic blocker for P0-F, and R0-I accepted five
  identity/storage blockers for P0-G before compiler integration.

P0-F integration evidence at
`461a8a85de622820eb627a24b8e53749489d020a`:

- integration review confirmed the exact declared parent and three-path ownership;
- 42 focused model-spec tests and 190 v3 tests passed, with only 40 pre-existing
  SciPy `SparseEfficiencyWarning` notices;
- both v3 Ruff gates, focused format, and `git diff --check` passed; and
- the worker's exact source commit additionally passed seven selected tests under
  a 640-digit interpreter limit and the full 379-test repository suite.

P0-G source and follow-up evidence before integration:

- exact source ancestry/four-path ownership and exact child ancestry/two-path
  ownership from the declared bases pass;
- the final child passes 47 focused identity tests under the default 4,300-digit
  limit, 212 v3 tests, and 401 full-repository tests;
- both Ruff gates, focused format, committed `git diff --check`, and the worker's
  clean worktree pass; and
- the integration-owner replay accepts a 5,001-digit exact integer without changing
  the interpreter limit and deterministically rejects overflow float, non-finite
  array, malformed unordered, and NumPy-impossible shape payloads.

Combined foundation evidence at
`afaac4d189979c861fb0463e6aa9c07bc1bc4ed5`:

- 89 focused spec/identity tests pass both normally and under the 640-digit
  interpreter limit;
- 219 v3 tests and 408 full-repository tests pass with only 40 pre-existing SciPy
  `SparseEfficiencyWarning` notices;
- both v3 Ruff gates, combined focused format, and `git diff --check` pass with a
  clean integration worktree; and
- the old `pyfem.v3.assembly` Python-reference scan has zero matches.

R0-L independent review evidence at exact dispatch base
`dc9e588a50c21596dfc2b7f2a879c3a4dd31ee92`:

- the task satisfied the corrected terminal contract with both a sentinel-bearing
  final response and a direct callback;
- 47 focused tests passed in both interpreter digit-limit modes, 162 bounded local
  checks passed per mode, both Ruff gates passed, and the 219-test v3 suite passed
  with only 40 existing SciPy warnings;
- the review accepted one frozen-registry P1 defect, reproduced by the integration
  owner: retained descriptor binding meaning can change under a stable snapshot
  manifest/fingerprint, while missing snapshot or nested descriptor fields leak raw
  `AttributeError`; and
- all other required identity, provenance, array, and registry checks passed. The
  compiler gate remains closed until the accepted defect is repaired and rechecked.

P0-H integration evidence at
`fb358fc0609b81a12cee4a1a66c2dae98edf5cae`:

- source commit `0947dd7ca3b70bef2ebdf986f2336f62f5b3c08b` has the exact declared parent,
  one-commit ancestry, two owned paths, clean committed worktree, and no scope
  expansion;
- the repair detaches descriptor and canonical carriers from caller-owned values,
  preserves exact selected callable objects, and validates the complete snapshot
  meaning before every resolution;
- I0 replayed the source-change and missing-snapshot-field cases after reviewing the
  committed diff, then integrated it without conflict; and
- the integrated branch passes 68 focused tests in both digit modes, both Ruff
  configurations, focused format, 240 v3 tests, 429 full tests, and whitespace
  checks. Only the 40 existing SciPy warnings plus four existing cold-cache Numba
  notices appeared.

R0-M independent closure evidence at exact integrated base
`fb358fc0609b81a12cee4a1a66c2dae98edf5cae`:

- 42 of 42 local correctness checks reproduced the original R0-L failure shapes
  and adjacent invalid-order, duplicate, manifest, fingerprint, binding-reference,
  required-key, source-rebinding, and stable-identity cases with zero blockers;
- 68 focused tests passed normally and under the 640-digit interpreter limit, both
  Ruff configurations passed, and the v3 suite passed 240 tests;
- exact HEAD, empty porcelain status, and `git diff --check` were reconfirmed after
  the one app-level output-capture interruption; the same running v3 process was
  recovered with exit code zero rather than duplicated; and
- R0-M returned a valid GO final plus callback. Combined with P0-H's writer and I0
  integration evidence, this closes the foundation gate without changing any
  recorded later-phase callable, transaction, or restore/rebind obligation.

R0-E inventory evidence at exact audit base
`c50ca70bff884157c5645dde276c25acc6672d4a`:

- both terminal signals are valid, and the durable report hash is
  `f9e326da3adc8e6bb3a777fb4134965ade6f9b364df352c53e68c2cbb7e377cf`;
- 154 JSONL rows contain all 15 required fields plus domain, with 52 preserve, 89
  change, and 13 retire working dispositions, ten valid semantic owners, E0 grade,
  inventoried source state, and the exact proof commit;
- 606/606 tracked paths, 118/118 legacy paths, 51/51 v3 paths, 218/218 examples,
  26/26 exercises, 85/85 docs, 33/33 skim paths, 122/122 `.pro` files, 100/100
  `.dat` files, and 41/41 test/support paths are classified with zero remnants;
- duplicate IDs, missing/blank required cells, invalid dispositions/owners/status/
  grades, unknown dependency IDs, missing local evidence paths, and coordinator
  terminology matches are all zero; and
- I0 independently reproduced the JSON/schema, tracked-manifest hash, dependency,
  `.pro` brace expansion, skim selector, test-path, and vocabulary checks. This is
  source-discovery evidence only; no behavior, compatibility, or retirement was
  proved.

Recovered R0-J supporting evidence at exact frozen ledger head
`8a952e7668f4f6c52d39352dee7e0685593906a3`:

- the complete exact-class/slot/container, custom-hook, depth/cycle, topology,
  detachment, source-context, and restrictive-digit probe matrix found zero
  blockers across all 11 spec/source families;
- all positive and negative 5,001-digit cases remained exact or failed with bounded
  deterministic diagnostics under the 640-digit interpreter limit without changing
  that global limit;
- 42 focused model-spec tests passed both normally and at the 640-digit limit, both
  Ruff configurations passed, and the cold-cache broad replay passed 219 v3 tests
  with 40 existing SciPy plus four existing Numba warnings; and
- two repeated 20,000-node / 5,000-cell / 5,000-reference samples retained full
  detachment at median 1.410 seconds, 11.36 MiB peak, and 2.50 times retained memory.
  This supports the recorded double-reconstruction obligation without crossing its
  blocker rule. Two task turns ended before a final result and callback. Under the
  corrected workflow this evidence is retained, but the R0-J task remains
  incomplete and cannot itself advance the packet state.

## Accepted later-phase obligations

- Registry snapshot fingerprints trust callable purity and truthful, behavior-bound
  `implementation_id` values. Compiler/restore work must preserve or strengthen
  that trust boundary; a mutable callable can otherwise drift behind a stable
  descriptor manifest.
- Repeated `base.next_accepted()` calls can identify sibling prospective
  generations equally. The future commit transaction must prevent multiple
  accepted transitions from one base and separately identify trial candidates;
  the primitive alone is not transaction enforcement.
- P0-E's preflight already constructs a detached canonical tree and
  `_reconstruct_model` defensively copies it again. R0-H measured 1.251 seconds and
  11.36 MiB peak for 20,000 nodes, 5,000 cells, and 5,000 references, with peak
  memory 2.37 times retained; R0-J repeated two fully detached samples at median
  1.410 seconds, the same 11.36 MiB peak, and 2.50 times retained. This preserves the
  ownership contract and is not a foundation blocker; remeasure at compiler-scale
  proof before optimizing, and retain caller-detachment edge cases through any change.
- Exact manually constructed `InstanceId`/`StateGeneration` fields can bypass helpers or raise raw
  exceptions after deliberate `object.__new__`/`object.__setattr__` construction.
  This is not an authored/live primitive blocker; future restore/rebind decoding
  must validate exact `UUID`/integer fields before invoking identity comparisons.
- Deep acyclic manifest behavior is now bounded by R0-L: capture and `to_bytes()`
  succeed through depth 247, while depth 248 returns the stable malformed-carrier
  `TypeError`. Iterative decoding or a larger explicit bound remains a later-phase
  restore decision; do not widen the current registry repair for it.

## Blocked condition

Stop a packet when its owned contract cannot be implemented without deciding an
API or invariant owned by another packet, editing a forbidden shared path, or
changing prototype behavior. Record the exact dependency and smallest required
integration decision; do not bridge it with a compatibility carrier.

## Milestone log

- 2026-07-17: architecture reset committed; FVM refactor playbook adapted; first
  three disjoint work packets defined for GPT-5.6 Sol with max reasoning.
- 2026-07-17: P0-A/P0-B/P0-C dispatched from `c75cbf3` into isolated worktrees;
  initial snapshots confirmed clean base/instruction reading and active execution.
- 2026-07-17: adopted stable `<ledger-id> · <semantic owner> — <target state>`
  thread titles and renamed the active first-chunk packets without changing scope.
- 2026-07-17: installed direct terminal callbacks from P0-A/P0-B/P0-C to `I0`;
  model-driven polling remains a documented fallback rather than the default.
- 2026-07-17: P0-A (`db486f5`) and P0-B (`c6527dd`) reported complete with
  focused, Ruff, and v3-suite evidence; both await delegator review/integration.
- 2026-07-17: P0-C (`168c49a`) reported complete with focused identity/provenance,
  Ruff, format, and v3-suite evidence; it awaits delegator review/integration.
- 2026-07-17: P0-A integrated as `92bc87d`; the combined checkout has zero old
  assembly imports, both Ruff gates pass, and v3 remains 130 passed/40 warnings.
- 2026-07-17: integration review returned P0-B for redundant authored geometry
  arity and unhashable-reference diagnostics, and P0-C for retained-array ownership;
  both repairs remain inside their original path contracts.
- 2026-07-17: integrated corrected P0-C as `d211afa` + `0834013` and corrected
  P0-B as `da1e60f` + `ba466cd`; combined focused, v3, full-suite, Ruff, and format
  gates pass. Independent edge-case review is next; compiler integration is not yet authorized.
- 2026-07-17: dispatched independent Sol/max reviewers R0-A and R0-B from `2bda241`
  with separate spec-semantics and identity/ownership review scopes and direct
  terminal callbacks. No polling automation is active.
- 2026-07-17: dispatched Sol/max meta-coordinator D0-A from `ad95149` to design a
  repository-specific, bounded semi-autonomous legacy-to-v3 migration workflow in
  one isolated proposal document; it has no production or existing-policy writes.
- 2026-07-17: R0-A and R0-B each returned three accepted blockers. P0-B now owns
  canonical nested-type guards, registry-owned interpolation arity, and 0D topology;
  P0-C owns lossless scalar policy, ndarray-subclass rejection, and exact identity
  hardening. Compiler integration remains closed pending repair and re-review.
- 2026-07-17: integrated the corrected P0-C boundary as `e1d7fe7`; 18 focused
  identity/provenance tests, 157 combined v3 tests, both Ruff gates, and format pass.
  Re-review will start from the final combined P0-B/P0-C repair head.
- 2026-07-17: integrated corrected P0-B canonicalization/topology as `f2a0cd2`;
  the combined repair head passes 41 focused, 171 v3, and 360 repository tests plus
  both Ruff gates and format. Fresh R0 spec and identity reviews are next.
- 2026-07-17: D0-A returned the one-file workflow source `50cc663`; integration
  review accepted its bounded packet conveyor, capability/evidence ledger,
  callback-first coordination, model/cost policy, and evidence-checkable completion bar.
- 2026-07-17: integrated the workflow proposal as `9b26574`, adopted it as the sole
  migration method, renamed this file from `phase0-execution.md`, and routed the
  document map without changing `design.md` or production code.
- 2026-07-17: dispatched fresh Sol/max reviewers `R0-C` and `R0-D` from exact combined
  head `faab0c938705f59fc5a22e702413f295af1dcadb`; both are active with disjoint spec
  and identity/storage lenses and direct callbacks. No compiler writer is active.
- 2026-07-17: the first R0-C and R0-D tasks did not finish correctly. R0-C resumed
  in replacement task `019f70af-2c98-7563-b303-0a5a66fd6ef5`; R0-D was archived and
  replaced exactly by Sol/max task `R0-G` (`019f70b1-793c-7c90-a051-07911fff3134`).
- 2026-07-17: replacement R0-C completed with 23 focused tests passing but three
  accepted blockers: returned caller-tree aliasing, custom scalar-subclass escape,
  and unvalidated manually altered slots/containers causing raw exceptions. P0-E owns the
  bounded canonical-tree and total-preflight repair; compiler integration stays shut.
- 2026-07-17: froze P0-E in `47e94752e93b7424c73e4d2979bebbe0f7567291`
  and dispatched Sol/max task `019f70b8-3e74-7d82-b27b-67991cf50e3c` with exclusive
  spec/test ownership. Its first status confirms the intended two-phase trusted
  capture then owned reconstruction; R0-G remains the only concurrent active reviewer.
- 2026-07-18: P0-E completed as source `246114ba32b8367675f2b3954e67f9e3847ca235`.
  Integration review found no correctness blocker and cherry-picked it as
  `108552ddd3382163a0e15c2fef7ca34e75f974fd`.
- 2026-07-18: the exact integrated P0-E head passes 35 focused spec tests, 183 v3
  tests, 372 full-repository tests, both Ruff gates, focused format, and
  `git diff --check`; 40 existing SciPy and four cold-cache Numba performance
  warnings remain. Fresh read-only R0-H is the next spec gate while R0-G continues.
- 2026-07-18: live status exposed R0-G as `systemError` before any terminal report.
  Its incomplete observations are not accepted evidence. Sol/max replacement R0-I
  (`019f711f-339b-7300-aa1c-17e6e7ea9974`) must reproduce or reject each lead.
- 2026-07-18: dispatched read-only Sol/max reviewers R0-H
  (`019f711d-c79c-7652-96dc-f07e55fdb71b`) and R0-I from exact base
  `13c68e302cf8f4e0f7e211f8af46eff09c863368`; both confirmed that base and are
  active with disjoint spec and identity/storage lenses.
- 2026-07-18: R0-H completed after one report-only resume from a system-erroring
  finalization turn. Its broad local edge-case matrix passed, but exact integers beyond the
  interpreter decimal limit leak raw `ValueError` in duplicate/reference/source
  diagnostics. The finding is accepted for bounded P0-F repair; its measured double
  reconstruction cost remains nonblocking.
- 2026-07-18: froze P0-F in `f9e1867a2c458531b61fd3d8e5107445c0229733`
  and dispatched Sol/max task `019f7136-c2f1-77b2-81bb-f9aa41627a92` with exclusive
  spec diagnostics/normalization/test ownership. It confirmed the exact clean base;
  R0-I remains the only concurrent read-only identity reviewer.
- 2026-07-18: R0-I completed after one report-only resume from a system-erroring
  probe turn. Eighteen focused tests passed, but five ordinary subclass/dtype/
  equality/snapshot seams violate exact canonical meaning. All five are accepted
  for bounded P0-G repair; manually constructed identity fields and missing depth evidence remain
  explicit later obligations rather than repair-scope inflation.
- 2026-07-18: froze P0-G in `7d1ededb6d11fc183ff506b989b6d1f72fd12600`
  and dispatched Sol/max task `019f7146-e871-76c2-a3e6-75d87fac83fd` with exclusive
  array/provenance/registry/test ownership. It confirmed the exact base gate; P0-F
  continues concurrently on disjoint spec paths.
- 2026-07-18: P0-F completed as source
  `c0b3c578e6641cad428244065221170136534b51`. Integration review found no correctness
  blocker and cherry-picked it as
  `461a8a85de622820eb627a24b8e53749489d020a`; 42 focused and 190 v3 tests plus both
  Ruff gates, focused format, and `git diff --check` pass on the integration branch.
  The worker additionally proved seven selected cases under a 640-digit conversion
  limit and 379 full-repository tests. P0-G remains the only active writer.
- 2026-07-18: P0-G completed its first source as
  `95044678c54a5d9f5141bed504d160fd1bfbbd33`, exactly one four-path commit over its
  declared base. Its 42 focused, 207 v3, and 396 full-repository tests plus static
  gates pass. Integration-owner probes nevertheless reproduced an ordinary huge-int
  conversion-limit escape, an overflow float escape, non-finite ndarray acceptance,
  and structurally invalid unordered-carrier acceptance. Integration is withheld;
  the same Sol/max task is producing one provenance/test-only child repair without
  changing manifest v1.
- 2026-07-18: P0-G completed direct child
  `8612d0140dc0cb7e68bc1bc61d3947acf3f81de0`, which repairs all four audit findings
  plus constructor-reachability gaps for canonical array payloads. Review replayed
  the local edge-case checks, then integrated source and child as
  `e04f86a67e5fa225f6191b7ef16ca5572598cbb9` and
  `afaac4d189979c861fb0463e6aa9c07bc1bc4ed5`.
- 2026-07-18: the combined repaired foundation passes 89 focused tests normally and
  at the 640-digit limit, 219 v3 tests, 408 full-repository tests, both Ruff gates,
  combined format, `git diff --check`, and the prototype-import quarantine scan;
  only 40 pre-existing SciPy warnings remain. Fresh repeat reviewers are now the sole
  gate before compiler integration.
- 2026-07-18: froze the combined proof and ledger at
  `8a952e7668f4f6c52d39352dee7e0685593906a3`, then dispatched read-only Sol/max
  reviewers R0-J (`019f7184-2fe6-7403-a5fd-683e58dcee79`) and R0-K
  (`019f7184-2fe6-7403-a5fd-685f9ee1995d`) from that exact clean head. Their scopes
  are disjoint spec versus identity/storage boundaries; no writer or compiler task
  is active.
- 2026-07-18: R0-J produced strong zero-blocker supporting evidence: its robustness
  cases, 42 focused tests both normally and under the 640-digit limit, both Ruff
  gates, and 219-test broad replay were green. It nevertheless ended without the
  required final result or callback and is therefore incomplete. R0-K also ended
  incomplete after creating but not executing a temporary script; it ran none of
  its required proof gates.
- 2026-07-18: audited all 17 user-visible migration tasks against their actual final
  turns. Twelve completed properly; initial R0-C, R0-D, R0-G, R0-J, and R0-K did
  not. Their visible titles now begin `INCOMPLETE`. The workflow now requires an
  explicit terminal-result final plus callback, uses concise local FEM correctness
  language for future reviewer prompts, and forbids promotion from idle state,
  commentary, or recovered partial evidence. Compiler integration remains closed
  pending one bounded identity/storage replacement review.
- 2026-07-18: dispatched exactly one Sol/max replacement, R0-L
  (`019f7551-0075-7383-a895-fd2d77f02419`), from exact audit commit
  `dc9e588a50c21596dfc2b7f2a879c3a4dd31ee92`. Its concise prompt links only the
  neutral local correctness card and relevant implementation/tests. Initial
  preflight confirmed the exact base and clean worktree; no other task is active.
- 2026-07-18: R0-L completed its mandatory focused/static block: 47 focused tests
  passed both normally and with the 640-digit interpreter limit, and both Ruff
  configurations passed. It is waiting for user approval to create the explicitly
  authorized task-specific `/private/tmp` script for its remaining independent
  cross-product and depth checks. No final result or callback exists yet, so the
  packet remains active and the compiler gate remains closed.
- 2026-07-18: after approval, R0-L completed 162 bounded local checks per digit mode,
  the 219-test v3 suite, a valid terminal final, and its direct callback. It returned
  NO-GO with one P1 frozen-registry defect. I0 reproduced the shared-descriptor
  binding change under a stable manifest/fingerprint and the two incidental
  `AttributeError` paths, accepted them as one root contract failure, and kept the
  compiler gate closed for a bounded registry/test repair plus one independent
  recheck.
- 2026-07-18: dispatched exactly one bounded Sol/max writer, P0-H
  (`019f7568-a1e0-7c33-bf83-0a490e54c520`), from exact clean adjudication commit
  `55fcf990da8f47f612f5193529c2a6b77a823a10`. It owns only the registry primitive
  and its focused identity tests, must return one commit plus both terminal signals,
  and cannot begin compiler work. No replacement or polling automation is active.
- 2026-07-18: P0-H completed both terminal signals with exact one-commit source
  `0947dd7ca3b70bef2ebdf986f2336f62f5b3c08b`. I0 reviewed its two-path diff,
  reproduced source detachment and missing-field behavior, integrated it without
  conflict as `fb358fc0609b81a12cee4a1a66c2dae98edf5cae`, and reran 68 focused tests
  in both modes, both Ruff gates, focused format, 240 v3 tests, and 429 full tests
  successfully.
- 2026-07-18: dispatched exactly one read-only Sol/max recheck, R0-M
  (`019f757a-e84f-70c3-a6d6-8c4ff7874204`), from exact clean integrated commit
  `fb358fc0609b81a12cee4a1a66c2dae98edf5cae`. It owns only the repaired registry
  invariant, has no writer overlap, and must provide both terminal signals before
  I0 can close the foundation gate.
- 2026-07-18: R0-M's first turn suffered an app-level output-capture `systemError`
  while its broad test process was still running. It had no final or callback, so
  I0 did not count it complete and issued the one workflow-permitted resume. That
  resume recovered the existing 240-pass process, confirmed 42/42 local checks, 68
  focused tests in both modes, both Ruff gates, exact HEAD and clean status, then
  supplied a GO final and callback with zero blockers. The foundation gate is now
  closed green at `fb358fc0609b81a12cee4a1a66c2dae98edf5cae`; no replacement chain
  was created.
- 2026-07-18: committed the foundation closure as
  `c50ca70bff884157c5645dde276c25acc6672d4a`, then dispatched exactly one
  read-only Sol/max inventory task, R0-E
  (`019f7587-31f2-7642-b162-5fd5b7a2d07b`), from that exact clean commit. Its sole
  output is the complete external E0 capability report for I0 ingestion; it cannot
  edit the repository or begin compiler work. P0-D remains closed until the
  inventory is terminal and adjudicated.
- 2026-07-18: R0-E confirmed its exact base and clean worktree, read all governing
  documents, and began repository-surface reconciliation. It is paused only on the
  app approval required to write the explicitly authorized external report at
  `/private/tmp/pyfem-r0e-capability-inventory.md`. No repository change exists and
  no compiler task is active.
- 2026-07-18: corrected the coordinator vocabulary after R0-M was incorrectly
  routed under an unrelated review frame. R0-M remains a completed finite-element
  registry-snapshot correctness review with zero blockers. Completed task titles
  now use neutral state descriptions, and active R0-E received a follow-up requiring
  the `failure_cases` schema field plus local finite-element correctness wording.
  Scope and evidence gates are unchanged.
- 2026-07-18: R0-E received the correction and resumed after its report-write
  approval resolved. It confirmed the mandatory sources and exact-base gate,
  completed the repository-surface census, and is assembling the per-capability
  ledger plus coverage appendix with `failure_cases` and neutral finite-element
  wording. No repository output or terminal callback exists yet, so R0-E remains
  active and P0-D remains closed.
- 2026-07-18: R0-E completed both terminal signals after producing and auditing a
  154-row E0 inventory with 606/606 tracked paths, 122/122 property files, 33/33
  skim paths, 41/41 test/support paths, and zero unclassified remnants. I0
  independently reproduced its schema, ID, dependency, manifest, mapping, and
  terminology checks, preserved the exact report under `evidence/` at SHA-256
  `f9e326da3adc8e6bb3a777fb4134965ade6f9b364df352c53e68c2cbb7e377cf`,
  classified the 18-row Q8 compiler dependency cut, and retained explicit blocks
  on six public retirement candidates. P0-D is not dispatched until its Horizon is
  frozen from the resulting clean ledger commit.
- 2026-07-18: froze the P0-D Horizon after comparing the authoritative compiler
  boundary, the normalized spec and identity/storage foundations, the existing Q8
  mathematical kernels, and the 18 immediate capability rows. The contract fixes
  one explicit per-unit-thickness Q8 plane-stress recipe, four versioned descriptor
  meanings, canonical dense/source/DOF maps, scale-relative reference-geometry
  rejection, derived capabilities, and a state layout with no evolving values. It
  assigns disjoint `compile/**`, compiled-model, and focused-test ownership and
  keeps program, section, assembly, state, public API, adapters, and prototype
  paths outside P0-D. Exactly one Sol/max writer may now be dispatched from this
  card's clean commit.
- 2026-07-18: committed the frozen Horizon as
  `2842ef84b86f04f82587f5fc378584f30f6b62bd` after 132 focused
  spec/identity/Q8 component tests and 110 strict-digit foundation tests passed,
  then dispatched exactly one Sol/max writer, P0-D
  (`019f75dd-f597-7c21-adfc-d78b1e2580c0`), from that exact clean commit. Its
  worktree preflight is active. No parallel compiler owner or R0-F reviewer exists;
  neither completion nor integration is claimed until both terminal signals and I0
  review are present.
