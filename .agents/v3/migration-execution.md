# PyFEM v3 migration execution ledger

- Status: P0-F integrated; P0-G identity/storage repair active
- Owner: delegating/integration thread
- Target branch: `v3`
- Design authority: [design.md](design.md)
- Migration method: [migration_workflow.md](migration_workflow.md)
- Supporting structural method: [refactor_playbook.md](refactor_playbook.md)
- Original dispatch base: `c75cbf3523349deb40bd2b07de7959e7606c3b1f`
- Integrated foundation implementation: `108552ddd3382163a0e15c2fef7ca34e75f974fd`
- Integrated exact-integer diagnostic repair: `461a8a85de622820eb627a24b8e53749489d020a`
- Workflow proposal integrated: `9b26574f52c39be756e2cdeb275dcfe7691e5bc4`
- Active milestone: `I0 · foundations — combined proof`

## Exact next safe action

Review and integrate P0-G's bounded identity/storage repair after its direct
callback, rerun the combined gates, then repeat the spec and identity/storage
critics from one exact repaired head. P0-F is integrated and green at its focused,
static, and v3 gates. R0-G system-failed without an acceptable terminal report;
R0-I independently reproduced and classified its useful leads. Do not start the
model compiler until both repeated critics are green at one exact combined
foundation commit.

Blocked condition: a critic demonstrates an invariant failure that cannot be
repaired inside the existing spec or identity/storage owner without a new design,
physics, compatibility, dependency, or authority decision.

## Semantic decisions and open questions

Decided:

- `design.md` remains authoritative; the legacy runtime and current v3 prototype are
  requirements and evidence, not target architecture.
- Development continues on local branch `v3`; the separate unmerged modernization
  line is not silently imported.
- Full migration uses the bounded packet conveyor in `migration_workflow.md`, direct
  terminal callbacks, serial integration, and independent adversarial proof.
- Thread IDs are never reused. `R0-E` and `R0-F` remain reserved for the planned
  legacy-breadth and compiler critics; replacement/follow-up foundation reviews use
  the next otherwise-unreserved IDs, hence `R0-G`, `R0-H`, and `R0-I`.
- dtype metadata is outside the current v1 manifest/finalization boundary and must
  fail closed; do not invent a recursive dtype-metadata schema or manifest v2 before
  an executable compiler slice demonstrates that semantic requirement;
- structured arrays remain unsupported by manifest v1, while metadata-free
  structured finalization may remain only if it preserves full dtype/value ownership;
  and
- total `FinalizedArray` identity semantics cover the carrier and standard NumPy
  comparison protocols, not arbitrary hostile unrelated left operands whose own
  equality method controls dispatch.

Open and not implicitly decided:

- root Python API, `pyfem` CLI, and `.pro`/`.dat` compatibility;
- optional GUI preservation, replacement, or retirement;
- RVE/FE2 and ROM preservation, redesign, or retirement; and
- the later-phase registry-callable trust and accepted-generation transaction
  obligations recorded below.

## Capability coverage ledger

The complete legacy capability ledger is not yet seeded. Until `R0-E` produces a
reviewable E0 inventory and the integration owner ingests it, [feature-parity.md](feature-parity.md)
is only a temporary source list. Its old `done`, `deferred`, and `out of scope`
labels are not preserve/change/retire decisions. No unclassified capability is
silently excluded from the migration-complete bar.

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
  dangerous cases for the replacement boundary.
- Do not optimize, add Numba policy, expose public exports, or continue the old
  roadmap.

## Strongest competing hypothesis

The prototype could be extended in place by adding fields to `ProblemDefinition`
and more registry/group switches. This chunk is falsified if the new foundations
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
are green and fresh adversarial review is required. The task was dispatched from
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

1. `normalize_model_spec()` returns the caller tree, so forged mutable slots and
   source values remain aliased.
2. `str`/`int` subclasses can retain polymorphic hash, comparison, stripping, and
   rendering behavior or escape as raw exceptions.
3. exact-but-forged dataclasses and wrong internal containers are trusted before
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

Required dangerous cases and evidence:

- normalized result and every nested spec/source/value/container are newly owned
  exact canonical values; mutating forged caller slots afterward cannot affect it;
- mutable-hash, unhashable, repr-bomb `str` subclasses and comparison-bomb `int`
  subclasses reject deterministically without invoking hostile behavior;
- exact uninitialized or wrong-slot `ModelSpec`, `MeshSpec`, `NodeSpec`, and
  `SourceContext`, plus non-iterable/list-forged collections, never escape raw
  `AttributeError`, `TypeError`, or rendering failures;
- ordinary multi-defect ordering, 0D topology, block-local identity, differing
  uniform block arities, caller detachment, and all earlier P0-B repairs remain;
- focused spec tests, both v3 Ruff configurations, focused format, `pytest -q
  test/v3`, full `pytest -q`, `git diff --check`, and a clean worker worktree pass.

Merge order: completed independently of the identity critic. If R0-I or R0-H
requires another owner repair, integrate disjoint repairs serially and re-run the
combined foundation gates before any compiler packet.

### R0-H — Re-falsify the repaired canonical model boundary

State: `COMPLETE` in task `019f711d-c79c-7652-96dc-f07e55fdb71b`, reviewed
read-only from exact base `13c68e302cf8f4e0f7e211f8af46eff09c863368`.
One blocker is accepted: exact integers beyond Python's decimal conversion limit
normalize successfully but duplicate/reference/source diagnostics interpolate them
with ordinary decimal rendering and leak raw `ValueError`.

Review lens:

- prove the returned tree is recursively exact, newly owned, and detached from
  hostile caller mutation at every nested spec/source/value/container boundary;
- attack every spec family with exact uninitialized objects, forged slots, wrong
  collection shapes, container subclasses, cycles, deep values, and malformed
  `SourceContext` values;
- prove hostile scalar subclasses cannot execute polymorphic hash, representation,
  comparison, or string-normalization behavior and every failure remains a
  deterministic `ModelSpecValidationError` at the nearest trusted source;
- preserve ordinary multi-defect ordering, 0D topology, block-local identity,
  differing uniform cross-block arities, and the absence of compiler/program/
  section policy in the authored boundary; and
- scrutinize the second defensive reconstruction, but report it as a blocker only
  if evidence shows a correctness failure or a required foundation-scale violation.

Owned output: terminal review report and reproducible hostile probes only. No file
edits, commits, merge, push, API expansion, or compiler work.

Evidence and adjudication:

- 35 focused spec tests passed in 0.76 seconds;
- all 11 exact spec/source families, 27 hostile scalar subclasses, 13 hostile
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

Required dangerous cases and evidence:

- duplicate 5,001-digit node/block/cell/field/material/region IDs use one bounded,
  stable representation and preserve diagnostic order;
- unknown/duplicate huge node, cell, block, field, and material references never
  invoke unbounded `repr`/`str` conversion;
- huge exact `SourceContext.line` and `.column` values render safely when another
  defect is reported, including nearest-trusted-source fallback;
- ordinary small integer/string diagnostic text, 0D topology, block-local identity,
  cross-block arity policy, and all P0-E hostile/detachment cases remain unchanged;
- focused spec tests, both v3 Ruff configurations, focused format, `pytest -q
  test/v3`, full `pytest -q`, `git diff --check`, and a clean worker worktree pass.

Forbidden: new ID-range policy, global interpreter-setting changes, compiler/
program/section work, double-reconstruction optimization, public exports,
dependency/config changes, shared-document edits, merge, or push.

### R0-I — Re-falsify identity, provenance, registry, and storage repairs

State: `COMPLETE` in task `019f711f-339b-7300-aa1c-17e6e7ea9974`, reviewed
read-only from exact base `13c68e302cf8f4e0f7e211f8af46eff09c863368`.

Accepted foundation blockers:

1. semantic NumPy scalar subclasses execute conversion methods during manifest
   capture, and nested semantic ndarray subclasses such as `MaskedArray` lose their
   meaning when hidden inside a list passed to `finalize_array`;
2. dtype metadata is omitted from fingerprints and its nested mutable objects remain
   caller-aliased across finalizations;
3. `FinalizedArray` comparison becomes elementwise under standard NumPy reverse/
   ufunc dispatch, and a public carrier subclass can spoof base-carrier equality;
4. canonical carriers plus mapping/list subclasses retain polymorphic execution
   paths, `ContentFingerprint` subclasses spoof equality, and invalid `id_key`
   values can escape as raw exceptions; and
5. `RegistrySnapshot` accepts and retains descriptor/key/source subclasses, allowing
   exposed key/binding meaning to drift under a stable snapshot fingerprint.

Evidence and exclusions:

- 18 focused identity tests passed in 2.06 seconds; independent hostile probes
  reproduced all five blockers under NumPy 2.3.5; ordinary exact scalar/array,
  endian/stride, cycle, detachment, and registry-rebinding cases held;
- exact forged identity fields required `object.__new__` plus deliberate
  `object.__setattr__` and are not a current primitive blocker; restore/rebind must
  validate exact `UUID`/integer fields before comparing externally reconstructed
  identity values;
- arbitrary unrelated left-operand equality and deliberate re-enabling of an owning
  NumPy array's write flag remain outside the carrier's accidental-mutation contract;
  and
- no depth-probe output was obtained, so R0-I makes no deep-acyclic-manifest claim.

### P0-G — Close polymorphic identity/storage semantic drift

State: `IMPLEMENTING` in task `019f7146-e871-76c2-a3e6-75d87fac83fd`,
dispatched from exact base `7d1ededb6d11fc183ff506b989b6d1f72fd12600`.

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

Required dangerous cases and evidence:

- NumPy integer/float subclasses with conversion bombs reject at every nested
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

## Merge and continuation order

1. Completed: review and integrate P0-A.
2. Completed: review P0-B and P0-C independently against the Horizon Gate.
3. Completed: integrate P0-B/P0-C, repair the six accepted original critic
   findings, and rerun combined v3 plus full repository tests.
4. Active: P0-F repaired the R0-H diagnostic blocker and is integrated; R0-I
   accepted five disjoint identity/storage blockers for P0-G. Integrate P0-G, then
   repeat both critics from one exact combined head.
5. Then dispatch the dependent compiler-integration packet: one explicit Q8 region ->
   immutable `CompiledModel` recipe with entity/source maps and empty physical-state
   layout.
6. Only after P0-D and its `R0-F` critic dispatch `ProgramSpec`/affine constraints and
   `PreparedAssemblyPlan` work.

## Packet ledger

Titles follow the compact coordinate/owner/outcome convention in
[refactor_playbook.md](refactor_playbook.md#thread-titles). Status and execution
metadata stay here rather than being encoded in the title.

Completion signalling: each worker has a direct terminal callback to `I0`; no
polling automation is active. A watchdog is unnecessary while callbacks and native
thread status are available.

| Exact title | Thread | Exact base | Owner/output | State and evidence |
|---|---|---|---|---|
| `I0 · foundations — combined proof` | `019f6f49-0b72-7d73-86da-c6b85519eeaf` | `c75cbf3523349deb40bd2b07de7959e7606c3b1f` | Integration, shared docs, combined proof | Active |
| `P0-A · assembly — prototype quarantined` | `019f7060-0bb3-7a72-bb6b-47697f1c5747` | `c75cbf3523349deb40bd2b07de7959e7606c3b1f` | Assembly quarantine | Integrated as `92bc87d` from `db486f5`; repair 0 |
| `P0-B · model spec — explicit immutable intent` | `019f7060-0bb9-7b40-8cfb-f056155afe37` | `c75cbf3523349deb40bd2b07de7959e7606c3b1f` | `pyfem/v3/spec/**` | Integrated through `f2a0cd2`; two integration repairs plus one critic repair |
| `P0-C · identity/storage — owned and frozen` | `019f7060-0bb1-7a72-b438-5c2274f3d5e8` | `c75cbf3523349deb40bd2b07de7959e7606c3b1f` | `pyfem/v3/model/**` | Integrated through `e1d7fe7`; one integration repair plus one critic repair |
| `R0-A · model spec — semantic gaps attacked` | `019f7087-61f2-78a2-9df7-5174dbc5b8a3` | `2bda241719e2c236abe4711528abd82f47b4633d` | Original spec critic | Complete; 3 accepted blockers repaired |
| `R0-B · identity/storage — invariants attacked` | `019f7087-61f0-71e0-9082-122e7ea75894` | `2bda241719e2c236abe4711528abd82f47b4633d` | Original identity/storage critic | Complete; 3 accepted blockers repaired |
| `D0-A · migration workflow — autonomy bounded` | `019f708b-2980-7703-8fca-7ea26d5826ba` | `ad95149e2e34b8eff55c0896c1dea53ac1cbc71d` | `migration_workflow.md` | Complete; source `50cc663`, integrated `9b26574`, adopted |
| `R0-C · model spec — repairs falsified` | interrupted `019f70a3-2650-7181-8a05-fc2b72b111a5`; replacement `019f70af-2c98-7563-b303-0a5a66fd6ef5` | `faab0c938705f59fc5a22e702413f295af1dcadb` | Fresh read-only spec critic | Complete; 3 accepted blockers; 23 focused tests passed |
| `R0-D · identity/storage — repairs falsified` | `019f70a3-264c-7831-8509-a3ffbf9235f4` | `faab0c938705f59fc5a22e702413f295af1dcadb` | Initial identity/storage critic | Failed to finish correctly; archived; no terminal report accepted |
| `R0-G · identity/storage — repairs falsified` | `019f70b1-793c-7c90-a051-07911fff3134` | `faab0c938705f59fc5a22e702413f295af1dcadb` | Exact R0-D replacement, same read-only lens | System-failed before terminal report; partial observations are unaccepted leads |
| `P0-E · model spec — canonical tree owned` | `019f70b8-3e74-7d82-b27b-67991cf50e3c` | `47e94752e93b7424c73e4d2979bebbe0f7567291` | `pyfem/v3/spec/**`, focused spec tests | Complete; source `246114b`, integrated `108552d`; 35 focused, 183 v3, 372 full tests passed |
| `R0-H · model spec — canonical boundary attacked` | `019f711d-c79c-7652-96dc-f07e55fdb71b` | `13c68e302cf8f4e0f7e211f8af46eff09c863368` | Fresh read-only P0-E critic | Complete; 1 accepted exact-integer rendering blocker; 35 focused passed |
| `R0-I · identity/storage — repairs falsified` | `019f711f-339b-7300-aa1c-17e6e7ea9974` | `13c68e302cf8f4e0f7e211f8af46eff09c863368` | Exact-scope R0-G replacement with independent reproduction | Complete; 5 accepted blockers; 18 focused passed |
| `P0-F · model spec — integer diagnostics total` | `019f7136-c2f1-77b2-81bb-f9aa41627a92` | `f9e1867a2c458531b61fd3d8e5107445c0229733` | Bounded exact-integer diagnostic repair | Complete; source `c0b3c57`, integrated `461a8a8`; 42 focused, 190 v3, 379 full tests passed |
| `P0-G · identity/storage — exact boundaries enforced` | `019f7146-e871-76c2-a3e6-75d87fac83fd` | `7d1ededb6d11fc183ff506b989b6d1f72fd12600` | Close five accepted R0-I semantic-drift seams | `IMPLEMENTING`; repair round 0; direct callback required |

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
  memory 2.37 times retained. This preserves the ownership contract and is not a
  foundation blocker; remeasure at compiler-scale proof before optimizing, and
  retain hostile detachment tests through any change.
- Exact forged `InstanceId`/`StateGeneration` fields can bypass helpers or raise raw
  exceptions after deliberate `object.__new__`/`object.__setattr__` construction.
  This is not an authored/live primitive blocker; future restore/rebind decoding
  must validate exact `UUID`/integer fields before invoking identity comparisons.
- Deep acyclic manifest behavior remains unproven because R0-I obtained no depth
  probe output. Bound or iteratively prove it before accepting untrusted detached
  manifest/restore input; do not widen P0-G solely for this missing evidence.

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
  gates pass. Adversarial review is next; compiler integration is not yet authorized.
- 2026-07-17: dispatched independent Sol/max critics R0-A and R0-B from `2bda241`
  with separate spec-semantics and identity/ownership attack lenses and direct
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
  callback-first coordination, model/cost policy, and falsifiable completion bar.
- 2026-07-17: integrated the workflow proposal as `9b26574`, adopted it as the sole
  migration method, renamed this file from `phase0-execution.md`, and routed the
  document map without changing `design.md` or production code.
- 2026-07-17: dispatched fresh Sol/max critics `R0-C` and `R0-D` from exact combined
  head `faab0c938705f59fc5a22e702413f295af1dcadb`; both are active with disjoint spec
  and identity/storage lenses and direct callbacks. No compiler writer is active.
- 2026-07-17: the first R0-C and R0-D tasks did not finish correctly. R0-C resumed
  in replacement task `019f70af-2c98-7563-b303-0a5a66fd6ef5`; R0-D was archived and
  replaced exactly by Sol/max task `R0-G` (`019f70b1-793c-7c90-a051-07911fff3134`).
- 2026-07-17: replacement R0-C completed with 23 focused tests passing but three
  accepted blockers: returned caller-tree aliasing, hostile scalar-subclass escape,
  and unvalidated forged slots/containers causing raw exceptions. P0-E owns the
  bounded canonical-tree and total-preflight repair; compiler integration stays shut.
- 2026-07-17: froze P0-E in `47e94752e93b7424c73e4d2979bebbe0f7567291`
  and dispatched Sol/max task `019f70b8-3e74-7d82-b27b-67991cf50e3c` with exclusive
  spec/test ownership. Its first status confirms the intended two-phase trusted
  capture then owned reconstruction; R0-G remains the only concurrent active critic.
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
- 2026-07-18: dispatched read-only Sol/max critics R0-H
  (`019f711d-c79c-7652-96dc-f07e55fdb71b`) and R0-I from exact base
  `13c68e302cf8f4e0f7e211f8af46eff09c863368`; both confirmed that base and are
  active with disjoint spec and identity/storage lenses.
- 2026-07-18: R0-H completed after one report-only resume from a system-erroring
  finalization turn. Its broad hostile matrix passed, but exact integers beyond the
  interpreter decimal limit leak raw `ValueError` in duplicate/reference/source
  diagnostics. The finding is accepted for bounded P0-F repair; its measured double
  reconstruction cost remains nonblocking.
- 2026-07-18: froze P0-F in `f9e1867a2c458531b61fd3d8e5107445c0229733`
  and dispatched Sol/max task `019f7136-c2f1-77b2-81bb-f9aa41627a92` with exclusive
  spec diagnostics/normalization/test ownership. It confirmed the exact clean base;
  R0-I remains the only concurrent read-only identity critic.
- 2026-07-18: R0-I completed after one report-only resume from a system-erroring
  probe turn. Eighteen focused tests passed, but five ordinary subclass/dtype/
  equality/snapshot seams violate exact canonical meaning. All five are accepted
  for bounded P0-G repair; forged identity fields and missing depth evidence remain
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
