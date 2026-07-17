# PyFEM v3 migration execution ledger

- Status: R0-C spec blockers accepted; P0-E repair contract frozen; R0-G identity re-review active
- Owner: delegating/integration thread
- Target branch: `v3`
- Design authority: [design.md](design.md)
- Migration method: [migration_workflow.md](migration_workflow.md)
- Supporting structural method: [refactor_playbook.md](refactor_playbook.md)
- Original dispatch base: `c75cbf3523349deb40bd2b07de7959e7606c3b1f`
- Integrated foundation implementation: `f2a0cd2247728f26bb2f8a641824ae490ba38ddf`
- Workflow proposal integrated: `9b26574f52c39be756e2cdeb275dcfe7691e5bc4`
- Active milestone: `I0 · foundations — combined proof`

## Exact next safe action

Commit this ledger transition, dispatch `P0-E` from that exact committed base, and
record its task ID. In parallel, consume the terminal callback from `R0-G`, the exact
replacement for failed identity critic `R0-D`. Review and integrate required repairs
serially, rerun fresh adversarial review where a boundary changed, then bind the
combined foundation proof to one exact commit. Do not dispatch the model compiler
writer until this gate is green.

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
- Thread IDs are never reused. `R0-C` and `R0-D` belong to the current foundation
  re-reviews; future legacy-breadth and compiler critics use `R0-E` and `R0-F`.

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

State: `HORIZON_FROZEN`; dispatch immediately after this ledger transition is
committed. This is a post-integration repair from the current `v3` head, not a
rewrite of P0-B history.

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

Merge order: review and integrate P0-E independently of R0-G; if R0-G also requires
an identity-owner repair, integrate both disjoint repairs serially and re-run the
combined foundation gates before any compiler packet.

## Merge and continuation order

1. Completed: review and integrate P0-A.
2. Completed: review P0-B and P0-C independently against the Horizon Gate.
3. Completed: integrate P0-B/P0-C, repair the six accepted original critic
   findings, and rerun combined v3 plus full repository tests.
4. Active: repair/re-prove the three accepted `R0-C` model-spec blockers in `P0-E`
   and consume replacement identity critic `R0-G`.
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
| `R0-G · identity/storage — repairs falsified` | `019f70b1-793c-7c90-a051-07911fff3134` | `faab0c938705f59fc5a22e702413f295af1dcadb` | Exact R0-D replacement, same read-only lens | Active; terminal callback required |
| `P0-E · model spec — canonical tree owned` | pending dispatch | commit containing the frozen P0-E card | `pyfem/v3/spec/**`, focused spec tests | `HORIZON_FROZEN`; repair round 0 |

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
`f2a0cd2247728f26bb2f8a641824ae490ba38ddf`:

- 41 focused foundation tests passed;
- 171 v3 tests passed with 40 pre-existing SciPy warnings;
- 360 full-repository tests passed with the same 40 warnings;
- both v3 Ruff gates and focused format checks passed; and
- the six accepted R0-A/R0-B blockers are repaired; fresh adversarial re-review
  from one combined head is the open gate before compiler integration.

## Accepted later-phase obligations

- Registry snapshot fingerprints trust callable purity and truthful, behavior-bound
  `implementation_id` values. Compiler/restore work must preserve or strengthen
  that trust boundary; a mutable callable can otherwise drift behind a stable
  descriptor manifest.
- Repeated `base.next_accepted()` calls can identify sibling prospective
  generations equally. The future commit transaction must prevent multiple
  accepted transitions from one base and separately identify trial candidates;
  the primitive alone is not transaction enforcement.

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
