# Phase 0/1 execution state

- Status: foundation critic repairs active; compiler gate closed
- Owner: delegating/integration thread
- Design authority: [design.md](design.md)
- Method: [refactor_playbook.md](refactor_playbook.md)
- Dispatch base: `c75cbf3` (`v3`)
- Integrated foundation: `ba466cd` (`v3`)

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

## Merge and continuation order

1. Review and integrate P0-A.
2. Review P0-B and P0-C independently against the Horizon Gate.
3. Integrate P0-B/P0-C and rerun combined v3 plus full repository tests.
4. Dispatch the dependent compiler-integration packet: one explicit Q8 region ->
   immutable `CompiledModel` recipe with entity/source maps and empty physical-state
   layout.
5. Only then dispatch `ProgramSpec`/affine constraints and
   `PreparedAssemblyPlan` work.

## Active threads

Titles follow the compact coordinate/owner/outcome convention in
[refactor_playbook.md](refactor_playbook.md#thread-titles). Status and execution
metadata stay here rather than being encoded in the title.

Completion signalling: each worker has a direct terminal callback to `I0`; no
polling automation is active. A watchdog is unnecessary while callbacks and native
thread status are available.

| Exact title | Thread | Status | Output |
|---|---|---|---|
| `I0 · foundations — combined proof` | `019f6f49-0b72-7d73-86da-c6b85519eeaf` | Active | — |
| `P0-A · assembly — prototype quarantined` | `019f7060-0bb3-7a72-bb6b-47697f1c5747` | Integrated | `92bc87d` from `db486f5` |
| `P0-B · model spec — explicit immutable intent` | `019f7060-0bb9-7b40-8cfb-f056155afe37` | Repairing critic findings | `da1e60f` + `ba466cd` + pending |
| `P0-C · identity/storage — owned and frozen` | `019f7060-0bb1-7a72-b438-5c2274f3d5e8` | Repairing critic findings | `d211afa` + `0834013` + pending |
| `R0-A · model spec — semantic gaps attacked` | `019f7087-61f2-78a2-9df7-5174dbc5b8a3` | Complete | 3 accepted blockers |
| `R0-B · identity/storage — invariants attacked` | `019f7087-61f0-71e0-9082-122e7ea75894` | Complete | 3 accepted blockers |
| `D0-A · migration workflow — autonomy bounded` | `019f708b-2980-7703-8fca-7ea26d5826ba` | Active | `migration_workflow.md` proposal |

## Required combined evidence

- focused tests owned by every packet;
- both v3 Ruff configurations;
- `pytest -q test/v3` after each merge;
- full `pytest -q` after the first combined foundation; and
- disconfirming review of identity, caller aliasing, explicit topology, and
  prototype-import quarantine before the integration packet begins.

Current combined evidence at `ba466cd`:

- 21 focused foundation tests passed;
- 151 v3 tests passed with 40 pre-existing SciPy warnings;
- 340 full-repository tests passed with the same 40 warnings;
- both v3 Ruff gates and focused format checks passed; and
- R0-A/R0-B found six accepted Phase-0 blockers; their bounded repairs and repeat
  adversarial review are the open gate before compiler integration.

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
