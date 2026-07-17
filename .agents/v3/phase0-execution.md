# Phase 0/1 execution state

- Status: dispatching first independent work packets
- Owner: delegating/integration thread
- Design authority: [design.md](design.md)
- Method: [refactor_playbook.md](refactor_playbook.md)

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

## Required combined evidence

- focused tests owned by every packet;
- both v3 Ruff configurations;
- `pytest -q test/v3` after each merge;
- full `pytest -q` after the first combined foundation; and
- disconfirming review of identity, caller aliasing, explicit topology, and
  prototype-import quarantine before the integration packet begins.

## Blocked condition

Stop a packet when its owned contract cannot be implemented without deciding an
API or invariant owned by another packet, editing a forbidden shared path, or
changing prototype behavior. Record the exact dependency and smallest required
integration decision; do not bridge it with a compatibility carrier.

## Milestone log

- 2026-07-17: architecture reset committed; FVM refactor playbook adapted; first
  three disjoint work packets defined for GPT-5.6 Sol with max reasoning.
