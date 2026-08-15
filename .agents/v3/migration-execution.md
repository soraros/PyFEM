# PyFEM v3 migration execution ledger

- Status: active after explicit user resumption; I1 and the bounded Phase 1
  reference slices remain component-qualified E3; the R2-E/P2-H/R2-F portfolio
  and S2-A simplification are complete and independently green; the current batch
  measures behavioral parity and refreshes the direct production replacement cut
- Owner: delegating/integration thread
- Target branch: `v3`
- Design authority: [design.md](design.md) plus the post-Phase-1
  [generic semantic-core amendment](generic_core.md)
- Migration method: [migration_workflow.md](migration_workflow.md)
- Supporting structural method: [refactor_playbook.md](refactor_playbook.md)
- Original dispatch base: `c75cbf3523349deb40bd2b07de7959e7606c3b1f`
- Integrated foundation implementation: `108552ddd3382163a0e15c2fef7ca34e75f974fd`
- Integrated exact-integer diagnostic repair: `461a8a85de622820eb627a24b8e53749489d020a`
- Combined repaired foundation code head: `fb358fc0609b81a12cee4a1a66c2dae98edf5cae`
- Integrated registry-snapshot repair: source `0947dd7ca3b70bef2ebdf986f2336f62f5b3c08b`, integrated `fb358fc0609b81a12cee4a1a66c2dae98edf5cae`
- Integrated P0-D compiler: source `f21aaed2c97e98e63f58b77d0e301c3a2107e243`, integrated `79060abb054d82fd34ac85f000e44d9de5947d60`
- P0-D independent component proof: `79060abb054d82fd34ac85f000e44d9de5947d60`
- P1-A frozen Horizon and R0-N adjudication: `b18b262b97068086b9c61618d06a2d14a50046cd`
- P1-A repair 1/2 replacement source: `587e3fc41a0a0fbf9ef3d7e069d8eca591006001` (I0-reviewed NO-GO; not integrated)
- Integrated P1-A program compiler: source `ab64e0219c11b5fc4155703eb8ade32939f90064`, integrated `b7316be8befba64df0a576df6b7fd8cdcf9b9873`
- P1-B frozen Horizon and dispatch base: `4ed975795a4917fe40ed0203c981980ac5bea9cc`
- P1-B initial source: `ab02297c052912cb027ac6963b37667bf61f3e97` (I0-reviewed NO-GO; not integrated)
- Integrated P1-B assembly plan: repaired source `8e94454c63a76fdc1505d8ec3ec6341128448aab`, integrated `daad22796a6b178ba9fdb33d0e444b5be0f830e7`
- P1-B independent proof: exact base `e11a07c0cc5c00e956c3d537ac94272f120621c0`; the sidebar R1-B task is visibly incomplete after a temporary-file approval stall, while the exact-base internal Sol/max replacement returned GO with zero findings
- P1-C frozen Horizon and planned common dispatch base: `02cff1ca76b02969b91eac6e889de78c7a4ef9e2`
- P1-C source chain: initial `c263d95830df8284ab55c82dbd771f4690301283`; repair 1/2 `8524c2337a408e69be215a78044d54d6af968a6b`; repair 2/2 `ba6fc4e16d62b5d2b1b6066694930e9cd1dede1b`; convergence closure `3ee920a4659d7bb37f09ed094fe5c0f9651805e1`
- Integrated P1-C chain: `c9b9480` -> `bfc1d80` -> `d78201d` -> `7ee65c3ea5c50278b17dce63f2569e22412e0ebd`
- P1-C integrated proof: 86 focused and 348 combined tests in both normal and 640-digit modes; 478 v3; 667 repository; both Ruff configurations, focused format, ancestry, ownership, and static gates green
- Workflow proposal integrated: `9b26574f52c39be756e2cdeb275dcfe7691e5bc4`
- Generic-core decision and common dispatch base: `f32e7e5b8eb02f4e58a410871f929a4df22e5ca9`
- Integrated P2-H generic proof: source
  `b1789d186a85be3505e359a020cb203888f3ea76`, integrated
  `f140b9eba0147c54993f35eaaea1e11d7f8204f4`
- S2-A simplified proof: `099b51f88022f70ad11c4cecfca35c4a84ecf41e`
- Workflow routing closure: `c5f83b791252f72afa7cc8271e2c51f491c22f1c`
- Active milestone: G1 direct unexported Q8 `ModelSpec -> CompiledSystem` compiler
  on the exclusive four-commit generic-spine branch rooted at `04baa4a`; no feature
  writer and no G2 work starts before G1's two independent reviews return GO

## Exact next safe action

The independently green Phase 1 reference remains at `7ee65c3`. That product path
passed 86 focused and 348 combined tests in both normal and 640-digit modes, 478 v3 tests,
667 repository tests, both Ruff configurations, focused format, and independent
physics, state-lifecycle, and public-contract reviews with zero accepted findings.
Those results are retained evidence; they were not rerun during this documentation
realignment.

The former P2-A worktree `/private/tmp/pyfem-p2a-7ee65c3` no longer exists. Git
records its worktree metadata as prunable, and branch `agnet/p2a-mixed-blocks`
still points exactly to `7ee65c3` with zero child commits. The six partial edits
recorded in `48fa0d1` were temporary uncommitted state and are gone. No product
commit or test evidence was lost from the main branch.

More importantly, [generic_core.md](generic_core.md) supersedes the old frozen
P2-A through P2-G plan. The Phase 1 implementation demonstrated the intended
ownership, state, balance, and result invariants, but the path from the closed
foundation `c50ca70` to `7ee65c3` added 14,572 production lines and 6,323 focused
slice-test lines for one stateless Q8 flow. The program, assembly, and result
layers also reconstruct compiler-owned meaning through shared private validators.
Native-width Q8/Q4/T3 blocks would not by themselves correct that extension seam.

The active D2-A decision makes entity blocks, discrete spaces, bound operator
blocks, typed contribution channels, coordinate maps, and observations the generic
semantic IR. Legacy element/material/section/model/load categories become authored
builders or evaluator composition choices. Validation is concentrated where data
changes authority or representation; downstream internal code checks identity,
generation, capabilities, and numerical preconditions without recursively
recompiling trusted carriers.

The R2-E/P2-H/R2-F portfolio completed from exact clean base `f32e7e5`:

1. R2-E mapped all 154 rows: 95 direct, 56 composed, two questionable
   (`MODEL-RVE-HOMOG`, `ROM-POD`), and one deliberately not representable
   (`V3-PROTOTYPE-CARRIER`). It found no seventh finite-element concept. The
   design is amended so observations are solve-free, auxiliary RVE tangent solves
   are explicit execution producing derivative channels, ROM maps are versioned
   artifacts, and ecosystem/adaptor rows remain outside the numerical IR.
2. P2-H source `b1789d186a85be3505e359a020cb203888f3ea76` proved Q4/T3
   continuum, directional spring, and program-owned point loads through one typed
   space/operator/channel boundary. Seven focused tests passed in both digit modes
   and 485 v3 tests passed in both modes. Its initial 812 nonblank production and
   453 nonblank test lines remain the pre-simplification evidence baseline; S2-A's
   accepted successor is recorded below.
3. R2-F selected one compiler/assembly vertical cut over an adapter or permanent
   second backend. Its future card owns 27 production and six test paths, deletes
   four predecessor files, and requires at least 5,500 gross/1,500 net production
   lines plus 900 trust-only test lines removed while retaining fresh numerical
   verification.

I0 accepts the numerical representation with the recorded amendments and accepts
R2-F strategy B in principle. S2-A integrated the unexported P2-H proof, then
removed accidental owner/role duplication, mirrored IDs, repeated factory
ceremony, redundant proof scaffolding, and naming drift. The result is 649
nonblank production lines and 360 nonblank focused-test lines, down from 812 and
453, without weakening typed ports/channels, native-width blocks, attribution,
independent oracles, or the normalize-once trust boundary. The independent final
review returned GO with P0/P1/P2 `0/0/0`; focused tests passed in both digit modes,
the v3 and repository suites passed 485 and 674 tests, and both Ruff, format,
ancestry, path, semantic-branch, exact-head, and clean-tree gates passed. The
polished work stopped here as requested. The user explicitly resumed work on
2026-08-16. The old P2-A writer remains superseded; the next production writer may
start only after the direct vertical cut is refreshed against `099b51f`, frozen at
the current clean head, and reconciled with an implementation-parity audit.

R3-A grades only 4 of 154 capabilities as strict public behavioral parity, 57 as
bounded executable coverage, and 139 as implemented-or-design-representable. The
complete row ledger is preserved in
[the R3-A evidence](evidence/2026-08-16-r3a-behavior-parity-audit.md). D3-A and
R3-B agree that feature writers must not compete with replacement of the current
Q8 production authority. R3-A's proposed immediate mixed-width public slice is
therefore deferred until the generic spine is the sole public path.

D3-A's [four-commit cut](evidence/2026-08-16-d3a-generic-spine-cut-freeze.md)
is authoritative. One exclusive branch rooted at exact `04baa4a` owns G1 through
G4. G1-G3 are unexported semantic cuts and are never integrated independently; G4
atomically switches the public authority and deletes both predecessor and proof
implementations. The accepted post-cut feature order is the state-first sequence in
[R3-B](evidence/2026-08-16-r3b-feature-frontier.md): real operator-local state,
typed nonlinear static with an incompatible second history schema, thermal/
thermoelastic multiple spaces, then mass/capacity evolution.

### D3-A frozen G1 writer card

G1 is the sole authorized implementation packet. Its exact parent is
`04baa4a79cbcddff888c57b239df5202e2b41424`; it returns exactly one direct-child
commit and owns only:

- new `pyfem/v3/model/operator.py`;
- new `pyfem/v3/model/system.py`;
- new `pyfem/v3/compile/system.py`;
- new `pyfem/v3/compile/continuum.py`;
- `pyfem/v3/spec/diagnostics.py`; and
- new `test/v3/test_v3_q8_generic_flow.py`.

It directly compiles normalized `ModelSpec` into an unexported `CompiledSystem`
with ordered multiple spaces, an open non-enumerating operator header, typed ports
and channels, a real zero-width operator-state schema, captured registry identity,
bounded diagnostics, owned arrays, attribution, and the qualified Q8 geometry and
convention oracles. It accepts no `CompiledModel`, changes no public export, and
performs no assembly or solve. Production is capped at 1,800 physical lines across
the five production paths; the focused test is capped at 650 physical lines.

The exact six required tests and all stop conditions are frozen in D3-A section 7.
At minimum they prove direct generic Q8 compilation, hard-coded shape/gradient/
quadrature conventions, disjoint native maps for mechanical-only, thermal-only,
coupled, and nonstandard-order spaces, detached metadata-free ownership and
attribution, scale/translation-stable geometry, and compile-boundary rejection of
invalid orientation and relative singularity. G1 stops before commit if it needs a
closed payload union, a singular system space, a predecessor input or adapter, a
downstream Q8 branch, a private cross-layer diagnostic import, an unowned path, or
a weakened Phase-1 oracle.

After G1 returns, one independent finite-element numerical reviewer and one
independent architecture/trust reviewer inspect the exact source commit. Accepted
findings return to the same writer; only a joint GO authorizes G2 on the same
exclusive branch. No feature writer is active.

### Basis for the selection

The original P1-B writer returned a valid terminal source and callback, but I0's
independent carrier review found two P1 gaps while the numerical and architecture
reviews returned GO: rebuilt compiled-model identity could hide altered Q8 parent
gradients, and metadata-bearing NumPy float64 binding outputs passed the exact-dtype
check. Both findings are accepted. The original repair turn was interrupted before
editing when it requested an avoidable destructive worktree transition. A forked
replacement then inherited the app-removed worktree path and was also stopped before
editing. Both are visibly marked `INCOMPLETE`; neither changed the main branch.
Replacement writer P1-E closed both findings in exact source `8e94454`, which I0
integrated as `daad227`. Integrated proof is 34 focused and 152 combined tests in
both digit modes, 27 named references, 392 v3 tests, 581 repository tests, both Ruff
configurations, focused format, zero forbidden imports, zero determinant-absolute-
value repair, clean diff, and clean status. The first sidebar reviewer then stalled
on an unnecessary temporary-file approval and is visibly marked `INCOMPLETE`; its
partial evidence is supporting only. An exact-base internal Sol/max replacement
independently read the source/contracts, reran 34 focused and 152 combined cases in
both digit modes, all 27 references, both Ruff configurations, format and static
gates, and returned GO with zero P0/P1/P2 findings from a clean tree. Only the
existing SciPy sparse-format and cold-cache Numba parallelization notices remain.

P1-A final repair source `ab64e02` has valid terminal signals, exact one-commit
ancestry and ten-path ownership. I0 reproduced 72 focused tests in both digit
modes, 228 combined P0-D/P1-A tests in both modes, 14 reference tests, both Ruff
configurations, focused format, a zero-hit forbidden-import scan, 358 v3 tests, and
547 full-repository tests. Independent carrier review returned GO with zero
P0/P1/P2 findings. The integrated commit is `b7316be`; P1-B may now consume its
private validation seam without copying that validation logic.

## Superseded large-chunk plan — historical Phase 1/2 decomposition

> **Superseded 2026-08-16.** This section records how Phase 1 was completed and
> how the old P2-A through P2-G plan was reasoned about. It is not executable
> authority. The live bounded action is in
> [generic_core.md](generic_core.md#11-current-bounded-simplification) and the exact next
> action above.

This is the adopted acceleration plan, not authority to bypass a Horizon or merge
gate. It targets one long execution pass and pauses after I2's combined Phase 2
proof. An accepted correctness finding, a required shared forbidden path, or a
failed exact-base/clean-state gate stops only the affected wave and returns the
smallest repair or prerequisite.

Three read-only Sol/max planning lanes independently checked the clean selection
base `1957902109ab6df6d6050d255b8bbf305d694a61`: one froze the P1-C state/public
boundary, one derived an exact rational Q8 solve and verification matrix, and one
constructed the Phase 2 dependency DAG. They edited no repository file. I0
adjudicated their recommendations into the plan and P1-C Horizon below.

### Critical path and parallel lanes

```text
P1-B provisional E2
  -> P1-C writer ───────────────> serial integration -> R1-A -> I1
       || R2-A Q4/T3 research                         |
       || R2-B ownership/backend research             v
                                             P2-A mixed blocks
                                               -> R2-C common-API review
                                               -> I0 leaf-Horizon freeze
                                               -> [P2-B material slots
                                                   || P2-C active fields/DOFs]
                                               -> serial integration/Horizon freeze
                                               -> [P2-D model boundary
                                                   || P2-E program boundary]
                                               -> serial integration
                                               -> P2-F complete recipe union
                                               -> P2-G sparse-slot backend
                                               -> R2-D -> I2 -> pause
```

Best case is eleven worker/reviewer dependency-depth stages: `P1-C -> R1-A -> I1`, then
`P2-A -> R2-C -> P2-B/P2-C -> P2-D/P2-E -> P2-F -> P2-G -> R2-D -> I2`.
Parallel brackets consume one stage, not two. Serial I0 adjudication, integration,
and Horizon-freeze gates occur between those stages. Every accepted repair adds one
bounded worker stage. I0 always retains one of the four available slots for
exact-base checks, adjudication, integration, the live ledger, and combined proof.

| Coordinate and exact title | Prerequisite and owner | Output and merge rule | Exit evidence |
|---|---|---|---|
| `P1-C · linear flow — public solution verified` | Frozen card below; sole state/analysis/result/API writer | One focused source commit; integrates first and serially | Authored Q8 to immutable verified solution through reusable and one-shot flows; exact rational and analytical oracles; full Phase 1 gates |
| `R2-A · topology — Q4/T3 contracts extracted` | P1-C dispatched; read-only topology research | Literal Q4/T3 interpolation, quadrature, element oracles; mixed-partition failure cases; smallest common block API; no merge | Terminal report and callback only; no grade advance |
| `R2-B · ownership — Phase 2 leaves separated` | P1-C dispatched; read-only ownership/backend research | Disjoint material-slot, field-layout, boundary-recipe and sparse-backend seams; recipe-union and memory measurement design; no merge | Terminal report and callback only; no grade advance |
| `R1-A · linear slice — physics and state verified` | P1-C integrated and I0-focused gates green; three disjoint read-only reviewers | Physics/balance, state lifecycle, and public-contract correctness reviews; no edits | Combined GO required on identities, generations, balance/reactions, storage isolation, failed acceptance, and fresh strong verification |
| `I1 · linear slice — public proof complete` | R1-A terminal and accepted findings repaired | Serial adjudication, combined proof, contract freeze, and ledger update | Exact Phase 1 component slices reach E3 only here; schedules, nonlinear history, adapters, prototype retirement, and root compatibility remain outside the claim |
| `P2-A · mixed blocks — Q4/T3 composition explicit` | I1 frozen plus R2-A/R2-B reports; sole compiler/block owner | Shared mixed-block carrier/compiler/assembly interfaces; serial integration | One genuine Q8/Q4/T3 mixed solve; heterogeneous local sizes; no padding, node-count dispatch, solver branch, or new whole-model optional field |
| `R2-C · mixed blocks — common API verified` | P2-A integrated; independent read-only owner | Review the proposed leaf-builder/recipe interfaces required by later writers; no edits or authority change | GO on topology meaning, partitions, ordering, attribution, identities, and source mapping |
| Serial I0 Horizon gate (no new task ID) | R2-C terminal and every finding adjudicated | Existing integration owner freezes exact interfaces, paths, bases, merge order, and separate P2-B/P2-C cards | Required before either parallel leaf writer is dispatched |
| `P2-B · material slots — region meaning stable` | R2-C GO plus I0-frozen card; material-region/slot owner | Disjoint new leaf modules and focused test; parallel with P2-C, merged first | Two regions differ as expected; layered semantic slots survive block/chunk/thread reorder; missing/overlap fails |
| `P2-C · field layout — active DOFs exact` | R2-C GO plus I0-frozen card; field-layout owner | Disjoint new leaf modules and focused test; parallel with P2-B, merged second | Nonstandard component order and mechanical/thermal/coupled structural fixtures allocate only active entity-field pairs; no thermal-physics claim |
| `P2-D · model boundary — fixed physics explicit` | P2-B/P2-C integrated plus a serial I0 Horizon freeze; model-boundary owner | Disjoint model-owned boundary recipe; parallel with P2-E, merged first | Explicit entity, orientation, measure, additive force/operator recipe, provenance, and no state mutation |
| `P2-E · program boundary — dead loads explicit` | P2-B/P2-C integrated plus a serial I0 Horizon freeze; program-boundary owner | Disjoint program-owned boundary recipe; parallel with P2-D, merged second | Repeated boundary and nodal contributions add; parameter derivative literal; field/orientation/measure errors fail; no follower/schedule claim |
| `P2-F · assembly plan — recipe union complete` | P2-B through P2-E integrated; sole assembly/prepared-analysis owner | Serial wiring of shared compiler exports, assembly preparation, and Phase 1 API | Final plan is the exact union of domain, model-boundary, program-boundary, affine reduction, request channels, and backend policy; mixed model solves publicly |
| `P2-G · sparse backend — slots equal COO` | Complete P2-F recipe union; sole backend owner | Frozen backend hook plus reusable sparse-slot implementation; serial | Slot values equal the COO oracle, topology is reused, duplicates/zero-free cases work, and retained/peak memory is reported; backend defines no physics |
| `R2-D · block generality — Phase 2 independently verified` | P2-G integrated; independent read-only owner | Whole Phase 2 finite-element correctness review | GO required before I2 |
| `I2 · block generality — combined proof complete` | R2-D terminal and accepted findings repaired | Serial combined gates, ledger adjudication, and pause | Phase 2 exit: second/third formulations required no solver branch or whole-model padding; exact capability slices only are advanced |

Phase 2 coordinates after P2-A remain proposed until R2-C reports and I0 serially
adjudicates its evidence and freezes exact interfaces, path ownership, bases, and
Horizon cards. The 154-row inventory has no dedicated row for
general fixed model-boundary physics, exact active-layout proof, layered slots, or
the sparse-slot backend. Track them as explicit component slices of existing rows;
if I1 splits a row, preserve its original capability ID as lineage. Structural
thermal fixtures do not advance `FORM-THERMAL`, `FORM-THERMOMECH`, or
`FORM-THERMAL-BC`; fixed boundary recipes do not advance the later interaction
meaning of `ASM-MODEL-ACTIONS`; and no Phase 2 packet advances
`V3-PROTOTYPE-ASSEMBLY` by implication.

### Four-slot and false-parallelism policy

- P1-C is one writer because the first state lifetime, transaction authority,
  reaction convention, result snapshot, verification boundary, and public API are
  one semantic cut. Splitting them would create mutually invented interfaces.
- P2-A is one writer because Q4 and T3 share partition, block, entity-map, compiler,
  and assembly-iteration owners even though their leaf kernels differ.
- P2-B and P2-C may run together only after R2-C reports and I0 freezes the leaf
  interfaces and exact paths; neither writer may touch `model/compiled.py`,
  `compile/model.py`, a shared export, the live ledger, or the same test file.
- P2-D and P2-E may run together only as new leaf recipes. Neither may wire itself
  into `assembly/prepare.py`; P2-F owns that serial union.
- P2-G cannot start against a partial recipe set. Backend optimization after P2-F
  preserves semantics; starting earlier would silently select them.
- Every `__init__.py`, shared test aggregation, integration, repair adjudication,
  execution-ledger edit, and phase-grade change has one serial owner.
- During P1-C the other two slots are read-only Phase 2 research. During a Phase 2
  writer wave use at most two disjoint writers and one read-only task. Shared-owner
  stages run one writer only.

## P1-C Horizon card — `HORIZON_FROZEN`

**ID/title:** `P1-C · linear flow — public solution verified`

**Outcome and ownership invariant:** compose the exact P0-D model, P1-A program,
and P1-B assembly plan into the first typed, verified Q8 linear-static flow.
`PreparedAnalysis` is the sole live owner of workspace reuse and one-accept
authority. Physical, evolution, trial, committed, ledger, and solution values are
immutable detached snapshots and never workspace views. The transaction boundary
is real even though this Phase 1 material and formulation have zero-width history.
No path-dependent Phase 3 behavior enters this packet.

**Selection base and dispatch parent:** the read-only design was checked at clean
base `1957902109ab6df6d6050d255b8bbf305d694a61`. The exact E1 proof commit and
common writer/research dispatch base is the single clean documentation commit
`02cff1ca76b02969b91eac6e889de78c7a4ef9e2`. P0-D, P1-A, P1-B, and their
independent proofs are required and complete. No concurrent state, request, result,
reaction, or public-API writer is permitted.

**Exact request and preparation contract:**

- `LinearStatic()` is one exact frozen/slotted zero-field request. Subclasses and
  foreign carriers fail. Its versioned manifest fixes the Phase 1 numeric,
  verification, and workspace policies; tolerances are not premature public knobs.
- `prepare_analysis(model, program, LinearStatic())` validates the complete exact
  compiled carriers, translates to P1-B's internal contribution request, calls the
  public P1-B preparation boundary exactly once, derives conservative capabilities,
  and creates a fresh `PreparedAnalysis.instance_id`.
- The prepared owner records exact model/program/plan content and live identities,
  the exact versioned request manifest, one algebraic `EvolutionLayout`, and a
  private serialized workspace.
  Two prepared analyses over content-equal inputs have distinct identities and
  cannot share transactions, state, solutions, caches, or factorization storage.
- The Phase 1 backend is a private correctness-first dense symmetric Cholesky of
  the P1-B canonical reduced COO operator. It is not a semantic carrier or public
  backend promise. Finiteness is required. The original canonical `K_q` remains
  the audit operator. P1-C's backend admission check, matching P1-B's `64 eps`
  local symmetry policy, is exactly
  `||K_q - K_q.T||_max <= 64 eps ||K_q||_max`, with an exactly zero right side
  requiring exact symmetry; otherwise preparation fails. Only after that check,
  the private solver projection is exactly
  `K_solve = 0.5 * (K_q + K_q.T)`. This projection is recorded as backend policy,
  never overwrites the audit ledger, and cannot hide a larger inconsistency.
  Cholesky must succeed; with `s = ||K_solve||_infinity`, `s > 0` and every
  unscaled Schur-complement pivot `d_i = L_ii**2` must satisfy `d_i > 1e-12 s`;
  and the fresh post-solve reduced residual against original `K_q` must pass the
  frozen verification normalization. Nonsymmetric beyond the frozen bound,
  indefinite, singular, near-singular, malformed, or nonfinite systems fail before
  acceptance.
- Before allocating, checked integer arithmetic must prove that the dense operator,
  factor, and required scratch fit a versioned `256 MiB` Phase 1 workspace budget.
  Larger reduced systems fail deterministically with a backend-capacity diagnostic
  and no state acceptance; they never rely on `MemoryError` as policy. P2-G removes
  this reference-backend ceiling rather than silently raising it.
- A bitwise-identical freshly evaluated canonical reduced operator may reuse the
  private factorization. This compiled slice declares a constant tangent, so any
  changed operator under the same exact model/program/plan identities and request
  manifest is a capability or implementation-identity failure and fails closed; it
  never replaces the cache. An empty reduced system bypasses factorization. P2-G
  replaces this reference backend with measured sparse slots; P1-C makes no scale
  or production-backend claim.
- Workspace may own reduced numeric buffers, a factorization, scratch arrays, and
  counters. It never owns authoritative fields, histories, reactions, convergence
  truth, accepted state, or result arrays.

**State and transaction spine:**

```text
PhysicalState
  model identity/fingerprint + schema + StateGeneration
  detached read-only full primary vector
  exact per-block material histories shaped (..., 0)
  exact per-block formulation histories shaped (..., 0)

EvolutionState
  prepared identity + exact request manifest/schema + same generation
  exact bound ProgramEvaluation
  algebraic field classification, accepted step index
  detached predictor and actual full increment

ProgramHistory
  exact program identity/schema; empty in P1-C

CommittedAnalysisState
  prepared identity + accepted generation
  PhysicalState + EvolutionState + empty ProgramHistory

StepTransaction
  unique transaction ID and exact prepared/model/program/plan identities
  exact request manifest
  exact base state/generation, bound target ProgramEvaluation
  retry=0, cutback=0, typed linear predictor

TrialAnalysisState
  unique trial ID and exact transaction/base identity
  prospective base.next_accepted() generation
  detached candidate physical/evolution/history snapshots
  candidate-specific linear balance ledger

LinearBalanceLedger
  unique ledger ID
  exact prepared/model/program/plan identities and fingerprints
  exact request manifest
  exact transaction and trial IDs
  exact base and candidate StateGeneration values
  exact bound ProgramEvaluation and detached candidate-specific arrays
```

`initialize(point=...)` requires an explicit `ProgramPoint`, creates one fresh
generation-0 lineage, evaluates that point's affine offset, initializes the full
primary vector to `u_bar`, and allocates the exact zero-width history arrays.
P1-C invents neither a schedule nor an implicit initial coordinate.

`PreparedAnalysis` keeps a locked private registry of transactions and consumed
base generations. Multiple trials or transactions may reference one base, but
only the first fully valid accept can consume it. Trials have distinct IDs even
when they share one prospective successor generation. Discarding a trial or
abandoning a transaction is byte-for-byte inert. Acceptance validates the entire
trial first, atomically consumes the exact base, copies trial arrays into distinct
committed storage, and closes the transaction. Double accept, a stale sibling,
foreign prepared analysis, wrong transaction, wrong trial, non-successor or forged
generation, and any trial/committed alias fail without partial state change.

The explicit reusable flow accepts an initialized base. A convenience prepared
solve may accept both explicit `initial_point` and target `point`, internally create
a fresh lineage, and reuse only workspace/factorization. Repeated independent
prepared solves therefore do not consume one another's generation-0 base. Every
successful direct solve—including zero-load and zero-free-DOF cases—accepts exactly
one transition `0 -> 1`; every failed solve accepts none.

**Frozen equilibrium, reaction, and work conventions:**

```text
K_q q = b_q
u = P q + u_bar
f_int = K u
r_full = f_ext - f_int
r_q = P.T r_full
c_constraint = f_int - f_ext = -r_full
balance = f_ext + c_constraint - f_int
```

The solution records full and reduced residuals, internal and external forces, the
full constraint-force vector, direct reactions, constraint violation, reduced
coordinates, constraint work, normalizations, and tolerances. Direct reactions are
`c_constraint` entries only for authored direct prescribed-DOF targets. P1-C does
not reinterpret one as a generalized force conjugate to a prescribed root that
also drives an MPC chain. P1-C does not report MPC multipliers because it has not
selected an independent dual basis; constrained residual entries are never
mislabeled as unique multipliers. Computing `K u` is the constant-linear audit
slice of `ASM-INTERNAL`, not a claim that the general nonlinear internal-force
ledger is implemented.

Every retained and recomputed balance ledger carries the exact prepared,
transaction, trial, base-generation, and candidate-generation provenance above.
`verify_record()` rejects a numerically identical ledger from a sibling trial or
generation before examining its values.

Use scale-relative verification with infinity norms and no unit floor:

```text
F_full = max(
  ||f_ext||,
  ||f_int||,
  ||c_constraint||,
  ||K|| * ||u||,
)
F_reduced = max(
  ||b_q||,
  ||K_q q||,
  ||P.T|| * F_full,
)
U_reconstruction = max(||u||, ||P q||, ||u_bar||)
W = max(
  |f_ext.T @ u|,
  |f_int.T @ u|,
  |c_constraint.T @ u|,
  F_full * max(||u||, ||u_bar||),
)
```

For any error/scale pair, a zero scale requires an exactly zero error; otherwise
compare the normalized ratio to `1e-12` using overflow/underflow-safe binary64
scaling rather than multiplying a possibly subnormal scale by the tolerance.
Reduced equilibrium and `P.T @ c_constraint` use `F_reduced`; full balance and
force-ledger comparisons use `F_full`; field reconstruction uses
`U_reconstruction`; and work uses `W`. Each authored prescribed or affine-tie
constraint uses its own algebraic row scale: the maximum magnitude of the target
value and every evaluated term in that row, with the same exact-zero rule. This
policy rejects an order-one wrong displacement even when valid stiffness and load
scales are far below one. P1-B's frozen operator/RHS tolerances remain unchanged.

**Immutable solution and verification:**

- `Solution` retains immutable model/program/request references, prepared
  provenance but no mutable workspace, the final committed state, the accepted
  transition record, convergence record, and detached balance/reaction ledger.
  It retains no factorization or cached solver residual as proof.
- `verify_record()` validates exact carriers, live/content identities, schemas,
  generation succession, accepted-transaction provenance, finite read-only
  detached arrays, convergence record, and internal algebraic consistency. It does
  not claim freshly recomputed equilibrium.
- `verify()` first verifies the record, then creates a fresh P1-B plan/evaluation
  and fresh verification workspace. It independently reconstructs `P`, `u_bar`,
  `K`, `f_ext`, constraint violation, reduced residual, full constraint force,
  direct reactions, work, and balance. It never reads the solve factorization,
  cached operator, stored norm, or mutable prepared workspace.
- The solve performs the same fresh candidate verification before atomic accept.
  Later `verify()` repeats it independently.
- Malformed carriers raise one structured verification error. A well-formed record
  whose recomputed numerical checks fail returns an immutable report with
  `passed=False`, named checks, tolerances, scales, and measured norms.

**Public flows and namespace decision:**

```python
from pyfem.v3.api import LinearStatic, prepare_analysis, solve

analysis = prepare_analysis(model, program, LinearStatic())
initial = analysis.initialize(point=base_point)
solution = analysis.solve(initial=initial, point=target_point)
report = solution.verify()
```

```python
solution = solve(
    model_spec,
    program_spec,
    LinearStatic(),
    registry=q8_reference_registry(),
    initial_point=base_point,
    point=target_point,
)
```

The one-shot path uses the same normalize, compile, prepare, initialize,
transaction, solve, accept, and verify path and requires explicit registry
injection. The approved Phase 1 public surface is `pyfem.v3.api`; neither
`pyfem/__init__.py` nor the current prototype-heavy `pyfem/v3/__init__.py` changes.
Consequently importing a `pyfem.v3` submodule still executes that existing package
initializer and may load prototype modules; P1-C makes no semantic call to them and
does not falsely claim their import-time retirement. Root exposure, initializer
cleanup/lazy compatibility, adapters, and prototype deletion remain later explicit
decisions.

**Coverage contracted at component-qualified E1:** `STATE-LAYOUT` for exact
stateless zero-width material/formulation layouts only; `STATE-GLOBAL` for the Q8
primary/evolution/workspace/result ownership split; `STATE-TRIAL` for detached
stateless candidate evaluation; `STATE-TRANSACTION` for one atomic algebraic
accept/discard; `STATE-EVOLUTION` for one explicit program-point transition;
`V3-GENERATION` for successor and sibling-accept enforcement; `ANAL-DISPATCH` for
exact `LinearStatic`; `ANAL-LINEAR` for the constrained Q8 solve; `RES-SOLUTION`
for the in-memory verified Q8 result; and `ASM-INTERNAL` for the constant-linear
`K u` audit derivation only. Successful code/integration/review may reach
provisional E2; only I1's complete public-flow proof may assign E3 to these exact
slices and the existing P0-D/P1-A/P1-B consumer paths.

All nonzero/path-dependent portions of the state rows remain E0, as do
`STATE-RESTORE`, `STATE-NODAL-ACCUM`, `PROG-SCHEDULE`, general
`ASM-INTERNAL`, nonlinear/follower/geometric contributions, all other
analysis/result families, adapters/ecosystem rows, and every
`V3-PROTOTYPE-*` retirement row.

**Owned paths:** new `pyfem/v3/api.py`; new `pyfem/v3/analysis/**`; new
`pyfem/v3/model/state.py`; minimal state exports in
`pyfem/v3/model/__init__.py`; new `pyfem/v3/results/**`; and new
`test/v3/test_v3_linear_analysis.py`. Splitting source files inside the two new
packages is allowed. One writer owns every path and returns one focused commit.

**Forbidden paths and non-goals:** `.agents/v3/**`, root configuration and lock
files, `pyfem/__init__.py`, `pyfem/v3/__init__.py`, existing `spec/**`,
`compile/**`, and `assembly/**`, existing model carriers except the single export
edit, prototype `types.py`, `pack.py`, `_prototype_assembly.py`, `registry.py`,
prototype `solver/**` and `io/**`, numeric kernels, and existing tests. Do not add
schedule stepping, authored initial conditions, checkpoint/rebind, path-dependent
history, nonlinear solve, line search/cutback, distributed/follower loads, general
constraints, output projectors, root compatibility, CLI/GUI, RVE/FE2, ROM, sparse
slots, or a wall-clock acceptance threshold.

**Independent exact Q8 oracle:** one unit-square Q8 uses the frozen P1-B node order,
`E=1`, `nu=0`, unit thickness, direct conditions `u_1x=u_1y=0`, affine MPC
`u_3y=u_3x+lambda`, and nodal load `f_5y=1`. At `lambda=1`, the exact rational
solution derived independently from literal `K=M/360` is:

```text
[0, 0, 29/12, 67/24, 29/6, 35/6, -23/24, 33/4,
 -7, 32/3, -77/12, 27/8, -35/6, -7/6, -37/24, -7/12]
```

The test pastes those fractions as literals and never calls production assembly or
solve to manufacture them. Its exact ledgers are:

```text
f_int = [-1,0,0,0, 1,-1,0,0,0,1, 0,0,0,0,0,0]
r_full = [1,0,0,0, -1,1,0,0,0,0, 0,0,0,0,0,0]
c_constraint = [-1,0,0,0, 1,-1,0,0,0,0, 0,0,0,0,0,0]
direct reactions = [-1, 0]
c_constraint.T @ u = -1
f_ext.T @ u = 32/3
f_int.T @ u = 29/3
f_ext.T @ u + c_constraint.T @ u = f_int.T @ u
P.T @ r_full = 0
c_constraint.T @ (du/dlambda) = -1
```

The affine derivative is the exact rigid-rotation field `du/dlambda=(-y,x)`, lies
in the nullspace of `K`, and leaves the constraint force unchanged. The independent
analytical oracle is the rational calculation itself. A calibration using current
P1-B reference assembly and a NumPy dense solve—not an independent assembly oracle
or the future P1-C solve—differed from the rational displacement by at most
`3.9e-14` and reduced equilibrium by at most `6.3e-15`.

**Required correctness matrix:**

1. the rational one-cell solve, reactions, work, rigid affine derivative, and
   global force/moment balance;
2. zero load with nonzero rigid offset, displacement-only, direct prescriptions,
   and a fully prescribed zero-free system that performs no factorization;
3. additive and constrained-DOF loads, affine MPC chains and nonzero offsets;
4. no-constraint identity reduction prepares successfully but solving rejects the
   exact rank-13 Q8 operator even for zero RHS; other near-singular, indefinite,
   nonsymmetric, nonfinite, malformed, unsupported, and capability-incompatible
   inputs also fail without commit;
5. content-equal but live-distinct model/program/prepared owners, altered or foreign
   request carriers, mismatched plan/request, and foreign state/transaction/trial
   carriers fail before workspace use or acceptance;
6. repeated prepared solves reuse only eligible factorization storage, while prior
   state, trial, ledger, and solution arrays remain detached, read-only, unchanged,
   and memory-disjoint;
7. same trial twice, sibling transactions, stale base, forged generation, discard,
   and failed solve prove exactly-once acceptance and byte-for-byte inert failure;
8. `verify_record()` catches identity, generation, provenance, convergence, ledger,
   and storage corruption;
9. for a coherent copied record with `u[2] += 1e-6`, fresh `verify()` recomputes
   `||P.T @ r_full||_infinity = 23/11250000`, approximately `2.04444e-6`, and
   fails the `1e-12 F_reduced` threshold; it likewise catches a broken affine constraint,
   changed force/reaction balance, and false convergence even when record-only
   checks are coherent, while poisoned solve caches do not affect it;
10. the book PatchTest8 analytical field
    `u_x = 1e-3 x + 5e-4 y`, `u_y = 5e-4 x + 1e-3 y` is a primary independent
    new-flow oracle at `rtol=0`, `atol=1e-12`; legacy/prototype agreement remains
    secondary disagreement evidence at `rtol=1e-10`, `atol=1e-12`; and
11. exact IDs and source locations remain total and bounded under
    `PYTHONINTMAXSTRDIGITS=640`.

**Required gates:** focused P1-C tests and combined P0-D/P1-A/P1-B/P1-C tests in
normal and 640-digit modes; P1-B's 27 named references plus the independent solve
cases; selected reusable and one-shot public flows; both Ruff configurations and
focused format; all `test/v3` and the full repository suite; source-dependency
scans proving no direct P1-C prototype/solver/I/O/legacy/root dependency; storage
alias and determinant-absolute-value scans; `git diff --check`; exact one-commit
ancestry and path ownership; final clean status. Record cold prepare, first solve,
twenty alternating-point warm solves, verification, factorization reuse, and
retained/peak memory for one-cell, five-cell, and bounded 8x8 Q8 cases. Report the
P1-A full-DOF witness cost separately. These are measurements, not timing gates.

**Strongest competing design and rejection:** a purely functional immutable solver
could return a successor state without a live accept owner. It is attractive but
cannot prevent sibling transactions from accepting the same base and supplies no
exact owner for cache/factorization identity. Adding a separate public run/session
owner would introduce another lifetime before evidence requires it. P1-C therefore
assigns both one-accept authority and private cache identity to the already-required
`PreparedAnalysis`.

**Causal prototype consequence:** the P1-C source and algorithm must have zero
direct production dependency on `ProblemDefinition`, `LoadedProblem`, `pack.py`,
the global string registry, prototype assembly, prototype solver/context/state, or
adapters. The new path replaces the Q8 direct-linear meaning but deletes nothing:
prototype carrier/registry/assembly/analysis rows remain E0 because Q4/T3/3D,
nonlinear/path references, import compatibility, and existing consumers still
depend on them.

**Blocked condition:** stop and return the smallest counterexample if P1-B cannot
supply fresh exact contributions without changing its carriers; if the one-accept
authority requires changing identity primitives; if correct full reactions require
pretending MPC multipliers are unique; if strong verification must reuse solver
caches; if an owned public flow cannot exist without a forbidden root/adapter path;
or if the frozen Q8 operator cannot satisfy the private Cholesky policy. Create a
separate prerequisite or design decision rather than widening P1-C informally.

## P2-A Horizon card — `SUPERSEDED`

> **Historical only as of 2026-08-16.** The card below was coherent for the
> topology-generalization design, but [generic_core.md](generic_core.md)
> supersedes that design before any P2-A product commit. Do not dispatch or resume
> it. Its constants remain useful topology/oracle evidence for later operator
> instances.

**ID/title:** `P2-A · mixed blocks — Q4/T3 composition explicit`

**Exact source parent:** `7ee65c3ea5c50278b17dce63f2569e22412e0ebd`.
The later I1 ledger commits are coordination evidence and do not change this product
parent. One sole writer owns P2-A. No concurrent compiler, model, assembly, state,
analysis, result, API, or mixed-block writer is permitted.

**Objective:** extend the proved domain-only linear-static slice from one Q8 block
to explicit homogeneous Q8, Q4, and T3 blocks and prove one genuine mixed public
solve through reusable and one-shot flows. The solver, transaction, reaction,
balance, convergence, and fresh-verification algorithms remain topology-blind.
Do not add a temporary Q8 guard: that would be the solver topology branch this
packet is required to prove unnecessary and would create later removal work.

**Bounded model contract:** accept any nonempty combination of these descriptors,
with one or more cells per homogeneous source block:

| Geometry interpolation | Topology | Quadrature | Nodes | Local DOFs | Points |
|---|---|---|---:|---:|---:|
| `serendipity-quad8` | quadrilateral | `gauss-3x3` | 8 | 16 | 9 |
| `bilinear-quad4` | quadrilateral | `gauss-2x2` | 4 | 8 | 4 |
| `linear-tria3` | triangle | `gauss-tria3-order1` | 3 | 6 | 1 |

Keep one node field `("x", "y")`, one shared plane-stress linear-elastic material,
one complete region per source block, the existing small-strain engineering-shear
formulation, zero material/formulation histories, and unit out-of-plane thickness.
Multiple materials or regions within one block belong to P2-B; active-field layout
belongs to P2-C. Canonical domain order is exact semantic
`(source_cell_block_id, source_region_id)` order; cells are semantic-ID ordered
within a block; descriptor-authored local node order is never sorted or repaired.

**Frozen schema families:** replace the Q8-only strings without aliases:

```text
pyfem-v3-compiled-model-domain-blocks-v1
pyfem-v3-prepared-assembly-plan-domain-coo-v1
pyfem-v3-physical-state-stateless-domain-blocks-v1
```

These are version 1 of new neutral schema families, not version 2 of the old
`...-q8-v1` families. Compiled-program, request, evolution-state, and empty-history
schemas remain unchanged. The persisted assembly geometry policy must describe
descriptor-driven signed audits rather than Q8 geometry.

**Registry and geometry contract:** preserve `q8_reference_registry()` and every
existing Q8 descriptor byte-for-byte. Add exact keys
`("topology", "bilinear-quad4")`, `("quadrature", "gauss-2x2")`,
`("topology", "linear-tria3")`, and
`("quadrature", "gauss-tria3-order1")`, plus a bounded
`linear_plane_stress_reference_registry()` containing the required union.
Compilation captures only descriptors required by authored blocks.

Q4 topology metadata contains exact
`reference_geometry_audit_parent_points = ((-1,-1),(1,-1),(1,1),(-1,1))`.
Evaluate its topology binding separately at those corners; never pad or alter its
four Gauss rows and add no compiled audit-point carrier. Q8 retains its nine-point
audit; T3's constant Jacobian uses its centroid quadrature row. Every audit requires
finite normalized geometry, strictly positive signed determinant, and the existing
scale-relative condition bound. Never use `abs(det J)` or reorder nodes. Resolve
bindings once per block signature, not once per cell.

**Native ownership and plan contract:** each `CompiledCellBlock.connectivity` is
the exact object also referenced by its `DomainBlock`; each domain DOF map is the
exact object referenced by its coupling recipe. No padded, ragged, object, sentinel,
or second connectivity owner is permitted. Prepared plan arrays remain separately
owned from compiled model/program arrays.

Add exactly one plan carrier:

```text
DomainCooPlan.full_raw_block_offsets: FinalizedArray
```

It is exact owning, read-only, canonical integer storage. Raw order is block, cell,
local row, local column. Do not persist reduced block offsets: the strictly ordered
`reduced_raw_source_indices` derive them from full offsets. Validate every block,
recipe, state-layout row, entity/source record, offset, and manifest entry; never
validate block zero and trust later blocks. Checked arithmetic covers summed cells,
DOFs, points, raw entries, offsets, and index capacity before allocation.

**Exact disconnected oracle:** semantic IDs canonically order one unit Q8, one unit
Q4, then one unit T3 cell. They use disjoint nodes: Q8 IDs 1..8 on `[0,1]^2`, Q4
IDs 9..12 on `[2,3]x[0,1]`, and T3 IDs 13..15 at `(4,0),(5,0),(4,1)`.
Use `E=1`, `nu=0`, and unit thickness.

```text
nodes / cells / blocks / regions        15 / 3 / 3 / 3
full / reduced DOFs                     30 / 21
direct prescribed DOFs                  9
integration points                      14
full raw / canonical entries            356 / 356
reduced raw / canonical entries         203 / 203
node offsets                            (0,8,12,15)
full-DOF offsets                        (0,16,24,30)
reduced-DOF offsets                     (0,13,18,21)
full raw block offsets                  (0,256,320,356)   # persisted
reduced raw oracle offsets              (0,169,194,203)   # derived only
integration-point offsets               (0,9,13,14)
cell offsets                            (0,1,2,3)
```

The test owns element-qualified hard-coded literals: the frozen Q8 integer operator
divided by 360, this Q4 integer operator divided by 8, and this T3 integer operator
divided by 4. They must not be generated through production bindings.

```text
Q4 = [
 [ 4, 1,-2,-1,-2,-1, 0, 1], [ 1, 4, 1, 0,-1,-2,-1,-2],
 [-2, 1, 4,-1, 0,-1,-2, 1], [-1, 0,-1, 4, 1,-2, 1,-2],
 [-2,-1, 0, 1, 4, 1,-2,-1], [-1,-2,-1,-2, 1, 4, 1, 0],
 [ 0,-1,-2, 1,-2, 1, 4,-1], [ 1,-2, 1,-2,-1, 0,-1, 4],
]
T3 = [
 [ 3, 1,-2,-1,-1, 0], [ 1, 3, 0,-1,-1,-2],
 [-2, 0, 2, 0, 0, 0], [-1,-1, 0, 1, 1, 0],
 [-1,-1, 0, 1, 1, 0], [ 0,-2, 0, 0, 0, 2],
]
```

Directly prescribe local DOFs `(0,1,3)` in every block; use no MPC in this literal
count fixture. The exact solution is `(u_x,u_y)=(x_local,y_local)`. Free external
entries equal the independent literal `K u`; prescribed entries are zero. Direct
reactions in local `(0,1,3)` order are Q8 `(-1/6,-1/6,-2/3)`, Q4
`(-1/2,-1/2,-1/2)`, and T3 `(-1/2,-1/2,0)`. Require `rtol=0`, local-operator
`atol<=2e-14`, complete-solve `atol<=5e-13`, and full force/moment balance.

**Public acceptance:** the same literal model must pass both
`compile_model -> compile_program -> prepare_analysis -> initialize -> solve` and
the existing one-shot `pyfem.v3.api.solve`. Require exact generation 0 to 1,
reusable/one-shot equality of primary values, reactions, ledgers, and verification
checks, repeated prepared solve through the same generic factorization path,
record verification, and cache-independent fresh verification that rebuilds the
mixed plan and response. Existing Q8-only flows remain unchanged. No Q4/T3 branch
may appear in analysis, results, or API.

**Ordering, provenance, and local edge cases:** authored node, block, region, cell,
and registry-order permutations preserve compiled fingerprint and physical result
after source remapping. Entity/source records cover all nodes, cells, blocks,
regions, DOFs, and 14 integration points. Live-distinct content-equal compiles have
distinct instance IDs; later registry mutation cannot change compiled meaning; all
arrays are exact dtype, metadata-free, owning, contiguous, read-only, and detached.
Cover Q4 corner-only inversion, uniform inversion, sign change, T3 degeneracy,
near-singular/nonfinite/extreme-scale geometry, wrong explicit descriptor pairing,
wrong binding type/dtype/shape/order, missing/duplicate/cross-block region membership,
wrong second/third block data, shared-node Q4/T3 COO duplicate addition, malformed
offsets/recipes/state rows/source records/manifests, declaration permutations, and
partially initialized carriers with structured diagnostics.

**Exact owned paths:**

```text
pyfem/v3/model/compiled.py             # topology-neutral wording only
pyfem/v3/model/state.py                # neutral physical-state schema and wording
pyfem/v3/compile/contracts.py
pyfem/v3/compile/model.py
pyfem/v3/compile/program.py
pyfem/v3/compile/__init__.py
pyfem/v3/assembly/contracts.py
pyfem/v3/assembly/prepare.py
pyfem/v3/assembly/reference.py
pyfem/v3/analysis/linear.py            # topology-neutral diagnostic only
pyfem/v3/results/solution.py           # topology-neutral module wording only
pyfem/v3/results/verification.py       # topology-neutral module wording only
test/v3/test_v3_mixed_blocks.py
```

The last three production paths may not change algorithms. Do not modify existing
tests, `api.py`, specs, kernels, prototype modules, solver/I/O/root paths, project
configuration, lock files, or the ledger. Add no helper module, whole-model optional
field, material slot, active-field layout, boundary recipe, sparse backend, output
projection, nonlinear/stateful meaning, compatibility shim, or timing threshold.

**Required gates:** run the new test plus model-spec, identity, model-compile,
program-compile, assembly-plan, and linear-analysis tests normally and with
`PYTHONINTMAXSTRDIGITS=640`; direct Q8/Q4/T3 shape, quadrature, stiffness, COO,
constraint, load, and MPC references; reusable and one-shot mixed flows; both Ruff
configurations; focused format; all `test/v3`; full repository pytest;
`git diff --check`; exact ancestry/path/final-clean checks. Static scans prove zero
prototype/legacy/solver/I/O dependencies, production `fem.element` dispatch,
determinant absolute-value repair, topology choice from node count or width,
padded/object connectivity, and Q4/T3 branches outside compiler/assembly. Report
existing SciPy/Numba notices separately. Measure compile, prepare, first solve, warm
reuse, verification, retained bytes, and peak bytes for the literal fixture without
using timing thresholds.

**Commit and callback:** exactly one focused commit, exactly one child of
`7ee65c3ea5c50278b17dce63f2569e22412e0ebd`, only the 13 paths above, no amend,
merge, rebase, push, install, or external action, and final clean worktree.

```text
P2-A COMPLETE · commit=<40> · parent=7ee65c3ea5c50278b17dce63f2569e22412e0ebd · paths=<exact list> · oracle=15 nodes/30 full/21 reduced; full offsets (0,256,320,356); integration offsets (0,9,13,14); 356 full/203 reduced · proof=<focused twice; references; public flows; Ruff; v3; repository; static/path/ancestry/clean> · warnings=<exact notices> · next=I0 inspect and integrate serially, then dispatch R2-C
```

After integration, one fresh read-only R2-C must independently rederive the oracle
and verify complete multi-block validation, descriptor meaning, native-width
ownership, offsets, ordering, provenance/source attribution, public solve and fresh
verification, and absence of topology branches outside compiler/assembly. R2-C GO
is required before P2-B/P2-C leaf cards freeze.

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
- Thread IDs are never reused. `R0-E` is the completed legacy-breadth inventory;
  `R0-F` stopped before review because I0 supplied a wrong expanded commit hash;
  replacement compiler review `R0-N` completed GO at the verified exact integrated
  hash. Earlier replacements likewise used the next otherwise-unreserved IDs.
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
- the later-phase registry-callable trust, restore/rebind, and path-dependent
  transaction obligations recorded below.

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
| P0-D immediate dependency cut | 18 | `verified`, component-qualified `E3` at I1 proof `7ee65c3`; compiler evidence consumed by the public linear slice and three R1-A GO reviews | 11 preserve, 7 change; exact Q8 component slice only | Extend through P2 mixed-block composition; no row-wide parity claim |
| P1-A immediate dependency cut | 3 | `verified`, component-qualified `E3` at I1 proof `7ee65c3`; affine program evidence consumed by the public linear slice and three R1-A GO reviews | 3 change; `PROG-DIRICHLET`, `PROG-MPC`, `PROG-NODAL-LOAD` | Extend through later field/material and adapter slices |
| P1-B immediate dependency cut | 5 | `verified`, component-qualified `E3` at I1 proof `7ee65c3`; assembly-plan evidence consumed by the public linear slice and three R1-A GO reviews | 5 change; component slices of `ASM-COO`, `ASM-PREPARE`, `ASM-EXTERNAL`, `ASM-GATHER`, `ASM-TANGENT` | Extend through P2 mixed-block and sparse-backend slices |
| P1-C immediate dependency cut | 10 | `verified`, component-qualified `E3` at I1 proof `7ee65c3`; integrated chain, closure, I0 proof, and three R1-A GO reviews complete | 1 preserve, 9 change; exact bounded slices listed below | Freeze and dispatch P2-A; deferred Phase 1 breadth remains separately inventoried |
| Later preserve/change portfolio | 105 | `inventoried`, `E0` | Working dispositions accepted; slice-specific E1 extraction and semantic adjudication still required | Select only when dependencies pass |
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

The three P1-A rows are:

```text
PROG-DIRICHLET        PROG-MPC              PROG-NODAL-LOAD
```

The five P1-B rows are:

```text
ASM-COO               ASM-PREPARE           ASM-EXTERNAL
ASM-GATHER            ASM-TANGENT
```

The ten P1-C rows are component-qualified rather than row-wide breadth claims:

```text
STATE-LAYOUT          STATE-GLOBAL           STATE-TRIAL
STATE-TRANSACTION     STATE-EVOLUTION        V3-GENERATION
ANAL-DISPATCH         ANAL-LINEAR            RES-SOLUTION
ASM-INTERNAL
```

Their exact E1 boundaries are frozen in the P1-C Horizon above. Nonzero local
history, schedules, restore/rebind, general nonlinear internal force, other
analysis families, exporters, compatibility, and prototype retirement remain E0.

The five P1-B rows' E1 contract is deliberately component-bounded. `ASM-COO`
covers canonical
raw Q8 element contributions and an explicitly coalesced reference operator;
`ASM-PREPARE` covers immutable request-specific topology and compatibility;
`ASM-EXTERNAL` covers P1-A nodal force and affine derivative channels only;
`ASM-GATHER` covers compiled Q8 coordinates/DOFs/material descriptors only; and
`ASM-TANGENT` covers constant small-strain plane-stress Q8 stiffness only.
Internal force, nonlinear/material state, geometric/follower tangents, distributed
loads, optimized backend storage, solve/reaction/result behavior, and prototype
retirement remain E0.

Their E1 reference is intentionally narrower than the eventual public program
surface. The exact affine relation is `u = P q + u_bar(p)`; one-master chains
compose factors and affine offsets analytically; and repeated nodal force
declarations add on the full DOF vector, including constrained DOFs. The existing
prototype/legacy constraint, MPC, and nodal-load checks pass 14 tests at
`79060abb`, while source inspection confirms the prototype's known last-write load
packing and solver-local constraint representation. P1-A preserves the physical
meaning and deliberately replaces those ownership failures. The E1 proof commit is
`b18b262b97068086b9c61618d06a2d14a50046cd`, while `79060abb` remains the exact
reference-test execution base.

I0 refines the coarse E0 dependency recorded on `PROG-DIRICHLET` and
`PROG-NODAL-LOAD`: their P1-A component proof requires the structured coordinate
schema and direct point evaluator owned inside this same packet, not schedule
stepping. `PROG-SCHEDULE` remains in the later E0 portfolio and becomes a hard
dependency only when an analysis owns stepping, acceptance, tables/functions,
restart, or schedule-driven binding. P1-A does not advance or partially claim that
row.

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
after combined proof does a fresh Sol/max R0-F task, or its explicitly recorded
replacement, independently verify the integrated block invariants. Accepted
findings return to the same writer for at most two bounded repair turns. P1-A stays
closed until that review is adjudicated.

**Blocked condition and minimum unlock:** stop without broadening scope if the
exact descriptor compatibility cannot represent the Q8 recipe, a correct geometry
audit requires modifying existing numeric kernels, the current spec cannot express
the supported slice without a new authored owner, a forbidden shared path is
required, or a physical convention other than the explicit per-unit-thickness
slice must be chosen. Return the smallest reproducible conflict; I0 decides whether
to amend design/card or create a new packet.

**Integration task ID:** `019f6f49-0b72-7d73-86da-c6b85519eeaf`.

## P1-A Horizon card — `HORIZON_FROZEN`

**ID/title:** `P1-A · program compiler — affine plan canonical`

**Outcome and ownership invariant:** `normalize_program_spec(...)` constructs a
new exact, recursively caller-detached authored program tree. `compile_program(...)`
consumes only that normalized meaning plus one exact validated `CompiledModel` and
returns a fresh immutable `CompiledProgram`. The program records both the model's
live instance identity and content fingerprint, owns a backend-neutral fixed affine
constraint plan and canonical nodal-load contribution plan, and contains no SciPy
object, evolving state, solver setting, or assembly workspace. `evaluate_program(...)`
binds one exact structured program point into immutable prescribed-offset and
full-space nodal-force values plus coordinate derivatives without mutating either
compiled input.

**Coverage rows advanced:** `PROG-DIRICHLET`, `PROG-MPC`, and
`PROG-NODAL-LOAD`. They are `contracted` at E1 from the analytical reference and
14 executed legacy/prototype checks above; the exact E1 proof is
`b18b262b97068086b9c61618d06a2d14a50046cd`, with `79060abb` retained as the test
execution base.
Successful source, I0 integration, and focused component evidence may advance them
only to E2. `PROG-SCHEDULE`, general
linear-combination constraints, distributed/follower loads, initial conditions,
program state, assembly, reactions, and analysis remain later rows and receive no
evidence advance from P1-A.

**Exact base and required parent packets:** dispatch only after this Horizon and
the R0-N GO adjudication are committed on a clean `v3` head. Required code proof is
P0-D integrated and independently verified at
`79060abb054d82fd34ac85f000e44d9de5947d60`. The dispatched prompt records the
resulting exact documentation head; no worker rebases onto a later coordinator
commit.

**Exact authored vocabulary:** add exact frozen/slotted internal values with these
roles and no opaque mapping/blob escape hatch:

```text
ProgramCoordinateSpec(name, kind, source)
DofRef(node_id, field_id, component)
AffineCoefficientSpec(coordinate, coefficient, source)
AffineValueSpec(constant, coefficients, source)
PrescribedDofSpec(id, target, value, source)
AffineTieSpec(id, slave, master, factor, offset, source)
NodalLoadSpec(id, target, value, source)
ProgramSpec(coordinates, constraints, loads, source)
ProgramCoordinateValue(name, value)
ProgramPoint(values)
```

`kind` is exactly `time`, `load`, or `continuation`; `kind == "time"` if and only
if `name == "time"`, and that coordinate appears at most once. Empty coordinates
are valid for a constant program. Coordinate names are nonempty and unique.
Constraint and load IDs use the
existing exact `str | int` semantic-ID policy and are unique within their entity
kind. A DOF reference is always the exact semantic triple
`(node_id, field_id, component)`; no global integer index is authored. Every
affine value is one finite float64-representable exact integer/float constant plus
zero or more unique named coordinate coefficients of the same numeric policy;
`bool`, subclasses, nonfinite values, narrowing overflow, unknown coordinates, and
duplicate terms fail before conversion or evaluation.

The first constraint slice contains prescribed DOFs and one-master affine ties
only:

```text
u_slave = factor * u_master + offset(p)
```

The factor is finite, float64-representable, constant, and nonzero. A factor or
prolongation coefficient may not depend on program coordinates. General
linear-combination rows are explicitly deferred rather than encoded in parameter
tuples. The offset and nodal load value may be affine in the declared coordinates.
Program points contain every declared name exactly once and no extra name, with
finite float64-representable exact integer/float values excluding `bool`.

**Exact compiled meaning:** add immutable internal carriers with at least these
semantic fields:

```text
AffineConstraintPlan
  full_dof_count, reduced_dof_count
  free_dofs
  row_offsets, column_indices, coefficients
  offset_constant, offset_coordinate_coefficients

NodalLoadPlan
  load_ids, dof_indices
  constant_values, coordinate_coefficients

ProgramCapabilities
ProgramProvenance
CompiledProgram
  fresh instance ID + deterministic content fingerprint + provenance
  compatible model instance ID + model content fingerprint
  coordinate names/kinds
  constraint plan + nodal-load plan
  capabilities + entity/source maps

ProgramEvaluation
  exact program/model identities + bound coordinate names/values
  prescribed offsets + their coordinate derivatives
  full-space nodal force + its coordinate derivatives
```

Every numeric carrier is a detached, owning, contiguous, read-only
`FinalizedArray`. The prolongation `P` is stored as canonical CSR-like arrays, not
a dense matrix and not a SciPy object. Rows for independent DOFs are identity;
prescribed rows have no column; one-master chains flatten to at most one canonical
free-root column with factors and affine offsets composed in dependency order.
Free reduced coordinates are ordered by compiled full DOF index. A no-constraint
program therefore yields identity `P`; a fully prescribed program yields a valid
`(n_full, 0)` plan rather than an accidental sparse-solve path.

The compiled numeric encoding is frozen for this slice:

- every program index array is exact `int64`, independent of the model's possibly
  narrower dense-index policy; every program coefficient/value array is exact
  `float64`;
- `free_dofs` has shape `(n_reduced,)`, is strictly increasing, and contains exact
  full DOF indices;
- `row_offsets` has shape `(n_full + 1,)`, begins at zero, is nondecreasing, and
  ends at `nnz`; `column_indices` and `coefficients` both have shape `(nnz,)`;
- each CSR row contains zero or one entry; every column is in
  `[0, n_reduced)`, independent rows contain coefficient `1.0`, and all other
  coefficients are the composed finite tie factor;
- `offset_constant` has shape `(n_full,)` and
  `offset_coordinate_coefficients` has shape `(n_full, n_coordinate)`;
- nodal-load `dof_indices`/`constant_values` have shape `(n_load,)`, and load
  coordinate coefficients have shape `(n_load, n_coordinate)`; and
- evaluation coordinate values have shape `(n_coordinate,)`; prescribed offsets
  and nodal force have shape `(n_full,)`; each derivative array has shape
  `(n_full, n_coordinate)`.

Counts, `n_full + 1`, and `nnz` are checked against `int64` before allocation, so
the terminal CSR offset is representable even when a model's maximum DOF index
fits only its own narrower dtype. The manifest records program index dtype
`int64`, floating dtype `float64`, the shapes above, and the exact reduction policy.

Canonical order is also frozen: coordinates use kind order
`time < load < continuation` then exact name; constraints and loads use the
existing type-tagged semantic-ID order; affine terms follow canonical coordinate
order. Constraint-chain constants and coefficients compose root-to-slave using
binary64 multiplication and `math.fsum` for the two-term addition, rejecting any
nonfinite intermediate. Load rows are ordered by target full DOF then semantic
load ID; evaluation uses `math.fsum` in that order separately for the constant and
each coordinate coefficient, followed by a `math.fsum` over canonical-coordinate
products at the bound point. This is the one deterministic float64 accumulation
owner; authored declaration order cannot select a different reduction path.

The nodal-load plan retains one canonical row per authored load declaration so
source attribution is not destroyed. Evaluation reduces all rows targeting the
same full DOF with a deterministic order and addition policy. Loads on constrained
DOFs remain in the full vector for later reaction/balance work. No pre-summed
mutable target vector becomes a second canonical owner.

Content identity includes the normalized program meaning, compatible model content
fingerprint, numeric/index policy, coordinate schema, constraint/load plans,
derived capabilities, and detached entity/source provenance. It excludes live
instance tokens. Compiling the same meaning against the same live model produces a
fresh program instance with the same content fingerprint; compiling against a
content-equivalent but distinct model records that distinct compatible live model
while retaining content equivalence. Declaration order that is semantically
unordered cannot change the result, while changing a DOF reference, affine
coefficient, coordinate kind/name, or source identity changes the corresponding
manifest meaning.

**Required constraint behavior:** reject unknown node/field/component references,
slave=master, zero factor, duplicate dependent ownership, a DOF that is both
prescribed and a slave, cycles of any length, malformed exact compiled-model
carriers, index overflow, and nonfinite arithmetic during chain composition with a
source-anchored program compilation error. Ties may chain through ties and through
a prescribed master; all factors, constants, and coordinate derivatives compose
exactly once. Multiple slaves may share one master. Constraint declarations never
silently replace one another, and evaluation never patches a candidate state.

**Required load and coordinate behavior:** repeated nodal contributions on one DOF
sum. Missing, extra, duplicate, malformed, nonfinite, or wrong-type program-point
coordinates fail deterministically before an evaluation carrier escapes. The
derivative arrays are the exact affine coefficients and do not depend on the bound
point. Constant zero-load and displacement-only programs are explicit valid
programs. Program capabilities are derived conservatively: this slice has fixed
constraint/load topology, no follower or interaction tangent, no program-owned
state, and only constant/coordinate-affine prescribed and external nodal channels.

**Independent mathematical oracle:** focused tests hard-code at least this chain,
using one named load coordinate `lambda` and one free root `q0`:

```text
u1 = 2*q0 + (3 + 4*lambda)
u2 = -0.5*u1 + (1 - lambda)
   = -q0 - 0.5 - 3*lambda
```

The expected `P` rows, constant offsets, and `du_bar/dlambda` are literal test
data, not reconstructed by the implementation. Two loads on `u2`,
`5 + 2*lambda` and `-1 + 3*lambda`, must evaluate to `4 + 5*lambda` with derivative
`5`. Tests also cover identity `P`, zero-free-DOF `P`, a chain whose root is
prescribed, negative factors, constrained-DOF loads, declaration-order invariance,
distinct live identity/content equivalence, and caller mutation after normalization,
compilation, and evaluation.

**Total-boundary correctness matrix:** exact-class/slot/container preflight occurs
before child access, iteration, hashing, equality, formatting, or conversion for
every new authored and runtime value. Malformed values produce stable
`ProgramSpecValidationError`, `ProgramCompilationError`, or
`ProgramEvaluationError` diagnostics, never raw incidental exceptions. Include
huge exact IDs/source locations under `PYTHONINTMAXSTRDIGITS=640`, missing/forged
children, list-to-tuple ownership, source detachment, cycles, unknown references,
duplicate/conflicting constraints, numeric overflow, nonfinite evaluation, and
separate finalizations. Diagnostics stay bounded and do not mutate interpreter
global state.

**Owned paths:** new `pyfem/v3/spec/program.py`, program-specific normalization and
diagnostic modules under `pyfem/v3/spec/`, and the minimal internal exports in
`pyfem/v3/spec/__init__.py`; new `pyfem/v3/model/program.py` plus minimal internal
exports in `pyfem/v3/model/__init__.py`; new program compiler/evaluator and
diagnostics under `pyfem/v3/compile/` plus minimal internal exports in
`pyfem/v3/compile/__init__.py`; and one new focused
`test/v3/test_v3_program_compile.py`. New files may be split inside those exact
subtrees for clarity, but all existing non-`__init__.py` files in those subtrees
are read-only and no existing model-compiler/spec behavior is rewritten.

**Forbidden paths and non-goals:** `.agents/v3/**`, root configuration,
`pyproject.toml`, `uv.lock`, `pyfem/v3/__init__.py`, prototype `types.py`,
`pack.py`, `_prototype_assembly.py`, existing solver/constraint/assembly/I/O
modules, existing numeric kernels, adapters, and existing tests outside the one
new focused file. Do not add SciPy, a sparse backend, `PreparedAssemblyPlan`,
physical/evolution/program state, transactions, reactions, result types, a solve
function, initial conditions, schedule stepping, distributed/follower loads,
general multi-master constraints, registry descriptors, public exports, or a
compatibility wrapper.

**Strongest competing design:** extend `ProblemDefinition`/`pack_problem`, reuse
`build_prescribed_constraints`, and carry one mutable dense external-load vector
plus a SciPy matrix into every solver. That design is rejected because it preserves
last-write load packing, solver-local constraint meaning, a scalar program
coordinate, and prototype carrier ownership. A second rejected design stores both
an authoritative dense `P` and sparse `P`; P1-A has one canonical backend-neutral
plan.

**Expected migration and API consequence:** no prototype or legacy path is deleted
in P1-A. Its types/functions are internal and are not exported from `pyfem.v3`.
P1-B is the first consumer and must compose this plan with model/request assembly;
P1-C may retire Q8 use of the prototype packing/constraint/load path only after the
verified public slice proves replacement. Legacy `.dat`/`.pro` names, syntax,
duplicate policies, root API, and CLI remain adapter/public decisions.

**Performance envelope:** compilation should be linear in full DOFs plus
coordinates, constraints, and load declarations, apart from canonical sorting.
Graph resolution uses explicit visitation/topological state rather than repeated
whole-pending scans. Do not allocate dense `P` or introduce a backend. Dense
`n_full x n_coordinate` offset/derivative storage is accepted in this bounded
slice because it is the required evaluated output; report, do not optimize, a
measured concern if coordinate breadth makes it material.

**Acceptance commands and evidence:** one focused commit and clean worker tree;
new program tests plus P0-D foundation/compiler tests normally and with
`PYTHONINTMAXSTRDIGITS=640`; the 14 existing constraint/MPC/load reference tests;
both required Ruff configurations and focused format; `pytest -q test/v3`; full
`pytest -q`; `git diff --check`; exact ancestry/path ownership; and a zero-hit scan
showing new program paths do not import prototype types/pack/assembly, solver, I/O,
SciPy, or root public exports. Report existing warnings separately.

**Merge order and repair loop:** exactly one Sol/max writer returns one source
commit. I0 verifies ancestry, ownership, total-boundary/affine/load semantics, and
all focused/static/v3/full evidence before serial integration. Accepted P0/P1
findings return to the same task for at most two bounded repair rounds. P1-B stays
closed until P1-A is integrated with combined proof; no independent reviewer is
dispatched in parallel.

**Blocked condition and minimum unlock:** stop without broadening scope if the
compiled model cannot address exact semantic DOFs, if correct one-master affine
composition requires changing P0-D carriers, if a general multi-master constraint
or schedule/state owner is required for the stated oracle, if a SciPy/backend owner
is unavoidable, or if an owned-path boundary must be crossed. Return the smallest
reproducible conflict; I0 amends the Horizon or creates a separate packet.

**Integration task ID:** `019f6f49-0b72-7d73-86da-c6b85519eeaf`.

## P1-B Horizon card — `HORIZON_FROZEN`

**ID/title:** `P1-B · assembly plan — contributions composed`

**Outcome and ownership invariant:** P1-B composes one exact validated
`CompiledModel` and compatible exact validated `CompiledProgram` into an immutable,
backend-neutral `PreparedAssemblyPlan`, then evaluates one structured program point
into immutable linear Q8 operator/vector contributions. Preparation owns topology,
compatibility, attribution, and deterministic reduction policy. Evaluation owns
values at one program point. Neither object owns evolving state, a sparse backend,
a factorization, a reduced unknown, a solution, or a public request.

P1-B consumes P1-A's private `_validated_model(...)`, `_validated_program(...)`,
and `evaluate_program(...)` internal seams. It must not copy their deep carrier
validation into a third implementation. Those names are frozen as internal
consumer seams for this packet; no shared-validator extraction or P1-A rewrite is
authorized unless the writer returns a concrete blocker first. The two validators
raise P1-A's private `_BoundaryError`; P1-B is authorized to import that type only
for immediate boundary translation. In `prepare_assembly_plan(...)`, validator and
model/program compatibility failures become stable `AssemblyPreparationError`.
In `assemble_reference_linear(...)`, the exact model, program, and plan are
revalidated; validator, plan/input compatibility, and live-identity failures become
stable `AssemblyEvaluationError`. P1-B catches `IdentityMismatchError` (or the exact
existing live-identity helper's equivalent) at both entry points and translates it
to the entry point's assembly error. `ProgramEvaluationError` from binding the
point through `evaluate_program(...)` also becomes `AssemblyEvaluationError` while
preserving every diagnostic's code, bounded message, and source in order. Assembly
errors use the nearest copied compiled source when available and otherwise fixed
assembly context.
P1-A-specific or raw incidental exceptions never cross the assembly boundary.

**Coverage rows advanced:** component slices only of `ASM-COO`, `ASM-PREPARE`,
`ASM-EXTERNAL`, `ASM-GATHER`, and `ASM-TANGENT`. Freezing this card contracts those
five rows at E1. Successful source, I0 integration, and focused component evidence
may advance only these slices to E2. `ASM-INTERNAL`, nonlinear or stateful tangent,
distributed/follower loads, `ANAL-LINEAR`, reactions/balance/results, public flow,
and prototype retirement remain E0.

**Exact base and required parent packets:** the sole writer was dispatched from
exact clean frozen-card commit `4ed975795a4917fe40ed0203c981980ac5bea9cc`, which
contains the integrated P1-A commit
`b7316be8befba64df0a576df6b7fd8cdcf9b9873`. P0-D is independently green at
`79060abb054d82fd34ac85f000e44d9de5947d60`; P1-A final source
`ab64e0219c11b5fc4155703eb8ade32939f90064` is independently green and integrated.
The exact task ID is recorded in the packet ledger. No other assembly writer runs
in parallel.

**Request ownership and exact vocabulary:** do not add an authored assembly spec.
P1-C's future typed `LinearStatic` request remains the sole authored analysis
request and source owner. P1-B adds one exact frozen/slotted zero-field internal
marker with versioned meaning:

```text
LinearStaticContributionRequest()
```

It requests exactly the constant model material tangent, P1-A external nodal force,
affine constraint reduction, full-balance inputs, and program-coordinate
derivatives. Subclasses, foreign values, extra fields, and optional channel names
reject. Future analysis families add exact request/result types rather than fields
on a universal carrier.

Add only this bounded internal vocabulary:

```text
DomainCooPlan
NodalVectorContributionPlan
AssemblyPlanProvenance
PreparedAssemblyPlan
CanonicalCooOperator
LinearStaticContributions
AssemblyPreparationDiagnostic / AssemblyPreparationError
AssemblyEvaluationDiagnostic / AssemblyEvaluationError
```

The two entry points are:

```text
prepare_assembly_plan(model, program, request) -> PreparedAssemblyPlan

assemble_reference_linear(
    model,
    program,
    plan,
    program_point,
) -> LinearStaticContributions
```

The evaluation entry point binds `program_point` through P1-A
`evaluate_program(...)`; it never trusts a caller-built `ProgramEvaluation`.

**Prepared topology and attribution:** `DomainCooPlan` records finalized exact
`int64` arrays for canonical raw full-operator rows and columns plus exact block,
cell, local-row, and local-column attribution. Raw order is block index, canonical
cell order, local row `0..15`, then local column `0..15`. Every Q8 cell has exactly
256 raw entries. Structural zeros and shared-DOF duplicates remain present in raw
topology and raw values; numeric nonzero detection never defines structure.

The plan also owns lexicographically sorted unique full `(row, column)` pairs, an
exact raw-to-canonical coalescing map, and the corresponding reduced topology and
maps derived from P1-A's zero-or-one-entry-per-full-row prolongation. A prescribed
full row maps to no reduced column. A retained row maps to its exact reduced column
and finite nonzero prolongation coefficient. Reduced raw values use frozen binary64
multiplication order `(left_factor * full_entry) * right_factor` before canonical
coalescing. All duplicate groups reduce with `math.fsum` in raw source order.

`NodalVectorContributionPlan` records the full-to-reduced mapping and deterministic
source order required to reduce full nodal/offset vectors. Full DOFs are visited in
ascending order and every reduced component uses `math.fsum`. No dense global
matrix and no dense prolongation are allocated. Count arithmetic, `256 * n_cell`,
shape products, terminal offsets, indices, and allocations are checked against both
exact `int64` and `np.intp` capacity before conversion or allocation.

**Exact result meaning and alias policy:** `CanonicalCooOperator` contains exact
shape, finalized `int64` row/column arrays in unique lexicographic order, and one
finalized `float64` value array. Its row/column carriers are the exact immutable
topology carriers owned and finalized once by the compatible plan; only operator
values are fresh per evaluation. `LinearStaticContributions` has exactly these
semantic fields:

```text
program_evaluation
plan_content_fingerprint
full and reduced CanonicalCooOperator
full raw Q8 operator values
reduced raw operator values
full affine-offset internal force and coordinate derivatives
full offset-corrected RHS and coordinate derivatives
reduced external force and coordinate derivatives
reduced affine-offset internal force and coordinate derivatives
reduced RHS and coordinate derivatives
```

The retained `program_evaluation` is the sole owner of bound coordinate values,
prescribed offsets and derivatives, and full external nodal force and derivatives;
P1-B references those exact P1-A carriers and does not duplicate them as sibling
fields. Every assembly-derived raw/canonical value, offset-internal, RHS, reduced,
and derivative array is a fresh exact `float64` detached, owning, contiguous,
read-only finalization with no cross-field or prior-evaluation value alias. The only
intentional sharing is immutable plan row/column topology and the exact retained
`ProgramEvaluation` channels.

Empty forms are first-class: zero coordinates produce `(n, 0)` and `(m, 0)`
derivative arrays; a fully prescribed program produces valid reduced `(0, 0)`,
`(0,)`, and `(0, c)` carriers. A load on a prescribed DOF remains in
`program_evaluation.nodal_force` even when its reduced contribution is zero.
Repeated evaluations reuse immutable plan topology but cannot mutate or alias one
another's value storage.

**Mathematical and sign contract:** for full tangent `K`, full external nodal force
`f`, P1-A prolongation `P`, and prescribed affine offset `u_bar(p)`:

```text
K_q       = P.T K P
f_q       = P.T f
g_q       = P.T K u_bar
b_full    = f - K u_bar
b_q       = f_q - g_q = P.T (f - K u_bar)
db_q / dp = P.T (df / dp - K du_bar / dp)
```

External force is positive. `K u_bar` is an explicitly attributed offset-internal
contribution. Reduced RHS is external minus offset-internal. P1-B does not define a
residual at a candidate displacement, internal force at `u`, reactions, or balance
acceptance. Full matrix-vector products traverse canonical raw COO source order and
use `math.fsum` per output row. Nonfinite product, integration, coalescing, matvec,
or reduction results fail through an assembly evaluation diagnostic.

**Q8 reference evaluation and registry meaning:** during preparation, validate the
captured `serendipity-quad8`, `gauss-3x3`, `small-strain-continuum`, and
`plane-stress-linear-elastic` descriptor identities against the exact model registry
snapshot. Do not bypass, rebind, or introspect a different registry. Preparation
reuses and validates the compiled domain block's quadrature weights, parent
gradients, shapes, points, and material parameters; it does not reevaluate topology
or quadrature at each program point. Reference evaluation invokes only the captured
formulation and material bindings needed to form the local kinematic/constitutive
response. Validate every binding result, array rank/shape/dtype/finiteness,
quadrature weight, and the declared constant/symmetric material meaning. Do not
symmetrize a computed operator to hide an inconsistent binding.

Each cell is evaluated with bounded local scratch. A naive physical-coordinate
Jacobian is forbidden because valid P0-D geometry may have uniform scales such as
`1e-200` or `1e200`. Use translation-free scalar-normalized geometry:

1. subtract a reference node, using a scaled finite fallback if direct subtraction
   overflows;
2. divide relative coordinates by their maximum finite magnitude;
3. form normalized `J_hat`, gradients, and `B_hat`; and
4. compute exactly
   `K_e = sum_p w_p * B_hat[p].T * C * B_hat[p] * det(J_hat[p])` in canonical
   quadrature order.

For two-dimensional per-unit-thickness elasticity the scalar scale cancels between
`B = B_hat / s` and `dOmega = s**2 dOmega_hat`. Let `t` be the exact validated
`model.provenance.geometry_relative_tolerance`. At every point, both
`det(J_hat) > t` and `det(J_hat) / sum(J_hat**2) > t` are required after finite,
nonzero-norm checks. This retains P0-D's near-singular policy while additionally
requiring positive orientation. Never use `abs(det J)`, silently reorder nodes, or
call a prototype wrapper whose orientation policy differs.

**Independent literal Q8 oracle:** focused tests embed the unit square in frozen
local order
`(0,0), (.5,0), (1,0), (1,.5), (1,1), (.5,1), (0,1), (0,.5)` with
`E=1`, `nu=0`, and unit thickness. Exact polynomial integration gives `K=M/360`.
The test embeds this literal integer matrix and does not reconstruct it through
production shape, quadrature, material, or assembly code:

```text
M = [
[312,85,-308,-100,146,15,-104,-20,138,35,-172,-20,124,-15,-136,20],
[85,312,20,-136,-15,124,-20,-172,35,138,-20,-104,15,146,-100,-308],
[-308,20,736,0,-308,-20,0,-80,-172,-20,224,0,-172,20,0,80],
[-100,-136,0,512,100,-136,-80,0,-20,-104,0,-32,20,-104,80,0],
[146,-15,-308,100,312,-85,-136,-20,124,15,-172,20,138,-35,-104,20],
[15,124,-20,-136,-85,312,100,-308,-15,146,20,-104,-35,138,20,-172],
[-104,-20,0,-80,-136,100,512,0,-136,-100,0,80,-104,20,-32,0],
[-20,-172,-80,0,-20,-308,0,736,20,-308,80,0,20,-172,0,224],
[138,35,-172,-20,124,-15,-136,20,312,85,-308,-100,146,15,-104,-20],
[35,138,-20,-104,15,146,-100,-308,85,312,20,-136,-15,124,-20,-172],
[-172,-20,224,0,-172,20,0,80,-308,20,736,0,-308,-20,0,-80],
[-20,-104,0,-32,20,-104,80,0,-100,-136,0,512,100,-136,-80,0],
[124,15,-172,20,138,-35,-104,20,146,-15,-308,100,312,-85,-136,-20],
[-15,146,20,-104,-35,138,20,-172,15,124,-20,-136,-85,312,100,-308],
[-136,-100,0,80,-104,20,-32,0,-104,-20,0,-80,-136,100,512,0],
[20,-308,80,0,20,-172,0,224,-20,-172,-80,0,-20,-308,0,736],
]
```

This oracle also proves symmetry, two rigid translations, infinitesimal rotation,
and zero row sums. Unit-scale full stiffness uses `rtol=0, atol=2e-14`; reduced
stiffness/RHS uses `rtol=0, atol=5e-13`.

Compose that literal operator with the P1-A chain oracle. At `lambda=2`,
`u_bar[1]=11`, `u_bar[2]=-6.5`, `f[2]=14`, and their coordinate derivatives are
`4`, `-3`, and `5`. The literal reduced RHS and derivative are:

```text
b_q = [
-3829/72, 187/45, -1837/360, -83/20,
11/18, 343/90, -167/40, -206/45,
419/90, 143/45, -1283/360, -41/10,
55/18, 977/90,
]

db_q/dlambda = [
-946/45, 68/45, -12/5, -139/90,
2/9, 56/45, -82/45, -17/10,
94/45, 52/45, -8/5, -131/90,
10/9, 184/45,
]
```

The test may use literal dense `P` and `M/360` only as an independent oracle;
production uses the prepared sparse maps. The existing two-cell P0-D fixture must
produce 512 raw entries, 476 distinct structural pairs, and exactly 36 shared-pair
duplicates, then coalesce two translated copies of the literal operator. This
proves duplicate addition and canonical cell-order invariance without claiming
multi-block order invariance or optimized sparse-slot equivalence.

**Required focused correctness matrix:** in addition to the literal matrix/RHS and
failure cases below, tests explicitly prove:

- uniformly scaled cells at `1e-200`, `1`, and `1e200` produce the same reference
  stiffness within the frozen tolerance;
- no constraints produce identity reduction;
- one directly prescribed full row is absent from reduced topology;
- all prescribed DOFs produce the exact empty reduced shapes stated above;
- a prescribed-DOF load survives in the full program force while its reduced force
  is zero;
- zero program coordinates produce exact full/reduced zero-width derivative shapes;
- model dense-index arrays narrower than `int64` lower safely into exact `int64`
  assembly topology; and
- repeated evaluation reuses exact immutable topology while every value carrier is
  fresh and prior results remain unchanged.

**Capability and failure policy:** P1-A currently has no follower or interaction
tangent. P1-B records an exact empty program-operator recipe set and rejects any
program capability that claims an operator contribution without a matching exact
recipe. It does not add placeholder arrays for future physics. Reject malformed,
subclassed, missing-slot, altered-manifest, writeable, cross-owner carrier reuse,
or mutable-storage aliasing where separate value ownership is required;
model/program live mismatch despite content equality; plan/input mismatch; invalid
counts/indices/maps; binding failures or wrong binding outputs; singular, inverted,
nonfinite, or orientation-changing geometry; and any dynamic-pattern, stateful,
follower, interaction, or unsupported contribution channel.

Exact-class/slot/container checks precede unsafe access, iteration, hashing,
comparison, formatting, or conversion. Diagnostics are stable, bounded, escaped,
and anchored to copied compiled model/program sources where meaningful. Huge exact
IDs and source coordinates remain total under `PYTHONINTMAXSTRDIGITS=640`; raw
incidental exceptions do not cross the preparation/evaluation boundary.

**Identity and content policy:** `PreparedAssemblyPlan` has deterministic content
fingerprint and provenance, but no fresh live `InstanceId`; P1-C's future
`PreparedAnalysis` owns live workspace/cache identity. The plan records exact
compatible model/program live IDs and content fingerprints. Its manifest includes
the fixed request meaning, model/program meaning, registry snapshot, numeric/index
policy, raw and canonical topology, reduction maps, shapes, attribution, and empty
program-operator recipe set. Separately preparing equivalent topology for the same
exact live model/program yields the same plan content fingerprint and is
interchangeable after full plan validation. A plan prepared from content-equivalent
but distinct live model/program instances remains incompatible. Contributions
record exactly the plan content fingerprint; their retained exact
`program_evaluation` is the sole owner of compatible model/program instance and
content identities, which result validation requires to equal the plan's recorded
compatibility. There is no plan live ID and no sibling copy of model/program
identity meaning. The result does not become a second request or state owner.

**Owned paths:** new `pyfem/v3/assembly/__init__.py`, `contracts.py`,
`diagnostics.py`, `prepare.py`, and `reference.py`; and one new focused
`test/v3/test_v3_assembly_plan.py`. Files may be split only inside the new
`pyfem/v3/assembly/**` namespace when necessary for clarity. All P0-D/P1-A
production paths are read-only dependencies.

**Forbidden paths and P1-C boundary:** `.agents/v3/**`, root configuration,
`pyproject.toml`, `uv.lock`, `pyfem/v3/__init__.py`, `pyfem/v3/spec/**`,
`pyfem/v3/model/**`, `pyfem/v3/compile/**`, prototype `types.py`, `pack.py`,
`_prototype_assembly.py`, `pyfem/v3/fem/assembly.py`, existing numeric kernels,
solver/analysis/state/result/I/O paths, and existing tests. Do not add SciPy, Numba,
a sparse backend, factorization, candidate displacement, internal force at a
candidate, physical/evolution/program state, transactions/generations, reactions,
balance/result/verification types, public/root exports, a solve function, schedule,
distributed/follower loads, mass/damping, general multi-master constraints,
adapters, compatibility wrappers, prototype edits, or deletion.

**Performance envelope:** with `raw_nnz = 256 * n_cell`, preparation time is
`O(raw_nnz log(raw_nnz) + n_full + nnz(P) + n_load)` and preparation memory is
`O(raw_nnz + canonical_nnz + n_full + nnz(P) + n_load)`. Reference evaluation time
is
`O(elements * 9 * 16**2 + raw_nnz * (1 + n_coordinate) +
(n_full + n_reduced) * (1 + n_coordinate) + P1-A validation/evaluation cost)`;
fresh evaluation-value memory is
`O(raw_nnz + canonical_nnz + (n_full + n_reduced) * (1 + n_coordinate))`.
Evaluation processes one element at a time with bounded local scratch and allocates
no dense `n x n` or dense `P`. The packet sets no timing, Numba, chunk, or backend
threshold. The correctness path retains raw attribution and canonical coalesced
views. Fixed optimized CSR slots and scale benchmarks belong to Phase 2.

P1-A's full-DOF `ProgramMeaningWitness` is a nonblocking O(full DOFs) duplication
and evaluation-cost obligation. P1-B must not expand that witness. Measure the
combined cost before Phase-1 performance claims and compare later with a compact
per-referenced-DOF witness without weakening P1-A's total-boundary proof.

**Acceptance commands and evidence:** one focused commit and clean worker tree;
focused assembly tests; P0-D/P1-A/P1-B combined tests normally and with
`PYTHONINTMAXSTRDIGITS=640`; existing element-stiffness, COO, chunked-assembly,
constraint, load, and MPC reference tests; both required Ruff configurations and
focused format; `pytest -q test/v3`; full `pytest -q`; `git diff --check`; exact
one-commit ancestry and owned-path proof; and zero direct imports from new assembly
code of prototype carriers/packing/assembly, solver, I/O, legacy `pyfem.fem`,
SciPy, Numba, or root public exports. A source scan must also find zero use of
`abs(det J)` in the new assembly namespace. Existing warnings are reported
separately.

**Merge order and repair loop:** exactly one Sol/max writer returns one source
commit. I0 verifies ancestry, ownership, total-boundary behavior, literal Q8 and
affine-reduction oracles, topology/identity semantics, and all focused/static/
v3/full evidence before serial integration. One independent Sol/max assembly
reviewer runs only after candidate proof is complete. Accepted P0/P1 findings
return to the same writer for at most two bounded repair rounds. P1-C remains
closed until P1-B is integrated and independently green.

**Blocked condition and minimum unlock:** stop without broadening scope if P1-A's
validators cannot be safely reused, the captured registry bindings cannot evaluate
the declared Q8 recipe with the required scale/orientation policy, correct
contribution attribution requires modifying P0-D/P1-A carriers, an authored
assembly request or state owner is required, a backend/SciPy object is unavoidable,
or an owned-path boundary must be crossed. Return the smallest reproducible
conflict; I0 amends this card or creates a prerequisite packet.

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
5. Completed: R0-L returned a valid terminal result and callback with one accepted
   frozen-registry defect. R0-J and R0-K remain incomplete; their statuses do not
   change.
6. Completed: P0-H repaired the R0-L defect, I0 reviewed/integrated its single
   commit, and all combined focused/static/v3/full gates passed.
7. Completed: R0-M independently returned GO with zero blockers at the integrated
   commit after one bounded resume recovered the already-running v3 result.
8. Completed: R0-E produced the complete E0 legacy capability inventory; I0
   preserved its exact report, independently checked it, and ingested the live state
   overlay into this ledger.
9. Completed: P0-D's repaired Q8 compiler was reviewed and integrated; replacement
   R0-N returned independent GO after the original R0-F stopped correctly on an
   I0-supplied base mismatch.
10. Completed: P1-A's final repair 2/2 closed all accepted program-carrier seams,
    I0 integrated it as `b7316be`, and independent carrier review returned GO.
11. Completed: P1-E closed both accepted P1-B findings, I0 integrated it as
    `daad227`, and the exact-base replacement independent review returned GO with
    zero findings. The five assembly rows are provisional E2; pause before P1-C.

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
| `ACTIVE · I0 · v3 integration coordinator` | `019f6f49-0b72-7d73-86da-c6b85519eeaf` | current main `f32e7e5`; original dispatch `c75cbf3` | Integration, shared docs, portfolio adjudication, and combined proof | Active; foundation and Phase 1 proofs complete, generic-core portfolio in flight |
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
| `P0-D · model compiler — Q8 block frozen` | `019f75dd-f597-7c21-adfc-d78b1e2580c0` | `2842ef84b86f04f82587f5fc378584f30f6b62bd` | `pyfem/v3/compile/**`, new compiled-model carriers, one focused compiler test | Complete after repair 1/2; replacement source `f21aaed2`, integrated `79060abb`; branch proof 156 focused twice, 22 direct Q8, 286 v3, 475 full |
| `INCOMPLETE R0-F · model compiler — base mismatch` | `019f761b-2fc2-7e20-b5d9-b9681cb83e7e` | incorrect requested hash `79060ab3186e38460595b3831b77855b3f824bc` | Intended read-only compiler review | Incomplete by coordinator error; stopped correctly at clean preflight after observing actual `79060abb`; no correctness matrix or finding was assessed |
| `R0-N · model compiler — block invariants verified` | `019f761d-4d85-7b72-9d80-857fab45aef8` | `79060abb054d82fd34ac85f000e44d9de5947d60` | Replacement read-only P0-D integrated review | Complete GO with both terminal signals; 99 independent checks, 156 focused twice, 22 direct Q8, 286 v3, 475 full, zero findings |
| `P1-A · program compiler — affine plan canonical` | `019f764f-2643-75b2-a8b5-323ed58351a2` | `b18b262b97068086b9c61618d06a2d14a50046cd` | New program spec/normalizer, immutable compiled-program carriers, compiler/evaluator, and one focused test | Complete after repair 2/2; source `ab64e02`, integrated `b7316be`; independent GO, 72 focused twice, 228 combined twice, 358 v3, 547 full |
| `INCOMPLETE P1-B · assembly plan — approval stalled` | `019f76f1-fffd-7772-ba9c-4e17d3ee9f63` | `4ed975795a4917fe40ed0203c981980ac5bea9cc` | Initial immutable reference assembly plan/evaluator source plus attempted repair turn | Initial source `ab02297` completed with valid final/callback; I0 accepted two P1 findings; repair turn was interrupted before editing at an avoidable worktree-transition approval |
| `INCOMPLETE P1-D · assembly plan — worktree missing` | `019f7732-d6a9-7c61-aa50-1410cb7b7f26` | intended source `ab02297c052912cb027ac6963b37667bf61f3e97` | First replacement for P1-B repair 1/2 | Incomplete before preflight/editing because the fork inherited the app-removed worktree path; archived without repository change |
| `P1-E · assembly plan — correspondence repaired` | `019f7736-ec41-7560-9d65-472551b4a6fe` | source `ab02297c052912cb027ac6963b37667bf61f3e97`, parent `4ed975795a4917fe40ed0203c981980ac5bea9cc` | Close exact Q8 reference-recipe correspondence and metadata-free binding output in the same six paths | Complete with both terminal signals; replacement `8e94454`, integrated `daad227`; 34 focused twice, 152 combined twice, 392 v3, 581 full |
| `INCOMPLETE R1-B · assembly plan — temp-file approval stalled` | `019f7750-8745-7e23-aaf2-35f50ec79fc8` | `e11a07c0cc5c00e956c3d537ac94272f120621c0` | Original independent read-only P1-B reviewer | Incomplete; preflight and repository gates were green, but an unnecessary temporary-file change waited on approval; archived without final/callback or repository change |
| `I0/R1-B · assembly plan — bounded replacement GO` | internal agent `/root/r1b_inline_review` | `e11a07c0cc5c00e956c3d537ac94272f120621c0` | Exact-base read-only Sol/max replacement | Complete GO; zero findings, 34 focused twice, 152 combined twice, 27 references, both Ruff configurations, format/static/diff/final-clean gates green |
| `P1-C · linear flow — public solution verified` | internal Sol/max agent `/root/p1c_writer`; source worktrees `/private/tmp/pyfem-p1c-02cff1c` and `/private/tmp/pyfem-p1c-convergence-closure` | base `02cff1c`; source chain `c263d958` -> `8524c233` -> `ba6fc4e` -> `3ee920a` | Sole state/analysis/result/API writer plus one bounded convergence closure | Complete; integrated chain `c9b9480` -> `bfc1d80` -> `d78201d` -> `7ee65c3`; 86 focused twice, 348 combined twice, 478 v3, 667 repository |
| `R0/P1-C · repair 1 solver/state check` | internal Sol/max agent `/root/p1c_repair_solver_review` | `8524c2337a408e69be215a78044d54d6af968a6b` | Read-only transaction, solver, convergence, and workspace check | Complete NO-GO; P0/P1/P2 `0/3/0`; opposite discard race, exact-ratio false negatives, and unverifiable cache history |
| `R0/P1-C · repair 1 result check` | internal Sol/max agent `/root/p1c_repair_result_review` | `8524c2337a408e69be215a78044d54d6af968a6b` | Read-only result, storage, scalar-preflight, and evaluation check | Complete NO-GO; P0/P1/P2 `0/2/0`; convergence evidence and exact/total scalar preflight |
| `R0/P1-C · repair 2 solver/state check` | internal Sol/max agent `/root/p1c_repair_solver_review` | `ba6fc4e16d62b5d2b1b6066694930e9cd1dede1b` | Read-only transaction, exact-ratio, explicit-factorization, convergence, and capacity check | Complete GO; P0/P1/P2 `0/0/0`; 85 focused twice, 750,000 exact-ratio cases, concurrency and 2,895/2,896 capacity boundary green |
| `R0/P1-C · repair 2 result check` | internal Sol/max agent `/root/p1c_repair_result_review` | `ba6fc4e16d62b5d2b1b6066694930e9cd1dede1b` | Read-only result and fresh-verification correspondence check | Complete NO-GO; P0/P1/P2 `0/1/0`; accepted exact fresh reduced-residual norm finding |
| `P1-C closure · fresh convergence correspondence` | internal Sol/max agent `/root/p1c_writer` | parent `ba6fc4e16d62b5d2b1b6066694930e9cd1dede1b`; source `3ee920a4659d7bb37f09ed094fe5c0f9651805e1` | One exact fresh-ledger convergence check plus focused regression in two paths | Complete; 86 focused twice, 348 combined twice, 478 v3, 667 repository; no public-schema change |
| `R0/P1-C closure · result correspondence verified` | internal Sol/max agent `/root/p1c_repair_result_review` | `3ee920a4659d7bb37f09ed094fe5c0f9651805e1` | Read-only residual, zero, subnormal, signed-zero, and admitted-asymmetry checks | Complete GO; P0/P1/P2 `0/0/0`; exact fresh check rejects coherent changed evidence while equilibrium remains tolerance-based |
| `R0/P1-C closure · solver regression verified` | internal Sol/max agent `/root/p1c_repair_solver_review` | `3ee920a4659d7bb37f09ed094fe5c0f9651805e1` | Read-only solver/state regression check | Complete GO; P0/P1/P2 `0/0/0`; 86 focused twice plus concurrency, exact-ratio, capacity, ownership, and cache-independence checks |
| `INCOMPLETE R0/P1-C · public-flow output routing` | internal Sol/max agent `/root/p1c_source_review` | first `ba6fc4e`, then closure `3ee920a` | Intended read-only Q8/public-flow completion | Incomplete; interim local tests were green, but two final turns were stopped by erroneous output classification; no verdict counted and no repository change |
| `INCOMPLETE R0/P1-C · public-flow Sol capacity` | internal Sol/max agent `/root/p1c_public_flow_redo` | intended closure `3ee920a` | Fresh minimal local FEM test replacement | Incomplete before work because the selected model was at capacity; no evidence or repository change |
| `R0/P1-C · public-flow replacement verified` | internal Terra/max agent `/root/p1c_public_flow_redo2` | `3ee920a4659d7bb37f09ed094fe5c0f9651805e1` | Fresh local Q8/public-flow and combined Phase 1 verification | Complete PASS; 86 focused twice, 348 combined twice, eight named flows, both Ruff configurations, focused format, clean tree |
| `I0/P1-C · integrated proof` | internal agents `/root/p1c_repair_solver_review`, `/root/p1c_repair_result_review`, `/root/p1c_public_flow_redo2` | integrated head `7ee65c3ea5c50278b17dce63f2569e22412e0ebd` | Parallel read-only focused/combined, broad, and static integration gates | Complete PASS; 86 focused twice, 348 combined twice, 478 v3, 667 repository, both Ruff configurations, focused format, ancestry/static/clean gates |
| `R1-A · physics and balance` | internal Sol/max agent `/root/r1a_physics_balance`; frozen `/private/tmp/pyfem-r1a-7ee65c3` | `7ee65c3ea5c50278b17dce63f2569e22412e0ebd` | Read-only rational Q8, PatchTest8, equilibrium, reaction, work, and orientation review | Complete GO; P0/P1/P2 `0/0/0`; 214 independent checks twice, 86 focused twice, 348 combined twice, hand-coded five-cell PatchTest8 oracle green |
| `R1-A · state lifecycle` | internal Sol/max agent `/root/r1a_state_lifecycle`; frozen `/private/tmp/pyfem-r1a-7ee65c3` | `7ee65c3ea5c50278b17dce63f2569e22412e0ebd` | Read-only identity, generation, transaction, storage, cache, and capacity review | Complete GO; P0/P1/P2 `0/0/0`; 2,020 normal and 1,628 restricted independent checks plus focused identity/linear suites |
| `R1-A · public contracts` | internal Terra/max agent `/root/r1a_public_contracts`; frozen `/private/tmp/pyfem-r1a-7ee65c3` | `7ee65c3ea5c50278b17dce63f2569e22412e0ebd` | Read-only public compile/assemble/solve/verify flow and diagnostics review | Complete GO; P0/P1/P2 `0/0/0`; 348 combined twice, nine named flows, 32 carrier-boundary cases, exact reusable/one-shot equality |
| `I1 · linear slice — public proof complete` | delegating/integration owner | product `7ee65c3`; ledger checkpoint follows | Reconcile P1-C integrated proof and three R1-A verdicts; freeze component-qualified Phase 1 contract | Complete; P0-D/P1-A/P1-B/P1-C bounded slices advance to E3; all deferred breadth remains explicit |
| `DONE · G1 · PyFEM sidebar normalized` | Luna/max `01a0061e-7c27-7d93-b994-878bf8c16a01` | app metadata only; common engineering base remains `f32e7e5` | Rename and archive old migration threads; retain only the active coordinator, current portfolio, and intentional pause points | Complete and archived; 30 threads renamed and 29 historical threads archived; no repository or engineering authority |
| `DONE · R2-E · Legacy coverage mapped` | Sol/max `01a0062b-5deb-79b3-8a73-e768a9ca5fe4` | exact clean `f32e7e5b8eb02f4e58a410871f929a4df22e5ca9` | Read-only mapping of all 154 E0 capabilities to the six generic concepts and seven variation axes | Complete and archived; 95 direct, 56 composed, two questionable, one rejected carrier; report SHA `480332f`; callback delivery was attempted but not acknowledged |
| `DONE · P2-H · Three-operator boundary proved` | Sol/max `01a0062b-6e84-7e22-bf8a-4fbb8f42db09` | exact clean `f32e7e5b8eb02f4e58a410871f929a4df22e5ca9`; source `b1789d1` | New-path-only experimental core for continuum, link/spring, and program-owned load operators sharing one space | Complete and archived; 7 focused twice, 485 v3 twice, exact balance/oracles, one clean commit; callback unavailable; no integration by implication |
| `DONE · R2-F · Vertical deletion cut derived` | Sol/max `01a0062e-c999-7740-89c3-4815948cb99a` | exact clean `f32e7e5b8eb02f4e58a410871f929a4df22e5ca9` | Read-only retain/migrate/delete/defer map for replacing the Phase 1 Q8 spine without a permanent second backend | Complete and archived; strategy B, 33 owned paths, 5,500 gross/1,500 net production deletion floor, report SHA `df8a813`; prompt lacked the now-mandatory literal callback instruction |
| `DONE · S2-A · generic proof simplified and reviewed` | Sol/high writer `/root/s2a_proof_simplifier`; Sol/max reviewer `/root/s2a_final_review` | source `099b51f` on integrated proof `f140b9e`; review base `c5f83b7` | Simplify only `pyfem/v3/core/{__init__.py,generic.py}` and `test/v3/test_v3_generic_core.py`; no feature expansion or public export | Complete GO; P0/P1/P2 `0/0/0`; 649 production and 360 test nonblank lines; 7 focused twice, 485 v3, 674 repository; independent Fraction and split/vectorized oracles plus Ruff, format, path, ancestry, branch, exact-head, and clean gates green |
| `DONE · R3-A · behavioral parity graded` | Luna/high `/root/r3a_behavior_parity_audit` | exact clean `04baa4a` | Grade all 154 capabilities by current public, bounded, experimental, design-only, deferred, or retired evidence | Complete; strict 4/154, bounded executable 57/154, design-backed 139/154; report SHA `f3223fb` |
| `DONE · D3-A · generic spine cut frozen` | Sol/max `/root/d3a_generic_spine_cut_freeze` | exact clean `04baa4a` | Refresh R2-F against simplified proof and freeze terminal ownership, sequence, budgets, reviewers, and first writer | Complete; four linear commits on one exclusive branch; G1 is the six-path unexported direct Q8 system compiler; report SHA `06ba6d7` |
| `DONE · R3-B · state-first frontier selected` | Sol/high `/root/r3b_feature_frontier` | exact clean `04baa4a` | Rank post-cut feature axes and dependencies without recreating v1 class boundaries | Complete; state transaction -> nonlinear/J2 -> thermal coupling -> evolution; no feature writer may overlap the cut; report SHA `f9bd194` |
| `G1 · direct Q8 generic system compiler` | sole Sol/high writer `/root/g1_q8_system_compiler`; branch `agnet/g1-generic-system`, worktree `/private/tmp/pyfem-g1-04baa4a` | exact parent `04baa4a`; initial source `f342b40` | Six paths frozen above; one direct-child commit plus bounded repair; unexported compiler only | Initial source NO-GO; G1-N `0/1/0`, G1-A `0/4/1`; repair 1/2 active; G2 closed |
| `P2-A · final freeze refresh` | internal Sol/max agent `/root/p2a_freeze_draft` | product `7ee65c3`; current main ledger | Read-only final writer-card refresh and public-scope adjudication | Complete; retained genuine public mixed solve, removed optional helpers, froze exact oracle and full gates |
| `R2-A · final topology delta` | internal Sol/max agent `/root/r2a_topology` | product `7ee65c3` | Read-only Q8/Q4/T3 oracle, schema, and public-scope check | Complete; confirmed 15/30/21 and 356/203/14 arithmetic, Q4 corner audit, native widths, and public mixed-slice design |
| `R2-B · final ownership delta` | internal Sol/max agent `/root/r2b_ownership` | product `7ee65c3` | Read-only final P2-A ownership and later-leaf separation check | Complete; confirmed compiler/program/assembly/state ownership and rejected a temporary solver guard |
| `P2-A · contract reconciliation` | internal Sol/max agent `/root/p2a_contract_reconcile` | product `7ee65c3` | Read-only schema/key/offset/audit/path decision comparison | Complete; froze new-family v1 schemas, `gauss-tria3-order1`, one persisted full-offset carrier, transient Q4 corner audit, and element-qualified literals; compiler-only scope recommendation was separately rejected |
| `INCOMPLETE P2-A · mixed blocks — superseded with no commit` | former internal Sol/max agent `/root/p2a_mixed_blocks_writer`; vanished `/private/tmp/pyfem-p2a-7ee65c3`; branch `agnet/p2a-mixed-blocks` | exact parent `7ee65c3ea5c50278b17dce63f2569e22412e0ebd`; zero child commits | Former 13-path Q8/Q4/T3 public mixed-slice writer | Incomplete and superseded; temporary six-path edits disappeared with the worktree, branch still equals the parent, no tests or commit existed, and no recovery/resume is required |
| `R2-C · acceptance matrix` | internal Sol/max agent `/root/r2c_acceptance_matrix` | planning base `7ee65c3` | Read-only independent post-integration review card | Complete; includes exact Fraction operators, disconnected and 13-node shared-interface oracles, all-block mutation matrix, public fresh verification, full gates, and no grade authority |
| `P2-B/P2-C · conditional leaf drafts` | internal Terra/max agent `/root/p2bc_leaf_drafts` | planning base `7ee65c3` | Read-only disjoint material-slot and active-field-layout drafts | Complete but unfrozen; two new paths per leaf; require R2-C confirmation of one immutable partition seed before parallel dispatch |
| `I0/P1-C · numeric repair card` | internal Sol/max agent `/root/p1c_numeric_repair_card` | repair parent `8524c2337a408e69be215a78044d54d6af968a6b` | Read-only exact-ratio, explicit factor/solve, and workspace acceptance design | Complete and clean; exact cross-products and deterministic 4-matrix/6-vector private-backend proposal; no edit or grade authority |
| `I0/P1-C · scalar repair card` | internal Sol/max agent `/root/p1c_scalar_repair_card` | repair parent `8524c2337a408e69be215a78044d54d6af968a6b` | Read-only complete result scalar/tuple preflight inventory | Complete and clean; canonical UUID/identity/generation and scalar/tuple matrix; no edit, grade, or integration authority |
| `I0/P1-C · repair gate matrix` | internal Sol/max agent `/root/p1c_repair_gate` | initial source `c263d95830df8284ab55c82dbd771f4690301283` | Read-only exact post-repair correctness-matrix design | Complete and clean; seven-area matrix plus cross-fix interactions and bounded command set; no edit, grade, or integration authority |
| `P2-A · mixed blocks — proposed freeze draft` | internal Sol/max agent `/root/p2a_freeze_draft`; main `d18800f7df2dcd8ceafb00254316d05cf94714f9` | post-I1 proposal only | Read-only packet design for descriptor-driven Q8/Q4/T3 composition | Complete and clean; proposed 15-node exact oracle and bounded ownership; not frozen, no edit or grade advance |
| `R2-A · topology — Q4/T3 contracts extracted` | internal Sol/max agent `/root/r2a_topology`; `/private/tmp/pyfem-r2a-02cff1c`; detached | `02cff1ca76b02969b91eac6e889de78c7a4ef9e2` | Read-only Phase 2 topology research | Complete and clean; Q4 corner audit, descriptor-local ordering, native-width homogeneous blocks, and literal Q4/T3 oracles proposed; no edit or grade advance |
| `R2-B · ownership — Phase 2 leaves separated` | internal Sol/max agent `/root/r2b_ownership`; `/private/tmp/pyfem-r2b-02cff1c`; detached | `02cff1ca76b02969b91eac6e889de78c7a4ef9e2` | Read-only Phase 2 ownership/backend research | Complete and clean; neutral partition/entity/recipe seam and disjoint leaf waves proposed; no edit or grade advance |

## Task completion audit

The 2026-07-18 audit inspected the actual final turns of the first 17 user-visible
migration tasks rather than relying on titles, idle state, or ledger summaries.
R0-L, P0-H, R0-M, R0-E, P0-D, and R0-N subsequently completed under the corrected
terminal contract. That historical sidebar set contains 29 tasks: 20 properly
complete and nine explicitly incomplete. Current internal tasks and replacements
are recorded separately because they are not part of that user-visible audit set.

- Properly completed: P0-A, P0-B, P0-C, R0-A, R0-B, D0-A, replacement R0-C,
  P0-E, R0-H, R0-I, P0-F, P0-G, R0-L, P0-H, R0-M, R0-E, P0-D, R0-N, P1-A,
  and P1-E.
- Incomplete: initial R0-C, R0-D, R0-G, R0-J, R0-K, R0-F, P1-B's interrupted
  repair turn, the P1-D missing-worktree replacement, and the sidebar R1-B review
  stalled on an unnecessary temporary-file approval.
- Paused: R2-E/P2-H/R2-F and S2-A are complete. The generic proof is integrated,
  simplified, and independently reviewed; no task may edit the Phase 1 production
  path. P2-A is incomplete and superseded; its temporary worktree is gone and its
  branch has zero child commits.
  The topology, ownership, reconciliation, R2-C-card, and leaf-draft tasks are
  historical planning evidence rather than executable authority. The three R1-A
  tasks are complete GO and I1 remains frozen. Luna G1 completed the sidebar
  cleanup and is archived; no implementation or review packet remains active. The
  two Kimi attempts, the twice-routed
  `/root/p1c_source_review`, and the Sol-capacity
  `/root/p1c_public_flow_redo` are explicitly incomplete and provide no evidence;
  replacement `/root/p1c_public_flow_redo2` completed cleanly.
- Completed replacements provide valid evidence for their own task IDs; they do not
  change the recorded status of the tasks they replaced.
- No incomplete writer commit was integrated. The exact-base internal replacement
  independently completed R1-B GO; the remaining incomplete entries are process
  history, not contamination or unfinished code on `v3`.

## P1-A component evidence — integrated and independently green

P1-A returned valid terminal final and callback for source
`a749972dca8a97ae4801017ac686d114225432f3`, exactly one commit on frozen parent
`b18b262b97068086b9c61618d06a2d14a50046cd` and exactly the ten authorized paths.
Its reported 53 focused, 209 combined twice, 14 reference, both Ruff, format, 339
v3, 528 full-suite, ancestry, path, whitespace, and import gates were complete.
I0 also reproduced the 53 focused tests normally and at the 640-digit limit.

The source remains unintegrated because I0 accepted these seven P1 findings from
the frozen total-boundary and semantic-consistency contract:

1. accepted exact string names and source labels can produce unbounded diagnostics
   and literal control characters instead of bounded escaped output;
2. `compile_program` accepts an exact `CompiledModel` missing half of its canonical
   slots because only six of twelve fields are preflighted;
3. a model DOF count that exceeds its recorded signed dense-index dtype wraps a
   semantic DOF negative and silently loses a prescribed constraint;
4. fixed-width subtraction in `free_dofs` ordering validation can wrap, accept
   out-of-range entries, and leak raw `IndexError`;
5. evaluation accepts two semantic array fields sharing the same finalized storage,
   contrary to the separate-finalization invariant;
6. plan-bearing programs can report empty capability channels and empty
   entity/source maps when the manifest and fingerprint are recomputed from those
   inconsistent fields; derived meaning is not revalidated; and
7. recursively nested invalid entity IDs can leak raw `RecursionError` instead of
   one deterministic `ProgramEvaluationError`.

Independent bounded review otherwise found the affine composition, hard-coded
oracle, canonical load reduction, derivatives, cycle handling, zero-free and
identity plans, and graph complexity sound, including a separate 40-case affine/load
matrix. Repair 1/2 preserves the frozen outcome, owned paths, numeric policy, and
non-goals; it adds only the smallest validation/diagnostic changes and regressions
for these seven cases.

Repair 1/2 returned replacement source
`587e3fc41a0a0fbf9ef3d7e069d8eca591006001`, exactly one commit on `b18b262` and
the same ten owned paths. Its own final and callback report all seven cases closed,
59 focused tests in both digit modes, 215 P0-D/P1-A tests in both modes, 14
references, both Ruff configurations, focused format, 345 v3 tests, 534 full-suite
tests, path/import/whitespace checks, and only the existing Numba/SciPy notices.
I0 then reviewed the repair delta and directly reproduced the original seven cases,
all of which now fail through the required structured program errors. It passed 59
focused tests in both digit modes, 105 model/program tests in both modes, the 14
reference tests, both Ruff checks, and bounded long/control-character probes. The
spec and math reviewers returned GO with zero findings.

The carrier reviewer returned NO-GO with three further P1 gaps accepted by I0:

1. six initialized `CompiledModel` fields are required but discarded without
   exact semantic validation, so invalid visible model meaning can retain the old
   model fingerprint and enter program compilation;
2. a dependent one-entry CSR row can carry a zero factor after outer program
   identity is recomputed; and
3. constraint/load IDs, sources, targets, affine values, and visible plans/maps can
   contradict the retained normalized program meaning after outer identity is
   recomputed.

These contradicted the frozen validated-model, flattened nonzero affine-plan,
content-identity, and total-boundary clauses. Repair 2/2 returned final replacement
source `ab64e0219c11b5fc4155703eb8ade32939f90064`, exactly one commit on the frozen
base and exactly the same ten paths. It closes the three general seams with complete
12-slot compiled-model reconstruction, finite nonzero dependent CSR coefficients,
and an exact typed compiler-owned normalized-meaning witness covering model,
coordinate, constraint, load, target, affine, entity, and source correspondence.

The worker reported 72 focused tests and 228 combined P0-D/P1-A tests in both
normal and strict-digit modes, 14 reference tests, both Ruff configurations,
focused format, 358 v3 tests, 547 full-repository tests, clean ancestry/path/import/
whitespace gates, and only existing warnings. I0 verified exact ancestry, path
ownership, and the final delta; reproduced 72 focused tests in both modes, the prior
failure matrix, bounded diagnostics, 228 combined tests in both modes, 14 references,
both Ruff configurations, focused format, and zero forbidden imports; and integrated
the source without conflict as `b7316be8befba64df0a576df6b7fd8cdcf9b9873`.
The integrated branch then passed 358 v3 tests with 40 existing SciPy and four
cold-cache Numba notices, followed by 547 full-repository tests with only the 40
existing SciPy notices.

Independent final carrier review returned GO with zero P0/P1/P2 findings. Its
12-slot model matrix rejected 12 of 12 alterations; its normalized-witness matrix
rejected 10 of 10 alterations through structured errors; and it confirmed detached,
owning, read-only witness storage. The full-DOF witness is correctness-valid but
duplicates O(full DOFs) storage and validation work per program. That measured-cost
obligation is nonblocking and is recorded for comparison with a later compact
per-referenced-DOF witness before performance claims. The three P1-A rows therefore
advance to provisional E2; P1-B may open from the frozen card above.

## P1-B component evidence — integrated and independently green

The original P1-B writer returned valid terminal final and callback for source
`ab02297c052912cb027ac6963b37667bf61f3e97`, exactly one commit on frozen parent
`4ed975795a4917fe40ed0203c981980ac5bea9cc` and exactly the six authorized paths.
It reported 26 focused assembly tests and 144 combined P0-D/P1-A/P1-B tests in both
normal and 640-digit modes, 27 named reference tests, both Ruff configurations,
focused format, 384 v3 tests, 573 full-repository tests, clean ancestry/path/import/
determinant/whitespace gates, and only existing SciPy/Numba notices. I0 reproduced
the 144 combined tests in both modes, 27 references, both Ruff configurations, and
focused format. The architecture and numerical reviewers returned GO with zero
findings; the numerical review independently confirmed the literal `K=M/360`,
affine RHS/derivative relations, two-cell raw/canonical counts, scale extremes,
positive-orientation policy, and stated complexity.

The carrier reviewer returned NO-GO with two P1 findings accepted by I0:

1. preparation checked compiled Q8 array shapes, finiteness, partition/gradient
   sums, descriptor identities, and rebuilt visible content identity, but did not
   prove that quadrature points/weights and topology shape/gradient arrays still
   corresponded to the exact captured descriptor bindings. Re-identifying a model
   with `2 * parent_gradients` passed preparation and produced four times the
   reference stiffness; and
2. NumPy float64 dtypes carrying metadata compare equal to canonical float64, so
   metadata-bearing topology/quadrature/formulation/material binding outputs could
   bypass the exact output boundary.

P1-E completed repair 1/2 as replacement source
`8e94454c63a76fdc1505d8ec3ec6341128448aab`, still exactly one child of the frozen
base and within the same six paths. Preparation now invokes the exact captured
quadrature and topology bindings once each, validates exact tuple/plain-array/shape/
finite/metadata-free-float64 output, and compares every reference carrier byte for
byte with the compiled values. Evaluation continues to invoke only formulation and
material. Metadata-bearing outputs reject on both preparation and evaluation
boundaries. The focused tests cover altered re-identified parent gradients and
quadrature points, stateful preparation binding failures, metadata-bearing outputs,
and repeated call counts.

I0 reviewed the repair delta, reproduced 152 combined model/program/assembly tests
in both digit modes, both Ruff configurations, and focused format, then integrated
the source without conflict as `daad22796a6b178ba9fdb33d0e444b5be0f830e7` on top
of the current ledger. The integrated branch passed the same 152 combined tests in
both modes, all 27 named references, both Ruff configurations, focused format, 392
v3 tests, and 581 full-repository tests. Forbidden-import, root-export,
determinant-absolute-value, whitespace, exact-head, and clean-status checks are all
green. Broad warnings are the existing 40 SciPy sparse-format notices and four
cold-cache Numba parallelization notices; the warm full suite emitted only the 40
SciPy notices. The original repair turn and its first fork changed no files and
remain recorded incomplete for process accuracy.

## R1-B reviewer card — `HORIZON_FROZEN`

**ID/title:** `R1-B · assembly plan — contribution invariants verified`

**Outcome:** independently decide whether integrated P1-B preserves the frozen
finite-element contribution meaning and closes both accepted repair findings. This
is one read-only component review of integrated code commit `daad227`; it does not
review or design P1-C state, solve, reaction, result, or public-flow ownership.

**Coverage rows:** `ASM-COO`, `ASM-PREPARE`, `ASM-EXTERNAL`, `ASM-GATHER`, and
`ASM-TANGENT`. They remain component-bounded E1 until this review is adjudicated;
even GO advances them only to provisional E2, not public-flow E3.

**Read-only scope:** the P1-B Horizon above; `pyfem/v3/assembly/**`;
`test/v3/test_v3_assembly_plan.py`; and the exact P0-D/P1-A model, registry,
program, and focused-test dependencies needed to check composition. Repository
edits, staging, commits, branch changes, installs, external access, new tasks, and
P1-C implementation are forbidden. Temporary local correctness data belongs under
`/private/tmp` only.

**Finite-element correctness matrix:**

1. reproduce the two accepted repair cases: a re-identified compiled model with
   altered parent gradients or another reference carrier must fail preparation, and
   metadata-bearing float64 outputs must fail at topology/quadrature/formulation/
   material boundaries;
2. verify preparation invokes the exact captured quadrature/topology bindings in
   the bounded count, compiled reference carriers correspond exactly, and repeated
   evaluation invokes only formulation/material;
3. independently check the literal `K=M/360` Q8 operator, affine full/reduced RHS
   and derivative signs, two-cell raw attribution/canonical coalescing, deterministic
   order, structural zeros, prescribed-load retention, identity/fully prescribed
   reduction, and fresh-value versus shared-topology ownership;
4. check scale extremes, positive orientation, sign-changing/inverted and relative
   singularity rejection, finite accumulation, and absence of determinant absolute-
   value repair;
5. check exact model/program/plan live/content compatibility, malformed carriers,
   plan reconstruction, separate/detached storage, binding failure translation,
   bounded diagnostics including the 640-digit mode, and no raw incidental exception;
6. verify no backend, dense global operator/prolongation, solver, state, candidate,
   reaction, result-ledger, adapter, prototype, legacy, or root-public ownership
   entered P1-B; and
7. report whether the stated preparation/evaluation complexity and the inherited
   P1-A full-DOF witness obligation remain honest and nonblocking.

**Required evidence:** exact clean-base preflight; a bounded independent local
correctness matrix; 34 focused assembly tests normally and with
`PYTHONINTMAXSTRDIGITS=640`; 152 combined model/program/assembly tests in both modes;
the 27 named references; both Ruff configurations; focused format; `test/v3` if the
configured environment permits; forbidden-import/root-export/determinant scans;
`git diff --check`; final exact HEAD and clean status. Existing SciPy/Numba notices
are separated from product findings. The integrated full-repository proof above is
I0 evidence and need not be repeated unless the reviewer finds a cross-suite reason.

**Terminal contract:** return P0/P1/P2 findings with file/line or command evidence,
then GO or NO-GO. Immediately before the final, send I0 one callback beginning
`R1-B COMPLETE` or `R1-B BLOCKED` with exact base, verdict, counts, strongest
evidence, warnings/deferred items, clean state, and smallest adjudication action.
The final begins `RESULT: COMPLETE` or `RESULT: BLOCKED`. Use ordinary local
finite-element correctness language only.

## R1-B component evidence — replacement GO

The original sidebar R1-B reviewer passed exact-base preflight, confirmed the exact
34/152/27 collections, ran the required repository selections and `test/v3` green,
then requested approval to create a temporary edge-case file despite the read-only
packet. That approval did not resolve. I0 archived and visibly renamed the task
`INCOMPLETE`; it produced neither a final verdict nor callback and changed no
repository file. Its recovered results are supporting evidence only.

I0 restored a detached clean worktree at exact frozen base
`e11a07c0cc5c00e956c3d537ac94272f120621c0` and assigned an internal Sol/max
replacement with an absolute no-file rule. It independently read the contracts and
implementation and returned GO with zero P0/P1/P2 findings. It reran 34 focused and
152 combined cases normally and at `PYTHONINTMAXSTRDIGITS=640`, all 27 named
references, both Ruff configurations, focused format, forbidden-import/root-export/
determinant scans, committed diff checks, and final exact-head/clean-status gates.
The named references emitted three existing cold-cache Numba notices. It relied on
I0's already-recorded 392-test v3 and 581-test repository proofs rather than
repeating the full repository suite.

The replacement authored no persistent helper and, after I0 bounded the review,
added no new inline runtime case. This is accepted for provisional E2 because the
frozen runtime matrix is explicitly present in the required checked-in cases, both
original repair findings have direct regressions, the replacement independently
checked captured-carrier correspondence, metadata-free boundaries, call separation,
order/sign, geometry policy, identity/storage/diagnostic behavior, scope and cost,
and the earlier independent numerical and carrier reviews supplied the literal Q8,
affine, geometry, ownership, and failure-case scrutiny that found the repaired gaps.
This is component proof only: `ASM-COO`, `ASM-PREPARE`, `ASM-EXTERNAL`,
`ASM-GATHER`, and `ASM-TANGENT` advance to provisional E2, never E3.

## Active watchdogs

None. Direct terminal callbacks and bounded native waits are active; unchanged
thread state does not trigger a model-consuming polling loop.

## P0-D component evidence — independent GO

The original source `92f57e82ac6639c9f5e769ea2b28c6178673f15a`
passed its reported gates but I0 accepted two P1 gaps: valid normalized exact
integers were rejected as material parameters, and the direct test set reused the
production Q8 shape implementation rather than independently establishing its
declared local order. Repair round 1/2 replaced it with source
`f21aaed2c97e98e63f58b77d0e301c3a2107e243`, still exactly one child of frozen base
`2842ef84b86f04f82587f5fc378584f30f6b62bd` and within the seven owned packet
paths.

I0 inspected the repair delta, confirmed exact integer/float-but-not-bool lowering
through finite `float64` before domain checks, bounded failure for nonrepresentable
integers, and hard-coded Q8 nodal and center-gradient expectations independent of
the shipped implementation. The repaired source worktree passed 156 focused tests
normally and with `PYTHONINTMAXSTRDIGITS=640`, 22 direct Q8 component tests, both
Ruff configurations, and focused format.

The exact repaired source was integrated without conflict as
`79060abb054d82fd34ac85f000e44d9de5947d60`. On that branch commit, I0 reproduced
156 focused tests in both digit modes, 22 direct Q8 component tests, both Ruff
configurations, focused format, 286 v3 tests, and 475 full-repository tests. The
prototype-import exclusion scan, `git diff --check`, exact-head, and clean-status
checks passed. The broad warm-cache proof emitted only 40 existing SciPy
`SparseEfficiencyWarning` notices; cold-cache component/v3 runs also emitted known
Numba performance notices from existing numeric kernels.

R0-F stopped before review because I0 supplied an incorrect expanded full hash; it
made no finding. Replacement R0-N completed both terminal signals at the verified
exact clean integrated commit with GO and zero P0/P1/P2 findings. Its 99 independent
checks covered hard-coded Q8 interpolation/gradients/quadrature, formulation and
material meaning, membership/maps, ownership, determinism, live/content identity,
geometry/numeric boundaries, malformed descriptor outputs, and architecture
exclusions. It independently reproduced 156 focused tests in both digit modes, 22
direct Q8 tests, 286 v3 tests, 475 full tests, both Ruff gates, focused format,
whitespace/ancestry/path checks, and the zero-hit forbidden-import scan. The 18 rows
therefore advance to `provisional` E2 at `79060abb`; no public vertical slice or E3
claim is made.

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

P1-A integration and independent component evidence at
`b7316be8befba64df0a576df6b7fd8cdcf9b9873`:

- final source `ab64e0219c11b5fc4155703eb8ade32939f90064` is exactly one commit on
  frozen base `b18b262b97068086b9c61618d06a2d14a50046cd`, changes exactly the ten
  authorized paths, and returned both terminal signals from the original task;
- I0 reproduced 72 focused program tests and 228 combined P0-D/P1-A tests in both
  normal and 640-digit modes, 14 constraint/load/MPC reference tests, both Ruff
  configurations, focused format, clean diff/path/ancestry checks, and a zero-hit
  forbidden-import scan before serial integration;
- the integrated branch passed 358 v3 tests with 40 existing SciPy plus four
  cold-cache Numba notices and 547 full-repository tests with only the 40 existing
  SciPy notices; and
- independent carrier review returned GO with zero P0/P1/P2 findings after a
  12-slot compiled-model matrix, a ten-case normalized-witness matrix, direct
  detachment/ownership checks, and strict-digit focused replay. The three program
  rows are provisional E2; no public flow or E3 claim is made.

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
- P1-A's exact `ProgramMeaningWitness` currently embeds a separately finalized full
  `DofPlan`. This is correctness-valid and preserves total-boundary proof, but it
  duplicates O(full DOFs) storage per compiled program and repeats full-layout
  validation during evaluation. Measure representative combined P1-B/P1-C cost
  before performance claims and compare with a compact per-referenced-DOF witness;
  any change must retain the adversarial model/program consistency matrix.

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
- 2026-07-18: the user requested a pause and a clearer status view while P0-D was
  already executing. P0-D may finish its bounded packet and send its terminal
  callback, but I0 will not integrate or request repair and will not dispatch R0-F
  or later work until explicit resume. At the pause request, 42 focused compiler
  tests, both Ruff gates, formatting, and 282 v3 tests were green; the repository-
  wide gate and source commit were still pending. A compact dashboard snapshot was
  prepared from this ledger; this ledger remains the only mutable authority.
- 2026-07-18: P0-D completed both terminal signals with source commit
  `92f57e82ac6639c9f5e769ea2b28c6178673f15a`, exactly one child of its frozen
  base. The task reported 152 focused tests in both digit modes, 22 direct Q8
  component tests, 282 v3 tests, 471 full tests, all required static gates, and a
  clean worktree. Per the requested pause, I0 recorded but did not review,
  cherry-pick, or integrate the source and did not dispatch R0-F or later work.
- 2026-07-19: the user resumed the migration. I0 verified P0-D ancestry and path
  ownership, inspected the full compiler/carrier/test diff, reproduced 152 focused
  tests in normal and strict-digit modes plus 22 direct component tests, and kept
  the target branch clean. It accepted two P1 findings: valid normalized integer
  material scalars were rejected, and the claimed direct Q8 evidence did not
  independently prove the declared local-node convention. P0-D repair round 1/2
  was sent to the original Sol/max task; no source was integrated and R0-F remains
  undispatched.
- 2026-07-19: P0-D repair round 1/2 returned exact replacement source
  `f21aaed2c97e98e63f58b77d0e301c3a2107e243`. I0 reviewed and reproduced the
  repair evidence, integrated it without conflict as
  `79060abb054d82fd34ac85f000e44d9de5947d60`, and passed 156 focused tests in both
  digit modes, 22 direct Q8 tests, both Ruff gates, focused format, 286 v3 tests,
  475 full-repository tests, prototype-import exclusion, whitespace, exact-head,
  and clean-status checks.
- 2026-07-19: I0 then supplied an incorrect expanded full hash to R0-F. R0-F
  correctly stopped at clean preflight and evaluated no compiler invariant; it is
  visibly marked incomplete. I0 verified the actual full hash, renamed the failed
  task, and dispatched exactly one Sol/max replacement R0-N
  (`019f761d-4d85-7b72-9d80-857fab45aef8`) from the exact clean integrated commit.
  No parallel migration writer is active.
- 2026-07-19: R0-N completed both terminal signals with GO and zero findings at
  exact clean `79060abb`. Its 99-check independent matrix and every focused,
  strict-digit, direct-Q8, static, v3, full-repository, path, whitespace, and import
  gate passed. I0 adjudicated the GO, advanced the 18 P0-D dependency rows to
  `provisional` E2, extracted the three P1-A program rows to contracted E1, and
  froze the P1-A affine-program Horizon. No public flow or E3 evidence is claimed.
- 2026-07-19: I0 committed the R0-N adjudication and frozen P1-A Horizon as
  `b18b262b97068086b9c61618d06a2d14a50046cd`, then dispatched exactly one Sol/max
  writer, `P1-A · program compiler — affine plan canonical`
  (`019f764f-2643-75b2-a8b5-323ed58351a2`), from that exact clean base. The three
  program rows are `implementing` at E1; P1-B and all other migration packets remain
  closed pending P1-A terminal final, callback, and I0 integration review.
- 2026-07-19: P1-A returned valid terminal final and callback with one exact source
  commit `a749972dca8a97ae4801017ac686d114225432f3` and all preregistered reported
  gates green. I0 verified ancestry and path ownership, reproduced focused proof,
  and completed three bounded semantic reviews. Seven P1 total-boundary and carrier
  consistency findings were accepted; source integration was withheld and repair
  round 1/2 was returned to the original task. P1-B remains closed.
- 2026-07-19: the user requested a pause after the already-running P1-A repair pass.
  I0 will validate and record only that task's terminal result, then stop before
  semantic re-review, integration, a second repair, an independent reviewer, P1-B,
  or any other packet. Resumption requires an explicit user instruction.
- 2026-07-19: P1-A repair 1/2 returned replacement source
  `587e3fc41a0a0fbf9ef3d7e069d8eca591006001` with a valid terminal final and
  callback. I0 mechanically verified its exact parent, one-commit ancestry, ten-path
  scope, clean diff, and clean worker head, and recorded its reported proof. The
  migration is now paused before semantic re-review, local gate reproduction,
  integration, reviewer dispatch, P1-B, or evidence advancement.
- 2026-07-19: on user resumption, I0 reviewed repair 1/2, reproduced the original
  seven cases as closed, reran focused/strict/reference/static proof, and received
  independent GO from spec and math reviewers. The carrier reviewer found three
  further P1 model/program cross-representation gaps; I0 accepted them against the
  frozen Horizon and withheld integration. Final repair 2/2 was dispatched to the
  original Sol/max P1-A task. P1-B and all other writers remain closed.
- 2026-07-19: P1-A repair 2/2 returned exact replacement source `ab64e02` with
  valid terminal signals, one-commit ancestry, and exact ten-path ownership. I0
  reviewed and integrated it as `b7316be`, reproduced 72 focused and 228 combined
  tests in both digit modes, 14 references, both Ruff configurations, format and
  import gates, then passed 358 v3 and 547 full-repository tests. Independent final
  carrier review returned GO with zero findings. The three P1-A rows advance to
  provisional E2.
- 2026-07-19: I0 froze the P1-B reference assembly Horizon from three independent
  read-only design/numerical reviews. It fixes one zero-field internal linear
  contribution request, raw-attributed plus canonical COO topology, scale-safe
  positive-Jacobian Q8 integration, exact affine reduction/sign semantics, literal
  operator/RHS oracles, five component rows, disjoint assembly-only ownership, and
  an explicit P1-C boundary. Carrier, numerical, and request-ownership rechecks
  returned final GO after the diagnostic, alias, compatibility, singularity, and
  cost clauses were made exact. The clean commit containing this card is the sole
  permitted P1-B dispatch base.
- 2026-07-19: I0 committed the frozen P1-B Horizon as
  `4ed975795a4917fe40ed0203c981980ac5bea9cc`, then dispatched exactly one Sol/max
  writer, `P1-B · assembly plan — contributions composed`
  (`019f76f1-fffd-7772-ba9c-4e17d3ee9f63`), from that exact clean base. Its
  preflight confirmed the exact commit and empty worktree. The five assembly rows
  are implementing at E1; no parallel assembly writer, reviewer, or P1-C owner is
  active.
- 2026-07-19: P1-B returned valid terminal final and callback with exact source
  `ab02297c052912cb027ac6963b37667bf61f3e97`, one child of the frozen base and
  exactly six owned paths. I0 reproduced the focused/combined/reference/static
  proof and completed three independent bounded reviews. Architecture and numerical
  reviews returned GO; the carrier review found two accepted P1 gaps in compiled
  reference-recipe correspondence and metadata-bearing binding-output dtypes.
  Integration was withheld and repair 1/2 was requested.
- 2026-07-19: the original P1-B repair turn requested an avoidable destructive
  worktree transition and was interrupted before editing. Its first fork, P1-D,
  inherited the app-removed worktree path and was also stopped before preflight or
  editing. Both tasks are visibly marked `INCOMPLETE`. I0 created one exact-source
  branch solely for a clean app worktree and dispatched replacement Sol/max writer
  P1-E (`019f7736-ec41-7560-9d65-472551b4a6fe`). P1-E passed exact source, parent,
  one-commit, and clean preflight; it is the sole assembly writer. No P1-B code is
  integrated, no reviewer is active, and P1-C remains closed.
- 2026-07-19: P1-E returned both terminal signals with replacement source
  `8e94454c63a76fdc1505d8ec3ec6341128448aab`, exactly one child of the frozen base
  and exactly six owned paths. I0 reviewed the repair, reproduced the 152 combined
  tests in both digit modes and all static gates, and integrated it without conflict
  as `daad22796a6b178ba9fdb33d0e444b5be0f830e7`. The integrated branch then passed
  152 combined tests twice, 27 references, 392 v3 tests, 581 repository tests, both
  Ruff configurations, format, import/root/determinant/whitespace scans, and clean
  status. Only the existing SciPy/Numba notices remain.
- 2026-07-19: I0 froze the R1-B independent assembly reviewer card after integration.
  It covers the five P1-B rows, both accepted repair cases, literal Q8 and affine
  oracles, geometry policy, identity/storage/diagnostic boundaries, scope exclusions,
  and cost honesty. P1-C remains closed until the one read-only Sol/max reviewer is
  terminal, every finding is adjudicated, and the verdict is GO.
- 2026-07-19: I0 committed the R1-B card as
  `e11a07c0cc5c00e956c3d537ac94272f120621c0` and dispatched exactly one read-only
  Sol/max reviewer, `R1-B · assembly plan — contribution invariants verified`
  (`019f7750-8745-7e23-aaf2-35f50ec79fc8`), from that exact clean base. No assembly
  writer, second reviewer, or P1-C owner is active.
- 2026-07-19: the sidebar R1-B reviewer passed exact preflight and the required
  repository gates, including 392 v3 tests, then stalled on approval for an
  unnecessary temporary file before its separate edge-case matrix. I0 renamed it
  `INCOMPLETE R1-B · assembly plan — temp-file approval stalled` and archived it.
  The task produced no final/callback and no repository change; its partial evidence
  is supporting only.
- 2026-07-19: I0 restored an exact detached `e11a07c` worktree and completed one
  no-file internal Sol/max replacement review. It returned GO with zero P0/P1/P2
  findings after 34 focused and 152 combined cases twice, 27 references, both Ruff
  configurations, format/static/diff/final-clean gates, and independent semantic
  source inspection. I0 adjudicated the five P1-B rows to provisional E2. The
  migration is paused with no active owner; the exact next action on resume is to
  freeze P1-C before dispatch.
- 2026-07-19: at the user's request, I0 planned the next large pass with three
  parallel read-only Sol/max analyses at exact clean `1957902`: P1-C state/public
  ownership, an independent exact rational Q8 solve/verification matrix, and the
  Phase 2 dependency DAG. I0 adjudicated the remaining choices, froze P1-C at
  component-qualified E1, adopted the eleven-stage Phase 1/2 critical path and two
  disjoint Phase 2 writer waves, and recorded an I2 pause boundary. No migration
  implementation task was dispatched. The next activation uses one P1-C writer,
  two read-only Phase 2 researchers, and reserves the fourth slot for I0.
- 2026-07-19: the frozen P1-C Horizon, exact rational oracle, eleven-stage worker
  path, serial I0 gates, and safe Phase 2 parallel lanes were committed as
  `02cff1ca76b02969b91eac6e889de78c7a4ef9e2`. That exact commit is the E1 proof
  anchor and common planned base for P1-C, R2-A, and R2-B. This follow-up records
  the hash only; no implementation or research task has been dispatched.
- 2026-07-19: the user resumed execution and authorized isolated branches,
  worktrees, Sol/max, and optional Kimi review. I0 created exact-base worktrees for
  one sole P1-C writer and two detached read-only Phase 2 researchers, verified all
  three clean at `02cff1c`, and dispatched internal Sol/max agents
  `/root/p1c_writer`, `/root/r2a_topology`, and `/root/r2b_ownership`. P1-C alone
  may edit its frozen paths. R2-A/R2-B may edit nothing. I0 remains the fourth-slot
  integration owner; Kimi will challenge the returned P1-C diff, not become a
  second writer.
- 2026-07-19: R2-A and R2-B returned terminal clean read-only reports. I0 adopted
  their proposed native-width homogeneous-block direction, descriptor-local source
  ordering, Q4 corner determinant audit, one numeric connectivity owner, neutral
  partition/boundary-entity/operator/vector seeds, typed leaf products, serial
  recipe union, and storage-only backend boundary as research inputs only. No code,
  capability grade, or Horizon state changed. Follow-up read-only planning produced
  proposed P2-B/P2-C, P2-D/P2-E, and P2-F/P2-G cards; they remain non-authoritative
  until their recorded serial gates.
- 2026-07-19: independent topology-oracle work rejected R2-A's copied
  `36-node / 72-DOF` mixed-fixture arithmetic. The exact small oracle is instead
  one disconnected Q8/Q4/T3 cell each: 15 nodes, 30 full DOFs, 21 reduced DOFs,
  literal rational operator/solution/reaction evidence, and raw block offsets
  `(0,256,320,356)`.
  A separately checked larger internally connected patch may later serve breadth
  and measurement, but node/DOF totals must always derive from literal coordinates
  and connectivity. `MESH-HETERO` is a workflow seed, not a row in the authoritative
  154-row inventory; P2-A may cite existing component-qualified IDs only.
- 2026-07-19: P1-C returned terminal source
  `c263d95830df8284ab55c82dbd771f4690301283`, exactly one clean child of `02cff1c`
  and exactly the frozen 13 paths. Its final worker proof was 31 focused and 183
  combined tests in both digit modes, 27 references, four public flows, 423 v3,
  612 repository, both Ruff configurations, format/static/ownership gates, and
  bounded dense-reference measurements. I0 withheld integration.
- 2026-07-19: Kimi was attempted as a read-only second opinion on P1-C. The first
  isolated challenge could not obtain ordinary read operations through the local
  ACP approval bridge. One final retry used a disposable exact clone so read tools
  could be approved; it timed out at ten minutes without a report. Both original
  source and disposable clone remained clean. These attempts are tooling evidence
  only, close no review gate, and will not be repeated in this pass.
- 2026-07-19: I0 and independent Sol/max checks reproduced seven P1-C repair areas:
  discard-versus-accept final-lock ordering; owning and identity-disjoint arrays;
  truthful retained and freshly recomputed convergence policy; an exact strict
  pivot boundary; first-factorization memory accounting; total result-carrier
  diagnostics; and retained/fresh program-evaluation correspondence. I0 dispatched
  repair 1/2 to the original sole writer as one child of `c263d958`. P1-C remains
  component-qualified E1 and unintegrated; R1-A and P2-A remain closed.
- 2026-07-19: after the user reaffirmed coordination-first execution, I0 used the
  remaining available agent slot for `/root/p1c_repair_gate`, a read-only Sol/max
  task that converts the seven accepted areas into an exact independent acceptance
  matrix. It owns no path and cannot integrate or advance a grade.
- 2026-07-19: `/root/p1c_repair_gate` completed cleanly with an exact seven-area
  matrix. It additionally required the 256 MiB boundary to distinguish the private
  solver workspace from fresh cache-independent verification. Repair 1 initially
  chose four reduced matrices plus eight vectors on reuse, with 2,895 reduced DOFs
  admitted and 2,896 rejected. Later I0 measurement showed that the two general
  `np.linalg.solve` calls allocate opaque matrix-sized scratch, so that formula is
  not yet truthful; repair 2 must own bounded solve scratch before retaining any
  boundary. Verification remains separate and must leave cache statistics unchanged.
- 2026-07-19: `/root/p2a_freeze_draft` completed cleanly with a proposed, non-frozen
  P2-A card. It derived 15 nodes, 30 full DOFs, 21 reduced DOFs, 356 raw full pairs,
  203 raw reduced pairs, raw offsets `(0,256,320,356)`, and integration-point
  offsets `(0,9,13,14)` from one literal disconnected Q8/Q4/T3 cell each. I1 must
  still inspect final P1-C schemas for Q8-named persisted meaning before freezing;
  no P2-A implementation is authorized.
- 2026-07-19: P1-C repair 1/2 returned clean source `8524c2337a408e69be215a78044d54d6af968a6b`,
  one child of `c263d958`, with 45 focused tests twice, 307 combined tests twice,
  27 references, four public flows, 437 v3 tests, 626 repository tests, and green
  Ruff/format/static/path gates. I0 withheld integration after direct reproduction
  and two fresh Sol/max reviews found five remaining roots: opposite-order discard
  atomicity; exact non-power-of-two pivot comparison; verifiable convergence
  meaning; bounded solve scratch; and exact scalar/tuple preflight. Repair 2/2 was
  dispatched to the original sole writer as one child of `8524c233`; P1-C remains
  E1 and R1-A/P2-A remain closed.
- 2026-07-19: repair-2 numeric support completed read-only. It selected one shared
  exact `float.as_integer_ratio()` cross-product comparator, explicit bounded
  Cholesky factorization and two-vector forward/back substitution, scalar private
  matrix checks, and the exact private-backend formula
  `8 * (4*n*n + 6*n)`: 2,895 reduced DOFs admit 268,331,760 bytes and 2,896 reject
  268,517,120 bytes. Neither support task edits source or advances evidence.
- 2026-07-19: repair-2 scalar support completed read-only. It enumerated exact
  schema/identifier/tuple/bool/int/float slots plus canonical inner
  `InstanceId`/`StateGeneration` UUID payloads, required direct state-validation
  totalization, and produced the normal/640-digit comparison-subclass and partial
  carrier matrix. Source preflights were present at final read; the sole writer
  still owns the required complete test evidence.
- 2026-07-19: P1-C repair 2/2 returned clean source `ba6fc4e`, one child of
  `8524c233`, with explicit bounded scalar Cholesky and triangular solve, exact
  integer cross-product pivot comparison, atomic accept/discard ordering, truthful
  convergence records, total result-carrier preflight, and exact private workspace
  formula `8 * (4*n*n + 6*n)`. Worker proof was 85 focused tests twice, 347
  combined tests twice, 477 v3, 666 repository, both Ruff configurations, focused
  format, and clean ownership/static gates. The solver/state reviewer returned GO;
  the result and public-flow reviewers independently found one remaining P1 in
  fresh reduced-residual norm correspondence, so I0 withheld integration.
- 2026-07-19: one-issue convergence closure `3ee920a`, direct child of `ba6fc4e`,
  exact-compared the recorded convergence norm to the independently rebuilt fresh
  residual while leaving equilibrium acceptance tolerance-based. It touched only
  `pyfem/v3/results/verification.py` and the focused linear-analysis test. Worker
  proof was 86 focused twice, 348 combined twice, 478 v3, and 667 repository.
  Independent result and solver/state checks returned GO with P0/P1/P2 `0/0/0`.
  The original public-flow completion task was explicitly incomplete after two
  erroneous output-routing stops, and a fresh Sol/max replacement did not start
  because the model was at capacity. Terra/max replacement
  `/root/p1c_public_flow_redo2` completed 86 focused twice, 348 combined twice,
  eight named FEM flows, Ruff, format, and clean-tree proof.
- 2026-07-19: I0 cherry-picked the four P1-C commits serially as `c9b9480`,
  `bfc1d80`, `d78201d`, and `7ee65c3`. Parallel integrated gates at exact clean
  `7ee65c3` passed 86 focused tests twice, 348 combined tests twice, 478 v3 tests,
  667 repository tests, both Ruff configurations, focused format, ancestry,
  ownership, dependency, and explicit-solver scans. Broad warnings were the
  existing 40 SciPy sparse-conversion notices plus four cold-cache Numba
  transformation notices. P1-C advances to component-qualified E2; only I1 after
  R1-A may assign E3.
- 2026-07-19: I0 dispatched three parallel read-only R1-A tasks at frozen detached
  worktree `/private/tmp/pyfem-r1a-7ee65c3`: Sol/max physics/balance, Sol/max state
  lifecycle, and Terra/max public contracts. They own no paths and cannot advance
  a grade. P2-A implementation remains closed until I1 adjudicates all three.
- 2026-07-19: all three R1-A tasks returned GO with P0/P1/P2 `0/0/0` at exact
  clean `7ee65c3`. Physics/balance passed 214 independent checks in both digit modes,
  exact fractional one-cell work/reaction balance, and an independently integrated
  five-cell PatchTest8 oracle. State lifecycle passed 2,020 normal and 1,628
  restricted independent checks across transaction orderings, identity/generation,
  storage, cache independence, and capacity. Public contracts passed 348 Phase 1
  tests in both modes, nine named flows, 32 carrier-boundary cases, and exact
  reusable/one-shot equality. The measured eager import path through
  `pyfem/v3/__init__.py` is real but explicitly deferred by the frozen P1-C card to
  the later root-package/CLI compatibility decision.
- 2026-07-19: I1 reconciled the unanimous R1-A verdicts with the existing I0
  integration proof and froze the bounded Phase 1 public contract at `7ee65c3`.
  The exact P0-D, P1-A, P1-B, and P1-C component slices advance to E3. This does
  not advance schedules, nonlinear/history behavior, mixed elements, adapters,
  broader outputs, root compatibility, or prototype retirement.
- 2026-07-19: before dispatching any Phase 2 writer, I0 reactivated three read-only
  planning tasks against the final I1 schemas: `/root/p2a_freeze_draft`,
  `/root/r2a_topology`, and `/root/r2b_ownership`. They must reconcile the literal
  disconnected 15-node Q8/Q4/T3 oracle, final topology/schema deltas, and exact
  ownership boundary into one frozen P2-A card. No Phase 2 source writer is active.
- 2026-07-19: the three planning reports and a fresh Sol/max reconciliation froze
  the P2-A constants: new-family v1 neutral schemas, T3 key
  `gauss-tria3-order1`, transient Q4 corner audits from descriptor metadata, one
  persisted full raw block-offset carrier, native-width ownership, and exact
  15-node/30-full/21-reduced, 356-full/203-reduced, 14-point arithmetic. A proposed
  compiler-only cut with a temporary Q8 analysis guard was not adopted: three
  independent call-graph adjudications showed the final consumers are already
  block-generic, the guard would violate the recorded public mixed-solve exit, and
  it would create later removal work. The frozen 13-path card therefore requires
  reusable and one-shot mixed public proof with no solver topology branch.
- 2026-07-19: I0 created isolated branch/worktree
  `agnet/p2a-mixed-blocks` at `/private/tmp/pyfem-p2a-7ee65c3` from exact product
  parent `7ee65c3` and dispatched fresh Sol/max sole writer
  `/root/p2a_mixed_blocks_writer`. It owns exactly the frozen 13 paths, must return
  one focused commit and every normal/640-digit, reference, public-flow, Ruff, v3,
  repository, static, ancestry, and clean gate. No parallel Phase 2 writer is active.
- 2026-07-19: the P2-A writer passed exact clean preflight and began the frozen
  compiler/carrier work. After six authorized paths were modified, its next local
  `apply_patch` to `compile/model.py` was rejected because an ordinary heartbeat
  reply was erroneously classified as unrelated content. The rejected hunk did not
  apply. I0 paused the task without a workaround; no test, commit, unowned path,
  revert, copy, or restart occurred. The preserved worktree remains at parent
  `7ee65c3` with changes only in `assembly/contracts.py`, `compile/__init__.py`,
  `compile/contracts.py`, `compile/model.py`, `model/compiled.py`, and
  `model/state.py`. Explicit post-disclosure user approval is required before the
  same bounded writer may resume.
- 2026-07-19: while the sole writer was active, two other tasks remained read-only.
  `/root/r2c_acceptance_matrix` completed the independent post-integration card,
  including exact Fraction-derived operators and a non-duplicative 13-node
  shared-Q4/T3 oracle with 356/340 full and 219/210 reduced raw/canonical counts.
  `/root/p2bc_leaf_drafts` completed disjoint two-path material-slot and active-field
  drafts. Both leaves remain unfrozen; R2-C must first confirm one immutable common
  partition seed, otherwise a serial prerequisite owns that seam.
- 2026-08-16: repository preflight found main `v3` clean at `48fa0d1`, while
  `/private/tmp/pyfem-p2a-7ee65c3` had disappeared and Git marked its worktree
  record prunable. Branch `agnet/p2a-mixed-blocks` still equals `7ee65c3` with zero
  child commits. The six uncommitted edits recorded in July are therefore gone;
  no product commit or test evidence was lost.
- 2026-08-16: D2-A re-evaluated the whole extension seam against the 154-row v1
  inventory, the integrated Phase 1 code, and `absim_fvm`'s normalize-once,
  concrete-array ownership model. The path from `c50ca70` to `7ee65c3` contains
  14,572 production insertions for one stateless Q8 flow and repeats
  compiler-owned meaning validation across program, assembly, and results.
  [generic_core.md](generic_core.md) now supersedes the legacy-shaped P2-A through
  P2-G plan with entity blocks, discrete spaces, bound operator blocks, typed
  channels, coordinate maps, and observations. The proven Phase 1 path is frozen
  as an oracle rather than extended.
- 2026-08-16: the next portfolio is R2-E legacy-to-instance mapping, P2-H isolated
  three-operator executable prototype, and R2-F trust/deletion-cut research. I0
  must adjudicate all three before any production migration writer is frozen.
- 2026-08-16: D2-A was committed as `f32e7e5`; R2-E
  `01a0062b-5deb-79b3-8a73-e768a9ca5fe4`, P2-H
  `01a0062b-6e84-7e22-bf8a-4fbb8f42db09`, and R2-F
  `01a0062e-c999-7740-89c3-4815948cb99a` were dispatched from that exact clean
  base. Luna/max G1 `01a0061e-7c27-7d93-b994-878bf8c16a01` then completed: 30
  titles were normalized, 29 historical threads were archived, and G1 itself was
  renamed `DONE` and archived. I0's next action is to adjudicate the three
  engineering results together; no result may independently authorize production
  integration.
- 2026-08-16: user required callback-only coordination for the live portfolio and
  a mandatory S2-A simplification/refactor pass after all three results return.
  I0 will not actively poll or inspect their worktrees. S2-A adds no capability;
  it squeezes the accepted outcome, updates the authoritative docs and evidence,
  proves a clean state, and is the terminal pause point for this long pass.
- 2026-08-16: R2-E/P2-H/R2-F all completed. I0 accepted the six-concept numerical
  IR with solve-free observation, explicit auxiliary-execution modality, and
  versioned coordinate-map artifact amendments; accepted P2-H as Proof-A evidence;
  and accepted R2-F's direct vertical-cut strategy without dispatching it. The
  missing literal callback clause in two original prompts and unacknowledged
  delivery in the third exposed a coordination defect; AGENTS.md now makes the
  callback target and literal prompt stanza a dispatch precondition. S2-A is the
  sole remaining implementation/refinement packet before the requested pause.
- 2026-08-16: P2-H source `b1789d1` was integrated as `f140b9e`. Sol/high S2-A
  simplified the proof in `099b51f`, reducing nonblank production/test lines from
  812/453 to 649/360 while retaining all seven focused proofs. Sol/max final review
  returned GO with P0/P1/P2 `0/0/0`, independently re-derived the exact Q4/T3 and
  split/vectorized assembly oracles, and passed 7 focused tests in both digit modes,
  485 v3 tests, 674 repository tests, both Ruff configurations, format, path,
  ancestry, semantic-branch, exact-head, and clean-tree gates. The migration is
  paused at this polished boundary; the accepted 33-path vertical cut remains
  undispatched.
- 2026-08-16: the user explicitly resumed the library migration and authorized
  either structural replacement or broader feature work. I0 opened one bounded
  read-only decision batch: a Luna capability-parity audit, a Sol/max refresh of
  the direct production cut against simplified proof `099b51f`, and a Sol/high
  dependency analysis of the fastest principled route toward v1 behavior. No
  production writer is authorized until those three reports are adjudicated.
- 2026-08-16: R3-A measured strict public parity at 4/154 capabilities, bounded
  executable coverage at 57/154, and design-backed coverage at 139/154. I0 accepted
  D3-A's four-commit direct replacement over R3-A's immediate mixed-width feature
  suggestion because feature breadth must not extend the predecessor authority.
  R3-B's state-first post-cut sequence is accepted. G1 is frozen as the sole next
  writer: six paths, direct unexported Q8 system compilation, exact parent
  `04baa4a`, then independent numerical and architecture reviews before G2.
- 2026-08-16: I0 created exclusive branch `agnet/g1-generic-system` and worktree
  `/private/tmp/pyfem-g1-04baa4a` at exact `04baa4a`, then dispatched sole
  Sol/high writer `/root/g1_q8_system_compiler`. Its prompt owns only the six G1
  paths, repeats every stop condition and named gate, and requires an explicit
  terminal callback. No other source or feature writer is active.
- 2026-08-16: G1 initial source `f342b40` returned within all six paths and line
  caps with 8 new, 156 retained, 493 v3, and 682 repository tests green in the
  required modes. Independent numerical review returned NO-GO `0/1/0`: callable
  outputs could contradict captured Q8 registry identity. Independent architecture
  review returned NO-GO `0/4/1`: the evaluator lacked typed program/request input;
  extra fields produced ghost/mislocated spaces; trusted carriers were ordinarily
  fabricable; source/fingerprint lookup was incomplete and non-exact; and normalized
  geometry was published as physical. I0 accepted all six roots and returned one
  same-path direct-child repair to the original writer. G2 remains closed.
