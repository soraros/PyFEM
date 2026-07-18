# Proofline — PyFEM v3 migration workflow

- Status: **active migration method**
- Method name: **Proofline**
- Prepared: 2026-07-17
- Investigation base: `ad95149e2e34b8eff55c0896c1dea53ac1cbc71d`
- Proposal source: `50cc6632490f5d185dd30ff6b85dd7cb10e432b6`
- Integrated proposal: `9b26574f52c39be756e2cdeb275dcfe7691e5bc4`
- Target branch: local `v3`; no merge or push is implied

This document is the full PyFEM specialization of
[Proofline](proofline.md), the repository's ledger-driven semi-automatic
coordination method. It defines the operating system for gradually replacing
legacy PyFEM with the architecture in [design.md](design.md). It does not authorize
a feature port or change any design invariant. Live state is owned only by
[migration-execution.md](migration-execution.md); the Proofline field guide is not
a second plan or ledger.

Navigation: [authority](#authority-and-adoption-status),
[state machine](#migration-state-machine),
[coverage ledger](#legacy-to-v3-coverage-ledger),
[initial waves](#dependency-aware-portfolio-and-initial-waves),
[contracts](#dispatch-and-integration-contracts),
[proof](#numerical-and-physical-proof-strategy),
[completion](#evidence-checkable-definition-of-migration-complete),
[adoption](#adoption-and-rollout-state), and
[coordinator loop](#coordinator-loop).

## Authority and adoption status

The authority order is unchanged:

1. the user's overall v3 goal and [design.md](design.md), including its amendment
   discipline;
2. edge-case tests and independent numerical or physical evidence;
3. the new v3 implementation;
4. legacy PyFEM as requirements, behavior evidence, and numerical reference;
5. the current v3 prototype as reusable evidence; and
6. historical plans, prototype APIs, old examples, and old benchmark policy.

### Finite-element task vocabulary

Every packet, title, callback, ledger entry, and coordinator handoff describes this
work as local finite-element migration and correctness work. Preferred terms are
**finite-element correctness review**, **local edge-case matrix**,
**correctness matrix**, **failure case**, and **independent reviewer**. Historical
wording is evidence about past execution, not reusable prompt text. Do not import
labels, metaphors, skills, or review frames from unrelated domains. This vocabulary
rule changes no invariant, evidence requirement, or acceptance gate.

The current foundation at adoption consists of:

- the prototype assembly module quarantined as
  `pyfem/v3/_prototype_assembly.py`;
- normalized authored model-specification primitives under `pyfem/v3/spec/`;
- live identity, state-generation, canonical-manifest, frozen-registry, and
  owned-array primitives under `pyfem/v3/model/`;
- six accepted `R0-A`/`R0-B` blockers repaired in the spec and identity/storage
  owners, with 41 focused, 171 v3, and 360 repository tests passing at the combined
  implementation head; and
- fresh, read-only Sol/max foundation review as a hard gate before any compiler
  packet; exact active, failed, replacement, and terminal task IDs live only in the
  execution ledger.

The prototype still exposes `load_problem`, `ProblemDefinition`, `LoadedProblem`,
`solve_linear`, `solve_nonlinear`, and `solve_riks`. It still contains the exact
failure seams listed in [design section 3.2](design.md#32-what-the-current-v3-does-not-prove).
Nothing in this workflow upgrades those paths from evidence to architecture.

The workflow keeps the Intel Mac, Python 3.13+, SciPy/NumPy/Numba development
baseline recorded in [AGENTS.md](AGENTS.md). The foreseeable development line is
`v3`. Modernization or feature work living on another unmerged branch is not
silently imported, rebased, or treated as a prerequisite.

### Adoption record

The D0-A source commit created only this document. The integration owner then
reviewed and adopted it by routing migration work here, preserving one live ledger,
and leaving [design.md](design.md) unchanged. Adoption itself does not:

- create or rename threads, worktrees, or a Goal;
- create a watchdog or recurring automation;
- classify an unresolved legacy capability by fiat;
- start the compiler writer;
- merge, push, or change production or test code; or
- expand compatibility, dependency, GUI, or retirement authority.

### Actions that remain separately bounded

Thread creation, scheduled automation, public compatibility decisions, legacy
retirement, design amendments, new dependencies, optional GUI policy, merge, and
push remain separately bounded actions. The user's standing authorization covers
the current delegated migration work and local focused commits; it does not turn
these higher-consequence decisions into automatic transitions.

## Operating thesis

Migration is not a queue of legacy files. It is a queue of user-observable or
physically meaningful capabilities, each assigned to one target semantic owner and
proved on a complete public vertical slice.

The durable rule is:

```text
discover behavior -> classify intent -> select a dependency-ready slice
-> freeze its Horizon contract -> implement in bounded ownership
-> receive a terminal callback -> review and integrate serially
-> obtain independent robustness review -> run combined proof
-> advance coverage or stop with an exact blocker
```

Only a verified transition changes migration state. A green worker branch, an old
example that happens to run, a final displacement parity number, or a reviewer's
approval sentence is not a transition by itself.

## Competing orchestration designs

### Design A: one long monolithic migration thread

One coordinator researches, implements, reviews, and integrates every phase in one
ever-growing thread.

Advantages:

- no cross-thread merge protocol;
- one mental model in the short term; and
- naturally serial changes to shared core owners.

Failure modes:

- context compaction becomes an accidental decision record;
- independent disconfirmation is weak;
- large unreviewed diffs accumulate before public-flow proof;
- a stale assumption can contaminate several downstream phases; and
- recovery requires replaying conversation rather than reading durable state.

### Design B: a permanent specialist swarm by legacy subsystem

Create long-lived element, material, solver, I/O, GUI, and test workers and let each
port its legacy namespace.

Advantages:

- apparent feature throughput; and
- easy initial division by directories.

Failure modes:

- it reproduces the legacy object and file boundaries;
- several workers invent competing state, registry, contribution, and result
  owners;
- shared compiler/analysis paths make ownership overlap inevitable;
- branches age against incompatible bases;
- integration debt is delayed rather than removed; and
- watchdogs and polls become a substitute for proof.

### Design C: Proofline's bounded packet conveyor

Keep one integration owner and one live ledger. Select a small dependency-aware
portfolio, then dispatch only packets with a frozen Horizon contract, disjoint path
ownership, exact base, exact evidence, and a terminal callback. Integrate one
commit at a time. Run read-only reviewers against combined semantic boundaries, not
against isolated code style.

This is less failure-prone because it combines the serial safety of Design A with
bounded independent work, while refusing the file-for-file partitioning and
long-lived branch drift of Design B. It also has a defined no-work result: if no
candidate is dependency-ready and testable, the coordinator records the missing
evidence and stops rather than manufacturing activity.

## Source of truth and persistence

The normal migration session should need exactly four durable inputs:

| Kind | Sole owner | What belongs there |
|---|---|---|
| Target design | `design.md` | Architecture, invariants, acceptance suite, phase order, amendments |
| Migration method | `migration_workflow.md` | This state machine, roles, contracts, automation, cost, and stop policy |
| Live execution ledger | `migration-execution.md` | Current head/base, capability rows, packets, decisions, callbacks, proof, blockers, next safe action |
| Dated phase evidence | one focused note only when needed | Raw numerical/performance measurements or adjudication facts not already represented by tests |

[refactor_playbook.md](refactor_playbook.md) is the source from which the Horizon
Gate and proof discipline were adapted. It remains the generic
method for non-migration structural refactors; it is not a second live migration
roadmap. [feature-parity.md](feature-parity.md) became historical after R0-E
absorbed its useful breadth and the complete repository into
[the dated E0 evidence note](evidence/2026-07-18-r0e-capability-inventory.md).
`WORKFLOW.md`, `roadmap.md`, `hardening.md`, and the Cursor-era artifacts remain
history.

The live ledger is updated only at a state transition, a material decision, a
changed base, a new blocker, or a completed proof gate. Routine shell commands and
unchanged status snapshots do not create ledger churn.

### Live execution ledger shape

The live `migration-execution.md` has these sections, in this order:

1. current integrated commit, dispatch base, branch, platform, and active
   milestone;
2. exact next safe action and blocked condition;
3. unresolved and decided semantic questions;
4. the [capability coverage ledger](#legacy-to-v3-coverage-ledger);
5. the packet ledger, including thread IDs, titles, bases, ownership, callbacks,
   source commits, integrated commits, repair count, and state;
6. the current phase proof matrix;
7. active watchdogs, if any, with owner, cadence, deduplication key, and shutdown
   condition; and
8. a short append-only transition log whose older detail is moved to dated
   evidence rather than duplicated.

Every commit hash is resolved to 40 hexadecimal characters before dispatch or
integration. `HEAD`, `v3`, “latest,” and a thread title are never accepted as an
exact base in a packet contract.

## Legacy to v3 coverage ledger

### Unit of coverage

A coverage row is one semantic capability with one observable contract, not one
legacy class or file. Examples include “fixed affine displacement constraints,”
“Q8 plane-stress linear static response,” “integration-point plastic history,”
“consistent mass for second-order mechanics,” “VTK projection with localization
metadata,” and “periodic RVE homogenization.”

Several legacy classes may be evidence for one row. One legacy class may produce
several rows. A row is split whenever it has a different semantic owner, failure
case, dependency, disposition, or proof surface.

The initial inventory must scan:

- root public entry points (`pyfem.run`, `pyfem` CLI, `PyFEMAPI`);
- the `.pro`/`.dat` loader, NodeSet, ElementSet, DofSpace, constraints, models,
  assembly actions, solvers, elements, materials, and writers;
- all legacy tests and all 98 checked-in `.pro` examples;
- the 12 parity skims and their v3 prototype tests; and
- public docs/notebooks that expose a behavior not represented by those sources.

Code existence is `E0` evidence. An example file that is not run is also `E0`; it
does not prove the capability works.

### Disposition

Every row must have exactly one disposition before implementation:

- **preserve** — the user-observable physical or numerical contract remains;
- **change** — the capability remains, but an unsafe, ambiguous, duplicate, or
  architecturally invalid behavior is deliberately replaced; or
- **retire** — the capability has no supported v3 successor.

`Change` names both the preserved intent and the intentional incompatibility.
`Retire` requires consumer evidence, a replacement or explicit loss statement,
and delegator/user approval when public behavior is affected. “Deferred,” “old,”
“not in Phase 1,” and “hard to port” are statuses, not dispositions.

### Required schema

| Field | Contract |
|---|---|
| `capability_id` | Stable semantic ID; never reused after retirement or split |
| `capability` | One observable physical, numerical, workflow, or output behavior |
| `legacy_surface` | Public call/CLI/input plus code, example, test, and documentation evidence pointers |
| `disposition` | Exactly `preserve`, `change`, or `retire`; decision ID if not self-evident |
| `intent_and_difference` | Preserved meaning and any deliberate incompatibility |
| `v3_semantic_owner` | One target owner from design: spec, compiler, block, program, state, assembly, analysis, result, adapter, or ecosystem |
| `dependencies` | Capability IDs and design/packet gates required first |
| `failure_cases` | Counterexamples that can disprove the proposed owner or behavior |
| `reference_evidence` | Legacy harness, analytical result, independent formulation, benchmark, or other oracle; never only the new implementation |
| `target_slice` | One vertical slice or explicit later decision packet |
| `compatibility` | Public names, files, outputs, checkpoints, or entry points affected |
| `status` | One value from the lifecycle below |
| `evidence_grade` | Highest completed grade `E0` through `E4` |
| `proof_commit` | Exact integrated commit on which the claimed evidence passed |
| `open_question` | Exact unresolved semantic choice or `none` |

Capability status is separate from packet state:

```text
inventoried -> classified -> contracted -> implementing -> provisional
-> proved | retired-proved
```

`blocked` may be entered from any nonterminal state with the exact blocker and
minimum unlock. `superseded` is allowed only when a row is split or merged and the
replacement IDs are recorded. Only `proved` and `retired-proved` are terminal for
migration completion.

### Evidence grades

| Grade | Meaning |
|---|---|
| `E0` | Source, example, or claim exists; behavior not extracted |
| `E1` | Behavior and conventions extracted into a reproducible legacy, analytical, or independent reference |
| `E2` | Target component plus edge cases pass, but no complete public v3 flow |
| `E3` | Public authored-to-verified-result vertical slice passes with provenance and failure behavior |
| `E4` | Compatibility/retirement, representative performance, docs/examples, and relevant full-suite gates pass |

Parity without ownership, rollback, balance, or provenance cannot exceed `E2`.
Rows with no performance relevance may reach `E4` with a recorded “not material”
decision rather than a fabricated benchmark.

### Example seed rows

These are seed hypotheses for the future ledger, not a substitute for the complete
inventory or a pre-approved compatibility decision.

| ID | Capability | Proposed disposition | Target owner/slice | Current evidence and required failure case |
|---|---|---|---|---|
| `LIN-Q8-PS` | Q8 plane-stress linear static solve | preserve | compiled block through verified `LinearStatic`, Phase 1 | Prototype/book parity exists; add orientation, aliasing, additive load, affine MPC, reactions, and independent verification |
| `CON-AFFINE` | fixed Dirichlet and linear MPC constraints | change | `CompiledProgram` affine plan, Phase 1 | Preserve affine meaning; retire chained mutable patching; prove cycles, conflicts, nonzero offsets, zero-free-DOF, and reaction basis |
| `LOAD-NODAL` | repeated nodal forces | change | program point-load plan, Phase 1 | Preserve additive force; explicitly reject prototype last-write-wins behavior |
| `MESH-HETERO` | mixed topologies, regions, and local layouts | change | compiler blocks/entity map, Phase 2 | Legacy breadth and spec foundation exist; prove Q4/T3, Tet4/Quad4 node-count collision, source mapping, and permutation |
| `STATE-HISTORY` | integration-point material and formulation history | change | committed/trial state transaction, Phase 3 | Legacy mutable history is requirements evidence; prove idempotence, rollback, exact commit, stable slot identity, and restart |
| `NL-NEWTON` | nonlinear static stepping | change | typed nonlinear analysis, Phase 3 | Preserve equilibrium path; replace caller mutation and fabricated increment; prove rejected iterations, line search, cutback, and tangent consistency |
| `FORM-FINITE` | finite-strain continuum and geometric tangent | preserve | domain-block kernel, Phase 4 | Prototype/legacy parity exists; add objectivity, current inversion, tangent attribution, and state safety |
| `PATH-ARC` | Riks continuation | change | `ArcLength` plus program derivatives, Phase 4 | Preserve limit-point traversal; replace solver-local constraint/load semantics; prove branch/load parameter and rollback |
| `PATH-DISS` | dissipated-energy continuation | change | continuation functional, post-Phase 4 portfolio | Legacy solver exists; extract local/classic semantics, dissipation state, switching rule, and failure behavior before selection |
| `EVOL-MECH` | explicit second-order mechanics and mass | change | Phase 5A evolution request | Legacy explicit/mass paths exist; prove consistent/lumped mass, energy/balance, stable step policy, constraints, and restart |
| `SPECTRAL` | modal and preloaded buckling analysis | change | separate Phase 5B requests | Preserve distinct eigen problems; prove base-state identity, homogeneous perturbation constraints, normalization, and residuals |
| `MULTI-THERMAL` | thermal, thermomechanical, phase-field, and staggered evolution | change | Phase 5C blocks/analysis | Legacy classes/examples exist; prove exact field ownership, mixed evolution order, coupled convergence, and atomic irreversible commit |
| `INTERACT-CONTACT` | contact interaction | change | Phase 5D `InteractionBlock` | Legacy model is evidence only; require active-set state, declared sparsity mode, complementarity/work, and rollback |
| `STRUCT-BREADTH` | beams, plates, shells, layered sections, condensation | preserve/change per subrow | post-Phase 4 block portfolio | Split by formulation and section semantics; shell objectivity test is useful evidence, not one-row parity |
| `MAT-BREADTH` | plasticity, damage, cohesive, viscous, crystal, failure criteria | preserve/change per subrow | Phase 3/5 material portfolio | Split by state schema and constitutive contract; numerical-tangent and output behavior require explicit decisions |
| `ECO-IO` | legacy input, CLI, VTK/HDF5/graph/ROM output | change | Phase 6 adapters/results | Input becomes spec adapter; outputs require localization/provenance; root entry-point compatibility is undecided |
| `MODEL-RVE-FE2` | RVE, periodic homogenization, and MicroModel FE2 | undecided until classification | explicit design packet | Cannot be silently retired by old “out of scope” text; identify users, nesting/state/restart semantics, and cost first |
| `GUI-OPTIONAL` | `pyfem-gui` | undecided until compatibility decision | Phase 6 ecosystem | Intel baseline intentionally lacks PySide6; preserve, replace, or retire only through explicit optional-dependency decision |

The two `undecided` seed rows are inventory warnings, not valid live-ledger
dispositions. They must become preserve/change/retire before their rows are
`classified`.

## Dependency aware portfolio and initial waves

### Selection rule

The coordinator compares a small portfolio before every new packet. A candidate is
eligible only if:

1. all hard dependencies are integrated and proved at their stated commit;
2. its owner-plus-consumer cone fits one reviewable vertical slice;
3. it has an independent reference and at least one edge case;
4. its path ownership and merge order are explicit;
5. its public/compatibility consequence is bounded; and
6. its blocked condition can be stated without “keep trying.”

Among eligible candidates, use this priority order:

```text
repair an invariant contradicted by evidence
-> unblock architecture ratification in Phases 1-3
-> replace a real public flow end to end
-> causally delete a prototype/legacy owner
-> add distinct semantic breadth
-> optimize a measured public bottleneck
```

Tie-break with the smaller semantic cone and stronger evidence. Do not select a
nearby class merely because its filename resembles the last packet.

### Wave 0: finish and operationalize foundations

No compiler writer is dispatched while required foundation independent review or
repair is active.

After the required callbacks:

1. the `I0` integrator adjudicates every finding;
2. accepted foundation defects are repaired inside the original owner;
3. combined focused/v3/full/Ruff proof is rerun at one exact commit;
4. the legacy coverage ledger is seeded from the complete inventory; and
5. only then is the compiler-integration writer selected.

The following migration IDs were reserved at adoption. Failed or replacement tasks
receive their own additional IDs in the live ledger; these reservations are not
reused for them:

| ID and title | Outcome | Dependency | Parallel policy |
|---|---|---|---|
| `R0-E · legacy breadth — semantic ledger seeded` | Complete E0 inventory and proposed preserve/change/retire classifications with evidence gaps | Foundation re-review terminal; adopted ledger schema | Read-only research may overlap P0-D; only integrator writes the ledger |
| `P0-D · model compiler — Q8 block frozen` | Normalized Q8 region compiles into immutable model/block recipe, entity/source maps, capabilities, and empty physical-state layout | Foundation re-review adjudicated and combined proof green | Serial core writer |
| `R0-F · model compiler — block invariants verified` | Check topology meaning, registry stability, membership, caller isolation, geometry validation, and identity compatibility | P0-D integrated | Read-only Sol/max reviewer |

`P0-D` must not introduce `ProgramSpec`, a public solver, a second whole-problem
carrier, or a compatibility shim. It is the exact next dependency already implied
by [migration-execution.md](migration-execution.md#merge-and-continuation-order).

### Wave 1: one honest Q8 linear slice

The early ownership boundaries are intentionally serial. Parallel writers here
would freeze mutually invented APIs before the architecture is executable.

| ID and title | Outcome | Depends on | Causal retirement/proof |
|---|---|---|---|
| `P1-A · program compiler — affine plan canonical` | `ProgramSpec`, additive nodal loads, fixed affine constraints, structured coordinates, and model-compatible `CompiledProgram` | P0-D and R0-F adjudication | Replaces prototype constraint/load packing on the new flow |
| `P1-B · assembly plan — contributions composed` | Reference COO oracle plus composed model/program/reduction/request plan | P1-A | Proves loads are added and program tangents cannot be absent from topology |
| `P1-C · linear slice — verified public flow` | Minimal physical/evolution state transaction, typed `LinearStatic`, reusable and one-shot solve, result ledger, `verify_record`, and fresh `verify` | P1-B | New Q8 public flow contains no `ProblemDefinition` or `LoadedProblem` |
| `R1-A · linear slice — physics and state verified` | Check geometry, constraints, loads, identity, reactions, balance, snapshot isolation, and verification integrity | P1-C integrated | Independent Sol/max report |
| `I1 · linear slice — combined proof` | Adjudicate findings and run all Phase 1 exit gates | R1-A terminal and repairs integrated | Capability rows reach E3; no breadth claim |

The Phase 1 exit includes the existing Q8 book oracle, but the success bar is the
full edge-case surface in [design section 11](design.md#11-testable-acceptance-suite).

### Wave 2: prove heterogeneous block generality

Once Phase 1 contracts are frozen, compare these candidates as one dependency
portfolio:

1. Q4 and T3 descriptors plus a genuinely mixed-topology solve;
2. multiple parameter/material regions and layered material slots;
3. exact active field/DOF layouts for mechanical, thermal-only, and coupled
   authored regions;
4. model-owned boundary physics and program-owned boundary loads; and
5. fixed sparse slot assembly compared with the reference COO plan.

The mixed-topology compiler/block packet lands first because all others consume its
partition and entity-map contracts. Material-slot and field-map research can be
parallel and read-only. Writers may run in parallel only after the integrator can
name disjoint modules and one frozen common block API. Sparse-slot work follows the
complete recipe union and may not define contribution semantics.

Phase 2 advances only when adding Q4/T3 and incompatible regions requires no solver
branch, no universal padded carrier, and no new optional field on every model.

### Wave 3: make accepted and trial state real

State ownership remains serial through the first path-dependent public solve:

1. material/IP and formulation/element state schemas with stable semantic slots;
2. pure candidate evaluation and `TrialAnalysisState`;
3. atomic accept/discard across physical, evolution, and program history;
4. Newton residual/tangent, line search, rejected attempt, and cutback;
5. checkpoint/fingerprint/schema rebind and uninterrupted-run comparison; and
6. one fixed-connectivity cohesive/interface block as ordinary stateful domain
   work.

Use an elastic block beside an incompatible stateful material and one
formulation-owned state example. Do not begin broad finite strain or material
ports until failed-step byte-for-byte rollback and exact accepted trial commit are
proved.

### Wave 4: nonlinear formulation and continuation breadth

After Phase 3 ratifies state semantics, finite-strain continuum, structural
truss/spring, and arc length may have separate kernel writers behind one frozen
block/evaluation contract. The integration order is:

1. finite-strain residual/material/geometric tangent and provenance;
2. structural contribution blocks;
3. program-parameter derivatives and continuation functionals;
4. arc-length public flow and limit-point proof; and
5. deletion of every remaining `_prototype_assembly.py` caller and then the bridge
   itself.

The existing cantilever and shallow-truss skims are numerical references. Balance,
objectivity, branch history, nonzero affine offsets, accepted load parameter, and
rollback are the acceptance surface.

### Wave 5: broaden by distinct semantics

Use separate portfolios, proof matrices, and decisions for:

- **5A evolution:** first-order thermal capacity/rate, second-order mass/damping,
  implicit/explicit dynamics, mixed field order, conservation, and restart;
- **5B spectral:** modal and preloaded buckling with distinct request/result and
  base-state contracts;
- **5C multiphysics:** thermal/phase-field/coupled blocks and monolithic/staggered
  atomic irreversible commit; and
- **5D dynamic interaction:** contact with pair/active-set state and explicit fixed,
  over-allocated, dynamic, or matrix-free structure.

Beam, plate, shell, layered-section, advanced material, dissipated-energy, RVE,
FE2, and ROM rows enter the portfolio according to their semantic dependencies,
not their chapter or directory. Research can be parallel. Any writer touching a
shared analysis/state owner remains serial.

### Wave 6: compatibility and ecosystem

Only after the core contracts survive Waves 1-5:

- expand `.pro`/`.dat`, TOML, Gmsh/meshio, and Python adapters into specs;
- decide root package and `pyfem` CLI compatibility;
- implement result-aware VTK/HDF5/graph/snapshot projections;
- migrate maintained examples and documentation to public v3 flows;
- decide ROM, RVE/FE2, and optional GUI outcomes; and
- quarantine, deprecate, or delete legacy runtime paths according to approved
  coverage dispositions.

This wave is not a cleanup appendix. Migration cannot be declared complete while
public compatibility, examples, or ecosystem capability rows remain merely
deferred.

## Migration state machine

Packet state and capability status are separate. One packet may advance several
capability rows, and one capability may require several packets.

| State | Entry evidence | Only allowed next states | Owner | Automatic transition allowed |
|---|---|---|---|---|
| `DISCOVERED` | Capability/evidence gap recorded | `CLASSIFIED`, `BLOCKED` | Coordinator/researcher | Inventory mechanics only |
| `CLASSIFIED` | Preserve/change/retire, owner, dependencies, failure cases | `READY`, `BLOCKED`, `SUPERSEDED` | Sol/max coordinator | No semantic classification automation |
| `READY` | Dependencies proved and candidate card complete | `SELECTED`, `BLOCKED` | Coordinator | Eligibility calculation may be mechanical |
| `SELECTED` | Portfolio comparison and reason recorded | `HORIZON_FROZEN`, `BLOCKED` | Sol/max coordinator | No |
| `HORIZON_FROZEN` | Outcome, ownership, scope, competing design, base, proof, merge order, blocked condition frozen | `DISPATCHED`, `BLOCKED` | Integrator | No |
| `DISPATCHED` | Thread/worktree ID, title, exact base, prompt, owned paths recorded | `IMPLEMENTING`, `CALLBACK_RECEIVED`, `BLOCKED` | Coordinator/worker | Native dispatch status only |
| `IMPLEMENTING` | Worker confirmed base, instructions, and first bounded action | `CALLBACK_RECEIVED`, `BLOCKED` | Worker | First native status may record it; no polling loop |
| `CALLBACK_RECEIVED` | Exactly one terminal COMPLETE/BLOCKED message | `INTEGRATION_REVIEW`, `REPAIR_REQUESTED`, `BLOCKED` | Integrator | Callback receipt and deterministic prechecks |
| `INTEGRATION_REVIEW` | Commit, diff, ownership, contract, evidence, base, and warnings reviewed | `REPAIR_REQUESTED`, `CHERRY_PICKED`, `REJECTED`, `BLOCKED` | Sol/max integrator | No semantic approval automation |
| `REPAIR_REQUESTED` | Accepted findings and round `1/2` or `2/2` recorded | `DISPATCHED`, `CALLBACK_RECEIVED`, `BLOCKED` | Integrator/original writer | Mechanical repair may be redispatched |
| `CHERRY_PICKED` | Source and integrated hashes plus clean focused gate recorded | `INDEPENDENT_REVIEW`, `REPAIR_REQUESTED`, `BLOCKED` | Integrator | Cherry-pick is serial and human/model reviewed |
| `INDEPENDENT_REVIEW` | Independent reviewer findings returned and adjudicated | `REPAIR_REQUESTED`, `COMBINED_PROOF`, `BLOCKED` | Sol/max reviewer/integrator | Callback receipt only |
| `COMBINED_PROOF` | Focused, public-flow, edge-case, relevant full-suite, static, and performance gates complete | `ADVANCED`, `REPAIR_REQUESTED`, `BLOCKED` | Sol/max integrator | Commands may run automatically; interpretation may not |
| `ADVANCED` | Capability statuses/evidence grades and next safe action updated | terminal | Integrator | Ledger bookkeeping after approved proof |

`REJECTED` means the commit is not integrated and the candidate returns to the
portfolio with evidence. `SUPERSEDED` applies only to capability rows, never as a
way to hide an incomplete packet. `BLOCKED` records safe attempts, exact blocker,
and minimum unlock.

A scope change after `HORIZON_FROZEN` creates a new ledger ID and contract. It does
not silently widen the existing worker.

## Roles and thread topology

### Coordinator and integrator

The integration thread owns:

- the portfolio and exact next safe action;
- the sole live execution ledger and shared migration documents;
- Horizon Gate adjudication;
- dispatch bases, path ownership, and merge order;
- commit review and serial cherry-pick;
- reviewer finding adjudication;
- combined proof and capability advancement; and
- shutdown of watchdogs and completed packet surfaces.

Architecture, numerical meaning, tolerance, compatibility, and retirement stay
with a Sol/max coordinator/integrator even when a cheaper worker wrote mechanical
code.

### Bounded writer

A writer receives one frozen outcome, exact committed base, disjoint owned paths,
forbidden paths, evidence commands, merge dependency, and callback target. It may
research within scope and repair its packet. It may not edit shared design/ledger
files, merge, push, absorb unrelated changes, create compatibility shims, or decide
an unresolved physical/API question.

### Read only researcher or reviewer

A researcher receives a question, evidence surface, validation focus, exact commit,
and named output. It has no file ownership and makes no commit. Useful validation
focuses are:

- strongest competing semantic owner;
- decisive numerical/physical failure case;
- missing state or provenance lifetime;
- unsupported compatibility claim;
- false independence in proof; and
- scope or authority expansion.

Reviewer reports are evidence, not votes. Findings are `P0` correctness/authority,
`P1` required proof/design, or `P2` advisory. Unresolved `P0` or `P1` findings block
advancement.

### Integration topology

Use the recorded title format exactly:

```text
<ledger-id> · <semantic owner> — <target state>
```

Use `P` for code/test packets, `R` for read-only research/review, `D` for bounded
design decisions, and `I` for phase integration. IDs are never reused. Titles stay
stable during normal execution; a task that ends without the required terminal
result is visibly prefixed `INCOMPLETE`, with its reason and replacement recorded
separately in the ledger.

Default topology for a risky slice:

```text
I<phase> integration owner
  <- terminal callback from P<phase>-A bounded writer
  <- terminal callback from optional disjoint P<phase>-B writer
  <- terminal callback from R<phase>-A independent reviewer
```

### Work that must remain serial

- changes to the same path or semantic owner;
- first definition of a compiler, state, constraint, contribution, result, or
  public API boundary;
- integration review and every cherry-pick;
- numerical/design/compatibility adjudication;
- live ledger and shared authoritative-document edits;
- repair of a failed combined gate; and
- deletion or quarantine of the superseded path.

Parallel work is limited to independent read-only research, disjoint kernels behind
a frozen descriptor/evaluation contract, disjoint fixture/test generation, or
mechanical ecosystem work with an already proved public contract.

## Dispatch and integration contracts

### Horizon contract

Every packet records this compact card before dispatch:

```text
ID/title:
Outcome and ownership invariant:
Coverage rows advanced:
Exact base and required parent packets:
Owned paths / forbidden paths:
Public flows and consumers:
Strongest competing design or explanation:
Baseline and independent reference:
Edge cases and failure behavior:
Expected migrations and causal deletions:
Compatibility/API consequence:
Performance relevance and envelope:
Acceptance commands and required evidence:
Merge order:
Blocked condition and minimum unlock:
Integration thread ID:
```

The integrator marks the card `HORIZON_FROZEN` and records its ledger anchor. A
writer may report a newly discovered issue but may not reinterpret the card.

### Writer dispatch prompt

```text
You are <ID>, the bounded writer for <title>.

Start exactly at <40-char base>. Before editing, read .agents/v3/AGENTS.md,
.agents/v3/design.md, .agents/v3/migration_workflow.md, and the packet card at
<ledger anchor>. Confirm `git rev-parse HEAD` and a clean `git status --short`.

Outcome/invariant: <...>
Coverage rows: <...>
Owned paths: <...>
Forbidden paths: <...>
Non-goals: <...>
Competing design: <...>
Edge cases/reference: <...>
Acceptance: <commands and public proof>
Merge dependency: <...>

Implement only this contract. Stop on a shared-owner decision, stale base, dirty
worktree, unresolved numerical meaning, or forbidden-path need. Make one focused
commit. Do not merge or push.

Immediately before your final response, send <integration-thread-id> exactly one
terminal callback:
<ID> COMPLETE|BLOCKED · <commit or none> · paths=<...> · proof=<...>
· warning/blocker=<...> · next=<review, repair, or dependency action>

Begin your final response with `RESULT: COMPLETE` or `RESULT: BLOCKED`, then return
the same data plus assumptions, unresolved findings, and clean worktree state.
```

### Terminal callback contract

The callback is single-shot per worker turn and begins with the exact ID and
terminal word. A repair turn sends a new callback with `repair=1/2` or `repair=2/2`.
It does not claim integration.

Valid example:

```text
P1-A COMPLETE · 0123456789abcdef0123456789abcdef01234567
· paths=pyfem/v3/program/**,test/v3/test_program.py
· proof=focused 18 pass; v3 Ruff pass; git diff --check pass
· warning/blocker=none · next=I1 review source commit
```

If the callback tool is unavailable, the worker writes `CALLBACK UNAVAILABLE` at
the start of its final response. The coordinator may then use one native status
snapshot; it does not start a polling loop.

### Deterministic callback precheck

Receipt may automatically run read-only checks:

1. callback ID and thread ID match the packet ledger;
2. commit exists and is a descendant of the exact base with the expected commit
   count;
3. changed paths are a subset of owned paths and exclude forbidden paths;
4. the worktree was reported clean after commit;
5. `git diff --check <base>..<commit>` is clean; and
6. required evidence and warnings are present.

A precheck can fail a contract mechanically. It cannot approve architecture,
physics, numerical semantics, API compatibility, or proof independence.

### Integration review contract

The integrator reviews, in this order:

1. exact base, commit ancestry, diff scope, and user-owned changes;
2. Horizon outcome and explicit non-goals;
3. target ownership and dependency direction;
4. edge cases, failure behavior, and independent reference;
5. public consumer migration and claimed causal deletion;
6. state, identity, provenance, and snapshot lifetimes;
7. numerical conventions, tolerances, balance, and performance claims; and
8. focused test/Ruff/static evidence.

Each finding is accepted, rejected with concrete evidence, or deferred outside the
packet's success bar. General approval without this reconciliation is insufficient.

### Repair contract

Before integration, repairs return to the original writer and exact worktree:

```text
Repair <ID>, round <1/2|2/2>, from source commit <hash>.
Accepted findings: <numbered exact failures>.
Required evidence: <commands/cases>.
Unchanged outcome, ownership, forbidden paths, and callback target: <ledger link>.
Do not widen scope. Amend the focused commit and return the replacement hash.
```

After a commit is integrated, do not rewrite it. A repair starts as a new packet
from the current integrated commit and lands as a new focused commit.

There are at most two repair rounds for one frozen packet. Repetition of the same
failure, a third failed round, or a repair that needs a new semantic decision
returns the work to portfolio/design state. Sunk cost never lowers the gate.

### Cherry pick contract

Only the integrator cherry-picks. It must:

1. start clean at the ledger's current integrated hash;
2. cherry-pick exactly one reviewed source commit, or the explicitly recorded
   ordered commits when an approved dependency requires more than one;
3. record source hash and resulting integrated hash separately;
4. run the packet's focused gate immediately;
5. stop on conflict or failure and avoid building another packet on top; and
6. never merge a worker branch, push, or resolve a semantic conflict as a textual
   conflict.

If a cherry-pick conflicts because the base is stale, redispatch or repair from the
current integrated base. Do not ask the worker to merge the integration branch.

### Reviewer prompt

```text
You are <R-ID>, a read-only Sol/max correctness reviewer for a local finite-element
Python library.
Review integrated commit <40-char hash> against design.md, migration_workflow.md,
the packet Horizon card, and coverage rows <IDs>. Make no edits or commits.

Validation focus: <one distinct local correctness lens>.
Check the strongest competing design, concrete numerical and input edge cases,
missing independent proof, duplicate semantic ownership, scope or authority drift,
and public-flow or numerical regressions. Use repository-local tests and temporary
local data only.

Return findings as P0/P1/P2 with file/line or command evidence, then GO or NO-GO.
Immediately before the final, send <integration-thread-id> one callback:
<R-ID> COMPLETE|BLOCKED · commit=none · verdict=<GO|NO-GO|blocked>
· findings=<counts/summary> · next=<smallest adjudication action>

Begin the final response with `RESULT: COMPLETE` or `RESULT: BLOCKED`, then record
the exact base, command evidence, warnings, worktree state, and smallest next action.
```

Use different focuses for multiple reviewers; duplicate generic reviews add cost but
not independence.

### Task prompt and terminal-result integrity

Task prompts use domain-specific correctness language only: finite-element model
meaning, numerical edge cases, type and shape validation, ownership, determinism,
exception safety, and independent verification. Avoid metaphorical language from
unrelated technical domains. Keep each prompt short by linking the frozen packet
card instead of restating the full migration history. Every task is limited to the
repository checkout, its configured test tools, and task-specific temporary local
data; it does not inspect external systems.

A task is terminal only when all of the following exist together:

1. its own final response begins with `RESULT: COMPLETE` or `RESULT: BLOCKED`;
2. the final response records the exact base, commit or `none`, required command
   results, warnings, worktree state, and smallest next action;
3. the required callback was received, or the final response places
   `CALLBACK UNAVAILABLE` immediately after the result line; and
4. the integration owner verifies the thread ID and evidence before changing the
   packet state.

An idle thread, a completed turn, commentary, partial command output, an approval
request, or recovered evidence is never task completion. A replacement task does
not retroactively complete the task it replaced. After one unsuccessful resume,
rename the task visibly as incomplete, record it separately, and either issue one
new bounded replacement or complete the remaining verification in the integration
thread. Do not create a chain of nominally active replacements.

### Phase completion contract

A phase integration packet can advance only when:

- every source commit is reviewed and recorded in merge order;
- all accepted reviewer findings are repaired;
- all rejected findings cite evidence and all deferrals are outside the exit bar;
- focused, public-flow, edge-case, v3, relevant full-suite, Ruff, format, and
  diff gates pass at one exact integrated commit;
- performance claims meet a preregistered representative envelope;
- affected coverage rows and evidence grades are updated;
- the superseded path is deleted/quarantined when its causal replacement is real;
- the next dependency or honest blocked condition is recorded; and
- all packet-specific watchdogs are disabled.

## Codex capability map

This map records the Codex desktop capabilities observed on 2026-07-17. Tool and
model names can change; re-verify them before dispatch rather than encoding them as
architecture.

| Capability | What is native | Operational limit and policy |
|---|---|---|
| Project worktrees | A project thread can be created in a new worktree from an existing branch/ref or current working tree | Creation is asynchronous; queued worktrees may return a client ID before a waitable thread ID. Always verify the resolved 40-character base and clean status inside the worker. Worktrees isolate checkouts, not repository objects or semantic conflicts. |
| Model/reasoning selection | Thread creation and follow-up can select a supported model and reasoning effort | Selection never expands authority. Preserve a thread's settings by omitting overrides. Record any intentional downgrade in the packet. |
| Thread creation | Native project or projectless user-visible threads | Use only when explicitly authorized and a named independent output justifies it. Record exact thread ID, host ID if relevant, title, base, and packet. Creation itself is not completion. |
| Thread titles | Native rename operation | Use stable `<ID> · <owner> — <state>` titles; never encode transient status or worker model in the title. |
| Native waits | `wait_threads` waits on up to eight targets and returns on terminal/attention state; cursors suppress delivered final text | Commentary does not wake the wait. Use a bounded wait while the coordinator is already active or `timeout=0` for one snapshot. A native wait transports status inside the current activation; it does not launch another model worker. |
| Direct callbacks | A worker can send one follow-up message directly to the integration thread | Default terminal signal. It uses the worker's already-running terminal turn and avoids another coordinator poll. It does not grant merge or shared-file authority. |
| Status reads | Native thread list/read operations | A read is a snapshot. Repeated model turns that call it are model-consuming polls and are not the default. |
| Recurring automation | Local recurring jobs and thread heartbeats can schedule later model activations | Not indefinite agency. Every run consumes model/tool capacity and can drift without durable state. It requires explicit authorization, a recorded owner/cadence, transition-only notifications, and shutdown. |
| Goal | Thread-scoped native completion contract when explicitly requested | Useful for an authorized multi-turn migration objective after the contract stabilizes. It is not a ledger, evidence store, permission expansion, or substitute for exact commits. |
| Git | Workers can commit in isolated worktrees; integrator can inspect objects and cherry-pick | A commit does not merge itself. Workers do not merge/push. Integrator remains clean, records source/integrated hashes, and preserves unrelated dirty state. |
| Execution ledger | Repository Markdown maintained by the coordinator | Not automatically synchronized with threads or git. Update only at verified transitions. It is the restart source when native thread history is unavailable. |
| Context compaction | Long conversations may be summarized automatically | Useful for conversational continuity, not an audit log. Exact decisions, commands, hashes, findings, and next action must be on disk before relying on compaction. |

### Native events versus model consuming polls

Preferred order:

1. worker terminal callback;
2. one native `wait_threads` call while the coordinator is already active;
3. one immediate native status snapshot after a missing callback; and
4. only for orphaned/long-lived work, a scheduled watchdog using the smallest
   capable model.

A loop that repeatedly wakes a coordinator model to call `read_thread`, reread
“active,” and narrate no change is polling. It spends context and cost without
advancing proof. No workflow step requires it.

## Automation policy

### Safe automatic progress

Within already authorized scope, automation may:

- ingest and deduplicate terminal callbacks;
- perform the deterministic callback precheck;
- update native thread status fields without interpreting code;
- run exact, preregistered test/Ruff/format/diff commands;
- collect test counts, timing samples, hashes, and changed paths;
- mark a mechanical gate pass/fail at an exact commit; and
- disable its own watchdog when all watched packets are terminal.

Automation may not move from integration review, reviewer adjudication, or combined
proof to advancement merely because commands are green.

### Required Sol max adjudication

Sol/max owns:

- preserve/change/retire classification;
- design amendments and semantic-owner selection;
- physics, signs, measures, units, state evolution, constraint basis, and
  conservation interpretation;
- tolerance or oracle conflicts;
- public API and compatibility consequences;
- commit integration and reviewer finding adjudication;
- representative performance tradeoffs; and
- the final migration-complete claim.

Ask the user when the decision changes intended physics, retires or breaks public
behavior, expands authority/cost/dependencies, changes the optional GUI baseline,
or remains genuinely ambiguous after the design and evidence are exhausted.

### Watchdog policy

Direct callbacks are mandatory by default. A watchdog is allowed only when a
packet is orphaned, callback tooling is unavailable, or work is expected to outlive
the active coordination session.

When authorized, record all of:

```text
watchdog_id, watched thread IDs, smallest capable model, low reasoning,
cadence >= 15 minutes, last native cursor, last terminal/attention state,
deduplication key, retry count, owner, and shutdown condition
```

As of adoption, a status-only watchdog can use the smallest available model
with reliable thread tools, such as `gpt-5.3-codex-spark` at low reasoning. It may
report status and terminal/attention transitions only. It may not review code,
send repair instructions, cherry-pick, alter the ledger's semantic decisions, or
claim completion.

Deduplicate notifications by:

```text
(thread_id, terminal_or_attention_state, commit_hash_or_blocker_digest)
```

Use at most three transient tool retries with backoff in one scheduled run. After
three consecutive inaccessible runs, notify once, record an attention transition,
and disable the watchdog. Unchanged active state produces no notification. Disable
immediately when all watched threads are terminal, the packet is abandoned, or the
integration phase closes.

## Model and cost policy

Use GPT-5.6 Sol with max reasoning for:

- migration coordination and portfolio selection;
- architecture and Horizon contracts;
- constraint, state, solver, material, formulation, and result semantics;
- numerical/physical reference design;
- integration review and finding adjudication;
- independent correctness reviewers; and
- phase and migration completion.

A cheaper model is eligible only after the Sol/max owner freezes the contract and
evidence surface, and only for genuinely mechanical work such as:

- exact import/path renames with behavior-preservation tests;
- generated inventory extraction without semantic classification;
- repetitive fixture conversion under a reviewed schema;
- formatting, link/anchor, stale-term, or manifest checks; and
- disjoint documentation synchronization that introduces no policy.

Possible bounded choices are a current Terra/Luna model or a mini/spark model at
low-to-medium reasoning, chosen by the integrator for that exact packet. The packet
must record the downgrade and cannot escalate its own scope. Any ambiguity about
physics, numerical behavior, ownership, compatibility, or proof returns to Sol/max.

Do not use cheaper writers for the first implementation of a semantic boundary,
kernel port, state schema, constraint compiler, nonlinear algorithm, capability
classification, or independent review. Savings from one cheap writer do not justify an
expensive repair/integration loop.

## Numerical and physical proof strategy

### Extract behavior before porting

For each preserve/change row, an `E1` extraction packet records:

1. the authored input and real legacy public path;
2. equations, tensor/Voigt convention, units, signs, measures, configuration,
   constraint/load convention, state variables, and commit timing;
3. outputs and failure behavior actually consumed;
4. a minimal deterministic reference case and an edge case;
5. the exact legacy commit, environment, command, tolerances, and output digest;
6. an independent analytical, textbook, finite-difference, or alternate
   implementation reference where possible; and
7. known legacy defects that the v3 invariant intentionally supersedes.

Legacy object layout, dynamic imports, `GlobalData`, element loops, mutable
history, `eval` load functions, and output side effects are not extracted as target
architecture.

### Proof ladder for a vertical slice

Every slice chooses the applicable layers:

1. **Schema and compiler:** validation, source IDs, compatibility, deterministic
   fingerprints, no aliasing, read-only ownership, total membership, geometry.
2. **Kernel mathematics:** direct shape/quadrature/material/formulation comparisons,
   exact conventions, scale-aware invalidity, directional derivatives.
3. **Contribution/assembly:** simple COO oracle, slot-plan equivalence, block-order
   invariance, sign-attributed force/tangent channels.
4. **Program/constraints:** additive loads, affine map, cycles/conflicts, derivatives,
   full and reduced balance, reaction/multiplier basis.
5. **State evolution:** repeat evaluation, A-then-B independence, line-search
   rejection, failed-step rollback, exact atomic commit, formulation and material
   state, checkpoint/rebind.
6. **Public analysis:** authored specs through compile/prepare/solve/verify, both
   reusable and one-shot flows, failure behavior, provenance, immutable snapshots.
7. **Representative system:** heterogeneous/stateful model, realistic scale,
   conservation/energy/restart as applicable, and complete public output.

### Parity and superseding invariants

Parity is useful when it compares meaningful quantities at the same configuration
and convention. Compare more than a final displacement:

- local response and consistent tangent;
- assembled internal/external/inertia/damping/constraint channels;
- reduced equilibrium, full balance, reactions, and constraint work;
- state/history and accepted generations;
- energies, dissipation, continuation parameters, and cutback trace;
- raw integration-point output and source localization; and
- eigen or transient residuals when applicable.

When legacy behavior contradicts the design, the new invariant wins only with a
`change` disposition, explicit counterexample, independent reference, compatibility
statement, and acceptance test. Examples already known include additive loads
instead of overwrite, invalid orientation rejection instead of `abs(det J)`, and
transactional rollback instead of caller-state mutation.

### Mandatory cross cutting cases

The portfolio maintains proof rows for:

- **heterogeneity:** Q4/T3, Tet4/Quad4 node-count collision, continuum/structural/
  boundary composition, material slots, field layouts, block and authored-order
  permutation;
- **state evolution:** load/unload/reload, path dependence, candidate idempotence,
  failed iteration/step/cutback, numerical tangent perturbation, simultaneous
  analyses, and exact accepted-trial commit;
- **constraints:** nonzero affine MPC offsets, cycles, dependency/conflict,
  parameter derivatives, fully prescribed and zero-load cases, reaction basis;
- **conservation:** attributable force/tangent channels, full/reduced balance,
  constraint work, mass/energy/dissipation where meaningful;
- **restart/provenance:** instance mismatch, content/schema/kernel verification,
  generation lineage, detached checkpoint rebind, foreign rejection, uninterrupted
  comparison;
- **geometry:** inverted and sign-changing Jacobian, scale-relative singularity,
  finite-strain current inversion, rigid translation/rotation objectivity;
- **stress scale:** a case large or irregular enough to expose assembly/state
  memory, dispatch, and solver behavior; and
- **results:** false convergence, perturbed field, stale workspace, material
  interface projection, raw localization, and independent strong verification.

### Performance proof

Performance is measured on public prepared and one-shot flows using the Intel Mac
baseline unless a packet preregisters another supported platform. Record hardware,
Python/NumPy/SciPy/Numba versions, thread settings, topology, fields, element/DOF
count, state complexity, cold compile/prepare, warm evaluation/solve, repeated
solve, peak and retained memory, and verification result.

Use the ladder from [design section 12](design.md#12-performance-proof-policy):
small correctness, medium representative public solve, scale/repeated solve, and a
heterogeneous or stateful stress case. Historical prototype numbers in
`scaling.md`, `plane_strain.md`, `tangent_assembly.md`, and `structural.md` are
hypotheses and fixtures, not gates. No tolerance, state invariant, or public flow is
weakened to recover a benchmark.

### Test gates

For code packets, run in increasing scope:

1. packet-focused tests and independent reference;
2. affected edge-case/public-flow tests;
3. both v3 Ruff configurations and format check;
4. `pytest -q test/v3`; and
5. `pytest -q` whenever the packet crosses legacy/v3, root API, input, packaging,
   examples, output, or shared dependency boundaries.

Run representative performance only when the Horizon card says it is material.
Record exact commands and commit. Do not call a flaky or nondeterministic result
green; follow the [failure policy](#failure-and-stop-policy).

## Context hygiene and fresh coordinator resume

Before ending any integration activation, persist:

- exact current integrated and dispatch-base commits;
- dirty/clean worktree status and protected user changes;
- active packet IDs, exact titles, thread/host IDs, bases, ownership, and native
  cursors;
- callbacks received and source/integrated commits;
- accepted/rejected/deferred findings and repair counts;
- proof commands/results at their exact commit;
- capability status/evidence changes;
- active watchdog and shutdown data; and
- one exact next safe action or blocked condition.

A fresh coordinator resumes without replaying chat:

1. read `AGENTS.md`, `design.md`, this method, and the live execution ledger;
2. verify the recorded branch, integrated hash, worktree status, and protected
   changes;
3. inspect only active packet threads or unadjudicated callbacks named in the
   ledger;
4. reconcile native state with the ledger, treating the ledger's last verified
   commit as authority over commentary;
5. execute the exact next safe action using the priority order in the
   [coordinator loop](#coordinator-loop); and
6. update the ledger only after the transition is proved.

Do not reread old thread history unless an exact missing decision or evidence item
cannot be recovered from the durable artifacts. Context compaction can summarize a
conversation but cannot replace this resume packet.

## Failure and stop policy

| Failure | Required response | Stop or ask condition |
|---|---|---|
| Scope/path collision | Freeze the overlapping writer; make the path review-only or serialize after the owning packet | Ask only if ownership cannot be assigned without changing requested scope |
| Bad or over-broad worker output | Do not cherry-pick; enumerate contract violations and return a bounded repair | Reject/reframe after two rounds or repeated same failure |
| Repeated failed repair | Preserve commits/evidence, mark packet blocked or rejected, return candidate to portfolio | Ask if minimum unlock is a new semantic/public decision |
| Unresolved numerical/physical choice | Create a bounded `D` decision with competing formulations, consequences, and tests; no implementation | Ask when design/evidence does not determine intended physics or compatibility |
| Stale base | Compare dependency cone; redispatch from current integrated commit for shared/core changes | Do not auto-merge/rebase across semantic conflicts |
| Dirty worktree | Identify exact owned and user-owned paths; preserve all changes | Stop if clean disjoint work cannot be established; never reset/checkout them away |
| Cherry-pick conflict | Abort integration transition and inspect semantic overlap; repair from current base | Never resolve by textual convenience or let worker merge integration branch |
| Flaky/nondeterministic evidence | Reproduce with fixed seed/environment, fresh processes, ordering/thread variants, and preregistered tolerance | No advancement until cause is fixed or nondeterminism is an explicit tested contract |
| Oracle disagreement | Check conventions/configuration and add independent reference; classify legacy defect if proved | Do not loosen tolerance to manufacture parity |
| Representative regression | Reproduce paired samples and causal memory/time model; repair, reject, or seek explicit tradeoff | Ask before accepting a material correctness-compatible public regression |
| Missing callback | Use one native status snapshot and request callback once | Watchdog only under its explicit policy; no continuous poll |
| Tool/environment failure | Record exact command/error and separate sandbox/platform failure from code failure | Block after three bounded retries or when external change/authority is required |

Safe in-scope alternatives must be exhausted before `BLOCKED`. “Hard,” “slow,”
“budget nearly used,” or “reviewer has not replied yet” is not blocked. Waiting work
with a direct callback simply yields; it does not promise continuous background
agency.

## Do not automate

Do not automate:

- design amendment or semantic-owner changes;
- physical/numerical interpretation, sign/measure/unit choices, or tolerance
  relaxation;
- preserve/change/retire decisions, public deprecation, or legacy deletion;
- commit integration, conflict resolution, merge, push, force operations, or
  destructive worktree cleanup;
- optional dependency, GUI, external service, secret, or authority expansion;
- acceptance of a performance/correctness tradeoff;
- conversion of unresolved reviewer findings into “advisory” status;
- deletion or rewriting of durable evidence/ledger history; or
- a migration-complete declaration.

These actions are irreversible, authority-expanding, physically ambiguous, or too
high-risk for status-driven automation.

## Evidence-checkable definition of migration complete

Migration is complete at one exact clean `v3` commit only when all of the following
are true:

1. **Coverage is total.** Every legacy public, solver, element, material, model,
   constraint, load, output, CLI, example, GUI, RVE/FE2, and ROM capability is in
   the ledger with an approved preserve/change/retire disposition and terminal
   `proved` or `retired-proved` status. There are no deferred, undecided,
   provisional, or hidden “out of scope” rows.
2. **Public flow is replaced.** Every preserved/changed capability runs through
   authored specs -> compiled model/program -> prepared analysis -> explicit state
   transaction -> typed analysis -> verified solution. Root package and CLI behavior
   match their approved compatibility decision.
3. **Old paths are gone or quarantined.** Production v3 has no runtime dependency
   on legacy element/material/model/solver objects. Legacy imports may exist only
   in isolated reference harnesses. `_prototype_assembly.py`, `ProblemDefinition`,
   `LoadedProblem`, duplicate solvers, and duplicate semantic owners have been
   deleted from active v3 after their evidence is exhausted.
4. **Compatibility is explicit.** Each public name, `.pro`/`.dat` form, result,
   checkpoint, CLI, writer, and optional GUI behavior is preserved through one
   adapter, deliberately changed with migration guidance, or retired with approval.
   No accidental shim silently selects the old runtime.
5. **Documentation and examples tell current truth.** Maintained docs, type
   contracts, examples, notebooks, and CLI help use the real public v3 flow. Every
   maintained example is executed or explicitly retired; archived material is
   visibly historical.
6. **Numerical/physical proof is complete.** Applicable parity, analytical
   references, edge cases, heterogeneity, state evolution, constraints,
   conservation, restart/provenance, result verification, and failure behavior
   pass at the completion commit.
7. **Performance is representative.** The public-flow benchmark ladder records
   cold/warm time and memory on the Intel Mac baseline, with no unadjudicated
   material regression and no hardware threshold elevated to architecture.
8. **Repository gates pass.** Focused capability matrices, `test/v3`, the full
   suite, Ruff/format, type/static checks in scope, examples/CLI, link/stale-term
   scans, and `git diff --check` are green at the same commit.
9. **No hidden duplicate owner remains.** Import/dependency and semantic scans show
   one active owner for model/program compilation, contribution topology,
   constraints, accepted/trial state, each analysis family, verification, adapters,
   and output projection. Independent reviewers find no unresolved P0/P1 duplicate.
10. **Operations are closed.** The ledger names the completion commit and evidence,
    all worker/reviewer packets are terminal, all watchdogs are disabled, the
    worktree is clean, and landing/push remains a separate explicit action.

This definition is not satisfied when one unclassified legacy example, one maintained
public flow that still constructs `GlobalData` or a prototype carrier, one
path-dependent result that cannot be restarted/verified, one hidden compatibility
fallback, or one unresolved required reviewer finding.

## Adoption and rollout state

### Adopted method and ledger

The integration owner reviewed D0-A after the original `R0-A`/`R0-B` findings were
repaired and the combined foundation gates passed. Adoption made these focused,
documentation-only changes:

1. `AGENTS.md` routes full legacy-to-v3 migration here while non-migration
   structural refactors continue to use `refactor_playbook.md`;
2. `refactor_playbook.md` links this workflow without duplicating the state machine;
3. `phase0-execution.md` became the sole live `migration-execution.md` ledger while
   retaining the foundation history;
4. `README.md` routes new sessions to the same method and ledger;
5. `feature-parity.md` is historical after R0-E absorption; its old statuses and
   exclusions have no classification authority; and
6. `design.md` remains unchanged because no architectural invariant was amended.

There is no second live status file and no watchdog. Existing direct callbacks and
bounded native waits remain sufficient.

### Foundation re-review at adoption

Fresh reviewers were launched against exact combined repaired commit
`faab0c938705f59fc5a22e702413f295af1dcadb`. Task failures, replacements,
callbacks, findings, and repairs are deliberately recorded only in
`migration-execution.md`, so this method does not become a second status ledger.
Compiler integration remains closed until required callbacks are adjudicated and
accepted P0/P1 findings are repaired and re-proved.

### Next trial

- Seed every legacy capability at E0 and classify high-level rows with Sol/max.
- Trial the workflow on `P0-D`, its `R0-F` reviewer, and then the serial Phase 1
  packets.
- Record friction as concrete transition failures, not another process roadmap.
- Change this method only if the trial exposes a testable operating defect.

### Retire overlapping guidance

After the first complete Phase 1 trial:

- verify that any unique legacy breadth from `feature-parity.md` remains represented
  in the R0-E evidence and live ledger overlay;
- mark old P0-P8/Cursor workflow artifacts historical in the document map;
- retain numerical notes as evidence only; and
- ensure the standard resume packet reads one design, this one migration method,
  and one live ledger.

## Coordinator loop

The loop performs one safe transition at a time. It does not busy-wait.

```text
CONSTANT MAX_REPAIR_ROUNDS = 2

function coordinator_activation():
    read AGENTS.md, design.md, migration_workflow.md, migration-execution.md
    assert recorded branch is v3
    resolve and compare current HEAD with ledger.integrated_commit
    inspect git status without changing user-owned files

    if base mismatch or dirty overlap exists:
        record exact mismatch and apply failure policy
        return BLOCKED_OR_RECONCILE

    # Highest priority: consume evidence already produced.
    callback = first unprocessed terminal callback in ledger order
    if callback exists:
        validate ID, thread, commit, ancestry, paths, evidence, warnings
        if callback is BLOCKED:
            classify blocker as repairable, semantic, authority, or external
            send bounded repair only if within frozen scope and rounds < 2
            otherwise record minimum unlock and return BLOCKED
        if mechanical precheck fails:
            request repair round + 1, or reject after round 2
            persist transition and return
        move packet to INTEGRATION_REVIEW
        persist exact next action and continue locally

    packet = first packet in INTEGRATION_REVIEW
    if packet exists:
        review contract, diff, ownership, consumers, proof, and warnings
        adjudicate every finding
        if required repair:
            request repair round + 1 or reframe after round 2
            persist and return
        if rejected:
            record evidence, return candidate to portfolio, persist, return
        require clean integration worktree
        cherry-pick exactly reviewed source commit in declared order
        record source hash and resulting integrated hash
        run focused gate immediately
        if gate fails:
            stop; create post-integration repair packet from current commit
            persist and return
        move packet to CHERRY_PICKED
        persist and continue locally

    packet = first CHERRY_PICKED packet requiring independent review
    if packet exists:
        if reviewer not dispatched:
            freeze reviewer focus/output/callback contract and dispatch if authorized
            persist thread ID/title/base; return and rely on callback
        if reviewer active and no callback:
            use at most one bounded native wait while this activation is useful
            persist only a transition; otherwise return without polling

    reviewer = first unadjudicated reviewer callback
    if reviewer exists:
        adjudicate each P0/P1/P2 finding with evidence
        if accepted P0/P1:
            create bounded repair packet from current integrated commit
            persist and return
        if an unresolved semantic choice remains:
            create a D packet or ask user under authority policy
            persist and return
        move slice to COMBINED_PROOF
        persist and continue locally

    slice = first item in COMBINED_PROOF
    if slice exists:
        run focused + public + edge-case + v3 + relevant full/static/perf gates
        bind every result to exact integrated commit and environment
        if flaky, failed, or semantically ambiguous:
            apply failure policy; do not advance
            persist and return
        update capability rows, evidence grades, causal deletions, and proof matrix
        disable packet watchdogs
        move slice to ADVANCED
        compute and persist exact next safe action
        return ADVANCED

    # No produced evidence is waiting. Select new work only now.
    if any active worker or reviewer exists:
        rely on direct callbacks; optionally one native wait in current activation
        return WAITING_WITHOUT_POLLING

    ready = capability rows whose dependencies and evidence prerequisites pass
    candidates = build portfolio cards for a small bounded subset of ready
    selected = priority_order(candidates)

    if selected is none:
        if every capability is proved or retired-proved:
            run evidence-checkable migration-complete audit with independent reviewer
            if all completion conditions pass:
                record completion commit; disable all automation; return COMPLETE
        record missing dependency/evidence or minimum user decision
        return BLOCKED_OR_SATURATED

    freeze selected Horizon contract, exact base, ownership, proof, merge order
    if contract contains unresolved physics/design/compatibility:
        dispatch/read a bounded D or R packet; do not dispatch writer
        persist and return
    dispatch only authorized, disjoint writers; record IDs/titles/thread/base
    return and rely on direct terminal callbacks
```

The exact current base, active tasks, findings, and next safe action live only in
`migration-execution.md`. This method defines how to consume that state; it does not
duplicate it. No compiler writer is selected while the ledger records an unresolved
foundation P0/P1 finding.
