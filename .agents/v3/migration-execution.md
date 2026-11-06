# PyFEM v3 NX8 coherent specialized-kernel charter

Live state reset on 2026-11-06 against `v3` at
`83aed1d9c29cf91d91ddde70e4f13508d4edacf2`. M127 changes this charter,
[generic_core.md](generic_core.md), and
[program-playbook.md](program-playbook.md) only. It changes no implementation,
test, notebook, benchmark workload, or benchmark result.

NX8 supersedes the live NX7 execution sequence, including generic-only prototype
deletion, the stopped NX7-4+ queue, and any implication that a baseline successor
is owed. Completed M121/NX7-1 retirements and M122/NX7-2 deletions remain
historical decisions; this charter does not restore their frozen runtimes.
[migration-history.md](migration-history.md), [evidence/](evidence/),
[parity.md](parity.md), [scaling.md](scaling.md), and `bench/results/` preserve
provenance, not additional dispatch queues. The semantic invariants in
[design.md](design.md) and [generic_core.md](generic_core.md) remain governing
constraints; their superseded migration sequences do not expand NX8.

## Exact next safe action

Request independent source-to-charter review of M127 at its exact branch tip.
The reviewer must check this frontier against current source, consumers, tests,
and benchmark envelopes, including the three program stop conditions below.
Documentation-only verification establishes a green base; it is not evidence for
any future candidate.

After a clean review, NX8-1 may begin with its read-only preflight and matched
base/candidate/base measurements. NX8-2 and NX8-3 are bounded later programs,
not automatic rewards for an NX8-1 pass. Each requires the preceding outcome to
be recorded, a fresh exact-base preflight, and its own independent review. A
failed program is abandoned without merge and without a remediation round; it
does not force the next program to run and does not create a partial migration.

## 1. NX8 north star

PyFEM v3 is one coherent scientific semantic kernel with the best measured
specialized implementation for each job. Coherence means shared mathematical
meaning, explicit ownership, auditable state, and independent numerical
conviction. It does **not** mean one concrete class hierarchy, one numerical
layout, one evaluator, or deletion of every specialized public route.

The generic spine is:

```text
ModelSpec + ProgramSpec + plain researcher declarations
                       |
                 normalization
                       |
                compile_system
                       |
                CompiledSystem
       entities + spaces + operators + specialized payloads
                       |
        CompiledConstraintMap + DriverAssemblyPlan
                       |
       NonlinearStaticDriver or RiksDriver schedule
                       |
 StateTransactionOwner + nested execution + disposable workspace
                       |
       typed accepted records + solve-free observations
```

`authoring/` and `io/legacy_deck.py` are adapters into this spine. Specialized
operator payloads and fused array kernels are first-class implementations of its
channels. They are not architectural failures and need no generic `einsum`
fallback when their semantics and independent oracles are strong.

The existing Q8/skim prototype is also supported production where it wins. Its
`ProblemDefinition`/`pack`/specialized-kernel/solver route is not a deprecation
queue, a fallback, or a deletion target by definition. NX8 does not require it to
become temporary, does not migrate its consumers to the generic route, and does
not make its eventual removal an architectural completion condition. A future
consumer-backed proposal may change that disposition only with fresh scientific,
performance, memory, and complexity evidence; generic-only deletion is already
falsified.

One semantic kernel therefore constrains meaning and duplicate ownership, not
the measured choice of implementation. New work must not add a compatibility
runtime that forwards between routes, a second accepted-state owner, or a second
result/verification stack merely to make the routes look identical.

## 2. Current production and scientific consumers

| Consumer or workload | Supported implementation and disposition |
|---|---|
| Uniform Q8 scale cases and linear/nonlinear/Riks skims | **PRESERVE** the current `ProblemDefinition`, `pack`, `_prototype_assembly`, `fem` kernels, and `solver/` route where matched evidence favors it. Keep independent analytic and legacy-v1 gates. |
| Generic Q4/T3/Q8/Hex8 small-strain and Q8 total-Lagrangian paths | **KEEP** the generic entities/spaces/operators, affine maps, driver plans, and independent geometry/force/tangent oracles. Improve Q8 selectively under NX8-1 without routing other consumers through it. |
| J2, plane-strain damage, fixed-entering-state sensitivities | **KEEP** the stateful payload/state-transaction semantics, pure references, derivative oracles, and bitwise/determinism gates. |
| FE², periodic RVE authoring, homogenization, nested execution | **KEEP** explicit nested execution, accepted-state ownership, retained-buffer identity, raw 1T/nT evidence, and resource closure. |
| Truss, grounded/researcher springs, scalar signals, Riks | **KEEP** the unlike entity layouts, researcher extension points, exact cycle/correction gates, and continuation semantics. |
| Maintained PatchTest8 notebook and retained legacy-deck routes | **KEEP** the generic authored-to-accepted-result flow and analytic assertion. Adapters do not acquire solver or state ownership. |
| Historical benchmark records | **PRESERVE** as revision/workload/environment-specific evidence. No historical JSON is rewritten. |

A maintained consumer is a notebook, registered benchmark workload, supported
deck route used by one, or a small authoring/extension flow required by those
scientific paths. Tests and public exports supply evidence or access; neither by
itself selects a capability. Source presence, legacy class boundaries, sunk cost,
and test count are not consumers.

The M121 retired scientific families remain retired, but retirement is not a
permanent support boundary. Section 4.4 defines their consumer-backed,
kernel-native return. The M122 frozen compiler/analysis/result vertical and
`Solution.verify` remain deleted; accepted driver records and fresh independent
numerical tests are the audit surfaces.

## 3. Falsification evidence and green production state

### 3.1 M123 — generic-only deletion failed

M123 was **abandoned on measured evidence, not merged**. Its branch
`feat/nx7-3-retire-prototype-solve-lane`, final source `b9ec035`, and final
documentation `1e55f90` are historical experiments, not production migrations
to resume. No partial consumer migration or prototype deletion was adopted.

Final `bench.common.measure` fresh-end-to-end ratios against the intact
`fbb72fa` prototype control were:

| Size | Final candidate / original deletion candidate / control (ms) | Final/control |
|---|---|---:|
| 2x2 | 7.2504 / 69.7946 / 0.9467 | 7.66x |
| 8x8 | 27.8077 / 177.2314 / 2.1784 | 12.77x |
| 32x32 | 403.9621 / 1619.2937 / 43.9407 | 9.19x |

The final candidate remained 31.58x/24.66x/25.32x control for load at those
sizes. Successive remediation reduced the unified-zero-diff nonblank deletion
from 2,426 production and 1,593 test lines to **1,703 production and 1,471 test
lines**, missing the then-required 1,800/1,500 floors by 97/29. Green correctness
did not compensate for performance or complexity failure.

The original clean reference-machine artifacts remain preserved outside the
commit graph:

- `.tower/worktrees/wt-123/bench/results/run_nx7_3_candidate.json`, at `58957f6`,
  212 records/756 cells, SHA-256
  `467158efb56faa6b781f3597feb9ed715f7e5768d88118bd0b0d5c9c4b04eef6`;
- `.tower/worktrees/wt-123/bench/results/run_nx7_3_control_base.json`, at
  `fbb72fa`, 212 records/756 cells, SHA-256
  `77eda94eed3c6514df8217c99647c6d7647c65529752aa5dfdfad549e6998f19`.

All 34 generated decks were byte-identical. Independent comparison found 360
flagged cells, no missing cells, and no correctness failures. The artifacts and
branch are falsification evidence, not merge candidates or a baseline request.

M124's source/profile survey identified the structural causes: eager recursive
array export/serialization/hash identity, object-rich per-entity declarations,
retained validation-only B/gradient tensors, generic `einsum` rather than the
existing fused Q8 kernel, and repeated int64 COO/index ownership. Its tangent
probe measured 18.9435 ms for generic `einsum` versus 0.8115 ms for warm fused
Numba, with 1.648e-14 maximum relative deviation. That supported a specialized
kernel hypothesis; it did not establish whole-path viability.

### 3.2 NX7-3P bounded falsification outcome — abandoned

M126 executed one integrated candidate from `c4930cf` and **abandoned it without
merge**. The candidate used lazy export/identity, columnar views, batched
geometry, fused tangent reuse, compact indices, selector-only reductions, and
explicit plan injection. It did not use a weak global cache, public
`PreparedExecution`, second runtime, or prototype fallback. Its focused 124-test
suite passed, but those techniques and correctness were not sufficient.

The full 2x2/8x8/32x32, PlaneStress/PlaneStrain, 1T/16T matrix was measured for
both programmatic-array and identical-deck envelopes. **Every comparable fresh
end-to-end cell failed both `ratio <= 1.25` and `delta <= 1 ms` against the
prototype.** The 32x32 results alone make the structural conclusion clear:

| Material / threads | Array candidate / prototype (ms) | Deck candidate / prototype (ms) | Generic load (ms) | Plan + cached solve (ms) |
|---|---:|---:|---:|---:|
| PlaneStress / 1T | 169.5545 / 54.2631 | 208.2232 / 63.4111 | 120.2872 | 76.0956 |
| PlaneStress / 16T | 170.9830 / 48.8930 | 199.6740 / 62.8428 | 121.4704 | 75.7381 |
| PlaneStrain / 1T | 173.4316 / 47.0801 | 195.2032 / 61.1287 | 123.1377 | 76.8481 |
| PlaneStrain / 16T | 176.9210 / 46.1985 | 209.6066 / 62.5330 | 134.1498 | 78.4380 |

The pre-candidate 32x32 generic fresh-deck cost was 1418.0507--1445.1778 ms at
1T. Thus M126 improved its generic base substantially while still failing the
prototype comparison. NX8 learns both facts: selective generic improvement can
be useful, and that usefulness does not justify deleting or replacing the
prototype where the prototype remains better.

Other independent gates also failed:

- COO exactly matched prototype bytes, but 32x32 `tracemalloc` was
  36.686--36.747 MiB versus 10.063 MiB. Logical COO bytes, traced allocation,
  and RSS are different quantities.
- Unified-zero-diff accounting against `fbb72fa`, excluding only the focused
  benchmark rewrite, was **+340 production and -134 v3-test nonblank lines**.
  Production complexity increased.
- Cached/cold states were raw-identical, all candidate 1T/16T accepted-state and
  tangent hashes matched, iteration/factorization counts stayed at 2/1, and
  generic-versus-legacy state deviation remained within 9.948e-18 absolute.
  Fused-versus-original tangent deviation reached 1.863e-9 at 32x32; raw old/new
  arithmetic equality was not claimed. Correctness could not rescue the other
  gates.

The complete protocol and stage tables remain recoverable at `f28e852`; the
summary anchor above is retained for the historical link in
[migration-history.md](migration-history.md). The candidate, focused harness,
and generated timing artifacts were removed. NX8-1 is not permission to recover
that branch or to rerun its broad integrated design.

### 3.3 Restored production is green and unchanged

At `83aed1d`, `pyfem/v3`, `test/v3`, `bench`, and `notebooks/v3` have no
implementation difference from `fbb72fa`; subsequent commits record the M126
outcome only. The restored-tree M126 controls passed:

- both complete repository digit-mode suites, **1,156 tests plus 12 subtests
  each** (151.70 s normally and 103.71 s with `PYTHONINTMAXSTRDIGITS=640`);
- the headless PatchTest8 notebook analytic assertion;
- `python -m bench.run gates` with **0 failures**, including Q8, legacy, FE²,
  J2, total-Lagrangian, and Riks controls; and
- every exact per-tree and bare Ruff check/format invocation.

These are production-base controls, not evidence for an abandoned candidate or
for this document diff. M127 then repeated the required documentation gates on
this NX8 tree: both complete suites again passed **1,156 tests plus 12 subtests
each** (160.38 s normally and 103.87 s with `PYTHONINTMAXSTRDIGITS=640`), with
the same 43 existing SciPy sparse-efficiency warnings, and all eight exact
per-tree Ruff check/format commands plus bare check and bare format passed.
These M127 results are likewise base-code controls, not implementation evidence;
no candidate benchmark or baseline artifact was created.

`bench/results/baseline_m3.json` remains unchanged at SHA-256
`d244664a58efdfcef208908cfac279d337e4c5a660cde59d3304edaf4117a172`.
There is no `baseline_nx7_3.json`, NX8 successor, rewritten historical result,
or commitment to create one. NX8 programs produce matched targeted evidence,
not baseline promotions.

## 4. Durable acceptance policy

### 4.1 Preserve scientific usefulness and specialized wins

A capability must have a truthful authored-to-accepted-result path, a maintained
consumer, and independent conviction appropriate to its science. A faster
implementation that weakens any of these fails. A generic implementation that is
clean but structurally slower is not automatically the best implementation.

Specialized payloads, fused kernels, range-appropriate arrays, and explicit
preparation are preferred when they express the same mathematical contract and
win on matched evidence. Keeping the prototype where it wins is a valid durable
disposition. Prototype deletion, generic consumer migration, and line-count
reduction are not independent goods.

### 4.2 Delete only duplicate ownership or dead code

DELETE is permitted only after a source-level inventory proves one of:

1. the code is dead: no maintained consumer, live adapter, independent oracle,
   or required extension boundary uses it; or
2. it duplicates a meaning whose single surviving owner is named, with every
   retained behavior and independent oracle mapped to that owner.

A specialized route that provides measured value is not duplicate ownership
merely because another route expresses similar mathematics. Deletion must remove
exports/adapters/maintenance obligations coherently, must not add a forwarding
facade, must reduce net complexity, and must cause **no measured performance
loss** on affected workloads under section 5. Historical source remains in Git;
no importable reference runtime or skipped test remains behind.

### 4.3 Optimize only independently useful, complexity-reducing changes

An optimization proceeds only when all are true:

- it is useful on its own, without a later prototype deletion or consumer
  migration to justify it;
- a predeclared public end-to-end workload improves beyond matched
  control/control variability, not merely a private microkernel;
- no affected comparable workload has a resolved performance loss;
- numerical, state, determinism, memory, and ownership contracts pass;
- net nonblank production **and** v3-test complexity both fall against the exact
  base, with benchmark instrumentation reported separately; and
- it adds no public lifecycle owner, compatibility runtime, or speculative
  abstraction.

An optimization that needs a second remediation round has failed. A line
reduction that removes an independent oracle, hides work in a new runtime, or
moves complexity into uncounted generated code is not a reduction.

### 4.4 Re-expand through consumers and kernel-native features

Retired or new physics returns through a named maintained consumer and an
explicit scientific contract: equations, state timing, observations, failure
behavior, literature or analytic references, and failing controls where
practical. The feature uses existing entities, discrete spaces, typed operator
payloads/channels, affine maps, driver plans, state transactions, and accepted
observations. It adds a second solver/state/result runtime only if a separate
scientific counterexample proves the current semantics insufficient; NX8 does
not authorize such a runtime.

Contact/cohesive, thermal/thermoelastic, rate-dependent, SOVS, crystal, beams,
shells, dynamics, ROM, transient integration, and total path sensitivities are
therefore possible future consumer-backed work, not hidden NX8 completion tasks.
Their historical evidence remains addressable. Re-expansion must not resurrect
the M122 frozen vertical or wrap deleted APIs in a compatibility runtime.

### 4.5 Fixed architectural prohibitions

- **No public `PreparedExecution`.** `DriverAssemblyPlan` owns prepared
  topology, `StateTransactionOwner` owns accepted state, and
  `PreparedNestedExecution` owns nested resources. Private helpers may remove
  duplication but may not become a fourth public lifetime.
- **No typed DSL.** `ModelSpec`, `ProgramSpec`, registry descriptors, and plain
  researcher declarations suffice. No code generator, universal dictionary
  protocol, or plugin-discovery runtime is a simplification.
- **No compatibility runtime or second result stack.** No facade dispatches to
  both prototype and generic solvers, and no replacement `Solution.verify` is
  introduced. Independent tests may recompute mathematical claims.
- **No hidden cache ownership.** Explicit validated plan reuse is allowed;
  process-wide weak caches and ambient accepted state are not.
- **No baseline successor commitment.** Existing baseline/history stays
  unchanged throughout NX8-1..3. Any later promotion needs a separate reviewed
  full-coverage proposal and is not an expected NX8 outcome.

## 5. Common evidence and hard-stop contract

The detailed numerical, instrumentation, and command rules live in
[program-playbook.md](program-playbook.md). Every implementation program obeys
this complete contract on one exact candidate tip.

### 5.1 Atomic candidate and complexity accounting

Start from a clean, named base revision. Preflight names affected consumers,
owners, tests, workloads, expected line delta, and measurement cells before hot
code changes. Build one integrated candidate only within that program's allowed
techniques. Do not leave partial consumer migration or combine programs to make
a failed gate pass.

Report unified-zero-diff nonblank Python deltas for `pyfem/v3`, `test/v3`, and
`bench` separately. Generated files are not complexity exclusions. Record public
export/signature changes explicitly. Historical counts are comparisons for their
revisions, not permanent test-count targets.

### 5.2 Matched performance and memory evidence

Use identical generated inputs and equivalent public envelopes for exact-base
control and candidate on the same machine, numerical-library versions, thread
counts, cache state, and thermal conditions. Keep deck read, compile, cold plan,
first solve, reused-plan solve, and complete fresh end-to-end costs distinct
where applicable. Never compare prototype solve-from-assembled with generic
fresh compile/driver, rename an envelope, omit a stage, or manufacture cache
warmth.

Run a matched **base/candidate/base** sequence and report all samples, medians,
environment provenance, correctness, and missing cells. The repeated base runs
predeclare the per-cell control/control variability used to distinguish movement
from noise. The established outer regression ceiling remains:

```text
candidate_median / control_median <= 1.25 OR candidate_median - control_median <= 1 ms
```

It is necessary, not a license for a resolved slowdown. A candidate slower than
control beyond the repeated-base envelope fails **no performance loss**, even if
it passes that outer ceiling. An optimization's declared primary cells must also
improve beyond that envelope. There is no averaging across workloads, materials,
threads, stages, or memory metrics.

Report logical array/COO/CSR bytes, `tracemalloc`, and RSS high-water separately
when used. None substitutes for another. No candidate may retain eager
array-valued runtime manifests solely to derive identity, duplicate owned index
copies, or validation-only tensors that the selected hot path no longer uses.
Cold/warm/fresh-process/JIT scopes remain distinct. Full baseline runs and
baseline promotion are outside all three programs.

### 5.3 Numerical and repository gates

Preserve independent analytic/legacy/literature oracles, accepted/trial
ownership, rejected-attempt byte identity, constraint diagnostics,
iteration/factorization/cycle/correction counts, and raw 1T/nT determinism.
Fused arithmetic requires a measured documented band and unchanged solve
counts; never claim raw old/new equality without proving it. Raw-bit pins follow
`test/v3/conftest.py`: the reference platform selects raw uint64, while other
platforms execute the documented tolerance branch without skipping.

Each candidate must pass its focused numerical matrix, the headless notebook
assertion, `python -m bench.run gates` with zero failures, both complete
repository digit-mode suites, and every exact and bare Ruff check/format command
in the playbook. A red gate invalidates timing. Existing failures must be shown
on the exact base and cannot be silently reclassified as candidate passes.

### 5.4 Independent review and abandonment

Independent review must inspect current source and consumers, the candidate diff,
raw matched artifacts, complexity accounting, and every program-specific gate.
Prose-only or test-count review is insufficient. Only a clean source-to-charter
review at the exact tip can recommend merge.

Any missing required cell, scientific or behavioral change outside scope,
performance or memory failure, complexity shortfall, public-lifecycle addition,
or need for another cache/lifecycle/attribution/micro-optimization remediation
round abandons that candidate **without merge**. Restore the base, retain only
the honest negative record and approved evidence artifacts, and stop. Do not
rescue the candidate by relaxing thresholds, moving to the next technique, or
starting the next program as an implicit retry.

## 6. NX8-1 — selective Q8 performance and data layout

**Purpose:** improve the existing generic small-strain Q8 route in place. This is
not a prototype-replacement candidate and has no deletion or consumer-migration
step.

### 6.1 Bounded scope

The only optimization targets are generic Q8 linear PlaneStress/PlaneStrain
construction and evaluation, its exact constraint/plan interaction, and identity
or export work directly required by that route. Q4/T3/Hex8, stateful J2/damage,
Q8 total-Lagrangian, FE², truss/spring/Riks, and prototype paths are controls,
not optimization targets; shared changes must preserve and measure every affected
one.

Only these five techniques are authorized:

1. **Fused tangent and reuse.** Bind the existing fused NumPy/Numba small-strain
   kernel in `fem/element.py` behind `Q8ContinuumOperator`'s existing channel
   contract, with immutable linear tangent precompute/reuse. Keep geometry and
   constitutive validation honest and diagnostic recovery explicit.
2. **Validation once.** Validate external/authored/restore boundaries once and
   trust exact immutable compiler-owned values downstream. Do not remove public
   input checks, restore compatibility checks, numerical precondition checks,
   or first-use sabotage coverage.
3. **Explicit plan reuse.** Allow an explicitly owned `DriverAssemblyPlan` to be
   reused through the existing driver boundary after exact system/map identity,
   fingerprint, load, and state compatibility validation. Define caller/driver
   lifetimes. No weak global cache or public `PreparedExecution`.
4. **Range-appropriate arrays.** Use the smallest safe index dtype for actual
   ranges and SciPy/Numba compatibility, adopt owned arrays without repeated
   copies, and preserve canonical stable reductions. Genuine MPC products and
   FE² maps retain their full semantics.
5. **Lazy export only.** Derive normalized-recipe/export/fingerprint arrays only
   when an actual export, persistence, provenance, or diagnostic boundary asks
   for them. Preserve semantic identity and restore behavior without eager
   derived-array manifests.

No columnar entity/source rewrite, selector-only reduction, new offset fast
path, new serialization format, general attribution rewrite, prototype edit, or
other optimization is authorized. In particular, do not recover M126's complete
candidate merely because its individual techniques appear similar.

### 6.2 Required measurements and acceptance

Preflight reproduces the generic base and records the current prototype only as
a supported-production reference. The candidate/control acceptance comparison is
against the **exact generic base**, not the abandoned M126 candidate and not a
prototype-parity deletion budget.

Measure programmatic-array and identical-deck public envelopes at 2x2, 8x8, and
32x32, for PlaneStress and PlaneStrain, at 1 and 16 threads. Report deck read,
compile, cold plan, first solve, explicit reused-plan solve, and complete fresh
array/deck end-to-end costs for every cell. The primary useful-improvement cells
are **all 32x32 fresh array and fresh deck end-to-end cells, for both materials
and both thread counts**. Every primary cell must improve beyond the repeated
base envelope. Every other comparable cell must have no resolved performance
loss and satisfy the outer regression ceiling. Any shared validation, identity,
array, constraint, plan, or driver change also requires matched affected cells
for every retained FE²/damage/J2/TL/truss/spring/signal/Riks/notebook route it
touches, under the same no-loss rule; Q8-only timing cannot clear a shared-code
change. A private tangent-only win or a prototype comparison cannot substitute
for a primary public-e2e improvement.

Acceptance additionally requires:

- analytic Q8/patch and independent legacy gates, documented fused arithmetic
  deviation, unchanged accepted states within the declared numerical policy,
  and exactly unchanged iteration/factorization counts;
- raw 1T/nT accepted-state, tangent, and sparse-reduction determinism on the
  reference platform or the documented always-executed tolerance branch;
- logical COO/index bytes and `tracemalloc` no greater than the exact generic
  base in every measured cell, with a resolved 32x32 traced-memory reduction;
  prototype memory is contextual, not a migration acceptance proxy;
- no eager derived-array manifest, duplicate owned index copy, or retained
  validation-only B/gradient tensors on the optimized linear Q8 path;
- negative nonblank deltas in **both** production and v3 tests, with benchmark
  instrumentation delta reported separately; and
- all common numerical/repository gates and clean independent review.

### 6.3 Hard stop

Stop without merge if any authorized technique cannot produce the required
public improvement and complexity reduction on its own, any non-primary cell has
a resolved loss, memory increases, any manifest/tensor prohibition is violated,
any required cell or numerical gate fails, or success needs an additional
optimization/remediation round. Do not migrate Q8/skim consumers, delete any
prototype file, promote a baseline, or leave an unused plan-injection API behind.

## 7. NX8-2 — private generic driver-runtime consolidation

**Purpose:** remove duplicated private orchestration from the existing generic
drivers without changing behavior. This is a complexity program, not a new
runtime design.

### 7.1 Preflight and bounded scope

Inventory `driver/nonlinear.py`, `driver/riks.py`, `driver/plan.py`,
`driver/nested.py`, `driver/task_pool.py`, `driver/continuation.py`, their private
workspace/record interactions, and the owning state interfaces. Name each
duplicated evaluation/factorization/nested-scope/accounting block, its sole
surviving owner, and the tests and workloads that falsify behavior changes.

The preflight must show a credible path to **at least 300 net production
nonblank lines removed** after all call-site and private-helper changes. If it
cannot, stop before implementation. Only private functions/classes within the
existing driver/state ownership boundary may change. Prototype `solver/`,
numerical payloads, public signatures/exports, and benchmark scientific gates
are outside scope.

### 7.2 Acceptance

- Remove **>=300 net production nonblank lines**, measured by unified zero diff
  against the exact base; v3-test nonblank lines must not grow and behavioral
  coverage must not decrease.
- Keep `DriverAssemblyPlan`, `StateTransactionOwner`, nested resources, drivers,
  and disposable workspaces as the existing owners. Add no public helper export,
  lifecycle object, accepted-state/factorization owner, typed DSL, or facade.
- Preserve public signatures and typed diagnostics, Newton/cutback and Riks
  prediction/two-solve/termination/resume policy, continuation evolution,
  fixed-entering-state sensitivities, nested commit/reject/resource closure,
  accepted/rejected bytes, and all iteration/factorization/cycle/correction
  counts. No tolerance or arithmetic-order change is permitted.
- Pass analytic, state, constraint, FE², J2/damage, sensitivity, TL,
  truss/spring/signal, Riks, and notebook controls plus every common repository
  gate.
- Run matched base/candidate/base timings on every affected registered generic
  workload and thread cell, including FE² and family routes and any generic Q8
  route touched. Require no resolved performance loss, no increase in measured
  retained/traced memory, no new eager manifest, and no missing correctness
  cells. A speedup is welcome but not required and cannot offset behavior drift.
- Obtain clean independent source-to-charter review at the exact tip.

### 7.3 Hard stop

Stop without merge if the final production reduction is below 300 lines, tests
or public surface grow to hide the extraction, any behavior/diagnostic/count or
raw-determinism result changes, any affected workload loses measured performance
or memory, a new lifecycle owner appears, or another extraction/remediation
round is needed. Do not proceed by reclassifying numerical changes as harmless
or by moving public driver behavior into the prototype.

## 8. NX8-3 — public surface, test, and benchmark consolidation

**Purpose:** make supported ownership and its evidence easier to understand
without introducing another lifecycle or deleting measured scientific value.

### 8.1 Preflight and bounded scope

Inventory the actual exports and consumers in `pyfem/v3/__init__.py`,
`authoring/__init__.py`, `compile/__init__.py`, `driver/__init__.py`, and
`io/__init__.py`; the notebook, deck, extension, and benchmark import sites;
duplicate test ownership and independent-oracle coverage; and duplicate bench
construction/timing/gate helpers. Every proposed deletion names its consumer
proof and surviving owner/oracle. No symbol is dead merely because it belongs to
the prototype or looks old.

Only dead/duplicate exports, duplicate tests, and duplicate benchmark harness
ownership are actionable. Numerical kernels, driver semantics, state ownership,
scientific workload definitions, and historical benchmark JSON do not change.
The current prototype public route remains where its consumers and measurements
justify it; no generic-only migration or baseline successor is added.

### 8.2 Acceptance

- Remove only proven duplicate/dead surface. Ordinary users retain a truthful
  notebook route; advanced users import owning modules. Do not add a mega-facade,
  compatibility wrapper, public `PreparedExecution`, or lifecycle owner, and do
  not re-export private plan/nested/task-pool helpers for convenience.
- Map every removed test to retained independent numerical/ownership/error
  coverage before deletion. Preserve analytic and legacy-v1 independence,
  raw-bit/tolerance behavior, failing controls, state rollback, and exact count
  assertions. Test-line reduction is not permission to weaken an oracle.
- Preserve every registered benchmark workload identity, generated input,
  scientific gate, stage/envelope meaning, size/material/thread coverage, cold
  versus warm distinction, result schema compatibility, and historical artifact.
  Harness helpers may unify only where they do not share the candidate with its
  independent oracle or drop inconvenient cells.
- Report production/test/bench nonblank deltas separately; their combined total
  must fall and no tree may grow without a separately justified consumer-backed
  reason recorded before review. There is no numeric deletion quota.
- Pass the complete notebook, numerical, benchmark-gate, both-digit, and Ruff
  contract. Run matched base/candidate/base evidence for every affected workload;
  require identical coverage/correctness, no resolved production performance
  loss, no increase in retained/traced memory, and no new eager derived-array
  manifest. If harness wall time changes, report it separately from solver
  timings.
- Obtain clean independent source-to-charter review of the export inventory,
  oracle map, workload/schema diff, and exact candidate tip.

### 8.3 Hard stop

Stop without merge if a claimed dead export has a maintained consumer, an
oracle or required diagnostic loses independent coverage, a workload/stage/cell
changes or disappears, a compatibility/lifecycle runtime becomes necessary,
combined complexity does not fall, production performance or memory regresses,
an eager manifest appears, or another cleanup round is needed. Do not delete the
prototype, rewrite historical results, skip missing benchmark cells, or promote
a baseline to make the consolidation appear successful.

## 9. Durable record and frontier ownership

This file is the only live NX8 frontier. Old mission cards, proof portfolios,
retirement maps, and "next" sections are evidence for their revisions, not
parallel work queues. A later mission amends this file with actual outcomes and
the next safe action; it does not infer authorization from historical imperative
wording.

Each outcome records the exact base/tip, consumers and scientific contract,
public surface and ownership changes, oracle migrations, production/test/bench
line deltas, all raw performance/memory evidence and missing cells, both complete
suite results, every Ruff result, benchmark routes/artifacts, and the independent
review decision. Negative results remain visible. Correctness green, reviewer
count, a "GO" label, and a lower line count never compensate for missing
scientific usefulness or measured performance evidence.
