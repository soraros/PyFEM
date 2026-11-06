# PyFEM v3 NX8 program playbook

This is the working discipline for the NX8 charter in
[migration-execution.md](migration-execution.md). It supersedes the former
parity-preservation, frozen-ABI, generic-only deletion, and feature-dispatch
lessons in this file. Numerical findings in
[migration-history.md](migration-history.md), [evidence/](evidence/),
[parity.md](parity.md), [scaling.md](scaling.md), and `bench/results/` remain
valuable; their old process instructions do not select new work.

## 1. Select scientific value before implementation shape

The north star is one coherent, explicitly stateful scientific kernel with the
best measured specialized implementation for each job. It is neither the largest
defensible legacy subset nor the smallest generic-only codebase.

Review every proposed capability in this order:

1. **Consumer:** who runs it — a maintained notebook, registered benchmark,
   retained deck route, or required extension of one of those flows?
2. **Scientific contract:** what equations, state timing, observations, failure
   behavior, and independent conviction make the answer meaningful?
3. **Semantic fit:** can existing entities/spaces/operators, affine maps, driver
   plans, transaction ownership, and typed records express it without duplicate
   lifecycle ownership?
4. **Specialized implementation:** which payload, layout, fused kernel, and
   preparation strategy best serve that contract on matched measurements?
5. **Cost:** how many implementations, exports, validators, tests, benchmark
   lanes, allocations, and lifetime rules does it add or remove?
6. **Disposition:** only then choose KEEP, SPECIALIZE, UNIFY, RE-EXPAND, or
   DELETE.

- **KEEP** means a real consumer and independent evidence justify the current
  boundary. Keeping the Q8/skim prototype where it wins is a valid durable KEEP,
  not a temporary failure to delete it.
- **SPECIALIZE** means a measured implementation improves an existing scientific
  route without changing its meaning. It must be independently useful and reduce
  complexity; a private-kernel speedup alone is insufficient.
- **UNIFY** means behavior survives while one existing owner replaces duplicate
  ownership. A forwarding compatibility wrapper is not unification.
- **RE-EXPAND** means a named consumer returns or adds physics through the
  kernel-native operator/state/observation contracts, with independent evidence.
- **DELETE** means only dead code or duplicate ownership is removed, with no
  performance loss and no loss of independent scientific evidence. Historical
  source remains addressable in Git.

Source presence, a public name, sunk implementation cost, an old ABI manifest,
or a large test file is not a consumer. Conversely, a cleaner generic route is
not automatically the best route, and prototype deletion is not a completion
metric.

## 2. One owner per meaning, specialized payloads per job

The generic semantic spine has exactly one owner for each of these meanings:

| Meaning | Owner |
|---|---|
| Authored model/program values and normalization | `spec/` |
| Immutable arrays, live/content/generation identity, export provenance | `model/arrays.py`, `identity.py`, `provenance.py` |
| Entities, discrete spaces, operator/channel/state declarations | `model/system.py`, `model/operator.py` |
| Family-specific payload construction and direct/nested evaluation | retained `compile/` family modules and numerical `fem/` kernels |
| Fixed affine coordinates, offsets, reactions, reductions | `constraints/` |
| Fixed COO/CSR topology, load/signal binding, stable value refill | `driver/plan.py` plus canonical sparse helpers |
| Accepted physical/operator state and atomic trial transitions | `state/owner.py` |
| Nested accepted state and execution resources | `state/nested.py`, `driver/nested.py` |
| Continuation evolution | `state/evolution.py` |
| Newton/cutback and Riks schedules | `driver/nonlinear.py`, `driver/riks.py` |
| Accepted run records and solve-free observations | `driver/contracts.py` and driver records |
| Plain authoring convenience | `authoring/` functions, not another runtime |
| Legacy deck syntax conversion | `io/legacy_deck.py` |

`DriverAssemblyPlan` is preparation; drivers are schedules; workspaces are
disposable. No public `PreparedExecution` may duplicate those lifetimes.
`PreparedNestedExecution` remains the nested-resource owner, not a new universal
execution facade. Private helpers are acceptable only when they remove actual
duplication without acquiring public lifecycle, accepted-state, or factorization
ownership.

Specialized numerical payloads are encouraged. They receive arrays and scalars
at hot boundaries, not registries, model dictionaries, or solver objects. Fused
evaluation may produce a total while diagnostic requests retain attributable
parts. The existing prototype route may remain the best Q8/skim implementation;
do not force it through a generic facade or reroute its consumers merely to make
the architecture diagram uniform.

No typed DSL, universal operator dictionary, plugin auto-discovery, alternative
accepted-state transaction, compatibility runtime, or second result verifier is
a simplification. New physics uses a typed specialized payload and explicit
channels rather than a second compiler/solver/state/result stack.

## 3. Acceptance by disposition

### 3.1 Deletion inventory

Before deleting dead or duplicate code, carry one atomic accounting unit:

- exact symbols and files removed and why they are dead or duplicate;
- the sole surviving owner for each retained meaning;
- exports, registry defaults, adapter branches, diagnostics, and docs updated;
- every in-repo consumer retained or explicitly rejected with a coded error;
- useful behavior and independent-oracle coverage mapped before duplicate tests
  are removed;
- historical source revision and independent oracle locations preserved outside
  live production;
- benchmark workload, envelope, and artifact consequences declared;
- net production/test/bench complexity reported; and
- numerical and matched performance gates passing on the resulting tree.

No measured performance loss is allowed on an affected workload. The matched
base/candidate/base protocol in section 6 decides whether a slowdown is resolved;
the outer 1.25/1 ms regression ceiling is necessary but does not excuse a slowdown
beyond repeated-base variability.

Do not leave importable reference implementations, skip old tests, or add a
facade that forwards to old and new paths. Do not call a route duplicate merely
because it is specialized: measured scientific or performance value blocks
deletion. No compatibility shim is assumed; a separately identified external
consumer and bounded migration window would require its own charter decision.

### 3.2 Optimization inventory

An independently useful optimization names before implementation:

- the existing public consumer and exact-base bottleneck;
- the bounded technique and why it preserves semantics;
- the primary public end-to-end cells that must improve beyond repeated-base
  variability;
- every secondary workload that must not regress;
- logical bytes, traced allocation, and RSS measurements where relevant;
- the expected production and v3-test nonblank reductions; and
- the condition that abandons the candidate without another optimization round.

Both production and v3-test nonblank complexity must fall for NX8-1. New
benchmark instrumentation is reported separately and cannot hide production
growth. An optimization whose value depends on later prototype deletion or
consumer migration is not independently useful.

### 3.3 Consumer-backed re-expansion

A returned or new scientific feature starts with a maintained consumer, not a
legacy inventory row. Its mission states equations, field/operator/channel
semantics, accepted/trial timing, observations, diagnostics, independent
literature/analytic/numerical evidence, resource ownership, and cost. It must use
the existing semantic spine and a specialized payload wherever that is the best
implementation.

A new file format produces normalized declarations, not another compiler. A new
analysis names channels and evolution semantics, not concrete element classes.
A new output reads accepted observations and never advances history. Retired
contact/cohesive, thermal, rate, SOVS, crystal, and other physics may return this
way; the M122 frozen analysis/result vertical and `Solution.verify` do not.

## 4. Oracles outlive implementations

Keep four kinds of evidence distinct:

1. **Independent live oracles** — analytic fields, rational/symbolic derivations,
   finite differences, literature equations, conservation/balance identities,
   and deliberately simple numerical references. These remain in tests when they
   cover retained behavior.
2. **Legacy-v1 comparisons** — an external implementation and shipped decks used
   to falsify v3 behavior at documented tolerances. The v1 side may retain an
   independent parser/settings helper; sharing implementation with the candidate
   weakens the instrument.
3. **Historical v3 implementations** — old prototypes, abandoned candidates, and
   frozen verticals. Recorded Git revisions preserve them. They are not hidden
   dependencies, fallbacks, or compatibility-runtime components.
4. **Historical performance records** — `bench/results/*.json`, including the
   M85, M116, and M118 controlled pairs. They establish exactly their recorded
   workload, revision, environment, and coverage, not universal scaling or a
   requirement to delete a route.

For retirement, record source revision, mathematical references, test/oracle
locations, known limitations, and corrected-legacy episodes. Never rewrite an
old result or turn a known bug into a parity target. Do not copy dead production
implementations into `.agents/` as an archive.

The [migration_workflow.md](migration_workflow.md) E0–E4 grades describe evidence,
not entitlement to support. A retained public scientific flow needs E3-level
authored-to-verified-result evidence; E4 documentation/compatibility/performance
claims are made only where applicable. Component E2 evidence cannot promote a
compile-only family to supported public breadth.

## 5. Numerical conviction rules

### 5.1 Independence and disconfirmation

Every significant numerical claim needs an independent leg and a failing control
where practical. A reference copied from the production formula, a stored output
with no derivation, or a test that only checks a flag is weak evidence. Rational
Q4/T3/Q8 constructions, analytic patch fields, symbolic differentiation, and
legacy-v1 comparisons are preferred according to the question.

Converged residuals and consistent tangents are separate claims. Use directional
finite differences at nonzero, non-kink states for assembled and homogenized
Jacobians; include a deliberately wrong construction failing the same probe.
Use symbolic differentiation for nontrivial closed-form reassembly when the dev
stack supports it. Preserve objectivity, geometry, balance, and constraint-work
checks where they address the physics.

Tolerance changes are scientific changes. Record normalization, state, load path,
observed deviation, and why the new band is honest. Four-way old/new × loose/tight
experiments remain the standard when convergence slack could explain movement.
Fused kernels use a measured band against the original contraction and unchanged
iteration/factorization counts; raw old/new arithmetic equality is never assumed.

### 5.2 State and generation

- Repeated evaluation from one accepted generation is repeatable; rejected trials
  change no accepted bytes.
- Commit is atomic across coefficients, operator rows, nested references, and
  applicable evolution. The accepted trial corresponds to the accepted candidate.
- Stateful derivative requests state exactly whether entering history is fixed.
  Current parameter columns are increment/fixed-entering-state sensitivities,
  not total path derivatives.
- Verification never advances final history under the guise of observation.
  Homogenization and reactions use committed responses and documented reductions.
- Nested tests cover accepted and rejected macro attempts, per-IP attribution,
  stable order, retained-buffer identity, and resource closure.

### 5.3 Determinism

Disjoint writes and canonical stable-order reductions are the parallelism
contract. Do not substitute SciPy/NumPy reduction order for the project's
canonical order without evidence. One-thread/n-thread and repeated-run pins
compare raw uint64 where required, including state payloads, counters, and signed
zero where relevant.

`test/v3/conftest.py::assert_bitwise_pin` defines the platform behavior: raw
uint64 on the recorded reference platform; an always-executed documented
rtol/atol branch elsewhere. A tolerance call site must cite observed
cross-platform deviation. Platform selection is not a skip mechanism.

### 5.4 Driver, Riks, and sensitivity semantics

Private driver consolidation must not change diagnostics, Newton/cutback,
factorization reuse, nested scope, continuation evolution, or accounting. Riks
prediction, two-solve normal-plane constraint, commit, termination, resume, and
exact cycle/correction-count gates are behavior, not implementation noise. A
later scientific redesign needs its own consumer and independent
branch/limit-point oracle; parity is not a reason to preserve a defect.

Homogenization derivatives along map coordinates and material-parameter
sensitivities are different channels. Keep their meanings, right-hand sides,
factorization reuse, and observations separate even when they share linear
algebra.

## 6. Performance and memory governance

Measure public prepared and end-to-end routes, not private kernels alone. Every
timing record states model/DOF/state complexity, exact base/candidate revisions,
working-tree provenance, hardware and numerical-library versions, thread count,
cold versus warm scope, correctness result, and all samples.

Use identical generated inputs and equivalent public envelopes under matched
conditions. Run **base/candidate/base** so the repeated base establishes per-cell
control/control variability before candidate movement is called real. Keep deck
read, compile, cold plan, first solve, reused-plan solve, and complete fresh
end-to-end costs separate where applicable. A historical prototype
solve-from-assembled cell and a fresh generic compile/driver cell are
non-comparable context, never acceptance evidence.

The established outer regression signal is:

```text
candidate_median / control_median <= 1.25 OR candidate_median - control_median <= 1 ms
```

Every comparable cell must satisfy it unless a stricter program rule applies. A
candidate slower than control beyond repeated-base variability still fails no
performance loss. An optimization's primary cells must improve beyond that
variability. Never average away a size/material/thread/stage failure, rename an
envelope, omit a costly stage, or use a weak global cache to manufacture warmth.
Fleet/thermal contamination requires the candidate/control overlap evidence in
`bench/README.md`.

Keep workload identities honest:

- Q8/skims compare against independent legacy/analytic gates on identical decks.
- J2 optimized kernels retain the pure reference and bitwise batch/status gates.
- FE² retains raw 1/n-thread identity and explicit serial/default versus opt-in
  host-pool policy. GIL-bound damage records are not parallel-scaling claims.
- FE²-memory is a synthetic copy/state protocol, not another physics solve.
- TL/Riks family routes retain independent force/tangent/branch and exact count
  gates.
- Cold, true-cold, warm, fresh-process, and JIT measurements remain distinct.

Logical array/copy/COO/CSR bytes, `tracemalloc`, and RSS high-water are different
measurements and are never interchanged. No eager derived-array runtime manifest
or duplicate owned index copy is an acceptable way to make identity faster.
Range-appropriate arrays must preserve genuine MPC/FE² semantics and canonical
reductions. Report missing cells as failures, not skipped passes.

A deleted implementation is not a baseline exception, but NX8 does not presume
deletion or baseline promotion. Preserve `baseline_m3.json` and every historical
artifact. NX8-1..3 produce targeted matched evidence only; a later baseline
successor would need a separate full-clean, full-coverage, reference-machine
proposal and review.

## 7. What M123 and M126 actually proved

M123's final generic fresh-e2e cost remained 7.66x/12.77x/9.19x prototype control
at 2x2/8x8/32x32, while remediation eroded its production/test deletion to
1,703/1,471 nonblank lines. M126 substantially improved its generic base yet
failed every comparable fresh-e2e prototype cell, exceeded prototype memory, and
added 340 production lines. Both were abandoned without merge.

The architectural lesson is not "generic semantics are slow" and not "specialized
code must be deleted." It is that eager derived-array identity, object-rich hot
preparation, retained validation tensors, unspecialized contractions, and
repeated wide-index ownership can make an otherwise clean implementation
structurally costly. A cache cannot repair an unsuitable representation, and a
fast private tangent cannot prove a useful public route.

Under NX8-1 only the following ideas are eligible for fresh, selective evaluation:

- existing fused Q8 tangent construction with immutable linear reuse and a
  documented arithmetic band;
- validation once at immutable authority boundaries, retaining public input and
  restore checks;
- explicit validated `DriverAssemblyPlan` reuse with clear ownership;
- range-appropriate owned arrays without duplicate copies or changed reductions;
  and
- lazy normalized-recipe/export identity without eager array manifests.

M126's columnar source/entity rewrite, selector-only reductions, new offset fast
paths, broad attribution work, and integrated candidate are outside NX8-1.
Eager V2 runtime manifests, weak global plan caches, envelope renames, stage
omissions, and prototype fallback are rejected as acceptance techniques. Do not
resume either abandoned branch or launch a remediation sequence after a failed
gate.

The new decision rule is deliberately different from generic-only deletion: a
selective generic improvement may be merged only if it is useful and simpler on
its own, while the prototype remains supported wherever matched evidence favors
it. Neither outcome requires consumer migration or a baseline successor.

## 8. Required repository gates

Run tests from the actual checkout root with the project interpreter, always as
`python -m pytest`, never bare `pytest` or `uv run pytest`. Before independent
review of a charter-level or cross-cutting change, run the complete repository
suite in both integer-string modes:

```bash
python -m pytest -q
PYTHONINTMAXSTRDIGITS=640 python -m pytest -q
```

In a worktree, use the primary checkout's absolute `.venv/bin/python` while
remaining in the worktree. Read-only review may add `PYTHONDONTWRITEBYTECODE=1`
and `-p no:cacheprovider`; these do not reduce coverage.

Implementation candidates also run the scientific controls:

```bash
MPLBACKEND=Agg python notebooks/v3/patch_test8.py
python -m bench.run gates
```

Run all exact CI Ruff invocations, using the pinned project Ruff rather than
`PATH` (CI spells the executable `uv run --no-sync ruff`):

```bash
uv run --no-sync ruff check pyfem/v3 test/v3 --config pyfem/v3/ruff.toml
uv run --no-sync ruff check test/v3 --config test/v3/ruff.toml
uv run --no-sync ruff check bench --config bench/ruff.toml
uv run --no-sync ruff check notebooks/v3 --config notebooks/v3/ruff.toml
uv run --no-sync ruff format --check pyfem/v3 test/v3 --config pyfem/v3/ruff.toml
uv run --no-sync ruff format --check test/v3 --config test/v3/ruff.toml
uv run --no-sync ruff format --check bench --config bench/ruff.toml
uv run --no-sync ruff format --check notebooks/v3 --config notebooks/v3/ruff.toml
```

Also run both invocation-invariant bare gates required by the root guidance:

```bash
ruff check pyfem/v3 test/v3 bench notebooks/v3
ruff format --check pyfem/v3 test/v3 bench notebooks/v3
```

The local substitute for either command form is the pinned absolute
`.venv/bin/ruff` with identical arguments. A changed-files-only lint is not a
whole-tree gate. Documentation-only missions still run the full test and Ruff
matrix so review starts from a known base, but must report that these are
base-code controls rather than implementation evidence for the document diff.

## 9. Hard stop, review, and durable record

Every NX8 program has one bounded candidate. Any missing required measurement,
scientific or out-of-scope behavior change, performance/memory regression,
complexity shortfall, lifecycle addition, or need for an extra remediation round
abandons it without merge. Do not relax a threshold, rename a failed cell, move
to another technique, or use the next program as an implicit retry. Restore the
base and preserve the honest negative result.

Independent review must inspect current source and consumers, not just prose or
test counts. For a plan, the reviewer checks named symbols/files, real consumers,
shared numerical dependencies, deletion claims, benchmark envelopes, and whether
acceptance can falsify the proposal. For implementation, the reviewer inspects
the exact diff and raw matched evidence and reproduces appropriate numerical and
performance gates at the exact branch tip. Only clean source-to-charter review
can recommend merge.

A mission record answers:

- what consumer and scientific contract changed;
- what was kept, specialized, unified, re-expanded, or deleted, with exact
  symbols and owners;
- which independent oracles moved and which historical records stayed behind;
- actual production/test/bench line deltas and public export changes;
- matched performance/memory artifacts, coverage, missing cells, and correctness;
- notebook and benchmark gates, both suite results, and all Ruff results;
- remaining unsupported combinations and the next safe action.

Append corrections rather than silently rewriting historical claims. The live
execution charter may change the frontier explicitly; old reviews and mission
cards remain provenance. Reviewer count, a "GO" word, green tests, or a smaller
line count cannot compensate for a missing consumer, shared-oracle conflict,
unsupported numerical semantics, or measured performance loss.


## 10. M131 direct-J2 total-path record

M131 is a consumer-backed RE-EXPANSION of total path sensitivities, not a
rewrite of the M56/M57 increment channel. `OperatorPathSeed` and
`OperatorPathEvaluation` are optional ABI carriers; the direct J2 binding owns
reference and batched forward kernels; `Q8StatefulContinuumOperator` owns
residual-layout assembly; `NonlinearStaticDriver` owns the two-sweep schedule;
the existing `StateTransactionOwner` remains the only accepted/staged lifecycle
owner. `TotalParameterSensitivityObservation` and authoring
`total_sensitivities` are new result/request names. No AD framework, general
sensitivity registry, second state owner, nested FE2 derivative lifecycle, or
new assembly-manifest schema was introduced.

The strict nonlinear IFT solve must use the final converged Jacobian batches.
Reusing the stash from the last Newton correction is not generally the same
operator. Ordinary total commits therefore add one shared final-tangent
factorization, two path evaluations, P right-hand-side assemblies/back-substitutions,
and P-1 factorization reuses; P=1 and P=4 have identical fresh-factorization
counts. Plateau commits use exact carry-forward and add no path work. Existing
increment-only evaluation/factorization behavior is unchanged.

### Numerical and lifecycle controls

The 59-test focused command is:

```bash
python -m pytest -q \
  test/v3/test_v3_param_derivatives.py test/v3/test_v3_sensitivities.py \
  test/v3/test_v3_state_transactions.py test/v3/test_v3_state_evolution.py
```

It passes. Independent complete-path FD covers both J2 forward implementations
and a joint state/port operator probe. Truncated history is a strong failing
control. Complete re-solve FD uses `max_iterations=100`, matching one-commit
schedules and active masks on every leg; requiring only 25 iterations caused an
oracle-side reload failure and was not hidden. The fine-stencil state deviation
has more than 7x margin to its `5e-7` gate; the coarse deviation is retained to
show FD truncation. Plastic-plateau fixed-state increments are numerical zero
only to their observed roundoff (`3.84e-19`), while total U,S carry bitwise.
All codec identity/order/schema/shape/generation and malformed-payload branches
are tested.

### Consumer performance evidence

Workload: the documented 4x1 Q8 direct-J2 cantilever, 36 integration points,
schedule `(0,20,10,25)`, tolerance `1e-12`, `max_iterations=100`, four material
parameters, smooth load/unload/reload active sets. P=1 requests
`youngs_modulus`; P=4 requests all four. The control is `2P` fresh complete
primal re-solves at the same central-FD parameters used by the scientific gate.
Every solve completed and branch masks matched.

Environment: macOS 26.7.2 x86_64, Intel i9-9980HK 2.40 GHz, Python 3.13.11,
NumPy 2.3.5, Numba 0.62.1, SciPy 1.18.0, 16 Numba threads. JIT was warmed before
all samples. "Fresh model" times full model/map/driver construction plus the
run. "Precompiled warm" reuses the compiled system/map but constructs a fresh
driver/plan and run for every solve. Each cell interleaves control/candidate/
control three times. Values below are milliseconds; the direct cell has three
samples and the 2P control six.

| P | Route | Direct samples | 2P samples | Direct median | 2P median | 2P/direct |
|---|---|---|---|---:|---:|---:|
| 1 | fresh model | 85.06, 90.26, 83.99 | 256.81, 256.93, 254.26, 253.99, 259.57, 252.60 | 85.06 | 255.53 | 3.00x |
| 1 | precompiled warm | 35.36, 32.67, 34.54 | 156.78, 156.79, 155.40, 154.02, 153.11, 156.47 | 34.54 | 155.94 | 4.51x |
| 4 | fresh model | 90.84, 93.17, 90.66 | 923.53, 912.46, 909.13, 911.89, 904.27, 912.36 | 90.84 | 912.12 | 10.04x |
| 4 | precompiled warm | 41.06, 41.05, 36.21 | 515.12, 505.54, 502.02, 503.55, 498.92, 507.80 | 41.05 | 504.54 | 12.29x |

The direct route wins every median; no performance rationale is needed to
excuse a loss. Memory remains linear in P: reduced U, full observation U, and
one derivative array per primal state block. The implementation does not retain
a dense global entering-state Jacobian or per-parameter factorization.

Full digit-mode repository suites, scientific controls, and the complete exact
and bare Ruff matrix are required after this record; their results are appended
below rather than presumed from the focused suite.

Change accounting at the implementation tip: production `+1625/-20` lines and
scoped v3 tests `+973/-1`; no benchmark implementation or artifact changed.
Design/playbook prose is separate. `state.__init__` exports the two derivative
codec constants and four layout/bundle/codec types. Total driver observations
are available through committed records and `driver.contracts`; assembly-plan
manifest exports and primal codec names are unchanged.

All eight explicit-config Ruff check/format commands and both bare whole-tree
Ruff check/format commands passed on the implementation tree.

Full-suite result at the implementation tip: the default integer-string mode
reported `1171 passed, 43 warnings, 12 subtests passed` in 145.25 s;
`PYTHONINTMAXSTRDIGITS=640` reported the same counts in 109.57 s. The warnings
are the existing SciPy sparse-CSC conversion warnings. PatchTest8 completed
under Agg with its known non-interactive-canvas warning. Benchmark-gate results
are appended when that long-running control completes.

The benchmark scientific gate also passed: `python -m bench.run gates` reported
`gates: 0 failure(s)`, including the direct-J2 `material/j2-isotropic-hardening/36864`
bitwise gate and all notebook/skim/FE2/TL/Riks controls. Together with the two
full suites and all exact+bare Ruff checks, this completes the required M131
verification matrix.
