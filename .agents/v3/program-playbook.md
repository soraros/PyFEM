# PyFEM v3 research-kernel program playbook

This is the working discipline for the NX7 charter in
[migration-execution.md](migration-execution.md). It supersedes the former
parity-preservation, frozen-ABI, and feature-dispatch lessons in this file.
Numerical findings in [migration-history.md](migration-history.md),
[evidence/](evidence/), [parity.md](parity.md), [scaling.md](scaling.md), and
`bench/results/` remain valuable; their old process instructions do not select
new work.

## 1. Selection before implementation

The north star is a compiled, data-oriented, explicitly stateful FE library whose
accepted answers are scientifically auditable. The means is the smallest
coherent research kernel with real consumers, not the largest defensible subset
of legacy or earlier v3.

Review every proposed capability in this order:

1. **Consumer:** who runs it — a maintained notebook, registered benchmark,
   retained deck route, or required extension of one of those flows?
2. **Scientific contract:** what equations, state timing, observations, failure
   behavior, and independent conviction make the answer meaningful?
3. **Architectural fit:** can it use the existing entities/spaces/operators,
   affine map, driver plan, transaction owner, and typed records without a second
   runtime?
4. **Cost:** how many implementations, exports, validators, tests, benchmark
   lanes, and mental lifetime rules does it add?
5. **Disposition:** only then choose KEEP, UNIFY, REBUILD, or DELETE.

- **KEEP** means a real consumer and independent evidence justify the capability
  at its current boundary.
- **UNIFY** means the behavior survives but one existing implementation becomes
  the only owner. Compatibility wrappers are not unification.
- **REBUILD** means the consumer and science survive while the inadequate
  implementation or public route is replaced and the old path deleted.
- **DELETE** means live support, exports, adapters, and maintenance obligations
  end. Historical evidence remains addressable.

Source presence, a public name, sunk implementation cost, an old ABI manifest,
or a large test file is not a consumer. A passing test can be excellent evidence
for a capability that should nevertheless be retired. Conversely, deleting a
family is not permission to delete its independent scientific findings.

## 2. Deletion is a supported-state transition

The retirement decisions named in the execution charter are approved dispositions
under [migration_workflow.md](migration_workflow.md); that document's coverage
vocabulary and evidence grades remain useful, but its generic warning that a
proposed retirement is not approved does not block an explicitly authorized NX7
retirement. Coverage of the legacy inventory is no longer the completion goal.

A deletion mission must carry one atomic accounting unit:

- exact symbols and files removed;
- exports, registry defaults, adapter branches, diagnostics, and docs updated;
- every in-repo consumer migrated or explicitly rejected with a coded error;
- useful behavior tests mapped to retained contracts before duplicate tests go;
- historical source revision and independent oracle location recorded outside
  the live implementation;
- benchmark workload/baseline consequences declared;
- numerical and performance gates run on the resulting tree.

Do not leave importable "reference" implementations in production, skip their
old tests, or add a facade that forwards to both old and new paths. A historical
revision is the archive; Git can execute it when an investigator truly needs the
old implementation. The current checkout keeps only live support and independent
oracles.

A compatibility shim is justified only by a separately identified external
consumer with a bounded migration window. No such consumer is assumed by NX7.
"Someone might use it" is not evidence.

## 3. One owner per meaning

Cross-mission duplication is a design defect even when each implementation is
correct. The kernel has exactly one owner for each of these meanings:

| Meaning | Owner |
|---|---|
| Authored model/program values and normalization | `spec/` |
| Immutable arrays, live/content/generation identity, manifests | `model/arrays.py`, `identity.py`, `provenance.py` |
| Entities, discrete spaces, operator/channel/state declarations | `model/system.py`, `model/operator.py` |
| Family-specific payload construction and direct/nested evaluation | retained `compile/` family modules |
| Fixed affine coordinates, offsets, reactions, reductions | `constraints/` |
| Fixed COO/CSR topology, load/signal binding, stable value refill | `driver/plan.py` plus numerical sparse helpers |
| Accepted physical/operator state and atomic trial transitions | `state/owner.py` |
| Nested accepted state and execution resources | `state/nested.py`, `driver/nested.py` |
| Continuation evolution | `state/evolution.py` |
| Newton/cutback and Riks schedules | `driver/nonlinear.py`, `driver/riks.py` |
| Accepted run records and solve-free observations | `driver/contracts.py` and driver records |
| Plain authoring convenience | `authoring/` functions, not another runtime |
| Legacy deck syntax conversion | `io/legacy_deck.py` |

`DriverAssemblyPlan` is preparation; drivers are schedules; workspaces are
disposable. No public `PreparedExecution` may duplicate those lifetimes. Private
helper extraction is acceptable only when it removes measured duplication and
adds no public lifecycle object or accepted-state owner.

No typed DSL, universal operator dictionary, plugin auto-discovery, second
compiler, alternative state transaction, or second result verifier is a
simplification. Specialized numerical payloads are encouraged; specialized
copies of generic validation and ownership are not.

## 4. Oracles outlive implementations

Keep four kinds of evidence distinct:

1. **Independent live oracles** — analytic fields, rational/symbolic derivations,
   finite differences, literature equations, conservation/balance identities,
   and deliberately simple numerical references. These remain in tests when they
   cover retained behavior.
2. **Legacy-v1 comparisons** — an external implementation and shipped decks used
   to falsify v3 behavior at documented tolerances. The v1 side may remain
   independent even where that duplicates a parser or settings helper; forcing
   the oracle and candidate to share code weakens the instrument.
3. **Historical v3 implementations** — old prototypes and frozen verticals.
   Their source and tests remain recoverable at recorded Git revisions. They are
   not live dependencies, fallback paths, or permanent benchmark sides.
4. **Historical performance records** — `bench/results/*.json`, including M85,
   M116, and M118 controlled pairs. They establish exactly their recorded
   workload, revision, environment, and coverage. They do not prove universal
   scaling or require the old route to remain available.

For a retirement, record the source revision, mathematical references, test/oracle
locations, known limitations, and corrected-legacy episodes in the historical
record. Do not silently rewrite an old result or turn a known bug into a parity
target. Do not copy thousands of lines of dead code into `.agents/` to make them
"archived".

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
Jacobians; include the deliberately wrong construction failing the same probe.
Use symbolic differentiation for nontrivial closed-form reassembly when the dev
stack supports it. Preserve objectivity, geometry, balance, and constraint-work
checks where they address the physics.

Tolerance changes are scientific changes. Record normalization, state, load path,
observed deviation, and why the new band is honest. Four-way old/new × loose/tight
experiments remain the standard when convergence slack could explain a movement.

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

### 5.4 Riks and sensitivity semantics

Algorithm consolidation must not silently change Riks prediction, two-solve
normal-plane constraint, commit, termination, resume, or continuation evolution.
A later scientific redesign may change legacy quirks, but it needs its own
consumer, independent branch/limit-point oracle, and explicit migration of exact
cycle/correction-count gates. Parity is not a reason to preserve a defect.

Homogenization derivatives along map coordinates and material-parameter
sensitivities are different channels. Keep their meanings, right-hand sides,
factorization reuse, and observations separate even when they share linear
algebra.

## 6. Performance governance

Measure the public prepared and end-to-end routes, not private kernels alone.
Every timing record states model/DOF/state complexity, hardware and numerical
library versions, thread count, cold versus warm scope, correctness result, and
working-tree provenance. Logical copy/byte accounting, `tracemalloc`, and RSS are
different measurements and are never interchanged.

Use matched candidate and base-revision control runs under comparable clean
conditions. The established regression signal is ratio above 1.25 with absolute
slowdown above 1 ms unless a cell has a documented variance override. A failed
ratio is investigated, not hidden by changing thresholds. Fleet/thermal
contamination requires the candidate/control overlap evidence described in
`bench/README.md`.

Keep workload identities honest:

- Q8/skims compare against independent legacy/analytic gates on identical decks.
- J2 optimized kernels retain the pure reference and bitwise batch/status gates.
- FE² retains raw 1/n-thread identity and explicit serial/default versus opt-in
  host-pool policy. GIL-bound damage records are not parallel-scaling claims.
- FE²-memory is a synthetic copy/state protocol, not another physics solve.
- TL/Riks family routes retain their independent force/tangent/branch gates.
- Cold, true-cold, warm, fresh-process, and JIT measurements remain distinct.

A deleted implementation is not a baseline exception. Migrate the workload,
state the envelope difference, and create a reviewed baseline successor only
from a full clean reference-machine run. Keep old baseline files as history;
never promote a baseline under load or silently drop missing full-coverage cells.

## 7. Required repository gates

Run tests from the actual checkout root with the project interpreter, always as
`python -m pytest`, never bare `pytest` or `uv run pytest`. Before requesting
independent review of a charter-level or cross-cutting change, run the complete
repository suite in both integer-string modes:

```bash
python -m pytest -q
PYTHONINTMAXSTRDIGITS=640 python -m pytest -q
```

In a worktree, use the primary checkout's absolute `.venv/bin/python` while
remaining in the worktree. Read-only review may add `PYTHONDONTWRITEBYTECODE=1`
and `-p no:cacheprovider`; these do not reduce coverage.

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

Also run the invocation-invariant bare check required by the root guidance:

```bash
ruff check pyfem/v3 test/v3 bench notebooks/v3
```

The local substitute for either command form is the pinned absolute
`.venv/bin/ruff` with identical arguments. A changed-files-only lint is not a
whole-tree gate. Documentation-only missions still run the full matrix so the
review starts from a known base, but must report that the tests are base-code
evidence rather than implementation evidence for the document diff.

## 8. Review and durable record

Independent review must inspect current source and consumers, not just prose or
test counts. For a plan, the reviewer checks every named symbol/file, whether the
claimed consumer exists, whether shared numerical dependencies survive, whether
deletion targets are actually unsupported, and whether acceptance gates can
falsify the proposal. For implementation, the reviewer reproduces numerical and
performance evidence at the exact branch tip.

A mission record answers:

- what consumer and scientific contract changed;
- what was kept, unified, rebuilt, or deleted, with exact symbols;
- which independent oracles moved and which historical records stayed behind;
- actual production/test/bench line deltas and public export changes;
- both suite results, all Ruff results, benchmark routes/artifacts and coverage;
- remaining unsupported combinations and the next safe action.

Append corrections rather than silently rewriting historical claims. The live
execution charter may change the frontier explicitly; old reviews and mission
cards remain provenance. Reviewer count, a "GO" word, or a green suite cannot
compensate for a missing consumer, a shared-oracle conflict, or unsupported
numerical semantics.

The durable product is a small, comprehensible scientific kernel and instruments
that can convict it. Preserve the instruments and evidence; do not preserve every
implementation that happened to produce them.
