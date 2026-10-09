# Program playbook — what 60+ tower missions taught about building this library

Distilled by the control tower (2026-10-09) at the human's request. This file is
the meta-layer: not what the architecture IS (see generic_core.md / design.md)
but what the program LEARNED about doing the work — which instruments detect
truth, which disciplines compound, and which preconceptions died. Evidence refs
are tower mission ids (M<n>) and findings; their full records live in
.tower/comms/ (missions/, reviews/, findings/).

## 1. Instruments, not tests

The parity battery is a measurement instrument, not a regression suite. Written
to pin exact arithmetic (bitwise where achievable, tolerance-class where
documented), it turned "rewrite the library" into "measure two implementations
against each other" — and every divergence became a bug candidate with receipts.
Nine legacy correctness bugs were found BY the pins, not by reading code
(spring tangent projector, J2 flow[3:] slot pollution, J2 shear +G accumulation,
self.hard undefined, damage depsdstrain never assigned, damage 0.5·O3 halving,
visco a↔(1−a) swap, constraint 2-cycle hang, MPC multi-hop flatten — all M55).

In a Newton-based library, residual correctness and tangent correctness are
SEPARATE axes with separate instruments. Converged-state comparison (R=0) is
blind to tangent bugs — they only move convergence rate and every
derivative-consumer (IFT sensitivities, bifurcation, arc-length quality). The
instruments that see tangents: FD probes of the assembled tangent against the
residual map at NONZERO committed states (X1 probes, M56), and iteration-count
protocol pins — restored quadratic convergence is a regression oracle
(41 linear → 5 quadratic on the J2 vehicle, M54).

Every pin ships with a conviction leg: the buggy construction measured FAILING
the same pin (e.g. truncated-unrolled sensitivity deviates 4.1e-3 vs the 1e-8
pin). A pin without a failing control is decoration.

For closed-form derivative claims, add a SYMBOLIC leg: cross-derive with sympy
(dev group; differentiate the map symbolically, compare against the coded form
by evaluation or expression simplification). FD convicts numerically at sampled
states; symbolic derivation convicts structurally everywhere. The SOVS tangent
episode (M67) is the cautionary case: per-leg FD identifications were correct
while the full-matrix reassembly was wrong — a symbolic check of the reassembly
would have caught it before review. Prefer generated evidence over hand-derived
claims whenever a formula enters the tree.

Escalation standard: when a hard invariant appears violated, the acceptable
proof is a four-way experiment (old/new × loose/tight tolerance) separating the
change's effect from the old run's own convergence slack — the creep_test 0.31
movement was the old code's own tol slack, R=0 states identical to 2e-10 (M55).

## 2. Performance governance

The D2 compute cost model (M5) is the design governor, not a report. Every
large win was an architecture correction at a model-named anti-pattern, and
each post-win bottleneck migrated exactly where the model predicted
(kernel 85×/203× M30 → fused continuum kernels 6-15× M51 → dedup 10-13× M59 →
production CSR wiring next). If the abstraction is right, the code cannot be
slow — measure against the model's floor, not against yesterday.

Bitwise discipline pays compound interest: 1T-vs-nT raw-uint64 pins make races
impossible to ship silently (disjoint-writes parallelism preserves bits), and
canonical reduction orders make results platform-bitwise. Own your reduction
orders: scipy's csr_sort_indices is an unstable std::sort (libc++/libstdc++
bits differ) and numpy's add.reduceat inner loop is SIMD-dependent — neither is
a valid canonical reference (M59; finding 20261009-agent-b6 for the remaining
driver/plan.py case). Strict-sequential stable-order accumulation is the
canonical form.

Bench adjudication: environment is a first-class variable. The accepted
evidence protocol is candidate run + base-revision control run under matched
conditions (M51 precedent, codified in bench/README.md, M60); manifests carry
working-tree state. The baseline JSON is tower-owned; never re-baseline under
fleet load.

## 3. ABI evolution without fear

The frozen ABI evolves by ADDITIVE OPTIONAL SLOTS under the getattr precedent
(initial_rows, then parameters/derivative_channels in M56): channel-free
headers carry no attribute, the frozen validator stays untouched, manifests
stay byte-identical (sha256-pinned). No version bump was needed.

Declare capability by BEHAVIOR (a binding member), never by metadata: metadata
is inherited verbatim by teaching bindings that cannot honor the declaration
(the student-law regression convicted the metadata-gate design, M56 notes).

Every declared channel is exercised fail-closed at virgin compile before the
operator escapes (finite/complete/byte-equal-trial-rows invariants). Declared
and unexercised ABI surface is worse than none (SpringKernelResult deliberately
left without param_derivatives until a spring law needs it).

## 4. Semantic boundaries are first-class design artifacts

The hard part of IFT sensitivities was not the linear algebra (one
back-substitution per parameter on the stashed factorization) — it was the
semantic boundary: the channel's fixed-entering-state semantics means later-step
columns are the INCREMENT's sensitivity, not total path sensitivity (M57).
State-transaction timing is physical: the same channel evaluated pre-commit vs
post-commit answers different questions. Such boundaries must be unmistakable in
user-facing docstrings — a user misreading increment for total gets silently
wrong numbers. Name the boundary, pin it with a test (the factor-2 boundary
test), and defer total propagation to an explicit v2 scope.

## 5. Fleet protocol lessons (tower + mission authors)

- Gates are the EXACT CI invocations on whole trees, format --check included.
  "Ruff green" claims covering only `check` on changed files let a format
  violation reach v3 (repaired in 6140cfa; the four invocations are now law).
- Append-only correction of record: mission notes accumulate claim +
  superseding retraction (53→39 shared cells, 9→8 records, 6→5 net-decreasing
  attempts). Never silently edit history; the audit trail is the product.
- Cross-mission coupling is the norm. Sequence merges so a semantic breaker
  carries the re-pin atomically on its own branch (M54 carried M52's near-miss
  vehicle redesign; M55's plasticity flip waited for M54's merge). A red tip
  window is a process failure, not an inevitability.
- Scope widenings are cheap, logged, and expected: request with file-level
  evidence (pyfem/fem/Constrainer.py, bench/v3_pipeline.py), never touch
  silently. A worker that discovers a dead scope glob has found a planning bug.
- Survey-before-build for anything capability-sized (M48 AD, M62/M63 breadth):
  the sketch must be executable by a worker that never reads the legacy source.

## 6. Implications for the remaining program

Ordering constraints discovered (not preferences):

- The production CSR wiring mission is ONE atomic mission: production assembly
  consumes the pattern-cached CSR AND the duplicate pattern compilers
  (driver/plan.py vs fem/assembly.py) consolidate AND driver/plan.py's
  reduceat order canonicalizes (killing the SIMD-dependence finding) — with the
  affected pins re-pinned in the same merge. Splitting it strands either the
  duplication or the platform-nondeterminism.
- Breadth order: cohesive/contact (M62) and ViscoPlasticity/SOVS (M63) surveys
  first; crystal plasticity needs a per-IP typed-status design step before its
  survey is actionable; FE² builds on the sensitivity channel's
  declared-auxiliary-solve ABI (M56 follow-up note) — after the channel has
  consumers, not before.
- The operator-DSL decision stays deferred until sensitivity-modality scoping
  forces it (M22 criteria); do not build it speculatively.

## 7. The deepest lesson

The program's durable product is the instrument suite and the protocol, not
any single merge: parity batteries that detect bugs, cost models that predict
bottlenecks, probes that convict tangents at real states, and a review loop
that reproduces every number independently. The library is what falls out of
running them. When in doubt, invest in the instrument — the code follows.
