# D3-A — generic-spine production cut freeze

Status: **COMPLETE**  
Repository base: `04baa4a79cbcddff888c57b239df5202e2b41424` (`04baa4a`)  
Qualified Phase-1 oracle: `7ee65c3ea5c50278b17dce63f2569e22412e0ebd`  
Accepted simplified proof: `099b51f88022f70ad11c4cecfca35c4a84ecf41e`  
Decision: **replace the public Q8 spine with one linear four-commit series on one
exclusive branch; the first three commits are unexported semantic cuts, and the
fourth is the single atomic public-authority/deletion cut.**

The old 33-path/one-commit R2-F direction remains correct about direct replacement,
no adapter, and no permanent second backend. One commit is no longer the safest unit:
at the current base it would mix a production-grade correction of the simplified
proof, program/coordinate semantics, sparse scheduling, state transactions, fresh
verification, public export changes, and more than six thousand lines of causal
deletion. Those are independently reviewable semantic boundaries. File count alone
is not the reason for splitting them.

No production file was changed for this report. Preflight and final checks used
`core.fsmonitor=false` because the configured fsmonitor socket emitted an
environmental IPC warning; full porcelain was empty.

## 1. Evidence at the current base

The required v3 authority documents, current/frontier execution sections, old R2-F
report, simplified proof, both proof files, and every current production/test path
named by R2-F were read completely.

The exact current facts are:

- `HEAD` is exactly `04baa4a79cbcddff888c57b239df5202e2b41424`, branch `v3`,
  and full porcelain is empty.
- Outside `pyfem/v3/core/generic.py` and
  `test/v3/test_v3_generic_core.py`, there is no production or test diff from
  either `7ee65c3` or `f32e7e5` to the current head. The accepted Phase-1 public
  code and its four focused matrices are therefore byte-identical to the qualified
  evidence surface.
- Phase 1 retains its component-qualified E3 evidence: 86 focused and 348 combined
  tests in normal and 640-digit modes, 478 v3 tests, 667 repository tests, and
  independent physics/balance, state-lifecycle, and public-contract GO reviews.
- S2-A `099b51f` left an unexported 746-physical-line proof module (648 nonblank by
  the current local count; the ledger records 649) and a 406-physical-line/360-
  nonblank test. Its exact independent review returned GO with P0/P1/P2 `0/0/0`,
  fresh Fraction-derived Q4/T3/global checks, split-versus-vectorized assembly,
  seven tests in both digit modes, 485 v3 tests, 674 repository tests, and all
  static/format/ancestry/clean gates.
- The old R2-F one-shot call census remains applicable because the public path is
  unchanged: one model and one program compilation still caused 12 complete-model
  validations, 17 complete-program validations, four plan preparations, eight
  plan validations, 12 Q8-recipe validations, four binding-correspondence passes,
  four assemblies, and three fresh verifications. The target is to retain the four
  numerical responses and three fresh proof points while deleting the repeated
  reconstruction and descriptor replay.

This report uses the Phase-1 results and the exact S2-A GO as evidence. It does not
claim that the isolated proof already supplies production identity, program,
transaction, sparse-plan, or result-verification semantics.

## 2. Exact target and trust boundary

The terminal public route is singular:

```text
ModelSpec
  -> normalize_model_spec exactly once
  -> direct generic CompiledSystem (entities, spaces, system-owned operators)
ProgramSpec + exact live CompiledSystem
  -> normalize_program_spec exactly once
  -> CompiledProgram (coordinate map, program-owned operators)
request + exact live system/program
  -> generic requested-channel schedule
  -> fresh workspace/evaluation
  -> linear solve and transaction
  -> candidate fresh verification
  -> pre-acceptance fresh verification
  -> public Solution fresh verification
```

There is no `CompiledModel -> generic` translation, no backend selector, no import
shim for a deleted predecessor, and no public interval in which two Q8 compilers or
assemblers are selectable.

### 2.1 Compiler boundary

Authored `ModelSpec`/`ProgramSpec`, registry descriptors, and future restored bytes
are untrusted boundary data. The compiler must:

- normalize each authored object once;
- resolve and freeze selected registry implementations once;
- validate binding outputs, exact dtypes, metadata absence, finite values, Q8
  conventions, and scale-relative geometry once;
- canonicalize identities/order and create detached owning read-only arrays;
- retain bounded source/entity attribution, implementation IDs, capability/state
  schemas, content fingerprints, and fresh live instance identities; and
- construct carriers only through compiler-owned constructors/tokens.

A restored compiled artifact is explicitly outside this cut; it would need a new
decoder/rebind boundary. The runtime must not preserve the current hostile-private-
carrier decoder merely in anticipation of that future feature.

### 2.2 Trusted internal execution

Assembly, analysis, state, and results consume compiler-produced objects as trusted
internal meaning. They may and must check:

- exact live system/program/prepared owner identity and compatible fingerprints;
- exact state lineage/generation and transaction/trial registry membership;
- requested channels and operator capabilities;
- numerical shape, finiteness, symmetry, pivot, and capacity preconditions;
- detached ownership at publication boundaries; and
- result/ledger/state internal correspondence.

They must not reconstruct a `ModelSpec`/`ProgramSpec`, rebuild a complete compiled
carrier, re-resolve registry metadata, re-run topology/quadrature recipes, or hash
every nested compiler array before each numerical response.

Fresh verification remains mathematically independent of retained solve state: it
creates a new generic schedule, new value workspace, new operator response, and new
constraint/balance/convergence evidence. It may share the trusted compiled
operator implementation and neutral scalar numerical primitives, but it may not
read or reuse the solve factor, solve projection, cached response, or solve
workspace. Literal Q8, rational global, and PatchTest8 oracles remain independent of
production-generated expected values.

## 3. Semantic invariants frozen for the cut

1. **Multiple spaces are structural now.** `CompiledSystem` owns an ordered tuple
   of `DiscreteSpace`; each port binds a stable `space_id` and native local
   coefficient map. There is no singular `CompiledSystem.space`, no rectangular
   cross-space connectivity, and no absent-field sentinel.
2. **The operator type is open.** The system stores one non-enumerating compiled-
   operator protocol/header. Production must not reproduce `AnyOperatorBlock =
   Continuum | Spring | PointLoad`, dispatch on payload class/name, or import a
   concrete Q8 payload outside the continuum builder/evaluator.
3. **Ownership is explicit.** System-owned operators and program-owned operators
   remain in distinct immutable owners. Preparation composes their schedules; it
   does not move loads into the system or copy system operators into the program.
4. **Coordinates are kinematics, not loads.** `CompiledProgram` owns a compiled
   affine coordinate map `x = P q + x_bar(p)` plus coordinate derivatives and the
   minimal compiled constraint-row/source evidence needed for violation and
   reaction checks. Prescribed values and affine ties are not residual operators.
5. **Program loads are operators.** Each authored load row remains separately
   attributable even when numerical values coalesce at one coefficient. Load
   amplitudes are evaluated from compiled affine coordinate data without rebuilding
   a `ProgramSpec` or `ProgramMeaningWitness`.
6. **Channels remain separate until the ledger.** Internal force/tangent, external
   force, affine-offset force, and requested derivatives retain typed channel and
   balance roles. A hard-coded sign multiplier must not collapse all contributions
   into one residual before attribution, reduction, reactions, work, and balance
   are formed.
7. **State has a schema, even when empty.** An `OperatorStateLayout` records stable
   block identity, named slot layout, entity offsets, dtype/shape, and accepted/
   trial lifetime. Q8 is a genuine zero-width instance. A lone
   `local_state_width: int` and material/formulation history split are not the
   production contract. This cut does not claim nonzero-history Proof C.
8. **Schedules are generic and structure-only.** Raw/canonical full and reduced COO
   plans, vector plans, native gathers, channel attribution, and capacity checks are
   derived once from ports/channels. Repeated evaluation replaces values only and
   publishes detached storage.
9. **Q8 knowledge is localized.** Registry keys, shape/gradient/quadrature rules,
   plane-stress payload construction, scale-relative geometry, and the Q8 evaluator
   live in `compile/continuum.py` (and its tests). `api.py`, analysis, results, and
   generic schedule code contain no topology/formulation/material/Q8 branch.
10. **Identity and ordering survive.** Canonically equivalent authored order yields
    equal content fingerprints but fresh live identities. Foreign content-equal
    owners are rejected. Operator/entity/channel/source term order is stable.
11. **The transaction proof survives unchanged in meaning.** Initialize, begin,
    evaluate, discard/abandon, exactly-once accept, sibling/stale rejection, atomic
    races, successor generation, cache isolation, and accepted-storage separation
    remain public behavior.
12. **All three fresh proof points survive.** Candidate evaluation, acceptance, and
    `Solution.verify()` each recompute using a fresh schedule/workspace; record-only
    verification remains distinct.

## 4. Conceptual audit of the six-concept proof

The simplified proof was accepted for its bounded purpose. The following are
production migration gaps, not findings against its exact reviewer GO.

| Gap in `core/generic.py` | Concrete consequence | Required production correction | Cut status |
|---|---|---|---|
| `AnyOperatorBlock` is a closed union of three payloads. | Adding a fourth family changes the supposedly generic core and invites class/name dispatch. | Non-enumerating operator header/protocol; payload and evaluator stay builder-owned. | Must be fixed in G1; stop otherwise. |
| `CompiledSystem.space`, `CompiledProgram.compatible_space`, and `PreparedExecution.space` are singular. | The current code cannot represent mechanical/thermal/coupled Proof B. | Ordered `spaces`, `space_id` ports, native per-space ranges/maps, and multi-space schedule checks. | Must be fixed in G1/G2. |
| The seven proof tests exercise one displacement space only. | The ledger's earlier Proof-B assumption is not executable evidence at `099b51f`. | Add mechanical-only, thermal-only, coupled, and nonstandard-component-order structural/allocation tests. | Concrete proof gap; not a Q8-cut blocker if closed before G2. |
| Program point loads contain fixed magnitudes; there is no coordinate map. | Phase-1 affine constraints, derivatives, and program coordinates cannot migrate honestly. | Compile affine coordinate map separately; compile load amplitudes as program-owned operators; retain constraint/source rows. | Must be fixed in G2. |
| `local_state_width` is only an integer and evaluators accept no accepted/trial state. | Future state either reintroduces material/formulation branches or changes every operator call. | `OperatorStateLayout` and evaluation context/result state channels; Q8 uses zero-width layout. | Schema fixed by G1/G3; nonzero state deferred. |
| There is no observation contract. | The proof does not demonstrate the sixth concept or result projection. | Keep observations solve-free and separate from operators. Phase-1 Q8 exposes no authored observation, so do not invent an empty backend; retain a results-side extension seam and defer the first concrete observation. | Explicit defer, not a reason to make `generic.py` larger. |
| `OperatorBlock` embeds a raw callable and has no implementation/fingerprint/capability identity. | A production schedule cannot prove which numerical implementation or state/channel schema it executes. | Compiler-bound implementation record with stable implementation ID, manifest, channels, state schema, and trusted evaluator binding. | Must be fixed in G1. |
| `_operator_block` derives channel IDs/signs from `BalanceRole`. | Attribution can be correct while external/internal/offset meanings are collapsed too early. | Explicit channel descriptors and ledger-side sign convention. | Must be fixed in G1/G2. |
| `assemble` allocates a dense global Jacobian and loops over one coefficient vector. | It discards the proven raw/canonical/reduced COO plan, capacity, duplicate, and multiple-space semantics. | Port-derived sparse schedule with requested channels and fresh values. | Must be fixed in G2. |
| Continuum compilation calls opaque `np.linalg.det`/`inv` and emits simple unbounded messages. | Blind transplantation would weaken the qualified scale/translation/orientation and bounded-diagnostic evidence. | Port the Phase-1 explicit 2x2, relative-singularity, source-bounded compiler oracles; do not use the proof geometry routine as production authority. | Must be fixed in G1. |
| The proof has no live/content identity, registry capture, transaction, sparse plan, or fresh result proof. | Passing its seven tests cannot authorize the public cut. | Preserve the exact Phase-1 identity/provenance/state/result evidence and add direct generic-flow oracles. | Closed only by G1-G4 plus four reviews. |

The module boundary therefore is deliberate:

- `model/operator.py`: entity/space ports, channel and operator-state schemas, open
  compiled-operator header;
- `model/system.py`: entities, ordered spaces, system identity/provenance and
  system-owned operators;
- `model/program.py`: coordinate map, compiled program identity and program-owned
  operators;
- `compile/continuum.py`: Q8-specific payload, registry binding and evaluator;
- `compile/system.py` and `compile/program.py`: authored-to-compiled boundaries;
- `assembly/*`: generic schedules and value evaluation;
- `model/state.py`, `analysis/*`, and `results/*`: lifecycle, solve, ledger,
  observation extension seam, and fresh verification.

`core/generic.py` is a proof seed, not the destination. It is deleted at the
terminal cut so it cannot become a permanent second implementation or god module.
The future observation law is already bounded: an observation consumes an accepted
state/result plus declared space/channel IDs, performs no solve or nested execution,
and publishes detached values. A first observation can therefore be added on the
results side without changing the operator, coordinate-map, schedule, or transaction
concepts; this Q8 cut does not pretend an empty tuple is executable Proof D.

## 5. Exact terminal ownership at `04baa4a`

### 5.1 Add — 6 paths

1. `pyfem/v3/model/operator.py`
2. `pyfem/v3/model/system.py`
3. `pyfem/v3/compile/system.py`
4. `pyfem/v3/compile/continuum.py`
5. `pyfem/v3/numerics.py`
6. `test/v3/test_v3_q8_generic_flow.py`

### 5.2 Rewrite in place — 18 production paths

1. `pyfem/v3/spec/diagnostics.py`
2. `pyfem/v3/api.py`
3. `pyfem/v3/model/__init__.py`
4. `pyfem/v3/model/program.py`
5. `pyfem/v3/model/state.py`
6. `pyfem/v3/compile/__init__.py`
7. `pyfem/v3/compile/program.py`
8. `pyfem/v3/assembly/__init__.py`
9. `pyfem/v3/assembly/contracts.py`
10. `pyfem/v3/assembly/prepare.py`
11. `pyfem/v3/assembly/reference.py`
12. `pyfem/v3/analysis/__init__.py`
13. `pyfem/v3/analysis/contracts.py`
14. `pyfem/v3/analysis/linear.py`
15. `pyfem/v3/results/__init__.py`
16. `pyfem/v3/results/contracts.py`
17. `pyfem/v3/results/solution.py`
18. `pyfem/v3/results/verification.py`

`assembly/reference.py` may retain its filename and public
`assemble_reference_linear` name, but after G4 its implementation is the one generic
channel evaluator; “reference” must not mean “legacy backend.” `compile_model` may
retain its public name but must be the direct function from `compile/system.py`, not
a selector or wrapper over a predecessor.

### 5.3 Rewrite in place — 5 focused-test paths

1. `test/v3/test_v3_generic_core.py`
2. `test/v3/test_v3_model_compile.py`
3. `test/v3/test_v3_program_compile.py`
4. `test/v3/test_v3_assembly_plan.py`
5. `test/v3/test_v3_linear_analysis.py`

`test_v3_generic_core.py` is rewritten to import the production operator/system/
schedule contracts. It retains the unlike-operator and independent Fraction oracles
and adds the missing multiple-space Proof-B structural cases; it no longer imports
`pyfem.v3.core.generic`.

### 5.4 Delete causally in G4 — 6 production paths

1. `pyfem/v3/model/compiled.py`
2. `pyfem/v3/compile/contracts.py`
3. `pyfem/v3/compile/model.py`
4. `pyfem/v3/analysis/numerics.py`
5. `pyfem/v3/core/generic.py`
6. `pyfem/v3/core/__init__.py`

The first four are the predecessor Q8 carrier/compiler/private-numerics spine. The
last two remove the isolated proof implementation after its laws/oracles have moved
to production contracts/tests. None may remain as an import shim, alias module, or
backend flag.

Total terminal owned set: **29 production paths and 6 test paths = 35 paths**. The
increase from R2-F's old 33 is entirely current-base accounting: the generic test now
exists and is rewritten rather than added, while both proof-package files must now
be deleted causally.

### 5.5 Retain unchanged dependencies — exact paths

- `pyfem/v3/spec/model.py`
- `pyfem/v3/spec/normalize.py`
- `pyfem/v3/spec/program.py`
- `pyfem/v3/spec/normalize_program.py`
- `pyfem/v3/spec/program_diagnostics.py`
- `pyfem/v3/model/arrays.py`
- `pyfem/v3/model/identity.py`
- `pyfem/v3/model/provenance.py`
- `pyfem/v3/model/registry.py`
- `pyfem/v3/compile/diagnostics.py`
- `pyfem/v3/compile/program_diagnostics.py`
- `pyfem/v3/assembly/diagnostics.py`
- `pyfem/v3/analysis/diagnostics.py`
- `pyfem/v3/results/diagnostics.py`

If the writer finds one of these must change, it is an interface finding and stops
for refreeze; it is not silently added to ownership.

### 5.6 Explicit defer/exclusion

Existing paths excluded from this cut are `pyfem/v3/__init__.py`,
`pyfem/v3/_prototype_assembly.py`, `pyfem/v3/loader.py`, `pyfem/v3/pack.py`,
`pyfem/v3/registry.py`, `pyfem/v3/types.py`, and every current path under
`pyfem/v3/elements/`, `pyfem/v3/fem/`, `pyfem/v3/io/`,
`pyfem/v3/materials/`, `pyfem/v3/mesh/`, and `pyfem/v3/solver/`. Root/CLI/TOML/
legacy adapters, legacy parity, and `_prototype_assembly.py` retirement remain later
packets.

Also deferred, with **no path allocation in this card**, are:

- persisted compiled-system/checkpoint decode, validation, and registry rebind;
- the first concrete solve-free observation family and observation persistence;
- nonzero operator history, nonlinear iteration/cutback, and Proof C;
- Q4/T3/spring production exposure, Proof D breadth, alternate sparse/performance
  backends, RVE auxiliary execution, ROM artifacts, and ecosystem adapters.

All repository paths not listed in 5.1-5.4 are forbidden to the writer. “Deferred”
does not authorize creating a guessed future module name.

## 6. Four-commit dependency and integration sequence

All commits are a linear series by one owner on one exclusive branch rooted exactly
at `04baa4a`. G1-G3 are not integrated independently. They may temporarily create a
second **unexported** direct implementation on that private branch, but they may not
add an adapter, public selector, union-typed public API, or alternate export.

### G1 — generic semantic carriers and direct Q8 system compiler

Exact paths:

- add `pyfem/v3/model/operator.py`;
- add `pyfem/v3/model/system.py`;
- add `pyfem/v3/compile/system.py`;
- add `pyfem/v3/compile/continuum.py`;
- rewrite `pyfem/v3/spec/diagnostics.py`; and
- add `test/v3/test_v3_q8_generic_flow.py`.

Outcome: a direct, unexported `ModelSpec -> CompiledSystem` Q8 compiler with ordered
spaces, open operator header, typed ports/channels/state layout, selected-registry
capture, bounded diagnostics, identities/provenance, owned arrays, and the qualified
Q8 compiler/geometry oracles. It accepts no `CompiledModel` and performs no
assembly/solve.

### G2 — coordinate map, program-owned loads, and generic schedule/evaluator

Exact paths:

- rewrite `pyfem/v3/model/program.py`;
- rewrite `pyfem/v3/compile/program.py`;
- rewrite `pyfem/v3/assembly/contracts.py`;
- rewrite `pyfem/v3/assembly/prepare.py`;
- rewrite `pyfem/v3/assembly/reference.py`;
- rewrite `test/v3/test_v3_program_compile.py`;
- rewrite `test/v3/test_v3_assembly_plan.py`; and
- extend `test/v3/test_v3_q8_generic_flow.py`.

Outcome: direct `ProgramSpec -> CompiledProgram`, an affine coordinate map with
derivatives/constraint evidence, separately owned load operators, and a requested-
channel raw/canonical/reduced schedule. The existing public exports still point only
to the old path on the private branch; tests call the unexported direct path. No old
carrier is input to the new path.

### G3 — state transaction, solve, ledger, and fresh verification

Exact paths:

- add `pyfem/v3/numerics.py`;
- rewrite `pyfem/v3/model/state.py`;
- rewrite `pyfem/v3/analysis/contracts.py`;
- rewrite `pyfem/v3/analysis/linear.py`;
- rewrite `pyfem/v3/results/contracts.py`;
- rewrite `pyfem/v3/results/solution.py`;
- rewrite `pyfem/v3/results/verification.py`;
- rewrite `test/v3/test_v3_linear_analysis.py`; and
- extend `test/v3/test_v3_q8_generic_flow.py`.

Outcome: a private direct generic Q8 compile/assemble/solve/verify flow retaining the
exact transaction, state, balance, bounded-solver, storage, and three fresh-
verification semantics. The pure Cholesky/ratio primitives move unchanged to the
neutral module; result verification has no private result-to-analysis import.

### G4 — atomic public authority switch, proof migration, and causal deletion

Exact G4 touch set (29 paths; additions from G1/G3 are already present):

- rewrite `pyfem/v3/api.py`;
- rewrite `pyfem/v3/model/__init__.py`;
- rewrite `pyfem/v3/model/program.py`;
- rewrite `pyfem/v3/model/state.py`;
- rewrite `pyfem/v3/compile/__init__.py`;
- rewrite `pyfem/v3/compile/program.py`;
- rewrite `pyfem/v3/assembly/__init__.py`;
- rewrite `pyfem/v3/assembly/contracts.py`;
- rewrite `pyfem/v3/assembly/prepare.py`;
- rewrite `pyfem/v3/assembly/reference.py`;
- rewrite `pyfem/v3/analysis/__init__.py`;
- rewrite `pyfem/v3/analysis/contracts.py`;
- rewrite `pyfem/v3/analysis/linear.py`;
- rewrite `pyfem/v3/results/__init__.py`;
- rewrite `pyfem/v3/results/contracts.py`;
- rewrite `pyfem/v3/results/solution.py`;
- rewrite `pyfem/v3/results/verification.py`;
- rewrite all six owned focused tests; and
- delete the six section-5.4 production paths.

G4:

- routes `api.py` and the five package `__init__.py` files directly to the generic
  implementations;
- removes the temporarily coexisting old implementations from the rewritten
  program/assembly/analysis/result modules;
- completes the five focused-test rewrites and production-contract rewrite of
  `test_v3_generic_core.py`;
- deletes the six predecessor/proof paths; and
- contains no new numerical or lifecycle semantics beyond G1-G3.

Only after G4 does the branch expose the replacement. I0 integrates the exact four-
commit series serially into an isolated integration worktree. It does not squash
away semantic review boundaries and does not merge an intermediate tip. Four
independent reviewers inspect the exact terminal tip and the G1-G4 range. An
accepted finding returns to the same owner as one direct-child repair commit; every
affected reviewer and gate reruns. No next packet overlaps the 35 paths until all
findings are adjudicated.

## 7. Exact first writer packet

Freeze **G1 only** as the next executable writer card.

Parent/base: `04baa4a79cbcddff888c57b239df5202e2b41424`  
Owned paths: the six G1 paths above  
Commit count: exactly one direct-child source commit  
Public export changes: zero  
Predecessor deletions: zero  
Production physical-line cap: 1,800 added/rewritten lines across the five production
paths  
Test physical-line cap: 650 lines in the new test

The new test must contain these named cases (names are frozen; parametrization is
allowed):

1. `test_direct_q8_system_compilation_uses_generic_spaces_ports_and_channels`
2. `test_q8_compiler_matches_hard_coded_shape_gradient_and_quadrature_oracle`
3. `test_multiple_spaces_have_disjoint_native_coefficient_maps`
4. `test_compiled_system_owns_metadata_free_arrays_identity_and_attribution`
5. `test_q8_geometry_classification_is_scale_and_translation_stable`
6. `test_q8_invalid_orientation_and_relative_singularity_fail_at_compile_boundary`

The hard-coded convention expected values must not be imported or manufactured by
the production builder. The multiple-space case must include mechanical-only,
thermal-only, coupled, and nonstandard component ordering; it proves carrier/allocation
structure only, not thermal physics. `compile/system.py` may be imported by the new
test directly but is not re-exported from `compile/__init__.py` in G1.

G1 stops without commit if it needs a closed concrete payload union, a singular
system space, a `CompiledModel` input, an adapter, a downstream Q8 branch, a private
cross-layer diagnostic import, a change to a retained dependency, or a broader path.
It also stops if the Phase-1 explicit geometry/registry/source/ownership evidence
cannot fit this direct compiler boundary within the cap; the cap is adjudicated,
not evaded by moving code into a dependency.

## 8. Required independent reviewers

All four are read-only, exact-terminal-tip reviews. All must return GO with no
accepted P0/P1/P2 finding before integration completes.

1. **Finite-element numerical/balance reviewer.** Re-derive the Q8 shape/gradient/
   quadrature convention, literal `K = M/360`, affine reduction, exact rational
   displacement, reactions, force/moment/work balance, two-cell attribution, scale/
   translation/orientation classification, and five-cell PatchTest8 independently
   of production expected-value helpers.
2. **State/transaction/ownership reviewer.** Audit live/content identities,
   operator state layout, initialize/begin/evaluate/discard/accept state machine,
   exact generations, sibling/double/race behavior, zero-free path, capacity/pivot,
   cache ownership, and disjoint immutable published storage.
3. **Result/fresh-verification reviewer.** Prove all three fresh calls create new
   schedules/workspaces/responses; poison the solve factor/cache; test coherent
   field, coordinate, constraint, load, derivative, balance, work, and convergence
   perturbations; verify record-vs-fresh separation; confirm no solver factor or
   retained response is an oracle.
4. **Architecture/trust/deletion reviewer.** Audit the six-concept boundaries,
   open operator type, multiple spaces, system/program ownership, coordinate map,
   zero-width future-state schema, observation defer, import/branch/call census,
   normalizer counts, exact path/deletion set, no adapter/backend/shim, and line
   budgets.

Production code is not its own mathematical oracle. Passing the migrated test suite
without the independent literal/Fraction/PatchTest checks is insufficient.

## 9. Named test and oracle gates

### 9.1 Existing cases that must retain their meaning

Compiler/program/assembly anchors:

- `test_shipped_q8_binding_matches_hard_coded_local_convention_oracle`
- `test_reference_geometry_is_valid_under_extreme_uniform_scaling`
- `test_reference_geometry_classification_is_translation_invariant`
- `test_inverted_and_sign_changing_reference_geometry_are_rejected`
- `test_horizon_chain_and_additive_load_oracle_is_literal`
- `test_chain_through_prescribed_root_composes_negative_factor_and_derivative`
- `test_literal_q8_operator_and_affine_rhs_oracles`
- `test_two_cell_raw_attribution_and_canonical_duplicate_coalescing`
- `test_repeated_evaluation_reuses_topology_but_all_values_are_fresh`

Solve/balance/state/result anchors:

- `test_rational_one_cell_solution_ledgers_and_rigid_derivative`
- `test_additive_and_constrained_dof_loads_keep_direct_reaction_semantics`
- `test_affine_mpc_chain_with_nonzero_offsets_solves_and_verifies`
- `test_factorization_reuse_and_all_published_storage_is_disjoint`
- `test_exactly_once_acceptance_siblings_double_accept_discard_and_forgery`
- `test_discard_and_accept_ordering_is_atomic`
- `test_accept_wins_racing_discard_and_only_one_concurrent_discard_succeeds`
- `test_fresh_verification_recomputes_complete_backend_evidence`
- `test_fresh_convergence_residual_matches_rebuilt_ledger_exactly`
- `test_coherent_perturbed_field_passes_record_but_fails_fresh_exactly`
- `test_tiny_scale_wrong_field_has_no_unit_floor`
- `test_fresh_verification_catches_constraint_and_coherent_balance_changes`
- `test_poisoned_solve_cache_cannot_affect_fresh_solution_verification`
- `test_patch_test8_analytical_field_is_primary_new_flow_oracle`
- `test_one_shot_uses_same_verified_transition_path`
- `test_huge_exact_constraint_id_remains_bounded_under_digit_limit`

Generic structural anchors:

- `test_three_unlike_blocks_share_typed_boundary_without_padding`
- `test_exact_constrained_solution_and_attributed_balance`
- `test_block_order_is_canonical_and_attribution_is_stable`
- `test_second_continuum_topology_changes_descriptor_instance_only`
- the six new G1 cases; and
- new terminal cases
  `test_public_q8_generic_flow_matches_literal_operator_reduction_and_solution`
  and
  `test_each_fresh_verification_owns_a_new_schedule_response_and_workspace`.

Names may remain in their current files or move only among the six owned test files.
A terminal case-by-case ledger must classify all current 138 Phase-1 test functions
as retained, rewritten/re-homed, or retired-with-reason.

The hostile scalar/tuple/UUID/partially initialized exact-private-carrier matrices,
`ProgramMeaningWitness` correspondence tests, reidentified complete compiled-
carrier validators, malformed private plan reconstruction, and assembly-time
descriptor replay tests are not public mathematical evidence. They are retired or
re-homed to the authored compiler boundary as appropriate. Public result corruption,
live identity, finite numeric inputs, publication ownership, and all mathematical/
transaction/fresh-verification cases remain.

### 9.2 Commands

Run the focused matrix in both modes with the shared project environment and no
pytest cache writes:

```text
PYTHONDONTWRITEBYTECODE=1 /Users/sora/Projects/python/PyFEM/.venv/bin/python \
  -m pytest -q -p no:cacheprovider \
  test/v3/test_v3_model_spec.py \
  test/v3/test_v3_model_identity.py \
  test/v3/test_v3_model_compile.py \
  test/v3/test_v3_program_compile.py \
  test/v3/test_v3_assembly_plan.py \
  test/v3/test_v3_linear_analysis.py \
  test/v3/test_v3_generic_core.py \
  test/v3/test_v3_q8_generic_flow.py

PYTHONINTMAXSTRDIGITS=640 PYTHONDONTWRITEBYTECODE=1 \
  /Users/sora/Projects/python/PyFEM/.venv/bin/python \
  -m pytest -q -p no:cacheprovider <the same eight paths>

PYTHONDONTWRITEBYTECODE=1 /Users/sora/Projects/python/PyFEM/.venv/bin/python \
  -m pytest -q -p no:cacheprovider test/v3

PYTHONINTMAXSTRDIGITS=640 PYTHONDONTWRITEBYTECODE=1 \
  /Users/sora/Projects/python/PyFEM/.venv/bin/python \
  -m pytest -q -p no:cacheprovider test/v3

PYTHONDONTWRITEBYTECODE=1 /Users/sora/Projects/python/PyFEM/.venv/bin/python \
  -m pytest -q -p no:cacheprovider

/Users/sora/Projects/python/PyFEM/.venv/bin/ruff check pyfem/v3 test/v3
/Users/sora/Projects/python/PyFEM/.venv/bin/ruff format --check pyfem/v3 test/v3
```

The numerical reviewer separately runs the literal/Fraction Q8, split-versus-
vectorized schedule, rational global, and independently assembled five-cell
PatchTest8 oracles in normal and 640-digit modes.

## 10. Static deletion, import, branch, and call-census gates

### 10.1 Exact deletion/private-import gates

- all six section-5.4 paths are absent (`test ! -e` for each);
- zero imports or references to
  `model.compiled`, `compile.contracts`, `compile.model`,
  `analysis.numerics`, or `core.generic`;
- zero production symbols/references named `CompiledModel`, `DomainBlock`,
  `ProgramMeaningWitness`, `_program_spec_from_witness`, `_validated_model`,
  `_validated_program`, `_validated_prepared_plan`, `AnyOperatorBlock`, or
  `compatible_space`;
- zero private imported names across distinct `spec`, `model`, `compile`,
  `assembly`, `analysis`, and `results` layers, verified by AST rather than only a
  text grep; and
- exactly one definition/export owner for each of `compile_model`,
  `compile_program`, `prepare_assembly_plan`, `assemble_reference_linear`,
  `prepare_analysis`, `solve`, and `fresh_verify`.

### 10.2 Semantic branch gates

- `api.py`, `analysis/`, and `results/` contain zero Q8/topology/formulation/
  material/descriptor-key dispatch;
- generic assembly contains zero Q8/topology/payload-type branch and imports no
  `compile.continuum` concrete payload;
- concrete Q8 keys/payload/evaluator references are confined to
  `compile/continuum.py`, `compile/system.py` builder selection, and owned tests;
- the compiled-system operator tuple is not a union enumerating implementations;
- `CompiledSystem` has plural spaces and no singular-space compatibility field;
- no absent-field sentinel, padded/rectangular mixed-space connectivity, backend
  selector, adapter, or import shim exists; and
- `compile/continuum.py` uses the qualified explicit scale-relative 2x2 geometry
  path, not blind `np.linalg.det`/`np.linalg.inv` transplantation from the proof.

### 10.3 One-shot dynamic call census

Instrument the public one-cell rational solve without changing production and
require:

- `normalize_model_spec == 1` and `normalize_program_spec == 1`;
- one direct model/system compiler and one direct program compiler;
- registry descriptor resolution and topology/quadrature recipe construction only
  during system compilation;
- four generic schedule preparations and four fresh value evaluations for the
  solve/candidate/accept/public sequence, with four distinct schedule, workspace,
  response, and value-array identities;
- `fresh_verify == 3`;
- zero authored-spec reconstruction, complete compiled-carrier reconstruction,
  descriptor replay after compilation, and witness correspondence pass; and
- zero reads of the prepared analysis factor/cache/workspace from result
  verification.

An exact count may change only if the public verification contract changes and is
separately adjudicated. Reducing the three fresh proof points is an automatic stop,
not an optimization.

## 11. Updated physical-line and deletion budgets

At `04baa4a`, the 24 existing production files in the terminal ownership set contain
13,836 physical lines (12,908 nonblank). The five existing focused files contain
6,729 physical lines (6,064 nonblank). The new production/test files do not yet
exist.

The production gross-retirement floor is derived from R2-F's measured predecessor
surfaces plus the now-integrated simplified proof:

- 1,624 lines in the old Q8 carrier/compiler files;
- about 2,086 lines of complete-model/program reconstruction, meaning-witness
  reconstruction, and repeated normalization in `compile/program.py`;
- about 1,494 lines of Q8 descriptor/recipe and complete-plan reconstruction in
  `assembly/prepare.py`;
- about 128 lines of duplicate evaluation reconstruction in assembly;
- about 72 lines of result-side duplicate reconstruction;
- 72 lines in the old analysis-private numerics module; and
- 747 physical lines in `core/generic.py` plus its package initializer.

Terminal gates:

- **at least 6,200 gross production lines deleted or mechanically retired**;
- **no more than 4,200 gross replacement production lines added**, including moved
  proof/numerical logic;
- **at least 2,000 net production physical lines removed** from the 13,836-line
  current owned baseline;
- **at most 11,836 physical production lines** across the corresponding terminal
  owned files;
- **at least 1,000 focused-test lines retired/replaced** whose only contract was
  recursive private-carrier/descriptor/plan decoding;
- **no more than 650 lines in the new generic-Q8 flow test** and at most 6,400
  physical lines across the six terminal focused proof files; and
- no production module above 1,200 physical lines without explicit architecture
  adjudication. `compile/program.py`, `assembly/prepare.py`, and
  `results/verification.py` are expected to fall below that ceiling, not merely
  move their validators elsewhere.

The arithmetic is a guardrail, not an objective function. A smaller diff that keeps
two authorities fails. A coherent implementation that exceeds a cap stops for an
evidence-backed refreeze; it does not compress names, hide code in tests, or move
logic to an unowned dependency.

## 12. Rollback and stop conditions

Rollback point is exact clean `04baa4a`; the independent numerical oracle remains
`7ee65c3` and the accepted generic proof remains `099b51f` in history.

Before G4, failure means abandon the exclusive branch series; no private generic
intermediate is integrated. After the full series has been placed only in an
isolated integration worktree, a failed terminal gate removes/reverts the whole
G1-G4 range, not just G4, so an unexported second spine cannot remain. If already
shared, rollback uses explicit revert commits; no destructive reset/rebase.

Stop and return a concrete blocker if any of these occurs:

- base/porcelain/ancestry/path ownership differs;
- a retained unchanged dependency must change;
- the open operator boundary requires a concrete payload union or downstream type-
  name/topology branch;
- multiple spaces require padding, sentinels, or a singular compatibility owner;
- direct program compilation cannot retain affine constraint/derivative/source
  evidence without `ProgramMeaningWitness` or authored reconstruction;
- program loads cannot remain program-owned through schedule attribution;
- zero-width Q8 cannot inhabit a credible operator-state layout without changing
  transaction semantics;
- fresh verification needs a solve cache/factor/response, loses one of its three
  proof points, or cannot reproduce constraint/balance/convergence evidence;
- an independent oracle changes to use production-generated expected values;
- any causal deletion, no-adapter, no-shim, import, branch, call-census, line-budget,
  or normal/restricted-digit gate fails; or
- any reviewer returns an unadjudicated P0/P1/P2 finding.

## 13. Frozen outcome

The lifecycle-safe minimum is not a monolithic 35-path commit and not an adapter.
It is one exclusive four-commit semantic series whose first three tips are private
and whose final tip makes one atomic public authority change and deletes both the
Phase-1 predecessor spine and the isolated proof implementation. The first executable
packet is G1's six paths only. It is large enough to establish the correct open,
multiple-space, identity-bearing production carrier and direct Q8 compiler, while
small enough for independent numerical and architecture review before program,
schedule, state, or public behavior moves.

RESULT COMPLETE  
Base: `04baa4a79cbcddff888c57b239df5202e2b41424`  
Decision: four linear commits on one exclusive branch; G4 alone switches public
authority and deletes all predecessor/proof implementations  
First packet: G1 direct unexported Q8 `ModelSpec -> CompiledSystem`, exactly six
paths, no public export/deletion/adapter  
Blocker: none to freeze; the concrete proof gaps above are mandatory G1/G2 stop
conditions  
Next: authorize only G1 from exact clean `04baa4a`, then jointly adjudicate its
numeric and architecture reviews before G2
