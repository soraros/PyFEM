# PyFEM v3 agent-neutral handoff

- Snapshot date: 2026-09-01 (Asia/Shanghai)
- Repository: `/Users/sora/Projects/python/PyFEM`
- Purpose: cold-start handoff for any coding agent without access to prior chat
- Live authority: [migration-execution.md](migration-execution.md)

This is a concise projection of the authoritative repository documents, not a
second execution ledger. If Git state or the live ledger differs from this
snapshot, stop and reconcile the difference before editing source.

## 1. Authority order

Read and obey these in order:

1. [design.md](design.md) — ownership, identity, state, contribution, and result
   invariants.
2. [generic_core.md](generic_core.md) — post-Phase-1 semantic IR and the decision
   to treat legacy cases as compositions rather than a taxonomy to reproduce.
3. [migration-execution.md](migration-execution.md) — sole live state, blockers,
   commits, and exact next action.
4. [evidence/2026-08-16-d3a-generic-spine-cut-freeze.md](evidence/2026-08-16-d3a-generic-spine-cut-freeze.md)
   — frozen G1-G4 replacement ownership and atomic-integration rule.
5. [migration_workflow.md](migration_workflow.md) and
   [AGENTS.md](AGENTS.md) — packet lifecycle, independent review, callbacks, and
   working rules.

The former prototype architecture snapshot, P0-P8 roadmap, old status matrix, and
Cursor-era execution prompts were removed from the durable tree after their useful
content was absorbed; they remain available through Git history. Old P2-A through
P2-G plans remain historical evidence inside the live ledger. None is executable
authority.

## 2. Authoritative overall goal

Build PyFEM v3 as a compiled, data-oriented finite-element library in the spirit
of `absim_fvm`, while preserving scientifically meaningful v1 behavior.

The intended flow is:

```text
authored declarations
  -> normalized exact specifications
  -> immutable CompiledSystem + CompiledProgram
  -> PreparedExecution with sparse/gather/reduction plans
  -> explicit accepted/trial state transaction
  -> analysis
  -> immutable result with provenance and fresh verification
```

The stable numerical IR is entity blocks, discrete spaces, bound operator blocks,
typed contribution channels, coordinate maps, and observations. Legacy element,
material, section, model-action, load, solver, and writer classes are requirements
and authoring evidence; their class boundaries are not architectural constraints.

Non-negotiable properties:

- normalize external meaning once at its authority boundary;
- compiler-owned arrays are detached, owning, read-only, and metadata-free;
- semantic IDs are independent of block order and dense execution indices;
- numeric kernels consume plain homogeneous arrays, not models or registries;
- compatible contributions add and remain attributable;
- accepted state, trial state, workspace, and published results have separate
  owners and lifetimes;
- failed evaluation or solve changes no accepted state;
- one valid acceptance commits one exact successor generation atomically;
- results retain provenance and use genuinely fresh verification where promised;
- broad green tests do not justify duplicate owners, adapters, or private
  revalidation at every layer.

## 3. Confirmed Git state at this handoff

State inspected locally and against `origin` on 2026-09-01. The exact
product/source baseline is `5287d20f96536f88b744d5715b27c46913926d14`.
This handoff and its documentation-only consolidation descend from that baseline
on `v3`. Inspect `git rev-parse HEAD` and the file history rather than treating the
baseline hash as the current local `v3` tip.

| Ref | Exact commit | Confirmed meaning |
|---|---|---|
| handoff parent; `origin/v3` at snapshot | `5287d20f96536f88b744d5715b27c46913926d14` | Clean product/source baseline; documentation records the repaired G1 review gate; no G1 source is integrated |
| `agnet/g1-generic-system`, remote peer | `2f6f972eef5a4c5f633478b1effecba834ca5fd4` | Pushed, isolated G1 candidate; five-commit source/repair chain rooted at `04baa4a` |
| `agnet/g1-integration-proof` | `72c48784b8a3718ed40b15c83dd60f0443278371` | Local-only conflict-free cherry-pick proof; not product authority and not to be pushed or merged by implication |
| integrated Phase-1 oracle | `7ee65c3ea5c50278b17dce63f2569e22412e0ebd` | Ancestor of `v3`; completed P1-C/I1 public linear reference |

The G1 source chain is:

```text
04baa4a
  -> f342b40
  -> 182bb3e
  -> 7929252
  -> 942a212
  -> 2f6f972
```

Its owned paths are exactly:

- `pyfem/v3/model/operator.py`
- `pyfem/v3/model/system.py`
- `pyfem/v3/compile/system.py`
- `pyfem/v3/compile/continuum.py`
- `pyfem/v3/spec/diagnostics.py`
- `test/v3/test_v3_q8_generic_flow.py`

Historical P1-C source branches and other migration branches remain local audit
anchors. The `update-install-script` stash is unrelated user work. Do not delete,
apply, or rewrite either category during v3 migration work.

## 4. Completed and independently verified work

### Foundation and bounded Phase 1

The immutable specification, identity/provenance, registry snapshot, Q8 compiled
model, program, assembly-plan, and verified linear-flow packets are complete. The
final P1-C chain was integrated as:

```text
c9b9480 -> bfc1d80 -> d78201d -> 7ee65c3
```

Accepted Phase-1 evidence at `7ee65c3`:

- 86 focused tests in normal and 640-digit modes;
- 348 combined Phase-1 tests in both modes;
- 478 v3 tests and 667 repository tests;
- independent physics/balance, state-lifecycle, and public-contract GO reviews;
- exact transaction, reaction, work, storage, cache, and fresh-verification
  checks.

Phase 1 is a frozen executable oracle. It is not the carrier stack to extend.

### Generic semantic proof and simplification

The R2-E/P2-H/R2-F portfolio and S2-A simplification are complete. The proof shows
Q4, T3, a directional spring, and program-owned point loads through the common
space/operator/channel boundary. S2-A reduced the proof to 649 production and 360
focused-test nonblank lines and received an independent GO with no findings.

### G1 candidate

G1 directly compiles one normalized Q8 `ModelSpec` into an unexported
`CompiledSystem` with ordered spaces, typed ports/channels, zero-width operator
state, selected registry identity, source attribution, owned arrays, and explicit
geometry/material qualification.

Candidate `2f6f972` replaces caller-context-sensitive Decimal qualification with
exact binary-rational qualification and a relative, zero-floor correspondence
envelope. Coordinator evidence is green:

- 15 focused G1 tests;
- 370 eight-file tests in normal and 640-digit modes;
- 500 v3 tests in both modes;
- 689 repository tests;
- both Ruff configurations and focused six-path format;
- 622 local material edge checks, including hostile Decimal contexts and
  minimum-subnormal 2x-17x perturbations.

This is strong candidate evidence, not the missing independent G1-N/G1-A verdict.

## 5. Critical correction: P1-C is not the next implementation

Any instruction saying “start P1-C” is stale relative to the current repository.
Do not restart it.

Historical P1-C facts retained for audit:

- frozen dispatch base: `02cff1ca76b02969b91eac6e889de78c7a4ef9e2`;
- source chain: `c263d958 -> 8524c233 -> ba6fc4e -> 3ee920a`;
- integrated chain: `c9b9480 -> bfc1d80 -> d78201d -> 7ee65c3`;
- outcome: typed `LinearStatic`, reusable and one-shot solve, explicit state
  transaction, exact reactions/balance/work, immutable solution,
  `verify_record()`, and fresh `verify()`;
- exact ownership and scientific contract: the
  [P1-C Horizon card](migration-execution.md#p1-c-horizon-card--horizon_frozen).

P1-C is preserved as a correctness oracle for G3/G4 migration. It is not a new
source packet and its historical branches must not be merged again.

## 6. Current pending work and blockers

### The one active blocker

G1 needs two independent, read-only reviews at exact clean `2f6f972`:

1. **G1-N finite-element numerical review** — rederive the Q8 shapes, gradients,
   quadrature, material law, geometry classification, local tangent, scaling, and
   edge behavior without using production-generated expected values.
2. **G1-A architecture/trust review** — check the generic boundary, exact carrier
   ownership, registry capture, multiple spaces, open operator contract, no
   predecessor input, no public selector, no reconstruction hole, six-path
   ownership, and line budgets.

Both must return GO with P0/P1/P2 `0/0/0`. Until then:

- G2 is closed;
- G1 remains unmerged;
- no feature writer may overlap the 35-path terminal cut;
- no adapter, shim, backend selector, or compatibility carrier may expose G1.

### Confirmed non-blockers

- P1-C and I1 are complete; their historical source branches are not unfinished
  product work.
- G1 is pushed and reproducible; the missing temporary worktree is not lost work.
- The nine full-tree Ruff-format differences recorded outside G1 ownership are
  baseline hygiene debt, not a G1 numerical or architecture finding. Do not widen
  a reviewer packet to format them.
- The existing SciPy sparse-efficiency and cold-cache Numba notices are known
  baseline warnings.
- PySide6 is absent on the Intel-Mac baseline; legacy `pyfem-gui` is not a v3 gate.

### Deferred decisions, not implied requirements

Root API/CLI compatibility, GUI packaging, `.pro`/`.dat` adapters, persisted
restore/rebind formats, observations, RVE/FE2, ROM, and public retirement decisions
remain explicit future packets. Do not guess their APIs or silently retire their
scientific intent.

## 7. Exact next action and next implementation packet

The exact next action is review, not implementation:

```text
review exact 2f6f972 independently as G1-N and G1-A
  -> adjudicate findings in migration-execution.md
  -> repair on the same exclusive branch if needed
  -> joint GO unlocks G2
```

After joint GO, the next implementation is **G2**, not P1-C. It stays on
`agnet/g1-generic-system` and owns exactly:

- `pyfem/v3/model/program.py`
- `pyfem/v3/compile/program.py`
- `pyfem/v3/assembly/contracts.py`
- `pyfem/v3/assembly/prepare.py`
- `pyfem/v3/assembly/reference.py`
- `test/v3/test_v3_program_compile.py`
- `test/v3/test_v3_assembly_plan.py`
- `test/v3/test_v3_q8_generic_flow.py`

G2 outcome:

- direct `ProgramSpec -> CompiledProgram`;
- explicit affine coordinate map with derivatives and constraint evidence;
- separately owned program load operators;
- requested-channel raw/canonical/reduced schedule and evaluator;
- no old compiled carrier as input;
- no public export switch yet.

If an unchanged dependency listed in D3-A section 5.5 must change, stop for an
interface refreeze. Do not silently widen G2 ownership.

## 8. Dependency path to the atomic cut and beyond

```text
G1 independent GO
  -> G2 coordinate map / program loads / generic schedule
  -> G3 state transaction / solve / ledger / fresh verification
  -> G4 public authority switch + causal deletion
  -> four independent terminal reviews + combined proof
  -> integrate the whole G1-G4 range atomically
```

G1-G3 are private semantic stages. They are never integrated independently. G4
alone routes the public API/package exports to the generic spine, rewrites the six
focused proof files, and deletes the predecessor/proof owners:

- `pyfem/v3/model/compiled.py`
- `pyfem/v3/compile/contracts.py`
- `pyfem/v3/compile/model.py`
- `pyfem/v3/analysis/numerics.py`
- `pyfem/v3/core/generic.py`
- `pyfem/v3/core/__init__.py`

After the atomic cut, the accepted feature frontier is:

1. real operator-local state transaction;
2. typed nonlinear static with an incompatible second history schema, including
   J2/path-dependent behavior and rejected-step rollback;
3. thermal/thermoelastic multiple spaces and coupling;
4. first-/second-order evolution with capacity/mass, conservation, and restart.

Feature breadth must not compete with replacement of the duplicate Q8 authority.

## 9. Required scientific and numerical oracles

Production code is never its own oracle. Preserve at least these independent laws:

### Q8 compiler and operator

- hard-coded 3x3 Gauss points and weights;
- hard-coded Q8 shape values and parent gradients, including partition of unity
  and zero gradient sum;
- explicit 2x2 Jacobian determinant/inverse path;
- inverted, sign-changing, near-singular, extreme-scale, and translated geometry;
- exact plane-stress law over ordinary, boundary, overflow, and subnormal inputs;
- affine displacement/constant-strain recovery and symmetric local tangent.

### Frozen rational linear solve

For the unit-square Q8 case with `E=1`, `nu=0`,
`u_1x=u_1y=0`, `u_3y=u_3x+lambda`, and `f_5y=1`, the exact displacement at
`lambda=1` is:

```text
[0, 0, 29/12, 67/24, 29/6, 35/6, -23/24, 33/4,
 -7, 32/3, -77/12, 27/8, -35/6, -7/6, -37/24, -7/12]
```

Required identities include `P.T @ r_full = 0`, direct reactions `[-1, 0]`,
constraint work `-1`, external work `32/3`, internal work `29/3`, and the rigid
affine derivative `du/dlambda=(-y,x)` in the nullspace of `K`.

### Balance, state, and verification

- `K_q q = b_q`, `u = P q + u_bar`, `f_int = K u`;
- `r_full = f_ext - f_int`, `c_constraint = -r_full`;
- `f_ext + c_constraint - f_int = 0`;
- exact-zero scale means exact-zero error; no unit tolerance floor;
- sibling/double/stale/foreign acceptance is inert and exactly one successor wins;
- published arrays do not alias trial or workspace storage;
- fresh verification rebuilds schedule, contributions, balance, work, constraint,
  and convergence evidence without reading the solve factor/cache;
- PatchTest8 analytical field:
  `u_x=1e-3*x+5e-4*y`, `u_y=5e-4*x+1e-3*y`, checked at `rtol=0`, `atol=1e-12`.

G2/G3 must retain additive repeated loads, nonzero affine MPC chains, raw versus
canonical contribution attribution, split-versus-vectorized assembly, zero-free
systems, singular/indefinite rejection, and the independently assembled five-cell
PatchTest8 oracle.

## 10. Verification commands and expected gates

Use the repository `.venv`; Python 3.13+ is required. At exact G1 tip `2f6f972`,
the accepted counts below are expected and should be explained if they drift.

```bash
git rev-parse HEAD
git status --porcelain=v1 --untracked-files=all

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  test/v3/test_v3_model_spec.py \
  test/v3/test_v3_model_identity.py \
  test/v3/test_v3_model_compile.py \
  test/v3/test_v3_program_compile.py \
  test/v3/test_v3_assembly_plan.py \
  test/v3/test_v3_linear_analysis.py \
  test/v3/test_v3_generic_core.py \
  test/v3/test_v3_q8_generic_flow.py

PYTHONINTMAXSTRDIGITS=640 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  test/v3/test_v3_model_spec.py \
  test/v3/test_v3_model_identity.py \
  test/v3/test_v3_model_compile.py \
  test/v3/test_v3_program_compile.py \
  test/v3/test_v3_assembly_plan.py \
  test/v3/test_v3_linear_analysis.py \
  test/v3/test_v3_generic_core.py \
  test/v3/test_v3_q8_generic_flow.py

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider test/v3
PYTHONINTMAXSTRDIGITS=640 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m pytest -q -p no:cacheprovider test/v3
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider

.venv/bin/ruff check pyfem/v3 test/v3 --config pyfem/v3/ruff.toml
.venv/bin/ruff check test/v3 --config test/v3/ruff.toml
.venv/bin/ruff format --check --config pyfem/v3/ruff.toml \
  pyfem/v3/compile/continuum.py pyfem/v3/compile/system.py \
  pyfem/v3/model/operator.py pyfem/v3/model/system.py \
  pyfem/v3/spec/diagnostics.py test/v3/test_v3_q8_generic_flow.py

git diff --check
```

Expected G1 counts: 370 eight-file tests in each mode, 500 v3 tests in each mode,
and 689 repository tests. Also require exact six-path ownership, clean ancestry,
zero forbidden predecessor/private-layer imports, no public export change, line
budgets at or below 1,925 production and 700 focused-test physical lines, and a
clean final worktree.

## 11. Ownership and safety constraints

- Do not merge, cherry-pick, or push an intermediate G1-G3 tip to `v3`.
- Do not squash away the semantic G1-G4 boundaries or repair provenance.
- Do not rebase, amend, force-update, or delete the pushed G1 branch.
- Do not delete historical P1-C branches, the integration-proof branch, evidence
  Markdown, detached Codex worktrees, or the unrelated stash without a separate
  exact audit and user authorization.
- Do not edit outside a packet's frozen paths.
- Do not import predecessor private validators into the generic path.
- Do not add an adapter, shim, public selector, duplicate backend, or compatibility
  carrier to bridge the two spines.
- Do not infer physics from array shape, connectivity width, or legacy type names.
- Do not weaken scientific or transaction oracles to make a candidate pass.
- Treat uncommitted and unrelated changes as user work.
- Use neutral finite-element terminology in packets and reviews.
- Delegated packets must name their callback target and send one terminal callback;
  coordinators do not poll in a loop.

## 12. Concise restart procedure

1. Read this file, the relevant invariants in `design.md` and `generic_core.md`,
   then the ledger header, [Exact next safe action](migration-execution.md#exact-next-safe-action),
   and [current D3-A/G1 card](migration-execution.md#d3-a-frozen-g1-writer-card).
   Query older ledger history only by packet ID or commit when auditing it.
2. Inspect `git status`, `git branch -vv`, `git worktree list`, and the exact local
   and remote hashes. Do not assume this snapshot is still current.
3. Confirm `v3` contains `7ee65c3` and does not contain `2f6f972`.
4. Create a detached review worktree at the exact candidate using an unused path.
   Detachment avoids branch-checkout collisions and makes the read-only base
   explicit:

   ```bash
   git worktree add --detach /private/tmp/pyfem-g1-review-new \
     2f6f972eef5a4c5f633478b1effecba834ca5fd4
   ```

5. Dispatch or perform the two read-only independent G1 reviews at exact
   `2f6f972`; record their exact commands, findings, warnings, and clean status.
6. Update only `migration-execution.md` when both results are adjudicated.
7. If either review finds a defect, repair only the six G1 paths on the same
   branch and rerun both affected reviews. If both return GO, freeze G2 against
   the exact reviewed tip before implementation.
8. Never expose or integrate the generic spine until G4 completes the atomic
   authority switch and causal deletion.

## 13. Facts, pending decisions, and speculation boundary

Confirmed facts are the exact refs, completed Phase-1 evidence, G1 candidate
evidence, frozen G1-G4 path ownership, and the pending independent review gate
recorded above.

Pending work includes G1-N/G1-A, G2-G4, terminal review/integration, and the
post-cut feature sequence. Deferred ecosystem and advanced-physics decisions are
not blockers to the current cut.

Speculation includes future public names, adapter syntax, persistence schemas,
GUI packaging, RVE/FE2 scheduling details, ROM artifact formats, and alternate
backend choices. Do not present any of these as accepted design without a new
Horizon/design decision and evidence.
