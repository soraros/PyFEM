# PyFEM v3 research-kernel execution charter

Live state checked on 2026-11-05 against `v3` at `d9e4a12`, after the
M113/M117/M118 consolidation. M120 wrote this charter without changing
implementation. It is the only current implementation queue.

This charter supersedes M119's preserve-and-consolidate framing, the former
parity-era dispatch in this file, and any instruction to port the next legacy
family or preserve a v3 facade for sunk cost. [migration-history.md](migration-history.md),
[evidence/](evidence/), [parity.md](parity.md), [scaling.md](scaling.md), and
`bench/results/` remain historical evidence. Their "current", "next", ABI-freeze,
and feature-parity instructions are not work to resume. The invariants in
[design.md](design.md) and [generic_core.md](generic_core.md) still govern the
surviving kernel; their old migration sequences do not expand this charter.

## NX7-1 implementation and acceptance record

Atomic implementation commit: `288e5f3`.

NX7-1 retired the complete D3 surface against the `d9e4a12` evidence boundary:
penalty contact, cohesive interfaces and their deck adapter, unsolved
thermal/thermoelastic operators, Prony viscoelasticity, both Perzyna mappings,
SOVS, crystal plasticity, and every associated registry, descriptor, authoring,
and focused-test branch. No compatibility facade or fallback replaced them.
Interface decks raise `unsupported-element-type`; retired authoring material
slices raise explicit input errors; direct retired-material specs fail registry
capture; retired thermal formulations fail with `incompatible-formulation`.
The exact symbols, fourteen test files, literature, and independent oracle
locations are indexed in
[migration-history.md](migration-history.md#nx7-1-retirement-non-kernel-scientific-breadth).
The wholly-v3 crystal page moved to
[evidence/2026-11-05-nx7-crystal-plasticity.md](evidence/2026-11-05-nx7-crystal-plasticity.md)
and left the Sphinx toctree. Valid legacy-only documentation and warnings stay
live with historical demotion links.

The actual nonblank Python totals, measured recursively against `v3` at
`64cbb92`, are **49,667 -> 41,876 production lines** (net **7,791 removed**,
exceeding 6,500) and **47,187 -> 35,697 v3-test lines** (net **11,490
removed**, exceeding 9,500). The fourteen deleted focused files account for
11,525 test lines. The retained partial-support multi-space case adds 32
nonblank lines to `test_v3_q8_generic_flow.py`, the explicit retired-interface
deck rejection adds eight, and removal of an obsolete thermal-only breadth
case removes another five, reconciling the final net count exactly.

Retained generic multi-space allocation, parameterized state layouts, J2,
damage, FE², total-Lagrangian, truss, spring, researcher-signal, and Riks paths
were not given replacement numerical implementations. The retained notebook's
analytic PatchTest8 assertion passed; 243 focused compiler/deck/state/authoring
tests passed; an AST audit parsed 199 production/test/benchmark/notebook files
and found no deleted-module import; the complete bare and per-tree Ruff
check/format matrix passed. Both full repository digit-mode suites passed
**1,366 tests plus 12 subtests each** (normal and `PYTHONINTMAXSTRDIGITS=640`).
`python -m bench.run gates` reported **0 failures**, including raw-zero FE²
identity comparisons and unchanged J2, total-Lagrangian, truss, Riks, and family
oracles. `python -m sphinx -M html doc doc/_build` succeeded with 117 existing
autoapi/legacy warnings, none naming the changed pages. Generated-HTML audits
confirmed the deleted v3 module pages and crystal toctree entry are absent and
all legacy demotions are present. No performance baseline changed, and no shared
hot numerical implementation changed, so no candidate/control timing or
performance claim is made.

## NX7-2 implementation and acceptance record

Atomic implementation commit: `f766bb2`.

NX7-2 deleted the complete D1 frozen Q8 vertical at the `df31ef4` evidence
boundary: `compile_model`, `compile_program`, `evaluate_program`, their
diagnostics and manifest constants, the model compiled/program/state carriers,
all of `analysis/`, `assembly/`, and `results/`, and their compile/model exports.
This includes `CompiledModel`, `CompiledProgram`, `PreparedAnalysis`,
`PreparedAssemblyPlan`, `assemble_reference_linear`, `CommittedAnalysisState`,
`Solution`, `LinearBalanceLedger`, and `fresh_verify`. The four frozen focused
test files were deleted after the generic migrations passed. No compatibility
runtime or replacement `Solution.verify` exists. The exact deleted surface and
old/new independent-oracle map are indexed in
[migration-history.md](migration-history.md#nx7-2-retirement-frozen-q8-analysisresult-vertical).

The actual nonblank Python totals after the R1 correction, measured recursively
against `v3` at `df31ef4`, are **41,876 -> 30,937 production lines** (net
**10,939 removed**, exceeding 10,500) and **35,697 -> 30,856 v3-test lines**
(net **4,841 removed**, exceeding 4,500). The direct D1 production set and
export cleanup contribute 10,964 removed lines; R1 commit `3eac574` adds 25
lines for finite additive-load accumulation checks. The four deleted test files
contribute 5,509 lines; retained migrations and regressions add 668. The
original 636 comprise 221 in `test_v3_q8_generic_flow.py`, 112 in
`test_v3_constraints.py`, 242 in `test_v3_driver_nonlinear.py`, and 61 in
`test_v3_driver_parity.py`; R1 adds 32 in `test_v3_driver_plan.py`.
`test_v3_model_identity.py` is unchanged because it contains only shared
array/identity/provenance/registry subjects.

The retained tests now cover literal Q8/rigid-body, authored material/topology
rejection, canonical provenance, prescribed-root/shared-master and overflowing
affine maps, rational equilibrium/reactions/work, zero-free rigid offsets,
chained MPC solving, and a real analytic PatchTest8 accepted-record observation
of full/reduced balance and direct reactions. R1 commit `3eac574` explicitly
covers finite-addend constant and coordinate-coefficient load overflow through
the public `NonlinearStaticDriver`, raising `DriverPreparationError` with
`non-finite-load-value` before manifest construction while preserving ordinary
ordered load arithmetic. Existing analytic Q4/T3/Hex8, constant-tangent reuse,
full/reduced projection, and rejected-attempt byte identity tests remain. The
migration gate passed **188 focused tests**; the R1 focused run passed **203
tests**. The notebook quickstart passed its analytic PatchTest8 assertion. An
AST audit parsed 174 production/test/benchmark/notebook files and found no
deleted-module import. At `3eac574`, both complete repository digit-mode suites
passed **1,156 tests plus 12 subtests each** (normal and
`PYTHONINTMAXSTRDIGITS=640`). The complete exact and bare Ruff check/format
matrix passed on the same commit.

The deletion acceptance run of `python -m bench.run gates` at `f766bb2` reported
**0 failures**, including raw-zero FE² and J2 comparisons and unchanged passing
Q8/skim, total-Lagrangian, truss, Riks, and family oracles. No `bench/results/`
file or performance baseline changed, and no candidate/control timing or
performance claim is made. The branch is ready for fresh R2 independent
source-to-charter review of the deletion, proof migration, and R1 correction.

## Exact next safe action

After NX7-2 is independently reviewed and merged, execute **NX7-3 — route every
maintained consumer through the generic spine, then delete the prototype**, below.
Migrate the Q8/skim benchmark consumers before removing their prototype runtime;
do not preserve a second solve lane or promote a performance baseline during the
migration.

Each later program starts only after the preceding program's consumers,
numerical gates, both digit-mode suites, Ruff gates, and required benchmark
evidence are green at the same commit. Update this file with the actual commit,
deleted surface, measured line delta, evidence artifacts, and the next safe
action. Never leave a deleted capability documented as supported.

## 1. North star and the change of axis

PyFEM v3 remains a scientific finite-element library in which convenient
authored physics is normalized and compiled once into immutable array-backed
operators; homogeneous numerical work has explicit state and contribution
contracts; accepted and trial state are separate; constraints and sparse
structure are reusable; analyses request mathematical channels rather than
element classes; and accepted results retain enough provenance and balance
information to audit the answer.

The implementation axis is now **the smallest coherent research kernel**, not
legacy coverage. A legacy class, a v3 prototype, a frozen ABI, a manifest, or a
large test battery is not a scientific consumer. Parity can convict a numerical
error, but it cannot justify a second runtime or a capability with no maintained
use. Deleting support is preferable to preserving an misleadingly broad facade.

Potential users of contact/cohesive, thermal/thermoelastic, rate-dependent,
SOVS, and crystal physics remain first-class concerns. Their absence from the
current maintained-consumer inventory is not evidence that the domains are
unimportant or that PyFEM has no intention to serve them. NX7 is a current-scope
simplification, not a permanent support boundary: each retirement must preserve
literature, independent oracles, and generic operator/state/driver seams so a
documented consumer-backed mission can reintroduce the capability on the minimal
kernel. Re-expansion must never resurrect duplicate frozen or prototype
runtimes.

A capability is supported only when it has a truthful authored-to-accepted-result
path and a maintained consumer. Maintained consumers are the v3 notebook,
registered benchmark workloads, supported legacy-deck routes used by those
workloads, and small authoring/extension flows needed by those scientific paths.
A compile-only operator, a test of its carrier, a legacy source row, or a public
re-export is not by itself a consumer. Tests provide evidence for a selected
capability; they do not select it.

## 2. Evidence at the `d9e4a12` boundary

Counts below are nonblank Python lines measured on the checkout, used as a
repeatable complexity metric rather than a claim about scientific value.

| Surface | Current size | What the consumers actually exercise |
|---|---:|---|
| `pyfem/v3/` | 49,667 | Three solve stacks plus direct, composed, and nested operators |
| `test/v3/` | 47,187 | Strong independent oracles mixed with duplicate carrier and parity tests |
| `bench/` | 3,863 | Prototype Q8/skim routes and generic FE²/family routes |
| `notebooks/v3/` | one paired notebook | `authoring` + legacy deck + `NonlinearStaticDriver` on PatchTest8 |
| `examples/` | one documented v3 authoring example | `examples/materials/viscoplasticity/README.md` uses `authoring.backward_euler_perzyna`; the other examples are legacy |

The decisive consumer facts are:

- The maintained notebook already uses `read_legacy_deck`, `authoring.compile`,
  `authoring.nonlinear_static`, typed driver status, and accepted coefficients.
- FE² and damage benchmarks use `CompiledSystem`, nested execution, J2/damage,
  homogenization, state codecs, and deterministic one-thread/n-thread gates.
  The M116 controlled pair reports a 51.2% harness-wall reduction for execution
  consolidation; the M85 pair covers the FE² and memory matrix.
- Finite-strain and Riks family benchmarks use the deck compiler and generic
  drivers. The M118 controlled pair covers 27 records/60 metric cells and exact
  cycle/correction counts with documented state tolerances.
- Uniform Q8 and skim benchmarks still use `ProblemDefinition`, `pack`,
  `_prototype_assembly`, and `solver/`; that is a migration debt, not a reason to
  keep the prototype. Their legacy-v1 comparisons and analytic patch gates are
  the durable oracles.
- The frozen `CompiledModel`/`CompiledProgram`/`PreparedAnalysis` vertical has
  its own assembly, state, `Solution`, and fresh verification implementation,
  but no notebook or benchmark consumer. Its production files total 10,861
  nonblank lines.
- Penalty contact, cohesive interfaces, thermal/thermoelastic operators, and
  the Prony, two Perzyna, SOVS, and crystal families have no v3 notebook or
  registered benchmark workload. They nevertheless have named live support:
  `examples/materials/viscoplasticity/README.md` contains a public compile-only
  `authoring.backward_euler_perzyna` example, while
  `doc/materials/{viscoplasticity,crystal_plasticity,sovs}.md` are live Sphinx
  support pages. D3/NX7-1 explicitly retires those surfaces rather than denying
  them. The direct implementation files total 5,314 nonblank lines before
  compiler/adapter cleanup; their fourteen focused test files total 11,525. The
  generic driver also rejects operators with more than one coefficient port in
  `driver/plan.py::_compile_operator_slices`, so the coupled thermoelastic
  compiler is not a supported solve path.
- Parameter sensitivities are different: the authored J2 flow has independent
  derivative evidence and shares the converged-factorization/IFT machinery used
  by homogenization. Keep its explicitly fixed-entering-state meaning; do not
  advertise total path sensitivity or add speculative state-derivative v2 work.

Historical numbers and revisions are evidence for those exact revisions, not
permanent minimum test counts or speedup promises.

## 3. Target kernel

### 3.1 One execution spine

```text
ModelSpec + ProgramSpec + plain researcher declarations
                       |
                 normalization
                       |
                compile_system
                       |
                CompiledSystem
       entities + spaces + operators + provenance
                       |
        CompiledConstraintMap + DriverAssemblyPlan
                       |
       NonlinearStaticDriver or RiksDriver schedule
                       |
 StateTransactionOwner + nested execution + disposable workspace
                       |
 typed records + accepted snapshots + solve-free observations
```

`read_legacy_deck` is an adapter into this spine. `authoring` is a small plain
Python convenience layer over it. Neither may grow another solver, state owner,
or result implementation. Linear static problems use the nonlinear driver's
constant-tangent path; they do not need a second linear runtime.

### 3.2 Supported modules and scientific roles

| Surface | KEEP / UNIFY / REBUILD decision |
|---|---|
| `spec/`, `model/arrays.py`, `identity.py`, `provenance.py`, `registry.py` | **KEEP** the single normalized authoring, ownership, identity, and injected-registry contracts. |
| `model/system.py`, `model/operator.py` | **KEEP** entities, spaces, ports/channels, state layouts, and direct/nested evaluation; **UNIFY** repeated input/channel validation in compiler helpers. |
| `compile/system.py`, retained family builders | **REBUILD** per-region composition and one operator-composition path; no legacy-shaped mandatory taxonomy and no typed DSL. |
| `constraints/` and `driver/plan.py` | **KEEP** one affine coordinate map and one cached CSR/COO assembly plan. |
| `driver/nonlinear.py`, `driver/riks.py` | **KEEP** Newton/cutback and two-solve arc-length algorithms; **UNIFY** evaluation, factorization, nested-scope, and accounting internals without changing their numerical policy in the refactor. |
| `state/` | **KEEP** the transaction owner, versioned row/nested/evolution codecs, continuation store, and nested rollback. No third accepted-state owner. |
| `driver/contracts.py` and accepted driver records | **KEEP** typed status, balance/reaction, sensitivity, homogenization, nested-work, and continuation observations. No universal observation registry is required. |
| Small-strain Tria3/Quad4/Quad8/Hex8 and plane-stress/plane-strain/3D elastic laws | **KEEP + UNIFY** behind geometry-profiled small-strain payload/evaluator code; remove Q8-only class names rather than scientific geometry support. |
| Q8 total-Lagrangian Saint-Venant–Kirchhoff | **KEEP** the benchmarked finite-strain route and independent tangent/force oracles. |
| Truss and grounded/researcher point springs | **KEEP** unlike entity layouts, Riks/skims, custom state, and signal extension points. |
| J2 isotropic hardening and plane-strain damage | **KEEP** the stateful, sensitivity, nonsymmetric, JIT, and FE² scientific vehicles and their independent references. |
| Fixed-entering-state parameter sensitivities | **KEEP** the existing bounded semantics and cost pins; total path propagation is out of scope until a real consumer defines it. |
| Periodic RVE authoring, homogenization, FE², deterministic host pool | **KEEP** the benchmarked multiscale route and retained-buffer/state identity gates; do not add more execution modalities speculatively. |
| `authoring/` and `io/legacy_deck.py` | **REBUILD** as small adapters over retained physics only; delete duplicate exports and rejected deck branches. |
| `fem/` numerical primitives | **KEEP** shapes, quadrature, kinematics, link/TL and required constitutive/assembly math; **UNIFY** canonical CSR helpers and remove prototype-only carriers/dispatch. |
| `notebooks/v3/patch_test8.py` + `.ipynb` | **KEEP** as the executable public quickstart and regenerate the paired artifact only from the Jupytext source. |

### 3.3 Explicit non-kernel decisions

- **No public `PreparedExecution`.** `DriverAssemblyPlan` is already the prepared
  topology, `StateTransactionOwner` already owns accepted state, and
  `PreparedNestedExecution` already owns nested resources. Adding a fourth public
  lifecycle object would duplicate ownership. NX7-5 may create only a private
  shared runtime helper to remove measured driver duplication.
- **No typed DSL.** `ModelSpec`, `ProgramSpec`, registry descriptors, and plain
  researcher declarations are sufficient. Do not add a language, code generator,
  general plugin discovery system, or universal dictionary protocol.
- **No second result/verification stack.** Accepted driver records and fresh
  numerical tests are the supported audit surfaces. The old Q8 `Solution` and
  reference evaluator are deleted, not wrapped.
- **No compatibility facade for deleted APIs.** Pre-1.0 callers migrate to the
  generic spine in the same change. A deprecation note cannot substitute for a
  truthful supported implementation.
- **No speculative physics.** Beams, shells, dynamics, general contact search,
  irreversible cohesive laws, multi-master nonlinear constraints, ROM, transient
  integrators, and total path sensitivities require a future consumer and a new
  charter amendment; they are not hidden remaining work.

## 4. Authorized dispositions and deletion map

These are approved retirements for this charter, not tentative parity-inventory
"retire" suggestions. Each still requires the consumer/test migration and
historical evidence entry in its owning program.

### 4.1 D1 — frozen Q8 vertical: DELETE

Delete `compile/model.py::compile_model`,
`compile/program.py::{compile_program,evaluate_program}`,
`compile/program_diagnostics.py`, `model/compiled.py`, `model/program.py`,
`model/state.py`, all of `analysis/`, `assembly/`, and `results/`, and their
exports in `compile/__init__.py` and `model/__init__.py`. This includes
`CompiledModel`, `CompiledProgram`, `PreparedAnalysis`, `PreparedAssemblyPlan`,
`assemble_reference_linear`, `CommittedAnalysisState`, `Solution`,
`LinearBalanceLedger`, and `fresh_verify`.

Delete `test_v3_linear_analysis.py`, `test_v3_assembly_plan.py`,
`test_v3_program_compile.py`, and `test_v3_model_compile.py` after migrating
still-relevant authored-input, geometry, constraint, and balance cases to the
generic compiler/driver tests. Keep and trim `test_v3_model_identity.py` because
most of it covers shared primitives; keep `test_v3_q8_generic_flow.py` because it
already tests the surviving system. Do not port tests whose only subject is a
deleted frozen carrier or deliberate mutation of compiler-owned objects.

### 4.2 D2 — prototype public solve lane: DELETE after consumer migration

Delete `types.py`, `pack.py`, `loader.py`, `registry.py`,
`_prototype_assembly.py`, `solver/`, `mesh/`, `elements/`, and
`io/{dat,legacy_pro,toml,solver_pro,load_ramp}.py`. Remove
`ProblemDefinition`, `load_problem`, `solve_linear`, `solve_nonlinear`, and
`solve_riks` from the root package and all `LoadedProblem`-based benchmark paths.
Replace `types.F64`/`I32` imports in retained numerical modules with explicit
NumPy dtype aliases owned by those modules; do not retain `types.py` for aliases.

Move `CooCsrPattern`, `compile_csr_pattern`, `dedup_coo_values`,
`dedup_coo_segments`, `_PARALLEL_MIN_ENTRIES`, `_dedup_coo_values_serial`, and
`_dedup_coo_values_parallel` from `fem/assembly.py` to `fem/sparse.py`; these are
the canonical pattern/refill helpers and thresholded serial/parallel reduction
kernels needed by `DriverAssemblyPlan` and generic benchmarks. Delete
`fem/assembly.py`, its `GROUP_*` dispatch/prototype assembly, and the prototype
re-exports in `fem/__init__.py`. Retain the small-strain stiffness/internal-force
math with generic callers or independent oracles in `fem/element.py`. Delete the
prototype-only Tet4 lane: `fem/element.py::{_tet4_rule,tet4_stiffness}`,
`fem/shapes.py::linear_tet4`, `fem/quadrature.py::gauss_tet4`, their exports, and
`test_tet4_solve.py`; Tet4 is not a hidden expansion obligation.

Keep the parallel reduction path: it is retained driver-plan behavior and part of
the generic performance envelope, not prototype-only dispatch. Before deleting
`test_assembly_coo.py`, migrate
`test_dedup_coo_values_bitwise_match_strict_sequential_reference` and
`test_dedup_coo_values_bitwise_identical_across_thread_counts` into a retained
`test_v3_sparse.py`. Replace their `LoadedProblem` fixtures with synthetic COO
streams containing duplicate runs and at least `_PARALLEL_MIN_ENTRIES` entries;
pin the explicit serial and parallel kernels and the dispatcher at 1T versus nT
with raw-uint64 equality above the threshold, check the strict left-to-right
reference, and restore the prior Numba thread count. Add the corresponding
above-threshold `refill_tangent` integration oracle to `test_v3_driver_plan.py`
using a compiled generic system rather than the prototype patch. Serial-only is
not an accepted simplification without separate numerical and matched
candidate/control performance evidence.

Migrate `test_v3_driver_parity.py` and the skim/linear parity tests to
`read_legacy_deck` plus generic drivers. Keep `test/v3/_legacy_parity.py` as an
independent legacy-v1 oracle, removing its imports of deleted v3 settings
parsers. Its deliberately independent settings handling may duplicate
`bench/legacy_settings.py`; do not make the oracle and the code under test share
an implementation merely to reduce lines.

### 4.3 D3 — unsupported scientific breadth: DELETE

Delete these complete live families and their authoring exports:

- `compile/contact.py`, `authoring/contacts.py::penalty_contact`, and the
  contact tests. The analytic-disc penalty operator has no search, friction,
  observation, deck, notebook, or benchmark consumer.
- `compile/interface.py`, all four helpers in `authoring/interfaces.py`, the
  `ConvertedDeck.interfaces` field, and
  `io/legacy_deck.py::{_InterfaceProfile,_interface_family,_interface_declaration,_check_interface_arities}`
  plus their call sites. Remove the interface/cohesive/authoring-interface tests.
  The frame/tangent investigations remain historical numerical evidence.
- `compile/continuum.py::{Q8ThermalPayload,Q8ThermalOperator,Q8ThermoElasticPayload,Q8ThermoElasticOperator,_compile_thermal,_compile_coupled}`,
  `THERMAL_*`/`THERMO_*` keys, descriptor branches, and dispatch entries. Delete
  `test_v3_multifield_continuum.py` after moving a small partial-support,
  multiple-discrete-space test to the generic system tests. Do not confuse
  deleting unsolved coupled operators with deleting multi-space representation.
- `materials/{prony_viscoelasticity,perzyna_viscoplasticity,perzyna_backward_euler,skorohod_olevsky,crystal_plasticity}.py`;
  their registry imports, bindings, descriptor constants and factories in
  `compile/continuum.py`; `authoring.materials` constructors
  `prony_viscoelasticity`, `viscoplasticity`, `backward_euler_perzyna`, and
  `skorohod_olevsky`; and the corresponding `*_law`/`*_registry` helpers in
  `authoring/registry.py` and `authoring/compile.py::_default_registry`. Delete
  their focused tests after recording the oracle locations. Scalar program
  signals remain supported by researcher springs; no built-in rate law is needed
  to justify that ABI.

Retire the affected public documentation explicitly, without deleting valid
legacy-only documentation:

- In `examples/materials/viscoplasticity/README.md`, remove the live section
  "True backward-Euler Perzyna physics in v3" and its authoring example, and link
  the remaining legacy deck README to the retirement entry. Apply the same
  removal or historical demotion to
  `examples/materials/viscoplasticity/bar_tension.pro:33-34`, whose shipped
  comment says the separate v3 law implements true rate physics, so no stale
  support statement survives. Preserve the legacy deck's rate-independent
  explanation and negative control.
- In `doc/materials/viscoplasticity.md`, remove the v3 mapping claims and the
  complete "True v3 `backward-euler-perzyna`" section, including its live
  authoring contract. Retain a legacy-only `ViscoPlasticity` page; demote the v3
  return-map/tangent equations, authoring example, and verification references to
  the retirement entry.
- Demote the wholly v3 `doc/materials/crystal_plasticity.md` page to
  `.agents/v3/evidence/2026-11-05-nx7-crystal-plasticity.md`, marked as retired
  historical evidence at `d9e4a12`, and remove `crystal_plasticity.md` from
  `doc/materials/overview.md`. Do not leave a live Sphinx link or present the law
  as supported.
- In `doc/materials/sovs.md`, retain the legacy model, quirks, repaired-tangent,
  and bibliography documentation but remove/demote the live v3
  `skorohod-olevsky` migration and parity claims in the tangent note. Update any
  v3 support links to point to the retirement entry.

Before deletion, append an NX7 retirement entry to `migration-history.md` naming
`d9e4a12`, these symbols/tests/documents, and the consumer/cost reason for
retirement. The entry must preserve these independent literature and oracle
locations rather than merely saying "see Git":

- For backward-Euler Perzyna, cite the README's retired v3 section,
  `doc/materials/viscoplasticity.md` sections "Verification" and "References",
  Perzyna (1966, DOI `10.1016/S0065-2156(08)70009-7`), Simo and Hughes (1998,
  DOI `10.1007/b98904`), and
  `test_v3_viscoplasticity.py::{test_backward_euler_perzyna_matches_scalar_bisection_and_central_fd,test_backward_euler_perzyna_symbolic_implicit_derivative,test_backward_euler_perzyna_elastic_and_rate_independent_limits,test_backward_euler_perzyna_q8_fast_loading_has_higher_overstress,test_backward_euler_perzyna_uninterrupted_and_restart_are_byte_equivalent}`.
- For crystal, cite the demoted page's "Slip systems and Schmid kinetics",
  "Saturation and latent hardening", and "Why this is fresh, not ported"
  sections; Peirce/Asaro/Needleman (1983, DOI
  `10.1016/0001-6160(83)90014-7`); Kocks and Mecking (2003, DOI
  `10.1016/S0079-6425(02)00003-8`); and
  `test_v3_crystal_plasticity.py::{test_single_slip_analytical_hardening_response,test_sympy_implicit_return_map_derivative,test_full_tangent_fd_conviction_and_measured_class,test_elastic_limits_are_exact,test_state_transactions_and_typed_rejections,test_two_system_competition_uses_the_complete_fcc_family,test_compiled_layout_derived_dtime_and_driver_commit}`.
  State explicitly that removed legacy `Crystal.py` is not a numerical oracle:
  conviction comes from literature, analytical single-slip response, SymPy,
  finite differences and tampered-tangent controls, elastic limits, transaction
  isolation (input purity, repeated evaluation, committed-state isolation, and
  typed rejection), full-family competition, and the compiled-driver
  derived-`dtime` path, not legacy arithmetic.
- For SOVS, cite `doc/materials/sovs.md` "References" (Skorohod 1972 and
  Olevsky 1998), the documented M67/M76 tangent-repair history, and
  `test_v3_sovs.py::{test_activated_tangent_repair_confirmed_by_fd,test_kernel_tangent_is_the_exact_stress_map_derivative_by_fd,test_symbolic_jacobian_confirms_the_closed_form}`.
  For the other D3 families, name their module docstrings, independent legacy or
  literature references, and focused tests at `d9e4a12` with the same precision.

Historical source remains recoverable from Git; do not copy dead implementations
into the documentation tree or keep them importable as "reference" production
modules.

### 4.4 D4 — facades and low-value public exports: UNIFY or DELETE

The root package stops re-exporting a solver facade. `authoring` retains the
mesh/model/material/program constructors, `compile`, `evaluate`, `trial_vector`,
`spring`, `fe2_continuum`, `periodic_cell`, `nonlinear_static`, and the typed
status/settings/result values needed by those flows. `driver.__init__` retains
the two drivers and their public request/result types, not plan internals,
`IntegrationPointTaskPool`, or `PreparedNestedExecution` as convenience exports.
`compile.__init__` no longer presents the frozen Q8 compiler as the package API;
direct `compile_system` and owning family modules are explicit. `io.__init__`
exports only the retained deck adapter.

Do not replace deleted re-exports with a new mega-facade. Advanced users import
an owning module; ordinary users follow the notebook.

## 5. Ordered implementation programs

Complexity targets are **net** changes after required test/oracle migrations,
measured with the same nonblank-line method. Partial-file estimates are not
additive; the final budget is the controlling check. A program that adds lines
must explain the unexpected scientific gain and amend this charter before merge.

### NX7-1 — retire non-kernel breadth and record its evidence

1. Add the historical retirement entry with every D3-required literature,
   independent-oracle, and non-legacy crystal verification location, plus the
   retained multi-space test.
2. Apply D3, including the named edits to
   `examples/materials/viscoplasticity/README.md` and
   `examples/materials/viscoplasticity/bar_tension.pro:33-34`,
   `doc/materials/{viscoplasticity,crystal_plasticity,sovs}.md`, the crystal
   historical move, and `doc/materials/overview.md`. Remove all exports,
   registry defaults, and deck support in the same commit. No rejected deck may
   silently fall back to another law.
3. Delete the fourteen obsolete focused test files except for the specifically
   migrated space case; do not retain skipped tests.

**Consumers affected:** interface decks and direct users of the named contact,
thermal, rate, SOVS, and crystal helpers now receive explicit unsupported-input
errors. Readers of the backward-Euler Perzyna README example and the three
Sphinx pages are directed to the exact historical retirement record; valid
legacy-only documentation remains live. No registered benchmark or v3 notebook
uses the deleted families. Researcher springs, J2/damage, FE², TL, and Riks
remain.

**Complexity gate:** at least 6,500 net production and 9,500 net v3-test
nonblank lines removed. Current direct files alone provide 5,314 production and
11,525 test lines; thermal, compiler, registry, and adapter cleanup supplies the
rest while allowing for the small retained test.

**Acceptance:** retained authored flows and `bench.run gates` pass; FE² raw
identity and J2/damage oracles are unchanged; both full digit-mode suites and
all Ruff gates pass; an import/search audit finds no live reference to D3
symbols. The Sphinx build and a documentation audit must show no live v3
support claim for the deleted families, no crystal entry in the materials
toctree, the historical crystal evidence at its exact new path, and all required
literature/oracle citations in `migration-history.md`. Run matched
candidate/control timings only if shared hot code changes; otherwise record that
no performance claim or baseline change is made.

### NX7-2 — delete the frozen Q8 analysis/result vertical

1. Migrate the unique generic-input, geometry, affine-map, equilibrium, reaction,
   rollback, and provenance cases identified in D1 to
   `test_v3_q8_generic_flow.py`, `test_v3_constraints.py`,
   `test_v3_driver_nonlinear.py`, and `test_v3_model_identity.py` as appropriate.
2. Add one PatchTest8 accepted-result check of full/reduced balance and reactions
   through the existing driver observation. Do not recreate `Solution.verify()`.
3. Apply D1 and remove the now-empty packages and exports.

**Consumers affected:** direct `compile_model`/`prepare_analysis` users migrate
to `compile_system` + `NonlinearStaticDriver`. The notebook and benchmarks need
no compatibility wrapper.

**Complexity gate:** at least 10,500 net production and 4,500 net v3-test lines
removed. The D1 production set is 10,861 lines; the four deleted test files are
5,509 lines before the focused migrations.

**Acceptance:** analytic Q4/T3/Q8/Hex8 and PatchTest8 oracles, affine MPC,
constant-tangent reuse, full/reduced residual, and rejected-attempt byte identity
pass. Run `bench.run gates`; require no FE²/family numerical change. Both full
suites and Ruff are mandatory. No performance baseline is promoted for a deletion.

### NX7-3 — route every maintained consumer through the generic spine, then delete the prototype

1. Rebuild `bench/v3_pipeline.py` on generic operator evaluation,
   `DriverAssemblyPlan`, constraint reduction, and the nonlinear constant-tangent
   path. Move `build_uniform_q8_patch` out of the deleted public mesh facade into
   the benchmark generator or make it emit retained specs; eliminate
   `build_uniform_q8_loaded` with `LoadedProblem`.
2. Migrate `bench/run.py::{run_q8_workload,_v3_skim_solver,run_skim_case}`,
   `bench/cold_worker.py`, `bench/gates.py`, and workload documentation to the
   generic deck/driver route. Preserve identical generated decks, stage meanings,
   cold/warm distinctions, analytic patch gates, and legacy-v1 comparisons.
3. Migrate generic parity tests and `_legacy_parity.py`; apply D2; delete the old
   prototype-only tests rather than maintaining two v3 sides.
4. Record a new baseline successor only after a full clean reference-machine run.
   Never compare timings from the deleted pipeline as if they were the same
   workload and never overwrite `baseline_m3.json` history.

**Consumers affected:** root imports, old TOML/`.pro` readers, old solver calls,
and prototype-based benchmark clients are removed. The notebook already uses the
target route; skim and family routes converge on it.

**Complexity gate:** at least 1,800 net production and 1,500 net v3-test lines
removed, excluding any benchmark rewrite whose stage coverage stays equivalent.
The exclusive root/solver/old-io/mesh/elements files provide about 1,936 lines;
partial `fem` cleanup is additional.

**Acceptance:** every registered skim preserves its independent legacy/analytic
gate; TL tangent/force and Riks branch tolerances pass; generic Q8 scale gates
include the analytic field; FE² gates stay raw-identical. The migrated synthetic
above-threshold serial/parallel and dispatcher 1T/nT raw-uint64 oracles and the
generic `refill_tangent` integration oracle must pass in both digit modes. A full
candidate and base-revision control run under matched clean conditions must show
no unexplained regression above the established 1.25 ratio plus 1 ms noise rule.
Missing cells are failures in full coverage, not "skipped" successes.

### NX7-4 — unify the surviving compiler and operator implementations

1. Split the retained stateful protocols/layout helpers in
   `compile/contracts.py` from duplicated Q8/TL descriptor factories; give each
   descriptor and registry factory one owner. Remove the frozen compiler export
   surface before renaming anything.
2. Replace `Q8ContinuumPayload`/`Q8ContinuumOperator` and
   `Hex8ContinuumPayload`/`Hex8ContinuumOperator` with geometry-profiled
   small-strain implementations. Reuse the same input/channel validation and
   contraction code across ranks and stateless/stateful/FE² paths where semantics
   are actually identical.
3. In `compile/system.py`, replace first-matching-family selection and
   `hasattr(slice_, "fields")` dispatch with per-region routing and an explicit
   builder signature. Replace `SpringFamilyBuilder.compose`,
   `spring.compose_system`, and `fe2.compose_fe2_system` duplication with one
   validated operator/entity composition helper that preserves source identity.
4. Extract only compiler-side validation used by retained continuum, truss,
   spring, and FE² operators. Do not recursively revalidate compiler-owned
   headers in hot loops and do not introduce a universal operator base class.

**Consumers affected:** all retained authored/deck/FE² flows; public scientific
behavior is unchanged, but internal class names and manifest schemas may change
without compatibility shims. Fingerprint changes must be intentional and listed.

**Complexity gate:** at least 1,000 net production and 1,000 net test lines
removed after migrations; no increase in the number of public operator lifecycle
objects or registry kinds.

**Acceptance:** rational Q4/T3 and analytic Q8/Hex8 oracles, TL finite
differences/objectivity, J2/damage tangent and failing controls, custom spring
signals/state, source diagnostics, and FE² virgin/nested rollback all pass.
Unchanged schemas remain byte-identical; changed schemas require explicit
migration tests, not a blanket re-pin. Measure warm generic family/FE² candidate
and control runs; canonical reductions stay raw-identical at 1/n threads and the
standard regression gate applies.

### NX7-5 — deduplicate driver execution without adding `PreparedExecution`

1. Extract the common substance of `nonlinear._DriverWorkspace` and
   `riks._RiksWorkspace`, operator-round evaluation, factorization/cache
   handling, nested begin/reject/validate/commit/close, activity deltas, and
   statistics into a **private** `driver/_runtime.py` composition helper.
2. Leave Newton/cutback and arc-length prediction/constraint/termination in their
   respective drivers. `StateTransactionOwner`, `NestedStateStore`, and
   `ContinuationEvolutionStore` remain the only authoritative owners in their
   domains. The helper owns no accepted physical history.
3. Keep `DriverAssemblyPlan` as the sole prepared topology and
   `PreparedNestedExecution` as the nested resource owner. Do not export or
   rename either into a new general `PreparedExecution` API.
4. Keep driver observations typed and solve-free. No output writer, observation
   registry, or hidden auxiliary solve is added by this program.

**Consumers affected:** none at the public numerical API; nested and Riks users
get one implementation of the same lifecycle invariants.

**Complexity gate:** at least 300 net production lines removed and no new public
exports. If the private extraction cannot meet that gate with simpler call sites,
do not land it; retaining two clear algorithms is better than another abstraction.

**Acceptance:** before/after driver records, counters, continuation evolution,
accepted/trial rows, and nested payloads are raw-uint64/byte identical where the
algorithm is unchanged. Rejected attempts leave every owner byte-identical.
Run the M85-style FE² matrix and M118-style Riks candidate/control routes, all
scientific gates, both full suites, and Ruff. Standard ratio/noise rules apply;
no universal threading-speedup claim is allowed.

### NX7-6 — close the public, test, and benchmark surfaces

1. Apply D4 and make the notebook the only quickstart facade. Regenerate its
   `.ipynb` pair and execute it headlessly against the final tree.
2. Organize remaining tests by scientific contract rather than mission history.
   Remove duplicate carrier sabotage and implementation-mirror tests only after
   an independent behavior/oracle mapping shows them redundant.
3. Keep benchmark lanes small and disjoint: generic Q8/skim, J2 kernel, FE² and
   nested memory, TL/Riks family, and cold startup. Legacy-v1 is an external
   oracle side, never a second maintained v3 implementation. Historical JSON is
   read-only evidence except for a reviewed successor baseline.
4. Update architecture/source docs to describe only live support, then record
   final counts and evidence here.

**Complexity gate:** finish at or below **30,000 nonblank production lines in
`pyfem/v3/` and 30,000 in `test/v3/`**, with `authoring.__all__` at or below 35
entries and no root solver facade. The budget is a maximum, not a target to fill.

**Final acceptance:** all maintained public flows, notebook execution, full
benchmark correctness matrix, clean candidate/control performance evidence,
both full digit-mode repository suites, all exact CI Ruff check/format commands,
and independent source-to-charter review pass on the same final commit. Every
D1-D4 symbol is absent or explicitly listed as a historical reference; every
remaining capability names a consumer and acceptance oracle.

## 6. Durable execution rules

- [program-playbook.md](program-playbook.md) defines deletion, independent-oracle,
  numerical, performance, and review discipline. It is part of this charter.
- A deletion is complete only when implementation, exports, adapters, live tests,
  benchmark routes, and support documentation agree. Preserving an oracle never
  requires preserving the deleted runtime.
- Legacy parity is one possible falsifier, never a requirement for arithmetic or
  architecture. Independent equations, analytic fields, finite/symbolic
  derivatives, literature, and corrected legacy behavior may supersede it with a
  recorded reason and tolerance.
- Raw-bit assertions follow `test/v3/conftest.py`: the reference platform selects
  raw-uint64 comparison; other platforms execute the documented tolerance branch,
  never skip. Tolerances cite observed cross-platform deviation.
- Performance evidence is workload-, revision-, environment-, and
  coverage-specific. Use matched candidate/base controls, keep cold and warm
  costs separate, and never promote a baseline under fleet load.
- The exact frontier lives only here. Historical mission cards and old "next"
  sections are provenance, not parallel dispatch queues.
