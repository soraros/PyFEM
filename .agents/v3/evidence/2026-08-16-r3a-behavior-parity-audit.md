# R3-A — PyFEM v1-to-v3 behavioral parity audit

## Result

At exact base `04baa4a79cbcddff888c57b239df5202e2b41424` (abbreviated
`04baa4a`), v3 does not have row-wide behavioral parity with v1. The honest
row grades are:

| Grade | Meaning used here | Rows |
|---:|---|---:|
| 1 | Public behavior implemented and independently verified | 4 |
| 2 | Implemented in a frozen/internal slice, but not general/public parity | 34 |
| 3 | Executable experimental proof only | 19 |
| 4 | Design-representable, not implemented | 82 |
| 5 | Absent/deferred, blocked, or design evidence still uncertain | 8 |
| 6 | Intentionally retired/replaced; not a parity gap | 7 |
| **Total** |  | **154** |

The grade is deliberately stricter than the R2-E six-concept mapping. R2-E
showed that 95 rows were direct and 56 composed, but that is representation
evidence, not implementation evidence. The current repository has only a
small Q8 authored-to-verified-result slice plus an executable generic proof and
the older v3 prototype.

SHA-256 (report bytes excluding this self-reference line): `51e61cd9d0cbd68fe8b62f4c2ef2c72f264cf780ad51d8e67f6ce64bbcc755b0`.
The terminal callback also supplies the exact final-file digest.

## Method and evidence boundary

I first required an exact clean base. `git rev-parse HEAD` was
`04baa4a79cbcddff888c57b239df5202e2b41424`; `git -c core.fsmonitor=false
status --porcelain=v1 --untracked-files=all` was empty. I read the mandated
agent guide, generic semantic core, current/frontier and capability-ledger
sections, and the complete 154-row R0-E inventory. I inspected current
`pyfem/v3` modules/tests and legacy implementation/docs only to establish the
behavior actually present.

Evidence vocabulary follows the migration workflow: E0 is source/example
existence; E1 is independently reproducible behavior; E2 is a target component
with edge cases but no complete public flow; E3 is a public
authored-to-verified-result flow; E4 additionally covers compatibility,
representative performance, docs/examples, and relevant repository gates.
The six grades above are this audit's parity grades, not replacements for E0-E4.

Observed verification: `/Users/sora/Projects/python/PyFEM/.venv/bin/python -m
pytest -q test/v3` passed **485 tests** at this base. The 40 SciPy
`SparseEfficiencyWarning` messages are existing warnings, not parity evidence
or a correctness failure. The exact Q8 path is component-qualified E3 in the
ledger, not row-wide parity. The seven focused generic-core tests prove Q4/T3,
directional spring, point load, typed channels, native-width blocks,
attribution, detachment, and literal oracles, but remain an experimental
internal proof.

## Progress measures

These measures intentionally answer different questions:

| Measure | Numerator / denominator | Result | Interpretation |
|---|---:|---:|---|
| Strict public behavioral parity | grade 1 / 154 | **4 / 154 = 2.6%** | Complete named rows with a public v3 path and independent behavior evidence. |
| Bounded executable coverage | grades 1–3 / 154 | **57 / 154 = 37.0%** | Includes frozen Q8/internal slices and experimental prototype proofs; not a compatibility claim. |
| Current unblocked design-backed coverage | grades 1–4 / 154 | **139 / 154 = 90.3%** | Implemented or explicitly represented by the generic-core design; design is not execution. |
| R2-E numerical representation mapping | direct + composed / 154 | **151 / 154 = 98.1%** | Historical architecture mapping only: 2 questionable and 1 rejected carrier remain. |

The first percentage is the only strict parity measure. The third and fourth
must not be read as “implemented.” Six public retirement candidates remain
blocked, and `MODEL-RVE-HOMOG` plus `ROM-POD` remain uncertain in the design
classification.

## Counts by domain

| Domain | Total | G1 | G2 | G3 | G4 | G5 | G6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| adapter | 14 | 0 | 0 | 8 | 5 | 1 | 0 |
| analysis | 11 | 0 | 2 | 2 | 6 | 0 | 1 |
| assembly | 9 | 0 | 6 | 0 | 3 | 0 | 0 |
| compiler | 1 | 0 | 1 | 0 | 0 | 0 | 0 |
| ecosystem | 10 | 0 | 0 | 0 | 6 | 4 | 0 |
| formulation | 22 | 2 | 1 | 1 | 18 | 0 | 0 |
| kernel | 5 | 0 | 0 | 4 | 1 | 0 | 0 |
| material | 24 | 2 | 2 | 1 | 19 | 0 | 0 |
| mesh | 6 | 0 | 6 | 0 | 0 | 0 | 0 |
| model | 9 | 0 | 0 | 0 | 8 | 1 | 0 |
| program | 6 | 0 | 3 | 1 | 2 | 0 | 0 |
| result | 8 | 0 | 1 | 0 | 6 | 1 | 0 |
| rom | 5 | 0 | 0 | 0 | 2 | 1 | 2 |
| section | 4 | 0 | 0 | 0 | 4 | 0 | 0 |
| state | 7 | 0 | 5 | 0 | 2 | 0 | 0 |
| v3-foundation | 7 | 0 | 7 | 0 | 0 | 0 | 0 |
| v3-prototype | 6 | 0 | 0 | 2 | 0 | 0 | 4 |
| **Total** | **154** | **4** | **34** | **19** | **82** | **8** | **7** |

## Largest missing behavior clusters

1. **Stateful and multi-stage behavior:** nonzero material/formulation history,
   schedule ownership, restore/rebind, nodal accumulation, nonlinear rollback,
   explicit dynamics, staggered/multi-stage acceptance, and path-dependent
   outputs are not in the public generic flow.
2. **Formulation breadth:** beams, plates, shells, SLS, axisymmetric response,
   cohesive/interface, phase field, and thermal/thermomechanical coupling are
   design mappings only.
3. **Material breadth:** plasticity, damage, cohesive laws, viscosity,
   crystal plasticity, sintering, sandwich/transverse laws, multi-material
   dispatch, and failure criteria have no v3 implementation evidence.
4. **Ecosystem and adapters:** root `pyfem.run`, full `.pro`/`.dat` grammar,
   includes/overrides, Gmsh, CLI contract, writers, logging, docs migration,
   GUI, and serialization compatibility are unresolved or absent.
5. **RVE/FE2 and ROM:** periodic/homogenized/nested solve semantics, snapshot
   schemas, POD, reduced solve, restart, and cost behavior remain unimplemented
   or uncertain.

## Exact rows by grades 1–3 and 6

### Grade 1 — public behavior independently verified (4)

`FORM-TRUSS`, `FORM-SPRING`, `MAT-PLANE-STRAIN`, `MAT-ISOTROPIC`.

Evidence: public v3 prototype element/constitutive paths with independent
reference/parity tests in `test/v3/test_link2_element.py`,
`test/v3/test_parity_shallow_truss_riks.py`, `test/v3/test_plane_strain.py`,
`test/v3/test_isotropic.py`, and the 3D parity/stiffness tests. These are
complete for the named narrow rows, not evidence that the whole structural or
material portfolio is complete.

### Grade 2 — frozen/internal slice, not general parity (34)

`MESH-NODES`, `MESH-CELLS`, `MESH-GROUPS`, `COMP-MESH`, `COMP-REGISTRY`,
`COMP-DOF`, `PROG-DIRICHLET`, `PROG-MPC`, `PROG-NODAL-LOAD`, `ASM-COO`,
`ASM-PREPARE`, `ASM-INTERNAL`, `ASM-EXTERNAL`, `ASM-TANGENT`, `ASM-GATHER`,
`STATE-GLOBAL`, `STATE-LAYOUT`, `STATE-TRIAL`, `STATE-TRANSACTION`,
`STATE-EVOLUTION`, `FORM-SMALL-CONT`, `MAT-DISPATCH`, `MAT-PLANE-STRESS`,
`ANAL-DISPATCH`, `ANAL-LINEAR`, `RES-SOLUTION`, `V3-AUTHORED-SPEC`,
`V3-SPEC-NORMALIZE`, `V3-ARRAY-OWNERSHIP`, `V3-LIVE-ID`, `V3-CONTENT-ID`,
`V3-GENERATION`, `V3-REGISTRY-SNAPSHOT`, `COMP-MODEL`.

Evidence: `pyfem/v3/compile/**`, `pyfem/v3/assembly/**`,
`pyfem/v3/analysis/**`, `pyfem/v3/results/**`, `pyfem/v3/spec/**`,
`pyfem/v3/model/**`, and `test/v3/test_v3_model_compile.py`,
`test/v3/test_v3_program_compile.py`, `test/v3/test_v3_assembly_plan.py`,
`test/v3/test_v3_linear_analysis.py`, and
`test/v3/test_v3_model_identity.py`. The proof is an exact detached Q8,
small-strain, constant-tangent, algebraic linear slice. It does not cover
general widths, fields, history, schedules, adapters, writers, or all public
v1 inputs.

### Grade 3 — executable experimental proof only (19)

`ADP-PRO`, `ADP-DAT`, `ADP-DAT-NODES`, `ADP-DAT-ELEMENTS`,
`ADP-DAT-DIRICHLET`, `ADP-DAT-MPC`, `ADP-DAT-LOADS`, `ADP-TOML`,
`KERN-SHAPES`, `KERN-QUADRATURE`, `KERN-TRANSFORM`, `KERN-KINEMATICS`,
`PROG-SCHEDULE`, `FORM-FINITE-CONT`, `MAT-HOOKE`, `ANAL-NONLINEAR`,
`ANAL-RIKS`, `V3-PROTOTYPE-FIXTURES`, `V3-PROTOTYPE-NOTEBOOK`.

Evidence: bounded source and tests in `pyfem/v3/io/**`, `pyfem/v3/fem/**`,
`pyfem/v3/solver/**`, `pyfem/v3/materials/**`, `pyfem/v3/mesh/**`, and the
corresponding `test/v3/test_parity_*.py`, `test/v3/test_dat_*.py`,
`test/v3/test_element_stiffness*.py`, `test/v3/test_quadrature_numba.py`,
`test/v3/test_nonlinear_solver.py`, and `test/v3/test_v3_generic_core.py`.
These are useful executable experiments and selected v1 comparisons, but they
run through the legacy-shaped prototype carrier or narrow helpers and do not
constitute the target general public v3 flow.

### Grade 6 — intentionally retired/replaced (7)

`ANAL-MODAL-DUP`, `ROM-LINEAR-MANIFOLD`, `ROM-QUADRATIC-MANIFOLD`,
`V3-PROTOTYPE-CARRIER`, `V3-PROTOTYPE-REGISTRY`, `V3-PROTOTYPE-ASSEMBLY`,
`V3-PROTOTYPE-ANALYSIS`.

Evidence: the live ledger's internal/duplicate retirement set and
`generic_core.md`'s explicit rejection of the flat carrier, string registry,
duplicate whole-problem assembly bridge, and direct prototype solve surface.
The modal duplicate and two manifold methods are not selected as separate
v3 capability obligations. This is distinct from the six public retirement
candidates, which remain grade 5 because approval and compatibility/loss
statements are still absent.

## Complete 154-row grade ledger

Evidence keys: **P** = independent public primitive/element oracle;
**Q** = exact Q8 Phase-1 authored-to-verified slice only; **X** = bounded
prototype/parser experiment; **D** = generic-core/R2-E representation only,
without implementation; **U** = blocked/deferred or uncertain mapping;
**R** = intentional internal/duplicate retirement replacement. The row-level
grade and key below preserve every original capability ID.

| Capability ID | Grade | Evidence |
|---|---:|:---:|
| ECO-PACKAGE | 4 | D |
| ECO-ROOT-RUN | 4 | D |
| ECO-ROOT-MESH-API | 5 | U |
| ECO-PYTHON-API | 5 | U |
| ECO-CLI | 4 | D |
| ECO-CLI-OVERRIDES | 4 | D |
| ECO-GUI | 5 | U |
| ECO-DOCS | 4 | D |
| ECO-LOGGING | 4 | D |
| ECO-ARCHIVES | 5 | U |
| ADP-PRO | 3 | X |
| ADP-PRO-INCLUDE | 4 | D |
| ADP-PRO-PARAM | 4 | D |
| ADP-DAT | 3 | X |
| ADP-DAT-NODES | 3 | X |
| ADP-DAT-ELEMENTS | 3 | X |
| ADP-DAT-GROUPS | 4 | D |
| ADP-DAT-DIRICHLET | 3 | X |
| ADP-DAT-MPC | 3 | X |
| ADP-DAT-LOADS | 3 | X |
| ADP-GMSH | 4 | D |
| ADP-GMSH-CONVERT | 4 | D |
| ADP-TOML | 3 | X |
| ADP-PICKLE-IN | 5 | U |
| MESH-NODES | 2 | Q |
| MESH-CELLS | 2 | Q |
| MESH-GROUPS | 2 | Q |
| COMP-MESH | 2 | Q |
| COMP-REGISTRY | 2 | Q |
| COMP-DOF | 2 | Q |
| KERN-SHAPES | 3 | X |
| KERN-QUADRATURE | 3 | X |
| KERN-BEZIER | 4 | D |
| KERN-TRANSFORM | 3 | X |
| KERN-KINEMATICS | 3 | X |
| PROG-DIRICHLET | 2 | Q |
| PROG-MPC | 2 | Q |
| PROG-SCHEDULE | 3 | X |
| PROG-LOAD-CASES | 4 | D |
| PROG-NODAL-LOAD | 2 | Q |
| PROG-DISTRIBUTED-LOAD | 4 | D |
| ASM-COO | 2 | Q |
| ASM-PREPARE | 2 | Q |
| ASM-INTERNAL | 2 | Q |
| ASM-EXTERNAL | 2 | Q |
| ASM-TANGENT | 2 | Q |
| ASM-MASS | 4 | D |
| ASM-DISSIPATION | 4 | D |
| ASM-MODEL-ACTIONS | 4 | D |
| ASM-GATHER | 2 | Q |
| STATE-GLOBAL | 2 | Q |
| STATE-LAYOUT | 2 | Q |
| STATE-TRIAL | 2 | Q |
| STATE-TRANSACTION | 2 | Q |
| STATE-EVOLUTION | 2 | Q |
| STATE-RESTORE | 4 | D |
| STATE-NODAL-ACCUM | 4 | D |
| FORM-SMALL-CONT | 2 | Q |
| FORM-SMALL-AXI | 4 | D |
| FORM-FINITE-CONT | 3 | X |
| FORM-FINITE-AXI | 4 | D |
| FORM-TRUSS | 1 | P |
| FORM-SPRING | 1 | P |
| FORM-BEAM-LIN | 4 | D |
| FORM-BEAM-NL | 4 | D |
| FORM-BEAM3D | 4 | D |
| FORM-KIRCHHOFF | 4 | D |
| FORM-TIMOSHENKO | 4 | D |
| FORM-PLATE | 4 | D |
| FORM-RM-SHELL | 4 | D |
| FORM-SLS | 4 | D |
| FORM-INTERFACE | 4 | D |
| FORM-PHASEFIELD | 4 | D |
| FORM-THERMAL | 4 | D |
| FORM-THERMAL-AXI | 4 | D |
| FORM-THERMOMECH | 4 | D |
| FORM-THERMOMECH-AXI | 4 | D |
| FORM-THERMAL-BC | 4 | D |
| FORM-THERMAL-SURFACE | 4 | D |
| SEC-BEAM | 4 | D |
| SEC-LAMINATE | 4 | D |
| SEC-SLS | 4 | D |
| SEC-CONDENSE | 4 | D |
| MAT-DISPATCH | 2 | Q |
| MAT-HOOKE | 3 | X |
| MAT-PLANE-STRESS | 2 | Q |
| MAT-PLANE-STRAIN | 1 | P |
| MAT-ISOTROPIC | 1 | P |
| MAT-TRANSVERSE | 4 | D |
| MAT-SANDWICH | 4 | D |
| MAT-DUMMY | 4 | D |
| MAT-MULTI | 4 | D |
| MAT-PLASTIC-ISO | 4 | D |
| MAT-PLASTIC-KIN | 4 | D |
| MAT-DAMAGE | 4 | D |
| MAT-COH-POWER | 4 | D |
| MAT-COH-THOULESS | 4 | D |
| MAT-COH-XU | 4 | D |
| MAT-VISCOELASTIC | 4 | D |
| MAT-VISCOPLASTIC | 4 | D |
| MAT-CRYSTAL | 4 | D |
| MAT-SINTER | 4 | D |
| MAT-FAIL-MAX-STRESS | 4 | D |
| MAT-FAIL-MAX-STRAIN | 4 | D |
| MAT-FAIL-TSAI-WU | 4 | D |
| MAT-FAIL-LARC03 | 4 | D |
| MAT-FAIL-VON-MISES | 4 | D |
| ANAL-DISPATCH | 2 | Q |
| ANAL-LINEAR | 2 | Q |
| ANAL-NONLINEAR | 3 | X |
| ANAL-RIKS | 3 | X |
| ANAL-DISSIPATED | 4 | D |
| ANAL-EXPLICIT | 4 | D |
| ANAL-DYN-EIG | 4 | D |
| ANAL-MODAL-DUP | 6 | R |
| ANAL-BUCKLING | 4 | D |
| ANAL-STAGGERED | 4 | D |
| ANAL-MULTI | 4 | D |
| MODEL-DISPATCH | 4 | D |
| MODEL-CONTACT | 4 | D |
| MODEL-RVE-BOUNDARY | 4 | D |
| MODEL-RVE-PERIODIC | 4 | D |
| MODEL-RVE-PRESCRIBED | 4 | D |
| MODEL-RVE-HOMOG | 5 | U |
| MODEL-FE2-SOLVE | 4 | D |
| MODEL-FE2-TANGENT | 4 | D |
| MODEL-FE2-OUTPUT | 4 | D |
| RES-SOLUTION | 2 | Q |
| RES-MANAGER | 4 | D |
| RES-TEXT | 4 | D |
| RES-GRAPH | 4 | D |
| RES-CONTOUR | 4 | D |
| RES-VTK | 4 | D |
| RES-HDF5 | 4 | D |
| RES-PICKLE | 5 | U |
| ROM-SNAPSHOT | 4 | D |
| ROM-POD | 5 | U |
| ROM-REDUCED | 4 | D |
| ROM-LINEAR-MANIFOLD | 6 | R |
| ROM-QUADRATIC-MANIFOLD | 6 | R |
| V3-AUTHORED-SPEC | 2 | Q |
| V3-SPEC-NORMALIZE | 2 | Q |
| V3-ARRAY-OWNERSHIP | 2 | Q |
| V3-LIVE-ID | 2 | Q |
| V3-CONTENT-ID | 2 | Q |
| V3-GENERATION | 2 | Q |
| V3-REGISTRY-SNAPSHOT | 2 | Q |
| V3-PROTOTYPE-CARRIER | 6 | R |
| V3-PROTOTYPE-REGISTRY | 6 | R |
| V3-PROTOTYPE-ASSEMBLY | 6 | R |
| V3-PROTOTYPE-ANALYSIS | 6 | R |
| V3-PROTOTYPE-FIXTURES | 3 | X |
| V3-PROTOTYPE-NOTEBOOK | 3 | X |
| COMP-MODEL | 2 | Q |

## Highest-leverage next ten capability slices

These are slices after the generic production cut, ordered by dependency and
behavioral leverage rather than by legacy directory size:

1. **Mixed native-width public solve:** Q8 + Q4 + T3 in one compiled system,
   preserving independent assembly, balance, provenance, and result checks.
2. **Production operator-block cut:** move the simplified generic proof into
   the target compiler/assembly path and delete the duplicated prototype bridge.
3. **General discrete spaces:** mechanical + thermal fields with nonstandard
   component ordering and no padded inactive DOFs.
4. **One real stateful operator:** repeat, reject, and accept a local history
   block beside a stateless block through the same transaction boundary.
5. **Schedule and continuation contract:** typed time/load schedules,
   cutback/restart semantics, and accepted evolution identity.
6. **Distributed/boundary load channels:** facet/body/follower contributions
   with attribution and independent balance oracles.
7. **Structural operator slice:** beam or shell payload with explicit frame/
   section data, then a public static solve without analysis type branches.
8. **Thermal/thermomechanical slice:** capacity/conduction plus coupled field
   channels and a staged or monolithic accepted transition.
9. **Restore and result projections:** versioned state/coordinate-map restore,
   nodal accumulation, and one provenance-carrying text/VTK/HDF5 observation
   sink.
10. **RVE/FE2 or ROM decision slice:** choose one explicit auxiliary nested
    execution or versioned reduced-space artifact and prove state, restart,
    tangent, cost, and verification semantics before broad implementation.

## Uncertainties and smallest next action

Uncertainties are concentrated in the six public retirement decisions,
`MODEL-RVE-HOMOG`, `ROM-POD`, exact adapter grammar/compatibility, and whether
the old prototype public paths remain temporarily supported. I did not upgrade
any row solely because legacy code, examples, docs, a registry string, or a
generic-core diagram exists.

Smallest next action: adjudicate this report together with the direct production
cut refresh, then freeze one exact clean implementation base and execute item 1
as a single mixed-width public correctness slice. Do not start broad adapters,
writers, or feature-family ports before that slice establishes the production
operator/block seam.
