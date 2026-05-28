# v1 → v3 feature parity matrix

Legacy PyFEM (`pyfem/`, excluding `pyfem/v3/`) vs the typed v3 core. Status: **done** | **next** (current phase) | **deferred**.

Phases **P0–P8** are defined in [roadmap.md](roadmap.md). Parity skims live under `skims/<case>/`; see [parity.md](parity.md).

## 1. Core platform

| v1 capability | v3 target | Status | Skim / example | Notes |
|---------------|-----------|--------|----------------|-------|
| `.pro` + `.dat` input | `load_problem`, literal parsers | done | `patch_test8` | No `eval`; skim `.pro` only |
| `problem.toml` snapshot | `io/toml.py` | done | `patch_test8` | Canonical layer B |
| Type registry | `registry.py` | done | — | Maps legacy type strings |
| `ProblemDefinition` (arrays) | `types.py` | done | — | Jitable `NamedTuple` |
| Python 3.13+ | `pyfem.v3` import guard | done | — | Legacy stays 3.11+ |
| CI v3 tests | `uv sync --group v3` in main `test` job | done | `test/v3/` | Skips on 3.11/3.12 |

## 2. Linear solids (book ch.2)

| v1 capability | v3 target | Status | Skim / example | Notes |
|---------------|-----------|--------|----------------|-------|
| `LinearSolver` | `solve_linear` | done | `patch_test8` | SciPy `spsolve` |
| `SmallStrainContinuum` (Q8/Quad4/Tria3) | `fem/element.py` dispatch | done | `patch_test8`, `patch_test4`, `patch_test3` | Numba K_e + COO |
| `PlaneStress` | `plane_stress` | done | `patch_test8` | 3×3 D matrix |
| Prescribed displacements | `solver/constraints.py` | done | `patch_test8` | Native `C` reduction |
| Nodal forces (`ExternalForces`) | `external_load` in `ProblemDefinition` | done | `patch_test8_loaded` | P1 |
| Quad4 continuum | `quad4` in `fem/element.py` | done | `patch_test4` | P1 |
| Tria3 continuum | `tria3` in `fem/element.py` | done | `patch_test3` | P1 |
| MPC / ties | extend constraints | next | — | P1, after loads |
| `PlaneStrain` | `plane_strain` | deferred | — | P2 |
| 3D continuum (hex/tet) | 3D kernels | deferred | `PatchTest8_3D` | P2 |

## 3. Nonlinear static (ch.3–4)

| v1 capability | v3 target | Status | Skim / example | Notes |
|---------------|-----------|--------|----------------|-------|
| `NonlinearSolver` (Newton) | `solve_nonlinear` + `SolverState` | deferred | ch.3 cantilever | P3 |
| Load ramp / `loadCases` | typed load tables | deferred | — | P3 |
| `FiniteStrainContinuum` | TL kernel | deferred | ch.3 | P3 |
| `RiksSolver` | arc-length driver | deferred | ch.4 | P4 |
| Truss / spring (linear) | structural elements | deferred | ch.4 | P4 |

## 4. Dynamics & eigen

| v1 capability | v3 target | Status | Skim / example | Notes |
|---------------|-----------|--------|----------------|-------|
| `ExplicitSolver` | explicit integrator | deferred | ch.5 wave | P6 |
| `ModalSolver` / `DynEigSolver` | eigen solve | deferred | `examples/solver/dynEig` | P6 |
| `BuckEigSolver` | buckling eigen | deferred | `eulerBuck` | P6 |
| Mass assembly | lumped/consistent M | deferred | — | P6 |

## 5. Materials (ch.6+)

| v1 capability | v3 target | Status | Skim / example | Notes |
|---------------|-----------|--------|----------------|-------|
| Plasticity (iso hardening) | tangent + state | deferred | ch.6 damage example | P5 |
| `PlaneStrainDamage` | damage law | deferred | ch.6 | P5 |
| Cohesive zone (interface) | interface element + laws | deferred | ch.13 peel | P5 |
| Viscoelastic / viscoplastic | rate laws | deferred | `examples/materials/` | P5+ |
| Failure criteria | post-process hooks | deferred | — | P5+ |
| `MicroModel` (FE²) | — | deferred | — | Out of scope |

## 6. Structural elements

| v1 capability | v3 target | Status | Skim / example | Notes |
|---------------|-----------|--------|----------------|-------|
| Truss, Spring | 1D elements | deferred | ch.4 | P4 |
| Kirchhoff / Timoshenko / BeamNL | beam kernels | deferred | ch.9 | P4+ |
| `Plate`, `ReissnerMindlinShell`, SLS | shell/plate | deferred | `examples/plate/` | P4+ |

## 7. Multiphysics & models

| v1 capability | v3 target | Status | Skim / example | Notes |
|---------------|-----------|--------|----------------|-------|
| `PhaseField` + `StaggeredSolver` | coupled fields | deferred | phase-field examples | P7 |
| Thermo continuum family | thermal DOFs | deferred | thermo examples | P7 |
| `RVE` model | periodic BC homogenization | deferred | `examples/models/rve/` | P7 |
| `Contact` model | penalty contact | deferred | `examples/contact/` | P7 |

## 8. Workflow & I/O

| v1 capability | v3 target | Status | Skim / example | Notes |
|---------------|-----------|--------|----------------|-------|
| `pyfem` CLI on v3 | optional bridge | deferred | — | P8 |
| Gmsh / meshio | mesh loader | deferred | `examples/gmsh/` | P8 |
| `MeshWriter` (VTK) | VTK export | deferred | — | P8 |
| `HDF5Writer` | HDF5 export | deferred | — | P8 |
| `pyfem-gui` | — | deferred | — | Out of scope |

## 9. ROM

| v1 capability | v3 target | Status | Skim / example | Notes |
|---------------|-----------|--------|----------------|-------|
| `ROMSnapshotWriter` | snapshot I/O | deferred | `examples/rom/` | P8+ |
| `ROMBasisBuilder` / `ReducedOrderSolver` | POD + reduced solve | deferred | `examples/rom/` | P8+ |

## 10. Explicitly out of scope

- Replacing legacy modules in place on `main`
- Full `ModelManager` port
- `MicroModel` FE² nested solves
- `pyfem-gui` parity
