# v3 roadmap

Phased delivery toward v1 feature parity. Each phase exits when its skim parity tests pass (see [parity.md](parity.md) and [feature-parity.md](feature-parity.md)).

## P0 — Reference linear patch (done)

- [x] Agent docs (`.agents/v3/`)
- [x] `pyfem/v3` package skeleton + 2-space Ruff
- [x] SoA mesh + `.dat` reader + `problem.toml` loader
- [x] Plane stress + Q8 small-strain element + linear solver
- [x] Skim `patch_test8` + parity test vs legacy
- [x] `@njit` Q8 element path; COO stiffness scatter
- [x] CI: `uv sync --group v3` in main `test` matrix; `pytest test/v3` (skips on Python &lt; 3.13)
- [x] Native prescribed-displacement constraints (`solver/constraints.py`)

## P1 — Linear 2D breadth (done)

- [x] Nodal loads from `<ExternalForces>` → `patch_test8_loaded` skim
- [x] Quad4 element + `patch_test4` skim
- [x] Tria3 element + `patch_test3` skim
- [x] MPC / multi-point ties (no legacy `Constrainer`)
- [x] Scale hardening: uniform Q8 patch generator, fused Q8 kernel, chunked assembly, factorized solve context, `_bench_solve_scale.py` ([scaling.md](scaling.md))

## P2 — Linear extensions

- [x] Plane strain material + scale hardening ([plane_strain.md](plane_strain.md))
- [x] 3D continuum (hex / tet)
- [x] Skim e.g. `PatchTest8_3D` when 3D kernel exists

## P3 — Nonlinear static

- [ ] `SolverState`, internal force + tangent assembly
- [ ] Newton loop; load factor / tables (no `eval`)
- [ ] Skims: ch.3 cantilever, ch.4 truss (subset)

## P4 — Path following & structures

- [ ] Riks arc-length
- [ ] Linear truss / beam elements
- [ ] ch.4 Riks examples (subset)

## P5 — Advanced materials

- [ ] Isotropic hardening plasticity
- [ ] Interface + cohesive laws (ch.13 peel subset)
- [ ] Continuum damage (ch.6 subset)

## P6 — Dynamics & eigen

- [ ] Mass assembly (lumped / consistent)
- [ ] Explicit dynamics
- [ ] Modal, dynamic, buckling eigen solvers

## P7 — Multiphysics & models

- [ ] Phase-field + staggered solve (1 reference case)
- [ ] Thermo-mechanical (1 reference case)
- [ ] RVE homogenization (1 reference case)
- [ ] Penalty contact (1 reference case)

## P8 — I/O & ergonomics

- [ ] VTK / HDF5 writers
- [ ] Richer safe `.pro` subset; optional Gmsh path
- [ ] Optional `pyfem` CLI bridge to v3

## Explicitly deferred (all phases)

- GUI, full `ModelManager`, ROM pipeline, FE² `MicroModel`
- Replacing legacy modules in place on `main`
