# v3 roadmap

## Done (milestone: patch_test8 + multi-element)

- [x] Agent docs (`.agents/v3/`)
- [x] `pyfem/v3` package skeleton + 2-space Ruff
- [x] SoA mesh + `.dat` reader + `problem.toml` loader
- [x] Plane stress + Q8 / Quad4 / Tria3 small-strain continuum + linear solver
- [x] Skim fixtures + parity tests vs legacy (`patch_test8`, `patch_test4`, `patch_test3`)
- [x] `@njit` element paths (`shapes`, `kinematics`, `element`); `prange` only on quadrature in `element.py`
- [x] `@njit` generic COO stiffness scatter in `fem/assembly.py`
- [x] CI runs `pytest test/v3` with `uv sync --group v3`
- [x] Native prescribed-displacement constraints (`solver/constraints.py`)

## Next

1. **Nonlinear** — Newton loop with typed `SolverState`; no `eval` load functions.
2. **MPC constraints** — Ties and multi-point constraints without legacy `Constrainer`.
3. **I/O** — Safe `.pro` subset; optional meshio path for Gmsh.
4. **Performance** — Fused assembly kernel; optional CI perf smoke from `test/v3/_bench_*.py`.

## Explicitly deferred

- GUI, VTK writers, `ModelManager`, ROM/multiphysics solvers.
- Replacing legacy modules in place on `main`.
