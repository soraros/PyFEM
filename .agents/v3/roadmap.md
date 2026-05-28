# v3 roadmap

## Done (milestone: patch_test8)

- [x] Agent docs (`.agents/v3/`)
- [x] `pyfem/v3` package skeleton + 2-space Ruff
- [x] SoA mesh + `.dat` reader + `problem.toml` loader
- [x] Plane stress + Q8 small-strain element + linear solver
- [x] Skim fixtures + parity test vs legacy
- [x] `@njit` Q8 element path (`shapes`, `kinematics`, `element`); `prange` only on quadrature in `element.py`
- [x] `@njit` COO stiffness scatter in `fem/assembly.py`
- [x] CI `test-v3` job (Python 3.13, `uv sync --group v3`, `pytest test/v3`)
- [x] Native prescribed-displacement constraints (`solver/constraints.py`)

## Next

1. **Nonlinear** — Newton loop with typed `SolverState`; no `eval` load functions.
2. **More elements** — Quad4, Tria3 via registry; shared integration tables.
3. **MPC constraints** — Ties and multi-point constraints without legacy `Constrainer`.
4. **I/O** — Safe `.pro` subset; optional meshio path for Gmsh.
5. **Performance** — Solver hot paths; optional CI perf smoke from `test/v3/_bench_*.py`.

## Explicitly deferred

- GUI, VTK writers, `ModelManager`, ROM/multiphysics solvers.
- Replacing legacy modules in place on `main`.
