# v3 roadmap

## Done (milestone: patch_test8)

- [x] Agent docs (`.agents/v3/`)
- [x] `pyfem/v3` package skeleton + 2-space Ruff
- [x] SoA mesh + `.dat` reader + `problem.toml` loader
- [x] Plane stress + Q8 small-strain element + linear solver
- [x] Skim fixtures + parity test vs legacy

## Next

1. **Nonlinear** — Newton loop with typed `SolverState`; no `eval` load functions.
2. **More elements** — Quad4, Tria3 via registry; shared integration tables.
3. **Constraints** — Native v3 `Constrainer` (prescribed + MPC) without legacy import.
4. **I/O** — Safe `.pro` subset; optional meshio path for Gmsh.
5. **CI** — `pytest test/v3` job on branch `v3` with `uv sync --group v3`.
6. **Performance** — Widen Numba coverage; optional parallel element loop.

## Explicitly deferred

- GUI, VTK writers, `ModelManager`, ROM/multiphysics solvers.
- Replacing legacy modules in place on `main`.
