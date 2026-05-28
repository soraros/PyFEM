# PyFEM v3 agent guide

Branch `v3`: typed, data-oriented FEM core under `pyfem/v3/`. Legacy `pyfem/` remains on Python 3.11+ with 4-space Ruff; **v3 requires Python 3.13+** and uses 2-space Ruff.

## Setup

```bash
uv sync --group v3    # jupyter stack (3.13+ only)
uv run pytest test/v3 -q
uv run ruff check pyfem/v3 test/v3 --config pyfem/v3/ruff.toml
```

## Public API

- `pyfem.v3.load_problem(path)` → `LoadedProblem` (metadata + `problem`)
- `pyfem.v3.solve_linear(loaded)` — global displacement `state`
- `pyfem.v3.pack_problem(...)` — build `ProblemDefinition` from mesh/DOF data

Importing `pyfem.v3` on Python &lt; 3.13 raises `ImportError`.

## Architecture

| Layer | Location | Notes |
|-------|----------|--------|
| Schema | `types.py` | `ProblemDefinition` (`NamedTuple`, arrays only), `LoadedProblem` (metadata) |
| I/O | `io/dat.py`, `io/toml.py`, `io/legacy_pro.py` | Literal parsing only (no `eval`) |
| Registry | `registry.py` | Legacy `.pro` type strings → implementations |
| FEM math | `fem/quadrature.py`, `fem/shapes.py`, `fem/kinematics.py`, `fem/element.py` | `@njit` Gauss rules, N, B, K_e = ∫ Bᵀ C B; `prange` over elements when `n_elems ≥ 32` |
| Assembly | `fem/assembly.py` | Batched element stiffness + `@njit` COO scatter (`prange` when `n_elems ≥ 32`) |
| Solver | `solver/linear.py` | SciPy `spsolve` + `coo_array`; legacy `Constrainer` for BCs |

`ProblemDefinition` is a **`NamedTuple`** of `F64`/`I32` fields — pass it to `@njit` directly or unpack arrays. No jitclass wrapper.

## Conventions

- PEP 8 `snake_case` / `PascalCase`; legacy names only in `registry.py`
- Annotate arrays with **`F64`** / **`I32`**; use `np.float64` / `np.int32` for `dtype=`
- Ruff: `pyfem/v3/ruff.toml` (`indent-width = 2`, `target-version = "py313"`)

## Parity (skims)

| Layer | File |
|-------|------|
| A | `skims/<case>/skim.pro` → legacy `InputRead` |
| B | `skims/<case>/problem.toml` → v3 canonical |
| C | `ProblemDefinition` NamedTuple |

First case: `skims/patch_test8/` vs `examples/ch02/PatchTest8.*`. See [parity.md](parity.md).

## Notebooks

- Jupytext: `py:percent` ↔ `ipynb` ([`tool.jupytext`](../pyproject.toml))
- Example: [`notebooks/v3/patch_test8.py`](../../notebooks/v3/patch_test8.py)

## Roadmap

See [roadmap.md](roadmap.md).
