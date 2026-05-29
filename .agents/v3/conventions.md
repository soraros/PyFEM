# v3 coding conventions

## Formatting

- **Indentation:** 2 spaces under `pyfem/v3/` and `test/v3/` only.
- **Ruff:** use nested config `pyfem/v3/ruff.toml` (extends repo root).

```bash
uv run ruff check pyfem/v3 test/v3 --config pyfem/v3/ruff.toml
uv run ruff format pyfem/v3 test/v3 --config pyfem/v3/ruff.toml
```

## Naming (PEP 8)

| Kind | Style | Example |
|------|--------|---------|
| Module | `snake_case` | `small_strain_quad8.py` |
| Function | `snake_case` | `assemble_stiffness` |
| Class | `PascalCase` | `ProblemDefinition` |
| Constant | `UPPER_SNAKE` | `DOF_TYPES_2D` |
| Legacy `.pro` names | Only in `registry.py` | `"SmallStrainContinuum"` |

Do not introduce Java-style names (`ContElem`) in public v3 APIs.

## Typing

- Annotate public functions and dataclass fields.
- Use **`F64`** / **`I32`** (`NDArray[np.float64]` / `NDArray[np.int32]`) from `types.py` for array annotations.
- **`ProblemDefinition`**: `typing.NamedTuple` of arrays only (jitable bundle).
- **`LoadedProblem`**: `dataclass` for metadata (strings, paths, registry).
- `frozen=True` on other immutable value objects (`PlaneStressMaterial`, `PlaneStrainMaterial`, `IsotropicMaterial`, `PrescribedDof`).
- Ship `pyfem/v3/py.typed` when the surface stabilizes.

## Python version

- **v3 requires Python 3.13+** (`pyfem.v3` raises on import otherwise).
- Legacy `pyfem` stays at project minimum 3.11.
- v3 dependency group uses `python_version >= '3.13'` markers.

## Dependencies

- Install v3 stack: `uv sync --group v3` (`jupytext`, `ipykernel`, `ipympl`, `ipywidgets`).

## Style

- Plain-Python FEM kernels use vectorized NumPy (`einsum`, broadcasting).
- ``@njit`` kernels use the nopython subset: 2D ``@``, slice assign inline (not
  ``einsum``); one ``prange`` site per top-level stiffness call — fused Q8 integrates
  ``B`` inside the element loop; Quad4/Tria3 still use staged kinematics + ``prange``
  on integration; COO scatter stays serial ``@njit`` with ``range``.
- Dev Numba benchmarks (not pytest-collected): ``test/v3/_bench_numba_stiffness.py``,
  ``test/v3/_bench_prange_investigation.py``, ``test/v3/_bench_solve_scale.py`` — run
  manually with ``uv run python ...``. See [scaling.md](scaling.md).
- **Scale-first performance:** optimize for mid-large meshes; no small-problem fallbacks
  or dual code paths without proven large-scale need ([scaling.md](scaling.md)).
- Name functions after the quantity they return (`strain_displacement`, `plane_stress_matrix`).
- Keep setup/I/O in plain Python; only numerical kernels need to be dense and array-oriented.
- Do not add v3-only deps to the default runtime list until v3 is promoted.

### Numba-safe abstraction

Unify and deduplicate code **without** patterns that block future `@njit` fusion or inlining.

**Python-only shell** (OK to use dicts, strings, dataclasses, `Callable`, dynamic `*args`):

- I/O, registry, `make_loaded`, `CachedLinearSystem`, COO buffer allocation, group loops in `assembly.py`.
- Dispatch from mesh metadata to a **fixed** kernel by rank / nodes-per-element / group kind.

**Inside or adjacent to `@njit` hot paths** (must stay nopython-safe):

- Array-only arguments; no closures, no dict-of-callables, no `Callable` parameters.
- Shared integration: one staged helper (e.g. `_integrate_btcb_batched`) called directly from each element kernel.
- Element-type dispatch: **`if` / `elif` on literal rank and node count**, calling `@njit` functions by name — not a runtime map of callables.
- State gather → kernel: allocate `element_states`, call `_gather_element_states`, then call the batched kernel **directly** (no generic `batch_fn(...)` wrapper).

**Rule of thumb:** if a refactor would need object mode or `numba.extending` to compile, keep the abstraction in the Python shell and duplicate a few lines at the `@njit` boundary instead.

### Short names in numerical kernels

In `@njit` FEM math and other hot, array-oriented code, **prefer short variable names** when they are unambiguous or match standard notation. The goal is **one line, dense with meaning** — avoid breaking expressions across lines just to satisfy long PEP 8 names.

| Context | Prefer | Over |
|---------|--------|------|
| Element stiffness / force | `ke`, `fe`, `k`, `f` | `element_stiffness`, `internal_force_local` |
| Local / global state | `a`, `u`, `da` | `element_displacement`, `displacement_increment` |
| Strain / stress scalars | `du`, `dv`, `eps`, `sig` | `axial_strain_increment`, `cauchy_stress` |
| Geometry | `l0`, `dx`, `dy` | `reference_length`, `delta_x` |
| Load / arc-length | `lam`, `dlam`, `fhat` | `load_factor`, `external_load_reference` |
| B-matrix / shape | `bl`, `N`, `dN` | `strain_displacement_row`, `shape_function` |
| Loop indices | `e`, `gp`, `i`, `j` | `elem_index`, `gauss_point` |

**Where this applies:** inside `fem/*` kernels, assembly inner loops, Newton/Riks increment math, and staged helpers that mirror textbook symbols.

**Where it does not apply:** public APIs, I/O, dataclass fields, registry keys, and test names — keep those descriptive (`assemble_tangent_loaded`, `ProblemDefinition`, `build_truss_fan`).

**Rule of thumb:** if a reviewer would recognize the symbol from FEM texts (B, K, u, λ, ε, σ) or from the immediately surrounding three lines, shorten it. If the name would need a comment to decode, spell it out.

Example (good — matches math, stays on one line):

```python
kl = youngs_modulus * area * l0 * np.outer(bl, bl)
f_bar = l0 * sigma * area * bl
```

Example (avoid in kernels — forces needless wraps):

```python
linear_stiffness_matrix = (
  youngs_modulus * cross_section_area * reference_length * np.outer(strain_displacement_row, strain_displacement_row)
)
```

## Notebooks

- Jupytext config in root `pyproject.toml` (`py:percent` ↔ `ipynb`).
- Notebooks live under `notebooks/v3/`.

## Tests

- Parity tests live in `test/v3/` and compare against legacy `InputRead` + `LinearSolver`.
- Tolerances come from `skims/<case>/parity.toml`.
