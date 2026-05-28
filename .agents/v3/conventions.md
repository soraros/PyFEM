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
- `frozen=True` on other immutable value objects (`PlaneStressMaterial`, `PrescribedDof`).
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
- Name functions after the quantity they return (`strain_displacement`, `plane_stress_matrix`).
- Keep setup/I/O in plain Python; only numerical kernels need to be dense and array-oriented.
- Do not add v3-only deps to the default runtime list until v3 is promoted.

## Notebooks

- Jupytext config in root `pyproject.toml` (`py:percent` ↔ `ipynb`).
- Notebooks live under `notebooks/v3/`.

## Tests

- Parity tests live in `test/v3/` and compare against legacy `InputRead` + `LinearSolver`.
- Tolerances come from `skims/<case>/parity.toml`.
