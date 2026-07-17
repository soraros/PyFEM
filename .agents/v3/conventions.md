# v3 coding conventions

These tooling and readability conventions are active. [design.md](design.md) owns
architecture, state, kernel/backend, and performance policy; no convention below
may recreate a prototype-specific carrier or dispatch rule.

## Formatting and checks

- Use 2-space indentation under `pyfem/v3/` and `test/v3/` only.
- Use `pyfem/v3/ruff.toml` for production plus v3 tests.
- Use `test/v3/ruff.toml` to verify the test/benchmark-specific policy.

```bash
.venv/bin/ruff check pyfem/v3 test/v3 --config pyfem/v3/ruff.toml
.venv/bin/ruff check test/v3 --config test/v3/ruff.toml
.venv/bin/ruff format pyfem/v3 test/v3 --config pyfem/v3/ruff.toml
```

## Naming

| Kind | Style | Example |
|---|---|---|
| Module | `snake_case` | `assembly_plan.py` |
| Function | `snake_case` | `compile_program` |
| Semantic type | `PascalCase` | `CompiledModel` |
| Constant | `UPPER_SNAKE` | `DEFAULT_TOLERANCE` |
| Authored field/component | descriptive lower case | `displacement`, `temperature` |

Legacy `.pro` names such as `SmallStrainContinuum` belong only in input adapters or
an explicit compatibility registry. Do not leak them into compiled or solver APIs.

Use descriptive names at public, compiler, state, and I/O boundaries. Standard
mathematical names such as `u`, `du`, `k`, `r`, `b`, `sigma`, or `lambda` are welcome
inside a short numeric kernel when their meaning is immediate from the formula.
Do not shorten semantic carrier fields or provenance names.

## Typing and arrays

- Annotate public functions, immutable descriptors, compiled carriers, and state
  transitions.
- Use NumPy typing aliases consistently within a numeric module. Existing `F64` and
  `I32` aliases are prototype conveniences, not a mandate to narrow every compiled
  index to 32 bits.
- Production floating-point state is `float64` initially. The compiler must validate
  any selected index dtype before conversion.
- Compiler-owned array carriers use identity equality (`eq=False` or equivalent),
  structural freezing, owned/non-aliased storage, and read-only arrays.
- Do not use a frozen dataclass as evidence that contained arrays are immutable.
- Keep `pyfem/v3/py.typed` and make public type contracts precise as the new surface
  stabilizes.

## Python and dependencies

- The project and v3 development environment require Python 3.13+ on this branch.
- Install the committed environment with `uv sync`.
- Do not add a dependency for convenience inside a numeric kernel. Add runtime or
  optional dependencies only with a public-flow need and an explicit packaging
  decision.
- PySide6 is not in the current Intel development baseline; GUI availability is not
  a v3 core gate.

## Numeric code

- Begin with a clear NumPy or deliberately simple reference implementation that can
  serve as an oracle.
- Dispatch once in Python per explicit homogeneous contribution block. Never infer
  formulation, topology, field layout, or material behavior from rank/node count in
  a kernel.
- Numeric kernels accept arrays/scalars/output buffers; they do not inspect file
  metadata, string registries, dictionaries, or a whole compiled model.
- Keep accepted physical state out of kernels and scratch buffers. Trial output has
  one explicit owner.
- Use Numba, fusion, chunking, parallel loops, and hardware thresholds only after
  the public-flow measurement required by
  [design.md](design.md#12-performance-proof-policy).
- A few repeated formula lines at a JIT boundary are preferable to a dynamic
  abstraction that obscures the mathematical operation, but duplication is not an
  excuse for inconsistent conventions or state semantics.

## Tests and notebooks

- Put v3 tests under `test/v3/`; name dangerous cases after the invariant they
  disprove or protect.
- Legacy parity skims are numerical oracles, not target API/schema fixtures.
- Every optimized kernel keeps a reference comparison and every new architecture
  boundary gets a counterexample test.
- Keep diagnostic benchmarks out of ordinary pytest collection and report hardware,
  cold/warm preparation, public flow, memory, and verification status.
- Jupytext configuration lives in root `pyproject.toml`; v3 notebooks live under
  `notebooks/v3/` and exercise public APIs rather than private benchmark paths.
