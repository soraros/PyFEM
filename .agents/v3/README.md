# PyFEM v3 (branch `v3`)

**Start here:** [AGENTS.md](AGENTS.md) (committed guide for agents and humans).

Typed, data-oriented rewrite under `pyfem/v3/`. Requires **Python 3.13+**. Legacy `pyfem/` stays on 4-space Ruff; v3 uses **2-space** via `pyfem/v3/ruff.toml`.

## Quick commands

```bash
uv sync --group v3
uv run ruff check pyfem/v3 test/v3 --config pyfem/v3/ruff.toml
uv run ruff format pyfem/v3 test/v3 --config pyfem/v3/ruff.toml
uv run pytest test/v3 -q
```

## Docs in this folder

| File | Contents |
|------|----------|
| [architecture.md](architecture.md) | SoA mesh, assembly, solver flow |
| [conventions.md](conventions.md) | Naming, typing, Ruff |
| [parity.md](parity.md) | Skims A+B+C and regression targets |
| [roadmap.md](roadmap.md) | Phased delivery |

## Entry points

- `pyfem.v3.load_problem(path)` → `ProblemDefinition`
- `pyfem.v3.solve_linear(problem)` → global displacement `state`
