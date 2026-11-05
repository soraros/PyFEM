# v3 tooling

- Python 3.13+; use the repository `.venv`. `uv sync` installs the committed
  environment when setup or dependency synchronization is needed.
- V3 production and tests use **2-space** indentation. Their Ruff configurations
  differ; root formatting defaults do not apply to these trees.
- PySide6 is absent from the Intel-Mac baseline. The legacy `pyfem-gui` entrypoint
  needs separately installed GUI dependencies and is not a core development gate.
- Running legacy example scripts or decks directly (evidence runs, parity oracles)
  must be headless: prefix with `MPLBACKEND=Agg` — several call `plt.show()` and
  will otherwise pop windows on the operator's machine.
- Tests live in `test/v3/`; notebooks in `notebooks/v3/`. Jupytext is configured in
  root `pyproject.toml`. Benchmark workloads and timing drivers live in `bench/`
  and run through `python -m bench.run`; see `bench/README.md` for the maintained
  gates, warm, cold, and regression-check routes.

## Worktree environments

A git worktree can reuse the primary checkout's environment; it does not need
its own `.venv`. Activate the primary checkout's `.venv` and then change to
the actual checkout root, or use that environment's absolute `python` and
`ruff` binaries while remaining in the actual checkout root. For example:

```bash
cd /path/to/actual/checkout
/path/to/primary/.venv/bin/python -m pytest -q
/path/to/primary/.venv/bin/python -m bench.run gates
/path/to/primary/.venv/bin/ruff check pyfem/v3 test/v3 --config pyfem/v3/ruff.toml
```

Always invoke tests as `python -m pytest` and benchmarks as
`python -m bench.run` from the checkout being tested, never bare `pytest` or
`uv run pytest`: module invocation puts that checkout first on `sys.path`,
whereas the path-style editable install can otherwise import the primary
checkout's code. Ruff must still come from the pinned project environment,
not `PATH`; keep its existing per-tree configuration arguments unchanged.

## Commands

Run the affected tests for a local change; the suite commands below are available
for broader integration evidence, not prerequisites for every edit.

```bash
.venv/bin/python -m pytest -q test/v3/<affected_test>.py
.venv/bin/python -m pytest -q test/v3
.venv/bin/python -m pytest -q
.venv/bin/ruff check pyfem/v3 test/v3 --config pyfem/v3/ruff.toml
.venv/bin/ruff check test/v3 --config test/v3/ruff.toml
.venv/bin/ruff format --check <changed_v3_paths> --config pyfem/v3/ruff.toml
```

Compiler/identity diagnostics have regressions for Python's integer-string limit.
When those paths change, run the affected cases again with
`PYTHONINTMAXSTRDIGITS=640`. Read-only reviews can use
`PYTHONDONTWRITEBYTECODE=1` and pytest `-p no:cacheprovider` to avoid generated files.
Historical test counts describe their exact revisions, not a permanent minimum.
