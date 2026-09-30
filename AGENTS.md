# PyFEM

PyFEM is a scientific finite-element library. The `v3` branch is a ground-up
rewrite; legacy behavior supplies requirements and numerical references, while
legacy class boundaries and prototype APIs are replaceable.

For v3 code or architectural decisions, use [the v3 guide](.agents/v3/AGENTS.md).
For migration status, use [the live record](.agents/v3/migration-execution.md).
A local documentation edit does not require reading the migration or design corpus.

## Worktree import trap

Always run the test suite as `python -m pytest` from the checkout root — never
bare `pytest` (or `uv run pytest`). From a git worktree, bare `pytest` imports
the **main checkout's** `pyfem` package (path-style editable install via
`.pth`, and `test/` has no `__init__.py`), while `python -m pytest` puts the
current checkout first on `sys.path`. The same holds for benchmark runs: use
`python -m bench.run` from the checkout root. See NUMBA_CACHING.md §7.
