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

## Ruff gates

Ruff gates use the PINNED `.venv` ruff (`.venv/bin/ruff` or `uv run ruff`;
floor in pyproject's dev group) — never the PATH ruff, which may be older and
disagree. The per-tree configs (`pyfem/v3/ruff.toml`, `test/v3/ruff.toml`,
`bench/ruff.toml`, `notebooks/v3/ruff.toml`) are invocation-invariant by
design: both bare (`ruff check pyfem/v3 test/v3 bench notebooks/v3`) and
explicit `--config` invocations (see .agents/v3/conventions.md) must be green.
The `known-first-party` / `known-third-party` isort entries and the dual
per-file-ignores patterns in those configs pin import classification and
ignore matching under both of ruff's project-root resolutions — do not drop
them.

## Bitwise pins and the reference platform

Byte-identity (raw-uint64) numerical pins are defined on one reference
platform: the machine recorded in the bench manifests
(`bench/results/*.json` — macOS x86_64 i9-9980HK, the pyproject-pinned
numba/numpy, Accelerate BLAS). On any other platform the same assertions run
through the documented tolerance branch of `assert_bitwise_pin` in
`test/v3/conftest.py`: the platform selects the comparison, it never skips
it, and each call site's tolerance must exceed the observed cross-platform
deviation, with that deviation cited in a comment.
