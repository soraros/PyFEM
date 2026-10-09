# Contributing to PyFEM

Thank you for your interest in contributing to PyFEM.

PyFEM is an educational and research-oriented finite element code. Its main
purpose is to make finite element formulations, numerical algorithms, and
software structures clear and accessible. Contributions should therefore
prioritize correctness, readability, and educational value.

## Ways to Contribute

Contributions may include:

- reporting bugs;
- correcting or improving documentation;
- adding examples or exercises;
- improving tests;
- implementing new elements, material models, solvers, or output modules;
- improving code quality, usability, or portability.

Before starting a substantial change, please open a GitHub issue to describe
your proposal. This allows the maintainers and contributors to discuss the
scope and approach before significant work is invested.

## Reporting Bugs

Please use [GitHub Issues](https://github.com/jjcremmers/PyFEM/issues) to
report bugs. A useful bug report includes:

- a clear description of the problem;
- the PyFEM version or commit used (`pyfem --version` or `git rev-parse HEAD`);
- the operating system and Python version;
- the input files or a minimal example needed to reproduce the problem;
- the expected behaviour;
- the actual behaviour, including relevant error messages or output.

Please remove confidential or proprietary data before attaching files.

Security vulnerabilities should not be reported in a public issue. Follow the
instructions in [`SECURITY.md`](SECURITY.md) instead.

## Requesting Features

Feature requests are welcome, particularly when they support education,
research, or the transparent implementation of finite element methods. Please
describe:

- the problem the feature would solve;
- its expected educational or scientific value;
- the proposed behaviour;
- relevant references, equations, or publications;
- possible alternatives, when applicable.

## Development Setup

PyFEM requires **Python 3.13 or newer** (see `requires-python` in
`pyproject.toml`).

Fork the repository on GitHub and clone your fork:

```bash
git clone https://github.com/<your-username>/PyFEM.git
cd PyFEM
```

The recommended development environment uses [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

This installs PyFEM in editable mode together with the default dependency
groups:

- `dev`: test and quality tools — pytest, coverage, ruff, and sympy (used for
  symbolic verification of derived formulations);
- `v3`: the scientific stack for the v3 rewrite (numba, Jupyter tooling).

If you prefer pip, create and activate a virtual environment first:

```bash
python3 -m venv .venv          # Linux / macOS
# py -m venv .venv             # Windows (cmd.exe / PowerShell)
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate.bat   # Windows (cmd.exe)
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install pytest coverage ruff
```

Create a focused branch for your contribution:

```bash
git checkout -b feature/short-description
```

Branch names such as `feature/add-new-element`, `fix/newton-convergence`,
`docs/material-model-example`, or `test/beam-elements` work well.

## Running the Tests

Run the full suite from the repository root:

```bash
uv run pytest
```

or a focused subset, e.g. `uv run pytest test/v3 -q`.

The v3 compiler and identity diagnostics have regressions for Python's
integer-string conversion limit. When those paths change, run the affected
tests in **both digit modes** — once normally and once with the limit raised:

```bash
uv run pytest
PYTHONINTMAXSTRDIGITS=640 uv run pytest
```

Numerical pins (byte-identity assertions) are defined on the reference
platform recorded in `bench/results/*.json`; on other platforms the same
assertions run through documented tolerances. See `AGENTS.md` and
`test/v3/conftest.py` for details.

When running example decks or scripts directly (outside pytest), set a
headless matplotlib backend so no plot windows appear:

```bash
MPLBACKEND=Agg uv run pyfem examples/ch02/PatchTest8.pro
```

## Lint and Format Gates

CI gates four source trees with ruff, each under its own configuration. Use
the ruff from your project environment (resolved from the dev-group floor
`ruff>=0.15.22`) — not a ruff from your PATH, which may be older and disagree.
The exact CI invocations, run from the repository root, are:

```bash
ruff check pyfem/v3 test/v3 --config pyfem/v3/ruff.toml
ruff check test/v3 --config test/v3/ruff.toml
ruff check bench --config bench/ruff.toml
ruff check notebooks/v3 --config notebooks/v3/ruff.toml
ruff format --check pyfem/v3 test/v3 --config pyfem/v3/ruff.toml
ruff format --check test/v3 --config test/v3/ruff.toml
ruff format --check bench --config bench/ruff.toml
ruff format --check notebooks/v3 --config notebooks/v3/ruff.toml
```

The format gates cover the whole tree, not just the files you changed. The
legacy trees (the top-level `pyfem/` modules outside `pyfem/v3/` and the
root-level `test/` files) predate these gates and are not yet clean under
them; please keep them out of unrelated clean-up commits.

## Code Guidelines

PyFEM is intended to be readable by students, researchers, and developers.
Please keep implementations as clear and direct as reasonably possible.

Contributions should:

- follow the ruff-enforced style (line length 88, `target-version = "py313"`);
- use descriptive names for classes, functions, and variables;
- prefer clarity over cleverness — this code is read for teaching;
- include tests for new elements, materials, solvers, and I/O modules;
- update documentation when behaviour or interfaces change.

## Documentation

The documentation is built with Sphinx, the same way as CI and Read the Docs:

```bash
uv sync --extra docs --no-dev
uv run sphinx-build -M html doc doc/_build
```

Open `doc/_build/html/index.html` to inspect the result.

## Commits and Pull Requests

- Keep commits focused; one logical change per commit.
- Commit messages follow the conventional-prefix style used in the history,
  e.g. `feat(v3): ...`, `fix: ...`, `docs: ...`, `test: ...`.
- Push your branch to your fork and open a pull request against `main`.
- Describe the change, link the motivating issue, and make sure all CI jobs
  (Ruff gates, tests, package build, documentation build) are green.

## License

PyFEM is distributed under the MIT License. By contributing, you agree that
your contributions are licensed under the same license.
