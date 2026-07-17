# PyFEM v3 (branch `v3`)

PyFEM v3 is being redesigned as a compiled, data-oriented finite-element library.
The current code is a runnable prototype and numerical-reference source, not the
target architecture.

**Start with [design.md](design.md).** It records the authoritative first-principles
design, invariants, dangerous-case proof suite, and phased migration. Then read
[AGENTS.md](AGENTS.md) for the working rules and commands.

V3 requires Python 3.13+ and uses 2-space Ruff. The branch's modernization and
Intel-Mac dependency baseline are intentional. PySide6 is not in the current
baseline, so the legacy `pyfem-gui` entrypoint requires a separate future optional-
dependency decision and is not a v3 gate.

## Quick commands

```bash
uv sync
.venv/bin/ruff check pyfem/v3 test/v3 --config pyfem/v3/ruff.toml
.venv/bin/ruff check test/v3 --config test/v3/ruff.toml
.venv/bin/python -m pytest -q test/v3
```

## Document status

| File | Status and purpose |
|---|---|
| [design.md](design.md) | **Authoritative:** target architecture, invariants, proof, migration |
| [AGENTS.md](AGENTS.md) | **Active:** session routing and working rules |
| [conventions.md](conventions.md) | **Active where compatible with the design:** style, typing, tooling |
| [parity.md](parity.md) | Numerical-reference workflow; parity is evidence, not architecture proof |
| [feature-parity.md](feature-parity.md) | Legacy requirements inventory; not a delivery sequence |
| [architecture.md](architecture.md) | Historical snapshot of the prototype carrier |
| [roadmap.md](roadmap.md) | Superseded P0-P8 checklist |
| [WORKFLOW.md](WORKFLOW.md), [hardening.md](hardening.md) | Superseded Cursor-era execution process |
| [scaling.md](scaling.md), [plane_strain.md](plane_strain.md), [structural.md](structural.md), [tangent_assembly.md](tangent_assembly.md) | Historical benchmark evidence; hardware/policy claims are not contracts |

## Current implementation direction

The next slice is deliberately narrow: authored and compiled carriers, one explicit
Q8 plane-stress contribution block, one compiled affine constraint/load program,
one composed prepared assembly plan, the minimal physical/evolution state
transaction, a typed linear-static request, and a result that independently
re-verifies constraints and equilibrium. The dangerous cases in
[design.md](design.md#11-falsifiable-acceptance-suite) must drive it before more
element or solver breadth is added.
