# PyFEM v3 (branch `v3`)

PyFEM v3 is being redesigned as a compiled, data-oriented finite-element library.
The current code is a runnable prototype and numerical-reference source, not the
target architecture.

**Start with [design.md](design.md), then read
[generic_core.md](generic_core.md).** The first records the authoritative
ownership/state/result invariants and the second is the authoritative post-Phase-1
semantic-IR amendment. Then read [AGENTS.md](AGENTS.md) for the working rules and
commands.

The full legacy-to-v3 migration follows
[Proofline](proofline.md), specialized in
[migration_workflow.md](migration_workflow.md), with current state in the sole live
[migration-execution.md](migration-execution.md) ledger. Structural refactors
outside that migration use [refactor_playbook.md](refactor_playbook.md).

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
| [design.md](design.md) | **Authoritative:** ownership, identity, state, contribution, and result invariants |
| [generic_core.md](generic_core.md) | **Authoritative amendment:** generic semantic IR, trust boundary, legacy-instance model, and next proof portfolio |
| [AGENTS.md](AGENTS.md) | **Active:** session routing and working rules |
| [proofline.md](proofline.md) | **Reusable field guide:** semi-automatic coordination, task integrity, callbacks, automation boundary, and dashboard projection |
| [migration_workflow.md](migration_workflow.md) | **Active:** legacy-to-v3 capability workflow, packet lifecycle, and completion contract |
| [migration-execution.md](migration-execution.md) | **Active:** sole restartable migration ledger and exact next action |
| [refactor_playbook.md](refactor_playbook.md) | **Active outside the migration:** generic structural refactor proof method |
| [conventions.md](conventions.md) | **Active where compatible with the design:** style, typing, tooling |
| [parity.md](parity.md) | Numerical-reference workflow; parity is evidence, not architecture proof |
| [evidence/2026-07-18-r0e-capability-inventory.md](evidence/2026-07-18-r0e-capability-inventory.md) | Immutable 154-row E0 capability inventory and 606-path coverage proof |
| [feature-parity.md](feature-parity.md) | Historical source list absorbed by R0-E; old statuses have no live authority |
| [architecture.md](architecture.md) | Historical snapshot of the prototype carrier |
| [roadmap.md](roadmap.md) | Superseded P0-P8 checklist |
| [WORKFLOW.md](WORKFLOW.md), [hardening.md](hardening.md) | Superseded Cursor-era execution process |
| [scaling.md](scaling.md), [plane_strain.md](plane_strain.md), [structural.md](structural.md), [tangent_assembly.md](tangent_assembly.md) | Historical benchmark evidence; hardware/policy claims are not contracts |

## Current implementation direction

The explicit Q8 linear slice is complete and frozen as a correctness oracle. It is
not the carrier stack to extend. The former P2-A mixed-Q8/Q4/T3 writer is
superseded and its temporary worktree no longer exists.

The R2-E/P2-H/R2-F portfolio and bounded S2-A polish pass are complete. The
unexported three-operator proof is integrated, simplified to 649 production and
360 focused-test nonblank lines, and independently reviewed GO with no findings.
The migration is paused at the boundary recorded in
[generic_core.md](generic_core.md#11-current-bounded-simplification). The future
direct vertical cut is recorded but remains undispatched until explicit resume.
