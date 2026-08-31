# PyFEM v3 (branch `v3`)

PyFEM v3 is being redesigned as a compiled, data-oriented finite-element library.
The current code is a runnable prototype and numerical-reference source, not the
target architecture.

**For a cold start, begin with the agent-neutral
[handoff.md](handoff.md).** It records the verified repository state, active
blocker, exact restart procedure, and authority order without requiring thread
history. It is a snapshot; changed lifecycle or next-action state belongs only in
the live [migration-execution.md](migration-execution.md) ledger.

Architecture authority remains [design.md](design.md), amended by
[generic_core.md](generic_core.md). Read [AGENTS.md](AGENTS.md) for working rules
and commands.

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

## Durable document map

The categories below cover every audited `.agents/v3` surface. Category 1 is
current durable state, authority, or method; category 2 is evidence required for
scientific or decision audit; category 3 is superseded material removed after its
unique value was absorbed; category 4 is stale coordination debris removed from
the durable tree. Removed files remain recoverable from Git history.

| Category | File or group | Durable role |
|---|---|---|
| 1 — current durable | `README.md` | Entry point and complete document classification; never a second status ledger |
| 1 — current durable | [handoff.md](handoff.md) | Agent-neutral state snapshot and cold restart procedure |
| 1 — current durable | [design.md](design.md), [generic_core.md](generic_core.md) | Architecture invariants and authoritative semantic-IR amendment |
| 1 — current durable | [AGENTS.md](AGENTS.md), [conventions.md](conventions.md) | Working rules, routing, tooling, typing, and tests |
| 1 — current durable | [migration_workflow.md](migration_workflow.md), [migration-execution.md](migration-execution.md) | Migration method and sole live execution ledger |
| 1 — current durable | [evidence/2026-08-16-d3a-generic-spine-cut-freeze.md](evidence/2026-08-16-d3a-generic-spine-cut-freeze.md) | Frozen G1-G4 ownership, dependency sequence, and atomic integration contract |
| 1 — current durable | [proofline.md](proofline.md), [refactor_playbook.md](refactor_playbook.md) | Reusable coordination field guide and non-migration structural refactor method |
| 2 — required evidence | [evidence/2026-07-18-r0e-capability-inventory.md](evidence/2026-07-18-r0e-capability-inventory.md) | Immutable 154-row E0 inventory and 606-path coverage proof |
| 2 — required evidence | [evidence/2026-08-16-r3a-behavior-parity-audit.md](evidence/2026-08-16-r3a-behavior-parity-audit.md) | Complete behavior grades and missing-cluster audit |
| 2 — required evidence | [evidence/2026-08-16-r3b-feature-frontier.md](evidence/2026-08-16-r3b-feature-frontier.md) | Evidence-backed post-cut dependency graph and state-first feature sequence |
| 2 — required evidence | [parity.md](parity.md) | Numerical-reference workflow and legacy skim oracles |
| 2 — required evidence | [scaling.md](scaling.md) | Consolidated prototype benchmark methods, measurements, fixtures, and observed failure evidence |
| 2 — consolidated | `plane_strain.md`, `structural.md`, `tangent_assembly.md` | Required evidence merged into `scaling.md`; original topic files remain in Git history |
| 3 — superseded, removed | `feature-parity.md` | Old source/status matrix absorbed by the complete R0-E inventory and R3-A grades |
| 3 — superseded, removed | `architecture.md`, `roadmap.md` | Prototype carrier snapshot and P0-P8 checklist absorbed by design/E0 evidence and commit history |
| 4 — removed | `WORKFLOW.md`, `hardening.md` | Redundant Cursor-era prompts/process deleted; Git history retains the original files |

Current implementation state intentionally appears only in [handoff.md](handoff.md)
and the live ledger, not in this index.
