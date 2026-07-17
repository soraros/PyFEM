# PyFEM v3 agent guide

Branch `v3` is a ground-up, data-oriented finite-element rewrite under
`pyfem/v3/`. Legacy `pyfem/` is a requirements and numerical-reference source, not
an architecture to reproduce.

## Read first

Read [design.md](design.md) before planning or editing v3. It is authoritative for
the architecture, invariants, acceptance suite, and migration order.

For the full legacy-to-v3 migration, also read
[migration_workflow.md](migration_workflow.md) and resume from the sole live ledger,
[migration-execution.md](migration-execution.md). For structural work outside that
migration which crosses ownership/state boundaries or multiple milestones, read
[refactor_playbook.md](refactor_playbook.md).

The current `pyfem/v3` code is an executable prototype. Its kernels, tests, and
benchmarks may be reused when they satisfy the new contracts, but these current
types are explicitly not architectural constraints:

- `ProblemDefinition`
- `LoadedProblem`
- shape/node-count formulation dispatch
- global material/solver strings
- fixed-width `group_props`
- current solver and result APIs

Do not continue the old P0-P8 checklist. Begin with Phase 0/1 in
[design.md](design.md#13-migration-plan), and prefer a complete, falsifiable vertical
slice over feature breadth.

## Platform baseline

V3 requires Python 3.13+ and uses 2-space Ruff. This branch's packaging,
dependency, CI, and Intel-Mac compatibility work is intentional and separate from
the architecture reset.

The current Intel baseline does not install PySide6. The legacy `pyfem-gui`
entrypoint therefore is not a development gate and will fail unless GUI dependencies
are installed separately; resolve that later as an explicit optional-dependency
decision rather than silently restoring it to the core environment.

```bash
uv sync
.venv/bin/python -m pytest -q test/v3
.venv/bin/ruff check pyfem/v3 test/v3 --config pyfem/v3/ruff.toml
.venv/bin/ruff check test/v3 --config test/v3/ruff.toml
```

Run the full suite when code changes cross the legacy/v3 boundary:

```bash
.venv/bin/python -m pytest -q
```

## Working rules

- Separate authored specification, compiled model, compiled program, physical
  state, solver workspace, and result by meaning and lifetime.
- Compiler output owns read-only arrays and never aliases caller-owned inputs.
- Formulation, topology, field layout, material kernel, quadrature, and state
  schema are explicit; never infer physics from array shape.
- Dispatch once per homogeneous contribution block; numeric kernels receive plain
  arrays/scalars and do not own accepted state.
- Separate model-owned physical state from program/request-owned evolution and
  interaction state, then commit their composed trial atomically or discard it.
- Compile model/program contribution topology separately and compose the final
  backend/reduction plan once in `PreparedAnalysis` when structure is fixed. Add
  load contributions; never silently overwrite them.
- Require dangerous-case tests and an independent reference before optimizing.
- Preserve source/entity identity and result provenance through compilation,
  batching, state evolution, and output projection.

## Documentation map

| Status | Files | Use |
|---|---|---|
| Authoritative | [design.md](design.md) | Architecture and migration contract |
| Migration method | [migration_workflow.md](migration_workflow.md) | Capability conveyor, packet lifecycle, integration, proof, and completion |
| Execution state | [migration-execution.md](migration-execution.md) | Sole live migration ledger: commits, packets, evidence, blockers, and next action |
| Refactor method | [refactor_playbook.md](refactor_playbook.md) | Generic Horizon Gate and proof loop for non-migration structural refactors |
| Active tooling | [conventions.md](conventions.md) | Style, typing, Ruff, tests |
| Numerical evidence | [parity.md](parity.md), [scaling.md](scaling.md), [plane_strain.md](plane_strain.md), [structural.md](structural.md), [tangent_assembly.md](tangent_assembly.md) | Oracles and historical measurements |
| Temporary requirements inventory | [feature-parity.md](feature-parity.md) | Legacy breadth awaiting absorption into the live capability ledger; not an implementation order |
| Historical/superseded | [architecture.md](architecture.md), [roadmap.md](roadmap.md), [WORKFLOW.md](WORKFLOW.md), [hardening.md](hardening.md) | Understand the prototype; do not execute as a plan |

## Decision discipline

If implementation evidence contradicts the design, stop and record the dangerous
case, competing design, falsifiable consequence, and updated tests in the
[design amendment log](design.md#16-design-amendment-log) before changing an
invariant.
Green legacy parity alone is not sufficient evidence: it proves a reference answer
for a skim, not general representation, state safety, or solver compatibility.
