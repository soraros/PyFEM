# PyFEM v3 agent guide

Branch `v3` is a ground-up, data-oriented finite-element rewrite under
`pyfem/v3/`. Legacy `pyfem/` is a requirements and numerical-reference source, not
an architecture to reproduce.

## Read first

For a cold start without thread context, begin with [handoff.md](handoff.md). Then
read [design.md](design.md) and [generic_core.md](generic_core.md) before planning
or editing v3. The first owns the durable ownership/state/result invariants; the
second is the authoritative post-Phase-1 semantic-IR amendment.

For the full legacy-to-v3 migration, also read
[migration_workflow.md](migration_workflow.md) and resume from the sole live ledger,
[migration-execution.md](migration-execution.md). For structural work outside that
migration which crosses ownership/state boundaries or multiple milestones, read
[refactor_playbook.md](refactor_playbook.md).

[Proofline](proofline.md) is the concise reusable field guide for the coordination
method. It is useful when transferring the workflow to another project or session,
but it never replaces this repository's full migration specialization or live
ledger.

The full 154-row E0 source inventory is preserved in
[evidence/2026-07-18-r0e-capability-inventory.md](evidence/2026-07-18-r0e-capability-inventory.md).
Read or query that large evidence note only when selecting, classifying, or auditing
a capability; mutable lifecycle and next-action state remains in the execution
ledger.

The current `pyfem/v3` code is an executable prototype. Its kernels, tests, and
benchmarks may be reused when they satisfy the new contracts, but these current
types are explicitly not architectural constraints:

- `ProblemDefinition`
- `LoadedProblem`
- shape/node-count formulation dispatch
- global material/solver strings
- fixed-width `group_props`
- current solver and result APIs

Do not continue the old P0-P8 checklist or the superseded P2-A through P2-G plan.
Phase 0/1 is a frozen reference and R2-E/P2-H/R2-F are complete. Resume only from
the exact action in [handoff.md](handoff.md) and the live execution ledger; section
11 of `generic_core.md` records completed simplification evidence, not a live
dispatch. Prefer a complete, testable generic instance over feature-by-feature
class ports.

## Task vocabulary and routing

PyFEM v3 packets are local finite-element design, implementation, and correctness
work. In task titles, prompts, callbacks, and ledger entries, use concrete terms
such as **finite-element correctness review**, **local edge-case matrix**,
**correctness matrix**, **failure case**, and **independent reviewer**. Do not
import labels, metaphors, skills, or review frames from unrelated domains, and do
not reuse historical packet wording as prompt text. This routing rule changes no
technical acceptance criterion.

## Delegated packet return contract

Do not dispatch a delegated packet unless its prompt itself names the exact source
integration thread and requires one terminal callback immediately before the
worker's final response. Do not assume that thread ancestry, a delegation wrapper,
an idle state, or a final visible only inside the child will return the result.

Every delegated prompt ends with this explicit instruction:

```text
Immediately before your final response, send source thread <thread-id> exactly one
terminal callback:
<packet-id> COMPLETE|BLOCKED · <commit or none> · paths=<...> · proof=<...>
· warning/blocker=<...> · next=<...>
If callback delivery is unavailable after one bounded attempt, put
CALLBACK UNAVAILABLE immediately after the RESULT line and close normally.
```

The coordinator records the callback target while freezing the packet and verifies
that the literal callback instruction is present before creation. Workers make one
bounded delivery attempt and do not remain alive waiting for acknowledgement. The
coordinator relies on callbacks; if one is unavailable, it may take one terminal
status snapshot, never start a polling loop.

## Delegated intelligence routing

Choose reasoning capacity by consequence, not by habit:

- use Luna for thread gardening, inventories, formatting, mechanical evidence
  collection, and ledger maintenance;
- use Sol at high reasoning for bounded implementation and ordinary design or
  correctness review;
- reserve Sol at max reasoning for architecture freezes, numerical-semantics
  decisions, integration adjudication, and the final independent review of a
  migration boundary.

Raise or lower a packet deliberately when its actual risk differs from these
defaults. Do not assign max reasoning merely because a packet belongs to the v3
migration. Record the exceptional choice in the frozen packet when it materially
affects cost or confidence.

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
- Treat entity blocks, discrete spaces, bound operator blocks, typed contribution
  channels, coordinate maps, and observations as the stable semantic IR. Legacy
  element/material/section/model/load categories are authoring or evaluator
  composition choices, not mandatory core registries.
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
- Require edge-case tests and an independent reference before optimizing.
- Preserve source/entity identity and result provenance through compilation,
  batching, state evolution, and output projection.
- Validate external/authored/restore representations once at their authority
  boundary. Internal consumers check live identity, generation, capabilities, and
  numerical preconditions; they do not recursively rebuild compiler-owned meaning
  at every layer.

## Documentation map

The complete durability classification for every retained file is in the
[README document map](README.md#durable-document-map). Operational routing is:

| Need | Read |
|---|---|
| Cold restart and exact current frontier | [handoff.md](handoff.md), then [migration-execution.md](migration-execution.md) |
| Architecture and semantic authority | [design.md](design.md), [generic_core.md](generic_core.md) |
| Migration method | [migration_workflow.md](migration_workflow.md), with [proofline.md](proofline.md) only as its reusable field guide |
| Structural method outside the migration | [refactor_playbook.md](refactor_playbook.md) |
| Tooling and local commands | [conventions.md](conventions.md) |
| Capability or numerical evidence | Use the category-2 records linked from [README.md](README.md#durable-document-map); never infer live state from historical files |

## Decision discipline

If implementation evidence contradicts the design, stop and record the failure
case, competing design, testable consequence, and updated tests in the
[design amendment log](design.md#16-design-amendment-log) before changing an
invariant.
Green legacy parity alone is not sufficient evidence: it proves a reference answer
for a skim, not general representation, state safety, or solver compatibility.
