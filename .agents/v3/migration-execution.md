# PyFEM v3 migration execution

Product/review state carried forward from the last recorded decision on 2026-08-21.
Local refs checked on 2026-09-09; no product tests or new G1 reviews were run during
the instruction redesign. This file is the only current migration status surface.
Prior decisions, exact evidence and superseded cards are in
[migration-history.md](migration-history.md).

## Exact next safe action

G1 candidate `2f6f972eef5a4c5f633478b1effecba834ca5fd4` still needs independent
numerical and architecture/trust assessment. The previously requested G1-N and
G1-A checks concern the repaired candidate, not an earlier revision. Establish both
kinds of evidence and address material findings before advancing G2; this
instruction cleanup neither supplies those reviews nor approves the candidate.
Reviewer count, verdict syntax and advisory-score totals are not the acceptance
criterion. The mathematical and architectural questions remain open until answered.

| Local ref | Verified revision and meaning |
|---|---|
| `v3` | `128d51d496dcdba897904461ca0d25f9bf1f2b15`; prior documentation tip; contains the completed Phase-1 reference, not G1 |
| `agnet/g1-generic-system` | `2f6f972eef5a4c5f633478b1effecba834ca5fd4`; isolated G1 candidate |
| `agnet/g1-integration-proof` | `72c48784b8a3718ed40b15c83dd60f0443278371`; local integration experiment, not accepted product code |
| Phase-1 oracle | `7ee65c3ea5c50278b17dce63f2569e22412e0ebd`; already integrated; P1-C is complete |

G1 uses exact binary-rational material qualification and a relative correspondence
envelope with no absolute/ULP floor. Its recorded coordinator results include 15
focused tests, 370 eight-file tests and 500 v3 tests in both digit modes, 689
repository tests and 622 material-edge checks. These are historical results, not a
fresh independent verdict. See the [G1 record](migration-history.md#exact-next-safe-action)
and the [repair history](migration-history.md#milestone-log) for detail.

## Current technical boundary

The accepted direction is the direct generic-spine replacement described in
[D3-A](evidence/2026-08-16-d3a-generic-spine-cut-freeze.md): G1 compiles the system,
G2 adds program/constraints and assembly, G3 supplies transaction/solve/verification,
and G4 switches the public path and removes predecessor and proof implementations.
Keep the public switch and its causal deletions together so an intermediate G1-G3
stage does not become a second public backend. This is a consistency requirement,
not a demand for exactly four commits, fixed filenames or physical-line budgets.

The candidate's numerical reference, array ownership, source identity, captured
registry, multiple spaces and open operator contract need independent scrutiny.
The rational solve and PatchTest8 oracles in the
[historical P1-C card](migration-history.md#p1-c-horizon-card--horizon_frozen) remain
behavioral references through G2/G3. Preserve their independent derivation and
fresh verification; reorganizing their tests does not retire their meaning.

After the public cut, the accepted direction is operator-local state, nonlinear
history/rollback, thermal coupling, then mass/capacity evolution. The
[frontier analysis](evidence/2026-08-16-r3b-feature-frontier.md) records the rationale;
it does not authorize feature implementation in an instruction-maintenance task.

## Coverage and open decisions

The latest row-level assessment is the
[R3-A audit](evidence/2026-08-16-r3a-behavior-parity-audit.md). No capability grades
are advanced by this cleanup. Older Phase-1 coverage overlays in the history have
narrower scopes and superseded next actions; they are not a current dispatch plan.
Use the [coverage vocabulary](migration_workflow.md) when recording new evidence.

Root API/CLI and legacy-input compatibility, GUI packaging, public retirement
candidates, persisted restore/rebind formats, and detailed RVE/FE2 and ROM contracts
remain unresolved. Their identities and prior decisions are in the
[decision record](migration-history.md#semantic-decisions-and-open-questions).

The unrelated stash `7d4e46a4e1c8a9eebe12da319cb63ddc4da02b51` is
`WIP on update-install-script: 55f8e96 Refactor version checking`; leave it out of
migration work. Existing branches and worktrees may contain retained evidence.
There is no active migration watchdog recorded. Reconcile changed refs or new task
results before relying on this status; ordinary documentation drift is not a
product failure.
