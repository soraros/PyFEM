# Instruction redesign audit — 2026-09-09

This is a change record, not an instruction source. Scope was instruction and
configuration maintenance; no migration implementation or numerical review ran.
Baseline: `v3` at `128d51d496dcdba897904461ca0d25f9bf1f2b15`, including the
uncommitted 2026-09-05 first pass. The user authorized replacing accumulated process
and then emphasized preserving information the agent cannot otherwise know.

## Source grounding and judgment

The [official Astra guide](https://developers.openai.com/api/docs/guides/latest-model)
was fetched again on 2026-09-09. It recommends auditing instruction files because
Astra is sensitive to conflicting guidance; it also describes autonomy, scoped
verification, style and delegation adjustments. Those are recommendations for
behavior, not requirements to install the example prompts verbatim.

The [official AGENTS.md guide](https://developers.openai.com/codex/guides/agents-md)
describes global-to-project instruction discovery. That supports retaining a root
router; a guide inside `.agents/v3` is not automatically found from the root.
The [configuration reference](https://developers.openai.com/codex/config-reference)
documents `features.context_management.experimental_mode` as notes/search-based
experimental context management, requiring eligible ChatGPT sign-in.

Deleting the overlapping manuals, retiring ceremony, and separating current state
from historical records are design judgments for this repository. No official
source mandates this file structure, a reviewer count or a model/effort profile.
Model competence is not evidence of numerical correctness; independent mathematical
references and observable transaction/result contracts remain essential.

## Removed constraints and their surviving information

| Former behavior | Result of the redesign |
|---|---|
| Handoff and ledger both named current refs and next work | Only the short live record does; prior records are historical |
| Proofline, full migration method and refactor playbook overlapped | Proofline/refactor/handoff deleted; migration file now defines only coverage and completion meaning |
| Every major task required cards, phases, owner roles and a portfolio | No generic project lifecycle; current work follows its requested outcome and actual dependencies |
| Sol/Astra max roles and cheaper-worker prohibitions | Model/effort belongs to host configuration and the task; no repository mandate |
| Task naming grammar, callbacks and RESULT prefixes controlled completion | Verified work and evidence determine completion; host result delivery suffices |
| Dirty tree, one commit, fixed paths or two failed repairs forced a stop | No standing rule of that kind; user changes and actual conflicts still matter |
| Approval prohibitions included reasoned integration and design decisions | Removed duplicated platform/permission rules; repository docs do not create extra authority or approval loops |
| Exact line caps, named tests, reviewer count and zero advisory-score totals | Retired procedural limits; preserve the scientific question and test meaning |
| Arbitrary percentage performance budgets | Removed; design's representative public-flow evidence remains |
| Old waves, Phase-0 rename and “next trial” looked executable | Removed from live guidance or explicitly historical |
| Full suites were routine packet gates | Commands are available by scope; the 640-digit diagnostic regression mode remains documented |
| Architecture authority competed with its own amendment | Scope map identifies generic-core as the current IR amendment and design as scientific contracts |

The G1-G4 public switch remains a technical consistency constraint: intermediate
private stages do not become a second public backend. It no longer implies exactly
four commits or an exclusive-path bureaucracy. Pending independent numerical and
architecture evidence at `2f6f972` remains pending; no review verdict was inferred
from this cleanup.

## Durability and preservation

- The three deleted manuals are recoverable at the baseline Git commit under
  `.agents/v3/handoff.md`, `.agents/v3/proofline.md`, and
  `.agents/v3/refactor_playbook.md`. The old workflow is recoverable there too.
- The previous 3,762-line ledger is retained as `migration-history.md`, with an
  archive notice and obsolete link annotations. Source/integrated hashes, rational
  displacement/reaction/work oracles, PatchTest8 fields, findings, capability IDs,
  and test results are retained. It is not a second live ledger.
- All four pre-existing dated evidence files, parity and scaling records retain
  their bytes. Scientific design sections remain; obsolete execution plans and
  routing were removed. The generic-core change removes a one-wave constraint,
  not the single-core architectural outcome.
- The unrelated `update-install-script` stash and existing source branches were
  inspected only to identify retained state. No worktree, stash or product file
  was modified.
- Global AGENTS and configuration had changed since the first pass. At this audit,
  global AGENTS contained the user's original personal sentence; the configured
  default was `gpt-5.6-sol` at `low`. These newer choices were preserved byte-for-byte.
  The experimental context flag was already true. No plugin cache or user memory
  was edited and no blanket model change was applied.

## Scenario review

These are static instruction walkthroughs, not measured model-behavior evaluations.
A separate read-only reviewer inspected the baseline and proposed replacement.

| Scenario | Reading and action supported by the final hierarchy |
|---|---|
| Small docs edit | Affected text and links; no design corpus, packet, model override, numerical suite or review ceremony |
| Ordinary v3 bugfix | Owning code/contract and local test commands; meaningful focused regression evidence without mandatory external review |
| Numerical or architecture change | Relevant design and generic-core sections; independent reference and failure behavior; evidence-backed amendment when needed |
| Bounded delegation | User/host lifecycle and a bounded task; no repository title, commit, callback or final-prefix requirement |
| Resume migration | Live record, actual refs, current G1 evidence questions; historical P1-C/P2-A text cannot dispatch obsolete work |

The reviewer found two remnants in the staged version: a deleted handoff link and
one-wave wording in generic-core. Both were removed before application. No unique
scientific fact was found in the three deleted manuals that lacked a retained home.

## Changed files

- Rewritten: root `AGENTS.md`; `.agents/v3/AGENTS.md`, `README.md`,
  `conventions.md`, `migration_workflow.md`, and `migration-execution.md`.
- Adjusted for scope/history: `.agents/v3/design.md`, `generic_core.md`, and root
  `.gitignore` (the root router remains shareable).
- Added: `.agents/v3/migration-history.md` and this audit record.
- Deleted: `.agents/v3/handoff.md`, `proofline.md`, and `refactor_playbook.md`.
- Audited without modification: global AGENTS/configuration, existing dated
  evidence, parity/scaling records, and the narrowly scoped Kimi delegation skill.

## Verification

The eight former guidance/router files totalled 2,830 lines / 142,773 bytes.
Their five surviving replacements total 156 lines / 8,946 bytes (about 94% less).
This measure excludes scientific design, evidence, and both the live and archived
ledger; it does not count retained history as deleted information.

Validation covers relative links and anchors, retired-instruction routing,
scientific-section and evidence preservation, configuration parse/flag recognition,
unchanged global files, unchanged product paths, and `git diff --check`.
No numerical tests are claimed from this documentation-only change.

Applied-file checks passed: 90 local references including 12 anchors; 15 unchanged
scientific/architecture sections; archived ledger-body equivalence; 629 other
tracked-file hashes unchanged; valid TOML with the enabled context flag recognized
by the installed client; unchanged global files, source refs and stash; clean diff
whitespace. Work remains uncommitted at the original HEAD.
