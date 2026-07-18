# Proofline

- Status: **reusable semi-automatic coordination field guide**
- Scope: long-running repository migrations and structural refactors
- Live state: never stored here
- PyFEM specialization: [migration_workflow.md](migration_workflow.md)
- PyFEM live ledger: [migration-execution.md](migration-execution.md)

**Proofline** is a ledger-driven method for moving a large codebase through a
sequence of bounded, independently checked changes. The name refers to the ordered
line from intent to evidence: every state change must be traceable through an exact
contract, source result, integration decision, and proof result.

It is semi-automatic rather than autonomous. Machines may dispatch bounded work,
transport callbacks, run deterministic checks, and refresh views. Semantic design,
integration, finding adjudication, compatibility, and completion remain explicit
decisions.

## The loop

```text
discover behavior -> classify intent -> select a dependency-ready slice
-> freeze its contract -> delegate bounded work -> receive a terminal callback
-> review and integrate serially -> obtain independent correctness evidence
-> run combined proof -> advance the ledger or stop with an exact blocker
```

The smallest useful unit is a **packet**: one outcome, one semantic owner, one exact
base, bounded paths, explicit non-goals, pre-registered proof, and one terminal
result. Packets follow dependency order rather than the legacy directory tree.

## Four durable surfaces

Proofline separates four kinds of information:

| Surface | Contains | Must not contain |
|---|---|---|
| Design authority | Intended architecture, invariants, acceptance criteria, amendment rules | Current task status |
| Method | Packet lifecycle, roles, delegation, review, automation, and stop policy | Project-specific live state |
| Sole live ledger | Exact commits, packets, decisions, evidence, blockers, and next safe action | Competing plans or decorative progress claims |
| Dated evidence | Large inventories, measurements, or adjudication records that should remain immutable | Mutable status |

Chat history, task titles, dashboards, worktrees, and context summaries are useful
interfaces. None is a source of truth. A fresh coordinator must be able to resume
from the durable surfaces without replaying conversation history.

## Semantic coverage and evidence

For migrations, track user-observable or physically meaningful capabilities rather
than files or legacy classes. Every capability has one target semantic owner, hard
dependencies, failure cases, an independent reference, and one disposition:

- **preserve:** retain the observable contract;
- **change:** retain the intent while explicitly replacing unsafe or ambiguous
  behavior; or
- **retire:** remove it only with consumer evidence, a replacement or loss
  statement, and the required approval.

Keep lifecycle separate from proof strength. A useful lifecycle is:

```text
inventoried -> classified -> contracted -> implementing -> provisional
-> proved | retired-proved
```

Use evidence grades when the portfolio is broad:

| Grade | Meaning |
|---|---|
| E0 | Source, example, or claim exists; behavior is not extracted |
| E1 | Behavior and conventions are reproducible independently |
| E2 | Target component and edge cases pass; no complete public flow |
| E3 | Public authored-to-verified-result slice passes |
| E4 | Compatibility, representative performance, docs/examples, and repository gates pass |

Green component parity cannot claim a public vertical slice. Select new work from a
small dependency-ready portfolio, prioritizing invariant repair, architectural
unblocking, complete public-flow replacement, and causal deletion before breadth or
optimization. If nothing is ready, recording the missing evidence or decision is a
successful no-work result; do not manufacture activity.

## Packet contract

Freeze this card before dispatching implementation:

```text
ID/title:
Outcome and ownership invariant:
Coverage or user-visible behavior advanced:
Exact base and dependencies:
Owned paths / forbidden paths:
Public consumers:
Strongest competing design:
Reference evidence and failure cases:
Expected migrations or causal deletions:
Compatibility consequence:
Performance relevance:
Acceptance commands:
Merge order:
Blocked condition and minimum unlock:
Integration task:
```

A scope change after freezing creates a new packet. A worker may report a conflict,
but may not silently reinterpret the card.

## Roles and serial ownership

- **Coordinator/integrator:** owns selection, the sole ledger, contract freezing,
  semantic review, serial integration, finding adjudication, and advancement.
- **Bounded writer:** owns only the frozen paths and outcome. It makes a focused
  source commit but does not merge, push, edit shared state, or decide a new public
  or architectural question.
- **Independent reviewer:** is read-only and checks a distinct correctness lens
  against the integrated commit. A reviewer supplies evidence, not a vote.

The first definition of a compiler, state, constraint, contribution, result, or
public API boundary stays serial. Parallel work is limited to genuinely disjoint
writers behind a frozen shared contract or independent read-only research.

## Names that carry coordinates, not prose

Use stable task titles:

```text
<packet-id> · <semantic owner> — <target state>
```

Examples:

```text
P1-A · program compiler — affine plan canonical
R1-A · linear slice — physics and state verified
I1 · linear slice — combined proof
```

Use `P` for writers, `R` for read-only research/review, `D` for bounded design
decisions, and `I` for integration. IDs are never reused. Transient execution
metadata belongs in the ledger, not the title. The only status prefix is
`INCOMPLETE`, applied when a task ended without the required terminal contract;
record its reason and replacement separately.

## Domain-native task language

Describe a task in the vocabulary of the actual repository and bounded outcome.
Avoid copying metaphors, labels, or review frames from unrelated work: they add no
technical value and can route tools or policy incorrectly. Prompts should state the
local domain, exact paths, exact base, correctness questions, and expected evidence.
This wording rule never weakens the review surface.

## Terminal result integrity

A task is complete only when all of these exist together:

1. its own final response begins with `RESULT: COMPLETE` or `RESULT: BLOCKED`;
2. the final records the exact base, commit or `none`, required command evidence,
   warnings, worktree state, and smallest next action;
3. exactly one terminal callback reaches the integration owner, or the final
   explicitly reports callback unavailability; and
4. the integrator verifies task ID, ancestry, paths, evidence, and callback before
   changing the ledger.

Idle state, commentary, partial commands, recovered output, or a replacement task
does not complete the original task. After one unsuccessful resume, mark the task
visibly incomplete and issue at most one bounded replacement rather than building a
chain of nominally active tasks.

## Event-driven coordination

Preferred completion signals, in order:

1. terminal callback from the already-running task;
2. one bounded native wait while the coordinator is active;
3. one immediate status snapshot when a callback is missing; and
4. only for orphaned or genuinely long-lived work, a smallest-capable-model
   watchdog with a long cadence and an explicit shutdown condition.

Do not wake a model repeatedly to reread an unchanged task. Polling consumes cost
and context without producing evidence. A watchdog may report a state transition;
it may not review code, integrate a commit, reinterpret findings, or advance the
ledger.

## Safe automation boundary

Safe mechanical automation may:

- validate callback IDs, exact ancestry, changed paths, and clean diffs;
- run pre-registered tests, formatting, static checks, and evidence scans;
- collect hashes, counts, timings, and warnings;
- deduplicate terminal notifications; and
- regenerate a status view from the ledger.

It may not automatically:

- amend design or choose a semantic owner;
- interpret physics, numerical conventions, units, signs, or tolerances;
- accept or reject reviewer findings;
- integrate commits or resolve conflicts;
- retire or break public behavior;
- accept a correctness/performance tradeoff; or
- declare a phase or migration complete.

Use the strongest reasoning model for coordination, architecture, semantic
boundaries, numerical meaning, integration, and independent review. A cheaper
model is appropriate only for a frozen, mechanical, disjoint task whose ambiguity
returns to the coordinator.

## Dashboard as projection

A status dashboard is a **read-only projection of the sole live ledger**. It is
valuable because it makes a long workflow inspectable without forcing the user to
read the ledger, but it must never become another mutable plan.

It should distinguish at least:

- completed and integrated work;
- source-complete but unreviewed or unintegrated work;
- active work;
- incomplete or superseded tasks;
- blockers and deliberate deferrals; and
- the exact next safe action.

Bind important claims to exact commits, test counts, or reviewer verdicts. Do not
invent percentages, confidence scores, dates, or qualitative status. Refresh the
view only after a verified transition, and repair the ledger first when the two
disagree.

For inline dashboards:

- design for roughly 736 px and reflow down to 320 px;
- let controls size to their content and wrap naturally;
- do not force long stage labels into equal-width columns;
- keep one selected-stage detail area rather than duplicating prose;
- use theme-aware colors and accessible native controls;
- validate interaction, overflow, overlap, and clipped labels before presenting;
  and
- keep generated presentation code outside the repository unless the dashboard is
  itself a maintained project deliverable.

Copyable dashboard prompt:

```text
Create or update an inline project-status dashboard from the repository's sole live
ledger. The dashboard is a read-only projection, never another project plan.

Inspect the design, ledger, exact Git state, task finals/callbacks, and test evidence.
Do not invent progress, confidence, dates, scores, or completion claims. Clearly
separate integrated, source-complete but unintegrated, active, incomplete,
superseded, blocked, and deferred work. Show the dependency path and exact next safe
action, binding important claims to commits, test counts, or reviewer verdicts.

Keep the view compact and interactive only where selection reduces clutter. Design
for approximately 736 px and reflow to 320 px. Let controls size to content and wrap
naturally; never force long labels into narrow equal-width columns. Use theme-aware
colors, accessible native controls, and one selected-stage detail area.

Before presenting it, read the fragment back and validate interaction, narrow-width
reflow, overflow, overlap, clipped labels, missing elements, and undefined script
identifiers. If the dashboard disagrees with the ledger, correct the ledger or mark
the conflict rather than guessing.
```

## Integration and proof

The integrator reviews source results in this order:

1. exact base, ancestry, path scope, and protected user changes;
2. frozen outcome and non-goals;
3. semantic ownership and dependency direction;
4. edge cases, failure behavior, and independent reference;
5. consumer migration and causal deletion claims;
6. state, identity, provenance, and snapshot lifetimes;
7. numerical conventions, tolerances, balance, and performance claims; and
8. focused test and static evidence.

Only reviewed commits are integrated, one at a time. Run a focused gate immediately
after integration. Independent review examines the integrated commit, not merely the
worker branch. Accepted required findings return through a bounded repair; green
commands alone do not overrule an unresolved semantic finding.

Allow at most two repair rounds under one frozen packet. A repeated failure, third
round, or newly required semantic decision returns the work to selection/design
state. Sunk cost never lowers the gate.

## Resume packet

Before yielding a long-running coordination session, persist:

- exact branch, integrated commit, dispatch bases, and clean/dirty state;
- protected user changes;
- active packet IDs, titles, task IDs, bases, and ownership;
- callbacks, source commits, integrated commits, and repair counts;
- findings and their adjudication;
- proof commands and exact results;
- active watchdog and shutdown data; and
- one exact next safe action or blocker.

A fresh session reads the design, method, and sole ledger; verifies Git state; then
executes one safe transition. It does not reconstruct project truth from task
commentary.

## Copyable coordinator prompt

```text
Use the repository's Proofline method for this long-running change.

First locate and read the design authority, Proofline specialization, and sole live
execution ledger. Verify the recorded branch, exact integrated commit, and worktree
state. Do not infer current state from chat history or task titles when durable state
exists.

Work one verified transition at a time:
1. consume any unprocessed terminal result;
2. perform deterministic callback prechecks;
3. review source semantics before serial integration;
4. obtain independent read-only correctness evidence after integration;
5. run combined proof at one exact commit;
6. update coverage and the exact next safe action; or
7. select and freeze one dependency-ready packet only when no evidence is waiting.

Use stable packet titles, exact bases, bounded path ownership, explicit non-goals,
pre-registered acceptance, and one terminal callback. Treat idle or partial tasks as
incomplete. Prefer callbacks and bounded waits over polling. Use cheaper models only
for frozen mechanical work; keep semantic decisions, integration, and independent
review with the strongest reasoning model.

If a status dashboard is useful, derive it from the sole ledger. Clearly separate
integrated, source-complete, active, incomplete, blocked, and deferred work. Show the
exact next safe action, use responsive wrapping controls, and never let the dashboard
become a second source of truth.

Do not merge, push, widen authority, change public compatibility, or alter a design
invariant unless separately authorized. Persist every material transition before
yielding.
```

## PyFEM adoption

For this repository, [design.md](design.md) is the architecture authority,
[migration_workflow.md](migration_workflow.md) is the full Proofline specialization,
and [migration-execution.md](migration-execution.md) is the only mutable migration
state. This field guide introduces no second roadmap, ledger, evidence grade, or
completion claim.
