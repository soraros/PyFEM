# PyFEM v3 structural refactor playbook

Adapted on 2026-07-17 from
`/Users/sora/Projects/Maxwell/absim_fvm/docs/refactor_playbook.md`.

Use this generic playbook for large or high-risk structural work outside the full
legacy-to-v3 migration: changes that cross ownership boundaries, retire a prototype
path, alter several production flows, carry numerical/physical/state risk, or
require multiple verified milestones. The migration specializes this discipline in
[migration_workflow.md](migration_workflow.md) and records state only in
[migration-execution.md](migration-execution.md). A localized one-shot change
follows [AGENTS.md](AGENTS.md) and [design.md](design.md) without adding long-horizon
machinery.

This file owns the generic refactor method. [migration_workflow.md](migration_workflow.md)
owns full-migration orchestration. [design.md](design.md) owns target architecture
and invariants. [AGENTS.md](AGENTS.md) owns branch-specific code, testing,
validation, tooling, and git rules. Put evolving milestone state in one named
execution-state artifact under `.agents/v3/`; put dated evidence in a focused review
note only when it adds facts not already represented by tests or the design.

## North star

```text
Make PyFEM simpler, smaller, and more correct by moving each semantic owner and
state lifetime onto the real compiled-model path.
```

Priority order:

```text
correctness -> representative performance -> code cleanliness
```

The refactor must make an ownership or behavioral invariant clearer on a real
public path. Preserve concepts carrying physical, numerical, state, validation,
restart, reconstruction, or API meaning. Retire hidden defaults, duplicate
orchestration, compatibility carriers, and stale names only after the replacement
and its evidence are real. A smaller design that hides a mismatch is wrong.

## Horizon Gate

Before implementation, establish a compact contract:

- outcome and ownership invariant;
- scope, authority boundary, path ownership, and non-goals;
- strongest competing explanation or design;
- baseline, edge cases, and testable evidence surface;
- consequence map: affected consumers, expected causal deletions, and migration
  order;
- acceptance bar, performance relevance, and an honest blocked condition; and
- exact merge dependencies for parallel work.

A small enabling phase need not shrink locally when the phase-gated final design
has a concrete causal deletion path and materially improves the system globally.
Conversely, do not credit unrelated cleanup merely because it lands in the same
diff, and do not call policy moved outside the measured owner a reduction.

### Portfolio pre-screen

Do not select the next deep task by walking to the nearest similar field, object, or
abstraction. Compare a small portfolio first:

- a narrow owner change;
- an owner-plus-consumer vertical slice;
- retirement of compatibility or duplicate orchestration;
- a measured performance change; and
- stopping or moving to a real application pain point.

Each candidate needs a compact card before research or delegation:

- measured pain or a concrete inconsistency;
- the current semantic owner and affected real public flows;
- the owner-plus-consumer dependency cone;
- exact migrations and causal deletions expected;
- compatibility/API and invariant burden;
- structural prediction for the owning path and whole v3 tree;
- representative runtime/allocation expectation, if performance-relevant; and
- a testable Phase 0 plus explicit stop condition.

A candidate passes only when Phase 0 can test a plausible global consequence:
causal downstream deletion, removal of real compatibility/orchestration burden, or
material improvement on a maintained public flow without weakening correctness.
Aesthetic duplication and owner-local line savings are not enough. Prefer no new
task when every candidate fails; record what evidence could reopen the decision.

### Measured space-for-time changes

Do not reject or accept a performance change from one allocation percentage. Before
implementation, define the operation's incremental workspace by phase, including
retained payloads, compiler temporaries, copies, sparse topology, state tables, and
backend work buffers.

Unless a Horizon contract justifies a different envelope, use these starting gates:

- candidate-minus-baseline post-return retained growth no greater than
  `max(1 MiB, 2% of baseline)` for unchanged semantics;
- traced-peak growth no greater than the causal phase model plus
  `max(1 MiB, 10% of that model)`;
- at least a 5% paired median improvement on one maintained public flow in two
  fresh processes, with no greater than 5% regression on another in-scope flow;
  and
- an explicit per-operation incremental workspace envelope distinct from total
  process RSS, solver-native memory, and result payloads.

For FEM features whose required physical state or sparse plan legitimately exceeds
those defaults, pre-register the domain-derived size model and acceptance envelope
instead of hiding it as an exception. For concurrent operations, bound the sum of
phase-peak workspaces by a declared pool. Keep exact runtime, allocation,
verification, hardware, and environment evidence in a dated review when making a
performance claim.

Performance policy in [design.md](design.md#12-performance-proof-policy) wins if
these defaults conflict with an owning phase's evidence contract.

### Goal and reasoning profile

If outcome, material design choices, or proof surface are unclear, remain in Plan or
research. Once the Horizon contract is stable for long-running implementation,
prefer a persistent Goal naming outcome, evidence, constraints, iteration policy,
and blocked stop. Skip Goal for bounded mechanical changes or read-only work.

Before starting a Goal, inspect any active one. Continue it only when objective and
proof match. Do not replace, clear, or mark a conflicting Goal complete merely to
make room; request the minimum lifecycle decision. A Goal is persistent completion
state, not broader authority.

For work entering this playbook, use the strongest quality-first profile available.
The current project mapping is **GPT-5.6 Sol with max reasoning**. Step down only for
a bounded mechanical slice or an explicit latency/cost constraint. Model and
reasoning effort never broaden authority.

## Authority contract

Route by requested deliverable and side effects, not isolated verbs. “Diagnose and
fix” authorizes the in-scope fix; “review and recommend” remains read-only. When
requested outputs conflict, follow the most specific explicit action and ask only
when a choice would materially change deliverable, physics, architecture, scope, or
authority.

An end-to-end refactor request authorizes safe local movement through research,
design, implementation, validation, and review. It does not authorize unrelated
cleanup, destructive/costly action, external writes, or publication. Tool choice,
Goal state, thread placement, and reasoning effort never expand authority. Follow
[AGENTS.md](AGENTS.md) for stage, commit, merge, push, and environment policy.

## Proof loop

Use these as proof gates, not a script to narrate:

1. **Orient.** Inspect the live tree, current diff, relevant commits, owning docs,
   and real high-level call paths. Treat dated claims and proposed design as
   hypotheses.
2. **Pre-register proof.** Record the Horizon Gate contract before finalizing the
   implementation design. Name baseline, edge cases, failure behavior, and
   acceptance evidence.
3. **Move one boundary.** Migrate meaningful behavior tests before deletion. Keep
   the prototype path frozen except for necessary shared API churn until the
   replacement is verified.
4. **Verify and repair.** Run focused proof, real integration flow, and relevant
   broader checks at every milestone. Compare baseline and repair a failed milestone
   before building on it.
5. **Seek disconfirmation.** For high-risk work, use independent read-only reviewers
   when available and authorized. Ask for the strongest competing design, decisive
   failure case, missing proof, and scope violation—not general approval.
6. **Adjudicate and persist.** Mark every finding accepted, rejected with evidence,
   or deferred outside the success bar. Update execution state and owning docs only
   when their distinct truths changed.

For work crossing turns, compaction, handoff, or several threads, keep one
execution-state artifact. A Goal owns the stable contract; the artifact owns
evolving milestone status, decisions, surprises, evidence, merge order, next
action, and blockers. It links to rather than restates [design.md](design.md).
Update it at milestones or material plan changes, not after routine tool calls.

## Delegation and coordination

Use the current thread by default. Create user-visible threads when explicitly
requested or when each thread has a named, independently reviewable output.

### Thread titles

Treat a title as a compact index entry, not a miniature prompt. The project,
branch, model, worktree, status, and generic action are already visible elsewhere.
A title should identify the stable work coordinate, the semantic owner, and the
terminal property being established:

```text
<ledger-id> · <semantic owner> — <target state>
```

Use these ledger-id classes:

- `P<phase>-<letter>` for a code-and-test work packet;
- `R<phase>-<letter>` for a read-only research or independent-review packet;
- `D<phase>-<letter>` for a bounded design decision owned by the delegator; and
- `I<phase>` for the phase integration and combined-proof thread.

The semantic owner is the smallest recognizable boundary, normally one to three
concrete nouns. The target state is an observable end condition, not an activity.
Prefer `model spec — explicit immutable intent` over `implement new model spec`,
and `assembly — prototype quarantined` over `refactor assembly module`.

Title rules:

- use the exact ID recorded in the execution-state artifact and never reuse it;
- keep the title stable during normal active, blocked, review, and complete states;
  if a task ends without the required terminal result, prefix its visible title
  with `INCOMPLETE` and record the exact reason and any replacement separately;
- omit `PyFEM`, `v3`, worker/model names, `thread`, `task`, `implement`, `work on`,
  and other context or motion words;
- aim for 30 to 60 characters and at most one punctuation-delimited qualification;
- give materially changed scope a new ledger ID and thread instead of stretching
  an old title; and
- record the exact title beside the thread ID at dispatch so titles and prompts can
  be audited independently.

Examples:

```text
P0-A · assembly — prototype quarantined
R0-A · topology — explicit meaning verified
D1-A · constraints — affine owner selected
I0 · foundations — combined proof
```

Before dispatching parallel writers:

- define disjoint file/path ownership;
- define dependency and merge order;
- name the owning invariant and acceptance commands;
- state forbidden shared files and API decisions;
- require a focused commit in the worker branch; and
- reserve shared design, routing, and execution-state documents for the delegator.

Read-only reviewers need a question, validation focus, evidence requirement, and output;
they do not need file ownership. Editing workers need both file ownership and a
merge contract. Only one writer owns a path at a time. If safe ownership cannot be
established, make the overlapping package review-only until dependencies are
reconciled.

Worktree workers start from the exact committed v3 base named in the execution
artifact. They must not merge, push, edit shared design docs, or absorb unrelated
working-tree changes. Each returns:

- commit hash and changed paths;
- acceptance evidence and warnings;
- design decisions or assumptions made;
- unresolved findings; and
- the exact next dependency/merge action.

### Completion signalling

Prefer a direct terminal callback over model-driven polling. Every worker prompt
names the integration thread ID. Immediately before its final response, the worker
sends that thread exactly one compact message containing:

```text
<ledger-id> COMPLETE|BLOCKED · <commit or none> · <evidence> · <warning/blocker>
· <next merge or dependency action>
```

The callback is part of handoff, costs no extra monitoring turn, and does not grant
merge, push, shared-file, or scope authority. If the callback tool is unavailable,
the worker marks `CALLBACK UNAVAILABLE` prominently in its own final response.

A task is complete only when its own final response begins with
`RESULT: COMPLETE` or `RESULT: BLOCKED`, contains the required evidence and clean
worktree state, and the callback is received or explicitly marked unavailable.
Idle status, commentary, partial command output, and recovered evidence do not
complete a task. After one unsuccessful resume, mark the task visibly incomplete
and record any replacement under a distinct ID.

While the delegator is already active, use the app's native thread wait/status
mechanism; do not spend a model turn repeatedly rediscovering `active`. Add a
scheduled watchdog only when direct callbacks are unavailable or a packet is
expected to outlive the active coordination session. A watchdog must use the
smallest capable model and reasoning profile, inspect only the recorded thread IDs,
poll no more often than every 15 minutes unless latency is material, notify only on
terminal/attention transitions, and be disabled as soon as all watched work is
terminal. Record any active watchdog and its cadence in the execution artifact.

The delegator integrates shared documents, adjudicates findings, checks combined
evidence, and updates the execution artifact. Surface choice never grants a new
side effect.

## Self-application

For a process/instruction refactor, inventory inherited behavior and classify each
item as **preserve**, **change**, or **retire**, with its invariant/evidence. Treat
current wording as an unverified behavior inventory. Change one policy group at a
time and replay the same cases. Pre-register edge cases, compare the
strongest alternative structure, scan stale terms/conflicts, and compare line/byte
counts. Do not describe author-written cases as independent proof.

## Completion

The refactor is technically complete when:

- stated acceptance and [AGENTS.md](AGENTS.md) evidence gates pass on the real path;
- no unresolved in-scope correctness, authority, or required-evidence finding
  disproves the invariant;
- performance claims stay within the representative and edge-case envelope;
- remaining old terms are active ownership, intentional compatibility, negative
  tests, or clearly labeled history;
- accepted findings are repaired, rejected findings have evidence, and advisory or
  out-of-scope improvements are named follow-ups;
- parallel work is integrated in declared dependency order and reverified; and
- long-running work is restartable from its execution state while owning docs
  describe current truth.

“Blocked” means safe in-scope alternatives are exhausted. Preserve attempted paths,
evidence, exact blocker, and minimum input/authority needed. Budget is not
completion. Technical completion does not imply landing: perform merge, push, or
publication only when requested.

## Minimal handoff

```text
Continue the PyFEM v3 structural refactor described in [execution-state artifact].
Read .agents/v3/AGENTS.md, .agents/v3/design.md, and
.agents/v3/refactor_playbook.md.

Outcome/invariant: [...]
Owned paths: [...]
Scope and non-goals: [...]
Competing hypothesis: [...]
Acceptance evidence: [...]
Merge dependency: [...]

Inspect the live path and apply the Horizon Gate. Continue a matching Goal when
active; otherwise remain in Plan/research until the contract is stable, then decide
Goal usage before implementation. Verify every milestone, seek disconfirming
review, commit only owned changes, and keep the execution state restartable. If
blocked, record evidence, exact blocker, and minimum unlock.
```
