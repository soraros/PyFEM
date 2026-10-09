# v3 guidance

## Read by question

| Question | Source |
|---|---|
| What does the architecture mean? | [generic_core.md](generic_core.md): current operator/space IR; [design.md](design.md): numerical, identity, state and result contracts. The generic-core amendment supersedes the older carrier decomposition. |
| How do I run or format this code? | [conventions.md](conventions.md) |
| What migration work is pending? | [migration-execution.md](migration-execution.md#exact-next-safe-action); check the actual checkout against its refs. |
| What does a coverage or completion claim mean? | [migration_workflow.md](migration_workflow.md) |
| What has the program learned about doing this work (instruments, disciplines, ordering)? | [program-playbook.md](program-playbook.md) |
| Where is a prior result or oracle? | [README.md](README.md#evidence-index); query the relevant case or revision. |

These are separate kinds of information, not a mandatory reading sequence. For a
bugfix, read the affected contract. For an architecture change, read the relevant
commits and amendment; the previous API is not a compatibility requirement by
itself. Change an inadequate design with its scientific justification and tests,
rather than treating a document as proof that the implementation is correct.

## Historical instructions

The live record owns current work. Old packet cards and dated reviews retain
scientific evidence and decision provenance; their procedural instructions are
retired. Model/effort mandates, task-title syntax, callback/result formats, fixed
repair or commit counts, path freezes, line budgets and exact test names are not
standing rules. Preserve the numerical and behavioral meaning of their tests.
Revision-specific scientific acceptance conditions still apply to the behavior
being replaced. A line cap or missing callback is not a correctness finding.

There is no additional repository approval or delegation protocol. Use the current
request and host capabilities; the documents here supply PyFEM-specific context.
