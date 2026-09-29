# Migration evidence

This file defines PyFEM's coverage vocabulary and the meaning of a completed
migration. Current work and decisions live in [migration-execution.md](migration-execution.md).
It does not prescribe an agent lifecycle.

## Coverage

The [legacy inventory](evidence/2026-07-18-r0e-capability-inventory.md) records 154
semantic capabilities and their source evidence. A capability is an observable
physical, numerical or public behavior, not a class or file. Its existing IDs
connect source evidence, decisions and tests; retain that traceability when rows
are split or combined.

The recorded dispositions mean:

- **preserve:** retain the observable contract;
- **change:** retain the intent while identifying the intentional difference;
- **retire:** no supported successor, with the lost behavior and affected consumers
  explicitly accounted for. A proposed retirement is not an approved deletion.

An unimplemented capability remains visible. Representation in the generic IR,
source code, or a passing component test does not establish public parity.

## Evidence grades

| Grade | Meaning |
|---|---|
| E0 | Source/example/claim identified; behavior not independently extracted |
| E1 | Equations, conventions and failure behavior reproduced by a reference |
| E2 | Target component and its edge cases pass |
| E3 | Public authored-to-verified-result flow passes, including ownership, rollback and provenance where applicable |
| E4 | Compatibility, documentation/examples and relevant system/performance evidence are complete |

Record the supported behavior and exact tested revision with a claim. A bounded
Q8 result does not upgrade an entire material, analysis or output family. The
[2026-08-16 parity audit](evidence/2026-08-16-r3a-behavior-parity-audit.md) distinguishes
strict parity from executable coverage and design representability.

For legacy comparisons, retain equations, tensor/Voigt conventions, units, signs,
measures, constraints, state timing and tolerances. If the legacy result is wrong,
record the counterexample and independent reference for the changed behavior.
[Design acceptance](design.md#11-testable-acceptance-suite) and
[performance evidence](design.md#12-performance-proof-policy) specify the technical
proof; use the applicable cases, not a universal test itinerary.

## Migration completion

A slice is complete when its claimed public behavior and required failure cases
are demonstrated and material findings are resolved. Independent assessment is
valuable at a numerical or architectural boundary; its value is the disconfirming
evidence, not a verdict word, reviewer count or priority-score tuple.

The whole migration claim additionally accounts for every in-scope capability,
public compatibility decision and maintained consumer, with a verified replacement
or explicit retirement. Deferred work cannot disappear into a completion claim.
The public path must use the intended architecture without a hidden predecessor
fallback. Relevant numerical, state, result, documentation and performance evidence
must refer to the actual resulting version. Completion does not require a particular
packet sequence or number of commits.
