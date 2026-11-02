# PyFEM v3 migration execution

Live state checked on 2026-11-02 against `v3` at `735d064`, the M85 merge.
This file is the only current migration status surface.
[migration-history.md](migration-history.md) is the archive of earlier decisions,
evidence and superseded dispatch cards; its "current" and "next" instructions
are historical, not work to resume.

## Exact next safe action

No M85 process or task remains, and this record dispatches no FE² recovery or
benchmark phase. M85 merged at `735d064` after a clean review of `06caa7d`; the
earlier paused state at `6073fc5` and its recovery instructions are superseded.

The committed performance evidence is the controlled pair
`bench/results/run_m85_candidate.json` (candidate `397849f`) and
`bench/results/run_m85_control_base.json` (control `4bf24ae`). Both are
clean-tree, explicitly partial runs covering all four FE² workloads at
1/2/4/8/16 threads and all four FE²-memory sizes, with every recorded
correctness gate passing. They are targeted candidate/control evidence, not a
full-harness run or a promoted baseline; `bench/results/baseline_m3.json` is
unchanged.

## Merged capability state at `735d064`

- The generic-spine migration and public cut are landed. The ancient G1
  candidate reviews, G1-G4 dispatch sequence and `agnet/*` refs in the archive
  are not open work.
- The M62/M63 breadth surveys and their cohesive/interface, penalty-contact and
  ViscoPlasticity/SOVS follow-ons are complete. M64 is also merged: `8728567`
  wired production assembly to pattern-cached CSR, consolidated the canonical
  pattern compiler and canonicalized stable-order reductions; `f837427` added
  its candidate/control benchmark evidence.
- The M71 crystal per-IP status study is complete. Following M80's removal of
  the dead legacy implementation, M81 landed the fresh stateful crystal law.
- M75-M85 are complete. The merged FE² capabilities include RVE homogenization,
  nested-state composition and rollback, periodic-cell authoring, the nested
  continuum operator, typed cross-level failure and cost propagation, and M85's
  deterministic opt-in integration-point host pool with stable result order.
  `nested_ip_thread_count` defaults to the direct serial schedule; the scoped
  nested trial buffers are retained across evaluations and steps. Raw-uint64
  1-thread/n-thread gates and retained-buffer identity checks pin those
  execution policies. The partial candidate/control artifacts above are the
  scale and memory evidence, not proof of universal threading speedups.

Reconcile refs and source before dispatching new work. Current architecture and
behavior are defined by [generic_core.md](generic_core.md),
[design.md](design.md) and the implementation; historical mission cards and
reviews retain provenance but do not supersede this live record.
