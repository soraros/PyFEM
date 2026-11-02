# PyFEM v3 migration execution

Live state checked on 2026-11-02 against `v3` at `04a51be`. This file is the
only current migration status surface. [migration-history.md](migration-history.md)
is the archive of earlier decisions, evidence and superseded dispatch cards; its
"current" and "next" instructions are historical, not work to resume.

## Exact next safe action

Resume the paused M85 FE² scale mission on `feat/fe9-fe2-scale` from its clean,
unmerged `6073fc5`, based on `04a51be`. No agent, benchmark or other background
process is running, and no resumable result JSON exists. The old full-harness
log is historical evidence only; do not repeat `bench.run all`.

The deterministic per-IP task pool, raw-uint64 N-thread/1-thread pins and
retained nested-buffer identity pins are implemented. Only these evidence and
gate tasks remain:

1. Run identical targeted warm candidate and serial-control measurements,
   preserving every FE² and FE²-memory workload and the full required thread
   sweep. Commit both as new result JSONs, compare candidate against control,
   report the memory-vs-IP evidence and honestly label the partial benchmark
   coverage. Leave `baseline_m3.json` unchanged.
2. Run the full repository suite in default and `PYTHONINTMAXSTRDIGITS=640`
   modes, then all whole-tree Ruff gates: the four exact CI check invocations
   and the bench check/format pair.

## Merged capability state at `04a51be`

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
- The M75-M84 wave is merged through M84's `5ec8f0b`: RVE homogenization,
  repair reconciliation, nested-state composition and rollback, periodic-cell
  authoring, the nested FE² continuum operator, and typed cross-level failure
  and cost propagation are current source and test capabilities, not remaining
  dispatch items.

Reconcile refs and source before dispatching new work. Current architecture and
behavior are defined by [generic_core.md](generic_core.md),
[design.md](design.md) and the implementation; historical mission cards and
reviews retain provenance but do not supersede this live record.
