# bench/ — in-repo benchmark harness

This harness is a **regression gate, not a performance goal**. PyFEM v3 bets
that data-oriented design with good abstractions cannot be slow; the harness
exists so that future work cannot quietly give that property back. Two rules
govern everything below:

1. **Correctness before timing.** Every benchmark first proves the candidate
   (v3) solution against the legacy reference (rtol 1e-10 for generated
   patches and linear skims; nonlinear finite-strain and Riks skims use their
   own `skims/*/parity.toml`, the repo's established oracle). A failed gate
   means no timings are recorded for that workload — a faster wrong result is
   a failed benchmark.
2. **Gate on ratios, not absolutes.** Absolute times move with machines;
   regressions are judged as `candidate_median / baseline_median` against the
   committed baseline JSON.

## Running it

All commands run from the repository root with the project venv (numba,
numpy, scipy are already in it; the harness adds no dependencies):

```bash
.venv/bin/python -m bench.run all        # gates + warm + cold + write JSON + ratio check
.venv/bin/python -m bench.run all --quick  # sizes<=16, threads 1,16 — fast iteration
.venv/bin/python -m bench.run warm       # stage-decomposed warm benchmarks only
.venv/bin/python -m bench.run cold       # fresh-subprocess cold benchmarks only
.venv/bin/python -m bench.run gates      # correctness gates only (no timing)
.venv/bin/python -m bench.run check --candidate bench/results/<run>.json
.venv/bin/python -m bench.run manifest   # print the environment manifest
```

A full run takes roughly 15–25 minutes on the reference machine, dominated by
the legacy side's superlinear assembly cost at 64x64 (that cost is exactly
what the harness quantifies). The warm phase runs skims **before** the scale
sweep: skim milliseconds degrade under thermal throttling once sustained
all-core load has heated the machine (measured 2.9x on cantilever8 after
13 minutes of load on the reference laptop), while the scale ratios are far
less clock-sensitive. For the same reason, compare runs made under similar
thermal conditions, and prefer a cool, plugged-in machine for baseline
promotion. Useful flags: `--sizes 2,4,8,16,32,64`,
`--materials PlaneStress,PlaneStrain`, `--threads 1,2,4,8,16`,
`--budget-s 0.6` (adaptive-rep time budget per stage), `--threshold 1.25`,
`--out <path>`, `--no-check`, `--no-skims`, `--no-true-cold`.

Results land in `bench/results/run_<utc>.json` and are checked against
`bench/results/baseline_m3.json` unless `--baseline` says otherwise; the exit
code is non-zero when the gate fails, so CI can block on it. Pre-warm the
numba cache before CI benchmarks (any warm run or the test suite does this);
see *Cold modes* below for the cache-cleared variant.

## Workloads and why they are comparable

**Uniform Q8 patches (`q8patch/<n>x<n>/<material>`, n = 2..64).** Both sides
are built from the *same* generator, `pyfem.v3.mesh.refined_patch
.build_uniform_q8_patch`: identical node numbering, coordinates, serendipity
connectivity, and prescribed boundary field (the book's PatchTest8 linear
displacement). The legacy side is materialized by
`bench/legacy_cases.write_legacy_q8_patch` as `.dat`/`.pro` files with
`%.17g` coordinates (identical mesh by construction, M3's method); the v3
side packs the same mesh via `build_uniform_q8_loaded`. PlaneStress and
PlaneStrain share geometry and differ only in the constitutive matrix, so
cross-material ratios isolate material-path costs. The workload scales
4 → 4096 elements (42 → 25090 DOFs), spanning overhead-, kernel-, and
factorization-dominated regimes. Comparability is enforced, not assumed: the
gate requires the two sides' solution vectors to agree at rtol 1e-10 *and*
the v3 solution to match the analytic patch field at rtol 1e-8 (so a bug
shared by both implementations cannot pass).

**Skim cases (`skim/<name>`).** The shipped engineering cases —
`patch_test8`, `patch_test3`, `patch_test4`, `patch_test8_mpc`,
`patch_test8_plane_strain`, `patch_test8_loaded` (linear),
`patch_test8_nonlinear`, `cantilever8` (nonlinear), `shallow_truss_riks`
(arc-length) — driven through the exact same files on both sides: v3 via
`load_problem(skim.pro)` + `solve_linear`/`solve_nonlinear`/`solve_riks`,
legacy via `InputRead` + the matching solver with the settings overrides
from `test/v3/_legacy_parity.py` (replicated numba-free in
`bench/legacy_settings.py`; verified bitwise-identical to the parity-test
oracle on all nonlinear skims). End-to-end skim times are **solve-only** —
per-rep loads are excluded by reloading outside the timed region
(`_per_rep_solver_loop`), matching the M3 envelope ("solve only; load
excludes"). Skims are correctness-first engineering anchors: they are small
on purpose, and their ratios mostly expose fixed overheads.

**Load path.** The `load` metric times v3 `build_uniform_q8_loaded` (mesh
generation + DOF map + `pack_problem`) against legacy `InputRead` on the
identical generated files. This is the benchmark that catches the known
quadratic `pack_problem._dof_node_type` regression class (M3 §4: v3 load
loses at >=32x32), which pure assembly timings would never see.

## What is measured

**Stage decomposition** (v3, per workload; `bench/v3_pipeline.py`):
`meshgen`, `load` (pack), `kernel` (batched numba element stiffness),
`scatter` (COO fill), `dedup` (COO→CSR), `assemble_coo` (the public
`assemble_loaded` = kernel+scatter+COO, matching M3's "asm" envelope),
`assemble` (plus CSR), `constrain` (C build + CᵀKC), `factor` (SuperLU),
`backsolve` (cached-factorization repeat solve), `solve` (single-shot
constrained solve), `e2e` (public `prepare_cached_linear(...).solve()`).
Stages scale differently (D2 §7.3); totals hide regressions. Legacy records
`load`, `assemble_coo`, `solve` — the stages its architecture exposes.

**Thread sweep 1/2/4/8/16.** Stages with a threaded code path (`kernel`,
`scatter`, `assemble_coo`, `assemble`, `e2e`) are measured at every sweep
count via `numba.set_num_threads` + `threadpool_limits`; stages with no
threaded code path (`meshgen`, `load`, `dedup`, `constrain`, `factor`,
`backsolve`, `solve`) are measured once at the reference count (16), inside
that record. Legacy is single-threaded (threads=1) — its per-element Python
loop does not thread, and its solve is single-threaded SuperLU like v3's.

**Repetitions and statistics.** Warm timings use warmup + adaptive reps
(minimum 5, up to 30, targeting `--budget-s` seconds of measured time per
cell) and report min/median/mean plus all samples. Two documented exceptions
match the M3 envelope: legacy assembly in its superlinear regime uses 5
reps ≤16x16, 3 at 32x32, 1 at 64x64 (marked in the cell's `note`), and the
pure-Python `meshgen`/`load` stages use 3 fixed reps. Cold subprocess cells
are single-shot by construction.

**Memory.** Per record: `logical_bytes` (K_e batch, COO, CSR array sizes)
and per-stage `rss_delta_mb` — the process peak-RSS high-water-mark increase
across that stage's rep block (not a per-rep allocation profile). Cold
records carry the worker's peak RSS in the `wall` metric's `rss_delta_mb`.

**Cold modes** (`bench.run cold`). Each measurement is a fresh
`<venv>/bin/python -m bench.cold_worker` subprocess with
`NUMBA_NUM_THREADS` and the BLAS thread env pinned; the parent times the
subprocess wall (M3's method) and the worker reports its own import /
load / first-assemble / solve breakdown plus peak RSS. Two flavors:
- *cold* (default): fresh process, numba disk cache present — the CI
  situation after a pre-warm step;
- *true cold* (`--true-cold`, default on): `NUMBA_CACHE_DIR` pointed at an
  empty temporary directory for one designated case (`skim:patch_test8`),
  forcing the ~10 s one-time JIT compile M3 measured. This is the
  cache-cleared mode; it is separate because it belongs to
  first-run-on-a-new-machine accounting, not to commit-to-commit gating.

**Environment manifest.** Every run JSON records CPU, OS, Python, git
revision, numpy/scipy/numba versions, the numpy BLAS/LAPACK backend, thread
environment variables, and the numba thread default (`bench/manifest.py`).

## Regression DB and gates

`bench/results/baseline_m3.json` is the committed seed: the M3 baseline
(agent-g1p, 2026-09-29, i9-9980HK), transcribed into the same schema with
provenance notes. `bench/results/run_*.json` are harness runs.

`bench.run check` compares a candidate against a baseline per cell
(`category, workload, side, mode, threads, metric`):

- **regression** — `ratio > threshold` (default 1.25) **and** the absolute
  slowdown exceeds the 1 ms noise floor (D2's measured warm fixed overhead;
  below it, ratios are timer jitter, and a gate that trips on jitter trains
  people to ignore it). Fails the gate. Baseline cells in a declared
  high-variance class carry a per-cell `gate_threshold` that overrides the
  default for that cell only — currently all legacy scale assemble/solve
  cells at 1.6 (M3 declares ±50% run-to-run variance at ≥32x32, and ±30%
  same-day cross-process flap was observed at 16x16: legacy assembly is
  interpreter-bookkeeping-bound and environment-sensitive, while the v3-side
  cells that gate v3 work stay at 1.25).
- **missing** — a baseline cell the candidate does not produce (a workload
  or metric silently disappeared). Fails the gate.
- **correctness failure** — any candidate record whose gate did not pass.
  Fails the gate, independent of timings.
- **improvement** (ratio < 0.8) and **new** cells are informational.

When v3 code changes intentionally move a number, regenerate the baseline
with a full `bench.run all` on the reference machine, review the diff like a
code change, and commit it as the new `baseline_m3.json` successor (keep the
old file; baselines are history).

## Layout

```
bench/
  run.py             CLI: gates | warm | cold | all | check | manifest
  v3_pipeline.py     stage-decomposed v3 timed callables (uniform Q8)
  legacy_cases.py    generated legacy .dat/.pro writer + legacy drivers (numba-free)
  legacy_settings.py numba-free replica of the v3 skim solver-settings parsers
  gates.py           correctness gates (legacy parity + analytic patch field)
  cold_worker.py     one fresh-subprocess cold measurement, JSON on stdout
  db.py              cell flattening + ratio-gate comparison
  workloads.py       workload registry (Q8 sizes/materials, skim cases)
  manifest.py        environment manifest
  common.py          timing core (adaptive reps, stats, RSS), run JSON IO
  results/           committed regression DB (baseline_m3.json + runs)
  generated/         generated legacy .dat/.pro (gitignored, reproducible)
```

Design sources: M3 baseline report and D2 §7 harness requirements
(`.tower/comms/inbox/20260929-agent-g1p-*-m3-*.md`,
`.tower/comms/inbox/20260929-agent-d2-*-m5-*.md`).
