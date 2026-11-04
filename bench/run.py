"""Benchmark harness CLI.

Usage (from the repository root, with the project venv)::

  .venv/bin/python -m bench.run all       # gates, warm, cold, write JSON, check
  .venv/bin/python -m bench.run warm      # stage-decomposed warm benchmarks
  .venv/bin/python -m bench.run fe2       # matched damage FE2 + JIT evidence
  .venv/bin/python -m bench.run cold      # fresh-subprocess cold benchmarks
  .venv/bin/python -m bench.run gates     # correctness gates only
  .venv/bin/python -m bench.run check --candidate results/<run>.json
  .venv/bin/python -m bench.run manifest  # print the environment manifest

Every benchmark is gated on legacy parity before timings are recorded; a
faster wrong result is a failed benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import warnings
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scipy.sparse import SparseEfficiencyWarning

from bench.common import (
  REPO_ROOT,
  RESULTS_DIR,
  BenchRecord,
  MetricStats,
  measure,
  measure_fixed,
  read_run,
  write_run,
)
from bench.db import DEFAULT_THRESHOLD, compare
from bench.family_v3 import (
  cantilever_family_load,
  cantilever_family_solve,
  riks_fan_load,
  riks_fan_solve,
)
from bench.fe2_pipeline import (
  FE2Pipeline,
  gate_fe2,
  run_fe2_memory_case,
  solve_fresh_fe2,
)
from bench.gates import (
  CANONICAL_LEGACY_MAX_SIZE,
  GateResult,
  check_fast_reference_selftest,
  gate_finite_strain,
  gate_material,
  gate_q8,
  gate_riks_fan,
  gate_skim,
  material_kernel_batch,
)
from bench.legacy_cases import (
  legacy_assembly,
  legacy_load,
  legacy_nonlinear_solver,
  legacy_riks_solver,
  legacy_solve,
  quiet_legacy,
  silence_legacy_logging,
  write_legacy_cantilever,
  write_legacy_q8_patch,
  write_legacy_truss_fan,
)
from bench.manifest import check_cache_coherence, collect_manifest
from bench.v3_pipeline import V3Q8Pipeline
from bench.workloads import (
  Q8_MATERIALS,
  Q8_SIZES,
  FE2MemoryCase,
  FE2Workload,
  FiniteStrainWorkload,
  MaterialKernelCase,
  Q8Workload,
  RiksFanWorkload,
  SkimCase,
  fe2_cases,
  fe2_memory_cases,
  finite_strain_cases,
  material_cases,
  q8_workloads,
  riks_fan_cases,
  skim_cases,
)

# Stages with a threaded (numba parallel) code path are measured at every
# sweep thread count. The other stages are measured once at the reference count;
# dedup deliberately uses this reference-count sampling despite its size-gated
# parallel implementation, while the remaining reference stages have no threaded path.
SWEPT_STAGES = ("kernel", "scatter", "assemble_coo", "assemble", "e2e")
REFERENCE_STAGES = (
  "meshgen",
  "load",
  "dedup",
  "constrain",
  "factor",
  "backsolve",
  "solve",
)
REFERENCE_THREADS = 16

DEFAULT_THREADS = (1, 2, 4, 8, 16)

COLD_V3_THREADS = (1, 16)
COLD_LEGACY_THREADS = (1,)
DEFAULT_COLD_Q8 = ((8, "PlaneStress"), (32, "PlaneStress"))
DEFAULT_COLD_SKIMS = ("patch_test8", "cantilever8", "shallow_truss_riks")
DEFAULT_COLD_FAMILY = ("family:tl-cantilever:16x2", "family:riks-fan:32")
TRUE_COLD_CASE = "skim:patch_test8"

BLAS_ENV_VARS = (
  "OMP_NUM_THREADS",
  "OPENBLAS_NUM_THREADS",
  "MKL_NUM_THREADS",
  "VECLIB_MAXIMUM_THREADS",
)


def _threadpool_limits(threads: int) -> AbstractContextManager:
  try:
    from threadpoolctl import threadpool_limits
  except ImportError:
    return nullcontext()
  return threadpool_limits(limits=threads)


def _set_numba_threads(threads: int) -> None:
  import numba

  numba.set_num_threads(threads)


def _legacy_assemble_rep_policy(nx: int) -> tuple[int, str]:
  """(reps, note) for legacy assembly in its superlinear regime (M3 envelope)."""
  if nx <= 16:
    return 5, ""
  if nx <= 32:
    return 3, "reduced reps: legacy superlinear cost (M3 used 2-3 here)"
  return 1, "single rep: legacy superlinear cost (M3 envelope)"


def _failed_record(
  category: str,
  workload: str,
  side: str,
  mode: str,
  gate: GateResult,
) -> BenchRecord:
  return BenchRecord(
    category=category,
    workload=workload,
    side=side,
    mode=mode,
    threads=0,
    n_elems=0,
    n_dofs=0,
    metrics={},
    correctness=gate.to_json(),
  )


def _per_rep_solver_loop(
  load: Callable[[], Any],
  solve: Callable[[Any], Any],
  *,
  warmup: int,
  reps: int,
) -> MetricStats:
  """
  Time a solver loop with a fresh, untimed load before every rep.

  Legacy nonlinear drivers need a fresh ``InputRead`` per rep to rerun
  honestly; only the solver loop enters the samples (M3: "solve only; load
  excludes").
  """
  for _ in range(warmup):
    solve(load())
  samples: list[float] = []
  for _ in range(reps):
    handle = load()
    t0 = time.perf_counter()
    solve(handle)
    samples.append((time.perf_counter() - t0) * 1e3)
  return MetricStats(
    min_ms=min(samples),
    median_ms=sorted(samples)[len(samples) // 2],
    mean_ms=sum(samples) / len(samples),
    reps=len(samples),
    samples_ms=samples,
    note="solver loop only; per-rep load excluded (M3 envelope)",
  )


def run_q8_workload(
  workload: Q8Workload,
  *,
  threads: list[int],
  budget_s: float,
) -> tuple[list[BenchRecord], GateResult]:
  """Gate one uniform Q8 workload, then record v3 and legacy stage timings."""
  from pyfem.v3.mesh.refined_patch import build_uniform_q8_loaded

  records: list[BenchRecord] = []
  pro_path = write_legacy_q8_patch(
    workload.nx, workload.ny, material_type=workload.material_type
  )

  # Legacy timed path; the first assembly also feeds the large-size gate.
  load_stats = measure_fixed(lambda: legacy_load(pro_path), warmup=0, reps=3)
  props, globdat = legacy_load(pro_path)
  reps, note = _legacy_assemble_rep_policy(workload.nx)
  assemble_stats = measure_fixed(
    lambda: legacy_assembly(props, globdat), warmup=0, reps=reps, note=note
  )
  matrix = legacy_assembly(props, globdat)
  solve_stats = measure(
    lambda: legacy_solve(props, globdat, matrix), warmup=1, budget_s=budget_s
  )
  legacy_state = legacy_solve(props, globdat, matrix)
  legacy_record = BenchRecord(
    category="scale",
    workload=workload.name,
    side="legacy",
    mode="warm",
    threads=1,
    n_elems=workload.n_elems,
    n_dofs=workload.n_dofs,
    metrics={"load": load_stats, "assemble_coo": assemble_stats, "solve": solve_stats},
  )

  loaded = build_uniform_q8_loaded(
    workload.nx, workload.ny, material_type=workload.material_type
  )
  if workload.nx > CANONICAL_LEGACY_MAX_SIZE:
    reference, reference_kind = legacy_state, "legacy-single-assembly"
  else:
    reference, reference_kind = None, None
  gate = gate_q8(
    workload,
    reference=reference,
    reference_kind=reference_kind,
    loaded=loaded,
  )
  if not gate.passed:
    for side in ("v3", "legacy"):
      records.append(_failed_record("scale", workload.name, side, "warm", gate))
    return records, gate

  legacy_record.correctness = gate.to_json()
  records.append(legacy_record)

  pipeline = V3Q8Pipeline(workload, loaded=loaded)
  logical = pipeline.logical_bytes
  for thread_count in threads:
    _set_numba_threads(thread_count)
    stage_names = list(SWEPT_STAGES)
    if thread_count == REFERENCE_THREADS:
      stage_names += list(REFERENCE_STAGES)
    metrics: dict[str, MetricStats] = {}
    with _threadpool_limits(thread_count):
      for stage in stage_names:
        fn = getattr(pipeline, stage)
        if stage in ("meshgen", "load"):
          metrics[stage] = measure_fixed(fn, warmup=0, reps=3)
        else:
          metrics[stage] = measure(fn, warmup=1, budget_s=budget_s)
    records.append(
      BenchRecord(
        category="scale",
        workload=workload.name,
        side="v3",
        mode="warm",
        threads=thread_count,
        n_elems=workload.n_elems,
        n_dofs=workload.n_dofs,
        metrics=metrics,
        logical_bytes=logical,
        correctness=gate.to_json(),
      )
    )
  return records, gate


def _legacy_linear_run(solver_type: type, handle: tuple) -> None:
  with quiet_legacy():
    solver_type(handle[0], handle[1]).run(handle[0], handle[1])


def _v3_skim_solver(case: SkimCase) -> Callable[[Any], Any]:
  from pyfem.v3 import solve_linear, solve_nonlinear, solve_riks

  if case.kind == "linear":
    return solve_linear
  if case.kind == "nonlinear":
    return lambda loaded: solve_nonlinear(loaded).state
  return lambda loaded: solve_riks(loaded).state


def run_skim_case(
  case: SkimCase,
  *,
  threads: list[int],
  budget_s: float,
) -> tuple[list[BenchRecord], GateResult]:
  """Gate one skim case, then record load and end-to-end (solve-only) times."""
  from pyfem.solvers.LinearSolver import LinearSolver
  from pyfem.v3 import load_problem

  records: list[BenchRecord] = []
  gate = gate_skim(case)
  if not gate.passed:
    for side in ("v3", "legacy"):
      records.append(_failed_record("skim", case.workload, side, "warm", gate))
    return records, gate

  solver = _v3_skim_solver(case)
  loaded = load_problem(case.pro_path)
  for thread_count in threads:
    _set_numba_threads(thread_count)
    with _threadpool_limits(thread_count):
      load_stats = measure(
        lambda: load_problem(case.pro_path), warmup=1, budget_s=budget_s
      )
      e2e_stats = measure(lambda: solver(loaded), warmup=1, budget_s=budget_s)
    records.append(
      BenchRecord(
        category="skim",
        workload=case.workload,
        side="v3",
        mode="warm",
        threads=thread_count,
        n_elems=0,
        n_dofs=int(loaded.problem.n_dofs),
        metrics={"load": load_stats, "e2e": e2e_stats},
        correctness=gate.to_json(),
      )
    )

  load_stats = measure(lambda: legacy_load(case.pro_path), warmup=1, budget_s=budget_s)
  if case.kind == "linear":
    legacy_e2e = _per_rep_solver_loop(
      lambda: legacy_load(case.pro_path),
      lambda handle: _legacy_linear_run(LinearSolver, handle),
      warmup=1,
      reps=5,
    )
    n_dofs = int(loaded.problem.n_dofs)
  else:
    solver_fn = (
      legacy_nonlinear_solver if case.kind == "nonlinear" else legacy_riks_solver
    )
    legacy_e2e = _per_rep_solver_loop(
      lambda: legacy_load(case.pro_path),
      lambda handle: solver_fn(handle[0], handle[1], case.pro_path),
      warmup=1,
      reps=5 if case.kind == "riks" else 3,
    )
    n_dofs = int(loaded.problem.n_dofs)
  records.append(
    BenchRecord(
      category="skim",
      workload=case.workload,
      side="legacy",
      mode="warm",
      threads=1,
      n_elems=0,
      n_dofs=n_dofs,
      metrics={"load": load_stats, "e2e": legacy_e2e},
      correctness=gate.to_json(),
    )
  )
  return records, gate


def run_material_case(
  case: MaterialKernelCase,
  *,
  threads: list[int],
  budget_s: float,
) -> tuple[list[BenchRecord], GateResult]:
  """Gate one material-kernel batch, then time reference vs optimized sides."""
  from pyfem.v3.materials.isotropic_hardening_plasticity import (
    isotropic_hardening_plasticity_kernel,
    isotropic_hardening_plasticity_kernel_reference,
  )

  records: list[BenchRecord] = []
  gate = gate_material(case)
  if not gate.passed:
    for side in ("reference", "optimized"):
      records.append(_failed_record("material", case.workload, side, "warm", gate))
    return records, gate

  strains, rows, calibration = material_kernel_batch(case)
  logical = {
    "strains": strains.nbytes,
    "accepted_rows": rows.nbytes,
    "stresses": strains.nbytes,
    "tangents": 6 * strains.nbytes,
    "trial_rows": rows.nbytes,
  }
  note = f"{case.n_entities} entities per call (documented batch, virgin rows)"
  reference_stats = measure_fixed(
    lambda: isotropic_hardening_plasticity_kernel_reference(strains, rows, calibration),
    warmup=1,
    reps=3,
    note=note + "; reduced reps: pure-NumPy per-entity reference loop",
  )
  records.append(
    BenchRecord(
      category="material",
      workload=case.workload,
      side="reference",
      mode="warm",
      threads=1,
      n_elems=0,
      n_dofs=case.n_entities,
      metrics={"kernel": reference_stats},
      logical_bytes=logical,
      correctness=gate.to_json(),
    )
  )
  for thread_count in threads:
    _set_numba_threads(thread_count)
    optimized_stats = measure(
      lambda: isotropic_hardening_plasticity_kernel(strains, rows, calibration),
      warmup=1,
      budget_s=budget_s,
    )
    optimized_stats.note = note
    records.append(
      BenchRecord(
        category="material",
        workload=case.workload,
        side="optimized",
        mode="warm",
        threads=thread_count,
        n_elems=0,
        n_dofs=case.n_entities,
        metrics={"kernel": optimized_stats},
        logical_bytes=logical,
        correctness=gate.to_json(),
      )
    )
  return records, gate


def _solved_fe2(case: FE2Workload, threads: int) -> FE2Pipeline:
  _set_numba_threads(threads)
  with _threadpool_limits(threads):
    pipeline = FE2Pipeline(case)
    pipeline.solve()
  return pipeline


def run_fe2_case(
  case: FE2Workload,
  *,
  threads: list[int],
  budget_s: float,
  e2e_reps: int = 1,
) -> tuple[list[BenchRecord], list[GateResult]]:
  """Race-gate each FE2 thread count before recording its stage timings."""
  records: list[BenchRecord] = []
  gates: list[GateResult] = []
  reference = _solved_fe2(case, 1)
  for thread_count in threads:
    candidate = _solved_fe2(case, thread_count)
    _set_numba_threads(thread_count)
    with _threadpool_limits(thread_count):
      gate = gate_fe2(reference, candidate)
      gates.append(gate)
      if not gate.passed:
        records.append(_failed_record("fe2", case.name, "v3", "warm", gate))
        continue
      nested_stats = measure(candidate.nested_evaluation, warmup=1, budget_s=budget_s)
      nested_stats.note = "committed macro state; nested trial rejected after each call"
      load_stats = measure_fixed(lambda: FE2Pipeline(case), warmup=0, reps=1)
      load_stats.note = "compile + real virgin nested probe"
      e2e_stats = measure_fixed(lambda: solve_fresh_fe2(case), warmup=0, reps=e2e_reps)
      e2e_stats.note = "fresh-e2e: fresh compile + virgin probe + macro nonlinear solve"
      evidence = candidate.evidence()
    records.append(
      BenchRecord(
        category="fe2",
        workload=case.name,
        side="v3",
        mode="warm",
        threads=thread_count,
        n_elems=case.n_elems,
        n_dofs=case.n_dofs,
        metrics={
          "load": load_stats,
          "nested_evaluation": nested_stats,
          "e2e": e2e_stats,
        },
        logical_bytes=candidate.logical_bytes,
        correctness={**gate.to_json(), "evidence": evidence},
      )
    )
  return records, gates


def _single_sample(milliseconds: float, note: str = "") -> MetricStats:
  return MetricStats(
    min_ms=milliseconds,
    median_ms=milliseconds,
    mean_ms=milliseconds,
    reps=1,
    samples_ms=[milliseconds],
    note=note,
  )


def _fe2_subprocess(
  arguments: list[str], *, threads: int, true_cold: bool
) -> tuple[dict[str, Any], float]:
  env = os.environ.copy()
  env["NUMBA_NUM_THREADS"] = str(threads)
  for var in BLAS_ENV_VARS:
    env[var] = str(threads)
  temporary = (
    tempfile.TemporaryDirectory(prefix="pyfem-fe2-jit-") if true_cold else None
  )
  if temporary is not None:
    env["NUMBA_CACHE_DIR"] = temporary.name
  try:
    start = time.perf_counter()
    output = subprocess.run(
      [sys.executable, "-m", "bench.fe2_pipeline", *arguments],
      env=env,
      cwd=REPO_ROOT,
      capture_output=True,
      text=True,
      check=True,
    )
    wall_ms = (time.perf_counter() - start) * 1e3
  finally:
    if temporary is not None:
      temporary.cleanup()
  return json.loads(output.stdout.strip().splitlines()[-1]), wall_ms


def run_fresh_fe2_subprocess(case: FE2Workload, *, true_cold: bool) -> BenchRecord:
  """Measure a fresh-process FE2 solve with cache present or an empty cache."""
  payload, wall_ms = _fe2_subprocess(
    ["--fresh-worker", case.name, "--threads", "1"], threads=1, true_cold=true_cold
  )
  evidence = payload["evidence"]
  passed = (
    payload["status"] == "COMPLETED"
    and evidence["accepted_nested_sha256"] == evidence["rejected_nested_sha256"]
  )
  mode = "truecold" if true_cold else "cache-present"
  return BenchRecord(
    category="fe2-fresh",
    workload=case.name,
    side="v3",
    mode=mode,
    threads=1,
    n_elems=case.n_elems,
    n_dofs=case.n_dofs,
    metrics={
      "load": _single_sample(
        payload["load_ms"], "fresh-process compile + virgin probe"
      ),
      "solve": _single_sample(payload["solve_ms"], "fresh-process macro solve"),
      "fresh_e2e": _single_sample(
        payload["fresh_e2e_ms"], "fresh-process compile + virgin probe + macro solve"
      ),
      "wall": _single_sample(wall_ms, "parent-measured subprocess wall"),
    },
    correctness={"passed": passed, "evidence": evidence},
  )


def run_damage_jit_subprocess(*, true_cold: bool) -> BenchRecord:
  """Isolate first-call JIT from second-call warm kernel execution."""
  payload, wall_ms = _fe2_subprocess(["--jit-worker"], threads=1, true_cold=true_cold)
  passed = payload["status"] == "OK" and payload["repeat_identical"]
  mode = "truecold" if true_cold else "cache-present"
  return BenchRecord(
    category="damage-jit",
    workload="damage/plane-strain-public-kernel",
    side="v3",
    mode=mode,
    threads=1,
    n_elems=0,
    n_dofs=16,
    metrics={
      "jit": _single_sample(
        payload["jit_ms"], "first public kernel call in a fresh interpreter"
      ),
      "kernel": _single_sample(payload["warm_ms"], "second public kernel call"),
      "wall": _single_sample(wall_ms, "parent-measured subprocess wall"),
    },
    correctness={"passed": passed, "response_sha256": payload["response_sha256"]},
  )


def run_fe2_memory_workload(
  case: FE2MemoryCase,
) -> tuple[list[BenchRecord], GateResult]:
  """Record production/legacy allocation curves for one macro-IP count."""
  production, legacy, gate = run_fe2_memory_case(case)
  return [production, legacy], gate


def run_finite_strain_case(
  case: FiniteStrainWorkload,
  *,
  threads: list[int],
  budget_s: float,
) -> tuple[list[BenchRecord], GateResult]:
  """Gate one refined cantilever, then time the F4 deck stack vs legacy."""
  records: list[BenchRecord] = []
  pro_path = write_legacy_cantilever(case.nx, case.ny)
  gate = gate_finite_strain(case)
  if not gate.passed:
    for side in ("v3", "legacy"):
      records.append(_failed_record("family", case.name, side, "warm", gate))
    return records, gate

  prepared = cantilever_family_load(pro_path)
  for thread_count in threads:
    _set_numba_threads(thread_count)
    with _threadpool_limits(thread_count):
      load_stats = measure(
        lambda: cantilever_family_load(pro_path), warmup=1, budget_s=budget_s
      )
      e2e_stats = measure(
        lambda: cantilever_family_solve(prepared), warmup=1, budget_s=budget_s
      )
    records.append(
      BenchRecord(
        category="family",
        workload=case.name,
        side="v3",
        mode="warm",
        threads=thread_count,
        n_elems=case.n_elems,
        n_dofs=case.n_dofs,
        metrics={"load": load_stats, "e2e": e2e_stats},
        correctness=gate.to_json(),
      )
    )

  load_stats = measure(lambda: legacy_load(pro_path), warmup=1, budget_s=budget_s)
  legacy_e2e = _per_rep_solver_loop(
    lambda: legacy_load(pro_path),
    lambda handle: legacy_nonlinear_solver(handle[0], handle[1], pro_path),
    warmup=1,
    reps=3,
  )
  records.append(
    BenchRecord(
      category="family",
      workload=case.name,
      side="legacy",
      mode="warm",
      threads=1,
      n_elems=case.n_elems,
      n_dofs=case.n_dofs,
      metrics={"load": load_stats, "e2e": legacy_e2e},
      correctness=gate.to_json(),
    )
  )
  return records, gate


def run_riks_fan_case(
  case: RiksFanWorkload,
  *,
  threads: list[int],
  budget_s: float,
) -> tuple[list[BenchRecord], GateResult]:
  """Gate one truss-only fan, then time the M33 Riks driver vs legacy."""
  records: list[BenchRecord] = []
  pro_path = write_legacy_truss_fan(case.n_rays)
  gate = gate_riks_fan(case)
  if not gate.passed:
    for side in ("v3", "legacy"):
      records.append(_failed_record("family", case.name, side, "warm", gate))
    return records, gate

  prepared = riks_fan_load(case.n_rays)
  for thread_count in threads:
    _set_numba_threads(thread_count)
    with _threadpool_limits(thread_count):
      load_stats = measure(
        lambda: riks_fan_load(case.n_rays), warmup=1, budget_s=budget_s
      )
      e2e_stats = measure(lambda: riks_fan_solve(prepared), warmup=1, budget_s=budget_s)
    records.append(
      BenchRecord(
        category="family",
        workload=case.name,
        side="v3",
        mode="warm",
        threads=thread_count,
        n_elems=case.n_elems,
        n_dofs=case.n_dofs,
        metrics={"load": load_stats, "e2e": e2e_stats},
        correctness=gate.to_json(),
      )
    )

  load_stats = measure(lambda: legacy_load(pro_path), warmup=1, budget_s=budget_s)
  legacy_e2e = _per_rep_solver_loop(
    lambda: legacy_load(pro_path),
    lambda handle: legacy_riks_solver(handle[0], handle[1], pro_path),
    warmup=1,
    reps=5,
  )
  records.append(
    BenchRecord(
      category="family",
      workload=case.name,
      side="legacy",
      mode="warm",
      threads=1,
      n_elems=case.n_elems,
      n_dofs=case.n_dofs,
      metrics={"load": load_stats, "e2e": legacy_e2e},
      correctness=gate.to_json(),
    )
  )
  return records, gate


def _cold_workload_name(case: str) -> str:
  kind, _, rest = case.partition(":")
  if kind == "q8patch":
    n_str, _, material_type = rest.partition(":")
    return f"q8patch/{n_str}x{n_str}/{material_type}"
  if kind == "family":
    name, _, size = rest.partition(":")
    return f"{name}/{size}"
  return f"skim/{rest}"


def run_cold_case(
  case: str,
  side: str,
  threads: int,
  *,
  true_cold: bool = False,
  gate_passed: bool = True,
) -> BenchRecord:
  """Spawn one fresh-subprocess cold measurement and pack it as a record."""
  env = os.environ.copy()
  env["NUMBA_NUM_THREADS"] = str(threads)
  for var in BLAS_ENV_VARS:
    env[var] = str(threads)
  tmp_dir = None
  if true_cold:
    tmp_dir = tempfile.TemporaryDirectory(prefix="pyfem-truecold-")
    env["NUMBA_CACHE_DIR"] = tmp_dir.name
  try:
    wall0 = time.perf_counter()
    out = subprocess.run(
      [sys.executable, "-m", "bench.cold_worker", "--case", case, "--side", side],
      env=env,
      cwd=REPO_ROOT,
      capture_output=True,
      text=True,
      check=True,
    )
    wall_ms = (time.perf_counter() - wall0) * 1e3
  finally:
    if tmp_dir is not None:
      tmp_dir.cleanup()
  payload = json.loads(out.stdout.strip().splitlines()[-1])

  metrics: dict[str, MetricStats] = {}
  for key, metric in (
    ("import_ms", "import"),
    ("load_ms", "load"),
    ("assemble_ms", "assemble"),
    ("solve_ms", "solve"),
  ):
    ms = payload.get(key)
    if ms is not None:
      metrics[metric] = MetricStats(
        min_ms=ms, median_ms=ms, mean_ms=ms, reps=1, samples_ms=[ms]
      )
  metrics["wall"] = MetricStats(
    min_ms=wall_ms,
    median_ms=wall_ms,
    mean_ms=wall_ms,
    reps=1,
    samples_ms=[wall_ms],
    rss_delta_mb=float(payload.get("peak_rss_mb", 0.0)),
    note="parent-measured subprocess wall; rss_delta_mb = worker peak RSS",
  )
  correctness: dict[str, Any] = {"passed": gate_passed, "detail": "gated by parent"}
  analytic_rel = payload.get("analytic_rel")
  if analytic_rel is not None:
    correctness["analytic_rel"] = analytic_rel
  return BenchRecord(
    category="cold",
    workload=_cold_workload_name(case),
    side=side,
    mode="truecold" if true_cold else "cold",
    threads=threads,
    n_elems=int(payload.get("n_elems", 0)),
    n_dofs=int(payload.get("n_dofs", 0)),
    metrics=metrics,
    correctness=correctness,
  )


def do_gates(
  sizes: tuple[int, ...], materials: tuple[str, ...], *, verbose: bool = True
) -> int:
  """Run all correctness gates; return the number of failures."""
  failures = 0
  for check in check_fast_reference_selftest(materials):
    failures += 0 if check.passed else 1
    if verbose:
      print(
        f"[{'PASS' if check.passed else 'FAIL'}] {check.name} "
        f"max_rel={check.max_rel_diff:.3e}"
      )
  for workload in q8_workloads(sizes, materials):
    gate = gate_q8(workload)
    failures += 0 if gate.passed else 1
    if verbose:
      worst = max(c.max_rel_diff for c in gate.checks)
      print(
        f"[{'PASS' if gate.passed else 'FAIL'}] {gate.workload} max_rel={worst:.3e}"
      )
  for case in skim_cases():
    gate = gate_skim(case)
    failures += 0 if gate.passed else 1
    if verbose:
      worst = max(c.max_rel_diff for c in gate.checks)
      print(
        f"[{'PASS' if gate.passed else 'FAIL'}] {gate.workload} max_rel={worst:.3e}"
      )
  for case in fe2_cases():
    reference = _solved_fe2(case, 1)
    candidate = _solved_fe2(case, 2)
    _set_numba_threads(2)
    with _threadpool_limits(2):
      gate = gate_fe2(reference, candidate)
    failures += 0 if gate.passed else 1
    if verbose:
      worst = max(c.max_rel_diff for c in gate.checks)
      print(
        f"[{'PASS' if gate.passed else 'FAIL'}] {gate.workload} max_rel={worst:.3e}"
      )
  for case in material_cases():
    gate = gate_material(case)
    failures += 0 if gate.passed else 1
    if verbose:
      worst = max(c.max_rel_diff for c in gate.checks)
      print(
        f"[{'PASS' if gate.passed else 'FAIL'}] {gate.workload} max_rel={worst:.3e}"
      )
  for case in finite_strain_cases():
    gate = gate_finite_strain(case)
    failures += 0 if gate.passed else 1
    if verbose:
      worst = max(c.max_rel_diff for c in gate.checks)
      print(
        f"[{'PASS' if gate.passed else 'FAIL'}] {gate.workload} max_rel={worst:.3e}"
      )
  for case in riks_fan_cases():
    gate = gate_riks_fan(case)
    failures += 0 if gate.passed else 1
    if verbose:
      worst = max(c.max_rel_diff for c in gate.checks)
      print(
        f"[{'PASS' if gate.passed else 'FAIL'}] {gate.workload} max_rel={worst:.3e}"
      )
  return failures


def do_fe2(args: argparse.Namespace) -> list[BenchRecord]:
  """Run the matched public damage FE2 evidence matrix and startup probes."""
  threads = _parse_ints(args.threads)
  if not threads or any(count < 1 for count in threads) or args.e2e_reps < 1:
    raise SystemExit("FE2 threads and --e2e-reps must be positive")
  records: list[BenchRecord] = []
  for case in fe2_cases(damage_only=True):
    start = time.perf_counter()
    case_records, gates = run_fe2_case(
      case,
      threads=threads,
      budget_s=args.budget_s,
      e2e_reps=args.e2e_reps,
    )
    records.extend(case_records)
    for true_cold in (False, True):
      records.append(run_fresh_fe2_subprocess(case, true_cold=true_cold))
    passed = all(gate.passed for gate in gates)
    print(
      f"[{'PASS' if passed else 'FAIL'}] {case.name} "
      f"({time.perf_counter() - start:.1f}s)"
    )
  for true_cold in (False, True):
    records.append(run_damage_jit_subprocess(true_cold=true_cold))
  return records


def do_warm(args: argparse.Namespace) -> list[BenchRecord]:
  """Run the warm phase: selftest, per-workload gates, stage timings.

  Skims run first: they are small, M3-headline comparisons whose absolute
  milliseconds degrade under thermal throttling once the heavy scale sweep
  has heated the machine (measured 2.9x on cantilever8 after 13 min of
  sustained all-core load). The scale ratios are far less sensitive to clock
  speed, and within them small sizes come first for the same reason.
  """
  sizes = tuple(_parse_ints(args.sizes))
  materials = _parse_strs(args.materials)
  threads = _parse_ints(args.threads)
  records: list[BenchRecord] = []

  try:
    import threadpoolctl  # noqa: F401
  except ImportError:
    print(
      "note: threadpoolctl not installed; warm-phase BLAS pinning degrades to "
      "numba thread setting only (cold subprocesses still pin BLAS env vars)"
    )

  selftest = check_fast_reference_selftest(materials)
  for check in selftest:
    print(
      f"[{'PASS' if check.passed else 'FAIL'}] {check.name} "
      f"max_rel={check.max_rel_diff:.3e}"
    )
  if not all(c.passed for c in selftest):
    raise SystemExit("fast legacy reference selftest failed; aborting")

  if not args.no_skims:
    for case in skim_cases():
      t0 = time.perf_counter()
      recs, gate = run_skim_case(case, threads=threads, budget_s=args.budget_s)
      records.extend(recs)
      print(
        f"[{'PASS' if gate.passed else 'FAIL'}] {case.workload} "
        f"({time.perf_counter() - t0:.1f}s)"
      )

  for case in fe2_cases():
    t0 = time.perf_counter()
    recs, gates = run_fe2_case(case, threads=threads, budget_s=args.budget_s)
    records.extend(recs)
    passed = all(gate.passed for gate in gates)
    print(
      f"[{'PASS' if passed else 'FAIL'}] {case.name} ({time.perf_counter() - t0:.1f}s)"
    )

  for case in fe2_memory_cases():
    t0 = time.perf_counter()
    recs, gate = run_fe2_memory_workload(case)
    records.extend(recs)
    print(
      f"[{'PASS' if gate.passed else 'FAIL'}] {case.workload} "
      f"({time.perf_counter() - t0:.1f}s)"
    )

  for case in material_cases():
    t0 = time.perf_counter()
    recs, gate = run_material_case(case, threads=threads, budget_s=args.budget_s)
    records.extend(recs)
    print(
      f"[{'PASS' if gate.passed else 'FAIL'}] {case.workload} "
      f"({time.perf_counter() - t0:.1f}s)"
    )

  for case in finite_strain_cases():
    t0 = time.perf_counter()
    recs, gate = run_finite_strain_case(case, threads=threads, budget_s=args.budget_s)
    records.extend(recs)
    print(
      f"[{'PASS' if gate.passed else 'FAIL'}] {case.name} "
      f"({time.perf_counter() - t0:.1f}s)"
    )

  for case in riks_fan_cases():
    t0 = time.perf_counter()
    recs, gate = run_riks_fan_case(case, threads=threads, budget_s=args.budget_s)
    records.extend(recs)
    print(
      f"[{'PASS' if gate.passed else 'FAIL'}] {case.name} "
      f"({time.perf_counter() - t0:.1f}s)"
    )

  for workload in q8_workloads(sizes, materials):
    t0 = time.perf_counter()
    recs, gate = run_q8_workload(workload, threads=threads, budget_s=args.budget_s)
    records.extend(recs)
    print(
      f"[{'PASS' if gate.passed else 'FAIL'}] {workload.name} "
      f"({time.perf_counter() - t0:.1f}s)"
    )
  return records


def do_cold(args: argparse.Namespace) -> list[BenchRecord]:
  """Run the cold phase: in-parent gates, then fresh-subprocess measurements."""
  sizes = tuple(_parse_ints(args.sizes))
  cold_q8 = [(n, mat) for n, mat in DEFAULT_COLD_Q8 if n in sizes]
  records: list[BenchRecord] = []

  gate_by_workload: dict[str, bool] = {}
  for n, mat in cold_q8:
    gate = gate_q8(Q8Workload(n, n, mat))
    gate_by_workload[gate.workload] = gate.passed
    print(f"[{'PASS' if gate.passed else 'FAIL'}] gate {gate.workload}")
  skim_by_name = {c.name: c for c in skim_cases()}
  for name in DEFAULT_COLD_SKIMS:
    gate = gate_skim(skim_by_name[name])
    gate_by_workload[gate.workload] = gate.passed
    print(f"[{'PASS' if gate.passed else 'FAIL'}] gate {gate.workload}")
  finite_strain_by_name = {c.name: c for c in finite_strain_cases()}
  riks_fan_by_name = {c.name: c for c in riks_fan_cases()}
  for spec in DEFAULT_COLD_FAMILY:
    workload = _cold_workload_name(spec)
    if workload in finite_strain_by_name:
      gate = gate_finite_strain(finite_strain_by_name[workload])
    else:
      gate = gate_riks_fan(riks_fan_by_name[workload])
    gate_by_workload[gate.workload] = gate.passed
    print(f"[{'PASS' if gate.passed else 'FAIL'}] gate {gate.workload}")

  cases = [f"q8patch:{n}:{mat}" for n, mat in cold_q8]
  cases += [f"skim:{name}" for name in DEFAULT_COLD_SKIMS]
  cases += list(DEFAULT_COLD_FAMILY)
  for case in cases:
    passed = gate_by_workload.get(_cold_workload_name(case), False)
    for threads in COLD_V3_THREADS:
      t0 = time.perf_counter()
      records.append(run_cold_case(case, "v3", threads, gate_passed=passed))
      print(f"cold {case} v3/{threads}T ({time.perf_counter() - t0:.1f}s)")
    for threads in COLD_LEGACY_THREADS:
      t0 = time.perf_counter()
      records.append(run_cold_case(case, "legacy", threads, gate_passed=passed))
      print(f"cold {case} legacy ({time.perf_counter() - t0:.1f}s)")
  if args.true_cold:
    passed = gate_by_workload.get(_cold_workload_name(TRUE_COLD_CASE), False)
    t0 = time.perf_counter()
    records.append(
      run_cold_case(
        TRUE_COLD_CASE, "v3", REFERENCE_THREADS, true_cold=True, gate_passed=passed
      )
    )
    print(
      f"true-cold {TRUE_COLD_CASE} v3/{REFERENCE_THREADS}T "
      f"({time.perf_counter() - t0:.1f}s)"
    )
  return records


def _parse_ints(text: str) -> list[int]:
  return [int(part) for part in text.split(",") if part.strip()]


def _parse_strs(text: str) -> tuple[str, ...]:
  return tuple(part.strip() for part in text.split(",") if part.strip())


def _default_out_path() -> Path:
  stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
  return RESULTS_DIR / f"run_{stamp}.json"


def _coverage(args: argparse.Namespace) -> str:
  """
  Full-coverage iff the run produced at least the complete default workload
  matrix: the ``all`` pipeline (warm + cold), skims included, sizes and
  materials covering the defaults, the full default thread sweep, and the
  true-cold case. Comparisons are set-containment, so reordered or superset
  arguments on a genuinely complete run still stamp "full" (missing-cell
  detection stays meaningful); anything reduced stamps "partial".
  """
  if getattr(args, "command", "") != "all":
    return "partial"
  if args.no_skims or not getattr(args, "true_cold", False):
    return "partial"
  if not set(Q8_SIZES) <= set(_parse_ints(args.sizes)):
    return "partial"
  if not set(Q8_MATERIALS) <= set(_parse_strs(args.materials)):
    return "partial"
  if not set(DEFAULT_THREADS) <= set(_parse_ints(args.threads)):
    return "partial"
  return "full"


def _write_and_maybe_check(records: list[BenchRecord], args: argparse.Namespace) -> int:
  out = Path(args.out) if args.out else _default_out_path()
  write_run(out, collect_manifest(), records, coverage=_coverage(args))
  print(f"wrote {out}")
  if args.no_check:
    failed = any(not r.correctness.get("passed", True) for r in records)
    return 1 if failed else 0
  baseline_path = Path(args.baseline)
  if not baseline_path.exists():
    print(f"baseline {baseline_path} not found; skipping check")
    return 0
  report = compare(read_run(baseline_path), read_run(out), threshold=args.threshold)
  print(report.render())
  return 0 if report.passed else 1


def cmd_gates(args: argparse.Namespace) -> int:
  failures = do_gates(tuple(_parse_ints(args.sizes)), _parse_strs(args.materials))
  print(f"gates: {failures} failure(s)")
  return 1 if failures else 0


def cmd_warm(args: argparse.Namespace) -> int:
  records = do_warm(args)
  return _write_and_maybe_check(records, args)


def cmd_fe2(args: argparse.Namespace) -> int:
  records = do_fe2(args)
  return _write_and_maybe_check(records, args)


def cmd_cold(args: argparse.Namespace) -> int:
  records = do_cold(args)
  return _write_and_maybe_check(records, args)


def cmd_all(args: argparse.Namespace) -> int:
  records = do_warm(args)
  records += do_cold(args)
  return _write_and_maybe_check(records, args)


def cmd_check(args: argparse.Namespace) -> int:
  report = compare(
    read_run(Path(args.baseline)),
    read_run(Path(args.candidate)),
    threshold=args.threshold,
  )
  print(report.render())
  return 0 if report.passed else 1


def cmd_manifest(_args: argparse.Namespace) -> int:
  print(json.dumps(collect_manifest(), indent=2))
  return 0


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  sub = parser.add_subparsers(dest="command", required=True)

  def add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--sizes", default=",".join(str(n) for n in Q8_SIZES))
    p.add_argument("--materials", default=",".join(Q8_MATERIALS))
    p.add_argument("--threads", default=",".join(str(t) for t in DEFAULT_THREADS))
    p.add_argument("--budget-s", type=float, default=0.6)
    p.add_argument(
      "--quick",
      action="store_true",
      help="sizes<=16, threads 1,16 (fast iteration)",
    )
    p.add_argument("--baseline", default=str(RESULTS_DIR / "baseline_m3.json"))
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--out", default="")
    p.add_argument("--no-check", action="store_true")
    p.add_argument("--no-skims", action="store_true")

  for name in ("all", "warm", "cold"):
    p = sub.add_parser(name)
    add_common(p)
    if name in ("all", "cold"):
      p.add_argument(
        "--true-cold",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also run one cache-cleared (true JIT cold) case",
      )
  focused_fe2 = sub.add_parser(
    "fe2", help="matched damage-only FE2 warm, fresh-process, and JIT evidence"
  )
  focused_fe2.add_argument("--threads", default="1")
  focused_fe2.add_argument("--budget-s", type=float, default=0.6)
  focused_fe2.add_argument("--e2e-reps", type=int, default=1)
  focused_fe2.add_argument("--baseline", default=str(RESULTS_DIR / "baseline_m3.json"))
  focused_fe2.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
  focused_fe2.add_argument("--out", default="")
  focused_fe2.add_argument("--no-check", action="store_true")
  gates = sub.add_parser("gates")
  gates.add_argument("--sizes", default=",".join(str(n) for n in Q8_SIZES))
  gates.add_argument("--materials", default=",".join(Q8_MATERIALS))
  check = sub.add_parser("check")
  check.add_argument("--candidate", required=True)
  check.add_argument("--baseline", default=str(RESULTS_DIR / "baseline_m3.json"))
  check.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
  sub.add_parser("manifest")
  return parser


def _warn_cache_incoherence() -> None:
  """Warn on stderr about stale caller cache entries (NUMBA_CACHING.md §6).

  The same coherence data is recorded in every run manifest via
  ``collect_manifest``; this startup warning keeps incoherent runs visible.
  """
  for pair in check_cache_coherence():
    if not pair["coherent"]:
      print(
        f"warning: {pair['caller']} has cache entries older than "
        f"{pair['callee']}; timings may measure stale embedded callees "
        "(NUMBA_CACHING.md §6)",
        file=sys.stderr,
      )


def main() -> int:
  silence_legacy_logging()
  warnings.simplefilter("ignore", SparseEfficiencyWarning)
  args = build_parser().parse_args()
  _warn_cache_incoherence()
  if getattr(args, "quick", False):
    args.sizes = "2,4,8,16"
    args.threads = "1,16"
  handlers = {
    "all": cmd_all,
    "warm": cmd_warm,
    "fe2": cmd_fe2,
    "cold": cmd_cold,
    "gates": cmd_gates,
    "check": cmd_check,
    "manifest": cmd_manifest,
  }
  return handlers[args.command](args)


if __name__ == "__main__":
  raise SystemExit(main())
