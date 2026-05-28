"""One-off prange / parallel investigation for v3 Q8 stiffness (not collected by pytest)."""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

if sys.version_info < (3, 13):
  raise SystemExit("requires Python 3.13+")

from numba import njit, prange

from pyfem.io.InputReader import InputRead
from pyfem.util.dataStructures import elementData
from pyfem.v3 import load_problem
from pyfem.v3.fem.element import (
  _stiffness_from_coords_batched,
  quad8_plane_stress_stiffness,
)
from pyfem.v3.fem.kinematics import physical_gradients, strain_displacement
from pyfem.v3.fem.quadrature import gauss_tensor_product_2d
from pyfem.v3.fem.shapes import serendipity_quad8
from pyfem.v3.materials.plane_stress import plane_stress_matrix

ROOT = Path(__file__).resolve().parents[2]
SKIM_PRO = ROOT / "skims" / "patch_test8" / "skim.pro"

# ---------------------------------------------------------------------------
# Inline Numba variants (benchmark-only; not production)
# ---------------------------------------------------------------------------


@njit(cache=False, parallel=True)
def _pg_prange(nodal_coords, parent_gradients):
  n_elems = nodal_coords.shape[0]
  n_pts = parent_gradients.shape[0]
  n_nodes = nodal_coords.shape[1]
  jacobian = np.empty((n_elems, n_pts, 2, 2))
  grad_n = np.empty((n_elems, n_pts, n_nodes, 2))
  for e in prange(n_elems):
    xt = nodal_coords[e].T
    for p in range(n_pts):
      jac = xt @ parent_gradients[p]
      jacobian[e, p] = jac
      grad_n[e, p] = parent_gradients[p] @ np.linalg.inv(jac)
  return jacobian, grad_n


@njit(cache=False, parallel=True)
def _pg_range(nodal_coords, parent_gradients):
  n_elems = nodal_coords.shape[0]
  n_pts = parent_gradients.shape[0]
  n_nodes = nodal_coords.shape[1]
  jacobian = np.empty((n_elems, n_pts, 2, 2))
  grad_n = np.empty((n_elems, n_pts, n_nodes, 2))
  for e in range(n_elems):
    xt = nodal_coords[e].T
    for p in range(n_pts):
      jac = xt @ parent_gradients[p]
      jacobian[e, p] = jac
      grad_n[e, p] = parent_gradients[p] @ np.linalg.inv(jac)
  return jacobian, grad_n


@njit(cache=False, parallel=False)
def _pg_serial(nodal_coords, parent_gradients):
  n_elems = nodal_coords.shape[0]
  n_pts = parent_gradients.shape[0]
  n_nodes = nodal_coords.shape[1]
  jacobian = np.empty((n_elems, n_pts, 2, 2))
  grad_n = np.empty((n_elems, n_pts, n_nodes, 2))
  for e in range(n_elems):
    xt = nodal_coords[e].T
    for p in range(n_pts):
      jac = xt @ parent_gradients[p]
      jacobian[e, p] = jac
      grad_n[e, p] = parent_gradients[p] @ np.linalg.inv(jac)
  return jacobian, grad_n


def _make_stiffness_kernel(
  *,
  kin_parallel: bool,
  int_parallel: bool,
  kin_prange: bool,
  int_prange: bool,
  fused: bool,
):
  """Build a batched stiffness @njit with chosen loop styles."""

  if fused:
    par = kin_parallel or int_parallel
    use_prange = kin_prange or int_prange

    if par and use_prange:

      @njit(cache=False, parallel=True)
      def stiff(nodal_coords, constitutive):
        parent_pts, parent_w = gauss_tensor_product_2d(3)
        _, dN = serendipity_quad8(parent_pts)
        n_elems, n_gp = nodal_coords.shape[0], parent_w.shape[0]
        n_nodes = nodal_coords.shape[1]
        n_dof = 2 * n_nodes
        stiffness = np.zeros((n_elems, n_dof, n_dof))
        for e in prange(n_elems):
          xt = nodal_coords[e].T
          for p in range(n_gp):
            jac = xt @ dN[p]
            det_j = abs(np.linalg.det(jac))
            grad_n = dN[p] @ np.linalg.inv(jac)
            b = np.zeros((3, n_dof))
            b[0, 0::2] = grad_n[:, 0]
            b[1, 1::2] = grad_n[:, 1]
            b[2, 0::2] = grad_n[:, 1]
            b[2, 1::2] = grad_n[:, 0]
            weight = parent_w[p] * det_j
            stiffness[e] += weight * (b.T @ constitutive @ b)
        return stiffness

    else:

      @njit(cache=False, parallel=par)
      def stiff(nodal_coords, constitutive):
        parent_pts, parent_w = gauss_tensor_product_2d(3)
        _, dN = serendipity_quad8(parent_pts)
        n_elems, n_gp = nodal_coords.shape[0], parent_w.shape[0]
        n_nodes = nodal_coords.shape[1]
        n_dof = 2 * n_nodes
        stiffness = np.zeros((n_elems, n_dof, n_dof))
        for e in range(n_elems):
          xt = nodal_coords[e].T
          for p in range(n_gp):
            jac = xt @ dN[p]
            det_j = abs(np.linalg.det(jac))
            grad_n = dN[p] @ np.linalg.inv(jac)
            b = np.zeros((3, n_dof))
            b[0, 0::2] = grad_n[:, 0]
            b[1, 1::2] = grad_n[:, 1]
            b[2, 0::2] = grad_n[:, 1]
            b[2, 1::2] = grad_n[:, 0]
            weight = parent_w[p] * det_j
            stiffness[e] += weight * (b.T @ constitutive @ b)
        return stiffness

    return stiff

  kin_fn = _pg_prange if kin_prange else (_pg_range if kin_parallel else _pg_serial)
  if not kin_parallel and kin_prange:
    kin_fn = _pg_serial

  outer_par = int_parallel
  outer_prange = int_prange

  if outer_par and outer_prange:

    @njit(cache=False, parallel=True)
    def stiff(nodal_coords, constitutive):
      parent_pts, parent_w = gauss_tensor_product_2d(3)
      _, dN = serendipity_quad8(parent_pts)
      jacobian, grad_n = kin_fn(nodal_coords, dN)
      b = strain_displacement(grad_n)
      n_elems, n_gp = nodal_coords.shape[0], parent_w.shape[0]
      n_dof = b.shape[-1]
      stiffness = np.zeros((n_elems, n_dof, n_dof))
      for e in prange(n_elems):
        for p in range(n_gp):
          jac = jacobian[e, p]
          weight = parent_w[p] * abs(np.linalg.det(jac))
          stiffness[e] += weight * (b[e, p].T @ constitutive @ b[e, p])
      return stiffness

  else:

    @njit(cache=False, parallel=outer_par)
    def stiff(nodal_coords, constitutive):
      parent_pts, parent_w = gauss_tensor_product_2d(3)
      _, dN = serendipity_quad8(parent_pts)
      jacobian, grad_n = kin_fn(nodal_coords, dN)
      b = strain_displacement(grad_n)
      n_elems, n_gp = nodal_coords.shape[0], parent_w.shape[0]
      n_dof = b.shape[-1]
      stiffness = np.zeros((n_elems, n_dof, n_dof))
      for e in range(n_elems):
        for p in range(n_gp):
          jac = jacobian[e, p]
          weight = parent_w[p] * abs(np.linalg.det(jac))
          stiffness[e] += weight * (b[e, p].T @ constitutive @ b[e, p])
      return stiffness

  return stiff


VARIANTS = {
  "prod_prange_both": None,  # production
  "kin_range+int_prange": _make_stiffness_kernel(
    kin_parallel=True, int_parallel=True, kin_prange=False, int_prange=True, fused=False
  ),
  "kin_prange+int_range": _make_stiffness_kernel(
    kin_parallel=True, int_parallel=True, kin_prange=True, int_prange=False, fused=False
  ),
  "range_both_parallel": _make_stiffness_kernel(
    kin_parallel=True, int_parallel=True, kin_prange=False, int_prange=False, fused=False
  ),
  "serial_both": _make_stiffness_kernel(
    kin_parallel=False, int_parallel=False, kin_prange=False, int_prange=False, fused=False
  ),
  "fused_prange": _make_stiffness_kernel(
    kin_parallel=True, int_parallel=True, kin_prange=True, int_prange=True, fused=True
  ),
  "fused_range": _make_stiffness_kernel(
    kin_parallel=False, int_parallel=False, kin_prange=False, int_prange=False, fused=True
  ),
}


def _make_coords(n_elems: int, rng: np.random.Generator) -> np.ndarray:
  base = np.array(
    [
      [0.0, 0.0],
      [1.0, 0.0],
      [1.0, 1.0],
      [0.0, 1.0],
      [0.5, 0.0],
      [1.0, 0.5],
      [0.5, 1.0],
      [0.0, 0.5],
    ],
    dtype=np.float64,
  )
  out = np.empty((n_elems, 8, 2), dtype=np.float64)
  for e in range(n_elems):
    out[e] = base + 0.02 * rng.standard_normal((8, 2))
  return out


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
  return float(np.max(np.abs(a - b)))


def _timeit(fn, *args, warmup: int, repeats: int) -> float:
  for _ in range(warmup):
    fn(*args)
  t0 = time.perf_counter()
  for _ in range(repeats):
    fn(*args)
  return (time.perf_counter() - t0) / repeats * 1e3


@dataclass(frozen=True)
class TimedRow:
  label: str
  n_elems: int
  ms: float


def _bench_sizes(
  fn,
  c: np.ndarray,
  sizes: list[int],
  rng: np.random.Generator,
  *,
  warmup_scale: float = 1.0,
) -> list[TimedRow]:
  rows: list[TimedRow] = []
  for n in sizes:
    coords = _make_coords(n, rng)
    fn(coords[: min(n, 8)], c)
    warmup = max(1, int(3 * warmup_scale) if n <= 200 else 1)
    repeats = max(10, 200 // max(n, 1))
    ms = _timeit(fn, coords, c, warmup=warmup, repeats=repeats)
    rows.append(TimedRow("", n, ms))
  return rows


def _print_table(title: str, header: str, rows: dict[str, list[TimedRow]], baseline: str) -> None:
  sizes = rows[baseline]
  n_list = [r.n_elems for r in sizes]
  print(f"\n=== {title} ===")
  print(f"{'n_elems':>8}  ", end="")
  labels = list(rows.keys())
  for lab in labels:
    print(f"{lab:>22}", end="")
  print(f"  {'best vs base':>12}")
  for i, n in enumerate(n_list):
    base_ms = rows[baseline][i].ms
    print(f"{n:8d}  ", end="")
    best_ms = min(rows[lab][i].ms for lab in labels)
    for lab in labels:
      ms = rows[lab][i].ms
      print(f"{ms:22.3f}", end="")
    ratio = base_ms / best_ms if best_ms else float("inf")
    print(f"  {ratio:11.2f}x")


def _legacy_element_setup():
  props, globdat = InputRead(str(SKIM_PRO))
  element = next(iter(globdat.elements.iterElementGroup("ContElem")))
  n_dof = element.dofCount()
  state = np.zeros(n_dof)
  template = elementData(state, state)
  element.globdat = globdat
  return element, template, n_dof, globdat


def _legacy_stiffness_one(element, template, coords_2d: np.ndarray, globdat) -> np.ndarray:
  element.globdat = globdat
  template.coords = coords_2d
  template.stiff.fill(0.0)
  template.fint.fill(0.0)
  if hasattr(element, "mat"):
    element.mat.reset()
  element.getTangentStiffness(template)
  return template.stiff.copy()


def _legacy_stiffness_batch(element, template, coords: np.ndarray, globdat) -> list[np.ndarray]:
  out = []
  for e in range(coords.shape[0]):
    out.append(_legacy_stiffness_one(element, template, coords[e], globdat))
  return out


def main() -> None:
  rng = np.random.default_rng(0)
  c = plane_stress_matrix(1.0e6, 0.25)
  sizes = [1, 5, 10, 50, 200, 1000, 5000]

  loaded = load_problem("skims/patch_test8/problem.toml").problem
  patch_coords = loaded.coords[loaded.conn]

  print("=== Investigation questions ===")
  questions = [
    "Q1: Does prange on the element axis beat range? At what n_elems?",
    "Q2: Kinematics prange vs integration prange — which matters more?",
    "Q3: Nested prange (kin + int both parallel) vs single prange site?",
    "Q4: Staged pipeline (gradients → B → integrate) vs fused element loop?",
    "Q5: Numba cold-compile cost vs warm throughput?",
    "Q6: v3 batched Numba vs legacy v1 Python element loop?",
    "Q7: Single-element API ([None,...] wrapper) overhead?",
    "Q8: PatchTest8-scale mesh (n=5) — is parallel harmful?",
    "Q9: NUMBA_NUM_THREADS sensitivity?",
  ]
  for q in questions:
    print(f"  {q}")

  print("\n=== Correctness vs production (PatchTest8 + random) ===")
  prod_fn = _stiffness_from_coords_batched
  check_coords = [
    ("patch n=5", patch_coords),
    ("random n=32", _make_coords(32, rng)),
  ]
  for vname, vfn in VARIANTS.items():
    if vfn is None:
      continue
    for label, coords in check_coords:
      k_prod = prod_fn(coords, c)
      k_var = vfn(coords, c)
      print(f"  {vname:24s} {label:12s} max|ΔK| = {_max_abs_diff(k_prod, k_var):.3e}")

  # Q5 cold compile
  print("\n=== Q5: Compile / cache (production kernel, n=1) ===")
  coords1 = _make_coords(1, rng)
  t0 = time.perf_counter()
  _stiffness_from_coords_batched(coords1, c)
  cold_ms = (time.perf_counter() - t0) * 1e3
  warm_ms = _timeit(_stiffness_from_coords_batched, coords1, c, warmup=0, repeats=500)
  print(f"  first call (compile): {cold_ms:8.2f} ms")
  print(f"  warm mean (500x):     {warm_ms:8.4f} ms/call")

  # Q1-Q4 variant sweep
  print("\n=== Q1-Q4: Loop variant timings (ms/call, warm) ===")
  variant_rows: dict[str, list[TimedRow]] = {}
  variant_rows["prod_prange_both"] = _bench_sizes(
    _stiffness_from_coords_batched, c, sizes, rng
  )
  for vname, vfn in VARIANTS.items():
    if vfn is None:
      continue
    variant_rows[vname] = _bench_sizes(vfn, c, sizes, rng, warmup_scale=1.5)

  _print_table(
    "Stiffness variants",
    "ms",
    variant_rows,
    baseline="prod_prange_both",
  )

  # Relative to serial_both at n=200 and n=5000
  for n_ref in (200, 5000):
    idx = next(i for i, r in enumerate(variant_rows["prod_prange_both"]) if r.n_elems == n_ref)
    serial_ms = variant_rows["serial_both"][idx].ms
    prod_ms = variant_rows["prod_prange_both"][idx].ms
    print(f"\n  At n={n_ref}: serial={serial_ms:.3f} ms  prod={prod_ms:.3f} ms  "
          f"speedup={serial_ms / prod_ms:.2f}x")

  # Q2 kinematics-only micro-benchmark
  print("\n=== Q2 supplement: physical_gradients only (ms/call) ===")
  pts, _ = gauss_tensor_product_2d(3)
  _, dN = serendipity_quad8(pts)
  kin_labels = [
    ("prod pg prange", physical_gradients),
    ("pg range", _pg_range),
    ("pg serial", _pg_serial),
  ]
  print(f"{'n_elems':>8}  {'prod prange':>14}  {'range':>14}  {'serial':>14}  {'prange/range':>12}")
  for n in [5, 50, 200, 1000, 5000]:
    batch = _make_coords(n, rng)
    for _, fn in kin_labels:
      fn(batch[: min(n, 4)], dN)
    times = []
    for _, fn in kin_labels:
      repeats = max(10, 200 // n)
      times.append(_timeit(fn, batch, dN, warmup=2, repeats=repeats))
    print(
      f"{n:8d}  {times[0]:14.3f}  {times[1]:14.3f}  {times[2]:14.3f}  "
      f"{times[0] / times[1]:12.2f}x"
    )

  # Q6 legacy vs v3
  print("\n=== Q6: v3 batched vs legacy v1 element loop (ms for n elems) ===")
  element, template, n_dof, globdat = _legacy_element_setup()
  k_legacy = _legacy_stiffness_one(element, template, patch_coords[0], globdat)
  k_v3 = quad8_plane_stress_stiffness(patch_coords[0], c)
  print(f"  patch elem 0 max|ΔK| legacy vs v3: {_max_abs_diff(k_legacy, k_v3):.3e}")

  print(f"{'n_elems':>8}  {'legacy loop':>14}  {'v3 batch':>14}  {'v3/legacy':>12}")
  for n in [1, 5, 10, 50, 200]:
    coords = patch_coords if n == 5 else _make_coords(n, rng)

    def legacy_batch():
      _legacy_stiffness_batch(element, template, coords, globdat)

    v3_batch = lambda: _stiffness_from_coords_batched(coords, c)  # noqa: E731
    legacy_batch()
    v3_batch()
    reps = max(5, 100 // max(n, 1))
    leg_ms = _timeit(legacy_batch, warmup=2, repeats=reps)
    v3_ms = _timeit(v3_batch, warmup=2, repeats=reps)
    print(f"{n:8d}  {leg_ms:14.3f}  {v3_ms:14.3f}  {leg_ms / v3_ms:12.1f}x")

  # Q7 single-element API overhead
  print("\n=== Q7: Single-element API overhead (n=1 coords, ms/call) ===")
  c1 = _make_coords(1, rng)[0]
  c1_batch = c1[None, ...]

  def via_wrapper():
    quad8_plane_stress_stiffness(c1, c)

  def via_batched():
    _stiffness_from_coords_batched(c1_batch, c)[0]

  via_wrapper()
  via_batched()
  wrap_ms = _timeit(via_wrapper, warmup=5, repeats=500)
  bat_ms = _timeit(via_batched, warmup=5, repeats=500)
  print(f"  quad8_plane_stress_stiffness (2D): {wrap_ms:.4f} ms")
  print(f"  _stiffness_from_coords_batched[0]: {bat_ms:.4f} ms")
  print(f"  wrapper overhead: {wrap_ms - bat_ms:.4f} ms ({wrap_ms / bat_ms:.2f}x)")

  # Q8 PatchTest8 end-to-end-ish (n=5 batch)
  print("\n=== Q8: PatchTest8 batch (n=5) prod vs serial ===")
  idx5 = next(i for i, r in enumerate(variant_rows["prod_prange_both"]) if r.n_elems == 5)
  prod5 = variant_rows["prod_prange_both"][idx5].ms
  ser5 = variant_rows["serial_both"][idx5].ms
  print(f"  prod prange both: {prod5:.3f} ms")
  print(f"  serial both:      {ser5:.3f} ms")
  print(f"  ratio prod/serial: {prod5 / ser5:.2f}x (>1 means parallel slower)")

  # Q9 thread count
  print("\n=== Q9: NUMBA_NUM_THREADS (production, n=1000) ===")
  import numba

  coords1k = _make_coords(1000, rng)
  _stiffness_from_coords_batched(coords1k[:10], c)
  thread_counts = sorted({1, 2, 4, os.cpu_count() or 4})
  thread_counts = [t for t in thread_counts if t <= (os.cpu_count() or 4)]
  print(f"{'threads':>8}  {'ms/call':>12}")
  base_threads = numba.get_num_threads()
  for nt in thread_counts:
    numba.set_num_threads(nt)
    ms = _timeit(_stiffness_from_coords_batched, coords1k, c, warmup=2, repeats=20)
    print(f"{nt:8d}  {ms:12.3f}")
  numba.set_num_threads(base_threads)

  print("\n=== Recommendations (see script output + parent summary) ===")


if __name__ == "__main__":
  main()
