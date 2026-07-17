"""One-off benchmark: Numba Q8 stiffness vs pre-commit NumPy (not collected by pytest)."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import numpy as np

if sys.version_info < (3, 13):
  raise SystemExit("requires Python 3.13+")

from pyfem.v3 import load_problem
from pyfem.v3.fem.element import (
  _stiffness_from_coords_batched,
  quad8_plane_stress_stiffness,
)
from pyfem.v3.fem.kinematics import physical_gradients
from pyfem.v3.fem.quadrature import gauss_tensor_product_2d
from pyfem.v3.fem.shapes import serendipity_quad8
from pyfem.v3.materials.plane_stress import plane_stress_matrix

# --- pre-commit reference (pure NumPy, no Numba) ---


def _physical_gradients_numpy(
  nodal_coords: np.ndarray,
  parent_gradients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  single = nodal_coords.ndim == 2
  if single:
    nodal_coords = nodal_coords[None, ...]
  jacobian = np.einsum("...na,pnb->...pab", nodal_coords, parent_gradients)
  jacobian_inv = np.linalg.inv(jacobian)
  grad_n = np.einsum("...pnb,...pbc->...pnc", parent_gradients, jacobian_inv)
  return jacobian, grad_n


def _strain_displacement_numpy(grad_n: np.ndarray) -> np.ndarray:
  *lead, n_nodes, _ = grad_n.shape
  n_dof = 2 * n_nodes
  b = np.zeros((*lead, 3, n_dof), dtype=np.float64)
  b[..., 0, 0::2] = grad_n[..., :, 0]
  b[..., 1, 1::2] = grad_n[..., :, 1]
  b[..., 2, 0::2] = grad_n[..., :, 1]
  b[..., 2, 1::2] = grad_n[..., :, 0]
  return b


def _stiffness_numpy(nodal_coords: np.ndarray, constitutive: np.ndarray) -> np.ndarray:
  single = nodal_coords.ndim == 2
  if single:
    nodal_coords = nodal_coords[None, ...]

  parent_pts, parent_w = gauss_tensor_product_2d(3)
  _, dN_parent = serendipity_quad8(parent_pts)
  _jacobian, grad_n = _physical_gradients_numpy(nodal_coords, dN_parent)
  det_j = np.linalg.det(_jacobian)
  b = _strain_displacement_numpy(grad_n)

  measure = parent_w * np.abs(det_j)
  bt_c_b = np.einsum("...paj,ac,...pck->...pjk", b, constitutive, b)
  stiffness = np.einsum("...p,...pjk->...jk", measure, bt_c_b)
  return stiffness[0] if single else stiffness


def _make_coords(n_elems: int, rng: np.random.Generator) -> np.ndarray:
  """Perturbed copies of a regular Q8 quad (keeps Jacobians well conditioned)."""
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


@dataclass(frozen=True)
class BenchRow:
  label: str
  numpy_ms: float
  numba_ms: float
  max_abs_diff: float


def _timeit(fn, *args, warmup: int, repeats: int) -> float:
  for _ in range(warmup):
    fn(*args)
  t0 = time.perf_counter()
  for _ in range(repeats):
    fn(*args)
  return (time.perf_counter() - t0) / repeats * 1e3


def main() -> None:
  rng = np.random.default_rng(0)
  c = plane_stress_matrix(1.0e6, 0.25)

  loaded = load_problem("skims/patch_test8/problem.toml").problem
  patch_coords = loaded.coords[loaded.conn]

  print("=== Correctness ===")
  for label, coords in [
    ("single random quad", _make_coords(1, rng)[0]),
    ("batch=10", _make_coords(10, rng)),
    ("patch_test8 (5 elems)", patch_coords),
  ]:
    k_np = _stiffness_numpy(coords, c)
    k_nb = quad8_plane_stress_stiffness(coords, c)
    diff = _max_abs_diff(k_np, k_nb)
    print(f"  {label:24s} max|ΔK| = {diff:.3e}")

  # kinematics-only check (isolates loop rewrite)
  pts, _ = gauss_tensor_product_2d(3)
  _, dN = serendipity_quad8(pts)
  batch = _make_coords(32, rng)
  gn_np, _ = _physical_gradients_numpy(batch, dN)
  gn_nb, det_nb = physical_gradients(batch, dN)
  jac_np, _ = _physical_gradients_numpy(batch, dN)
  print(f"  physical_gradients J     max|Δ| = {_max_abs_diff(jac_np[:, :, 0, 0] * jac_np[:, :, 1, 1] - jac_np[:, :, 0, 1] * jac_np[:, :, 1, 0], det_nb):.3e}")
  print(f"  physical_gradients gradN max|Δ| = {_max_abs_diff(gn_np, gn_nb):.3e}")

  print("\n=== Numba compile / cache ===")
  coords1 = _make_coords(1, rng)
  t0 = time.perf_counter()
  _stiffness_from_coords_batched(coords1, c)
  cold_ms = (time.perf_counter() - t0) * 1e3
  t0 = time.perf_counter()
  _stiffness_from_coords_batched(coords1, c)
  warm_ms = (time.perf_counter() - t0) * 1e3
  print(f"  first call (compile): {cold_ms:8.2f} ms")
  print(f"  second call (cached): {warm_ms:8.4f} ms")

  print("\n=== Micro-benchmark (warm Numba, ms per call) ===")
  rows: list[BenchRow] = []
  for n_elems in [1, 5, 10, 50, 200, 1000, 5000]:
    coords = _make_coords(n_elems, rng)
    # prime numba
    quad8_plane_stress_stiffness(coords[: min(n_elems, 5)], c)
    warmup = 3 if n_elems <= 200 else 1
    repeats = max(20, 200 // max(n_elems, 1))
    np_ms = _timeit(_stiffness_numpy, coords, c, warmup=2, repeats=repeats)
    nb_ms = _timeit(
      quad8_plane_stress_stiffness, coords, c, warmup=warmup, repeats=repeats
    )
    diff = _max_abs_diff(
      _stiffness_numpy(coords, c), quad8_plane_stress_stiffness(coords, c)
    )
    rows.append(BenchRow(str(n_elems), np_ms, nb_ms, diff))
    ratio = np_ms / nb_ms if nb_ms else float("inf")
    winner = "Numba" if ratio > 1 else "NumPy"
    print(
      f"  n_elems={n_elems:5d}  NumPy={np_ms:8.3f}  Numba={nb_ms:8.3f}  "
      f"ratio={ratio:6.2f}x  -> {winner}  max|ΔK|={diff:.2e}"
    )

  print("\n=== End-to-end assembly (patch_test8, 5 elems) ===")
  from pyfem.v3._prototype_assembly import assemble_loaded

  loaded_full = load_problem("skims/patch_test8/problem.toml")
  for _ in range(3):
    assemble_loaded(loaded_full)
  t0 = time.perf_counter()
  for _ in range(200):
    assemble_loaded(loaded_full)
  asm_ms = (time.perf_counter() - t0) / 200 * 1e3
  print(
    f"  assemble_loaded: {asm_ms:.3f} ms/call (dominated by tiny mesh + Python COO scatter)"
  )

  print("\n=== Summary ===")
  cross = next(r for r in rows if r.label == "200")
  print(
    f"  At n_elems=200: NumPy {cross.numpy_ms:.3f} ms vs Numba {cross.numba_ms:.3f} ms"
  )
  if cross.numba_ms < cross.numpy_ms:
    print("  Numba wins at moderate batch sizes once compiled.")
  else:
    print("  NumPy still competitive or faster at tested sizes — check loop structure.")


if __name__ == "__main__":
  main()
