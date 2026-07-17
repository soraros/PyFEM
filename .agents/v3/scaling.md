# v3 scaling notes

> **Historical prototype measurements.** Reuse the fixtures and evidence, not the
> thresholds or architecture claims. New performance proof follows
> [design.md](design.md#12-performance-proof-policy).

How to interpret performance and memory for the P1 linear 2D stack. **Do not extrapolate from 5-element parity skims alone.**

Hardening workflow for other phases: [hardening.md](hardening.md).

## Scale-first performance

Optimize for **mid-to-large meshes** (hundreds–thousands of elements; 10³–10⁴+ DOFs). That is where batched Numba, chunked COO, and factorization reuse matter.

- **Small book skims** (e.g. 5 elements) are for parity correctness only—not perf targets.
- **Do not add cheeky fallbacks** (serial-vs-parallel thresholds, dual paths for tiny `n_elems`) unless a large-scale regression is proven. Prefer one clear hot path.
- **Bench at 16×16 and up** when judging assembly or solver changes; see tables below.

### Numba discipline (production kernels)

- One `prange` site per top-level stiffness call; Q8 uses a fused element loop (J, B, quadrature).
- No `einsum` / `tensordot` in `@njit`; COO scatter stays serial.
- `cache=True`; Gauss order via compile-time `@overload`.

### v1 vs v3 (qualitative)

- **Assembly:** v3 batched Numba is orders of magnitude faster than legacy Python element loops at scale (see `_bench_prange_investigation.py` Q6).
- **Solve:** both use SciPy direct methods; wins come from `factorized` reuse (`LinearSolutionContext`, Newton with cached `K`).
- **Newton (P3):** cached tangent → `f_int = K @ u`; factorize `K_red` once per load step when `K` is constant.

## Run scale benchmarks

```bash
uv sync --group v3
uv run python test/v3/_bench_solve_scale.py
uv run python test/v3/_bench_plane_strain_scale.py
uv run python test/v3/_bench_tangent_assembly.py
uv run python test/v3/_bench_numba_stiffness.py
uv run python test/v3/_bench_prange_investigation.py
```

`_bench_solve_scale.py` sweeps uniform Q8 patch grids (`build_uniform_q8_patch`) with staged timings:

| Column | Meaning |
|--------|---------|
| `asm_ms` | `assemble_loaded` (chunked when `n_elems > 2048`) |
| `solve_ms` | amortized `LinearSolutionContext.solve` after one-time factorization |
| `repeat_ms` | same context, repeated back-solve only |
| `rss_mb` | peak RSS after case (platform-dependent) |

Minimum grid size is **2×2** (1×1 has no interior DOFs). Legacy comparison is limited to the 5-element `patch_test8` skim reference row. The `patch_test8_loaded` row validates factorization reuse on a loaded skim.

### Sample results (macOS, warm Numba)

| mesh | elems | dofs | asm_ms | solve_ms | repeat_ms | rss_mb |
|------|-------|------|--------|----------|-----------|--------|
| 2×2 | 4 | 42 | 0.37 | 0.01 | 0.01 | 135 |
| 4×4 | 16 | 130 | 0.37 | 0.01 | 0.01 | 135 |
| 8×8 | 64 | 450 | 0.47 | 0.03 | 0.03 | 136 |
| 16×16 | 256 | 1666 | 0.92 | 0.09 | 0.09 | 140 |
| 32×32 | 1024 | 6402 | 2.15 | 0.51 | 0.48 | 170 |
| 64×64 | 4096 | 25090 | 8.08 | 4.04 | 4.02 | 336 |

Legacy 5-elem skim solve-only: ~1.6 ms/call. Loaded skim repeat solve: ~0.01 ms/call.

## Mesh topology (uniform Q8 patch)

Serendipity Q8 elements have no bubble node. Grid points at both-odd indices `(ix % 2 == 1 and iy % 2 == 1)` are omitted:

```
n_nodes = (2 * nx + 1) * (2 * ny + 1) - nx * ny
```

Outer-boundary nodes receive the PatchTest8 prescribed displacement field. Grid sweeps measure assembly/solve scaling; the loaded skim row validates `LinearSolutionContext`.

## Known bottlenecks (large mesh)

1. **Element formation** — fused Q8 kernel (`prange` over elements); Quad4/Tria3 still use staged kinematics.
2. **COO buffer** — `256 × n_elems` triplets for Q8 before `tocsr()` deduplication; chunked assembly caps Ke batch size.
3. **Direct solve** — SciPy `spsolve` / `factorized` on reduced system; dominates as DOFs grow (~O(n^1.5–2)).

## Architecture (scale-related)

| Component | Location | Notes |
|-----------|----------|-------|
| Uniform Q8 patch mesh | `pyfem/v3/mesh/refined_patch.py` | Benchmark / programmatic problems |
| Loaded Q8 patch problem | `build_uniform_q8_loaded(..., material_type=...)` | Scale sweeps for PlaneStress / PlaneStrain |
| Chunked assembly | `pyfem/v3/assembly.py` | Auto when `n_elems > 2048`, `chunk_size=4096` |
| Factorized reuse | `pyfem/v3/solver/context.py` | `factorized_reduced_solve`; `prepare_linear_solve` → `LinearSolutionContext` |
| Tangent + `f_int` | `pyfem/v3/assembly.py`, `solver/tangent_context.py`, `solver/nonlinear.py` | Fused `K_e`; cached `K @ u`; factorized `K_red` per NR load step — [tangent_assembly.md](tangent_assembly.md) |
| Parity path | `solve_linear` | Unchanged; always full assemble + solve |

## Accuracy

Parity skims (`rtol=1e-10`, `atol=1e-12`) remain on book-scale meshes. Uniform patch grids use the same linear prescribed field as `PatchTest8.py` but are **not** parity-checked against legacy at large scale.

## Out of scope here

- Iterative / AMG solvers
- 3D Hex8 scale sweeps (optional P2 pass B; parity via `patch_test8_3d` skim)
- Gmsh-in-v3 (P8)
- Quad4/Tria3 fused kernels
