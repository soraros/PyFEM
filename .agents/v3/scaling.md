# Prototype benchmark and scaling evidence

> **Historical prototype measurements.** This file consolidates the former linear,
> plane-strain, tangent-assembly, and structural benchmark notes. Reuse the
> fixtures, numerical observations, and failure cases—not the thresholds or
> prototype architecture claims. Future performance work follows
> [design.md](design.md#12-performance-proof-policy).

These P1-P4 measurements describe the prototype's performance and memory behavior.
Do not extrapolate from the small parity skims alone. Paths and APIs are recorded
as measured at the historical prototype base; they may have moved and are not
current interface authority.

The original notes did not completely capture hardware identity, dependency
versions, or CPU-thread settings. Treat every timing as directional evidence, not
as a reproducible performance gate.

## Historical scale-first policy

The prototype pass targeted **mid-to-large meshes** (hundreds to thousands of
elements; 10³ to 10⁴+ DOFs), where batched Numba, chunked COO, and factorization
reuse mattered.

- Small book skims, such as the five-element cases, served parity correctness,
  not performance claims.
- The pass treated unmeasured serial/parallel thresholds and tiny-mesh dual paths
  as unsupported unless a large-scale regression demonstrated their need.
- Assembly and solver comparisons started at 16×16; the tables below preserve the
  recorded observations.

### Historical Numba policy

- Each top-level stiffness call used one `prange` site; Q8 used a fused element
  loop for Jacobian, strain-displacement matrix, and quadrature.
- The compiled kernels avoided `einsum` and `tensordot`; COO scatter stayed
  serial.
- Kernels used `cache=True`; Gauss order used compile-time `@overload`.

### Observed qualitative comparison

- **Assembly:** the prototype's batched Numba path was orders of magnitude faster
  than the legacy Python element loops at scale; see
  `_bench_prange_investigation.py` Q6.
- **Solve:** both paths used SciPy direct methods; measured gains came from
  `factorized` reuse through `LinearSolutionContext` and the cached Newton path.
- **Newton (P3):** the prototype cached the tangent, evaluated
  `f_int = K @ u`, and factorized `K_red` once per load step when `K` was
  constant.

## Run scale benchmarks

```bash
uv sync --group v3
uv run python test/v3/_bench_solve_scale.py
uv run python test/v3/_bench_plane_strain_scale.py
uv run python test/v3/_bench_tangent_assembly.py
uv run python test/v3/_bench_structural_scale.py
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

This rule preserves an observed failure case from the original scale pass: the
first fixture retained center points as orphaned nodes and produced a singular
global matrix even though the small parity skims passed. The corrected generator
excludes those points.

Outer-boundary nodes receive the PatchTest8 prescribed displacement field. Grid sweeps measure assembly/solve scaling; the loaded skim row validates `LinearSolutionContext`.

## Known bottlenecks (large mesh)

1. **Element formation** — fused Q8 kernel (`prange` over elements); Quad4/Tria3 still use staged kinematics.
2. **COO buffer** — `256 × n_elems` triplets for Q8 before `tocsr()` deduplication; chunked assembly caps Ke batch size.
3. **Direct solve** — SciPy `spsolve` / `factorized` on reduced system; dominates as DOFs grow (~O(n^1.5–2)).

## Prototype paths (scale-related)

| Component | Location | Notes |
|-----------|----------|-------|
| Uniform Q8 patch mesh | `pyfem/v3/mesh/refined_patch.py` | Benchmark / programmatic problems |
| Loaded Q8 patch problem | `build_uniform_q8_loaded(..., material_type=...)` | Scale sweeps for PlaneStress / PlaneStrain |
| Chunked assembly | `pyfem/v3/assembly.py` | Auto when `n_elems > 2048`, `chunk_size=4096` |
| Factorized reuse | `pyfem/v3/solver/context.py` | `factorized_reduced_solve`; `prepare_linear_solve` → `LinearSolutionContext` |
| Tangent + `f_int` | `pyfem/v3/assembly.py`, `solver/tangent_context.py`, `solver/nonlinear.py` | Fused `K_e`; cached `K @ u`; factorized `K_red` per NR load step — see [tangent-assembly evidence](#tangent-assembly-evidence) |
| Parity path | `solve_linear` | Unchanged; always full assemble + solve |

## Linear Q8 accuracy

Parity skims (`rtol=1e-10`, `atol=1e-12`) remain on book-scale meshes. Uniform patch grids use the same linear prescribed field as `PatchTest8.py` but are **not** parity-checked against legacy at large scale.

## Linear Q8 baseline exclusions

- Iterative / AMG solvers
- 3D Hex8 scale sweeps (optional P2 pass B; parity via `patch_test8_3d` skim)
- Gmsh-in-v3 (P8)
- Quad4/Tria3 fused kernels

## Plane-strain evidence

The historical P2 pass compared the `patch_test8_plane_strain` skim with legacy
behavior. Do not extrapolate from the five-element skim alone.

### Plane-strain material path

Plane strain differs from plane stress only in the packed 3×3 `constitutive`
matrix (`plane_strain_matrix` in `pyfem/v3/materials/plane_strain.py`). Element
kernels, assembly chunking, and solver paths are unchanged from the linear Q8
baseline.

`pack_problem(..., material_type="PlaneStrain")` selects the D matrix; I/O loaders
pass through from `.pro` / `problem.toml`.

### Plane-strain fixtures

| API | Location | Role |
|---|---|---|
| `build_uniform_q8_patch` | `pyfem/v3/mesh/refined_patch.py` | Mesh + PatchTest8 BCs |
| `build_uniform_q8_loaded` | same | `LoadedProblem` with `material_type` kwarg |

Use generators for scale sweeps, not for replacing book parity skims.

### Plane-strain benchmark

```bash
uv sync --group v3
uv run python test/v3/_bench_plane_strain_scale.py
```

Columns match the linear baseline: `asm_ms` (chunked when `n_elems > 2048`),
amortized `LinearSolutionContext.solve`, repeat back-solve, and peak RSS. Minimum
grid size is 2×2 because 1×1 has no interior DOFs.

| mesh | elems | dofs | asm_ms | solve_ms | repeat_ms | rss_mb |
|---|---:|---:|---:|---:|---:|---:|
| 2×2 | 4 | 42 | 0.33 | 0.01 | 0.01 | 134 |
| 4×4 | 16 | 130 | 0.38 | 0.01 | 0.01 | 135 |
| 8×8 | 64 | 450 | 0.49 | 0.03 | 0.03 | 136 |
| 16×16 | 256 | 1666 | 0.98 | 0.10 | 0.10 | 140 |
| 32×32 | 1024 | 6402 | 2.81 | 0.51 | 0.50 | 169 |
| 64×64 | 4096 | 25090 | 8.74 | 4.18 | 4.21 | 335 |

At 16×16, the reference run measured plane-stress and plane-strain assembly at
approximately 0.83 ms and 0.81 ms per call. The plane-strain skim spot-check gave
`||u|| ≈ 9.53×10⁻⁴`.

| Test | Historical role |
|---|---|
| `test/v3/test_parity_linear.py` | v3 versus legacy on the book mesh |
| `test/v3/test_plane_strain.py` | D matrix versus closed form |
| `test/v3/test_refined_patch.py` | 2×2/8×8 solve, stiffness diagonal, and factorized context |

Parity skim tolerances were `rtol=1e-10`, `atol=1e-12`. Uniform patch grids used
the same prescribed field as PatchTest8 but were not parity-checked against legacy
at large scale. Separate plane-strain fused kernels, 3D continuum, and Q4/T3
plane-strain scale benches were outside this pass.

## Tangent-assembly evidence

The historical P3 `SolverState` pass compared
`test/v3/test_tangent_assembly.py` with legacy `assembleTangentStiffness`. Do not
extrapolate from the five-element skims alone.

### Tangent path

| API | Location | Role |
|---|---|---|
| `assemble_tangent_coo` | `pyfem/v3/fem/assembly.py` | One `K_e` batch to COO scatter plus optional `f_e = K_e @ u_e` |
| `assemble_tangent_loaded` | `pyfem/v3/assembly.py` | Chunked global tangent plus `internal_force` |
| `TangentAssemblyContext` | `pyfem/v3/solver/tangent_context.py` | Cache CSR `K`; repeated `f_int = K @ u` |
| `prepare_tangent_assembly` | same | One-shot context builder |

Fused assembly avoids the second element-quadrature pass previously performed by
`assemble_internal_force`. `TangentAssemblyContext` is valid for linear
small-strain tangents with constant `K`; nonlinear materials must call
`assemble_tangent_*` on each update. The prototype Newton path used
`TangentAssemblyContext.internal_force`, cached `f_int = K @ u`, and factorized
`K_red = C.T @ K @ C` once per load step when `K` was constant.

The fixture was `build_uniform_q8_loaded` from
`pyfem/v3/mesh/refined_patch.py`, with a minimum 2×2 grid.

```bash
uv sync --group v3
uv run python test/v3/_bench_tangent_assembly.py
```

| Column | Meaning |
|---|---|
| `linear_ms` | `assemble_loaded` stiffness only |
| `tangent_ms` | `assemble_tangent_loaded` at converged `u` |
| `fint_ms` | first cached-context internal-force evaluation |
| `repeat_fint_ms` | amortized cached `K @ u` only |
| `rss_mb` | peak RSS after the case |

| mesh | elems | dofs | linear_ms | tangent_ms | fint_ms | repeat_fint_ms | rss_mb |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2×2 | 4 | 42 | 0.36 | 0.33 | 0.001 | 0.001 | 134 |
| 4×4 | 16 | 130 | 0.37 | 0.36 | 0.002 | 0.003 | 135 |
| 8×8 | 64 | 450 | 0.51 | 0.50 | 0.005 | 0.005 | 136 |
| 16×16 | 256 | 1666 | 0.95 | 0.95 | 0.017 | 0.016 | 141 |
| 32×32 | 1024 | 6402 | 2.68 | 2.42 | 0.060 | 0.062 | 170 |
| 64×64 | 4096 | 25090 | 8.48 | 8.85 | 0.243 | 0.253 | 336 |

The loaded five-element skim measured about 0.31 ms per tangent assembly and
0.001 ms per cached `K @ u`. At 64×64, internal-force-only updates were roughly
35× cheaper than full tangent assembly.

`test/v3/test_tangent_assembly.py` checked legacy parity plus fused/separate
internal force, and `test/v3/test_refined_patch.py` checked cached versus full
tangent assembly on 2×2/8×8 grids. Book tolerances were unchanged; large uniform
patches were not legacy-parity-checked. Nonlinear `K(u)` reuse, 3D Hex8 tangent
benches, and the Newton/load-ramp feature were outside this benchmark pass.

## Structural evidence

The historical P4 pass covered Truss/Spring plus Riks arc length against the
three-element `shallow_truss_riks` skim. Do not extrapolate from that book mesh.

### Structural path and fixtures

| API | Location | Role |
|---|---|---|
| `link2_tangent_single` / `link2_tangent_batched` | `pyfem/v3/fem/link2.py` | Corotational TL truss and axial spring in one two-node kernel |
| `local_to_global_4`, `rotation_matrix_2d`, `to_element_vector_4` | `pyfem/v3/fem/link2.py` | Element rotation matrix and local/global transforms (colocated after M20) |
| `assemble_tangent_loaded` | `pyfem/v3/assembly.py` | Multi-group tangent plus `f_int` |
| `solve_riks` | `pyfem/v3/solver/riks.py` | Riks arc length with `solve_reduced_displacement` |

Truss and Spring shared one batched kernel; prototype assembly dispatched by
`group_kind` with truss group 0 and spring group 1.
`build_truss_fan(n_rays)` in `pyfem/v3/mesh/truss_fan.py` produced
`n_rays` trusses plus one apex spring; `n_rays=2` reproduced the chapter-4
`ShallowtrussRiks` topology. `build_truss_fan_loaded(n_rays)` added
`RiksSolverSettings(fixed_step=True)`.

```bash
uv sync --group v3
uv run python test/v3/_bench_structural_scale.py
```

| Column | Meaning |
|---|---|
| `tangent_ms` | zero-state `assemble_tangent_loaded` |
| `riks_ms` | full `solve_riks` with `max_lam=2.0` |
| `rss_mb` | peak RSS after the case |

| mesh | elems | dofs | tangent_ms | riks_ms | rss_mb |
|---|---:|---:|---:|---:|---:|
| skim | 3 | 8 | — | 71.2 | — |
| fan_8 | 9 | 20 | 0.88 | 7.05 | 248 |
| fan_32 | 33 | 68 | 0.91 | 7.53 | 248 |
| fan_128 | 129 | 260 | 0.89 | 5.52 | 248 |
| fan_512 | 513 | 1028 | 0.96 | 4.95 | 249 |

The skim included one-time Numba compilation; fan rows amortized compilation.
Tangent assembly stayed near 1 ms through hundreds of link elements, while Riks
iterations and sparse factorization dominated. `test_link2_element.py` checked
element stiffness, and `test_structural_mesh.py` checked topology, assembly, and
the `n_rays=2` Riks correspondence.
