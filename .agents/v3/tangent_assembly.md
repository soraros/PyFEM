# Tangent assembly scaling notes

Pass B hardening for P3 `SolverState` + tangent / internal-force assembly. Pass A parity: `test/v3/test_tangent_assembly.py` vs legacy `assembleTangentStiffness`. **Do not extrapolate from 5-element skims alone.**

Hardening workflow: [hardening.md](hardening.md). Linear Q8 baseline: [scaling.md](scaling.md).

## Assembly path

| API | Location | Role |
|-----|----------|------|
| `assemble_tangent_coo` | `pyfem/v3/fem/assembly.py` | One `K_e` batch → COO scatter + optional `f_e = K_e @ u_e` |
| `assemble_tangent_loaded` | `pyfem/v3/assembly.py` | Chunked global tangent + `internal_force` (same chunking as linear) |
| `TangentAssemblyContext` | `pyfem/v3/solver/tangent_context.py` | Cache `K` (CSR); repeated `f_int = K @ u` via sparse matvec |
| `prepare_tangent_assembly` | same | One-shot context builder |

Fused assembly avoids a second element quadrature pass that the initial P3 path incurred (stiffness COO + separate `assemble_internal_force`).

`TangentAssemblyContext` is valid for **linear small-strain** tangents (constant `K`). Nonlinear materials must call `assemble_tangent_*` each update.

Newton with cached tangent (`solver/nonlinear.py`): `f_int = K @ u` each iteration; factorize `K_red = C.T @ K @ C` once per load step via `factorized_reduced_solve` when `K` is constant.

## Programmatic fixtures

Use `build_uniform_q8_loaded` from `pyfem/v3/mesh/refined_patch.py` for scale sweeps (same topology as P1). Minimum grid **2×2**.

## Run scale benchmark

```bash
uv sync --group v3
uv run python test/v3/_bench_tangent_assembly.py
```

| Column | Meaning |
|--------|---------|
| `linear_ms` | `assemble_loaded` (stiffness only) |
| `tangent_ms` | `assemble_tangent_loaded` at converged `u` (fused `K_e` + `f_int`) |
| `fint_ms` | first `TangentAssemblyContext.internal_force` after one-time `K` cache |
| `repeat_fint_ms` | amortized cached `K @ u` only |
| `rss_mb` | peak RSS after case (platform-dependent) |

### Sample results (macOS, warm Numba)

| mesh | elems | dofs | linear_ms | tangent_ms | fint_ms | repeat_fint_ms | rss_mb |
|------|-------|------|-----------|------------|---------|----------------|--------|
| 2×2 | 4 | 42 | 0.36 | 0.33 | 0.001 | 0.001 | 134 |
| 4×4 | 16 | 130 | 0.37 | 0.36 | 0.002 | 0.003 | 135 |
| 8×8 | 64 | 450 | 0.51 | 0.50 | 0.005 | 0.005 | 136 |
| 16×16 | 256 | 1666 | 0.95 | 0.95 | 0.017 | 0.016 | 141 |
| 32×32 | 1024 | 6402 | 2.68 | 2.42 | 0.060 | 0.062 | 170 |
| 64×64 | 4096 | 25090 | 8.48 | 8.85 | 0.243 | 0.253 | 336 |

Loaded skim (5 elems): tangent assemble ~0.31 ms/call; cached `K @ u` ~0.001 ms/call.

At 16×16 and below, `tangent_ms` ≈ `linear_ms` (single `K_e` pass). At 64×64, tangent is within noise of linear; internal-force-only updates are ~35× cheaper than full tangent assembly.

## Tests

| Test | Role |
|------|------|
| `test/v3/test_tangent_assembly.py` | Pass A: legacy parity; fused vs separate `f_int` path |
| `test/v3/test_tangent_scale.py` | 2×2 / 8×8 programmatic assemble; context `f_int` vs full tangent |

## Accuracy

Book skim tolerances unchanged. Uniform patches are not legacy-parity-checked at large scale.

## Out of scope

- Newton loop / load ramp (next P3 pass A item)
- Nonlinear `K(u)` reuse (needs per-step full tangent assembly)
- 3D Hex8 tangent benches (optional; use `patch_test8_3d` skim for parity)
