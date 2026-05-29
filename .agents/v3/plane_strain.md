# Plane-strain scaling notes

Pass B hardening for P2 plane strain. Pass A parity: `patch_test8_plane_strain` skim vs legacy. **Do not extrapolate from 5-element skims alone.**

Hardening workflow: [hardening.md](hardening.md). P1 Q8 scale baseline: [scaling.md](scaling.md).

## Material path

Plane strain differs from plane stress only in the packed 3×3 `constitutive` matrix (`plane_strain_matrix` in `pyfem/v3/materials/plane_strain.py`). Element kernels, assembly chunking, and solver paths are unchanged from P1.

`pack_problem(..., material_type="PlaneStrain")` selects the D matrix; I/O loaders pass through from `.pro` / `problem.toml`.

## Programmatic fixtures

| API | Location | Role |
|-----|----------|------|
| `build_uniform_q8_patch` | `pyfem/v3/mesh/refined_patch.py` | Mesh + PatchTest8 BCs |
| `build_uniform_q8_loaded` | same | `LoadedProblem` with `material_type` kwarg |

Use generators for **scale sweeps**, not for replacing book parity skims.

## Run scale benchmark

```bash
uv sync --group v3
uv run python test/v3/_bench_plane_strain_scale.py
```

Columns match [scaling.md](scaling.md): `asm_ms` (chunked when `n_elems > 2048`), amortized `LinearSolutionContext.solve`, repeat back-solve, peak RSS.

Minimum grid **2×2** (1×1 has no interior DOFs).

### Sample results (macOS, warm Numba)

| mesh | elems | dofs | asm_ms | solve_ms | repeat_ms | rss_mb |
|------|-------|------|--------|----------|-----------|--------|
| 2×2 | 4 | 42 | 0.33 | 0.01 | 0.01 | 134 |
| 4×4 | 16 | 130 | 0.38 | 0.01 | 0.01 | 135 |
| 8×8 | 64 | 450 | 0.49 | 0.03 | 0.03 | 136 |
| 16×16 | 256 | 1666 | 0.98 | 0.10 | 0.10 | 140 |
| 32×32 | 1024 | 6402 | 2.81 | 0.51 | 0.50 | 169 |
| 64×64 | 4096 | 25090 | 8.74 | 4.18 | 4.21 | 335 |

16×16 assembly is material-agnostic (same mesh, different C only in `constitutive` pointer): PlaneStress ~0.83 ms, PlaneStrain ~0.81 ms/call on the reference run.

Skim spot-check: `patch_test8_plane_strain` solve ok, `||u|| ≈ 9.53×10⁻⁴`.

## Tests

| Test | Role |
|------|------|
| `test/v3/test_parity_patch_test8_plane_strain.py` | Pass A: v3 vs legacy on book mesh |
| `test/v3/test_plane_strain.py` | D matrix vs closed form |
| `test/v3/test_plane_strain_scale.py` | 2×2 and 8×8 programmatic assemble + solve |
| `test/v3/test_refined_patch.py` | Plane-strain stiffness diagonal, factorized 2×2 |

## Accuracy

Parity skim tolerances unchanged (`rtol=1e-10`, `atol=1e-12`). Uniform patch grids use the same prescribed field as PatchTest8 but are **not** parity-checked against legacy at large scale.

## Out of scope here

- Separate fused kernel for plane strain (not needed — same Bᵀ C B path)
- 3D continuum (next P2 item)
- Quad4/Tria3 plane-strain scale benches (reuse P1 element path when needed)
