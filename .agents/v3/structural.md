# Structural scaling notes

Pass B hardening for P4 Truss/Spring + Riks arc-length. Pass A parity: `shallow_truss_riks` skim vs legacy. **Do not extrapolate from the 3-element book mesh alone.**

Hardening workflow: [hardening.md](hardening.md). Continuum tangent baseline: [tangent_assembly.md](tangent_assembly.md).

## Element kernel

| API | Location | Role |
|-----|----------|------|
| `link2_tangent_single` / `link2_tangent_batched` | `pyfem/v3/fem/link2.py` | Unified 2-node Truss + Spring (corotational TL truss, axial spring) |
| `local_to_global_4` | `pyfem/v3/fem/transforms.py` | Single rotation matrix per element |
| `assemble_tangent_loaded` | `pyfem/v3/assembly.py` | Multi-group COO tangent + `f_int` (Truss group 0, Spring group 1) |
| `solve_riks` | `pyfem/v3/solver/riks.py` | Riks arc-length with `solve_reduced_displacement` |

Truss and Spring share one batched kernel; assembly dispatches by `group_kind` without separate module paths.

## Programmatic fixtures

`build_truss_fan(n_rays)` in `pyfem/v3/mesh/truss_fan.py` — shallow-truss fan: `n_rays` truss members + one apex spring. `n_rays=2` reproduces ch.4 `ShallowtrussRiks` topology (validated in `test_structural_mesh.py`).

`build_truss_fan_loaded(n_rays)` returns a `LoadedProblem` with `RiksSolverSettings(fixed_step=True)`.

## Run scale benchmark

```bash
uv sync --group v3
uv run python test/v3/_bench_structural_scale.py
```

| Column | Meaning |
|--------|---------|
| `tangent_ms` | `assemble_tangent_loaded` at zero state (one batched pass per group) |
| `riks_ms` | full `solve_riks` with `max_lam=2.0` (includes repeated tangent + factorize) |
| `rss_mb` | peak RSS after case (platform-dependent) |

### Sample results (macOS, warm Numba)

| mesh | elems | dofs | tangent_ms | riks_ms | rss_mb |
|------|-------|------|------------|---------|--------|
| skim | 3 | 8 | — | 71.2 | — |
| fan_8 | 9 | 20 | 0.88 | 7.05 | 248 |
| fan_32 | 33 | 68 | 0.91 | 7.53 | 248 |
| fan_128 | 129 | 260 | 0.89 | 5.52 | 248 |
| fan_512 | 513 | 1028 | 0.96 | 4.95 | 249 |

Skim `riks_ms` includes one-time Numba compile on first call; fan rows amortize compile over repeated solves. Tangent assembly stays ~1 ms for hundreds of link elements — cost is dominated by Riks Newton loops and sparse factorization, not element kernel batching.

## Equality checks

- `test_link2_element.py` — truss/spring stiffness at element level
- `test_structural_mesh.py` — fan topology, tangent assembly, `n_rays=2` Riks vs skim
