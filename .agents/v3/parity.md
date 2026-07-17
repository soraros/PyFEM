# Parity workflow (skims)

> **Numerical-reference evidence.** These skims remain useful oracles, but parity
> does not validate the new representation, state transactions, compatibility, or
> result provenance. Use them under the acceptance suite in [design.md](design.md).

The prototype parity harness currently has three layers. This is historical test
plumbing, not the new compiler contract:

| Layer | File | Purpose |
|-------|------|---------|
| **A** | `skims/<case>/skim.pro` | Thin legacy input; `input` points at book `.dat` |
| **B** | `skims/<case>/problem.toml` | Canonical constants and mesh path for v3 |
| **C** | `types.ProblemDefinition` | Prototype carrier both old loaders satisfy |

## Reference cases

| Skim | Legacy example | Element | Test |
|------|----------------|---------|------|
| `patch_test8` | [`PatchTest8.pro`](../../examples/ch02/PatchTest8.pro) + [`PatchTest8.dat`](../../examples/ch02/PatchTest8.dat) | Q8 | [`test_parity_linear.py`](../../test/v3/test_parity_linear.py) |
| `patch_test8_loaded` | `skim.pro` + [`PatchTest8_loaded.dat`](../../skims/patch_test8_loaded/PatchTest8_loaded.dat) | Q8 + nodal load | [`test_parity_linear.py`](../../test/v3/test_parity_linear.py) |
| `patch_test4` | [`PatchTest4.pro`](../../examples/ch02/PatchTest4.pro) + [`PatchTest4.dat`](../../examples/ch02/PatchTest4.dat) | Quad4 | [`test_parity_linear.py`](../../test/v3/test_parity_linear.py) |
| `patch_test3` | [`PatchTest3.pro`](../../examples/ch02/PatchTest3.pro) + [`PatchTest3.dat`](../../examples/ch02/PatchTest3.dat) | Tria3 | [`test_parity_linear.py`](../../test/v3/test_parity_linear.py) |
| `patch_test8_mpc` | [`skim.pro`](../../skims/patch_test8_mpc/skim.pro) + [`PatchTest8_mpc.dat`](../../skims/patch_test8_mpc/PatchTest8_mpc.dat) | Q8 + MPC ties | [`test_parity_linear.py`](../../test/v3/test_parity_linear.py) |
| `patch_test8_plane_strain` | [`skim.pro`](../../skims/patch_test8_plane_strain/skim.pro) + [`PatchTest8.dat`](../../examples/ch02/PatchTest8.dat) | Q8 + PlaneStrain | [`test_parity_linear.py`](../../test/v3/test_parity_linear.py) |
| `patch_test8_3d` | [`PatchTest8_3D.pro`](../../examples/ch02/PatchTest8_3D.pro) + [`PatchTest8_3D.dat`](../../examples/ch02/PatchTest8_3D.dat) | Hex8 + Isotropic | [`test_parity_patch_test8_3d.py`](../../test/v3/test_parity_patch_test8_3d.py) |
| `patch_test8_nonlinear` | [`skim.pro`](../../skims/patch_test8_nonlinear/skim.pro) + loaded `.dat` | Q8 + `NonlinearSolver` | [`test_nonlinear_solver.py`](../../test/v3/test_nonlinear_solver.py) |
| `patch_test8_nonlinear_ramp` | multi-step `loadTable` | Q8 + ramp | same |
| `patch_test8_nonlinear_prescribed` | PatchTest8 `.dat` + ramp | prescribed-driven | same |
| `cantilever8` | [`cantilever8.pro`](../../examples/ch03/cantilever8.pro) + [`cantilever8.dat`](../../examples/ch03/cantilever8.dat) | Q8 + `FiniteStrainContinuum` | [`test_parity_cantilever8.py`](../../test/v3/test_parity_cantilever8.py) |
| `shallow_truss_riks` | [`ShallowtrussRiks.pro`](../../examples/ch04/ShallowtrussRiks.pro) + [`ShallowtrussRiks.dat`](../../examples/ch04/ShallowtrussRiks.dat) | Truss + Spring + `RiksSolver` | [`test_parity_shallow_truss_riks.py`](../../test/v3/test_parity_shallow_truss_riks.py) |

## Running parity

```bash
uv sync
uv run pytest test/v3/test_parity_linear.py -q
# or all v3 tests:
uv run pytest test/v3 -q
```

## Expected result

For the current prototype, global displacement `state` from `solve_linear` matches
legacy `LinearSolver` within each skim's `parity.toml` (`rtol` / `atol`). New flows
reuse the numerical oracle without preserving this API or carrier.

Document any intentional mismatch here with a physical or numerical reason before loosening tolerances.
