# Parity workflow (skims)

Three layers share one schema (`ProblemDefinition`):

| Layer | File | Purpose |
|-------|------|---------|
| **A** | `skims/<case>/skim.pro` | Thin legacy input; `input` points at book `.dat` |
| **B** | `skims/<case>/problem.toml` | Canonical constants and mesh path for v3 |
| **C** | `types.ProblemDefinition` | Contract both loaders must satisfy |

## Reference cases

| Skim | Legacy example | Element | Test |
|------|----------------|---------|------|
| `patch_test8` | [`PatchTest8.pro`](../../examples/ch02/PatchTest8.pro) + [`PatchTest8.dat`](../../examples/ch02/PatchTest8.dat) | Q8 | [`test_parity_patch_test8.py`](../../test/v3/test_parity_patch_test8.py) |
| `patch_test8_loaded` | `skim.pro` + [`PatchTest8_loaded.dat`](../../skims/patch_test8_loaded/PatchTest8_loaded.dat) | Q8 + nodal load | [`test_parity_patch_test8_loaded.py`](../../test/v3/test_parity_patch_test8_loaded.py) |
| `patch_test4` | [`PatchTest4.pro`](../../examples/ch02/PatchTest4.pro) + [`PatchTest4.dat`](../../examples/ch02/PatchTest4.dat) | Quad4 | [`test_parity_patch_test4.py`](../../test/v3/test_parity_patch_test4.py) |
| `patch_test3` | [`PatchTest3.pro`](../../examples/ch02/PatchTest3.pro) + [`PatchTest3.dat`](../../examples/ch02/PatchTest3.dat) | Tria3 | [`test_parity_patch_test3.py`](../../test/v3/test_parity_patch_test3.py) |
| `patch_test8_mpc` | [`skim.pro`](../../skims/patch_test8_mpc/skim.pro) + [`PatchTest8_mpc.dat`](../../skims/patch_test8_mpc/PatchTest8_mpc.dat) | Q8 + MPC ties | [`test_parity_patch_test8_mpc.py`](../../test/v3/test_parity_patch_test8_mpc.py) |
| `patch_test8_plane_strain` | [`skim.pro`](../../skims/patch_test8_plane_strain/skim.pro) + [`PatchTest8.dat`](../../examples/ch02/PatchTest8.dat) | Q8 + PlaneStrain | [`test_parity_patch_test8_plane_strain.py`](../../test/v3/test_parity_patch_test8_plane_strain.py) |
| `patch_test8_3d` | [`PatchTest8_3D.pro`](../../examples/ch02/PatchTest8_3D.pro) + [`PatchTest8_3D.dat`](../../examples/ch02/PatchTest8_3D.dat) | Hex8 + Isotropic | [`test_parity_patch_test8_3d.py`](../../test/v3/test_parity_patch_test8_3d.py) |

## Running parity

```bash
uv sync --group v3
uv run pytest test/v3/test_parity_patch_test8.py -q
uv run pytest test/v3/test_parity_patch_test4.py -q
uv run pytest test/v3/test_parity_patch_test3.py -q
# or all v3 tests:
uv run pytest test/v3 -q
```

## Expected result

Global displacement `state` from v3 `solve_linear` matches legacy `LinearSolver` within each skim's `parity.toml` (`rtol` / `atol`).

Document any intentional mismatch here with a physical or numerical reason before loosening tolerances.
