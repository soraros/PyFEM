# Parity workflow (skims)

Three layers share one schema (`ProblemDefinition`):

| Layer | File | Purpose |
|-------|------|---------|
| **A** | `skims/<case>/skim.pro` | Thin legacy input; `input` points at book `.dat` |
| **B** | `skims/<case>/problem.toml` | Canonical constants and mesh path for v3 |
| **C** | `types.ProblemDefinition` | Contract both loaders must satisfy |

## Reference case: `patch_test8`

- Legacy: [`examples/ch02/PatchTest8.pro`](../../examples/ch02/PatchTest8.pro) + [`PatchTest8.dat`](../../examples/ch02/PatchTest8.dat)
- Skim: [`skims/patch_test8/`](../../skims/patch_test8/)
- Test: [`test/v3/test_parity_patch_test8.py`](../../test/v3/test_parity_patch_test8.py)

## Running parity

```bash
uv sync --group v3
uv run pytest test/v3/test_parity_patch_test8.py -q
```

## Expected result

Global displacement `state` from v3 `solve_linear` matches legacy `LinearSolver` within `parity.toml` (`rtol` / `atol`).

Document any intentional mismatch here with a physical or numerical reason before loosening tolerances.
