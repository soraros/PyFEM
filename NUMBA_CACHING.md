# Numba `cache=True` — stale-kernel hazard notes

Status: **final** — root cause established by PyFEM tower mission M19
(numba 0.62.1, CPython 3.13, macOS). Everything below is verified by
reproduction; the earlier preliminary version of this document had two
errors, corrected in §4 and §5.

These notes exist because the hazard was found here, but nothing in them is
PyFEM-specific: any package using `@njit(cache=True)` with **call edges or
globals crossing module boundaries** is affected.

## 1. The incident

While fixing a one-line bug in a `cache=True` numba kernel call path
(`pyfem/v3/fem/transforms.py`), we discovered that after editing the source:

- the **stale, pre-edit kernel was still served** from numba's on-disk cache
  (`__pycache__` next to the source file);
- this produced **false-green** results (old source + new cache → new tests
  "passed" against the stale kernel) and, in the opposite direction,
  **false-red** results (new source + old cache → correct code "failed");
- only clearing the package `__pycache__` produced truthful before/after
  measurements.

Reproduced independently by two engineers during the incident, then again by
the M19 production replica on a read-only tree copy with the real test suite:
post-fix cold = 39 passed (truthful) → pre-fix source + warm cache = 39
passed (**false-green**) → cleared = 35 failed (truthful) → post-fix restored
+ warm cache = 35 failed (**false-red**) → cleared = 39 passed.

## 2. Root cause (verified against numba 0.62.1 source and docs)

Numba's on-disk cache works per source **file**:

- a per-function `.nbi` index in `__pycache__` stamps the source file's
  `(st_mtime, st_size)` — a mismatch empties the whole index
  (`numba/core/caching.py`, ~lines 581–604);
- each entry is keyed by `(signature, codegen magic,
  sha256 of the function's own bytecode, sha256 of closure cell contents)`
  (~lines 764–782).

**Neither input covers symbols defined in other files.** Two documented
upstream limitations (numba's own manual states both):

1. *"Cache invalidation fails to recognize changes in symbols defined in a
   different file."* A cached **caller** embeds the machine code of any
   `@njit` callee from another module, frozen at the caller's compile time.
   Edit the callee's file and the callee's own entry invalidates — but the
   caller's file/bytecode are unchanged, so the caller's cache entry still
   hits and keeps executing the **old callee code embedded inside it**.
2. *"Global variables are treated as constants … will not rebind."* A kernel
   reading a module-level constant defined in another file freezes its value
   at compile time; editing the constants file does not invalidate the
   reader.

Same-file edits always invalidate (the file stamp + own-bytecode hash both
cover them) — including literal-only edits. This is why single-file probes
can never reproduce the hazard (see §4).

In our incident the result was a **franken-build**: the edited module's
kernels were fresh while its callers' kernels, in the same process, kept
serving the pre-edit callee they had embedded. Clearing `__pycache__` "fixed"
measurements only because it forced every caller to recompile.

## 3. "Did we do anything ad-hoc to trigger it?" — No

The trigger is intrinsic to the code structure: `cache=True` callers in one
module calling `cache=True` callees in another, or reading cross-file
globals. Any ordinary edit to the callee module — editor, git, sed —
triggers it in fresh processes. Explicitly ruled out as requirements:

- factory-produced dispatchers / decorator or qualname patterns (all our
  production kernels are direct `@njit(cache=True)` decorations);
- dispatcher singleton reuse within a process;
- environment variables (no `NUMBA_*` / `PYTHONDONTWRITEBYTECODE` anywhere);
- mtime manipulation, worktree artifacts, filesystem timestamp granularity
  (a `(mtime, size)`-preserving edit *can* also stale the cache, but is not
  what happened here).

## 4. How to check your own package (corrected methodology)

The faithful minimal reproduction needs **two modules**, not one:

```python
# /tmp/cache_probe_callee.py
from numba import njit

@njit(cache=True)
def callee(x):
    return 2.0 * x          # <- edit this line between runs (e.g. 5.0 * x)
```

```python
# /tmp/cache_probe_caller.py
from numba import njit
from cache_probe_callee import callee

@njit(cache=True)
def caller(x):
    return callee(x) + 1.0

print(caller(10.0))
```

```sh
python /tmp/cache_probe_caller.py   # run 1 → 21.0, caches populated
# edit ONLY the callee module
python /tmp/cache_probe_caller.py   # run 2 → still 21.0 (fresh would be 51.0): STALE
```

A single-file probe (editing the kernel you call directly) **always
invalidates** and will falsely reassure you. The cross-file **globals**
variant is the same shape: a kernel reading a constant imported from another
module keeps the frozen value after that module is edited.

How to audit a real package: enumerate every `cache=True` kernel, build its
call graph, and flag any edge (jit callee or global read) whose target lives
in a different file. Those sites are cache-unfriendly.

## 5. Policy (in force here)

1. **No manual cache manipulation as a standing practice** — no blanket
   `__pycache__` clearing rituals, no per-checkout cache-dir surgery. (This
   supersedes the interim "clear between runs" discipline in the preliminary
   version of this document; it masked the bug rather than fixing it.)
2. **Cache-unfriendly kernel code is a bug.** Every `cache=True` kernel is
   classified: (a) verifiably cache-friendly — all cross-references
   same-file — keep; (b) cache-unfriendly — filed as a bug and fixed; or
   (c) caching explicitly disabled per kernel with a documented reason where
   (b) is impractical.
3. **No unspecified behavior**: no code path whose behavior depends on
   unstated cache state.

Fix patterns, in preference order:

- **Colocate**: move/inlinesingle-consumer helpers into the caller's module,
  so the per-file stamp gate covers the whole call chain (zero runtime
  cost). Define or copy cross-file constants locally.
- **Documented `cache=False`** on exactly the cross-module kernels where
  colocation is too invasive — keeps same-file helpers cached, costs seconds
  of compile per fresh process.
- **Regression canary**: an AST scan in the test suite that fails on any
  `cache=True` kernel referencing names imported from other modules of the
  package, unless the site carries an explicit documented `cache=False`.
  This enforces the policy permanently without touching cache state.

Rejected: blanket `__pycache__` clearing (masks the bug), per-checkout
`NUMBA_CACHE_DIR` (surgery; doesn't fix cross-file invalidation), blanket
`cache=False` (punishes the majority of kernels that are fine), relying on a
numba upgrade (the limitation is documented upstream — re-check when the
pin is bumped), process discipline alone (that *is* unspecified behavior).

## 6. Measurement discipline

With the bugs fixed and the canary in place, no fresh-cache-per-run rituals
are needed. Until then, treat any before/after, parity, or benchmark result
involving a `cache=True` kernel as cache-suspect if a helper module was
edited since the last cold compile.

Benchmark harnesses deserve special care: parity/correctness gates catch
numerically significant staleness (loud, safe direction) but **cannot catch
performance-only helper edits** — numerics identical, gate passes, timings
silently measure old embedded code. If you time `cache=True` kernels, either
fix cross-file edges first or record a coherence check (caller cache-entry
mtime ≥ callee source mtime) in the run manifest.

A deliberate `NUMBA_CACHE_DIR` pointing at an empty directory is the
legitimate way to measure true-cold compile cost — that is measurement, not
manipulation.

## 7. Separate, unrelated trap: which tree your test run imports

Verified on this repo: running plain `pytest` (or `uv run pytest`) from a
git **worktree** imported the **main checkout's** package (path-style
editable install via `.pth`, and the test dir has no `__init__.py`), while
`python -m pytest` from the checkout root imports that checkout's code. If
you benchmark or test across worktrees, standardize on
`python -m pytest` / `python -m bench.run` from the checkout root, and
consider a conftest warning when the imported package's `__file__` resolves
outside the current working tree.

## 8. PyFEM blast radius (for the record)

48 `cache=True` kernels audited: 41 production + 7 test probes.

- **Cache-friendly, keep — 33**: all call-graph edges same-file, no
  cross-file globals (transforms 4, kinematics 4, tl_kinematics 6, shapes 5,
  quadrature 7, assembly 4, link2 local kernels 2, one element integrator).
- **Cache-unfriendly bugs — 8 production + 7 test probes**, fixed under
  follow-up mission N2: `link2_tangent_single/batched` (callees in
  transforms.py + `GROUP_TRUSS`/`GROUP_SPRING` globals from types.py), 5
  batched stiffness kernels in element.py and `quad8_tl_tangent_batched` in
  tl_element.py (callees in quadrature/shapes/kinematics/tl_kinematics), and
  the 7 test probes (callees in quadrature.py).
- **Past results**: only the incident's pre-clear before/after numbers were
  known-false (superseded by cleared re-measurement; the merged fix itself
  verified three ways). All benchmark baselines stand — they measured
  unmodified kernels with no edits between cache population and measurement.
