# H1 — Spring tangent semantics decision and transforms.py doc sweep (M24)

Date: 2026-09-30. Branch: feat/h1-spring-tangent-semantics-and-doc-swee (wt-24).
Base: v3 (green; the earlier 35-failure scare was the M19 stale-cache artifact).

## Decision: v3 keeps the axial spring tangent; legacy tangent is not the oracle

`pyfem/v3/fem/link2.py::_spring_local` pins the axial spring:

- elongation `e = -(b.a)` with `b = (c, s, -c, -s)`, `(c, s)` the element axis;
- internal force `f = -k e b = k (b.a) b`;
- tangent `K = d f / d a = k b b^T` — the exact derivative of the residual;
- relative transverse motion `t = (-s, c, s, -c)` is unresisted (`K t = 0`).

Legacy `pyfem/elements/Spring.py` assembles an isotropic `k * eye(2)` block
tangent `k * [[I, -I], [-I, I]]` (rotation-invariant) whose transverse
stiffness is not the derivative of its own axial residual (filed finding
20260929-agent-b3, M14). Resolution of that finding: the legacy spring
TANGENT is invalid as a v3 oracle; the legacy spring RESIDUAL stays
authoritative (fint parity holds, verified). v3 numerics are unchanged —
the pinned semantics already were the consistent ones. The legacy tangent
differs from the v3 tangent by exactly the transverse projector `k * t t^T`.

## Pins added (test/v3/test_link2_element.py)

- `test_spring_tangent_matches_closed_form` — docstring now states the axial
  `k * b b^T` semantics and names the non-oracle test (5 orientations).
- `test_spring_transverse_motion_is_unresisted` — `K t = 0` (5 orientations).
- `test_spring_legacy_tangent_is_not_the_oracle` — drives the real legacy
  `Spring` element from `skims/shallow_truss_riks/skim.pro`: legacy tangent
  equals the isotropic closed form, legacy fint equals v3 fint (residual
  parity), and `K_legacy - K_v3 = k * t t^T` exactly (5 orientations).
- Pre-existing `test_tangent_is_directional_derivative_of_force` already
  covers tangent/residual consistency by central finite difference.

The `_spring_local` kernel carries a comment recording the same decision.

## Doc sweep

Stale references to the deleted `pyfem/v3/fem/transforms.py` now cite the
post-M20 colocated layout (kernels in `pyfem/v3/fem/link2.py`):

- `.agents/v3/scaling.md` structural-fixtures table row;
- `.agents/v3/evidence/2026-07-18-r0e-capability-inventory.md` line 81
  (KERN-TRANSFORM `legacy_surface` now `pyfem/v3/fem/link2.py:56`).

Repo-wide grep confirms no `fem/transforms` references remain in `.agents/`,
`pyfem/`, `test/`, or `bench/`. Two adjacent stale rows found in the same
scaling.md table (`assemble_tangent_loaded` path, group-kind numbering) were
outside the task list and are filed as a separate finding.

## Verification counts (this branch, wt-24, repo `.venv`, `python -m pytest` from worktree root)

- `test/v3/test_link2_element.py`: 49 passed (39 before this mission; +10 new pins).
- Full repository suite, normal mode: **985 passed**, 44 warnings (124 s).
- Full repository suite, `PYTHONINTMAXSTRDIGITS=640`: **985 passed**, 44 warnings (104 s).
- `ruff check pyfem/v3 test/v3 --config pyfem/v3/ruff.toml`: pass.
- `ruff check test/v3 --config test/v3/ruff.toml`: pass.
- `ruff format --check pyfem/v3/fem/link2.py test/v3/test_link2_element.py --config pyfem/v3/ruff.toml`: pass.
