# Crystal plasticity (v3)

> **Retired historical evidence — NX7-1, 2026-11-05.** This wholly-v3 page was
> moved out of the live Sphinx materials toctree. It records the implementation
> and independent verification at boundary revision `d9e4a12`; it does not
> describe currently supported physics. Crystal users remain first-class
> concerns. A future documented consumer-backed mission may reintroduce this
> capability on the minimal generic kernel, using the literature and oracles
> below, but must not resurrect a duplicate frozen or prototype runtime.

At `d9e4a12`, `crystal-plasticity` was a small-strain, rate-dependent
single-crystal law for the v3 stateful continuum interface. It was registered by
`pyfem.v3.compile.continuum.crystal_reference_registry()` with implementation
id `pyfem-v3-crystal-plasticity-v1`. The descriptor used isotropic elasticity
and crystal axes aligned with the global axes; two-dimensional models used the
standard plane-strain embedding into the internal 3D law.

## Slip systems and Schmid kinetics

The default `crystal_structure = 0.0` generates the **complete FCC
`{111}<110>` family**: four distinct slip planes with three directions per
plane, giving 12 systems. `crystal_structure = 1.0` selects the **complete BCC
`{110}<111>` family**: six planes with two directions per plane, also giving
12 systems. Partial families are not accepted; `slip_system_count` must be
`12.0` for either option. Signed slip on a generated system represents motion
in either direction along that system's slip vector.

For unit plane normal `n` and orthonormal slip direction `s`, the Schmid
tensor is `P = sym(s ⊗ n)`. Resolved shear stress and slip rate are

```text
tau = sigma : P
g   = gamma0 sign(tau) |tau / tau_c|^m.
```

The implementation uses engineering 6-Voigt order
`[xx, yy, zz, yz, zx, xy]`. Frames are validated at binding time. A
non-unit or nonorthogonal normal/direction pair raises the coded
`nonorthogonal-slip-frame` diagnostic with the pair and dot product; the code
never silently orthonormalizes an invalid specification.

The power law follows D. Peirce, R. J. Asaro, and A. Needleman, “Material rate
dependence and localized deformation in crystalline solids,” *Acta
Metallurgica* **31** (1983), 1951–1976,
[doi:10.1016/0001-6160(83)90014-7](https://doi.org/10.1016/0001-6160(83)90014-7).

## Saturation and latent hardening

Each system has absolute accumulated slip `gamma_cum_alpha`. Self and latent
hardening enter through

```text
Gamma_alpha = sum_beta Q_alpha_beta gamma_cum_beta
Q_alpha_beta = delta_alpha_beta + qab (1 - delta_alpha_beta)
d tau_c / d Gamma = h0 (1 - tau_c / taus)^a,    a >= 1.
```

Thus `h0` is the initial hardening slope, `taus > tau0` is the saturation
resistance, and `qab >= 1` controls latent hardening. The integrated law is
evaluated directly. With `y = 1 - tau_c / taus` and
`y0 = 1 - tau0 / taus`,

```text
y = y0 exp(-h0 Gamma / taus)                         (a = 1)
y^(1-a) = y0^(1-a) + (a - 1) h0 Gamma / taus        (a > 1).
```

This is the Kocks–Mecking saturation form reviewed by U. F. Kocks and H.
Mecking, “Physics and phenomenology of strain hardening: the FCC case,”
*Progress in Materials Science* **48** (2003), 171–273,
[doi:10.1016/S0079-6425(02)00003-8](https://doi.org/10.1016/S0079-6425(02)00003-8).
Resistance is recomputed at every current Newton iterate, including the latent
contribution; it is not a constant hardening matrix.

## Implicit return map and tangent

For step slip increments `x_alpha = Delta gamma_alpha`, the theta method solves

```text
r = x - dtime [(1 - theta) g_accepted + theta g(x)] = 0
sigma = C : (strain - sum_alpha P_alpha (gamma_alpha + x_alpha)).
```

The accepted rate comes from the accepted state. Every other rate, stress,
and resistance in the residual is evaluated at the **current** `x`. The local
Newton uses at most 50 iterations and the authored `newton_tolerance`.
`theta = 0` is the explicit accepted-rate update; `theta = 1` is backward
Euler. A zero time increment freezes slip, and a converged zero-increment
step returns exactly the elastic stress and tangent.

The consistent algorithmic tangent is

```text
D = C - C P J^-1 theta dtime diag(dg/dtau) P^T C,
J = dr/dx.
```

The full `theta*dtime*diag(dg/dtau)` factor and the hardening derivatives in
`J` are retained. General multislip tangents are nonsymmetric, so the
descriptor declares `algorithmic-nonsymmetric`; the driver must not impose a
symmetric solver. At the verification battery's mixed plastic state, central
finite differences match the tangent to `2.51e-9` relative at a `1e-7`
stencil (`2.25e-10` at `3e-8`), and measured relative asymmetry is `6.823e-5`.
A tangent omitting the implicit rate factor fails that same finite-difference
check by `2.25e4` relative.

## State, time, and failure behavior

Each integration point owns `4*nslip + 7` floats (55 for either complete
family):

1. `tau_c[nslip]`: current slip resistances;
2. `gamma[nslip]`: signed accumulated slips;
3. `tau[nslip]`: resolved shear stresses;
4. `gamma_cum[nslip]`: absolute accumulated slips;
5. `sigma[6]`: committed stress;
6. `gamma_total`: sum of absolute slip increments.

Initial rows set every resistance to `initial_critical_stress` and all other
values to zero. The schedule step arrives through the **derived `dtime`
signal port**, not an identity time port; state rows therefore need no time
slot. Accepted arrays are input-pure. Trial rows become history only when the
owner commits them. Non-finite input or results, negative time increments,
singular local solves, and Newton exhaustion return `REJECT_STEP` with trial
rows byte-equal to the accepted rows.

## Parameters

The v3 parameter order is:

| Parameter | Meaning | Domain |
|---|---|---|
| `youngs_modulus` | Isotropic Young's modulus | positive |
| `poisson_ratio` | Isotropic Poisson ratio | `-1 < nu < 0.5` |
| `crystal_structure` | `0.0` FCC, `1.0` BCC | enumeration |
| `slip_system_count` | Complete-family width | `12.0` |
| `reference_slip_rate` | `gamma0` | positive |
| `rate_exponent` | `m` | greater than 1 |
| `initial_critical_stress` | `tau0` | positive |
| `hardening_modulus` | `h0` | non-negative |
| `saturation_stress` | `taus` | greater than `tau0` |
| `hardening_exponent` | `a` | at least 1 |
| `latent_hardening_ratio` | `qab` | at least 1 |
| `theta` | Theta-method weight | `[0, 1]` |
| `newton_tolerance` | Scaled local residual tolerance | positive |

## Why this is fresh, not ported

The removed legacy `Crystal.py` is not a numerical oracle. This implementation
was authored from the references above and verified independently. It
intentionally designs out five convicted legacy failure modes:

1. **A zero initial residual:** legacy evaluated the old rate at the trial
   state, so its implicit Newton loop exited before changing stress. Here the
   old rate is accepted-state data and the residual is evaluated at the
   current iterate. The single-system battery measures a nonzero initial
   residual before independently solving the analytical scalar equation.
2. **Ad-hoc slip subsets:** legacy generated six-of-twelve systems and
   singleton sets. Here cubic symmetry generates all 12 FCC or BCC systems.
   The old deck's `[110]` normal and `[111]` direction are nonorthogonal; the
   analytical anchor uses a corrected orthonormal FCC frame instead.
3. **A constant hardening matrix:** legacy ignored cumulative slip in its
   hardening-matrix calculation. Here self and latent resistance evolve with
   current-iterate accumulated slip under the cited saturation law.
4. **Silent singularity handling:** legacy bare-except paths hid singular
   solves and non-convergence. Here expected numerical failures are typed
   `REJECT_STEP` results with untouched accepted history.
5. **An inconsistent tangent:** legacy omitted the implicit
   `theta*dtime*diag(g')` factor. Here it appears explicitly and is convicted
   by both finite-difference and tampered-tangent tests, with a sympy
   implicit-differentiation leg on the per-system return map.

Verification uses no legacy arithmetic: analytical single-slip hardening and
stress response, sympy, finite differences, elastic-limit exactness,
transaction isolation, two-system competition on the full FCC family, and an
end-to-end compiled-driver derived-`dtime` path. The exact test locations,
including `test_state_transactions_and_typed_rejections`, are indexed in
[the NX7-1 retirement record](../migration-history.md#nx7-1-retirement-non-kernel-scientific-breadth).
