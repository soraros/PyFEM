# SkorohodOlevsky (SOVS)

The `SkorohodOlevsky` model describes viscous sintering of ceramic powder
compacts: densification driven by the sintering stress and hydrostatic
stress, plus deviatoric creep, through the Skorohod viscosity functions
`η = η_ref · (ρ^−n − 1)` with the Arrhenius reference viscosity
`η_ref = eta0 · exp(Q / (R·T))`.

## Overview
- **Material type:** `SkorohodOlevsky`
- **Integration:** explicit forward Euler per step (density and viscous
  strain from committed-density viscosities and the trial stress).
- **State:** committed total strain, viscous strain, relative density `ρ`,
  and the last evaluation time.

## Parameters
### Mandatory
- `type`: Must be set to `"SkorohodOlevsky"`
- `eta0`: Reference viscosity (Pa·s)
- `Q`: Activation energy for viscous flow (J/mol)
- `T`: Temperature (K)
- `rho0`: Initial relative density, in `(0, 1]`
- `sigma_sint`: Sintering stress (Pa)

### Optional
- `R`: Universal gas constant (J/(mol·K)). Default: `8.314`
- `n_vol`: Volumetric viscosity exponent. Default: `2.0`
- `n_shear`: Shear viscosity exponent. Default: `1.0`

## Documented quirks (finding
`20261009-agent-vp1-improve-sovs-vp-migration-relevant-quirks-hard-coded-moduli-dormant`)

- **Hard-coded modulus law:** `E = 100 GPa · (ρ/ρ0)^2.5`, `ν = 0.25` — there
  are no `E`/`nu` properties; the elastic stiffness is a density-scaled
  constant law.
- **Full-density guard:** for `ρ ≥ 0.999` the viscosities are replaced by
  `1e20` Pa·s (essentially elastic), avoiding the Skorohod singularity.
- **Density clamp:** `ρ` is clamped into `[ρ0, 1]`; the volumetric viscous
  strain increment reads the UNCLAMPED `dρ`, so the clamp kinks the density
  trajectory but not the stress response of the step.
- **Output lag:** the reported density output channel lags one step (it
  reports the pre-update `ρ`). The committed state is the post-update value;
  the v3 law carries no reporting channel, so the lag has no v3 consequence.
- **Dormant decks:** the shipped sintering examples carry `eta0 = 1e12`,
  `Q = 5e5`, `T = 1600`, so `η_ref ≈ 2.1e28` Pa·s and the viscous machinery
  is inert (`dρ` rounds to zero) — the decks are effectively elastic. Any
  activated use (realistic viscosities) exercises the viscous update.

## Tangent note

The coded tangent is the closed form of an IMPLICIT step while the update is
explicit — an O(dt) inconsistency (finding
`20261009-agent-vp1-bug-sovs-tangent-is-the-implicit-step-form-of-an-explicit-update`:
9.4e-3 relative against a finite difference of the law's own response at
`dtime = 0.01` on activated constants, 9.9e-4 at `dtime = 0.001`; the
elastic branch is exact). Converged states are unaffected. The true
explicit-map derivative has closed form at the committed density —
`K_alg = K(1 − 3K·dt/2η_v)` volumetric, `G_dev = G(1 − G·dt/η_s)` on the
normal-deviatoric block, `G_alg = G(1 − G·dt/2η_s)` shear — exact because the
map is affine in the strain increment at fixed state. The legacy repair rides
the L3 wave; the v3 migration (`pyfem.v3` law `skorohod-olevsky`) ships the
true tangent from day one and starts its state at `ρ = ρ0`.

## References

- Skorohod, V.V. (1972). *Rheological basis of the theory of sintering.*
  Naukova Dumka, Kiev.
- Olevsky, E.A. (1998). *Theory of sintering: from discrete to continuum.*
  Materials Science and Engineering: R, 23(2), 41-100.
