# ViscoPlasticity

The `ViscoPlasticity` model is a J2 plasticity law with isotropic linear
hardening and a time-step gate. Despite the Perzyna branding, **the
implemented integration is rate-INDEPENDENT** — see the correction below.

## Overview
- **Material type:** `ViscoPlasticity`
- **State:** elastic strain, plastic strain, accumulated equivalent plastic
  strain, and the last evaluation time.

## Parameters
### Mandatory
- `type`: Must be set to `"ViscoPlasticity"`
- `E`: Young's modulus
- `nu`: Poisson's ratio
- `syield`: Initial yield stress
- `gamma`: Fluidity parameter (see the correction below: numerically inert)

### Optional
- `n`: Rate sensitivity exponent. Default: `1.0` (numerically inert)
- `hard`: Linear hardening modulus. Default: `0.0` (perfectly plastic)

## Correction: the implemented map is rate-independent J2

The class documentation and `examples/materials/viscoplasticity/README.md`
describe Perzyna overstress theory — rate-dependent yielding, "fast loading →
higher peak stress", gamma controlling the flow rate. **These claims are
false for the implemented algorithm** (finding
`20261009-agent-vp1-bug-viscoplasticity-stress-update-is-rate-independent-j2-perzyna`):

- The local Newton iteration solves the rate-independent J2 consistency
  equation `σ_vm − 3G·Δε_p − σ_y(ε̄_p) = 0`. No rate term enters the
  residual; `gamma`, `n`, and `dtime` only seed the initial guess, which the
  linear-residual Newton discards. Measured: running a 12-step plastic ramp
  with `gamma = 1e-4, n = 1` versus `gamma = 1e2, n = 2` changes the
  converged stress by **1.95e-16 relative** — pure Newton-tolerance slack.
- The ONLY time effect is the `dtime > 0` gate: a supra-yield step at
  constant (or backward) time returns the **elastic** trial response above
  the yield surface. Schedules that hold time constant while advancing the
  load therefore commit elastic states above yield — a semantic trap.

Consequences: the shipped deck cannot exhibit the rate-dependent behavior
its README promises, and any calibration against `gamma`/`n` is meaningless.
A true-Perzyna rate law is a deferred feature, not a bugfix of this class.

## Notes

- **Tangent:** the returned plastic-branch tangent adds a spurious
  `rate_factor` term (the derivative of the discarded initial guess, not of
  the converged map — finding
  `20261009-agent-vp1-bug-viscoplasticity-tangent-inconsistent-with-its-own-stress-upd`).
  Converged states are unaffected; measured against a finite difference of
  the law's own stress response the coded tangent errs by 6.6e-2 relative at
  an amplified state (`gamma = 1e4`, `dtime = 1e3`), while the
  rate-factor-free J2 consistent tangent matches to 3e-8 class. The term is
  dormant at deck-like time steps (~1e-11 relative). The legacy repair rides
  the L3 wave; the v3 migration (`pyfem.v3` law `perzyna-viscoplasticity`)
  ships the true tangent from day one.
- **Non-convergence:** the legacy local iteration warns and continues; the
  v3 law reports a typed step rejection instead.
