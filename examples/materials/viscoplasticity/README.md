# Viscoplastic Material Example

## What the shipped deck actually demonstrates

`bar_tension.pro` exercises the legacy `ViscoPlasticity` material on a notched
bar in tension. Despite the historical Perzyna branding, **this deck's
implemented constitutive map is rate-independent J2 plasticity with isotropic
linear hardening and a `dtime > 0` gate**. It must not be used as evidence of
rate-dependent yielding.

The local legacy residual is

```text
σ_vm,trial − 3G Δκ − (syield + hard (κ_n + Δκ)) = 0.
```

Neither `gamma`, `n`, nor `dtime` occurs in that residual. They only seed a
Newton initial guess that the linear solve discards. Consequently:

- all schedules with positive time increments give the same converged stress,
  apart from Newton-tolerance roundoff;
- changing `gamma` or `n` does not calibrate rate physics;
- a supra-yield step at constant or backward time takes the gate and commits
  the elastic trial response above yield, with plastic state frozen;
- the deck cannot show “fast loading → higher peak stress”.

The old Perzyna equation and rate-dependent claims formerly shown in this
README described the intended branding, not the algorithm in the deck.

## Problem and material data

The bar has length 8.0, width 0.5 (0.45 at the notch), thickness 0.2, and a
0.05-deep central notch. The material parameters are:

- `E = 200000.0`: Young's modulus;
- `nu = 0.3`: Poisson's ratio;
- `syield = 250.0`: initial von Mises yield stress;
- `hard = 1000.0`: linear isotropic hardening slope;
- `gamma = 0.001`, `n = 1.0`: numerically inert converged-map parameters in
  this legacy material, retained to preserve the deck's meaning.

Run the legacy example headlessly with:

```bash
MPLBACKEND=Agg pyfem bar_tension.pro
```

`GraphWriter` records time, horizontal displacement at node 17, reaction at
the loaded nodes, and equivalent plastic strain at the notch. Stress and
plastic strain concentrate at the notch; the response is elastic below yield
and follows rate-independent J2 hardening above it whenever time advances.

## Retired v3 rate-law record

NX7-1 retired the separate experimental v3 Perzyna mappings and their public
authoring example. Their implicit return-map equations, literature, independent
scalar-bisection/SymPy/finite-difference oracles, fast-loading overstress check,
and restart byte-equivalence evidence are preserved in the
[NX7-1 retirement record](../../../.agents/v3/migration-history.md#nx7-1-retirement-non-kernel-scientific-breadth).
The legacy gamma-invariance negative control is indexed there as well. None of
that retired functionality changes this legacy deck's meaning.
