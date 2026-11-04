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

## True backward-Euler Perzyna physics in v3

The separate v3 material key **`backward-euler-perzyna`** implements the
actual implicit rate law without changing this legacy deck or the
legacy-mapped v3 key `perzyna-viscoplasticity`:

```python
from pyfem.v3 import authoring

material = authoring.backward_euler_perzyna(
  200000.0, 0.3, 250.0, 1000.0, 0.001, 1.0
)
model = authoring.small_strain_continuum(
  authoring.quad8_patch(), material=material
)
system = authoring.compile(model)
```

Its equivalent viscoplastic strain rate is

```text
κ̇ = gamma <(σ_vm − σ_y(κ))/σ_y(κ)>^n,
σ_y(κ) = syield + hard κ.
```

A nonlinear static program must declare a `time` coordinate. The driver derives
`dtime` from the previous committed program point, and the material solves

```text
Δκ = gamma Δt <(σ_vm,trial − 3GΔκ − σ_y(κ_n+Δκ))/σ_y(κ_n+Δκ)>^n.
```

For the same displacement history, a faster physical schedule has less time
for viscoplastic flow and therefore carries higher overstress; a slower
schedule flows more and relaxes toward the yield surface. For this law,
`gamma Δt → 0` approaches the elastic predictor and `gamma Δt → ∞` approaches
rate-independent J2 consistency. (The directions formerly printed here were
incorrect.)

The Q8 regression `test_backward_euler_perzyna_q8_fast_loading_has_higher_overstress`
in `test/v3/test_v3_viscoplasticity.py` compares schedules differing by a
factor of 100 and checks overstress at every integration point. Other tests
cover unload/reload state, rejected-trial byte identity, and restart
byte-equivalence. The legacy gamma-invariance test remains as the negative
control.
