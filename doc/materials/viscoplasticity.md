# Legacy ViscoPlasticity

This page documents only the legacy `ViscoPlasticity` material. NX7-1 retired
the experimental v3 `perzyna-viscoplasticity` and `backward-euler-perzyna`
mappings. Their return-map and tangent derivations, authoring example,
verification references, and independent oracle locations are preserved in the
[NX7-1 retirement record](https://github.com/jjcremmers/PyFEM/blob/v3/.agents/v3/migration-history.md#nx7-1-retirement-non-kernel-scientific-breadth).
No v3 authoring or registry contract is claimed here.

## Implemented legacy constitutive map

The legacy class implements J2 plasticity with isotropic linear hardening and a
time-step gate. Its converged local residual is

```text
σ_vm,trial − 3G Δκ − σ_y(κ_n + Δκ) = 0,
σ_y(κ) = syield + hard κ.
```

This is **rate-independent** consistency plasticity, not a Perzyna integration.
`gamma`, `n`, and `dtime` only seed a discarded Newton initial guess. Measured
on the documented ramp, changing `(gamma=1e-4, n=1)` to `(gamma=1e2, n=2)`
changes stress by only about `2e-16` relative. The retired v3 verification's
legacy gamma-invariance negative control is indexed in the retirement record.

The sole time effect is the `dtime > 0` gate. A supra-yield step at constant or
backward time returns and commits the elastic trial stress above yield while
freezing `epsilon_p` and `kappa`. Changing time schedules therefore does not
demonstrate physical rate dependence.

The legacy local iteration warns and continues on non-convergence. Its returned
J2 tangent is the consistent rate-factor-free form after the M67 repair removed
the spurious derivative of the discarded initial guess. The retirement record
preserves the independent finite-difference and symbolic checks of that repair.
