# Viscoplasticity

PyFEM has two deliberately distinct material mappings. The legacy mapping is
preserved for deck compatibility; true Perzyna rate physics is new v3
functionality under another key.

## Legacy `ViscoPlasticity` / v3 `perzyna-viscoplasticity`

The legacy class and its v3 mapping implement J2 plasticity with isotropic
linear hardening and a time-step gate. Their converged local residual is

```text
σ_vm,trial − 3G Δκ − σ_y(κ_n + Δκ) = 0,
σ_y(κ) = syield + hard κ.
```

This is **rate-independent** consistency plasticity, not a Perzyna integration.
`gamma`, `n`, and `dtime` only seed a discarded Newton initial guess. Measured
on the documented ramp, changing `(gamma=1e-4, n=1)` to `(gamma=1e2, n=2)`
changes stress by only about `2e-16` relative. The regression suite retains
this gamma invariance as a negative control.

The sole time effect is the `dtime > 0` gate. A supra-yield step at constant or
backward time returns and commits the elastic trial stress above yield while
freezing `epsilon_p` and `kappa`. Changing time schedules therefore does not
demonstrate physical rate dependence.

The legacy-mapped v3 descriptor keeps 14 state floats (`epsilon_e[6]`,
`epsilon_p[6]`, `kappa[1]`, `time[1]`) and its identity `time` port. Its name,
parameters, state schema, and parity tests are unchanged. The legacy local
iteration warns and continues on non-convergence; v3 reports a typed step
rejection. The returned J2 tangent is the consistent rate-factor-free form
(the M67 repair removed the spurious derivative of the discarded initial
guess).

## True v3 `backward-euler-perzyna`

This fresh constitutive descriptor implements associative J2 Perzyna flow with
linear isotropic hardening. It is not selected by a legacy `ViscoPlasticity`
deck and must not be confused with the mapping above.

### Rate equation and implicit return map

The equivalent viscoplastic strain rate is

```text
κ̇ = gamma <(q − y)/y>^n,
q = σ_vm,
y = syield + hard κ.
```

`<x>` denotes `max(x,0)`. For trial von Mises stress `q_trial`, accepted
`kappa_n`, `l = Δκ`, `a = gamma Δt`, and `A = 3G + hard`, backward Euler gives
the scalar residual

```text
r(l) = l − a <(q_trial − A l − y_n)/(y_n + hard l)>^n = 0
q(l) = q_trial − 3G l
y(l) = y_n + hard l.
```

For positive trial overstress and positive `a`, the root is bracketed by zero
and `(q_trial-y_n)/A`. The implementation uses safeguarded Newton iterations
with the analytical residual derivative and bisection fallback. Zero `dtime`
is exactly elastic; negative or non-finite `dtime` rejects the step with trial
state byte-equal to accepted state.

The response is genuinely time-dependent: at the same displacement, faster
loading permits less viscoplastic flow and produces higher overstress. As
`gamma Δt → 0`, the map approaches the elastic predictor. As
`gamma Δt → ∞`, normalized overstress approaches zero and the map approaches
rate-independent J2 consistency.

### Algorithmic tangent

On the active branch, define

```text
φ = (q−y)/y
B = A y + hard (q−y) = 3G y + hard q
w = a n φ^(n−1).
```

Implicit differentiation of the same residual gives

```text
dl/dq_trial = w y / (y² + w B)
dq/dq_trial = 1 − 3G dl/dq_trial.
```

With `N = dev(σ_trial)/q_trial`, the stress is
`σ = p_trial I + qN`. The consistent engineering-Voigt tangent is the radial
elastic block plus
`(3G dq/dq_trial − 3G q/q_trial) N⊗N`; it is symmetric. Elastic steps return
the elastic stiffness.

### Parameters, state, and authoring

The six required parameters are `youngs_modulus`, `poisson_ratio`,
`initial_yield_stress`, `hardening_slope`, `fluidity`, and `rate_exponent`.
Fluidity has units of inverse time because the overstress ratio is
nondimensional. State contains 13 floats: `epsilon_e[6]`, `epsilon_p[6]`, and
monotone `kappa[1]`. The descriptor declares a derived `dtime` port with a
`time` derivative coordinate, so absolute time need not be stored in each
integration-point row.

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

Declare `time` among the nonlinear program coordinates. The driver computes
each `dtime` from the previous committed point, including after a run-boundary
restart.

### Verification

`test/v3/test_v3_viscoplasticity.py` validates the return map against an
independent scalar bisection solve and the tangent against central finite
differences away from branch kinks. Across exponents 0.7, 1, and 2, with and
without hardening, the worst measured root deviation is `2.86e-14` relative
and the worst tangent deviation is `3.24e-11` relative of `max|FD|`. A Sympy
implicit differentiation checks the analytical tangent structurally. Elastic
and rate-independent limiting controls are included.

A displacement-controlled Q8 test compares physical schedules differing by a
factor of 100 and requires higher overstress everywhere in the fast schedule.
Further tests prove unload/reload state correctness, rejected-trial byte
identity, and uninterrupted-versus-restart byte equivalence.

## References

- P. Perzyna, “Fundamental Problems in Viscoplasticity,” *Advances in Applied
  Mechanics* 9 (1966), 243–377,
  [doi:10.1016/S0065-2156(08)70009-7](https://doi.org/10.1016/S0065-2156(08)70009-7).
- J.C. Simo and T.J.R. Hughes, *Computational Inelasticity*, Springer, 1998,
  chapters 2 and 5,
  [doi:10.1007/b98904](https://doi.org/10.1007/b98904).
