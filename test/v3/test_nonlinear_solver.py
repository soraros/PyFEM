# SPDX-License-Identifier: MIT

"""Nonlinear static solver vs legacy and linear reference."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest

if sys.version_info < (3, 13):
    pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from _legacy_parity import legacy_nonlinear_state

from pyfem.v3 import load_problem, solve_linear, solve_nonlinear
from pyfem.v3.solver.constraints import build_prescribed_constraints
from pyfem.v3.solver.nonlinear import newton_step
from pyfem.v3.solver.tangent_context import prepare_tangent_assembly
from pyfem.v3.types import NonlinearSolverSettings

ROOT = Path(__file__).resolve().parents[2]


def _tolerances(skim_name: str) -> tuple[float, float]:
    with (ROOT / "skims" / skim_name / "parity.toml").open("rb") as fh:
        data = tomllib.load(fh)
    return float(data["rtol"]), float(data["atol"])


def test_nonlinear_loaded_matches_legacy_and_linear() -> None:
    rtol, atol = _tolerances("patch_test8_nonlinear")
    skim_pro = ROOT / "skims" / "patch_test8_nonlinear" / "skim.pro"
    loaded = load_problem(skim_pro)
    v3_state = solve_nonlinear(loaded).state
    legacy = legacy_nonlinear_state(skim_pro)
    loaded_linear = load_problem(ROOT / "skims" / "patch_test8_loaded" / "skim.pro")
    linear = solve_linear(loaded_linear)
    np.testing.assert_allclose(v3_state, legacy, rtol=rtol, atol=atol)
    np.testing.assert_allclose(v3_state, linear, rtol=rtol, atol=atol)


def test_nonlinear_ramp_matches_legacy_and_linear() -> None:
    rtol, atol = _tolerances("patch_test8_nonlinear_ramp")
    skim_pro = ROOT / "skims" / "patch_test8_nonlinear_ramp" / "skim.pro"
    v3_state = solve_nonlinear(load_problem(skim_pro)).state
    legacy = legacy_nonlinear_state(skim_pro)
    loaded_linear = load_problem(ROOT / "skims" / "patch_test8_loaded" / "skim.pro")
    linear = solve_linear(loaded_linear)
    np.testing.assert_allclose(v3_state, legacy, rtol=rtol, atol=atol)
    np.testing.assert_allclose(v3_state, linear, rtol=rtol, atol=atol)


def test_nonlinear_prescribed_ramp_matches_legacy() -> None:
    rtol, atol = _tolerances("patch_test8_nonlinear_prescribed")
    skim_pro = ROOT / "skims" / "patch_test8_nonlinear_prescribed" / "skim.pro"
    v3_state = solve_nonlinear(load_problem(skim_pro)).state
    legacy = legacy_nonlinear_state(skim_pro)
    linear = solve_linear(load_problem(ROOT / "skims" / "patch_test8" / "skim.pro"))
    np.testing.assert_allclose(v3_state, legacy, rtol=rtol, atol=atol)
    np.testing.assert_allclose(v3_state, linear, rtol=rtol, atol=atol)


def test_nonlinear_fails_when_not_converged() -> None:
    loaded = load_problem(ROOT / "skims" / "patch_test8_nonlinear" / "skim.pro")
    loaded.nonlinear_settings = NonlinearSolverSettings(
        tol=1.0e-10,
        iter_max=0,
        load_table=np.array([0.0, 1.0]),
    )
    with pytest.raises(RuntimeError, match="Newton-Raphson"):
        solve_nonlinear(loaded)


def test_newton_recovers_from_perturbed_state_with_cached_k() -> None:
    rtol, atol = _tolerances("patch_test8_nonlinear")
    loaded = load_problem(ROOT / "skims" / "patch_test8_nonlinear" / "skim.pro")
    loaded_linear = load_problem(ROOT / "skims" / "patch_test8_loaded" / "skim.pro")
    reference = solve_linear(loaded_linear)
    constraints = build_prescribed_constraints(loaded.problem)
    tangent_ctx = prepare_tangent_assembly(loaded)
    settings = NonlinearSolverSettings(
        tol=1.0e-10,
        iter_max=10,
        load_table=np.array([0.0, 1.0]),
    )
    perturbed = reference * 0.85
    recovered = newton_step(
        loaded,
        perturbed,
        1.0,
        settings=settings,
        constraints=constraints,
        tangent_ctx=tangent_ctx,
    )
    np.testing.assert_allclose(recovered, reference, rtol=rtol, atol=atol)


def test_load_func_rejects_unsafe_expression() -> None:
    from pyfem.v3.io.load_ramp import validate_load_func

    with pytest.raises(ValueError, match="loadFunc"):
        validate_load_func("sin(t)")
