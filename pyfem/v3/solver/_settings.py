"""Shared solver settings accessors."""

from __future__ import annotations

from pyfem.v3.types import LoadedProblem, NonlinearSolverSettings, RiksSolverSettings


def nonlinear_settings(loaded: LoadedProblem) -> NonlinearSolverSettings:
  if loaded.nonlinear_settings is not None:
    return loaded.nonlinear_settings
  return NonlinearSolverSettings()


def riks_settings(loaded: LoadedProblem) -> RiksSolverSettings:
  if loaded.riks_settings is not None:
    return loaded.riks_settings
  return RiksSolverSettings()
