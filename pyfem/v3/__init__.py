"""Typed, data-oriented PyFEM core (v3). Requires Python 3.13+."""

from __future__ import annotations

import sys

if sys.version_info < (3, 13):
  msg = "pyfem.v3 requires Python 3.13 or newer"
  raise ImportError(msg)

from pyfem.v3.loader import load_problem
from pyfem.v3.pack import pack_problem
from pyfem.v3.solver.linear import solve_linear
from pyfem.v3.solver.nonlinear import solve_nonlinear
from pyfem.v3.solver.riks import solve_riks
from pyfem.v3.types import F64, I32, LoadedProblem, ProblemDefinition

__all__ = [
  "F64",
  "I32",
  "LoadedProblem",
  "ProblemDefinition",
  "load_problem",
  "pack_problem",
  "solve_linear",
  "solve_nonlinear",
  "solve_riks",
]
