"""Typed, data-oriented PyFEM core (v3). Requires Python 3.13+.

Root exports are deliberately limited to the legacy loader, numerical solvers,
and ``ProblemDefinition``. Dtype aliases and packing helpers stay in their
owning modules rather than adding unused facade surface. Generic authored
solves use ``compile.system``, ``constraints``, and ``driver`` directly.
"""

from __future__ import annotations

import sys

if sys.version_info < (3, 13):
  msg = "pyfem.v3 requires Python 3.13 or newer"
  raise ImportError(msg)

from pyfem.v3.loader import load_problem
from pyfem.v3.solver.linear import solve_linear
from pyfem.v3.solver.nonlinear import solve_nonlinear
from pyfem.v3.solver.riks import solve_riks
from pyfem.v3.types import ProblemDefinition

__all__ = [
  "ProblemDefinition",
  "load_problem",
  "solve_linear",
  "solve_nonlinear",
  "solve_riks",
]
