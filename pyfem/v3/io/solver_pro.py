"""Parse nonlinear solver blocks from skim ``.pro`` files."""

from __future__ import annotations

import re

from pyfem.v3.io.load_ramp import parse_load_table, validate_load_func
from pyfem.v3.types import NonlinearSolverSettings

_SOLVER_BLOCK_RE = re.compile(r"solver\s*=\s*\{([^}]*)\}", re.S | re.I)
_FLOAT_FIELD_RE = re.compile(
  r"(?P<name>tol|iterMax|maxCycle|dtime)\s*=\s*(?P<value>[\d.eE+-]+)",
  re.I,
)
_LOAD_FUNC_RE = re.compile(r'loadFunc\s*=\s*"?(\w+)"?', re.I)


def parse_nonlinear_solver_settings(pro_text: str) -> NonlinearSolverSettings | None:
  """Return settings when ``solver.type`` is ``NonlinearSolver``."""
  solver_match = re.search(r'solver\s*=\s*\{[^}]*type\s*=\s*"(\w+)"', pro_text, re.S)
  if not solver_match or solver_match.group(1) != "NonlinearSolver":
    return None

  tol = 1.0e-3
  iter_max = 10
  max_cycle = 5
  dtime = 1.0
  load_func = "t"

  block_match = _SOLVER_BLOCK_RE.search(pro_text)
  block = block_match.group(1) if block_match else pro_text

  for match in _FLOAT_FIELD_RE.finditer(block):
    name = match.group("name").lower()
    value = float(match.group("value"))
    if name == "tol":
      tol = value
    elif name == "itermax":
      iter_max = int(value)
    elif name == "maxcycle":
      max_cycle = int(value)
    elif name == "dtime":
      dtime = value

  load_func_match = _LOAD_FUNC_RE.search(block)
  if load_func_match:
    load_func = load_func_match.group(1)
  validate_load_func(load_func)

  load_table = parse_load_table(pro_text)
  if load_table is not None:
    max_cycle = int(load_table.shape[0] - 1)

  return NonlinearSolverSettings(
    tol=tol,
    iter_max=iter_max,
    max_cycle=max_cycle,
    dtime=dtime,
    load_func=load_func,
    load_table=load_table,
  )
