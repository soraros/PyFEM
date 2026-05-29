"""Parse solver blocks from skim ``.pro`` files."""

from __future__ import annotations

import re

from pyfem.v3.io.load_ramp import parse_load_table, validate_load_func
from pyfem.v3.types import NonlinearSolverSettings, RiksSolverSettings

_SOLVER_BLOCK_RE = re.compile(r"solver\s*=\s*\{([^}]*)\}", re.S | re.I)
_FLOAT_FIELD_RE = re.compile(
  r"(?P<name>tol|iterMax|maxCycle|dtime|optiter|maxLam|maxFactor)\s*=\s*(?P<value>[\d.eE+-]+)",
  re.I,
)
_LOAD_FUNC_RE = re.compile(r'loadFunc\s*=\s*"?(\w+)"?', re.I)
_BOOL_FIELD_RE = re.compile(r"fixedStep\s*=\s*(true|false)", re.I)


def _solver_type(pro_text: str) -> str | None:
  pattern = r'solver\s*=\s*\{[^}]*type\s*=\s*["\']?(\w+)'
  solver_match = re.search(pattern, pro_text, re.S)
  if not solver_match:
    return None
  return solver_match.group(1)


def _solver_block(pro_text: str) -> str:
  block_match = _SOLVER_BLOCK_RE.search(pro_text)
  return block_match.group(1) if block_match else pro_text


def _parse_float_fields(block: str) -> dict[str, float]:
  """Extract numeric solver fields shared by Nonlinear and Riks blocks."""
  fields: dict[str, float] = {}
  for match in _FLOAT_FIELD_RE.finditer(block):
    fields[match.group("name").lower()] = float(match.group("value"))
  return fields


def parse_nonlinear_solver_settings(pro_text: str) -> NonlinearSolverSettings | None:
  """Return settings when ``solver.type`` is ``NonlinearSolver``."""
  if _solver_type(pro_text) != "NonlinearSolver":
    return None

  fields = _parse_float_fields(_solver_block(pro_text))
  tol = fields.get("tol", 1.0e-3)
  iter_max = int(fields.get("itermax", 10))
  max_cycle = int(fields.get("maxcycle", 5))
  dtime = fields.get("dtime", 1.0)

  block = _solver_block(pro_text)
  load_func = "t"
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


def parse_riks_solver_settings(pro_text: str) -> RiksSolverSettings | None:
  """Return settings when ``solver.type`` is ``RiksSolver``."""
  if _solver_type(pro_text) != "RiksSolver":
    return None

  fields = _parse_float_fields(_solver_block(pro_text))
  block = _solver_block(pro_text)
  fixed_step = False
  bool_match = _BOOL_FIELD_RE.search(block)
  if bool_match:
    fixed_step = bool_match.group(1).lower() == "true"

  return RiksSolverSettings(
    tol=fields.get("tol", 1.0e-5),
    iter_max=int(fields.get("itermax", 10)),
    opt_iter=int(fields.get("optiter", 5)),
    fixed_step=fixed_step,
    max_lam=fields.get("maxlam", 1.0e20),
    max_factor=fields.get("maxfactor", 1.0e20),
  )
