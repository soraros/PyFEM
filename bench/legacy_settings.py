"""Numba-free replica of the v3 skim solver-settings parsers.

The legacy benchmark drivers must apply the same solver-attribute overrides
as ``test/v3/_legacy_parity.py`` (legacy solvers do not read these fields from
the solver block natively), but importing ``pyfem.v3.io.solver_pro`` would pull
numba into legacy cold processes. This module reproduces the exact parsing
semantics with stdlib + numpy only; the correctness gates protect it, since a
divergence from the v3 parse shows up as a parity failure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

_SOLVER_BLOCK_RE = re.compile(r"solver\s*=\s*\{([^}]*)\}", re.S | re.I)
_FLOAT_FIELD_RE = re.compile(
  r"(?P<name>tol|iterMax|maxCycle|dtime|optiter|maxLam|maxFactor)\s*=\s*(?P<value>[\d.eE+-]+)",
  re.I,
)
_LOAD_FUNC_RE = re.compile(r'loadFunc\s*=\s*"?(\w+)"?', re.I)
_BOOL_FIELD_RE = re.compile(r"fixedStep\s*=\s*(true|false)", re.I)
_LOAD_TABLE_RE = re.compile(r"loadTable\s*=\s*\[\s*([^\]]+)\s*\]", re.IGNORECASE)
_FLOAT_RE = re.compile(r"[\d.eE+-]+")
_SOLVER_TYPE_RE = re.compile(r'solver\s*=\s*\{[^}]*type\s*=\s*["\']?(\w+)', re.S)


@dataclass(frozen=True)
class LegacyNonlinearSettings:
  tol: float = 1.0e-3
  iter_max: int = 10
  max_cycle: int = 5
  dtime: float = 1.0
  load_func: str = "t"
  load_table: np.ndarray | None = field(default=None, compare=False)


@dataclass(frozen=True)
class LegacyRiksSettings:
  tol: float = 1.0e-5
  iter_max: int = 10
  opt_iter: int = 5
  fixed_step: bool = False
  max_lam: float = 1.0e20
  max_factor: float = 1.0e20


def _solver_type(pro_text: str) -> str | None:
  match = _SOLVER_TYPE_RE.search(pro_text)
  return match.group(1) if match else None


def _solver_block(pro_text: str) -> str:
  match = _SOLVER_BLOCK_RE.search(pro_text)
  return match.group(1) if match else pro_text


def _float_fields(block: str) -> dict[str, float]:
  return {
    match.group("name").lower(): float(match.group("value"))
    for match in _FLOAT_FIELD_RE.finditer(block)
  }


def _load_table(pro_text: str) -> np.ndarray | None:
  match = _LOAD_TABLE_RE.search(pro_text)
  if not match:
    return None
  values = [float(token) for token in _FLOAT_RE.findall(match.group(1))]
  if not values:
    msg = "loadTable is empty"
    raise ValueError(msg)
  table = np.zeros(len(values) + 1, dtype=np.float64)
  table[1:] = values
  return table


def nonlinear_settings(pro_text: str) -> LegacyNonlinearSettings | None:
  """Mirror of ``parse_nonlinear_solver_settings`` (defaults included)."""
  if _solver_type(pro_text) != "NonlinearSolver":
    return None
  block = _solver_block(pro_text)
  fields = _float_fields(block)
  load_func_match = _LOAD_FUNC_RE.search(block)
  load_func = load_func_match.group(1) if load_func_match else "t"
  if load_func != "t":
    msg = f"Unsupported loadFunc {load_func!r}; only 't' is allowed"
    raise ValueError(msg)
  load_table = _load_table(pro_text)
  max_cycle = int(fields.get("maxcycle", 5))
  if load_table is not None:
    max_cycle = int(load_table.shape[0] - 1)
  return LegacyNonlinearSettings(
    tol=fields.get("tol", 1.0e-3),
    iter_max=int(fields.get("itermax", 10)),
    max_cycle=max_cycle,
    dtime=fields.get("dtime", 1.0),
    load_func=load_func,
    load_table=load_table,
  )


def riks_settings(pro_text: str) -> LegacyRiksSettings | None:
  """Mirror of ``parse_riks_solver_settings`` (defaults included)."""
  if _solver_type(pro_text) != "RiksSolver":
    return None
  block = _solver_block(pro_text)
  fields = _float_fields(block)
  fixed_step = False
  bool_match = _BOOL_FIELD_RE.search(block)
  if bool_match:
    fixed_step = bool_match.group(1).lower() == "true"
  return LegacyRiksSettings(
    tol=fields.get("tol", 1.0e-5),
    iter_max=int(fields.get("itermax", 10)),
    opt_iter=int(fields.get("optiter", 5)),
    fixed_step=fixed_step,
    max_lam=fields.get("maxlam", 1.0e20),
    max_factor=fields.get("maxfactor", 1.0e20),
  )
