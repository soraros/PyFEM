"""Typed load ramping for nonlinear solvers (no ``eval``)."""

from __future__ import annotations

import re

import numpy as np

from pyfem.v3.types import F64, NonlinearSolverSettings

_LOAD_TABLE_RE = re.compile(
  r"loadTable\s*=\s*\[\s*([^\]]+)\s*\]",
  re.IGNORECASE,
)
_FLOAT_RE = re.compile(r"[\d.eE+-]+")


def parse_load_table(pro_text: str) -> F64 | None:
  """Parse ``loadTable = [ ... ]`` from a ``.pro`` solver block."""
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


def validate_load_func(name: str) -> None:
  """Allow only identity ramp ``t`` (legacy default)."""
  if name != "t":
    msg = f"Unsupported loadFunc {name!r}; only 't' is allowed"
    raise ValueError(msg)


def load_factor(step: int, settings: NonlinearSolverSettings) -> tuple[float, float]:
  """
  Return absolute load factor ``lam`` and increment ``dlam`` for 1-based ``step``.

  Matches legacy ``NonlinearSolver.setLoadAndConstraints`` indexing.
  """
  if settings.load_table is not None:
    table = settings.load_table
    if step < 1 or step >= table.shape[0]:
      msg = f"Load step {step} out of range for table of length {table.shape[0]}"
      raise ValueError(msg)
    lam = float(table[step])
    dlam = float(table[step] - table[step - 1])
    return lam, dlam

  time = step * settings.dtime
  time0 = (step - 1) * settings.dtime
  validate_load_func(settings.load_func)
  lam = time
  dlam = time - time0
  return lam, dlam


def n_load_steps(settings: NonlinearSolverSettings) -> int:
  """Number of load steps (legacy ``maxCycle`` when using a table)."""
  if settings.load_table is not None:
    return int(settings.load_table.shape[0] - 1)
  return settings.max_cycle
