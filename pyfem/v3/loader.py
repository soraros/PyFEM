"""Unified problem loader."""

from __future__ import annotations

from pathlib import Path

from pyfem.v3.io.legacy_pro import read_legacy_pro
from pyfem.v3.io.toml import read_problem_toml
from pyfem.v3.types import LoadedProblem


def load_problem(path: str | Path) -> LoadedProblem:
  """Load from ``problem.toml`` or a skim ``.pro`` file."""
  resolved = Path(path).resolve()
  if resolved.suffix == ".toml":
    return read_problem_toml(resolved)
  if resolved.suffix == ".pro":
    return read_legacy_pro(resolved)
  msg = f"Unsupported problem file: {resolved}"
  raise ValueError(msg)
