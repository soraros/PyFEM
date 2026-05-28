"""Minimal legacy ``.pro`` reader for skims (no ``eval``)."""

from __future__ import annotations

import re
from pathlib import Path

from pyfem.v3.io.dat import read_dat_mesh
from pyfem.v3.mesh import build_dof_map
from pyfem.v3.pack import pack_problem
from pyfem.v3.types import LoadedProblem


def read_legacy_pro(path: Path) -> LoadedProblem:
  """Load skim ``.pro`` files that mirror book examples."""
  base = path.parent
  text = path.read_text(encoding="utf-8")

  input_match = re.search(r'input\s*=\s*"([^"]+)"', text)
  if not input_match:
    msg = f"No input = in {path}"
    raise ValueError(msg)

  element_match = re.search(
    r'type\s*=\s*"(\w+)"\s*;\s*\n\s*material\s*=',
    text,
  )
  mat_pat = r'material\s*=\s*\{[^}]*type\s*=\s*"(\w+)"'
  material_type_match = re.search(mat_pat, text, re.S)
  e_match = re.search(r"\bE\s*=\s*([\d.eE+-]+)", text)
  nu_match = re.search(r"\bnu\s*=\s*([\d.eE+-]+)", text)
  solver_match = re.search(r'solver\s*=\s*\{[^}]*type\s*=\s*"(\w+)"', text, re.S)

  if not all([element_match, material_type_match, e_match, nu_match, solver_match]):
    msg = f"Could not parse required blocks in {path}"
    raise ValueError(msg)

  mesh_path = (base / input_match.group(1)).resolve()
  mesh, constraints, ties, loads = read_dat_mesh(mesh_path)
  dof_map = build_dof_map(mesh)

  return LoadedProblem(
    problem=pack_problem(
      mesh,
      dof_map,
      float(e_match.group(1)),
      float(nu_match.group(1)),
      constraints,
      ties,
      loads,
    ),
    name=path.stem,
    element_type=element_match.group(1),
    material_type=material_type_match.group(1),
    solver_type=solver_match.group(1),
    element_group="ContElem",
    mesh_path=mesh_path,
  )
