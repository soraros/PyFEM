"""Minimal legacy ``.pro`` reader for skims (no ``eval``)."""

from __future__ import annotations

import re
from pathlib import Path

from pyfem.v3.io.dat import read_dat_mesh
from pyfem.v3.io.solver_pro import (
  parse_nonlinear_solver_settings,
  parse_riks_solver_settings,
)
from pyfem.v3.mesh import build_dof_map
from pyfem.v3.pack import make_loaded, pack_problem
from pyfem.v3.types import ElementGroupSpec, LoadedProblem

_STRUCTURAL_TYPES = frozenset({"Truss", "Spring"})
_FLOAT_RE = re.compile(r"[\d.eE+-]+")


def _parse_pro_blocks(text: str) -> dict[str, str]:
  """Return top-level ``Name -> block body`` for element property blocks."""
  blocks: dict[str, str] = {}
  skip = frozenset({"solver", "output", "graph", "logger", "input", "outputModules"})
  for match in re.finditer(r"(\w+)\s*=\s*\{", text):
    name = match.group(1)
    if name in skip or name.startswith("output"):
      continue
    start = match.end()
    depth = 1
    index = start
    while index < len(text) and depth > 0:
      char = text[index]
      if char == "{":
        depth += 1
      elif char == "}":
        depth -= 1
      index += 1
    blocks[name] = text[start : index - 1]
  return blocks


def _block_type(body: str) -> str | None:
  match = re.search(r'type\s*=\s*"(\w+)"', body)
  return match.group(1) if match else None


def _block_float(body: str, name: str) -> float | None:
  match = re.search(rf"\b{name}\s*=\s*([\d.eE+-]+)", body)
  return float(match.group(1)) if match else None


def _parse_element_group(name: str, body: str) -> ElementGroupSpec:
  element_type = _block_type(body)
  if element_type is None:
    msg = f"No type in element block {name!r}"
    raise ValueError(msg)

  if element_type == "Truss":
    e_val = _block_float(body, "E")
    area = _block_float(body, "Area")
    if e_val is None or area is None:
      msg = f"Truss block {name!r} requires E and Area"
      raise ValueError(msg)
    return ElementGroupSpec(name=name, element_type=element_type, props=(e_val, area))

  if element_type == "Spring":
    k_val = _block_float(body, "k")
    if k_val is None:
      msg = f"Spring block {name!r} requires k"
      raise ValueError(msg)
    return ElementGroupSpec(name=name, element_type=element_type, props=(k_val,))

  mat_pat = r'material\s*=\s*\{[^}]*type\s*=\s*"(\w+)"'
  material_match = re.search(mat_pat, body, re.S)
  e_match = re.search(r"\bE\s*=\s*([\d.eE+-]+)", body)
  nu_match = re.search(r"\bnu\s*=\s*([\d.eE+-]+)", body)
  if not all([material_match, e_match, nu_match]):
    msg = f"Could not parse continuum block {name!r}"
    raise ValueError(msg)
  return ElementGroupSpec(
    name=name,
    element_type=element_type,
    props=(float(e_match.group(1)), float(nu_match.group(1))),
  )


def _ordered_groups(
  mesh_group_names: tuple[str, ...],
  blocks: dict[str, str],
) -> tuple[ElementGroupSpec, ...]:
  groups: list[ElementGroupSpec] = []
  for group_name in mesh_group_names:
    if group_name not in blocks:
      msg = f"Missing .pro block for mesh group {group_name!r}"
      raise ValueError(msg)
    groups.append(_parse_element_group(group_name, blocks[group_name]))
  return tuple(groups)


def read_legacy_pro(path: Path) -> LoadedProblem:
  """Load skim ``.pro`` files that mirror book examples."""
  base = path.parent
  text = path.read_text(encoding="utf-8")

  input_match = re.search(r'input\s*=\s*"([^"]+)"', text)
  if not input_match:
    msg = f"No input = in {path}"
    raise ValueError(msg)

  solver_match = re.search(
    r'solver\s*=\s*\{[^}]*type\s*=\s*["\']?(\w+)',
    text,
    re.S,
  )
  if not solver_match:
    msg = f"No solver block in {path}"
    raise ValueError(msg)
  solver_type = solver_match.group(1)

  mesh_path = (base / input_match.group(1)).resolve()
  mesh, constraints, ties, loads = read_dat_mesh(mesh_path)
  dof_map = build_dof_map(mesh)
  blocks = _parse_pro_blocks(text)
  groups = _ordered_groups(mesh.group_names, blocks)

  if any(g.element_type in _STRUCTURAL_TYPES for g in groups):
    first = groups[0]
    return make_loaded(
      pack_problem(
        mesh,
        dof_map,
        0.0,
        0.0,
        constraints=constraints,
        ties=ties,
        loads=loads,
        groups=groups,
      ),
      name=path.stem,
      element_type=first.element_type,
      material_type=first.element_type,
      solver_type=solver_type,
      element_group=first.name,
      mesh_path=mesh_path,
      riks_settings=parse_riks_solver_settings(text),
      groups=groups,
    )

  continuum = groups[0]
  mat_pat = r'material\s*=\s*\{[^}]*type\s*=\s*"(\w+)"'
  material_type_match = re.search(mat_pat, blocks[continuum.name], re.S)
  if material_type_match is None:
    msg = f"No material type in {continuum.name}"
    raise ValueError(msg)

  return make_loaded(
    pack_problem(
      mesh,
      dof_map,
      continuum.props[0],
      continuum.props[1],
      material_type=material_type_match.group(1),
      constraints=constraints,
      ties=ties,
      loads=loads,
    ),
    name=path.stem,
    element_type=continuum.element_type,
    material_type=material_type_match.group(1),
    solver_type=solver_type,
    element_group=continuum.name,
    mesh_path=mesh_path,
    nonlinear_settings=parse_nonlinear_solver_settings(text),
    groups=groups,
  )
