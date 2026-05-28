"""Load canonical ``problem.toml`` into :class:`LoadedProblem`."""

from __future__ import annotations

import tomllib
from pathlib import Path

from pyfem.v3.io.dat import read_dat_mesh
from pyfem.v3.mesh import build_dof_map
from pyfem.v3.pack import pack_problem
from pyfem.v3.types import LoadedProblem


def read_problem_toml(path: Path) -> LoadedProblem:
  """Read a v3 problem snapshot (paths relative to the TOML file)."""
  base = path.parent
  with path.open("rb") as fh:
    data = tomllib.load(fh)

  mesh_path = (base / data["mesh"]).resolve()
  mesh, constraints, ties, loads = read_dat_mesh(mesh_path)
  dof_map = build_dof_map(mesh)
  material = data["material"]

  return LoadedProblem(
    problem=pack_problem(
      mesh,
      dof_map,
      float(material["E"]),
      float(material["nu"]),
      constraints,
      ties,
      loads,
    ),
    name=str(data.get("name", path.stem)),
    element_type=str(data["element_type"]),
    material_type=str(data["material_type"]),
    solver_type=str(data["solver_type"]),
    element_group=str(data.get("element_group", "ContElem")),
    mesh_path=mesh_path,
  )
