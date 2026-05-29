"""Programmatic shallow-truss fan meshes for structural scale benches."""

from __future__ import annotations

import numpy as np

from pyfem.v3.types import (
  ElementGroupSpec,
  LoadedProblem,
  Mesh,
  NodalLoad,
  PrescribedDof,
  RiksSolverSettings,
)


def build_truss_fan(
  n_rays: int,
  *,
  span: float = 20.0,
  height: float = 0.5,
) -> tuple[
  Mesh,
  tuple[PrescribedDof, ...],
  tuple[NodalLoad, ...],
  tuple[ElementGroupSpec, ...],
]:
  """
  Shallow-truss fan: ``n_rays`` truss members meet at a loaded apex.

  Node layout (``n_rays=2`` matches ch.4 ``ShallowtrussRiks`` topology):

  - anchor at ``(0, 0)`` — spring attachment
  - ``n_rays`` base nodes on ``y=0`` from ``-span/2`` to ``span/2``
  - apex at ``(0, height)``

  Elements: ``n_rays`` truss (group 0) + one spring anchor→apex (group 1).
  """
  if n_rays < 1:
    msg = "n_rays must be at least 1"
    raise ValueError(msg)

  n_nodes = n_rays + 2
  coords = np.zeros((n_nodes, 2), dtype=np.float64)
  coords[0] = (0.0, 0.0)
  for i in range(n_rays):
    x = -span / 2.0 + i * span / max(n_rays - 1, 1)
    coords[i + 1] = (x, 0.0)
  apex = n_rays + 1
  coords[apex] = (0.0, height)

  n_elems = n_rays + 1
  conn = np.zeros((n_elems, 2), dtype=np.int32)
  elem_group_id = np.zeros(n_elems, dtype=np.int32)
  for i in range(n_rays):
    conn[i] = (i + 1, apex)
  conn[n_rays] = (0, apex)
  elem_group_id[n_rays] = 1

  node_ids = np.arange(n_nodes, dtype=np.int32)
  node_id_to_index = {int(nid): int(nid) for nid in node_ids}
  mesh = Mesh(
    coords=coords,
    conn=conn,
    node_ids=node_ids,
    elem_group_id=elem_group_id,
    node_id_to_index=node_id_to_index,
    group_names=("TrussElem", "SpringElem"),
  )

  constraints: list[PrescribedDof] = []
  for node_id in range(n_rays + 1):
    constraints.append(PrescribedDof(node_id=node_id, dof_type="u", value=0.0))
    constraints.append(PrescribedDof(node_id=node_id, dof_type="v", value=0.0))

  loads = (NodalLoad(node_id=apex, dof_type="v", value=-100.0),)
  groups = (
    ElementGroupSpec(name="TrussElem", element_type="Truss", props=(5.0e6, 1.0)),
    ElementGroupSpec(name="SpringElem", element_type="Spring", props=(100.0,)),
  )
  return mesh, tuple(constraints), loads, groups


def build_truss_fan_loaded(
  n_rays: int,
  *,
  span: float = 20.0,
  height: float = 0.5,
  max_lam: float = 10.0,
) -> LoadedProblem:
  """Fan mesh as a :class:`LoadedProblem` for structural assembly and Riks benches."""
  from pyfem.v3.mesh import build_dof_map
  from pyfem.v3.pack import make_loaded, pack_problem

  mesh, constraints, loads, groups = build_truss_fan(
    n_rays,
    span=span,
    height=height,
  )
  dof_map = build_dof_map(mesh)
  problem = pack_problem(
    mesh,
    dof_map,
    0.0,
    0.0,
    constraints=constraints,
    loads=loads,
    groups=groups,
  )
  return make_loaded(
    problem,
    name=f"truss_fan_{n_rays}",
    element_type="Truss",
    material_type="Truss",
    solver_type="RiksSolver",
    element_group="TrussElem",
    riks_settings=RiksSolverSettings(fixed_step=True, max_lam=max_lam),
    groups=groups,
  )
