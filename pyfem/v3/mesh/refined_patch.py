"""Structured Q8 patch-test meshes for scaling benchmarks."""

from __future__ import annotations

import numpy as np

from pyfem.v3.types import LoadedProblem, Mesh, PrescribedDof

PATCH_WIDTH = 0.24
PATCH_HEIGHT = 0.12


def patch_displacement(x: float, y: float) -> tuple[float, float]:
  """Book PatchTest8 prescribed field (examples/ch02/PatchTest8.py)."""
  u = 1.0e-3 * (x + y / 2.0)
  v = 1.0e-3 * (y + x / 2.0)
  return u, v


def build_uniform_q8_patch(
  nx: int,
  ny: int,
  *,
  width: float = PATCH_WIDTH,
  height: float = PATCH_HEIGHT,
) -> tuple[Mesh, tuple[PrescribedDof, ...]]:
  """
  Uniform Q8 grid on ``[0, width] × [0, height]``.

  Serendipity local node order matches :func:`pyfem.v3.fem.shapes.serendipity_quad8`:
  BL, mid-bottom, BR, mid-right, TR, mid-top, TL, mid-left.

  Grid points at both-odd indices ``(ix % 2 == 1 and iy % 2 == 1)`` are omitted
  because serendipity Q8 elements have no bubble node::

      n_nodes = (2 * nx + 1) * (2 * ny + 1) - nx * ny
  """
  if nx < 1 or ny < 1:
    msg = "nx and ny must be at least 1"
    raise ValueError(msg)

  n_nodes_x = 2 * nx + 1
  n_nodes_y = 2 * ny + 1
  grid_to_node: dict[tuple[int, int], int] = {}
  coords_list: list[tuple[float, float]] = []

  for iy in range(n_nodes_y):
    y = iy * height / (2 * ny)
    for ix in range(n_nodes_x):
      if ix % 2 == 1 and iy % 2 == 1:
        continue
      x = ix * width / (2 * nx)
      grid_to_node[(ix, iy)] = len(coords_list)
      coords_list.append((x, y))

  coords = np.asarray(coords_list, dtype=np.float64)
  n_nodes = coords.shape[0]
  n_elems = nx * ny
  conn = np.zeros((n_elems, 8), dtype=np.int32)
  local_nodes = (
    (0, 0),
    (1, 0),
    (2, 0),
    (2, 1),
    (2, 2),
    (1, 2),
    (0, 2),
    (0, 1),
  )
  elem_id = 0
  for ey in range(ny):
    for ex in range(nx):
      ix0 = 2 * ex
      iy0 = 2 * ey
      for local_id, (dx, dy) in enumerate(local_nodes):
        conn[elem_id, local_id] = grid_to_node[(ix0 + dx, iy0 + dy)]
      elem_id += 1

  node_ids = np.arange(n_nodes, dtype=np.int32)
  node_id_to_index = {int(nid): int(nid) for nid in node_ids}
  mesh = Mesh(
    coords=coords,
    conn=conn,
    node_ids=node_ids,
    elem_group_id=np.zeros(n_elems, dtype=np.int32),
    node_id_to_index=node_id_to_index,
  )

  tol = 1.0e-12
  constraints: list[PrescribedDof] = []
  for node_id in node_ids:
    x, y = coords[int(node_id)]
    on_boundary = (
      x <= tol
      or y <= tol
      or x >= width - tol
      or y >= height - tol
    )
    if not on_boundary:
      continue
    u, v = patch_displacement(float(x), float(y))
    constraints.append(PrescribedDof(node_id=int(node_id), dof_type="u", value=u))
    constraints.append(PrescribedDof(node_id=int(node_id), dof_type="v", value=v))

  return mesh, tuple(constraints)


def build_uniform_q8_loaded(
  nx: int,
  ny: int,
  *,
  width: float = PATCH_WIDTH,
  height: float = PATCH_HEIGHT,
  youngs_modulus: float = 1.0e6,
  poisson_ratio: float = 0.25,
  material_type: str = "PlaneStress",
) -> LoadedProblem:
  """Uniform Q8 patch as a :class:`LoadedProblem` for assembly and solve benches."""
  from pyfem.v3.mesh import build_dof_map
  from pyfem.v3.pack import pack_problem

  mesh, constraints = build_uniform_q8_patch(nx, ny, width=width, height=height)
  dof_map = build_dof_map(mesh)
  problem = pack_problem(
    mesh,
    dof_map,
    youngs_modulus,
    poisson_ratio,
    constraints,
    material_type=material_type,
  )
  tag = material_type.removeprefix("Plane").lower()
  return LoadedProblem(
    problem=problem,
    name=f"patch_{nx}x{ny}_{tag}",
    element_type="SmallStrainContinuum",
    material_type=material_type,
    solver_type="LinearSolver",
    element_group="ContElem",
    mesh_path=None,
  )
