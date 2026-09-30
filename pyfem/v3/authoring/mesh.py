"""Plain-data mesh helpers for the authoring layer.

Meshes are authored as ordinary Python values: a sequence of ``(x, y)`` pairs
(integer node ids ``1..n``) or a mapping of node id to coordinates, and a
sequence of node-id tuples (cell ids ``"cell-1".."cell-k"``) or a mapping of
cell id to its node ids. ``quad8_patch`` generates the uniform serendipity
grids used by teaching patch tests directly.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from pyfem.v3.spec.diagnostics import SourceContext
from pyfem.v3.spec.model import CellBlockSpec, CellSpec, MeshSpec, NodeSpec, SpecId


def _source(label: str) -> SourceContext:
  return SourceContext(source=label)


def _semantic_id(value: object, *, label: str) -> SpecId:
  if type(value) is str and value:
    return value
  if type(value) is int:
    return value
  msg = f"{label} must be a non-empty string or integer id"
  raise TypeError(msg)


def _coordinates(value: object, *, label: str) -> tuple[float, float]:
  if not isinstance(value, Sequence) or isinstance(value, str) or len(value) != 2:
    msg = f"{label} must be an (x, y) coordinate pair"
    raise TypeError(msg)
  components: list[float] = []
  for component in value:
    if type(component) not in (int, float):
      msg = f"{label} components must be exact numbers"
      raise TypeError(msg)
    try:
      converted = float(component)
    except OverflowError:
      msg = f"{label} components must fit finite float64"
      raise ValueError(msg) from None
    if not math.isfinite(converted):
      msg = f"{label} components must be finite"
      raise ValueError(msg)
    components.append(converted)
  return components[0], components[1]


def _node_entries(nodes: object, *, source: str) -> tuple[NodeSpec, ...]:
  if isinstance(nodes, Mapping):
    items = list(nodes.items())
  elif isinstance(nodes, Sequence) and not isinstance(nodes, str):
    items = list(enumerate(nodes, start=1))
  else:
    msg = (
      "mesh nodes must be a sequence of (x, y) pairs or a mapping of "
      "node id to coordinates"
    )
    raise TypeError(msg)
  entries: list[NodeSpec] = []
  for raw_id, raw_point in items:
    node_id = _semantic_id(raw_id, label="mesh node id")
    point = _coordinates(raw_point, label=f"mesh node {node_id!r}")
    entries.append(
      NodeSpec(
        id=node_id,
        coordinates=point,
        source=_source(f"{source}:node:{node_id}"),
      )
    )
  if not entries:
    msg = "a mesh requires at least one node"
    raise ValueError(msg)
  return tuple(entries)


def _cell_entries(
  cells: object,
  *,
  arity: int,
  topology: str,
  source: str,
) -> tuple[CellSpec, ...]:
  if isinstance(cells, Mapping):
    items = list(cells.items())
  elif isinstance(cells, Sequence) and not isinstance(cells, str):
    items = [(f"cell-{index}", cell) for index, cell in enumerate(cells, start=1)]
  else:
    msg = (
      "mesh cells must be a sequence of node-id tuples or a mapping of "
      "cell id to node ids"
    )
    raise TypeError(msg)
  entries: list[CellSpec] = []
  for raw_id, raw_node_ids in items:
    cell_id = _semantic_id(raw_id, label="mesh cell id")
    if not isinstance(raw_node_ids, Sequence) or isinstance(raw_node_ids, str):
      msg = f"mesh cell {cell_id!r} must reference a sequence of node ids"
      raise TypeError(msg)
    node_ids = tuple(
      _semantic_id(node_id, label=f"mesh cell {cell_id!r} node id")
      for node_id in raw_node_ids
    )
    if len(node_ids) != arity:
      msg = (
        f"{topology} cell {cell_id!r} must reference exactly {arity} nodes, "
        f"got {len(node_ids)}"
      )
      raise ValueError(msg)
    entries.append(
      CellSpec(
        id=cell_id,
        node_ids=node_ids,
        source=_source(f"{source}:cell:{cell_id}"),
      )
    )
  if not entries:
    msg = "a mesh requires at least one cell"
    raise ValueError(msg)
  return tuple(entries)


def _mesh(
  nodes: object,
  cells: object,
  *,
  arity: int,
  reference_topology: str,
  topological_dimension: int,
  embedding_dimension: int,
  geometry_interpolation: str,
  block_id: SpecId,
  source: str,
) -> MeshSpec:
  node_entries = _node_entries(nodes, source=source)
  block = CellBlockSpec(
    id=_semantic_id(block_id, label="mesh block id"),
    reference_topology=reference_topology,
    topological_dimension=topological_dimension,
    embedding_dimension=embedding_dimension,
    geometry_interpolation=geometry_interpolation,
    cells=_cell_entries(
      cells, arity=arity, topology=geometry_interpolation, source=source
    ),
    source=_source(f"{source}:block"),
  )
  return MeshSpec(
    nodes=node_entries,
    cell_blocks=(block,),
    source=_source(source),
  )


def quad8_mesh(
  nodes: Sequence[tuple[float, float]] | Mapping[SpecId, tuple[float, float]],
  cells: Sequence[Sequence[SpecId]] | Mapping[SpecId, Sequence[SpecId]],
  *,
  block_id: SpecId = "cells",
  source: str = "authoring.quad8_mesh",
) -> MeshSpec:
  """Build a serendipity-quad8 mesh from plain node coordinates and cells.

  Every cell must reference exactly eight node ids in serendipity local
  order: bottom-left, mid-bottom, bottom-right, mid-right, top-right,
  mid-top, top-left, mid-left.
  """
  return _mesh(
    nodes,
    cells,
    arity=8,
    reference_topology="quadrilateral",
    topological_dimension=2,
    embedding_dimension=2,
    geometry_interpolation="serendipity-quad8",
    block_id=block_id,
    source=source,
  )


def line2_mesh(
  nodes: Sequence[tuple[float, float]] | Mapping[SpecId, tuple[float, float]],
  cells: Sequence[Sequence[SpecId]] | Mapping[SpecId, Sequence[SpecId]],
  *,
  block_id: SpecId = "cells",
  source: str = "authoring.line2_mesh",
) -> MeshSpec:
  """Build a two-node line mesh (truss bars) from plain coordinates and cells."""
  return _mesh(
    nodes,
    cells,
    arity=2,
    reference_topology="line",
    topological_dimension=1,
    embedding_dimension=2,
    geometry_interpolation="line2",
    block_id=block_id,
    source=source,
  )


def _positive_int(value: object, *, label: str) -> int:
  if type(value) is not int or value < 1:
    msg = f"{label} must be a positive exact integer"
    raise ValueError(msg)
  return value


def _positive_float(value: object, *, label: str) -> float:
  if type(value) not in (int, float):
    msg = f"{label} must be an exact number"
    raise TypeError(msg)
  try:
    converted = float(value)
  except OverflowError:
    msg = f"{label} must fit finite float64"
    raise ValueError(msg) from None
  if not math.isfinite(converted) or converted <= 0.0:
    msg = f"{label} must be a positive finite number"
    raise ValueError(msg)
  return converted


def quad8_patch(
  nx: int = 1,
  ny: int = 1,
  *,
  width: float = 1.0,
  height: float = 1.0,
  origin: tuple[float, float] = (0.0, 0.0),
  block_id: SpecId = "cells",
  source: str = "authoring.quad8_patch",
) -> MeshSpec:
  """Generate a uniform ``nx`` by ``ny`` serendipity-quad8 patch.

  Nodes are numbered ``1..n`` row by row over the refined ``(2*nx + 1)`` by
  ``(2*ny + 1)`` grid, skipping cell centers (serendipity elements carry no
  bubble node), so ``n = (2*nx + 1)*(2*ny + 1) - nx*ny``. Cells ``"cell-1"``
  onward run row by row; each references its eight nodes in serendipity local
  order (bottom-left, mid-bottom, bottom-right, mid-right, top-right,
  mid-top, top-left, mid-left).
  """
  columns = _positive_int(nx, label="quad8_patch nx")
  rows = _positive_int(ny, label="quad8_patch ny")
  patch_width = _positive_float(width, label="quad8_patch width")
  patch_height = _positive_float(height, label="quad8_patch height")
  x0, y0 = _coordinates(origin, label="quad8_patch origin")

  grid_ids: dict[tuple[int, int], int] = {}
  nodes: list[tuple[float, float]] = []
  for iy in range(2 * rows + 1):
    y = y0 + iy * patch_height / (2 * rows)
    for ix in range(2 * columns + 1):
      if ix % 2 == 1 and iy % 2 == 1:
        continue
      grid_ids[ix, iy] = len(nodes) + 1
      nodes.append((x0 + ix * patch_width / (2 * columns), y))

  cells: list[tuple[int, ...]] = []
  for row in range(rows):
    for column in range(columns):
      i, j = 2 * column, 2 * row
      cells.append(
        (
          grid_ids[i, j],
          grid_ids[i + 1, j],
          grid_ids[i + 2, j],
          grid_ids[i + 2, j + 1],
          grid_ids[i + 2, j + 2],
          grid_ids[i + 1, j + 2],
          grid_ids[i, j + 2],
          grid_ids[i, j + 1],
        )
      )
  return _mesh(
    nodes,
    cells,
    arity=8,
    reference_topology="quadrilateral",
    topological_dimension=2,
    embedding_dimension=2,
    geometry_interpolation="serendipity-quad8",
    block_id=block_id,
    source=source,
  )
