"""Legacy .dat mesh and constraint reader (literal parsing only)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from pyfem.v3.types import Mesh, PrescribedDof


def _parse_float(text: str) -> float:
  return float(text.strip())


def _parse_int(text: str) -> int:
  return int(text.strip())


def read_dat_mesh(path: Path) -> tuple[Mesh, tuple[PrescribedDof, ...]]:
  """Read ``<Nodes>``, ``<Elements>``, and ``<NodeConstraints>`` sections."""
  node_ids: list[int] = []
  coords: list[list[float]] = []
  elements: list[tuple[int, str, list[int]]] = []
  constraints: list[PrescribedDof] = []

  section: str | None = None
  text = path.read_text(encoding="utf-8")

  for raw_line in text.splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#") or line.startswith("//"):
      continue

    if line.startswith("<") and line.endswith(">"):
      if line == "<Nodes>":
        section = "nodes"
      elif line == "</Nodes>":
        section = None
      elif line == "<Elements>":
        section = "elements"
      elif line == "</Elements>":
        section = None
      elif line.startswith("<NodeConstraints"):
        section = "constraints"
      elif line == "</NodeConstraints>":
        section = None
      else:
        section = None
      continue

    if section == "nodes":
      for chunk in line.rstrip(";").split(";"):
        chunk = chunk.strip()
        if not chunk:
          continue
        parts = re.sub(r"\s{2,}", " ", chunk).split(" ")
        nid = _parse_int(parts[0])
        node_ids.append(nid)
        coords.append([_parse_float(x) for x in parts[1:]])

    elif section == "elements":
      for chunk in line.rstrip(";").split(";"):
        chunk = chunk.strip()
        if not chunk:
          continue
        parts = chunk.split()
        eid = _parse_int(parts[0])
        group = parts[1].strip('"')
        nodes = [_parse_int(x) for x in parts[2:]]
        elements.append((eid, group, nodes))

    elif section == "constraints":
      for chunk in line.rstrip(";").split(";"):
        chunk = chunk.strip()
        if not chunk or "=" not in chunk:
          continue
        lhs, rhs = chunk.split("=", 1)
        dof_type, node_part = lhs.split("[", 1)
        node_id = _parse_int(node_part.split("]")[0])
        value = _parse_float(rhs)
        constraints.append(
          PrescribedDof(
            node_id=node_id,
            dof_type=dof_type.strip(),
            value=value,
          ),
        )

  if not node_ids:
    msg = f"No nodes found in {path}"
    raise ValueError(msg)

  node_id_to_index = {nid: i for i, nid in enumerate(node_ids)}
  coords_arr = np.asarray(coords, dtype=np.float64)
  rank = coords_arr.shape[1]

  conn = np.zeros((len(elements), len(elements[0][2])), dtype=np.int32)
  elem_group_ids = np.zeros(len(elements), dtype=np.int32)
  group_names: dict[str, int] = {}

  for i, (_eid, group, nodes) in enumerate(elements):
    if group not in group_names:
      group_names[group] = len(group_names)
    elem_group_ids[i] = group_names[group]
    conn[i, :] = np.asarray(
      [node_id_to_index[n] for n in nodes],
      dtype=np.int32,
    )

  mesh = Mesh(
    coords=coords_arr,
    conn=conn,
    node_ids=np.asarray(node_ids, dtype=np.int32),
    elem_group_id=elem_group_ids,
    node_id_to_index=node_id_to_index,
  )
  if mesh.rank != rank:
    msg = "Inconsistent mesh rank"
    raise ValueError(msg)

  return mesh, tuple(constraints)
