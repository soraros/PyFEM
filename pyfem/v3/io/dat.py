"""Legacy .dat mesh and constraint reader (literal parsing only)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from pyfem.v3.types import Mesh, MpcTie, NodalLoad, PrescribedDof


def _parse_float(text: str) -> float:
  return float(text.strip())


def _parse_int(text: str) -> int:
  return int(text.strip())


def _parse_dof_lhs(line: str) -> tuple[str, int] | None:
  chunk = line.strip()
  if not chunk or "=" not in chunk:
    return None
  lhs = chunk.split("=", 1)[0]
  if "[" not in lhs:
    return None
  dof_type, node_part = lhs.split("[", 1)
  node_id = _parse_int(node_part.split("]")[0])
  return dof_type.strip(), node_id


def _parse_prescribed_rhs(rhs: str) -> float:
  return _parse_float(rhs.strip().rstrip(";"))


def _parse_mpc_rhs(rhs: str) -> tuple[float, float, str, int]:
  """Parse ``offset + factor * master_dof`` without ``eval``."""
  chunk = rhs.strip().rstrip(";")
  normalized = chunk.replace(" ", "").replace("+", " +").replace("-", " -")
  raw_tokens = [token for token in normalized.split(" ") if token]
  tokens: list[str] = []
  for token in raw_tokens:
    if "*" in token:
      tokens.extend(part for part in token.split("*") if part)
    else:
      tokens.append(token)

  offset = 0.0
  factor = 1.0
  sign = 1.0
  master_type: str | None = None
  master_node: int | None = None

  index = 0
  while index < len(tokens):
    token = tokens[index]
    if token == "+":
      sign = 1.0
      index += 1
      continue
    if token == "-":
      sign = -1.0
      index += 1
      continue

    master_match = re.fullmatch(r"([uvw])\[(\d+)\]", token)
    if master_match is not None:
      master_type = master_match.group(1)
      master_node = int(master_match.group(2))
      factor *= sign
      break

    if index + 1 < len(tokens) and tokens[index + 1] == "*":
      factor = sign * _parse_float(token)
      index += 2
      continue

    if index + 1 < len(tokens) and re.fullmatch(
      r"([uvw])\[(\d+)\]",
      tokens[index + 1],
    ):
      factor = sign * _parse_float(token)
      sign = 1.0
      index += 1
      continue

    offset += sign * _parse_float(token)
    sign = 1.0
    index += 1

  if master_type is None or master_node is None:
    msg = f"No master DOF in MPC tie: {rhs!r}"
    raise ValueError(msg)

  return offset, factor, master_type, master_node


def read_dat_mesh(
  path: Path,
) -> tuple[Mesh, tuple[PrescribedDof, ...], tuple[MpcTie, ...], tuple[NodalLoad, ...]]:
  """Read mesh, constraints, MPC ties, and external forces from a legacy ``.dat``."""
  node_ids: list[int] = []
  coords: list[list[float]] = []
  elements: list[tuple[int, str, list[int]]] = []
  constraints: list[PrescribedDof] = []
  ties: list[MpcTie] = []
  loads: list[NodalLoad] = []

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
      elif line == "<ExternalForces>":
        section = "forces"
      elif line == "</ExternalForces>":
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
        group = parts[1].strip().strip('"').strip("'")
        nodes = [_parse_int(x) for x in parts[2:]]
        elements.append((eid, group, nodes))

    elif section == "constraints":
      for chunk in line.rstrip(";").split(";"):
        chunk = chunk.strip()
        if not chunk:
          continue
        lhs = _parse_dof_lhs(chunk)
        if lhs is None:
          continue
        slave_dof_type, slave_node_id = lhs
        rhs = chunk.split("=", 1)[1]
        if "[" in rhs:
          offset, factor, master_dof_type, master_node_id = _parse_mpc_rhs(rhs)
          ties.append(
            MpcTie(
              slave_node_id=slave_node_id,
              slave_dof_type=slave_dof_type,
              offset=offset,
              master_node_id=master_node_id,
              master_dof_type=master_dof_type,
              factor=factor,
            ),
          )
        else:
          constraints.append(
            PrescribedDof(
              node_id=slave_node_id,
              dof_type=slave_dof_type,
              value=_parse_prescribed_rhs(rhs),
            ),
          )

    elif section == "forces":
      for chunk in line.rstrip(";").split(";"):
        chunk = chunk.strip()
        if not chunk:
          continue
        lhs = _parse_dof_lhs(chunk)
        if lhs is None:
          continue
        dof_type, node_id = lhs
        value = _parse_prescribed_rhs(chunk.split("=", 1)[1])
        loads.append(NodalLoad(node_id=node_id, dof_type=dof_type, value=value))

  if not node_ids:
    msg = f"No nodes found in {path}"
    raise ValueError(msg)

  node_id_to_index = {nid: i for i, nid in enumerate(node_ids)}
  coords_arr = np.asarray(coords, dtype=np.float64)
  rank = coords_arr.shape[1]

  conn = np.zeros((len(elements), len(elements[0][2])), dtype=np.int32)
  elem_group_ids = np.zeros(len(elements), dtype=np.int32)
  group_names: dict[str, int] = {}
  group_name_list: list[str] = []

  for i, (_eid, group, nodes) in enumerate(elements):
    if group not in group_names:
      group_names[group] = len(group_names)
      group_name_list.append(group)
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
    group_names=tuple(group_name_list),
    node_id_to_index=node_id_to_index,
  )
  if mesh.rank != rank:
    msg = "Inconsistent mesh rank"
    raise ValueError(msg)

  return mesh, tuple(constraints), tuple(ties), tuple(loads)
