"""Benchmark workload registry: uniform Q8 patches and shipped skim cases."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from bench.common import REPO_ROOT

SKIMS_DIR = REPO_ROOT / "skims"

Q8_SIZES: tuple[int, ...] = (2, 4, 8, 16, 32, 64)
Q8_MATERIALS: tuple[str, ...] = ("PlaneStress", "PlaneStrain")

# Mission tolerance for generated patches and linear skims.
GATE_RTOL = 1.0e-10
GATE_ATOL = 1.0e-12


@dataclass(frozen=True)
class Q8Workload:
  """Uniform Q8 small-strain patch on a structured ``nx x ny`` grid."""

  nx: int
  ny: int
  material_type: str

  @property
  def name(self) -> str:
    return f"q8patch/{self.nx}x{self.ny}/{self.material_type}"

  @property
  def n_elems(self) -> int:
    return self.nx * self.ny

  @property
  def n_dofs(self) -> int:
    n_nodes = (2 * self.nx + 1) * (2 * self.ny + 1) - self.n_elems
    return 2 * n_nodes


def q8_workloads(
  sizes: tuple[int, ...] = Q8_SIZES,
  materials: tuple[str, ...] = Q8_MATERIALS,
) -> list[Q8Workload]:
  return [Q8Workload(n, n, mat) for n in sizes for mat in materials]


@dataclass(frozen=True)
class SkimCase:
  """Shipped skim case: solver kind, input file, and parity tolerances."""

  name: str
  kind: str  # "linear" | "nonlinear" | "riks"
  rtol: float
  atol: float

  @property
  def workload(self) -> str:
    return f"skim/{self.name}"

  @property
  def pro_path(self) -> Path:
    return SKIMS_DIR / self.name / "skim.pro"


LINEAR_SKIM_NAMES = (
  "patch_test8",
  "patch_test3",
  "patch_test4",
  "patch_test8_mpc",
  "patch_test8_plane_strain",
  "patch_test8_loaded",
)
NONLINEAR_SKIM_NAMES = ("patch_test8_nonlinear", "cantilever8")
RIKS_SKIM_NAMES = ("shallow_truss_riks",)


def _parity_tolerances(name: str) -> tuple[float, float]:
  with (SKIMS_DIR / name / "parity.toml").open("rb") as fh:
    data = tomllib.load(fh)
  return float(data["rtol"]), float(data["atol"])


def skim_cases(names: tuple[str, ...] | None = None) -> list[SkimCase]:
  """All registered skim cases, tolerances from their ``parity.toml``."""
  cases: list[SkimCase] = []
  for kind, group in (
    ("linear", LINEAR_SKIM_NAMES),
    ("nonlinear", NONLINEAR_SKIM_NAMES),
    ("riks", RIKS_SKIM_NAMES),
  ):
    for name in group:
      if names is not None and name not in names:
        continue
      rtol, atol = _parity_tolerances(name)
      cases.append(SkimCase(name=name, kind=kind, rtol=rtol, atol=atol))
  return cases
