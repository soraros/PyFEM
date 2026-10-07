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


@dataclass(frozen=True)
class MaterialKernelCase:
  """Documented stateful-kernel batch workload (M30: reference vs optimized).

  ``n_entities`` is the flattened integration-point count of the batch; the
  documented size 36864 is the 64x64 Q8 patch (4096 elements x 9 Gauss
  points), the largest gated patch's material-kernel call size.
  """

  name: str
  n_entities: int

  @property
  def workload(self) -> str:
    return f"material/{self.name}/{self.n_entities}"


MATERIAL_KERNEL_CASES: tuple[MaterialKernelCase, ...] = (
  MaterialKernelCase("j2-isotropic-hardening", 36864),
)


def material_cases() -> list[MaterialKernelCase]:
  """All registered material-kernel batch workloads."""
  return list(MATERIAL_KERNEL_CASES)


@dataclass(frozen=True)
class FiniteStrainWorkload:
  """Refined cantilever8 strip through the finite-strain TL deck stack.

  The generated deck is the shipped ``skims/cantilever8`` case refined to
  ``nx x ny`` serendipity-quad8 cells over the same 8.0 x 0.5 strip: same
  clamped left edge, same 0.01 tip load, same NonlinearSolver ramp. The v3
  side drives the landed finite-strain family (deck conversion + compilation
  + ``NonlinearStaticDriver``), not the prototype solver the skim times; the
  gate is legacy parity per the landed F4 oracle (final state plus tangent
  and internal force at the converged state), within
  ``skims/cantilever8/parity.toml``.
  """

  nx: int
  ny: int
  rtol: float
  atol: float

  @property
  def name(self) -> str:
    return f"tl-cantilever/{self.nx}x{self.ny}"

  @property
  def n_elems(self) -> int:
    return self.nx * self.ny

  @property
  def n_nodes(self) -> int:
    return (self.ny + 1) * (2 * self.nx + 1) + self.ny * (self.nx + 1)

  @property
  def n_dofs(self) -> int:
    return 2 * self.n_nodes


# Documented family sizes: 2x and 4x refinements of the shipped 8x1 skim in
# each direction. The 32x4 legacy side is ~24 s per nonlinear run on the
# reference machine (16x the skim's ~1.5 s), inside the harness's legacy
# budget at the standard 3-rep nonlinear envelope.
FINITE_STRAIN_SIZES: tuple[tuple[int, int], ...] = ((16, 2), (32, 4))


def finite_strain_cases() -> list[FiniteStrainWorkload]:
  """All registered finite-strain TL family workloads (cantilever8-derived)."""
  rtol, atol = _parity_tolerances("cantilever8")
  return [FiniteStrainWorkload(nx, ny, rtol, atol) for nx, ny in FINITE_STRAIN_SIZES]


@dataclass(frozen=True)
class RiksFanWorkload:
  """Truss-only shallow-truss fan through the landed Riks arc-length driver.

  ``n_rays`` truss members from fully constrained base nodes on
  ``[-span/2, span/2]`` (span 20) to a loaded apex (0, 0.5, v = -100);
  ``n_rays=2`` reproduces the ch.4 ShallowtrussRiks truss-only geometry of
  the landed M33 oracle. The fan drops the skim's spring on both sides
  exactly like that oracle — the point-spring family is declaration-routed
  and not ModelSpec-expressible. The gate is legacy parity per the oracle:
  exact cycle-count equality and per-cycle (lam, state, correction-count)
  parity within ``skims/shallow_truss_riks/parity.toml``.
  """

  n_rays: int
  rtol: float
  atol: float

  @property
  def name(self) -> str:
    return f"riks-fan/{self.n_rays}"

  @property
  def n_elems(self) -> int:
    return self.n_rays

  @property
  def n_dofs(self) -> int:
    return 2 * (self.n_rays + 1)


# Documented fan sizes, mirroring the landed structural scale bench
# (test/v3/_bench_structural_scale.py): 8/32/128 rays.
RIKS_FAN_RAYS: tuple[int, ...] = (8, 32, 128)


def riks_fan_cases() -> list[RiksFanWorkload]:
  """All registered Riks arc-length family workloads (shallow-truss-derived)."""
  rtol, atol = _parity_tolerances("shallow_truss_riks")
  return [RiksFanWorkload(n, rtol, atol) for n in RIKS_FAN_RAYS]
