"""Plane-strain Q8 scale benchmark — delegates to ``_bench_solve_scale``."""

from __future__ import annotations

from _bench_solve_scale import run_material

if __name__ == "__main__":
  run_material("PlaneStrain")
