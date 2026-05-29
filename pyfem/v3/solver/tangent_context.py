"""Backward-compatible re-exports for tangent assembly caching."""

from pyfem.v3.solver.context import (
  CachedLinearSystem,
  TangentAssemblyContext,
  prepare_cached_linear,
  prepare_tangent_assembly,
)

__all__ = [
  "CachedLinearSystem",
  "TangentAssemblyContext",
  "prepare_cached_linear",
  "prepare_tangent_assembly",
]
