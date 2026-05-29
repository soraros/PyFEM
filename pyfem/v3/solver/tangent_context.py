"""Cached tangent stiffness for fast internal-force updates (linear elasticity)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix

from pyfem.v3.assembly import assemble_linear_system, assemble_loaded
from pyfem.v3.registry import (
  resolve_element_type,
  resolve_material_type,
  resolve_solver_type,
)
from pyfem.v3.types import F64, LoadedProblem, ProblemDefinition


@dataclass
class TangentAssemblyContext:
  """
  Cached global tangent for repeated ``f_int = K @ u`` updates.

  Valid while mesh, constitutive law, and sparsity pattern are fixed (linear
  small-strain elasticity). Nonlinear materials need full ``assemble_tangent_*``.
  """

  k_csr: csr_matrix
  n_dofs: int

  def internal_force(self, state: F64) -> F64:
    """Return ``K @ state`` using the cached tangent."""
    state = np.ascontiguousarray(state, dtype=np.float64)
    return np.asarray(self.k_csr @ state, dtype=np.float64)

  @classmethod
  def from_loaded(
    cls,
    loaded: LoadedProblem,
    *,
    chunk_size: int | None = None,
  ) -> TangentAssemblyContext:
    resolve_element_type(loaded.element_type)
    resolve_material_type(loaded.material_type)
    system = assemble_loaded(loaded, chunk_size=chunk_size)
    return cls.from_stiffness(system.stiffness.tocsr(), system.n_dofs)

  @classmethod
  def from_problem(
    cls,
    problem: ProblemDefinition,
    *,
    element_type: str,
    material_type: str,
    solver_type: str = "LinearSolver",
    chunk_size: int | None = None,
  ) -> TangentAssemblyContext:
    resolve_element_type(element_type)
    resolve_material_type(material_type)
    resolve_solver_type(solver_type)
    system = assemble_linear_system(
      problem,
      element_type=element_type,
      material_type=material_type,
      chunk_size=chunk_size,
    )
    return cls.from_stiffness(system.stiffness.tocsr(), system.n_dofs)

  @classmethod
  def from_stiffness(cls, k_csr: csr_matrix, n_dofs: int) -> TangentAssemblyContext:
    return cls(k_csr=k_csr, n_dofs=n_dofs)


def prepare_tangent_assembly(
  loaded: LoadedProblem,
  *,
  chunk_size: int | None = None,
) -> TangentAssemblyContext:
  """Assemble and cache tangent stiffness once for repeated internal-force updates."""
  return TangentAssemblyContext.from_loaded(loaded, chunk_size=chunk_size)
