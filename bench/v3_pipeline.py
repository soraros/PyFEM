"""Stage-decomposed v3 pipeline for the uniform Q8 workloads.

Each stage is a separately timed callable over the real v3 code path:

- ``meshgen`` — :func:`build_uniform_q8_patch` (structured grid construction)
- ``load`` — :func:`build_uniform_q8_loaded` (mesh + DOF map + ``pack_problem``);
  this is the load path whose quadratic ``_dof_node_type`` regression class the
  harness exists to catch
- ``kernel`` — batched numba element stiffness (gather of ``coords[conn]`` is
  precomputed here and shows up inside ``assemble_coo`` instead)
- ``scatter`` — COO buffer fill into preallocated arrays
- ``dedup`` — COO -> CSR conversion (duplicate summation)
- ``assemble_coo`` — the public ``assemble_loaded`` (kernel + scatter + COO),
  matching the M3 baseline's "v3 asm" envelope
- ``assemble`` — ``assemble_coo`` plus CSR conversion (what a solve consumes)
- ``constrain`` — constraint matrix build and ``C.T @ K @ C``
- ``factor`` — SuperLU factorization of the reduced system
- ``backsolve`` — one cached-factorization back-solve (repeat-solve cost)
- ``solve`` — single-shot constrained solve from an assembled stiffness
- ``e2e`` — the public ``prepare_cached_linear(...).solve()`` end to end
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_array
from scipy.sparse.linalg import factorized

from bench.workloads import Q8Workload
from pyfem.v3._prototype_assembly import assemble_loaded
from pyfem.v3.fem.assembly import _fill_stiffness_coo
from pyfem.v3.fem.element import continuum_stiffness_batched
from pyfem.v3.mesh.refined_patch import (
  build_uniform_q8_loaded,
  build_uniform_q8_patch,
)
from pyfem.v3.solver.constraints import build_prescribed_constraints
from pyfem.v3.solver.context import CachedLinearSystem, prepare_cached_linear
from pyfem.v3.types import LinearSystem, LoadedProblem

ENTRIES_PER_Q8_ELEM = 16 * 16


class V3Q8Pipeline:
  """Prebuilt inputs and per-stage callables for one uniform Q8 workload."""

  def __init__(self, workload: Q8Workload, loaded: LoadedProblem | None = None) -> None:
    self.workload = workload
    self.loaded = loaded or build_uniform_q8_loaded(
      workload.nx,
      workload.ny,
      material_type=workload.material_type,
    )
    problem = self.loaded.problem
    self.problem = problem

    n_elems = problem.n_elems
    n_dof = 2 * int(problem.conn.shape[1])
    self.nodal_coords = problem.coords[problem.conn]
    self.element_dofs = problem.global_dofs[problem.conn].reshape(n_elems, n_dof)
    self.ke = continuum_stiffness_batched(self.nodal_coords, problem.constitutive)

    nnz = n_elems * n_dof * n_dof
    self.row = np.empty(nnz, dtype=np.int32)
    self.col = np.empty(nnz, dtype=np.int32)
    self.val = np.empty(nnz, dtype=np.float64)
    _fill_stiffness_coo(self.element_dofs, self.ke, self.row, self.col, self.val)

    n_dofs = problem.n_dofs
    self.k_coo = coo_array(
      (self.val, (self.row, self.col)),
      shape=(n_dofs, n_dofs),
    )
    self.k_csr = self.k_coo.tocsr()
    self.constraints = build_prescribed_constraints(problem)
    self.k_red = (self.constraints.C.T @ (self.k_csr @ self.constraints.C)).tocsr()
    self.solve_red = factorized(self.k_red)

    a = np.zeros(n_dofs, dtype=np.float64)
    a += self.constraints.prescribed
    rhs = problem.external_load - self.k_csr @ a
    self.b_red = self.constraints.C.T @ rhs
    self.system = LinearSystem(
      stiffness=self.k_coo,
      load=np.ascontiguousarray(problem.external_load, dtype=np.float64),
      n_dofs=n_dofs,
    )

  @property
  def logical_bytes(self) -> dict[str, int]:
    """Logical sizes of the big arrays (D2 §7.5: RSS plus logical sizes)."""
    return {
      "ke_batch": int(self.ke.nbytes),
      "coo": int(self.row.nbytes + self.col.nbytes + self.val.nbytes),
      "csr": int(
        self.k_csr.data.nbytes + self.k_csr.indices.nbytes + self.k_csr.indptr.nbytes
      ),
      "nodal_coords": int(self.nodal_coords.nbytes),
    }

  def meshgen(self) -> None:
    build_uniform_q8_patch(self.workload.nx, self.workload.ny)

  def load(self) -> None:
    build_uniform_q8_loaded(
      self.workload.nx,
      self.workload.ny,
      material_type=self.workload.material_type,
    )

  def kernel(self) -> None:
    continuum_stiffness_batched(self.nodal_coords, self.problem.constitutive)

  def scatter(self) -> None:
    _fill_stiffness_coo(self.element_dofs, self.ke, self.row, self.col, self.val)

  def dedup(self) -> None:
    coo_array(
      (self.val, (self.row, self.col)),
      shape=self.k_coo.shape,
    ).tocsr()

  def assemble_coo(self) -> None:
    assemble_loaded(self.loaded)

  def assemble(self) -> None:
    assemble_loaded(self.loaded).stiffness.tocsr()

  def constrain(self) -> None:
    constraints = build_prescribed_constraints(self.problem)
    (constraints.C.T @ (self.k_csr @ constraints.C)).tocsr()

  def factor(self) -> None:
    factorized(self.k_red)

  def backsolve(self) -> None:
    self.solve_red(self.b_red)

  def solve(self) -> None:
    CachedLinearSystem.from_system(self.problem, self.system).solve()

  def e2e(self) -> None:
    prepare_cached_linear(self.loaded).solve()
