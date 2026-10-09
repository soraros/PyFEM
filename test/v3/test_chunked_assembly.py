# SPDX-License-Identifier: MIT

"""Chunked vs monolithic COO assembly."""

from __future__ import annotations

import sys

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from pyfem.v3._prototype_assembly import assemble_linear_system, canonical_csr
from pyfem.v3.mesh import build_dof_map
from pyfem.v3.mesh.refined_patch import build_uniform_q8_patch
from pyfem.v3.pack import pack_problem


def test_chunked_matches_monolithic() -> None:
  mesh, constraints = build_uniform_q8_patch(8, 8)
  dof_map = build_dof_map(mesh)
  problem = pack_problem(mesh, dof_map, 1.0e6, 0.25, constraints)
  mono = assemble_linear_system(
    problem,
    element_type="SmallStrainContinuum",
    material_type="PlaneStress",
    chunk_size=10_000,
  )
  chunked = assemble_linear_system(
    problem,
    element_type="SmallStrainContinuum",
    material_type="PlaneStress",
    chunk_size=16,
  )
  k_mono = mono.stiffness.toarray()
  k_chunk = chunked.stiffness.toarray()
  np.testing.assert_allclose(k_mono, k_chunk, rtol=0, atol=1e-9)


def test_chunked_canonical_csr_matches_monolithic_bitwise() -> None:
  """Chunked and monolithic streams produce bit-identical canonical CSR.

  For one continuum group the chunked concatenation reproduces the
  monolithic element-major stream exactly, so the pattern cache serves
  both chunkings the same compiled topology and the strict-sequential
  dedup yields raw-uint64-identical CSR data.
  """
  mesh, constraints = build_uniform_q8_patch(8, 8)
  dof_map = build_dof_map(mesh)
  problem = pack_problem(mesh, dof_map, 1.0e6, 0.25, constraints)
  mono = assemble_linear_system(
    problem,
    element_type="SmallStrainContinuum",
    material_type="PlaneStress",
    chunk_size=10_000,
  )
  chunked = assemble_linear_system(
    problem,
    element_type="SmallStrainContinuum",
    material_type="PlaneStress",
    chunk_size=16,
  )
  csr_mono = canonical_csr(mono.stiffness)
  csr_chunked = canonical_csr(chunked.stiffness)
  np.testing.assert_array_equal(csr_mono.indptr, csr_chunked.indptr)
  np.testing.assert_array_equal(csr_mono.indices, csr_chunked.indices)
  np.testing.assert_array_equal(
    csr_mono.data.view(np.uint64), csr_chunked.data.view(np.uint64)
  )
