"""Thin plain-Python authoring layer over the in-tree v3 builders.

The public extension surface: users write plain functions and values — meshes
as coordinate lists, constitutive laws as callables, springs as batched
kernels — and the layer composes them into exactly the ``ModelSpec`` slices
and registry descriptors the landed builders consume. Nothing here replaces
the builders' coded validation; the layer adds authoring-time ergonomics and
field-level metadata-mismatch diffs on top of it.

A first continuum model is one readable call per stage::

    mesh = quad8_patch(2, 2, width=0.24, height=0.12)
    model = small_strain_continuum(mesh, material=linear_elastic(70.0e9, 0.33))
    system = compile(model)

A new material is a plain law function plus one descriptor line::

    def my_law(youngs_modulus, poisson_ratio):
      ...

    law = plane_stress_law(my_law, implementation_id="my-law-v1")
    system = compile(model, q8_registry(material=law))
"""

from pyfem.v3.authoring.compile import compile
from pyfem.v3.authoring.evaluate import evaluate, trial_vector
from pyfem.v3.authoring.materials import linear_elastic, uniaxial_elastic
from pyfem.v3.authoring.mesh import line2_mesh, quad8_mesh, quad8_patch
from pyfem.v3.authoring.models import small_strain_continuum, truss
from pyfem.v3.authoring.registry import (
  check_registry,
  plane_stress_law,
  q8_registry,
  truss_registry,
  uniaxial_law,
)
from pyfem.v3.authoring.springs import damage_envelope_spring, spring
from pyfem.v3.compile.spring import SpringKernelResult
from pyfem.v3.model.operator import EvaluationStatus

__all__ = [
  "EvaluationStatus",
  "SpringKernelResult",
  "check_registry",
  "compile",
  "damage_envelope_spring",
  "evaluate",
  "line2_mesh",
  "linear_elastic",
  "plane_stress_law",
  "q8_registry",
  "quad8_mesh",
  "quad8_patch",
  "small_strain_continuum",
  "spring",
  "trial_vector",
  "truss",
  "truss_registry",
  "uniaxial_elastic",
  "uniaxial_law",
]
