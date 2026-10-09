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

A stateful law authors exactly like an elastic one::

    model = small_strain_continuum(
      mesh,
      material=plasticity(210.0e3, 0.3, 250.0, 1000.0),
    )
    system = compile(model)  # the plasticity reference registry is the default

A stateful workflow adds a spring kernel with state slots and steps it through
the landed driver — transactions stay begin/stage/commit plain, never touching
owner internals::

    system = spring(
      compile(model),
      nodes={"tip": 5},
      kernel=my_kernel,
      state=(("max_extension", 1),),
      parameters=(3.0, 0.01),
      name="memory-spring",
      implementation_id="my-spring-v1",
    )
    session = nonlinear_static(
      system,
      constraints=fixed(nodes=(1, 2)),
      loads=(nodal_load(5, "x", 100.0),),
    )
    result = session.run({"load": 0.0}, {"load": 0.5}, {"load": 1.0})
    rows = session.accepted_state("springs")
    snapshot = session.snapshot()  # byte-exact rollback reference

A sensitivity study names qualified material parameters at run time and reads
the typed per-parameter columns off the committed records::

    sensed = session.run({"load": 0.0}, {"load": 1.0},
                         sensitivities=("initial_yield_stress",))
    column = sensed.records[-1].observation.sensitivities[0].coefficients
"""

from pyfem.v3.authoring.compile import compile
from pyfem.v3.authoring.evaluate import evaluate, trial_vector
from pyfem.v3.authoring.materials import (
  damage,
  linear_elastic,
  plasticity,
  prony_viscoelasticity,
  uniaxial_elastic,
)
from pyfem.v3.authoring.mesh import line2_mesh, quad8_mesh, quad8_patch
from pyfem.v3.authoring.models import small_strain_continuum, truss
from pyfem.v3.authoring.program import fixed, nodal_load
from pyfem.v3.authoring.registry import (
  check_registry,
  damage_law,
  damage_registry,
  plane_stress_law,
  plasticity_law,
  plasticity_registry,
  prony_viscoelasticity_law,
  q8_registry,
  truss_registry,
  uniaxial_law,
  viscoelasticity_registry,
)
from pyfem.v3.authoring.springs import damage_envelope_spring, spring
from pyfem.v3.authoring.transactions import (
  NonlinearStaticSession,
  StateOwner,
  StateSnapshot,
  StateTrial,
  nonlinear_static,
  state_owner,
)
from pyfem.v3.compile.spring import SpringKernelResult
from pyfem.v3.driver import (
  DriverStatus,
  NonlinearStaticResult,
  NonlinearStaticSettings,
  ParameterSensitivityObservation,
  SubstepStatus,
)
from pyfem.v3.model.operator import EvaluationStatus

__all__ = [
  "DriverStatus",
  "EvaluationStatus",
  "NonlinearStaticResult",
  "NonlinearStaticSession",
  "NonlinearStaticSettings",
  "ParameterSensitivityObservation",
  "SpringKernelResult",
  "StateOwner",
  "StateSnapshot",
  "StateTrial",
  "SubstepStatus",
  "check_registry",
  "compile",
  "damage",
  "damage_envelope_spring",
  "damage_law",
  "damage_registry",
  "evaluate",
  "fixed",
  "line2_mesh",
  "linear_elastic",
  "nodal_load",
  "nonlinear_static",
  "plane_stress_law",
  "plasticity",
  "plasticity_law",
  "plasticity_registry",
  "prony_viscoelasticity",
  "prony_viscoelasticity_law",
  "q8_registry",
  "quad8_mesh",
  "quad8_patch",
  "small_strain_continuum",
  "spring",
  "state_owner",
  "trial_vector",
  "truss",
  "truss_registry",
  "uniaxial_elastic",
  "uniaxial_law",
  "viscoelasticity_registry",
]
