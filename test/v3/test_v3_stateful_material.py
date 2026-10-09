# SPDX-License-Identifier: MIT

"""v2 stateful material descriptor ABI: schema, emission, initial state, owner.

The parameterized-width mock law in this module is the ABI future-proofing
witness: it compiles through the same generic stateful path as the shipped
isotropic-hardening law with no compiler changes, its slot widths resolve from
spec parameters (the ViscoElasticity ``6 * nterms + 7`` shape), and its nonzero
initial state rows reach the transaction owner unchanged.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
import pytest

if sys.version_info < (3, 13):
  pytest.skip("pyfem.v3 requires Python 3.13+", allow_module_level=True)

from pyfem.v3.compile.continuum import (
  PLASTIC_MATERIAL_KEY,
  Q8_MATERIAL_KEY,
  plasticity_reference_registry,
  q8_reference_registry,
)
from pyfem.v3.compile.contracts import (
  STATEFUL_MATERIAL_DESCRIPTOR_SCHEMA,
  StatefulContinuumKernelResult,
  build_material_state_layout,
  material_state_layout_schema,
  resolve_material_state_slots,
  stateful_tangent_channel_flags,
  validate_stateful_material_metadata,
)
from pyfem.v3.compile.diagnostics import ModelCompilationError
from pyfem.v3.compile.system import compile_system
from pyfem.v3.materials.isotropic_hardening_plasticity import (
  ISOTROPIC_HARDENING_PLASTICITY_BINDING,
  isotropic_hardening_plasticity_metadata,
)
from pyfem.v3.model.arrays import FinalizedArray
from pyfem.v3.model.operator import (
  ChannelRequest,
  EvaluationStatus,
  OperatorEvaluationInput,
  evaluation_derivative_values,
  evaluation_status,
)
from pyfem.v3.model.provenance import CanonicalManifest
from pyfem.v3.model.registry import RegistryDescriptor
from pyfem.v3.spec import (
  CellBlockSpec,
  CellRef,
  CellSpec,
  FieldSpec,
  MaterialParameterSpec,
  MaterialSpec,
  MeshSpec,
  ModelSpec,
  NodeSpec,
  RegionSpec,
  SourceContext,
)
from pyfem.v3.state import StateCodecError, StateTransactionOwner

_UNIT_COORDINATES = (
  (0.0, 0.0),
  (0.5, 0.0),
  (1.0, 0.0),
  (1.0, 0.5),
  (1.0, 1.0),
  (0.5, 1.0),
  (0.0, 1.0),
  (0.0, 0.5),
)

_MOCK_MODEL = "mock-parameterized-visco"
_MOCK_STATE_SCHEMA = "pyfem-v3-mock-visco-state-v1"


def _source(label: str) -> SourceContext:
  return SourceContext(source=label)


def _mock_metadata() -> dict[str, object]:
  return {
    "schema": STATEFUL_MATERIAL_DESCRIPTOR_SCHEMA,
    "law": "mock-parameterized-visco",
    "stress_state": "plane-strain",
    "parameter_names": ["youngs_modulus", "term_count"],
    "parameter_dtype": "float64",
    "stress_voigt_order": ["xx", "yy", "xy"],
    "strain_shear_convention": "engineering",
    "internal_voigt_order": ["xx", "yy", "zz", "yz", "zx", "xy"],
    "tangent_class": "algorithmic-symmetric",
    "state_schema": _MOCK_STATE_SCHEMA,
    "state_slots": [
      {
        "name": "q",
        "width": {"parameter": "term_count", "scale": 6},
        "dtype": "float64",
        "lifetime": "accepted-trial",
      },
      {
        "name": "sigma",
        "width": 6,
        "dtype": "float64",
        "lifetime": "accepted-trial",
      },
      {
        "name": "kappa",
        "width": 1,
        "dtype": "float64",
        "lifetime": "accepted-trial",
        "annotation": "monotone-nondecreasing",
      },
    ],
  }


def _mock_kernel(
  strains: np.ndarray,
  accepted_rows: np.ndarray,
  calibration: np.ndarray,
) -> StatefulContinuumKernelResult:
  """Linear echo kernel: sigma slot tracks E * strain, kappa is an envelope."""
  modulus = float(calibration[0])
  row_width = accepted_rows.shape[1]
  trial_rows = np.array(accepted_rows, copy=True)
  trial_rows[:, -1] = np.maximum(accepted_rows[:, -1], np.abs(strains[:, 0]))
  trial_rows[:, row_width - 7 : row_width - 1] = modulus * strains
  return StatefulContinuumKernelResult(
    stresses=modulus * strains,
    tangents=np.broadcast_to(modulus * np.eye(6), (len(strains), 6, 6)).copy(),
    trial_rows=trial_rows,
    status=EvaluationStatus.OK,
  )


@dataclass(frozen=True, slots=True)
class _MockViscoBinding:
  """Parameterized-width mock binding with a nonzero parameter-derived state."""

  def __call__(self, *parameters: float) -> np.ndarray:
    return np.array([parameters[0]], dtype=np.float64)

  def descriptor_metadata(self) -> dict[str, object]:
    return _mock_metadata()

  def kernel(
    self,
    strains: np.ndarray,
    accepted_rows: np.ndarray,
    calibration: np.ndarray,
  ) -> StatefulContinuumKernelResult:
    return _mock_kernel(strains, accepted_rows, calibration)

  def initial_state(
    self,
    parameters: tuple[float, ...],
    layout: object,
  ) -> np.ndarray:
    term_count = int(parameters[1])
    row_width = 6 * term_count + 7
    rows = np.zeros((layout.row_shape[0], row_width), dtype=np.float64)
    rows[:, : 6 * term_count] = 7.0
    rows[:, -1] = 0.25
    return rows


def _mock_registry(
  *,
  binding: object | None = None,
  metadata: dict[str, object] | None = None,
  model: str = _MOCK_MODEL,
) -> dict[tuple[str, str], RegistryDescriptor]:
  registry = dict(q8_reference_registry())
  descriptor = RegistryDescriptor(
    kind="material",
    name=model,
    version="1",
    implementation_id="pyfem-v3-test-mock-visco-v1",
    metadata=_mock_metadata() if metadata is None else metadata,
    binding=_MockViscoBinding() if binding is None else binding,
  )
  registry[descriptor.key] = descriptor
  return registry


def _stateful_model(
  model: str,
  parameters: tuple[tuple[str, float], ...],
) -> ModelSpec:
  nodes = tuple(
    NodeSpec(
      id=index + 1,
      coordinates=point,
      source=_source(f"node-{index + 1}"),
    )
    for index, point in enumerate(_UNIT_COORDINATES)
  )
  cell = CellSpec(
    id="cell-1",
    node_ids=tuple(node.id for node in nodes),
    source=_source("cell"),
  )
  return ModelSpec(
    mesh=MeshSpec(
      nodes=nodes,
      cell_blocks=(
        CellBlockSpec(
          id="cells",
          reference_topology="quadrilateral",
          topological_dimension=2,
          embedding_dimension=2,
          geometry_interpolation="serendipity-quad8",
          cells=(cell,),
          source=_source("block"),
        ),
      ),
      source=_source("mesh"),
    ),
    fields=(
      FieldSpec(
        id="displacement",
        components=("x", "y"),
        location="node",
        source=_source("field"),
      ),
    ),
    materials=(
      MaterialSpec(
        id="material",
        model=model,
        parameters=tuple(
          MaterialParameterSpec(name, value, _source(f"parameter:{name}"))
          for name, value in parameters
        ),
        source=_source("material"),
      ),
    ),
    regions=(
      RegionSpec(
        id="domain",
        cell_refs=(CellRef("cells", "cell-1"),),
        field_ids=("displacement",),
        material_id="material",
        formulation="small-strain-continuum",
        quadrature="gauss-3x3",
        source=_source("region"),
      ),
    ),
    source=_source("model"),
  )


def _mock_spec(term_count: float = 3.0, model: str = _MOCK_MODEL) -> ModelSpec:
  return _stateful_model(
    model,
    (("youngs_modulus", 1000.0), ("term_count", term_count)),
  )


def test_slot_resolution_fixed_and_parameter_computed_widths() -> None:
  slots = resolve_material_state_slots(
    _mock_metadata()["state_slots"],
    {"youngs_modulus": 1000.0, "term_count": 3.0},
  )
  assert [(slot.name, slot.width) for slot in slots] == [
    ("q", 18),
    ("sigma", 6),
    ("kappa", 1),
  ]
  assert slots[2].annotation == "monotone-nondecreasing"
  # The Crystal-style 4 * nslip + 7 shape resolves through the same fields.
  crystal_slots = resolve_material_state_slots(
    [
      {
        "name": f"slip_{name}",
        "width": {"parameter": "slip_system_count"},
        "dtype": "float64",
        "lifetime": "accepted-trial",
      }
      for name in ("tau_c", "gamma", "tau", "gamma_cum")
    ]
    + [
      {"name": "sigma", "width": 6, "dtype": "float64", "lifetime": "accepted-trial"},
      {
        "name": "gamma_total",
        "width": 1,
        "dtype": "float64",
        "lifetime": "accepted-trial",
        "annotation": "monotone-nondecreasing",
      },
    ],
    {"slip_system_count": 12.0},
  )
  assert [slot.width for slot in crystal_slots] == [12, 12, 12, 12, 6, 1]


def test_slot_resolution_rejects_invalid_declarations() -> None:
  valid = _mock_metadata()["state_slots"]
  with pytest.raises(ValueError, match="undeclared material parameter"):
    resolve_material_state_slots(valid, {"youngs_modulus": 1000.0})
  with pytest.raises(ValueError, match="positive integer value"):
    resolve_material_state_slots(valid, {"youngs_modulus": 1000.0, "term_count": 2.5})
  with pytest.raises(ValueError, match="unique"):
    resolve_material_state_slots(
      [
        {"name": "dup", "width": 1, "dtype": "float64", "lifetime": "accepted-trial"},
        {"name": "dup", "width": 2, "dtype": "float64", "lifetime": "accepted-trial"},
      ],
      {},
    )
  with pytest.raises(ValueError, match="positive exact integers"):
    resolve_material_state_slots(
      [{"name": "bad", "width": 0, "dtype": "float64", "lifetime": "accepted-trial"}],
      {},
    )
  with pytest.raises(ValueError, match="annotations"):
    resolve_material_state_slots(
      [
        {
          "name": "bad",
          "width": 1,
          "dtype": "float64",
          "lifetime": "accepted-trial",
          "annotation": "sometimes",
        }
      ],
      {},
    )
  with pytest.raises(TypeError, match="non-empty exact string"):
    resolve_material_state_slots(
      [{"name": "bad", "width": 1, "dtype": 64, "lifetime": "accepted-trial"}],
      {},
    )
  with pytest.raises(ValueError, match="float64"):
    resolve_material_state_slots(
      [
        {
          "name": "bad",
          "width": 1,
          "dtype": "float32",
          "lifetime": "accepted-trial",
        }
      ],
      {},
    )


def test_stateful_metadata_validation_is_fail_closed() -> None:
  metadata = _mock_metadata()
  assert validate_stateful_material_metadata(metadata) is metadata
  with pytest.raises(ValueError, match="frozen v2 field set"):
    validate_stateful_material_metadata({**metadata, "extra": 1})
  missing = {key: value for key, value in metadata.items() if key != "state_slots"}
  with pytest.raises(ValueError, match="frozen v2 field set"):
    validate_stateful_material_metadata(missing)
  with pytest.raises(ValueError, match="schema"):
    validate_stateful_material_metadata(
      {**metadata, "schema": "pyfem-v3-material-descriptor-v1"}
    )
  with pytest.raises(ValueError, match="tangent_class"):
    validate_stateful_material_metadata(
      {**metadata, "tangent_class": "constant-symmetric"}
    )
  with pytest.raises(ValueError, match="unique"):
    validate_stateful_material_metadata({**metadata, "parameter_names": ["a", "a"]})
  with pytest.raises(TypeError, match="non-empty exact list"):
    validate_stateful_material_metadata({**metadata, "state_slots": ()})


def test_tangent_class_flags_are_never_linear() -> None:
  # Factorization honesty is structural: every stateful class is nonlinear.
  assert stateful_tangent_channel_flags("algorithmic-symmetric") == (False, True)
  assert stateful_tangent_channel_flags("algorithmic-nonsymmetric") == (False, False)
  assert stateful_tangent_channel_flags("secant-branch") == (False, False)
  with pytest.raises(ValueError, match="tangent class"):
    stateful_tangent_channel_flags("constant-symmetric")


def test_layout_schema_versions_the_resolved_slot_layout() -> None:
  slots_3 = resolve_material_state_slots(
    _mock_metadata()["state_slots"], {"youngs_modulus": 1000.0, "term_count": 3.0}
  )
  slots_4 = resolve_material_state_slots(
    _mock_metadata()["state_slots"], {"youngs_modulus": 1000.0, "term_count": 4.0}
  )
  schema_3 = material_state_layout_schema(_MOCK_STATE_SCHEMA, slots_3)
  schema_4 = material_state_layout_schema(_MOCK_STATE_SCHEMA, slots_4)
  assert schema_3 == f"{_MOCK_STATE_SCHEMA}|q:18,sigma:6,kappa:1"
  assert schema_4 == f"{_MOCK_STATE_SCHEMA}|q:24,sigma:6,kappa:1"
  assert schema_3 != schema_4


def test_build_layout_flattens_the_entity_axis() -> None:
  slots = resolve_material_state_slots(
    _mock_metadata()["state_slots"], {"youngs_modulus": 1000.0, "term_count": 2.0}
  )
  entity_count = 2 * 9  # two elements x nine integration points x one slot
  layout = build_material_state_layout(
    schema_prefix=_MOCK_STATE_SCHEMA,
    block_id=("cells", "domain"),
    entity_count=entity_count,
    slots=slots,
    initial_rows=None,
    index_dtype=np.dtype(np.int64),
  )
  assert layout.entity_count == entity_count
  assert layout.row_width == 19
  assert layout.row_shape == (18, 19)
  np.testing.assert_array_equal(
    layout.entity_offsets.values,
    np.arange(entity_count + 1) * 19,
  )
  assert layout.initial_rows is None
  with pytest.raises(ValueError, match="overflows"):
    build_material_state_layout(
      schema_prefix=_MOCK_STATE_SCHEMA,
      block_id=("cells", "domain"),
      entity_count=entity_count,
      slots=slots,
      initial_rows=None,
      index_dtype=np.dtype(np.int8),
    )
  with pytest.raises(TypeError, match="initial state rows"):
    build_material_state_layout(
      schema_prefix=_MOCK_STATE_SCHEMA,
      block_id=("cells", "domain"),
      entity_count=entity_count,
      slots=slots,
      initial_rows=np.zeros((entity_count, 18)),
      index_dtype=np.dtype(np.int64),
    )
  with pytest.raises(ValueError, match="finite"):
    build_material_state_layout(
      schema_prefix=_MOCK_STATE_SCHEMA,
      block_id=("cells", "domain"),
      entity_count=entity_count,
      slots=slots,
      initial_rows=np.full((entity_count, 19), np.nan),
      index_dtype=np.dtype(np.int64),
    )


def test_mock_law_compiles_through_the_generic_stateful_path() -> None:
  system = compile_system(_mock_spec(term_count=3.0), _mock_registry())
  operator = system.operators[0]
  layout = operator.header.state_layout
  assert layout.schema == f"{_MOCK_STATE_SCHEMA}|q:18,sigma:6,kappa:1"
  assert layout.row_shape == (9, 25)
  assert [slot.width for slot in layout.slots] == [18, 6, 1]
  assert layout.slots[2].annotation == "monotone-nondecreasing"
  jacobian = operator.header.jacobian_channels[0]
  assert jacobian.linear is False
  assert jacobian.symmetric is True
  # The descriptor-bound initial state reaches the owner unchanged (F3 G2).
  assert layout.initial_rows is not None
  owner = StateTransactionOwner(system)
  accepted = owner.accepted_state(layout.block_id).values
  np.testing.assert_array_equal(accepted, layout.initial_rows.values)
  assert np.all(accepted[:, :18] == 7.0)
  assert np.all(accepted[:, -1] == 0.25)
  # Codec round-trip over the nonzero-width rows.
  payload = owner.encode_state(layout.block_id)
  decoded = owner.decode_state(layout.block_id, payload)
  np.testing.assert_array_equal(decoded.values, accepted)
  with pytest.raises(StateCodecError):
    owner.decode_state(layout.block_id, payload.replace(b"q:18", b"q:24"))


def test_mock_law_width_follows_the_spec_parameter() -> None:
  narrow = compile_system(_mock_spec(term_count=2.0), _mock_registry())
  wide = compile_system(_mock_spec(term_count=4.0), _mock_registry())
  narrow_layout = narrow.operators[0].header.state_layout
  wide_layout = wide.operators[0].header.state_layout
  assert narrow_layout.row_shape == (9, 19)
  assert wide_layout.row_shape == (9, 31)
  assert narrow_layout.schema != wide_layout.schema
  assert narrow.content_fingerprint != wide.content_fingerprint


def test_binding_without_initial_state_keeps_zero_initialization() -> None:
  @dataclass(frozen=True, slots=True)
  class ZeroInitBinding:
    def __call__(self, *parameters: float) -> np.ndarray:
      return np.array([parameters[0]], dtype=np.float64)

    def descriptor_metadata(self) -> dict[str, object]:
      return _mock_metadata()

    def kernel(
      self,
      strains: np.ndarray,
      accepted_rows: np.ndarray,
      calibration: np.ndarray,
    ) -> StatefulContinuumKernelResult:
      return _mock_kernel(strains, accepted_rows, calibration)

  system = compile_system(_mock_spec(), _mock_registry(binding=ZeroInitBinding()))
  layout = system.operators[0].header.state_layout
  assert layout.initial_rows is None
  owner = StateTransactionOwner(system)
  accepted = owner.accepted_state(layout.block_id).values
  assert np.all(accepted == 0.0)


def test_plasticity_descriptor_is_pinned_and_self_consistent() -> None:
  registry = plasticity_reference_registry()
  descriptor = registry[PLASTIC_MATERIAL_KEY]
  assert descriptor.metadata.to_bytes() == (
    CanonicalManifest(isotropic_hardening_plasticity_metadata()).to_bytes()
  )
  assert (
    ISOTROPIC_HARDENING_PLASTICITY_BINDING.descriptor_metadata()
    == isotropic_hardening_plasticity_metadata()
  )
  # The reference Q8 registry is untouched by the stateful extension.
  assert PLASTIC_MATERIAL_KEY not in q8_reference_registry()
  assert Q8_MATERIAL_KEY in registry


def test_stateful_compile_boundary_fails_coded() -> None:
  # Binding metadata drifting from the descriptor manifest.
  drifted = {**_mock_metadata(), "law": "drifted"}
  with pytest.raises(ModelCompilationError, match="incompatible-registry-descriptor"):
    compile_system(_mock_spec(), _mock_registry(metadata=drifted))
  # A v1-schema descriptor cannot bind the stateful path, even when the
  # binding re-declares it consistently.

  @dataclass(frozen=True, slots=True)
  class V1Binding:
    def __call__(self, *parameters: float) -> np.ndarray:
      return np.array([parameters[0]], dtype=np.float64)

    def descriptor_metadata(self) -> dict[str, object]:
      return {
        "schema": "pyfem-v3-material-descriptor-v1",
        "law": "linear-elastic",
      }

    def kernel(
      self,
      strains: np.ndarray,
      accepted_rows: np.ndarray,
      calibration: np.ndarray,
    ) -> StatefulContinuumKernelResult:
      return _mock_kernel(strains, accepted_rows, calibration)

  with pytest.raises(ModelCompilationError, match="malformed-registry-descriptor"):
    compile_system(
      _mock_spec(),
      _mock_registry(binding=V1Binding(), metadata=V1Binding().descriptor_metadata()),
    )
  # Unknown parameter reference inside a slot width.
  bad_width = _mock_metadata()
  bad_width["state_slots"] = [
    {
      "name": "q",
      "width": {"parameter": "ghost"},
      "dtype": "float64",
      "lifetime": "accepted-trial",
    }
  ]

  @dataclass(frozen=True, slots=True)
  class GhostBinding(_MockViscoBinding):
    def descriptor_metadata(self) -> dict[str, object]:
      return bad_width

  with pytest.raises(ModelCompilationError, match="invalid-material-state-schema"):
    compile_system(
      _mock_spec(),
      _mock_registry(binding=GhostBinding(), metadata=bad_width),
    )
  # Non-integral parameter-computed width.
  with pytest.raises(ModelCompilationError, match="invalid-material-state-schema"):
    compile_system(_mock_spec(term_count=2.5), _mock_registry())
  # Missing declared parameter.
  with pytest.raises(ModelCompilationError, match="invalid-material-parameter-schema"):
    compile_system(
      _stateful_model(_MOCK_MODEL, (("youngs_modulus", 1000.0),)),
      _mock_registry(),
    )


def test_initial_state_binding_failures_fail_coded() -> None:
  @dataclass(frozen=True, slots=True)
  class WrongShapeBinding(_MockViscoBinding):
    def initial_state(
      self,
      parameters: tuple[float, ...],
      layout: object,
    ) -> np.ndarray:
      return np.zeros((1, 1), dtype=np.float64)

  with pytest.raises(ModelCompilationError, match="invalid-material-binding-output"):
    compile_system(_mock_spec(), _mock_registry(binding=WrongShapeBinding()))

  @dataclass(frozen=True, slots=True)
  class NonFiniteBinding(_MockViscoBinding):
    def initial_state(
      self,
      parameters: tuple[float, ...],
      layout: object,
    ) -> np.ndarray:
      rows = np.full(layout.row_shape, np.inf)
      return rows

  with pytest.raises(ModelCompilationError, match="invalid-material-binding-output"):
    compile_system(_mock_spec(), _mock_registry(binding=NonFiniteBinding()))

  @dataclass(frozen=True, slots=True)
  class RaisingBinding(_MockViscoBinding):
    def initial_state(
      self,
      parameters: tuple[float, ...],
      layout: object,
    ) -> np.ndarray:
      raise RuntimeError("boom")

  with pytest.raises(ModelCompilationError, match="material-binding-failed"):
    compile_system(_mock_spec(), _mock_registry(binding=RaisingBinding()))


def test_kernel_probe_failures_fail_coded() -> None:
  def mutating_kernel(
    strains: np.ndarray,
    accepted_rows: np.ndarray,
    calibration: np.ndarray,
  ) -> StatefulContinuumKernelResult:
    result = _mock_kernel(strains, accepted_rows, calibration)
    mutated = np.array(result.trial_rows, copy=True)
    mutated[:, -1] += 1.0
    return StatefulContinuumKernelResult(
      stresses=result.stresses,
      tangents=result.tangents,
      trial_rows=mutated,
      status=EvaluationStatus.OK,
    )

  @dataclass(frozen=True, slots=True)
  class MutatingBinding(_MockViscoBinding):
    def kernel(
      self,
      strains: np.ndarray,
      accepted_rows: np.ndarray,
      calibration: np.ndarray,
    ) -> StatefulContinuumKernelResult:
      return mutating_kernel(strains, accepted_rows, calibration)

  with pytest.raises(ModelCompilationError, match="invalid-kernel-probe"):
    compile_system(_mock_spec(), _mock_registry(binding=MutatingBinding()))

  def nonsymmetric_kernel(
    strains: np.ndarray,
    accepted_rows: np.ndarray,
    calibration: np.ndarray,
  ) -> StatefulContinuumKernelResult:
    result = _mock_kernel(strains, accepted_rows, calibration)
    tangents = np.array(result.tangents, copy=True)
    tangents[:, 0, 1] += 1.0
    return StatefulContinuumKernelResult(
      stresses=result.stresses,
      tangents=tangents,
      trial_rows=result.trial_rows,
      status=EvaluationStatus.OK,
    )

  @dataclass(frozen=True, slots=True)
  class NonsymmetricBinding(_MockViscoBinding):
    def kernel(
      self,
      strains: np.ndarray,
      accepted_rows: np.ndarray,
      calibration: np.ndarray,
    ) -> StatefulContinuumKernelResult:
      return nonsymmetric_kernel(strains, accepted_rows, calibration)

  with pytest.raises(ModelCompilationError, match="invalid-kernel-probe"):
    compile_system(_mock_spec(), _mock_registry(binding=NonsymmetricBinding()))

  @dataclass(frozen=True, slots=True)
  class KernellessBinding:
    def __call__(self, *parameters: float) -> np.ndarray:
      return np.array([1.0], dtype=np.float64)

    def descriptor_metadata(self) -> dict[str, object]:
      return _mock_metadata()

  with pytest.raises(ModelCompilationError, match="malformed-registry-descriptor"):
    compile_system(_mock_spec(), _mock_registry(binding=KernellessBinding()))


def test_unknown_material_model_fails_at_registry_capture() -> None:
  with pytest.raises(ModelCompilationError, match="registry-capture-failed"):
    compile_system(
      _stateful_model(
        "no-such-law",
        (("youngs_modulus", 1000.0), ("term_count", 3.0)),
      ),
      q8_reference_registry(),
    )


class _MockDerivativeBinding(_MockViscoBinding):
  """The mock law with the optional derivative kernel member attached.

  The echo map is exactly linear in ``youngs_modulus`` (sigma = E * strain)
  and independent of ``term_count``, so the exact columns are ``strains``
  and zero respectively.
  """

  def param_derivative_kernel(
    self,
    strains: np.ndarray,
    accepted_rows: np.ndarray,
    calibration: np.ndarray,
  ) -> StatefulContinuumKernelResult:
    base = _mock_kernel(strains, accepted_rows, calibration)
    columns = np.zeros((2, len(strains), 6), dtype=np.float64)
    columns[0] = strains
    return StatefulContinuumKernelResult(
      stresses=base.stresses,
      tangents=base.tangents,
      trial_rows=base.trial_rows,
      status=base.status,
      param_derivatives=columns,
    )


def test_derivative_capable_binding_opens_the_channel_on_the_generic_path() -> None:
  """The parameter-derivative channel is generic, not J2-specific.

  The parameterized-width mock binding gains the optional
  ``param_derivative_kernel`` member, and the same generic stateful compiler
  path emits one ``ParameterBinding`` and one ``dinternal-force/d<name>``
  channel per declared ``parameter_names`` entry and answers derivative
  requests with the assembled exact columns. The echo map's exact linearity
  in the modulus makes the oracle bitwise: the ``dR/dyoungs_modulus``
  channel equals the unit-modulus recompile's residual (``1.0 * x`` is
  exact), and the ``term_count`` channel — which the stress map does not
  reference — is exactly zero.
  """
  system = compile_system(
    _mock_spec(), _mock_registry(binding=_MockDerivativeBinding())
  )
  operator = system.operators[0]
  header = operator.header
  assert tuple(item.parameter_id for item in header.parameters) == (
    "youngs_modulus",
    "term_count",
  )
  assert tuple(item.channel_id for item in header.derivative_channels) == (
    "dinternal-force/dyoungs_modulus",
    "dinternal-force/dterm_count",
  )
  layout = header.state_layout
  assert layout.initial_rows is not None
  displacements = np.zeros((1, 16))
  displacements[0, 0::2] = 1.0e-3 * np.array([point[0] for point in _UNIT_COORDINATES])
  accepted = FinalizedArray(
    np.array(layout.initial_rows.values, copy=True), dtype=np.float64
  )
  evaluation = operator.evaluate(
    OperatorEvaluationInput(
      port_values=(FinalizedArray(displacements, dtype=np.float64),),
      accepted_state=accepted,
      signals=(),
      request=ChannelRequest(
        ("internal-force",),
        (),
        ("dinternal-force/dyoungs_modulus", "dinternal-force/dterm_count"),
      ),
    )
  )
  assert evaluation_status(evaluation) is EvaluationStatus.OK
  columns = evaluation_derivative_values(evaluation)
  assert len(columns) == 2
  assert np.all(columns[1].values == 0.0)
  unit = compile_system(
    _stateful_model(_MOCK_MODEL, (("youngs_modulus", 1.0), ("term_count", 3.0))),
    _mock_registry(),
  )
  reference = unit.operators[0].evaluate(
    OperatorEvaluationInput(
      port_values=(FinalizedArray(displacements, dtype=np.float64),),
      accepted_state=accepted,
      signals=(),
      request=ChannelRequest(("internal-force",), ()),
    )
  )
  np.testing.assert_array_equal(columns[0].values, reference.residual_values[0].values)
  # The primal-only request on the same operator serves no derivative values.
  primal = operator.evaluate(
    OperatorEvaluationInput(
      port_values=(FinalizedArray(displacements, dtype=np.float64),),
      accepted_state=accepted,
      signals=(),
      request=ChannelRequest(("internal-force",), ()),
    )
  )
  assert evaluation_derivative_values(primal) == ()
  np.testing.assert_array_equal(
    evaluation.residual_values[0].values, primal.residual_values[0].values
  )
