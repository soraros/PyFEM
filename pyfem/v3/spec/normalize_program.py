"""Exact normalization for authored finite-element program values."""

from __future__ import annotations

import math

from pyfem.v3.spec.diagnostics import SourceContext
from pyfem.v3.spec.program import (
  AffineCoefficientSpec,
  AffineTieSpec,
  AffineValueSpec,
  DofRef,
  NodalLoadSpec,
  PrescribedDofSpec,
  ProgramConstraintSpec,
  ProgramCoordinateSpec,
  ProgramSpec,
)
from pyfem.v3.spec.program_diagnostics import (
  ProgramSpecDiagnostic,
  ProgramSpecValidationError,
  render_program_source,
  render_program_value,
)

_COORDINATE_KIND_ORDER = {"time": 0, "load": 1, "continuation": 2}
_MISSING = object()
_INVALID = object()


class _Validator:
  def __init__(self) -> None:
    self.diagnostics: list[ProgramSpecDiagnostic] = []

  def error(self, code: str, message: str, source: SourceContext) -> None:
    self.diagnostics.append(
      ProgramSpecDiagnostic(code=code, message=message, source=source)
    )


def _read_slot(value: object, name: str) -> object:
  try:
    return object.__getattribute__(value, name)
  except AttributeError:
    return _MISSING


def _is_exact_type(
  value: object,
  expected: type[object],
  *,
  code: str,
  label: str,
  source: SourceContext,
  validator: _Validator,
) -> bool:
  if type(value) is expected:
    return True
  validator.error(code, f"{label} must be exactly {expected.__name__}", source)
  return False


def _required_slots(
  value: object,
  names: tuple[str, ...],
  *,
  code: str,
  label: str,
  source: SourceContext,
  validator: _Validator,
) -> tuple[object, ...] | None:
  slots = tuple(_read_slot(value, name) for name in names)
  if any(item is _MISSING for item in slots):
    validator.error(code, f"{label} must initialize every canonical slot", source)
    return None
  return slots


def _exact_tuple(
  value: object,
  *,
  code: str,
  label: str,
  source: SourceContext,
  validator: _Validator,
) -> tuple[object, ...] | None:
  if type(value) is tuple:
    return value
  validator.error(code, f"{label} must be exactly tuple", source)
  return None


def _trusted_source(
  value: object,
  *,
  label: str,
  fallback: SourceContext,
  validator: _Validator,
) -> SourceContext:
  if type(value) is not SourceContext:
    validator.error(
      "invalid-program-source-type",
      f"{label} source must be exactly SourceContext",
      fallback,
    )
    return fallback
  source = _read_slot(value, "source")
  line = _read_slot(value, "line")
  column = _read_slot(value, "column")
  if (
    source is _MISSING
    or line is _MISSING
    or column is _MISSING
    or type(source) is not str
    or (line is not None and type(line) is not int)
    or (column is not None and type(column) is not int)
  ):
    validator.error(
      "invalid-program-source-value",
      f"{label} source must contain plain string/integer values",
      fallback,
    )
    return fallback
  return SourceContext(source=source, line=line, column=column)


def _spec_source(
  value: object,
  *,
  label: str,
  fallback: SourceContext,
  validator: _Validator,
) -> SourceContext:
  source = _read_slot(value, "source")
  if source is _MISSING:
    validator.error(
      "invalid-program-source-value",
      f"{label} source must initialize every canonical slot",
      fallback,
    )
    return fallback
  return _trusted_source(
    source,
    label=label,
    fallback=fallback,
    validator=validator,
  )


def _canonical_id(
  value: object,
  *,
  code: str,
  label: str,
  source: SourceContext,
  validator: _Validator,
) -> object:
  if type(value) is str:
    return str.strip(value)
  if type(value) is int:
    return value
  validator.error(code, f"{label} must be a plain string or integer", source)
  return _INVALID


def _canonical_text(
  value: object,
  *,
  code: str,
  label: str,
  source: SourceContext,
  validator: _Validator,
) -> object:
  if type(value) is str:
    return str.strip(value)
  validator.error(code, f"{label} must be a plain string", source)
  return _INVALID


def _canonical_float64(
  value: object,
  *,
  code: str,
  label: str,
  source: SourceContext,
  validator: _Validator,
) -> object:
  if type(value) is not int and type(value) is not float:
    validator.error(
      code,
      f"{label} must be a finite plain integer or float, excluding bool",
      source,
    )
    return _INVALID
  try:
    converted = float(value)
  except OverflowError:
    validator.error(code, f"{label} cannot be represented as finite float64", source)
    return _INVALID
  if not math.isfinite(converted):
    validator.error(code, f"{label} cannot be represented as finite float64", source)
    return _INVALID
  return converted


def _preflight_coordinate(
  value: object,
  index: int,
  program_source: SourceContext,
  validator: _Validator,
) -> ProgramCoordinateSpec | None:
  label = f"program coordinate {render_program_value(index)}"
  if not _is_exact_type(
    value,
    ProgramCoordinateSpec,
    code="invalid-program-coordinate-type",
    label=label,
    source=program_source,
    validator=validator,
  ):
    return None
  source = _spec_source(
    value,
    label=label,
    fallback=program_source,
    validator=validator,
  )
  slots = _required_slots(
    value,
    ("name", "kind"),
    code="invalid-program-coordinate-value",
    label=label,
    source=source,
    validator=validator,
  )
  if slots is None:
    return None
  name = _canonical_text(
    slots[0],
    code="invalid-program-coordinate-name",
    label="program coordinate name",
    source=source,
    validator=validator,
  )
  kind = _canonical_text(
    slots[1],
    code="invalid-program-coordinate-kind",
    label="program coordinate kind",
    source=source,
    validator=validator,
  )
  if name is _INVALID or kind is _INVALID:
    return None
  return ProgramCoordinateSpec(name=name, kind=kind, source=source)


def _preflight_dof_ref(
  value: object,
  *,
  label: str,
  source: SourceContext,
  validator: _Validator,
) -> DofRef | None:
  if not _is_exact_type(
    value,
    DofRef,
    code="invalid-dof-reference-type",
    label=label,
    source=source,
    validator=validator,
  ):
    return None
  slots = _required_slots(
    value,
    ("node_id", "field_id", "component"),
    code="invalid-dof-reference-value",
    label=label,
    source=source,
    validator=validator,
  )
  if slots is None:
    return None
  node_id = _canonical_id(
    slots[0],
    code="invalid-dof-node-id",
    label="DOF node ID",
    source=source,
    validator=validator,
  )
  field_id = _canonical_id(
    slots[1],
    code="invalid-dof-field-id",
    label="DOF field ID",
    source=source,
    validator=validator,
  )
  component = _canonical_text(
    slots[2],
    code="invalid-dof-component",
    label="DOF component",
    source=source,
    validator=validator,
  )
  if node_id is _INVALID or field_id is _INVALID or component is _INVALID:
    return None
  return DofRef(node_id=node_id, field_id=field_id, component=component)


def _preflight_affine_coefficient(
  value: object,
  *,
  index: int,
  affine_source: SourceContext,
  validator: _Validator,
) -> AffineCoefficientSpec | None:
  label = f"affine coefficient {render_program_value(index)}"
  if not _is_exact_type(
    value,
    AffineCoefficientSpec,
    code="invalid-affine-coefficient-type",
    label=label,
    source=affine_source,
    validator=validator,
  ):
    return None
  source = _spec_source(
    value,
    label=label,
    fallback=affine_source,
    validator=validator,
  )
  slots = _required_slots(
    value,
    ("coordinate", "coefficient"),
    code="invalid-affine-coefficient-value",
    label=label,
    source=source,
    validator=validator,
  )
  if slots is None:
    return None
  coordinate = _canonical_text(
    slots[0],
    code="invalid-affine-coordinate-name",
    label="affine coordinate name",
    source=source,
    validator=validator,
  )
  coefficient = _canonical_float64(
    slots[1],
    code="invalid-affine-coefficient",
    label="affine coefficient",
    source=source,
    validator=validator,
  )
  if coordinate is _INVALID or coefficient is _INVALID:
    return None
  return AffineCoefficientSpec(
    coordinate=coordinate,
    coefficient=coefficient,
    source=source,
  )


def _preflight_affine_value(
  value: object,
  *,
  label: str,
  fallback: SourceContext,
  validator: _Validator,
) -> AffineValueSpec | None:
  if not _is_exact_type(
    value,
    AffineValueSpec,
    code="invalid-affine-value-type",
    label=label,
    source=fallback,
    validator=validator,
  ):
    return None
  source = _spec_source(
    value,
    label=label,
    fallback=fallback,
    validator=validator,
  )
  slots = _required_slots(
    value,
    ("constant", "coefficients"),
    code="invalid-affine-value",
    label=label,
    source=source,
    validator=validator,
  )
  if slots is None:
    return None
  constant = _canonical_float64(
    slots[0],
    code="invalid-affine-constant",
    label="affine constant",
    source=source,
    validator=validator,
  )
  coefficients = _exact_tuple(
    slots[1],
    code="invalid-affine-value",
    label="affine coefficients",
    source=source,
    validator=validator,
  )
  canonical_coefficients: list[AffineCoefficientSpec] = []
  valid = constant is not _INVALID and coefficients is not None
  if coefficients is not None:
    for index, coefficient in enumerate(coefficients):
      canonical = _preflight_affine_coefficient(
        coefficient,
        index=index,
        affine_source=source,
        validator=validator,
      )
      if canonical is None:
        valid = False
      else:
        canonical_coefficients.append(canonical)
  if not valid:
    return None
  return AffineValueSpec(
    constant=constant,
    coefficients=tuple(canonical_coefficients),
    source=source,
  )


def _preflight_prescribed(
  value: PrescribedDofSpec,
  *,
  index: int,
  program_source: SourceContext,
  validator: _Validator,
) -> PrescribedDofSpec | None:
  label = f"program constraint {render_program_value(index)}"
  source = _spec_source(
    value,
    label=label,
    fallback=program_source,
    validator=validator,
  )
  slots = _required_slots(
    value,
    ("id", "target", "value"),
    code="invalid-prescribed-dof-value",
    label=label,
    source=source,
    validator=validator,
  )
  if slots is None:
    return None
  constraint_id = _canonical_id(
    slots[0],
    code="invalid-constraint-id",
    label="constraint ID",
    source=source,
    validator=validator,
  )
  target = _preflight_dof_ref(
    slots[1],
    label="prescribed DOF target",
    source=source,
    validator=validator,
  )
  affine_value = _preflight_affine_value(
    slots[2],
    label="prescribed affine value",
    fallback=source,
    validator=validator,
  )
  if constraint_id is _INVALID or target is None or affine_value is None:
    return None
  return PrescribedDofSpec(
    id=constraint_id,
    target=target,
    value=affine_value,
    source=source,
  )


def _preflight_tie(
  value: AffineTieSpec,
  *,
  index: int,
  program_source: SourceContext,
  validator: _Validator,
) -> AffineTieSpec | None:
  label = f"program constraint {render_program_value(index)}"
  source = _spec_source(
    value,
    label=label,
    fallback=program_source,
    validator=validator,
  )
  slots = _required_slots(
    value,
    ("id", "slave", "master", "factor", "offset"),
    code="invalid-affine-tie-value",
    label=label,
    source=source,
    validator=validator,
  )
  if slots is None:
    return None
  constraint_id = _canonical_id(
    slots[0],
    code="invalid-constraint-id",
    label="constraint ID",
    source=source,
    validator=validator,
  )
  slave = _preflight_dof_ref(
    slots[1],
    label="affine tie slave",
    source=source,
    validator=validator,
  )
  master = _preflight_dof_ref(
    slots[2],
    label="affine tie master",
    source=source,
    validator=validator,
  )
  factor = _canonical_float64(
    slots[3],
    code="invalid-affine-tie-factor",
    label="affine tie factor",
    source=source,
    validator=validator,
  )
  offset = _preflight_affine_value(
    slots[4],
    label="affine tie offset",
    fallback=source,
    validator=validator,
  )
  if (
    constraint_id is _INVALID
    or slave is None
    or master is None
    or factor is _INVALID
    or offset is None
  ):
    return None
  return AffineTieSpec(
    id=constraint_id,
    slave=slave,
    master=master,
    factor=factor,
    offset=offset,
    source=source,
  )


def _preflight_constraint(
  value: object,
  *,
  index: int,
  program_source: SourceContext,
  validator: _Validator,
) -> ProgramConstraintSpec | None:
  if type(value) is PrescribedDofSpec:
    return _preflight_prescribed(
      value,
      index=index,
      program_source=program_source,
      validator=validator,
    )
  if type(value) is AffineTieSpec:
    return _preflight_tie(
      value,
      index=index,
      program_source=program_source,
      validator=validator,
    )
  validator.error(
    "invalid-program-constraint-type",
    f"program constraint {render_program_value(index)} must be exactly "
    "PrescribedDofSpec or AffineTieSpec",
    program_source,
  )
  return None


def _preflight_load(
  value: object,
  *,
  index: int,
  program_source: SourceContext,
  validator: _Validator,
) -> NodalLoadSpec | None:
  label = f"program load {render_program_value(index)}"
  if not _is_exact_type(
    value,
    NodalLoadSpec,
    code="invalid-nodal-load-type",
    label=label,
    source=program_source,
    validator=validator,
  ):
    return None
  source = _spec_source(
    value,
    label=label,
    fallback=program_source,
    validator=validator,
  )
  slots = _required_slots(
    value,
    ("id", "target", "value"),
    code="invalid-nodal-load-value",
    label=label,
    source=source,
    validator=validator,
  )
  if slots is None:
    return None
  load_id = _canonical_id(
    slots[0],
    code="invalid-load-id",
    label="load ID",
    source=source,
    validator=validator,
  )
  target = _preflight_dof_ref(
    slots[1],
    label="nodal load target",
    source=source,
    validator=validator,
  )
  affine_value = _preflight_affine_value(
    slots[2],
    label="nodal load affine value",
    fallback=source,
    validator=validator,
  )
  if load_id is _INVALID or target is None or affine_value is None:
    return None
  return NodalLoadSpec(
    id=load_id,
    target=target,
    value=affine_value,
    source=source,
  )


def _preflight_program(spec: object, validator: _Validator) -> ProgramSpec | None:
  fallback = SourceContext()
  if not _is_exact_type(
    spec,
    ProgramSpec,
    code="invalid-program-spec-type",
    label="program",
    source=fallback,
    validator=validator,
  ):
    return None
  source = _spec_source(
    spec,
    label="program",
    fallback=fallback,
    validator=validator,
  )
  slots = _required_slots(
    spec,
    ("coordinates", "constraints", "loads"),
    code="invalid-program-spec-value",
    label="program",
    source=source,
    validator=validator,
  )
  if slots is None:
    return None
  coordinates = _exact_tuple(
    slots[0],
    code="invalid-program-spec-value",
    label="program coordinates",
    source=source,
    validator=validator,
  )
  constraints = _exact_tuple(
    slots[1],
    code="invalid-program-spec-value",
    label="program constraints",
    source=source,
    validator=validator,
  )
  loads = _exact_tuple(
    slots[2],
    code="invalid-program-spec-value",
    label="program loads",
    source=source,
    validator=validator,
  )
  canonical_coordinates: list[ProgramCoordinateSpec] = []
  canonical_constraints: list[ProgramConstraintSpec] = []
  canonical_loads: list[NodalLoadSpec] = []
  valid = coordinates is not None and constraints is not None and loads is not None
  if coordinates is not None:
    for index, coordinate in enumerate(coordinates):
      canonical = _preflight_coordinate(coordinate, index, source, validator)
      if canonical is None:
        valid = False
      else:
        canonical_coordinates.append(canonical)
  if constraints is not None:
    for index, constraint in enumerate(constraints):
      canonical = _preflight_constraint(
        constraint,
        index=index,
        program_source=source,
        validator=validator,
      )
      if canonical is None:
        valid = False
      else:
        canonical_constraints.append(canonical)
  if loads is not None:
    for index, load in enumerate(loads):
      canonical = _preflight_load(
        load,
        index=index,
        program_source=source,
        validator=validator,
      )
      if canonical is None:
        valid = False
      else:
        canonical_loads.append(canonical)
  if not valid:
    return None
  return ProgramSpec(
    coordinates=tuple(canonical_coordinates),
    constraints=tuple(canonical_constraints),
    loads=tuple(canonical_loads),
    source=source,
  )


def _valid_id(value: object) -> bool:
  return type(value) is int or type(value) is str and bool(value)


def _semantic_id_key(value: str | int) -> tuple[int, object]:
  return (0, value) if type(value) is int else (1, value)


def _typed_id(value: str | int) -> tuple[str, str | int]:
  return ("int", value) if type(value) is int else ("str", value)


def _dof_key(value: DofRef) -> tuple[tuple[str, str | int], ...]:
  return (
    _typed_id(value.node_id),
    _typed_id(value.field_id),
    ("str", value.component),
  )


def _validate_dof_ref(
  value: DofRef,
  *,
  label: str,
  source: SourceContext,
  validator: _Validator,
) -> None:
  if not _valid_id(value.node_id):
    validator.error(
      "invalid-dof-node-id",
      f"{label} node ID must be a non-empty string or integer",
      source,
    )
  if not _valid_id(value.field_id):
    validator.error(
      "invalid-dof-field-id",
      f"{label} field ID must be a non-empty string or integer",
      source,
    )
  if type(value.component) is not str or not value.component:
    validator.error(
      "invalid-dof-component",
      f"{label} component must be a non-empty string",
      source,
    )


def _validate_affine(
  value: AffineValueSpec,
  *,
  coordinate_names: set[str],
  validator: _Validator,
) -> None:
  seen: set[str] = set()
  for coefficient in value.coefficients:
    name = coefficient.coordinate
    if not name:
      validator.error(
        "invalid-affine-coordinate-name",
        "affine coordinate name must be non-empty",
        coefficient.source,
      )
      continue
    if name in seen:
      validator.error(
        "duplicate-affine-coordinate",
        f"affine value repeats coordinate {render_program_value(name)}",
        coefficient.source,
      )
    elif name not in coordinate_names:
      validator.error(
        "unknown-affine-coordinate",
        f"affine value references unknown coordinate {render_program_value(name)}",
        coefficient.source,
      )
    seen.add(name)


def _copy_source(source: SourceContext) -> SourceContext:
  return SourceContext(source=source.source, line=source.line, column=source.column)


def _copy_dof_ref(value: DofRef) -> DofRef:
  return DofRef(
    node_id=value.node_id,
    field_id=value.field_id,
    component=value.component,
  )


def _canonical_affine(
  value: AffineValueSpec,
  coordinate_indices: dict[str, int],
) -> AffineValueSpec:
  coefficients = tuple(
    AffineCoefficientSpec(
      coordinate=item.coordinate,
      coefficient=item.coefficient,
      source=_copy_source(item.source),
    )
    for item in sorted(
      value.coefficients,
      key=lambda item: coordinate_indices[item.coordinate],
    )
  )
  return AffineValueSpec(
    constant=value.constant,
    coefficients=coefficients,
    source=_copy_source(value.source),
  )


def normalize_program_spec(spec: ProgramSpec) -> ProgramSpec:
  """Return a canonical exact program tree detached from every caller value."""
  validator = _Validator()
  snapshot = _preflight_program(spec, validator)
  if validator.diagnostics or snapshot is None:
    raise ProgramSpecValidationError(validator.diagnostics)

  coordinate_sources: dict[str, SourceContext] = {}
  time_count = 0
  for coordinate in snapshot.coordinates:
    if not coordinate.name:
      validator.error(
        "invalid-program-coordinate-name",
        "program coordinate name must be non-empty",
        coordinate.source,
      )
    elif coordinate.name in coordinate_sources:
      first = coordinate_sources[coordinate.name]
      validator.error(
        "duplicate-program-coordinate-name",
        f"duplicate program coordinate {render_program_value(coordinate.name)}; "
        f"first declared at {render_program_source(first)}",
        coordinate.source,
      )
    else:
      coordinate_sources[coordinate.name] = coordinate.source
    if coordinate.kind not in _COORDINATE_KIND_ORDER:
      validator.error(
        "invalid-program-coordinate-kind",
        "program coordinate kind must be exactly time, load, or continuation",
        coordinate.source,
      )
    if coordinate.kind == "time":
      time_count += 1
    if (coordinate.name == "time") != (coordinate.kind == "time"):
      validator.error(
        "invalid-time-coordinate-reservation",
        "coordinate name 'time' is reserved exactly for kind 'time'",
        coordinate.source,
      )
  if time_count > 1:
    validator.error(
      "duplicate-time-coordinate",
      "a program may declare at most one time coordinate",
      snapshot.source,
    )

  coordinate_names = set(coordinate_sources)
  constraint_ids: dict[tuple[str, str | int], SourceContext] = {}
  for constraint in snapshot.constraints:
    identifier = constraint.id
    if not _valid_id(identifier):
      validator.error(
        "invalid-constraint-id",
        "constraint ID must be a non-empty string or integer",
        constraint.source,
      )
    else:
      key = _typed_id(identifier)
      first = constraint_ids.get(key)
      if first is not None:
        validator.error(
          "duplicate-constraint-id",
          f"duplicate constraint ID {render_program_value(identifier)}; "
          f"first declared at {render_program_source(first)}",
          constraint.source,
        )
      else:
        constraint_ids[key] = constraint.source
    if type(constraint) is PrescribedDofSpec:
      _validate_dof_ref(
        constraint.target,
        label="prescribed DOF",
        source=constraint.source,
        validator=validator,
      )
      _validate_affine(
        constraint.value,
        coordinate_names=coordinate_names,
        validator=validator,
      )
    else:
      _validate_dof_ref(
        constraint.slave,
        label="affine tie slave",
        source=constraint.source,
        validator=validator,
      )
      _validate_dof_ref(
        constraint.master,
        label="affine tie master",
        source=constraint.source,
        validator=validator,
      )
      if constraint.factor == 0.0:
        validator.error(
          "zero-affine-tie-factor",
          "affine tie factor must be nonzero",
          constraint.source,
        )
      _validate_affine(
        constraint.offset,
        coordinate_names=coordinate_names,
        validator=validator,
      )

  load_ids: dict[tuple[str, str | int], SourceContext] = {}
  for load in snapshot.loads:
    if not _valid_id(load.id):
      validator.error(
        "invalid-load-id",
        "load ID must be a non-empty string or integer",
        load.source,
      )
    else:
      key = _typed_id(load.id)
      first = load_ids.get(key)
      if first is not None:
        validator.error(
          "duplicate-load-id",
          f"duplicate load ID {render_program_value(load.id)}; "
          f"first declared at {render_program_source(first)}",
          load.source,
        )
      else:
        load_ids[key] = load.source
    _validate_dof_ref(
      load.target,
      label="nodal load",
      source=load.source,
      validator=validator,
    )
    _validate_affine(
      load.value,
      coordinate_names=coordinate_names,
      validator=validator,
    )

  if validator.diagnostics:
    raise ProgramSpecValidationError(validator.diagnostics)

  coordinates = tuple(
    ProgramCoordinateSpec(
      name=item.name,
      kind=item.kind,
      source=_copy_source(item.source),
    )
    for item in sorted(
      snapshot.coordinates,
      key=lambda item: (_COORDINATE_KIND_ORDER[item.kind], item.name),
    )
  )
  coordinate_indices = {item.name: index for index, item in enumerate(coordinates)}
  canonical_constraints: list[ProgramConstraintSpec] = []
  for constraint in sorted(
    snapshot.constraints,
    key=lambda item: _semantic_id_key(item.id),
  ):
    if type(constraint) is PrescribedDofSpec:
      canonical_constraints.append(
        PrescribedDofSpec(
          id=constraint.id,
          target=_copy_dof_ref(constraint.target),
          value=_canonical_affine(constraint.value, coordinate_indices),
          source=_copy_source(constraint.source),
        )
      )
    else:
      canonical_constraints.append(
        AffineTieSpec(
          id=constraint.id,
          slave=_copy_dof_ref(constraint.slave),
          master=_copy_dof_ref(constraint.master),
          factor=constraint.factor,
          offset=_canonical_affine(constraint.offset, coordinate_indices),
          source=_copy_source(constraint.source),
        )
      )
  loads = tuple(
    NodalLoadSpec(
      id=item.id,
      target=_copy_dof_ref(item.target),
      value=_canonical_affine(item.value, coordinate_indices),
      source=_copy_source(item.source),
    )
    for item in sorted(snapshot.loads, key=lambda item: _semantic_id_key(item.id))
  )
  return ProgramSpec(
    coordinates=coordinates,
    constraints=tuple(canonical_constraints),
    loads=loads,
    source=_copy_source(snapshot.source),
  )
