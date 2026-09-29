"""Data-declared periodic pairings for affine constraint maps.

Periodic and RVE-style boundary conditions are ordinary one-master affine
ties: the image side follows the primary side plus a (possibly
signal-dependent) offset. A researcher declares the pairing as data — two
node sequences, a field, its components, and per-component offsets — and this
helper expands it into ``AffineTieSpec`` declarations that
:func:`pyfem.v3.constraints.compile_constraint_map` compiles into the same
canonical coordinate map as every other affine constraint. No core edits, no
constrainer rebuilds.
"""

from __future__ import annotations

import math

from pyfem.v3.model.operator import SemanticId
from pyfem.v3.spec.diagnostics import SourceContext
from pyfem.v3.spec.program import AffineTieSpec, AffineValueSpec, DofRef


def _node_ids(value: object, *, label: str) -> tuple[SemanticId, ...]:
  if type(value) is not tuple or any(type(item) not in (str, int) for item in value):
    msg = f"{label} must be an exact tuple of semantic node ids"
    raise TypeError(msg)
  return value


def periodic_ties(
  *,
  primary_node_ids: tuple[SemanticId, ...],
  image_node_ids: tuple[SemanticId, ...],
  field_id: SemanticId,
  components: tuple[str, ...],
  offsets: tuple[AffineValueSpec, ...] = (),
  factor: int | float = 1.0,
  id_prefix: str = "periodic",
  source: SourceContext = SourceContext(),
) -> tuple[AffineTieSpec, ...]:
  """Expand a periodic pairing declaration into affine tie declarations.

  Each pair ``(primary, image)`` and each component yields one tie
  ``u_image = factor * u_primary + offset``. ``offsets`` carries one
  :class:`AffineValueSpec` per component (empty means homogeneous offsets
  everywhere); signal-scaled offsets declare RVE-style macroscopic strain
  coordinates directly as data. Ties are emitted in pair-major, then
  component, order with ids ``"<id_prefix>:<pair index>:<component>"``.
  """
  primary = _node_ids(primary_node_ids, label="primary node ids")
  image = _node_ids(image_node_ids, label="image node ids")
  if not primary:
    msg = "periodic pairings require at least one node pair"
    raise ValueError(msg)
  if len(primary) != len(image):
    msg = "periodic pairings require equally many primary and image nodes"
    raise ValueError(msg)
  if len(set(primary)) != len(primary):
    msg = "periodic pairings cannot repeat a primary node"
    raise ValueError(msg)
  if len(set(image)) != len(image):
    msg = "periodic pairings cannot repeat an image node"
    raise ValueError(msg)
  if set(primary) & set(image):
    msg = "one node cannot be both primary and image in a periodic pairing"
    raise ValueError(msg)
  if type(field_id) not in (str, int):
    msg = "periodic pairing field id must be an exact semantic id"
    raise TypeError(msg)
  if type(components) is not tuple or any(
    type(component) is not str or not component for component in components
  ):
    msg = "periodic pairing components must be non-empty exact strings"
    raise TypeError(msg)
  if not components:
    msg = "periodic pairings require at least one field component"
    raise ValueError(msg)
  if type(offsets) is not tuple or any(
    type(offset) is not AffineValueSpec for offset in offsets
  ):
    msg = "periodic pairing offsets must be exact AffineValueSpec values"
    raise TypeError(msg)
  if offsets and len(offsets) != len(components):
    msg = "periodic pairing offsets must align one-to-one with the components"
    raise ValueError(msg)
  if type(factor) is not int and type(factor) is not float:
    msg = "periodic pairing factor must be an exact int or float"
    raise TypeError(msg)
  if not math.isfinite(float(factor)):
    msg = "periodic pairing factor must be finite"
    raise ValueError(msg)
  if type(id_prefix) is not str or not id_prefix:
    msg = "periodic pairing id prefix must be a non-empty exact string"
    raise TypeError(msg)
  if type(source) is not SourceContext:
    msg = "periodic pairings require an exact SourceContext"
    raise TypeError(msg)

  homogeneous = AffineValueSpec()
  ties: list[AffineTieSpec] = []
  for pair_index, (primary_id, image_id) in enumerate(zip(primary, image, strict=True)):
    for component_index, component in enumerate(components):
      ties.append(
        AffineTieSpec(
          id=f"{id_prefix}:{pair_index}:{component}",
          slave=DofRef(
            node_id=image_id,
            field_id=field_id,
            component=component,
          ),
          master=DofRef(
            node_id=primary_id,
            field_id=field_id,
            component=component,
          ),
          factor=factor,
          offset=offsets[component_index] if offsets else homogeneous,
          source=source,
        )
      )
  return tuple(ties)
