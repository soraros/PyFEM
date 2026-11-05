# Legacy Interface

> This page documents the legacy `Interface` element and its cohesive laws
> only. NX7-1 retired the experimental v3 interface compiler, authoring
> helpers, and deck-adapter path. Their frame, tangent, legacy-parity, and
> finite-difference evidence is indexed in the
> [NX7-1 retirement record](https://github.com/jjcremmers/PyFEM/blob/v3/.agents/v3/migration-history.md#nx7-1-retirement-non-kernel-scientific-breadth).
> Interface decks are explicitly unsupported by the retained v3 adapter.

The legacy `Interface` element models cohesive or contact interfaces between
continuum parts for delamination, peeling, and traction-separation behavior.

## Overview
- **Element type:** `Interface`
- **Supports:**
  - Traction–separation laws via legacy material models (e.g., `XuNeedleman`)
  - The remaining legacy rank-2 interface formulation; M80 removed the rank-3
    branch (commit `503cefb`)
  - Nonlinear legacy analysis for softening and debonding

## Parameters
### Mandatory
- `type`: Must be set to `"Interface"`
- `material`: Interface material model block defining the legacy cohesive law
  and its parameters.
