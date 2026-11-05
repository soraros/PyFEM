# Legacy Contact Mechanics Model

> This page documents the legacy contact model only. NX7-1 retired the
> experimental v3 contact compiler and authoring helper. Their independent
> legacy, finite-difference, and tampered-tangent evidence is indexed in the
> [NX7-1 retirement record](https://github.com/jjcremmers/PyFEM/blob/v3/.agents/v3/migration-history.md#nx7-1-retirement-non-kernel-scientific-breadth).

## Overview
The legacy contact model implements a penalty-based algorithm for mechanical interactions between deformable bodies and a moving rigid disc in 2D. A simple penalty method enforces non-penetration constraints.

The legacy model is a special model type added to a two-dimensional structural analysis to impose contact constraints between the mesh and the moving obstacle. Its 3D sphere-contact branch was removed by M80 (commit `435244b`). No v3 contact compiler or 3D diagnostic is advertised here.

## Theory
### Penalty Method
The contact algorithm uses the penalty method to enforce contact constraints. When a node penetrates a rigid obstacle, a penalty force is applied:

$$
\mathbf{f}_c = k_p \, g \, \mathbf{n}
$$
where:
- $k_p$ is the penalty stiffness parameter
- $g$ is the penetration depth (overlap)
- $\mathbf{n}$ is the outward normal direction from the obstacle surface

### Contact Detection
For a rigid disc in 2D with center $\mathbf{c}$ and radius $R$, contact is detected when:

$$
g = R - \|\mathbf{x} - \mathbf{c}\| > 0
$$
where $\mathbf{x}$ is the current position of a node. The penetration depth $g$ measures how far the node has penetrated into the obstacle.

The outward normal from the obstacle surface is:

(See documentation for further details)
