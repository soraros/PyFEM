# v3 architecture

## Data flow

```mermaid
flowchart LR
  load[load_problem] --> loaded[LoadedProblem]
  loaded --> meta[strings registry]
  loaded --> problem[ProblemDefinition NamedTuple]
  problem --> assemble[assemble_linear_system]
  assemble --> solve[solve_linear]
```

## `ProblemDefinition` (NamedTuple)

Array-only bundle for assembly and future `@njit` kernels:

| Field | Type | Role |
|-------|------|------|
| `coords` | `F64` (N, dim) | Node coordinates |
| `conn` | `I32` (E, nodes) | Element connectivity |
| `global_dofs` | `I32` (n_dof,) | Global DOF numbering |
| `constitutive` | `F64` (3, 3) | Plane-stress D matrix |
| `constraint_dof` | `I32` | Prescribed DOF indices |
| `constraint_val` | `F64` | Prescribed values |

Properties: `n_nodes`, `n_elems`, `n_dofs`.

Built by `pack_problem()` from load-time `Mesh` / `DofMap`. No jitclass layer — pass the NamedTuple or unpacked arrays to Numba when needed.

## `LoadedProblem` (dataclass)

Holds `problem: ProblemDefinition` plus metadata (`element_type`, `material_type`, `mesh_path`, etc.) for registry dispatch. Strings and paths stay out of the jitable bundle.

## FEM stack

1. **Quadrature** — `@njit` Gauss–Legendre (`@overload` + compile-time literal `order`); 2D rule composes 1D with `_meshgrid_2d` (loop tensor product, no `meshgrid`).
2. **Shapes** — `@njit` Serendipity Q8 `N`, `∂N/∂ξ`.
3. **Kinematics** — `@njit` `B = ∂N/∂x` via `∂N/∂ξ · J⁻¹` (batched ranks).
4. **Element** — `@njit` batched `K_e = Σ w |J| Bᵀ C B`; Python wrapper adds a leading axis for one element.
5. **Assembly** — batched element matrices → COO global stiffness.

## Solver

- Global system: SciPy `coo_array` + `spsolve`.
- BCs: legacy `Constrainer` bridge (`solver/constraints.py`) until native v3 constraints land.

## Skims

See [parity.md](parity.md). Layer C is always `ProblemDefinition`.
