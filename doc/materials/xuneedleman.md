# XuNeedleman

`XuNeedleman` is a cohesive interface material model defining traction–separation behavior with parameters governing peak traction and fracture energy. The remaining rank-2 surface outputs the normal and shear traction components `Tn` and `Ts`; M80 (commit `503cefb`) removed the legacy rank-3 branch.

## Overview
- **Material type:** `XuNeedleman`

## Parameters
### Mandatory
- `type`: Must be set to `"XuNeedleman"`
- `Gc`: Fracture energy
- `Tult`: Ultimate traction
