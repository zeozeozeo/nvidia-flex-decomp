# _Z13CollideShapes22FlexCollisionShapeData8Vec3IterPKiiPiP8XVector4IfES6_ff

- unit: `nvflex.flex#_Z13CollideShapes22FlexCollisionShapeData8Vec3IterPKiiPiP8XVector4IfES6_ff`
- original: `src/cuda/flex.cu:3622`
- signature: `CollideShapes(FlexCollisionShapeData, Vec3Iter, int const*, int, int*, XVector4<float>*, XVector4<float>*, float, float)`
- parity: kernel (PTX instruction-count band + device call set; symbol, signature and
  referenced-type layouts come from the ELF object).

## Evidence
- device PTX: `work/binary/gpu/dbg/nvflex.flex/sm_30.ptx` (`.entry _Z13CollideShapes22FlexCollisionShapeData8Vec3IterPKiiPiP8XVector4IfES6_ff` body)
- DWARF: `work/dwarf/nvflex.flex.json` (declaration line, parameter types)
- host launch site: `work/decomp/nvflex.flex/cudaflex.c`
- constant-buffer offsets: byte offsets into `__constant__ KernelParams gParams`

## Status
Static gate green (`bin/flexkit gate static --unit 'nvflex.flex#_Z13CollideShapes22FlexCollisionShapeData8Vec3IterPKiiPiP8XVector4IfES6_ff'`).
GPU differential (G3) is exercised only with `FLEX_GPU=1`; this host has no NVIDIA
device, so `flexkit gate gpu` reports `skipped:no-gpu`.
