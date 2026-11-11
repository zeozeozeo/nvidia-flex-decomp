# _Z20TransformShapeBoundsP8XVector4IfES1_PKS0_PK5XQuatIfES3_S7_PK23NvFlexCollisionGeometryPKiiPK22FlexConvexMeshInternalPK24FlexTriangleMeshInternal

- unit: `nvflex.flex#_Z20TransformShapeBoundsP8XVector4IfES1_PKS0_PK5XQuatIfES3_S7_PK23NvFlexCollisionGeometryPKiiPK22FlexConvexMeshInternalPK24FlexTriangleMeshInternal`
- original: `src/cuda/flex.cu:6009`
- signature: `TransformShapeBounds(XVector4<float>*, XVector4<float>*, XVector4<float> const*, XQuat<float> const*, XVector4<float> const*, XQuat<float> const*, NvFlexCollisionGeometry const*, int const*, int, FlexConvexMeshInternal const*, FlexTriangleMeshInternal const*)`
- parity: kernel (PTX instruction-count band + device call set; symbol, signature and
  referenced-type layouts come from the ELF object).

## Evidence
- device PTX: `work/binary/gpu/dbg/nvflex.flex/sm_30.ptx` (`.entry _Z20TransformShapeBoundsP8XVector4IfES1_PKS0_PK5XQuatIfES3_S7_PK23NvFlexCollisionGeometryPKiiPK22FlexConvexMeshInternalPK24FlexTriangleMeshInternal` body)
- DWARF: `work/dwarf/nvflex.flex.json` (declaration line, parameter types)
- host launch site: `work/decomp/nvflex.flex/cudaflex.c`
- constant-buffer offsets: byte offsets into `__constant__ KernelParams gParams`

## Status
Static gate green (`bin/flexkit gate static --unit 'nvflex.flex#_Z20TransformShapeBoundsP8XVector4IfES1_PKS0_PK5XQuatIfES3_S7_PK23NvFlexCollisionGeometryPKiiPK22FlexConvexMeshInternalPK24FlexTriangleMeshInternal'`).
GPU differential (G3) is exercised only with `FLEX_GPU=1`; this host has no NVIDIA
device, so `flexkit gate gpu` reports `skipped:no-gpu`.
