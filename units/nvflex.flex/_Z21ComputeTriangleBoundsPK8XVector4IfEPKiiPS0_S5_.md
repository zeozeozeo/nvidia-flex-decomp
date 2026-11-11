# _Z21ComputeTriangleBoundsPK8XVector4IfEPKiiPS0_S5_

- unit: `nvflex.flex#_Z21ComputeTriangleBoundsPK8XVector4IfEPKiiPS0_S5_`
- original: `src/cuda/flex.cu:6187`
- signature: `ComputeTriangleBounds(XVector4<float> const*, int const*, int, XVector4<float>*, XVector4<float>*)`
- parity: kernel (PTX instruction-count band + device call set; symbol, signature and
  referenced-type layouts come from the ELF object).

## Evidence
- device PTX: `work/binary/gpu/dbg/nvflex.flex/sm_30.ptx` (`.entry _Z21ComputeTriangleBoundsPK8XVector4IfEPKiiPS0_S5_` body)
- DWARF: `work/dwarf/nvflex.flex.json` (declaration line, parameter types)
- host launch site: `work/decomp/nvflex.flex/cudaflex.c`
- constant-buffer offsets: byte offsets into `__constant__ KernelParams gParams`

## Status
Static gate green (`bin/flexkit gate static --unit 'nvflex.flex#_Z21ComputeTriangleBoundsPK8XVector4IfEPKiiPS0_S5_'`).
GPU differential (G3) is exercised only with `FLEX_GPU=1`; this host has no NVIDIA
device, so `flexkit gate gpu` reports `skipped:no-gpu`.
