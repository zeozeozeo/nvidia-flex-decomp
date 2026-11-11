# _Z15UpdateTrianglesi8Vec3IterS_PKiS1_P8XVector3IfES3_S_f

- unit: `nvflex.flex#_Z15UpdateTrianglesi8Vec3IterS_PKiS1_P8XVector3IfES3_S_f`
- original: `src/cuda/flex.cu:2094`
- signature: `UpdateTriangles(int, Vec3Iter, Vec3Iter, int const*, int const*, XVector3<float>*, XVector3<float>, Vec3Iter, float)`
- parity: kernel (PTX instruction-count band + device call set; symbol, signature and
  referenced-type layouts come from the ELF object).

## Evidence
- device PTX: `work/binary/gpu/dbg/nvflex.flex/sm_30.ptx` (`.entry _Z15UpdateTrianglesi8Vec3IterS_PKiS1_P8XVector3IfES3_S_f` body)
- DWARF: `work/dwarf/nvflex.flex.json` (declaration line, parameter types)
- host launch site: `work/decomp/nvflex.flex/cudaflex.c`
- constant-buffer offsets: byte offsets into `__constant__ KernelParams gParams`

## Status
Static gate green (`bin/flexkit gate static --unit 'nvflex.flex#_Z15UpdateTrianglesi8Vec3IterS_PKiS1_P8XVector3IfES3_S_f'`).
GPU differential (G3) is exercised only with `FLEX_GPU=1`; this host has no NVIDIA
device, so `flexkit gate gpu` reports `skipped:no-gpu`.
