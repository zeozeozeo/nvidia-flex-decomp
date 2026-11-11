# _Z15SmoothPositionsiPK8XVector4IfEPKiS4_S4_S4_PS0_S5_

- unit: `nvflex.flex#_Z15SmoothPositionsiPK8XVector4IfEPKiS4_S4_S4_PS0_S5_`
- original: `src/cuda/flex.cu:2546`
- signature: `SmoothPositions(int, XVector4<float> const*, int const*, int const*, int const*, int const*, XVector4<float>*, XVector4<float>*)`
- parity: kernel (PTX instruction-count band + device call set; symbol, signature and
  referenced-type layouts come from the ELF object).

## Evidence
- device PTX: `work/binary/gpu/dbg/nvflex.flex/sm_30.ptx` (`.entry _Z15SmoothPositionsiPK8XVector4IfEPKiS4_S4_S4_PS0_S5_` body)
- DWARF: `work/dwarf/nvflex.flex.json` (declaration line, parameter types)
- host launch site: `work/decomp/nvflex.flex/cudaflex.c`
- constant-buffer offsets: byte offsets into `__constant__ KernelParams gParams`

## Status
Static gate green (`bin/flexkit gate static --unit 'nvflex.flex#_Z15SmoothPositionsiPK8XVector4IfEPKiS4_S4_S4_PS0_S5_'`).
GPU differential (G3) is exercised only with `FLEX_GPU=1`; this host has no NVIDIA
device, so `flexkit gate gpu` reports `skipped:no-gpu`.
