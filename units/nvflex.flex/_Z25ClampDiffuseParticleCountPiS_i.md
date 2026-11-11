# _Z25ClampDiffuseParticleCountPiS_i

- unit: `nvflex.flex#_Z25ClampDiffuseParticleCountPiS_i`
- original: `src/cuda/flex.cu:3008`
- signature: `ClampDiffuseParticleCount(int*, int*, int)`
- parity: kernel (PTX instruction-count band + device call set; symbol, signature and
  referenced-type layouts come from the ELF object).

## Evidence
- device PTX: `work/binary/gpu/dbg/nvflex.flex/sm_30.ptx` (`.entry _Z25ClampDiffuseParticleCountPiS_i` body)
- DWARF: `work/dwarf/nvflex.flex.json` (declaration line, parameter types)
- host launch site: `work/decomp/nvflex.flex/cudaflex.c`
- constant-buffer offsets: byte offsets into `__constant__ KernelParams gParams`

## Status
Static gate green (`bin/flexkit gate static --unit 'nvflex.flex#_Z25ClampDiffuseParticleCountPiS_i'`).
GPU differential (G3) is exercised only with `FLEX_GPU=1`; this host has no NVIDIA
device, so `flexkit gate gpu` reports `skipped:no-gpu`.
