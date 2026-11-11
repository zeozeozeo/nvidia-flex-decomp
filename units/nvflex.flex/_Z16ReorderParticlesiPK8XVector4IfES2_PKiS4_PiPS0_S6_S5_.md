# ReorderParticles(int, XVector4<float> const*, XVector4<float> const*, int const*, int const*, int*, XVector4<float>*, XVector4<float>*, int*)

- unit: `nvflex.flex#_Z16ReorderParticlesiPK8XVector4IfES2_PKiS4_PiPS0_S6_S5_`
- source: src/cuda/flex.cu:946
- PTX: `work/binary/gpu/dbg/nvflex.flex/sm_30.ptx` lines 315-379
- launch: block [256, 1, 1], grid ceil(numParticles/256) (from the host launch site or the shipped stub)

## Behaviour

sortedX[i] = X[indices[i]] for positions/newPositions/phases; reverseLookup[indices[i]] = i (PTX 315-379).

## Evidence

- ptx:315-379
- decomp:10872 (launch)
- decomp:3161 (DWARF parameter names)

## Inference only

- none

## Gate

- static: PTX branch, instruction ratio inside [0.4, 2.5], no PTX call operands on either side
- gpu: `skipped:no-gpu` on this host (no NVIDIA device); the kernel is reconstructed from the shipped sm_30 PTX and launches with the block size recorded above
