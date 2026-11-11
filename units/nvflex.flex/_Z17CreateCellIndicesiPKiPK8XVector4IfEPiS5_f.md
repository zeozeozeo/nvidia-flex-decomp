# CreateCellIndices(int, int const*, XVector4<float> const*, int*, int*, float)

- unit: `nvflex.flex#_Z17CreateCellIndicesiPKiPK8XVector4IfEPiS5_f`
- source: src/cuda/flex.cu:725
- PTX: `work/binary/gpu/dbg/nvflex.flex/sm_30.ptx` lines 182-248
- launch: block [256, 1, 1], grid ceil(numParticles/256) (from the host launch site or the shipped stub)

## Behaviour

cellIndices[i] = (cz<<14)|(cy<<7)|cx with c = int((pos - gBounds.kLower)*invCellSize) & 127; particleIndices[i] = indices[i] (PTX 182-248).

## Evidence

- ptx:182-248
- decomp:10513-10514 (6th arg = _LC3/(radius+margin); _LC3 == 1.0f in .rodata)
- recovered.h KernelBounds kLower

## Inference only

- invCellSize (the shipped source multiplied by 1/cellSize)

## Gate

- static: PTX branch, instruction ratio inside [0.4, 2.5], no PTX call operands on either side
- gpu: `skipped:no-gpu` on this host (no NVIDIA device); the kernel is reconstructed from the shipped sm_30 PTX and launches with the block size recorded above
