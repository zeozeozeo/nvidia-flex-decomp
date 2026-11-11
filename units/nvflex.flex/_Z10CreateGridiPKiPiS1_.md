# CreateGrid(int, int const*, int*, int*)

- unit: `nvflex.flex#_Z10CreateGridiPKiPiS1_`
- source: src/cuda/flex.cu:740
- PTX: `work/binary/gpu/dbg/nvflex.flex/sm_30.ptx` lines 251-312
- launch: block [256, 1, 1], grid ceil(numParticles/256) (from the host launch site or the shipped stub)

## Behaviour

Run-length pass over the sorted cell ids: emits cellStarts[cell]/cellEnds[prevCell] on a boundary and cellEnds[cell] = n at the last index (PTX 251-312).

## Evidence

- ptx:251-312

## Inference only

- none

## Gate

- static: PTX branch, instruction ratio inside [0.4, 2.5], no PTX call operands on either side
- gpu: `skipped:no-gpu` on this host (no NVIDIA device); the kernel is reconstructed from the shipped sm_30 PTX and launches with the block size recorded above
