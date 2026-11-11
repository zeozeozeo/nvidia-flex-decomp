# InitActiveIndices(int, int*)

- unit: `nvflex.flex#_Z17InitActiveIndicesiPi`
- source: src/cuda/flex.cu:680
- PTX: `work/binary/gpu/dbg/nvflex.flex/sm_30.ptx` lines 43-69
- launch: block [256, 1, 1], grid ceil(numParticles/256) (from the host launch site or the shipped stub)

## Behaviour

indices[i] = i for i < numParticles (PTX 43-69).

## Evidence

- ptx:43-69
- decomp:6715 (launch site, block 256)
- xxd: flex.cu:680 in DWARF

## Inference only

- parameter names are DWARF-less (__cuda_0/1); derived from the launch site

## Gate

- static: PTX branch, instruction ratio inside [0.4, 2.5], no PTX call operands on either side
- gpu: `skipped:no-gpu` on this host (no NVIDIA device); the kernel is reconstructed from the shipped sm_30 PTX and launches with the block size recorded above
