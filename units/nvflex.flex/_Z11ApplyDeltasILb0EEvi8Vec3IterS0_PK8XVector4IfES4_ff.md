# ApplyDeltas<false>(int, Vec3Iter, Vec3Iter, XVector4<float> const*, XVector4<float> const*, float, float)

- unit: `nvflex.flex#_Z11ApplyDeltasILb0EEvi8Vec3IterS0_PK8XVector4IfES4_ff`
- source: src/cuda/flex.cu (no DWARF line)
- PTX: `work/binary/gpu/dbg/nvflex.flex/sm_30.ptx` lines 19654-19698
- launch: block [256, 1, 1], grid ceil(numParticles/256) (from the host launch site or the shipped stub)

## Behaviour

newPositions[i].xyz += deltas[i].xyz * relaxation (no contact-count normalisation; damp unused) (PTX 19654-19698).

## Evidence

- ptx:19654-19698

## Inference only

- none

## Gate

- static: PTX branch, instruction ratio inside [0.4, 2.5], no PTX call operands on either side
- gpu: `skipped:no-gpu` on this host (no NVIDIA device); the kernel is reconstructed from the shipped sm_30 PTX and launches with the block size recorded above
