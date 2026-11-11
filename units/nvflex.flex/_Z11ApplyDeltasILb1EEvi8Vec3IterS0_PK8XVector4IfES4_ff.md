# ApplyDeltas<true>(int, Vec3Iter, Vec3Iter, XVector4<float> const*, XVector4<float> const*, float, float)

- unit: `nvflex.flex#_Z11ApplyDeltasILb1EEvi8Vec3IterS0_PK8XVector4IfES4_ff`
- source: src/cuda/flex.cu (no DWARF line)
- PTX: `work/binary/gpu/dbg/nvflex.flex/sm_30.ptx` lines 19603-19651
- launch: block [256, 1, 1], grid ceil(numParticles/256) (from the host launch site or the shipped stub)

## Behaviour

newPositions[i].xyz += deltas[i].xyz * (1/max(deltas[i].w*relaxation, damp)); returns early when deltas is null. deltas.w is the contact count from SolveDensities (PTX 19603-19651).

## Evidence

- ptx:19603-19651
- decomp:13586 (DWARF parameter names)
- decomp:12757 (launch site)

## Inference only

- relaxation/damp names follow DWARF; the parameter pair is unused by the <false> variant

## Gate

- static: PTX branch, instruction ratio inside [0.4, 2.5], no PTX call operands on either side
- gpu: `skipped:no-gpu` on this host (no NVIDIA device); the kernel is reconstructed from the shipped sm_30 PTX and launches with the block size recorded above
