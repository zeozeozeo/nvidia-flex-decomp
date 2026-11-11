# ContinuousShockPropagation(int, XVector4<float>*, int*)

- unit: `nvflex.flex#_Z26ContinuousShockPropagationiP8XVector4IfEPi`
- source: src/cuda/flex.cu:1087
- PTX: `work/binary/gpu/dbg/nvflex.flex/sm_30.ptx` lines 958-1073
- launch: block [256, 1, 1], grid ceil(numParticles/256) (from the host launch site or the shipped stub)

## Behaviour

thread 0 picks the bounds corner furthest along gravity as __shared__ minPoint and minDir = normalized gravity; every particle scales pos.w by expf(kShockPropagation * dot(minPoint - pos, minDir)) (PTX 958-1073).

## Evidence

- ptx:958-1073
- decomp:11101 (launch)
- recovered.h KernelParams offset 328 = kShockPropagation, 260 = kGravity

## Inference only

- CUDA 9.2 lowers expf to the trunc/ln2-hi-lo/two-ex2 form with +-105 clamps (PTX 1040-1062); CUDA 13 uses a different but equivalent expf lowering

## Gate

- static: PTX branch, instruction ratio inside [0.4, 2.5], no PTX call operands on either side
- gpu: `skipped:no-gpu` on this host (no NVIDIA device); the kernel is reconstructed from the shipped sm_30 PTX and launches with the block size recorded above
