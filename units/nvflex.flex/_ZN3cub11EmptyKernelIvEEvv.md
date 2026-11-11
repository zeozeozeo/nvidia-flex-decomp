# cub::EmptyKernel<void>()

- unit: `nvflex.flex#_ZN3cub11EmptyKernelIvEEvv`
- source: src/cuda/flex.cu (no DWARF line)
- PTX: `work/binary/gpu/dbg/nvflex.flex/sm_30.ptx` lines 22746-22753
- launch: block [1, 1, 1], grid [1,1,1] (from the host launch site or the shipped stub)

## Behaviour

Empty kernel (CUB 1.3.2 util_device.cuh) instantiated for void; the shipped body is an empty ret (PTX 22746-22753).

## Evidence

- ptx:22746-22753
- FleX/external/cub-1.3.2/cub/util_device.cuh (template definition)

## Inference only

- none

## Gate

- static: PTX branch, instruction ratio inside [0.4, 2.5], no PTX call operands on either side
- gpu: `skipped:no-gpu` on this host (no NVIDIA device); the kernel is reconstructed from the shipped sm_30 PTX and launches with the block size recorded above
