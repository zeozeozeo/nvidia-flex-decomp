# FleX decomp

Reverse engineering the [NVIDIA FleX](https://developer.nvidia.com/flex) library.

## Progress

Currently decompiled:

- CUDA kernels
- HLSL shaders from DXBC
- Core device methods

Left to decompile:

- DX11 backend (started)

## Modules

Main parts of the code.

- [recon](./recon) source root
- [recon/src/cuda](./recon/src/cuda) CUDA kernel sources
- [recon/src/device](./recon/src/device) CPU device host
- [recon/src/device](./recon/src/dx/shaders) HLSL compute shader sources
- [recon/src/dx](./recon/src/dx) DX11 backend (WIP)
