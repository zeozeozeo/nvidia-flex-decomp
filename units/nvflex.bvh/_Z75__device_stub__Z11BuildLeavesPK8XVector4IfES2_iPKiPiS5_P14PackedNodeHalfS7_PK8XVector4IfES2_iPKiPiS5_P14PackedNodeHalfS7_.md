# __device_stub__Z11BuildLeavesPK8XVector4IfES2_iPKiPiS5_P14PackedNodeHalfS7_(XVector4<float> const*, XVector4<float> const*, int, int const*, int*, int*, PackedNodeHalf*, PackedNodeHalf*) — device launch stub

nvcc-generated launch stub for the kernel of the same name (the shipped object pairs it with a
5-byte `_Z<kernel>` tail-jump wrapper). It is toolchain output, not reconstruction source: this
TU's kernels live in `bvh.part1.cu` and nvcc 13.4 emits `__cudaGetKernel`/`__cudaLaunchKernel`
plus a function-local guard where the CUDA 9.2 object calls `cudaLaunchKernel`.

## Spec allowances (recorded)

- `allow_missing_calls: ["cudaLaunchKernel"]` — the 9.2-era call cannot be produced by the
  installed toolchain. The extra `__cuda*` calls are reported by the gate as `extra`
  (non-fatal). Instruction count, signature, literals and layout match.

`parity: needs-gpu` — the stub launches a device kernel.
