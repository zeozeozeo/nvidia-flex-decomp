# __device_stub__Z14BuildHierarchyiPiPKiS_PViS3_PV14PackedNodeHalfS6_(int, int*, int const*, int*, int volatile*, int volatile*, PackedNodeHalf volatile*, PackedNodeHalf volatile*) — device launch stub

nvcc-generated launch stub for the kernel of the same name (the shipped object pairs it with a
5-byte `_Z<kernel>` tail-jump wrapper). It is toolchain output, not reconstruction source: this
TU's kernels live in `bvh.part1.cu` and nvcc 13.4 emits `__cudaGetKernel`/`__cudaLaunchKernel`
plus a function-local guard where the CUDA 9.2 object calls `cudaLaunchKernel`.

## Spec allowances (recorded)

- `allow_missing_calls: ["cudaLaunchKernel"]` — the 9.2-era call cannot be produced by the
  installed toolchain. The extra `__cuda*` calls are reported by the gate as `extra`
  (non-fatal). Instruction count, signature, literals and layout match.

`parity: needs-gpu` — the stub launches a device kernel.
