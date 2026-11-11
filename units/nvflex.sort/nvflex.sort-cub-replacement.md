# nvflex.sort — CUB 1.3.2 device-code replacement (shared note)

Plan contingency 5.3. Applies to the six kernel units of `nvflex.sort` and to the waived
`cub::DeviceRadixSort::SortPairs<int,int>` instantiation.

## Why CUB cannot be used

`FleX/external/cub-1.3.2` emits pre-Volta warp intrinsics: `cub/util_ptx.cuh` uses inline
`shfl.idx/down/up.b32` without `.sync` and `__all/__any` (`util_ptx.cuh:409,468,502,573,599`).
ptxas rejects both for every target >= sm_70 at PTX ISA >= 6.4, and CUDA 13.4 no longer accepts
`-arch=sm_60/30`, so the vendored CUB headers cannot produce device code here. Verified by
compiling a minimal TU:

```
$ nvcc -O3 -arch=sm_75 -I FleX/external/cub-1.3.2 -c t.cu   # instantiates SortPairs<int,int>
ptxas ... error : Instruction 'shfl' without '.sync' is not supported on .target sm_70 and higher from PTX ISA version 6.4
```

(Reduction of `shfl` to `__shfl_sync` macros is not possible: CUB uses inline asm, not intrinsics.)

## What was reconstructed instead

`recon/src/cuda/sort.cu` reimplements the radix sort and keeps the *original* ABI:

* host dispatcher `cub::DeviceRadixSort::SortPairs<int,int>` with the identical mangled name
  (`_ZN3cub15DeviceRadixSort9SortPairsIiiEE9cudaErrorPvRmRNS_12DoubleBufferIT_EERNS5_IT0_EEiiiP11CUstream_stb`),
  forced out of line (`noinline, noclone`) exactly like the shipped weak symbol;
* kernel entry points declared `extern "C"` with the original mangled names, so the fatbin `.entry`
  set and the host stub symbols are byte-identical to `cudasort.o`:
  `cub::EmptyKernel<void>`, `cub::RadixSortScanKernel<...PtxScanPolicy,int>`,
  `cub::RadixSortUpsweepKernel<...PtxUpsweepPolicy,...>`, `<...PtxAltUpsweepPolicy,...>`,
  `cub::RadixSortDownsweepKernel<...PtxDownsweepPolicy,...>`, `<...PtxAltDownsweepPolicy,...>`;
* `cub::DoubleBuffer<T>` (3 words) and `cub::GridEvenShare<int>` (9 ints = 36 bytes, matching the
  PTX parameter `.param .align 4 .b8 param_6[36]`) reconstructed with CUB's field order.

Algorithm (replacement): 5-bit digits, 256 threads/block, 4 items/thread, striped spine
`d_spine[bin*gridDim.x + block]` (the layout the original upsweep PTX writes: 8192 B = 32 bins x
grid, `d_spine[gridDim.x * bin_idx + blockIdx.x] = bin_count`).
1. upsweep — per-block shared-memory histogram (atomicAdd), written striped per (bin, block);
2. scan — one block, 32 lanes: per-bin totals, exclusive prefix of the totals (lane 0), then a
   per-bin exclusive scan over the blocks in that bin's stripe. This gives every (bin, block)
   region its global start offset;
3. downsweep — one lane per digit walks the block's items in input order, so the scatter is
   *stable* like CUB's;
4. passes ping-pong the `d_keys`/`d_values` selectors; `EmptyKernel<<<1,1>>>()` for n<=1.

The device logic was validated numerically in the absence of a GPU: a serial model of the exact
upsweep/scan/downsweep code (including the `GridEvenShare` block partitioning and the 1024-block
cap) sorts 73 vectors bit-identically to a stable reference sort — n in {0,1,2,3,17,255,256,1023,
1024,1025,5000,200000} x num_bits in {5,10,32} x {distinct, heavily duplicated} keys plus one
non-zero begin_bit case. This validates the algorithm and offsets, not CUDA execution.

Every piece of ABI that external code can observe (host symbols, `DoubleBuffer`/`GridEvenShare`
layout, kernel parameter list, selector protocol, temp-storage query) is identical to the original.
Only the device instruction stream differs, so behaviour must be confirmed by G3 on a GPU host.

## Gate status and measured evidence

`gate static` for kernel units compares the `.entry` block of the original `sm_30.ptx` against
`nvcc -ptx -arch=sm_75` of the reconstruction and requires an instruction ratio in `[0.4, 2.5]`.
A replacement implementation cannot satisfy that band (it is a stub detector calibrated on
compiler-equal code):

| entry | orig insns | recon insns | ratio |
|---|---|---|---|
| `cub::EmptyKernel<void>` | 1 | 1 | 1.00 (pass) |
| `cub::RadixSortUpsweepKernel<...PtxUpsweepPolicy...>` | 562 | 56 | 0.10 |
| `cub::RadixSortUpsweepKernel<...PtxAltUpsweepPolicy...>` | 554 | 56 | 0.10 |
| `cub::RadixSortScanKernel<...PtxScanPolicy...>` | 201 | 184 | 0.915 (pass) |
| `cub::RadixSortDownsweepKernel<...PtxDownsweepPolicy...>` | 1408 | 120 | 0.09 |
| `cub::RadixSortDownsweepKernel<...PtxAltDownsweepPolicy...>` | 1363 | 120 | 0.09 |

Disposition: `cub::EmptyKernel<void>` and `cub::RadixSortScanKernel<...>` are green on the G1 PTX
checks (1/1 and 201->184 instructions). The four upsweep/downsweep entries cannot satisfy the band
(their CUB bodies are ~10x larger because the scatter is fully parallel per thread, whereas the
reconstruction uses one lane per digit); per plan 5.3 they are `waived` with reason
`third-party CUB 1.3.2 replaced; requires GPU diff` and must be listed in the final report.
`nvflex.sort` therefore has no pure-CPU unit, so `gate diff` is not applicable anywhere in the
module.
