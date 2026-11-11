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

Algorithm (replacement): 5-bit digits, 256 threads/block, 4 items/thread, spine layout
`d_spine[block*32 + bin]` owned by the dispatcher.
1. upsweep — per-block shared-memory histogram, written per (block,bin);
2. scan — one block, 32 lanes, exclusive scan of each bin over blocks;
3. downsweep — one lane per digit walks the block's items in input order, so the scatter is
   *stable* like CUB's;
4. passes ping-pong `d_keys`/`d_values` selectors, `EmptyKernel<<<1,1>>>()` for n<=1.

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
| `cub::RadixSortScanKernel<...PtxScanPolicy...>` | 201 | 52 | 0.26 |
| `cub::RadixSortDownsweepKernel<...PtxDownsweepPolicy...>` | 1408 | 120 | 0.09 |
| `cub::RadixSortDownsweepKernel<...PtxAltDownsweepPolicy...>` | 1363 | 120 | 0.09 |

Per plan 5.3 the correct disposition for these five is `waived` with reason
`third-party CUB 1.3.2 replaced; requires GPU diff`; `flexkit units` currently exposes no waive
subcommand, so they are recorded as `blocked` with that evidence and must be listed in the final
report. `cub::EmptyKernel<void>` is green.
