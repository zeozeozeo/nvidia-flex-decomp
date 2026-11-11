# SortCellIndices(NvFlexLibrary*, int*, int*, int, int)

Unit: `nvflex.sort#_Z15SortCellIndicesP13NvFlexLibraryPiS1_ii` — original `src/cuda/sort.cu:50`, 520 bytes at `.text+0x160`.

## Evidence used

| Fact | Evidence |
|---|---|
| two `cub::DoubleBuffer<int>` locals `(keys, keys+n)` / `(values, values+n)`, selector 0 | `cudasort.disasm.txt` `178: lea (%rsi,%r13,1),%rax` with `r13 = n*4`; stores at `0x40/0x48/0x50` and `0x60/0x68/0x70` |
| first call: query with `d_temp_storage = NULL` | `187: xor %edi,%edi` then `call cub::DeviceRadixSort::SortPairs<int,int>` with `rsi = &sortTempSize@0x30`, `rdx = &d_keys`, `rcx = &d_values`, `r8d = n`, `r9d = 0`, stack `numBits, 0, 0` |
| grow path: `if ((size_t)lib->mSortMaxTempSize < sortTempSize) { SortReset; cudaMalloc(&lib->mSortTemp, size); ... }` | `1f9: movslq 0x220(%rbx),%rax; cmp 0x30(%rsp),%rax; jb 308`; `308: call SortReset`; `315: lea 0x218(%rbx),%rdi; call cudaMalloc`; `lib->mSortMaxTempSize = (int)sortTempSize` |
| second call uses `lib->mSortTemp` | `20b: mov 0x218(%rbx),%rdi` |
| copy back only when the selector moved | `241: movslq 0x50(%rsp),%rax; mov 0x40(%rsp,%rax,8),%rsi; cmp %rsi,%r12; je` |
| copy shape | `cudaMemcpyAsync(dst, src, n*4, 3 /*cudaMemcpyDeviceToDevice*/, 0)` |
| literals | `.LC1` `Cuda error: %d in %s`, `.LC0` `../../../src/cuda/sort.cu`, `.LC13` `cudaMalloc(&lib->mSortTemp, sortTempSize)`, `.LC14` `cudaMemcpyAsync(keys, d_keys.Current(), sizeof(int)*n, cudaMemcpyDeviceToDevice)`, `.LC15` the same for `values` |
| line immediates | `r8d = 0x3f` (63, malloc), `0x47` (71, keys copy), `0x4a` (74, values copy) |

## Reconstruction notes

* `d_keys.Current()` is used for the copy-back comparisons and sources, exactly as the original
  (`d_buffers[selector]` at `0x40 + selector*8`).
* `sizeof(int)*n` is evaluated as a 64-bit size (`movslq %ecx,%r13; shl $0x2`).
* All five literals are byte-identical to the shipped `.rodata`. No literal allowlist is needed.
* The dispatch itself (`cub::DeviceRadixSort::SortPairs<int,int>`, weak, 0x41e1 bytes in the shipped object)
  is reconstructed in `recon/src/cuda/sort.cu` with the same mangled name and marked
  `__attribute__((noinline, noclone))` so `SortCellIndices` stays a thin driver (127 original / 154
  reconstructed instructions, ratio 1.21).
* `lib->mSortTemp` / `lib->mSortMaxTempSize` are reached through `SortTempSlot` / `SortTempSizeSlot`
  (typed-vs-opaque tolerant accessors, see the `SortReset` note); DWARF offsets 536/544 are unchanged.
* Temp-storage protocol: query returns `grid_size * 32 * sizeof(int)`; the second call launches the
  reconstructed upsweep/scan/downsweep and flips both selectors per pass, matching CUB's contract.

Inference only: member-by-member the original is the standard CUB `SortPairs` call sequence; the
`DoubleBuffer`/`GridEvenShare` reconstructions live in `sort.cu` (CUB 1.3.2 cannot be compiled for sm_75 —
see `nvflex.sort-cub-replacement.md`).

Parity: `needs-gpu` (G3 on a GPU host); static gate green (layout probe: `sizeof(NvFlexLibrary)` 648,
`mResourceTracker` 552).
