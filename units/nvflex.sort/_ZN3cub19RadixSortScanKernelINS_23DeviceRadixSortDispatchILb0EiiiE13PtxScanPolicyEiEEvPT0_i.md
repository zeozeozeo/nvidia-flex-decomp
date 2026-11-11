# _ZN3cub19RadixSortScanKernelINS_23DeviceRadixSortDispatchILb0EiiiE13PtxScanPolicyEiEEvPT0_i

Kernel unit of `nvflex.sort`, instantiated by `cub::DeviceRadixSort::SortPairs<int,int>`.

* Original device code: `work/binary/gpu/dbg/nvflex.sort/sm_30.ptx` (`.entry _ZN3cub19RadixSortScanKernelINS_23DeviceRadixSortDispatchILb0EiiiE13PtxScanPolicyEiEEvPT0_i`,
  parameter list verified against the reconstruction's `cudaLaunchKernel` packing).
* Reconstruction: `recon/src/cuda/sort.cu`, `extern "C" __global__` definition with this exact
  mangled entry name.
* CUB 1.3.2 cannot be built for sm_75 (pre-Volta `shfl`/`vote`), so the body is a replacement;
  ABI (entry name, parameter list, spine/selector protocol) is preserved. See
  `nvflex.sort-cub-replacement.md` for the algorithm and the measured PTX instruction ratios.
* Gate: `gate static` PTX band fails by construction for a replacement implementation; the unit is
  recorded `blocked` per plan 5.3 pending G3 on a GPU host (waiver reason
  `third-party CUB 1.3.2 replaced; requires GPU diff`).
