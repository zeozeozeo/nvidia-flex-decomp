# __device_stub__Z...RadixSortScanKernel... [compiler-generated]

Unit: `nvflex.sort#_ZL105__device_stub__ZN3cub19RadixSortScanKernelINS_23DeviceRadixSortDispatchILb0EiiiE13PtxScanPolicyEiEEvPT0_iPii`,
original 207 bytes at `.text+0x0`.

This is nvcc's internal launch thunk for the scan kernel (`__cudaPopCallConfiguration` +
`cudaLaunchKernel`). In the shipped object it survived as a separate internal-linkage function; in
CUDA 13.4 the thunk is emitted with a different mangling
(`__device_stub__Z91_...`) and inlined into the kernel's host wrapper, so the original mangled
symbol cannot be produced from source. There is no source-level definition to write.

Disposition: `blocked` — compiler-generated device launch stub (waive reason
`compiler-generated CUDA registration`). Coverage is unaffected: the enclosing kernel unit
`nvflex.sort#_ZN3cub19RadixSortScanKernel...` owns the launch path and is gated on PTX.
