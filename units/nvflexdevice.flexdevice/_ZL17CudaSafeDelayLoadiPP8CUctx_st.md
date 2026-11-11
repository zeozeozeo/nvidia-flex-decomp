# CudaSafeDelayLoad(int, CUcontext*)

The CUDA driver loader: `dlopen("libcuda.so", RTLD_NOW)`, six `dlsym` lookups,
then a context created through the driver's PhysX low-latency export table.

Object: `work/binary/obj/dbg/NvFlexDeviceDebug_x64/flexDevice.cpp.o`
(debug archive; the release object carries no DWARF for this module).

CU producer: `GNU C++ 4.8.1 -m64 -mtune=generic -march=x86-64 -g -O0 -std=c++11
-fPIC -fpermissive -fno-strict-aliasing -fstack-protector`, compiled against
CUDA 9.2 `cuda.h` (DWARF decl_file 3) and `./../../../src/device/PhysXDevice.h`
(decl_file 4).

The source keeps the shipped `-O0` codegen via a file-scope
`#pragma GCC optimize ("O0")`: the object was built at -O0 (see
DW_AT_producer), and at the project's -O3 the two static helpers are folded
into their callers, which changes both the call graph and the instruction
budget the static gate measures. This is a codegen-level reproduction only.

## Evidence (line-for-line from `.debug_line`)

| line | address | statement |
|---|---|---|
| 114 | 0x67 | `void* hCudaDriver = dlopen("libcuda.so", RTLD_NOW)` (`mov $0x2,%esi`) |
| 117/119 | 0x7c/0x83 | early `return false` when the driver is absent |
| 122 | 0x8d | `cuGetExportTable_f` ← "cuGetExportTable" (.rodata+0x17) |
| 123 | 0xa7 | `cuCtxCreate_f` ← "cuCtxCreate" (.rodata+0x28) |
| 124 | 0xc1 | `cuDeviceGet_f` ← "cuDeviceGet" (.rodata+0x34) |
| 125 | 0xdb | `cuCtxSetCurrent_f` ← "cuCtxSetCurrent" (.rodata+0x40) |
| 126 | 0xf5 | `cuCtxGetCurrent_f` ← "cuCtxGetCurrent" (.rodata+0x50) |
| 127 | 0x10f | `cuCtxDestroy_f` ← "cuCtxDestroy" (.rodata+0x60) |
| 130 | 0x129 | `CUetblPhysXInterface* lowlatency = NULL` |
| 134 | 0x131 | `int flags` = 0x1c |
| 135 | 0x138 | `cuGetExportTable_f(&lowlatency, &CU_ETID_PhysXInterface)` |
| 139/140 | 0x152/0x158 | `if (deviceOrdinal == -1) deviceOrdinal = 0;` |
| 146 | 0x15f | `int suggestedOrdinal = GetSuggestedDeviceOrdinal();` |
| 149 | 0x167 | `if (lowlatency)` |
| 152 | 0x170 | `cuDeviceGet_f(&device, deviceOrdinal)` |
| 154 | 0x185 | `result = lowlatency->cuCtxCreateOnNode3DLowLatency(outContext, flags, device)` |
| 155-158 | 0x19f-0x1b5 | `if (result != CUDA_SUCCESS) { *outContext = NULL; return false; }` |
| 163/164 | 0x1b7/0x1ca | `cuCtxSetCurrent_f(*outContext); return true;` |
| 168 | 0x1d1 | `return false;` (no export table) |

* `flags = 0x1c` = `CU_CTX_SCHED_BLOCKING_SYNC(4) | CU_CTX_MAP_HOST(8) |
  CU_CTX_LMEM_RESIZE_TO_MAX(16)`; the enum values are the DWARF
  `CUctx_flags_enum` enumerators (cuda.h:310).
* `CU_ETID_PhysXInterface` is 16 bytes at `.rodata+0` (readelf: GLOBAL OBJECT,
  size 16) initialised with `0c a5 0b 8c 10 04 92 9a 89 a7 d0 df 10 e7 72 86`.
* The export-table struct is DWARF `CUetblPhysXInterface_st` (16 B,
  `size_t size` @0, fn pointer @8) with typedef `CUetblPhysXInterface`.
* `suggestedOrdinal` (decl_line 146) is stored at `-0x18(%rbp)` and never read
  — the shipped source computes and discards it; reproduced as a dead call so
  the object keeps the `GetSuggestedDeviceOrdinal` call edge.

## Inference

* The `-fpermissive` narrowing of the UUID bytes: modern GCC rejects
  `int`→`char` narrowing inside the brace initialiser, so the reconstruction
  casts each byte explicitly (same 16 bytes).
* The `_Unwind_Resume` landing pads in this object (`.gcc_except_table`, one
  cold path per function that makes an indirect/PLT call) are GCC 4.8
  `-O0 -fexceptions` artefacts; the reconstructed file carries no explicit
  `try`/`catch` and none is needed for behaviour.

## Parity

`needs-gpu`: requires an NVIDIA driver (`libcuda.so` plus the PhysX export
table). G1 only on this host, and the symbol is internal anyway.

## Unknowns

None.
