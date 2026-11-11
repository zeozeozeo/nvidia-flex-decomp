# physx::physxDevGetCudaOrdinalWrapper(int*)

```c
// flexdecomp: PhysXDevice.cpp:37
PHYSX_DEV_STATUS physxDevGetCudaOrdinalWrapper(int* cudaDevOrdinal)
{
	const char* deviceOrdinalString = getenv("PHYSX_GPU_DEVICE");  // :39
	if (deviceOrdinalString == NULL)                               // :40
		*cudaDevOrdinal = 0;                                        // :41
	else
		*cudaDevOrdinal = atoi(deviceOrdinalString);                // :43
	return PHYSX_DEV_OK;                                           // :44
}
```

Object: `work/binary/obj/dbg/NvFlexDeviceDebug_x64/PhysXDevice.cpp.o`
(the archive's release member carries no DWARF for this module).

CU producer: `GNU C++ 4.8.1 -m64 -mtune=generic -march=x86-64 -g -O0 -std=c++11
-fPIC -fpermissive -fno-strict-aliasing -fstack-protector`.

This is the Linux "PhysX device" shim layer shipped with the SDK: there is no
PhysX GPU runtime on Linux, so all three wrappers are degraded. The module
keeps the shipped `-O0` codegen through a file-scope
`#pragma GCC optimize ("O0")` — see `_ZL17CudaSafeDelayLoadiPP8CUctx_st.md`
for the rationale (the static gate's call-graph/instruction budget measures
the un-inlined object).

The recovered `recon/src/device/PhysXDevice.h` carries only the
`namespace physx` contents the DWARF records for decl_file 2: the anonymous
`PHYSX_DEV_*` enumerators (:52) and `typedef int PHYSX_DEV_STATUS` (:47).

## Evidence

* `.debug_line`: 38 @0x0, 39 @0xc (`getenv` call), 40 @0x1c (`cmpq $0x0`),
  41 @0x23 (`movl $0x0,(%rax)`), 43 @0x2f (`atoi` call), 44 @0x41
  (`mov $0x0,%eax`), 45 @0x46.
* disasm: `getenv@plt` at 0x14, `atoi@plt` at 0x37; `.rodata+0` is the single
  literal `PHYSX_GPU_DEVICE`.
* DWARF: param `cudaDevOrdinal` (int*), local `deviceOrdinalString` decl_line 39.
* `return PHYSX_DEV_OK` (0) is the constant 0 stored in `%eax`; the DWARF
  enumerator is `PHYSX_DEV_OK = 0`.

## `atoi` vs `strtol`

The shipped source calls `atoi()`, but the two shipped objects resolve it
differently: the debug object calls `atoi@plt` (0x36), the release object calls
`strtol@plt` with base 10 / NULL endptr (rel `PhysXDevice.cpp.o` @0x1f), i.e.
glibc's inline `atoi` was active in the `-O3` build. Since `(int)strtol(nptr,
NULL, 10)` *is* `atoi`, the reconstruction writes that call explicitly and
declares `strtol`/`getenv` `extern "C"` (including `<stdlib.h>` would map
`strtol` to `__isoc23_strtol` on this glibc, which neither shipped object
calls). Static gate: `calls 2/2 matched` against the release baseline,
25 instructions vs 20 (ratio 1.25).

## Parity

`pure-cpu` — G2 `64/64 identical` (`*cudaDevOrdinal` and the return value
compared bitwise against `libnvflexdevice_rel.so`). G1 green.

## Unknowns

None.
