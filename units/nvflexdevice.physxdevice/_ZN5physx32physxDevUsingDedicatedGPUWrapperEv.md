# physx::physxDevUsingDedicatedGPUWrapper()

```c
// flexdecomp: PhysXDevice.cpp:47
PHYSX_DEV_STATUS physxDevUsingDedicatedGPUWrapper()
{
	return PHYSX_DEV_OK;      // :50, mov $0x0,%eax at 0x4c
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

* `.debug_line`: 48 @0x48 (prologue), 50 @0x4c (`mov $0x0,%eax`), 51 @0x51.
* disasm 0x48..0x52: 11 bytes, no calls, returns 0.
* DWARF return type is typedef `PHYSX_DEV_STATUS` = `int` (header :47).

## Parity

`pure-cpu` — G2 `64/64 identical` on the return value. G1 green.

## Unknowns

None.
