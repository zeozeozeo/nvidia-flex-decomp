# physx::physxDevSLIEnabledWrapper(void*)

```c
// flexdecomp: PhysXDevice.cpp:54
bool physxDevSLIEnabledWrapper(void*)
{
	return false;             // :56, mov $0x0,%eax at 0x5b
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

* disasm 0x53: the shipped function stores its (unnamed) pointer argument at
  `-0x8(%rbp)` and returns 0 — the parameter is unnamed in the DWARF
  (`DW_TAG_formal_parameter` without `DW_AT_name`), reproduced as `void*`.
* `.debug_line`: 55 @0x53, 56 @0x5b, 57 @0x60.
* DWARF return type is `bool` (1-byte base type, encoding boolean).

## Parity

`pure-cpu` — G2 `64/64 identical` on `bool` plus the caller-visible buffer.
Harness note: the generated differential harness calls the function through
`bool (*)(int*)` because `gate diff` cannot size a `void`-typed buffer; the
argument is an ignored opaque pointer, so the pointer ABI is identical.

## Unknowns

None.
