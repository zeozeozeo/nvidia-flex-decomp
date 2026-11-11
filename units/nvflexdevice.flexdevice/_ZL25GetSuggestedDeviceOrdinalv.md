# GetSuggestedDeviceOrdinal()

```c
// flexdecomp: flexDevice.cpp:92
static int GetSuggestedDeviceOrdinal()
{
	int ordinal;                       // flexDevice.cpp:94
	if (physx::physxDevGetCudaOrdinalWrapper(&ordinal) != 0)
		return -1;
	return ordinal;
}
```

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

## Evidence

* DWARF: decl_line 92, local `ordinal` at decl_line 94 (`int`), lexical block is
  the whole body; no other locals, so no named `bool` temporary.
* disasm 0x25..0x57: `test %eax,%eax; setne %al; test %al,%al; je` — that
  double conversion is `-O0` codegen for `x != 0`, not `if (x)`.
* disasm 0x42: `mov $0xffffffff,%eax; jmp epilogue` — the failure path returns
  -1 directly, so the source is `return -1;`, not `ordinal = -1;`.
* disasm 0x49: `mov -0x4(%rbp),%eax` — success returns the wrapper's out value.
* `physx::physxDevGetCudaOrdinalWrapper` is declared in this TU (the recovered
  `PhysXDevice.h` declares no functions — DWARF namespace child list).

## Inference

* The comparison constant is `0` (`PHYSX_DEV_OK` is 0); provenance of the
  literal form is not observable.

## Parity

G1 only: internal linkage, not `dlsym`-able. The exported wrapper
`NvFlexDeviceGetSuggestedOrdinal()` covers this path bitwise in G2.

## Unknowns

None.
