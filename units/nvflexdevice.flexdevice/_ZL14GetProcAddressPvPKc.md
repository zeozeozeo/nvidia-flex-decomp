# GetProcAddress(void*, char const*)

```c
// flexdecomp: flexDevice.cpp:84
static void* GetProcAddress(void* handle, const char* name)
{
	return dlsym(handle, name);
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

* `work/dwarf/nvflexdevice.flexdevice.json` — DIE `GetProcAddress` decl_line 84,
  params `void* handle`, `const char* name`, return `void*`.
* disasm `dump/flexDevice.cpp.disasm.txt` @0x0..0x24 — one `dlsym@PLT` call.
* `.debug_line`: the whole body (0x0..0x24) maps to line 84, i.e. the shipped
  source writes this three-line function such that GCC 4.8 attributes the body
  to the declaration line. The reconstruction keeps the readable form; the
  attribution difference is not observable in the object's code.

## Parity

`parity: needs-gpu` is a *G1-only* marker here (the field vocabulary has no
"internal function" value): the symbol has internal linkage (`t`, not in
`.dynsym`), so the differential driver cannot `dlsym` it. Behaviour is a single
`dlsym` forward.

## Unknowns

None.

## Codegen shape (both shipped objects)

The release object (the static gate's baseline for this module) folds this
forward into `CudaSafeDelayLoad`; the debug object keeps it out of line
(12 instructions). Both shapes are in the object: `GetProcAddressImpl` is
`always_inline` and is what the callers contain, while the shipped symbol
`_ZL14GetProcAddressPvPKc` is the out-of-line wrapper (18 instructions,
baseline 12, ratio 1.5, calls `dlsym` 1/1).
