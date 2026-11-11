# NvFlexBuffer::map()

`nvflex.util#_ZN12NvFlexBuffer3mapEv` - host unit, `src/cuda/util.cpp:499`

## Evidence
- disasm 0xab2..0xb21: `if (mapped && lib->mErrorFunc)` -> "Mapping a buffer that was already
  mapped" (eNvFlexLogWarning = 2 in `esi`, line 0x1f7); then `mapped = true; return ptr;`.
- Differential: 16/16 bitwise identical (the harness's zeroed buffer takes the no-warning
  path; `mapped` and the returned pointer are compared).

## Unknowns
- `NvFlexBuffer`'s members are DWARF-recovered data members only; the member function is
  defined with its shipped mangled name (assembler label) because recovered.h has no member
  declaration. See the module report.
