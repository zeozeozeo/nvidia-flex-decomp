# DumpMemory(char const*)

`nvflex.util#_Z10DumpMemoryPKc` - host unit, `src/cuda/util.cpp:335`

## Evidence
- disasm 0x58f..0x598: parameter spill and return only - the body is empty in the shipped
  object (a debug-only helper compiled out).
- Differential: 16/16 bitwise identical against `libnvflex_rel.so` (both sides are `ret`).

## Unknowns
- none.
