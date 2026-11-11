# NvFlexBuffer::NvFlexBuffer(int, int, int, NvFlexLibrary*) (C2)

`nvflex.util#_ZN12NvFlexBufferC2EiiiP13NvFlexLibrary` - host unit, `src/cuda/util.cpp:471` (decl `util.h:138`)

Same body as the C1 alias, emitted once with the shipped mangled name
(`_ZN12NvFlexBufferC2EiiiP13NvFlexLibrary`) and aliased to C1. See the C1 note for evidence.

## Gate status
`static`: the C2 symbol is an alias of C1 in the shipped object (`nm`: both at 0x894), so
objdump prints only one function header per address; the gate's disassembly map has no C2
entry on **either** side (`orig=0, recon=0`) and the instruction-ratio check cannot pass.
Implementation is complete; awaiting the gate's alias handling.

## Unknowns
- none.
