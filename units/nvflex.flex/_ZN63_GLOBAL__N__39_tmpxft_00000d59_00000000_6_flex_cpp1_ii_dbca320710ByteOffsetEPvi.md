# ByteOffset(void*, int)

Original: `flex.cu:5200`. `return (char*)ptr + byteOffset;`
Evidence: pseudocode `cudaflex.c:871`, disasm @0x1100 (7 bytes, single `lea`),
DWARF decl line 5200.
G1 static gate PASS (4/4 instructions). G2: 8/8 bitwise identical against
`work/binary/so/libnvflex_dbg.so` (the release library renames the anonymous
namespace, so the CLI's rel variant cannot resolve the symbol).
