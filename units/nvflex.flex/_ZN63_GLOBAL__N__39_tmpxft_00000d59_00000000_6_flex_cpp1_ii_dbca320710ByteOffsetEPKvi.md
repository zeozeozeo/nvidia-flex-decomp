# ByteOffset(const void*, int)

Original: `flex.cu:5201`. `return (char*)ptr + byteOffset;` (const dropped by the cast).
Evidence: pseudocode `cudaflex.c:861`, disasm @0x1110, DWARF decl line 5201.
G1 static gate PASS (4/4). G2: 8/8 bitwise identical against `libnvflex_dbg.so`.
