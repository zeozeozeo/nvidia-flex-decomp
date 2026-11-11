# WriteData(NvFlexErrorCallback, void*, int, NvFlexBuffer*, const NvFlexCopyDesc*, int)

Original: `flex.cu:5324`. Evidence: pseudocode `cudaflex.c:1105`, disasm @0x1620
(63 instructions), DWARF decl line 5324. Returns immediately when srcBuffer == 0.
G1 PASS (63 vs 56, ratio 0.89). Parity needs-gpu.
