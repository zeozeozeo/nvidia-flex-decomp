# ReadData(NvFlexErrorCallback, NvFlexBuffer*, const void*, int, const NvFlexCopyDesc*, int)

Original: `flex.cu:5297`. Evidence: pseudocode `cudaflex.c:1061`, disasm @0x1540
(52 instructions), DWARF decl line 5297.
desc == 0 clamps n = min(srcLength, dstBuffer->elementCount); otherwise the
destination-range and source-range checks with their two literal messages.
G1 PASS (52 vs 61, ratio 1.17). Parity needs-gpu (delegates to the CUDA copy).
