# WriteData(void*, int, NvFlexBuffer*, int, int)

Original: `flex.cu:5245`. Evidence: pseudocode `cudaflex.c:956`, disasm @0x1350
(88 instructions), DWARF decl line 5245.
Calls matched: FlexLogFunc, NvFlexSetFence, ByteOffset (both), cudaMemcpyAsync.
kind = HostToDevice when type == 0, else DeviceToDevice (disasm @0x1499:
sbb/and 0xfffffffe/add 3).
G1 PASS (88 vs 71, ratio 0.81). Parity needs-gpu.
