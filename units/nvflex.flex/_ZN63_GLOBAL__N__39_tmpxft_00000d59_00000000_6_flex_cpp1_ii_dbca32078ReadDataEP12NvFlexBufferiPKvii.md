# ReadData(NvFlexBuffer*, int, const void*, int, int)

Original: `flex.cu:5204`. Evidence: pseudocode `cudaflex.c:881` (all four error
branches with their literals and flex.cu lines 0x145a/0x146a/0x146b/0x1472/0x1477),
disasm @0x1120 (148 instructions), DWARF decl line 5204.

Call set matched exactly: FlexLogFunc, NvFlexSetFence, ByteOffset (both overloads),
cudaGraphicsMapResources, cudaGraphicsResourceGetMappedPointer,
cudaGraphicsUnmapResources, cudaMemcpyAsync.

- kind = DeviceToHost when type == eNvFlexBufferHost, else DeviceToDevice
  (disasm @0x118f).
- type == 3 is the graphics-resource buffer (literal; NvFlexBuffer::type is int).
- NvFlexLibrary::mErrorFunc is an opaque 8-byte member in recovered.h, read via
  memcpy in libErrorFunc().
- G1 static gate PASS (148 vs 183, ratio 1.24). Parity needs-gpu.
