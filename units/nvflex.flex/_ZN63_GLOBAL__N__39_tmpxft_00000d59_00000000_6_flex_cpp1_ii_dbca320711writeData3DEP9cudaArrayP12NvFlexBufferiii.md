# writeData3D(cudaArray_t, NvFlexBuffer*, int, int, int)

Original: `flex.cu:5275`. Evidence: pseudocode `cudaflex.c:1005`, disasm @0x1470.
cudaMemcpy3DParms offsets written: srcPtr 0x20 (ptr/pitch/xsize/ysize), dstArray
0x40, extent 0x80 (width/height/height), kind 0x98 - identical to CUDA 13's
measured layout (sizeof 160; the original zeroes 0x14 qwords).
extent.depth == height in the shipped code; reproduced literally.
G1 PASS (51 vs 65, ratio 1.27). Parity needs-gpu.
