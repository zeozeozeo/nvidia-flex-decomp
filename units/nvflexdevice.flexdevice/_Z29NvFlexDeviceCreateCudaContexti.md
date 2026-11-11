# NvFlexDeviceCreateCudaContext(int)

```c
// flexdecomp: flexDevice.cpp:180
bool NvFlexDeviceCreateCudaContext(int ordinal)
{
	CUcontext context;                          // flexDevice.cpp:182
	return CudaSafeDelayLoad(ordinal, &context);// flexDevice.cpp:183
}
```

## Evidence

* Public header `recon/include/NvFlexDevice.h`
  (`NV_FLEX_API bool NvFlexDeviceCreateCudaContext(int ordinal);`).
* DWARF: decl_line 180, param `ordinal` (int), local `context` (CUcontext)
  decl_line 182, 40 bytes.
* disasm 0x1ef..0x216: `lea -0x8(%rbp),%rdx; mov ordinal,%eax; call
  CudaSafeDelayLoad`, return value passes through untouched (jmp to epilogue).

## Parity

`needs-gpu`: the only path is `CudaSafeDelayLoad`, which requires
`libcuda.so` and the driver PhysX export table. G1 green on this host.

## Unknowns

None.
