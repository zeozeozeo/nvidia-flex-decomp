# NvFlexDeviceDestroyCudaContext()

```c
// flexdecomp: flexDevice.cpp:186
void NvFlexDeviceDestroyCudaContext()
{
	CUcontext context;                     // flexDevice.cpp:188
	cuCtxGetCurrent_f(&context);           // flexDevice.cpp:189
	cuCtxDestroy_f(context);               // flexDevice.cpp:191
}
```

## Evidence

* Public header `recon/include/NvFlexDevice.h`
  (`NV_FLEX_API void NvFlexDeviceDestroyCudaContext();`).
* DWARF: decl_line 186, no params, local `context` decl_line 188, 52 bytes.
* disasm 0x217..0x24a: indirect `call *cuCtxGetCurrent_f` with `&context`,
  then indirect `call *cuCtxDestroy_f` with `context`. Both pointers are
  file-scope `.bss` slots initialised by `CudaSafeDelayLoad`.
* `.debug_line` rows: 187 @0x217, 189 @0x21f, 191 @0x22f, 192 @0x23f.

## Parity

`needs-gpu`: the call targets are the driver entry points and are NULL until
`CudaSafeDelayLoad` has run, so the function cannot be exercised without an
NVIDIA driver. G1 green.

## Unknowns

None.
