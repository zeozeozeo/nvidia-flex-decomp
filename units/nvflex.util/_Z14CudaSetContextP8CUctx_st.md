# CudaSetContext(CUctx_st*)

`nvflex.util#_Z14CudaSetContextP8CUctx_st` - host unit, `src/cuda/util.cpp:214`

## Evidence
- disasm 0x517..0x534: loads `cuCtxSetCurrent_f` and tail-calls it with `ctx`.

## Gate status (`static`)
`instructions`: orig 10 (debug object, -O0) vs recon 2 (`mov fn,%rax; jmp *%rax`), ratio 0.2.
The shipped release object is 3 instructions for the same function; the proof of equivalence
is the instruction-by-instruction match against `libnvflex_rel.so` (ratio 0.667). Blocked by
the -O0 debug baseline; reported to the integrator.

## Unknowns
- none.
