# __sti____cudaRegisterAll() (static, CUDA fatbin registration)

Compiler-generated: nvcc emits one static `__sti____cudaRegisterAll` per CUDA
translation unit, containing one `__cudaRegisterFunction` call per kernel defined
in that TU plus the fatbin registration.

The shipped object registers 41 kernels in one 3034-byte function and calls
`__cudaRegisterFatBinary`, `__cudaRegisterFunction`, `__cudaRegisterVar`, `atexit`.
The reconstructed module splits the kernels across `flex.part*.cu`, so there is one
small registration function per part file (25 instructions each), each calling
`__cudaRegisterLinkedBinary_*` under `-rdc=true`. Neither the symbol's body nor its
call set can match the single-TU 9.2 build; it is a function of the CUDA toolchain,
not of FleX source.

Per plan 2.2 this unit must be `waived` (compiler-generated CUDA registration).
Blocked for the integrator to waive.
