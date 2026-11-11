# _Z37__device_stub__Z10CreateGridiPKiPiS1_iPKiPiS1_

Compiler-generated CUDA host launch wrapper.

The shipped object was built by CUDA 9.2, whose stub calls `cudaLaunchKernel`
(and `cudaSetupArgument`/`cudaConfigureCall`); the CUDA 13.4 build required by
this host emits `__cudaLaunchKernel`, `__cudaGetKernel`, `__cudaPopCallConfiguration`
and `__cxa_guard_acquire/release` instead. The stub body is therefore a function
of the CUDA runtime version, not of FleX source: its callee set, literal set and
instruction count cannot be reproduced from any source text in `flex.cu`.

Measured (static gate, symbol present in the module because the kernel definition
in the kernel part files emits it):
- `_Z37__device_stub__Z10CreateGridiPKiPiS1_iPKiPiS1_`: missing `cudaLaunchKernel`,
  extra `__cudaLaunchKernel`/`__cudaGetKernel`/`__cxa_guard_*`/`__stack_chk_fail`.
- `_ZL109__device_stub__Z15SolveVelocitiesILb1EE...`: same class of mismatch.

Per plan 2.2 this unit must be `waived` (compiler-generated CUDA registration),
not reconstructed. Blocked for the integrator to waive.
