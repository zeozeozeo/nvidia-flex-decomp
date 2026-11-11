# cub::EmptyKernel<void>()

Kernel unit of `nvflex.sort`. Device entry is empty in both the original `sm_30.ptx` and the
reconstruction (`ret;`), so the PTX check passes with `orig = recon = 1` instruction.

* Reconstruction: `recon/src/cuda/sort.cu` (`extern "C" __global__ void _ZN3cub11EmptyKernelIvEEvv()`).
* Launched as `<<<1,1>>>` when `num_items <= 1` or `end_bit <= begin_bit`, mirroring CUB's dummy
  launch (surfaces sticky stream errors and preserves stream ordering).
* Parity: `kernel`. Static gate green.
