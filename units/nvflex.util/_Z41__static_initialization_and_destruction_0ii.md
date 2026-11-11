# __static_initialization_and_destruction_0(int, int)

`nvflex.util#_Z41__static_initialization_and_destruction_0ii` - compiler-generated, `src/cuda/util.cpp` (no source line)

## Evidence
- disasm 0xb8d..0xbd4 (debug object): `if (__initialize_p == 1 && __priority == 0xffff)`,
  then `std::ios_base::Init::Init()` (0xbb1) and `__cxa_atexit(std::ios_base::Init::~Init,
  &std::__ioinit, &__dso_handle)` (0xbd2). `_GLOBAL__sub_I_util.cpp` (waived) calls it.
- The shipped **release** object has no such symbol: at -O3 GCC 4.8 inlined the trampoline
  into `_GLOBAL__sub_I_util.cpp` (0x2e bytes) - so it exists only in the -O0 debug object.

## Why there is no reconstruction
- This is GCC 4.8's TU static-initialisation trampoline, emitted for `std::__ioinit`.
- Modern libstdc++ removed `std::__ioinit` (iostream initialisation moved to
  `std::ios_base_library_init()` inside libstdc++.so), so including `<iostream>` no longer
  materialises any static initialiser: verified with the pinned g++ + exact gate flags
  (`nm` shows only `U std::ios_base_library_init()`).
- GCC 13 emits static initialisation straight into `_GLOBAL__sub_I_<file>` with the init and
  `__cxa_atexit` calls inlined (verified with a TU-local object that has a destructor);
  it never emits `_Z41__static_initialization_and_destruction_0ii`.
- Writing the function by hand would mean emitting compiler-internal code with a hard-coded
  `std::__ioinit` symbol that no longer exists.

Requested disposition: `waived` with reason "compiler-generated TU static-initialisation
trampoline (GCC 4.8); modern libstdc++ has no `std::__ioinit` and GCC 13 folds static
initialisation into `_GLOBAL__sub_I_util.cpp`". The ledger already waives
`_GLOBAL__sub_I_util.cpp` for the same reason.
