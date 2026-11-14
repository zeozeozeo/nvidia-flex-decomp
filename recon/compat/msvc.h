#pragma once
// MSVC-hosted build neutralizers for the recon CUDA targets (Windows).
//
// Forced into every translation unit by recon/CMakeLists.txt on WIN32
// (cl: /FI..., nvcc: -Xcompiler /FI...).  Per-site #if defined(_MSC_VER)
// edits in the sources keep the Linux/ELF path untouched: the PE ABI is the
// export surface win32/*.def generated mechanically from the shipped DLLs
// (tools/apitest/gen_exports.py), so internal C++ symbols are free under
// MSVC and the GCC-only symbol tricks (asm labels, aliases, visibility
// attributes) exist only for the ELF static gate.
//
// The behavioural parity target is the shipped release DLL; it carries no
// assert strings (asserts compiled out), so the Windows build sets NDEBUG
// and hand-rolled glibc assert calls in the sources are compiled out with
// the guard below.
#if defined(_MSC_VER)

#define FLEX_MSVC_BUILD 1

// The Windows targets compile with NDEBUG; guards on hand-rolled assert
// implementations use this instead of repeating the macro test.
#define FLEX_HAS_ASSERTS 0

// MSVC noinline hint for sites that used __attribute__((noinline)) purely as
// an optimization nudge on the ELF path (alias sites need real edits, not a
// macro, and are handled per-site).
#define FLEX_NOINLINE __declspec(noinline)

// bvh.cu calls glibc's fortified printf and assert-failure hook directly to
// reproduce the shipped ELF literals.  The shipped Windows release DLL
// imports no CRT (KERNEL32 + cudart only), so neither path exists there:
// compile both out under MSVC.  (Evidence: dumpbin /DEPENDENTS of
// NvFlexReleaseCUDA_x64.dll; no "assert" strings in the image.)
#define __printf_chk(...) ((void)0)
#define __assert_fail(assertion, file, line, function) ((void)0)

// NOTE: no namespace-std declarations here.  A force-included header that
// declares into namespace std breaks nvcc's -FI cuda_runtime.h prepass (CRT
// headers get double-parsed).  The C++17 std::unary_function shim lives in
// compat/std_cpp17_shim.h, included by the CUDA C++17 TUs that need it.

#endif // _MSC_VER
