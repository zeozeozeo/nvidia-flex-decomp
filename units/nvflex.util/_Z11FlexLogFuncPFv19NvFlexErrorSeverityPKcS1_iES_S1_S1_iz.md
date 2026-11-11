# FlexLogFunc(void (*)(NvFlexErrorSeverity, char const*, char const*, int), NvFlexErrorSeverity, char const*, char const*, int, ...)

`nvflex.util#_Z11FlexLogFuncPFv19NvFlexErrorSeverityPKcS1_iES_S1_S1_iz` - host unit, `src/cuda/util.cpp:49`

## Evidence
- disasm 0x0..0x108: variadic prologue (XMM0-7 saves, `test %al`), canary at -0xb8,
  `char buf[2048]` at -0x8c0, `vsnprintf(buf, 0x800, msg, args)` at 0xbe, then
  `func(type, buf, file, line)` through the saved pointer at 0xe4.
- `va_list` layout from the object: `gp_offset = 0x28`, `fp_offset = 0x30`,
  `overflow_arg_area = &stack0x8`, `reg_save_area = local_b8` (i.e. `va_start(args, line)`).
- signature/DWARF: `void FlexLogFunc(NvFlexErrorCallback, NvFlexErrorSeverity, const char*, const char*, int, ...)`.

## Inference only
- `buf` is a named local of 2048 bytes; the shipped code writes at most that.
- Every log site in this TU passes `__FILE__` ("./../../../src/cuda/util.cpp") and one empty
  variadic argument (`.rodata+0xe`, a lone NUL); both are reproduced literally so the call
  ABI and the literal set match (`src/cuda/util.cpp:49`).
- `va_end` is a no-op here (no visible code).

## Differential
`pure-cpu`, 64/64 bitwise identical. The harness cannot mint a C callback, and it always passes
a valid pointer for a pointer parameter, so the spec spells the callback argument as `long`
(integer-class pointer argument on x86-64 SysV) to reach the real null-callback path; `type`,
`msg`, `file` and `line` are varied by the harness.

## Unknowns
- none.
