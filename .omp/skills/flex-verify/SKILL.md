---
name: flex-verify
description: Independently verify a submitted FleX reconstruction unit by re-running the static, differential and GPU gates, auditing evidence and layouts, and accepting or rejecting with a concrete diff. Use when a unit is in state `decompiled` and needs a second reviewer.
---

# Verify a unit

You are the second agent. You may not have submitted the unit under review
(`flexkit unit verify` enforces this).

## 1. Inspect

```bash
bin/flexkit unit show '<unit-id>'          # status, claims, gate history
bin/flexkit context '<unit-id>'            # evidence bundle
cat units/<module>/<symbol>.md             # author's notes
```

Check specifically:

* the author's locals/field names match DWARF (`referenced_types`);
* every magic constant has an evidence reference;
* `inlined_away` entries are real inline candidates (header-defined or
  same-module), not an excuse to drop a call;
* no file outside the module was modified (`git status --porcelain`).

## 2. Re-run the gates yourself

```bash
bin/flexkit gate static --unit '<unit-id>'   # exits 0?
bin/flexkit gate diff   --unit '<unit-id>'   # 64/64 bitwise?
bin/flexkit gate gpu    --unit '<unit-id>'   # skipped:no-gpu is acceptable
```

What each gate proves:

| Check | Proves | Does not prove |
|---|---|---|
| symbol/signature | exact ABI name + demangled signature | body correctness |
| calls | no call silently dropped (minus declared inlining) | argument correctness |
| instructions | not a stub (ratio band 0.5–2.0, relaxed when calls were inlined) | fidelity |
| strings | no literal lost or invented at library scope | formatting |
| layout | `sizeof`/`offsetof` for every type in the signature | semantic use of fields |
| diff | bitwise identical observable behaviour over 64 vectors | untested paths |

## 3. Decide

Accept:

```bash
bin/flexkit unit verify --unit '<unit-id>' --agent "$AGENT"
```

Reject with a concrete diff — never a vague objection:

```bash
bin/flexkit unit verify --unit '<unit-id>' --agent "$AGENT" --reject \
  --detail 'calls: unexplained_missing=[_Z9ResizeBVHR3BVHi]; disasm 0x1c90 shows a direct call'
```

Rejection returns the unit to `open` with your detail in `notes`.

## 4. Never

* never relax a gate or edit a spec to make a unit pass;
* never accept a unit whose `must_review` list is non-empty;
* never accept `waived` unless the reason matches the waiver class
  (compiler-generated registration, third-party header template).
