// Hand-maintained additions appended to the generated recovered.h.
//
// Only facts DWARF cannot express but the shipped objects require, each with
// its evidence.  The header generator appends this file verbatim, so a
// regeneration never loses them.
#pragma once

#include <cstring>

// ltstr is the comparator of std::map<const char*, int, ltstr> used by
// CudaTimers::mNameToTimerMap (DWARF: util.h).  The map's ordered lookups and
// the shipped `_Rb_tree<... ltstr ...>::equal_range` symbol require a strict
// weak ordering over C strings; strcmp ordering is the only semantics that
// makes the shipped map behave as a name->timer index.  INFERRED body.
struct ltstr
{
    bool operator()(const char* a, const char* b) const { return strcmp(a, b) < 0; }
};
