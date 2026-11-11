// Interface between the generic driver and a generated per-unit harness.
#pragma once

extern "C" {
void fd_setup(long iter);      // deterministic inputs for this iteration
void fd_call(void *fn);        // invoke the unit through fn
void fd_serialize(int fd);     // write [uint64 len][bytes] per output
}
