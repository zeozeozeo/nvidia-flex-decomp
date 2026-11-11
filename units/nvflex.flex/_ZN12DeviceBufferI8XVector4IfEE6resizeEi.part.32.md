# DeviceBuffer<XVector4<float> >::resize(int) [clone .part.32]

GCC 4.8.1 IPA partial-inline clone of `DeviceBuffer<XVector4<float> >::resize(int)`
(`util.h:111`). It is not a source entity: the compiler split the function and gave
the out-of-line remainder the `.part.32` suffix (nm: `00000000000004a0 t`, 126 bytes,
section `.text`). The body is the cold half of resize (destroy texture, free, grow
to n*3/2, create texture) as shown in `cudaflex.c:237`.

A modern compiler reproduces neither the split point nor the `.part.N` numbering, so
the exact symbol cannot be emitted from source. The master unit
`_ZN12DeviceBufferI8XVector4IfEE6resizeEi` (not in this slice) is the reconstructable
one. Blocked: compiler-generated partial-inline clone.
