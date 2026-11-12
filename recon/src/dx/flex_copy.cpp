// flexdecomp: reconstructed D3D backend — flex.obj, buffer copy layer.
//
// Evidence: work/decomp/nvflexd3d/NvFlexReleaseD3D_x64.dll.c
//   `anonymous namespace'::ReadData(Context*, Buffer*, ...)  @ 0x347d0 (125 bytes)
//   `anonymous namespace'::WriteData(Context*, Buffer*, ...) @ 0x34850 (114 bytes)
//   `anonymous namespace'::ReadData(NvFlexLibrary*, NvFlexBuffer*, Buffer*, NvFlexCopyDesc*) @ 0x348d0 (349 bytes)
//   `anonymous namespace'::WriteData(NvFlexLibrary*, Buffer*, NvFlexBuffer*, NvFlexCopyDesc*) @ 0x34a30 (363 bytes)
//
// Every NvFlexSet*() / NvFlexGet*() entry point is one call into this layer.  The
// library-level function validates the user buffer's element stride against the
// solver buffer's stride and reports the three shipped error strings; the
// context-level function picks the transport from the user buffer's type:
//
//   type 3 (registered / caller-owned pointer) -> copyToNative
//   type 0 (eNvFlexBufferHost)                 -> copyToHost
//   type 1 (eNvFlexBufferDevice)               -> copyToDevice
#include <cstring>

#include "FlexObjects.h"

namespace NvFlex
{
namespace flexobj
{

// ---------------------------------------------------------------------------
// Context-level transport
// ---------------------------------------------------------------------------
// @0x347d0 (125 bytes): solver buffer -> user buffer.
void ReadData(NvFlex::ContextEx* context, NvFlexBuffer* destination, unsigned dstOffset,
              NvFlex::Buffer* source, unsigned srcOffset, unsigned size)
{
    if (destination == 0 || size == 0)
        return;

    switch ((int)destination->type)
    {
    case 3:
        context->copyToNative(destination->ptr, dstOffset, source, srcOffset, size);
        return;
    case eNvFlexBufferHost:
        context->copyToHost(source, srcOffset, destination->mBuffer, dstOffset, size);
        return;
    case eNvFlexBufferDevice:
        context->copyToDevice(destination->mBuffer, dstOffset, source, srcOffset, size,
                              NvFlex::eDefaultBuffer);
        return;
    default:
        return;
    }
}

// @0x34850 (114 bytes): user buffer -> solver buffer.  All three transports go
// through ContextEx::copyToDevice; only the source view and the resource-kind
// tag differ (disassembly of 0x34850: host -> tag 8 + mBuffer, device -> tag 2
// + mBuffer, native pointer -> tag 16 + ptr).
void WriteData(NvFlex::ContextEx* context, NvFlex::Buffer* destination, unsigned dstOffset,
               NvFlexBuffer* source, unsigned srcOffset, unsigned size)
{
    if (source == 0 || size == 0)
        return;

    switch ((int)source->type)
    {
    case eNvFlexBufferHost:
        context->copyToDevice(destination, dstOffset, source->mBuffer, srcOffset, size,
                              NvFlex::eDownloadBuffer);
        return;
    case eNvFlexBufferDevice:
        context->copyToDevice(destination, dstOffset, source->mBuffer, srcOffset, size,
                              NvFlex::eDefaultBuffer);
        return;
    case 3:
        context->copyToDevice(destination, dstOffset, (NvFlex::Buffer*)source->ptr, srcOffset,
                              size, NvFlex::eNativeBuffer);
        return;
    default:
        return;
    }
}

// ---------------------------------------------------------------------------
// Library-level validation
// ---------------------------------------------------------------------------
// @0x348d0 (349 bytes)
void ReadData(NvFlexLibrary* lib, NvFlexBuffer* destination, NvFlex::Buffer* source,
              const NvFlexCopyDesc* desc)
{
    if (destination == 0)
        return;

    const unsigned sourceStride = source->desc().stride;
    const unsigned destStride = (unsigned)destination->elementByteStride;
    NvFlexErrorCallback errorFunc = lib->mErrorFunc;

    if (sourceStride != destStride)
    {
        if (errorFunc != 0)
            FlexLogFunc(errorFunc, eNvFlexLogError,
                        "Call to an NvFlexGet*() method with incompatible buffer strides, "
                        "source stride: %d dest stride: %d",
                        "..\\..\\dx\\flex.cpp", 0xc6,
                        (int)sourceStride, (int)destStride);
        return;
    }

    if (desc == 0)
    {
        int count = (int)source->desc().dim;
        if (destination->elementCount <= count)
            count = destination->elementCount;
        ReadData(lib->mContext, destination, 0, source, 0, (unsigned)count * sourceStride);
        return;
    }

    const int elementCount = desc->elementCount;
    if ((int)source->desc().dim < desc->dstOffset + elementCount)
    {
        if (errorFunc != 0)
            FlexLogFunc(errorFunc, eNvFlexLogError,
                        "Call to an NvFlexGet*() method with destination range outside of "
                        "destination buffer", "..\\..\\dx\\flex.cpp", 0xd0);
        return;
    }

    if (desc->srcOffset + elementCount <= (int)source->desc().dim)
    {
        ReadData(lib->mContext, destination, (unsigned)(desc->dstOffset * (int)destStride),
                 source, (unsigned)(desc->srcOffset * (int)sourceStride),
                 (unsigned)elementCount * sourceStride);
        return;
    }

    if (errorFunc != 0)
        FlexLogFunc(errorFunc, eNvFlexLogError,
                    "Call to an NvFlexGet*() method with source range outside of source buffer",
                    "..\\..\\dx\\flex.cpp", 0xd6);
}

// @0x34a30 (363 bytes)
void WriteData(NvFlexLibrary* lib, NvFlex::Buffer* destination, NvFlexBuffer* source,
               const NvFlexCopyDesc* desc)
{
    if (source == 0)
        return;

    const unsigned destStride = destination->desc().stride;
    const unsigned sourceStride = (unsigned)source->elementByteStride;
    NvFlexErrorCallback errorFunc = lib->mErrorFunc;

    if (sourceStride != destStride)
    {
        if (errorFunc != 0)
            FlexLogFunc(errorFunc, eNvFlexLogError,
                        "Call to an NvFlexSet*() method with incompatible buffer strides, "
                        "source stride: %d dest stride: %d",
                        "..\\..\\dx\\flex.cpp", 0xf0,
                        (int)sourceStride, (int)destStride);
        return;
    }

    if (desc == 0)
    {
        int count = (int)destination->desc().dim;
        if (source->elementCount <= count)
            count = source->elementCount;
        WriteData(lib->mContext, destination, 0, source, 0, (unsigned)count * sourceStride);
        return;
    }

    const int elementCount = desc->elementCount;
    if ((int)destination->desc().dim < desc->dstOffset + elementCount)
    {
        if (errorFunc != 0)
            FlexLogFunc(errorFunc, eNvFlexLogError,
                        "Call to an NvFlexSet*() method with destination range outside of "
                        "destination buffer", "..\\..\\dx\\flex.cpp", 0xfa);
        return;
    }

    if (desc->srcOffset + elementCount <= source->elementCount)
    {
        WriteData(lib->mContext, destination, (unsigned)(desc->dstOffset * (int)destStride),
                  source, (unsigned)(desc->srcOffset * (int)sourceStride),
                  (unsigned)elementCount * sourceStride);
        return;
    }

    if (errorFunc != 0)
        FlexLogFunc(errorFunc, eNvFlexLogError,
                    "Call to an NvFlexSet*() method with source range outside of source buffer",
                    "..\\..\\dx\\flex.cpp", 0x100);
}

} // namespace flexobj
} // namespace NvFlex
