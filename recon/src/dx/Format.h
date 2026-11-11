// flexdecomp: reconstructed D3D backend — Format.obj.
//
// The backend carries two conversion tables between the internal NvFlexFormat
// enum and DXGI_FORMAT:
//
//   NvFlex::DXGI_table        Element { DXGI_FORMAT format; unsigned sizeInBytes; }
//                             table[22], indexed by NvFlexFormat.
//   NvFlex::NvFlexFormatTable NvFlexFormat table[70], indexed by DXGI_FORMAT
//                             (only the first DXGI_FORMAT_BC1_TYPELESS entries
//                              are addressable).
//
// Both tables and their constructors are emitted in Format.obj; the shipped
// code also has a dynamic initializer for `g_DXGI_table` because its
// constructor is not constexpr-able in the original translation unit.
#pragma once

#include <dxgi.h>

#include "utils.h"

namespace NvFlex
{

struct DXGI_table
{
    struct Element
    {
        DXGI_FORMAT format;
        unsigned sizeInBytes;
    };

    Element table[22];

    DXGI_table();
};

struct NvFlexFormatTable
{
    NvFlexFormat table[70];

    NvFlexFormatTable();
};

extern DXGI_table g_DXGI_table;
extern NvFlexFormatTable g_NvFlexFormatTable;

// Format.obj @0x4ffb0 (14 bytes).
DXGI_FORMAT convertToDXGI(NvFlexFormat format);

// Format.obj @0x4ffc0 (14 bytes).
unsigned getFormatSizeInBytes(NvFlexFormat format);

// Format.obj @0x500f0 (22 bytes).
NvFlexFormat convertToNvFlex(DXGI_FORMAT format);

} // namespace NvFlex
