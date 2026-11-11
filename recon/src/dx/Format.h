// flexdecomp: reconstructed D3D backend — Format.obj.
//
// The backend carries two conversion tables between the internal NvFlexFormat
// enum and DXGI_FORMAT:
//
//   NvFlex::DXGI_table        Element { DXGI_FORMAT format; unsigned sizeInBytes; }
//                             table[22], indexed by NvFlexFormat.
//   NvFlex::NvFlexFormatTable NvFlexFormat table[70], indexed by DXGI_FORMAT
//                             (only formats below DXGI_FORMAT_BC1_TYPELESS are
//                              addressable).
//
// Both tables and their constructors are emitted in Format.obj; the shipped
// code also has a dynamic initializer for `g_DXGI_table`.
#pragma once

#include <dxgi.h>

#include "utils.h"

// Frame.h spells out only the NvFlexFormat enumerators the radix-sort slice
// uses (8..22).  The full PDB enum (field list 0x1033) starts at 0; the low
// values are needed by the DXGI <-> NvFlex format tables and are named here.
namespace nvfmt
{
enum
{
    unknown              = 0,
    r32_typeless         = 1,
    r32_float            = 2,
    r32g32_float         = 3,
    r32g32b32a32_float   = 4,
    r16_float            = 5,
    r16g16_float         = 6,
    r16g16b16a16_float   = 7
};
}

inline NvFlexFormat NvFmt(int value) { return static_cast<NvFlexFormat>(value); }

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
