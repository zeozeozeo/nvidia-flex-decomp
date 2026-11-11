// flexdecomp: reconstructed D3D backend — Format.obj (6 functions).
//
// Evidence: work/decomp/nvflexd3d/NvFlexReleaseD3D_x64.dll.c
//   `dynamic_initializer_for_'g_DXGI_table'' @0x29ba0 (541 bytes)
//   NvFlex::DXGI_table::DXGI_table           @0x4fd80 (552 bytes)
//   NvFlex::convertToDXGI                    @0x4ffb0 (14 bytes)
//   NvFlex::getFormatSizeInBytes             @0x4ffc0 (14 bytes)
//   NvFlex::NvFlexFormatTable::NvFlexFormatTable @0x4ffd0 (285 bytes)
//   NvFlex::convertToNvFlex                  @0x500f0 (22 bytes)
//
// The table contents were additionally verified byte-for-byte against the
// shipped image: the 70-entry NvFlexFormat array lives at .data RVA 0x146990
// (PDB public `?g_NvFlexFormatTable@NvFlex@@3UNvFlexFormatTable@1@A`).
#include "Format.h"

namespace NvFlex
{

// Initial contents of the two tables.  They are copied by the constructors
// (the shipped constructors write the same values one by one), so the tables
// stay ordinary mutable globals like the originals.
static const DXGI_table::Element kDXGITable[22] = {
    { DXGI_FORMAT_UNKNOWN,              0 },   // [0]  eNvFlexFormat_unknown
    { DXGI_FORMAT_R32_TYPELESS,         0 },   // [1]
    { DXGI_FORMAT_R32_FLOAT,            4 },   // [2]
    { DXGI_FORMAT_R32G32_FLOAT,         8 },   // [3]
    { DXGI_FORMAT_R32G32B32A32_FLOAT,  16 },   // [4]
    { DXGI_FORMAT_R16_FLOAT,            2 },   // [5]
    { DXGI_FORMAT_R16G16_FLOAT,         4 },   // [6]
    { DXGI_FORMAT_R16G16B16A16_FLOAT,   8 },   // [7]
    { DXGI_FORMAT_R32_UINT,             4 },   // [8]
    { DXGI_FORMAT_R32G32_UINT,          8 },   // [9]
    { DXGI_FORMAT_R32G32B32A32_UINT,   16 },   // [10]
    { DXGI_FORMAT_R8_UNORM,             1 },   // [11]
    { DXGI_FORMAT_R8G8_UNORM,           2 },   // [12]
    { DXGI_FORMAT_R8G8B8A8_UNORM,       4 },   // [13]
    { DXGI_FORMAT_R16_UNORM,            2 },   // [14]
    { DXGI_FORMAT_R16G16_UNORM,         4 },   // [15]
    { DXGI_FORMAT_R16G16B16A16_UNORM,   8 },   // [16]
    { DXGI_FORMAT_D32_FLOAT,            4 },   // [17]
    { DXGI_FORMAT_D24_UNORM_S8_UINT,    4 },   // [18]
    { DXGI_FORMAT_R8_SNORM,             1 },   // [19]
    { DXGI_FORMAT_R8G8_SNORM,           2 },   // [20]
    { DXGI_FORMAT_R8G8B8A8_SNORM,       4 },   // [21]
};

// Indexed by DXGI_FORMAT; values are the NvFlexFormat the shipped table holds
// (verified against the .data image at RVA 0x146990).
static const NvFlexFormat kFormatTable[70] = {
    eNvFlexFormat_unknown,               //  0 DXGI_FORMAT_UNKNOWN
    eNvFlexFormat_unknown,               //  1 R32G32B32A32_TYPELESS
    eNvFlexFormat_r32g32b32a32_float,    //  2 R32G32B32A32_FLOAT
    eNvFlexFormat_r32g32b32a32_uint,     //  3 R32G32B32A32_UINT
    eNvFlexFormat_unknown,               //  4 R32G32B32A32_SINT
    eNvFlexFormat_unknown,               //  5 R32G32B32_TYPELESS
    eNvFlexFormat_unknown,               //  6 R32G32B32_FLOAT
    eNvFlexFormat_unknown,               //  7 R32G32B32_UINT
    eNvFlexFormat_unknown,               //  8 R32G32B32_SINT
    eNvFlexFormat_unknown,               //  9 R16G16B16A16_TYPELESS
    eNvFlexFormat_r16g16b16a16_float,    // 10 R16G16B16A16_FLOAT
    eNvFlexFormat_r16g16b16a16_unorm,    // 11 R16G16B16A16_UNORM
    eNvFlexFormat_unknown,               // 12 R16G16B16A16_UINT
    eNvFlexFormat_unknown,               // 13 R16G16B16A16_SNORM
    eNvFlexFormat_unknown,               // 14 R16G16B16A16_SINT
    eNvFlexFormat_unknown,               // 15 R32G32_TYPELESS
    eNvFlexFormat_r32g32_float,          // 16 R32G32_FLOAT
    eNvFlexFormat_r32g32_uint,           // 17 R32G32_UINT
    eNvFlexFormat_unknown,               // 18 R32G32_SINT
    eNvFlexFormat_unknown,               // 19 R32G8X24_TYPELESS
    eNvFlexFormat_unknown,               // 20 D32_FLOAT_S8X24_UINT
    eNvFlexFormat_unknown,               // 21 R32_FLOAT_X8X24_TYPELESS
    eNvFlexFormat_unknown,               // 22 X32_TYPELESS_G8X24_UINT
    eNvFlexFormat_unknown,               // 23 R10G10B10A2_TYPELESS
    eNvFlexFormat_unknown,               // 24 R10G10B10A2_UNORM
    eNvFlexFormat_unknown,               // 25 R10G10B10A2_UINT
    eNvFlexFormat_unknown,               // 26 R11G11B10_FLOAT
    eNvFlexFormat_unknown,               // 27 R8G8B8A8_TYPELESS
    eNvFlexFormat_r8g8b8a8_unorm,        // 28 R8G8B8A8_UNORM
    eNvFlexFormat_unknown,               // 29 R8G8B8A8_UNORM_SRGB
    eNvFlexFormat_unknown,               // 30 R8G8B8A8_UINT
    eNvFlexFormat_r8g8b8a8_snorm,        // 31 R8G8B8A8_SNORM
    eNvFlexFormat_unknown,               // 32 R8G8B8A8_SINT
    eNvFlexFormat_unknown,               // 33 R16G16_TYPELESS
    eNvFlexFormat_r16g16_float,          // 34 R16G16_FLOAT
    eNvFlexFormat_r16g16_unorm,          // 35 R16G16_UNORM
    eNvFlexFormat_unknown,               // 36 R16G16_UINT
    eNvFlexFormat_unknown,               // 37 R16G16_SNORM
    eNvFlexFormat_unknown,               // 38 R16G16_SINT
    eNvFlexFormat_r32_typeless,          // 39 R32_TYPELESS
    eNvFlexFormat_d32_float,             // 40 D32_FLOAT
    eNvFlexFormat_r32_float,             // 41 R32_FLOAT
    eNvFlexFormat_r32_uint,              // 42 R32_UINT
    eNvFlexFormat_unknown,               // 43 R32_SINT
    eNvFlexFormat_unknown,               // 44 R24G8_TYPELESS
    eNvFlexFormat_d24_unorm_s8_uint,     // 45 D24_UNORM_S8_UINT
    eNvFlexFormat_unknown,               // 46 R24_UNORM_X8_TYPELESS
    eNvFlexFormat_unknown,               // 47 X24_TYPELESS_G8_UINT
    eNvFlexFormat_unknown,               // 48 R8G8_TYPELESS
    eNvFlexFormat_r8g8_unorm,            // 49 R8G8_UNORM
    eNvFlexFormat_unknown,               // 50 R8G8_UINT
    eNvFlexFormat_r8g8_snorm,            // 51 R8G8_SNORM
    eNvFlexFormat_unknown,               // 52 R8G8_SINT
    eNvFlexFormat_unknown,               // 53 R16_TYPELESS
    eNvFlexFormat_r16_float,             // 54 R16_FLOAT
    eNvFlexFormat_unknown,               // 55 D16_UNORM
    eNvFlexFormat_r16_unorm,             // 56 R16_UNORM
    eNvFlexFormat_unknown,               // 57 R16_UINT
    eNvFlexFormat_unknown,               // 58 R16_SNORM
    eNvFlexFormat_unknown,               // 59 R16_SINT
    eNvFlexFormat_unknown,               // 60 R8_TYPELESS
    eNvFlexFormat_r8_unorm,              // 61 R8_UNORM
    eNvFlexFormat_unknown,               // 62 R8_UINT
    eNvFlexFormat_r8_snorm,              // 63 R8_SNORM
    eNvFlexFormat_unknown,               // 64 R8_SINT
    eNvFlexFormat_unknown,               // 65 A8_UNORM
    eNvFlexFormat_unknown,               // 66 R1_UNORM
    eNvFlexFormat_unknown,               // 67 R9G9B9E5_SHAREDEXP
    eNvFlexFormat_unknown,               // 68 R8G8_B8G8_UNORM
    eNvFlexFormat_unknown,               // 69 G8R8_G8B8_UNORM
};

DXGI_table g_DXGI_table;
NvFlexFormatTable g_NvFlexFormatTable;

// Format.obj @0x4fd80: fills the table and returns &g_DXGI_table.
DXGI_table::DXGI_table()
{
    for (int i = 0; i < 22; ++i)
        table[i] = kDXGITable[i];
}

// Format.obj @0x4ffd0: fills the table with eNvFlexFormat_unknown and installs
// the 21 known mappings.
NvFlexFormatTable::NvFlexFormatTable()
{
    for (int i = 0; i < 70; ++i)
        table[i] = kFormatTable[i];
}

// Format.obj @0x4ffb0 (14 bytes): direct table lookup.
DXGI_FORMAT convertToDXGI(NvFlexFormat format)
{
    return g_DXGI_table.table[format].format;
}

// Format.obj @0x4ffc0 (14 bytes): direct table lookup.
unsigned getFormatSizeInBytes(NvFlexFormat format)
{
    return g_DXGI_table.table[format].sizeInBytes;
}

// Format.obj @0x500f0 (22 bytes).
NvFlexFormat convertToNvFlex(DXGI_FORMAT format)
{
    // Formats at or above DXGI_FORMAT_BC1_TYPELESS (70) are not in the
    // 70-entry table and map to eNvFlexFormat_unknown.
    if (format < DXGI_FORMAT_BC1_TYPELESS)
        return g_NvFlexFormatTable.table[format];

    return eNvFlexFormat_unknown;
}

} // namespace NvFlex
