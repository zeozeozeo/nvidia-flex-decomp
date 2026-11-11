// flexdecomp: D3D compute shader 041 (cs_5_0), reconstructed from
// work/dxbc/NvFlexReleaseD3D_x64/041_cs_5_0.asm.
// Interface: cb0[26] (element count in cb0[25].y), RWBuffer<float4> u0 (raw),
// 256x1x1. Body: zero the 16-byte slot for each thread below the count.

cbuffer CB0 : register(b0)
{
    float4 cb0[26];
};

RWBuffer<float4> u0 : register(u0);

[numthreads(256, 1, 1)]
void clearRaw(uint3 id : SV_DispatchThreadID)
{
    if (id.x < (uint)cb0[25].y)
        u0[id.x] = float4(0, 0, 0, 0);
}
