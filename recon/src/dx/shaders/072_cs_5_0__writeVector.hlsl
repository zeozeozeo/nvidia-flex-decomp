// flexdecomp: D3D compute shader 072 (cs_5_0), reconstructed from
// work/dxbc/NvFlexReleaseD3D_x64/072_cs_5_0.asm.
// Interface: cb0[2] (count in cb0[0].x, payload in cb0[1].xyzw),
// RWStructuredBuffer<float4> u0 (stride 16), 256x1x1.

cbuffer CB0 : register(b0)
{
    float4 cb0[2];
};

RWStructuredBuffer<float4> u0 : register(u0);

[numthreads(256, 1, 1)]
void writeVector(uint3 id : SV_DispatchThreadID)
{
    if (id.x < (uint)cb0[0].x)
        u0[id.x] = cb0[1];
}
