// flexdecomp: D3D compute shader 059 (cs_5_0), reconstructed from
// work/dxbc/NvFlexReleaseD3D_x64/059_cs_5_0.asm.
// Interface (from the original): cb0[1] (count in .x, value in .y),
// RWStructuredBuffer<float> u0 (stride 4), 256x1x1.
// Body: if (id.x < count) u0[id.x] = value;

cbuffer CB0 : register(b0)
{
    float4 cb0[1];
};

RWStructuredBuffer<float> u0 : register(u0);

[numthreads(256, 1, 1)]
void writeValue(uint3 id : SV_DispatchThreadID)
{
    if (id.x < (uint)cb0[0].x)
        u0[id.x] = cb0[0].y;
}
