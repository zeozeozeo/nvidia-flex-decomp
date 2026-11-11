// Reconstructed from the shipped DXBC bytecode of NvFlexReleaseD3D_x64.dll,
// shader 013 (cs_5_0).  Interface (constant-buffer layout, resource bindings,
// thread group) comes from the RDEF chunk; the body is derived from the SHEX
// assembly 013_cs_5_0.asm.
//
// Evidence: work/dxbc/NvFlexReleaseD3D_x64/013.dxbc,
//           work/dxbc/NvFlexReleaseD3D_x64/013_cs_5_0.asm, work/dxbc.json

cbuffer constBVH : register(b0)
{
    int gNumItems;
    float3 _fdpad0_0;
};

Buffer<uint> keys : register(t0);

RWStructuredBuffer<int> deltas : register(u0);

[numthreads(256, 1, 1)]
void CalculateKeyDeltas(uint3 tid : SV_DispatchThreadID)
{
	// CalculateKeyDeltas: XOR of adjacent sorted Morton keys
	if ((int)tid.x < gNumItems - 1)
	{
		deltas[tid.x] = keys[tid.x] ^ keys[tid.x + 1];
	}
}
