// Reconstructed from the shipped DXBC bytecode of NvFlexReleaseD3D_x64.dll,
// shader 002 (cs_5_0).  Interface (constant-buffer layout, resource bindings,
// thread group) comes from the RDEF chunk; the body is derived from the SHEX
// assembly 002_cs_5_0.asm.
//
// Evidence: work/dxbc/NvFlexReleaseD3D_x64/002.dxbc,
//           work/dxbc/NvFlexReleaseD3D_x64/002_cs_5_0.asm, work/dxbc.json

cbuffer constBVH : register(b0)
{
    int gNumItems;
    float3 _fdpad0_0;
};

StructuredBuffer<float4> itemLowers : register(t0);

StructuredBuffer<float4> itemUppers : register(t1);

StructuredBuffer<float3> gridLower : register(t2);

StructuredBuffer<float3> gridInvEdges : register(t3);

RWBuffer<uint> indices : register(u0);

RWBuffer<uint> keys : register(u1);

[numthreads(256, 1, 1)]
void CalculateMortonCodes(uint3 tid : SV_DispatchThreadID)
{
	// CalculateMortonCodes: quantise the item centre to the 512^3 grid and
	// interleave the cell coordinates into a Morton key.
	uint key = 0xffffffffu;
	uint index = 0xffffffffu;

	if ((int)tid.x < gNumItems)
	{
		const float4 lower = itemLowers[tid.x];
		const float4 upper = itemUppers[tid.x];
		const float3 gridOrigin = gridLower[0];
		const float3 invEdges = gridInvEdges[0];

		const float3 centre = (lower.xyz + upper.xyz) * 0.5f;
		const float3 cellf = (centre - gridOrigin) * invEdges * 512.0f;

		// cvt.rzi + clamp to [0, 511]
		const int3 cell = clamp((int3)cellf, int3(0, 0, 0), int3(511, 511, 511));

		// Part1by2 over the mirrored (z, y, x) register order of the original.
		const uint3 v0 = uint3((uint)cell.z, (uint)cell.y, (uint)cell.x);
		uint3 v = v0 ^ (v0 << 16);
		uint3 m = v & uint3(0x010000FFu, 0x010000FFu, 0x010000FFu);
		v = ((v & 0xFFu) << 8) ^ m;
		v = v & uint3(0x0100F00Fu, 0x0100F00Fu, 0x0100F00Fu);
		v = v ^ (v << 4);
		v = v & uint3(0x010C30C3u, 0x010C30C3u, 0x010C30C3u);
		v = v ^ (v << 2);

		key = ((v.x << 2) & 0x04924924u) + ((v.y << 1) & 0x02492492u) + (v.z & 0x01249249u);
		index = tid.x;
	}

	keys[tid.x] = key;
	indices[tid.x] = index;
}
