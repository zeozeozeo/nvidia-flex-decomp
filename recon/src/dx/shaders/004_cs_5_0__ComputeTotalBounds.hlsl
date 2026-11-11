// Reconstructed from the shipped DXBC bytecode of NvFlexReleaseD3D_x64.dll,
// shader 004 (cs_5_0).  Interface (constant-buffer layout, resource bindings,
// thread group) comes from the RDEF chunk; the body is derived from the SHEX
// assembly 004_cs_5_0.asm.
//
// Evidence: work/dxbc/NvFlexReleaseD3D_x64/004.dxbc,
//           work/dxbc/NvFlexReleaseD3D_x64/004_cs_5_0.asm, work/dxbc.json

	// ComputeTotalBounds (first level): block-wide min/max over the item bounds.
groupshared float3 sBounds[256];

cbuffer constBVH : register(b0)
{
    int gNumItems;
    float3 _fdpad0_0;
};

StructuredBuffer<float4> itemLowers : register(t0);

StructuredBuffer<float4> itemUppers : register(t1);

RWStructuredBuffer<float4> writeBoundsLower : register(u0);

RWStructuredBuffer<float4> writeBoundsUpper : register(u1);

[numthreads(256, 1, 1)]
void ComputeTotalBounds(uint3 tid : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
	const bool active = (int)tid.x < gNumItems;

	sBounds[gtid.x] = active ? itemLowers[tid.x].xyz
	                         : float3(3.40282347e+38f, 3.40282347e+38f, 3.40282347e+38f);
	for (uint a = 128; a > 0; a >>= 1)
	{
		GroupMemoryBarrierWithGroupSync();
		if (gtid.x < a)
		{
			sBounds[gtid.x] = min(sBounds[gtid.x], sBounds[gtid.x + a]);
		}
	}
	GroupMemoryBarrierWithGroupSync();
	const float3 totalLower = sBounds[0];
	GroupMemoryBarrierWithGroupSync();

	sBounds[gtid.x] = active ? itemUppers[tid.x].xyz
	                         : float3(-3.40282347e+38f, -3.40282347e+38f, -3.40282347e+38f);
	for (uint b = 128; b > 0; b >>= 1)
	{
		GroupMemoryBarrierWithGroupSync();
		if (gtid.x < b)
		{
			sBounds[gtid.x] = max(sBounds[gtid.x], sBounds[gtid.x + b]);
		}
	}
	GroupMemoryBarrierWithGroupSync();

	if (gtid.x == 0)
	{
		writeBoundsLower[gid.x] = float4(totalLower, 0.0f);
		writeBoundsUpper[gid.x] = float4(sBounds[0], 0.0f);
	}
}
