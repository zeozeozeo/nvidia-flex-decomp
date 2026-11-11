// Reconstructed from the shipped DXBC bytecode of NvFlexReleaseD3D_x64.dll,
// shader 003 (cs_5_0).  Interface (constant-buffer layout, resource bindings,
// thread group) comes from the RDEF chunk; the body is derived from the SHEX
// assembly 003_cs_5_0.asm.
//
// Evidence: work/dxbc/NvFlexReleaseD3D_x64/003.dxbc,
//           work/dxbc/NvFlexReleaseD3D_x64/003_cs_5_0.asm, work/dxbc.json

	// ComputeTotalBounds (second level): reduce the per-group bounds of the
	// previous pass, one output entry per group.
groupshared float3 sBounds[256];

cbuffer constBVH : register(b0)
{
    int gNumItems;
    float3 _fdpad0_0;
};

StructuredBuffer<float4> readBoundsLower : register(t2);

StructuredBuffer<float4> readBoundsUpper : register(t3);

RWStructuredBuffer<float4> writeBoundsLower : register(u0);

RWStructuredBuffer<float4> writeBoundsUpper : register(u1);

[numthreads(256, 1, 1)]
void ComputeTotalBounds(uint3 tid : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
	const uint numGroups = ((uint)gNumItems + 255u) >> 8;
	const bool active = tid.x < numGroups;

	sBounds[gtid.x] = active ? readBoundsLower[tid.x].xyz
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

	sBounds[gtid.x] = active ? readBoundsUpper[tid.x].xyz
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
