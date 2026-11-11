// Reconstructed from the shipped DXBC bytecode of NvFlexReleaseD3D_x64.dll,
// shader 006 (cs_5_0).  Interface (constant-buffer layout, resource bindings,
// thread group) comes from the RDEF chunk; the body is derived from the SHEX
// assembly 006_cs_5_0.asm.
//
// Evidence: work/dxbc/NvFlexReleaseD3D_x64/006.dxbc,
//           work/dxbc/NvFlexReleaseD3D_x64/006_cs_5_0.asm, work/dxbc.json

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

RWByteAddressBuffer AmdDxExtShaderIntrinsicsUAV : register(u7);

// The shipped bytecode drives the AMD shader-intrinsics UAV (u7) with one
// imm_atomic_cmp_exch per component at a fixed opcode address.  This is the
// extension's encoding, not a retry loop; it is transcribed verbatim.
uint3 CasGroup(uint addr, float3 v)
{
	uint3 old;
	InterlockedCompareExchange(AmdDxExtShaderIntrinsicsUAV[addr], asuint(v.x), 0u, old.x);
	InterlockedCompareExchange(AmdDxExtShaderIntrinsicsUAV[addr], asuint(v.y), 0u, old.y);
	InterlockedCompareExchange(AmdDxExtShaderIntrinsicsUAV[addr], asuint(v.z), 0u, old.z);
	return old;
}

[numthreads(256, 1, 1)]
void ComputeTotalBounds(uint3 tid : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
	// ComputeTotalBounds (first level, AMD extension path): block min/max over
	// the item bounds published through the extension UAV's vector atomics.
	const float3 big = float3(3.40282347e+38f, 3.40282347e+38f, 3.40282347e+38f);
	const float3 small = float3(-3.40282347e+38f, -3.40282347e+38f, -3.40282347e+38f);

	const bool active = (int)tid.x < gNumItems;
	const uint lane = gtid.x & 31u;
	const uint slot = gtid.x >> 5u;

	float3 lo = active ? itemLowers[tid.x].xyz : big;
	lo = min(lo, asfloat(CasGroup(1346379524u, lo)));
	lo = min(lo, asfloat(CasGroup(1344282372u, lo)));
	lo = min(lo, asfloat(CasGroup(1343233796u, lo)));
	lo = min(lo, asfloat(CasGroup(1342709508u, lo)));
	const uint3 loOld = CasGroup(1342447364u, lo);

	if (lane == 0u)
	{
		lo = min(lo, asfloat(loOld));
		sBounds[slot] = lo;
	}
	GroupMemoryBarrierWithGroupSync();

	if (gtid.x == 0u)
	{
		float3 r = sBounds[0];
		r = min(r, sBounds[1]);
		r = min(r, sBounds[2]);
		r = min(r, sBounds[3]);
		r = min(r, sBounds[4]);
		r = min(r, sBounds[5]);
		r = min(r, sBounds[6]);
		r = min(r, sBounds[7]);
		sBounds[0] = r;
	}
	GroupMemoryBarrierWithGroupSync();
	const float3 totalLower = sBounds[0];
	GroupMemoryBarrierWithGroupSync();

	float3 hi = active ? itemUppers[tid.x].xyz : small;
	hi = max(hi, asfloat(CasGroup(1346379524u, hi)));
	hi = max(hi, asfloat(CasGroup(1344282372u, hi)));
	hi = max(hi, asfloat(CasGroup(1343233796u, hi)));
	hi = max(hi, asfloat(CasGroup(1342709508u, hi)));
	const uint3 hiOld = CasGroup(1342447364u, hi);

	if (lane == 0u)
	{
		hi = max(hi, asfloat(hiOld));
		sBounds[slot] = hi;
	}
	GroupMemoryBarrierWithGroupSync();

	if (gtid.x == 0u)
	{
		float3 r = sBounds[0];
		r = max(r, sBounds[1]);
		r = max(r, sBounds[2]);
		r = max(r, sBounds[3]);
		r = max(r, sBounds[4]);
		r = max(r, sBounds[5]);
		r = max(r, sBounds[6]);
		r = max(r, sBounds[7]);
		sBounds[0] = r;
	}
	GroupMemoryBarrierWithGroupSync();

	if (gtid.x == 0u)
	{
		writeBoundsLower[gid.x] = float4(totalLower, 0.0f);
		writeBoundsUpper[gid.x] = float4(sBounds[0], 0.0f);
	}
}
