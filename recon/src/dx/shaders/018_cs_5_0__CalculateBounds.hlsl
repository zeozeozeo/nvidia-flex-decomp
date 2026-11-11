// Reconstructed from the shipped DXBC bytecode of NvFlexReleaseD3D_x64.dll,
// shader 018 (cs_5_0).  Interface (constant-buffer layout, resource bindings,
// thread group) comes from the RDEF chunk; the body is derived from the SHEX
// assembly 018_cs_5_0.asm.
//
// Evidence: work/dxbc/NvFlexReleaseD3D_x64/018.dxbc,
//           work/dxbc/NvFlexReleaseD3D_x64/018_cs_5_0.asm, work/dxbc.json

groupshared float3 sBounds[128];

cbuffer consts : register(b0)
{
    float4 kPlanes[8];
    int kNumPlanes;
    float kRadius;
    float kRadiusSq;
    float kInvRadius;
    float kPoly6;
    float kSpiky1;
    float kSpiky2;
    float kAkinci1;
    float kAkinci2;
    float kCohesion1;
    float kCohesion2;
    float _pad0;
    float kRestDensity;
    float kInvRestDensity;
    float kFluidRestDistance;
    float kSolidRestDistance;
    float kLambdaScale;
    float kCohesion;
    float kSurfaceTension;
    float kSolidPressure;
    float kVorticityConfinement;
    float kAnisotropy;
    float kAnisotropyMin;
    float kAnisotropyMax;
    float kSmoothing;
    float kViscosity;
    float kFreeSurfaceDrag;
    float kBuoyancy;
    float kDiffuseThreshold;
    float kDiffuseBuoyancy;
    float kDiffuseDrag;
    int kDiffuseBallistic;
    float3 kGravity;
    float kDynamicFriction;
    float kStaticFriction;
    float kParticleFriction;
    float kAdhesion;
    float kDissipation;
    float kDamping;
    float kInertiaBias;
    float kCollisionDistance;
    float kCollisionThreshold;
    float kCollisionMargin;
    float kCollisionMarginSq;
    float _pad1;
    float _pad2;
    float kDrag;
    float kLift;
    float kPlasticThreshold_deprecated;
    float kPlasticCreep_deprecated;
    float kSleepThreshold;
    float kSleepThresholdSq;
    float kShockPropagation;
    float kRestitution;
    float kSOR;
    float3 _pad3;
    int kNumParticles;
    int kNumParticlesAligned;
    float kDt;
    float kInvDt;
};

StructuredBuffer<int> activeIndices : register(t0);

StructuredBuffer<float4> positions : register(t1);

RWStructuredBuffer<float4> writeBoundsLower : register(u0);

RWStructuredBuffer<float4> writeBoundsUpper : register(u1);

RWByteAddressBuffer AmdDxExtShaderIntrinsicsUAV : register(u7);

// AMD shader-intrinsics encoding, transcribed verbatim (see 006).
uint3 CasGroup(uint addr, float3 v)
{
	uint3 old;
	InterlockedCompareExchange(AmdDxExtShaderIntrinsicsUAV[addr], asuint(v.x), 0u, old.x);
	InterlockedCompareExchange(AmdDxExtShaderIntrinsicsUAV[addr], asuint(v.y), 0u, old.y);
	InterlockedCompareExchange(AmdDxExtShaderIntrinsicsUAV[addr], asuint(v.z), 0u, old.z);
	return old;
}

[numthreads(128, 1, 1)]
void CalculateBounds(uint3 tid : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
	// CalculateBounds (AMD extension path): block min/max over the active
	// particles' positions published through the extension UAV's vector atomics.
	const float3 big = float3(3.40282347e+38f, 3.40282347e+38f, 3.40282347e+38f);
	const float3 small = float3(-3.40282347e+38f, -3.40282347e+38f, -3.40282347e+38f);

	const bool active = (int)tid.x < kNumParticles;
	const uint index = activeIndices[tid.x];
	const float3 pos = positions[index].xyz;

	const uint lane = gtid.x & 31u;
	const uint slot = gtid.x >> 5u;

	float3 lo = active ? pos : big;
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
		sBounds[0] = r;
	}
	GroupMemoryBarrierWithGroupSync();
	const float3 totalLower = sBounds[0];
	GroupMemoryBarrierWithGroupSync();

	float3 hi = active ? pos : small;
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
		sBounds[0] = r;
	}
	GroupMemoryBarrierWithGroupSync();

	if (gtid.x == 0u)
	{
		writeBoundsLower[gid.x] = float4(totalLower, 0.0f);
		writeBoundsUpper[gid.x] = float4(sBounds[0], 0.0f);
	}
}
