// Reconstructed from the shipped DXBC bytecode of NvFlexReleaseD3D_x64.dll,
// shader 021 (cs_5_0).  Interface (constant-buffer layout, resource bindings,
// thread group) comes from the RDEF chunk; the body is derived from the SHEX
// assembly 021_cs_5_0.asm.
//
// Evidence: work/dxbc/NvFlexReleaseD3D_x64/021.dxbc,
//           work/dxbc/NvFlexReleaseD3D_x64/021_cs_5_0.asm, work/dxbc.json

	// CalculateBounds: block-wide min/max over the positions of the active particles
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

[numthreads(128, 1, 1)]
void CalculateBounds(uint3 tid : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
	const bool active = (int)tid.x < kNumParticles;
	const uint index = activeIndices[tid.x];
	const float3 pos = positions[index].xyz;

	sBounds[gtid.x] = active ? pos
	                         : float3(3.40282347e+38f, 3.40282347e+38f, 3.40282347e+38f);
	for (uint a = 64; a > 0; a >>= 1)
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

	sBounds[gtid.x] = active ? pos
	                         : float3(-3.40282347e+38f, -3.40282347e+38f, -3.40282347e+38f);
	for (uint b = 64; b > 0; b >>= 1)
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
