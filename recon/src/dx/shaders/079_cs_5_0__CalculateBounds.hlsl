// Reconstructed from the shipped DXBC bytecode of NvFlexReleaseD3D_x64.dll,
// shader 079 (cs_5_0).  Interface (constant-buffer layout, resource
// bindings, thread group) comes from the RDEF chunk; the body is derived from
// the SHEX assembly 079_cs_5_0.asm.
//
// Evidence: work/dxbc/NvFlexReleaseD3D_x64/079.dxbc,
//           work/dxbc/NvFlexReleaseD3D_x64/079_cs_5_0.asm, work/dxbc.json

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

StructuredBuffer<float4> positions : register(t2);

StructuredBuffer<float4> predicted : register(t3);

RWStructuredBuffer<float4> boundsMin : register(u0);

RWStructuredBuffer<float4> boundsMax : register(u1);

// One float3 per thread of the group, reduced in-place in a shared tree.
groupshared float3 sBounds[128];

[numthreads(128, 1, 1)]
void CalculateBounds(uint3 tid : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint gi : SV_GroupIndex)
{
	// The shipped bytecode bounds the global thread id by ceil(n / 128), so only
	// the first ceil(n/128) threads contribute; the rest reduce the identity.
	bool valid = tid.x < ((kNumParticles + 127) >> 7);

	float3 nonexistent = float3(3.40282347e+38, 3.40282347e+38, 3.40282347e+38);
	sBounds[gi] = valid ? positions[tid.x].xyz : nonexistent;

	for (uint s = 64; s > 0; s >>= 1)
	{
		GroupMemoryBarrierWithGroupSync();
		if (gi < s)
		{
			sBounds[gi] = min(sBounds[gi], sBounds[gi + s]);
		}
	}
	GroupMemoryBarrierWithGroupSync();

	// groupMin is broadcast through shared memory before the max pass overwrites it
	float3 groupMin = sBounds[0];
	GroupMemoryBarrierWithGroupSync();

	float3 nonexistentMax = float3(-3.40282347e+38, -3.40282347e+38, -3.40282347e+38);
	sBounds[gi] = valid ? predicted[tid.x].xyz : nonexistentMax;

	for (uint s2 = 64; s2 > 0; s2 >>= 1)
	{
		GroupMemoryBarrierWithGroupSync();
		if (gi < s2)
		{
			sBounds[gi] = max(sBounds[gi], sBounds[gi + s2]);
		}
	}
	GroupMemoryBarrierWithGroupSync();

	if (gi == 0)
	{
		boundsMin[gid.x] = float4(groupMin, 0.0f);
		boundsMax[gid.x] = float4(sBounds[0], 0.0f);
	}
}
