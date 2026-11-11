// Reconstructed from the shipped DXBC bytecode of NvFlexReleaseD3D_x64.dll,
// shader 039 (cs_5_0).  Interface (constant-buffer layout, resource bindings,
// thread group) comes from the RDEF chunk; the body is derived from the SHEX
// assembly 039_cs_5_0.asm.
//
// Evidence: work/dxbc/NvFlexReleaseD3D_x64/039.dxbc,
//           work/dxbc/NvFlexReleaseD3D_x64/039_cs_5_0.asm, work/dxbc.json
//
// Inferred-only facts are listed in units/nvflexd3d/039_cs_5_0.md.

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

StructuredBuffer<float4> lowerPositions : register(t2);

StructuredBuffer<float4> upperPositions : register(t3);

// u2 is a two-entry float3 buffer: [0] lower bound, [1] upper bound.
RWStructuredBuffer<float3> totalBounds : register(u2);

groupshared float3 sReduction[128];

static const float kFloatMax = 3.402823466e+38f;

[numthreads(128, 1, 1)]
void CalculateTotalBounds(uint3 tid : SV_DispatchThreadID, uint3 gid : SV_GroupThreadID)
{
	// CalculateTotalBounds: block-wide min/max reduction over the element range
	// followed by a single write of the (lower - kRadius, upper + kRadius) pair.
	// The element predicate is derived from the dispatch counter with two
	// successive round-up divisions by the group size (see the unit notes).
	const int numItems = ((kNumParticles + 127) / 128 + 127) / 128;
	const bool active = (int)tid.x < numItems;

	float3 lo = lowerPositions[tid.x].xyz;
	if (!active)
	{
		lo = float3(kFloatMax, kFloatMax, kFloatMax);
	}
	sReduction[gid.x] = lo;

	for (uint stride = 64u; stride > 0u; stride >>= 1)
	{
		GroupMemoryBarrierWithGroupSync();
		if (gid.x < stride)
		{
			sReduction[gid.x] = min(sReduction[gid.x], sReduction[gid.x + stride]);
		}
	}

	GroupMemoryBarrierWithGroupSync();
	const float3 reducedLo = sReduction[0];
	GroupMemoryBarrierWithGroupSync();

	float3 hi = upperPositions[tid.x].xyz;
	if (!active)
	{
		hi = float3(-kFloatMax, -kFloatMax, -kFloatMax);
	}
	sReduction[gid.x] = hi;

	for (uint stride2 = 64u; stride2 > 0u; stride2 >>= 1)
	{
		GroupMemoryBarrierWithGroupSync();
		if (gid.x < stride2)
		{
			sReduction[gid.x] = max(sReduction[gid.x], sReduction[gid.x + stride2]);
		}
	}

	GroupMemoryBarrierWithGroupSync();

	if (gid.x == 0)
	{
		totalBounds[0] = reducedLo - kRadius;
		totalBounds[1] = sReduction[0] + kRadius;
	}
}
