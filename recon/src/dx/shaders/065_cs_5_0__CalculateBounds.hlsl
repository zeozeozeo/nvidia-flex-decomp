// Reconstructed from the shipped DXBC bytecode of NvFlexReleaseD3D_x64.dll,
// shader 065 (cs_5_0).  Interface (constant-buffer layout, resource
// bindings, thread group) comes from the RDEF chunk; the body is derived from
// the SHEX assembly 065_cs_5_0.asm.
//
// Evidence: work/dxbc/NvFlexReleaseD3D_x64/065.dxbc,
//           work/dxbc/NvFlexReleaseD3D_x64/065_cs_5_0.asm, work/dxbc.json

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

// Lock-free five-level bounds tree (raw UAV); the node byte offsets are the
// immediates 0x50401f04, 0x50201f04, 0x50101f04, 0x50081f04, 0x50041f04 that
// the shipped bytecode passes to imm_atomic_cmp_exch.
RWByteAddressBuffer boundsTree : register(u7);

groupshared float3 sBounds[128];

void AtomicMinFloat(uint addr, inout float v)
{
	uint expected = 0u;
	for (;;)
	{
		uint old;
		InterlockedCompareExchange(boundsTree[addr], expected, asuint(min(asfloat(expected), v)), old);
		if (old == expected)
		{
			return;
		}
		expected = old;
	}
}

void AtomicMaxFloat(uint addr, inout float v)
{
	uint expected = 0u;
	for (;;)
	{
		uint old;
		InterlockedCompareExchange(boundsTree[addr], expected, asuint(max(asfloat(expected), v)), old);
		if (old == expected)
		{
			return;
		}
		expected = old;
	}
}

void ReduceMin(uint3 gid, float3 v)
{
	AtomicMinFloat(0x50401f04u, v.x);
	AtomicMinFloat(0x50401f04u, v.y);
	AtomicMinFloat(0x50401f04u, v.z);
	AtomicMinFloat(0x50201f04u, v.x);
	AtomicMinFloat(0x50201f04u, v.y);
	AtomicMinFloat(0x50201f04u, v.z);
	AtomicMinFloat(0x50101f04u, v.x);
	AtomicMinFloat(0x50101f04u, v.y);
	AtomicMinFloat(0x50101f04u, v.z);
	AtomicMinFloat(0x50081f04u, v.x);
	AtomicMinFloat(0x50081f04u, v.y);
	AtomicMinFloat(0x50081f04u, v.z);
	v = min(v, min(v, v));
	AtomicMinFloat(0x50041f04u, v.x);
	AtomicMinFloat(0x50041f04u, v.y);
	AtomicMinFloat(0x50041f04u, v.z);
}

void ReduceMax(uint3 gid, float3 v)
{
	AtomicMaxFloat(0x50401f04u, v.x);
	AtomicMaxFloat(0x50401f04u, v.y);
	AtomicMaxFloat(0x50401f04u, v.z);
	AtomicMaxFloat(0x50201f04u, v.x);
	AtomicMaxFloat(0x50201f04u, v.y);
	AtomicMaxFloat(0x50201f04u, v.z);
	AtomicMaxFloat(0x50101f04u, v.x);
	AtomicMaxFloat(0x50101f04u, v.y);
	AtomicMaxFloat(0x50101f04u, v.z);
	AtomicMaxFloat(0x50081f04u, v.x);
	AtomicMaxFloat(0x50081f04u, v.y);
	AtomicMaxFloat(0x50081f04u, v.z);
	AtomicMaxFloat(0x50041f04u, v.x);
	AtomicMaxFloat(0x50041f04u, v.y);
	AtomicMaxFloat(0x50041f04u, v.z);
}

[numthreads(128, 1, 1)]
void CalculateBounds(uint3 tid : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint gi : SV_GroupIndex)
{
	const bool valid = tid.x < ((kNumParticles + 127) >> 7);
	const uint lane = gi & 31;
	const uint warp = gi >> 5;

	// ---- minimum pass ----
	float3 mn = valid ? positions[tid.x].xyz : float3(3.40282347e+38, 3.40282347e+38, 3.40282347e+38);
	ReduceMin(gid, mn);

	if (lane == 0)
	{
		sBounds[warp] = mn;
	}
	GroupMemoryBarrierWithGroupSync();

	if (gi == 0)
	{
		float3 v = min(min(sBounds[0], sBounds[1]), min(sBounds[2], sBounds[3]));
		sBounds[0] = v;
	}
	GroupMemoryBarrierWithGroupSync();
	const float3 groupMin = sBounds[0];
	GroupMemoryBarrierWithGroupSync();

	// ---- maximum pass ----
	float3 mx = valid ? predicted[tid.x].xyz : float3(-3.40282347e+38, -3.40282347e+38, -3.40282347e+38);
	ReduceMax(gid, mx);

	if (lane == 0)
	{
		sBounds[warp] = mx;
	}
	GroupMemoryBarrierWithGroupSync();

	if (gi == 0)
	{
		const float3 v = max(max(sBounds[0], sBounds[1]), max(sBounds[2], sBounds[3]));
		boundsMin[gid.x] = float4(groupMin, 0.0f);
		boundsMax[gid.x] = float4(v, 0.0f);
	}
}
