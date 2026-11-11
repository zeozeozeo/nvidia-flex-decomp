// Reconstructed from the shipped DXBC bytecode of NvFlexReleaseD3D_x64.dll,
// shader 036 (cs_5_0).  Interface (constant-buffer layout, resource bindings,
// thread group) comes from the RDEF chunk; the body is derived from the SHEX
// assembly 036_cs_5_0.asm.
//
// Evidence: work/dxbc/NvFlexReleaseD3D_x64/036.dxbc,
//           work/dxbc/NvFlexReleaseD3D_x64/036_cs_5_0.asm, work/dxbc.json
//
// The RDEF chunk records the original variable names: inflatables (t0),
// positions (t1), reverseLookup (t2), indices (t3), lambdas (t4) and the raw
// UAV accum (u0).  One thread group handles one inflatable; every triangle of
// the inflatable scatters the same lambda-scaled corner gradient onto its three
// vertices with float atomics (nine inlined compare-exchange loops).

struct Inflatable
{
    int offset;
    int count;
    float targetVolume;
    float stiffness;
};

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
    float kSubStepLength;
    float kMaxSpeed;
    float kInvCellEdge;
    float kStaticTriGridInvCellEdge;
};
StructuredBuffer<Inflatable> inflatables : register(t0);

StructuredBuffer<float4> positions : register(t1);

StructuredBuffer<int> reverseLookup : register(t2);

StructuredBuffer<int> indices : register(t3);

StructuredBuffer<float> lambdas : register(t4);

RWByteAddressBuffer accum : register(u0);

groupshared int2 sRange;

// Float atomic add through an integer compare-exchange loop; the shipped code
// is nine inlined copies of exactly this loop.
void AtomicAddFloat(uint addr, float value)
{
	uint expected = 0u;
	for (;;)
	{
		uint old;
		InterlockedCompareExchange(accum[addr], expected, asuint(asfloat(expected) + value), old);
		if (old == expected)
		{
			return;
		}
		expected = old;
	}
}

[numthreads(512, 1, 1)]
void SolveInflatableVolume(uint3 gid : SV_GroupThreadID, uint3 bid : SV_GroupID)
{
	const float maxDelta = kDt * kMaxSpeed;
	const float maxDeltaSq = maxDelta * maxDelta;

	if (gid.x == 0)
	{
		sRange = int2(inflatables[bid.x].offset, inflatables[bid.x].count);
	}
	GroupMemoryBarrierWithGroupSync();

	const float lambda = lambdas[bid.x];
	const int offset = sRange.x;
	const int count = sRange.y;

	for (int i = (int)gid.x; i < count; i += 512)
	{
		const int tri = offset + i;
		const int v0 = reverseLookup[indices[tri * 3 + 0]];
		const int v1 = reverseLookup[indices[tri * 3 + 1]];
		const int v2 = reverseLookup[indices[tri * 3 + 2]];

		const float3 p0 = positions[v0].xyz;
		const float3 p1 = positions[v1].xyz;
		const float3 p2 = positions[v2].xyz;

		float3 n = -(lambda * cross(p1 - p0, p2 - p0));

		const float lenSq = dot(n, n);
		if (maxDeltaSq < lenSq)
		{
			n *= maxDelta * rsqrt(lenSq);
		}

		AtomicAddFloat((uint)v0 * 16u + 0u, n.x);
		AtomicAddFloat((uint)v0 * 16u + 4u, n.y);
		AtomicAddFloat((uint)v0 * 16u + 8u, n.z);

		AtomicAddFloat((uint)v1 * 16u + 0u, n.x);
		AtomicAddFloat((uint)v1 * 16u + 4u, n.y);
		AtomicAddFloat((uint)v1 * 16u + 8u, n.z);

		AtomicAddFloat((uint)v2 * 16u + 0u, n.x);
		AtomicAddFloat((uint)v2 * 16u + 4u, n.y);
		AtomicAddFloat((uint)v2 * 16u + 8u, n.z);
	}
}
