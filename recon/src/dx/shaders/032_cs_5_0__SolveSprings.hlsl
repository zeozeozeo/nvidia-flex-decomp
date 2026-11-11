// Reconstructed from the shipped DXBC bytecode of NvFlexReleaseD3D_x64.dll,
// shader 032 (cs_5_0).  Interface (constant-buffer layout, resource bindings,
// thread group) comes from the RDEF chunk; the body is derived from the SHEX
// assembly 032_cs_5_0.asm.
//
// Evidence: work/dxbc/NvFlexReleaseD3D_x64/032.dxbc,
//           work/dxbc/NvFlexReleaseD3D_x64/032_cs_5_0.asm, work/dxbc.json

	// SolveSprings: distribute each half-edge spring correction over the
	// particle's springs in a block-wide reduction, one mass-weighted delta per
	// body.
groupshared float3 sDelta[512];

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
    float kCollideParticlesRadiusSq;
    int kMaxParticles;
    int kRelaxationMode;
    int kShapesCount;
};

StructuredBuffer<int> halfSpringOpposites : register(t1);

StructuredBuffer<float> halfSpringLengths : register(t2);

StructuredBuffer<float> halfSpringStiffness : register(t3);

StructuredBuffer<int> particleSpringBegin : register(t4);

StructuredBuffer<int> particleSpringEnd : register(t5);

StructuredBuffer<int> reverseLookup : register(t6);

StructuredBuffer<float4> sortedNewPositionsTex : register(t7);

RWByteAddressBuffer deltas : register(u0);

[numthreads(512, 1, 1)]
void SolveSprings(uint3 tid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID)
{
	const uint body0 = tid.x >> 3;
	const uint lane = tid.x & 7;

	const bool overflow = body0 >= (uint)kMaxParticles;
	const uint body = overflow ? 0u : body0;

	const uint begin = particleSpringBegin[body];
	const uint end = particleSpringEnd[body];
	const uint count = overflow ? 0u : (end - begin);

	const uint index = reverseLookup[body];
	const float4 pos = sortedNewPositionsTex[index];

	const bool active = 0.0f < pos.w;

	float3 accum = float3(0.0f, 0.0f, 0.0f);
	if (active && count > 0u)
	{
		for (uint si = begin + lane; si < end; si += 8u)
		{
			const uint other = (uint)halfSpringOpposites[si];
			const uint otherIndex = reverseLookup[other];
			const float4 otherPos = sortedNewPositionsTex[otherIndex];

			const float3 d = pos.xyz - otherPos.xyz;
			const float wsum = pos.w + otherPos.w;
			const bool wValid = 0.0f < wsum;

			const float d2 = dot(d, d);
			const bool dValid = 0.0f < d2;
			const float invw = 1.0f / wsum;
			const float dist = sqrt(d2);

			float diff = halfSpringLengths[si] - dist;
			float stiffness = halfSpringStiffness[si];
			if (stiffness < 0.0f)
			{
				diff = min(diff, 0.0f);
				stiffness = -stiffness;
			}

			const float3 cand = accum + (d / dist) * (diff * (invw * stiffness));
			const float3 next = dValid ? cand : accum;
			accum = wValid ? next : accum;
		}
	}

	sDelta[gtid.x] = accum;
	GroupMemoryBarrierWithGroupSync();

	if (lane == 0u && count > 0u && active)
	{
		float3 sum = accum + sDelta[gtid.x + 1u];
		sum += sDelta[gtid.x + 2u];
		sum += sDelta[gtid.x + 3u];
		sum += sDelta[gtid.x + 4u];
		sum += sDelta[gtid.x + 5u];
		sum += sDelta[gtid.x + 6u];
		sum += sDelta[gtid.x + 7u];

		const uint byteOffset = body * 16u;
		const float4 old = float4(asfloat(deltas.Load(byteOffset)),
		                          asfloat(deltas.Load(byteOffset + 4u)),
		                          asfloat(deltas.Load(byteOffset + 8u)),
		                          asfloat(deltas.Load(byteOffset + 12u)));
		deltas.Store(byteOffset, asuint(float4(sum * pos.w, (float)count) + old));
	}
}
