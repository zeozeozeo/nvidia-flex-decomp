// Reconstructed from the shipped DXBC bytecode of NvFlexReleaseD3D_x64.dll,
// shader 032 (cs_5_0).  Interface (constant-buffer layout, resource bindings,
// thread group) comes from the RDEF chunk; the body is derived from the SHEX
// assembly 032_cs_5_0.asm.
//
// Evidence: work/dxbc/NvFlexReleaseD3D_x64/032.dxbc,
//           work/dxbc/NvFlexReleaseD3D_x64/032_cs_5_0.asm, work/dxbc.json
//
// The RDEF chunk records the original variable names: halfSpringOpposites (t1),
// halfSpringLengths (t2), halfSpringStiffness (t3), particleSpringBegin (t4),
// particleSpringEnd (t5), reverseLookup (t6), sortedNewPositionsTex (t7) and
// deltas (u0).  Eight threads per particle walk that particle's half springs
// with stride 8, the eight partial corrections are summed in shared memory by
// lane 0, and lane 0 adds them to the particle's delta accumulator.
//
// DEVIATION, forced by the SM5 front end: vkd3d-shader exposes neither Load nor
// a plain raw read on RWByteAddressBuffer, so the shipped
//   ld_raw_indexable r3.xyzw, r0.x, u0.xyzw
// is spelled as four non-committing 32-bit atomics (InterlockedAdd with zero),
// which return exactly the stored word and never write.  The accumulator is
// owned by this one lane, so the value read is identical; only the instruction
// used to read it differs.

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

// float4 per particle: xyz accumulated spring correction, w spring count.
RWByteAddressBuffer deltas : register(u0);

groupshared float3 sDeltas[512];

// Read-only 32-bit access to a raw UAV word (the front end has no raw Load).
uint LoadRaw(uint addr)
{
	uint old;
	InterlockedAdd(deltas[addr], 0u, old);
	return old;
}

[numthreads(512, 1, 1)]
void SolveSprings(uint3 tid : SV_DispatchThreadID, uint3 gid : SV_GroupThreadID)
{
	// Eight threads share one particle; the thread index is clamped so the
	// tail threads index particle 0 and are then gated out by the count.
	const uint group = tid.x >> 3;
	const uint lane = tid.x & 7u;
	const bool inRange = group < (uint)kMaxParticles;
	const uint particle = inRange ? group : 0u;

	const int begin = particleSpringBegin[particle];
	const int end = particleSpringEnd[particle];
	const int count = inRange ? (end - begin) : 0;

	const int lookup = reverseLookup[particle];
	const float4 pos = sortedNewPositionsTex[lookup];

	const bool movable = 0.0f < pos.w;
	const bool hasSprings = 0 < count;

	float3 accum = float3(0.0f, 0.0f, 0.0f);
	if (movable && hasSprings)
	{
		for (int i = begin + (int)lane; i < end; i += 8)
		{
			const int other = reverseLookup[halfSpringOpposites[i]];
			const float4 p = sortedNewPositionsTex[other];

			const float3 d = pos.xyz - p.xyz;
			const float massSum = pos.w + p.w;
			const float lenSq = dot(d, d);
			const bool nonDegenerate = 0.0f < lenSq;

			const float invMassSum = 1.0f / massSum;
			const float len = sqrt(lenSq);

			const float rest = halfSpringLengths[i];
			float stiffness = halfSpringStiffness[i];
			float diff = rest - len;
			if (stiffness < 0.0f)
			{
				diff = min(diff, 0.0f);
				stiffness = -stiffness;
			}

			const float3 dir = d / len;
			const float scale = invMassSum * stiffness * diff;

			const float3 candidate = accum + dir * scale;
			const float3 corrected = nonDegenerate ? candidate : accum;
			accum = (0.0f < massSum) ? corrected : accum;
		}
	}

	sDeltas[gid.x] = accum;
	GroupMemoryBarrierWithGroupSync();

	if (movable && hasSprings && lane == 0u)
	{
		// Fold this particle's eight lanes, scale by the inverse mass and add
		// the particle's spring count into the accumulator's fourth channel.
		float3 sum = sDeltas[gid.x];
		sum += sDeltas[gid.x + 1];
		sum += sDeltas[gid.x + 2];
		sum += sDeltas[gid.x + 3];
		sum += sDeltas[gid.x + 4];
		sum += sDeltas[gid.x + 5];
		sum += sDeltas[gid.x + 6];
		sum += sDeltas[gid.x + 7];

		const uint addr = (uint)lookup * 16u;

		float4 accumulated;
		accumulated.x = asfloat(LoadRaw(addr + 0u));
		accumulated.y = asfloat(LoadRaw(addr + 4u));
		accumulated.z = asfloat(LoadRaw(addr + 8u));
		accumulated.w = asfloat(LoadRaw(addr + 12u));

		accumulated += float4(pos.w * sum, (float)count);

		deltas.Store4(addr, asuint(accumulated));
	}
}
