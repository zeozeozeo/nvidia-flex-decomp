// Reconstructed from the shipped DXBC bytecode of NvFlexReleaseD3D_x64.dll,
// shader 056 (cs_5_0).  Interface (constant-buffer layout, resource bindings,
// thread group) comes from the RDEF chunk; the body is derived from the SHEX
// assembly 056_cs_5_0.asm.
//
// Entry point: SolveContacts (RDEF name table: "SolveContacts::deltas").
// Evidence: work/dxbc/NvFlexReleaseD3D_x64/056.dxbc,
//           work/dxbc/NvFlexReleaseD3D_x64/056_cs_5_0.asm, work/dxbc.json

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
    int kNumShapes;
    int kNumRigids;
    int kNumSprings;
    int kNumTriangles;
    int kMaxDiffuseParticles;
    int kDiffuseRadixSortSize;
    float kDiffuseDt;
    float kDiffuseMaxVelocity;
    float3 kDiffuseSortAxis;
    float kRelaxationFactor;
    float3 kStaticTriGridLower;
    float kDiffuseLifetime;
    float3 kWind;
    int kMaxNeighborsPerParticle;
    int kNumGeometryEntries;
    float kMaxAcceleration;
    float kMaxVelocityDelta;
    int kMaxContactsPerParticle;
};

StructuredBuffer<int> collisionCounts : register(t0);
StructuredBuffer<float4> collisionPlanes : register(t1);
StructuredBuffer<float4> prevPositions : register(t2);
StructuredBuffer<float4> collisionVelocities : register(t3);

RWStructuredBuffer<float4> positions : register(u0);
RWStructuredBuffer<float4> deltas : register(u1);

[numthreads(256, 1, 1)]
void SolveContacts(uint3 tid : SV_DispatchThreadID)
{
	// Resolve the static collision planes collected for this particle: push out
	// of every penetrating half-space, then apply adhesion and a tangential
	// friction correction against the plane's collision velocity.
	if ((int)tid.x >= kNumParticles)
	{
		return;
	}

	int count = collisionCounts[tid.x];
	if (count == 0)
	{
		return;
	}

	const float4 p = positions[tid.x];
	if (p.w == 0.0f)
	{
		return;
	}
	count -= 1;

	const bool useAdhesion = (kAdhesion != 0.0f);
	const bool useFriction = (0.0f < kDynamicFriction);
	const float3 d = p.xyz - prevPositions[tid.x].xyz;

	float3 accum = float3(0.0f, 0.0f, 0.0f);
	float numContacts = 0.0f;

	const uint base = tid.x * (uint)kMaxContactsPerParticle;
	for (int k = count; k >= 0; k--)
	{
		const uint ci = base + (uint)k;
		const float4 plane = collisionPlanes[ci];
		const float dist = dot(p.xyz, plane.xyz) + plane.w;
		const float pen = dist - kCollisionDistance;

		if (useAdhesion)
		{
			accum -= (dist * kAdhesion) * plane.xyz;
		}

		if (pen < 0.0f)
		{
			const float3 push = accum - pen * plane.xyz;
			const bool haveNormal = (dot(plane.xyz, plane.xyz) > 0.0f) && useFriction;

			float3 res = push;
			if (haveNormal)
			{
				const float3 rel = d - collisionVelocities[ci].xyz;
				const float3 tang = rel - dot(rel, plane.xyz) * plane.xyz;
				const float t2 = dot(tang, tang);
				const float limit = pen * kStaticFriction;

				if (t2 < limit * limit)
				{
					res = push - plane.xyz;
				}
				else if (0.0f < t2)
				{
					const float f = min(rsqrt(t2) * (-pen * kDynamicFriction), 1.0f);
					res = push - tang * f;
				}
			}

			accum = res;
			numContacts += 1.0f;
		}
	}

	if (0.0f < numContacts)
	{
		deltas[tid.x] += float4(accum, numContacts);
	}
}
