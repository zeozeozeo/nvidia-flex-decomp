// Reconstructed from the shipped DXBC bytecode of NvFlexReleaseD3D_x64.dll,
// shader 061 (cs_5_0).  Interface (constant-buffer layout, resource bindings,
// thread group) comes from the RDEF chunk; the body is derived from the SHEX
// assembly 061_cs_5_0.asm.
//
// Entry point: UpdateDiffuseParticles
// (RDEF name table: "UpdateDiffuseParticles::diffusePositions").
// Evidence: work/dxbc/NvFlexReleaseD3D_x64/061.dxbc,
//           work/dxbc/NvFlexReleaseD3D_x64/061_cs_5_0.asm, work/dxbc.json

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

StructuredBuffer<int> numDiffuseParticles : register(t0);
StructuredBuffer<int> cellBucketStarts : register(t1);
StructuredBuffer<int> cellBucketEnds : register(t2);
StructuredBuffer<int> contactCounts : register(t3);
StructuredBuffer<float4> sortedNewPositionsTex : register(t4);
StructuredBuffer<float4> sortedNewVelocitiesTex : register(t5);
StructuredBuffer<float3> gBounds : register(t7);

RWStructuredBuffer<float4> diffusePositions : register(u0);
RWStructuredBuffer<float4> diffuseVelocities : register(u1);

static float3 ClampVelocity(float3 v)
{
	const float sp2 = dot(v, v);
	const float maxV2 = kDiffuseMaxVelocity * kDiffuseMaxVelocity;
	return (maxV2 < sp2) ? (v * (rsqrt(sp2) * kDiffuseMaxVelocity)) : v;
}

[numthreads(256, 1, 1)]
void UpdateDiffuseParticles(uint3 tid : SV_DispatchThreadID)
{
	if (!((int)tid.x < numDiffuseParticles[0]))
	{
		return;
	}

	const float4 dp = diffusePositions[tid.x];
	const float3 dv = diffuseVelocities[tid.x].xyz;
	const float3 origin = gBounds[0];
	const int3 cell = (int3)((dp.xyz - origin) * kInvCellEdge);

	// linear bucket index: one 7-bit field per cell axis
	const uint cellIdx = (uint)((cell.x & 127) | ((cell.y & 127) << 7) | ((cell.z & 127) << 14));

	const int start = cellBucketStarts[cellIdx];
	const int end = cellBucketEnds[cellIdx];

	float3 sumVel = float3(0.0f, 0.0f, 0.0f);
	float sumWeight = 0.0f;
	float density = 0.0f;

	for (int j = start; j < end; j++)
	{
		const float3 d = dp.xyz - sortedNewPositionsTex[j].xyz;
		const float d2 = dot(d, d);
		const float3 nv = sortedNewVelocitiesTex[j].xyz;
		const float w = 1.0f - d2 * kInvRadius;

		if (d2 < kRadiusSq)
		{
			sumWeight += w;
			sumVel += nv * w;
			if (density == 0.0f)
			{
				density = contactCounts[j];
			}
		}
	}

	const float3 avgVel = sumVel / sumWeight;
	const bool ballistic = (density >= kDiffuseBallistic);

	// velocity from the neighbourhood average, buoyancy and drag
	const float3 vel = ClampVelocity(dv - kDiffuseBuoyancy * kGravity * kDiffuseDt
	                                 + kDiffuseDrag * (avgVel - dv));
	const float3 pos = dp.xyz + vel * kDiffuseDt;

	// pure ballistic integration
	const float3 dv2 = ClampVelocity(dv * (1.0f - kDiffuseDt) + kGravity * kDiffuseDt);
	const float3 pos2 = dp.xyz + dv2 * kDiffuseDt;

	float3 newVel = ballistic ? vel : dv2;
	float3 newPos = ballistic ? pos : pos2;

	// collide against the static planes
	for (int i = 0; i < kNumPlanes; i++)
	{
		const float4 plane = kPlanes[i];
		const float dist = dot(newPos, plane.xyz) + plane.w - kCollisionDistance;
		if (dist <= 0.0f)
		{
			newPos -= dist * plane.xyz;
			const float vdn = dot(newVel, plane.xyz);
			newVel = (vdn < 0.0f) ? (newVel - 1.5f * vdn * plane.xyz) : newVel;
		}
	}

	diffuseVelocities[tid.x] = float4(newVel, 0.0f);
	diffusePositions[tid.x] = float4(newPos, dp.w - kDiffuseDt);
}
