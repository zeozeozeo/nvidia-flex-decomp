// Reconstructed from the shipped DXBC bytecode of NvFlexReleaseD3D_x64.dll,
// shader 030 (cs_5_0).  Interface (constant-buffer layout, resource bindings,
// thread group) comes from the RDEF chunk; the body is derived from the SHEX
// assembly 030_cs_5_0.asm.
//
// Evidence: work/dxbc/NvFlexReleaseD3D_x64/030.dxbc,
//           work/dxbc/NvFlexReleaseD3D_x64/030_cs_5_0.asm, work/dxbc.json

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

StructuredBuffer<float4> sortedNewPositions : register(t0);

StructuredBuffer<float4> sortedNewVelocities : register(t1);

StructuredBuffer<float> sortedDensities : register(t2);

ByteAddressBuffer impulses : register(t3);

Buffer<uint> indices : register(t4);

RWStructuredBuffer<float4> positions : register(u0);

RWStructuredBuffer<float3> velocities : register(u1);

RWStructuredBuffer<float> densities : register(u2);

[numthreads(256, 1, 1)]
void Finalize(uint3 tid : SV_DispatchThreadID)
{
	// Finalize: write the sorted positions back to their particle slot, apply the
	// contact impulse to the velocity (clamped to kMaxVelocityDelta) and rescale
	// the density.
	if ((int)tid.x < kNumParticles)
	{
		const float4 newPos = sortedNewPositions[tid.x];
		if (0.0f < newPos.w)
		{
			const uint index = indices[tid.x];

			float4 pos = positions[index];
			pos.xyz = newPos.xyz;
			positions[index] = pos;

			const float4 impulse = asfloat(impulses.Load4(tid.x * 16u));
			const float3 oldVel = velocities[index];
			const float3 newVel = sortedNewVelocities[tid.x].xyz;

			const float3 delta = (newVel + impulse.xyz / max(impulse.w, 1.0f)) - oldVel;
			const float len2 = dot(delta, delta);
			const float limit = kMaxVelocityDelta * kMaxVelocityDelta;

			const float3 clamped = (limit < len2)
				? delta * (rsqrt(len2) * kMaxVelocityDelta)
				: delta;

			velocities[index] = oldVel + clamped;
			densities[index] = sortedDensities[tid.x] * kInvRestDensity;
		}
	}
}
