// Reconstructed from the shipped DXBC bytecode of NvFlexReleaseD3D_x64.dll,
// shader 023 (cs_5_0).  Interface (constant-buffer layout, resource bindings,
// thread group) comes from the RDEF chunk; the body is derived from the SHEX
// assembly 023_cs_5_0.asm.
//
// Evidence: work/dxbc/NvFlexReleaseD3D_x64/023.dxbc,
//           work/dxbc/NvFlexReleaseD3D_x64/023_cs_5_0.asm, work/dxbc.json

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
};

StructuredBuffer<float> potentials : register(t0);

StructuredBuffer<float4> sortedNewPositionsTex : register(t1);

StructuredBuffer<float4> sortedNewVelocitiesTex : register(t2);

StructuredBuffer<float> random : register(t3);

RWStructuredBuffer<int> numDiffuseParticles : register(u0);

RWStructuredBuffer<float4> diffusePositions : register(u1);

RWStructuredBuffer<float4> diffuseVelocities : register(u2);

[numthreads(256, 1, 1)]
void CreateDiffuseParticles(uint3 tid : SV_DispatchThreadID)
{
	// CreateDiffuseParticles: emit a diffuse spray particle for every sorted
	// particle whose potential exceeds kDiffuseThreshold.
	if ((int)tid.x < kNumParticles)
	{
		const int age = potentials[tid.x];
		if (kDiffuseThreshold < (float)age)
		{
			uint slot;
			InterlockedAdd(numDiffuseParticles[0], 1, slot);

			if ((int)slot < kMaxDiffuseParticles)
			{
				const float3 pos = sortedNewPositionsTex[tid.x].xyz;
				const float3 vel = sortedNewVelocitiesTex[tid.x].xyz;

				const float rnd0 = asfloat(random[tid.x & 255u]);
				const float rnd1 = asfloat(random[(tid.x + 1u) & 255u]);
				const float rnd2 = asfloat(random[(tid.x + 2u) & 255u]);

				const float lifetime = min((float)age / kDiffuseThreshold * kDiffuseLifetime,
				                           kDiffuseLifetime);

				const float3 base = pos - vel * rnd0 * kDiffuseDt;
				const float3 spray = vel * rnd0 * kFluidRestDistance * 0.25f + base;

				diffusePositions[slot] = float4(spray, lifetime);
				diffuseVelocities[slot] = float4(vel, 0.0f);
			}
		}
	}
}
