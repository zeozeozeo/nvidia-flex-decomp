// Reconstructed from the shipped DXBC bytecode of NvFlexReleaseD3D_x64.dll,
// shader 014 (cs_5_0).  Interface (constant-buffer layout, resource bindings,
// thread group) comes from the RDEF chunk; the body is derived from the SHEX
// assembly 014_cs_5_0.asm.
//
// Evidence: work/dxbc/NvFlexReleaseD3D_x64/014.dxbc,
//           work/dxbc/NvFlexReleaseD3D_x64/014_cs_5_0.asm, work/dxbc.json

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

StructuredBuffer<float4> positions : register(t0);

StructuredBuffer<float4> newPositions : register(t1);

StructuredBuffer<float3> velocities : register(t2);

StructuredBuffer<int> phases : register(t3);

Buffer<uint> indices : register(t4);

StructuredBuffer<float4> normals : register(t5);

RWStructuredBuffer<float4> sortedPositions : register(u0);

RWStructuredBuffer<float4> sortedNewPositions : register(u1);

RWStructuredBuffer<float4> sortedVelocities : register(u2);

RWStructuredBuffer<int> sortedPhases : register(u3);

RWStructuredBuffer<int> reverseLookup : register(u4);

RWStructuredBuffer<float4> sortedNormals : register(u5);

[numthreads(256, 1, 1)]
void ReorderParticles(uint3 tid : SV_DispatchThreadID)
{
	// ReorderParticles: scatter into sorted order and write the inverse permutation
	if ((int)tid.x < kNumParticles)
	{
		const uint index = indices[tid.x];

		sortedPositions[tid.x] = positions[index];
		sortedNewPositions[tid.x] = newPositions[index];
		sortedPhases[tid.x] = phases[index];
		sortedNormals[tid.x] = normals[index];

		float4 v = sortedVelocities[tid.x];
		v.xyz = velocities[index];
		sortedVelocities[tid.x] = v;

		reverseLookup[index] = tid.x;
	}
}
