// Reconstructed from the shipped DXBC bytecode of NvFlexReleaseD3D_x64.dll,
// shader 019 (cs_5_0).  Interface (constant-buffer layout, resource bindings,
// thread group) comes from the RDEF chunk; the body is derived from the SHEX
// assembly 019_cs_5_0.asm.
//
// Evidence: work/dxbc/NvFlexReleaseD3D_x64/019.dxbc,
//           work/dxbc/NvFlexReleaseD3D_x64/019_cs_5_0.asm, work/dxbc.json

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

StructuredBuffer<int> activeIndices : register(t0);

StructuredBuffer<float4> newPositions : register(t1);

StructuredBuffer<float3> gBounds : register(t2);

RWStructuredBuffer<int> cellBucketStarts : register(u0);

RWStructuredBuffer<int> cellBucketEnds : register(u1);

[numthreads(256, 1, 1)]
void ClearCellBuckets(uint3 tid : SV_DispatchThreadID)
{
	// ClearCellBuckets: zero the hash bucket of every active particle's cell
	if ((int)tid.x < kNumParticles)
	{
		const uint index = activeIndices[tid.x];
		const float4 pos = newPositions[index];
		const float3 lower = gBounds[0];

		const float3 delta = (pos.xyz - lower) * kInvCellEdge;

		const uint cx = (uint)clamp(delta.x, 0.0f, 511.0f);
		const uint cy = (uint)clamp(delta.y, 0.0f, 511.0f);
		const uint cz = (uint)clamp(delta.z, 0.0f, 511.0f);

		const uint hash = (cx & 127u) | ((cy & 127u) << 7) | ((cz & 127u) << 14);

		cellBucketStarts[hash] = 0;
		cellBucketEnds[hash] = 0;
	}
}
