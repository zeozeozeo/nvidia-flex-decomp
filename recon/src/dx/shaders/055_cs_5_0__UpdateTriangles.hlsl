// Reconstructed from the shipped DXBC bytecode of NvFlexReleaseD3D_x64.dll,
// shader 055 (cs_5_0).  Interface (constant-buffer layout, resource bindings,
// thread group) comes from the RDEF chunk; the body is derived from the SHEX
// assembly 055_cs_5_0.asm.
//
// Entry point: UpdateTriangles (RDEF name table: "UpdateTriangles::deltas").
// Evidence: work/dxbc/NvFlexReleaseD3D_x64/055.dxbc,
//           work/dxbc/NvFlexReleaseD3D_x64/055_cs_5_0.asm, work/dxbc.json

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

RWByteAddressBuffer deltas : register(u0);

[numthreads(256, 1, 1)]
void UpdateTriangles(uint3 tid : SV_DispatchThreadID)
{
	// Zero the per-particle delta accumulator (float4 at particle index).
	if ((int)tid.x < kNumParticles)
	{
		deltas.Store(tid.x * 16, uint4(0u, 0u, 0u, 0u));
	}
}
