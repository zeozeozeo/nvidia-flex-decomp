// Reconstructed from the shipped DXBC bytecode of NvFlexReleaseD3D_x64.dll,
// shader 064 (cs_5_0).  Interface (constant-buffer layout, resource
// bindings, thread group) comes from the RDEF chunk; the body is derived from
// the SHEX assembly 064_cs_5_0.asm.
//
// Evidence: work/dxbc/NvFlexReleaseD3D_x64/064.dxbc,
//           work/dxbc/NvFlexReleaseD3D_x64/064_cs_5_0.asm, work/dxbc.json

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

RWBuffer<uint> cellHash : register(u0);

RWBuffer<uint> cellIndices : register(u1);

[numthreads(256, 1, 1)]
void CalculateParticleHash(uint3 tid : SV_DispatchThreadID)
{
	if ((int)tid.x < kNumParticles)
	{
		const int index = activeIndices[tid.x];
		const float3 position = newPositions[index].xyz;
		// gBounds holds a single float3 (stride 12, read at index 0).
		const float3 lower = gBounds[0];
		// Truncation to int matches the ftoi in the bytecode.
		const int3 coord = (int3)((position - lower) * kInvCellEdge);

		// 7-bit per-axis packing: bitfield inserts of 7-bit fields at 0, 7, 14.
		const uint hz = (uint)coord.z & 0x7f;
		const uint hy = (uint)coord.y & 0x7f;
		const uint hx = (uint)coord.x & 0x7f;
		const uint hash = (hx << 14) + (hy << 7) + hz;

		cellHash[tid.x] = hash;
		cellIndices[tid.x] = (uint)index;
	}
	else
	{
		// The bytecode initialises the pair to 0xffffffff in the inactive lanes.
		cellHash[tid.x] = 0xffffffff;
		cellIndices[tid.x] = 0xffffffff;
	}
}
