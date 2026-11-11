// Reconstructed from the shipped DXBC bytecode of NvFlexReleaseD3D_x64.dll,
// shader 063 (cs_5_0).  Interface (constant-buffer layout, resource bindings,
// thread group) comes from the RDEF chunk; the body is derived from the SHEX
// assembly 063_cs_5_0.asm.
//
// Evidence: work/dxbc/NvFlexReleaseD3D_x64/063.dxbc,
//           work/dxbc/NvFlexReleaseD3D_x64/063_cs_5_0.asm, work/dxbc.json

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

Buffer<uint> cellIds : register(t0);

RWStructuredBuffer<int> cellStarts : register(u0);

RWStructuredBuffer<int> cellEnds : register(u1);

[numthreads(256, 1, 1)]
void CreateGrid(uint3 tid : SV_DispatchThreadID)
{
	// CreateGrid: emit start/end offsets for every run of equal cell ids
	if ((int)tid.x < kNumParticles)
	{
		const uint cellId = cellIds[tid.x];

		if (tid.x == 0)
		{
			cellStarts[cellId] = 0;
		}
		else
		{
			const uint prevCellId = cellIds[tid.x - 1];
			if (prevCellId != cellId)
			{
				cellStarts[cellId] = tid.x;
				cellEnds[prevCellId] = tid.x;
			}
		}

		if ((int)tid.x == kNumParticles - 1)
		{
			cellEnds[cellId] = tid.x + 1;
		}
	}
}
