// Reconstructed from the shipped DXBC bytecode of NvFlexReleaseD3D_x64.dll,
// shader 058 (cs_5_0).  Interface (constant-buffer layout, resource bindings,
// thread group) comes from the RDEF chunk; the body is derived from the SHEX
// assembly 058_cs_5_0.asm.
//
// Entry point: CollideParticles (RDEF name table: "CollideParticles::contactCounts").
// Evidence: work/dxbc/NvFlexReleaseD3D_x64/058.dxbc,
//           work/dxbc/NvFlexReleaseD3D_x64/058_cs_5_0.asm, work/dxbc.json

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
};

StructuredBuffer<int> cellStarts : register(t0);
StructuredBuffer<int> cellEnds : register(t1);
Buffer<uint> indices : register(t2);
StructuredBuffer<float4> sortedNewPositions : register(t3);
StructuredBuffer<float4> restPositions : register(t4);
StructuredBuffer<uint> phases : register(t5);
StructuredBuffer<float3> bounds : register(t6);

RWStructuredBuffer<int> contacts : register(u0);
RWStructuredBuffer<int> contactCounts : register(u1);

static const uint kPhaseGroupMask = 1048575u;  // 0xfffff
static const uint kPhaseFlagB = 1048576u;      // 1 << 20
static const uint kPhaseFlagA = 2097152u;      // 1 << 21

[numthreads(256, 1, 1)]
void CollideParticles(uint3 tid : SV_DispatchThreadID)
{
	if ((int)tid.x >= kNumParticles)
	{
		return;
	}

	const uint i = tid.x;
	const float3 p = sortedNewPositions[i].xyz;
	const float3 origin = bounds[0];
	const int3 cell = (int3)((p - origin) * kInvCellEdge);

	const uint phase = phases[i];
	const uint group = phase & kPhaseGroupMask;
	const bool m = (phase & kPhaseFlagA) != 0u;
	const bool flag = (phase & kPhaseFlagB) != 0u;

	const float3 myRest = (m ? restPositions[indices[i]].xyz : p);

	int count = 0;
	int slot = (int)i;

	// scan the 3x3x3 cell neighbourhood; the linear cell index packs the three
	// cell coordinates into 7 bits each (z in bits 0..6, y 7..13, x 14..20)
	for (int cz = cell.z - 1; cz <= cell.z + 1; cz++)
	{
		const int base = (cz & 127);

		for (int cy = cell.y - 1; cy <= cell.y + 1; cy++)
		{
			const int cellYZ = base | ((cy & 127) << 7);

			for (int cx = cell.x - 1; cx <= cell.x + 1; cx++)
			{
				const int cellIdx = cellYZ | ((cx & 127) << 14);
				const int start = cellStarts[cellIdx];
				const int end = cellEnds[cellIdx];

				for (int j = start; j < end; j++)
				{
					if (j == (int)i)
					{
						continue;
					}

					const uint otherPhase = phases[j];
					const bool sameGroup = (group == (otherPhase & kPhaseGroupMask));
					if (!(m || !sameGroup))
					{
						continue;
					}

					if (flag && sameGroup)
					{
						const float3 otherRest = restPositions[indices[j]].xyz;
						if (dot(myRest - otherRest, myRest - otherRest) < kRadiusSq)
						{
							continue;
						}
					}

					const float3 q = sortedNewPositions[j].xyz;
					if (dot(p - q, p - q) < kCollideParticlesRadiusSq)
					{
						if (count < kMaxNeighborsPerParticle)
						{
							contacts[slot] = j;
							slot += kNumParticlesAligned;
							count += 1;
						}
					}
				}
			}
		}
	}

	contactCounts[i] = count;
}
