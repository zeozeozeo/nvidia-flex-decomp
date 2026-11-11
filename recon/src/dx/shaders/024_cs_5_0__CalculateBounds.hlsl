// Reconstructed from the shipped DXBC bytecode of NvFlexReleaseD3D_x64.dll,
// shader 024 (cs_5_0).  Interface (constant-buffer layout, resource bindings,
// thread group) comes from the RDEF chunk; the body is derived from the SHEX
// assembly 024_cs_5_0.asm.
//
// Evidence: work/dxbc/NvFlexReleaseD3D_x64/024.dxbc,
//           work/dxbc/NvFlexReleaseD3D_x64/024_cs_5_0.asm, work/dxbc.json
//
// The RDEF chunk records the original variable names: readBoundsLower (t2),
// readBoundsUpper (t3), bounds (u2) and the AMD shader-intrinsics UAV (u7).
// This is the AMD cross-lane variant of shader 039: instead of the
// power-of-two shared-memory walk it reduces through five rounds of AMD
// atomic-memory reads, then folds the four 32-lane slices in shared memory and
// writes the (lower - kRadius, upper + kRadius) pair from thread 0.
//
// The shipped u7 accesses are imm_atomic_cmp_exch with the read value as the
// compare operand and zero as the stored value, i.e. a read that can never
// change memory; the HLSL below spells exactly that out.

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
StructuredBuffer<float4> readBoundsLower : register(t2);

StructuredBuffer<float4> readBoundsUpper : register(t3);

// u2 is a two-entry float3 buffer: [0] lower bound, [1] upper bound.
RWStructuredBuffer<float3> bounds : register(u2);

// AMD shader-intrinsics UAV: the shipped bytecode only touches it through the
// folded atomic-memory addresses below.
RWByteAddressBuffer amdIntrinsics : register(u7);

groupshared float3 sSlices[128];

static const float kFloatMax = 3.402823466e+38f;

// AMD atomic-memory addresses folded into the shipped instruction stream.
static const uint kAmdReduce0 = 0x50401f04u;
static const uint kAmdReduce1 = 0x50201f04u;
static const uint kAmdReduce2 = 0x50101f04u;
static const uint kAmdReduce3 = 0x50081f04u;
static const uint kAmdReduce4 = 0x50041f04u;

// Atomic read that never writes: the compare operand is the value itself and
// the stored value is zero, so a match would store zero over zero.
float AmdAtomicRead(uint addr, float value)
{
	uint old;
	InterlockedCompareExchange(amdIntrinsics[addr], asuint(value), 0u, old);
	return asfloat(old);
}

float3 AmdReduceRound(uint addr, float3 value)
{
	return float3(AmdAtomicRead(addr, value.x), AmdAtomicRead(addr, value.y), AmdAtomicRead(addr, value.z));
}

[numthreads(128, 1, 1)]
void CalculateBounds(uint3 tid : SV_DispatchThreadID, uint3 gid : SV_GroupThreadID)
{
	// CalculateBounds: block-wide min/max reduction by 32-lane slices.
	// The element predicate is derived from the dispatch counter with two
	// successive round-up divisions by the group size (see the unit notes).
	const int numItems = ((kNumParticles + 127) / 128 + 127) / 128;
	const bool active = (int)tid.x < numItems;

	const uint lane = gid.x & 31u;
	const uint slice = gid.x >> 5;

	float3 lo = readBoundsLower[tid.x].xyz;
	if (!active)
	{
		lo = float3(kFloatMax, kFloatMax, kFloatMax);
	}

	lo = min(lo, AmdReduceRound(kAmdReduce0, lo));
	lo = min(lo, AmdReduceRound(kAmdReduce1, lo));
	lo = min(lo, AmdReduceRound(kAmdReduce2, lo));
	lo = min(lo, AmdReduceRound(kAmdReduce3, lo));
	lo = min(lo, AmdReduceRound(kAmdReduce4, lo));

	if (lane == 0u)
	{
		sSlices[slice] = lo;
	}
	GroupMemoryBarrierWithGroupSync();

	if (gid.x == 0u)
	{
		float3 sliceMin = sSlices[0];
		sliceMin = min(sliceMin, sSlices[1]);
		sliceMin = min(sliceMin, sSlices[2]);
		sliceMin = min(sliceMin, sSlices[3]);
		sSlices[0] = sliceMin;
	}
	GroupMemoryBarrierWithGroupSync();

	const float3 totalLo = sSlices[0];
	GroupMemoryBarrierWithGroupSync();

	float3 hi = readBoundsUpper[tid.x].xyz;
	if (!active)
	{
		hi = float3(-kFloatMax, -kFloatMax, -kFloatMax);
	}

	hi = max(hi, AmdReduceRound(kAmdReduce0, hi));
	hi = max(hi, AmdReduceRound(kAmdReduce1, hi));
	hi = max(hi, AmdReduceRound(kAmdReduce2, hi));
	hi = max(hi, AmdReduceRound(kAmdReduce3, hi));
	hi = max(hi, AmdReduceRound(kAmdReduce4, hi));

	if (lane == 0u)
	{
		sSlices[slice] = hi;
	}
	GroupMemoryBarrierWithGroupSync();

	if (gid.x == 0u)
	{
		float3 sliceMax = sSlices[0];
		sliceMax = max(sliceMax, sSlices[1]);
		sliceMax = max(sliceMax, sSlices[2]);
		sliceMax = max(sliceMax, sSlices[3]);
		sSlices[0] = sliceMax;
	}
	GroupMemoryBarrierWithGroupSync();

	if (gid.x == 0u)
	{
		bounds[0] = totalLo - kRadius;
		bounds[1] = sSlices[0] + kRadius;
	}
}
