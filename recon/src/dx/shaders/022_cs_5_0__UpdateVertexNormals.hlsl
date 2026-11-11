// Reconstructed from the shipped DXBC bytecode of NvFlexReleaseD3D_x64.dll,
// shader 022 (cs_5_0).  Interface (constant-buffer layout, resource bindings,
// thread group) comes from the RDEF chunk; the body is derived from the SHEX
// assembly 022_cs_5_0.asm.
//
// Evidence: work/dxbc/NvFlexReleaseD3D_x64/022.dxbc,
//           work/dxbc/NvFlexReleaseD3D_x64/022_cs_5_0.asm, work/dxbc.json
//
// The RDEF chunk records the original variable names: this is
// UpdateVertexNormals with t1 = indices, t2 = triNormals and the raw UAV
// u0 = InterlockedFloat::accum.  Atomically accumulating a float into a raw
// UAV is written with the SM5 InterlockedCompareExchange machine on a raw
// buffer element: `InterlockedCompareExchange(accum[addr], ...)` compiles to
// imm_atomic_cmp_exch against dcl_uav_raw, which is exactly what the shipped
// bytecode contains (nine inline CAS loops, addresses idx*16 + {0,4,8}).

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
};

StructuredBuffer<int> indices : register(t1);

StructuredBuffer<float3> triNormals : register(t2);

// float4 accumulator over vertices; only .xyz is touched.
RWByteAddressBuffer accum : register(u0);

// Float atomic add through an integer compare-exchange loop: the shipped code
// is nine inlined copies of exactly this loop.
void AtomicAddFloat(uint addr, float value)
{
	uint expected = 0u;
	for (;;)
	{
		uint old;
		InterlockedCompareExchange(accum[addr], expected, asuint(asfloat(expected) + value), old);
		if (old == expected)
		{
			return;
		}
		expected = old;
	}
}

[numthreads(256, 1, 1)]
void UpdateVertexNormals(uint3 tid : SV_DispatchThreadID)
{
	// UpdateVertexNormals: scatter this triangle's normal onto its three
	// vertices' running normal sums (one thread per triangle).
	if ((int)tid.x < kNumTriangles)
	{
		const int i0 = indices[tid.x * 3 + 0];
		const int i1 = indices[tid.x * 3 + 1];
		const int i2 = indices[tid.x * 3 + 2];

		const float3 n = triNormals[tid.x];

		AtomicAddFloat((uint)i0 * 16u + 0u, n.x);
		AtomicAddFloat((uint)i0 * 16u + 4u, n.y);
		AtomicAddFloat((uint)i0 * 16u + 8u, n.z);

		AtomicAddFloat((uint)i1 * 16u + 0u, n.x);
		AtomicAddFloat((uint)i1 * 16u + 4u, n.y);
		AtomicAddFloat((uint)i1 * 16u + 8u, n.z);

		AtomicAddFloat((uint)i2 * 16u + 0u, n.x);
		AtomicAddFloat((uint)i2 * 16u + 4u, n.y);
		AtomicAddFloat((uint)i2 * 16u + 8u, n.z);
	}
}
