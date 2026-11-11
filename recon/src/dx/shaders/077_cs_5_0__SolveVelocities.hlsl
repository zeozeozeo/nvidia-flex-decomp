// Reconstructed from the shipped DXBC bytecode of NvFlexReleaseD3D_x64.dll,
// shader 077 (cs_5_0).  Interface (constant-buffer layout, resource
// bindings, thread group) comes from the RDEF chunk; the body is derived from
// the SHEX assembly 077_cs_5_0.asm.
//
// Evidence: work/dxbc/NvFlexReleaseD3D_x64/077.dxbc,
//           work/dxbc/NvFlexReleaseD3D_x64/077_cs_5_0.asm, work/dxbc.json

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

// Vorticity confinement force: cross(curl, accumulated vorticity), normalised
// and scaled by dt^2 / kInvRestDensity / kVorticityConfinement.
float3 vortexForce(float3 acc, float3 curlAt)
{
	float4 a = float4(acc.y, acc.z, acc.x, acc.y);
	const float lenSq = dot(a.xzw, a.xzw);
	const float scale = (kDt * kDt) * kInvRestDensity * kVorticityConfinement;
	const float3 w = (lenSq > 0.0f) ? (a.xyw * rsqrt(lenSq)) : a.xyw;
	float3 c = curlAt;
	return float3(
		w.y * c.z - w.z * c.y,
		w.z * c.x - w.x * c.z,
		w.x * c.y - w.y * c.x) * scale;
}


StructuredBuffer<int> contacts : register(t0);

StructuredBuffer<int> contactCounts : register(t1);

StructuredBuffer<float4> collisionPlanes : register(t2);

StructuredBuffer<int> collisionCounts : register(t3);

StructuredBuffer<float4> sortedPositions : register(t4);

StructuredBuffer<float4> sortedVelocities : register(t5);

StructuredBuffer<float4> sortedNewVelocities : register(t6);

StructuredBuffer<uint> phases : register(t7);

StructuredBuffer<float4> curl : register(t8);

RWStructuredBuffer<float4> sortedNewPositions : register(u0);

RWByteAddressBuffer deltas : register(u1);

RWStructuredBuffer<float> densities : register(u2);

RWStructuredBuffer<float> potentials : register(u3);

[numthreads(256, 1, 1)]
void SolveVelocities(uint3 tid : SV_DispatchThreadID)
{
	if ((int)tid.x >= kNumParticles)
	{
		return;
	}

	const float3 pos = sortedNewPositions[tid.x].xyz;
	const float3 vel = sortedNewVelocities[tid.x].xyz;
	const uint phase = phases[tid.x];
	const int numContacts = contactCounts[tid.x];

	const bool diffuse = (phase & 0x400000) != 0;
	const bool doVorticity = kVorticityConfinement > 0.0f;
	const bool doCohesion = kDiffuseThreshold > 0.0f;

	// Viscosity, vorticity and the cohesion/surface-tension accumulators.
	float3 viscosity = float3(0, 0, 0);
	float3 vorticity = float3(0, 0, 0);
	float2 accum = float2(0, 0);

	int j = tid.x;
	for (int c = 0; c < numContacts; ++c)
	{
		const int n = contacts[j];
		j += kNumParticlesAligned;

		const float3 rvec = pos - sortedNewPositions[n].xyz;
		const float dSq = dot(rvec, rvec);
		const bool near = (kRadiusSq >= dSq) && (dSq > 0.0f);
		if (!near)
		{
			continue;
		}

		const bool bothDiffuse = diffuse && ((phases[n] & 0x400000) != 0);
		const float d = sqrt(dSq);
		const float3 dir = rvec / d;
		const float3 dvel = sortedVelocities[n].xyz - vel;
		const float w = 1.0f - d * kInvRadius;
		const float w2 = w * w;

		// Viscosity: w^2 * kSpiky1 * kFreeSurfaceDrag * kDt * (vj - vi)
		const float viscCoef = w2 * kSpiky1;
		const float3 visc = (dvel * kFreeSurfaceDrag * viscCoef) * kDt + viscosity;

		// Vorticity: -(1 - d/h) * kSpiky2 * curlror(n) * dir, plus the
		// accumulated difference (r5.z holds the neighbour's curl weight).
		const float3 vort = (dir * (-w * kSpiky2)) * curl[n].x + vorticity;
		const float3 vortEnabled = doVorticity ? vort : vorticity;

		// Cohesion: (|vj - vi| - dot(vj - vi, dir)) * w^2 * kSpiky1
		const float cohTerm = (length(dvel) - dot(dvel, dir)) * viscCoef + accum.y;

		const float2 newAccum = float2(kSpiky1 * w2 + accum.x, doCohesion ? cohTerm : accum.y);

		viscosity = bothDiffuse ? visc : viscosity;
		vorticity = bothDiffuse ? vortEnabled : vorticity;
		accum = bothDiffuse ? newAccum : accum;
	}

	// Vorticity confinement force from the curl fields.
	float3 force = vortexForce(vorticity, curl[tid.x].xyz);
	float3 dv = doVorticity && (accum.x > 0.0f) ? force : viscosity;

	// Dissipation and damping, both scaled by dt.
	const float3 dvdt = vel * kDt;
	const float3 dvScaled = dvdt * accum.x;
	dv = kDissipation > 0.0f ? dv - dvScaled * kDissipation : dv;
	dv = kDamping > 0.0f ? dv - dvdt * kDamping : dv;

	// Collision plane response.
	if (kRestitution > 0.0f)
	{
		const int numPlanes = collisionCounts[tid.x];
		if (numPlanes != 0)
		{
			float4 response = float4(0, 0, 0, 0);
			for (int p = 0; p < numPlanes; ++p)
			{
				const float4 plane = collisionPlanes[tid.x * kMaxContactsPerParticle + p];
				const float planeD = dot(pos, plane.xyz) + plane.w;
				const float vn = dot(vel, plane.xyz);
				if (planeD - kCollisionDistance * 1.001f <= 0.0f && vn < 0.0f)
				{
					const float vnNew = vn * kRestitution + dot(dv, plane.xyz);
					response = float4(response.xyz - plane.xyz * vnNew, response.w + 1.0f);
				}
			}
			dv += response.xyz / max(response.w, 1.0f);
		}
	}

	const float3 newVel = vel + dv;

	// Sleep threshold: particles below the threshold are put back.
	if (dot(newVel, newVel) < kSleepThresholdSq)
	{
		sortedNewPositions[tid.x] = float4(sortedPositions[tid.x].xyz, sortedNewPositions[tid.x].w);
		dv = float3(-vel);
	}

	deltas.Store4(tid.x * 16, asuint(float4(dv, 0.0f)));
	densities[tid.x] = accum.x;
}
