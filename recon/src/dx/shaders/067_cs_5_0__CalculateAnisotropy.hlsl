// Reconstructed from the shipped DXBC bytecode of NvFlexReleaseD3D_x64.dll,
// shader 067 (cs_5_0).  Interface (constant-buffer layout, resource
// bindings, thread group) comes from the RDEF chunk; the body is derived from
// the SHEX assembly 067_cs_5_0.asm.
//
// Evidence: work/dxbc/NvFlexReleaseD3D_x64/067.dxbc,
//           work/dxbc/NvFlexReleaseD3D_x64/067_cs_5_0.asm, work/dxbc.json

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

StructuredBuffer<int> neighbors : register(t0);

StructuredBuffer<int> neighborCounts : register(t1);

Buffer<uint> indices : register(t2);

StructuredBuffer<uint> particleFlags : register(t3);

StructuredBuffer<float4> positions : register(t4);

RWStructuredBuffer<float4> anisotropy0 : register(u0);

RWStructuredBuffer<float4> anisotropy1 : register(u1);

RWStructuredBuffer<float4> anisotropy2 : register(u2);

[numthreads(256, 1, 1)]
void CalculateAnisotropy(uint3 tid : SV_DispatchThreadID)
{
	if ((int)tid.x >= kNumParticles)
	{
		return;
	}

	const float3 pos = positions[tid.x].xyz;
	const int count = neighborCounts[tid.x];

	// Pass 1: weighted centroid of the non-diffuse neighbours inside the radius.
	float4 sum = float4(0, 0, 0, 0);
	int j = tid.x;
	for (int c = 0; c < count; ++c)
	{
		const int n = neighbors[j];
		j += kNumParticlesAligned;

		const bool diffuse = (particleFlags[n] & 0x400000) != 0;
		const float3 r = pos - positions[n].xyz;
		const float dSq = dot(r, r);
		const float d = sqrt(dSq);
		const float t = d * kInvRadius;
		const float w = 1.0f - t * t * t;

		if (dSq > 0.0f && dSq < kRadiusSq)
		{
			const float4 candidate = float4(sum.xyz + positions[n].xyz * w, sum.w + w);
			sum = diffuse ? candidate : sum;
		}
	}

	const uint index = indices[tid.x];
	if (sum.w == 0.0f || count < 0)
	{
		// Degenerate neighbourhood: identity frame.
		anisotropy0[index] = float4(1, 0, 0, kAnisotropyMin);
		anisotropy1[index] = float4(0, 1, 0, kAnisotropyMin);
		anisotropy2[index] = float4(0, 0, 1, kAnisotropyMin);
		return;
	}

	const float invW = 1.0f / sum.w;
	const float3 center = sum.xyz * invW;

	// Pass 2: weighted covariance of the same neighbourhood around the centroid.
	float cxx = 0, cxy = 0, cxz = 0, cyy = 0, cyz = 0, czz = 0;
	j = tid.x;
	for (int c2 = 0; c2 < count; ++c2)
	{
		const int n = neighbors[j];
		j += kNumParticlesAligned;

		if ((particleFlags[n] & 0x400000) != 0)
		{
			const float3 r = positions[n].xyz - center;
			const float dSq = dot(r, r);
			if (dSq > 0.0f && dSq < kRadiusSq)
			{
				const float d = sqrt(dSq);
				const float t = d * kInvRadius;
				const float w = 1.0f - t * t * t;
				cxx += w * r.x * r.x;
				cxy += w * r.x * r.y;
				cxz += w * r.x * r.z;
				cyy += w * r.y * r.y;
				cyz += w * r.y * r.z;
				czz += w * r.z * r.z;
			}
		}
	}
	cxx *= invW; cxy *= invW; cxz *= invW;
	cyy *= invW; cyz *= invW; czz *= invW;

	// Symmetric 3x3 eigen decomposition by Jacobi rotations (4 sweeps, as the
	// bytecode's fixed trip count).
	float3 v0 = float3(1, 0, 0);
	float3 v1 = float3(0, 1, 0);
	float3 v2 = float3(0, 0, 1);
	float a00 = cxx, a01 = cxy, a02 = cxz, a11 = cyy, a12 = cyz, a22 = czz;

	for (int sweep = 0; sweep < 4; ++sweep)
	{
		// largest off-diagonal element
		float m01 = abs(a01), m02 = abs(a02), m12 = abs(a12);
		int p = 0, q = 1;
		float m = m01;
		if (m02 > m) { p = 0; q = 2; m = m02; }
		if (m12 > m) { p = 1; q = 2; m = m12; }
		if (m <= 0.0f)
		{
			continue;
		}

		float app, aqq, apq;
		if (p == 0 && q == 1) { app = a00; aqq = a11; apq = a01; }
		else if (p == 0) { app = a00; aqq = a22; apq = a02; }
		else { app = a11; aqq = a22; apq = a12; }

		const float theta = (aqq - app) / (2.0f * apq);
		const float tSign = theta < 0.0f ? -1.0f : 1.0f;
		const float tt = tSign / (abs(theta) + sqrt(theta * theta + 1.0f));
		const float cs = rsqrt(tt * tt + 1.0f);
		const float sn = cs * tt;

		// rotate rows/columns p and q
		float3 rp = p == 0 ? float3(a00, a01, a02) : float3(a01, a11, a12);
		float3 rq = q == 1 ? float3(a01, a11, a12) : float3(a02, a12, a22);
		float3 np = cs * rp - sn * rq;
		float3 nq = sn * rp + cs * rq;

		if (p == 0) { a00 = np.x; a01 = np.y; a02 = np.z; }
		else { a01 = np.x; a11 = np.y; a12 = np.z; }
		if (q == 1) { a01 = nq.x; a11 = nq.y; a12 = nq.z; }
		else { a02 = nq.x; a12 = nq.y; a22 = nq.z; }

		// accumulate the rotation into the eigenvector columns
		float3 vp = p == 0 ? v0 : v1;
		float3 vq = q == 1 ? v1 : v2;
		float3 w0 = cs * vp - sn * vq;
		float3 w1 = sn * vp + cs * vq;
		if (p == 0) { v0 = w0; } else { v1 = w0; }
		if (q == 1) { v1 = w1; } else { v2 = w1; }
	}

	const float3 scale = clamp(sqrt(abs(float3(a00, a11, a22))) * kAnisotropy, kAnisotropyMin, kAnisotropyMax);

	anisotropy0[index] = float4(v0, scale.x);
	anisotropy1[index] = float4(v1, scale.y);
	anisotropy2[index] = float4(v2, scale.z);
}
