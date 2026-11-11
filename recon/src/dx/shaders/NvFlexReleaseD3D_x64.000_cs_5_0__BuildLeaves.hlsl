// Reconstructed from the shipped DXBC bytecode of NvFlexReleaseD3D_x64.dll,
// shader 000 (cs_5_0).  Interface (constant-buffer layout, resource bindings,
// thread group) comes from the RDEF chunk; the body is derived from the SHEX
// assembly 000_cs_5_0.asm.
//
// Evidence: work/dxbc/NvFlexReleaseD3D_x64/000.dxbc,
//           work/dxbc/NvFlexReleaseD3D_x64/000_cs_5_0.asm, work/dxbc.json

struct lowersType
{
    float x;
    float y;
    float z;
    uint ib;
};

struct uppersType
{
    float x;
    float y;
    float z;
    uint ib;
};

cbuffer constBVH : register(b0)
{
    int gNumItems;
    float3 _fdpad0_0;
};

StructuredBuffer<float4> itemLowers : register(t0);

StructuredBuffer<float4> itemUppers : register(t1);

Buffer<uint> indices : register(t2);

RWStructuredBuffer<int> rangeLefts : register(u0);

RWStructuredBuffer<int> rangeRights : register(u1);

RWStructuredBuffer<lowersType> lowers : register(u2);

RWStructuredBuffer<uppersType> uppers : register(u3);

[numthreads(256, 1, 1)]
void BuildLeaves(uint3 tid : SV_DispatchThreadID)
{
	// BuildLeaves: pack every (sorted) item bound into a BVH leaf node.
	if ((int)tid.x < gNumItems)
	{
		const uint index = indices[tid.x];

		const float4 lo = itemLowers[index];
		const float4 up = itemUppers[index];

		lowersType lower;
		lower.x = lo.x;
		lower.y = lo.y;
		lower.z = lo.z;
		lower.ib = index | 0x80000000u;	// BVH leaf flag
		lowers[tid.x] = lower;

		uppersType upper;
		upper.x = up.x;
		upper.y = up.y;
		upper.z = up.z;
		upper.ib = index & 0x7fffffffu;
		uppers[tid.x] = upper;

		rangeLefts[tid.x] = (int)tid.x;
		rangeRights[tid.x] = (int)tid.x;
	}
}
