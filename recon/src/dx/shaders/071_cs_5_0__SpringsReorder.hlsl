// Reconstructed from the shipped DXBC bytecode of NvFlexReleaseD3D_x64.dll,
// shader 071 (cs_5_0).  Interface (constant-buffer layout, resource
// bindings, thread group) comes from the RDEF chunk; the body is derived from
// the SHEX assembly 071_cs_5_0.asm.
//
// Evidence: work/dxbc/NvFlexReleaseD3D_x64/071.dxbc,
//           work/dxbc/NvFlexReleaseD3D_x64/071_cs_5_0.asm, work/dxbc.json

cbuffer consts : register(b0)
{
    int gNumSprings;
};

Buffer<uint> sortedIndices : register(t0);

StructuredBuffer<int> indices : register(t1);

StructuredBuffer<float> lengths : register(t2);

StructuredBuffer<float> stiffness : register(t3);

RWStructuredBuffer<int> outIndices : register(u0);

RWStructuredBuffer<float> outLengths : register(u1);

RWStructuredBuffer<float> outStiffness : register(u2);

[numthreads(256, 1, 1)]
void SpringsReorder(uint3 tid : SV_DispatchThreadID)
{
	// Each spring contributes two entries (start and end particle).
	if (tid.x < (uint)(gNumSprings * 2))
	{
		const uint key = sortedIndices[tid.x];
		const uint spring = key >> 1;
		const uint parity = key & 1;
		// Odd keys take the previous particle, even keys the next one.
		const uint other = parity ? (key - 1) : (key + 1);

		outIndices[tid.x] = indices[other];
		outLengths[tid.x] = lengths[spring];
		outStiffness[tid.x] = stiffness[spring];
	}
}
