// flexdecomp — nvflex.bvh device kernels (src/cuda/bvh.cu:418-660)
//
// Reconstructed from the sm_30 PTX of cudabvh.o (work/binary/gpu/dbg/nvflex.bvh/
// sm_30.ptx) and the DWARF of the debug object.  The host side of this module
// (LinearBVHBuilderGPU::Build and the CPU builder) lives in bvh.cu; this file
// owns only the seven kernel units of the module.
//
// Kernel parameters are spelled with the FleX class templates XVector3<float> /
// XVector4<float> (recovered.h) because that spelling is what produces the
// shipped mangled names (`PK8XVector4IfE`, `PK8XVector3IfE`).

#include "nvflex/all.h"

#include <cub/block/block_reduce.cuh>

// nvcc (>= 11) marks every __global__ launch wrapper and its __device_stub__
// hidden; the shipped objects (CUDA 9.2) exported them as GLOBAL DEFAULT and
// they are part of the library ABI (`_Z20CalculateMortonCodes...` etc. are
// `T` in NvFlexDebugCUDA_x64.a and in the relinked library).  Restoring default
// visibility keeps the exported symbol set identical; it does not affect the
// mangled name, the PTX entry or intra-DSO references.
#if defined(_MSC_VER)
// PE ABI: kernels need no export; win32/nvflex.def pins the 81 API exports.
#define FLEX_KERNEL_API
#else
// The shipped 9.2 objects export the launch wrappers GLOBAL DEFAULT; modern
// nvcc emits them hidden, so every kernel carries an explicit visibility.
#define FLEX_KERNEL_API __attribute__((visibility("default")))
#endif

// ---------------------------------------------------------------------------
// flexdecomp: src/cuda/bvh.cu:69 Morton3<1024>
// ---------------------------------------------------------------------------
// Interleaves the low 10 bits of three cell coordinates into a 30-bit Morton
// key.  PTX evidence (CalculateMortonCodes, sm_30.ptx:83-140): the key is
// spread(x) | spread(y)<<1 | spread(z)<<2 with the Part1by2 mask chain
// 0xff0000ff / 0x0300f00f / 0x030c30c3 / 0x09249249.
template <int N>
__device__ __forceinline__ unsigned int Morton3(unsigned int x, unsigned int y, unsigned int z);

// Part1by2 for a 10-bit input (one axis of Morton3<1024>).
__device__ __forceinline__ unsigned int ExpandBits(unsigned int v)
{
	v &= 0x000003ffu;	// sm_30.ptx: no explicit mask survives (inputs are <= 511)

	v = (v ^ (v << 16)) & 0xff0000ffu;
	v = (v ^ (v << 8)) & 0x0300f00fu;
	v = (v ^ (v << 4)) & 0x030c30c3u;
	v = (v ^ (v << 2)) & 0x09249249u;
	return v;
}

template <int N>
__device__ __forceinline__ unsigned int Morton3(unsigned int x, unsigned int y, unsigned int z)
{
	return ExpandBits(x) | (ExpandBits(y) << 1) | (ExpandBits(z) << 2);
}

// ---------------------------------------------------------------------------
// float atomics used by ComputeTotalBounds
// (sm_30.ptx:1442-1530: compare-and-swap loops, guarded by the "unordered or
// already better" test, storing `val` itself on success)
// ---------------------------------------------------------------------------
__device__ __forceinline__ void AtomicMax(float* address, float val)
{
	if (*address < val)
	{
		int old = *reinterpret_cast<int*>(address);
		int assumed;

		do
		{
			assumed = old;
			old = atomicCAS(reinterpret_cast<int*>(address), assumed, __float_as_int(val));
		}
		while (__int_as_float(old) < val);
	}
}

__device__ __forceinline__ void AtomicMin(float* address, float val)
{
	if (*address > val)
	{
		int old = *reinterpret_cast<int*>(address);
		int assumed;

		do
		{
			assumed = old;
			old = atomicCAS(reinterpret_cast<int*>(address), assumed, __float_as_int(val));
		}
		while (__int_as_float(old) > val);
	}
}

// ---------------------------------------------------------------------------
// flexdecomp: src/cuda/bvh.cu:418 _Z20CalculateMortonCodesPK8XVector4IfES2_PKiiPK8XVector3IfES8_PiS9_
// ---------------------------------------------------------------------------
// For each item: key = Morton3 of the item centre in normalised grid space,
// with the (inverted) priority packed into the top five bits.  Writes the item
// index to `indices` (radix-sort payload) and the key to `keys`.
//
// PTX: sm_30.ptx:17-163.  centre = (lower+upper)*0.5f, delta = centre-origin,
// cell = delta*invEdges*512.0f, clamped with cvt.rzi + min(.,511) + max(0,.).
__global__ void FLEX_KERNEL_API CalculateMortonCodes(const XVector4<float>* itemLowers,
	const XVector4<float>* itemUppers,
	const int* itemPriorities,
	int numItems,
	const XVector3<float>* origin,
	const XVector3<float>* invEdges,
	int* indices,
	int* keys)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= numItems)
		return;

	const XVector4<float>& lower = itemLowers[i];
	const XVector4<float>& upper = itemUppers[i];

	// 0f3F000000 (0.5f), 0f44000000 (512.0f), cells clamped to [0, 511]
	int x = max(0, min((int)(((lower.x + upper.x) * 0.5f - origin->x) * invEdges->x * 512.0f), 511));
	int y = max(0, min((int)(((lower.y + upper.y) * 0.5f - origin->y) * invEdges->y * 512.0f), 511));
	int z = max(0, min((int)(((lower.z + upper.z) * 0.5f - origin->z) * invEdges->z * 512.0f), 511));

	unsigned int key = Morton3<1024>((unsigned int)x, (unsigned int)y, (unsigned int)z);

	if (itemPriorities)
		key |= (unsigned int)(~itemPriorities[i]) << 27;

	indices[i] = i;
	keys[i] = (int)key;
}

// ---------------------------------------------------------------------------
// flexdecomp: src/cuda/bvh.cu:445 _Z18CalculateKeyDeltasPKiPii
// ---------------------------------------------------------------------------
// Deltas between adjacent (sorted) Morton keys; the launch count is
// numItems-1 so keys[i+1] is always in range.  PTX: sm_30.ptx:165-199.
__global__ void FLEX_KERNEL_API CalculateKeyDeltas(const int* keys, int* deltas, int numItems)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= numItems)
		return;

	deltas[i] = keys[i + 1] ^ keys[i];
}

// ---------------------------------------------------------------------------
// flexdecomp: src/cuda/bvh.cu:460 _Z11BuildLeavesPK8XVector4IfES2_iPKiPiS5_P14PackedNodeHalfS7_
// ---------------------------------------------------------------------------
// Emits leaf node i from item `index = indices[i]`: the lower half carries the
// item's lower bound and `i = index`, `b = 1`; the upper half the upper bound
// and `i = index`, `b = 0` (PackedNodeHalf bitfield, DWARF).  Range leaf/right
// arrays start out as the single-item range [i, i].
// PTX: sm_30.ptx:201-283.
__global__ void FLEX_KERNEL_API BuildLeaves(const XVector4<float>* itemLowers,
	const XVector4<float>* itemUppers,
	int numItems,
	const int* indices,
	int* rangeLefts,
	int* rangeRights,
	PackedNodeHalf* nodeLowers,
	PackedNodeHalf* nodeUppers)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= numItems)
		return;

	int index = indices[i];

	nodeLowers[i].x = itemLowers[index].x;
	nodeLowers[i].y = itemLowers[index].y;
	nodeLowers[i].z = itemLowers[index].z;
	nodeLowers[i].i = (unsigned int)index;
	nodeLowers[i].b = 1;

	nodeUppers[i].x = itemUppers[index].x;
	nodeUppers[i].y = itemUppers[index].y;
	nodeUppers[i].z = itemUppers[index].z;
	nodeUppers[i].i = (unsigned int)index;
	nodeUppers[i].b = 0;

	rangeLefts[i] = i;
	rangeRights[i] = i;
}

// ---------------------------------------------------------------------------
// flexdecomp: src/cuda/bvh.cu:485 _Z14BuildHierarchyiPiPKiS_PViS3_PV14PackedNodeHalfS6_
// ---------------------------------------------------------------------------
// Builds the internal nodes bottom-up from the leaf ranges.  A range that
// reaches the left end of the key array extends to the right and vice versa;
// otherwise the smaller adjacent key delta wins (the "longest common prefix"
// criterion).  The second of the two children to arrive (atomicAdd on
// numChildren returns 1) merges both children's bounds into the parent and
// climbs.  The thread whose range covers all items writes the root index.
// PTX: sm_30.ptx:284-471.
__global__ void FLEX_KERNEL_API BuildHierarchy(int numItems,
	int* rootNode,
	const int* deltas,
	int* numChildren,
	volatile int* rangeLefts,
	volatile int* rangeRights,
	volatile PackedNodeHalf* nodeLowers,
	volatile PackedNodeHalf* nodeUppers)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= numItems)
		return;

	int index = i;
	int left = rangeLefts[index];
	int right = rangeRights[index];

	if (left == 0 && right == numItems - 1)
	{
		*rootNode = index;
		return;
	}

	while (true)
	{
		int nodeIndex;
		int count;

		// PTX: (left == 0) -> extend right, else (right == numItems-1) -> extend
		// left, else the smaller key delta picks the side.
		if (left == 0 || (right != numItems - 1 && deltas[right] < deltas[left - 1]))
		{
			nodeIndex = right + numItems;
			nodeLowers[nodeIndex].i = (unsigned int)index;
			rangeLefts[nodeIndex] = right;
			__threadfence_system();
			count = atomicAdd(&numChildren[nodeIndex], 1);
		}
		else
		{
			nodeIndex = (numItems - 1) + left;
			nodeUppers[nodeIndex].i = (unsigned int)index;
			rangeRights[nodeIndex] = left;
			__threadfence_system();
			count = atomicAdd(&numChildren[nodeIndex], 1);
		}

		// first child to arrive only registers itself
		if (count != 1)
			return;

		int leftChild = (int)nodeLowers[nodeIndex].i;
		int rightChild = (int)nodeUppers[nodeIndex].i;

		float lowerX = nodeLowers[leftChild].x;
		float lowerY = nodeLowers[leftChild].y;
		float lowerZ = nodeLowers[leftChild].z;
		nodeLowers[nodeIndex].x = lowerX < nodeLowers[rightChild].x ? lowerX : nodeLowers[rightChild].x;
		nodeLowers[nodeIndex].y = lowerY < nodeLowers[rightChild].y ? lowerY : nodeLowers[rightChild].y;
		nodeLowers[nodeIndex].z = lowerZ < nodeLowers[rightChild].z ? lowerZ : nodeLowers[rightChild].z;
		nodeLowers[nodeIndex].i = (unsigned int)leftChild;
		nodeLowers[nodeIndex].b = 0;

		float upperX = nodeUppers[leftChild].x;
		float upperY = nodeUppers[leftChild].y;
		float upperZ = nodeUppers[leftChild].z;
		nodeUppers[nodeIndex].x = upperX > nodeUppers[rightChild].x ? upperX : nodeUppers[rightChild].x;
		nodeUppers[nodeIndex].y = upperY > nodeUppers[rightChild].y ? upperY : nodeUppers[rightChild].y;
		nodeUppers[nodeIndex].z = upperZ > nodeUppers[rightChild].z ? upperZ : nodeUppers[rightChild].z;
		nodeUppers[nodeIndex].i = (unsigned int)rightChild;
		nodeUppers[nodeIndex].b = 0;

		index = nodeIndex;
		left = rangeLefts[index];
		right = rangeRights[index];

		if (left == 0 && right == numItems - 1)
		{
			*rootNode = index;
			return;
		}
	}
}

// ---------------------------------------------------------------------------
// flexdecomp: src/cuda/bvh.cu:581 _Z18ComputeTotalBoundsPK8XVector4IfES2_P8XVector3IfES5_i
// ---------------------------------------------------------------------------
// Reduces the item bounds into the global grid bounds: block-wide max of the
// uppers, block-wide min of the lowers, combined with float atomics by thread
// 0 of each block.  The shipped object inlines cub::BlockReduce<Vec3, 256>
// (external/cub-1.3.2) twice; the toolkit CUB provides the same collective and
// is used instead (the vendored 1.3.2 headers do not compile with CUDA 13).
// PTX: sm_30.ptx:472-1593 (765 instructions, both the full-tile and the
// num_valid-guarded paths).
// ---------------------------------------------------------------------------
struct Vec3Max
{
	__device__ __forceinline__ XVector3<float> operator()(const XVector3<float>& a, const XVector3<float>& b) const
	{
		XVector3<float> r;
		r.x = a.x > b.x ? a.x : b.x;
		r.y = a.y > b.y ? a.y : b.y;
		r.z = a.z > b.z ? a.z : b.z;
		return r;
	}
};

struct Vec3Min
{
	__device__ __forceinline__ XVector3<float> operator()(const XVector3<float>& a, const XVector3<float>& b) const
	{
		XVector3<float> r;
		r.x = a.x < b.x ? a.x : b.x;
		r.y = a.y < b.y ? a.y : b.y;
		r.z = a.z < b.z ? a.z : b.z;
		return r;
	}
};

__global__ void FLEX_KERNEL_API ComputeTotalBounds(const XVector4<float>* itemLowers,
	const XVector4<float>* itemUppers,
	XVector3<float>* totalLower,
	XVector3<float>* totalUpper,
	int numItems)
{
	__shared__ typename cub::BlockReduce<XVector3<float>, 256>::TempStorage temp_storage;

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= numItems)
		return;

	// valid items in this block (PTX: min(numItems - blockIdx*blockDim, blockDim))
	int numValid = min(numItems - blockIdx.x * blockDim.x, (int)blockDim.x);

	XVector3<float> lower;
	lower.x = itemLowers[i].x;
	lower.y = itemLowers[i].y;
	lower.z = itemLowers[i].z;

	XVector3<float> upper;
	upper.x = itemUppers[i].x;
	upper.y = itemUppers[i].y;
	upper.z = itemUppers[i].z;

	XVector3<float> maxUpper = cub::BlockReduce<XVector3<float>, 256>(temp_storage).Reduce(upper, Vec3Max(), numValid);
	XVector3<float> minLower = cub::BlockReduce<XVector3<float>, 256>(temp_storage).Reduce(lower, Vec3Min(), numValid);

	if (threadIdx.x == 0)
	{
		AtomicMax(&totalUpper->x, maxUpper.x);
		AtomicMax(&totalUpper->y, maxUpper.y);
		AtomicMax(&totalUpper->z, maxUpper.z);

		AtomicMin(&totalLower->x, minLower.x);
		AtomicMin(&totalLower->y, minLower.y);
		AtomicMin(&totalLower->z, minLower.z);
	}
}

// ---------------------------------------------------------------------------
// flexdecomp: src/cuda/bvh.cu:614 _Z20ComputeTotalInvEdgesPK8XVector3IfES2_PS0_
// ---------------------------------------------------------------------------
// Reciprocal of the grid extent per axis, with the 1e-4f guard that keeps the
// value finite for a degenerate (single-item) grid.
// PTX: sm_30.ptx:1594-1631 (rcp.rn.f32 of (upper-lower)+1e-4f).
__global__ void FLEX_KERNEL_API ComputeTotalInvEdges(const XVector3<float>* totalLower,
	const XVector3<float>* totalUpper,
	XVector3<float>* totalInvEdges)
{
	totalInvEdges->x = 1.0f / ((totalUpper->x - totalLower->x) + 1e-4f);
	totalInvEdges->y = 1.0f / ((totalUpper->y - totalLower->y) + 1e-4f);
	totalInvEdges->z = 1.0f / ((totalUpper->z - totalLower->z) + 1e-4f);
}

// ---------------------------------------------------------------------------
// flexdecomp: (third party) external/cub-1.3.2/cub/util_device.cuh:60
// _ZN3cub11EmptyKernelIvEEvv
// ---------------------------------------------------------------------------
// cub::EmptyKernel<void> is instantiated by the shipped object (fatbin entry,
// sm_30.ptx:1632-1642) through CUB's device-properties probe.  The toolkit's
// CUB only provides cub::detail::EmptyKernel, so the 1.3.2 instantiation is
// reproduced verbatim here to keep the exact symbol and PTX entry name.
namespace cub
{
	template <typename T>
	__global__ void FLEX_KERNEL_API EmptyKernel(void) { }

	template __global__ void EmptyKernel<void>(void);

	// CUB 1.3.2 odr-uses the instantiation as a function pointer in the
	// device-properties probe (`EmptyKernelPtr Empty() { return
	// EmptyKernel<T>; }`, util_device.cuh:122-130), which is what emits the
	// exported host launch wrapper `w _ZN3cub11EmptyKernelIvEEvv`.  Reproduce
	// that use so the shipped weak symbol stays part of the ABI.
	typedef void (*EmptyKernelPtr)();

#if !defined(_MSC_VER)
	__attribute__((used)) static EmptyKernelPtr kEmptyKernel = EmptyKernel<void>;
#else
	// PE build: kernels are not exported (win32/nvflex.def pins the ABI); the
	// odr-use of the instantiation is what matters, not the GCC 'used' attr.
	static EmptyKernelPtr kEmptyKernel = EmptyKernel<void>;
#endif
}
