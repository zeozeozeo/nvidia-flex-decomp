// flexdecomp: reconstructed D3D backend — flex.obj object model.
//
// Evidence: work/decomp/nvflexd3d/NvFlexReleaseD3D_x64.dll.c; every function
// names its `// ---- <Name> @ <rva>` block and byte size.
//
// The three stores are slot pools: a `std::vector` of textures / host records
// plus a `DxComputeBuffer` of GPU-side records and an int free list.  Allocation
// pops the last free index, Free releases the owned object at that index and
// pushes the index back.  That is exactly the shape the shipped code emits.
#include "FlexObjects.h"

#include <cmath>
#include <cstdlib>
#include <cstring>

namespace NvFlex
{
namespace flexobj
{

namespace
{

// Every store's GPU record buffer is created with eBuffer | eUAV_SRV | eStage |
// eStructured (0x17): the CPU reads the handles back through a staging copy.
void CreateReadbackBuffer(DxComputeBuffer& buffer, NvFlex::ContextEx* context,
                          unsigned dim, unsigned stride)
{
    unsigned* zeros = (unsigned*)std::malloc((size_t)dim * stride);
    if (zeros != 0)
        std::memset(zeros, 0, (size_t)dim * stride);

    BufferDesc desc;
    desc.format = NvFlexFormat(0);
    desc.dim = dim;
    desc.stride = stride;
    desc.bufferType = eBuffer | eUAV_SRV | eStage | eStructured;   // 0x17
    desc.data = zeros;

    buffer.Release();
    buffer.mBufferDesc = desc;
    buffer.mBuffer = context->createBuffer(&desc, 0);

    std::free(zeros);
}

// Same as above but for a plain eBuffer | eUAV_SRV | eStructured (0x13) buffer,
// which is what the flattening arrays use.
void CreateStructuredBuffer(DxBufferExt& buffer, NvFlex::ContextEx* context,
                            unsigned dim, unsigned stride)
{
    buffer.Release();

    BufferDesc desc;
    desc.format = NvFlexFormat(0);
    desc.dim = dim;
    desc.stride = stride;
    desc.bufferType = eBuffer | eUAV_SRV | eStructured;   // 0x13
    desc.data = 0;

    buffer.mBufferDesc = desc;
    buffer.mBuffer = context->createBuffer(&desc, 0);
}

// The free lists are plain int arrays of maxSlots entries.
int* CreateFreeList(unsigned slots)
{
    int* freeList = (int*)std::malloc((size_t)slots * sizeof(int));
    for (unsigned i = 0; i < slots; ++i)
        freeList[i] = (int)i;
    return freeList;
}

} // namespace

// ---------------------------------------------------------------------------
// FlexTriangleMeshHost — @0x2f4f0 (82 bytes)
// ---------------------------------------------------------------------------
void TriangleMeshHostInit(FlexTriangleMeshHost* host)
{
    std::memset(host, 0, sizeof(*host));
}

// ---------------------------------------------------------------------------
// FlexTriangleMeshData
// ---------------------------------------------------------------------------
// @0x2fdb0 (312 bytes): a host record is appended (or a freed slot reused) and
// its handle returned.
int TriangleMeshAllocate(FlexTriangleMeshData* data, NvFlex::ContextEx* context)
{
    (void)context;

    FlexTriangleMeshHost* host = new FlexTriangleMeshHost();
    TriangleMeshHostInit(host);

    data->mTriMeshTableHost.push_back(host);
    return (int)data->mTriMeshTableHost.size() - 1;
}

// @0x2fef0 (84 bytes)
void TriangleMeshFree(FlexTriangleMeshData* data, int index)
{
    if (index < 0 || (size_t)index >= data->mTriMeshTableHost.size())
        return;

    FlexTriangleMeshHost* host = data->mTriMeshTableHost[index];
    delete host;
    data->mTriMeshTableHost.erase(data->mTriMeshTableHost.begin() + index);
}

// @0x2f630 (403 bytes)
void TriangleMeshRelease(FlexTriangleMeshData* data)
{
    data->mVerticesArray.Release();
    data->mIndicesArray.Release();
    data->mTriMeshTableDevice.Release();
    data->mBvhData.mNodeLowersArray.Release();
    data->mBvhData.mNodeUppersArray.Release();
    data->mBvhData.mRootNodeArray.Release();
}

// @0x2f7d0 (1490 bytes)
//
// Each array grows to 1.5x the requirement so a slowly growing mesh set does not
// reallocate every frame; the node and mesh arrays are sized from the triangle
// count because a BVH over N triangles has at most 2N nodes.
void TriangleMeshResize(FlexTriangleMeshData* data, NvFlex::ContextEx* context,
                        int numVertices, int numTris, int numMeshes)
{
    if (data->mMaxNumVertices < numVertices)
    {
        const unsigned capacity = (unsigned)((numVertices * 3) / 2);
        CreateStructuredBuffer(data->mVerticesArray, context, capacity, 0x10);
        data->mMaxNumVertices = (int)capacity;
    }

    if (data->mMaxNumTriangles < numTris)
    {
        const int triCapacity = (numTris * 3) / 2;
        const unsigned nodeCapacity = (unsigned)(triCapacity * 2);

        CreateStructuredBuffer(data->mIndicesArray, context, (unsigned)(triCapacity * 3), 4);
        CreateStructuredBuffer(data->mBvhData.mNodeLowersArray, context, nodeCapacity, 0x10);
        CreateStructuredBuffer(data->mBvhData.mNodeUppersArray, context, nodeCapacity, 0x10);
        CreateStructuredBuffer(data->mBvhData.mRootNodeArray, context, nodeCapacity, 4);
        CreateStructuredBuffer(data->mTriMeshTableDevice, context, nodeCapacity,
                               sizeof(FlexTriangleMeshDevice));

        data->mMaxNumTriangles = triCapacity;
    }

    if (numMeshes > data->mMaxNumMeshes)
        data->mMaxNumMeshes = numMeshes;
}

// ---------------------------------------------------------------------------
// FlexSDFData — @0x2ff90 (297 bytes) / @0x300c0 / @0x30180 / @0x301a0
// ---------------------------------------------------------------------------
void SDFCreate(FlexSDFData* data, NvFlex::ContextEx* context, bool shflSupported)
{
    (void)shflSupported;

    data->mNumSDFs = 0;
    data->mMaxSDFs = 0x10;
    data->mTextures.resize(0x10);

    CreateReadbackBuffer(data->mSDFs, context, (unsigned)data->mMaxSDFs, 0x20);
    data->mFreeList = CreateFreeList((unsigned)data->mMaxSDFs);
}

void SDFRelease(FlexSDFData* data)
{
    for (size_t i = 0; i < data->mTextures.size(); ++i)
    {
        if (data->mTextures[i] != 0)
            data->mTextures[i]->release();
        data->mTextures[i] = 0;
    }

    data->mSDFs.Release();
    std::free(data->mFreeList);
    data->mFreeList = 0;
    data->mNumSDFs = 0;
}

int SDFAllocate(FlexSDFData* data)
{
    if (data->mNumSDFs >= data->mMaxSDFs)
        return -1;
    const int index = data->mFreeList[data->mNumSDFs];
    data->mNumSDFs += 1;
    return index;
}

void SDFFree(FlexSDFData* data, int index)
{
    if (data->mTextures[index] != 0)
        data->mTextures[index]->release();
    data->mTextures[index] = 0;
    data->mNumSDFs -= 1;
    data->mFreeList[data->mNumSDFs] = index;
}

// ---------------------------------------------------------------------------
// FlexConvexMeshData — @0x2f130 (431 bytes) / @0x2f2e0 / @0x2e380 / @0x2e3a0
// ---------------------------------------------------------------------------
void ConvexCreate(FlexConvexMeshData* data, NvFlex::ContextEx* context,
                  bool shflSupported, int smCount)
{
    (void)shflSupported;
    (void)smCount;

    data->mMaxConvexes = 0x1000;
    data->mMaxPlanesPerConvex = 0x40;
    data->mNumConvexes = 0;

    // 0x1000 convexes * 0x40 planes each, float4 per plane.
    CreateReadbackBuffer(data->mPlanes, context, 0x40000, 0x10);
    CreateReadbackBuffer(data->mConvexes, context, (unsigned)data->mMaxConvexes,
                         sizeof(FlexConvexMeshDevice));
    data->mFreeList = CreateFreeList((unsigned)data->mMaxConvexes);
}

void ConvexRelease(FlexConvexMeshData* data)
{
    data->mPlanes.Release();
    data->mConvexes.Release();
    std::free(data->mFreeList);
    data->mFreeList = 0;
    data->mNumConvexes = 0;
}

int ConvexAllocate(FlexConvexMeshData* data)
{
    if (data->mNumConvexes >= data->mMaxConvexes)
        return -1;
    const int index = data->mFreeList[data->mNumConvexes];
    data->mNumConvexes += 1;
    return index;
}

void ConvexFree(FlexConvexMeshData* data, int index)
{
    data->mNumConvexes -= 1;
    data->mFreeList[data->mNumConvexes] = index;
}

// ---------------------------------------------------------------------------
// NvResourceTracker — @0x303a0 / @0x305f0 / @0x308f0 / @0x30a50
// ---------------------------------------------------------------------------
void TrackerAdd(NvResourceTracker* tracker, NvFlexSolver* solver)
{
    tracker->mSolver.push_back(solver);
}

int TrackerGet(NvResourceTracker* tracker, const std::vector<unsigned>& pool, unsigned handle)
{
    (void)tracker;
    for (size_t i = 0; i < pool.size(); ++i)
    {
        if (pool[i] == handle)
            return (int)i;
    }
    return -1;
}

void TrackerRemove(NvResourceTracker* tracker, std::vector<unsigned>& pool, unsigned handle)
{
    (void)tracker;
    for (size_t i = 0; i < pool.size(); ++i)
    {
        if (pool[i] == handle)
        {
            pool.erase(pool.begin() + i);
            return;
        }
    }
}

// @0x30a50 (217 bytes): drain both pools.  Mesh handles cannot exist yet (the
// NvFlexCreate*Mesh entry points are not reconstructed), so the mesh, SDF and
// convex pools are empty by construction; the solver list is what this call
// services today.
void TrackerCleanup(NvResourceTracker* tracker, NvFlexLibrary* lib)
{
    (void)lib;
    while (!tracker->mSolver.empty())
        NvFlexDestroySolver(tracker->mSolver[0]);
}

} // namespace flexobj
} // namespace NvFlex

// ===========================================================================
// Mesh / SDF / convex public API + FlattenTriangleMeshes
// ===========================================================================
//
// Appended by the mesh-API slice.  Everything above this banner is untouched.
// Evidence: work/decomp/nvflexd3d/NvFlexReleaseD3D_x64.dll.c
//   FlattenTriangleMeshes                 @ 0x37c40 (785 bytes)
//   FlexSDFData::Update                   @ 0x30200 (414 bytes)
//   FlexConvexMeshData::Update            @ 0x2f3b0 (305 bytes)
//   TightPack3D                           @ 0x2e520 (450 bytes)
//   NvFlexCreateTriangleMesh              @ 0x417a0 (183 bytes)
//   NvFlexDestroyTriangleMesh             @ 0x41860 (185 bytes)
//   NvFlexGetTriangleMeshes               @ 0x41920 (106 bytes)
//   NvFlexUpdateTriangleMesh              @ 0x41990 (1814 bytes)
//   NvFlexGetTriangleMeshBounds           @ 0x420b0 (241 bytes)
//   NvFlexCreateDistanceField             @ 0x3fb60 (209 bytes)
//   NvFlexDestroyDistanceField            @ 0x3fc40 (404 bytes)
//   NvFlexGetDistanceFields               @ 0x3fde0 (106 bytes)
//   NvFlexUpdateDistanceField             @ 0x3fe50 (175 bytes)
//   NvFlexCreateConvexMesh                @ 0x412b0 (206 bytes)
//   NvFlexDestroyConvexMesh               @ 0x41380 (316 bytes)
//   NvFlexGetConvexMeshes                 @ 0x414c0 (106 bytes)
//   NvFlexUpdateConvexMesh                @ 0x41530 (291 bytes)
//   NvFlexGetConvexMeshBounds             @ 0x41660 (308 bytes)
//
// The handles are 1-based slots: NvFlexCreate* allocates a 0-based store slot,
// stores `slot + 1` in the matching mResourceTracker pool and hands the value
// back; every other entry point decrements it and never validates it (the
// shipped code only checks for the null handle).
namespace
{

// The shipped code allocates the per-mesh record with `operator new(0xe0)` and
// zero-fills the members that the constructor writes; ~FlexTriangleMeshHost
// (@0x2f550, 152 bytes) only frees the BVH buffers and the two staging buffers.
void ReleaseTriangleMeshHost(FlexTriangleMeshHost* host)
{
    ::FreeBVH(&host->mBVH);

    if (host->mVertices.mBuffer != 0)
    {
        host->mVertices.mBuffer->release();
        host->mVertices.mBuffer = 0;
        host->mVertices.mBufferDesc = NvFlex::BufferDesc();
    }

    if (host->mIndices.mBuffer != 0)
    {
        host->mIndices.mBuffer->release();
        host->mIndices.mBuffer = 0;
        host->mIndices.mBufferDesc = NvFlex::BufferDesc();
    }
}

// Host-side growth used by NvFlexUpdateTriangleMesh: a staging buffer
// (bufferType 0x17 == eBuffer | eUAV_SRV | eStage | eStructured) that is only
// recreated when the existing descriptor cannot hold the request.
void EnsureHostMeshBuffer(DxBuffer* buffer, NvFlex::ContextEx* context,
                          unsigned dim, unsigned stride)
{
    if (buffer->mBuffer != 0)
    {
        if (buffer->mBufferDesc.dim >= dim && buffer->mBufferDesc.stride >= stride)
            return;

        buffer->mBuffer->release();
        buffer->mBuffer = 0;
    }

    NvFlex::BufferDesc desc;
    std::memset(&desc, 0, sizeof(desc));
    desc.dim = dim;
    desc.stride = stride;
    desc.bufferType = 0x17;   // eBuffer | eUAV_SRV | eStage | eStructured
    desc.data = 0;

    buffer->mBufferDesc = desc;
    buffer->mBuffer = context->createBuffer(&desc, 0);
}

// FlexSDFDevice: the 32-byte record the SDF shaders read.  The shipped write at
// 0x30200 stores {dimX, dimY, dimZ, <unused>} then the reciprocal dimensions as
// three floats; the trailing word is left unwritten there, zeroed here.
struct FlexSDFDeviceRecord
{
    int   dimX;
    int   dimY;
    int   dimZ;
    int   pad0;
    float invX;
    float invY;
    float invZ;
    float pad1;
};

static_assert(sizeof(FlexSDFDeviceRecord) == 32, "FlexSDFDevice is 32 bytes");

void LogError(NvFlexLibrary* lib, const char* message, int line)
{
    if (lib->mErrorFunc != 0)
        FlexLogFunc(lib->mErrorFunc, eNvFlexLogError, message, "..\\..\\dx\\flex.cpp", line);
}

} // namespace

namespace NvFlex
{
namespace flexobj
{

// ---------------------------------------------------------------------------
// FlexSDFData::Update — @0x30200 (414 bytes)
// ---------------------------------------------------------------------------
// Rebuilds mTextures[index] from the mapped source volume and publishes
// {dims, 1/dims} in mSDFs[index].  The texture descriptor is
// {format = 2, dim = (x,y,z), uploadAccess = true, downloadAccess = false,
//  data = mapped source}.
void SDFUpdate(FlexSDFData* data, NvFlex::ContextEx* context, int index,
               int dimX, int dimY, int dimZ, NvFlex::Buffer* field)
{
    const float invX = 1.0f / (float)dimX;
    const float invY = 1.0f / (float)dimY;
    const float invZ = 1.0f / (float)dimZ;

    NvFlex::Texture3D*& texture = data->mTextures[(size_t)index];
    if (texture != 0)
    {
        texture->release();
        texture = 0;
    }

    void* mapped = context->map(field, eMapReadWrite, true);

    NvFlex::Texture3DDesc desc;
    std::memset(&desc, 0, sizeof(desc));
    desc.format = NvFlexFormat(2);
    desc.dim.x = (unsigned)dimX;
    desc.dim.y = (unsigned)dimY;
    desc.dim.z = (unsigned)dimZ;
    desc.uploadAccess = true;
    desc.downloadAccess = false;
    desc.data = mapped;

    texture = context->createTexture3D(&desc);

    context->unmap(field);

    NvFlex::Buffer* sdfBuffer = data->mSDFs.mBuffer;
    FlexSDFDeviceRecord* records =
        (FlexSDFDeviceRecord*)context->map(sdfBuffer, eMapWrite, true);

    FlexSDFDeviceRecord& record = records[index];
    record.dimX = dimX;
    record.dimY = dimY;
    record.dimZ = dimZ;
    record.pad0 = 0;
    record.invX = invX;
    record.invY = invY;
    record.invZ = invZ;
    record.pad1 = 0.0f;

    context->unmap(sdfBuffer);
    context->upload(sdfBuffer, (unsigned)index << 5, 0x20);
}

// ---------------------------------------------------------------------------
// FlexConvexMeshData::Update — @0x2f3b0 (305 bytes)
// ---------------------------------------------------------------------------
// Uploads the plane set to mPlanes[index * mMaxPlanesPerConvex] and publishes
// the record {lower, upper, planeOffset, planeCount} in mConvexes[index].
void ConvexUpdate(FlexConvexMeshData* data, NvFlex::ContextEx* context, int index,
                  NvFlexBuffer* planes, int numPlanes,
                  const XVector3<float>* lower, const XVector3<float>* upper)
{
    const int planeOffset = data->mMaxPlanesPerConvex * index;

    if (numPlanes != 0)
    {
        context->copyToDevice(data->mPlanes.mBuffer, (unsigned)planeOffset << 4,
                              planes->mBuffer, 0, (unsigned)numPlanes << 4,
                              eDownloadBuffer);
    }

    NvFlex::Buffer* convexBuffer = data->mConvexes.mBuffer;
    FlexConvexMeshDevice* records =
        (FlexConvexMeshDevice*)context->map(convexBuffer, eMapWrite, true);

    FlexConvexMeshDevice& record = records[index];
    record.mLower.x = lower->x;
    record.mLower.y = lower->y;
    record.mLower.z = lower->z;
    record.mLower.w = 0.0f;
    record.mUpper.x = upper->x;
    record.mUpper.y = upper->y;
    record.mUpper.z = upper->z;
    record.mUpper.w = 0.0f;
    record.mPlaneOffset = planeOffset;
    record.mPlaneCount = numPlanes;
    record.mPad[0] = 0;
    record.mPad[1] = 0;

    context->unmap(convexBuffer);
    context->upload(convexBuffer, (unsigned)index * 0x30, 0x30);
}

// ---------------------------------------------------------------------------
// TightPack3D — @0x2e520 (450 bytes)
// ---------------------------------------------------------------------------
// `dim = ceilf(radius / separation)` and then the same tight hexagonal lattice
// the CUDA library generates: stride `separation` in x, sqrt(3/4)*separation in
// y/z, an alternating half-stride offset in x, centre skipped, points outside
// `radius` rejected, at most `maxPoints` written.  Returns the count.
int TightPack3D(float radius, float separation, XVector3<float>* points, int maxPoints)
{
    const int dim = (int)std::ceil(radius / separation);
    const float root3over4 = std::sqrt(0.75f);

    int count = 0;

    for (int z = -dim; z <= dim; ++z)
    {
        for (int y = -dim; y <= dim; ++y)
        {
            for (int x = -dim; x <= dim; ++x)
            {
                const float xpos = (float)x * separation
                                 + (((y + z) & 1) ? separation * 0.5f : 0.0f);
                const float ypos = (float)y * root3over4 * separation;
                const float zpos = (float)z * root3over4 * separation;

                const float lengthSq = xpos * xpos + ypos * ypos + zpos * zpos;

                if (lengthSq == 0.0f)
                    continue;

                if (count < maxPoints && std::sqrt(lengthSq) <= radius)
                {
                    points[count].x = xpos;
                    points[count].y = ypos;
                    points[count].z = zpos;
                    ++count;
                }
            }
        }
    }

    return count;
}

} // namespace flexobj
} // namespace NvFlex

// ---------------------------------------------------------------------------
// FlattenTriangleMeshes — @0x37c40 (785 bytes), global scope
// ---------------------------------------------------------------------------
// Concatenates every live host mesh into the shared arrays that
// mConstTriMeshInfo / ComputeTriangleBounds and the collide stage consume, then
// uploads the per-mesh FlexTriangleMeshDevice table
// ({lower, upper, indexStart, vertexStart, nodeStart}).
void FlattenTriangleMeshes(NvFlexLibrary* lib)
{
    FlexTriangleMeshData& data = lib->mTriangleMeshData;

    const int numMeshes = (int)data.mTriMeshTableHost.size();

    int numTris = 0;
    int numVertices = 0;
    for (int i = 0; i < numMeshes; ++i)
    {
        FlexTriangleMeshHost* host = data.mTriMeshTableHost[(size_t)i];
        if (host != 0)
        {
            numTris += host->mNumTris;
            numVertices += host->mNumVertices;
        }
    }

    NvFlex::flexobj::TriangleMeshResize(&data, lib->mContext, numVertices, numTris,
                                        numMeshes);

    std::vector<FlexTriangleMeshDevice> table((size_t)numMeshes);

    int vertexStart = 0;
    int indexStart = 0;
    int nodeStart = 0;

    for (int i = 0; i < numMeshes; ++i)
    {
        FlexTriangleMeshHost* host = data.mTriMeshTableHost[(size_t)i];
        if (host == 0)
            continue;

        if (host->mNumVertices != 0)
        {
            lib->mContext->copy(data.mVerticesArray.mBuffer, (unsigned)vertexStart << 4,
                                host->mVertices.mBuffer, 0,
                                (unsigned)host->mNumVertices << 4);
        }

        if (host->mNumIndices != 0)
        {
            lib->mContext->copy(data.mIndicesArray.mBuffer, (unsigned)indexStart << 2,
                                host->mIndices.mBuffer, 0,
                                (unsigned)host->mNumIndices << 2);
        }

        if (host->mBVH.mNumNodes != 0)
        {
            lib->mContext->copy(data.mBvhData.mNodeLowersArray.mBuffer,
                                (unsigned)nodeStart << 4,
                                host->mBVH.mNodeLowers.mBuffer, 0,
                                (unsigned)host->mBVH.mNumNodes << 4);
            lib->mContext->copy(data.mBvhData.mNodeUppersArray.mBuffer,
                                (unsigned)nodeStart << 4,
                                host->mBVH.mNodeUppers.mBuffer, 0,
                                (unsigned)host->mBVH.mNumNodes << 4);
        }

        lib->mContext->copy(data.mBvhData.mRootNodeArray.mBuffer, (unsigned)i << 2,
                            host->mBVH.mRootNode.mBuffer, 0, 4);

        FlexTriangleMeshDevice& entry = table[(size_t)i];
        entry.mLower.x = host->mLower.x;
        entry.mLower.y = host->mLower.y;
        entry.mLower.z = host->mLower.z;
        entry.mLower.w = 0.0f;
        entry.mUpper.x = host->mUpper.x;
        entry.mUpper.y = host->mUpper.y;
        entry.mUpper.z = host->mUpper.z;
        entry.mUpper.w = 0.0f;
        entry.mIndexStart = indexStart;
        entry.mVertexStart = vertexStart;
        entry.mNodeStart = nodeStart;
        entry.mPad = 0;

        indexStart += host->mNumIndices;
        vertexStart += host->mNumVertices;
        nodeStart += host->mBVH.mNumNodes;
    }

    if (numMeshes != 0)
    {
        NvFlex::Buffer* tableBuffer = data.mTriMeshTableDevice.mBuffer;
        void* mapped = lib->mContext->map(tableBuffer, NvFlex::eMapWrite, true);
        std::memcpy(mapped, table.data(),
                    (size_t)numMeshes * sizeof(FlexTriangleMeshDevice));
        lib->mContext->unmap(tableBuffer);
        lib->mContext->upload(tableBuffer, 0, (unsigned)numMeshes * 0x30);
    }
}

// ===========================================================================
// Public entry points
// ===========================================================================
extern "C"
{

// ---------------------------------------------------------------------------
// @0x417a0 (183 bytes)
// ---------------------------------------------------------------------------
NvFlexTriangleMeshId NvFlexCreateTriangleMesh(NvFlexLibrary* lib)
{
    const int index = NvFlex::flexobj::TriangleMeshAllocate(&lib->mTriangleMeshData,
                                                            lib->mContext);
    const unsigned handle = (unsigned)(index + 1);
    lib->mResourceTracker.mTriangleMesh.push_back(handle);
    return handle;
}

// ---------------------------------------------------------------------------
// @0x41860 (185 bytes)
// ---------------------------------------------------------------------------
void NvFlexDestroyTriangleMesh(NvFlexLibrary* lib, NvFlexTriangleMeshId mesh)
{
    if (mesh == 0)
    {
        LogError(lib, "Trying to destroy an invalid triangle mesh", 0xa13);
        return;
    }

    NvFlex::flexobj::TrackerRemove(&lib->mResourceTracker,
                                   lib->mResourceTracker.mTriangleMesh, mesh);

    FlexTriangleMeshHost* host =
        lib->mTriangleMeshData.mTriMeshTableHost[(size_t)(mesh - 1)];
    if (host != 0)
    {
        ReleaseTriangleMeshHost(host);
        delete host;
    }
    lib->mTriangleMeshData.mTriMeshTableHost[(size_t)(mesh - 1)] = 0;
}

// ---------------------------------------------------------------------------
// @0x41920 (106 bytes)
// ---------------------------------------------------------------------------
// Copies min(count, n) handles; returns the pool count (see the shipped
// `memcpy(dst, begin, min(count,n)*4)` at 0x18004196d).
int NvFlexGetTriangleMeshes(NvFlexLibrary* lib, NvFlexTriangleMeshId* meshes, int n)
{
    const std::vector<unsigned>& pool = lib->mResourceTracker.mTriangleMesh;
    const size_t count = pool.size();
    const size_t limit = (size_t)n;
    const size_t copyCount = (count >= limit) ? limit : count;

    if (copyCount != 0)
        std::memcpy(meshes, pool.data(), copyCount * sizeof(unsigned));

    return (int)count;
}

// ---------------------------------------------------------------------------
// @0x41990 (1814 bytes)
// ---------------------------------------------------------------------------
void NvFlexUpdateTriangleMesh(NvFlexLibrary* lib, NvFlexTriangleMeshId mesh,
                              NvFlexBuffer* vertices, NvFlexBuffer* indices,
                              int numVertices, int numTriangles,
                              const float* lower, const float* upper)
{
    if (mesh == 0)
    {
        LogError(lib, "Trying to update an invalid triangle mesh", 0xa24);
        return;
    }

    FlexTriangleMeshHost* host =
        lib->mTriangleMeshData.mTriMeshTableHost[(size_t)(mesh - 1)];
    if (host == 0)
    {
        LogError(lib, "Trying to update an invalid triangle mesh", 0xa2c);
        return;
    }

    if (lower == 0 || upper == 0)
    {
        LogError(lib, "Spatial bounds of mesh cannot be null", 0xa32);
        return;
    }

    if (numTriangles == 0 || numVertices == 0)
    {
        // The shipped body stores one 8-byte zero across mNumTris/mNumVertices.
        host->mNumTris = 0;
        host->mNumVertices = 0;
        return;
    }

    NvFlex::ContextEx* context = lib->mContext;

    // Index buffer: triCapacity = (numTriangles * 3) / 2, 3 indices per tri.
    if (host->mMaxTris < numTriangles)
    {
        const int triCapacity = (numTriangles * 3) / 2;
        EnsureHostMeshBuffer(&host->mIndices, context, (unsigned)(triCapacity * 3), 4);
        host->mMaxTris = triCapacity;
    }

    // Vertex buffer: vertexCapacity = (numVertices * 3) / 2, float4 per vertex.
    if (host->mMaxVertices < numVertices)
    {
        const int vertexCapacity = (numVertices * 3) / 2;
        EnsureHostMeshBuffer(&host->mVertices, context, (unsigned)vertexCapacity, 0x10);
        host->mMaxVertices = vertexCapacity;
    }

    host->mNumVertices = numVertices;
    host->mNumTris = numTriangles;
    host->mNumIndices = numTriangles * 3;

    NvFlex::flexobj::WriteData(context, host->mVertices.mBuffer, 0, vertices, 0,
                               (unsigned)numVertices << 4);
    NvFlex::flexobj::WriteData(context, host->mIndices.mBuffer, 0, indices, 0,
                               (unsigned)numTriangles * 0xc);

    host->mLower.x = lower[0];
    host->mLower.y = lower[1];
    host->mLower.z = lower[2];
    host->mUpper.x = upper[0];
    host->mUpper.y = upper[1];
    host->mUpper.z = upper[2];

    lib->mConstTriMeshInfo.Create(context, 1, 4);
    lib->mConstTriMeshInfo.Copy(context, &host->mNumTris, 4);

    if (numTriangles > lib->mMaxStaticTris)
    {
        const unsigned capacity = (unsigned)((numTriangles * 3) / 2);

        lib->mStaticTriLowers.Create(context, capacity, 0x10, 0);
        lib->mStaticTriUppers.Create(context, capacity, 0x10, 0);

        lib->mMaxStaticTris = (int)capacity;
    }

    NvFlex::DispatchParams params;
    std::memset(&params, 0, sizeof(params));

    // Shader 029 (ComputeTriangleBounds): t0 = flattened vertices,
    // t1 = flattened indices, u0 = static triangle lowers, u1 = uppers,
    // b0 = mConstTriMeshInfo {numTris}; one 256-thread group per 256 triangles.
    params.shader = lib->mShaderComputeTriangleBounds;
    params.gridDim.x = ((unsigned)numTriangles + 0xff) >> 8;
    params.gridDim.y = 1;
    params.gridDim.z = 1;
    params.rootConstantBuffer = lib->mConstTriMeshInfo.mBuffer;
    params.readOnly[0] = host->mVertices.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[1] = host->mIndices.mBuffer->getResource(NvFlex::eBuffer);
    params.readWrite[0] = lib->mStaticTriLowers.mBuffer->getResourceRW();
    params.readWrite[1] = lib->mStaticTriUppers.mBuffer->getResourceRW();

    context->dispatch(&params);

    Bounds bounds;
    bounds.lower.x = lower[0];
    bounds.lower.y = lower[1];
    bounds.lower.z = lower[2];
    bounds.upper.x = upper[0];
    bounds.upper.y = upper[1];
    bounds.upper.z = upper[2];

    lib->mBVHBuilder->Build(&host->mBVH, &lib->mStaticTriLowers, &lib->mStaticTriUppers,
                            0, numTriangles, &bounds, 0);

    lib->mRebuildTriangleData = true;
}

// ---------------------------------------------------------------------------
// @0x420b0 (241 bytes)
// ---------------------------------------------------------------------------
void NvFlexGetTriangleMeshBounds(NvFlexLibrary* lib, const NvFlexTriangleMeshId mesh,
                                 float* lower, float* upper)
{
    if (mesh == 0)
    {
        LogError(lib, "Trying to query bounds of an invalid triangle mesh", 0xa8b);
        return;
    }

    FlexTriangleMeshHost* host =
        lib->mTriangleMeshData.mTriMeshTableHost[(size_t)(mesh - 1)];
    if (host == 0)
    {
        LogError(lib, "Trying to query bounds of an invalid triangle mesh", 0xa92);
        return;
    }

    lower[0] = host->mLower.x;
    lower[1] = host->mLower.y;
    lower[2] = host->mLower.z;
    upper[0] = host->mUpper.x;
    upper[1] = host->mUpper.y;
    upper[2] = host->mUpper.z;
}

// ---------------------------------------------------------------------------
// @0x3fb60 (209 bytes)
// ---------------------------------------------------------------------------
NvFlexDistanceFieldId NvFlexCreateDistanceField(NvFlexLibrary* lib)
{
    const int index = NvFlex::flexobj::SDFAllocate(&lib->mSDFData);
    const unsigned handle = (unsigned)(index + 1);
    lib->mResourceTracker.mSDF.push_back(handle);
    return handle;
}

// ---------------------------------------------------------------------------
// @0x3fc40 (404 bytes)
// ---------------------------------------------------------------------------
void NvFlexDestroyDistanceField(NvFlexLibrary* lib, NvFlexDistanceFieldId sdf)
{
    if (sdf == 0)
    {
        LogError(lib, "Trying to update an invalid SDF", 0x8fe);
        return;
    }

    NvFlex::flexobj::TrackerRemove(&lib->mResourceTracker,
                                   lib->mResourceTracker.mSDF, sdf);
    NvFlex::flexobj::SDFFree(&lib->mSDFData, (int)(sdf - 1));
}

// ---------------------------------------------------------------------------
// @0x3fde0 (106 bytes)
// ---------------------------------------------------------------------------
int NvFlexGetDistanceFields(NvFlexLibrary* lib, NvFlexDistanceFieldId* sdfs, int n)
{
    const std::vector<unsigned>& pool = lib->mResourceTracker.mSDF;
    const size_t count = pool.size();
    const size_t limit = (size_t)n;
    const size_t copyCount = (count >= limit) ? limit : count;

    if (copyCount != 0)
        std::memcpy(sdfs, pool.data(), copyCount * sizeof(unsigned));

    return (int)count;
}

// ---------------------------------------------------------------------------
// @0x3fe50 (175 bytes)
// ---------------------------------------------------------------------------
void NvFlexUpdateDistanceField(NvFlexLibrary* lib, NvFlexDistanceFieldId sdf,
                               int dimx, int dimy, int dimz, NvFlexBuffer* field)
{
    if (sdf == 0)
    {
        LogError(lib, "Trying to update an invalid SDF", 0x911);
        return;
    }

    NvFlex::flexobj::SDFUpdate(&lib->mSDFData, lib->mContext, (int)(sdf - 1),
                               dimx, dimy, dimz, field->mBuffer);
}

// ---------------------------------------------------------------------------
// @0x412b0 (206 bytes)
// ---------------------------------------------------------------------------
NvFlexConvexMeshId NvFlexCreateConvexMesh(NvFlexLibrary* lib)
{
    const int index = NvFlex::flexobj::ConvexAllocate(&lib->mConvexMeshData);
    const unsigned handle = (unsigned)(index + 1);
    lib->mResourceTracker.mConvexMesh.push_back(handle);
    return handle;
}

// ---------------------------------------------------------------------------
// @0x41380 (316 bytes)
// ---------------------------------------------------------------------------
void NvFlexDestroyConvexMesh(NvFlexLibrary* lib, NvFlexConvexMeshId convex)
{
    if (convex == 0)
    {
        LogError(lib, "Trying to destroy an invalid convex mesh", 0x9d6);
        return;
    }

    NvFlex::flexobj::TrackerRemove(&lib->mResourceTracker,
                                   lib->mResourceTracker.mConvexMesh, convex);
    NvFlex::flexobj::ConvexFree(&lib->mConvexMeshData, (int)(convex - 1));
}

// ---------------------------------------------------------------------------
// @0x414c0 (106 bytes)
// ---------------------------------------------------------------------------
int NvFlexGetConvexMeshes(NvFlexLibrary* lib, NvFlexConvexMeshId* meshes, int n)
{
    const std::vector<unsigned>& pool = lib->mResourceTracker.mConvexMesh;
    const size_t count = pool.size();
    const size_t limit = (size_t)n;
    const size_t copyCount = (count >= limit) ? limit : count;

    if (copyCount != 0)
        std::memcpy(meshes, pool.data(), copyCount * sizeof(unsigned));

    return (int)count;
}

// ---------------------------------------------------------------------------
// @0x41530 (291 bytes)
// ---------------------------------------------------------------------------
void NvFlexUpdateConvexMesh(NvFlexLibrary* lib, NvFlexConvexMeshId convex,
                            NvFlexBuffer* planes, int numPlanes,
                            const float* lower, const float* upper)
{
    if (convex == 0)
    {
        // The shipped code reuses the destroy string here (line 0x9e7).
        LogError(lib, "Trying to destroy an invalid convex mesh", 0x9e7);
        return;
    }

    if (numPlanes > 0x40)
    {
        if (lib->mErrorFunc != 0)
            FlexLogFunc(lib->mErrorFunc, eNvFlexLogWarning,
                        "Trying to update a convex mesh with too many planes %d > %d",
                        "..\\..\\dx\\flex.cpp", 0x9ed, numPlanes, 0x40);
        return;
    }

    XVector3<float> lowerCopy(lower[0], lower[1], lower[2]);
    XVector3<float> upperCopy(upper[0], upper[1], upper[2]);

    NvFlex::flexobj::ConvexUpdate(&lib->mConvexMeshData, lib->mContext,
                                  (int)(convex - 1), planes, numPlanes,
                                  &lowerCopy, &upperCopy);
}

// ---------------------------------------------------------------------------
// @0x41660 (308 bytes)
// ---------------------------------------------------------------------------
void NvFlexGetConvexMeshBounds(NvFlexLibrary* lib, NvFlexConvexMeshId mesh,
                               float* lower, float* upper)
{
    if (mesh == 0)
    {
        LogError(lib, "Trying to query bounds for an invalid convex mesh", 0x9f8);
        return;
    }

    NvFlex::Buffer* convexBuffer = lib->mConvexMeshData.mConvexes.mBuffer;
    const unsigned offset = (unsigned)(mesh - 1) * 0x30;

    lib->mContext->download(convexBuffer, offset, 0x30);
    const char* mapped =
        (const char*)lib->mContext->map(convexBuffer, NvFlex::eMapReadWrite, true);
    const FlexConvexMeshDevice* record =
        (const FlexConvexMeshDevice*)(mapped + offset);

    const float lowerX = record->mLower.x;
    const float lowerY = record->mLower.y;
    const float lowerZ = record->mLower.z;
    const float upperX = record->mUpper.x;
    const float upperY = record->mUpper.y;
    const float upperZ = record->mUpper.z;

    lib->mContext->unmap(convexBuffer);

    lower[0] = lowerX;
    lower[1] = lowerY;
    lower[2] = lowerZ;
    upper[0] = upperX;
    upper[1] = upperY;
    upper[2] = upperZ;
}

} // extern "C"
