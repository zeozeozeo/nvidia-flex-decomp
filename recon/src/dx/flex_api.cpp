// flexdecomp: reconstructed D3D backend — flex.obj, library and public API.
//
// Evidence: work/decomp/nvflexd3d/NvFlexReleaseD3D_x64.dll.c.  Each function
// below names its `// ---- <Name> @ <rva>` block and byte size.
//
// This translation unit holds the library object, the device selection, the
// initialisation / shutdown path (including the complete solver shader table
// construction) and the small public entry points that do not belong to the
// solver or the object model.  The shipped flex.obj is a single object file;
// splitting it by topic changes nothing observable.
#include <cstring>
#include <new>
#include <string>

#include "Bvh.h"
#include "FlexObjects.h"
#include "NvApiShim.h"
#include "contextD3D11/ContextD3D11.h"
#include "contextD3D11/DeviceD3D11.h"
#include "generated/shaders_flex.h"

namespace
{

const char kFlexSourceFile[] = "..\\..\\dx\\flex.cpp";

// The shipped code builds a ComputeShaderDesc from the DXBC bytes it has in
// .rdata.  In this reconstruction those bytes are the generated table.
NvFlex::ComputeShader* MakeShader(NvFlex::ContextEx* context, const char* symbol,
                                  const wchar_t* label, unsigned slot)
{
    const FlexShaderBlob* blob = findFlexShader(symbol);
    if (blob == 0)
        return 0;

    NvFlex::ComputeShaderDesc desc;
    desc.cs = blob->data;
    desc.cs_length = blob->size;
    desc.label = label;
    desc.NvAPI_Slot = slot;
    return context->createComputeShader(&desc);
}

// `sDefaultDesc` — the NvFlexInitDesc NvFlexInit uses when desc == nullptr.
// Evidence: NvFlexInit @0x35ba0 reads the global at 0x18012cf00.
struct DefaultInitDesc
{
    int    deviceIndex;
    bool   enableExtensions;
    void*  renderDevice;
    void*  renderContext;
    void*  computeContext;
    bool   runOnRenderContext;
    int    computeType;
};

DefaultInitDesc makeDefaultDesc()
{
    DefaultInitDesc desc;
    std::memset(&desc, 0, sizeof(desc));
    desc.deviceIndex = 0;
    desc.enableExtensions = false;
    desc.renderDevice = 0;
    desc.renderContext = 0;
    desc.computeContext = 0;
    desc.runOnRenderContext = true;
    desc.computeType = eNvFlexD3D11;
    return desc;
}

} // namespace

// ---------------------------------------------------------------------------
// NvFlexLibrary — @0x37040 (263 bytes) / ~NvFlexLibrary @0x37bb0 (114 bytes)
// ---------------------------------------------------------------------------
NvFlexLibrary::NvFlexLibrary()
{
    mErrorFunc = 0;
    mDevice = 0;
    mContext = 0;
    std::memset(mDeviceName, 0, sizeof(mDeviceName));

    mShaderPredict = 0;
    mShaderCalculateBounds = 0;
    mShaderCalculateBoundsGroup = 0;
    mShaderCalculateBoundsFinalize = 0;
    mShaderCalculateParticleHash = 0;
    mShaderCreateGrid = 0;
    mShaderClearInt = 0;
    mShaderClearFloat4 = 0;
    mShaderClearCellBuckets = 0;
    mShaderReorderParticles = 0;
    mShaderCollideParticles = 0;
    mShaderCollideShapes = 0;
    mShaderContinuousShockPropagation = 0;
    mShaderCalculateDensity = 0;
    mShaderCalculateDensitySurfaceTension = 0;
    mShaderSolveDensities = 0;
    mShaderSolveDensitiesNonFluid = 0;
    mShaderSolveDensitiesSurfaceTension = 0;
    mShaderCalculateInflatableVolume = 0;
    mShaderSolveInflatableVolume = 0;
    mShaderSolveSprings = 0;
    mShaderSolveShapes = 0;
    mShaderSolveShapes32 = 0;
    mShaderSolveShapes128 = 0;
    mShaderSolveShapesPlasticDeformation = 0;
    mShaderSolveShapesPlasticDeformation32 = 0;
    mShaderSolveShapesPlasticDeformation128 = 0;
    mShaderApplyDeltas = 0;
    mShaderSolveContactsSequential = 0;
    mShaderSolveContactsAveraged = 0;
    mShaderSolveContactsAccumulate = 0;
    mShaderCollideTriangles = 0;
    mShaderUpdateVelocities = 0;
    mShaderCalculateVorticity = 0;
    mShaderSolveVelocities = 0;
    mShaderUpdateTriangles = 0;
    mShaderUpdateTrianglesInit = 0;
    mShaderUpdateVertexNormalsInit = 0;
    mShaderUpdateVertexNormals = 0;
    mShaderNormalizeVertexNormals = 0;
    mShaderFinalize = 0;
    mShaderCreateDiffuseParticles = 0;
    mShaderUpdateDiffuseParticles = 0;
    mShaderClampDiffuseParticleCount = 0;
    mShaderCompactDiffuseParticles = 0;
    mShaderSmoothPositions = 0;
    mShaderCalculateAnisotropy = 0;
    mShaderComputeTriangleBounds = 0;
    mShaderTransformShapeBounds = 0;
    mShaderSpringsGenerateIndices = 0;
    mShaderSpringsParticleRange = 0;
    mShaderSpringsReorder = 0;

    mShaderClearConstantBufferInt.mBuffer = 0;
    mShaderClearConstantBufferFloat4.mBuffer = 0;
    mCellBucketStarts.mBuffer = 0;
    mCellBucketEnds.mBuffer = 0;
    mConstTriMeshInfo.mBuffer = 0;
    mStaticTriLowers.mBuffer = 0;
    mStaticTriUppers.mBuffer = 0;

    mBVHBuilder = 0;
    mMaxStaticTris = 0;
    mIsSHFLSupported = false;
    mIsFP32ATOMICSupported = false;
    mAGSContext = 0;
    mIsSwizzleSupported = false;
    mEnableAmdDriverBugWorkaround = false;
    mGpuVendorId = VENDOR_ID_OTHERS;
    mSMCount = -1;
    mSyncQuery = 0;

    mRebuildTriangleData = false;

    mTriangleMeshData.mMaxNumVertices = 0;
    mTriangleMeshData.mMaxNumTriangles = 0;
    mTriangleMeshData.mMaxNumMeshes = 0;
    mTriangleMeshData.mTriMeshTableDevice.mBuffer = 0;
    mTriangleMeshData.mVerticesArray.mBuffer = 0;
    mTriangleMeshData.mIndicesArray.mBuffer = 0;
    mTriangleMeshData.mBvhData.mNodeLowersArray.mBuffer = 0;
    mTriangleMeshData.mBvhData.mNodeUppersArray.mBuffer = 0;
    mTriangleMeshData.mBvhData.mRootNodeArray.mBuffer = 0;

    mConvexMeshData.mPlanes.mBuffer = 0;
    mConvexMeshData.mConvexes.mBuffer = 0;

    mSDFData.mSDFs.mBuffer = 0;
}

NvFlexLibrary::~NvFlexLibrary()
{
    // The shipped destructor only frees the four std::vector storages; the
    // scalar members are released by NvFlexShutdown.
    mResourceTracker.mSolver.clear();
    mResourceTracker.mTriangleMesh.clear();
    mResourceTracker.mSDF.clear();
    mResourceTracker.mConvexMesh.clear();
    mSDFData.mTextures.clear();
    mTriangleMeshData.mTriMeshTableHost.clear();
}

// ---------------------------------------------------------------------------
// DeviceSelect — @0x359f0 (417 bytes)
// ---------------------------------------------------------------------------
// Returns non-zero on failure.  DeviceD3D12 is not part of this reconstruction:
// eNvFlexD3D12 is reported by NvFlexInit before it gets here.
int DeviceSelect(NvFlexLibrary* lib, const NvFlexInitDesc* init)
{
    FlexDeviceDesc desc;
    desc.mDeviceNumber = 0xffffffffu;
    desc.mDevice = 0;
    desc.mComputeCommandQueue = 0;
    desc.mRenderCommandQueue = 0;
    desc.useComputeQueue = false;
    desc.enablePresent = false;
    desc.mErrorFunc = lib->mErrorFunc;

    if (init != 0)
    {
        desc.mDeviceNumber = init->deviceIndex == 0xffffffffu ? 0 : (unsigned)init->deviceIndex;
        desc.mDevice = init->renderDevice;
        desc.mRenderCommandQueue = init->renderContext;
        desc.mComputeCommandQueue = init->computeContext;

        if (init->computeType == eNvFlexD3D11)
        {
            desc.enablePresent = false;
            desc.useComputeQueue = init->runOnRenderContext == false;

            void* memory = NvFlex::g_FlexMalloc(sizeof(NvFlex::DeviceD3D11));
            lib->mDevice = memory != 0 ? new (memory) NvFlex::DeviceD3D11(&desc) : 0;
        }
        // eNvFlexD3D12: rejected by NvFlexInit (see below).
    }

    NvFlex::DeviceD3D11* device = static_cast<NvFlex::DeviceD3D11*>(lib->mDevice);
    if (device == 0 || (device->d3dDevice() == 0 && device->renderDevice() == 0))
        return 1;

    lib->mContext = device->createContext();
    if (lib->mContext == 0)
        return 1;

    std::strncpy(lib->mDeviceName, device->deviceName(), sizeof(lib->mDeviceName) - 1);
    lib->mDeviceName[sizeof(lib->mDeviceName) - 1] = '\0';
    lib->mGpuVendorId = GpuVendorId(device->gpuVendorId());
    lib->mSMCount = (int)device->smCount();
    return 0;
}

// ---------------------------------------------------------------------------
// NvFlexInit — @0x35ba0 (5269 bytes)
// ---------------------------------------------------------------------------
extern "C" NvFlexLibrary* NvFlexInit(int version, NvFlexErrorCallback errorFunc,
                                     NvFlexInitDesc* desc)
{
    // `sDefaultDesc`: the desc NvFlexInit substitutes when the caller passes
    // nullptr.  It requests D3D11 on adapter 0 with runOnRenderContext = true.
    DefaultInitDesc fallback;
    const NvFlexInitDesc* effective = desc;
    if (effective == 0)
    {
        fallback = makeDefaultDesc();
        effective = reinterpret_cast<const NvFlexInitDesc*>(&fallback);
    }

    if (version != 0x78)
    {
        if (errorFunc != 0)
            FlexLogFunc(errorFunc, eNvFlexLogError,
                        "Could not initialize Flex, header/NvFlexLibrary mismatch, "
                        "provided: %d, expected: %d", kFlexSourceFile, 699);
        return 0;
    }

    if (effective != 0 && effective->computeType == eNvFlexCUDA)
    {
        if (errorFunc == 0)
            return 0;
        FlexLogFunc(errorFunc, eNvFlexLogError,
                    "Trying to initialize D3D Flex with a CUDA compute type.",
                    kFlexSourceFile, 0x2c1);
        return 0;
    }

    // D3D12 is out of scope for this reconstruction; report it through the
    // caller's callback instead of silently creating a D3D11 device.
    if (effective != 0 && effective->computeType == eNvFlexD3D12)
    {
        if (errorFunc != 0)
            FlexLogFunc(errorFunc, eNvFlexLogError,
                        "D3D12 backend is not available in this reconstruction",
                        kFlexSourceFile, 0x2c1);
        return 0;
    }

    void* memory = NvFlex::g_FlexMalloc(sizeof(NvFlexLibrary));
    NvFlexLibrary* lib = memory != 0 ? new (memory) NvFlexLibrary() : 0;
    if (lib == 0)
        return 0;

    lib->mErrorFunc = errorFunc;

    if (DeviceSelect(lib, effective) != 0)
    {
        if (lib->mErrorFunc != 0)
            FlexLogFunc(lib->mErrorFunc, eNvFlexLogError,
                        "Could not initialize Flex, error querying devices.",
                        kFlexSourceFile, 0x2ca);
        return 0;
    }

    lib->mEnableAmdDriverBugWorkaround = false;

    // The extension probe: only NVIDIA and AMD can answer it, and any other
    // vendor (or a failed NVAPI init) disables extensions entirely.
    bool enableExtensions = effective->enableExtensions;
    if (lib->mGpuVendorId == VENDOR_ID_NVIDIA &&
        (NvApiInitialize(), lib->mSMCount == -1))
        enableExtensions = false;
    if (lib->mGpuVendorId == VENDOR_ID_AMD && enableExtensions)
    {
        // AMD driver extensions (AGS) are not shimmed: without them the
        // workaround stays off, exactly as on a machine without amd_ags_x64.dll.
        lib->mEnableAmdDriverBugWorkaround = false;
        enableExtensions = false;
    }
    if (lib->mGpuVendorId == VENDOR_ID_OTHERS)
        enableExtensions = false;

    lib->mSyncQuery = lib->mContext->createFence();
    lib->mIsSHFLSupported = false;
    lib->mIsFP32ATOMICSupported = false;
    lib->mIsSwizzleSupported = false;

    if (!enableExtensions)
    {
        lib->mSMCount = -1;
    }
    else
    {
        if (lib->mGpuVendorId == VENDOR_ID_NVIDIA)
        {
            bool supported = false;
            lib->mDevice->queryCapability(1, &supported);
            lib->mIsSHFLSupported = supported;
            lib->mDevice->queryCapability(0xd, &supported);
            lib->mIsFP32ATOMICSupported = supported;
        }
    }

    NvFlex::ContextEx* context = lib->mContext;

    // --- bounds shaders -----------------------------------------------------
    if (lib->mIsSHFLSupported)
    {
        lib->mShaderCalculateBounds = MakeShader(context, "g_Flex_CalculateBoundsNV",
                                                 L"CalculateBoundsNV", 7);
        lib->mShaderCalculateBoundsGroup = MakeShader(context, "g_Flex_CalculateBoundsGroupNV",
                                                      L"CalculateBoundsGroupNV", 7);
        lib->mShaderCalculateBoundsFinalize =
            MakeShader(context, "g_Flex_CalculateBoundsFinalizeNV",
                       L"CalculateBoundsFinalizeNV", 7);
        lib->mShaderCalculateInflatableVolume =
            MakeShader(context, "g_Flex_CalculateInflatableVolumeNV",
                       L"CalculateInflatableVolumeNV", 7);
    }
    else if (lib->mIsSwizzleSupported)
    {
        lib->mShaderCalculateBounds = MakeShader(context, "g_Flex_CalculateBoundsAMD",
                                                 L"CalculateBoundsAMD", 0xffffffffu);
        lib->mShaderCalculateBoundsGroup = MakeShader(context, "g_Flex_CalculateBoundsGroupAMD",
                                                      L"CalculateBoundsGroupAMD", 0xffffffffu);
        lib->mShaderCalculateBoundsFinalize =
            MakeShader(context, "g_Flex_CalculateBoundsFinalizeAMD",
                       L"CalculateBoundsFinalizeAMD", 0xffffffffu);
        lib->mShaderCalculateInflatableVolume =
            MakeShader(context, "g_Flex_CalculateInflatableVolumeAMD",
                       L"CalculateInflatableVolumeAMD", 0xffffffffu);
    }
    else
    {
        lib->mShaderCalculateBounds = MakeShader(context, "g_Flex_CalculateBounds",
                                                 L"CalculateBounds", 0xffffffffu);
        lib->mShaderCalculateBoundsGroup = MakeShader(context, "g_Flex_CalculateBoundsGroup",
                                                      L"CalculateBoundsGroup", 0xffffffffu);
        lib->mShaderCalculateBoundsFinalize =
            MakeShader(context, "g_Flex_CalculateBoundsFinalize",
                       L"CalculateBoundsFinalize", 0xffffffffu);
        lib->mShaderCalculateInflatableVolume =
            MakeShader(context, "g_Flex_CalculateInflatableVolume",
                       L"CalculateInflatableVolume", 0xffffffffu);
    }

    // --- inflatable / springs / triangles -----------------------------------
    if (lib->mIsFP32ATOMICSupported)
    {
        lib->mShaderSolveInflatableVolume =
            MakeShader(context, "g_Flex_SolveInflatableVolumeNV", L"SolveInflatableVolumeNV", 7);
        lib->mShaderSolveSprings = MakeShader(context, "g_Flex_SolveSpringsNV",
                                              L"SolveSpringsNV", 7);
        lib->mShaderUpdateTriangles = MakeShader(context, "g_Flex_UpdateTrianglesNV",
                                                 L"UpdateTrianglesNV", 7);
        lib->mShaderUpdateVertexNormals = MakeShader(context, "g_Flex_UpdateVertexNormalsNV",
                                                     L"UpdateVertexNormalsNV", 7);
    }
    else
    {
        lib->mShaderSolveInflatableVolume =
            MakeShader(context, "g_Flex_SolveInflatableVolume", L"SolveInflatableVolume",
                       0xffffffffu);
        lib->mShaderSolveSprings = MakeShader(context, "g_Flex_SolveSprings",
                                              L"SolveSprings", 0xffffffffu);
        lib->mShaderUpdateTriangles = MakeShader(context, "g_Flex_UpdateTriangles",
                                                 L"UpdateTriangles", 0xffffffffu);
        lib->mShaderUpdateVertexNormals = MakeShader(context, "g_Flex_UpdateVertexNormals",
                                                     L"UpdateVertexNormals", 0xffffffffu);
    }

    lib->mShaderUpdateTrianglesInit = MakeShader(context, "g_Flex_UpdateTrianglesInit",
                                                 L"UpdateTrianglesInit", 0xffffffffu);
    lib->mShaderUpdateVertexNormalsInit =
        MakeShader(context, "g_Flex_UpdateVertexNormalsInit", L"UpdateVertexNormalsInit",
                   0xffffffffu);
    lib->mShaderNormalizeVertexNormals =
        MakeShader(context, "g_Flex_NormalizeVertexNormals", L"NormalizeVertexNormals",
                   0xffffffffu);
    lib->mShaderCalculateVorticity = MakeShader(context, "g_Flex_CalculateVorticity",
                                                L"CalculateVorticity", 0xffffffffu);
    lib->mShaderPredict = MakeShader(context, "g_Flex_Predict", L"Predict", 0xffffffffu);
    lib->mShaderCalculateParticleHash =
        MakeShader(context, "g_Flex_CalculateParticleHash", L"CalculateParticleHash",
                   0xffffffffu);
    lib->mShaderCreateGrid = MakeShader(context, "g_Flex_CreateGrid", L"CreateGrid",
                                        0xffffffffu);
    lib->mShaderClearCellBuckets = MakeShader(context, "g_Flex_ClearCellBuckets",
                                              L"ClearCellBuckets", 0xffffffffu);
    lib->mShaderClearInt = MakeShader(context, "g_Flex_ClearInt", L"Clear", 0xffffffffu);

    // One element of 16 bytes (int2 clear parameters) and 32 bytes (float4).
    lib->mShaderClearConstantBufferInt.Create(lib->mContext, 1, 0x10);

    lib->mShaderClearFloat4 = MakeShader(context, "g_Flex_ClearFloat4", L"Clear", 0xffffffffu);
    lib->mShaderClearConstantBufferFloat4.Create(lib->mContext, 1, 0x20);

    lib->mShaderReorderParticles = MakeShader(context, "g_Flex_ReorderParticles",
                                              L"ReorderParticles", 0xffffffffu);
    lib->mShaderCollideParticles = MakeShader(context, "g_Flex_CollideParticles",
                                              L"CollideParticles", 0xffffffffu);
    lib->mShaderCollideShapes = MakeShader(context, "g_Flex_CollideShapes", L"CollideShapes",
                                           0xffffffffu);
    lib->mShaderContinuousShockPropagation =
        MakeShader(context, "g_Flex_ContinuousShockPropagation", L"ContinuousShockPropagation",
                   0xffffffffu);
    lib->mShaderCalculateDensity = MakeShader(context, "g_Flex_CalculateDensity",
                                              L"CalculateDensity", 0xffffffffu);
    lib->mShaderCalculateDensitySurfaceTension =
        MakeShader(context, "g_Flex_CalculateDensitySurfaceTension",
                   L"CalculateDensitySurfaceTension", 0xffffffffu);
    lib->mShaderSolveDensities = MakeShader(context, "g_Flex_SolveDensities",
                                            L"SolveDensities", 0xffffffffu);
    lib->mShaderSolveDensitiesNonFluid = MakeShader(context, "g_Flex_SolveDensitiesNonFluid",
                                                    L"SolveDensitiesNonFluid", 0xffffffffu);
    lib->mShaderSolveDensitiesSurfaceTension =
        MakeShader(context, "g_Flex_SolveDensitiesSurfaceTension",
                   L"SolveDensitiesSurfaceTension", 0xffffffffu);

    if (lib->mIsSHFLSupported == false || lib->mIsFP32ATOMICSupported == false)
    {
        lib->mShaderSolveShapes = MakeShader(context, "g_Flex_SolveShapes", L"SolveShapes",
                                             0xffffffffu);
        lib->mShaderSolveShapesPlasticDeformation =
            MakeShader(context, "g_Flex_SolveShapesPlasticDeformation",
                       L"SolveShapesPlasticDeformation", 0xffffffffu);
    }
    else
    {
        lib->mShaderSolveShapes = MakeShader(context, "g_Flex_SolveShapesNV", L"SolveShapesNV", 7);
        lib->mShaderSolveShapes32 = MakeShader(context, "g_Flex_SolveShapes32NV",
                                               L"SolveShapes32NV", 7);
        lib->mShaderSolveShapes128 = MakeShader(context, "g_Flex_SolveShapes128NV",
                                                L"SolveShapes128NV", 7);
        lib->mShaderSolveShapesPlasticDeformation =
            MakeShader(context, "g_Flex_SolveShapesPlasticDeformationNV",
                       L"SolveShapesPlasticDeformationNV", 7);
        lib->mShaderSolveShapesPlasticDeformation32 =
            MakeShader(context, "g_Flex_SolveShapesPlasticDeformation32NV",
                       L"SolveShapesPlasticDeformation32NV", 7);
        lib->mShaderSolveShapesPlasticDeformation128 =
            MakeShader(context, "g_Flex_SolveShapesPlasticDeformation128NV",
                       L"SolveShapesPlasticDeformation128NV", 7);
    }

    lib->mShaderApplyDeltas = MakeShader(context, "g_Flex_ApplyDeltas", L"ApplyDeltas",
                                         0xffffffffu);
    lib->mShaderSolveContactsSequential =
        MakeShader(context, "g_Flex_SolveContactsSequential", L"SolveContactsSequential",
                   0xffffffffu);
    lib->mShaderSolveContactsAveraged = MakeShader(context, "g_Flex_SolveContactsAveraged",
                                                    L"SolveContactsAveraged", 0xffffffffu);
    lib->mShaderSolveContactsAccumulate =
        MakeShader(context, "g_Flex_SolveContactsAccumulate", L"SolveContactsAccumulate",
                   0xffffffffu);
    lib->mShaderCollideTriangles = MakeShader(context, "g_Flex_CollideTriangles",
                                              L"CollideTriangles", 0xffffffffu);
    lib->mShaderUpdateVelocities = MakeShader(context, "g_Flex_UpdateVelocities",
                                              L"UpdateVelocities", 0xffffffffu);
    lib->mShaderSolveVelocities = MakeShader(context, "g_Flex_SolveVelocities",
                                             L"SolveVelocities", 0xffffffffu);
    lib->mShaderFinalize = MakeShader(context, "g_Flex_Finalize", L"Finalize", 0xffffffffu);
    lib->mShaderCreateDiffuseParticles = MakeShader(context, "g_Flex_CreateDiffuseParticles",
                                                    L"CreateDiffuseParticles", 0xffffffffu);
    lib->mShaderUpdateDiffuseParticles = MakeShader(context, "g_Flex_UpdateDiffuseParticles",
                                                    L"UpdateDiffuseParticles", 0xffffffffu);
    lib->mShaderClampDiffuseParticleCount =
        MakeShader(context, "g_Flex_ClampDiffuseParticleCount", L"ClampDiffuseParticleCount",
                   0xffffffffu);
    lib->mShaderCompactDiffuseParticles =
        MakeShader(context, "g_Flex_CompactDiffuseParticles", L"CompactDiffuseParticles",
                   0xffffffffu);
    lib->mShaderSmoothPositions = MakeShader(context, "g_Flex_SmoothPositions",
                                             L"SmoothPositions", 0xffffffffu);
    lib->mShaderCalculateAnisotropy = MakeShader(context, "g_Flex_CalculateAnisotropy",
                                                 L"CalculateAnisotropy", 0xffffffffu);
    lib->mShaderComputeTriangleBounds = MakeShader(context, "g_Flex_ComputeTriangleBounds",
                                                   L"ComputeTriangleBounds", 0xffffffffu);
    lib->mShaderTransformShapeBounds = MakeShader(context, "g_Flex_TransformShapeBounds",
                                                  L"TransformShapeBounds", 0xffffffffu);
    lib->mShaderSpringsGenerateIndices = MakeShader(context, "g_Flex_SpringsGenerateIndices",
                                                    L"SpringsGenerateIndices", 0xffffffffu);
    lib->mShaderSpringsParticleRange = MakeShader(context, "g_Flex_SpringsParticleRange",
                                                  L"SpringsParticleRange", 0xffffffffu);
    lib->mShaderSpringsReorder = MakeShader(context, "g_Flex_SpringsReorder",
                                            L"ShaderSpringsReorder", 0xffffffffu);

    // --- cell bucket tables -------------------------------------------------
    // 2 M structured uint elements, created from a zeroed host copy and then
    // released (the GPU copy keeps the initial data).
    {
        const unsigned elements = 0x200000;
        void* zeros = std::malloc(elements * 4);
        if (zeros != 0)
            std::memset(zeros, 0, elements * 4);

        NvFlex::BufferDesc desc;
        desc.format = NvFlexFormat(0);
        desc.dim = elements;
        desc.stride = 4;
        desc.bufferType = NvFlex::eBuffer | NvFlex::eUAV_SRV | NvFlex::eStructured; // 0x13
        desc.data = zeros;

        lib->mCellBucketStarts.mBufferDesc = desc;
        lib->mCellBucketStarts.mBuffer = lib->mContext->createBuffer(&desc, 0);

        lib->mCellBucketEnds.mBufferDesc = desc;
        lib->mCellBucketEnds.mBuffer = lib->mContext->createBuffer(&desc, 0);

        std::free(zeros);
    }

    void* builderMemory = NvFlex::g_FlexMalloc(sizeof(LinearBVHBuilderGPU));
    lib->mBVHBuilder = builderMemory != 0
        ? new (builderMemory) LinearBVHBuilderGPU(lib->mContext, lib->mIsSHFLSupported,
                                                  lib->mSMCount, lib->mIsSwizzleSupported)
        : 0;

    lib->mMaxStaticTris = 0;
    NvFlex::flexobj::ConvexCreate(&lib->mConvexMeshData, lib->mContext,
                                  lib->mIsSHFLSupported, lib->mSMCount);
    NvFlex::flexobj::SDFCreate(&lib->mSDFData, lib->mContext, lib->mIsSHFLSupported);

    return lib;
}

namespace
{

// Every solver shader owned by the library (NvFlexLibrary field list 0x2F4B).
typedef NvFlex::ComputeShader* NvFlexLibrary::*ShaderMember;

const ShaderMember kShaderMembers[] =
{
    &NvFlexLibrary::mShaderPredict, &NvFlexLibrary::mShaderCalculateBounds,
    &NvFlexLibrary::mShaderCalculateBoundsGroup, &NvFlexLibrary::mShaderCalculateBoundsFinalize,
    &NvFlexLibrary::mShaderCalculateParticleHash, &NvFlexLibrary::mShaderCreateGrid,
    &NvFlexLibrary::mShaderClearCellBuckets, &NvFlexLibrary::mShaderClearInt,
    &NvFlexLibrary::mShaderClearFloat4, &NvFlexLibrary::mShaderReorderParticles,
    &NvFlexLibrary::mShaderCollideParticles, &NvFlexLibrary::mShaderCollideShapes,
    &NvFlexLibrary::mShaderContinuousShockPropagation,
    &NvFlexLibrary::mShaderCalculateDensity,
    &NvFlexLibrary::mShaderCalculateDensitySurfaceTension,
    &NvFlexLibrary::mShaderSolveDensities, &NvFlexLibrary::mShaderSolveDensitiesNonFluid,
    &NvFlexLibrary::mShaderSolveDensitiesSurfaceTension,
    &NvFlexLibrary::mShaderCalculateInflatableVolume,
    &NvFlexLibrary::mShaderSolveInflatableVolume, &NvFlexLibrary::mShaderSolveSprings,
    &NvFlexLibrary::mShaderSolveShapes, &NvFlexLibrary::mShaderSolveShapes32,
    &NvFlexLibrary::mShaderSolveShapes128,
    &NvFlexLibrary::mShaderSolveShapesPlasticDeformation,
    &NvFlexLibrary::mShaderSolveShapesPlasticDeformation32,
    &NvFlexLibrary::mShaderSolveShapesPlasticDeformation128,
    &NvFlexLibrary::mShaderApplyDeltas, &NvFlexLibrary::mShaderSolveContactsSequential,
    &NvFlexLibrary::mShaderSolveContactsAveraged, &NvFlexLibrary::mShaderSolveContactsAccumulate,
    &NvFlexLibrary::mShaderCollideTriangles, &NvFlexLibrary::mShaderUpdateVelocities,
    &NvFlexLibrary::mShaderCalculateVorticity, &NvFlexLibrary::mShaderSolveVelocities,
    &NvFlexLibrary::mShaderUpdateTriangles, &NvFlexLibrary::mShaderUpdateTrianglesInit,
    &NvFlexLibrary::mShaderUpdateVertexNormalsInit,
    &NvFlexLibrary::mShaderUpdateVertexNormals, &NvFlexLibrary::mShaderNormalizeVertexNormals,
    &NvFlexLibrary::mShaderFinalize, &NvFlexLibrary::mShaderCreateDiffuseParticles,
    &NvFlexLibrary::mShaderUpdateDiffuseParticles,
    &NvFlexLibrary::mShaderClampDiffuseParticleCount,
    &NvFlexLibrary::mShaderCompactDiffuseParticles, &NvFlexLibrary::mShaderSmoothPositions,
    &NvFlexLibrary::mShaderCalculateAnisotropy, &NvFlexLibrary::mShaderComputeTriangleBounds,
    &NvFlexLibrary::mShaderTransformShapeBounds,
    &NvFlexLibrary::mShaderSpringsGenerateIndices, &NvFlexLibrary::mShaderSpringsParticleRange,
    &NvFlexLibrary::mShaderSpringsReorder
};

const unsigned kShaderMemberCount = sizeof(kShaderMembers) / sizeof(kShaderMembers[0]);

} // namespace

// ---------------------------------------------------------------------------
// NvFlexShutdown — @0x372a0 (2255 bytes)
// ---------------------------------------------------------------------------
extern "C" void NvFlexShutdown(NvFlexLibrary* lib)
{
    if (lib == 0)
        return;

    // The shipped sequence releases every owned resource in this order; the
    // patterns are identical, so the transcription is a table walk.
    lib->mStaticTriLowers.Release();
    lib->mStaticTriUppers.Release();
    lib->mConstTriMeshInfo.Release();

    if (lib->mBVHBuilder != 0)
    {
        lib->mBVHBuilder->~LinearBVHBuilderGPU();
        NvFlex::g_FlexFree(lib->mBVHBuilder);
        lib->mBVHBuilder = 0;
    }

    lib->mCellBucketStarts.Release();
    lib->mCellBucketEnds.Release();

    // Every shader is released and cleared the same way; a pointer-to-member
    // table keeps the list in one place (the shipped body repeats the same
    // eight instructions per shader).
    for (unsigned i = 0; i < kShaderMemberCount; ++i)
    {
        NvFlex::ComputeShader*& shader = lib->*kShaderMembers[i];
        if (shader != 0)
            shader->release();
        shader = 0;
    }

    if (lib->mSyncQuery != 0)
        lib->mSyncQuery->release();
    lib->mSyncQuery = 0;

    NvFlex::flexobj::TrackerCleanup(&lib->mResourceTracker, lib);
    NvFlex::flexobj::TriangleMeshRelease(&lib->mTriangleMeshData);
    NvFlex::flexobj::ConvexRelease(&lib->mConvexMeshData);
    NvFlex::flexobj::SDFRelease(&lib->mSDFData);

    if (lib->mDevice != 0 && lib->mDevice->gpuVendorId() == VENDOR_ID_AMD)
        lib->mDevice->queryVendorExtension(lib->mAGSContext, 0, 0);

    // Context and device are not reference counted: the shipped code calls
    // their scalar deleting destructors, which is a destructor + free here.
    if (lib->mContext != 0)
    {
        lib->mContext->~ContextEx();
        NvFlex::g_FlexFree(lib->mContext);
        lib->mContext = 0;
    }
    if (lib->mDevice != 0)
    {
        lib->mDevice->~Device();
        NvFlex::g_FlexFree(lib->mDevice);
        lib->mDevice = 0;
    }

    lib->~NvFlexLibrary();
    NvFlex::g_FlexFree(lib);
}

// ---------------------------------------------------------------------------
// NvFlexBuffer — @0x2efc0 (184 bytes) / ~NvFlexBuffer @0x2f080 (49 bytes)
// ---------------------------------------------------------------------------
NvFlexBuffer::NvFlexBuffer(int elementCount_, int stride_, NvFlexBufferType type_,
                           NvFlexLibrary* lib, NvFlex::ContextEx* context, void* data)
{
    mapped = false;
    type = type_;
    elementCount = elementCount_;
    mBuffer = 0;
    ptr = 0;
    elementByteStride = stride_;
    mFlexLib = lib;
    mContext = context;

    // type 3 is a registered (caller-owned) buffer: only the pointer is kept.
    if ((int)type_ == 3)
    {
        ptr = data;
        return;
    }

    NvFlex::BufferDesc desc;
    desc.format = NvFlexFormat(0);
    desc.dim = elementCount;
    desc.stride = stride_;
    desc.data = data;
    desc.bufferType = 0;

    if (type_ == eNvFlexBufferHost)
        desc.bufferType = NvFlex::eStage | NvFlex::eStructured;          // 0x14
    else if (type_ == eNvFlexBufferDevice)
        desc.bufferType = NvFlex::eBuffer | NvFlex::eStructured;         // 0x11
    else if ((int)type_ == 2)
        desc.bufferType = NvFlex::eDynamic | NvFlex::eStructured;        // 0x90

    mBuffer = context->createBuffer(&desc, data);
}

NvFlexBuffer::~NvFlexBuffer()
{
    if (mBuffer != 0)
        mBuffer->release();
    mBuffer = 0;
    mContext = 0;
    ptr = 0;
}

// ---------------------------------------------------------------------------
// Small public entry points
// ---------------------------------------------------------------------------
// @0x37c30 (6 bytes)
extern "C" int NvFlexGetVersion()
{
    return 0x78;
}

// @0x422a0 (15 bytes)
extern "C" void NvFlexGetDeviceAndContext(NvFlexLibrary* lib, void** device, void** context)
{
    *device = lib->mDevice;
    *context = lib->mContext;
}

// @0x422b0 (5 bytes)
extern "C" const char* NvFlexGetDeviceName(NvFlexLibrary* lib)
{
    return lib->mDeviceName;
}

// @0x422c0 (14 bytes)
extern "C" void NvFlexFlush(NvFlexLibrary* lib)
{
    lib->mContext->flush();
}

// @0x43c20 (14 bytes)
extern "C" void NvFlexComputeWaitForGraphics(NvFlexLibrary* lib)
{
    lib->mContext->computeWaitForGraphics();
}

// @0x41290 (3 bytes) / @0x412a0 (3 bytes) — empty in the shipped binary.
extern "C" void NvFlexAcquireContext(NvFlexLibrary* lib)
{
    (void)lib;
}

extern "C" void NvFlexRestoreContext(NvFlexLibrary* lib)
{
    (void)lib;
}

// @0x40350 (7 bytes)
extern "C" void NvFlexSetDebug(NvFlexSolver* solver, bool debug)
{
    solver->mDebug = debug;
}

// @0x40070 (106 bytes)
extern "C" NvFlexBuffer* NvFlexAllocBuffer(NvFlexLibrary* lib, int elementCount,
                                           int elementByteStride, NvFlexBufferType type)
{
    void* memory = NvFlex::g_FlexMalloc(sizeof(NvFlexBuffer));
    return memory != 0 ? new (memory) NvFlexBuffer(elementCount, elementByteStride, type, lib,
                                                   lib->mContext, 0)
                       : 0;
}

// @0x400e0 (61 bytes)
extern "C" void NvFlexFreeBuffer(NvFlexBuffer* buffer)
{
    if (buffer == 0)
        return;
    if (buffer->mBuffer != 0)
        buffer->mBuffer->release();
    buffer->mBuffer = 0;
    buffer->mContext = 0;
    buffer->ptr = 0;
    buffer->~NvFlexBuffer();
    NvFlex::g_FlexFree(buffer);
}

// @0x40160 (78 bytes)
extern "C" void* NvFlexMap(NvFlexBuffer* buffer, int flags)
{
    if (buffer == 0)
        return 0;
    return buffer->map(flags);
}

// @0x401b0 (47 bytes)
extern "C" void NvFlexUnmap(NvFlexBuffer* buffer)
{
    if (buffer != 0)
        buffer->unmap();
}

// NvFlexRegisterOGLBuffer is not defined and not exported by the shipped D3D
// backend (it belongs to the GL/CUDA interop build), so this unit omits it.

// @0x401f0 (3 bytes) — the shipped body is empty: the D3D backend has no GL
// resource to release, and the handle was never registered with it.
extern "C" void NvFlexUnregisterOGLBuffer(NvFlexBuffer* buffer)
{
    (void)buffer;
}

// @0x40230 (108 bytes)
extern "C" NvFlexBuffer* NvFlexRegisterD3DBuffer(NvFlexLibrary* lib, void* nativeBuffer,
                                                 int elementCount, int elementByteStride)
{
    void* memory = NvFlex::g_FlexMalloc(sizeof(NvFlexBuffer));
    if (memory == 0)
        return 0;

    NvFlexBuffer* buffer = static_cast<NvFlexBuffer*>(memory);
    buffer->mapped = false;
    buffer->type = NvFlexBufferType(3);   // registered/caller-owned
    buffer->mBuffer = 0;
    buffer->elementCount = elementCount;
    buffer->elementByteStride = elementByteStride;
    buffer->mContext = lib->mContext;
    buffer->mFlexLib = lib;
    buffer->ptr = nativeBuffer;
    return buffer;
}

// @0x402a0 (61 bytes)
extern "C" void NvFlexUnregisterD3DBuffer(NvFlexBuffer* buffer)
{
    NvFlexFreeBuffer(buffer);
}

// @0x43ae0 (319 bytes)
extern "C" void NvFlexCopyDeviceToHost(NvFlexSolver* solver, NvFlexBuffer* src, void* dst,
                                       int elementCount, int elementByteStride)
{
    NvFlexLibrary* lib = solver->mLib;

    void* memory = NvFlex::g_FlexMalloc(sizeof(NvFlexBuffer));
    if (memory == 0)
        return;
    NvFlexBuffer* staging = new (memory) NvFlexBuffer(elementCount, elementByteStride,
                                                      eNvFlexBufferHost, lib, lib->mContext, 0);

    if (elementCount > 0)
    {
        if (elementCount * elementByteStride != 0)
            lib->mContext->copyToDevice(staging->mBuffer, 0, src->mBuffer, 0,
                                        elementCount * elementByteStride,
                                        NvFlex::eDownloadBuffer);

        void* mapped = staging->map(0);
        std::memcpy(dst, mapped, elementCount * elementByteStride);
        staging->unmap();
        staging->~NvFlexBuffer();
        NvFlex::g_FlexFree(staging);
    }
}

// @0x42960 (143 bytes)
extern "C" float NvFlexGetDeviceLatency(NvFlexSolver* solver, unsigned long long* a,
                                        unsigned long long* b, unsigned long long* c)
{
    if (!solver->mLatencyTimerEnable)
    {
        solver->mLatencyTimerEnable = true;
    }
    else if (solver->mLatencyTimerBeginRecorded && solver->mLatencyTimerEndRecorded)
    {
        solver->mTimers->mTimerPool->sync();
        return solver->mTimers->mTimerPool->get(solver->mLatencyTimerIndex, a, b, c);
    }
    return 0.0f;
}

// @0x402e0 (104 bytes) — stores the callback and returns the previous one.
extern "C" NvFlexSolverCallback NvFlexRegisterSolverCallback(NvFlexSolver* solver,
                                                            NvFlexSolverCallback callback,
                                                            NvFlexSolverCallbackStage stage)
{
    NvFlexSolverCallback previous = solver->mCallbacks[(int)stage];
    solver->mCallbacks[(int)stage] = callback;
    return previous;
}
