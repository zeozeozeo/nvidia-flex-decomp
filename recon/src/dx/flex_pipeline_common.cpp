// flexdecomp: reconstructed D3D backend — the helpers every solver stage uses.
//
// `Dispatch` is the shipped ContextEx::dispatch call (slot +0xe0 of the shipped
// vtable, which the reconstruction names directly).  `Synchronize` is
// @0x30de0 (98 bytes) and only runs on the AMD driver path.
#include <cstring>

#include "FlexPipeline.h"

namespace nvpipe
{

// @0x30de0 (98 bytes): flush, signal and wait on the library's sync fence.
void Synchronize(NvFlexLibrary* lib)
{
    lib->mContext->flush();
    lib->mContext->fenceSet(lib->mSyncQuery);
    lib->mContext->fenceWait(lib->mSyncQuery, 0);
}

void ClearDispatch(NvFlex::DispatchParams& params)
{
    std::memset(&params, 0, sizeof(params));
}

void Dispatch(NvFlexSolver* solver, const NvFlex::DispatchParams& params)
{
    solver->mLib->mContext->dispatch(&params);
}

// The stage dispatchers only patch individual KernelParams fields after
// InitParams ran; one upload of the whole mirror per stage keeps the constant
// buffer coherent without tracking per-field dirty state.
void UploadKernelParams(NvFlexSolver* solver)
{
    NvFlex::ContextEx* context = solver->mLib->mContext;
    void* mapped = context->map(solver->mKernelParams.mBuffer);
    std::memcpy(mapped, &solver->mKernelParamsHost, sizeof(KernelParams));
    context->unmap(solver->mKernelParams.mBuffer);
}

} // namespace nvpipe
