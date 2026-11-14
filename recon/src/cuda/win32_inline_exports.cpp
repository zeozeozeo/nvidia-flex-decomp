// PE build (MSVC only): out-of-line emission of the four header-inline API
// helpers the shipped Windows DLLs export (dumpbin of NvFlexReleaseCUDA_x64.dll
// and NvFlexExtReleaseCUDA_x64.dll both list them).
//
// NvFlex.h defaults NV_FLEX_API to __declspec(dllexport) on _WIN32; including
// the header in a TU whose only content is these declarations makes MSVC
// emit external COMDAT copies of the inline definitions, which the frozen
// export lists in recon/win32/*.def resolve against.  Added to both the nvflex
// and nvflexext targets on WIN32.  The ELF/Linux path never compiles this
// file (the ELF gate pins the inline helpers differently).
#if defined(_MSC_VER)

#include "NvFlex.h"

// force external emission of every helper (MSVC emits dllexport inline
// functions as COMDATs; the references below make the intent explicit and
// keep the TU from being optimized to nothing)
namespace
{
struct FlexInlineExportAnchor
{
	int (*const makePhaseWithChannels)(int, int, int);
	int (*const makePhase)(int, int);
	int (*const makeShapeFlagsWithChannels)(NvFlexCollisionShapeType, bool, int);
	int (*const makeShapeFlags)(NvFlexCollisionShapeType, bool);
};
const FlexInlineExportAnchor g_flexInlineExportAnchor = {
	&NvFlexMakePhaseWithChannels,
	&NvFlexMakePhase,
	&NvFlexMakeShapeFlagsWithChannels,
	&NvFlexMakeShapeFlags,
};
}  // namespace

#endif  // _MSC_VER
