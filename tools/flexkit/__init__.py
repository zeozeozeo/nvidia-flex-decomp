"""flexkit - FleX 1.2.0 decompilation framework toolchain."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FLEX = ROOT / "FleX"
WORK = ROOT / "work"
STATE = ROOT / "state"
UNITS = ROOT / "units"
RECON = ROOT / "recon"
TOOLS = ROOT / "tools"

BINARY = WORK / "binary"
OBJ = BINARY / "obj"
SO = BINARY / "so"
GPU = BINARY / "gpu"
EXE = BINARY / "exe"
STUBS = BINARY / "stubs"
DWARF_DIR = WORK / "dwarf"
DECOMP = WORK / "decomp"
PDB_DIR = WORK / "pdb"
DXBC_DIR = WORK / "dxbc"
GEN = WORK / "gen"
DB = STATE / "units.db"

# module id -> (archive member .o name, kind of origin)
MODULES = {
    "nvflex.util": ("util.cpp.o", "nvflex"),
    "nvflex.bvh": ("cudabvh.o", "nvflex"),
    "nvflex.flex": ("cudaflex.o", "nvflex"),
    "nvflex.sort": ("cudasort.o", "nvflex"),
    "nvflexext.cloth": ("flexExtCloth.cpp.o", "nvflexext"),
    "nvflexext.container": ("flexExtContainer.cpp.o", "nvflexext"),
    "nvflexext.movingframe": ("flexExtMovingFrame.cpp.o", "nvflexext"),
    "nvflexext.rigid": ("flexExtRigid.cpp.o", "nvflexext"),
    "nvflexext.soft": ("flexExtSoft.cpp.o", "nvflexext"),
    "nvflexext.sdf": ("sdf.cpp.o", "nvflexext"),
    "nvflexext.voxelize": ("voxelize.cpp.o", "nvflexext"),
    "nvflexext.maths": ("maths.cpp.o", "nvflexext"),
    "nvflexext.aabbtree": ("aabbtree.cpp.o", "nvflexext"),
    "nvflexext.device": ("cudaflexExt.o", "nvflexext"),
    "nvflexdevice.flexdevice": ("flexDevice.cpp.o", "nvflexdevice"),
    "nvflexdevice.physxdevice": ("PhysXDevice.cpp.o", "nvflexdevice"),
}

# archive member -> module (reverse of MODULES)
MEMBER_MODULE = {m: k for k, (m, _) in MODULES.items()}

ARCHIVES = {
    "dbg": {
        "nvflex": "NvFlexDebugCUDA_x64.a",
        "nvflexext": "NvFlexExtDebugCUDA_x64.a",
        "nvflexdevice": "NvFlexDeviceDebug_x64.a",
    },
    "rel": {
        "nvflex": "NvFlexReleaseCUDA_x64.a",
        "nvflexext": "NvFlexExtReleaseCUDA_x64.a",
        "nvflexdevice": "NvFlexDeviceRelease_x64.a",
    },
}

# modules whose objects carry CUDA fatbins with kernels
KERNEL_MODULES = ("nvflex.flex", "nvflex.bvh", "nvflex.sort", "nvflexext.device")

EXPECTED_KERNELS = {"nvflex.flex": 41, "nvflex.bvh": 7, "nvflex.sort": 6}
EXPECTED_DXBC = 84


def module_of_member(member: str) -> str | None:
    return MEMBER_MODULE.get(member)
