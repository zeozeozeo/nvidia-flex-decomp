# apitest coverage

## Export sets

| target | shipped | .def | recon DLL | status |
|---|---|---|---|---|
| nvflex | 81 | 81 | 81 | def ok; DLL exports match |
| nvflexext | 40 | 40 | 40 | def ok; DLL exports match |
| nvflexdevice | 3 | 3 | 3 | def ok; DLL exports match |

## API coverage by scenario packs

Packs: core_scenarios.cpp, ext_scenarios.cpp, sim_scenarios.cpp

| target | shipped exports | covered | uncovered |
|---|---|---|---|
| nvflex | 81 | 80 | 1 |
| nvflexext | 40 | 40 | 0 |
| nvflexdevice | 3 | 3 | 0 |

Uncovered exports:
- nvflex:NvFlexRegisterD3DBuffer

**Problems (hard)**
- 1 shipped exports not exercised by any scenario pack: ['nvflex:NvFlexRegisterD3DBuffer']
