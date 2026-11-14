// tools/apitest/runner.cpp -- flex_apitest.exe
//
// One process per implementation DLL (see tools/apitest/README.md): the runner
// LoadLibraryA's the target FleX DLLs, resolves every API entry point through
// the generated flex_api.h bindings, installs the FleX error callback as a
// record sink, and runs one --scenario, serializing every Reporter::emit to
// <out>/<scenario>.bin as [u32 nameLen][name][u64 dataLen][data] and mirroring
// the record metadata into <out>/<scenario>.manifest.json.
//
// CLI:
//   flex_apitest.exe --dll <core.dll> [--ext <ext.dll>] [--device <device.dll>]
//                    --scenario <name> --out <dir> [--seed N]
//   flex_apitest.exe --list
//
// Exit codes:
//   0  scenario ran to completion
//   3  binding resolution failure (a loaded module missed a GetProcAddress)
//   4  unknown --scenario (or --scenario missing)
//   5  scenario aborted (null lib/solver/binding guard; ScenarioAbort thrown)
//
// Determinism: no wall-clock and no rand() anywhere; scenario inputs come from
// fixed constants or apitest::Rng seeded from --seed (apitest::g_seed).

#include <windows.h>

#include "scenario.h"

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Reporter: serializes each emit immediately; mirrors metadata for compare.py.
// ---------------------------------------------------------------------------
class FileReporter : public apitest::Reporter {
public:
  FileReporter(const char* scenario, const std::string& out_dir)
      : scenario_(scenario) {
    bin_path_ = out_dir + "\\" + scenario_ + ".bin";
    manifest_path_ = out_dir + "\\" + scenario_ + ".manifest.json";
    file_ = std::fopen(bin_path_.c_str(), "wb");
    ok_ = file_ != nullptr;
  }

  ~FileReporter() override {
    if (file_ != nullptr) {
      std::fclose(file_);
      file_ = nullptr;
    }
  }

  bool ok() const { return ok_; }
  const std::string& bin_path() const { return bin_path_; }

  // note() is protected in the base Reporter, but the runner (and the error
  // sink) call it through this static type; overriding it public is the
  // simplest way to keep that legal.
  void note(const char* fmt, ...) override {
    va_list ap;
    va_start(ap, fmt);
    std::fprintf(stderr, "flex_apitest: ");
    std::vfprintf(stderr, fmt, ap);
    std::fprintf(stderr, "\n");
    va_end(ap);
  }

  // Writes <scenario>.manifest.json.  Called after every scenario attempt,
  // including aborts, so compare.py always sees the record metadata.
  void write_manifest() {
    if (file_ != nullptr) {
      std::fclose(file_);
      file_ = nullptr;
    }
    FILE* m = std::fopen(manifest_path_.c_str(), "wb");
    if (m == nullptr) return;
    std::fprintf(m, "{\n  \"scenario\": \"%s\",\n  \"records\": [\n",
                 scenario_.c_str());
    for (size_t i = 0; i < entries_.size(); ++i) {
      const Entry& e = entries_[i];
      std::fprintf(m,
                   "    {\"name\": \"%s\", \"dtype\": \"%s\", \"strict\": %s, "
                   "\"len\": %llu, \"masks\": [",
                   e.name.c_str(), e.dtype.c_str(),
                   e.strict ? "true" : "false",
                   (unsigned long long)e.len);
      for (size_t k = 0; k < e.masks.size(); ++k) {
        std::fprintf(m, "%s{\"offset\": %llu, \"len\": %llu}",
                     k ? ", " : "",
                     (unsigned long long)e.masks[k].offset,
                     (unsigned long long)e.masks[k].len);
      }
      std::fprintf(m, "]}%s\n", i + 1 < entries_.size() ? "," : "");
    }
    std::fprintf(m, "  ]\n}\n");
    std::fclose(m);
  }

protected:
  void emit(const char* name, const void* data, uint64_t len,
            const char* dtype, bool strict,
            std::initializer_list<apitest::MaskRange> masks,
            const char* /*extra*/) override {
    if (!ok_) return;
    const uint32_t name_len = (uint32_t)std::strlen(name);
    if (std::fwrite(&name_len, sizeof(name_len), 1, file_) != 1 ||
        std::fwrite(name, 1, name_len, file_) != name_len ||
        std::fwrite(&len, sizeof(len), 1, file_) != 1 ||
        (len > 0 && std::fwrite(data, 1, (size_t)len, file_) != len)) {
      std::fprintf(stderr, "flex_apitest: failed writing %s\n", bin_path_.c_str());
      ok_ = false;
    }
    Entry e;
    e.name = name;
    e.dtype = dtype;
    e.strict = strict;
    e.len = len;
    e.masks.assign(masks.begin(), masks.end());
    entries_.push_back(e);
  }

private:
  struct Entry {
    std::string name;
    std::string dtype;
    bool strict;
    uint64_t len;
    std::vector<apitest::MaskRange> masks;
  };

  std::string scenario_;
  std::string bin_path_;
  std::string manifest_path_;
  FILE* file_ = nullptr;
  bool ok_ = false;
  std::vector<Entry> entries_;
};

// ---------------------------------------------------------------------------
// Error-callback sink: installed before any API call can run.  flexErrorCallback
// (flex_api.h) names the record error.<n> and hands the message here; it is
// recorded as non-strict text (build paths may differ between binaries) plus
// an informational note with the source location.
// ---------------------------------------------------------------------------
FileReporter* g_reporter = nullptr;

void ErrorSink(const char* name, const char* msg, const char* file, int line) {
  if (g_reporter == nullptr) return;
  g_reporter->text(name, msg ? msg : "", /*strict=*/false);
  g_reporter->note("error callback %s from %s:%d", name, file, line);
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------
struct Args {
  std::string dll, ext, device, scenario, out;
  unsigned int seed = 0;
  bool list = false;
  bool valid = false;
};

void usage() {
  std::fprintf(stderr,
               "usage: flex_apitest.exe --dll <core.dll> [--ext <ext.dll>] "
               "[--device <device.dll>] --scenario <name> --out <dir> [--seed N]\n"
               "       flex_apitest.exe --list\n");
}

Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const bool has_next = i + 1 < argc;
    if (arg == "--list") {
      a.list = true;
    } else if (arg == "--dll" && has_next) {
      a.dll = argv[++i];
    } else if (arg == "--ext" && has_next) {
      a.ext = argv[++i];
    } else if (arg == "--device" && has_next) {
      a.device = argv[++i];
    } else if (arg == "--scenario" && has_next) {
      a.scenario = argv[++i];
    } else if (arg == "--out" && has_next) {
      a.out = argv[++i];
    } else if (arg == "--seed" && has_next) {
      a.seed = (unsigned int)std::strtoul(argv[++i], nullptr, 10);
    } else {
      std::fprintf(stderr, "flex_apitest: unknown/invalid argument %s\n", arg.c_str());
      usage();
      return a;
    }
  }
  if (a.list) {
    a.valid = true;
    return a;
  }
  if (a.dll.empty() || a.scenario.empty() || a.out.empty()) {
    usage();
    return a;
  }
  a.valid = true;
  return a;
}

void mkdir_recursive(const std::string& dir) {
  std::string path = dir;
  for (size_t i = 0; i < path.size(); ++i) {
    if (path[i] == '\\' || path[i] == '/') {
      const std::string part = path.substr(0, i);
      if (!part.empty() && part.find(':') == std::string::npos)
        CreateDirectoryA(part.c_str(), nullptr);
    }
  }
  CreateDirectoryA(path.c_str(), nullptr);
}

// ---------------------------------------------------------------------------
// Scenario registry
// ---------------------------------------------------------------------------
class Registry : public apitest::ScenarioSink {
public:
  std::vector<apitest::Scenario> scenarios;
  void add(const apitest::Scenario& s) override { scenarios.push_back(s); }
};

const apitest::Scenario* find_scenario(const Registry& reg,
                                       const std::string& name) {
  for (const apitest::Scenario& s : reg.scenarios)
    if (name == s.name) return &s;
  return nullptr;
}

} // namespace

int main(int argc, char** argv) {
  const Args args = parse_args(argc, argv);
  if (!args.valid) return 4;  // bad CLI behaves like "unknown scenario"

  // Register every pack.  With zero tools/apitest/*_scenarios.cpp files the
  // CMake target falls back to a work/-only stub pack (see CMakeLists.txt).
  Registry registry;
  register_core_pack(registry);
  register_sim_pack(registry);
  register_ext_pack(registry);

  if (args.list) {
    for (const apitest::Scenario& s : registry.scenarios)
      std::printf("%s\t%s\n", s.name, s.desc);
    return 0;
  }

  const apitest::Scenario* scenario = find_scenario(registry, args.scenario);
  if (scenario == nullptr) {
    std::fprintf(stderr, "flex_apitest: unknown scenario %s; use --list\n",
                 args.scenario.c_str());
    return 4;
  }

  // Determinism: fixed seed only, propagated to packs via apitest::g_seed.
  apitest::g_seed = args.seed;

  mkdir_recursive(args.out);

  // Load the implementation DLLs.  A DLL that is not provided stays null and
  // its bindings stay null; scenarios touching them abort with exit code 5.
  HMODULE core = nullptr, ext = nullptr, device = nullptr;
  if (!args.dll.empty()) {
    core = LoadLibraryA(args.dll.c_str());
    if (core == nullptr) {
      std::fprintf(stderr, "flex_apitest: LoadLibraryA(%s) failed (%lu)\n",
                   args.dll.c_str(), GetLastError());
      return 3;
    }
  }
  if (!args.ext.empty()) {
    ext = LoadLibraryA(args.ext.c_str());
    if (ext == nullptr) {
      std::fprintf(stderr, "flex_apitest: LoadLibraryA(%s) failed (%lu)\n",
                   args.ext.c_str(), GetLastError());
      return 3;
    }
  }
  if (!args.device.empty()) {
    device = LoadLibraryA(args.device.c_str());
    if (device == nullptr) {
      std::fprintf(stderr, "flex_apitest: LoadLibraryA(%s) failed (%lu)\n",
                   args.device.c_str(), GetLastError());
      return 3;
    }
  }

  // Resolve the full binding surface; a loaded module that misses any export
  // is a hard failure listing the missing names.
  if (!resolve_api(core, ext, device)) {
    std::fprintf(stderr, "flex_apitest: binding resolution failed, %d missing:\n",
                 g_api_missing_count);
    for (int i = 0; i < g_api_missing_count; ++i)
      std::fprintf(stderr, "  %s\n", g_api_missing[i]);
    return 3;
  }

  FileReporter reporter(scenario->name, args.out);
  if (!reporter.ok()) {
    std::fprintf(stderr, "flex_apitest: cannot create %s\n",
                 reporter.bin_path().c_str());
    return 5;
  }
  g_reporter = &reporter;
  apitest::g_flex_error_sink = &ErrorSink;
  apitest::g_flex_error_seq = 0;

  reporter.note("scenario %s: seed=%u dll=%s ext=%s device=%s",
                scenario->name, args.seed, args.dll.c_str(), args.ext.c_str(),
                args.device.c_str());

  int exit_code = 0;
  try {
    scenario->fn(reporter);
    reporter.note("scenario %s: completed", scenario->name);
  } catch (const apitest::ScenarioAbort&) {
    reporter.note("scenario %s: aborted (null lib/solver/binding guard)",
                  scenario->name);
    exit_code = 5;
  } catch (...) {
    reporter.note("scenario %s: aborted (unknown exception)", scenario->name);
    exit_code = 5;
  }

  // Manifest is written for every outcome, including aborts, so compare.py
  // always sees the record metadata.  Records themselves were serialized
  // incrementally, so data captured before an abort survives.
  reporter.write_manifest();
  g_reporter = nullptr;
  apitest::g_flex_error_sink = nullptr;

  // Do not FreeLibrary: the DLL owns worker threads/contexts we already asked
  // to shut down inside the scenario; unloading a CUDA runtime DLL here would
  // only add nondeterministic teardown noise after records are written.
  return exit_code;
}
