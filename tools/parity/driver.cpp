// flexdecomp differential driver (G2).
//
// Loads the original and the reconstructed shared library with
// RTLD_LOCAL|RTLD_LAZY, resolves one symbol from each, and runs the generated
// per-unit harness in forked children so that a crash on one side is recorded
// as data instead of killing the run.
//
// The generated unit file (diff_<unit>.cpp) owns all type knowledge and
// exposes:
//
//   extern "C" void  fd_setup(long iter);      // deterministic inputs
//   extern "C" void  fd_call(void* fn);        // invoke through fn
//   extern "C" void  fd_serialize(int fd);     // write outputs to fd
//   extern "C" const char* fd_output_table();  // names of serialized outputs
//
// Output framing: for each output, [uint64 length][bytes].

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#include "driver_support.h"

namespace {

struct RunResult {
  bool loaded = false;
  bool crashed = false;
  int signal = 0;
  int exit_code = 0;
  std::vector<unsigned char> payload;
  std::string error;
};

std::vector<unsigned char> read_all(int fd) {
  std::vector<unsigned char> out;
  unsigned char buf[8192];
  for (;;) {
    ssize_t n = read(fd, buf, sizeof buf);
    if (n < 0) {
      if (errno == EINTR) continue;
      break;
    }
    if (n == 0) break;
    out.insert(out.end(), buf, buf + n);
  }
  return out;
}

// Run fd_setup/fd_call in a child; capture fd_serialize output.
RunResult run_in_child(void *fn, long iter, const char *what) {
  RunResult r;
  int pipefd[2];
  if (pipe(pipefd) != 0) {
    r.error = "pipe failed";
    return r;
  }
  pid_t pid = fork();
  if (pid == 0) {
    close(pipefd[0]);
    fd_setup(iter);
    fd_call(fn);
    fd_serialize(pipefd[1]);
    close(pipefd[1]);
    _exit(0);
  }
  close(pipefd[1]);
  r.payload = read_all(pipefd[0]);
  close(pipefd[0]);
  int status = 0;
  waitpid(pid, &status, 0);
  if (WIFSIGNALED(status)) {
    r.crashed = true;
    r.signal = WTERMSIG(status);
  } else if (WIFEXITED(status)) {
    r.exit_code = WEXITSTATUS(status);
    r.crashed = r.exit_code != 0;
  }
  r.loaded = true;
  (void)what;
  return r;
}

void split_payload(const std::vector<unsigned char> &p,
                   std::vector<std::vector<unsigned char>> &outs,
                   std::vector<std::string> &names) {
  size_t off = 0;
  size_t idx = 0;
  while (off + 8 <= p.size()) {
    uint64_t len = 0;
    memcpy(&len, p.data() + off, 8);
    off += 8;
    if (off + len > p.size()) break;
    outs.emplace_back(p.begin() + off, p.begin() + off + len);
    off += len;
    names.push_back(idx < 64 ? "" : "");
    idx++;
  }
}

const char *basename_of(const char *path) {
  const char *s = strrchr(path, '/');
  return s ? s + 1 : path;
}

}  // namespace

int main(int argc, char **argv) {
  const char *orig_path = nullptr;
  const char *recon_path = nullptr;
  const char *symbol = nullptr;
  long iterations = 8;
  bool verbose = false;

  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--orig") && i + 1 < argc) orig_path = argv[++i];
    else if (!strcmp(argv[i], "--recon") && i + 1 < argc) recon_path = argv[++i];
    else if (!strcmp(argv[i], "--symbol") && i + 1 < argc) symbol = argv[++i];
    else if (!strcmp(argv[i], "--iterations") && i + 1 < argc) iterations = atol(argv[++i]);
    else if (!strcmp(argv[i], "--verbose")) verbose = true;
    else {
      fprintf(stderr, "driver: unknown argument %s\n", argv[i]);
      return 2;
    }
  }
  if (!orig_path || !recon_path || !symbol) {
    fprintf(stderr, "usage: driver --orig L --recon L --symbol S [--iterations N]\n");
    return 2;
  }

  void *h_orig = dlopen(orig_path, RTLD_LOCAL | RTLD_LAZY);
  if (!h_orig) {
    printf("{\"ok\":false,\"error\":\"dlopen_orig\",\"detail\":\"%s\"}\n", dlerror());
    return 1;
  }
  void *h_recon = dlopen(recon_path, RTLD_LOCAL | RTLD_LAZY);
  if (!h_recon) {
    printf("{\"ok\":false,\"error\":\"dlopen_recon\",\"detail\":\"%s\"}\n", dlerror());
    return 1;
  }

  void *fn_orig = dlsym(h_orig, symbol);
  void *fn_recon = dlsym(h_recon, symbol);
  if (!fn_orig || !fn_recon) {
    printf("{\"ok\":false,\"error\":\"symbol_missing\",\"orig\":%s,\"recon\":%s,\"symbol\":\"%s\"}\n",
           fn_orig ? "true" : "false", fn_recon ? "true" : "false", symbol);
    return 1;
  }

  long identical = 0, rejected = 0, diverged = 0, first_bad = -1;
  std::string first_detail;

  for (long it = 0; it < iterations; it++) {
    RunResult a = run_in_child(fn_orig, it, "orig");
    RunResult b = run_in_child(fn_recon, it, "recon");
    if (a.crashed && b.crashed) {
      rejected++;
      continue;
    }
    if (a.crashed != b.crashed) {
      diverged++;
      if (first_bad < 0) {
        first_bad = it;
        char buf[256];
        snprintf(buf, sizeof buf,
                 "%s crashed (signal %d exit %d) while %s did not",
                 a.crashed ? "orig" : "recon", a.crashed ? a.signal : b.signal,
                 a.crashed ? a.exit_code : b.exit_code,
                 a.crashed ? "recon" : "orig");
        first_detail = buf;
      }
      continue;
    }
    if (a.payload == b.payload) {
      identical++;
      continue;
    }
    diverged++;
    if (first_bad < 0) {
      first_bad = it;
      std::vector<std::vector<unsigned char>> oa, ob;
      std::vector<std::string> na, nb;
      split_payload(a.payload, oa, na);
      split_payload(b.payload, ob, nb);
      size_t n = oa.size() < ob.size() ? oa.size() : ob.size();
      char buf[256];
      for (size_t k = 0; k < n; k++) {
        if (oa[k] != ob[k]) {
          snprintf(buf, sizeof buf,
                   "output %zu differs (%zu vs %zu bytes)", k, oa[k].size(), ob[k].size());
          first_detail = buf;
          if (verbose) {
            size_t m = oa[k].size() < ob[k].size() ? oa[k].size() : ob[k].size();
            size_t at = 0;
            while (at < m && oa[k][at] == ob[k][at]) at++;
            char hex[96];
            snprintf(hex, sizeof hex, "first differing byte at %zu", at);
            first_detail += ", ";
            first_detail += hex;
          }
          break;
        }
      }
      if (first_detail.empty()) first_detail = "payload length differs";
    }
  }

  bool ok = diverged == 0 && identical > 0;
  printf("{\"ok\":%s,\"identical\":%ld,\"rejected\":%ld,\"diverged\":%ld,"
         "\"iterations\":%ld,\"first_bad\":%ld,\"detail\":\"%s\","
         "\"orig\":\"%s\",\"recon\":\"%s\"}\n",
         ok ? "true" : "false", identical, rejected, diverged, iterations, first_bad,
         first_detail.c_str(), basename_of(orig_path), basename_of(recon_path));
  dlclose(h_recon);
  dlclose(h_orig);
  return ok ? 0 : 1;
}
