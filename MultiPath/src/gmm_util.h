#pragma once
#include <cxxabi.h>
#include <execinfo.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <iostream>
#include <memory>
#include <mutex>

#ifdef SYS_gettid
#define gettid() ((pid_t)syscall(SYS_gettid))
#else
#error "SYS_gettid is unavailable on this system"
#endif

#define alignSize(x, n) (((x) + ((n)-1)) & ~((n)-1))

static std::string demangle(const char* const symbol) {
  const std::unique_ptr<char, decltype(&std::free)> demangled(
      abi::__cxa_demangle(symbol, 0, 0, 0), &std::free);
  if (demangled) {
    return demangled.get();
  } else {
    return symbol;
  }
}

#define gtrace()                                                         \
  {                                                                      \
    void* array[96];                                                     \
    size_t size = backtrace(array, 96);                                  \
    char** strings = backtrace_symbols(array, size);                     \
    if (strings) {                                                       \
      printf("------------------\n");                                    \
      for (int i = 0; i < size; ++i) printf("[%d] %s\n", i, strings[i]); \
      printf("------------------\n");                                    \
      free(strings);                                                     \
      strings = NULL;                                                    \
    }                                                                    \
  }

static void g_backtrace(const char* file_name, const char* func_name,
                        const int line) {
  std::cout << "\n==============" << file_name << ":" << func_name << ":"
            << line << " call stack===============\n";
  void* addresses[256];
  const int n = ::backtrace(addresses, std::extent<decltype(addresses)>::value);
  const std::unique_ptr<char*, decltype(&std::free)> symbols(
      ::backtrace_symbols(addresses, n), &std::free);
  for (int i = 0; i < n; ++i) {
    // we parse the symbols retrieved from backtrace_symbols() to
    // extract the "real" symbols that represent the mangled names.
    char* const symbol = symbols.get()[i];
    char* end = symbol;
    while (*end) {
      ++end;
    }
    // scanning is done backwards, since the module name
    // might contain both '+' or '(' characters.
    while (end != symbol && *end != '+') {
      --end;
    }
    char* begin = end;
    while (begin != symbol && *begin != '(') {
      --begin;
    }

    if (begin != symbol) {
      std::cout << std::string(symbol, ++begin - symbol);
      *end++ = '\0';
      std::cout << demangle(begin) << '+' << end;
    } else {
      std::cout << symbol;
    }
    std::cout << std::endl;
  }

  std::cout << "==============================================================="
               "=====\n\n";
}

static bool is_call_by(const char* parent_func_name) {
  void* addresses[256];
  const int n = ::backtrace(addresses, std::extent<decltype(addresses)>::value);
  const std::unique_ptr<char*, decltype(&std::free)> symbols(
      ::backtrace_symbols(addresses, n), &std::free);
  for (int i = 0; i < n; ++i) {
    char* const symbol = symbols.get()[i];
    char* end = symbol;
    while (*end) {
      ++end;
    }
    while (end != symbol && *end != '+') {
      --end;
    }
    char* begin = end;
    while (begin != symbol && *begin != '(') {
      --begin;
    }

    if (begin != symbol) {
      *end++ = '\0';
      std::string func_full_name = demangle(++begin);
      if (func_full_name.find(parent_func_name) != std::string::npos)
        return true;
      else
        continue;
    }
  }
  return false;
}

typedef enum {
  FATAL = 0,
  ERROR = 1,
  WARN = 2,

  INFO = 3,
  LOG_DEFAULT = 4,

  DEBUG = 4,
  VERBOSE = 5,
} log_level_t;

static volatile thread_local log_level_t gmm_log_level;

static inline void gmm_set_log_level() {
  int log_level =
      std::getenv("GMM_LOG") ? std::atoi(std::getenv("GMM_LOG")) : LOG_DEFAULT;

  if (log_level >= FATAL && log_level <= VERBOSE) {
    gmm_log_level = (log_level_t)log_level;
  }
}

static std::mutex gmm_log_lock;
static thread_local FILE* gmm_log_file = nullptr;
//FILE *gmm_log_fd = fp ? fp : stdout;                              \

#define gmm_logger(level, fp, format, ...)                                  \
  ({                                                                        \
    if (level <= gmm_log_level) {                                           \
      FILE* gmm_log_fd = stdout;                                            \
      setvbuf(gmm_log_fd, NULL, _IOLBF, 0);                                 \
      switch (level) {                                                      \
        case VERBOSE:                                                       \
          fprintf(gmm_log_fd, "[VERB] ");                                   \
          break;                                                            \
        case DEBUG:                                                         \
          fprintf(gmm_log_fd, "[DEBG] ");                                   \
          break;                                                            \
        case INFO:                                                          \
          fprintf(gmm_log_fd, "[INFO] ");                                   \
          break;                                                            \
        case WARN:                                                          \
          fprintf(gmm_log_fd, "[WARN] ");                                   \
          break;                                                            \
        case ERROR:                                                         \
          fprintf(gmm_log_fd, "[ERRO] ");                                   \
          break;                                                            \
        case FATAL:                                                         \
          fprintf(gmm_log_fd, "[FATAL] ");                                  \
          break;                                                            \
      }                                                                     \
      std::lock_guard<std::mutex> lock_(gmm_log_lock);                      \
      fprintf(gmm_log_fd, "%s:%s:%d: " format "\n", __FILE__, __FUNCTION__, \
              __LINE__, ##__VA_ARGS__);                                     \
      if (0 && level == FATAL) {                                            \
        exit(-1);                                                           \
      }                                                                     \
    }                                                                       \
  })

#define LOGGER(level, format, ...) \
  gmm_logger(level, gmm_log_file, format, ##__VA_ARGS__)

static inline void do_assert(bool check, const char* tok, const char* file,
                             unsigned line, const char* msg) {
  if (check != true) {
    LOGGER(ERROR, "assert failure at %s:%d check:%s msg:%s\n", file, line, tok,
           msg);
    fflush(stdout);
    abort();
  }
}

#define ASSERT(x, ...) do_assert(x, #x, __FILE__, __LINE__, ##__VA_ARGS__);

static bool gmm_is_file_exist(const std::string& path) {
  struct stat buffer;
  return (stat(path.c_str(), &buffer) == 0);
}
