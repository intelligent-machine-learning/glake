#pragma once

#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>

#define CODENAME "xgpu"
static const char MODULE_NAME[] = "CUDA-Driver";

#define LOGEMPTY(format, ...)
#define LOG(format, ...) fprintf(stdout, format "\n", ##__VA_ARGS__);
#define LOGE(format, ...)                                         \
  fprintf(stdout, "\n!!!%s L%d:" format "\n", __FILE__, __LINE__, \
          ##__VA_ARGS__);                                         \
  fflush(stdout);
#define LOGW(format, ...)                                                \
  fprintf(stdout, CODENAME ": %s L-%d:" format "\n", __FILE__, __LINE__, \
          ##__VA_ARGS__);                                                \
  fflush(stdout);
#define LOGI(format, ...)                                                \
  fprintf(stdout, CODENAME ": %s L-%d:" format "\n", __FILE__, __LINE__, \
          ##__VA_ARGS__);                                                \
  fflush(stdout);

#ifdef DEBUG_
#define LOGD(format, ...)                                          \
  fprintf(stdout, CODENAME ":%s %s() L-%d:" format "\n", __FILE__, \
          __FUNCTION__, __LINE__, ##__VA_ARGS__);                  \
  fflush(stdout);
#else
#define LOGD(format, ...)
#endif

#define LOGT(format, ...) LOGD(format, __VA_ARGS__)

struct callStats {
  int cnt;
  long avgTm;
  const char *name;
};

struct thrStats {
  unsigned long thrId;
  long avgTm;
  int cnt;
};

struct api_stats {
  pthread_spinlock_t statLock;
  int api_num;
  int thread_num;

  struct callStats g_stats[512];
  struct thrStats g_thr[64];

 public:
  api_stats() {
    pthread_spin_init(&statLock, PTHREAD_PROCESS_PRIVATE);
    api_num = 0;
    thread_num = 0;
    memset(g_stats, 0, sizeof(g_stats));
    memset(g_thr, 0, sizeof(g_thr));
  }

  ~api_stats() {}

 public:
  void print_stat();
};

#define API_STATS()                                                   \
  {                                                                   \
    pthread_spin_lock(&statLock);                                     \
    int index = api_num, i = 0, hit = 0;                              \
    for (; i < api_num; i++) {                                        \
      if (strcmp(__FUNCTION__, g_stats[i].name) == 0) {               \
        index = i;                                                    \
        hit = 1;                                                      \
        break;                                                        \
      }                                                               \
    }                                                                 \
    if (hit == 0) g_stats[index].name = __FUNCTION__;                 \
    g_stats[index].cnt++;                                             \
    if (index >= api_num) api_num++;                                  \
    index = thread_num, hit = 0;                                      \
    for (i = 0; i < thread_num; i++) {                                \
      if (g_thr[i].thrId == (unsigned long)pthread_self()) {          \
        index = i;                                                    \
        hit = 1;                                                      \
        break;                                                        \
      }                                                               \
    }                                                                 \
    if (hit == 0) g_thr[index].thrId = (unsigned long)pthread_self(); \
    g_thr[index].cnt++;                                               \
    if (index >= thread_num) thread_num++;                            \
    pthread_spin_unlock(&statLock);                                   \
  }

#define __CF(func) ((MODULE_STATUS(*)(...))dlsym(libP, func))
//#define __I(func)  API_STATS(); return __CF(func)
#define __I(func) return __CF(func)
#define __C() __I(__FUNCTION__)

#define __TODO(func) ASSERT(0, "%s is to implement/test", func);
#define TODO() __TODO(__FUNCTION__)
