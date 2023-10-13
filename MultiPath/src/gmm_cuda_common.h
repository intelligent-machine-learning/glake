#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "gmm_queue.h"
#include "gmm_util.h"

static void check(cudaError_t result, char const *const func,
                  const char *const file, int const line) {
  if (result) {
    LOGGER(ERROR, "pid:%d tid:%d CUDA error at %s:%d code:%d message:%s\n",
           getpid(), gettid(), file, line, static_cast<unsigned int>(result),
           cudaGetErrorName(result));
    abort();
  }
}

// check result from CUDA RT
#define CHECK_CUDA(val) check((val), #val, __FILE__, __LINE__)
static inline CUresult checkDrvError(CUresult res, const char *tok,
                                     const char *file, unsigned line) {
  if (res != CUDA_SUCCESS) {
    const char *errStr = NULL;
    int dev = -1;

    CHECK_CUDA(cudaGetDevice(&dev));
    (void)cuGetErrorString(res, &errStr);
    LOGGER(ERROR, "%s:%d func:%s gpu:%d error:%d str:%s\n", file, line, tok,
           dev, res, errStr);
    fflush(stdout);
  }

  return res;
}

// check result from CUDA Driver
#define CHECK_DRV(x) checkDrvError(x, #x, __FILE__, __LINE__);

struct gmm_evt_queue {
  // pid_t pid; // evt user's pid
  // CUstream stream; //evt user's stream

  // the idx of pre-created evt
  fifo_queue<uint32_t> evt_queue;

 public:
  gmm_evt_queue() {}

  void fill(uint32_t nr) {
    for (uint32_t i = 0; i < nr; ++i) {
      evt_queue.push(i);
    }
  }

  void get_free_idx(uint32_t *idx) { *idx = *evt_queue.pop().get(); }
};
