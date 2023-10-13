#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#define gettid() syscall(SYS_gettid)

static inline void check_cudart(cudaError_t result, char const *const func,
                                const char *const file, int const line) {
  if (result) {
    printf("pid:%d tid:%d CUDA error at %s:%d code:%d message:%s\n", getpid(),
           gettid(), file, line, static_cast<unsigned int>(result),
           cudaGetErrorName(result));
    abort();
  }
}

#define CHECK_CUDA(val) check_cudart((val), #val, __FILE__, __LINE__)

static inline CUresult checkDrvError(CUresult res, const char *tok,
                                     const char *file, unsigned line) {
  if (res != CUDA_SUCCESS) {
    const char *errStr = NULL;
    int dev = -1;

    CHECK_CUDA(cudaGetDevice(&dev));
    (void)cuGetErrorString(res, &errStr);
    printf("%s:%d func:%s gpu:%d error:%d str:%s\n", file, line, tok, dev, res,
           errStr);
    fflush(stdout);
  }
  return res;
}
#define CHECK_DRV(x) checkDrvError(x, #x, __FILE__, __LINE__);
